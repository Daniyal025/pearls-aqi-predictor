# Final Report — Pearls AQI Predictor

**An end-to-end Air Quality Index forecasting system with a custom MongoDB feature store**

---

## 1. Introduction and problem statement

Air pollution is a serious public-health concern in Karachi, Pakistan, where
particulate matter and other pollutants regularly reach unhealthy levels. The
goal of this project is to build a complete, automated system that forecasts
the Air Quality Index (AQI) for the next three days (24h, 48h, 72h) for a
configurable city, defaulting to Karachi.

The project is built to demonstrate real-world ML engineering practice:
automated data pipelines, a feature store, a model registry, a prediction API,
an interactive dashboard, scheduled retraining, continuous integration, and
explainability — rather than a single notebook.

## 2. Objectives

1. Ingest air-pollution and weather data automatically from a public API.
2. Engineer model-ready features and store them in a deduplicated feature store.
3. Backfill historical data to create a training dataset.
4. Train and compare several models, selecting the best by error.
5. Serve forecasts through a REST API and a dashboard.
6. Automate the hourly feature and daily training pipelines.
7. Explain model behaviour (SHAP) and raise alerts for hazardous air quality.

## 3. Data source

Data comes from the **OpenWeather Air Pollution API** (current, forecast, and
historical endpoints) and the **OpenWeather Weather API** for meteorological
context. Pollutants captured: CO, NO, NO2, O3, SO2, NH3, PM2.5, and PM10.

**Important note on the AQI scale.** OpenWeather reports AQI on a **1–5 scale**,
which is *not* the US 0–500 AQI scale. The mapping used throughout the project
is: 1 = Good, 2 = Fair, 3 = Moderate, 4 = Poor, 5 = Very Poor. The system labels
this honestly and never presents it as US AQI, because no conversion between the
two scales is implemented.

## 4. System architecture

The system follows a "thin entry points, fat library" design: reusable logic
lives in the `src/aqi_predictor/` package, while pipelines, the API, and the
dashboard are thin scripts that call into it.

The data flow is: OpenWeather → feature pipeline → MongoDB Atlas → training
pipeline → model registry → FastAPI and Streamlit serving layers. The hourly
feature pipeline and daily training pipeline run on GitHub Actions cron
schedules, giving a serverless-style system where nothing runs continuously
except the lightweight serving apps. See `architecture.md` for the full diagram
and collection-level detail.

## 5. The custom feature store (MongoDB Atlas)

MongoDB is **not** a dedicated ML feature store such as Hopsworks or Vertex AI
Feature Store. This project implements a feature-store-*like* layer using plain
MongoDB collections together with deterministic identifiers and indexes that
provide the core functions a feature store offers:

- **Deduplication and idempotent writes** via a deterministic `feature_id`
  built from city and hour (for example, `Karachi_2026-06-05T10:00:00Z`), with a
  unique index so re-running a pipeline never creates duplicate rows.
- **Time-range retrieval** via a compound `(city, timestamp)` index.
- **Model versioning** via a `model_registry` collection with an `is_active`
  flag, so exactly one model per horizon serves predictions.
- **Logging** of predictions and alerts in dedicated collections.

The eight collections are: `raw_weather_data`, `raw_pollution_data`,
`aqi_features`, `training_datasets`, `model_registry`, `model_metrics`,
`predictions`, and `alerts`.

## 6. Feature engineering

Each feature row represents one city at one hour. Features include calendar
fields (hour, day, month, day of week, weekend flag), the eight pollutant
values, the AQI and its category label, lag features (AQI at 1, 3, and 24 hours
prior), rolling means (3-hour and 24-hour windows for AQI and for PM2.5/PM10),
and an AQI change rate. The forecast targets — `target_aqi_24h`,
`target_aqi_48h`, and `target_aqi_72h` — are produced by shifting the AQI value
forward by 24, 48, and 72 hourly steps respectively.

## 7. Modelling

For each of the three horizons, the training pipeline trains and compares four
scikit-learn regressors — Ridge Regression, Random Forest, Gradient Boosting,
and Extra Trees — and an optional TensorFlow MLP that is attempted only when at
least 500 rows are available. Models are evaluated on a held-out split using
RMSE, MAE, and R². The model with the lowest RMSE is selected, saved as a joblib
artifact, recorded in the model registry, and marked active. The full comparison
table for every horizon is stored in `model_metrics` for display on the
dashboard.

## 8. Serving layer

**FastAPI** exposes seven endpoints: a welcome root, a `/health` check (API +
MongoDB + active-model status), `/predict` (predict from a supplied feature
row), `/forecast` (three-day forecast from the latest stored features),
`/model-info` (active models and their metrics), `/latest-features`, and
`/alerts`. Forecast requests are logged to the `predictions` collection, and
hazardous predictions trigger alert logging.

**Streamlit** provides a presentation-friendly dashboard showing the current
AQI, the three-day forecast with category labels, pollutant trend charts, the
model comparison table, the SHAP feature-importance chart, and alert cards for
Poor or Very Poor air quality. The dashboard reads MongoDB directly, so it
functions even when the API is asleep.

## 9. Automation and CI/CD

Three GitHub Actions workflows run the system: an hourly feature pipeline, a
daily training pipeline (which also regenerates SHAP charts), and a test
workflow that runs on every push and pull request. All secrets are injected
through GitHub Secrets; none are hardcoded. The test suite is fully offline,
requiring no database or network, so CI is fast and reliable.

## 10. Explainability and alerts

SHAP is used to compute feature importance for the active tree-based models,
producing a ranked chart per horizon that is saved as an image and stored in
MongoDB. The alert subsystem raises an alert when a predicted AQI is Poor (4) or
Very Poor (5). Alerts are deduplicated to one per city, horizon, severity, and
calendar day via a unique `alert_key`, so a frequently-running pipeline does not
produce repeated alerts, and each alert carries short health guidance.

## 11. Deployment

The FastAPI service deploys to Render using the included `render.yaml`, with
secrets set in the Render dashboard. The dashboard deploys to Streamlit
Community Cloud, with secrets entered through the Streamlit secrets manager; the
dashboard bridges those secrets into environment variables so the same code runs
both locally and in the cloud.

## 12. Testing

The project includes offline unit tests covering AQI utility functions, feature
engineering (using a synthetic pollution payload), MongoDB schema constants, and
the alert decision and deduplication logic. All tests pass and run without
external services.

## 13. Limitations

- The OpenWeather AQI scale is 1–5, coarser than the US 0–500 scale; no
  conversion is implemented.
- Free tiers (MongoDB Atlas M0, OpenWeather, Render, Streamlit Cloud) impose
  storage, rate, and cold-start limits.
- Atlas network access is opened to all IPs for convenience during the project,
  which is acceptable for a demo but not production-secure.
- The TensorFlow model is optional and only trained with sufficient data.
- Forecast quality depends on how much historical data has been backfilled.

## 14. Future improvements

- Support multiple cities concurrently with per-city models.
- Implement a validated conversion to the US AQI scale.
- Add time-series-specific models (for example, LSTM or temporal models) and
  proper backtesting.
- Add model-drift monitoring and automated rollback in the registry.
- Add authentication and rate limiting to the API.
- Restrict Atlas network access to known deployment IPs.

## 15. Conclusion

Pearls AQI Predictor delivers a complete, automated, and explainable AQI
forecasting system. It demonstrates the full ML engineering lifecycle — from
data ingestion and a custom feature store through training, a model registry,
automated retraining, serving, explainability, and deployment — while remaining
beginner-friendly and reproducible.
