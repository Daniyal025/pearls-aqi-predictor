# Demo Script — Pearls AQI Predictor

A timed walkthrough for presenting the project (about 8–10 minutes). Each
section lists **what to say** and **what to click/run**. Do a dry run once
beforehand, and warm up the Render API a minute early (free tier sleeps).

---

## Before you start (do this 2 minutes early)

- Open the Render API URL `/health` in a browser tab so the service wakes up.
- Open the live Streamlit dashboard in another tab.
- Have a terminal open in the project folder with the virtual environment
  active.
- Have the MongoDB Atlas "Browse Collections" view open in a tab.
- Have the GitHub repo's Actions tab open in a tab.

---

## 1. Introduction (about 1 minute)

**Say:** "Pearls AQI Predictor forecasts the Air Quality Index for the next
three days for Karachi, and the city is configurable. It is a full ML
engineering system, not a single notebook: automated data pipelines, a custom
feature store on MongoDB, a model registry, a prediction API, a dashboard,
scheduled retraining, and explainability."

**Say (important honesty point):** "The AQI here is OpenWeather's 1-to-5 scale —
1 Good through 5 Very Poor — not the US 0-to-500 scale. I label it accurately
throughout."

---

## 2. The dashboard — the headline demo (about 2 minutes)

**Click:** the live Streamlit dashboard.

**Show and say:**
- The current AQI card with its colour-coded category.
- The three forecast cards (24h, 48h, 72h) with predicted AQI and category.
- "These forecasts come from the active models for each horizon."
- Scroll to pollutant trend charts: "Recent PM2.5, PM10, ozone and others."
- The model metrics table: "This is the live comparison of all models from the
  most recent training run."
- The SHAP feature-importance chart: "This explains what drives the 24-hour
  model — lag features and the change rate dominate, which makes sense for a
  time series."
- The alert cards at the bottom (if any): "Alerts fire automatically for Poor or
  Very Poor air, with health guidance, and they are deduplicated so we do not
  spam the same alert."

---

## 3. The API (about 1.5 minutes)

**Click:** the Render API `/docs` page.

**Show and say:**
- "FastAPI auto-generates this interactive documentation."
- Expand `GET /health` → Execute: "It reports the API, the MongoDB connection,
  and how many active models are loaded."
- Expand `GET /forecast` → Execute: "A three-day forecast as JSON. Every forecast
  call is also logged to MongoDB, and hazardous predictions create alerts."
- Briefly point out `/model-info`, `/latest-features`, and `/alerts`.

---

## 4. The custom feature store on MongoDB (about 1.5 minutes)

**Click:** the Atlas "Browse Collections" tab.

**Say:** "MongoDB is not a dedicated feature store like Hopsworks. I built a
feature-store-like layer on plain collections."

**Show and say:**
- `aqi_features`: "One row per city per hour. The `feature_id` — city plus hour —
  has a unique index, so re-running pipelines never creates duplicates."
- `model_registry`: "One document per model per horizon. Notice `is_active`:
  exactly one model per horizon serves predictions, which is the versioning
  mechanism."
- `predictions` and `alerts`: "Logged forecasts and alert history."

---

## 5. The pipelines and automation (about 1.5 minutes)

**Option A — run live (if time and connectivity allow):**
```
python pipelines/feature_pipeline.py
python pipelines/batch_predict_pipeline.py
```
**Say:** "The feature pipeline fetches current and forecast pollution, builds
features, and upserts them. The batch predictor writes forecasts and any
alerts."

**Option B — show the schedules (safer for a fixed time slot):**

**Click:** the GitHub Actions tab.

**Say and show:**
- The three workflows: hourly feature pipeline, daily training pipeline, tests.
- Open a recent successful run: "This is the hourly pipeline running on a cron
  schedule on GitHub's servers — no server of my own runs continuously."
- "The test workflow runs on every push. Tests are fully offline, so CI is fast
  and needs no secrets."

---

## 6. Wrap-up (about 1 minute)

**Say:** "To summarise: data flows from OpenWeather through the feature pipeline
into the MongoDB feature store; the daily pipeline trains and compares four
models plus an optional neural network, registers the best, and regenerates SHAP
charts; and the API and dashboard serve everything live."

**Say (limitations, shows maturity):** "Known limitations: the 1-to-5 AQI scale
is coarse, the free tiers have cold starts and rate limits, and database network
access is open for the demo. Future work includes multiple cities, a validated
US-AQI conversion, time-series models, and drift monitoring."

**End:** "Thank you — happy to take questions or dig into any part of the code."

---

## Quick recovery if something fails live

- **Dashboard shows no data:** run `python pipelines/feature_pipeline.py`, then
  refresh.
- **Forecast endpoint errors with "no active models":** run
  `python pipelines/train_pipeline.py` (needs backfilled data first).
- **Render API is slow:** it was asleep; wait for the cold start, and explain
  that free-tier services spin down when idle.
- **Anything cloud-related is down:** fall back to running the dashboard and API
  locally — the same code runs both ways.
