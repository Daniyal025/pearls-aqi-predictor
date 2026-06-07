# Architecture — Pearls AQI Predictor

## Overview
End-to-end system that forecasts AQI (next 24h/48h/72h) for a configurable
city (default Karachi). Data flows: OpenWeather → feature pipeline →
MongoDB Atlas (custom feature store) → training pipeline → model registry →
FastAPI + Streamlit serving.

## Why MongoDB as a "custom feature store"
MongoDB is **not** a dedicated ML feature store (like Hopsworks or Vertex AI
Feature Store). We implement a feature-store-*like* layer using:
- deterministic `feature_id` (city + hour) for idempotent upserts (dedup),
- compound index `(city, timestamp)` for time-range reads,
- a `model_registry` collection with an `is_active` flag for versioning,
- `predictions` and `alerts` collections for logging.

## Collections
| Collection | Purpose |
|---|---|
| raw_weather_data | raw weather API responses (audit) |
| raw_pollution_data | raw pollution API responses (audit) |
| aqi_features | model-ready feature rows (one per city per hour) |
| training_datasets | metadata about backfilled datasets |
| model_registry | one doc per model per horizon; `is_active` marks serving model |
| model_metrics | full model-comparison tables |
| predictions | logged forecasts |
| alerts | Poor/Very Poor AQI events |

## AQI scale (important)
OpenWeather AQI is **1–5**, not US 0–500:
1 Good · 2 Fair · 3 Moderate · 4 Poor · 5 Very Poor. We label it honestly and
never call it "US AQI".

## Automation
- Hourly feature pipeline (GitHub Actions cron).
- Daily training pipeline (GitHub Actions cron).
- Tests on every push/PR.

## Limitations
- M0 Atlas tier and OpenWeather free tier limits.
- Network access `0.0.0.0/0` is convenient but not production-secure.
- TensorFlow model is optional; trained only with enough data.
