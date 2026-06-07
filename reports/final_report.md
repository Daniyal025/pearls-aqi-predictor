# Final Report — Pearls AQI Predictor

## 1. Problem
Forecast Air Quality Index for the next three days for a configurable city
(default: Karachi, Pakistan) using an automated, serverless-style MLOps pipeline.

## 2. Data
OpenWeather Air Pollution API (current, forecast, history) + Weather API.
AQI is on the 1–5 OpenWeather scale (Good → Very Poor). Pollutants: CO, NO,
NO2, O3, SO2, NH3, PM2.5, PM10.

## 3. Feature engineering
Calendar features, pollutant values, AQI + category, lag features (1/3/24h),
rolling means (3/24h), AQI change rate. Targets: AQI shifted 24/48/72h ahead.

## 4. Custom feature store (MongoDB Atlas)
Described in architecture.md. Deduplication via unique `feature_id`.

## 5. Models
Ridge, Random Forest, Gradient Boosting, Extra Trees (+ optional TensorFlow
MLP). Selection by lowest RMSE on a held-out split. Metrics: RMSE, MAE, R².

## 6. Model registry
MongoDB `model_registry` with `is_active` per horizon; artifacts via joblib
(local) or GridFS (optional).

## 7. Serving
FastAPI endpoints (`/predict`, `/forecast`, `/model-info`, `/health`,
`/latest-features`) and a Streamlit dashboard with forecasts, trends, metrics,
SHAP importance, and alert cards.

## 8. Explainability & alerts
SHAP feature importance for tree models; alerts logged for Poor/Very Poor AQI.

## 9. Limitations
Free-tier API/DB limits; optional TF model; demo-level security.

## 10. Future improvements
Multi-city, true US-AQI conversion, better time-series models (e.g. LSTM),
backtesting, drift monitoring, authentication on the API.
