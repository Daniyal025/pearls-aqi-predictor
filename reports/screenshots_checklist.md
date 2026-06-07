# Screenshots Checklist — Pearls AQI Predictor

Capture these for your submission and slides. Group them as below; this order
also mirrors the demo script. Blur or crop any secret values before saving.

## A. Setup and infrastructure
- [ ] MongoDB Atlas cluster overview (cluster name and tier visible)
- [ ] Atlas "Browse Collections" showing all eight collections present
- [ ] Atlas indexes view for `aqi_features` (unique `feature_id`, `city+timestamp`)
- [ ] Atlas indexes view for `alerts` (unique `alert_key`)
- [ ] OpenWeather account API-keys page (key value blurred)
- [ ] GitHub repository home page (README rendered)
- [ ] GitHub Secrets page, Settings → Secrets and variables → Actions (values hidden)

## B. Data and feature store
- [ ] `aqi_features` collection with several rows expanded (show `feature_id`,
      `aqi`, `aqi_category`, pollutants, lag/rolling features)
- [ ] `training_datasets` document from a backfill run
- [ ] Terminal output of `python pipelines/backfill_pipeline.py --days 30`
- [ ] Terminal output of `python pipelines/feature_pipeline.py`

## C. Training and model registry
- [ ] Terminal output of `python pipelines/train_pipeline.py` showing the model
      comparison (RMSE/MAE/R² per model) and the selected best model
- [ ] `model_registry` document with `is_active: true` for a horizon
- [ ] `model_metrics` document showing the full comparison table
- [ ] The saved artifact file under `artifacts/models/`

## D. Explainability and alerts
- [ ] SHAP feature-importance chart (`artifacts/shap/shap_24h.png`)
- [ ] `alerts` collection showing an alert document (severity, message, guidance)
- [ ] Dashboard alert card for Poor or Very Poor AQI

## E. API (FastAPI)
- [ ] `/docs` interactive documentation page
- [ ] `GET /health` response (api, mongodb, active_models)
- [ ] `GET /forecast` JSON response (24h/48h/72h with categories)
- [ ] `GET /model-info` response
- [ ] `GET /alerts` response

## F. Dashboard (Streamlit)
- [ ] Current AQI card with category
- [ ] Three-day forecast cards
- [ ] Pollutant trend charts
- [ ] Model metrics table
- [ ] SHAP chart section

## G. Automation (GitHub Actions)
- [ ] Actions tab listing all three workflows
- [ ] A successful run of the hourly feature pipeline (logs expanded)
- [ ] A successful run of the daily training pipeline
- [ ] A successful tests workflow run (showing tests passed)
- [ ] A manual "Run workflow" trigger via workflow_dispatch

## H. Deployment
- [ ] Render service overview (status "Live")
- [ ] Render environment variables page (values hidden)
- [ ] Live Render API URL open in a browser at `/docs`
- [ ] Streamlit Community Cloud app overview (status running)
- [ ] Streamlit secrets settings page (values hidden)
- [ ] Live Streamlit dashboard URL in a browser

## Capture tips
- Use a clean browser window without unrelated tabs or bookmarks visible.
- For terminal screenshots, scroll so the command and its key output both show.
- Keep a consistent zoom level so images look uniform in the report.
- Save with descriptive names, e.g. `C_model_registry_active.png`, so they are
  easy to place into the final document.
