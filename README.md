# 🌫️ Pearls AQI Predictor

End-to-end Air Quality Index forecasting (next 24h/48h/72h) for a configurable
city (default: **Karachi, Pakistan**). Serverless-style MLOps using a **custom
MongoDB Atlas feature store + model registry**, OpenWeather data, scikit-learn
(+ optional TensorFlow), FastAPI, Streamlit, SHAP, and GitHub Actions.

> **AQI note:** OpenWeather AQI is a **1–5** scale (1 Good · 2 Fair · 3 Moderate ·
> 4 Poor · 5 Very Poor), *not* the US 0–500 scale.

## Quick start
```bash
python -m venv .venv
# Windows: .venv\Scripts\Activate.ps1   |  macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env          # then edit .env with your real keys
python -m aqi_predictor.database.indexes   # create collections + indexes
python pipelines/backfill_pipeline.py --days 30   # get training data
python pipelines/train_pipeline.py                # train models
uvicorn app.api.main:app --reload                 # API at http://127.0.0.1:8000/docs
streamlit run app/dashboard/streamlit_app.py      # dashboard
```

## Project layout
- `src/aqi_predictor/` — core library (data, database, features, models, utils)
- `pipelines/` — feature, backfill, train, batch-predict scripts
- `app/api/` — FastAPI · `app/dashboard/` — Streamlit
- `.github/workflows/` — hourly feature, daily training, tests
- `reports/` — architecture, final report, screenshots checklist

## Secrets
Never commit `.env`. Use GitHub Secrets in CI, Streamlit secrets on Streamlit
Cloud, and platform env vars on Render/Cloud Run. See `.env.example`.

## Tests
```bash
pytest -q
```
Tests are offline (no DB/network needed).
