# 🌫️ Pearls AQI Predictor

End-to-end Air Quality Index forecasting (next 24h/48h/72h) for a configurable
city (default: **Karachi, Pakistan**). Serverless-style MLOps using a **custom
MongoDB Atlas feature store + model registry**, OpenWeather data, scikit-learn
(+ optional TensorFlow), FastAPI, Streamlit, SHAP, and GitHub Actions.

> **AQI note:** OpenWeather AQI is a **1–5** scale (1 Good · 2 Fair · 3 Moderate ·
> 4 Poor · 5 Very Poor), *not* the US 0–500 scale.

## 🚀 Live deployments

- **Dashboard (Streamlit Cloud):** https://pearls-aqi-predictor-syed-daniyal-ali.streamlit.app/
- **API (Render):** _add your Render URL here, e.g._ `https://pearls-aqi-api.onrender.com` (try `/docs` and `/health`)

> The Render free tier sleeps when idle, so the first request after a pause may
> take ~30–50s to wake.

> **Deployment requires GridFS.** Training and serving run on different machines
> that do not share a filesystem, so models are stored in MongoDB GridFS (not
> just on local disk). Training uploads the artifact automatically; the API and
> dashboard download it on demand. See "Deployment note" below.

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
- `reports/` — architecture, final report, demo script, screenshots checklist

## Secrets
Never commit `.env`. Use GitHub Secrets in CI, Streamlit secrets on Streamlit
Cloud, and platform env vars on Render/Cloud Run. See `.env.example`.

### `API_BASE_URL` — set it correctly per environment
`API_BASE_URL` tells any caller where the FastAPI service lives.

- **Locally:** `http://127.0.0.1:8000` is correct — the API runs on your machine.
- **On Streamlit Cloud:** `127.0.0.1` would point at the Streamlit server itself,
  where no API runs. Set it to your **Render API URL**
  (e.g. `https://pearls-aqi-api.onrender.com`) in the Streamlit **Secrets**
  manager.

> Note: the dashboard currently reads MongoDB directly and computes forecasts
> in-process, so it does **not** call the API for forecasts — meaning a wrong
> `API_BASE_URL` will not break the forecast cards today. Set it correctly
> anyway so it is right if/when the dashboard is wired to call the API.

## Deployment note — why models live in GridFS
When you train locally (or in GitHub Actions) and serve on Streamlit Cloud /
Render, the machines do **not** share a filesystem. A model saved only to
`artifacts/models/*.joblib` on the training machine cannot be opened by the
deployed app — you would see an error like
`No such file or directory: 'artifacts/models/aqi_24h_..._extra_trees.joblib'`.

To fix this, the training pipeline uploads each model artifact to **MongoDB
GridFS** and records its `gridfs_file_id` in the model registry. The API and
dashboard load the model from the local file when present, and otherwise
download it from GridFS automatically (caching it locally). This is on by
default; set `USE_GRIDFS=false` to disable it for purely local single-machine
use.

**If your deployed dashboard shows the "models not ready" file error:** retrain
once with GridFS enabled so the artifact is uploaded —
`python pipelines/train_pipeline.py` (or trigger the daily training workflow) —
then reload the dashboard.

## Tests
```bash
pytest -q
```
Tests are offline (no DB/network needed).
