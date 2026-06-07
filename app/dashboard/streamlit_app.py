"""
Streamlit dashboard for Pearls AQI Predictor.

Reads directly from MongoDB (works even if the FastAPI service is offline).
Shows current AQI, 3-day forecast, pollutant trends, model metrics, SHAP
image if present, and alert cards for Poor/Very Poor AQI.

Run:
    streamlit run app/dashboard/streamlit_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

import pandas as pd
import streamlit as st

# On Streamlit Cloud, secrets arrive via st.secrets (not env vars). Bridge them
# into os.environ BEFORE get_settings() reads them. Harmless locally (no-op if
# no secrets file is present).
import os
try:
    for _k in ("MONGODB_URI", "MONGODB_DATABASE", "CITY_NAME", "LATITUDE",
               "LONGITUDE", "TIMEZONE", "OPENWEATHER_API_KEY", "API_BASE_URL"):
        if _k in st.secrets:
            os.environ.setdefault(_k, str(st.secrets[_k]))
except Exception:
    pass  # no secrets configured (pure-local run); rely on .env

from aqi_predictor.alerts import recent_alerts
from aqi_predictor.config import get_settings
from aqi_predictor.database import collections as C
from aqi_predictor.database.mongodb_client import get_db
from aqi_predictor.models.predict import predict_all
from aqi_predictor.utils.aqi import aqi_category

CATEGORY_COLORS = {
    "Good": "#2ecc71", "Fair": "#a3d977", "Moderate": "#f1c40f",
    "Poor": "#e67e22", "Very Poor": "#e74c3c", "Unknown": "#95a5a6",
}

st.set_page_config(page_title="Pearls AQI Predictor", page_icon="🌫️", layout="wide")

settings = get_settings(require_secrets=False)
db = get_db()

st.title("🌫️ Pearls AQI Predictor")
st.caption("3-day Air Quality forecast • OpenWeather AQI scale 1–5 (Good → Very Poor)")

city = st.sidebar.text_input("City", value=settings.city_name)

latest = db[C.AQI_FEATURES].find_one({"city": city}, sort=[("timestamp", -1)])
if not latest:
    st.warning(f"No features found for {city}. Run the feature pipeline first.")
    st.stop()

cat = aqi_category(latest.get("aqi"))
col1, col2 = st.columns([1, 2])
with col1:
    st.subheader("Current AQI")
    st.markdown(
        f"<div style='padding:20px;border-radius:12px;background:{CATEGORY_COLORS[cat]};"
        f"color:white;text-align:center'><h1 style='margin:0'>{latest.get('aqi')}</h1>"
        f"<p style='margin:0'>{cat}</p></div>", unsafe_allow_html=True)
    st.caption(f"As of {latest.get('timestamp')}")

with col2:
    st.subheader("Next 3 days forecast")
    try:
        preds = predict_all(latest)
    except Exception as exc:  # noqa: BLE001
        preds = []
        st.info(f"Models not ready: {exc}")
    if preds:
        cols = st.columns(len(preds))
        for c, p in zip(cols, preds):
            color = CATEGORY_COLORS.get(p["aqi_category"], "#95a5a6")
            c.markdown(
                f"<div style='padding:14px;border-radius:10px;background:{color};"
                f"color:white;text-align:center'><b>{p['horizon']}</b><br>"
                f"<span style='font-size:28px'>{p['predicted_aqi']}</span><br>"
                f"{p['aqi_category']}</div>", unsafe_allow_html=True)
        for p in preds:
            if p["alert_level"] == "alert":
                st.error(f"⚠️ {p['horizon']}: {p['aqi_category']} air quality expected.")
    else:
        st.info("Run the training pipeline to enable forecasts.")

st.divider()

# Pollutant trends
st.subheader("Pollutant trends (recent)")
hist = list(db[C.AQI_FEATURES].find({"city": city}).sort("timestamp", -1).limit(72))
if hist:
    dfh = pd.DataFrame(hist).sort_values("timestamp")
    pollutant_cols = [c for c in ["pm2_5", "pm10", "o3", "no2", "so2", "co"] if c in dfh]
    if pollutant_cols:
        st.line_chart(dfh.set_index("timestamp")[pollutant_cols])
    st.line_chart(dfh.set_index("timestamp")[["aqi"]])

st.divider()

# Model metrics
st.subheader("Model metrics")
metrics_doc = db[C.MODEL_METRICS].find_one(sort=[("created_at", -1)])
if metrics_doc:
    st.caption(f"Latest comparison • version {metrics_doc.get('model_version')}")
    st.dataframe(pd.DataFrame(metrics_doc.get("results", [])))
else:
    st.info("No model metrics yet. Run the training pipeline.")

# SHAP image (if generated)
shap_png = Path("artifacts/shap/shap_24h.png")
if shap_png.exists():
    st.subheader("SHAP feature importance (24h model)")
    st.image(str(shap_png))

st.divider()

# Recent alerts (Poor / Very Poor) with health guidance.
st.subheader("Recent alerts")
alerts = recent_alerts(city, limit=10)
if not alerts:
    st.success("No recent Poor/Very Poor AQI alerts for this city.")
else:
    for a in alerts:
        box = st.error if a.get("severity") == "very_poor" else st.warning
        guidance = f" — {a['guidance']}" if a.get("guidance") else ""
        box(f"**{a['aqi_category']}** ({a['horizon']}): {a['message']}{guidance}")
