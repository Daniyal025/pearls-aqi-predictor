"""
FastAPI prediction service.

Endpoints:
  GET  /                  welcome
  GET  /health            API + MongoDB + active-model status
  POST /predict           predict from a supplied feature row
  GET  /forecast          3-day forecast from latest stored features
  GET  /model-info        active models + metrics
  GET  /latest-features   latest stored feature row

Run locally:
    uvicorn app.api.main:app --reload
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from aqi_predictor.config import get_settings
from aqi_predictor.database import collections as C
from aqi_predictor.database.mongodb_client import get_db, ping
from aqi_predictor.models.predict import predict_all
from aqi_predictor.models.registry import get_all_active_models
from aqi_predictor.utils.aqi import aqi_category, is_hazardous

app = FastAPI(title="Pearls AQI Predictor API", version="0.1.0")


class FeatureRow(BaseModel):
    # A flexible feature row. Only the fields the model needs are used.
    features: dict[str, Any]


def _latest_features(city: str) -> dict[str, Any] | None:
    return get_db()[C.AQI_FEATURES].find_one({"city": city}, sort=[("timestamp", -1)])


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "Pearls AQI Predictor API", "docs": "/docs"}


@app.get("/health")
def health() -> dict[str, Any]:
    active = get_all_active_models()
    return {
        "api": "ok",
        "mongodb": "ok" if ping() else "down",
        "active_models": len(active),
    }


@app.post("/predict")
def predict(row: FeatureRow) -> dict[str, Any]:
    preds = predict_all(row.features)
    if not preds:
        raise HTTPException(status_code=503, detail="No active models. Train first.")
    return {"predictions": preds}


@app.get("/forecast")
def forecast(city: str | None = None) -> dict[str, Any]:
    s = get_settings()
    city = city or s.city_name
    latest = _latest_features(city)
    if not latest:
        raise HTTPException(status_code=404, detail=f"No features for {city}.")

    preds = predict_all(latest)
    if not preds:
        raise HTTPException(status_code=503, detail="No active models. Train first.")

    now = datetime.now(timezone.utc)
    record = {
        "city": city,
        "based_on_timestamp": latest.get("timestamp"),
        "current_aqi": latest.get("aqi"),
        "current_aqi_category": aqi_category(latest.get("aqi")),
        "predictions": preds,
        "created_at": now,
    }
    db = get_db()
    db[C.PREDICTIONS].insert_one(dict(record))  # log prediction
    for p in preds:
        if is_hazardous(p["predicted_aqi"]):
            db[C.ALERTS].insert_one({
                "city": city, "horizon": p["horizon"],
                "predicted_aqi": p["predicted_aqi"],
                "aqi_category": p["aqi_category"], "created_at": now,
            })
    record.pop("_id", None)
    return record


@app.get("/model-info")
def model_info() -> dict[str, Any]:
    models = get_all_active_models()
    for m in models:
        m.pop("_id", None)
    return {"active_models": models}


@app.get("/latest-features")
def latest_features(city: str | None = None) -> dict[str, Any]:
    s = get_settings()
    city = city or s.city_name
    latest = _latest_features(city)
    if not latest:
        raise HTTPException(status_code=404, detail=f"No features for {city}.")
    latest.pop("_id", None)
    return latest
