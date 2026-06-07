"""
Batch prediction pipeline. Loads the latest feature row for the city and
writes 24h/48h/72h predictions + any alerts to MongoDB. Handy for testing the
full path without the API, and for scheduled batch forecasting.

Run:
    python pipelines/batch_predict_pipeline.py
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from aqi_predictor.config import get_settings
from aqi_predictor.database import collections as C
from aqi_predictor.database.mongodb_client import get_db
from aqi_predictor.logging_config import get_logger
from aqi_predictor.models.predict import predict_all
from aqi_predictor.utils.aqi import is_hazardous

logger = get_logger(__name__)


def run() -> dict:
    s = get_settings()
    db = get_db()
    latest = db[C.AQI_FEATURES].find_one(
        {"city": s.city_name}, sort=[("timestamp", -1)]
    )
    if not latest:
        logger.error("No features available. Run the feature pipeline first.")
        return {}

    preds = predict_all(latest)
    now = datetime.now(timezone.utc)
    record = {
        "city": s.city_name,
        "based_on_timestamp": latest.get("timestamp"),
        "current_aqi": latest.get("aqi"),
        "predictions": preds,
        "created_at": now,
    }
    db[C.PREDICTIONS].insert_one(record)

    for p in preds:
        if is_hazardous(p["predicted_aqi"]):
            db[C.ALERTS].insert_one({
                "city": s.city_name, "horizon": p["horizon"],
                "predicted_aqi": p["predicted_aqi"],
                "aqi_category": p["aqi_category"],
                "created_at": now,
            })
            logger.warning("ALERT %s: AQI %s (%s)",
                           p["horizon"], p["predicted_aqi"], p["aqi_category"])
    return record


if __name__ == "__main__":
    out = run()
    print("Batch prediction complete.")
    for p in out.get("predictions", []):
        print(f"  {p['horizon']}: AQI {p['predicted_aqi']} ({p['aqi_category']})")
