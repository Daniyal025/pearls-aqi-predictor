"""
Hourly feature pipeline.

  1. Fetch current pollution (+ current weather) for the configured city.
  2. Also pull the short pollution forecast so we capture upcoming hours.
  3. Store raw payloads in MongoDB (raw_pollution_data / raw_weather_data).
  4. Build feature rows + time-series features using recent history from DB.
  5. Upsert feature rows into aqi_features (dedup via unique feature_id).

Run:
    python pipelines/feature_pipeline.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make src/ importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd
from pymongo import UpdateOne

from aqi_predictor.config import get_settings
from aqi_predictor.data.openweather_client import OpenWeatherClient
from aqi_predictor.data.validation import validate_pollution_response
from aqi_predictor.database import collections as C
from aqi_predictor.database.mongodb_client import get_db
from aqi_predictor.features.build_features import (
    add_timeseries_features,
    rows_from_pollution_payload,
)
from aqi_predictor.logging_config import get_logger
from aqi_predictor.utils.time_utils import utc_now

logger = get_logger(__name__)


def run() -> int:
    s = get_settings()
    client = OpenWeatherClient(s.openweather_api_key)
    db = get_db()

    pollution = client.current_pollution(s.latitude, s.longitude)
    if not validate_pollution_response(pollution):
        logger.error("Invalid pollution response; aborting.")
        return 0

    try:
        weather = client.current_weather(s.latitude, s.longitude)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Weather fetch failed (%s); continuing without it.", exc)
        weather = None

    forecast = client.pollution_forecast(s.latitude, s.longitude)

    # Store raw payloads for auditing.
    now = utc_now()
    db[C.RAW_POLLUTION].insert_one(
        {"city": s.city_name, "fetched_at": now, "payload": pollution})
    if weather:
        db[C.RAW_WEATHER].insert_one(
            {"city": s.city_name, "fetched_at": now, "payload": weather})

    # Build feature rows from current + forecast items.
    rows = rows_from_pollution_payload(pollution, s.city_name, s.latitude, s.longitude, weather)
    rows += rows_from_pollution_payload(forecast, s.city_name, s.latitude, s.longitude, weather)

    # Pull recent stored features so lag/rolling features have history.
    history = list(
        db[C.AQI_FEATURES].find({"city": s.city_name}).sort("timestamp", -1).limit(72)
    )
    combined = pd.DataFrame(history + rows).drop_duplicates("feature_id")
    combined = add_timeseries_features(combined)

    # Upsert (dedup on feature_id).
    ops = [
        UpdateOne({"feature_id": r["feature_id"]}, {"$set": r}, upsert=True)
        for r in combined.to_dict(orient="records")
    ]
    if ops:
        res = db[C.AQI_FEATURES].bulk_write(ops)
        logger.info("Upserted features: matched=%s upserts=%s",
                    res.matched_count, len(res.upserted_ids))
    return len(ops)


if __name__ == "__main__":
    count = run()
    print(f"Feature pipeline complete. Rows processed: {count}")
