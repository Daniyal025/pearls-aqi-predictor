"""
Historical backfill pipeline.

Uses OpenWeather's air_pollution/history endpoint to pull past hourly data,
build features + time-series features + targets, and upsert into aqi_features.

Run (last 30 days):
    python pipelines/backfill_pipeline.py --days 30

Run an explicit range:
    python pipelines/backfill_pipeline.py --start 2026-01-01 --end 2026-02-01
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd
from pymongo import UpdateOne

from aqi_predictor.config import get_settings
from aqi_predictor.data.openweather_client import OpenWeatherClient
from aqi_predictor.database import collections as C
from aqi_predictor.database.mongodb_client import get_db
from aqi_predictor.features.build_features import (
    add_targets,
    add_timeseries_features,
    rows_from_pollution_payload,
)
from aqi_predictor.logging_config import get_logger

logger = get_logger(__name__)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Backfill historical AQI features.")
    p.add_argument("--days", type=int, default=30, help="Days back from now.")
    p.add_argument("--start", type=str, help="Start date YYYY-MM-DD (overrides --days).")
    p.add_argument("--end", type=str, help="End date YYYY-MM-DD.")
    return p.parse_args()


def run(start: datetime, end: datetime) -> int:
    s = get_settings()
    client = OpenWeatherClient(s.openweather_api_key)
    db = get_db()

    start_ts, end_ts = int(start.timestamp()), int(end.timestamp())
    logger.info("Backfilling %s from %s to %s", s.city_name, start.date(), end.date())

    payload = client.pollution_history(s.latitude, s.longitude, start_ts, end_ts)
    rows = rows_from_pollution_payload(payload, s.city_name, s.latitude, s.longitude)
    if not rows:
        logger.warning("No historical rows returned.")
        return 0

    df = pd.DataFrame(rows).drop_duplicates("feature_id")
    df = add_timeseries_features(df)
    df = add_targets(df)

    ops = [
        UpdateOne({"feature_id": r["feature_id"]}, {"$set": r}, upsert=True)
        for r in df.to_dict(orient="records")
    ]
    db[C.AQI_FEATURES].bulk_write(ops)

    # Record dataset metadata.
    db[C.TRAINING_DATASETS].insert_one({
        "city": s.city_name,
        "start": start, "end": end,
        "row_count": len(df),
        "created_at": datetime.now(timezone.utc),
    })
    logger.info("Backfill upserted %s rows.", len(df))
    return len(df)


if __name__ == "__main__":
    args = _parse_args()
    if args.start:
        start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        end = (datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
               if args.end else datetime.now(timezone.utc))
    else:
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=args.days)
    n = run(start, end)
    print(f"Backfill complete. Rows: {n}")
