"""
Create all collections and indexes. Run once after setting up Atlas:

    python -m aqi_predictor.database.indexes

Indexes:
  - aqi_features.feature_id        UNIQUE  (dedup feature rows)
  - aqi_features.(city, timestamp)         (time-range reads per city)
  - raw_*.(city, fetched_at)               (audit / dedup raw)
  - model_registry.(model_name, model_version, target_horizon) UNIQUE
  - model_registry.is_active               (find active model fast)
  - predictions.created_at / alerts.created_at
"""
from __future__ import annotations

from pymongo import ASCENDING, DESCENDING

from aqi_predictor.database import collections as C
from aqi_predictor.database.mongodb_client import get_db
from aqi_predictor.logging_config import get_logger

logger = get_logger(__name__)


def create_indexes() -> None:
    db = get_db()

    db[C.AQI_FEATURES].create_index(
        [("feature_id", ASCENDING)], unique=True, name="uniq_feature_id"
    )
    db[C.AQI_FEATURES].create_index(
        [("city", ASCENDING), ("timestamp", ASCENDING)], name="city_timestamp"
    )

    for raw in (C.RAW_WEATHER, C.RAW_POLLUTION):
        db[raw].create_index(
            [("city", ASCENDING), ("fetched_at", DESCENDING)], name="city_fetched_at"
        )

    db[C.MODEL_REGISTRY].create_index(
        [("model_name", ASCENDING), ("model_version", ASCENDING),
         ("target_horizon", ASCENDING)],
        unique=True, name="uniq_model_version_horizon",
    )
    db[C.MODEL_REGISTRY].create_index([("is_active", ASCENDING)], name="is_active")
    db[C.MODEL_REGISTRY].create_index([("created_at", DESCENDING)], name="created_at")

    db[C.MODEL_METRICS].create_index([("created_at", DESCENDING)], name="created_at")
    db[C.PREDICTIONS].create_index([("created_at", DESCENDING)], name="created_at")
    db[C.ALERTS].create_index([("created_at", DESCENDING)], name="created_at")

    logger.info("All indexes created/verified.")


if __name__ == "__main__":
    create_indexes()
    print("MongoDB indexes are ready.")
