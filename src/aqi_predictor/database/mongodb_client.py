"""Thin MongoDB connection helper using PyMongo."""
from __future__ import annotations

from functools import lru_cache

from pymongo import MongoClient
from pymongo.database import Database

from aqi_predictor.config import get_settings
from aqi_predictor.logging_config import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def get_client() -> MongoClient:
    """Single shared client (cached). Atlas connection string comes from env."""
    settings = get_settings()
    logger.info("Connecting to MongoDB Atlas...")
    return MongoClient(settings.mongodb_uri, serverSelectionTimeoutMS=20000)


def get_db() -> Database:
    settings = get_settings()
    return get_client()[settings.mongodb_database]


def ping() -> bool:
    """Return True if the server responds. Used by /health."""
    try:
        get_client().admin.command("ping")
        return True
    except Exception as exc:  # noqa: BLE001
        logger.error("MongoDB ping failed: %s", exc)
        return False
