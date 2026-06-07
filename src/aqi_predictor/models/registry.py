"""
Custom MongoDB-based model registry.

Each registry document describes ONE trained model for ONE target horizon:
  model_name, model_version, model_type, target_horizon, artifact_path,
  gridfs_file_id, features_used, rmse, mae, r2, is_active, trained_at, created_at

register_model() inserts a doc and (optionally) flips is_active so only the
newest best model per horizon is active.
"""
from __future__ import annotations

from datetime import timezone
from typing import Any

from aqi_predictor.database import collections as C
from aqi_predictor.database.mongodb_client import get_db
from aqi_predictor.logging_config import get_logger
from aqi_predictor.utils.time_utils import utc_now

logger = get_logger(__name__)


def register_model(
    *, model_name: str, model_version: str, model_type: str,
    target_horizon: str, features_used: list[str], metrics: dict[str, float],
    artifact_path: str | None = None, gridfs_file_id: str | None = None,
    make_active: bool = True,
) -> dict[str, Any]:
    db = get_db()
    now = utc_now()
    doc = {
        "model_name": model_name,
        "model_version": model_version,
        "model_type": model_type,
        "target_horizon": target_horizon,
        "features_used": features_used,
        "artifact_path": artifact_path,
        "gridfs_file_id": gridfs_file_id,
        "rmse": metrics.get("rmse"),
        "mae": metrics.get("mae"),
        "r2": metrics.get("r2"),
        "is_active": make_active,
        "trained_at": now,
        "created_at": now,
    }

    if make_active:
        # Deactivate previously-active models for this horizon.
        db[C.MODEL_REGISTRY].update_many(
            {"target_horizon": target_horizon, "is_active": True},
            {"$set": {"is_active": False}},
        )

    db[C.MODEL_REGISTRY].update_one(
        {"model_name": model_name, "model_version": model_version,
         "target_horizon": target_horizon},
        {"$set": doc},
        upsert=True,
    )
    logger.info("Registered %s v%s (%s) active=%s",
                model_name, model_version, target_horizon, make_active)
    return doc


def get_active_model(target_horizon: str) -> dict[str, Any] | None:
    return get_db()[C.MODEL_REGISTRY].find_one(
        {"target_horizon": target_horizon, "is_active": True}
    )


def get_all_active_models() -> list[dict[str, Any]]:
    return list(get_db()[C.MODEL_REGISTRY].find({"is_active": True}))
