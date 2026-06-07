"""
Load active models from the registry and produce 24h/48h/72h forecasts from a
single latest-feature row.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from aqi_predictor.logging_config import get_logger
from aqi_predictor.models.registry import get_active_model
from aqi_predictor.utils.aqi import aqi_category, alert_level

logger = get_logger(__name__)


def _ensure_local_artifact(meta: dict[str, Any]) -> str | None:
    """Return a local path to the artifact, downloading from GridFS if the
    recorded local file is not present on this machine (the deployment case).

    Returns None if neither a local file nor a GridFS copy is available.
    """
    path = meta.get("artifact_path")
    # 1. If the local file exists, use it directly (fast same-machine path).
    if path and os.path.exists(path):
        return path

    # 2. Otherwise try GridFS using the stored file id.
    file_id = meta.get("gridfs_file_id")
    if not file_id:
        return None

    # Download into a stable local cache so repeated predictions reuse it.
    cache_dir = Path("artifacts/models")
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Use the original filename if we can recover it, else the file id.
    fname = Path(path).name if path else f"{file_id}.joblib"
    dest = cache_dir / fname
    if dest.exists():
        return str(dest)

    from aqi_predictor.database.gridfs_store import download_artifact
    logger.info("Artifact not local; downloading from GridFS (%s).", file_id)
    return download_artifact(file_id, str(dest))


def _load_artifact(meta: dict[str, Any]):
    """Load a model artifact described by a registry doc. Supports joblib and
    (optionally) Keras. Falls back to GridFS when the local file is absent."""
    local_path = _ensure_local_artifact(meta)
    if local_path is None:
        raise RuntimeError(
            "Model artifact not found locally or in GridFS. Re-run the training "
            "pipeline (with GridFS enabled) so the model is available to all machines."
        )
    if local_path.endswith(".keras"):
        from tensorflow import keras
        model = keras.models.load_model(local_path)
        return model, meta.get("features_used", []), "keras"
    bundle = joblib.load(local_path)
    return bundle["model"], bundle["features"], "sklearn"


def predict_horizon(feature_row: dict[str, Any], horizon: str) -> dict[str, Any]:
    """Predict one horizon ('24h'/'48h'/'72h') from a feature row dict."""
    meta = get_active_model(horizon)
    if meta is None:
        raise RuntimeError(
            f"No active model for horizon {horizon}. Run the training pipeline."
        )
    model, feats, kind = _load_artifact(meta)
    X = np.array([[float(feature_row.get(f, 0.0)) for f in feats]], dtype=float)
    raw = model.predict(X)
    value = float(np.asarray(raw).ravel()[0])
    aqi = int(round(max(1, min(5, value))))
    return {
        "horizon": horizon,
        "predicted_aqi": aqi,
        "predicted_aqi_raw": round(value, 3),
        "aqi_category": aqi_category(aqi),
        "alert_level": alert_level(aqi),
        "model_type": meta.get("model_type"),
        "model_version": meta.get("model_version"),
    }


def predict_all(feature_row: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for horizon in ("24h", "48h", "72h"):
        try:
            out.append(predict_horizon(feature_row, horizon))
        except RuntimeError as exc:
            logger.warning(str(exc))
    return out
