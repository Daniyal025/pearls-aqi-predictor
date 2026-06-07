"""
Load active models from the registry and produce 24h/48h/72h forecasts from a
single latest-feature row.
"""
from __future__ import annotations

from typing import Any

import joblib
import numpy as np

from aqi_predictor.logging_config import get_logger
from aqi_predictor.models.registry import get_active_model
from aqi_predictor.utils.aqi import aqi_category, alert_level

logger = get_logger(__name__)


def _load_artifact(meta: dict[str, Any]):
    """Load a model artifact described by a registry doc. Supports joblib and
    (optionally) Keras."""
    path = meta.get("artifact_path")
    if path and path.endswith(".keras"):
        from tensorflow import keras
        model = keras.models.load_model(path)
        return model, meta.get("features_used", []), "keras"
    bundle = joblib.load(path)
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
