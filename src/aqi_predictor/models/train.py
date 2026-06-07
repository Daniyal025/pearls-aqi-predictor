"""
Train and compare models for each forecast horizon, then register the best.

Models compared (scikit-learn):
  - Ridge Regression
  - Random Forest Regressor
  - Gradient Boosting Regressor
  - Extra Trees Regressor
An optional TensorFlow model is trained only if tensorflow is installed AND
there are enough rows (see TF_MIN_ROWS).

Artifacts are saved with joblib to artifacts/models/. The registry stores the
path and metrics in MongoDB; the best model per horizon is marked active.
"""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

from aqi_predictor.features.build_features import FEATURE_COLUMNS, TARGET_COLUMNS
from aqi_predictor.logging_config import get_logger
from aqi_predictor.models.evaluate import compute_metrics
from aqi_predictor.models.registry import register_model

logger = get_logger(__name__)

ARTIFACT_DIR = Path("artifacts/models")
TF_MIN_ROWS = 500  # only attempt TensorFlow when we have plenty of data


def _candidate_models() -> dict[str, Any]:
    return {
        "ridge": Ridge(alpha=1.0),
        "random_forest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        "gradient_boosting": GradientBoostingRegressor(random_state=42),
        "extra_trees": ExtraTreesRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    }


def _maybe_train_tf(X_train, y_train, X_test, y_test) -> tuple[Any, dict] | None:
    """Train a small Keras MLP if TensorFlow is available. Returns None if not."""
    try:
        import tensorflow as tf  # noqa: F401
        from tensorflow import keras
    except Exception:
        logger.info("TensorFlow not installed; skipping TF model.")
        return None

    model = keras.Sequential([
        keras.layers.Input(shape=(X_train.shape[1],)),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X_train, y_train, epochs=30, batch_size=32, verbose=0)
    preds = model.predict(X_test, verbose=0).ravel()
    return model, compute_metrics(y_test, preds)


def train_for_horizon(df: pd.DataFrame, horizon_col: str, version: str) -> dict[str, Any]:
    """Train all candidates for one target column, register & return the best."""
    horizon = horizon_col.replace("target_aqi_", "")  # '24h'
    data = df.dropna(subset=[horizon_col]).copy()
    feats = [c for c in FEATURE_COLUMNS if c in data.columns]

    if len(data) < 20:
        raise ValueError(
            f"Not enough rows to train ({len(data)}). Run the backfill pipeline "
            f"first to populate historical features."
        )

    X = data[feats].fillna(0.0).values
    y = data[horizon_col].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    results: list[dict[str, Any]] = []
    fitted: dict[str, Any] = {}

    for name, model in _candidate_models().items():
        model.fit(X_train, y_train)
        metrics = compute_metrics(y_test, model.predict(X_test))
        results.append({"model_type": name, **metrics})
        fitted[name] = model
        logger.info("[%s] %s -> RMSE=%.3f MAE=%.3f R2=%.3f",
                    horizon, name, metrics["rmse"], metrics["mae"], metrics["r2"])

    tf_out = None
    if len(data) >= TF_MIN_ROWS:
        tf_out = _maybe_train_tf(X_train, y_train, X_test, y_test)
        if tf_out is not None:
            _, tf_metrics = tf_out
            results.append({"model_type": "tensorflow_mlp", **tf_metrics})
            logger.info("[%s] tensorflow_mlp -> RMSE=%.3f", horizon, tf_metrics["rmse"])

    # Best = lowest RMSE.
    best = min(results, key=lambda r: r["rmse"])
    best_type = best["model_type"]

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    if best_type == "tensorflow_mlp" and tf_out is not None:
        artifact_path = str(ARTIFACT_DIR / f"aqi_{horizon}_{version}_tf.keras")
        tf_out[0].save(artifact_path)
    else:
        artifact_path = str(ARTIFACT_DIR / f"aqi_{horizon}_{version}_{best_type}.joblib")
        joblib.dump({"model": fitted[best_type], "features": feats}, artifact_path)

    register_model(
        model_name=f"aqi_forecaster_{horizon}",
        model_version=version,
        model_type=best_type,
        target_horizon=horizon,
        features_used=feats,
        metrics={k: best[k] for k in ("rmse", "mae", "r2")},
        artifact_path=artifact_path,
        make_active=True,
    )

    # Persist the full comparison table for the dashboard.
    from aqi_predictor.database import collections as C
    from aqi_predictor.database.mongodb_client import get_db
    get_db()[C.MODEL_METRICS].insert_one({
        "target_horizon": horizon,
        "model_version": version,
        "results": results,
        "best_model_type": best_type,
        "created_at": datetime.utcnow(),
    })

    return {"horizon": horizon, "best_model_type": best_type,
            "artifact_path": artifact_path, "results": results}


def train_all(df: pd.DataFrame, version: str | None = None) -> list[dict[str, Any]]:
    version = version or datetime.utcnow().strftime("%Y%m%d%H%M%S")
    out = []
    for target in TARGET_COLUMNS:
        if target in df.columns:
            out.append(train_for_horizon(df, target, version))
    return out
