"""
SHAP explainability pipeline.

Loads recent feature rows from MongoDB, builds a background sample, and
computes + saves a SHAP feature-importance chart (PNG) for the active model of
each horizon (24h/48h/72h). Also stores the ranked importances in MongoDB so
the dashboard can render them without reading PNGs.

Run:
    python pipelines/explain_pipeline.py
    python pipelines/explain_pipeline.py --horizons 24h
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from aqi_predictor.config import get_settings
from aqi_predictor.database import collections as C
from aqi_predictor.database.mongodb_client import get_db
from aqi_predictor.features.build_features import FEATURE_COLUMNS
from aqi_predictor.logging_config import get_logger
from aqi_predictor.models.explain import explain_horizon
from aqi_predictor.models.registry import get_active_model

logger = get_logger(__name__)

# How many recent rows to use as the SHAP background sample. SHAP is O(rows),
# so we cap this to keep it fast on the free tiers.
BACKGROUND_LIMIT = 200


def _background_matrix(city: str, feats: list[str]) -> np.ndarray:
    """Build an (n_rows x n_features) matrix from recent stored features."""
    db = get_db()
    docs = list(
        db[C.AQI_FEATURES]
        .find({"city": city})
        .sort("timestamp", -1)
        .limit(BACKGROUND_LIMIT)
    )
    if not docs:
        raise RuntimeError(f"No features for {city}. Run the feature pipeline first.")
    rows = [[float(d.get(f, 0.0)) for f in feats] for d in docs]
    return np.array(rows, dtype=float)


def run(horizons: list[str]) -> list[dict]:
    s = get_settings()
    db = get_db()
    out = []

    for horizon in horizons:
        meta = get_active_model(horizon)
        if meta is None:
            logger.warning("No active model for %s; skipping.", horizon)
            continue

        feats = meta.get("features_used") or [
            f for f in FEATURE_COLUMNS
        ]
        try:
            background = _background_matrix(s.city_name, feats)
            result = explain_horizon(background, horizon=horizon)
        except RuntimeError as exc:
            logger.warning("Explain skipped for %s: %s", horizon, exc)
            continue

        # Persist ranked importances for the dashboard.
        db[C.MODEL_METRICS].update_one(
            {"target_horizon": horizon, "model_version": meta.get("model_version")},
            {"$set": {
                "shap_importance": result["importance"][:20],
                "shap_plot_path": result["plot_path"],
                "shap_updated_at": datetime.now(timezone.utc),
            }},
            upsert=True,
        )
        logger.info("SHAP done for %s -> %s", horizon, result["plot_path"])
        out.append(result)

    return out


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate SHAP importance charts.")
    p.add_argument(
        "--horizons", nargs="+", default=["24h", "48h", "72h"],
        help="Which horizons to explain (default: all three).",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    results = run(args.horizons)
    print(f"SHAP explainability complete. Charts generated: {len(results)}")
    for r in results:
        print(f"  {r['horizon']}: {r['plot_path']}")
