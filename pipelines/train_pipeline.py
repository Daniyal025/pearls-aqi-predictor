"""
Daily training pipeline.

  1. Load feature rows (with targets) from MongoDB.
  2. Train + compare models per horizon, register best, mark active.

Run:
    python pipelines/train_pipeline.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd

from aqi_predictor.config import get_settings
from aqi_predictor.database import collections as C
from aqi_predictor.database.mongodb_client import get_db
from aqi_predictor.features.build_features import add_targets
from aqi_predictor.logging_config import get_logger
from aqi_predictor.models.train import train_all

logger = get_logger(__name__)


def run() -> list:
    s = get_settings()
    db = get_db()
    docs = list(db[C.AQI_FEATURES].find({"city": s.city_name}))
    if not docs:
        logger.error("No features found. Run backfill_pipeline.py first.")
        return []

    df = pd.DataFrame(docs).drop(columns=["_id"], errors="ignore")
    # Ensure targets exist (backfill adds them; recompute to be safe).
    if not any(c.startswith("target_aqi_") for c in df.columns):
        df = add_targets(df)

    results = train_all(df)
    for r in results:
        logger.info("Best for %s: %s", r["horizon"], r["best_model_type"])
    return results


if __name__ == "__main__":
    out = run()
    print(f"Training complete. Horizons trained: {len(out)}")
