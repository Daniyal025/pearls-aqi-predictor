"""Tests for feature engineering. Uses a synthetic pollution payload."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd

from aqi_predictor.features.build_features import (
    FEATURE_COLUMNS,
    add_targets,
    add_timeseries_features,
    rows_from_pollution_payload,
)


def _payload(n=80):
    items = []
    base = 1609459200  # 2021-01-01 00:00 UTC
    for i in range(n):
        items.append({
            "dt": base + i * 3600,
            "main": {"aqi": (i % 5) + 1},
            "components": {"co": 200.0, "no": 0.1, "no2": 5.0, "o3": 60.0,
                           "so2": 2.0, "nh3": 1.0, "pm2_5": 15.0, "pm10": 30.0},
        })
    return {"list": items}


def test_rows_have_core_fields():
    rows = rows_from_pollution_payload(_payload(3), "Karachi", 24.86, 67.0)
    assert len(rows) == 3
    r = rows[0]
    for key in ("feature_id", "city", "timestamp", "aqi", "aqi_category", "pm2_5"):
        assert key in r


def test_feature_id_is_deterministic():
    a = rows_from_pollution_payload(_payload(1), "Karachi", 24.86, 67.0)[0]
    b = rows_from_pollution_payload(_payload(1), "Karachi", 24.86, 67.0)[0]
    assert a["feature_id"] == b["feature_id"]


def test_timeseries_and_targets():
    rows = rows_from_pollution_payload(_payload(80), "Karachi", 24.86, 67.0)
    df = add_timeseries_features(pd.DataFrame(rows))
    df = add_targets(df)
    # All declared feature columns should be present.
    for col in FEATURE_COLUMNS:
        assert col in df.columns
    assert "target_aqi_24h" in df.columns
    # Early rows should have a non-null 24h target.
    assert df["target_aqi_24h"].notna().sum() > 0
