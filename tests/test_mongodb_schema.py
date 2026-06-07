"""
Schema-shape tests that do NOT require a live MongoDB connection.
We validate collection-name constants and the registry document shape.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from aqi_predictor.database import collections as C


def test_all_collections_present():
    expected = {
        "raw_weather_data", "raw_pollution_data", "aqi_features",
        "training_datasets", "model_registry", "model_metrics",
        "predictions", "alerts",
    }
    assert set(C.ALL_COLLECTIONS) == expected


def test_collection_names_are_strings():
    assert all(isinstance(name, str) and name for name in C.ALL_COLLECTIONS)
