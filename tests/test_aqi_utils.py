"""Tests for AQI label/alert helpers. Pure functions, no network."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from aqi_predictor.utils.aqi import aqi_category, alert_level, is_hazardous


def test_categories():
    assert aqi_category(1) == "Good"
    assert aqi_category(3) == "Moderate"
    assert aqi_category(5) == "Very Poor"


def test_category_clamps_and_rounds():
    assert aqi_category(0) == "Good"        # clamped up to 1
    assert aqi_category(9) == "Very Poor"   # clamped down to 5
    assert aqi_category(3.6) == "Poor"      # rounds to 4


def test_alert_levels():
    assert alert_level(2) == "none"
    assert alert_level(3) == "watch"
    assert alert_level(5) == "alert"


def test_is_hazardous():
    assert is_hazardous(4) is True
    assert is_hazardous(2) is False
