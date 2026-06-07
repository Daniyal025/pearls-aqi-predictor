"""
AQI helpers.

IMPORTANT: OpenWeather's Air Pollution API returns AQI on a 1-5 scale, which
is NOT the US 0-500 AQI scale. We keep the 1-5 scale and label it honestly.

  1 = Good
  2 = Fair
  3 = Moderate
  4 = Poor
  5 = Very Poor

Alert levels are derived from this 1-5 category.
"""
from __future__ import annotations

# OpenWeather 1-5 categories.
AQI_CATEGORIES: dict[int, str] = {
    1: "Good",
    2: "Fair",
    3: "Moderate",
    4: "Poor",
    5: "Very Poor",
}

# Alert level per category. Used by the API and dashboard alert cards.
ALERT_LEVELS: dict[int, str] = {
    1: "none",
    2: "none",
    3: "watch",
    4: "alert",
    5: "alert",
}


def aqi_category(aqi: int | float | None) -> str:
    """Map a 1-5 AQI value to its label. Rounds floats (model outputs)."""
    if aqi is None:
        return "Unknown"
    key = int(round(float(aqi)))
    key = max(1, min(5, key))  # clamp into the valid 1-5 range
    return AQI_CATEGORIES[key]


def alert_level(aqi: int | float | None) -> str:
    """Return 'none', 'watch', or 'alert' for a 1-5 AQI value."""
    if aqi is None:
        return "none"
    key = int(round(float(aqi)))
    key = max(1, min(5, key))
    return ALERT_LEVELS[key]


def is_hazardous(aqi: int | float | None) -> bool:
    """True for Poor (4) or Very Poor (5) -- worth logging an alert."""
    if aqi is None:
        return False
    return int(round(float(aqi))) >= 4
