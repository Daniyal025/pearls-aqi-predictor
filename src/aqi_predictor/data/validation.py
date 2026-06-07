"""Light validation of OpenWeather responses before we trust them."""
from __future__ import annotations

from typing import Any

from aqi_predictor.logging_config import get_logger

logger = get_logger(__name__)


def validate_pollution_response(payload: dict[str, Any]) -> bool:
    """A valid pollution payload has a non-empty 'list' whose items contain
    'main.aqi' and 'components'."""
    items = payload.get("list")
    if not items:
        logger.warning("Pollution payload has no 'list' entries.")
        return False
    first = items[0]
    if "main" not in first or "aqi" not in first["main"]:
        logger.warning("Pollution item missing main.aqi.")
        return False
    if "components" not in first:
        logger.warning("Pollution item missing components.")
        return False
    return True


def validate_weather_response(payload: dict[str, Any]) -> bool:
    if "main" not in payload:
        logger.warning("Weather payload missing 'main'.")
        return False
    return True
