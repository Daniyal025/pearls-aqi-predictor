"""
AQI alert logic.

Centralises "should we raise an alert and log it?" so the API, batch pipeline,
and dashboard all behave the same way.

Design choices:
  - We alert when predicted AQI is Poor (4) or Very Poor (5).
  - We DEDUPLICATE: one alert per (city, horizon, severity) per calendar day,
    so an hourly job doesn't write the same alert 24 times.
  - Each alert stores a human-readable message and a severity string.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from aqi_predictor.database import collections as C
from aqi_predictor.database.mongodb_client import get_db
from aqi_predictor.logging_config import get_logger
from aqi_predictor.utils.aqi import aqi_category, is_hazardous

logger = get_logger(__name__)

# Severity label per AQI level (only 4 and 5 trigger alerts).
SEVERITY = {4: "poor", 5: "very_poor"}

# Short health guidance per category, shown on dashboard alert cards.
GUIDANCE = {
    "Poor": "Sensitive groups should limit prolonged outdoor exertion.",
    "Very Poor": "Everyone should avoid outdoor exertion; wear a mask if outside.",
}


def _alert_key(city: str, horizon: str, severity: str, day: str) -> str:
    """Deterministic key so the same alert on the same day is written once."""
    return f"{city}_{horizon}_{severity}_{day}"


def maybe_log_alert(
    city: str, horizon: str, predicted_aqi: int | float,
    *, now: datetime | None = None,
) -> dict[str, Any] | None:
    """Log an alert if the prediction is hazardous and not already logged today.
    Returns the alert doc if written, else None."""
    if not is_hazardous(predicted_aqi):
        return None

    now = now or datetime.now(timezone.utc)
    level = int(round(max(1, min(5, float(predicted_aqi)))))
    severity = SEVERITY[level]
    category = aqi_category(level)
    day = now.strftime("%Y-%m-%d")
    key = _alert_key(city, horizon, severity, day)

    doc = {
        "alert_key": key,
        "city": city,
        "horizon": horizon,
        "predicted_aqi": level,
        "aqi_category": category,
        "severity": severity,
        "message": f"{category} air quality expected in {horizon} for {city}.",
        "guidance": GUIDANCE.get(category, ""),
        "created_at": now,
    }

    db = get_db()
    # upsert on alert_key => one row per city/horizon/severity/day.
    res = db[C.ALERTS].update_one(
        {"alert_key": key}, {"$setOnInsert": doc}, upsert=True
    )
    if res.upserted_id is not None:
        logger.warning("ALERT logged: %s", doc["message"])
        return doc
    return None  # already existed today


def recent_alerts(city: str, limit: int = 10) -> list[dict[str, Any]]:
    """Fetch recent alerts for the dashboard."""
    docs = list(
        get_db()[C.ALERTS].find({"city": city}).sort("created_at", -1).limit(limit)
    )
    for d in docs:
        d.pop("_id", None)
    return docs
