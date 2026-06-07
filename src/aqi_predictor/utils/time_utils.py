"""Time helpers. We standardise on timezone-aware UTC timestamps and a single
canonical ISO string format used to build feature_ids."""
from __future__ import annotations

from datetime import datetime, timezone


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def to_iso_hour(dt: datetime) -> str:
    """Canonical hour-resolution ISO timestamp, e.g. '2026-06-05T10:00:00Z'.
    Truncating to the hour means one feature row per city per hour."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    dt = dt.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)
    return dt.strftime("%Y-%m-%dT%H:00:00Z")


def from_unix(ts: int) -> datetime:
    """OpenWeather returns unix seconds; convert to aware UTC datetime."""
    return datetime.fromtimestamp(ts, tz=timezone.utc)


def make_feature_id(city: str, dt: datetime) -> str:
    """Deterministic id like 'Karachi_2026-06-05T10:00:00Z'. Same city+hour
    always yields the same id, which lets us upsert without duplicates."""
    return f"{city}_{to_iso_hour(dt)}"
