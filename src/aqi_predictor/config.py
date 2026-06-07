"""
Central configuration. Loads environment variables from a local .env file
(via python-dotenv) when present, and otherwise reads the real process
environment (this is how GitHub Actions / Streamlit / Render inject secrets).

Nothing here is hardcoded. If a required secret is missing we fail loudly
with a clear message instead of silently using a bad default.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

# Load .env if it exists. In CI/cloud there is no .env and that is fine.
load_dotenv()


def _get(name: str, default: str | None = None, required: bool = False) -> str:
    value = os.getenv(name, default)
    if required and (value is None or value == ""):
        raise RuntimeError(
            f"Missing required environment variable: {name}. "
            f"Set it in your .env file (local) or as a secret (CI/cloud)."
        )
    return value if value is not None else ""


@dataclass(frozen=True)
class Settings:
    openweather_api_key: str
    mongodb_uri: str
    mongodb_database: str
    city_name: str
    latitude: float
    longitude: float
    timezone: str
    api_base_url: str


def get_settings(require_secrets: bool = True) -> Settings:
    """Build a Settings object. Set require_secrets=False for offline unit
    tests that should not need real credentials."""
    return Settings(
        openweather_api_key=_get("OPENWEATHER_API_KEY", required=require_secrets),
        mongodb_uri=_get("MONGODB_URI", required=require_secrets),
        mongodb_database=_get("MONGODB_DATABASE", default="pearls_aqi"),
        city_name=_get("CITY_NAME", default="Karachi"),
        latitude=float(_get("LATITUDE", default="24.8607")),
        longitude=float(_get("LONGITUDE", default="67.0011")),
        timezone=_get("TIMEZONE", default="Asia/Karachi"),
        api_base_url=_get("API_BASE_URL", default="http://127.0.0.1:8000"),
    )
