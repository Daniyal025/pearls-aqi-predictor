"""
OpenWeather API client (Air Pollution + Weather).

Endpoints used:
  - /air_pollution           current pollution + AQI (1-5)
  - /air_pollution/forecast  hourly pollution forecast
  - /air_pollution/history   historical pollution (free tier from 2020-11-27)
  - /weather                 current weather (temp, humidity, pressure, wind)

All methods return parsed JSON dicts. Network errors raise requests
exceptions which callers handle/log.
"""
from __future__ import annotations

from typing import Any

import requests

from aqi_predictor.logging_config import get_logger

logger = get_logger(__name__)

BASE = "https://api.openweathermap.org/data/2.5"


class OpenWeatherClient:
    def __init__(self, api_key: str, timeout: int = 20) -> None:
        if not api_key:
            raise ValueError("OpenWeather API key is required.")
        self.api_key = api_key
        self.timeout = timeout

    def _get(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        params = {**params, "appid": self.api_key}
        url = f"{BASE}/{path}"
        logger.info("GET %s", path)
        resp = requests.get(url, params=params, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def current_pollution(self, lat: float, lon: float) -> dict[str, Any]:
        return self._get("air_pollution", {"lat": lat, "lon": lon})

    def pollution_forecast(self, lat: float, lon: float) -> dict[str, Any]:
        return self._get("air_pollution/forecast", {"lat": lat, "lon": lon})

    def pollution_history(
        self, lat: float, lon: float, start: int, end: int
    ) -> dict[str, Any]:
        """start/end are unix timestamps (seconds)."""
        return self._get(
            "air_pollution/history",
            {"lat": lat, "lon": lon, "start": start, "end": end},
        )

    def current_weather(self, lat: float, lon: float) -> dict[str, Any]:
        return self._get("weather", {"lat": lat, "lon": lon, "units": "metric"})
