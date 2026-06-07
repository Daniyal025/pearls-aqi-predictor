"""
Turn raw OpenWeather pollution/weather data into model-ready feature rows.

A single feature row = one city at one hour. We build:
  - calendar features (hour, day, month, day_of_week, is_weekend)
  - the 8 pollutant components (co, no, no2, o3, so2, nh3, pm2_5, pm10)
  - aqi (1-5) + aqi_category label
  - optional weather (temp, humidity, pressure, wind_speed)

Time-series features (lags, rolling averages, change rate) and the forecast
targets are computed across a *sorted DataFrame of many rows* -- see
add_timeseries_features() and add_targets().
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from aqi_predictor.utils.aqi import aqi_category
from aqi_predictor.utils.time_utils import from_unix, make_feature_id, to_iso_hour

POLLUTANTS = ["co", "no", "no2", "o3", "so2", "nh3", "pm2_5", "pm10"]


def pollution_item_to_row(
    item: dict[str, Any], city: str, lat: float, lon: float,
    weather: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Convert one OpenWeather pollution 'list' item into a flat feature row."""
    dt = from_unix(item["dt"])
    aqi = int(item["main"]["aqi"])
    comps = item.get("components", {})

    row: dict[str, Any] = {
        "feature_id": make_feature_id(city, dt),
        "city": city,
        "latitude": lat,
        "longitude": lon,
        "timestamp": to_iso_hour(dt),
        "hour": dt.hour,
        "day": dt.day,
        "month": dt.month,
        "day_of_week": dt.weekday(),
        "is_weekend": 1 if dt.weekday() >= 5 else 0,
        "aqi": aqi,
        "aqi_category": aqi_category(aqi),
    }
    for p in POLLUTANTS:
        row[p] = float(comps.get(p, 0.0))

    if weather and "main" in weather:
        m = weather["main"]
        wind = weather.get("wind", {})
        row["temp"] = float(m.get("temp", 0.0))
        row["humidity"] = float(m.get("humidity", 0.0))
        row["pressure"] = float(m.get("pressure", 0.0))
        row["wind_speed"] = float(wind.get("speed", 0.0))

    return row


def rows_from_pollution_payload(
    payload: dict[str, Any], city: str, lat: float, lon: float,
    weather: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Convert every item in a pollution payload's 'list' to feature rows."""
    return [
        pollution_item_to_row(item, city, lat, lon, weather)
        for item in payload.get("list", [])
    ]


def add_timeseries_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lag, rolling-average and change-rate features. Expects rows for a
    single city sorted by timestamp. Returns a new sorted DataFrame."""
    if df.empty:
        return df
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Lags of AQI (previous 1h, 3h, 24h).
    for lag in (1, 3, 24):
        df[f"aqi_lag_{lag}"] = df["aqi"].shift(lag)

    # Rolling averages of AQI (3h, 24h windows).
    for win in (3, 24):
        df[f"aqi_roll_mean_{win}"] = df["aqi"].rolling(win, min_periods=1).mean()

    # Rolling averages for the two most health-relevant pollutants.
    for p in ("pm2_5", "pm10"):
        if p in df.columns:
            df[f"{p}_roll_mean_24"] = df[p].rolling(24, min_periods=1).mean()

    # AQI change rate vs previous hour.
    df["aqi_change_rate"] = df["aqi"].diff().fillna(0.0)

    # Fill lag NaNs (start of series) with the current aqi value.
    for lag in (1, 3, 24):
        df[f"aqi_lag_{lag}"] = df[f"aqi_lag_{lag}"].fillna(df["aqi"])
    return df


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Add target_aqi_24h/48h/72h by shifting AQI backwards in time.

    Data is hourly, so 24h ahead = 24 rows ahead. Rows near the end have no
    future value and get NaN targets -- training drops those rows.
    """
    if df.empty:
        return df
    df = df.sort_values("timestamp").reset_index(drop=True)
    for horizon, steps in (("24h", 24), ("48h", 48), ("72h", 72)):
        df[f"target_aqi_{horizon}"] = df["aqi"].shift(-steps)
    return df


# Feature columns the models train on (must exist at predict time too).
FEATURE_COLUMNS = [
    "hour", "day", "month", "day_of_week", "is_weekend",
    *POLLUTANTS,
    "aqi",
    "aqi_lag_1", "aqi_lag_3", "aqi_lag_24",
    "aqi_roll_mean_3", "aqi_roll_mean_24",
    "pm2_5_roll_mean_24", "pm10_roll_mean_24",
    "aqi_change_rate",
]

TARGET_COLUMNS = ["target_aqi_24h", "target_aqi_48h", "target_aqi_72h"]
