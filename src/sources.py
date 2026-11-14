"""API clients. Each fetcher returns a DataFrame indexed by hourly UTC timestamp using the
canonical column names (aqi, pm2_5, pm10, temperature, humidity, pressure, wind_speed,
wind_dir, precip_mm, uv_index) or None when the source is down."""
from __future__ import annotations

import os
import time
from datetime import date, datetime, timedelta, timezone

import pandas as pd
import requests

from . import config as C
from .logs import log, now_utc

# ------------------------------------------------------------------ helpers

def _get(url: str, params: dict, source: str, retries: int = 3, timeout: int = 25):
    last = None
    for i in range(retries):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code == 200:
                return r.json()
            last = f"HTTP {r.status_code}: {r.text[:150]}"
        except requests.RequestException as e:
            last = str(e)
        time.sleep(2 ** i)
    print(f"[sources] {source} failed: {last}")
    log("failure_log", source=source, kind="data_source", agent="ingestion", error=last, prompt="", model="")
    return None


def _floor_now() -> pd.Timestamp:
    return now_utc().floor("h")


def pm25_to_aqi(pm: float) -> float:
    """US EPA AQI from PM2.5 (µg/m³), 2024 breakpoints."""
    if pm is None or pd.isna(pm):
        return float("nan")
    pm = int(max(pm, 0) * 10) / 10
    bps = [(0, 9.0, 0, 50), (9.1, 35.4, 51, 100), (35.5, 55.4, 101, 150),
           (55.5, 125.4, 151, 200), (125.5, 225.4, 201, 300), (225.5, 500.4, 301, 500)]
    for lo, hi, alo, ahi in bps:
        if pm <= hi:
            return round((ahi - alo) / (hi - lo) * (pm - lo) + alo)
    return 500.0


# ------------------------------------------------------------------ OpenWeather (current)

def parse_openweather(weather: dict | None, pollution: dict | None) -> pd.DataFrame | None:
    row = {}
    if weather:
        row.update(
            temperature=weather.get("main", {}).get("temp"),
            humidity=weather.get("main", {}).get("humidity"),
            pressure=weather.get("main", {}).get("pressure"),
            wind_speed=weather.get("wind", {}).get("speed"),
            wind_dir=weather.get("wind", {}).get("deg"),
            precip_mm=(weather.get("rain", {}) or {}).get("1h", 0.0) + (weather.get("snow", {}) or {}).get("1h", 0.0),
        )
    if pollution and pollution.get("list"):
        comp = pollution["list"][0].get("components", {})
        row.update(pm2_5=comp.get("pm2_5"), pm10=comp.get("pm10"), aqi=pm25_to_aqi(comp.get("pm2_5")))
    if not row:
        return None
    return pd.DataFrame([row], index=pd.DatetimeIndex([_floor_now()], name="timestamp"))


def fetch_openweather() -> pd.DataFrame | None:
    key = os.getenv("OPENWEATHER_API_KEY")
    if not key:
        return None
    p = {"lat": C.LAT, "lon": C.LON, "appid": key, "units": "metric"}
    w = _get("https://api.openweathermap.org/data/2.5/weather", p, "openweather")
    a = _get("https://api.openweathermap.org/data/2.5/air_pollution", {k: p[k] for k in ("lat", "lon", "appid")}, "openweather")
    return parse_openweather(w, a)


# ------------------------------------------------------------------ AQICN (current)

def parse_aqicn(js: dict | None) -> pd.DataFrame | None:
    if not js or js.get("status") != "ok":
        return None
    d = js["data"]
    iaqi = d.get("iaqi", {})
    aqi = d.get("aqi")
    row = {"aqi": float(aqi) if str(aqi).replace(".", "", 1).isdigit() else None,
           "pm2_5": iaqi.get("pm25", {}).get("v"), "pm10": iaqi.get("pm10", {}).get("v")}
    return pd.DataFrame([row], index=pd.DatetimeIndex([_floor_now()], name="timestamp"))


def fetch_aqicn() -> pd.DataFrame | None:
    token = os.getenv("AQICN_API_TOKEN")
    if not token:
        return None
    js = _get(f"https://api.waqi.info/feed/geo:{C.LAT};{C.LON}/", {"token": token}, "aqicn")
    return parse_aqicn(js)


# ------------------------------------------------------------------ Open-Meteo (hourly, history + recent + UV)

OM_WEATHER = {"temperature_2m": "temperature", "relative_humidity_2m": "humidity",
              "surface_pressure": "pressure", "precipitation": "precip_mm",
              "wind_speed_10m": "wind_speed", "wind_direction_10m": "wind_dir"}
OM_AQ = {"us_aqi": "aqi", "pm2_5": "pm2_5", "pm10": "pm10", "uv_index": "uv_index"}


def parse_open_meteo_hourly(js: dict | None, mapping: dict) -> pd.DataFrame | None:
    if not js or "hourly" not in js:
        return None
    h = js["hourly"]
    df = pd.DataFrame({new: h.get(old) for old, new in mapping.items() if old in h})
    df.index = pd.DatetimeIndex(pd.to_datetime(h["time"]), name="timestamp")
    return df


def fetch_open_meteo(start: date, end: date) -> pd.DataFrame | None:
    """Hourly observations/analysis for [start, end] (inclusive). Uses the archive API for old
    dates and the forecast API (past 90 days) for recent ones; air quality incl. UV from the CAMS API."""
    today = datetime.now(timezone.utc).date()
    parts = []
    split = today - timedelta(days=7)
    common = {"latitude": C.LAT, "longitude": C.LON, "timezone": "UTC", "wind_speed_unit": "ms"}
    wvars = ",".join(OM_WEATHER)
    if start <= split:
        js = _get("https://archive-api.open-meteo.com/v1/archive",
                  {**common, "start_date": start.isoformat(), "end_date": min(end, split).isoformat(), "hourly": wvars},
                  "openmeteo")
        parts.append(parse_open_meteo_hourly(js, OM_WEATHER))
    if end > split:
        s = max(start, split + timedelta(days=1))
        js = _get("https://api.open-meteo.com/v1/forecast",
                  {**common, "start_date": s.isoformat(), "end_date": end.isoformat(), "hourly": wvars}, "openmeteo")
        parts.append(parse_open_meteo_hourly(js, OM_WEATHER))
    weather = pd.concat([p for p in parts if p is not None]) if any(p is not None for p in parts) else None

    js = _get("https://air-quality-api.open-meteo.com/v1/air-quality",
              {"latitude": C.LAT, "longitude": C.LON, "timezone": "UTC", "start_date": start.isoformat(),
               "end_date": end.isoformat(), "hourly": ",".join(OM_AQ)}, "openmeteo")
    aq = parse_open_meteo_hourly(js, OM_AQ)

    frames = [f for f in (weather, aq) if f is not None]
    if not frames:
        return None
    out = pd.concat(frames, axis=1)
    out = out.loc[~out.index.duplicated(keep="last")].sort_index()
    return out[out.index <= _floor_now()]


def fetch_recent_open_meteo(days: int = 2) -> pd.DataFrame | None:
    today = datetime.now(timezone.utc).date()
    return fetch_open_meteo(today - timedelta(days=days), today)
