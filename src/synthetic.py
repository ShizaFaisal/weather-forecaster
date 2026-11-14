"""Synthetic hourly data (Karachi-like climate) for offline development, tests and demos.
NOT used in production – only when you pass --synthetic."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .logs import now_utc


def make_observations(days: int = 60, end: pd.Timestamp | None = None, seed: int = C.SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    end = (end or now_utc()).floor("h")
    idx = pd.date_range(end - pd.Timedelta(hours=24 * days - 1), end, freq="h", name="timestamp")
    n = len(idx)
    h = idx.hour.values
    doy = idx.dayofyear.values
    weather_noise = np.cumsum(rng.normal(0, 0.15, n)) * 0.2
    weather_noise -= np.convolve(weather_noise, np.ones(72) / 72, mode="same")
    temp = 27 + 5 * np.sin(2 * np.pi * (doy - 100) / 365) + 4 * np.sin(2 * np.pi * (h - 9) / 24) + weather_noise + rng.normal(0, .4, n)
    hum = np.clip(65 - 2.0 * (temp - 27) + 8 * np.cos(2 * np.pi * (h - 4) / 24) + rng.normal(0, 4, n), 15, 100)
    wind = np.clip(4.5 + 2 * np.sin(2 * np.pi * (h - 11) / 24) + rng.gamma(2, .6, n) - 1.5, 0.1, None)
    wdir = (230 + 40 * np.sin(2 * np.pi * h / 24) + rng.normal(0, 25, n)) % 360
    uv = np.clip(9 * np.sin(np.pi * (h - 6) / 12), 0, None) * np.clip(1 - 0.2 * rng.random(n), 0, 1)
    rain_event = rng.random(n) < 0.015
    precip = np.where(rain_event, rng.gamma(2, 1.5, n), 0.0)
    aqi = np.clip(120 + 35 * np.cos(2 * np.pi * (h - 8) / 24) - 6 * (wind - 4.5) + np.cumsum(rng.normal(0, 1, n)) * 0.3
                  - np.convolve(np.cumsum(rng.normal(0, 1, n)), np.ones(48) / 48, mode="same") * 0.3 + rng.normal(0, 8, n), 10, 400)
    df = pd.DataFrame({
        "aqi": aqi, "pm2_5": aqi * 0.35, "pm10": aqi * 0.8, "temperature": temp, "humidity": hum,
        "pressure": 1008 + 3 * np.sin(2 * np.pi * h / 12) + rng.normal(0, .8, n),
        "wind_speed": wind, "wind_dir": wdir, "precip_mm": precip, "uv_index": uv}, index=idx)
    return df
