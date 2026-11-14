"""Fuse per-source frames into one record per timestamp and impute gaps.

Fusion strategy (config.FUSION): continuous weather variables are averaged across the sources that
report them; AQI/PM follow a priority list (AQICN station > OpenWeather > Open-Meteo/CAMS model);
wind direction (circular) and UV use priority. Open-Meteo is the only hourly-history source, so
backfilled rows come from Open-Meteo alone and live rows blend all three.

Imputation strategy (documented in docs/REPORT.md):
  1. time-linear interpolation across gaps of up to 6 h,
  2. forward fill up to 3 more hours at the leading edge,
  3. hour-of-day median of the available window,
  4. global median of the column.
`n_imputed` records how many columns of a row were filled so quality can be audited."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

CANON = ["aqi", "pm2_5", "pm10", "temperature", "humidity", "pressure",
         "wind_speed", "wind_dir", "precip_mm", "uv_index"]


def fuse(raw: dict[str, pd.DataFrame | None]) -> pd.DataFrame:
    raw = {k: v for k, v in raw.items() if v is not None and not v.empty}
    if not raw:
        return pd.DataFrame()
    idx = sorted(set().union(*[set(v.index) for v in raw.values()]))
    out = pd.DataFrame(index=pd.DatetimeIndex(idx, name="timestamp"))
    for col in CANON:
        strategy, order = C.FUSION[col]
        cols = {s: raw[s][col].reindex(out.index) for s in order if s in raw and col in raw[s]}
        if not cols:
            out[col] = np.nan
            continue
        stack = pd.DataFrame(cols)
        out[col] = stack.mean(axis=1) if strategy == "mean" else stack.bfill(axis=1).iloc[:, 0]
    out["n_sources"] = pd.DataFrame({s: raw[s].reindex(out.index).notna().any(axis=1) for s in raw}).sum(axis=1)
    return out


def add_wind_components(df: pd.DataFrame) -> pd.DataFrame:
    rad = np.deg2rad(df["wind_dir"])
    df["wind_dir_sin"], df["wind_dir_cos"] = np.sin(rad), np.cos(rad)
    return df.drop(columns=["wind_dir"])


INPUT_COLS = ["aqi", "pm2_5", "pm10", "temperature", "humidity", "pressure",
              "wind_speed", "wind_dir_sin", "wind_dir_cos", "precip_mm", "uv_index"]
BOUNDS = {"aqi": (0, 500), "pm2_5": (0, 1000), "pm10": (0, 1500), "humidity": (0, 100),
          "uv_index": (0, 16), "precip_mm": (0, 500), "wind_speed": (0, 80), "pressure": (850, 1100),
          "temperature": (-60, 60), "wind_dir_sin": (-1, 1), "wind_dir_cos": (-1, 1)}


def impute(df: pd.DataFrame) -> pd.DataFrame:
    """Regularise to an hourly grid and fill gaps. Input/Output indexed by timestamp."""
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    grid = pd.date_range(df.index.min(), df.index.max(), freq="h", name="timestamp")
    df = df.reindex(grid)
    for c in INPUT_COLS:
        if c not in df:
            df[c] = np.nan
    miss = df[INPUT_COLS].isna()
    out = df.copy()
    for c in INPUT_COLS:
        s = out[c].astype(float)
        lo, hi = BOUNDS[c]
        s = s.where((s >= lo) & (s <= hi))            # out-of-range sensor values -> missing
        s = s.interpolate(method="time", limit=6, limit_area="inside")
        s = s.ffill(limit=3)
        s = s.fillna(s.groupby(s.index.hour).transform("median"))
        s = s.fillna(s.median())
        out[c] = s.fillna(0.0)
    out["n_imputed"] = (miss | df[INPUT_COLS].isna()).sum(axis=1).astype(int)
    out["rain_flag"] = (out["precip_mm"] > C.RAIN_THRESHOLD_MM).astype(float)
    return out
