"""Exploratory data analysis for every metric (used by the dashboard and `python -m scripts.eda`)."""
from __future__ import annotations

import numpy as np
import pandas as pd

EDA_COLS = {"aqi": "AQI", "temperature": "Temperature (°C)", "precip_mm": "Precipitation (mm/h)",
            "rain_flag": "Rain occurrence", "humidity": "Humidity (%)", "uv_index": "UV Index",
            "wind_speed": "Wind speed (m/s)", "wind_dir": "Wind direction (°)"}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["wind_dir"] = np.degrees(np.arctan2(d["wind_dir_sin"], d["wind_dir_cos"])) % 360
    return d


def acf(s: pd.Series, nlags: int = 72) -> np.ndarray:
    s = s.astype(float) - s.mean()
    den = float((s ** 2).sum()) or 1.0
    return np.array([float((s.iloc[k:].values * s.iloc[:len(s) - k].values).sum() / den) for k in range(nlags + 1)])


def profile(d: pd.DataFrame, col: str) -> dict:
    s = d[col].astype(float)
    q1, q3 = s.quantile([.25, .75])
    iqr = q3 - q1
    out_pct = float(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).mean() * 100) if iqr > 0 else 0.0
    a = acf(s, 48)
    p = {"count": int(s.count()), "mean": s.mean(), "std": s.std(), "min": s.min(), "p25": q1, "median": s.median(),
         "p75": q3, "max": s.max(), "skew": s.skew(), "outliers_%": out_pct,
         "acf_lag_1h": a[1], "acf_lag_24h": a[24], "acf_lag_48h": a[48]}
    if "n_imputed" in d:
        p["rows_with_imputation_%"] = float((d["n_imputed"] > 0).mean() * 100)
    try:
        from statsmodels.tsa.stattools import adfuller
        p["adf_p_value"] = float(adfuller(s.dropna().iloc[-2000:], autolag="AIC")[1])
    except Exception:
        pass
    return p


def hourly_profile(d, col): return d.groupby(d.index.hour)[col].agg(["mean", "std", "min", "max"])
def monthly_profile(d, col): return d.groupby(d.index.month)[col].mean()


def correlation(d: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in EDA_COLS if c in d] + ["pm2_5", "pressure"]
    return d[cols].corr()
