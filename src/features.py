"""Time-based + derived features and 24 h-ahead targets."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

ROLL_COLS = C.BASE_COLS + ["pm2_5", "pressure"]
META_COLS = {"city", "timestamp", "n_imputed", "n_sources"}


def build_features(obs: pd.DataFrame, with_targets: bool = True) -> pd.DataFrame:
    """obs: hourly-regular, indexed by timestamp, already imputed. Returns a frame with the same index."""
    out = obs.copy()
    idx = out.index
    out["hour"], out["dayofweek"], out["month"] = idx.hour, idx.dayofweek, idx.month
    out["hour_sin"], out["hour_cos"] = np.sin(2 * np.pi * idx.hour / 24), np.cos(2 * np.pi * idx.hour / 24)
    out["month_sin"], out["month_cos"] = np.sin(2 * np.pi * idx.month / 12), np.cos(2 * np.pi * idx.month / 12)
    new = {}
    for c in ROLL_COLS:
        s = out[c]
        new[f"{c}_diff_1h"] = s.diff(1)                       # rate of change
        new[f"{c}_diff_3h"] = s.diff(3)
        new[f"{c}_roll_mean_6h"] = s.rolling(6, min_periods=1).mean()
        new[f"{c}_roll_mean_24h"] = s.rolling(24, min_periods=1).mean()
        new[f"{c}_roll_std_24h"] = s.rolling(24, min_periods=2).std()
        new[f"{c}_lag_24h"] = s.shift(24)
    out = pd.concat([out, pd.DataFrame(new, index=idx)], axis=1)
    diff_cols = [c for c in out if c.endswith(("_diff_1h", "_diff_3h", "_roll_std_24h"))]
    out[diff_cols] = out[diff_cols].fillna(0.0)
    if with_targets:
        for c in C.BASE_COLS:
            out[f"target_{c}"] = obs[c].shift(-C.HORIZON)    # NaN for the newest 24 h
    return out


def feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in META_COLS and not c.startswith("target_")
            and c not in ("hour", "dayofweek", "month")]


def to_store_frame(feat: pd.DataFrame) -> pd.DataFrame:
    df = feat.reset_index()
    df.insert(0, "city", C.CITY)
    return df


def from_store_frame(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=["city"], errors="ignore").sort_values("timestamp").set_index("timestamp")


def obs_columns() -> list[str]:
    return C.BASE_COLS + C.EXTRA_OBS_COLS
