from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


def rmse(y, p): return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))
def mae(y, p): return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))


def r2(y, p):
    y, p = np.asarray(y), np.asarray(p)
    d = np.sum((y - y.mean()) ** 2)
    return float(1 - np.sum((y - p) ** 2) / d) if d > 0 else float("nan")


def angular_mae(true_deg, pred_deg):
    d = np.abs((np.asarray(true_deg) - np.asarray(pred_deg) + 180) % 360 - 180)
    return float(np.mean(d))


def evaluate(truth: pd.DataFrame, pred: pd.DataFrame) -> pd.DataFrame:
    """rows: target; columns rmse/mae/r2 (NaN predictions are dropped pairwise)."""
    rows = {}
    for t in C.TARGETS:
        m = pred[t].notna() & truth[t].notna()
        rows[t] = {"rmse": rmse(truth[t][m], pred[t][m]), "mae": mae(truth[t][m], pred[t][m]), "r2": r2(truth[t][m], pred[t][m]), "n": int(m.sum())}
    return pd.DataFrame(rows).T


def wind_dir_mae(truth: pd.DataFrame, pred: pd.DataFrame) -> float:
    from .models import postprocess
    m = pred.notna().all(axis=1) & truth.notna().all(axis=1)
    t = np.degrees(np.arctan2(truth.loc[m, "target_wind_dir_sin"], truth.loc[m, "target_wind_dir_cos"])) % 360
    p = postprocess(pred[m])["wind_dir"]
    return angular_mae(t, p)


def score(metrics: pd.DataFrame, std: dict) -> float:
    """Mean RMSE normalised by target std – comparable across targets with different units."""
    return float(np.mean([metrics.loc[t, "rmse"] / max(std[t], 1e-9) for t in C.TARGETS]))
