"""Drift Agent logic: data drift (live vs training features, PSI + KS) and model drift (recent forecast error
vs usual validation error)."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import requests
from scipy import stats

from . import config as C
from . import logs
from .evaluate import angular_mae, rmse
from .features import from_store_frame

DATA_DRIFT_COLS = ["aqi", "pm2_5", "temperature", "humidity", "pressure", "wind_speed", "uv_index",
                   "wind_dir_sin", "wind_dir_cos"]


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    expected, actual = np.asarray(expected, float), np.asarray(actual, float)
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-4, None), np.clip(a, 1e-4, None)
    return float(np.sum((a - e) * np.log(a / e)))


def data_drift(store, bundle, live_hours: int = 168, now=None) -> list[dict]:
    now = now or logs.now_utc()
    feats = store.read(C.FG_FEATURES)
    if feats.empty:
        return []
    feats = from_store_frame(feats)
    train = feats.loc[bundle.train_start:bundle.train_end]
    live = feats[feats.index > now.floor("h") - pd.Timedelta(hours=live_hours)]
    if len(train) < 100 or len(live) < 48:
        return []
    out = []
    for c in DATA_DRIFT_COLS:
        if c not in feats:
            continue
        ks = stats.ks_2samp(train[c], live[c])
        out.append(dict(metric=c, drift_type="data", score=psi(train[c], live[c]), threshold=C.PSI_THRESHOLD,
                        detail=f"ks={ks.statistic:.3f} p={ks.pvalue:.3g} live_n={len(live)}"))
    return out


def model_drift(store, bundle, now=None, window_hours: int = 72, min_points: int = 24) -> list[dict]:
    now = now or logs.now_utc()
    led = store.read(C.FG_LEDGER, since=now - pd.Timedelta(hours=window_hours + 30), since_col="made_at")
    feats = store.read(C.FG_FEATURES, since=now - pd.Timedelta(hours=window_hours + 30))
    if led.empty or feats.empty:
        return []
    obs = from_store_frame(feats)
    led = led[(led["lead_hours"] == C.HORIZON) & (led["target_time"] <= obs.index.max())
              & (led["target_time"] > now - pd.Timedelta(hours=window_hours))]
    out = []
    wd_val = max(bundle.val_wind_dir_mae, 1.0)
    for metric in list(C.LEDGER_METRICS) + ["wind_dir"]:
        d = led[led["metric"] == metric].drop_duplicates("target_time", keep="last").set_index("target_time")
        if len(d) < min_points:
            continue
        if metric == "wind_dir":
            actual = np.degrees(np.arctan2(obs["wind_dir_sin"], obs["wind_dir_cos"])).mod(360).reindex(d.index)
            recent, val = angular_mae(actual.dropna(), d.loc[actual.notna(), "forecast"]), wd_val
        else:
            col = C.LEDGER_METRICS[metric]
            actual = obs[col].reindex(d.index)
            m = actual.notna()
            tkey = f"target_{col}"
            recent = rmse(actual[m], d.loc[m, "forecast"])
            val = max(bundle.val_rmse[tkey], 0.1 * bundle.val_std[tkey], 1e-6)
        out.append(dict(metric=metric, drift_type="model", score=float(recent / val), threshold=C.MODEL_DRIFT_RATIO,
                        detail=f"recent_err={recent:.3f} val_err={val:.3f} n={len(d)}"))
    return out


def notify_team(message: str) -> str:
    url = os.getenv("TEAM_WEBHOOK_URL")
    print(f"[drift] TEAM ALERT: {message}")
    if not url:
        return "logged_only"
    try:
        r = requests.post(url, json={"text": message, "content": message}, timeout=15)
        return "webhook_sent" if r.status_code < 300 else f"webhook_http_{r.status_code}"
    except Exception as e:
        logs.log("failure_log", agent="drift", kind="notify", source="webhook", error=str(e), prompt="", model="")
        return "webhook_failed"


def dispatch_training(reason: str) -> str:
    """Ask GitHub Actions to run the training workflow (workflow_dispatch)."""
    token, repo = os.getenv("GH_DISPATCH_TOKEN") or os.getenv("GITHUB_TOKEN"), os.getenv("GITHUB_REPOSITORY")
    if not (token and repo):
        return "training_requested_no_dispatch_configured"
    try:
        r = requests.post(f"https://api.github.com/repos/{repo}/actions/workflows/training.yml/dispatches",
                          headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"},
                          json={"ref": os.getenv("GITHUB_REF_NAME", "main"), "inputs": {"reason": reason[:200]}}, timeout=15)
        return "training_dispatched" if r.status_code == 204 else f"dispatch_http_{r.status_code}"
    except Exception as e:
        logs.log("failure_log", agent="drift", kind="dispatch", source="github", error=str(e), prompt="", model="")
        return "dispatch_failed"
