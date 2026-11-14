"""Forecast logic used by the Forecast Agent."""
from __future__ import annotations

import pandas as pd

from . import config as C
from . import logs
from .features import from_store_frame
from .models import postprocess

STALE_AFTER_H = 3


def make_forecast(store, now: pd.Timestamp | None = None) -> dict:
    now = (now or logs.now_utc()).floor("h")
    bundle, info = store.load_model(C.MODEL_NAME)
    if bundle is None:
        return {"status": "no_model", "forecast": pd.DataFrame()}
    feats = store.read(C.FG_FEATURES, since=now - pd.Timedelta(hours=96))
    if feats.empty:
        return {"status": "no_features", "forecast": pd.DataFrame()}
    feats = from_store_frame(feats)
    last_ts = feats.index[-1]
    window = feats.tail(24 + C.LSTM_WINDOW)
    live = window.tail(C.HORIZON)                       # newest 24 rows: no target yet -> forecast only
    pred = postprocess(bundle.predict(window).loc[live.index])
    if pred.isna().any().any():
        pred = pred.ffill().bfill()
    pred = pred.reset_index().rename(columns={"timestamp": "feature_time"})
    pred["target_time"] = pred["feature_time"] + pd.Timedelta(hours=C.HORIZON)
    pred["made_at"] = now
    pred["city"] = C.CITY
    pred["model_version"] = int(info["version"])
    pred["lead_hours"] = (pred["target_time"] - pred["feature_time"]).dt.total_seconds() / 3600
    status = "stale_features" if (now - last_ts) > pd.Timedelta(hours=STALE_AFTER_H) else "ok"
    return {"status": status, "forecast": pred, "model_version": info["version"], "choice": bundle.choice,
            "last_feature_time": last_ts}


def ledger_rows(fc: pd.DataFrame, choice: dict) -> pd.DataFrame:
    """Forecast Ledger: one row per (made_at, target_time, metric)."""
    rows = []
    fam = {m: choice.get(f"target_{o}") for m, o in C.LEDGER_METRICS.items()}
    fam["wind_dir"] = choice.get("target_wind_dir_sin")
    for metric in list(C.LEDGER_METRICS) + ["wind_dir"]:
        d = pd.DataFrame({"made_at": fc["made_at"], "target_time": fc["target_time"], "feature_time": fc["feature_time"],
                          "lead_hours": fc["lead_hours"], "metric": metric, "forecast": fc[metric],
                          "model_family": fam[metric], "model_version": fc["model_version"]})
        rows.append(d)
    out = pd.concat(rows, ignore_index=True)
    out["id"] = (out["made_at"].dt.strftime("%Y%m%d%H") + "-" + out["target_time"].dt.strftime("%Y%m%d%H") + "-" + out["metric"])
    out["city"] = C.CITY
    return out
