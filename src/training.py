"""Training pipeline: load features -> time-based split -> train families -> compare with naive baseline
-> evaluate (RMSE/MAE/R²) -> promote to Model Registry only if better than the current champion."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from . import config as C
from . import evaluate as E
from . import logs
from .features import feature_cols, from_store_frame
from .models import FAMILIES, FAMILY_TYPE, Bundle, NaiveBaseline

PURGE = C.HORIZON  # gap between train and val so targets cannot leak across the split


def load_labelled(store) -> pd.DataFrame:
    df = store.read(C.FG_FEATURES)
    if df.empty:
        raise SystemExit("No features in the Feature Store – run scripts/backfill.py first.")
    df = from_store_frame(df)
    return df.dropna(subset=C.TARGETS + feature_cols(df))   # drops the first 24 h (no lag_24h) and the newest 24 h (no target)


def time_split(df: pd.DataFrame):
    cut = int(len(df) * (1 - C.VAL_FRACTION))
    return df.iloc[:cut - PURGE], df.iloc[cut:], cut


def _predict_val(model, df, cut):
    warm = max(cut - C.LSTM_WINDOW, 0)                    # LSTM needs history rows before the first val row
    return model.predict(df.iloc[warm:]).loc[df.index[cut:]]


def evaluate_bundle(bundle: Bundle, df: pd.DataFrame, cut: int):
    """Re-score an existing champion on the *current* validation window (fair comparison)."""
    warm = max(cut - C.LSTM_WINDOW, 0)
    pred = bundle.predict(df.iloc[warm:]).loc[df.index[cut:]]
    val = df.iloc[cut:]
    m = E.evaluate(val[C.TARGETS], pred)
    std = val[C.TARGETS].std().to_dict()
    return E.score(m, std), m


def train(store, families=None, promote_always: bool = False) -> dict:
    df = load_labelled(store)
    if len(df) < C.MIN_TRAIN_ROWS:
        raise SystemExit(f"Only {len(df)} labelled rows (< {C.MIN_TRAIN_ROWS}). Backfill more history.")
    fcols = feature_cols(df)
    train_df, val_df, cut = time_split(df)
    std = val_df[C.TARGETS].std().to_dict()
    print(f"[train] rows={len(df)} train={len(train_df)} val={len(val_df)} features={len(fcols)}")

    families = families or list(FAMILIES)
    results, models = {}, {}
    naive = NaiveBaseline()
    results["naive"] = E.evaluate(val_df[C.TARGETS], naive.predict(val_df))
    preds = {"naive": naive.predict(val_df)}
    for fam in families:
        try:
            print(f"[train] fitting {fam} ...")
            m = FAMILIES[fam]().fit(train_df, fcols)
            p = _predict_val(m, df, cut)
            results[fam], models[fam], preds[fam] = E.evaluate(val_df[C.TARGETS], p), m, p
        except ImportError as e:
            print(f"[train] skipping {fam}: {e}")
        except Exception as e:
            print(f"[train] {fam} failed: {e}")
            logs.log("failure_log", source="training", kind="model_fit", agent="training", error=f"{fam}: {e}", prompt="", model=fam)
    if not models:
        raise SystemExit("No model family trained successfully.")

    # best family per target (naive is a baseline, not a candidate)
    choice, val_rmse = {}, {}
    for t in C.TARGETS:
        best = min(models, key=lambda f: results[f].loc[t, "rmse"])
        choice[t], val_rmse[t] = best, float(results[best].loc[t, "rmse"])
    chosen_pred = pd.DataFrame({t: preds[choice[t]][t] for t in C.TARGETS})
    new_metrics = pd.DataFrame({t: results[choice[t]].loc[t] for t in C.TARGETS}).T
    new_score = E.score(new_metrics, std)
    wd_mae = E.wind_dir_mae(val_df[C.TARGETS], chosen_pred)
    naive_wd = E.wind_dir_mae(val_df[C.TARGETS], preds["naive"])

    bundle = Bundle(models=models, choice=choice, feature_cols=fcols, val_rmse=val_rmse,
                    val_std={t: float(std[t]) for t in C.TARGETS}, val_wind_dir_mae=wd_mae, score=new_score,
                    train_start=str(train_df.index[0]), train_end=str(train_df.index[-1]),
                    trained_at=str(logs.now_utc()))

    # ---- champion/challenger
    run_id = logs.now_utc().strftime("%Y%m%d%H%M%S")
    champion, info = store.load_model(C.MODEL_NAME)
    promoted, old_score, reason = True, None, "no existing model"
    if champion is not None and not promote_always:
        try:
            old_score, _ = evaluate_bundle(champion, df, cut)
            promoted = new_score < old_score
            reason = f"new {new_score:.4f} vs current {old_score:.4f} (lower is better)"
        except Exception as e:
            reason = f"current model incompatible ({e}); replacing"
    version = None
    if promoted:
        metrics = {"score": new_score, "wind_dir_mae_deg": wd_mae,
                   **{f"rmse_{t}": v for t, v in val_rmse.items()}}
        version = store.save_model(C.MODEL_NAME, bundle, metrics, "24h-ahead weather & AQ forecaster")
    print(f"[train] promoted={promoted} ({reason}) version={version}")

    # ---- persist metrics of every model/target for the dashboard + report
    rows = []
    for fam, m in results.items():
        for t in C.TARGETS:
            rows.append(dict(id=f"{run_id}-{fam}-{t}", ts=logs.now_utc(), run_id=run_id, family=fam,
                             family_type=FAMILY_TYPE[fam], target=t, rmse=m.loc[t, "rmse"], mae=m.loc[t, "mae"],
                             r2=m.loc[t, "r2"], beats_naive=bool(m.loc[t, "rmse"] < results["naive"].loc[t, "rmse"]),
                             chosen=bool(choice.get(t) == fam), promoted=bool(promoted), n_val=int(len(val_df))))
    store.write(C.FG_MODEL_METRICS, pd.DataFrame(rows))

    try:
        from .explain import explain_rf
        if promoted and "rf" in models:
            explain_rf(store, models["rf"], val_df, fcols, run_id)
    except Exception as e:
        print(f"[train] explainability skipped: {e}")
        logs.log("failure_log", source="training", kind="explain", agent="training", error=str(e), prompt="", model="")

    return {"promoted": promoted, "score": new_score, "old_score": old_score, "choice": choice,
            "version": version, "naive_wind_dir_mae": naive_wd, "wind_dir_mae": wd_mae, "reason": reason}
