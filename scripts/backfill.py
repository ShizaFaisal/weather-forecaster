"""Historical backfill: same fusion/imputation/feature code as the hourly pipeline, applied to past dates.

Targets are created by shifting observations 24 h ahead. The newest 24 rows therefore have NaN targets and are
used for forecasting only."""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config as C
from src import logs, sources, synthetic
from src.features import build_features, to_store_frame
from src.fusion import add_wind_components, fuse, impute
from src.store import get_store


def run(store, days: int = 180, synthetic_data: bool = False):
    if synthetic_data:
        om = synthetic.make_observations(days=days)
    else:
        today = datetime.now(timezone.utc).date()
        om = sources.fetch_open_meteo(today - timedelta(days=days), today)
    if om is None or om.empty:
        raise SystemExit("Backfill failed: Open-Meteo returned no data (see failure_log).")
    store.write(C.FG_RAW, _raw(om))
    obs = impute(add_wind_components(fuse({"openmeteo": om})).drop(columns=["n_sources"]))
    feat = build_features(obs)
    store.write(C.FG_FEATURES, to_store_frame(feat))
    n_lab = int(feat[C.TARGETS].notna().all(axis=1).sum())
    print(f"[backfill] {len(feat)} rows ({feat.index[0]} -> {feat.index[-1]}); {n_lab} labelled; "
          f"{len(feat) - n_lab} newest rows kept for forecasting only")


def _raw(om):
    d = om.reset_index()
    d.insert(0, "source", "openmeteo")
    d.insert(0, "city", C.CITY)
    return d


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=180)
    ap.add_argument("--synthetic", action="store_true")
    a = ap.parse_args()
    store = get_store()
    try:
        run(store, a.days, a.synthetic)
    finally:
        logs.flush(store)
