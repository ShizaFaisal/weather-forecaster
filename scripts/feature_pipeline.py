"""Hourly feature + ingestion pipeline (run by GitHub Actions every hour).

fetch OpenWeather / AQICN / Open-Meteo -> store raw -> fuse -> impute -> features
-> upsert into the `weather_features` feature group (also fills the 24 h-ahead target of rows that just matured)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src import config as C
from src import logs, sources, synthetic
from src.features import build_features, from_store_frame, to_store_frame
from src.fusion import add_wind_components, fuse, impute
from src.store import get_store

LOOKBACK_H = 24 * 8   # history window needed to recompute rolling features / regularise the grid
WRITE_LAST_H = 30     # rows re-upserted each run (rows 24h back receive their targets)


def raw_to_store(raw: dict) -> pd.DataFrame:
    frames = []
    for src, df in raw.items():
        if df is not None and not df.empty:
            d = df.reset_index()
            d.insert(0, "source", src)
            d.insert(0, "city", C.CITY)
            frames.append(d)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def run(store, raw: dict | None = None) -> pd.DataFrame | None:
    if raw is None:
        raw = {"openweather": sources.fetch_openweather(), "aqicn": sources.fetch_aqicn(),
               "openmeteo": sources.fetch_recent_open_meteo(days=2)}
    live = {k: v for k, v in raw.items() if v is not None and not v.empty}
    print(f"[feature_pipeline] sources available: {sorted(live)}")
    if not live:
        logs.log("failure_log", source="all", kind="ingestion", agent="ingestion",
                 error="no data source returned data", prompt="", model="")
        return None
    store.write(C.FG_RAW, raw_to_store(live))

    fused = add_wind_components(fuse(live))
    now_h = logs.now_utc().floor("h")

    hist = store.read(C.FG_FEATURES, since=now_h - pd.Timedelta(hours=LOOKBACK_H))
    obs_cols = [c for c in fused.columns if c != "n_sources"]
    if not hist.empty:
        hist = from_store_frame(hist)
        # fresh rows replace stored rows only for new hours or previously imputed hours
        keep_fresh = ~fused.index.isin(hist.index) | (hist["n_imputed"].reindex(fused.index).fillna(1) > 0).values
        fused = fused[keep_fresh]
        base = hist[[c for c in obs_cols if c in hist.columns]]
        obs = pd.concat([base, fused[obs_cols]]).sort_index()
        obs = obs[~obs.index.duplicated(keep="last")]
        prev_imp = hist["n_imputed"].reindex(obs.index)
    else:
        obs, prev_imp = fused[obs_cols], None

    obs = impute(obs)
    if prev_imp is not None:   # keep the imputation count of rows we did not touch
        obs["n_imputed"] = obs["n_imputed"].where(prev_imp.reindex(obs.index).isna() | obs.index.isin(fused.index), prev_imp.reindex(obs.index))
    feat = build_features(obs)
    out = to_store_frame(feat.tail(WRITE_LAST_H))
    store.write(C.FG_FEATURES, out)
    print(f"[feature_pipeline] wrote {len(out)} rows, latest={feat.index[-1]}, imputed_in_latest={int(feat['n_imputed'].iloc[-1])}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true", help="offline demo: inject synthetic data instead of calling APIs")
    a = ap.parse_args()
    store = get_store()
    raw = {"openmeteo": synthetic.make_observations(days=3)} if a.synthetic else None
    try:
        run(store, raw)
    finally:
        logs.flush(store)
