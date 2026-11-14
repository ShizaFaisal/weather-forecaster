"""Storage abstraction.

* HopsworksStore – Feature Store (feature groups) + Model Registry. Used when HOPSWORKS_API_KEY is set.
* LocalStore     – pickle files under data/store. Used for offline dev / tests / CI dry-runs.

All tables are pandas DataFrames with tz-naive UTC timestamps.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path

import joblib
import pandas as pd

from . import config as C

# name -> (primary_key, event_time, description)
SCHEMAS = {
    C.FG_FEATURES: (["city", "timestamp"], "timestamp", "Hourly fused observations, engineered features and 24h-ahead targets"),
    C.FG_RAW: (["city", "source", "timestamp"], "timestamp", "Raw per-source observations before fusion"),
    C.FG_FORECASTS: (["city", "target_time"], "target_time", "Latest 24h forecast for every hour"),
    C.FG_LEDGER: (["id"], "made_at", "Forecast Ledger: every model forecast"),
    C.FG_AGENT_CALLS: (["id"], "ts", "Every LLM call: prompt, response, latency, status"),
    C.FG_FAILURES: (["id"], "ts", "Failed LLM calls and agent errors"),
    C.FG_DRIFT: (["id"], "ts", "Drift log"),
    C.FG_ALERTS: (["id"], "created_at", "Alerts written by the Alert Agent"),
    C.FG_REPORTS: (["id"], "created_at", "News-style weather reports"),
    C.FG_STATE: (["key"], "updated_at", "Shared agent state between runs"),
    C.FG_MODEL_METRICS: (["id"], "ts", "Validation metrics for every model/target of each training run"),
    C.FG_EXPLAIN: (["id"], "ts", "SHAP feature importances"),
}


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in df.columns:
        if str(df[c].dtype).startswith("datetime64") and getattr(df[c].dt, "tz", None) is not None:
            df[c] = df[c].dt.tz_convert("UTC").dt.tz_localize(None)
    return df


class BaseStore:
    def read(self, name, since=None, since_col=None) -> pd.DataFrame: ...
    def write(self, name, df): ...
    def save_model(self, name, bundle, metrics, description=""): ...
    def load_model(self, name): ...  # -> (bundle, info) or (None, None)


class LocalStore(BaseStore):
    kind = "local"

    def __init__(self, root: Path | str | None = None):
        self.root = Path(root or os.getenv("LOCAL_STORE_DIR", C.ROOT / "data" / "store"))
        self.root.mkdir(parents=True, exist_ok=True)

    def _p(self, name):
        return self.root / f"{name}.pkl"

    def read(self, name, since=None, since_col=None):
        p = self._p(name)
        if not p.exists():
            return pd.DataFrame()
        df = pd.read_pickle(p)
        if since is not None:
            col = since_col or SCHEMAS[name][1]
            df = df[df[col] >= pd.Timestamp(since)]
        return df.reset_index(drop=True)

    def write(self, name, df):
        if df is None or df.empty:
            return
        df = _clean(df)
        old = self.read(name)
        pk = SCHEMAS[name][0]
        out = pd.concat([old, df], ignore_index=True) if not old.empty else df
        out = out.drop_duplicates(subset=pk, keep="last")
        out.reset_index(drop=True).to_pickle(self._p(name))

    def save_model(self, name, bundle, metrics, description=""):
        d = self.root / "models" / name
        d.mkdir(parents=True, exist_ok=True)
        version = len([x for x in d.iterdir() if x.is_dir()]) + 1
        vd = d / f"v{version}"
        vd.mkdir()
        joblib.dump(bundle, vd / "bundle.joblib")
        (vd / "metrics.json").write_text(json.dumps(metrics, default=float))
        return version

    def load_model(self, name):
        d = self.root / "models" / name
        if not d.exists():
            return None, None
        versions = sorted([int(x.name[1:]) for x in d.iterdir() if x.is_dir()])
        if not versions:
            return None, None
        v = versions[-1]
        bundle = joblib.load(d / f"v{v}" / "bundle.joblib")
        metrics = json.loads((d / f"v{v}" / "metrics.json").read_text())
        return bundle, {"version": v, "metrics": metrics}


class HopsworksStore(BaseStore):
    kind = "hopsworks"

    def __init__(self):
        import hopsworks  # lazy
        self.project = hopsworks.login(
            api_key_value=os.environ["HOPSWORKS_API_KEY"],
            project=os.getenv("HOPSWORKS_PROJECT") or None,
        )
        self.fs = self.project.get_feature_store()
        self.mr = self.project.get_model_registry()
        self._fgs = {}

    def _fg(self, name):
        if name not in self._fgs:
            pk, et, desc = SCHEMAS[name]
            self._fgs[name] = self.fs.get_or_create_feature_group(
                name=name, version=1, primary_key=pk, event_time=et,
                description=desc, online_enabled=False)
        return self._fgs[name]

    def read(self, name, since=None, since_col=None):
        try:
            fg = self._fg(name)
            if since is not None:
                col = since_col or SCHEMAS[name][1]
                df = fg.filter(fg.get_feature(col) >= pd.Timestamp(since).to_pydatetime()).read()
            else:
                df = fg.read()
        except Exception as e:  # empty FG / not materialised yet
            print(f"[store] read({name}) failed: {e}")
            return pd.DataFrame()
        df = _clean(df)
        if since is not None:
            col = since_col or SCHEMAS[name][1]
            df = df[df[col] >= pd.Timestamp(since)]
        return df.reset_index(drop=True)

    def write(self, name, df):
        if df is None or df.empty:
            return
        self._fg(name).insert(_clean(df), write_options={"wait_for_job": True})

    def save_model(self, name, bundle, metrics, description=""):
        tmp = tempfile.mkdtemp()
        try:
            joblib.dump(bundle, Path(tmp) / "bundle.joblib")
            flat = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
            m = self.mr.python.create_model(name=name, metrics=flat, description=description)
            m.save(tmp)
            return m.version
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def load_model(self, name):
        try:
            models = self.mr.get_models(name)
        except Exception:
            return None, None
        if not models:
            return None, None
        m = max(models, key=lambda x: x.version)
        path = Path(m.download())
        bundle = joblib.load(path / "bundle.joblib")
        return bundle, {"version": m.version, "metrics": m.training_metrics or {}}


def get_store() -> BaseStore:
    if os.getenv("HOPSWORKS_API_KEY"):
        return HopsworksStore()
    print("[store] HOPSWORKS_API_KEY not set -> using LocalStore (data/store)")
    return LocalStore()
