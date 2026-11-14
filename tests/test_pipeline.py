import os
import tempfile
from unittest import mock

import numpy as np
import pandas as pd

from src import agents, config as C, logs, synthetic
from src.drift import psi
from src.features import build_features, feature_cols
from src.fusion import add_wind_components, fuse, impute
from src.llm import call_llm
from src.rules import evaluate_rules
from src.sources import parse_aqicn, parse_openweather, pm25_to_aqi
from src.store import LocalStore
from src.training import train


def _obs(days=10):
    om = synthetic.make_observations(days=days)
    return impute(add_wind_components(fuse({"openmeteo": om})).drop(columns=["n_sources"]))


def test_pm25_to_aqi():
    assert pm25_to_aqi(0) == 0 and 100 < pm25_to_aqi(45) < 150 and pm25_to_aqi(600) == 500


def test_parsers():
    ow = parse_openweather({"main": {"temp": 30, "humidity": 50, "pressure": 1000}, "wind": {"speed": 3, "deg": 200}},
                           {"list": [{"components": {"pm2_5": 40, "pm10": 80}}]})
    assert ow["temperature"].iloc[0] == 30 and ow["aqi"].iloc[0] > 100
    assert parse_aqicn({"status": "ok", "data": {"aqi": 150, "iaqi": {"pm25": {"v": 60}}}})["aqi"].iloc[0] == 150
    assert parse_aqicn({"status": "error"}) is None


def test_fusion_mean_and_priority():
    idx = pd.DatetimeIndex(["2026-01-01 00:00"], name="timestamp")
    a = pd.DataFrame({"temperature": [30.0], "aqi": [100.0], "wind_dir": [10.0]}, index=idx)
    b = pd.DataFrame({"temperature": [32.0], "aqi": [140.0], "wind_dir": [350.0], "uv_index": [3.0]}, index=idx)
    f = fuse({"openweather": a, "openmeteo": b, "aqicn": pd.DataFrame({"aqi": [170.0]}, index=idx)})
    assert f["temperature"].iloc[0] == 31.0       # mean
    assert f["aqi"].iloc[0] == 170.0               # priority: aqicn first
    assert f["wind_dir"].iloc[0] == 10.0           # circular -> priority, never averaged
    assert f["uv_index"].iloc[0] == 3.0


def test_imputation_fills_gaps_and_flags():
    obs = synthetic.make_observations(days=5)
    obs.iloc[40:44, obs.columns.get_loc("temperature")] = np.nan      # short gap -> interpolation
    obs = obs.drop(obs.index[70:75])                                    # missing rows -> grid regularised
    out = impute(add_wind_components(fuse({"openmeteo": obs})).drop(columns=["n_sources"]))
    assert out.isna().sum().sum() == 0
    assert out.index.to_series().diff().dropna().eq(pd.Timedelta(hours=1)).all()
    assert out["n_imputed"].max() > 0


def test_targets_shift_and_newest_24_unlabelled():
    f = build_features(_obs())
    assert f[C.TARGETS].iloc[-24:].isna().all().all()
    assert f[C.TARGETS].iloc[:-24].notna().all().all()
    assert np.isclose(f["target_temperature"].iloc[10], f["temperature"].iloc[34])
    assert not any(c.startswith("target_") for c in feature_cols(f))   # no leakage into inputs


def test_rules_cover_metrics():
    t = pd.date_range("2026-01-01", periods=24, freq="h")
    fc = pd.DataFrame({"target_time": t, "aqi": 320.0, "temperature": 41.0, "humidity": 50.0, "uv_index": 12.0,
                       "precip_mm": 9.0, "rain_prob": 0.9, "wind_speed": 21.0})
    got = {(h["metric"], h["level"]) for h in evaluate_rules(fc)}
    assert {("aqi", "danger"), ("temperature", "warning"), ("uv_index", "danger"), ("precip_mm", "warning"),
            ("wind_speed", "danger"), ("rain_prob", "watch")} <= got
    fc["aqi"], fc["temperature"], fc["uv_index"], fc["precip_mm"], fc["rain_prob"], fc["wind_speed"] = 50, 25, 2, 0, 0.1, 3
    assert evaluate_rules(fc) == []


def test_psi():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 2000)
    assert psi(a, rng.normal(0, 1, 500)) < 0.1
    assert psi(a, rng.normal(2, 1, 500)) > 0.25


def test_llm_fallbacks_are_logged():
    logs._BUFFER.clear()
    with mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop("GROQ_API_KEY", None); os.environ.pop("LLM_API_KEY", None)
        res, meta = call_llm("alert", "t", "s", "p")
    assert res is None and meta["status"] == "no_api_key"
    assert logs.peek("failure_log") and logs.peek("agent_calls")
    logs._BUFFER.clear()
    bad = mock.Mock(status_code=500, text="boom")
    with mock.patch.dict(os.environ, {"GROQ_API_KEY": "x"}), mock.patch("src.llm.requests.post", return_value=bad), mock.patch("src.llm.time.sleep"):
        res, meta = call_llm("report", "t", "s", "p")
    assert res is None and meta["status"] == "all_failed"
    assert len(logs.peek("failure_log")) == 2 * len(C.LLM_MODELS) * 1 or len(logs.peek("failure_log")) >= 2
    assert logs.peek("agent_calls")[-1]["status"] == "fallback_used"
    logs._BUFFER.clear()


def test_llm_json_validation():
    ok = mock.Mock(status_code=200)
    ok.json.return_value = {"choices": [{"message": {"content": '{"aqi": {"general": "a"}}'}}]}
    with mock.patch.dict(os.environ, {"GROQ_API_KEY": "x"}), mock.patch("src.llm.requests.post", return_value=ok):
        res, meta = call_llm("alert", "t", "s", "p", as_json=True, json_keys=["aqi"])
    assert res == {"aqi": {"general": "a"}} and meta["status"] == "success"
    logs._BUFFER.clear()


def test_end_to_end_local():
    with tempfile.TemporaryDirectory() as d:
        store = LocalStore(d)
        from scripts import backfill, feature_pipeline
        backfill.run(store, days=45, synthetic_data=True)
        info = train(store, families=["ridge", "rf"])
        assert info["promoted"]
        again = train(store, families=["ridge", "rf"])
        assert not again["promoted"]                      # same data -> not better -> champion kept
        # hourly pipeline with new data for the next hour
        feat = store.read(C.FG_FEATURES)
        assert feat["target_aqi"].isna().sum() == 24
        now = logs.now_utc().floor("h")
        state = {"store": store, "now": now, "memory": {}, "errors": []}
        state = agents.forecast_agent(state)
        assert len(state["forecast"]) == 24 and state["forecast_status"] == "ok"
        assert state["forecast"]["target_time"].min() > now and state["forecast"]["target_time"].max() <= now + pd.Timedelta(hours=24)
        assert len(store.read(C.FG_LEDGER)) == 24 * 8
        state = agents.alert_agent(state)
        state = agents.report_agent(state)
        assert len(state["report"].split()) > 40
        state = agents.drift_agent(state)
        assert "last_drift_check" in state["memory"]
        agents.save_memory(store, state["memory"]); logs.flush(store)
        assert not store.read(C.FG_DRIFT).empty and not store.read(C.FG_MODEL_METRICS).empty
        assert set(agents.load_memory(store)) >= {"last_report_at", "alert_signature"}
        # model drift needs matured forecasts: none yet -> only data-drift rows
        assert set(store.read(C.FG_DRIFT)["drift_type"]) == {"data"}
