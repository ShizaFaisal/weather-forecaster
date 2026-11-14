"""Public dashboard: 24 h forecasts for all six metrics, news-style report, Agent Activity (alerts),
EDA, SHAP/LIME explanations, drift events/metrics and failed agent calls."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import config as C
from src import eda
from src.features import from_store_frame
from src.store import get_store

st.set_page_config(page_title=f"{C.CITY} 24h Weather & Air Quality", page_icon="🌤️", layout="wide")


@st.cache_resource
def store():
    return get_store()


@st.cache_data(ttl=600, show_spinner="Loading from the Feature Store…")
def load(name, days=None, col=None):
    since = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=days) if days else None
    return store().read(name, since=since, since_col=col)


st.title(f"🌤️ {C.CITY}: 24-hour weather & air-quality forecast")
if st.sidebar.button("🔄 Refresh data"):
    st.cache_data.clear()
st.sidebar.caption(f"Storage backend: **{store().kind}**")

fc = load(C.FG_FORECASTS, days=3, col="target_time")
state = load(C.FG_STATE)
mem = {r.key: json.loads(r.value) for r in state.itertuples()} if not state.empty else {}
if not fc.empty:
    fc = fc.sort_values("target_time").tail(C.HORIZON)      # newest 24 target hours = the current 24 h forecast
    st.sidebar.caption(f"Last forecast run: {mem.get('last_forecast_at', 'n/a')} UTC · model v{mem.get('model_version', '?')}")

tabs = st.tabs(["📈 Forecast", "📰 Weather report", "🤖 Agent activity", "🔎 EDA", "🧠 Explainability", "🛰️ Monitoring"])

# ------------------------------------------------------------ forecast
with tabs[0]:
    if fc.empty:
        st.info("No forecast yet – run the backfill, training and hourly pipelines.")
    else:
        latest = fc.iloc[0]
        c = st.columns(6)
        c[0].metric("AQI", f"{fc['aqi'].max():.0f}", "peak 24h")
        c[1].metric("Temp °C", f"{fc['temperature'].max():.1f}", f"min {fc['temperature'].min():.1f}")
        c[2].metric("Precip mm/h", f"{fc['precip_mm'].max():.1f}", f"rain {fc['rain_prob'].max() * 100:.0f}%")
        c[3].metric("Humidity %", f"{fc['humidity'].mean():.0f}", "mean")
        c[4].metric("UV", f"{fc['uv_index'].max():.1f}", "peak")
        c[5].metric("Wind m/s", f"{fc['wind_speed'].mean():.1f}", f"{fc['wind_dir'].iloc[-1]:.0f}°")
        cols = st.columns(2)
        for i, (m, ttl) in enumerate([("aqi", "AQI"), ("temperature", "Temperature (°C)"), ("humidity", "Humidity (%)"),
                                      ("uv_index", "UV index")]):
            f = px.line(fc, x="target_time", y=m, title=ttl, markers=True)
            cols[i % 2].plotly_chart(f, use_container_width=True)
        p = go.Figure()
        p.add_bar(x=fc["target_time"], y=fc["precip_mm"], name="Precipitation (mm/h)")
        p.add_scatter(x=fc["target_time"], y=fc["rain_prob"] * 100, name="Rain probability (%)", yaxis="y2")
        p.update_layout(title="Precipitation", yaxis2=dict(overlaying="y", side="right", range=[0, 100]))
        w = go.Figure()
        w.add_scatter(x=fc["target_time"], y=fc["wind_speed"], name="Speed (m/s)")
        w.add_scatter(x=fc["target_time"], y=fc["wind_dir"], name="Direction (°)", yaxis="y2", mode="markers")
        w.update_layout(title="Wind", yaxis2=dict(overlaying="y", side="right", range=[0, 360]))
        cols[0].plotly_chart(p, use_container_width=True)
        cols[1].plotly_chart(w, use_container_width=True)

# ------------------------------------------------------------ report
with tabs[1]:
    rep = load(C.FG_REPORTS, days=14, col="created_at")
    if rep.empty:
        st.info("No report yet.")
    else:
        r = rep.sort_values("created_at").iloc[-1]
        st.caption(f"Generated {r['created_at']} UTC · trigger: {r['trigger']} · writer: {r['source']}")
        st.markdown(r["report"].replace("\n", "\n\n"))
        with st.expander("Previous reports"):
            for _, x in rep.sort_values("created_at", ascending=False).iloc[1:8].iterrows():
                st.markdown(f"**{x['created_at']}** ({x['source']})\n\n{x['report']}")

# ------------------------------------------------------------ agent activity
with tabs[2]:
    st.subheader("Active alerts")
    active = mem.get("active_alerts", [])
    alerts = load(C.FG_ALERTS, days=7, col="created_at")
    if not active:
        st.success("No active hazards in the next 24 hours.")
    badge = {"watch": "🟡", "warning": "🟠", "danger": "🔴"}
    for h in active:
        st.markdown(f"{badge[h['level']]} **{h['label']}** – {h['agg']} {h['peak_value']} {h['unit']} (rule `{h['rule']}`)")
    if not alerts.empty:
        latest_ts = alerts["created_at"].max()
        cur = alerts[alerts["created_at"] == latest_ts]
        aud = st.radio("Audience", ["general", "sensitive_groups", "outdoor_workers"], horizontal=True)
        for _, a in cur[cur["audience"] == aud].iterrows():
            st.info(f"**{a['label']}** ({a['source']}): {a['message']}")
        st.subheader("Alert history")
        st.dataframe(alerts.sort_values("created_at", ascending=False)[["created_at", "metric", "level", "audience", "source", "message"]], use_container_width=True)
    calls = load(C.FG_AGENT_CALLS, days=7, col="ts")
    if not calls.empty:
        st.subheader("Agent call log")
        c1, c2, c3 = st.columns(3)
        c1.metric("LLM calls (7d)", len(calls))
        c2.metric("Success rate", f"{(calls['status'] == 'success').mean() * 100:.0f}%")
        ok = calls[calls["status"] == "success"]
        c3.metric("Median latency", f"{ok['latency_ms'].median():.0f} ms" if len(ok) else "n/a")
        st.dataframe(calls.sort_values("ts", ascending=False)[["ts", "agent", "purpose", "model", "status", "latency_ms", "prompt", "response"]].head(100), use_container_width=True)

# ------------------------------------------------------------ EDA
with tabs[3]:
    hist = load(C.FG_FEATURES)
    if hist.empty:
        st.info("No data yet.")
    else:
        d = eda.prepare(from_store_frame(hist))
        col = st.selectbox("Metric", list(eda.EDA_COLS), format_func=eda.EDA_COLS.get)
        prof = eda.profile(d, col)
        st.dataframe(pd.DataFrame(prof, index=["value"]).T, use_container_width=True)
        a, b = st.columns(2)
        a.plotly_chart(px.line(d.reset_index(), x="timestamp", y=col, title="Time series"), use_container_width=True)
        b.plotly_chart(px.histogram(d, x=col, nbins=50, title="Distribution"), use_container_width=True)
        hp = eda.hourly_profile(d, col)
        a.plotly_chart(px.line(hp.reset_index(), x="index", y="mean", error_y="std", title="Diurnal profile (mean ± sd)"), use_container_width=True)
        ac = eda.acf(d[col], 72)
        b.plotly_chart(px.bar(x=list(range(len(ac))), y=ac, title="Autocorrelation (lags in hours)"), use_container_width=True)
        st.plotly_chart(px.imshow(eda.correlation(d), text_auto=".2f", title="Correlation matrix", aspect="auto"), use_container_width=True)

# ------------------------------------------------------------ explainability
with tabs[4]:
    ex = load(C.FG_EXPLAIN)
    tgt = st.selectbox("Target", C.TARGETS)
    if ex.empty:
        st.info("SHAP values are written by the training pipeline.")
    else:
        run = ex["run_id"].max()
        e = ex[(ex["run_id"] == run) & (ex["target"] == tgt)].sort_values("mean_abs_shap")
        st.plotly_chart(px.bar(e, x="mean_abs_shap", y="feature", orientation="h", title=f"SHAP – mean |impact| ({tgt}, run {run})"), use_container_width=True)
    if st.button("Explain the latest forecast with LIME"):
        bundle, _ = store().load_model(C.MODEL_NAME)
        feats = from_store_frame(load(C.FG_FEATURES))
        if bundle is None or "rf" not in bundle.models:
            st.warning("No Random-Forest model available.")
        else:
            from src.explain import lime_explain
            X = feats[bundle.feature_cols].dropna()
            w = lime_explain(bundle.models["rf"], X.iloc[:-24].tail(1500), X.iloc[-1], tgt)
            st.plotly_chart(px.bar(w.sort_values("weight"), x="weight", y="feature", orientation="h", title="LIME – local explanation"), use_container_width=True)

# ------------------------------------------------------------ monitoring
with tabs[5]:
    mm = load(C.FG_MODEL_METRICS)
    st.subheader("Model comparison (validation, latest run)")
    if not mm.empty:
        last = mm[mm["run_id"] == mm["run_id"].max()]
        st.caption(f"Run {last['run_id'].iloc[0]} · promoted: {bool(last['promoted'].iloc[0])}")
        st.dataframe(last.pivot_table(index="target", columns="family", values="rmse").round(3), use_container_width=True)
        st.dataframe(last[["family", "family_type", "target", "rmse", "mae", "r2", "beats_naive", "chosen"]].sort_values(["target", "rmse"]), use_container_width=True)
    dl = load(C.FG_DRIFT, days=14, col="ts")
    st.subheader("Drift")
    if dl.empty:
        st.info("No drift checks logged yet.")
    else:
        st.metric("Drift events (14d)", int(dl["drifted"].sum()))
        st.plotly_chart(px.line(dl.sort_values("ts"), x="ts", y="score", color="metric", facet_row="drift_type", title="Drift scores (PSI for data, error ratio for model)"), use_container_width=True)
        st.dataframe(dl[dl["drifted"]].sort_values("ts", ascending=False)[["ts", "metric", "drift_type", "score", "threshold", "action"]], use_container_width=True)
    st.subheader("Failed agent calls & errors")
    fl = load(C.FG_FAILURES, days=14, col="ts")
    st.dataframe(fl.sort_values("ts", ascending=False) if not fl.empty else pd.DataFrame({"info": ["no failures"]}), use_container_width=True)
    led = load(C.FG_LEDGER, days=3, col="made_at")
    st.subheader("Forecast Ledger (latest)")
    if not led.empty:
        st.dataframe(led.sort_values("made_at", ascending=False).head(200), use_container_width=True)
