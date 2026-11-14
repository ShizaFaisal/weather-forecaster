"""Four agents as nodes of ONE LangGraph graph. Shared state is persisted between runs in the
`agent_state` feature group (the Forecast Ledger's companion) so each run knows what the last one did.

START -> forecast -> alert -> (report if daily-due or new alert) -> drift -> END
"""
from __future__ import annotations

import json
from typing import Any, TypedDict

import pandas as pd

from . import config as C
from . import drift as D
from . import logs
from .forecast import ledger_rows, make_forecast
from .llm import call_llm
from .rules import ADVICE, AUDIENCES, evaluate_rules, signature


class AgentState(TypedDict, total=False):
    store: Any
    now: Any
    memory: dict            # persisted between runs
    forecast: Any           # DataFrame
    forecast_status: str
    hazards: list
    alerts: list
    new_alert: bool
    report: str
    drift_events: list
    trigger_training: bool
    errors: list


# ------------------------------------------------------------------ persistent state
def load_memory(store) -> dict:
    df = store.read(C.FG_STATE)
    return {r.key: json.loads(r.value) for r in df.itertuples()} if not df.empty else {}


def save_memory(store, mem: dict):
    now = logs.now_utc()
    store.write(C.FG_STATE, pd.DataFrame([{"key": k, "value": json.dumps(v, default=str), "updated_at": now} for k, v in mem.items()]))


# ------------------------------------------------------------------ 1. Forecast Agent
def forecast_agent(state: AgentState) -> AgentState:
    store, now = state["store"], state["now"]
    res = make_forecast(store, now)
    state["forecast_status"] = res["status"]
    if res["status"] in ("no_model", "no_features") or res["forecast"].empty:
        state.setdefault("errors", []).append(f"forecast: {res['status']}")
        state["forecast"] = pd.DataFrame()
        logs.log("failure_log", agent="forecast", kind="forecast", source="forecast", error=res["status"], prompt="", model="")
        return state
    fc = res["forecast"]
    store.write(C.FG_FORECASTS, fc)
    store.write(C.FG_LEDGER, ledger_rows(fc, res["choice"]))
    state["forecast"] = fc
    state["memory"]["last_forecast_at"] = str(now)
    state["memory"]["model_version"] = res["model_version"]
    if res["status"] == "stale_features":
        state.setdefault("errors", []).append("forecast produced from stale features")
    return state


# ------------------------------------------------------------------ 2. Alert Agent
ALERT_SYSTEM = ("You write concise public-safety weather alerts. Use ONLY the hazards given. "
                "Return a JSON object keyed by metric; each value has keys general, sensitive_groups, outdoor_workers "
                "(one or two plain sentences each, actionable, no invented numbers).")


def _template_messages(h: dict) -> dict:
    base = f"{h['label']}: forecast {h['agg']} {h['peak_value']} {h['unit']} around {h['peak_time'][:16].replace('T', ' ')} UTC. "
    adv = ADVICE[h["metric"]]
    return {a: base + adv[a] for a in AUDIENCES}


def alert_agent(state: AgentState) -> AgentState:
    mem, fc = state["memory"], state.get("forecast")
    if fc is None or fc.empty:
        state["hazards"], state["alerts"], state["new_alert"] = [], [], False
        return state
    hazards = evaluate_rules(fc)
    sig = signature(hazards)
    state["hazards"] = hazards
    state["new_alert"] = bool(hazards) and sig != mem.get("alert_signature", "")
    alerts = []
    if state["new_alert"]:
        keys = sorted({h["metric"] for h in hazards})
        prompt = "Hazards (JSON):\n" + json.dumps(hazards, indent=1) + f"\nCity: {C.CITY}. Next 24 hours."
        res, meta = call_llm("alert", "write_alerts", ALERT_SYSTEM, prompt, as_json=True, json_keys=keys)
        for h in hazards:
            msgs, src = None, "template"
            if res and isinstance(res.get(h["metric"]), dict) and all(isinstance(res[h["metric"]].get(a), str) and res[h["metric"]].get(a) for a in AUDIENCES):
                msgs, src = res[h["metric"]], "llm"
            msgs = msgs or _template_messages(h)
            for a in AUDIENCES:
                alerts.append(logs.log("alerts", city=C.CITY, metric=h["metric"], level=h["level"], label=h["label"],
                                       peak_value=h["peak_value"], peak_time=pd.Timestamp(h["peak_time"]), rule=h["rule"],
                                       audience=a, message=msgs[a], source=src))
    mem["alert_signature"] = sig
    mem["active_alerts"] = [{k: v for k, v in h.items()} for h in hazards]
    state["alerts"] = alerts
    return state


def route_after_alert(state: AgentState) -> str:
    mem, now = state["memory"], state["now"]
    last = mem.get("last_report_at")
    daily_due = last is None or (now - pd.Timestamp(last)) >= pd.Timedelta(hours=23)
    has_fc = state.get("forecast") is not None and not state["forecast"].empty
    return "report" if has_fc and (daily_due or state.get("new_alert")) else "drift"


# ------------------------------------------------------------------ 3. Weather Report Agent
REPORT_SYSTEM = ("You are a TV/online weather journalist. Write ONE news-style report (130-220 words, no headings, "
                 "one headline line first) for the next 24 hours using ONLY the supplied numbers and alerts. "
                 "Mention any alerts prominently. Use plain text.")


def _summary(fc: pd.DataFrame) -> dict:
    s = {}
    for m in ["aqi", "temperature", "humidity", "uv_index", "precip_mm", "rain_prob", "wind_speed"]:
        s[m] = dict(min=round(float(fc[m].min()), 1), max=round(float(fc[m].max()), 1), mean=round(float(fc[m].mean()), 1),
                    peak_at=pd.Timestamp(fc.loc[fc[m].idxmax(), "target_time"]).strftime("%H:%M UTC"))
    s["wind_dir_deg_mean_last"] = round(float(fc["wind_dir"].iloc[-1]), 0)
    return s


def _compass(deg: float) -> str:
    return ["N", "NE", "E", "SE", "S", "SW", "W", "NW"][int((deg + 22.5) % 360 // 45)]


def _template_report(fc: pd.DataFrame, hazards: list) -> str:
    s = _summary(fc)
    head = f"{C.CITY} outlook: " + ("alerts in force" if hazards else "no weather hazards expected")
    t = (f"{head}\n\nTemperatures over the next 24 hours will range from {s['temperature']['min']}°C to {s['temperature']['max']}°C "
         f"(peaking around {s['temperature']['peak_at']}), with humidity between {s['humidity']['min']:.0f}% and {s['humidity']['max']:.0f}%. "
         f"Winds will average {s['wind_speed']['mean']} m/s, mainly from the {_compass(s['wind_dir_deg_mean_last'])}. "
         f"Air quality is forecast to reach a US AQI of {s['aqi']['max']:.0f} at {s['aqi']['peak_at']}, and the UV index may climb to {s['uv_index']['max']}. ")
    t += (f"The chance of rain peaks at {s['rain_prob']['max'] * 100:.0f}% with up to {s['precip_mm']['max']} mm/h of precipitation. ")
    if hazards:
        t += "Active alerts: " + "; ".join(f"{h['label']} ({h['peak_value']} {h['unit']})" for h in hazards) + ". "
    return t.strip()


def report_agent(state: AgentState) -> AgentState:
    fc, hazards, mem = state["forecast"], state.get("hazards", []), state["memory"]
    prompt = (f"City: {C.CITY}\nForecast summary (JSON):\n{json.dumps(_summary(fc))}\nActive alerts:\n"
              f"{json.dumps([{k: h[k] for k in ('label', 'level', 'peak_value', 'unit', 'peak_time')} for h in hazards])}")
    text, meta = call_llm("report", "write_report", REPORT_SYSTEM, prompt, min_words=60)
    src = "llm" if text else "template"
    text = text or _template_report(fc, hazards)
    logs.log("weather_reports", city=C.CITY, report=text, source=src, n_alerts=len(hazards),
             trigger="new_alert" if state.get("new_alert") else "daily")
    state["report"] = text
    mem["last_report_at"] = str(state["now"])
    return state


# ------------------------------------------------------------------ 4. Drift Agent
def drift_agent(state: AgentState) -> AgentState:
    store, now, mem = state["store"], state["now"], state["memory"]
    last = mem.get("last_drift_check")
    if last and (now - pd.Timestamp(last)) < pd.Timedelta(hours=C.DRIFT_CHECK_EVERY_H):
        return state
    bundle, _ = store.load_model(C.MODEL_NAME)
    if bundle is None:
        return state
    events = []
    for fn in (D.data_drift, D.model_drift):
        try:
            events += fn(store, bundle, now=now)
        except Exception as e:
            logs.log("failure_log", agent="drift", kind="drift_check", source=fn.__name__, error=str(e), prompt="", model="")
    data_hit = [e for e in events if e["drift_type"] == "data" and e["score"] > e["threshold"]]
    model_hit = [e for e in events if e["drift_type"] == "model" and e["score"] > e["threshold"]]
    retrain = len(data_hit) >= C.DATA_DRIFT_MIN_COLS or len(model_hit) >= C.MODEL_DRIFT_MIN_METRICS
    last_train = mem.get("last_training_trigger")
    cooling = bool(last_train) and (now - pd.Timestamp(last_train)) < pd.Timedelta(hours=C.RETRAIN_COOLDOWN_H)
    drifted = data_hit + model_hit
    team_status = "n/a"
    if drifted:
        team_status = D.notify_team(f"[{C.CITY} forecaster] drift detected: " + ", ".join(f"{e['drift_type']}:{e['metric']}={e['score']:.2f}" for e in drifted))
    train_status = "not_needed"
    if retrain and not cooling:
        train_status = D.dispatch_training("drift: " + ", ".join(e["metric"] for e in drifted)[:150])
        mem["last_training_trigger"] = str(now)
        state["trigger_training"] = True
    elif retrain:
        train_status = "suppressed_cooldown"
    for e in events:
        hit = e["score"] > e["threshold"]
        action = "none" if not hit else (f"alert_team({team_status})" + (f"+{train_status}" if retrain else ""))
        logs.log("drift_log", city=C.CITY, metric=e["metric"], drift_type=e["drift_type"], score=float(e["score"]),
                 threshold=float(e["threshold"]), drifted=bool(hit), action=action, detail=e["detail"])
    state["drift_events"] = events
    mem["last_drift_check"] = str(now)
    return state


# ------------------------------------------------------------------ graph
def build_graph():
    from langgraph.graph import END, START, StateGraph
    g = StateGraph(AgentState)
    g.add_node("forecast", forecast_agent)
    g.add_node("alert", alert_agent)
    g.add_node("report", report_agent)
    g.add_node("drift", drift_agent)
    g.add_edge(START, "forecast")
    g.add_edge("forecast", "alert")
    g.add_conditional_edges("alert", route_after_alert, {"report": "report", "drift": "drift"})
    g.add_edge("report", "drift")
    g.add_edge("drift", END)
    return g.compile()


def run_graph(store, now=None) -> AgentState:
    now = (now or logs.now_utc()).floor("h")
    memory = load_memory(store)
    state: AgentState = {"store": store, "now": now, "memory": memory, "errors": []}
    try:
        state = build_graph().invoke(state)
    finally:
        try:
            save_memory(store, state.get("memory", memory))
        finally:
            logs.flush(store)
    return state
