"""Runs the LangGraph agent graph (Forecast -> Alert -> Weather Report -> Drift). Hourly, after ingestion."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agents import run_graph
from src.store import get_store

if __name__ == "__main__":
    s = run_graph(get_store())
    print("forecast:", s.get("forecast_status"), "| new_alert:", s.get("new_alert"),
          "| report:", bool(s.get("report")), "| retrain:", s.get("trigger_training", False), "| errors:", s.get("errors"))
