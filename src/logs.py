"""Buffered logging into the Feature Store (one insert per table per run, not per event)."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pandas as pd

_BUFFER: dict[str, list[dict]] = {}


def now_utc() -> pd.Timestamp:
    return pd.Timestamp(datetime.now(timezone.utc)).tz_localize(None)


def log(table: str, **row) -> dict:
    row.setdefault("id", uuid.uuid4().hex)
    ts_col = "created_at" if table in ("alerts", "weather_reports") else "ts"
    row.setdefault(ts_col, now_utc())
    _BUFFER.setdefault(table, []).append(row)
    return row


def flush(store) -> None:
    for table, rows in list(_BUFFER.items()):
        if rows:
            try:
                store.write(table, pd.DataFrame(rows))
            except Exception as e:  # never lose the run because logging failed
                print(f"[logs] could not flush {table}: {e}")
                continue
        _BUFFER[table] = []


def peek(table: str) -> list[dict]:
    return list(_BUFFER.get(table, []))
