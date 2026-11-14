"""LLM client with retries, model fallback, output validation and full call logging.

Every call (success or not) lands in `agent_calls` (prompt, response, latency, status);
failures additionally land in `failure_log`. Callers must supply a deterministic fallback."""
from __future__ import annotations

import json
import os
import time

import requests

from . import config as C
from .logs import log


def _validate_json(text: str, keys: list[str] | None):
    try:
        obj = json.loads(text)
    except Exception as e:
        return None, f"invalid_json: {e}"
    if keys and not all(k in obj for k in keys):
        return None, f"missing_keys: {sorted(set(keys) - set(obj))}"
    return obj, None


def call_llm(agent: str, purpose: str, system: str, prompt: str, *, json_keys: list[str] | None = None,
             min_words: int = 0, as_json: bool = False):
    """Returns (result|None, meta). result is dict when as_json else str. None => caller uses fallback."""
    key = os.getenv("LLM_API_KEY") or os.getenv("GROQ_API_KEY")
    if not key:
        _fail(agent, purpose, "", prompt, "", 0, "skipped_no_api_key", "LLM_API_KEY/GROQ_API_KEY not set")
        return None, {"status": "no_api_key"}
    for model in C.LLM_MODELS:
        for attempt in (1, 2):
            t0 = time.time()
            body = {"model": model, "temperature": 0.3, "max_tokens": 1000,
                    "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}]}
            if as_json:
                body["response_format"] = {"type": "json_object"}
            text, status, err = "", "success", None
            try:
                r = requests.post(C.LLM_URL, json=body, timeout=C.LLM_TIMEOUT,
                                  headers={"Authorization": f"Bearer {key}"})
                if r.status_code != 200:
                    status, err = f"http_{r.status_code}", r.text[:300]
                else:
                    text = r.json()["choices"][0]["message"]["content"].strip()
            except Exception as e:
                status, err = "exception", str(e)[:300]
            latency = int((time.time() - t0) * 1000)
            result = None
            if status == "success":
                if as_json:
                    result, err = _validate_json(text, json_keys)
                elif len(text.split()) < min_words:
                    err = f"too_short: {len(text.split())} words"
                else:
                    result = text
                if err:
                    status = "invalid_output"
            if result is not None:
                log("agent_calls", agent=agent, purpose=purpose, model=model, attempt=attempt, prompt=prompt,
                    response=text, latency_ms=latency, status="success", error="")
                return result, {"status": "success", "model": model, "latency_ms": latency}
            _fail(agent, purpose, model, prompt, text, latency, status, err, attempt)
            if status in ("http_429", "exception"):
                time.sleep(1.5 * attempt)
            if status == "http_401":
                return None, {"status": "auth_error"}
    log("agent_calls", agent=agent, purpose=purpose, model="template_fallback", attempt=0, prompt=prompt,
        response="", latency_ms=0, status="fallback_used", error="all LLM attempts failed")
    return None, {"status": "all_failed"}


def _fail(agent, purpose, model, prompt, response, latency, status, err, attempt=1):
    log("agent_calls", agent=agent, purpose=purpose, model=model, attempt=attempt, prompt=prompt,
        response=response, latency_ms=latency, status=status, error=err or "")
    log("failure_log", agent=agent, kind="llm_call", source=purpose, model=model, prompt=prompt[:2000],
        error=f"{status}: {err}", status=status, latency_ms=latency)
