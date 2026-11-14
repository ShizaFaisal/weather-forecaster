"""Alert Agent – fixed, auditable hazard rules over all six metrics (AQI, temperature, precipitation,
humidity, UV, wind). Rules decide *whether* there is a hazard; the LLM only phrases the messages."""
from __future__ import annotations

import pandas as pd

LEVELS = {"watch": 1, "warning": 2, "danger": 3}

# metric -> list of (level, label, predicate(value)->bool, aggregation, unit)
# aggregation: max / min over the next 24 h
RULES = [
    ("aqi", "max", "AQI", [(300, "danger", "Hazardous air quality"), (200, "danger", "Very unhealthy air quality"),
                           (150, "warning", "Unhealthy air quality"), (100, "watch", "Air unhealthy for sensitive groups")]),
    ("temperature", "max", "°C", [(45, "danger", "Extreme heat"), (40, "warning", "Severe heat"), (35, "watch", "Hot weather")]),
    ("temperature", "min", "°C", [(-10, "danger", "Extreme cold"), (0, "warning", "Freezing temperatures"), (5, "watch", "Cold weather")]),
    ("precip_mm", "max", "mm/h", [(15, "danger", "Torrential rain / flash-flood risk"), (7.6, "warning", "Heavy rain"), (2.5, "watch", "Moderate rain")]),
    ("rain_prob", "max", "", [(0.8, "watch", "Rain very likely")]),
    ("humidity", "max", "%", [(95, "watch", "Very high humidity")]),
    ("humidity", "min", "%", [(15, "watch", "Very dry air")]),
    ("uv_index", "max", "", [(11, "danger", "Extreme UV"), (8, "warning", "Very high UV"), (6, "watch", "High UV")]),
    ("wind_speed", "max", "m/s", [(20, "danger", "Storm-force wind"), (14, "warning", "Strong wind"), (10, "watch", "Windy")]),
]
DIRECTION_LOWER_IS_WORSE = {("temperature", "min"), ("humidity", "min")}

ADVICE = {
    "aqi": {"general": "Reduce prolonged outdoor exertion; keep windows closed if possible.",
            "sensitive_groups": "Children, older adults and people with asthma/heart disease should stay indoors and keep medication handy.",
            "outdoor_workers": "Wear a certified N95/FFP2 mask and take frequent breaks in cleaner air."},
    "temperature": {"general": "Stay hydrated and avoid the midday sun (or dress warmly in cold).",
                    "sensitive_groups": "Check on elderly neighbours, infants and people with chronic illness.",
                    "outdoor_workers": "Schedule heavy work for cooler hours, take shaded breaks, drink water regularly."},
    "precip_mm": {"general": "Expect waterlogging and slow traffic; avoid flooded roads.",
                  "sensitive_groups": "Avoid unnecessary travel; keep emergency supplies ready.",
                  "outdoor_workers": "Secure equipment, avoid low-lying work areas and electrical hazards."},
    "rain_prob": {"general": "Carry an umbrella.", "sensitive_groups": "Plan travel with extra time.",
                  "outdoor_workers": "Plan for weather-protected work."},
    "humidity": {"general": "Humidity may make conditions feel uncomfortable.",
                 "sensitive_groups": "People with respiratory conditions should take it easy.",
                 "outdoor_workers": "Increase water intake and rest breaks."},
    "uv_index": {"general": "Use SPF 30+, a hat and sunglasses.",
                 "sensitive_groups": "Keep children and fair-skinned people out of direct sun at midday.",
                 "outdoor_workers": "Cover up, reapply sunscreen every 2 hours, work in shade when possible."},
    "wind_speed": {"general": "Secure loose objects; take care when driving high-sided vehicles.",
                   "sensitive_groups": "Stay indoors during gusts; dust may aggravate breathing problems.",
                   "outdoor_workers": "Suspend work at height and crane operations if gusts rise further."},
}
AUDIENCES = ["general", "sensitive_groups", "outdoor_workers"]


def evaluate_rules(fc: pd.DataFrame) -> list[dict]:
    """fc: forecast frame with columns target_time + metrics. Returns one hazard per (metric, agg) at its highest level."""
    hazards = []
    for metric, agg, unit, tiers in RULES:
        if metric not in fc or fc[metric].isna().all():
            continue
        series = fc.set_index("target_time")[metric]
        val = series.max() if agg == "max" else series.min()
        when = series.idxmax() if agg == "max" else series.idxmin()
        for thr, level, label in tiers:                  # tiers ordered from most to least severe
            hit = val >= thr if agg == "max" else val <= thr
            if hit:
                hazards.append(dict(metric=metric, agg=agg, level=level, severity=LEVELS[level], label=label,
                                    peak_value=round(float(val), 2), threshold=thr, unit=unit,
                                    peak_time=pd.Timestamp(when).isoformat(),
                                    rule=f"{agg}({metric}) {'>=' if agg == 'max' else '<='} {thr}"))
                break
    return sorted(hazards, key=lambda h: -h["severity"])


def signature(hazards: list[dict]) -> str:
    return "|".join(sorted(f"{h['metric']}:{h['agg']}:{h['level']}" for h in hazards))
