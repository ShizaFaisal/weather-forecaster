"""Central configuration. Everything city/threshold specific is overridable via env vars."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# ---- City ---------------------------------------------------------------
CITY = os.getenv("CITY_NAME", "Karachi")
LAT = float(os.getenv("CITY_LAT", "24.8607"))
LON = float(os.getenv("CITY_LON", "67.0011"))
TZ = os.getenv("CITY_TZ", "Asia/Karachi")

# ---- Forecast definition ------------------------------------------------
HORIZON = 24  # hours ahead

# Base observation columns that become both inputs and (shifted) targets.
# Six user-facing metrics: AQI, temperature, precipitation (mm + rain prob),
# humidity, UV index, wind (speed + direction).
BASE_COLS = ["aqi", "temperature", "humidity", "uv_index",
             "precip_mm", "rain_flag", "wind_speed", "wind_dir_sin", "wind_dir_cos"]
EXTRA_OBS_COLS = ["pm2_5", "pm10", "pressure"]            # inputs only
TARGETS = [f"target_{c}" for c in BASE_COLS]
TARGET_BASE = dict(zip(TARGETS, BASE_COLS))

# forecast-name -> observation column (for the Forecast Ledger / drift checks)
LEDGER_METRICS = {
    "aqi": "aqi", "temperature": "temperature", "humidity": "humidity",
    "uv_index": "uv_index", "precip_mm": "precip_mm", "rain_prob": "rain_flag",
    "wind_speed": "wind_speed",
}
METRIC_LABELS = {
    "aqi": "AQI (US)", "temperature": "Temperature (°C)", "humidity": "Humidity (%)",
    "uv_index": "UV Index", "precip_mm": "Precipitation (mm/h)",
    "rain_prob": "Rain probability", "wind_speed": "Wind speed (m/s)",
    "wind_dir": "Wind direction (°)",
}

# ---- Source fusion --------------------------------------------------------
# "mean": average the sources that have a value; "priority": first non-null in order.
FUSION = {
    "aqi": ("priority", ["aqicn", "openweather", "openmeteo"]),
    "pm2_5": ("priority", ["aqicn", "openweather", "openmeteo"]),
    "pm10": ("priority", ["aqicn", "openweather", "openmeteo"]),
    "temperature": ("mean", ["openweather", "openmeteo"]),
    "humidity": ("mean", ["openweather", "openmeteo"]),
    "pressure": ("mean", ["openweather", "openmeteo"]),
    "wind_speed": ("mean", ["openweather", "openmeteo"]),
    "wind_dir": ("priority", ["openweather", "openmeteo"]),   # circular -> no averaging
    "precip_mm": ("priority", ["openweather", "openmeteo"]),
    "uv_index": ("priority", ["openmeteo"]),
}
RAIN_THRESHOLD_MM = 0.1

# ---- Storage names --------------------------------------------------------
MODEL_NAME = "weather_forecaster"
FG_FEATURES = "weather_features"
FG_RAW = "raw_observations"
FG_FORECASTS = "forecasts"
FG_LEDGER = "forecast_ledger"
FG_AGENT_CALLS = "agent_calls"
FG_FAILURES = "failure_log"
FG_DRIFT = "drift_log"
FG_ALERTS = "alerts"
FG_REPORTS = "weather_reports"
FG_STATE = "agent_state"
FG_MODEL_METRICS = "model_metrics"
FG_EXPLAIN = "explanations"

# ---- Training -------------------------------------------------------------
VAL_FRACTION = 0.2
MIN_TRAIN_ROWS = int(os.getenv("MIN_TRAIN_ROWS", str(24 * 14)))
LSTM_WINDOW = 24
LSTM_EPOCHS = int(os.getenv("LSTM_EPOCHS", "30"))
SEED = 42

# ---- LLM ------------------------------------------------------------------
LLM_URL = os.getenv("LLM_URL", "https://api.groq.com/openai/v1/chat/completions")
LLM_MODELS = os.getenv("LLM_MODELS", "llama-3.3-70b-versatile,llama-3.1-8b-instant").split(",")
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "30"))

# ---- Drift ----------------------------------------------------------------
PSI_THRESHOLD = float(os.getenv("PSI_THRESHOLD", "0.25"))
MODEL_DRIFT_RATIO = float(os.getenv("MODEL_DRIFT_RATIO", "1.5"))
DATA_DRIFT_MIN_COLS = 3       # columns over threshold to trigger retraining
MODEL_DRIFT_MIN_METRICS = 2   # metrics over threshold to trigger retraining
DRIFT_CHECK_EVERY_H = 6
RETRAIN_COOLDOWN_H = 24
