# 24-hour Weather & Air-Quality Forecaster (serverless ML + LangGraph agents)

Forecasts **AQI, temperature, precipitation (mm + rain probability), humidity, UV index and wind (speed + direction)**
for one city, 24 h ahead, on a fully managed stack: **Hopsworks** (Feature Store + Model Registry) · **GitHub Actions**
(scheduler) · **Streamlit** (dashboard) · **LangGraph** (agents) · **Groq** (LLM, with deterministic fallbacks).
Default city is Karachi; change `CITY_NAME / CITY_LAT / CITY_LON` (env vars or Actions variables).

```
OpenWeather ─┐                                         ┌─ Forecast Agent ─► forecasts + Forecast Ledger
AQICN ───────┼► fuse ► impute ► features ► Hopsworks ──┤  Alert Agent  (fixed rules + LLM wording)
Open-Meteo ──┘   (hourly, GH Actions)      FS          │  Report Agent (LLM news report, template fallback)
                                             ▲          └─ Drift Agent  (data + model drift ► alert team ► retrain)
backfill (history) ─────────────────────────┘                    │
training (daily + on drift) ► Model Registry (champion/challenger) ◄┘
                                   Streamlit dashboard reads everything from the Feature Store
```

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in keys, then: export $(grep -v '^#' .env | xargs)

python scripts/backfill.py --days 180      # history -> features + 24h-shifted targets
python scripts/training_pipeline.py        # SARIMAX, Ridge, RF, LSTM vs naive baseline -> registry
python scripts/feature_pipeline.py         # one hourly ingestion
python scripts/run_agents.py               # LangGraph: forecast -> alert -> report -> drift
streamlit run streamlit_app.py             # dashboard
```
**No keys? No problem for a dry run:** without `HOPSWORKS_API_KEY` everything runs against a local store
(`data/store`); `--synthetic` on backfill/feature_pipeline generates fake data so you can exercise the whole system offline.
Without an LLM key the Alert/Report agents use templates (and log the failure).

## Deployment checklist
1. **Hopsworks**: create a free project at app.hopsworks.ai → API key. Feature groups are created automatically.
2. **Repo secrets** (Settings → Secrets → Actions): `HOPSWORKS_API_KEY`, `HOPSWORKS_PROJECT`, `OPENWEATHER_API_KEY`,
   `AQICN_API_TOKEN`, `GROQ_API_KEY`; optional `TEAM_WEBHOOK_URL`, `GH_DISPATCH_TOKEN` (PAT, `actions:write`; falls back to the
   workflow token).
3. Run workflow **backfill** once, then **training** once, then let **hourly** (cron) take over. **training** also runs daily
   and is dispatched by the Drift Agent.
4. **Dashboard**: Streamlit Community Cloud → New app → this repo, main file `streamlit_app.py`, requirements file
   `requirements-dashboard.txt`, add `HOPSWORKS_API_KEY` (+ `HOPSWORKS_PROJECT`) in app secrets. (HF Spaces/Render work too.)
5. Put the dashboard URL in `docs/REPORT.md` and submit the repo URL.

## Design decisions (details in `docs/REPORT.md`)
| Topic | Choice |
|---|---|
| Sources | OpenWeather (current + air pollution), AQICN (station AQI), Open-Meteo (hourly history, recent hours, UV). |
| Fusion | Mean for temperature/humidity/pressure/wind speed; priority for AQI/PM (AQICN > OpenWeather > Open-Meteo), wind direction, precip, UV. |
| Imputation | Time-interpolation ≤ 6 h → ffill ≤ 3 h → hour-of-day median → global median; out-of-range values masked; `n_imputed` audited. |
| Features | hour/day/month (+ sin/cos), 1 h & 3 h diffs, 6 h/24 h rolling mean, 24 h rolling std, 24 h lag. |
| Targets | `obs.shift(-24)`; wind direction as sin/cos; rain probability = P(rain ≥ 0.1 mm) regressed on the 0/1 outcome. Newest 24 rows have no target and are used for forecasting only – each predicts one hour of the next 24 h. |
| Models | Naive (value 24 h ago) · SARIMAX (regression with AR(1) errors) · Ridge · Random Forest · LSTM (PyTorch). Split is chronological with a 24 h purge gap. |
| Promotion | Mean(RMSE/σ) over all targets. The *current champion is re-scored on the new validation window*; replaced only if the challenger is strictly better. |
| Agents | One LangGraph graph; shared state persisted in `agent_state`; conditional edge runs the Report Agent daily or after a new alert. |
| Drift | Data: PSI (+KS) live-7d vs training window; model: recent 24 h-lead error ÷ validation error. Retrain when ≥3 features PSI>0.25 or ≥2 metrics ratio>1.5, 24 h cooldown. |
| Logs (all in the Feature Store) | `forecast_ledger`, `agent_calls`, `failure_log`, `drift_log`, `alerts`, `weather_reports`, `model_metrics`, `explanations`. |

## Known limitations (be upfront in your report)
* Backfilled history is Open-Meteo only (re-analysis/CAMS), while live rows blend three sources → some distribution shift
  between train and live (esp. AQI). The Drift Agent will flag it; daily retraining on accumulated fused rows reduces it.
* A 24 h-ahead *direct* forecast; no intermediate horizons. The 24-point curve on the dashboard is the 24 newest rows
  each forecasting their own T+24.
* SARIMAX at prediction time reduces to intercept + exogenous terms (the AR error cannot be propagated to an unseen future).
* Rain/precipitation are zero-inflated; regression RMSE looks good but skill over naive is the number to read.
* GitHub cron can lag several minutes and is disabled after 60 days of repo inactivity.
* **Only the offline path was tested here** (synthetic data, local store; SARIMAX/LSTM/SHAP/LangGraph/Hopsworks need their
  packages and keys). Do a first real run via the workflows and read the logs.

## Tests
`pytest -q` — fusion, imputation, target alignment/no leakage, rules, PSI, LLM fallback/logging, end-to-end on synthetic data.
