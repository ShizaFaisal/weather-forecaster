# Project report – 24 h weather & air-quality forecaster

Dashboard: <PASTE PUBLIC URL> · Repo: <PASTE URL> · City: Karachi (24.86, 67.00)

> Fill the *italic* placeholders with numbers from your own runs (dashboard → Monitoring tab, or the
> `model_metrics`, `drift_log`, `agent_calls`, `failure_log` feature groups).

## 1. Data sources
OpenWeather current weather + air pollution (PM2.5 converted to US AQI with EPA 2024 breakpoints); AQICN geo feed
(station AQI, PM2.5/PM10); Open-Meteo archive/forecast/air-quality (hourly temperature, humidity, pressure, precipitation,
wind, AQI, PM, UV – the only source with hourly history and UV). Fusion rules: see README. *Coverage achieved: X days, Y% rows with ≥2 sources.*

## 2. Imputation
Strategy and rationale in `src/fusion.py`. *Share of imputed cells: X%; longest gap: Y h; effect on validation RMSE (with vs without): Z.*

## 3. Model choices
Five candidates (naive baseline + SARIMAX, Ridge, Random Forest, LSTM), chronological 80/20 split with 24 h purge, RMSE/MAE/R².
Per-target selection of the best family; promotion by champion/challenger re-scoring.
*Table: model × metric (RMSE/MAE/R²) with a column “beats naive?” – copy from the dashboard.*
*Discussion: which family wins where (expect tree/linear models for AQI, LSTM competitive for temperature, naive hard to beat for precipitation).*

## 4. Explainability
SHAP (TreeExplainer on RF, stored per run) and LIME (on demand). *Top drivers per metric and whether they make physical sense
(e.g. 24 h lag and diurnal harmonics for temperature, wind speed and PM trends for AQI).*

## 5. Issues and fixes
| Issue | Fix |
|---|---|
| Rows with NaN lag/target in training | Drop first 24 h (no lag) and newest 24 h (no target); newest 24 h kept for forecasting |
| Wind direction is circular | Predict sin/cos, convert with atan2, evaluate with angular MAE |
| Train/live source mismatch | Drift Agent + daily retraining; fused rows accumulate in the store |
| Hourly rows revised by sources | Fresh data replaces stored rows only for new or previously imputed hours |
| LLM failures / invalid JSON | 2 retries × 2 models, JSON/length validation, template fallback, every attempt logged |
| *Add your own (API quotas, Hopsworks latency, cron delays …)* | |

## 6. Agent performance
* **Alert quality** – rules are deterministic (`src/rules.py`). *Alerts raised: N; share worded by LLM vs template; spot-check of 10 alerts for correctness; false alarm / missed hazard review against observations.*
* **Drift detection** – *events: data N, model M; retrains triggered K; did retraining improve the next validation score?*
* **LLM call failures** – *calls: N; success rate: X%; median latency: Y ms; failure causes (429, invalid output, timeout); fallback rate.*
