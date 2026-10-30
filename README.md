# Greenko Generation Forecasting & Variance Explanation Agent

A runnable baseline application for renewable generation prediction and forecast-versus-actual variance analysis.

## Features
- ## Features

- Loads the bundled `data/generation_forecast.csv` dataset.
- Trains and evaluates multiple machine learning regression models to estimate actual generation.
- Compares model performance with the supplied forecast benchmark.
- Uses a chronological, date-grouped holdout split.
- Exposes FastAPI endpoints for health, summary, records and interactive prediction.
- Provides a responsive HTML dashboard with forecast-versus-actual charts.
- Generates rule-based operational driver notes for irradiance, wind speed, temperature and outages..

## Run locally

Python 3.10+ is recommended.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
uvicorn backend.app:app --reload
```

Open http://127.0.0.1:8000

API documentation: http://127.0.0.1:8000/docs

## API endpoints
- `GET /health` - health check
- `GET /api/demo` - project status and evaluation
- `GET /api/summary` - dataset KPIs and holdout metrics
- `GET /api/records?limit=60` - generation observations
- `POST /api/run` - estimate actual generation and explain variance

Example request:

```json
{
  "irradiance_wm2": 650,
  "wind_speed_ms": 7.2,
  "temperature_c": 33,
  "outage_flag": 0,
  "forecast_mwh": 55
}
```

## Methodology

The target variable is `actual_mwh`. Input features include irradiance, wind speed, temperature, outage flag and the supplied `forecast_mwh`.

The application evaluates multiple regression models using a chronological, date-grouped holdout split. Linear Regression is the currently selected model in the dashboard.

The supplied forecast is also evaluated as a benchmark using the same test observations.

The supplied forecast benchmark currently performs better than the selected Linear Regression model on the holdout dataset. Therefore, the results should be treated as exploratory and not as proof of improved forecasting accuracy..

Metrics:
- MAE: mean absolute error in MWh.
- RMSE: root mean squared error in MWh.
- R²: coefficient of determination (interpret cautiously on a small dataset).

Median imputation is included in a pipeline for resilience to future missing feature values.

## Variance explanation design
The agent compares the model-estimated actual generation with the provided forecast and reports the signed difference and percentage. It then surfaces simple diagnostic signals:
- outage flag
- relatively low/high irradiation
- relatively low/high wind speed
- high ambient temperature

These signals are **heuristics**, not causal attribution and not an LLM-generated conclusion. The output explicitly recommends validation against SCADA, plant alarms, curtailment, asset availability and site-specific engineering information.

## Assumptions and limitations
1. The supplied data is small (60 rows) and synthetic/demo-oriented; it is not sufficient to establish production-grade accuracy.
2. There are 30 unique dates represented by 60 rows. The dataset has no asset/site identifier, so the two observations per date cannot be interpreted as distinct plants with certainty.
3. The input combines solar- and wind-related weather features without identifying generation technology or plant capacity.
4. `forecast_mwh` is used as an input feature, and the provided forecast is also used as a benchmark. This is a correction/actual-estimation baseline, not a pure day-ahead forecast model.
5. A date-grouped chronological holdout is used to reduce leakage from duplicate dates. The small number of unique dates means metrics have high uncertainty.
6. Driver rules use broad thresholds for demonstration. They must be calibrated to site-specific operating ranges and power curves.
7. No external LLM is required. This keeps the application reproducible and avoids hard-coded API keys.

## Production-readiness gaps
- Collect a larger time-series dataset with plant ID, installed capacity, generation technology and timestamp granularity.
- Add asset availability, curtailment, grid constraints, maintenance and SCADA alarm data.
- Compare persistence, seasonal naive, gradient boosting and technology-specific models using rolling-origin backtesting.
- Add prediction intervals, drift monitoring and model/version tracking.
- Validate explanations with operations engineers; use causal methods only when the data supports them.
- Add authentication, request logging, observability and data governance.

## Railway deployment
1. Push this project to a GitHub repository.
2. In Railway, create a new project and deploy from the GitHub repository.
3. Railway detects the `Procfile`; it starts `uvicorn backend.app:app --host 0.0.0.0 --port $PORT`.
4. Confirm the service is healthy at `/health` and open the root URL for the dashboard.
5. Keep the dataset committed in the repository for this demo. Do not commit secrets or credentials.

## Suggested demo flow
1. Open the dashboard and explain the dataset KPIs.
2. Review the MAE/RMSE and compare the model with the supplied forecast benchmark.
3. Submit sample weather and outage values.
4. Explain that the variance is model-estimated and that driver notes are transparent operational signals, not proof of causality.
5. Show `/docs` to demonstrate the API contract.
