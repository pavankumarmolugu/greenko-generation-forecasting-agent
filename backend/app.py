from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "generation_forecast.csv"
FRONTEND_DIR = BASE_DIR / "frontend"

FEATURES = [
    "irradiance_wm2", "wind_speed_ms", "temperature_c",
    "outage_flag", "forecast_mwh"
]
TARGET = "actual_mwh"

app = FastAPI(
    title="Greenko Generation Forecasting & Variance Agent",
    description="Baseline renewable generation prediction and evidence-based variance analysis.",
    version="1.0.0",
)

class RunRequest(BaseModel):
    irradiance_wm2: float = Field(ge=0, le=1500)
    wind_speed_ms: float = Field(ge=0, le=100)
    temperature_c: float = Field(ge=-30, le=80)
    outage_flag: int = Field(ge=0, le=1)
    forecast_mwh: float = Field(ge=0, le=100000)

def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    required = ["date", *FEATURES, TARGET]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in FEATURES + [TARGET]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["date", TARGET]).sort_values("date").reset_index(drop=True)
    if len(df) < 10:
        raise ValueError("At least 10 valid rows are required to train the baseline.")
    return df

def train_and_evaluate(df: pd.DataFrame):
    
    dates = np.array(sorted(df["date"].dropna().unique()))

    split_idx = max(1, int(len(dates) * 0.8))
    if split_idx >= len(dates):
        split_idx = len(dates) - 1

    train_dates = set(dates[:split_idx])
    test_dates = set(dates[split_idx:])

    train = df[df["date"].isin(train_dates)]
    test = df[df["date"].isin(test_dates)]

    if len(test) == 0 or len(train) < 5:
        raise ValueError("Not enough distinct dates for a date-based split.")

    candidates = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=250, max_depth=5,
            min_samples_leaf=2, random_state=42
        ),
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=300, max_depth=5,
            min_samples_leaf=2, random_state=42
        ),
        "Histogram Gradient Boosting": HistGradientBoostingRegressor(
            max_iter=150, max_leaf_nodes=7,
            l2_regularization=1.0, random_state=42
        ),
    }

    results = []
    trained_models = {}

    for name, estimator in candidates.items():
        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("regressor", estimator)
        ])

        pipeline.fit(train[FEATURES], train[TARGET])
        predictions = pipeline.predict(test[FEATURES])

        results.append({
            "name": name,
            "mae": float(mean_absolute_error(test[TARGET], predictions)),
            "rmse": float(np.sqrt(mean_squared_error(test[TARGET], predictions)))
        })
        trained_models[name] = pipeline

    results.sort(key=lambda item: item["mae"])
    best_name = results[0]["name"]
    best_model = trained_models[best_name]

    predictions = best_model.predict(test[FEATURES])
    benchmark = test["forecast_mwh"].to_numpy()

    metrics = {
        "model_name": best_name,
        "model_mae": float(mean_absolute_error(test[TARGET], predictions)),
        "model_rmse": float(np.sqrt(mean_squared_error(test[TARGET], predictions))),
        "model_r2": float(r2_score(test[TARGET], predictions)) if len(test) > 1 else None,
        "benchmark_mae": float(mean_absolute_error(test[TARGET], benchmark)),
        "benchmark_rmse": float(np.sqrt(mean_squared_error(test[TARGET], benchmark))),
        "candidate_results": results,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_unique_dates": int(len(train_dates)),
        "test_unique_dates": int(len(test_dates)),
    }

    return best_model, metrics, best_name

DATA = load_data()
MODEL, METRICS, MODEL_NAME = train_and_evaluate(DATA)

def explain_variance(payload: dict, predicted_mwh: float) -> dict:
    forecast = float(payload["forecast_mwh"])
    variance = predicted_mwh - forecast
    pct = (variance / forecast * 100) if forecast else None

    evidence = []
    # These are transparent diagnostic heuristics, not causal estimates.
    if payload["outage_flag"] == 1:
        evidence.append({
            "driver": "Outage flag",
            "direction": "Potentially lower generation",
            "detail": "An outage is flagged. Check asset availability, downtime duration and affected capacity."
        })
    if payload["irradiance_wm2"] < 450:
        evidence.append({
            "driver": "Low irradiation",
            "direction": "Potentially lower solar generation",
            "detail": "Irradiation is relatively low in this input. Compare with site-specific expected irradiation."
        })
    elif payload["irradiance_wm2"] > 800:
        evidence.append({
            "driver": "High irradiation",
            "direction": "Potentially supports higher solar generation",
            "detail": "Irradiation is relatively high. Confirm that plant availability and inverter performance allow this resource to be converted."
        })
    if payload["wind_speed_ms"] < 4:
        evidence.append({
            "driver": "Low wind speed",
            "direction": "Potentially lower wind generation",
            "detail": "Wind speed is low relative to this dataset's operating range; turbine power curves are needed for a reliable estimate."
        })
    elif payload["wind_speed_ms"] > 9:
        evidence.append({
            "driver": "Higher wind speed",
            "direction": "Potentially supports higher wind generation",
            "detail": "Wind speed is relatively high. Check turbine-specific power curves and cut-out limits."
        })
    if payload["temperature_c"] > 38:
        evidence.append({
            "driver": "High temperature",
            "direction": "Possible efficiency reduction",
            "detail": "High ambient temperature can reduce PV module efficiency and may increase thermal constraints."
        })
    if not evidence:
        evidence.append({
            "driver": "No strong rule-based signal",
            "direction": "Uncertain",
            "detail": "The supplied variables do not trigger the simple diagnostic rules. Review plant-level alarms, curtailment, soiling and data quality."
        })

    if variance > 0.5:
        summary = "The model estimate is above the supplied forecast."
    elif variance < -0.5:
        summary = "The model estimate is below the supplied forecast."
    else:
        summary = "The model estimate is close to the supplied forecast."

    return {
        "forecast_mwh": round(forecast, 2),
        "predicted_actual_mwh": round(float(predicted_mwh), 2),
        "variance_mwh": round(float(variance), 2),
        "variance_percent": round(float(pct), 2) if pct is not None else None,
        "summary": summary,
        "drivers": evidence,
        "method_note": "Driver notes are transparent heuristic signals, not causal attribution. Validate against SCADA, asset alarms, curtailment and site-specific engineering data."
    }

@app.get("/health")
def health():
    return {"status": "ok", "service": "greenko-generation-agent"}

@app.get("/api/demo")
def demo():
    return {
        "case_study": "Power Generation Forecasting & Variance Explanation Agent",
        "status": "running",
        "records": int(len(DATA)),
        "unique_dates": int(DATA["date"].nunique()),
        "metrics": METRICS
    }

@app.get("/api/summary")
def summary():
    return {
        "records": int(len(DATA)),
        "unique_dates": int(DATA["date"].nunique()),
        "average_forecast_mwh": round(float(DATA["forecast_mwh"].mean()), 2),
        "average_actual_mwh": round(float(DATA["actual_mwh"].mean()), 2),
        "average_variance_mwh": round(float((DATA["actual_mwh"] - DATA["forecast_mwh"]).mean()), 2),
        "outage_records": int(DATA["outage_flag"].sum()),
        "metrics": METRICS
    }

@app.get("/api/records")
def records(limit: int = 60):
    limit = max(1, min(limit, 500))
    result = DATA.tail(limit).copy()
    result["date"] = result["date"].dt.strftime("%Y-%m-%d")
    result["variance_mwh"] = result["actual_mwh"] - result["forecast_mwh"]
    return result.round(2).to_dict(orient="records")

@app.post("/api/run")
def run_case(request: RunRequest):
    payload = request.model_dump()
    frame = pd.DataFrame([payload], columns=FEATURES)
    try:
        prediction = float(MODEL.predict(frame)[0])
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}")
    result = explain_variance(payload, prediction)
    result["model"] = MODEL_NAME
    return result

# Serve the dashboard and API from the same process for simple Railway deployment.
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
