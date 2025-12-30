"""FastAPI application for serving fraud detection predictions.

Includes Prometheus metrics and health endpoints (equivalent to Spring Actuator).
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
import uvicorn
import xgboost as xgb
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from prometheus_client import Counter, Histogram, Gauge, Info, generate_latest, CONTENT_TYPE_LATEST
from prometheus_fastapi_instrumentator import Instrumentator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.append(str(SRC_PATH))

from models.train_xgb import TopKCategoryEncoder  # noqa: E402
from preprocessing.model_features import FEATURE_COLUMNS, RAW_COLUMNS, transform_raw_transactions  # noqa: E402

sys.modules.setdefault("__main__", sys.modules[__name__])
setattr(sys.modules["__main__"], "TopKCategoryEncoder", TopKCategoryEncoder)

MODEL_PATH = PROJECT_ROOT / "models" / "xgb_model.pkl"
ENV_PATH = PROJECT_ROOT / ".env"
FLAG_THRESHOLD = 0.5


class TransactionPayload(BaseModel):
    """Raw transaction payload matching the stored historical dataset."""

    user_id: str = Field(..., description="Unique identifier for the user/account")
    amount: float = Field(..., description="Transaction amount in currency units")
    timestamp: str = Field(
        ..., description="Transaction timestamp in ISO-8601 format (UTC preferred)"
    )
    account_balance: float = Field(..., description="Account balance at transaction time")
    previous_fraudulent_activity: int = Field(
        ..., description="Historical fraud indicator (0 or 1)"
    )
    daily_transaction_count: float = Field(
        ..., description="Transactions performed by the user on that day"
    )
    avg_transaction_amount_7d: float = Field(
        ..., description="Average transaction amount over the previous 7 days"
    )
    failed_transaction_count_7d: float = Field(
        ..., description="Failed transaction count over the previous 7 days"
    )
    card_type: str = Field(..., description="Payment card type")
    card_age: float = Field(..., description="Age of the card in days")
    risk_score: float = Field(..., description="Upstream risk score")
    is_weekend: int = Field(..., description="1 if weekend transaction, else 0")

    model_config = ConfigDict(extra="forbid")


def load_pipeline() -> Any:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model artifact not found at '{MODEL_PATH}'. Train the model first.")
    return joblib.load(MODEL_PATH)


@lru_cache(maxsize=1)
def get_pipeline() -> Any:
    return load_pipeline()


def build_feature_row(payload: TransactionPayload) -> pd.DataFrame:
    payload_df = pd.DataFrame([payload.dict()])
    try:
        transformed = transform_raw_transactions(payload_df[RAW_COLUMNS])
    except ValueError as exc:
        raise ValueError(str(exc)) from exc
    return transformed[FEATURE_COLUMNS]


def compute_explanation(pipeline: Any, features_df: pd.DataFrame, probability: float) -> Dict[str, float]:
    preprocessor = pipeline.named_steps.get("preprocess")
    model = pipeline.named_steps.get("model")

    if preprocessor is None or model is None:
        return {"detail": "Explanation unavailable: pipeline missing expected steps."}

    transformed = preprocessor.transform(features_df)

    try:
        feature_names = preprocessor.get_feature_names_out()
    except AttributeError:
        feature_names = [f"f{i}" for i in range(transformed.shape[1])]

    dmatrix = xgb.DMatrix(transformed, feature_names=feature_names)

    try:
        contribs = model.get_booster().predict(dmatrix, pred_contribs=True)
    except Exception:
        importances = model.feature_importances_
        ranking = np.argsort(importances)[::-1]
        top_indices = ranking[:5]
        return {
            feature_names[i]: float(importances[i])
            for i in top_indices
            if importances[i] > 0
        } or {"detail": "Explanation unavailable: unable to compute contributions."}

    shap_values = contribs[0][:-1]  # drop bias term
    top_indices = np.argsort(np.abs(shap_values))[::-1][:5]
    explanation = {feature_names[idx]: float(shap_values[idx]) for idx in top_indices}
    if not explanation:
        explanation["detail"] = "Explanation unavailable: no informative features."
    return explanation


# ============================================================================
# Application Setup with Prometheus Metrics (Spring Actuator equivalent)
# ============================================================================

APPLICATION_NAME = "fraud-detection-service"
MODEL_VERSION = "1.0.0"
START_TIME = datetime.utcnow()

app = FastAPI(
    title="Fraud Detection Serving API",
    description="ML-based fraud detection with Prometheus metrics",
    version=MODEL_VERSION
)

# ============================================================================
# Prometheus Metrics Definition (equivalent to Micrometer metrics)
# ============================================================================

# Counter metrics - equivalent to Counter in Micrometer
PREDICTION_COUNTER = Counter(
    "fraud_predictions_total",
    "Total number of fraud predictions made",
    ["result", "application"]
)

FRAUD_FLAGGED_COUNTER = Counter(
    "fraud_flagged_total",
    "Total number of transactions flagged as fraud",
    ["application"]
)

PREDICTION_ERRORS = Counter(
    "fraud_prediction_errors_total",
    "Total number of prediction errors",
    ["error_type", "application"]
)

# Histogram metrics - equivalent to Timer in Micrometer
PREDICTION_LATENCY = Histogram(
    "fraud_prediction_duration_seconds",
    "Time spent processing fraud predictions",
    ["application"],
    buckets=[0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5]
)

RISK_SCORE_HISTOGRAM = Histogram(
    "fraud_risk_score_distribution",
    "Distribution of fraud risk scores",
    ["application"],
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

# Gauge metrics - for current state
MODEL_LOADED = Gauge(
    "fraud_model_loaded",
    "Whether the fraud detection model is loaded (1=yes, 0=no)",
    ["application"]
)

# Info metric - application metadata (like Spring info endpoint)
APP_INFO = Info(
    "fraud_detection_app",
    "Fraud Detection Application Information"
)
APP_INFO.info({
    "version": MODEL_VERSION,
    "application": APPLICATION_NAME,
    "model_type": "xgboost"
})

# Initialize Prometheus FastAPI Instrumentator (auto HTTP metrics)
instrumentator = Instrumentator(
    should_group_status_codes=True,
    should_ignore_untemplated=True,
    should_respect_env_var=True,
    should_instrument_requests_inprogress=True,
    excluded_handlers=["/metrics", "/health", "/health/liveness", "/health/readiness"],
    inprogress_name="http_requests_inprogress",
    inprogress_labels=True
)
instrumentator.instrument(app)


# ============================================================================
# Health Check Endpoints (equivalent to Spring Actuator health endpoints)
# ============================================================================

@app.get("/health")
def health_check() -> Dict[str, Any]:
    """Main health endpoint - equivalent to /actuator/health in Spring.
    
    Returns overall application health status with component details.
    Equivalent to: management.endpoint.health.show-details=always
    """
    components = {}
    overall_status = "UP"
    
    # Check model availability
    try:
        pipeline = get_pipeline()
        components["model"] = {
            "status": "UP",
            "details": {
                "path": str(MODEL_PATH),
                "loaded": True
            }
        }
        MODEL_LOADED.labels(application=APPLICATION_NAME).set(1)
    except Exception as e:
        components["model"] = {
            "status": "DOWN",
            "details": {
                "error": str(e)
            }
        }
        MODEL_LOADED.labels(application=APPLICATION_NAME).set(0)
        overall_status = "DOWN"
    
    # Disk space check
    try:
        import shutil
        disk_usage = shutil.disk_usage(PROJECT_ROOT)
        disk_free_percent = (disk_usage.free / disk_usage.total) * 100
        components["diskSpace"] = {
            "status": "UP" if disk_free_percent > 10 else "DOWN",
            "details": {
                "total": disk_usage.total,
                "free": disk_usage.free,
                "threshold": "10%"
            }
        }
    except Exception:
        components["diskSpace"] = {"status": "UNKNOWN"}
    
    return {
        "status": overall_status,
        "components": components,
        "details": {
            "application": APPLICATION_NAME,
            "version": MODEL_VERSION,
            "uptime_seconds": (datetime.utcnow() - START_TIME).total_seconds()
        }
    }


@app.get("/health/liveness")
def liveness_probe() -> Dict[str, str]:
    """Kubernetes liveness probe - equivalent to /actuator/health/liveness.
    
    Returns UP if the application is running.
    """
    return {"status": "UP"}


@app.get("/health/readiness")
def readiness_probe() -> Dict[str, Any]:
    """Kubernetes readiness probe - equivalent to /actuator/health/readiness.
    
    Returns UP if the application is ready to serve traffic.
    """
    try:
        pipeline = get_pipeline()
        return {
            "status": "UP",
            "details": {
                "model_ready": True
            }
        }
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "DOWN",
                "details": {
                    "model_ready": False,
                    "error": str(e)
                }
            }
        )


# ============================================================================
# Prometheus Metrics Endpoint (equivalent to /actuator/prometheus)
# ============================================================================

@app.get("/metrics")
def prometheus_metrics() -> Response:
    """Prometheus metrics endpoint - equivalent to /actuator/prometheus.
    
    Exposes all application metrics in Prometheus text format.
    """
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


# ============================================================================
# Application Info Endpoint (equivalent to /actuator/info)
# ============================================================================

@app.get("/info")
def app_info() -> Dict[str, Any]:
    """Application info endpoint - equivalent to /actuator/info."""
    return {
        "app": {
            "name": APPLICATION_NAME,
            "version": MODEL_VERSION,
            "description": "ML-based Fraud Detection Service"
        },
        "model": {
            "type": "XGBoost",
            "version": MODEL_VERSION,
            "threshold": FLAG_THRESHOLD
        },
        "build": {
            "timestamp": START_TIME.isoformat() + "Z"
        }
    }


# ============================================================================
# Prediction Endpoint with Metrics
# ============================================================================

@app.post("/predict")
def predict(payload: TransactionPayload) -> Dict[str, Any]:
    start_time = time.time()
    
    try:
        pipeline = get_pipeline()
    except FileNotFoundError as exc:
        PREDICTION_ERRORS.labels(error_type="model_not_found", application=APPLICATION_NAME).inc()
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        PREDICTION_ERRORS.labels(error_type="model_load_error", application=APPLICATION_NAME).inc()
        raise HTTPException(status_code=500, detail=f"Failed to load model: {exc}") from exc

    try:
        features_df = build_feature_row(payload)
    except ValueError as exc:
        PREDICTION_ERRORS.labels(error_type="validation_error", application=APPLICATION_NAME).inc()
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        PREDICTION_ERRORS.labels(error_type="feature_error", application=APPLICATION_NAME).inc()
        raise HTTPException(status_code=400, detail=f"Feature engineering failed: {exc}") from exc

    try:
        proba = pipeline.predict_proba(features_df)[0, 1]
    except Exception as exc:
        PREDICTION_ERRORS.labels(error_type="inference_error", application=APPLICATION_NAME).inc()
        raise HTTPException(status_code=400, detail=f"Inference failed: {exc}") from exc

    # Record metrics
    is_flagged = bool(proba >= FLAG_THRESHOLD)
    result_label = "fraud" if is_flagged else "normal"
    
    PREDICTION_COUNTER.labels(result=result_label, application=APPLICATION_NAME).inc()
    RISK_SCORE_HISTOGRAM.labels(application=APPLICATION_NAME).observe(proba)
    
    if is_flagged:
        FRAUD_FLAGGED_COUNTER.labels(application=APPLICATION_NAME).inc()
    
    # Record latency
    duration = time.time() - start_time
    PREDICTION_LATENCY.labels(application=APPLICATION_NAME).observe(duration)

    explanation = compute_explanation(pipeline, features_df, proba)
    return {
        "anomaly_score": float(proba),
        "is_flagged": is_flagged,
        "explain": explanation,
    }


def load_port() -> int:
    env_port = os.getenv("PORT")
    if env_port is None and ENV_PATH.exists():
        for line in ENV_PATH.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key.strip() == "PORT":
                env_port = value.strip()
                break

    if env_port is None:
        return 8000

    try:
        return int(env_port)
    except ValueError as exc:
        raise RuntimeError(f"Invalid PORT value: {env_port}") from exc


def main() -> None:
    port = load_port()
    uvicorn.run(app, host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Failed to start server: {exc}", file=sys.stderr)
        sys.exit(1)
