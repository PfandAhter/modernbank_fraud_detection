"""FastAPI application for serving fraud detection predictions."""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
import uvicorn
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

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


app = FastAPI(title="Fraud Detection Serving API")


@app.post("/predict")
def predict(payload: TransactionPayload) -> Dict[str, Any]:
    try:
        pipeline = get_pipeline()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {exc}") from exc

    try:
        features_df = build_feature_row(payload)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Feature engineering failed: {exc}") from exc

    try:
        proba = pipeline.predict_proba(features_df)[0, 1]
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Inference failed: {exc}") from exc

    explanation = compute_explanation(pipeline, features_df, proba)
    return {
        "anomaly_score": float(proba),
        "is_flagged": bool(proba >= FLAG_THRESHOLD),
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
