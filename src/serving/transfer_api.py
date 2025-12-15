"""FastAPI application for Transfer Fraud Detection Service.

This module provides REST API endpoints for evaluating money transfer
transactions for fraud risk. It can be used alongside or instead of
the Kafka consumer for integration flexibility.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.append(str(SRC_PATH))

from schemas.kafka_schemas import (
    FraudEvaluationRequest,
    FraudEvaluationResult,
    RiskLevel,
    RecommendedAction,
)
from serving.fraud_evaluator import FraudEvaluator, get_evaluator

# FastAPI application
app = FastAPI(
    title="Transfer Fraud Detection API",
    description="Evaluate money transfer transactions for fraud risk",
    version="2.0.0",
)


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool


class EvaluationResponse(BaseModel):
    """API response for fraud evaluation."""
    transactionId: str
    riskScore: float = Field(..., ge=0.0, le=1.0)
    riskLevel: str
    recommendedAction: str
    evaluatedAt: str
    featureImportance: Dict[str, float] | None = None
    
    model_config = ConfigDict(extra="ignore")


@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Health check endpoint."""
    try:
        evaluator = get_evaluator()
        _ = evaluator.pipeline  # Trigger model load
        return HealthResponse(status="healthy", model_loaded=True)
    except Exception:
        return HealthResponse(status="degraded", model_loaded=False)


@app.post("/evaluate", response_model=EvaluationResponse)
def evaluate_transaction(request: FraudEvaluationRequest) -> Dict[str, Any]:
    """Evaluate a single transaction for fraud risk.
    
    This endpoint accepts a transaction event matching the Kafka schema
    and returns a fraud risk assessment.
    
    Decision Policy:
    - riskScore < 0.30 → LOW → APPROVE
    - 0.30 ≤ riskScore < 0.70 → MEDIUM → HOLD  
    - riskScore ≥ 0.70 → HIGH → BLOCK
    """
    try:
        evaluator = get_evaluator()
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Model not available: {exc}"
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load evaluator: {exc}"
        ) from exc
    
    try:
        result = evaluator.evaluate_transaction(request)
        return result.model_dump(by_alias=True)
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid request data: {exc}"
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Evaluation failed: {exc}"
        ) from exc


@app.post("/evaluate/batch")
def evaluate_batch(requests: list[FraudEvaluationRequest]) -> list[Dict[str, Any]]:
    """Evaluate multiple transactions for fraud risk.
    
    Useful for batch processing or backfill operations.
    """
    try:
        evaluator = get_evaluator()
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Model not available: {exc}"
        ) from exc
    
    try:
        results = evaluator.evaluate_batch(requests)
        return [r.model_dump(by_alias=True) for r in results]
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Batch evaluation failed: {exc}"
        ) from exc


def load_port() -> int:
    """Load server port from environment."""
    port_str = os.getenv("PORT", "8000")
    try:
        return int(port_str)
    except ValueError:
        return 8000


def main() -> None:
    """Entry point for running the API server."""
    port = load_port()
    uvicorn.run(
        "serving.transfer_api:app",
        host="0.0.0.0",
        port=port,
        reload=False,
    )


if __name__ == "__main__":
    main()
