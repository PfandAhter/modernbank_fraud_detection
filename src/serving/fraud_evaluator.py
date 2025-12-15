"""Fraud Evaluation Service for Money Transfers.

This module provides the core fraud evaluation logic that can be used
with Kafka consumers or REST API endpoints.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

# Add src to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.append(str(SRC_PATH))

from preprocessing.transfer_features import (
    transform_single_transaction,
    TRANSFER_FEATURE_COLUMNS,
)
from schemas.kafka_schemas import (
    FraudEvaluationRequest,
    FraudEvaluationResult,
)
from models.train_transfer_model import TopKCategoryEncoder

# Register custom class for pickle deserialization
import sys
sys.modules.setdefault("__main__", sys.modules[__name__])
setattr(sys.modules["__main__"], "TopKCategoryEncoder", TopKCategoryEncoder)

MODEL_PATH = PROJECT_ROOT / "models" / "transfer_fraud_model.pkl"


class FraudEvaluator:
    """Fraud evaluator for money transfer transactions.
    
    This class encapsulates the fraud detection pipeline including
    feature engineering, model inference, and explanation generation.
    
    Attributes
    ----------
    model_path : Path
        Path to the trained model artifact.
    pipeline : Any
        Loaded sklearn pipeline with preprocessor and XGBoost model.
    """
    
    def __init__(self, model_path: Optional[Path] = None):
        """Initialize the fraud evaluator.
        
        Parameters
        ----------
        model_path : Path, optional
            Path to model artifact. Defaults to models/xgb_model.pkl.
        """
        self.model_path = model_path or MODEL_PATH
        self._pipeline = None
    
    @property
    def pipeline(self) -> Any:
        """Lazy-load the model pipeline."""
        if self._pipeline is None:
            self._pipeline = self._load_pipeline()
        return self._pipeline
    
    def _load_pipeline(self) -> Any:
        """Load the trained model pipeline from disk."""
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model artifact not found at '{self.model_path}'. "
                "Train the model first using train_transfer_model.py"
            )
        return joblib.load(self.model_path)
    
    def evaluate_transaction(
        self, 
        request: FraudEvaluationRequest
    ) -> FraudEvaluationResult:
        """Evaluate a single transaction for fraud risk.
        
        Parameters
        ----------
        request : FraudEvaluationRequest
            Transaction data from Kafka event.
        
        Returns
        -------
        FraudEvaluationResult
            Risk assessment with score, level, and recommended action.
        """
        # Build features directly matching training data format
        features_df = self._build_features(request)
        
        # Get probability prediction
        risk_score = self._predict(features_df)
        
        # Get feature importance for explainability
        importance = self._compute_explanation(features_df)
        
        # Create result with decision logic applied
        return FraudEvaluationResult.from_evaluation(
            transaction_id=request.transaction_id,
            risk_score=risk_score,
            feature_importance=importance
        )
    
    def _build_features(self, request: FraudEvaluationRequest) -> pd.DataFrame:
        """Build features matching the realistic training data format.
        
        This computes derived features identically to generate_realistic_data.py.
        """
        import numpy as np
        from datetime import datetime
        
        # Parse timestamp
        ts = pd.to_datetime(request.timestamp, utc=True)
        txn_hour = ts.hour
        txn_day_of_week = ts.dayofweek
        
        # Amount ratios - same computation as generate_realistic_data.py
        avg_safe = request.avg_transaction_amount_7d if request.avg_transaction_amount_7d > 0 else 1.0
        amount_to_avg_ratio = min(request.transaction_amount / avg_safe, 100.0)
        
        balance_safe = request.account_balance_before if request.account_balance_before > 0 else 1.0
        balance_drain_ratio = min(request.transaction_amount / balance_safe, 1.0)
        
        # Velocity features - same computation as generate_realistic_data.py
        velocity_24h = request.transaction_count_24h / 24.0
        velocity_7d = request.transaction_count_7d / 168.0
        velocity_7d_safe = velocity_7d if velocity_7d > 0 else 0.01
        velocity_burst = min(velocity_24h / velocity_7d_safe, 50.0)
        
        # Binary indicators - same computation as generate_realistic_data.py
        is_new_card = 1 if request.card_age_months < 3 else 0
        is_round_amount = 1 if request.transaction_amount % 100 == 0 else 0
        is_large_amount = 1 if request.transaction_amount > 5000 else 0
        is_off_hours = 1 if (txn_hour < 6 or txn_hour > 22) else 0
        
        # Build feature DataFrame matching TRANSFER_FEATURE_COLUMNS order
        # Note: no raw amounts, only ratios for scale-independence
        features = pd.DataFrame([{
            # Ratio features
            "amount_to_avg_ratio": amount_to_avg_ratio,
            "balance_drain_ratio": balance_drain_ratio,
            "velocity_24h": velocity_24h,
            "velocity_7d": velocity_7d,
            "velocity_burst": velocity_burst,
            # Card features  
            "card_age_months": request.card_age_months,
            "card_type": request.card_type,
            # Risk indicators
            "is_new_receiver": int(request.is_new_receiver),
            "is_new_card": is_new_card,
            "is_round_amount": is_round_amount,
            "is_large_amount": is_large_amount,
            "is_off_hours": is_off_hours,
            "previous_fraud_flag": int(request.previous_fraud_flag),
            "is_weekend": int(request.is_weekend),
            # Time features
            "txn_hour": txn_hour,
            "txn_day_of_week": txn_day_of_week,
        }])
        
        return features
    
    def evaluate_batch(
        self, 
        requests: list[FraudEvaluationRequest]
    ) -> list[FraudEvaluationResult]:
        """Evaluate multiple transactions for fraud risk.
        
        Parameters
        ----------
        requests : list[FraudEvaluationRequest]
            Batch of transaction data.
        
        Returns
        -------
        list[FraudEvaluationResult]
            List of risk assessments.
        """
        return [self.evaluate_transaction(req) for req in requests]
    
    def _predict(self, features_df: pd.DataFrame) -> float:
        """Get fraud probability from model.
        
        Parameters
        ----------
        features_df : pd.DataFrame
            Transformed features ready for model.
        
        Returns
        -------
        float
            Probability of fraud (0-1).
        """
        try:
            proba = self.pipeline.predict_proba(features_df)[0, 1]
            return float(proba)
        except Exception as exc:
            raise RuntimeError(f"Model inference failed: {exc}") from exc
    
    def _compute_explanation(
        self, 
        features_df: pd.DataFrame,
        top_k: int = 5
    ) -> Dict[str, float]:
        """Compute feature importance for explainability.
        
        Parameters
        ----------
        features_df : pd.DataFrame
            Transformed features.
        top_k : int
            Number of top features to return.
        
        Returns
        -------
        Dict[str, float]
            Top contributing features and their importance scores.
        """
        try:
            preprocessor = self.pipeline.named_steps.get("preprocess")
            model = self.pipeline.named_steps.get("model")
            
            if preprocessor is None or model is None:
                return {"detail": "Explanation unavailable"}
            
            # Transform features
            transformed = preprocessor.transform(features_df)
            
            # Get feature names
            try:
                feature_names = preprocessor.get_feature_names_out()
            except AttributeError:
                feature_names = [f"f{i}" for i in range(transformed.shape[1])]
            
            # Get SHAP values if available
            dmatrix = xgb.DMatrix(transformed, feature_names=feature_names)
            
            try:
                contribs = model.get_booster().predict(dmatrix, pred_contribs=True)
                shap_values = contribs[0][:-1]  # Exclude bias term
                top_indices = np.argsort(np.abs(shap_values))[::-1][:top_k]
                return {
                    feature_names[idx]: round(float(shap_values[idx]), 4) 
                    for idx in top_indices
                }
            except Exception:
                # Fallback to feature importances
                importances = model.feature_importances_
                top_indices = np.argsort(importances)[::-1][:top_k]
                return {
                    feature_names[i]: round(float(importances[i]), 4)
                    for i in top_indices
                    if importances[i] > 0
                }
        except Exception:
            return {"detail": "Explanation computation failed"}


# Singleton instance for reuse
_evaluator: Optional[FraudEvaluator] = None


def get_evaluator() -> FraudEvaluator:
    """Get or create the fraud evaluator singleton."""
    global _evaluator
    if _evaluator is None:
        _evaluator = FraudEvaluator()
    return _evaluator


def evaluate_fraud_risk(transaction_data: dict) -> dict:
    """Convenience function for evaluating a single transaction.
    
    Parameters
    ----------
    transaction_data : dict
        Raw transaction data as dictionary.
    
    Returns
    -------
    dict
        Fraud evaluation result as dictionary.
    """
    request = FraudEvaluationRequest(**transaction_data)
    evaluator = get_evaluator()
    result = evaluator.evaluate_transaction(request)
    return result.model_dump(by_alias=True)


__all__ = [
    "FraudEvaluator",
    "get_evaluator",
    "evaluate_fraud_risk",
]
