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

# Human-readable feature name mapping for API responses
# Maps internal/sklearn feature names to user-friendly camelCase names
FEATURE_NAME_MAPPING: dict[str, str] = {
    # Ratio features
    "amount_to_avg_ratio": "amountToAvgRatio",
    "num__amount_to_avg_ratio": "amountToAvgRatio",
    "balance_drain_ratio": "balanceDrainRatio",
    "num__balance_drain_ratio": "balanceDrainRatio",
    "velocity_24h": "velocity24h",
    "num__velocity_24h": "velocity24h",
    "velocity_7d": "velocity7d",
    "num__velocity_7d": "velocity7d",
    "velocity_burst": "velocityBurst",
    "num__velocity_burst": "velocityBurst",
    # Card features
    "card_age_months": "cardAgeMonths",
    "num__card_age_months": "cardAgeMonths",
    "card_type": "cardType",
    # Risk indicators
    "is_new_receiver": "isNewReceiver",
    "num__is_new_receiver": "isNewReceiver",
    "is_new_card": "isNewCard",
    "num__is_new_card": "isNewCard",
    "is_round_amount": "isRoundAmount",
    "num__is_round_amount": "isRoundAmount",
    "is_large_amount": "isLargeAmount",
    "num__is_large_amount": "isLargeAmount",
    "is_off_hours": "isOffHours",
    "num__is_off_hours": "isOffHours",
    "previous_fraud_flag": "previousFraudFlag",
    "num__previous_fraud_flag": "previousFraudFlag",
    "is_weekend": "isWeekend",
    "num__is_weekend": "isWeekend",
    # Time features
    "txn_hour": "transactionHour",
    "num__txn_hour": "transactionHour",
    "txn_day_of_week": "transactionDayOfWeek",
    "num__txn_day_of_week": "transactionDayOfWeek",
    # Raw amount (if used)
    "transaction_amount": "transactionAmount",
    "num__transaction_amount": "transactionAmount",
    # Index-based fallback mapping (based on TRANSFER_FEATURE_COLUMNS order)
    # Numeric columns processed first by ColumnTransformer:
    # f0: amount_to_avg_ratio, f1: balance_drain_ratio, f2: velocity_24h,
    # f3: velocity_7d, f4: velocity_burst, f5: card_age_months,
    # f6: is_new_receiver, f7: is_new_card, f8: is_round_amount,
    # f9: is_large_amount, f10: is_off_hours, f11: previous_fraud_flag,
    # f12: is_weekend, f13: txn_hour, f14: txn_day_of_week
    # Then one-hot encoded card_type columns at the end
    "f0": "amountToAvgRatio",
    "f1": "balanceDrainRatio",
    "f2": "velocity24h",
    "f3": "velocity7d",
    "f4": "velocityBurst",
    "f5": "cardAgeMonths",
    "f6": "isNewReceiver",
    "f7": "isNewCard",
    "f8": "isRoundAmount",
    "f9": "isLargeAmount",
    "f10": "isOffHours",
    "f11": "previousFraudFlag",
    "f12": "isWeekend",
    "f13": "transactionHour",
    "f14": "transactionDayOfWeek",
    # One-hot encoded card_type (indices 15+)
    "f15": "cardType",
    "f16": "cardType",
    "f17": "cardType",
    "f18": "cardType",
    "f19": "cardType",
    "f20": "cardType",
}

# Register custom class for pickle deserialization
import sys
sys.modules.setdefault("__main__", sys.modules[__name__])
setattr(sys.modules["__main__"], "TopKCategoryEncoder", TopKCategoryEncoder)

MODEL_PATH = PROJECT_ROOT / "models" / "transfer_fraud_model.pkl"

# Model version - update this when retraining the model
MODEL_VERSION = "1.0.0"


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
            feature_importance=importance,
            model_version=MODEL_VERSION
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
        def _map_feature_name(name: str) -> str:
            """Map internal feature name to human-readable name."""
            # First check exact match
            if name in FEATURE_NAME_MAPPING:
                return FEATURE_NAME_MAPPING[name]
            # Check for partial match (for encoded categorical features)
            for key, readable in FEATURE_NAME_MAPPING.items():
                if key in name:
                    return readable
            # Return original if no mapping found
            return name
        
        try:
            preprocessor = self.pipeline.named_steps.get("preprocess")
            model = self.pipeline.named_steps.get("model")
            
            if preprocessor is None or model is None:
                return {"detail": "Explanation unavailable"}
            
            # Transform features
            transformed = preprocessor.transform(features_df)
            
            # Get feature names
            try:
                feature_names = list(preprocessor.get_feature_names_out())
            except AttributeError:
                feature_names = [f"f{i}" for i in range(transformed.shape[1])]
            
            # Get SHAP values if available
            dmatrix = xgb.DMatrix(transformed, feature_names=feature_names)
            
            try:
                contribs = model.get_booster().predict(dmatrix, pred_contribs=True)
                shap_values = contribs[0][:-1]  # Exclude bias term
                top_indices = np.argsort(np.abs(shap_values))[::-1][:top_k]
                return {
                    _map_feature_name(feature_names[idx]): round(float(shap_values[idx]), 2) 
                    for idx in top_indices
                }
            except Exception:
                # Fallback to feature importances
                importances = model.feature_importances_
                top_indices = np.argsort(importances)[::-1][:top_k]
                return {
                    _map_feature_name(feature_names[i]): round(float(importances[i]), 2)
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
