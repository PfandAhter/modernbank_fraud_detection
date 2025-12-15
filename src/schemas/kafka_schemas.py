"""Kafka schemas and data models for Fraud Detection Service.

This module defines Pydantic models for Kafka message validation
for the fraud-evaluation-request and fraud-evaluation-result topics.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Dict, Optional

from pydantic import BaseModel, ConfigDict, Field


class RiskLevel(str, Enum):
    """Risk classification levels."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class RecommendedAction(str, Enum):
    """Recommended actions based on risk assessment."""
    APPROVE = "APPROVE"
    HOLD = "HOLD"
    BLOCK = "BLOCK"


class FraudEvaluationRequest(BaseModel):
    """Schema for fraud-evaluation-request Kafka topic.
    
    This matches the money transfer transaction events from the Transaction Service.
    """
    
    transaction_id: str = Field(
        ..., 
        alias="transactionId",
        description="Unique transaction identifier"
    )
    user_id: str = Field(
        ..., 
        alias="userId",
        description="User/account identifier"
    )
    transaction_amount: float = Field(
        ..., 
        alias="transactionAmount",
        ge=0,
        description="Transfer amount in currency units"
    )
    transaction_type: str = Field(
        default="TRANSFER",
        alias="transactionType",
        description="Transaction type (always TRANSFER for this service)"
    )
    merchant_category: str = Field(
        default="P2P_TRANSFER",
        alias="merchantCategory",
        description="Merchant/transfer category"
    )
    card_type: str = Field(
        ...,
        alias="cardType",
        description="Payment card type (DEBIT/CREDIT)"
    )
    card_age_months: int = Field(
        ...,
        alias="cardAgeMonths",
        ge=0,
        description="Age of the card in months"
    )
    account_balance_before: float = Field(
        ...,
        alias="accountBalanceBefore",
        ge=0,
        description="Account balance before transaction"
    )
    avg_transaction_amount_7d: float = Field(
        ...,
        alias="avgTransactionAmount7d",
        ge=0,
        description="User's 7-day average transaction amount"
    )
    transaction_count_24h: int = Field(
        ...,
        alias="transactionCount24h",
        ge=0,
        description="User's transaction count in last 24 hours"
    )
    transaction_count_7d: int = Field(
        ...,
        alias="transactionCount7d",
        ge=0,
        description="User's transaction count in last 7 days"
    )
    previous_fraud_flag: bool = Field(
        ...,
        alias="previousFraudFlag",
        description="Historical fraud indicator"
    )
    is_new_receiver: bool = Field(
        ...,
        alias="isNewReceiver",
        description="First-time transfer to this recipient"
    )
    is_weekend: bool = Field(
        ...,
        alias="isWeekend",
        description="Weekend transaction indicator"
    )
    timestamp: str = Field(
        ...,
        description="Transaction timestamp in ISO-8601 format"
    )
    
    model_config = ConfigDict(
        populate_by_name=True,
        extra="ignore"
    )
    
    def to_feature_dict(self) -> dict:
        """Convert to dictionary format expected by feature engineering."""
        return {
            "transaction_id": self.transaction_id,
            "user_id": self.user_id,
            "transaction_amount": self.transaction_amount,
            "transaction_type": self.transaction_type,
            "merchant_category": self.merchant_category,
            "card_type": self.card_type,
            "card_age_months": self.card_age_months,
            "account_balance_before": self.account_balance_before,
            "avg_transaction_amount_7d": self.avg_transaction_amount_7d,
            "transaction_count_24h": self.transaction_count_24h,
            "transaction_count_7d": self.transaction_count_7d,
            "previous_fraud_flag": self.previous_fraud_flag,
            "is_new_receiver": self.is_new_receiver,
            "is_weekend": self.is_weekend,
            "timestamp": self.timestamp,
        }


class FraudEvaluationResult(BaseModel):
    """Schema for fraud-evaluation-result Kafka topic.
    
    This is the decision output published after fraud evaluation.
    """
    
    transaction_id: str = Field(
        ...,
        alias="transactionId",
        description="Original transaction identifier"
    )
    risk_score: float = Field(
        ...,
        alias="riskScore",
        ge=0.0,
        le=1.0,
        description="Fraud probability score (0-1)"
    )
    risk_level: RiskLevel = Field(
        ...,
        alias="riskLevel",
        description="Risk classification (LOW/MEDIUM/HIGH)"
    )
    recommended_action: RecommendedAction = Field(
        ...,
        alias="recommendedAction",
        description="Recommended action (APPROVE/HOLD/BLOCK)"
    )
    evaluated_at: str = Field(
        ...,
        alias="evaluatedAt",
        description="Evaluation timestamp in ISO-8601 format"
    )
    feature_importance: Optional[Dict[str, float]] = Field(
        default=None,
        alias="featureImportance",
        description="Top contributing features to the decision"
    )
    
    model_config = ConfigDict(
        populate_by_name=True,
        use_enum_values=True
    )
    
    @classmethod
    def from_evaluation(
        cls,
        transaction_id: str,
        risk_score: float,
        feature_importance: Optional[Dict[str, float]] = None
    ) -> "FraudEvaluationResult":
        """Create result from risk score applying decision logic.
        
        Decision Policy:
        - riskScore < 0.30 → LOW → APPROVE
        - 0.30 ≤ riskScore < 0.70 → MEDIUM → HOLD
        - riskScore ≥ 0.70 → HIGH → BLOCK
        """
        if risk_score < 0.30:
            risk_level = RiskLevel.LOW
            action = RecommendedAction.APPROVE
        elif risk_score < 0.70:
            risk_level = RiskLevel.MEDIUM
            action = RecommendedAction.HOLD
        else:
            risk_level = RiskLevel.HIGH
            action = RecommendedAction.BLOCK
        
        return cls(
            transaction_id=transaction_id,
            risk_score=round(risk_score, 4),
            risk_level=risk_level,
            recommended_action=action,
            evaluated_at=datetime.utcnow().isoformat() + "Z",
            feature_importance=feature_importance
        )


__all__ = [
    "RiskLevel",
    "RecommendedAction",
    "FraudEvaluationRequest",
    "FraudEvaluationResult",
]
