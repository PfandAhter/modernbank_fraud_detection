"""Train XGBoost model for transfer fraud detection.

This script trains an XGBoost classifier optimized for money transfer
fraud detection using the transfer-specific feature engineering.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.append(str(SRC_PATH))

from preprocessing.transfer_features import TRANSFER_FEATURE_COLUMNS


class TopKCategoryEncoder(BaseEstimator, TransformerMixin):
    """Keep only top-K most frequent categories; replace others with 'RARE'."""
    
    def __init__(self, top_k: int = 20):
        self.top_k = top_k
        self.top_values_: Dict[str, set] = {}

    def fit(self, X, y=None):
        X = X.copy()
        for col in X.columns:
            freq = X[col].value_counts()
            self.top_values_[col] = set(freq.head(self.top_k).index)
        return self

    def transform(self, X):
        X_ = X.copy()
        for col, top_vals in self.top_values_.items():
            X_[col] = X_[col].where(X_[col].isin(top_vals), "RARE")
        return X_


def _one_hot_encoder_kwargs() -> Dict[str, object]:
    """Return OneHotEncoder kwargs compatible with sklearn version."""
    version = tuple(int(part) for part in sklearn.__version__.split(".")[:2])
    kwargs: Dict[str, object] = {"handle_unknown": "ignore"}
    if version >= (1, 2):
        kwargs["sparse_output"] = False
    else:
        kwargs["sparse"] = False
    return kwargs


def load_transfer_dataset(csv_path: Path) -> Tuple[pd.DataFrame, pd.Series]:
    """Load and prepare fraud dataset for transfer model training.
    
    Supports both the old synthetic format and the new realistic format.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    # Detect dataset format based on column names
    if "transaction_amount" in df.columns:
        # New realistic format - already has correct column names
        features = df[TRANSFER_FEATURE_COLUMNS].copy()
        y = df["fraud_label"].astype(int)
    else:
        # Old synthetic format - need to map columns
        features = pd.DataFrame({
            "transaction_amount": df["Transaction_Amount"],
            "account_balance_before": df["Account_Balance"],
            "card_age_months": df["Card_Age"],
            "card_type": df["Card_Type"],
            "avg_transaction_amount_7d": df["Avg_Transaction_Amount_7d"],
            "transaction_count_24h": df["Daily_Transaction_Count"],
            "transaction_count_7d": df["Daily_Transaction_Count"] * 3,
            "previous_fraud_flag": df["Previous_Fraudulent_Activity"],
            "is_new_receiver": (np.random.rand(len(df)) > 0.7).astype(int),
            "is_weekend": df["Is_Weekend"],
            "timestamp": df["Timestamp"],
        })
        
        # Add derived features
        avg_safe = features["avg_transaction_amount_7d"].replace(0, 1.0)
        features["amount_to_avg_ratio"] = (
            features["transaction_amount"] / avg_safe
        ).clip(upper=100.0)
        
        balance_safe = features["account_balance_before"].replace(0, 1.0)
        features["balance_drain_ratio"] = (
            features["transaction_amount"] / balance_safe
        ).clip(upper=1.0)
        
        features["velocity_24h"] = features["transaction_count_24h"] / 24.0
        features["velocity_7d"] = features["transaction_count_7d"] / 168.0
        velocity_7d_safe = features["velocity_7d"].replace(0, 0.01)
        features["velocity_burst"] = (
            features["velocity_24h"] / velocity_7d_safe
        ).clip(upper=50.0)
        
        features["is_new_card"] = (features["card_age_months"] < 3).astype(int)
        features["is_round_amount"] = (
            features["transaction_amount"] % 100 == 0
        ).astype(int)
        features["is_large_amount"] = (
            features["transaction_amount"] > 5000
        ).astype(int)
        
        # Time features
        features["timestamp"] = pd.to_datetime(features["timestamp"], errors="coerce")
        features["txn_hour"] = features["timestamp"].dt.hour.fillna(12).astype(int)
        features["txn_day_of_week"] = features["timestamp"].dt.dayofweek.fillna(0).astype(int)
        features["is_off_hours"] = (
            (features["txn_hour"] < 6) | (features["txn_hour"] > 22)
        ).astype(int)
        
        features = features.drop(columns=["timestamp"])
        features = features[TRANSFER_FEATURE_COLUMNS]
        y = df["Fraud_Label"].astype(int)
    
    return features, y


def build_transfer_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Build preprocessing pipeline for transfer features."""
    numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    transformers = []
    if numeric_cols:
        transformers.append((
            "num",
            Pipeline(steps=[
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]),
            numeric_cols,
        ))
    if categorical_cols:
        transformers.append((
            "cat",
            Pipeline(steps=[
                ("topk", TopKCategoryEncoder(top_k=20)),
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(**_one_hot_encoder_kwargs())),
            ]),
            categorical_cols,
        ))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def compute_scale_pos_weight(y: pd.Series) -> float:
    """Compute class weight for imbalanced dataset."""
    positives = float((y == 1).sum())
    negatives = float((y == 0).sum())
    if positives == 0:
        raise ValueError("No positive samples in training data.")
    return negatives / positives


def find_optimal_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Find threshold that maximizes F1 score."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_idx = np.argmax(f1_scores)
    return float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5


def train_transfer_model(data_path: Path, model_path: Path) -> Dict[str, float]:
    """Train XGBoost model for transfer fraud detection.
    
    Returns
    -------
    Dict[str, float]
        Validation metrics including AUC-PR, AUC-ROC, and optimal threshold.
    """
    print("Loading dataset...")
    X, y = load_transfer_dataset(data_path)
    
    print(f"Dataset shape: {X.shape}")
    print(f"Fraud rate: {y.mean():.2%}")
    
    # Stratified split
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_valid)}")
    
    # Build pipeline
    preprocessor = build_transfer_preprocessor(X_train)
    scale_pos_weight = compute_scale_pos_weight(y_train)
    
    print(f"Class imbalance weight: {scale_pos_weight:.2f}")
    
    classifier = XGBClassifier(
        objective="binary:logistic",
        n_estimators=400,
        learning_rate=0.03,
        max_depth=6,
        min_child_weight=3,
        subsample=0.85,
        colsample_bytree=0.85,
        gamma=0.1,
        reg_alpha=0.05,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        n_jobs=-1,
        random_state=42,
    )
    
    pipeline = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", classifier),
    ])
    
    print("Training model...")
    pipeline.fit(X_train, y_train)
    
    # Evaluate
    print("Evaluating model...")
    proba_valid = pipeline.predict_proba(X_valid)[:, 1]
    
    auc_pr = average_precision_score(y_valid, proba_valid)
    auc_roc = roc_auc_score(y_valid, proba_valid)
    optimal_threshold = find_optimal_threshold(y_valid.values, proba_valid)
    
    preds = (proba_valid >= optimal_threshold).astype(int)
    f1 = f1_score(y_valid, preds)
    
    metrics = {
        "auc_pr": auc_pr,
        "auc_roc": auc_roc,
        "optimal_threshold": optimal_threshold,
        "f1_score": f1,
    }
    
    print(f"\n{'='*50}")
    print("Validation Metrics:")
    print(f"  AUC-PR (Primary): {auc_pr:.4f}")
    print(f"  AUC-ROC: {auc_roc:.4f}")
    print(f"  Optimal Threshold: {optimal_threshold:.4f}")
    print(f"  F1 Score @ Threshold: {f1:.4f}")
    print(f"{'='*50}\n")
    
    # Save model
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    print(f"Model saved to: {model_path}")
    
    return metrics


def main() -> None:
    """Entry point for training."""
    data_path = PROJECT_ROOT / "data" / "realistic_fraud_dataset.csv"
    model_path = PROJECT_ROOT / "models" / "transfer_fraud_model.pkl"
    
    try:
        metrics = train_transfer_model(data_path, model_path)
        print("\nTraining completed successfully!")
    except Exception as exc:
        print(f"Training failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
