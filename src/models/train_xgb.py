"""Train an XGBoost classifier on processed fraud detection features."""

from __future__ import annotations

import sys
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
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR.parent) not in sys.path:
    sys.path.append(str(SCRIPT_DIR.parent))

from preprocessing.model_features import FEATURE_COLUMNS


DEFAULT_TARGET_CANDIDATES = ("is_fraud", "fraud_label", "target")


def load_features(parquet_path: Path) -> Tuple[pd.DataFrame, pd.Series]:
    """Load features and binary target vector from a parquet file."""
    if not parquet_path.exists():
        raise FileNotFoundError(f"Feature file not found: {parquet_path}")

    df = pd.read_parquet(parquet_path)

    target_column = None
    for candidate in DEFAULT_TARGET_CANDIDATES:
        if candidate in df.columns:
            target_column = candidate
            break
    if target_column is None:
        raise ValueError(
            f"Could not infer target column. Expected one of {DEFAULT_TARGET_CANDIDATES}"
        )

    y = df[target_column]
    if y.dtype == bool:
        y = y.astype(int)
    elif set(np.unique(y)) <= {0, 1}:
        y = pd.Series(y, name=target_column)
    else:
        raise ValueError(
            f"Target column '{target_column}' must be binary with values {{0, 1}} or boolean."
        )

    expected_missing = set(FEATURE_COLUMNS) - set(df.columns)
    if expected_missing:
        raise KeyError(
            "Processed dataset missing expected feature columns: "
            f"{', '.join(sorted(expected_missing))}"
        )

    X = df[FEATURE_COLUMNS]
    return X, y


def _one_hot_encoder_kwargs() -> Dict[str, object]:
    """Return OneHotEncoder keyword arguments compatible with sklearn version."""
    version = tuple(int(part) for part in sklearn.__version__.split(".")[:2])
    kwargs: Dict[str, object] = {"handle_unknown": "ignore"}
    if version >= (1, 2):
        kwargs["sparse_output"] = False
    else:  # pragma: no cover - legacy sklearn support
        kwargs["sparse"] = False
    return kwargs

class TopKCategoryEncoder(BaseEstimator, TransformerMixin):
    """Keep only the top-K most frequent categories per column; replace the rest with 'RARE'."""
    def __init__(self, top_k: int = 30):
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


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Memory-efficient preprocessing pipeline."""
    numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    transformers = []
    if numeric_cols:
        transformers.append(
            (
                "num",
                Pipeline(
                    steps=[
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric_cols,
            )
        )
    if categorical_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    steps=[
                        ("topk", TopKCategoryEncoder(top_k=30)),
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(**_one_hot_encoder_kwargs())),
                    ]
                ),
                categorical_cols,
            )
        )

    return ColumnTransformer(transformers=transformers, remainder="drop")



def compute_scale_pos_weight(y: pd.Series) -> float:
    """Compute XGBoost scale_pos_weight given binary labels."""
    positives = float((y == 1).sum())
    negatives = float((y == 0).sum())
    if positives == 0:
        raise ValueError("Training data contains no positive samples.")
    return negatives / positives if positives else 1.0


def train_model(parquet_path: Path, model_path: Path) -> None:
    """Train the XGBoost model and persist the fitted pipeline."""
    X, y = load_features(parquet_path)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    preprocessor = build_preprocessor(X_train)
    scale_pos_weight = compute_scale_pos_weight(y_train)

    classifier = XGBClassifier(
        objective="binary:logistic",
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.9,
        colsample_bytree=0.9,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        n_jobs=-1,
        random_state=42,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", classifier),
        ]
    )

    pipeline.fit(X_train, y_train)

    proba_valid = pipeline.predict_proba(X_valid)[:, 1]
    auc_pr = average_precision_score(y_valid, proba_valid)
    auc_roc = roc_auc_score(y_valid, proba_valid)

    print(f"Validation AUC-PR: {auc_pr:.4f}")
    print(f"Validation AUC-ROC: {auc_roc:.4f}")

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)


def main() -> None:
    """Entry-point for training the XGBoost fraud detection model."""
    project_root = Path(__file__).resolve().parents[2]
    features_path = project_root / "data" / "processed" / "train.parquet"
    model_path = project_root / "models" / "xgb_model.pkl"

    try:
        train_model(features_path, model_path)
    except Exception as exc:  # pragma: no cover - console output path
        print(f"Training failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
