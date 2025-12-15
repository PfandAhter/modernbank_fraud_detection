
# Fraud Detection Prediction API

## Overview

- **Endpoint**: `POST /predict`
- **Purpose**: Score a single transaction and return a fraud likelihood alongside top contributing features.
- **Authentication**: None (add according to your deployment requirements).
- **Content-Type**: `application/json`

## Request Schema

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `user_id` | string | ✓ | Unique identifier for the customer / account |
| `amount` | number | ✓ | Transaction amount in currency units |
| `timestamp` | string | ✓ | ISO-8601 timestamp (e.g., `"2023-08-14T19:30:00Z"`) |
| `account_balance` | number | ✓ | Account balance at the time of the transaction |
| `previous_fraudulent_activity` | integer | ✓ | Historical fraud indicator (0/1) |
| `daily_transaction_count` | number | ✓ | User's transaction count for the current day |
| `avg_transaction_amount_7d` | number | ✓ | Average transaction amount over the past 7 days |
| `failed_transaction_count_7d` | number | ✓ | Failed transaction count over the past 7 days |
| `card_type` | string | ✓ | Payment card type (e.g., `"Visa"`, `"Mastercard"`) |
| `card_age` | number | ✓ | Age of the card in days |
| `risk_score` | number | ✓ | Upstream risk score (0-1 range in synthetic data) |
| `is_weekend` | integer | ✓ | `1` if the transaction occurs on a weekend, else `0` |

> **Note**: The service now performs all model-side feature engineering (time-derived fields, encoding, scaling) internally. Send only the raw transaction attributes listed above.

### Example Request Body

```json
{
  "user_id": "USER_1834",
  "amount": 125.55,
  "timestamp": "2023-08-14T19:30:00Z",
  "account_balance": 93213.17,
  "previous_fraudulent_activity": 0,
  "daily_transaction_count": 7,
  "avg_transaction_amount_7d": 437.63,
  "failed_transaction_count_7d": 3,
  "card_type": "Amex",
  "card_age": 65,
  "risk_score": 0.84,
  "is_weekend": 0
}
```

## Response Schema

```json
{
  "anomaly_score": 0.4123,
  "is_flagged": false,
  "explain": {
    "feature_name": 0.1234,
    "...": -0.0567
  }
}
```

| Field | Type | Description |
| --- | --- | --- |
| `anomaly_score` | number | Probability of fraud produced by the model (0-1). |
| `is_flagged` | boolean | Indicates whether the transaction exceeds the configured threshold (`0.5` by default). |
| `explain` | object | Top SHAP feature contributions explaining the score. Keys are transformed feature names; values are contribution magnitudes. |

## Running the Service

1. Ensure the trained pipeline artifact exists at `models/xgb_model.pkl`.
2. Optionally edit `.env` to set `PORT=<desired_port>` (default `8000`).
3. Start the server:
   ```bash
   python -m uvicorn serving.app:app --host 0.0.0.0 --port %PORT%
   ```
   (The script also calls `uvicorn` directly if you run `python src/serving/app.py`.)
4. Send a request:
   ```bash
   curl -X POST http://localhost:8000/predict \
        -H "Content-Type: application/json" \
        -d '{ ... }'
   ```
