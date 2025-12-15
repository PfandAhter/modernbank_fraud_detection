# 🔍 Fraud Detection System

A machine learning-based fraud detection system for banking money transfers. This project uses XGBoost to identify potentially fraudulent transactions in real-time.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-ML-orange.svg)
![Kafka](https://img.shields.io/badge/Kafka-Streaming-black.svg)

## 📋 Table of Contents

- [Features](#-features)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Model Architecture](#-model-architecture)
- [API Reference](#-api-reference)
- [Contributing](#-contributing)

## ✨ Features

- **Real-time Fraud Detection**: Evaluates transactions in milliseconds
- **XGBoost Classifier**: High-accuracy machine learning model with AUC-PR optimization
- **Feature Engineering**: Sophisticated derived features including:
  - Balance drain ratio detection
  - Transaction velocity analysis
  - Amount anomaly detection
  - Time-based risk patterns
- **Kafka Integration**: Stream processing for high-throughput transaction evaluation
- **REST API**: FastAPI-based endpoints for synchronous fraud checks
- **Explainability**: SHAP-based feature importance for each prediction

## 📁 Project Structure

```
fraud_detection/
├── data/                          # Dataset directory (not tracked in git)
├── models/                        # Trained model artifacts
├── src/
│   ├── models/
│   │   └── train_transfer_model.py    # Model training pipeline
│   ├── preprocessing/
│   │   ├── generate_realistic_data.py # Synthetic data generation
│   │   └── transfer_features.py       # Feature engineering
│   ├── schemas/
│   │   └── kafka_schemas.py           # Pydantic data models
│   └── serving/
│       ├── app.py                     # FastAPI application
│       ├── fraud_evaluator.py         # Core evaluation logic
│       ├── kafka_consumer.py          # Kafka stream consumer
│       └── transfer_api.py            # REST API endpoints
└── docs/                          # Documentation
```

## 🚀 Installation

### Prerequisites

- Python 3.8+
- pip or conda

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/PfandAhter/fraud-detection.git
   cd fraud-detection
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   
   # Windows
   .venv\Scripts\activate
   
   # Linux/Mac
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install pandas numpy scikit-learn xgboost joblib fastapi uvicorn pydantic
   ```

4. **Generate training data**
   ```bash
   python src/preprocessing/generate_realistic_data.py
   ```

5. **Train the model**
   ```bash
   python src/models/train_transfer_model.py
   ```

## 📖 Usage

### Training the Model

Generate synthetic fraud data and train the XGBoost model:

```bash
# Generate 30,000 samples with 15% fraud rate
python src/preprocessing/generate_realistic_data.py

# Train the model
python src/models/train_transfer_model.py
```

### Running the API Server

```bash
python src/serving/app.py
```

The API will be available at `http://localhost:8000`

### Evaluating a Transaction

```python
from src.serving.fraud_evaluator import evaluate_fraud_risk

transaction = {
    "transaction_id": "TXN_001",
    "transaction_amount": 5000.0,
    "account_balance_before": 10000.0,
    "avg_transaction_amount_7d": 200.0,
    "transaction_count_24h": 3,
    "transaction_count_7d": 10,
    "card_age_months": 24,
    "card_type": "DEBIT",
    "is_new_receiver": True,
    "previous_fraud_flag": False,
    "is_weekend": False,
    "timestamp": "2024-01-15T14:30:00Z"
}

result = evaluate_fraud_risk(transaction)
print(f"Risk Score: {result['risk_score']}")
print(f"Risk Level: {result['risk_level']}")
print(f"Action: {result['recommended_action']}")
```

## 🧠 Model Architecture

### Features Used

| Feature | Description |
|---------|-------------|
| `balance_drain_ratio` | Transaction amount / Account balance |
| `amount_to_avg_ratio` | Transaction amount / 7-day average |
| `velocity_burst` | 24h velocity / 7d velocity ratio |
| `is_new_receiver` | First-time recipient flag |
| `is_new_card` | Card issued < 3 months ago |
| `is_off_hours` | Transaction between 10pm - 6am |
| `card_age_months` | Age of the card in months |
| `previous_fraud_flag` | Historical fraud indicator |

### Fraud Patterns Detected

1. **Balance Drain**: Large transactions (>70% of balance)
2. **Amount Spike**: Unusual amounts (10x-100x average)
3. **Velocity Burst**: Many transactions in short time
4. **New Card Fraud**: Suspicious activity on new cards
5. **Combined Patterns**: Multiple risk factors together

### Model Performance

| Metric | Score |
|--------|-------|
| AUC-PR | ~0.95+ |
| AUC-ROC | ~0.98+ |
| F1 Score | ~0.90+ |

## 🔌 API Reference

### POST `/api/v1/fraud/evaluate`

Evaluate a single transaction for fraud risk.

**Request Body:**
```json
{
  "transaction_id": "TXN_001",
  "transaction_amount": 5000.0,
  "account_balance_before": 10000.0,
  "avg_transaction_amount_7d": 200.0,
  "transaction_count_24h": 3,
  "transaction_count_7d": 10,
  "card_age_months": 24,
  "card_type": "DEBIT",
  "is_new_receiver": true,
  "previous_fraud_flag": false,
  "is_weekend": false,
  "timestamp": "2024-01-15T14:30:00Z"
}
```

**Response:**
```json
{
  "transaction_id": "TXN_001",
  "risk_score": 0.85,
  "risk_level": "HIGH",
  "recommended_action": "BLOCK",
  "feature_importance": {
    "balance_drain_ratio": 0.45,
    "amount_to_avg_ratio": 0.32,
    "is_new_receiver": 0.12
  }
}
```

### Risk Levels

| Risk Score | Level | Action |
|------------|-------|--------|
| 0.0 - 0.3 | LOW | ALLOW |
| 0.3 - 0.7 | MEDIUM | REVIEW |
| 0.7 - 1.0 | HIGH | BLOCK |

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License.

## 👤 Author

**Ataberk BAKIR**
- GitHub: [@PfandAhter](https://github.com/PfandAhter)

---

⭐ Star this repo if you find it helpful!
