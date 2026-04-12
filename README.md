# SentinelML

A real-time fraud detection system built with Python, FastAPI, and scikit-learn.

![Python](https://img.shields.io/badge/Python-3.11+-blue?style=flat&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-green?style=flat)
![scikit-learn](https://img.shields.io/badge/scikit--learn-orange?style=flat)
![License](https://img.shields.io/badge/License-MIT-blue?style=flat)

## Overview

SentinelML is a complete end-to-end fraud detection system. It includes everything from data generation and feature engineering to model training, API serving, and monitoring.

## Key Features

- **Real-time Inference**: Sub-100ms prediction latency with Redis caching
- **Feature Engineering**: 25+ engineered features including user behavior patterns, risk scores, and temporal features
- **Model Training**: Random Forest, Gradient Boosting, and Logistic Regression support
- **MLflow Integration**: Experiment tracking and model versioning
- **Web Dashboard**: Beautiful Apple-inspired interface for testing predictions
- **Docker Support**: Full stack deployment with PostgreSQL, Redis, MLflow, and monitoring tools

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the API

```bash
python -m uvicorn sentinel_ml.api.main:app --reload
```

### 3. Open the Dashboard

Navigate to: **http://localhost:8000**

## Web Dashboard

The built-in dashboard provides:
- Single transaction prediction with full form input
- Batch prediction for multiple transactions
- Real-time prediction history
- Live statistics tracking
- Model information display

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web dashboard |
| `/health` | GET | Service health check |
| `/api/v1/predict` | POST | Single transaction prediction |
| `/api/v1/predict/batch` | POST | Batch predictions |
| `/api/v1/model/info` | GET | Model metadata |
| `/docs` | GET | Swagger API documentation |

## Example Usage

### Single Prediction

```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_123",
    "amount": 1500.00,
    "merchant_id": "merchant_001",
    "merchant_category": "online_shopping",
    "transaction_type": "purchase",
    "device_type": "mobile",
    "location_country": "US"
  }'
```

**Response:**

```json
{
  "transaction_id": "abc-123",
  "fraud_score": 0.45,
  "prediction": false,
  "confidence": "low",
  "model_version": "v20260410_092544",
  "inference_time_ms": 65.0,
  "cached": false
}
```

### Batch Prediction

```bash
curl -X POST http://localhost:8000/api/v1/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "transactions": [
      {"user_id": "user_1", "amount": 50, "merchant_id": "m1", "merchant_category": "grocery", "transaction_type": "purchase"},
      {"user_id": "user_2", "amount": 5000, "merchant_id": "m2", "merchant_category": "electronics", "transaction_type": "purchase"}
    ]
  }'
```

## Project Structure

```
sentinel_ml/
├── api/                 # FastAPI application
├── config/              # Configuration management
├── core/                # Database, caching, logging
├── data/                # Synthetic data generation
├── features/            # Feature engineering pipeline
├── models/              # Model training & evaluation
├── monitoring/          # Drift detection & alerting
├── pipelines/           # Training & retraining pipelines
├── dashboard/            # Web UI
└── tests/               # Unit tests
```

## Running Tests

```bash
pytest sentinel_ml/tests/ -v
```

**Test Results**: 22/22 tests passing

## Docker Deployment

```bash
# Start full stack (API, PostgreSQL, Redis, MLflow)
docker-compose up -d

# Access services
# API: http://localhost:8000
# MLflow: http://localhost:5000
# Grafana: http://localhost:3000
```

## Technology Stack

| Component | Technology |
|-----------|------------|
| Web Framework | FastAPI |
| ML Library | scikit-learn |
| Database | PostgreSQL |
| Cache | Redis |
| Experiment Tracking | MLflow |
| Containerization | Docker |

## Learning Topics

This project demonstrates:
- Building ML pipelines for production
- Creating REST APIs for ML models
- Implementing caching strategies
- Database design for ML systems
- Model monitoring and drift detection
- Docker containerization

## License

MIT License - Feel free to use this for learning or as a starting point for your own projects.