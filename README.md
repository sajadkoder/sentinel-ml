# SentinelML - Production Fraud Detection System

## Overview

SentinelML is a production-ready, real-time fraud detection system that demonstrates:
- Strong ML fundamentals with scikit-learn
- Real-time inference via FastAPI
- PostgreSQL for transaction storage
- Redis caching for performance
- MLflow for experiment tracking
- Docker containerization
- Comprehensive monitoring and alerting

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Client    │────▶│   FastAPI   │────▶│    Redis    │
│  Requests   │     │   Gateway   │     │   Cache     │
└─────────────┘     └──────┬──────┘     └──────┬──────┘
                           │                    │
           ┌───────────────┼────────────┬───────┘
           ▼               ▼            ▼
    ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
    │  ML Model   │ │ PostgreSQL  │ │   MLflow    │
    │  Inference  │ │  Database   │ │  Tracking   │
    └─────────────┘ └─────────────┘ └─────────────┘
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start Infrastructure (Docker)

```bash
docker-compose up -d postgres redis mlflow
```

### 3. Run Training Pipeline

```bash
python scripts/train_model.py
```

### 4. Start API Server

```bash
python -m sentinel_ml.api.main
```

### 5. Make Predictions

```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_123",
    "amount": 150.00,
    "merchant_id": "merchant_001",
    "merchant_category": "online_shopping",
    "transaction_type": "purchase"
  }'
```

## Project Structure

```
sentinel_ml/
├── api/                    # FastAPI application
│   └── main.py            # API endpoints
├── config/                 # Configuration
│   └── settings.py        # Settings and config
├── core/                   # Core utilities
│   ├── cache.py           # Redis caching
│   ├── database.py        # Database models
│   └── logging.py         # Structured logging
├── data/                   # Data processing
│   └── preprocessing.py   # Data generation & preprocessing
├── features/              # Feature engineering
│   └── engineering.py     # Feature pipeline
├── models/                # Model training
│   └── trainer.py         # Training & MLflow
├── monitoring/            # Monitoring
│   └── drift.py           # Drift detection & alerting
├── pipelines/             # Training pipelines
│   └── training.py        # Retraining & A/B testing
└── tests/                 # Unit tests
    └── test_pipeline.py   # Test suite
```

## Key Features

### 1. Feature Engineering
- Time-based features (hour, day, weekend)
- User behavioral features (transaction patterns)
- Risk scores (merchant, country)
- Change detection (device, location)

### 2. Model Training
- Multiple model types (Random Forest, Gradient Boosting, Logistic Regression)
- Cross-validation
- MLflow experiment tracking
- Model versioning

### 3. Real-time Inference
- FastAPI REST endpoints
- Batch prediction support
- Response caching
- Async operations

### 4. Monitoring
- Data drift detection (KS test, PSI)
- Model performance tracking
- Alerting system
- Metrics collection

### 5. Retraining
- Scheduled retraining
- Performance-triggered retraining
- Drift-triggered retraining
- A/B testing simulation

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/api/v1/predict` | POST | Single prediction |
| `/api/v1/predict/batch` | POST | Batch predictions |
| `/api/v1/model/info` | GET | Model information |

## Configuration

Set environment variables:

```bash
# Database
export DB_HOST=localhost
export DB_PORT=5432
export DB_NAME=sentinel_ml
export DB_USER=postgres
export DB_PASSWORD=postgres

# Redis
export REDIS_HOST=localhost
export REDIS_PORT=6379

# MLflow
export MLFLOW_TRACKING_URI=http://localhost:5000
```

## Testing

```bash
# Run all tests
pytest sentinel_ml/tests/ -v

# Run with coverage
pytest sentinel_ml/tests/ --cov=sentinel_ml --cov-report=html
```

## Docker Deployment

```bash
# Build image
docker build -t sentinel-ml .

# Run container
docker run -p 8000:8000 sentinel-ml
```

## Load Testing

```bash
locust -f scripts/load_test.py --host http://localhost:8000
```

## Monitoring Dashboard

Access the monitoring metrics at `/api/v1/monitoring/metrics`

## Database Schema

### transactions
- Transaction details with fraud labels
- Indexed by user_id, merchant_id, timestamp

### model_metrics
- Model performance history
- Precision, recall, F1, AUC

### prediction_logs
- Audit trail of all predictions
- Latency and cache status

### data_drift_logs
- Drift detection results
- Feature-level drift tracking

## Scaling Considerations

1. **Horizontal Scaling**: Deploy API behind load balancer
2. **Database**: Use read replicas for queries
3. **Caching**: Increase Redis cluster size
4. **Model Serving**: Use model sharding for large models
5. **Async Processing**: Use message queues for batch predictions

## Interview Topics Covered

1. **ML Fundamentals**: Feature engineering, model selection, evaluation
2. **System Design**: Microservices, caching, database design
3. **MLOps**: Model versioning, experiment tracking, retraining
4. **Production Engineering**: Logging, monitoring, alerting
5. **Performance**: Latency optimization, batch processing

## License

MIT License
