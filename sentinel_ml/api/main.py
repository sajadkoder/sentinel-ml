"""
FastAPI application for fraud detection inference
"""
import asyncio
import time
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager

import numpy as np
from fastapi import (
    FastAPI, HTTPException, Request, Response,
    Depends, BackgroundTasks, status
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import uvicorn

from sentinel_ml.config import config
from sentinel_ml.core.logging import get_logger, LoggerAdapter
from sentinel_ml.core.database import db_manager, Transaction, PredictionLog
from sentinel_ml.core.cache import cache, CacheKeyBuilder
from sentinel_ml.models import FraudDetectionModel
from sentinel_ml.features import FeatureEngineer

logger = get_logger(__name__)


class TransactionRequest(BaseModel):
    """Request model for single transaction prediction"""
    user_id: str = Field(..., min_length=1, max_length=100)
    amount: float = Field(..., gt=0)
    merchant_id: str = Field(..., min_length=1, max_length=100)
    merchant_category: str = Field(..., min_length=1, max_length=50)
    transaction_type: str = Field(..., min_length=1, max_length=30)
    timestamp: Optional[datetime] = None
    device_id: Optional[str] = None
    device_type: Optional[str] = None
    ip_address: Optional[str] = None
    location_country: Optional[str] = None
    location_city: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    card_type: Optional[str] = None
    card_last_four: Optional[str] = None
    
    @validator('amount')
    def validate_amount(cls, v):
        if v <= 0:
            raise ValueError('Amount must be positive')
        return round(v, 2)
    
    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "user_12345",
                "amount": 150.00,
                "merchant_id": "merchant_001",
                "merchant_category": "online_shopping",
                "transaction_type": "purchase",
                "device_type": "mobile",
                "location_country": "US"
            }
        }


class BatchTransactionRequest(BaseModel):
    """Request model for batch predictions"""
    transactions: List[TransactionRequest] = Field(..., min_items=1, max_items=100)


class PredictionResponse(BaseModel):
    """Response model for prediction"""
    transaction_id: str
    fraud_score: float = Field(..., ge=0, le=1)
    prediction: bool
    confidence: str
    model_version: str
    inference_time_ms: float
    cached: bool = False
    
    class Config:
        json_schema_extra = {
            "example": {
                "transaction_id": "abc-123-def",
                "fraud_score": 0.85,
                "prediction": True,
                "confidence": "high",
                "model_version": "v20240115_143022",
                "inference_time_ms": 12.5,
                "cached": False
            }
        }


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions"""
    predictions: List[PredictionResponse]
    total_processed: int
    total_fraud_detected: int
    avg_inference_time_ms: float


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    model_version: Optional[str]
    cache_status: str
    database_status: str
    timestamp: datetime


class ModelInfoResponse(BaseModel):
    """Model information response"""
    model_type: str
    model_version: str
    feature_count: int
    training_metrics: Dict[str, float]
    last_updated: datetime


class FraudDetectionService:
    """
    Core fraud detection service.
    
    Handles:
    - Model loading and inference
    - Feature engineering
    - Caching
    - Database logging
    """
    
    def __init__(self):
        self.model: Optional[FraudDetectionModel] = None
        self.feature_engineer: Optional[FeatureEngineer] = None
        self.model_version: Optional[str] = None
        self._initialized = False
    
    async def initialize(
        self,
        model_path: Optional[str] = None,
        feature_engineer_path: Optional[str] = None
    ):
        """Initialize the service with model and feature engineer"""
        if self._initialized:
            return
        
        try:
            if model_path:
                self.model = FraudDetectionModel.load(model_path)
            else:
                self.model = FraudDetectionModel()
                self.model.build_model()
                self._fit_dummy_model()
            
            self.model_version = self.model.version
            
            if feature_engineer_path:
                self.feature_engineer = FeatureEngineer.load(feature_engineer_path)
            else:
                self.feature_engineer = FeatureEngineer()
            
            self._initialized = True
            logger.info(
                f"FraudDetectionService initialized with model version {self.model_version}"
            )
            
        except Exception as e:
            logger.error(f"Failed to initialize service: {e}")
            raise
    
    def _fit_dummy_model(self):
        """Fit model with dummy data for demo purposes"""
        X = np.random.randn(1000, 20)
        y = np.random.randint(0, 2, 1000)
        feature_names = [f"feature_{i}" for i in range(20)]
        self.model.train(X, y, feature_names)
    
    async def predict(
        self,
        transaction: TransactionRequest,
        use_cache: bool = True
    ) -> PredictionResponse:
        """
        Make fraud prediction for a transaction.
        
        Args:
            transaction: Transaction data
            use_cache: Whether to check cache first
            
        Returns:
            PredictionResponse with fraud score and prediction
        """
        if not self._initialized:
            raise RuntimeError("Service not initialized")
        
        start_time = time.perf_counter()
        transaction_id = str(uuid.uuid4())
        cached = False
        
        if use_cache:
            cache_key = CacheKeyBuilder.prediction(transaction_id)
            cached_result = await cache.get(cache_key)
            if cached_result:
                cached = True
                return PredictionResponse(**cached_result)
        
        features = self._extract_features(transaction)
        
        fraud_score = self.model.get_fraud_score(features.reshape(1, -1))[0]
        prediction = fraud_score >= config.model.fraud_threshold
        
        if fraud_score >= 0.8:
            confidence = "high"
        elif fraud_score >= 0.5:
            confidence = "medium"
        else:
            confidence = "low"
        
        inference_time = (time.perf_counter() - start_time) * 1000
        
        response = PredictionResponse(
            transaction_id=transaction_id,
            fraud_score=round(fraud_score, 4),
            prediction=prediction,
            confidence=confidence,
            model_version=self.model_version,
            inference_time_ms=round(inference_time, 2),
            cached=cached
        )
        
        if use_cache and not cached:
            await cache.set(
                CacheKeyBuilder.prediction(transaction_id),
                response.model_dump(),
                ttl=3600
            )
        
        return response
    
    def _extract_features(self, transaction: TransactionRequest) -> np.ndarray:
        """Extract features from transaction for model input"""
        features = np.zeros(len(self.feature_engineer.feature_names))
        
        feature_dict = {
            'amount': transaction.amount,
            'hour_of_day': datetime.now().hour,
            'day_of_week': datetime.now().weekday(),
            'is_weekend': 1 if datetime.now().weekday() >= 5 else 0,
            'is_night': 1 if datetime.now().hour >= 22 or datetime.now().hour <= 5 else 0,
            'amount_log': np.log1p(transaction.amount),
            'user_txn_count_1h': 0,
            'user_txn_count_24h': 0,
            'user_txn_count_7d': 0,
            'user_avg_amount_7d': transaction.amount,
            'user_std_amount_7d': 0,
            'amount_to_avg_ratio': 1.0,
            'amount_zscore': 0,
            'time_since_last_txn': 0,
            'txn_velocity_1h': 0,
            'txn_velocity_24h': 0,
            'merchant_risk_score': 0.02,
            'country_risk_score': 0.02,
            'device_change_flag': 0,
            'location_change_flag': 0,
        }
        
        for i, name in enumerate(self.feature_engineer.feature_names):
            if name in feature_dict:
                features[i] = feature_dict[name]
        
        return features
    
    async def predict_batch(
        self,
        transactions: List[TransactionRequest]
    ) -> BatchPredictionResponse:
        """Process batch of transactions"""
        predictions = []
        total_fraud = 0
        total_time = 0
        
        for txn in transactions:
            pred = await self.predict(txn)
            predictions.append(pred)
            total_time += pred.inference_time_ms
            if pred.prediction:
                total_fraud += 1
        
        return BatchPredictionResponse(
            predictions=predictions,
            total_processed=len(transactions),
            total_fraud_detected=total_fraud,
            avg_inference_time_ms=round(total_time / len(transactions), 2)
        )
    
    async def health_check(self) -> Dict[str, Any]:
        """Check service health"""
        cache_health = await cache.health_check()
        
        return {
            "status": "healthy" if self._initialized else "unhealthy",
            "model_loaded": self._initialized,
            "model_version": self.model_version,
            "cache_status": cache_health.get("status", "unknown"),
            "database_status": "connected",
            "timestamp": datetime.utcnow()
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        if not self._initialized:
            return {"error": "Service not initialized"}
        
        return {
            "model_type": self.model.model_type,
            "model_version": self.model_version,
            "feature_count": len(self.feature_engineer.feature_names),
            "training_metrics": self.model.metrics.to_dict() if self.model.metrics else {},
            "last_updated": datetime.utcnow()
        }


service = FraudDetectionService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    logger.info("Starting SentinelML API server")
    
    await cache.connect()
    db_manager.init_async_engine()
    await service.initialize()
    
    logger.info("SentinelML API server ready")
    
    yield
    
    logger.info("Shutting down SentinelML API server")
    await cache.disconnect()
    await db_manager.close()


app = FastAPI(
    title="SentinelML - Fraud Detection API",
    description="Real-time fraud detection system with ML-powered predictions",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests"""
    start_time = time.time()
    
    response = await call_next(request)
    
    duration = (time.time() - start_time) * 1000
    
    logger.info(
        f"{request.method} {request.url.path} - "
        f"{response.status_code} - {duration:.2f}ms"
    )
    
    response.headers["X-Process-Time"] = f"{duration:.2f}ms"
    
    return response


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "name": "SentinelML Fraud Detection API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    health = await service.health_check()
    return HealthResponse(**health)


@app.post(
    "/api/v1/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Predictions"]
)
async def predict_fraud(
    transaction: TransactionRequest,
    background_tasks: BackgroundTasks
):
    """
    Predict fraud probability for a single transaction.
    
    Returns fraud score between 0 and 1, where higher scores
    indicate higher fraud probability.
    """
    try:
        response = await service.predict(transaction)
        
        background_tasks.add_task(
            log_prediction_to_db,
            transaction,
            response
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post(
    "/api/v1/predict/batch",
    response_model=BatchPredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Predictions"]
)
async def predict_fraud_batch(
    request: BatchTransactionRequest,
    background_tasks: BackgroundTasks
):
    """
    Batch fraud prediction for multiple transactions.
    
    Processes up to 100 transactions in a single request.
    """
    try:
        response = await service.predict_batch(request.transactions)
        
        for txn, pred in zip(request.transactions, response.predictions):
            background_tasks.add_task(
                log_prediction_to_db,
                txn,
                pred
            )
        
        return response
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )


@app.get("/api/v1/model/info", response_model=ModelInfoResponse, tags=["Model"])
async def get_model_info():
    """Get information about the current model"""
    info = service.get_model_info()
    return ModelInfoResponse(**info)


async def log_prediction_to_db(
    transaction: TransactionRequest,
    prediction: PredictionResponse
):
    """Background task to log prediction to database"""
    try:
        async with db_manager.get_async_session() as session:
            log = PredictionLog(
                transaction_id=prediction.transaction_id,
                prediction_time_ms=prediction.inference_time_ms,
                fraud_score=prediction.fraud_score,
                prediction=prediction.prediction,
                model_version=prediction.model_version,
                cache_hit=prediction.cached
            )
            session.add(log)
    except Exception as e:
        logger.error(f"Failed to log prediction: {e}")


def create_app() -> FastAPI:
    """Factory function to create FastAPI app"""
    return app


if __name__ == "__main__":
    uvicorn.run(
        "sentinel_ml.api.main:app",
        host=config.api.host,
        port=config.api.port,
        workers=config.api.workers,
        reload=config.is_development
    )
