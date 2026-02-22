"""
Database connection and session management
"""
import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional
from datetime import datetime

from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Boolean, Index
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.dialects.postgresql import JSONB

from sentinel_ml.config import config
from sentinel_ml.core.logging import get_logger

logger = get_logger(__name__)

Base = declarative_base()


class Transaction(Base):
    """Transaction table for storing all transaction records"""
    __tablename__ = "transactions"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    transaction_id = Column(String(36), unique=True, nullable=False, index=True)
    user_id = Column(String(36), nullable=False, index=True)
    
    amount = Column(Float, nullable=False)
    merchant_id = Column(String(36), nullable=False, index=True)
    merchant_category = Column(String(50), nullable=False)
    transaction_type = Column(String(30), nullable=False)
    
    device_id = Column(String(36))
    device_type = Column(String(30))
    ip_address = Column(String(45))
    location_country = Column(String(3))
    location_city = Column(String(100))
    latitude = Column(Float)
    longitude = Column(Float)
    
    card_type = Column(String(20))
    card_last_four = Column(String(4))
    
    timestamp = Column(DateTime, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    
    is_fraud = Column(Boolean, nullable=True)
    fraud_score = Column(Float)
    model_version = Column(String(20))
    
    features = Column(JSONB)
    metadata = Column(JSONB)
    
    __table_args__ = (
        Index('idx_transactions_user_timestamp', 'user_id', 'timestamp'),
        Index('idx_transactions_merchant_timestamp', 'merchant_id', 'timestamp'),
        Index('idx_transactions_created_at', 'created_at'),
    )


class ModelMetrics(Base):
    """Model performance metrics table"""
    __tablename__ = "model_metrics"
    
    id = Column(Integer, primary_key=True)
    model_version = Column(String(20), nullable=False, index=True)
    model_name = Column(String(50), nullable=False)
    
    precision = Column(Float)
    recall = Column(Float)
    f1_score = Column(Float)
    roc_auc = Column(Float)
    accuracy = Column(Float)
    
    true_positives = Column(Integer)
    true_negatives = Column(Integer)
    false_positives = Column(Integer)
    false_negatives = Column(Integer)
    
    sample_size = Column(Integer)
    evaluation_period_start = Column(DateTime)
    evaluation_period_end = Column(DateTime)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    additional_metrics = Column(JSONB)


class PredictionLog(Base):
    """Prediction audit log"""
    __tablename__ = "prediction_logs"
    
    id = Column(Integer, primary_key=True)
    transaction_id = Column(String(36), nullable=False, index=True)
    prediction_time_ms = Column(Float, nullable=False)
    fraud_score = Column(Float, nullable=False)
    prediction = Column(Boolean, nullable=False)
    model_version = Column(String(20), nullable=False)
    cache_hit = Column(Boolean, default=False)
    
    feature_vector_hash = Column(String(64))
    explanation = Column(JSONB)
    
    created_at = Column(DateTime, default=datetime.utcnow, index=True)


class DataDriftLog(Base):
    """Data drift detection results"""
    __tablename__ = "data_drift_logs"
    
    id = Column(Integer, primary_key=True)
    feature_name = Column(String(100), nullable=False)
    drift_score = Column(Float, nullable=False)
    drift_detected = Column(Boolean, nullable=False)
    test_statistic = Column(Float)
    p_value = Column(Float)
    
    reference_period_start = Column(DateTime)
    reference_period_end = Column(DateTime)
    test_period_start = Column(DateTime)
    test_period_end = Column(DateTime)
    
    created_at = Column(DateTime, default=datetime.utcnow)


class DatabaseManager:
    """Database connection manager with connection pooling"""
    
    def __init__(self):
        self._engine = None
        self._async_engine = None
        self._session_factory = None
        self._async_session_factory = None
    
    def init_sync_engine(self):
        """Initialize synchronous database engine"""
        self._engine = create_engine(
            config.database.connection_string,
            pool_size=config.database.pool_size,
            max_overflow=config.database.max_overflow,
            pool_pre_ping=True,
            echo=config.debug
        )
        self._session_factory = sessionmaker(bind=self._engine)
        logger.info("Sync database engine initialized")
    
    def init_async_engine(self):
        """Initialize asynchronous database engine"""
        async_connection_string = config.database.connection_string.replace(
            "postgresql://", "postgresql+asyncpg://"
        )
        self._async_engine = create_async_engine(
            async_connection_string,
            pool_size=config.database.pool_size,
            max_overflow=config.database.max_overflow,
            echo=config.debug
        )
        self._async_session_factory = async_sessionmaker(
            self._async_engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
        logger.info("Async database engine initialized")
    
    def create_tables(self):
        """Create all database tables"""
        if not self._engine:
            self.init_sync_engine()
        Base.metadata.create_all(self._engine)
        logger.info("Database tables created")
    
    @asynccontextmanager
    async def get_async_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get async database session"""
        if not self._async_session_factory:
            self.init_async_engine()
        
        async with self._async_session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception as e:
                await session.rollback()
                logger.error(f"Database session error: {e}")
                raise
    
    def get_sync_session(self):
        """Get synchronous database session"""
        if not self._session_factory:
            self.init_sync_engine()
        return self._session_factory()
    
    async def close(self):
        """Close all database connections"""
        if self._async_engine:
            await self._async_engine.dispose()
        if self._engine:
            self._engine.dispose()
        logger.info("Database connections closed")


db_manager = DatabaseManager()
