"""
Configuration settings for SentinelML
"""
import os
from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum


class Environment(Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


@dataclass
class DatabaseConfig:
    host: str = os.getenv("DB_HOST", "localhost")
    port: int = int(os.getenv("DB_PORT", "5432"))
    database: str = os.getenv("DB_NAME", "sentinel_ml")
    user: str = os.getenv("DB_USER", "postgres")
    password: str = os.getenv("DB_PASSWORD", "postgres")
    pool_size: int = 10
    max_overflow: int = 20
    
    @property
    def connection_string(self) -> str:
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"


@dataclass
class RedisConfig:
    host: str = os.getenv("REDIS_HOST", "localhost")
    port: int = int(os.getenv("REDIS_PORT", "6379"))
    db: int = int(os.getenv("REDIS_DB", "0"))
    password: Optional[str] = os.getenv("REDIS_PASSWORD")
    ttl: int = 3600
    max_connections: int = 50


@dataclass
class MLflowConfig:
    tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    experiment_name: str = "fraud_detection"
    model_name: str = "fraud_detector"
    s3_endpoint_url: Optional[str] = os.getenv("MLFLOW_S3_ENDPOINT_URL")


@dataclass
class ModelConfig:
    model_type: str = "random_forest"
    n_estimators: int = 100
    max_depth: int = 10
    min_samples_split: int = 5
    min_samples_leaf: int = 2
    random_state: int = 42
    class_weight: str = "balanced"
    
    fraud_threshold: float = 0.5
    feature_importance_threshold: float = 0.01


@dataclass
class FeatureConfig:
    numerical_features: List[str] = field(default_factory=lambda: [
        "amount",
        "hour_of_day",
        "day_of_week",
        "transaction_count_24h",
        "avg_transaction_amount_7d",
        "std_transaction_amount_7d",
        "time_since_last_transaction",
        "distance_from_home",
        "merchant_risk_score",
    ])
    
    categorical_features: List[str] = field(default_factory=lambda: [
        "merchant_category",
        "transaction_type",
        "device_type",
        "location_country",
        "card_type",
    ])
    
    derived_features: List[str] = field(default_factory=lambda: [
        "amount_to_avg_ratio",
        "velocity_1h",
        "velocity_24h",
        "is_night_transaction",
        "is_weekend",
    ])


@dataclass
class MonitoringConfig:
    drift_detection_window: int = 1000
    drift_threshold: float = 0.05
    performance_threshold: float = 0.85
    alert_email: str = os.getenv("ALERT_EMAIL", "alerts@sentinelml.com")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")


@dataclass
class APIConfig:
    host: str = os.getenv("API_HOST", "0.0.0.0")
    port: int = int(os.getenv("API_PORT", "8000"))
    workers: int = int(os.getenv("API_WORKERS", "4"))
    timeout: int = 30
    max_request_size: int = 10 * 1024 * 1024
    rate_limit: int = 1000


@dataclass
class TrainingConfig:
    test_size: float = 0.2
    validation_size: float = 0.1
    cross_validation_folds: int = 5
    early_stopping_rounds: int = 10
    retrain_interval_hours: int = 24
    min_samples_for_retrain: int = 10000


@dataclass
class Config:
    environment: Environment = Environment.DEVELOPMENT
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"
    
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)
    mlflow: MLflowConfig = field(default_factory=MLflowConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    api: APIConfig = field(default_factory=APIConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    @property
    def is_production(self) -> bool:
        return self.environment == Environment.PRODUCTION
    
    @property
    def is_development(self) -> bool:
        return self.environment == Environment.DEVELOPMENT


config = Config()
