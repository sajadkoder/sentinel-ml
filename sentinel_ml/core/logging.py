"""
Logging configuration for SentinelML
"""
import logging
import sys
from datetime import datetime
from typing import Optional
from pathlib import Path
import json
from pythonjsonlogger import jsonlogger

from sentinel_ml.config import config


class CustomJsonFormatter(jsonlogger.JsonFormatter):
    """Custom JSON formatter for structured logging"""
    
    def add_fields(self, log_record, record, message_dict):
        super().add_fields(log_record, record, message_dict)
        log_record['timestamp'] = datetime.utcnow().isoformat()
        log_record['level'] = record.levelname
        log_record['logger'] = record.name
        log_record['environment'] = config.environment.value
        
        if hasattr(record, 'transaction_id'):
            log_record['transaction_id'] = record.transaction_id
        if hasattr(record, 'model_version'):
            log_record['model_version'] = record.model_version
        if hasattr(record, 'prediction_score'):
            log_record['prediction_score'] = record.prediction_score


def setup_logging(
    name: str = "sentinel_ml",
    log_level: Optional[str] = None,
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    Setup structured logging for the application.
    
    Args:
        name: Logger name
        log_level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for logs
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    if logger.handlers:
        return logger
    
    level = log_level or config.monitoring.log_level
    logger.setLevel(getattr(logging, level.upper()))
    
    formatter = CustomJsonFormatter(
        '%(timestamp)s %(level)s %(name)s %(message)s'
    )
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


class LoggerAdapter(logging.LoggerAdapter):
    """Context-aware logger adapter for adding extra fields"""
    
    def process(self, msg, kwargs):
        extra = kwargs.get('extra', {})
        extra.update(self.extra)
        kwargs['extra'] = extra
        return msg, kwargs


def get_logger(name: str = "sentinel_ml") -> logging.Logger:
    """Get a configured logger instance"""
    return setup_logging(name)


logger = get_logger()
