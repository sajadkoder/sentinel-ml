"""
Monitoring and observability for SentinelML

Includes:
- Data drift detection
- Model performance monitoring
- Alerting system
- Metrics collection
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from scipy import stats
from collections import deque
import asyncio
import json

from sentinel_ml.config import config
from sentinel_ml.core.logging import get_logger
from sentinel_ml.core.database import db_manager, DataDriftLog, ModelMetrics

logger = get_logger(__name__)


@dataclass
class DriftResult:
    """Result of drift detection test"""
    feature_name: str
    drift_score: float
    drift_detected: bool
    test_statistic: float
    p_value: float
    test_type: str
    reference_mean: float
    test_mean: float
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
    
    def to_dict(self) -> Dict:
        return {
            'feature_name': self.feature_name,
            'drift_score': self.drift_score,
            'drift_detected': self.drift_detected,
            'test_statistic': self.test_statistic,
            'p_value': self.p_value,
            'test_type': self.test_type,
            'reference_mean': self.reference_mean,
            'test_mean': self.test_mean,
            'timestamp': self.timestamp.isoformat()
        }


class DataDriftDetector:
    """
    Detect data drift in feature distributions.
    
    Uses statistical tests to compare recent data against
    reference distributions from training data.
    
    Methods:
    - Kolmogorov-Smirnov test for numerical features
    - Chi-squared test for categorical features
    - Population Stability Index (PSI)
    """
    
    def __init__(
        self,
        drift_threshold: float = 0.05,
        window_size: int = 1000
    ):
        self.drift_threshold = drift_threshold
        self.window_size = window_size
        self.reference_distributions: Dict[str, np.ndarray] = {}
        self.recent_data: Dict[str, deque] = {}
        self.drift_history: List[DriftResult] = []
    
    def set_reference(self, feature_name: str, data: np.ndarray):
        """Set reference distribution for a feature"""
        self.reference_distributions[feature_name] = np.array(data)
        self.recent_data[feature_name] = deque(maxlen=self.window_size)
        logger.info(f"Set reference distribution for {feature_name}")
    
    def update(self, feature_name: str, value: float):
        """Add new observation to recent data window"""
        if feature_name not in self.recent_data:
            self.recent_data[feature_name] = deque(maxlen=self.window_size)
        self.recent_data[feature_name].append(value)
    
    def detect_drift_ks(
        self,
        feature_name: str,
        test_data: Optional[np.ndarray] = None
    ) -> Optional[DriftResult]:
        """
        Detect drift using Kolmogorov-Smirnov test.
        
        Null hypothesis: samples are drawn from the same distribution
        """
        if feature_name not in self.reference_distributions:
            return None
        
        if test_data is None:
            if feature_name not in self.recent_data or len(self.recent_data[feature_name]) < 100:
                return None
            test_data = np.array(self.recent_data[feature_name])
        
        reference = self.reference_distributions[feature_name]
        
        statistic, p_value = stats.ks_2samp(reference, test_data)
        
        drift_detected = p_value < self.drift_threshold
        drift_score = 1 - p_value
        
        result = DriftResult(
            feature_name=feature_name,
            drift_score=drift_score,
            drift_detected=drift_detected,
            test_statistic=statistic,
            p_value=p_value,
            test_type='kolmogorov_smirnov',
            reference_mean=np.mean(reference),
            test_mean=np.mean(test_data)
        )
        
        if drift_detected:
            logger.warning(
                f"Drift detected for {feature_name}: "
                f"p-value={p_value:.4f}, statistic={statistic:.4f}"
            )
            self.drift_history.append(result)
        
        return result
    
    def calculate_psi(
        self,
        feature_name: str,
        test_data: Optional[np.ndarray] = None,
        buckets: int = 10
    ) -> float:
        """
        Calculate Population Stability Index (PSI).
        
        PSI interpretation:
        - < 0.1: No significant change
        - 0.1 - 0.25: Moderate change
        - > 0.25: Significant change
        """
        if feature_name not in self.reference_distributions:
            return 0.0
        
        if test_data is None:
            if feature_name not in self.recent_data:
                return 0.0
            test_data = np.array(self.recent_data[feature_name])
        
        reference = self.reference_distributions[feature_name]
        
        breakpoints = np.percentile(reference, np.linspace(0, 100, buckets + 1))
        breakpoints[0] = -np.inf
        breakpoints[-1] = np.inf
        
        ref_counts, _ = np.histogram(reference, bins=breakpoints)
        test_counts, _ = np.histogram(test_data, bins=breakpoints)
        
        ref_pct = ref_counts / len(reference) + 1e-10
        test_pct = test_counts / len(test_data) + 1e-10
        
        psi = np.sum((test_pct - ref_pct) * np.log(test_pct / ref_pct))
        
        return psi
    
    def detect_drift_all_features(
        self,
        feature_data: Optional[Dict[str, np.ndarray]] = None
    ) -> List[DriftResult]:
        """Run drift detection on all tracked features"""
        results = []
        
        for feature_name in self.reference_distributions.keys():
            test_data = None
            if feature_data and feature_name in feature_data:
                test_data = feature_data[feature_name]
            
            result = self.detect_drift_ks(feature_name, test_data)
            if result:
                results.append(result)
        
        return results
    
    def get_drift_summary(self) -> Dict[str, Any]:
        """Get summary of drift detection results"""
        if not self.drift_history:
            return {"total_drifts": 0, "features_affected": []}
        
        recent_drifts = [
            d for d in self.drift_history
            if d.timestamp > datetime.utcnow() - timedelta(days=7)
        ]
        
        features_affected = list(set(d.feature_name for d in recent_drifts))
        
        return {
            "total_drifts": len(self.drift_history),
            "recent_drifts": len(recent_drifts),
            "features_affected": features_affected,
            "last_drift": self.drift_history[-1].to_dict() if self.drift_history else None
        }


class ModelPerformanceMonitor:
    """
    Monitor model performance over time.
    
    Tracks:
    - Prediction accuracy
    - Precision/Recall trends
    - Confusion matrix evolution
    - Prediction latency
    """
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.predictions: deque = deque(maxlen=window_size)
        self.actuals: deque = deque(maxlen=window_size)
        self.scores: deque = deque(maxlen=window_size)
        self.latencies: deque = deque(maxlen=window_size)
        self.metrics_history: List[Dict] = []
    
    def log_prediction(
        self,
        prediction: bool,
        score: float,
        actual: Optional[bool] = None,
        latency_ms: float = 0
    ):
        """Log a prediction for monitoring"""
        self.predictions.append(prediction)
        self.scores.append(score)
        self.latencies.append(latency_ms)
        
        if actual is not None:
            self.actuals.append(actual)
    
    def compute_metrics(self) -> Dict[str, float]:
        """Compute current performance metrics"""
        if len(self.predictions) < 10:
            return {}
        
        predictions = np.array(self.predictions)
        scores = np.array(self.scores)
        latencies = np.array(self.latencies)
        
        metrics = {
            'total_predictions': len(predictions),
            'fraud_rate': np.mean(predictions),
            'avg_score': np.mean(scores),
            'score_std': np.std(scores),
            'avg_latency_ms': np.mean(latencies),
            'p95_latency_ms': np.percentile(latencies, 95),
            'p99_latency_ms': np.percentile(latencies, 99),
        }
        
        if len(self.actuals) > 10:
            actuals = np.array(self.actuals)
            min_len = min(len(predictions), len(actuals))
            preds = predictions[-min_len:]
            acts = actuals[-min_len:]
            
            tp = np.sum((preds == True) & (acts == True))
            fp = np.sum((preds == True) & (acts == False))
            tn = np.sum((preds == False) & (acts == False))
            fn = np.sum((preds == False) & (acts == True))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            metrics.update({
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'true_positives': int(tp),
                'false_positives': int(fp),
                'true_negatives': int(tn),
                'false_negatives': int(fn),
            })
        
        return metrics
    
    def check_performance_degradation(
        self,
        baseline_precision: float = 0.9,
        baseline_recall: float = 0.8,
        threshold: float = 0.1
    ) -> Dict[str, Any]:
        """Check if performance has degraded significantly"""
        current_metrics = self.compute_metrics()
        
        if not current_metrics:
            return {"degradation_detected": False, "reason": "insufficient_data"}
        
        degradation = {}
        
        if 'precision' in current_metrics:
            precision_drop = baseline_precision - current_metrics['precision']
            if precision_drop > threshold:
                degradation['precision'] = {
                    'baseline': baseline_precision,
                    'current': current_metrics['precision'],
                    'drop': precision_drop
                }
        
        if 'recall' in current_metrics:
            recall_drop = baseline_recall - current_metrics['recall']
            if recall_drop > threshold:
                degradation['recall'] = {
                    'baseline': baseline_recall,
                    'current': current_metrics['recall'],
                    'drop': recall_drop
                }
        
        return {
            "degradation_detected": len(degradation) > 0,
            "degraded_metrics": degradation,
            "current_metrics": current_metrics
        }
    
    def get_performance_trend(self, periods: int = 7) -> Dict[str, List[float]]:
        """Get performance trend over time"""
        if len(self.metrics_history) < periods:
            periods = len(self.metrics_history)
        
        trend = {
            'timestamps': [],
            'fraud_rate': [],
            'avg_latency_ms': []
        }
        
        for metrics in self.metrics_history[-periods:]:
            trend['timestamps'].append(metrics.get('timestamp', ''))
            trend['fraud_rate'].append(metrics.get('fraud_rate', 0))
            trend['avg_latency_ms'].append(metrics.get('avg_latency_ms', 0))
        
        return trend


class AlertingSystem:
    """
    Alerting system for fraud detection.
    
    Triggers alerts for:
    - Model performance degradation
    - Data drift detection
    - High fraud rates
    - System errors
    """
    
    def __init__(self):
        self.alert_history: List[Dict] = []
        self.alert_rules: Dict[str, Dict] = {}
        self._setup_default_rules()
    
    def _setup_default_rules(self):
        """Setup default alert rules"""
        self.alert_rules = {
            'high_fraud_rate': {
                'condition': lambda m: m.get('fraud_rate', 0) > 0.5,
                'severity': 'high',
                'message': 'Fraud rate exceeds 50%'
            },
            'low_precision': {
                'condition': lambda m: m.get('precision', 1) < 0.7,
                'severity': 'high',
                'message': 'Model precision below 70%'
            },
            'high_latency': {
                'condition': lambda m: m.get('p99_latency_ms', 0) > 100,
                'severity': 'medium',
                'message': 'P99 latency exceeds 100ms'
            },
            'drift_detected': {
                'condition': lambda m: m.get('drift_detected', False),
                'severity': 'high',
                'message': 'Data drift detected'
            }
        }
    
    def check_alerts(
        self,
        metrics: Dict[str, Any],
        drift_results: Optional[List[DriftResult]] = None
    ) -> List[Dict]:
        """Check metrics against alert rules"""
        alerts = []
        
        enhanced_metrics = metrics.copy()
        if drift_results:
            enhanced_metrics['drift_detected'] = any(d.drift_detected for d in drift_results)
        
        for rule_name, rule in self.alert_rules.items():
            try:
                if rule['condition'](enhanced_metrics):
                    alert = {
                        'rule_name': rule_name,
                        'severity': rule['severity'],
                        'message': rule['message'],
                        'metrics': metrics,
                        'timestamp': datetime.utcnow().isoformat()
                    }
                    alerts.append(alert)
                    self.alert_history.append(alert)
                    
                    logger.warning(
                        f"Alert triggered: {rule_name} - {rule['message']}"
                    )
            except Exception as e:
                logger.error(f"Error checking alert rule {rule_name}: {e}")
        
        return alerts
    
    def add_alert_rule(
        self,
        name: str,
        condition: callable,
        severity: str,
        message: str
    ):
        """Add custom alert rule"""
        self.alert_rules[name] = {
            'condition': condition,
            'severity': severity,
            'message': message
        }
        logger.info(f"Added alert rule: {name}")
    
    def get_recent_alerts(self, hours: int = 24) -> List[Dict]:
        """Get alerts from last N hours"""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        return [
            a for a in self.alert_history
            if datetime.fromisoformat(a['timestamp']) > cutoff
        ]


class MonitoringDashboard:
    """
    Aggregates all monitoring data for dashboard display.
    """
    
    def __init__(self):
        self.drift_detector = DataDriftDetector()
        self.performance_monitor = ModelPerformanceMonitor()
        self.alerting = AlertingSystem()
    
    def collect_metrics(self) -> Dict[str, Any]:
        """Collect all monitoring metrics"""
        performance_metrics = self.performance_monitor.compute_metrics()
        drift_summary = self.drift_detector.get_drift_summary()
        alerts = self.alerting.check_alerts(performance_metrics)
        
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'performance': performance_metrics,
            'drift': drift_summary,
            'alerts': alerts,
            'alert_count': len(alerts),
            'system_health': self._compute_health_status(performance_metrics, alerts)
        }
    
    def _compute_health_status(
        self,
        metrics: Dict,
        alerts: List[Dict]
    ) -> str:
        """Compute overall system health status"""
        if not metrics:
            return 'unknown'
        
        critical_alerts = [a for a in alerts if a['severity'] == 'high']
        
        if len(critical_alerts) > 0:
            return 'critical'
        elif len(alerts) > 0:
            return 'warning'
        else:
            return 'healthy'
    
    async def log_metrics_to_db(self, metrics: Dict):
        """Log metrics to database for historical tracking"""
        try:
            async with db_manager.get_async_session() as session:
                model_metrics = ModelMetrics(
                    model_version=metrics.get('model_version', 'unknown'),
                    model_name='fraud_detector',
                    precision=metrics.get('precision'),
                    recall=metrics.get('recall'),
                    f1_score=metrics.get('f1_score'),
                    sample_size=metrics.get('total_predictions', 0),
                    additional_metrics=metrics
                )
                session.add(model_metrics)
        except Exception as e:
            logger.error(f"Failed to log metrics to database: {e}")
