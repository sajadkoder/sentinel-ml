"""
Training and retraining pipelines for SentinelML
"""
import os
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Optional, Any
from pathlib import Path
import json

import numpy as np
import pandas as pd

from sentinel_ml.config import config
from sentinel_ml.core.logging import get_logger
from sentinel_ml.core.database import db_manager, Transaction
from sentinel_ml.data import SyntheticDataGenerator, DataPreprocessor
from sentinel_ml.features import FeatureEngineer
from sentinel_ml.models import FraudDetectionModel, MLflowTracker, ModelTrainer
from sentinel_ml.monitoring import DataDriftDetector, MonitoringDashboard

logger = get_logger(__name__)


class TrainingPipeline:
    """
    Complete training pipeline for fraud detection model.
    
    Steps:
    1. Load/generate training data
    2. Preprocess data
    3. Engineer features
    4. Train model
    5. Evaluate model
    6. Save artifacts
    7. Log to MLflow
    """
    
    def __init__(
        self,
        output_dir: str = "artifacts",
        use_mlflow: bool = True
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.use_mlflow = use_mlflow
        self.mlflow_tracker = MLflowTracker() if use_mlflow else None
        
        self.data_generator = SyntheticDataGenerator()
        self.preprocessor = DataPreprocessor()
        self.feature_engineer = FeatureEngineer()
        self.model_trainer = ModelTrainer(
            feature_engineer=self.feature_engineer,
            mlflow_tracker=self.mlflow_tracker
        )
    
    def run(
        self,
        data_path: Optional[str] = None,
        n_samples: int = 100000,
        fraud_rate: float = 0.02,
        model_type: str = "random_forest",
        hyperparams: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Run the complete training pipeline.
        
        Args:
            data_path: Path to training data (optional)
            n_samples: Number of samples to generate if no data_path
            fraud_rate: Fraud rate for synthetic data
            model_type: Type of model to train
            hyperparams: Model hyperparameters
            
        Returns:
            Dictionary with training results
        """
        logger.info("Starting training pipeline")
        pipeline_start = datetime.now()
        
        results = {
            'start_time': pipeline_start.isoformat(),
            'config': {
                'n_samples': n_samples,
                'fraud_rate': fraud_rate,
                'model_type': model_type
            }
        }
        
        try:
            if data_path and os.path.exists(data_path):
                logger.info(f"Loading data from {data_path}")
                df = pd.read_csv(data_path)
            else:
                logger.info(f"Generating synthetic data: {n_samples} samples")
                df = self.data_generator.generate_dataset(
                    n_transactions=n_samples,
                    fraud_rate=fraud_rate
                )
                data_path = self.output_dir / "training_data.csv"
                df.to_csv(data_path, index=False)
                results['data_path'] = str(data_path)
            
            results['data_shape'] = df.shape
            results['fraud_distribution'] = df['is_fraud'].value_counts().to_dict()
            
            logger.info("Training model")
            model, metrics = self.model_trainer.run_full_training(
                df=df,
                model_type=model_type,
                hyperparams=hyperparams,
                log_to_mlflow=self.use_mlflow
            )
            
            results['metrics'] = metrics
            results['model_version'] = model.version
            
            model_path = self.output_dir / f"model_{model.version}.joblib"
            model.save(str(model_path))
            results['model_path'] = str(model_path)
            
            fe_path = self.output_dir / f"feature_engineer_{model.version}.joblib"
            self.feature_engineer.save(str(fe_path))
            results['feature_engineer_path'] = str(fe_path)
            
            if self.use_mlflow:
                results['mlflow_run_id'] = metrics.get('mlflow_run_id')
            
            pipeline_end = datetime.now()
            results['end_time'] = pipeline_end.isoformat()
            results['total_duration_seconds'] = (pipeline_end - pipeline_start).total_seconds()
            
            results_path = self.output_dir / f"training_results_{model.version}.json"
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(
                f"Training pipeline completed in {results['total_duration_seconds']:.2f}s"
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {e}")
            results['error'] = str(e)
            raise


class RetrainingPipeline:
    """
    Automated retraining pipeline triggered by:
    - Scheduled intervals
    - Performance degradation
    - Data drift detection
    """
    
    def __init__(
        self,
        training_pipeline: Optional[TrainingPipeline] = None,
        min_new_samples: int = 10000,
        performance_threshold: float = 0.85
    ):
        self.training_pipeline = training_pipeline or TrainingPipeline()
        self.min_new_samples = min_new_samples
        self.performance_threshold = performance_threshold
        
        self.drift_detector = DataDriftDetector()
        self.last_retrain_time: Optional[datetime] = None
        self.retrain_history: list = []
    
    async def check_retrain_needed(self) -> Dict[str, Any]:
        """
        Check if retraining is needed based on multiple factors.
        
        Returns:
            Dictionary with retrain decision and reasons
        """
        reasons = []
        should_retrain = False
        
        if self.last_retrain_time:
            hours_since_retrain = (
                datetime.now() - self.last_retrain_time
            ).total_seconds() / 3600
            
            if hours_since_retrain >= config.training.retrain_interval_hours:
                reasons.append(f"Scheduled retrain: {hours_since_retrain:.1f} hours since last retrain")
                should_retrain = True
        
        new_samples_count = await self._count_new_transactions()
        if new_samples_count >= self.min_new_samples:
            reasons.append(f"New data available: {new_samples_count} new transactions")
            should_retrain = True
        
        drift_summary = self.drift_detector.get_drift_summary()
        if drift_summary.get('recent_drifts', 0) > 0:
            reasons.append(f"Data drift detected: {drift_summary['recent_drifts']} features affected")
            should_retrain = True
        
        return {
            'should_retrain': should_retrain,
            'reasons': reasons,
            'new_samples_count': new_samples_count,
            'last_retrain': self.last_retrain_time.isoformat() if self.last_retrain_time else None,
            'drift_summary': drift_summary
        }
    
    async def _count_new_transactions(self) -> int:
        """Count new transactions since last retrain"""
        try:
            async with db_manager.get_async_session() as session:
                from sqlalchemy import select, func
                
                query = select(func.count(Transaction.id))
                
                if self.last_retrain_time:
                    query = query.where(
                        Transaction.created_at > self.last_retrain_time
                    )
                
                result = await session.execute(query)
                return result.scalar() or 0
        except Exception as e:
            logger.error(f"Failed to count new transactions: {e}")
            return 0
    
    async def run_retraining(
        self,
        force: bool = False,
        **training_kwargs
    ) -> Dict[str, Any]:
        """
        Run retraining if conditions are met.
        
        Args:
            force: Force retrain regardless of conditions
            **training_kwargs: Arguments for training pipeline
            
        Returns:
            Training results or skip reason
        """
        if not force:
            check = await self.check_retrain_needed()
            if not check['should_retrain']:
                logger.info("Retraining not needed")
                return {
                    'status': 'skipped',
                    'reason': 'Retrain conditions not met',
                    'check_results': check
                }
        
        logger.info("Starting retraining")
        
        df = await self._get_training_data()
        
        if df is None or len(df) < self.min_new_samples:
            logger.warning("Insufficient data for retraining")
            return {
                'status': 'skipped',
                'reason': 'Insufficient training data'
            }
        
        results = self.training_pipeline.run(
            data=None,
            n_samples=len(df),
            **training_kwargs
        )
        
        self.last_retrain_time = datetime.now()
        self.retrain_history.append({
            'timestamp': self.last_retrain_time.isoformat(),
            'samples_used': len(df),
            'model_version': results.get('model_version'),
            'metrics': results.get('metrics')
        })
        
        results['status'] = 'completed'
        results['retrain_reason'] = 'forced' if force else 'automatic'
        
        logger.info(f"Retraining completed: {results.get('model_version')}")
        
        return results
    
    async def _get_training_data(self) -> Optional[pd.DataFrame]:
        """Fetch training data from database"""
        try:
            async with db_manager.get_async_session() as session:
                from sqlalchemy import select
                
                query = select(Transaction).order_by(Transaction.created_at.desc()).limit(100000)
                result = await session.execute(query)
                transactions = result.scalars().all()
                
                if not transactions:
                    return None
                
                data = []
                for t in transactions:
                    data.append({
                        'transaction_id': t.transaction_id,
                        'user_id': t.user_id,
                        'amount': t.amount,
                        'merchant_id': t.merchant_id,
                        'merchant_category': t.merchant_category,
                        'transaction_type': t.transaction_type,
                        'device_type': t.device_type,
                        'location_country': t.location_country,
                        'card_type': t.card_type,
                        'timestamp': t.timestamp,
                        'is_fraud': t.is_fraud
                    })
                
                return pd.DataFrame(data)
                
        except Exception as e:
            logger.error(f"Failed to fetch training data: {e}")
            return None


class ABTestSimulation:
    """
    A/B testing simulation for model comparison.
    
    Simulates traffic splitting between two models
    and compares performance metrics.
    """
    
    def __init__(
        self,
        model_a: FraudDetectionModel,
        model_b: FraudDetectionModel,
        traffic_split: float = 0.5
    ):
        self.model_a = model_a
        self.model_b = model_b
        self.traffic_split = traffic_split
        
        self.results_a = {
            'predictions': [],
            'scores': [],
            'actuals': [],
            'latencies': []
        }
        self.results_b = {
            'predictions': [],
            'scores': [],
            'actuals': [],
            'latencies': []
        }
    
    def predict(self, X: np.ndarray, actual: Optional[np.ndarray] = None) -> Dict:
        """
        Route prediction to model A or B based on traffic split.
        
        Args:
            X: Feature matrix
            actual: Ground truth labels (optional)
            
        Returns:
            Prediction result with model assignment
        """
        import time
        
        use_model_a = np.random.random() < self.traffic_split
        
        if use_model_a:
            start = time.perf_counter()
            score = self.model_a.get_fraud_score(X)[0]
            latency = (time.perf_counter() - start) * 1000
            prediction = score >= 0.5
            
            self.results_a['predictions'].append(prediction)
            self.results_a['scores'].append(score)
            self.results_a['latencies'].append(latency)
            
            if actual is not None:
                self.results_a['actuals'].append(actual[0])
            
            return {
                'model': 'A',
                'score': score,
                'prediction': prediction,
                'latency_ms': latency
            }
        else:
            start = time.perf_counter()
            score = self.model_b.get_fraud_score(X)[0]
            latency = (time.perf_counter() - start) * 1000
            prediction = score >= 0.5
            
            self.results_b['predictions'].append(prediction)
            self.results_b['scores'].append(score)
            self.results_b['latencies'].append(latency)
            
            if actual is not None:
                self.results_b['actuals'].append(actual[0])
            
            return {
                'model': 'B',
                'score': score,
                'prediction': prediction,
                'latency_ms': latency
            }
    
    def get_comparison(self) -> Dict[str, Any]:
        """Compare performance of model A vs B"""
        from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
        
        def compute_metrics(results):
            if not results['predictions']:
                return {}
            
            metrics = {
                'total_predictions': len(results['predictions']),
                'fraud_rate': np.mean(results['predictions']),
                'avg_score': np.mean(results['scores']),
                'avg_latency_ms': np.mean(results['latencies']),
                'p95_latency_ms': np.percentile(results['latencies'], 95)
            }
            
            if results['actuals']:
                preds = np.array(results['predictions'][:len(results['actuals'])])
                actuals = np.array(results['actuals'])
                scores = np.array(results['scores'][:len(results['actuals'])])
                
                metrics.update({
                    'precision': precision_score(actuals, preds, zero_division=0),
                    'recall': recall_score(actuals, preds, zero_division=0),
                    'f1_score': f1_score(actuals, preds, zero_division=0),
                    'roc_auc': roc_auc_score(actuals, scores) if len(np.unique(actuals)) > 1 else 0
                })
            
            return metrics
        
        metrics_a = compute_metrics(self.results_a)
        metrics_b = compute_metrics(self.results_b)
        
        winner = None
        if metrics_a.get('roc_auc') and metrics_b.get('roc_auc'):
            if metrics_a['roc_auc'] > metrics_b['roc_auc'] + 0.02:
                winner = 'A'
            elif metrics_b['roc_auc'] > metrics_a['roc_auc'] + 0.02:
                winner = 'B'
            else:
                winner = 'tie'
        
        return {
            'model_a': metrics_a,
            'model_b': metrics_b,
            'winner': winner,
            'traffic_split': self.traffic_split,
            'total_samples': len(self.results_a['predictions']) + len(self.results_b['predictions'])
        }
    
    def reset(self):
        """Reset A/B test results"""
        self.results_a = {
            'predictions': [],
            'scores': [],
            'actuals': [],
            'latencies': []
        }
        self.results_b = {
            'predictions': [],
            'scores': [],
            'actuals': [],
            'latencies': []
        }
