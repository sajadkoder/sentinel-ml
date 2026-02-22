"""
Training script for SentinelML
"""
import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sentinel_ml.pipelines import TrainingPipeline
from sentinel_ml.core.logging import get_logger

logger = get_logger(__name__)


def main():
    logger.info("=" * 60)
    logger.info("SentinelML Training Pipeline")
    logger.info("=" * 60)
    
    pipeline = TrainingPipeline(
        output_dir="artifacts",
        use_mlflow=False
    )
    
    results = pipeline.run(
        n_samples=50000,
        fraud_rate=0.02,
        model_type="random_forest",
        hyperparams={
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_split": 5,
            "class_weight": "balanced"
        }
    )
    
    logger.info("=" * 60)
    logger.info("Training Results:")
    logger.info(f"  Model Version: {results.get('model_version')}")
    logger.info(f"  Test ROC-AUC: {results.get('metrics', {}).get('test_roc_auc', 'N/A'):.4f}")
    logger.info(f"  Test Precision: {results.get('metrics', {}).get('test_precision', 'N/A'):.4f}")
    logger.info(f"  Test Recall: {results.get('metrics', {}).get('test_recall', 'N/A'):.4f}")
    logger.info(f"  Duration: {results.get('total_duration_seconds', 0):.2f}s")
    logger.info("=" * 60)
    
    return results


if __name__ == "__main__":
    main()
