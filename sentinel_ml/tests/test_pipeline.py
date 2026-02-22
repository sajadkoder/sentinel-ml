"""
Unit tests for SentinelML
"""
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import tempfile
import os

from sentinel_ml.config import config
from sentinel_ml.data import SyntheticDataGenerator, DataPreprocessor
from sentinel_ml.features import FeatureEngineer
from sentinel_ml.models import FraudDetectionModel, ModelTrainer
from sentinel_ml.monitoring import DataDriftDetector, ModelPerformanceMonitor


class TestDataGenerator:
    """Tests for synthetic data generation"""
    
    def test_generate_dataset_shape(self):
        """Test dataset has correct shape"""
        generator = SyntheticDataGenerator(seed=42)
        df = generator.generate_dataset(n_transactions=1000, fraud_rate=0.1)
        
        assert df.shape[0] == 1000
        assert 'is_fraud' in df.columns
        assert 'amount' in df.columns
        assert 'user_id' in df.columns
    
    def test_fraud_rate(self):
        """Test fraud rate is approximately correct"""
        generator = SyntheticDataGenerator(seed=42)
        df = generator.generate_dataset(n_transactions=10000, fraud_rate=0.05)
        
        actual_rate = df['is_fraud'].mean()
        assert 0.04 < actual_rate < 0.06
    
    def test_transaction_ids_unique(self):
        """Test all transaction IDs are unique"""
        generator = SyntheticDataGenerator(seed=42)
        df = generator.generate_dataset(n_transactions=1000)
        
        assert df['transaction_id'].nunique() == len(df)


class TestDataPreprocessor:
    """Tests for data preprocessing"""
    
    @pytest.fixture
    def sample_data(self):
        generator = SyntheticDataGenerator(seed=42)
        return generator.generate_dataset(n_transactions=1000)
    
    def test_clean_data(self, sample_data):
        """Test data cleaning"""
        preprocessor = DataPreprocessor()
        df_clean = preprocessor.clean_data(sample_data)
        
        assert df_clean.isnull().sum().sum() == 0 or df_clean['device_type'].isnull().sum() > 0
    
    def test_fit_transform(self, sample_data):
        """Test fit_transform pipeline"""
        preprocessor = DataPreprocessor()
        df_processed = preprocessor.fit_transform(sample_data)
        
        assert 'merchant_category_encoded' in df_processed.columns
        assert 'amount_normalized' in df_processed.columns


class TestFeatureEngineer:
    """Tests for feature engineering"""
    
    @pytest.fixture
    def sample_data(self):
        generator = SyntheticDataGenerator(seed=42)
        return generator.generate_dataset(n_transactions=1000)
    
    def test_fit_transform(self, sample_data):
        """Test feature engineering fit_transform"""
        fe = FeatureEngineer()
        X, feature_names = fe.fit_transform(sample_data)
        
        assert X.shape[0] == len(sample_data)
        assert len(feature_names) > 0
        assert fe.is_fitted
    
    def test_transform_after_fit(self, sample_data):
        """Test transform after fitting"""
        fe = FeatureEngineer()
        fe.fit_transform(sample_data)
        
        new_data = sample_data.iloc[:100]
        X = fe.transform(new_data)
        
        assert X.shape[0] == 100
    
    def test_time_features(self, sample_data):
        """Test time feature extraction"""
        fe = FeatureEngineer()
        df = fe.extract_time_features(sample_data)
        
        assert 'hour_of_day' in df.columns
        assert 'is_weekend' in df.columns
        assert 'is_night' in df.columns
    
    def test_save_load(self, sample_data):
        """Test saving and loading feature engineer"""
        fe = FeatureEngineer()
        fe.fit_transform(sample_data)
        
        with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as f:
            fe.save(f.name)
            loaded_fe = FeatureEngineer.load(f.name)
            
            assert loaded_fe.is_fitted
            assert loaded_fe.feature_names == fe.feature_names
            
            os.unlink(f.name)


class TestFraudDetectionModel:
    """Tests for fraud detection model"""
    
    @pytest.fixture
    def sample_data(self):
        generator = SyntheticDataGenerator(seed=42)
        return generator.generate_dataset(n_transactions=1000)
    
    @pytest.fixture
    def trained_model(self, sample_data):
        fe = FeatureEngineer()
        X, feature_names = fe.fit_transform(sample_data)
        y = sample_data['is_fraud'].values
        
        model = FraudDetectionModel(model_type='random_forest')
        model.train(X[:800], y[:800], feature_names, X[800:], y[800:])
        
        return model, X, y
    
    def test_model_initialization(self):
        """Test model initialization"""
        model = FraudDetectionModel(model_type='random_forest')
        
        assert model.model_type == 'random_forest'
        assert model.version is not None
        assert model.model is None
    
    def test_model_training(self, trained_model):
        """Test model training"""
        model, X, y = trained_model
        
        assert model.is_fitted
        assert model.metrics is not None
        assert model.metrics.roc_auc > 0.5
    
    def test_prediction(self, trained_model):
        """Test model prediction"""
        model, X, y = trained_model
        
        predictions = model.predict(X[:10])
        probabilities = model.predict_proba(X[:10])
        scores = model.get_fraud_score(X[:10])
        
        assert len(predictions) == 10
        assert probabilities.shape == (10, 2)
        assert len(scores) == 10
        assert all(0 <= s <= 1 for s in scores)
    
    def test_model_save_load(self, trained_model):
        """Test saving and loading model"""
        model, X, y = trained_model
        
        with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as f:
            model.save(f.name)
            loaded_model = FraudDetectionModel.load(f.name)
            
            assert loaded_model.is_fitted
            assert loaded_model.version == model.version
            
            np.testing.assert_array_equal(
                model.predict(X[:5]),
                loaded_model.predict(X[:5])
            )
            
            os.unlink(f.name)
    
    def test_cross_validation(self, sample_data):
        """Test cross-validation"""
        fe = FeatureEngineer()
        X, _ = fe.fit_transform(sample_data)
        y = sample_data['is_fraud'].values
        
        model = FraudDetectionModel(model_type='random_forest')
        cv_results = model.cross_validate(X, y, cv_folds=3)
        
        assert 'roc_auc_mean' in cv_results
        assert cv_results['roc_auc_mean'] > 0


class TestDataDriftDetector:
    """Tests for data drift detection"""
    
    def test_drift_detection_no_drift(self):
        """Test drift detection with no drift"""
        detector = DataDriftDetector()
        
        reference = np.random.normal(0, 1, 1000)
        test = np.random.normal(0, 1, 1000)
        
        detector.set_reference('feature_1', reference)
        result = detector.detect_drift_ks('feature_1', test)
        
        assert result is not None
        assert result.p_value > 0.01
    
    def test_drift_detection_with_drift(self):
        """Test drift detection with actual drift"""
        detector = DataDriftDetector(drift_threshold=0.05)
        
        reference = np.random.normal(0, 1, 1000)
        test = np.random.normal(2, 1, 1000)
        
        detector.set_reference('feature_1', reference)
        result = detector.detect_drift_ks('feature_1', test)
        
        assert result is not None
        assert result.drift_detected
        assert result.p_value < 0.05
    
    def test_psi_calculation(self):
        """Test PSI calculation"""
        detector = DataDriftDetector()
        
        reference = np.random.normal(0, 1, 1000)
        test = np.random.normal(0, 1, 1000)
        
        detector.set_reference('feature_1', reference)
        psi = detector.calculate_psi('feature_1', test)
        
        assert 0 <= psi < 0.1


class TestModelPerformanceMonitor:
    """Tests for model performance monitoring"""
    
    def test_log_prediction(self):
        """Test logging predictions"""
        monitor = ModelPerformanceMonitor()
        
        for _ in range(100):
            monitor.log_prediction(
                prediction=np.random.choice([True, False]),
                score=np.random.random(),
                actual=np.random.choice([True, False]),
                latency_ms=np.random.uniform(1, 50)
            )
        
        assert len(monitor.predictions) == 100
    
    def test_compute_metrics(self):
        """Test computing metrics"""
        monitor = ModelPerformanceMonitor()
        
        for _ in range(100):
            monitor.log_prediction(
                prediction=np.random.choice([True, False]),
                score=np.random.random(),
                actual=np.random.choice([True, False]),
                latency_ms=np.random.uniform(1, 50)
            )
        
        metrics = monitor.compute_metrics()
        
        assert 'total_predictions' in metrics
        assert 'avg_latency_ms' in metrics
        assert metrics['total_predictions'] == 100


class TestConfig:
    """Tests for configuration"""
    
    def test_config_exists(self):
        """Test config is loaded"""
        assert config is not None
        assert config.database is not None
        assert config.model is not None
    
    def test_database_config(self):
        """Test database configuration"""
        assert config.database.host is not None
        assert config.database.port > 0
    
    def test_model_config(self):
        """Test model configuration"""
        assert config.model.model_type is not None
        assert config.model.n_estimators > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
