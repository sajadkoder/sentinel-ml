"""
Feature engineering pipeline for fraud detection
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
from pathlib import Path

from sentinel_ml.config import config
from sentinel_ml.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class FeatureSet:
    """Container for engineered features"""
    numerical: np.ndarray
    categorical: np.ndarray
    feature_names: List[str]
    metadata: Dict


class FeatureEngineer:
    """
    Feature engineering class for fraud detection.
    
    Creates three types of features:
    1. Transaction-level: Direct attributes of transaction
    2. User-level: Historical behavior patterns
    3. Contextual: Time, location, device patterns
    """
    
    def __init__(self):
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.scaler = StandardScaler()
        self.is_fitted = False
        
        self.numerical_feature_names = [
            'amount',
            'hour_of_day',
            'day_of_week',
            'is_weekend',
            'is_night',
            'amount_log',
            'user_txn_count_1h',
            'user_txn_count_24h',
            'user_txn_count_7d',
            'user_avg_amount_7d',
            'user_std_amount_7d',
            'amount_to_avg_ratio',
            'amount_zscore',
            'time_since_last_txn',
            'txn_velocity_1h',
            'txn_velocity_24h',
            'merchant_risk_score',
            'country_risk_score',
            'device_change_flag',
            'location_change_flag',
        ]
        
        self.categorical_feature_names = [
            'merchant_category',
            'transaction_type',
            'device_type',
            'card_type',
            'location_country',
        ]
    
    def extract_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract time-based features"""
        df = df.copy()
        
        df['hour_of_day'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_night'] = ((df['hour_of_day'] >= 22) | (df['hour_of_day'] <= 5)).astype(int)
        df['amount_log'] = np.log1p(df['amount'])
        
        return df
    
    def compute_user_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute user-level behavioral features"""
        df = df.copy()
        df = df.sort_values(['user_id', 'timestamp'])
        
        df['user_txn_count_1h'] = df.groupby('user_id').apply(
            lambda g: g.set_index('timestamp')['transaction_id']
            .rolling('1h').count()
        ).values
        
        df['user_txn_count_24h'] = df.groupby('user_id').apply(
            lambda g: g.set_index('timestamp')['transaction_id']
            .rolling('24h').count()
        ).values
        
        df['user_txn_count_7d'] = df.groupby('user_id').cumcount()
        
        df['user_avg_amount_7d'] = df.groupby('user_id')['amount'].transform(
            lambda x: x.rolling(window=100, min_periods=1).mean()
        )
        
        df['user_std_amount_7d'] = df.groupby('user_id')['amount'].transform(
            lambda x: x.rolling(window=100, min_periods=1).std().fillna(0)
        )
        
        df['amount_to_avg_ratio'] = df['amount'] / (df['user_avg_amount_7d'] + 1e-8)
        
        df['amount_zscore'] = (
            df['amount'] - df['user_avg_amount_7d']
        ) / (df['user_std_amount_7d'] + 1e-8)
        
        df['time_since_last_txn'] = df.groupby('user_id')['timestamp'].diff().dt.total_seconds()
        df['time_since_last_txn'] = df['time_since_last_txn'].fillna(0)
        
        df['txn_velocity_1h'] = 3600 / (df['time_since_last_txn'] + 1)
        df['txn_velocity_24h'] = df['user_txn_count_24h'] / 24
        
        df['user_txn_count_1h'] = df['user_txn_count_1h'].fillna(0)
        df['user_txn_count_24h'] = df['user_txn_count_24h'].fillna(0)
        df['user_txn_count_7d'] = df['user_txn_count_7d'].fillna(0)
        
        return df
    
    def compute_risk_scores(
        self,
        df: pd.DataFrame,
        fraud_rates: Optional[Dict] = None
    ) -> pd.DataFrame:
        """Compute risk scores for merchants and countries"""
        df = df.copy()
        
        if fraud_rates is None:
            merchant_fraud = df.groupby('merchant_id')['is_fraud'].mean()
            country_fraud = df.groupby('location_country')['is_fraud'].mean()
        else:
            merchant_fraud = fraud_rates.get('merchant', {})
            country_fraud = fraud_rates.get('country', {})
        
        avg_fraud_rate = df['is_fraud'].mean() if 'is_fraud' in df.columns else 0.02
        
        df['merchant_risk_score'] = df['merchant_id'].map(merchant_fraud).fillna(avg_fraud_rate)
        df['country_risk_score'] = df['location_country'].map(country_fraud).fillna(avg_fraud_rate)
        
        return df
    
    def compute_change_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect changes in user behavior patterns"""
        df = df.copy()
        df = df.sort_values(['user_id', 'timestamp'])
        
        df['prev_device_type'] = df.groupby('user_id')['device_type'].shift(1)
        df['prev_location_country'] = df.groupby('user_id')['location_country'].shift(1)
        
        df['device_change_flag'] = (
            df['device_type'] != df['prev_device_type']
        ).astype(int).fillna(0)
        
        df['location_change_flag'] = (
            df['location_country'] != df['prev_location_country']
        ).astype(int).fillna(0)
        
        df = df.drop(['prev_device_type', 'prev_location_country'], axis=1, errors='ignore')
        
        return df
    
    def encode_categorical(
        self,
        df: pd.DataFrame,
        fit: bool = True
    ) -> pd.DataFrame:
        """Encode categorical features"""
        df = df.copy()
        
        for col in self.categorical_feature_names:
            if col not in df.columns:
                continue
            
            df[col] = df[col].fillna('UNKNOWN').astype(str)
            
            if fit:
                self.label_encoders[col] = LabelEncoder()
                df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df[col])
            else:
                if col in self.label_encoders:
                    known_classes = set(self.label_encoders[col].classes_)
                    df[col] = df[col].apply(
                        lambda x: x if x in known_classes else 'UNKNOWN'
                    )
                    df[f'{col}_encoded'] = self.label_encoders[col].transform(df[col])
                else:
                    df[f'{col}_encoded'] = 0
        
        return df
    
    def normalize_features(
        self,
        df: pd.DataFrame,
        fit: bool = True
    ) -> pd.DataFrame:
        """Normalize numerical features"""
        df = df.copy()
        
        feature_cols = [col for col in self.numerical_feature_names if col in df.columns]
        
        if len(feature_cols) == 0:
            return df
        
        if fit:
            df[feature_cols] = self.scaler.fit_transform(df[feature_cols])
        else:
            df[feature_cols] = self.scaler.transform(df[feature_cols])
        
        return df
    
    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """
        Fit feature engineering pipeline and transform data.
        
        Args:
            df: Input DataFrame with transaction data
            
        Returns:
            Tuple of (feature matrix, feature names)
        """
        logger.info("Fitting feature engineering pipeline")
        
        df = self.extract_time_features(df)
        df = self.compute_user_features(df)
        df = self.compute_risk_scores(df)
        df = self.compute_change_features(df)
        df = self.encode_categorical(df, fit=True)
        
        feature_cols = []
        for col in self.numerical_feature_names:
            if col in df.columns:
                feature_cols.append(col)
        
        for col in self.categorical_feature_names:
            encoded_col = f'{col}_encoded'
            if encoded_col in df.columns:
                feature_cols.append(encoded_col)
        
        df = self.normalize_features(df, fit=True)
        
        X = df[feature_cols].values
        df[feature_cols] = df[feature_cols].fillna(0)
        X = df[feature_cols].values
        
        self.is_fitted = True
        self.feature_names = feature_cols
        
        logger.info(f"Created {len(feature_cols)} features")
        
        return X, feature_cols
    
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transform data using fitted pipeline.
        
        Args:
            df: Input DataFrame with transaction data
            
        Returns:
            Feature matrix
        """
        if not self.is_fitted:
            raise ValueError("FeatureEngineer must be fitted before transform")
        
        logger.info("Transforming data with fitted pipeline")
        
        df = self.extract_time_features(df)
        df = self.compute_user_features(df)
        df = self.compute_risk_scores(df, fraud_rates=self._get_fraud_rates())
        df = self.compute_change_features(df)
        df = self.encode_categorical(df, fit=False)
        df = self.normalize_features(df, fit=False)
        
        df[self.feature_names] = df[self.feature_names].fillna(0)
        X = df[self.feature_names].values
        
        return X
    
    def _get_fraud_rates(self) -> Dict:
        """Get stored fraud rates for inference"""
        return {}
    
    def save(self, path: str):
        """Save feature engineering pipeline"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            'label_encoders': self.label_encoders,
            'scaler': self.scaler,
            'is_fitted': self.is_fitted,
            'feature_names': self.feature_names,
            'numerical_feature_names': self.numerical_feature_names,
            'categorical_feature_names': self.categorical_feature_names,
        }, path)
        logger.info(f"Feature engineering pipeline saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'FeatureEngineer':
        """Load feature engineering pipeline"""
        data = joblib.load(path)
        fe = cls()
        fe.label_encoders = data['label_encoders']
        fe.scaler = data['scaler']
        fe.is_fitted = data['is_fitted']
        fe.feature_names = data['feature_names']
        fe.numerical_feature_names = data['numerical_feature_names']
        fe.categorical_feature_names = data['categorical_feature_names']
        logger.info(f"Feature engineering pipeline loaded from {path}")
        return fe
    
    def get_feature_importance_names(self) -> List[str]:
        """Get list of feature names for importance analysis"""
        return self.feature_names if self.is_fitted else []
