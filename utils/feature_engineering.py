import numpy as np
import pandas as pd
from datetime import datetime
import math

class FeatureEngineer:
    """Feature engineering utilities for developer metrics"""
    
    @staticmethod
    def extract_time_features(timestamp):
        """Extract cyclical time features from timestamp"""
        dt = datetime.fromisoformat(timestamp) if isinstance(timestamp, str) else timestamp
        
        hour = dt.hour
        day_of_week = dt.weekday()
        
        return {
            'hour_sin': math.sin(2 * math.pi * hour / 24),
            'hour_cos': math.cos(2 * math.pi * hour / 24),
            'day_sin': math.sin(2 * math.pi * day_of_week / 7),
            'day_cos': math.cos(2 * math.pi * day_of_week / 7),
            'is_weekend': 1 if day_of_week >= 5 else 0,
            'is_work_hours': 1 if 9 <= hour <= 17 else 0
        }
    
    @staticmethod
    def calculate_rolling_metrics(df, window_size=5):
        """Calculate rolling statistics for time series features"""
        rolling_features = {}
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            rolling_features[f'{col}_rolling_mean'] = df[col].rolling(window=window_size).mean()
            rolling_features[f'{col}_rolling_std'] = df[col].rolling(window=window_size).std()
            rolling_features[f'{col}_rolling_trend'] = df[col].diff()
        
        return pd.DataFrame(rolling_features)
    
    @staticmethod
    def normalize_metrics(metrics_dict):
        """Normalize developer metrics to standard ranges"""
        normalized = metrics_dict.copy()
        
        # Define normalization ranges for each metric
        normalization_rules = {
            'keystrokes_per_min': {'min': 0, 'max': 500},
            'context_switch_rate': {'min': 0, 'max': 1},
            'focus_ratio': {'min': 0, 'max': 1},
            'emotion_variance': {'min': 0, 'max': 1},
            'commit_density': {'min': 0, 'max': 10},
            'idle_ratio': {'min': 0, 'max': 1},
            'files_opened': {'min': 1, 'max': 20}
        }
        
        for metric, rule in normalization_rules.items():
            if metric in normalized:
                value = normalized[metric]
                normalized[metric] = max(0, min(1, (value - rule['min']) / (rule['max'] - rule['min'])))
        
        return normalized
    
    @staticmethod
    def detect_anomalies(metrics_dict, historical_data=None):
        """Detect anomalous patterns in developer metrics"""
        anomalies = []
        
        # Simple threshold-based anomaly detection
        if metrics_dict.get('keystrokes_per_min', 0) > 400:
            anomalies.append('Unusually high typing speed detected')
        
        if metrics_dict.get('context_switch_rate', 0) > 0.8:
            anomalies.append('Excessive context switching detected')
        
        if metrics_dict.get('focus_ratio', 1) < 0.2:
            anomalies.append('Very low focus ratio detected')
        
        if metrics_dict.get('idle_ratio', 0) > 0.7:
            anomalies.append('High idle time detected')
        
        return anomalies
    
    @staticmethod
    def create_feature_vector(raw_metrics, timestamp=None):
        """Create a complete feature vector from raw developer metrics"""
        features = raw_metrics.copy()
        
        # Add time features if timestamp provided
        if timestamp:
            time_features = FeatureEngineer.extract_time_features(timestamp)
            features.update(time_features)
        
        # Normalize metrics
        features = FeatureEngineer.normalize_metrics(features)
        
        # Add derived features
        features['productivity_score'] = (
            features.get('focus_ratio', 0) * 0.4 +
            (1 - features.get('context_switch_rate', 0)) * 0.3 +
            (1 - features.get('idle_ratio', 0)) * 0.3
        )
        
        features['stress_indicator'] = (
            features.get('emotion_variance', 0) * 0.5 +
            features.get('context_switch_rate', 0) * 0.5
        )
        
        return features