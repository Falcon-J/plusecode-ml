import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from prophet import Prophet
import joblib
import os
from datetime import datetime, timedelta
from config import Config
import logging

logger = logging.getLogger(__name__)

class ModelTrainer:
    """Model training pipeline for fatigue prediction"""
    
    def __init__(self):
        self.config = Config()
        self._ensure_model_directory()
    
    def _ensure_model_directory(self):
        """Create model directory if it doesn't exist"""
        os.makedirs(self.config.MODEL_PATH, exist_ok=True)
    
    def generate_synthetic_data(self, n_samples=1000):
        """Generate synthetic training data for model development"""
        np.random.seed(42)
        
        data = pd.DataFrame({
            'keystrokes_per_min': np.random.normal(250, 50, n_samples),
            'context_switch_rate': np.random.uniform(0, 1, n_samples),
            'focus_ratio': np.random.uniform(0.3, 1.0, n_samples),
            'emotion_variance': np.random.uniform(0, 0.5, n_samples),
            'commit_density': np.random.normal(3, 1, n_samples),
            'idle_ratio': np.random.uniform(0, 0.6, n_samples),
            'files_opened': np.random.randint(1, 15, n_samples),
            'hour_sin': np.sin(np.linspace(0, 24, n_samples) * (2 * np.pi / 24)),
            'hour_cos': np.cos(np.linspace(0, 24, n_samples) * (2 * np.pi / 24)),
        })
        
        # Generate realistic fatigue score based on features
        data['fatigue_score'] = (
            0.3 * (1 - data['focus_ratio']) +
            0.2 * data['context_switch_rate'] +
            0.2 * data['idle_ratio'] +
            0.1 * data['emotion_variance'] +
            0.1 * np.maximum(0, (data['keystrokes_per_min'] - 300) / 100) +  # High typing = fatigue
            np.random.normal(0, 0.05, n_samples)
        )
        
        # Clamp fatigue score between 0 and 1
        data['fatigue_score'] = np.clip(data['fatigue_score'], 0, 1)
        
        return data
    
    def train_xgboost_model(self, data):
        """Train XGBoost model for real-time fatigue prediction"""
        logger.info("Training XGBoost model...")
        
        X = data.drop(columns=['fatigue_score'])
        y = data['fatigue_score']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        model = XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=4,
            random_state=42
        )
        
        model.fit(X_train, y_train)
        
        # Evaluate
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        
        logger.info(f"XGBoost - Train R²: {train_score:.3f}, Test R²: {test_score:.3f}")
        
        # Save model
        model_path = os.path.join(self.config.MODEL_PATH, self.config.XGB_MODEL_FILE)
        joblib.dump(model, model_path)
        logger.info(f"XGBoost model saved to {model_path}")
        
        return model
    
    def train_prophet_model(self, data):
        """Train Prophet model for trend forecasting"""
        logger.info("Training Prophet model...")
        
        # Prepare time series data
        df_time = pd.DataFrame({
            'ds': pd.date_range(start='2024-01-01', periods=len(data), freq='H'),
            'y': data['fatigue_score'].values
        })
        
        # Train Prophet model
        model = Prophet(
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=False,
            changepoint_prior_scale=0.05
        )
        
        model.fit(df_time)
        
        # Save model
        model_path = os.path.join(self.config.MODEL_PATH, self.config.PROPHET_MODEL_FILE)
        joblib.dump(model, model_path)
        logger.info(f"Prophet model saved to {model_path}")
        
        return model
    
    def train_models(self, n_samples=None):
        """Train both XGBoost and Prophet models"""
        try:
            logger.info("Starting model training pipeline...")
            
            # Generate or load training data
            if n_samples is None:
                n_samples = self.config.MIN_TRAINING_SAMPLES
            
            data = self.generate_synthetic_data(n_samples)
            logger.info(f"Generated {len(data)} training samples")
            
            # Train models
            xgb_model = self.train_xgboost_model(data)
            prophet_model = self.train_prophet_model(data)
            
            logger.info("✅ All models trained successfully!")
            
            return {
                'xgb_model': xgb_model,
                'prophet_model': prophet_model,
                'training_samples': len(data)
            }
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise
    
    def evaluate_models(self):
        """Evaluate trained models performance"""
        try:
            # Load models
            xgb_path = os.path.join(self.config.MODEL_PATH, self.config.XGB_MODEL_FILE)
            prophet_path = os.path.join(self.config.MODEL_PATH, self.config.PROPHET_MODEL_FILE)
            
            if not os.path.exists(xgb_path) or not os.path.exists(prophet_path):
                logger.error("Models not found. Please train models first.")
                return None
            
            xgb_model = joblib.load(xgb_path)
            prophet_model = joblib.load(prophet_path)
            
            # Generate test data
            test_data = self.generate_synthetic_data(200)
            X_test = test_data.drop(columns=['fatigue_score'])
            y_test = test_data['fatigue_score']
            
            # XGBoost evaluation
            xgb_predictions = xgb_model.predict(X_test)
            xgb_mse = np.mean((y_test - xgb_predictions) ** 2)
            
            logger.info(f"XGBoost MSE: {xgb_mse:.4f}")
            
            return {
                'xgb_mse': xgb_mse,
                'test_samples': len(test_data)
            }
            
        except Exception as e:
            logger.error(f"Evaluation failed: {str(e)}")
            return None