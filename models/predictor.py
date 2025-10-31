import numpy as np
import pandas as pd
import joblib
import os
from datetime import datetime, timedelta
from config import Config
import logging

logger = logging.getLogger(__name__)

class FatiguePredictionEngine:
    """Hybrid XGBoost/Prophet fatigue prediction engine"""
    
    def __init__(self):
        self.config = Config()
        self.xgb_model = None
        self.prophet_model = None
        self.load_models()
    
    def load_models(self):
        """Load trained models from disk"""
        try:
            xgb_path = os.path.join(self.config.MODEL_PATH, self.config.XGB_MODEL_FILE)
            prophet_path = os.path.join(self.config.MODEL_PATH, self.config.PROPHET_MODEL_FILE)
            
            if os.path.exists(xgb_path):
                self.xgb_model = joblib.load(xgb_path)
                logger.info("XGBoost model loaded successfully")
            
            if os.path.exists(prophet_path):
                self.prophet_model = joblib.load(prophet_path)
                logger.info("Prophet model loaded successfully")
                
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            # Initialize with default models if loading fails
            self._initialize_default_models()
    
    def _initialize_default_models(self):
        """Initialize models with default training if no saved models exist"""
        logger.info("Initializing default models...")
        from .trainer import ModelTrainer
        trainer = ModelTrainer()
        trainer.train_models()
        self.load_models()
    
    def predict_fatigue(self, metrics_dict):
        """
        Predict fatigue score using hybrid XGBoost/Prophet approach
        
        Args:
            metrics_dict: Dictionary containing developer metrics
            
        Returns:
            Dictionary with prediction results
        """
        try:
            # Prepare input data
            X = pd.DataFrame([metrics_dict])
            
            # XGBoost prediction (real-time score)
            realtime_score = 0.5  # Default fallback
            if self.xgb_model:
                realtime_score = self.xgb_model.predict(X)[0]
            
            # Prophet prediction (trend forecast)
            trend_score = 0.5  # Default fallback
            if self.prophet_model:
                future = self.prophet_model.make_future_dataframe(periods=1)
                forecast = self.prophet_model.predict(future)
                trend_score = forecast['yhat'].iloc[-1]
            
            # Hybrid score calculation
            final_score = 0.7 * realtime_score + 0.3 * trend_score
            final_score = max(0, min(1, final_score))  # Clamp between 0-1
            
            # Generate recommendations
            recommendations = self._generate_recommendations(final_score, metrics_dict)
            
            # Calculate confidence based on model availability and input quality
            confidence = self._calculate_confidence(metrics_dict)
            
            return {
                'fatigue_score': round(final_score, 3),
                'confidence': round(confidence, 3),
                'trend_forecast': self._get_trend_forecast(),
                'recommendations': recommendations
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return {
                'fatigue_score': 0.5,
                'confidence': 0.1,
                'trend_forecast': [],
                'recommendations': ['Unable to generate prediction. Please check input data.']
            }
    
    def _generate_recommendations(self, fatigue_score, metrics):
        """Generate actionable recommendations based on fatigue score and metrics"""
        recommendations = []
        
        if fatigue_score > 0.7:
            recommendations.append("High fatigue detected. Consider taking a break.")
            if metrics.get('focus_ratio', 0) < 0.5:
                recommendations.append("Focus ratio is low. Try eliminating distractions.")
            if metrics.get('context_switch_rate', 0) > 0.6:
                recommendations.append("High context switching detected. Focus on single tasks.")
        
        elif fatigue_score > 0.4:
            recommendations.append("Moderate fatigue. Monitor your energy levels.")
            if metrics.get('idle_ratio', 0) > 0.3:
                recommendations.append("Consider more active coding sessions.")
        
        else:
            recommendations.append("Good energy levels. Keep up the productive work!")
        
        return recommendations
    
    def _calculate_confidence(self, metrics):
        """Calculate prediction confidence based on input quality and model availability"""
        confidence = 0.5  # Base confidence
        
        # Increase confidence if both models are available
        if self.xgb_model and self.prophet_model:
            confidence += 0.3
        elif self.xgb_model or self.prophet_model:
            confidence += 0.1
        
        # Adjust based on input data quality
        required_fields = ['keystrokes_per_min', 'focus_ratio', 'context_switch_rate']
        valid_fields = sum(1 for field in required_fields if field in metrics and metrics[field] is not None)
        confidence += (valid_fields / len(required_fields)) * 0.2
        
        return min(1.0, confidence)
    
    def _get_trend_forecast(self):
        """Get 7-day fatigue trend forecast"""
        if not self.prophet_model:
            return []
        
        try:
            future = self.prophet_model.make_future_dataframe(periods=7)
            forecast = self.prophet_model.predict(future)
            
            # Return last 7 predictions
            trend_data = []
            for i in range(-7, 0):
                trend_data.append({
                    'date': forecast['ds'].iloc[i].strftime('%Y-%m-%d'),
                    'predicted_fatigue': round(forecast['yhat'].iloc[i], 3)
                })
            
            return trend_data
        except Exception as e:
            logger.error(f"Trend forecast error: {str(e)}")
            return []
    
    def retrain_models(self):
        """Trigger model retraining"""
        try:
            from .trainer import ModelTrainer
            trainer = ModelTrainer()
            trainer.train_models()
            self.load_models()
            logger.info("Models retrained successfully")
        except Exception as e:
            logger.error(f"Retraining error: {str(e)}")
            raise