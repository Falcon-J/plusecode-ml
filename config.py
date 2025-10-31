import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Application configuration"""
    
    # Flask settings
    DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
    PORT = int(os.getenv('PORT', 8000))
    
    # Model settings
    MODEL_PATH = os.getenv('MODEL_PATH', 'models/saved/')
    XGB_MODEL_FILE = 'xgb_model.pkl'
    PROPHET_MODEL_FILE = 'prophet_model.pkl'
    
    # Database settings (if needed)
    DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///pulsecode.db')
    
    # API settings
    API_KEY = os.getenv('API_KEY', None)
    RATE_LIMIT = os.getenv('RATE_LIMIT', '100/hour')
    
    # Logging
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    
    # Model training settings
    RETRAIN_INTERVAL_HOURS = int(os.getenv('RETRAIN_INTERVAL_HOURS', 24))
    MIN_TRAINING_SAMPLES = int(os.getenv('MIN_TRAINING_SAMPLES', 1000))