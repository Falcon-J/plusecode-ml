import sqlite3
import json
from datetime import datetime
from config import Config
import logging

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Simple database manager for storing predictions and metrics"""
    
    def __init__(self):
        self.config = Config()
        self.db_path = 'pulsecode.db'  # Simplified for demo
        self.init_database()
    
    def init_database(self):
        """Initialize database tables"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Predictions table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS predictions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        input_metrics TEXT,
                        fatigue_score REAL,
                        confidence REAL,
                        recommendations TEXT
                    )
                ''')
                
                # Metrics table for historical data
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        user_id TEXT,
                        keystrokes_per_min REAL,
                        context_switch_rate REAL,
                        focus_ratio REAL,
                        emotion_variance REAL,
                        commit_density REAL,
                        idle_ratio REAL,
                        files_opened INTEGER
                    )
                ''')
                
                conn.commit()
                logger.info("Database initialized successfully")
                
        except Exception as e:
            logger.error(f"Database initialization error: {str(e)}")
    
    def store_prediction(self, input_metrics, prediction_result):
        """Store prediction result in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO predictions (input_metrics, fatigue_score, confidence, recommendations)
                    VALUES (?, ?, ?, ?)
                ''', (
                    json.dumps(input_metrics),
                    prediction_result['fatigue_score'],
                    prediction_result['confidence'],
                    json.dumps(prediction_result['recommendations'])
                ))
                
                conn.commit()
                logger.info("Prediction stored successfully")
                
        except Exception as e:
            logger.error(f"Error storing prediction: {str(e)}")
    
    def store_metrics(self, metrics, user_id=None):
        """Store developer metrics in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO metrics (
                        user_id, keystrokes_per_min, context_switch_rate,
                        focus_ratio, emotion_variance, commit_density,
                        idle_ratio, files_opened
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    user_id,
                    metrics.get('keystrokes_per_min'),
                    metrics.get('context_switch_rate'),
                    metrics.get('focus_ratio'),
                    metrics.get('emotion_variance'),
                    metrics.get('commit_density'),
                    metrics.get('idle_ratio'),
                    metrics.get('files_opened')
                ))
                
                conn.commit()
                logger.info("Metrics stored successfully")
                
        except Exception as e:
            logger.error(f"Error storing metrics: {str(e)}")
    
    def get_recent_predictions(self, limit=10):
        """Get recent predictions from database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT timestamp, fatigue_score, confidence
                    FROM predictions
                    ORDER BY timestamp DESC
                    LIMIT ?
                ''', (limit,))
                
                results = cursor.fetchall()
                
                return [
                    {
                        'timestamp': row[0],
                        'fatigue_score': row[1],
                        'confidence': row[2]
                    }
                    for row in results
                ]
                
        except Exception as e:
            logger.error(f"Error retrieving predictions: {str(e)}")
            return []
    
    def get_historical_metrics(self, user_id=None, days=7):
        """Get historical metrics for analysis"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                if user_id:
                    cursor.execute('''
                        SELECT * FROM metrics
                        WHERE user_id = ? AND timestamp >= datetime('now', '-{} days')
                        ORDER BY timestamp DESC
                    '''.format(days), (user_id,))
                else:
                    cursor.execute('''
                        SELECT * FROM metrics
                        WHERE timestamp >= datetime('now', '-{} days')
                        ORDER BY timestamp DESC
                    '''.format(days))
                
                results = cursor.fetchall()
                return results
                
        except Exception as e:
            logger.error(f"Error retrieving historical metrics: {str(e)}")
            return []