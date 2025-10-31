from flask import Flask, request, jsonify
from models.predictor import FatiguePredictionEngine
from utils.database import DatabaseManager
from config import Config
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config.from_object(Config)

# Initialize services
predictor = FatiguePredictionEngine()
db_manager = DatabaseManager()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'pulsecode-ml-service',
        'version': '1.0.0'
    })

@app.route('/predict/fatigue', methods=['POST'])
def predict_fatigue():
    """
    Predict developer fatigue score
    Expected payload:
    {
        "keystrokes_per_min": 250,
        "context_switch_rate": 0.3,
        "focus_ratio": 0.8,
        "emotion_variance": 0.2,
        "commit_density": 3,
        "idle_ratio": 0.1,
        "files_opened": 5,
        "hour_sin": 0.5,
        "hour_cos": 0.8
    }
    """
    try:
        data = request.get_json(force=True)
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
    except Exception as json_error:
        return jsonify({'error': 'Invalid JSON format'}), 400
    
    try:
        
        # Validate required fields
        required_fields = [
            'keystrokes_per_min', 'context_switch_rate', 'focus_ratio',
            'emotion_variance', 'commit_density', 'idle_ratio',
            'files_opened', 'hour_sin', 'hour_cos'
        ]
        
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({
                'error': f'Missing required fields: {missing_fields}'
            }), 400
        
        # Get prediction
        result = predictor.predict_fatigue(data)
        
        # Log prediction (optional)
        logger.info(f"Fatigue prediction: {result}")
        
        return jsonify({
            'fatigue_score': result['fatigue_score'],
            'confidence': result['confidence'],
            'trend_forecast': result['trend_forecast'],
            'recommendations': result['recommendations']
        })
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/train', methods=['POST'])
def retrain_models():
    """Trigger model retraining"""
    try:
        # This would typically be secured with authentication
        predictor.retrain_models()
        return jsonify({'message': 'Models retrained successfully'})
    except Exception as e:
        logger.error(f"Training error: {str(e)}")
        return jsonify({'error': 'Training failed'}), 500

if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=int(app.config.get('PORT', 8000)),
        debug=app.config.get('DEBUG', False)
    )