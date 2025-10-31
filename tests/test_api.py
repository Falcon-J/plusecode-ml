import pytest
import json
from app import app

@pytest.fixture
def client():
    """Create test client"""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_health_check(client):
    """Test health check endpoint"""
    response = client.get('/health')
    assert response.status_code == 200
    
    data = json.loads(response.data)
    assert data['status'] == 'healthy'
    assert data['service'] == 'pulsecode-ml-service'

def test_predict_fatigue_success(client):
    """Test successful fatigue prediction"""
    test_data = {
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
    
    response = client.post('/predict/fatigue', 
                          data=json.dumps(test_data),
                          content_type='application/json')
    
    assert response.status_code == 200
    
    data = json.loads(response.data)
    assert 'fatigue_score' in data
    assert 'confidence' in data
    assert 'recommendations' in data
    assert isinstance(data['fatigue_score'], float)
    assert 0 <= data['fatigue_score'] <= 1

def test_predict_fatigue_missing_fields(client):
    """Test prediction with missing required fields"""
    test_data = {
        "keystrokes_per_min": 250,
        "focus_ratio": 0.8
        # Missing other required fields
    }
    
    response = client.post('/predict/fatigue',
                          data=json.dumps(test_data),
                          content_type='application/json')
    
    assert response.status_code == 400
    
    data = json.loads(response.data)
    assert 'error' in data
    assert 'Missing required fields' in data['error']

def test_predict_fatigue_no_data(client):
    """Test prediction with no data"""
    response = client.post('/predict/fatigue',
                          data='',
                          content_type='application/json')
    
    assert response.status_code == 400

def test_predict_fatigue_invalid_json(client):
    """Test prediction with invalid JSON"""
    response = client.post('/predict/fatigue',
                          data='invalid json',
                          content_type='application/json')
    
    assert response.status_code == 400