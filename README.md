"# PulseCode ML Service

ML microservice for PulseCode developer wellness monitoring - hybrid XGBoost/Prophet fatigue prediction

## Overview

This service provides real-time fatigue prediction for developers using a hybrid machine learning approach combining XGBoost for immediate predictions and Prophet for trend forecasting.

## Features

- **Real-time Fatigue Prediction**: Instant fatigue scoring based on developer metrics
- **Trend Forecasting**: 7-day fatigue trend predictions using Prophet
- **Hybrid ML Approach**: Combines XGBoost and Prophet models for robust predictions
- **RESTful API**: Simple HTTP endpoints for integration
- **Docker Support**: Containerized deployment with Docker Compose
- **Health Monitoring**: Built-in health checks and logging

## API Endpoints

### Health Check

```
GET /health
```

### Predict Fatigue

```
POST /predict/fatigue
Content-Type: application/json

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
```

**Response:**

```json
{
  "fatigue_score": 0.342,
  "confidence": 0.85,
  "trend_forecast": [
    { "date": "2025-11-02", "predicted_fatigue": 0.35 },
    { "date": "2025-11-03", "predicted_fatigue": 0.32 }
  ],
  "recommendations": ["Good energy levels. Keep up the productive work!"]
}
```

## Quick Start

### Local Development

1. **Clone the repository**

```bash
git clone https://github.com/Falcon-J/pulsecode-ml-service.git
cd pulsecode-ml-service
```

2. **Set up virtual environment**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Configure environment**

```bash
cp .env.example .env
# Edit .env with your configuration
```

5. **Run the service**

```bash
python app.py
```

The service will be available at `http://localhost:8000`

### Docker Deployment

1. **Build and run with Docker Compose**

```bash
docker-compose up --build
```

2. **Test the service**

```bash
curl http://localhost:8000/health
```

## Configuration

Environment variables can be set in `.env` file:

- `DEBUG`: Enable debug mode (default: false)
- `PORT`: Service port (default: 8000)
- `MODEL_PATH`: Path to saved models (default: models/saved/)
- `LOG_LEVEL`: Logging level (default: INFO)

## Model Training

The service automatically initializes with pre-trained models. To retrain:

```bash
curl -X POST http://localhost:8000/train
```

## Testing

Run the test suite:

```bash
pytest tests/
```

## Architecture

```
├── app.py                 # Main Flask application
├── config.py             # Configuration management
├── models/
│   ├── predictor.py      # Prediction engine
│   └── trainer.py        # Model training pipeline
├── utils/
│   ├── feature_engineering.py  # Feature processing
│   └── database.py       # Data persistence
└── tests/                # Test suite
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License - see LICENSE file for details"
