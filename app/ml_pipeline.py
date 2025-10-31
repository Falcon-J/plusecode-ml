import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from prophet import Prophet
import joblib
import datetime

# ---------- 1️⃣ Synthetic Data Generation ----------
def generate_mock_data(n_samples=1000):
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

    # Fatigue score: synthetic target (random but correlated)
    data['fatigue_score'] = (
        0.3 * (1 - data['focus_ratio'])
        + 0.2 * data['context_switch_rate']
        + 0.2 * data['idle_ratio']
        + 0.1 * data['emotion_variance']
        + np.random.normal(0, 0.05, n_samples)
    )
    return data


# ---------- 2️⃣ Model Training ----------
def train_models():
    data = generate_mock_data()
    X = data.drop(columns=['fatigue_score'])
    y = data['fatigue_score']

    # Split for XGBoost
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # XGBoost Model
    xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=4)
    xgb_model.fit(X_train, y_train)

    # Prophet model (simulate time-based fatigue trend)
    df_time = pd.DataFrame({
        'ds': pd.date_range(start='2025-01-01', periods=len(y), freq='H'),
        'y': y.values
    })
    prophet_model = Prophet(daily_seasonality=True)
    prophet_model.fit(df_time)

    # Save models
    joblib.dump(xgb_model, 'models/xgb_model.pkl')
    joblib.dump(prophet_model, 'models/prophet_model.pkl')
    print("✅ Models trained and saved successfully.")


# ---------- 3️⃣ Prediction Logic ----------
def predict_fatigue(metrics_dict):
    """Combine XGBoost and Prophet outputs"""
    xgb_model = joblib.load('models/xgb_model.pkl')
    prophet_model = joblib.load('models/prophet_model.pkl')

    X = pd.DataFrame([metrics_dict])
    realtime_score = xgb_model.predict(X)[0]

    # Forecast next 7 days fatigue
    future = prophet_model.make_future_dataframe(periods=7)
    forecast = prophet_model.predict(future)
    trend_score = forecast['yhat'].iloc[-1]

    # Hybrid score
    final_score = 0.7 * realtime_score + 0.3 * trend_score
    return round(final_score, 3)


if __name__ == "__main__":
    train_models()
