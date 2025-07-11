import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import streamlit as st

class BasePredictor:
    """Base class for all prediction models"""
    
    def __init__(self):
        self.model = None
        self.is_trained = False
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
    
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for training/prediction"""
        features = data.copy()
        
        # Technical indicators as features
        if 'price' in features.columns:
            # Moving averages
            features['sma_5'] = features['price'].rolling(5).mean()
            features['sma_10'] = features['price'].rolling(10).mean()
            features['sma_20'] = features['price'].rolling(20).mean()
            
            # Price changes
            features['price_change'] = features['price'].pct_change()
            features['price_change_5'] = features['price'].pct_change(5)
            features['price_change_10'] = features['price'].pct_change(10)
            
            # Volatility
            features['volatility'] = features['price_change'].rolling(10).std()
            
            # RSI approximation
            delta = features['price'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            features['rsi'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            bb_period = 20
            bb_std = 2
            bb_ma = features['price'].rolling(bb_period).mean()
            bb_std_val = features['price'].rolling(bb_period).std()
            features['bb_upper'] = bb_ma + (bb_std_val * bb_std)
            features['bb_lower'] = bb_ma - (bb_std_val * bb_std)
            features['bb_position'] = (features['price'] - bb_lower) / (bb_upper - bb_lower)
        
        # Volume features if available
        if 'volume' in features.columns:
            features['volume_ma'] = features['volume'].rolling(10).mean()
            features['volume_ratio'] = features['volume'] / features['volume_ma']
        
        # Time-based features
        if 'timestamp' in features.columns:
            features['hour'] = features['timestamp'].dt.hour
            features['day_of_week'] = features['timestamp'].dt.dayofweek
            features['month'] = features['timestamp'].dt.month
        
        return features
    
    def create_sequences(self, data: np.ndarray, sequence_length: int = 60) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training"""
        X, y = [], []
        for i in range(sequence_length, len(data)):
            X.append(data[i-sequence_length:i])
            y.append(data[i])
        return np.array(X), np.array(y)

class ProphetPredictor(BasePredictor):
    """Facebook Prophet time series forecasting model"""
    
    def __init__(self):
        super().__init__()
        self.prophet_available = PROPHET_AVAILABLE
    
    def predict(self, data: pd.DataFrame, days: int = 7) -> Dict:
        """
        Generate Prophet predictions
        
        Args:
            data: DataFrame with timestamp and price columns
            days: Number of days to predict
            
        Returns:
            Dictionary with predictions and confidence intervals
        """
        if not self.prophet_available:
            return self._fallback_prediction(data, days)
        
        try:
            # Prepare data for Prophet
            prophet_data = data[['timestamp', 'price']].copy()
            prophet_data.columns = ['ds', 'y']
            prophet_data = prophet_data.dropna()
            
            if len(prophet_data) < 30:
                raise ValueError("Insufficient data for Prophet model")
            
            # Initialize and fit Prophet model
            model = Prophet(
                changepoint_prior_scale=0.05,
                seasonality_prior_scale=10,
                seasonality_mode='multiplicative',
                daily_seasonality=True,
                weekly_seasonality=True,
                yearly_seasonality=False
            )
            
            model.fit(prophet_data)
            
            # Create future dataframe
            future = model.make_future_dataframe(periods=days)
            forecast = model.predict(future)
            
            # Extract predictions
            predictions = forecast.tail(days)
            
            return {
                'predictions': predictions['yhat'].tolist(),
                'upper_bound': predictions['yhat_upper'].tolist(),
                'lower_bound': predictions['yhat_lower'].tolist(),
                'dates': predictions['ds'].tolist(),
                'confidence': 0.8,
                'model_type': 'Prophet'
            }
            
        except Exception as e:
            st.warning(f"Prophet prediction failed: {e}")
            return self._fallback_prediction(data, days)
    
    def _fallback_prediction(self, data: pd.DataFrame, days: int) -> Dict:
        """Fallback prediction using simple trend analysis"""
        if len(data) < 10:
            return {'predictions': [], 'confidence': 0}
        
        # Simple linear trend
        prices = data['price'].values
        x = np.arange(len(prices))
        z = np.polyfit(x, prices, 1)
        trend = z[0]
        
        # Add some randomness based on historical volatility
        volatility = data['price'].pct_change().std()
        
        last_price = prices[-1]
        predictions = []
        
        for i in range(1, days + 1):
            # Trend + random walk
            pred = last_price + (trend * i) + np.random.normal(0, volatility * last_price * np.sqrt(i))
            predictions.append(max(0, pred))  # Ensure non-negative
        
        return {
            'predictions': predictions,
            'upper_bound': [p * 1.1 for p in predictions],
            'lower_bound': [p * 0.9 for p in predictions],
            'confidence': 0.6,
            'model_type': 'Trend-based'
        }

class LSTMPredictor(BasePredictor):
    """Long Short-Term Memory neural network for price prediction"""
    
    def __init__(self):
        super().__init__()
        self.tensorflow_available = TENSORFLOW_AVAILABLE
        self.sequence_length = 60
    
    def predict(self, data: pd.DataFrame, days: int = 7) -> Dict:
        """
        Generate LSTM predictions
        
        Args:
            data: DataFrame with price data
            days: Number of days to predict
            
        Returns:
            Dictionary with predictions
        """
        if not self.tensorflow_available:
            return self._fallback_prediction(data, days)
        
        try:
            # Prepare data
            features = self.prepare_features(data)
            
            # Select features for LSTM
            feature_columns = ['price', 'volume'] if 'volume' in features.columns else ['price']
            features = features[feature_columns].dropna()
            
            if len(features) < self.sequence_length + 30:
                raise ValueError("Insufficient data for LSTM model")
            
            # Scale features
            scaled_features = self.feature_scaler.fit_transform(features.values)
            
            # Create sequences
            X, y = self.create_sequences(scaled_features, self.sequence_length)
            
            if len(X) < 20:
                raise ValueError("Insufficient sequences for training")
            
            # Split data
            train_size = int(len(X) * 0.8)
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]
            
            # Build LSTM model
            model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(self.sequence_length, len(feature_columns))),
                Dropout(0.2),
                LSTM(50, return_sequences=False),
                Dropout(0.2),
                Dense(25),
                Dense(len(feature_columns))
            ])
            
            model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
            
            # Train model
            model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0, validation_split=0.1)
            
            # Generate predictions
            predictions = []
            current_sequence = X[-1].reshape(1, self.sequence_length, len(feature_columns))
            
            for _ in range(days):
                pred = model.predict(current_sequence, verbose=0)
                predictions.append(pred[0])
                
                # Update sequence for next prediction
                current_sequence = np.roll(current_sequence, -1, axis=1)
                current_sequence[0, -1] = pred[0]
            
            # Inverse scale predictions (only price column)
            predictions_array = np.array(predictions)
            
            # Create dummy array for inverse scaling
            dummy_array = np.zeros((len(predictions), len(feature_columns)))
            dummy_array[:, 0] = predictions_array[:, 0]  # Price predictions
            
            predictions_rescaled = self.feature_scaler.inverse_transform(dummy_array)
            price_predictions = predictions_rescaled[:, 0].tolist()
            
            # Calculate confidence based on model performance
            test_pred = model.predict(X_test, verbose=0)
            mse = mean_squared_error(y_test[:, 0], test_pred[:, 0])
            confidence = max(0.5, 1 - (mse / np.var(y_test[:, 0])))
            
            return {
                'predictions': price_predictions,
                'confidence': min(0.95, confidence),
                'model_type': 'LSTM'
            }
            
        except Exception as e:
            st.warning(f"LSTM prediction failed: {e}")
            return self._fallback_prediction(data, days)
    
    def _fallback_prediction(self, data: pd.DataFrame, days: int) -> Dict:
        """Fallback prediction using moving average"""
        if len(data) < 10:
            return {'predictions': [], 'confidence': 0}
        
        # Use exponential moving average for prediction
        prices = data['price'].values
        ema_span = min(20, len(prices) // 2)
        ema = pd.Series(prices).ewm(span=ema_span).mean()
        
        # Predict using trend continuation
        trend = ema.iloc[-1] - ema.iloc[-min(5, len(ema))]
        last_price = prices[-1]
        
        predictions = []
        for i in range(1, days + 1):
            pred = last_price + (trend * i * 0.5)  # Dampen trend
            predictions.append(max(0, pred))
        
        return {
            'predictions': predictions,
            'confidence': 0.65,
            'model_type': 'EMA-based'
        }

class XGBoostPredictor(BasePredictor):
    """XGBoost gradient boosting for price prediction"""
    
    def __init__(self):
        super().__init__()
        self.xgboost_available = XGBOOST_AVAILABLE
    
    def predict(self, data: pd.DataFrame, days: int = 7) -> Dict:
        """
        Generate XGBoost predictions
        
        Args:
            data: DataFrame with price data
            days: Number of days to predict
            
        Returns:
            Dictionary with predictions
        """
        if not self.xgboost_available:
            return self._fallback_prediction(data, days)
        
        try:
            # Prepare features
            features = self.prepare_features(data)
            
            # Select feature columns (exclude timestamp and target)
            exclude_cols = ['timestamp', 'price']
            feature_cols = [col for col in features.columns if col not in exclude_cols]
            
            # Create target variable (next day price)
            features['target'] = features['price'].shift(-1)
            
            # Drop rows with NaN values
            clean_data = features.dropna()
            
            if len(clean_data) < 50:
                raise ValueError("Insufficient clean data for XGBoost model")
            
            # Prepare training data
            X = clean_data[feature_cols]
            y = clean_data['target']
            
            # Split data
            train_size = int(len(X) * 0.8)
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]
            
            # Train XGBoost model
            model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )
            
            model.fit(X_train, y_train)
            
            # Generate predictions
            predictions = []
            current_features = X.iloc[-1:].copy()
            
            for i in range(days):
                pred = model.predict(current_features)[0]
                predictions.append(max(0, pred))
                
                # Update features for next prediction
                # This is simplified - in practice, you'd update all relevant features
                current_features = current_features.copy()
                # Update moving averages and other features based on new prediction
                # For simplicity, we'll just use the last known features
            
            # Calculate confidence based on model performance
            test_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, test_pred)
            confidence = max(0.5, 1 - (mse / np.var(y_test)))
            
            return {
                'predictions': predictions,
                'confidence': min(0.9, confidence),
                'model_type': 'XGBoost',
                'feature_importance': dict(zip(feature_cols, model.feature_importances_))
            }
            
        except Exception as e:
            st.warning(f"XGBoost prediction failed: {e}")
            return self._fallback_prediction(data, days)
    
    def _fallback_prediction(self, data: pd.DataFrame, days: int) -> Dict:
        """Fallback prediction using Random Forest"""
        try:
            # Prepare simple features
            features = data[['price']].copy()
            features['price_lag1'] = features['price'].shift(1)
            features['price_lag2'] = features['price'].shift(2)
            features['price_ma5'] = features['price'].rolling(5).mean()
            features['target'] = features['price'].shift(-1)
            
            clean_data = features.dropna()
            
            if len(clean_data) < 20:
                return {'predictions': [], 'confidence': 0}
            
            X = clean_data[['price_lag1', 'price_lag2', 'price_ma5']]
            y = clean_data['target']
            
            # Use Random Forest as fallback
            model = RandomForestRegressor(n_estimators=50, random_state=42)
            model.fit(X, y)
            
            # Generate predictions
            predictions = []
            last_features = X.iloc[-1:].values
            
            for _ in range(days):
                pred = model.predict(last_features)[0]
                predictions.append(max(0, pred))
                # Update features (simplified)
                last_features[0, 0] = last_features[0, 1]  # shift lags
                last_features[0, 1] = pred
            
            return {
                'predictions': predictions,
                'confidence': 0.7,
                'model_type': 'Random Forest'
            }
            
        except Exception:
            # Ultimate fallback
            if len(data) > 0:
                last_price = data['price'].iloc[-1]
                return {
                    'predictions': [last_price] * days,
                    'confidence': 0.5,
                    'model_type': 'Last Price'
                }
            return {'predictions': [], 'confidence': 0}

class EnsemblePredictor:
    """Ensemble model combining multiple prediction models"""
    
    def __init__(self):
        self.prophet = ProphetPredictor()
        self.lstm = LSTMPredictor()
        self.xgboost = XGBoostPredictor()
        
        # Model weights based on typical performance
        self.weights = {
            'prophet': 0.4,
            'lstm': 0.35,
            'xgboost': 0.25
        }
    
    def predict(self, data: pd.DataFrame, days: int = 7, individual_predictions: Dict = None) -> Dict:
        """
        Generate ensemble predictions combining multiple models
        
        Args:
            data: DataFrame with price data
            days: Number of days to predict
            individual_predictions: Pre-computed individual model predictions
            
        Returns:
            Dictionary with ensemble predictions
        """
        try:
            predictions = {}
            confidences = {}
            
            if individual_predictions:
                # Use provided predictions
                predictions = individual_predictions
                confidences = {k: v.get('confidence', 0.5) for k, v in predictions.items()}
            else:
                # Generate individual predictions
                with st.spinner("Training Prophet model..."):
                    prophet_pred = self.prophet.predict(data, days)
                    if prophet_pred.get('predictions'):
                        predictions['prophet'] = prophet_pred
                        confidences['prophet'] = prophet_pred.get('confidence', 0.5)
                
                with st.spinner("Training LSTM model..."):
                    lstm_pred = self.lstm.predict(data, days)
                    if lstm_pred.get('predictions'):
                        predictions['lstm'] = lstm_pred
                        confidences['lstm'] = lstm_pred.get('confidence', 0.5)
                
                with st.spinner("Training XGBoost model..."):
                    xgb_pred = self.xgboost.predict(data, days)
                    if xgb_pred.get('predictions'):
                        predictions['xgboost'] = xgb_pred
                        confidences['xgboost'] = xgb_pred.get('confidence', 0.5)
            
            if not predictions:
                return {'predictions': [], 'confidence': 0, 'model_type': 'Ensemble (Failed)'}
            
            # Calculate weighted ensemble
            ensemble_predictions = []
            total_weight = 0
            
            for day in range(days):
                weighted_sum = 0
                day_weight = 0
                
                for model_name, pred_data in predictions.items():
                    if pred_data.get('predictions') and len(pred_data['predictions']) > day:
                        weight = self.weights.get(model_name, 0) * confidences.get(model_name, 0.5)
                        weighted_sum += pred_data['predictions'][day] * weight
                        day_weight += weight
                
                if day_weight > 0:
                    ensemble_predictions.append(weighted_sum / day_weight)
                    total_weight += day_weight
                else:
                    # Fallback to last price if no predictions available
                    ensemble_predictions.append(data['price'].iloc[-1] if len(data) > 0 else 0)
            
            # Calculate ensemble confidence
            avg_confidence = np.mean(list(confidences.values())) if confidences else 0.5
            ensemble_confidence = min(0.95, avg_confidence * 1.1)  # Slight boost for ensemble
            
            # Calculate prediction bounds
            if len(predictions) > 1:
                # Use prediction variance for bounds
                all_preds = []
                for day in range(days):
                    day_preds = []
                    for pred_data in predictions.values():
                        if pred_data.get('predictions') and len(pred_data['predictions']) > day:
                            day_preds.append(pred_data['predictions'][day])
                    all_preds.append(day_preds)
                
                upper_bounds = []
                lower_bounds = []
                
                for day_preds in all_preds:
                    if day_preds:
                        std = np.std(day_preds)
                        mean = np.mean(day_preds)
                        upper_bounds.append(mean + 1.96 * std)  # 95% confidence
                        lower_bounds.append(max(0, mean - 1.96 * std))
                    else:
                        price = ensemble_predictions[len(upper_bounds)] if ensemble_predictions else 0
                        upper_bounds.append(price * 1.1)
                        lower_bounds.append(price * 0.9)
            else:
                # Single model bounds
                single_pred = list(predictions.values())[0]
                upper_bounds = single_pred.get('upper_bound', [p * 1.1 for p in ensemble_predictions])
                lower_bounds = single_pred.get('lower_bound', [p * 0.9 for p in ensemble_predictions])
            
            return {
                'predictions': ensemble_predictions,
                'upper_bound': upper_bounds,
                'lower_bound': lower_bounds,
                'confidence': ensemble_confidence,
                'model_type': 'Ensemble',
                'individual_models': {k: v.get('model_type', k) for k, v in predictions.items()},
                'model_weights': {k: self.weights.get(k, 0) * confidences.get(k, 0.5) 
                                for k in predictions.keys()}
            }
            
        except Exception as e:
            st.error(f"Ensemble prediction failed: {e}")
            return {'predictions': [], 'confidence': 0, 'model_type': 'Ensemble (Error)'}
    
    def quick_predict(self, data: pd.DataFrame, days: int = 1) -> Dict:
        """Quick prediction using simplified models for real-time signals"""
        try:
            if len(data) < 10:
                return {'prediction': data['price'].iloc[-1] if len(data) > 0 else 0, 'confidence': 0.5}
            
            # Simple trend-based prediction for speed
            prices = data['price'].values
            
            # Linear trend
            x = np.arange(len(prices))
            z = np.polyfit(x, prices, 1)
            trend_pred = prices[-1] + z[0] * days
            
            # EMA prediction
            ema = pd.Series(prices).ewm(span=10).mean().iloc[-1]
            ema_pred = ema + (ema - pd.Series(prices).ewm(span=20).mean().iloc[-1]) * days * 0.5
            
            # Weighted average
            prediction = 0.6 * trend_pred + 0.4 * ema_pred
            
            return {
                'prediction': max(0, prediction),
                'confidence': 0.7,
                'model_type': 'Quick Ensemble'
            }
            
        except Exception:
            return {'prediction': data['price'].iloc[-1] if len(data) > 0 else 0, 'confidence': 0.5}
