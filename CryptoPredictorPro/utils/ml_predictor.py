import pandas as pd
import numpy as np
from prophet import Prophet
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class CryptoPricePredictor:
    def __init__(self):
        self.model = None
        self.is_trained = False
        self.training_data = None
        
    def prepare_data(self, df):
        """
        Prepare data for Prophet model
        """
        if df.empty:
            return pd.DataFrame()
        
        # Prophet expects columns named 'ds' and 'y'
        data = df.copy()
        data = data.rename(columns={'timestamp': 'ds', 'price': 'y'})
        
        # Ensure we have the required columns
        if 'ds' not in data.columns or 'y' not in data.columns:
            raise ValueError("Data must contain 'timestamp' and 'price' columns")
        
        # Remove any NaN values
        data = data.dropna()
        
        # Ensure ds is datetime
        data['ds'] = pd.to_datetime(data['ds'])
        
        # Sort by date
        data = data.sort_values('ds')
        
        return data[['ds', 'y']]
    
    def train_model(self, df, coin_name="Cryptocurrency"):
        """
        Train Prophet model on historical data
        """
        try:
            # Prepare data
            data = self.prepare_data(df)
            
            if data.empty or len(data) < 10:
                raise ValueError("Insufficient data for training (need at least 10 data points)")
            
            # Initialize Prophet model
            self.model = Prophet(
                changepoint_prior_scale=0.05,
                seasonality_prior_scale=10,
                holidays_prior_scale=10,
                daily_seasonality=False,
                weekly_seasonality=True,
                yearly_seasonality=True,
                interval_width=0.95
            )
            
            # Add custom seasonalities for crypto markets
            self.model.add_seasonality(
                name='monthly',
                period=30.5,
                fourier_order=5
            )
            
            # Fit the model
            with st.spinner(f"Training prediction model for {coin_name}..."):
                self.model.fit(data)
            
            self.is_trained = True
            self.training_data = data
            
            return True
            
        except Exception as e:
            st.error(f"Error training model: {str(e)}")
            return False
    
    def predict_price(self, days_ahead=7):
        """
        Predict future prices
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        try:
            # Create future dataframe
            future = self.model.make_future_dataframe(periods=days_ahead, freq='D')
            
            # Make predictions
            forecast = self.model.predict(future)
            
            # Extract predictions for future dates only
            future_predictions = forecast.tail(days_ahead)
            
            # Calculate prediction confidence
            predictions = []
            for _, row in future_predictions.iterrows():
                predictions.append({
                    'date': row['ds'],
                    'predicted_price': row['yhat'],
                    'lower_bound': row['yhat_lower'],
                    'upper_bound': row['yhat_upper'],
                    'confidence': min(100, max(0, 100 - abs(row['yhat_upper'] - row['yhat_lower']) / row['yhat'] * 100))
                })
            
            return predictions
            
        except Exception as e:
            st.error(f"Error making predictions: {str(e)}")
            return []
    
    def get_model_performance(self):
        """
        Calculate model performance metrics
        """
        if not self.is_trained or self.model is None or self.training_data is None:
            return {}
        
        try:
            # Make predictions on training data
            forecast = self.model.predict(self.training_data)
            
            # Calculate metrics
            actual = self.training_data['y'].values
            predicted = forecast['yhat'].values
            
            mae = mean_absolute_error(actual, predicted)
            mse = mean_squared_error(actual, predicted)
            rmse = np.sqrt(mse)
            mape = np.mean(np.abs((actual - predicted) / actual)) * 100
            
            return {
                'mae': mae,
                'mse': mse,
                'rmse': rmse,
                'mape': mape,
                'data_points': len(actual)
            }
            
        except Exception as e:
            st.error(f"Error calculating performance: {str(e)}")
            return {}
    
    def predict_trend(self, days_ahead=7):
        """
        Predict price trend (up/down/sideways)
        """
        predictions = self.predict_price(days_ahead)
        
        if not predictions:
            return "Unknown"
        
        current_price = self.training_data['y'].iloc[-1]
        future_price = predictions[-1]['predicted_price']
        
        price_change = (future_price - current_price) / current_price * 100
        
        if price_change > 5:
            return "Bullish (Strong Up)"
        elif price_change > 1:
            return "Bullish (Up)"
        elif price_change > -1:
            return "Neutral (Sideways)"
        elif price_change > -5:
            return "Bearish (Down)"
        else:
            return "Bearish (Strong Down)"
    
    def get_key_insights(self):
        """
        Generate key insights from the model
        """
        if not self.is_trained or self.model is None:
            return []
        
        insights = []
        
        try:
            # Get components
            future = self.model.make_future_dataframe(periods=1, freq='D')
            forecast = self.model.predict(future)
            
            # Trend analysis
            trend = forecast['trend'].iloc[-1] - forecast['trend'].iloc[0]
            if trend > 0:
                insights.append("📈 Overall trend is positive")
            else:
                insights.append("📉 Overall trend is negative")
            
            # Seasonality insights
            weekly_effect = forecast['weekly'].iloc[-1]
            if abs(weekly_effect) > 0.01:
                if weekly_effect > 0:
                    insights.append("📅 Weekly seasonality shows positive effect")
                else:
                    insights.append("📅 Weekly seasonality shows negative effect")
            
            # Volatility
            price_std = self.training_data['y'].std()
            price_mean = self.training_data['y'].mean()
            cv = price_std / price_mean * 100
            
            if cv > 50:
                insights.append("⚠️ High volatility detected")
            elif cv > 25:
                insights.append("⚡ Moderate volatility")
            else:
                insights.append("✅ Low volatility")
            
        except Exception as e:
            insights.append(f"⚠️ Error generating insights: {str(e)}")
        
        return insights
    
    def simple_trend_prediction(self, df, days_ahead=7):
        """
        Simple trend-based prediction as fallback
        """
        if df.empty or len(df) < 3:
            return []
        
        # Calculate simple moving average trend
        df = df.sort_values('timestamp')
        prices = df['price'].values
        
        # Calculate trend using linear regression
        x = np.arange(len(prices))
        coeffs = np.polyfit(x, prices, 1)
        trend_slope = coeffs[0]
        
        # Generate predictions
        last_price = prices[-1]
        predictions = []
        
        for i in range(1, days_ahead + 1):
            predicted_price = last_price + (trend_slope * i)
            
            # Add some uncertainty bounds
            uncertainty = last_price * 0.1  # 10% uncertainty
            
            predictions.append({
                'date': df['timestamp'].iloc[-1] + timedelta(days=i),
                'predicted_price': predicted_price,
                'lower_bound': predicted_price - uncertainty,
                'upper_bound': predicted_price + uncertainty,
                'confidence': 60  # Lower confidence for simple model
            })
        
        return predictions
