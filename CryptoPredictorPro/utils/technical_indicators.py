import pandas as pd
import numpy as np
import ta
from ta.utils import dropna
import streamlit as st

class TechnicalIndicators:
    def __init__(self):
        pass
    
    def calculate_sma(self, df, period=20):
        """
        Calculate Simple Moving Average
        """
        if 'price' not in df.columns:
            return df
        
        df = df.copy()
        df[f'SMA_{period}'] = df['price'].rolling(window=period).mean()
        return df
    
    def calculate_ema(self, df, period=20):
        """
        Calculate Exponential Moving Average
        """
        if 'price' not in df.columns:
            return df
        
        df = df.copy()
        df[f'EMA_{period}'] = df['price'].ewm(span=period).mean()
        return df
    
    def calculate_rsi(self, df, period=14):
        """
        Calculate Relative Strength Index
        """
        if 'price' not in df.columns:
            return df
        
        df = df.copy()
        try:
            df['RSI'] = ta.momentum.RSIIndicator(
                close=df['price'],
                window=period
            ).rsi()
        except Exception as e:
            st.warning(f"Error calculating RSI: {str(e)}")
            df['RSI'] = 50  # Default neutral value
        
        return df
    
    def calculate_macd(self, df, fast=12, slow=26, signal=9):
        """
        Calculate MACD (Moving Average Convergence Divergence)
        """
        if 'price' not in df.columns:
            return df
        
        df = df.copy()
        try:
            macd_indicator = ta.trend.MACD(
                close=df['price'],
                window_fast=fast,
                window_slow=slow,
                window_sign=signal
            )
            
            df['MACD'] = macd_indicator.macd()
            df['MACD_signal'] = macd_indicator.macd_signal()
            df['MACD_histogram'] = macd_indicator.macd_diff()
        except Exception as e:
            st.warning(f"Error calculating MACD: {str(e)}")
            df['MACD'] = 0
            df['MACD_signal'] = 0
            df['MACD_histogram'] = 0
        
        return df
    
    def calculate_bollinger_bands(self, df, period=20, std_dev=2):
        """
        Calculate Bollinger Bands
        """
        if 'price' not in df.columns:
            return df
        
        df = df.copy()
        try:
            bollinger = ta.volatility.BollingerBands(
                close=df['price'],
                window=period,
                window_dev=std_dev
            )
            
            df['BB_upper'] = bollinger.bollinger_hband()
            df['BB_middle'] = bollinger.bollinger_mavg()
            df['BB_lower'] = bollinger.bollinger_lband()
        except Exception as e:
            st.warning(f"Error calculating Bollinger Bands: {str(e)}")
            sma = df['price'].rolling(window=period).mean()
            std = df['price'].rolling(window=period).std()
            df['BB_upper'] = sma + (std * std_dev)
            df['BB_middle'] = sma
            df['BB_lower'] = sma - (std * std_dev)
        
        return df
    
    def calculate_stochastic(self, df, k_period=14, d_period=3):
        """
        Calculate Stochastic Oscillator
        """
        if 'price' not in df.columns:
            return df
        
        df = df.copy()
        try:
            # For stochastic, we need high, low, close
            # Using price as close, and approximating high/low
            high = df['price'].rolling(window=k_period).max()
            low = df['price'].rolling(window=k_period).min()
            
            df['%K'] = 100 * (df['price'] - low) / (high - low)
            df['%D'] = df['%K'].rolling(window=d_period).mean()
        except Exception as e:
            st.warning(f"Error calculating Stochastic: {str(e)}")
            df['%K'] = 50
            df['%D'] = 50
        
        return df
    
    def calculate_all_indicators(self, df):
        """
        Calculate all technical indicators
        """
        if df.empty or 'price' not in df.columns:
            return df
        
        df = df.copy()
        
        # Moving Averages
        df = self.calculate_sma(df, 10)
        df = self.calculate_sma(df, 20)
        df = self.calculate_sma(df, 50)
        df = self.calculate_ema(df, 12)
        df = self.calculate_ema(df, 26)
        
        # Momentum Indicators
        df = self.calculate_rsi(df, 14)
        df = self.calculate_macd(df)
        df = self.calculate_stochastic(df)
        
        # Volatility Indicators
        df = self.calculate_bollinger_bands(df)
        
        return df
    
    def get_trading_signals(self, df):
        """
        Generate trading signals based on technical indicators
        """
        if df.empty or len(df) < 50:
            return []
        
        signals = []
        latest = df.iloc[-1]
        
        try:
            # RSI Signals
            if 'RSI' in df.columns:
                rsi = latest['RSI']
                if rsi < 30:
                    signals.append({
                        'type': 'BUY',
                        'indicator': 'RSI',
                        'reason': f'RSI oversold at {rsi:.1f}',
                        'strength': 'Strong' if rsi < 25 else 'Moderate'
                    })
                elif rsi > 70:
                    signals.append({
                        'type': 'SELL',
                        'indicator': 'RSI',
                        'reason': f'RSI overbought at {rsi:.1f}',
                        'strength': 'Strong' if rsi > 75 else 'Moderate'
                    })
            
            # MACD Signals
            if 'MACD' in df.columns and 'MACD_signal' in df.columns:
                macd = latest['MACD']
                macd_signal = latest['MACD_signal']
                
                if macd > macd_signal and len(df) > 1:
                    prev_macd = df.iloc[-2]['MACD']
                    prev_signal = df.iloc[-2]['MACD_signal']
                    
                    if prev_macd <= prev_signal:  # Crossover
                        signals.append({
                            'type': 'BUY',
                            'indicator': 'MACD',
                            'reason': 'MACD bullish crossover',
                            'strength': 'Moderate'
                        })
                elif macd < macd_signal and len(df) > 1:
                    prev_macd = df.iloc[-2]['MACD']
                    prev_signal = df.iloc[-2]['MACD_signal']
                    
                    if prev_macd >= prev_signal:  # Crossover
                        signals.append({
                            'type': 'SELL',
                            'indicator': 'MACD',
                            'reason': 'MACD bearish crossover',
                            'strength': 'Moderate'
                        })
            
            # Bollinger Bands Signals
            if all(col in df.columns for col in ['BB_upper', 'BB_lower', 'price']):
                price = latest['price']
                bb_upper = latest['BB_upper']
                bb_lower = latest['BB_lower']
                
                if price <= bb_lower:
                    signals.append({
                        'type': 'BUY',
                        'indicator': 'Bollinger Bands',
                        'reason': 'Price at lower Bollinger Band',
                        'strength': 'Moderate'
                    })
                elif price >= bb_upper:
                    signals.append({
                        'type': 'SELL',
                        'indicator': 'Bollinger Bands',
                        'reason': 'Price at upper Bollinger Band',
                        'strength': 'Moderate'
                    })
            
            # Moving Average Signals
            if 'SMA_10' in df.columns and 'SMA_20' in df.columns:
                sma_10 = latest['SMA_10']
                sma_20 = latest['SMA_20']
                
                if sma_10 > sma_20 and len(df) > 1:
                    prev_sma_10 = df.iloc[-2]['SMA_10']
                    prev_sma_20 = df.iloc[-2]['SMA_20']
                    
                    if prev_sma_10 <= prev_sma_20:  # Golden Cross
                        signals.append({
                            'type': 'BUY',
                            'indicator': 'Moving Average',
                            'reason': 'Golden Cross (SMA 10 > SMA 20)',
                            'strength': 'Strong'
                        })
                elif sma_10 < sma_20 and len(df) > 1:
                    prev_sma_10 = df.iloc[-2]['SMA_10']
                    prev_sma_20 = df.iloc[-2]['SMA_20']
                    
                    if prev_sma_10 >= prev_sma_20:  # Death Cross
                        signals.append({
                            'type': 'SELL',
                            'indicator': 'Moving Average',
                            'reason': 'Death Cross (SMA 10 < SMA 20)',
                            'strength': 'Strong'
                        })
            
        except Exception as e:
            st.warning(f"Error generating signals: {str(e)}")
        
        return signals
    
    def get_market_sentiment(self, df):
        """
        Calculate overall market sentiment based on indicators
        """
        if df.empty or len(df) < 20:
            return "Neutral", 50
        
        try:
            latest = df.iloc[-1]
            bullish_score = 0
            total_indicators = 0
            
            # RSI sentiment
            if 'RSI' in df.columns:
                rsi = latest['RSI']
                if rsi < 30:
                    bullish_score += 2
                elif rsi < 50:
                    bullish_score += 1
                elif rsi > 70:
                    bullish_score -= 2
                elif rsi > 50:
                    bullish_score -= 1
                total_indicators += 1
            
            # MACD sentiment
            if 'MACD' in df.columns and 'MACD_signal' in df.columns:
                if latest['MACD'] > latest['MACD_signal']:
                    bullish_score += 1
                else:
                    bullish_score -= 1
                total_indicators += 1
            
            # Moving Average sentiment
            if 'SMA_10' in df.columns and 'SMA_20' in df.columns:
                if latest['SMA_10'] > latest['SMA_20']:
                    bullish_score += 1
                else:
                    bullish_score -= 1
                total_indicators += 1
            
            # Bollinger Bands sentiment
            if all(col in df.columns for col in ['BB_upper', 'BB_lower', 'BB_middle', 'price']):
                price = latest['price']
                bb_middle = latest['BB_middle']
                
                if price > bb_middle:
                    bullish_score += 1
                else:
                    bullish_score -= 1
                total_indicators += 1
            
            if total_indicators == 0:
                return "Neutral", 50
            
            # Calculate sentiment score (0-100)
            sentiment_score = max(0, min(100, 50 + (bullish_score / total_indicators * 25)))
            
            # Determine sentiment label
            if sentiment_score >= 70:
                sentiment = "Very Bullish"
            elif sentiment_score >= 60:
                sentiment = "Bullish"
            elif sentiment_score >= 40:
                sentiment = "Neutral"
            elif sentiment_score >= 30:
                sentiment = "Bearish"
            else:
                sentiment = "Very Bearish"
            
            return sentiment, sentiment_score
            
        except Exception as e:
            st.warning(f"Error calculating market sentiment: {str(e)}")
            return "Neutral", 50
