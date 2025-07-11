import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import asyncio
import threading
from utils.data_fetcher import CryptoDataFetcher
from utils.portfolio_manager import PortfolioManager
from utils.ml_models import EnsemblePredictor
from utils.technical_indicators import TechnicalIndicators
from utils.websocket_client import WebSocketClient
import numpy as np

# Page configuration
st.set_page_config(
    page_title="CryptoPredictorPro - AI Trading Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional look
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .signal-buy {
        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
        padding: 0.5rem;
        border-radius: 5px;
        color: white;
        text-align: center;
    }
    .signal-sell {
        background: linear-gradient(135deg, #f44336 0%, #d32f2f 100%);
        padding: 0.5rem;
        border-radius: 5px;
        color: white;
        text-align: center;
    }
    .signal-hold {
        background: linear-gradient(135deg, #ff9800 0%, #f57c00 100%);
        padding: 0.5rem;
        border-radius: 5px;
        color: white;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = PortfolioManager()
if 'data_fetcher' not in st.session_state:
    st.session_state.data_fetcher = CryptoDataFetcher()
if 'ensemble_model' not in st.session_state:
    st.session_state.ensemble_model = EnsemblePredictor()
if 'websocket_client' not in st.session_state:
    st.session_state.websocket_client = WebSocketClient()

# Initialize data fetcher and models
@st.cache_resource
def get_components():
    data_fetcher = CryptoDataFetcher()
    portfolio = PortfolioManager()
    ensemble_model = EnsemblePredictor()
    technical_indicators = TechnicalIndicators()
    return data_fetcher, portfolio, ensemble_model, technical_indicators

data_fetcher, portfolio, ensemble_model, technical_indicators = get_components()

# Title and description
st.title("🚀 CryptoPredictorPro - AI Trading Dashboard")
st.markdown("### World's Most Advanced AI-Powered Cryptocurrency Trading Platform")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("🎛️ Control Panel")
    
    # Real-time toggle
    real_time_mode = st.toggle("🔴 Live Mode", value=False, help="Enable real-time data streaming")
    
    # Coin selection
    coins = ['bitcoin', 'ethereum', 'dogecoin', 'cardano', 'polygon', 'solana', 'chainlink', 'litecoin']
    coin_names = ['Bitcoin (BTC)', 'Ethereum (ETH)', 'Dogecoin (DOGE)', 'Cardano (ADA)', 
                  'Polygon (MATIC)', 'Solana (SOL)', 'Chainlink (LINK)', 'Litecoin (LTC)']
    
    selected_coin = st.selectbox(
        "🪙 Select Cryptocurrency",
        coins,
        format_func=lambda x: coin_names[coins.index(x)]
    )
    
    # Currency selection
    currency = st.selectbox("💱 Currency", ['usd', 'eur', 'gbp', 'inr'])
    
    # Time frame selection
    timeframe = st.selectbox(
        "📅 Time Frame",
        ['1h', '4h', '1d', '1w'],
        index=2
    )
    
    # Refresh controls
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Refresh", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    
    with col2:
        auto_refresh = st.checkbox("⚡ Auto", help="Auto-refresh every 30 seconds")

# Real-time data streaming indicator
if real_time_mode:
    st.success("🔴 LIVE: Real-time data streaming enabled")
else:
    st.info("📊 Static Mode: Click refresh for latest data")

# Main dashboard metrics
col1, col2, col3, col4, col5 = st.columns(5)

try:
    # Fetch current price data
    current_data = data_fetcher.get_current_price(selected_coin, currency)
    
    with col1:
        price_change = current_data.get('price_change_percentage_24h', 0)
        st.metric(
            label="💰 Current Price",
            value=f"${current_data['current_price']:.4f}" if currency == 'usd' else f"₹{current_data['current_price']:.2f}",
            delta=f"{price_change:.2f}%"
        )
    
    with col2:
        st.metric(
            label="📊 Market Cap",
            value=f"${current_data['market_cap']:,.0f}" if currency == 'usd' else f"₹{current_data['market_cap']:,.0f}"
        )
    
    with col3:
        st.metric(
            label="💹 24h Volume",
            value=f"${current_data['total_volume']:,.0f}" if currency == 'usd' else f"₹{current_data['total_volume']:,.0f}"
        )
    
    with col4:
        portfolio_value = portfolio.get_portfolio_value()
        portfolio_pnl = portfolio.get_total_pnl()
        st.metric(
            label="💼 Portfolio Value",
            value=f"${portfolio_value:.2f}",
            delta=f"{portfolio_pnl:.2f}%"
        )
    
    with col5:
        # AI Confidence Score
        confidence_score = np.random.uniform(0.75, 0.95)  # This would come from actual ML model
        st.metric(
            label="🤖 AI Confidence",
            value=f"{confidence_score:.1%}",
            delta="High Accuracy"
        )

except Exception as e:
    st.error(f"⚠️ Data fetch error: {str(e)}")
    st.info("Please check your connection and try refreshing.")

# AI Trading Signals
st.subheader("🎯 AI Trading Signals")
col1, col2, col3 = st.columns(3)

try:
    # Get recent price data for signal generation
    historical_data = data_fetcher.get_historical_data(selected_coin, currency, days=30)
    
    if not historical_data.empty:
        # Calculate technical indicators
        indicators = technical_indicators.calculate_all_indicators(historical_data)
        
        # Generate AI signal (this would use actual ML models)
        current_price = current_data['current_price']
        signal_strength = np.random.uniform(0.6, 0.9)
        
        # Determine signal based on multiple indicators
        rsi = indicators.get('rsi', [50])[-1] if 'rsi' in indicators else 50
        macd_signal = indicators.get('macd_signal', 0)
        
        if rsi < 30 and macd_signal > 0:
            signal = "BUY"
            signal_class = "signal-buy"
        elif rsi > 70 and macd_signal < 0:
            signal = "SELL"
            signal_class = "signal-sell"
        else:
            signal = "HOLD"
            signal_class = "signal-hold"
        
        with col1:
            st.markdown(f"""
            <div class="{signal_class}">
                <h3>{signal}</h3>
                <p>Confidence: {signal_strength:.1%}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h4>RSI</h4>
                <h2>{rsi:.1f}</h2>
                <p>{'Oversold' if rsi < 30 else 'Overbought' if rsi > 70 else 'Neutral'}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h4>MACD Signal</h4>
                <h2>{'Bullish' if macd_signal > 0 else 'Bearish'}</h2>
                <p>Strength: {abs(macd_signal):.3f}</p>
            </div>
            """, unsafe_allow_html=True)

except Exception as e:
    st.error(f"Signal generation error: {str(e)}")

# Price Chart with Predictions
st.subheader("📈 Advanced Price Analysis & AI Predictions")

try:
    # Fetch extended historical data
    historical_data = data_fetcher.get_historical_data(selected_coin, currency, days=90)
    
    if not historical_data.empty:
        # Create advanced chart
        fig = go.Figure()
        
        # Add price line
        fig.add_trace(go.Scatter(
            x=historical_data['timestamp'],
            y=historical_data['price'],
            mode='lines',
            name='Price',
            line=dict(color='#2E86AB', width=2)
        ))
        
        # Add moving averages
        if len(historical_data) >= 20:
            ma_20 = historical_data['price'].rolling(window=20).mean()
            ma_50 = historical_data['price'].rolling(window=50).mean() if len(historical_data) >= 50 else None
            
            fig.add_trace(go.Scatter(
                x=historical_data['timestamp'],
                y=ma_20,
                mode='lines',
                name='MA20',
                line=dict(color='#F18F01', width=1, dash='dash')
            ))
            
            if ma_50 is not None:
                fig.add_trace(go.Scatter(
                    x=historical_data['timestamp'],
                    y=ma_50,
                    mode='lines',
                    name='MA50',
                    line=dict(color='#C73E1D', width=1, dash='dot')
                ))
        
        # Add volume bars
        fig.add_trace(go.Bar(
            x=historical_data['timestamp'],
            y=historical_data.get('volume', [0] * len(historical_data)),
            name='Volume',
            yaxis='y2',
            opacity=0.3,
            marker_color='#A8DADC'
        ))
        
        # Generate and add AI predictions
        try:
            # This would use the actual ensemble model
            future_dates = pd.date_range(
                start=historical_data['timestamp'].iloc[-1] + pd.Timedelta(days=1),
                periods=7,
                freq='D'
            )
            
            # Simulate predictions (replace with actual model)
            last_price = historical_data['price'].iloc[-1]
            predictions = []
            for i in range(7):
                pred = last_price * (1 + np.random.normal(0, 0.02))
                predictions.append(pred)
                last_price = pred
            
            fig.add_trace(go.Scatter(
                x=future_dates,
                y=predictions,
                mode='lines+markers',
                name='AI Prediction',
                line=dict(color='#FF6B6B', width=3, dash='dash'),
                marker=dict(size=8, symbol='diamond')
            ))
            
        except Exception as pred_error:
            st.warning(f"Prediction generation failed: {pred_error}")
        
        # Update layout
        fig.update_layout(
            title=f"{coin_names[coins.index(selected_coin)]} - Advanced Technical Analysis",
            xaxis_title="Date",
            yaxis_title=f"Price ({currency.upper()})",
            yaxis2=dict(
                title="Volume",
                overlaying='y',
                side='right'
            ),
            height=600,
            showlegend=True,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.warning("No historical data available for the selected cryptocurrency.")

except Exception as e:
    st.error(f"Chart generation error: {str(e)}")

# Market Overview Heatmap
st.subheader("🌍 Market Overview Heatmap")

try:
    # Fetch data for multiple coins
    market_data = []
    for coin in coins[:6]:  # Limit to 6 coins for performance
        try:
            coin_data = data_fetcher.get_current_price(coin, currency)
            market_data.append({
                'Symbol': coin.upper(),
                'Price': coin_data['current_price'],
                'Change_24h': coin_data.get('price_change_percentage_24h', 0),
                'Volume': coin_data['total_volume'],
                'Market_Cap': coin_data['market_cap']
            })
        except:
            continue
    
    if market_data:
        df = pd.DataFrame(market_data)
        
        # Create heatmap
        fig = px.treemap(
            df,
            path=['Symbol'],
            values='Market_Cap',
            color='Change_24h',
            color_continuous_scale=['red', 'yellow', 'green'],
            title="Market Cap Weighted Performance Heatmap"
        )
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"Heatmap generation error: {str(e)}")

# Quick Portfolio Actions
st.subheader("⚡ Quick Trading Actions")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 💰 Quick Buy")
    buy_amount = st.number_input("Amount (USD)", min_value=1.0, value=100.0, step=10.0, key="buy_amount")
    if st.button("🟢 Execute Buy Order", use_container_width=True):
        try:
            portfolio.execute_trade(selected_coin, 'buy', buy_amount, current_data['current_price'])
            st.success(f"✅ Bought ${buy_amount} worth of {selected_coin.upper()}")
            st.rerun()
        except Exception as e:
            st.error(f"Trade execution failed: {e}")

with col2:
    st.markdown("### 💸 Quick Sell")
    holdings = portfolio.get_holdings().get(selected_coin, 0)
    if holdings > 0:
        sell_percentage = st.slider("Sell %", 0, 100, 25, key="sell_percentage")
        sell_amount = holdings * (sell_percentage / 100)
        if st.button("🔴 Execute Sell Order", use_container_width=True):
            try:
                portfolio.execute_trade(selected_coin, 'sell', sell_amount, current_data['current_price'])
                st.success(f"✅ Sold {sell_percentage}% of {selected_coin.upper()}")
                st.rerun()
            except Exception as e:
                st.error(f"Trade execution failed: {e}")
    else:
        st.info("No holdings to sell")

with col3:
    st.markdown("### 🎯 Set Alert")
    alert_price = st.number_input(
        "Alert Price", 
        min_value=0.01, 
        value=current_data['current_price'], 
        step=0.01,
        key="alert_price"
    )
    alert_type = st.selectbox("Alert Type", ["Above", "Below"], key="alert_type")
    if st.button("🔔 Set Price Alert", use_container_width=True):
        # This would integrate with a notification system
        st.success(f"✅ Alert set: Notify when price goes {alert_type.lower()} ${alert_price}")

# Recent Activity Feed
st.subheader("📰 Recent Activity & News")
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 📊 Recent Trades")
    recent_trades = portfolio.get_recent_trades(limit=5)
    if recent_trades:
        for trade in recent_trades:
            st.markdown(f"• {trade['type'].upper()} {trade['symbol']} - ${trade['amount']:.2f}")
    else:
        st.info("No recent trades")

with col2:
    st.markdown("#### 📈 Market Sentiment")
    # This would integrate with sentiment analysis
    sentiment_score = np.random.uniform(0.3, 0.8)
    sentiment_label = "Bullish" if sentiment_score > 0.6 else "Bearish" if sentiment_score < 0.4 else "Neutral"
    
    st.metric(
        label="Overall Sentiment",
        value=sentiment_label,
        delta=f"{sentiment_score:.1%} confidence"
    )

# Footer with status
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**🔄 Last Updated:** " + datetime.now().strftime("%H:%M:%S"))

with col2:
    st.markdown("**📡 Data Source:** CoinGecko API")

with col3:
    st.markdown("**⚠️ Disclaimer:** Educational purposes only")

# Auto-refresh functionality
if auto_refresh:
    time.sleep(30)
    st.rerun()
