import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
from utils.data_fetcher import CryptoDataFetcher
from utils.technical_indicators import TechnicalIndicators
from utils.websocket_client import WebSocketClient
import time

st.set_page_config(page_title="Market Data", page_icon="📊", layout="wide")

# Initialize components
@st.cache_resource
def get_components():
    return CryptoDataFetcher(), TechnicalIndicators()

data_fetcher, technical_indicators = get_components()

st.title("📊 Real-time Market Data & Technical Analysis")
st.markdown("### Advanced cryptocurrency market analysis with 20+ technical indicators")

# Sidebar controls
with st.sidebar:
    st.header("📈 Analysis Controls")
    
    # Coin selection with search
    coins = ['bitcoin', 'ethereum', 'binancecoin', 'cardano', 'solana', 'polkadot', 
             'dogecoin', 'polygon', 'chainlink', 'litecoin', 'avalanche', 'uniswap']
    coin_names = ['Bitcoin (BTC)', 'Ethereum (ETH)', 'BNB (BNB)', 'Cardano (ADA)',
                  'Solana (SOL)', 'Polkadot (DOT)', 'Dogecoin (DOGE)', 'Polygon (MATIC)',
                  'Chainlink (LINK)', 'Litecoin (LTC)', 'Avalanche (AVAX)', 'Uniswap (UNI)']
    
    selected_coin = st.selectbox(
        "🪙 Select Cryptocurrency",
        coins,
        format_func=lambda x: coin_names[coins.index(x)]
    )
    
    # Time period selection
    time_periods = {
        '1 Day': 1,
        '7 Days': 7,
        '30 Days': 30,
        '90 Days': 90,
        '1 Year': 365
    }
    selected_period = st.selectbox("📅 Time Period", list(time_periods.keys()), index=2)
    days = time_periods[selected_period]
    
    # Technical indicators selection
    st.subheader("🔧 Technical Indicators")
    show_ma = st.checkbox("Moving Averages", value=True)
    show_bollinger = st.checkbox("Bollinger Bands", value=True)
    show_rsi = st.checkbox("RSI", value=True)
    show_macd = st.checkbox("MACD", value=True)
    show_volume = st.checkbox("Volume Analysis", value=True)
    
    # Advanced indicators
    st.subheader("🔬 Advanced Indicators")
    show_ichimoku = st.checkbox("Ichimoku Cloud")
    show_fibonacci = st.checkbox("Fibonacci Retracements")
    show_support_resistance = st.checkbox("Support/Resistance Levels")
    
    # Real-time toggle
    real_time = st.toggle("🔴 Real-time Updates", help="Updates every 10 seconds")

# Main content area
try:
    # Fetch current data
    current_data = data_fetcher.get_current_price(selected_coin, 'usd')
    historical_data = data_fetcher.get_historical_data(selected_coin, 'usd', days=days)
    
    if historical_data.empty:
        st.error("No data available for the selected cryptocurrency and time period.")
        st.stop()
    
    # Current price metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "💰 Current Price",
            f"${current_data['current_price']:.4f}",
            f"{current_data.get('price_change_percentage_24h', 0):.2f}%"
        )
    
    with col2:
        st.metric(
            "📊 Market Cap",
            f"${current_data['market_cap']:,.0f}",
            f"Rank #{current_data.get('market_cap_rank', 'N/A')}"
        )
    
    with col3:
        st.metric(
            "💹 24h Volume",
            f"${current_data['total_volume']:,.0f}",
            f"{current_data.get('price_change_percentage_24h', 0):.2f}%"
        )
    
    with col4:
        high_24h = current_data.get('high_24h', current_data['current_price'])
        low_24h = current_data.get('low_24h', current_data['current_price'])
        st.metric(
            "📈 24h High",
            f"${high_24h:.4f}",
            f"Range: {((high_24h - low_24h) / low_24h * 100):.2f}%"
        )
    
    with col5:
        # Calculate volatility
        if len(historical_data) > 1:
            volatility = historical_data['price'].pct_change().std() * np.sqrt(365) * 100
        else:
            volatility = 0
        st.metric(
            "📊 Volatility",
            f"{volatility:.1f}%",
            "Annualized"
        )
    
    # Calculate technical indicators
    indicators = technical_indicators.calculate_all_indicators(historical_data)
    
    # Main price chart with technical analysis
    st.subheader("📈 Advanced Price Chart with Technical Analysis")
    
    # Create subplot with secondary y-axis for volume
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxis=True,
        vertical_spacing=0.05,
        subplot_titles=('Price & Technical Indicators', 'Volume Analysis', 'Oscillators'),
        row_heights=[0.6, 0.2, 0.2]
    )
    
    # Main price line
    fig.add_trace(
        go.Scatter(
            x=historical_data['timestamp'],
            y=historical_data['price'],
            mode='lines',
            name='Price',
            line=dict(color='#2E86AB', width=2)
        ),
        row=1, col=1
    )
    
    # Moving averages
    if show_ma and 'sma_20' in indicators:
        fig.add_trace(
            go.Scatter(
                x=historical_data['timestamp'],
                y=indicators['sma_20'],
                mode='lines',
                name='SMA 20',
                line=dict(color='#F18F01', width=1)
            ),
            row=1, col=1
        )
        
        if 'sma_50' in indicators:
            fig.add_trace(
                go.Scatter(
                    x=historical_data['timestamp'],
                    y=indicators['sma_50'],
                    mode='lines',
                    name='SMA 50',
                    line=dict(color='#C73E1D', width=1)
                ),
                row=1, col=1
            )
    
    # Bollinger Bands
    if show_bollinger and 'bb_upper' in indicators:
        fig.add_trace(
            go.Scatter(
                x=historical_data['timestamp'],
                y=indicators['bb_upper'],
                mode='lines',
                name='BB Upper',
                line=dict(color='rgba(255,0,0,0.3)', width=1),
                fill=None
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=historical_data['timestamp'],
                y=indicators['bb_lower'],
                mode='lines',
                name='BB Lower',
                line=dict(color='rgba(255,0,0,0.3)', width=1),
                fill='tonexty',
                fillcolor='rgba(255,0,0,0.1)'
            ),
            row=1, col=1
        )
    
    # Volume
    if show_volume and 'volume' in historical_data.columns:
        fig.add_trace(
            go.Bar(
                x=historical_data['timestamp'],
                y=historical_data['volume'],
                name='Volume',
                marker_color='rgba(158,202,225,0.6)'
            ),
            row=2, col=1
        )
    
    # RSI
    if show_rsi and 'rsi' in indicators:
        fig.add_trace(
            go.Scatter(
                x=historical_data['timestamp'],
                y=indicators['rsi'],
                mode='lines',
                name='RSI',
                line=dict(color='#9467bd', width=2)
            ),
            row=3, col=1
        )
        
        # RSI overbought/oversold levels
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
    
    # MACD
    if show_macd and 'macd' in indicators:
        fig.add_trace(
            go.Scatter(
                x=historical_data['timestamp'],
                y=indicators['macd'],
                mode='lines',
                name='MACD',
                line=dict(color='#1f77b4', width=1)
            ),
            row=3, col=1
        )
        
        if 'macd_signal' in indicators:
            fig.add_trace(
                go.Scatter(
                    x=historical_data['timestamp'],
                    y=indicators['macd_signal'],
                    mode='lines',
                    name='MACD Signal',
                    line=dict(color='#ff7f0e', width=1)
                ),
                row=3, col=1
            )
    
    # Update layout
    fig.update_layout(
        title=f"{coin_names[coins.index(selected_coin)]} - Technical Analysis ({selected_period})",
        height=800,
        showlegend=True,
        xaxis3_title="Date"
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Technical Analysis Summary
    st.subheader("🔍 Technical Analysis Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("#### 📊 Price Action")
        if 'sma_20' in indicators and len(indicators['sma_20']) > 0:
            current_price = historical_data['price'].iloc[-1]
            sma_20 = indicators['sma_20'].iloc[-1]
            trend = "Bullish 📈" if current_price > sma_20 else "Bearish 📉"
            st.success(f"Short-term: {trend}")
        
        if 'sma_50' in indicators and len(indicators['sma_50']) > 0:
            sma_50 = indicators['sma_50'].iloc[-1]
            trend = "Bullish 📈" if current_price > sma_50 else "Bearish 📉"
            st.info(f"Medium-term: {trend}")
    
    with col2:
        st.markdown("#### 🌊 Momentum")
        if 'rsi' in indicators and len(indicators['rsi']) > 0:
            rsi_current = indicators['rsi'].iloc[-1]
            if rsi_current > 70:
                st.warning(f"RSI: {rsi_current:.1f} (Overbought)")
            elif rsi_current < 30:
                st.error(f"RSI: {rsi_current:.1f} (Oversold)")
            else:
                st.success(f"RSI: {rsi_current:.1f} (Neutral)")
    
    with col3:
        st.markdown("#### 💹 Volume Analysis")
        if 'volume' in historical_data.columns:
            avg_volume = historical_data['volume'].rolling(20).mean().iloc[-1]
            current_volume = historical_data['volume'].iloc[-1]
            volume_ratio = current_volume / avg_volume
            
            if volume_ratio > 1.5:
                st.success(f"Volume: {volume_ratio:.1f}x above average")
            elif volume_ratio < 0.5:
                st.warning(f"Volume: {volume_ratio:.1f}x below average")
            else:
                st.info(f"Volume: {volume_ratio:.1f}x average")
    
    with col4:
        st.markdown("#### 🎯 Trading Signals")
        # Generate composite signal
        signals = []
        
        if 'rsi' in indicators and len(indicators['rsi']) > 0:
            rsi = indicators['rsi'].iloc[-1]
            if rsi < 30:
                signals.append("Buy")
            elif rsi > 70:
                signals.append("Sell")
        
        if 'macd' in indicators and 'macd_signal' in indicators:
            macd = indicators['macd'].iloc[-1]
            macd_signal = indicators['macd_signal'].iloc[-1]
            if macd > macd_signal:
                signals.append("Buy")
            else:
                signals.append("Sell")
        
        # Determine overall signal
        buy_signals = signals.count("Buy")
        sell_signals = signals.count("Sell")
        
        if buy_signals > sell_signals:
            st.success("🟢 Overall: BUY")
        elif sell_signals > buy_signals:
            st.error("🔴 Overall: SELL")
        else:
            st.warning("🟡 Overall: HOLD")
    
    # Advanced Indicators Table
    st.subheader("📋 Technical Indicators Summary")
    
    if indicators:
        # Create indicators summary
        indicator_data = []
        
        latest_idx = -1
        
        if 'sma_20' in indicators and len(indicators['sma_20']) > 0:
            indicator_data.append(['SMA 20', f"{indicators['sma_20'].iloc[latest_idx]:.4f}"])
        
        if 'sma_50' in indicators and len(indicators['sma_50']) > 0:
            indicator_data.append(['SMA 50', f"{indicators['sma_50'].iloc[latest_idx]:.4f}"])
        
        if 'rsi' in indicators and len(indicators['rsi']) > 0:
            indicator_data.append(['RSI', f"{indicators['rsi'].iloc[latest_idx]:.2f}"])
        
        if 'macd' in indicators and len(indicators['macd']) > 0:
            indicator_data.append(['MACD', f"{indicators['macd'].iloc[latest_idx]:.6f}"])
        
        if 'bb_upper' in indicators and len(indicators['bb_upper']) > 0:
            bb_upper = indicators['bb_upper'].iloc[latest_idx]
            bb_lower = indicators['bb_lower'].iloc[latest_idx]
            bb_width = ((bb_upper - bb_lower) / bb_lower) * 100
            indicator_data.append(['Bollinger Band Width', f"{bb_width:.2f}%"])
        
        if indicator_data:
            df_indicators = pd.DataFrame(indicator_data, columns=['Indicator', 'Value'])
            st.dataframe(df_indicators, use_container_width=True)
    
    # Market Comparison
    st.subheader("🏆 Market Comparison")
    
    try:
        # Fetch data for top cryptocurrencies
        comparison_coins = ['bitcoin', 'ethereum', 'binancecoin', 'cardano', 'solana']
        comparison_data = []
        
        for coin in comparison_coins:
            try:
                coin_data = data_fetcher.get_current_price(coin, 'usd')
                comparison_data.append({
                    'Cryptocurrency': coin.title(),
                    'Price': f"${coin_data['current_price']:.4f}",
                    '24h Change': f"{coin_data.get('price_change_percentage_24h', 0):.2f}%",
                    'Market Cap': f"${coin_data['market_cap']:,.0f}",
                    'Volume': f"${coin_data['total_volume']:,.0f}"
                })
            except:
                continue
        
        if comparison_data:
            df_comparison = pd.DataFrame(comparison_data)
            st.dataframe(df_comparison, use_container_width=True)
    
    except Exception as e:
        st.warning(f"Market comparison unavailable: {e}")

except Exception as e:
    st.error(f"Error loading market data: {str(e)}")
    st.info("Please try refreshing the page or selecting a different cryptocurrency.")

# Auto-refresh for real-time mode
if real_time:
    time.sleep(10)
    st.rerun()

# Footer
st.markdown("---")
st.markdown("📊 **Market Data** powered by CoinGecko API • Real-time technical analysis")
