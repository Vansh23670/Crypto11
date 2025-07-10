import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from utils.data_fetcher import CryptoDataFetcher
from utils.technical_indicators import TechnicalIndicators

st.set_page_config(page_title="Market Data", page_icon="📊", layout="wide")

# Initialize data fetcher and technical indicators
@st.cache_resource
def get_data_fetcher():
    return CryptoDataFetcher()

@st.cache_resource
def get_technical_indicators():
    return TechnicalIndicators()

data_fetcher = get_data_fetcher()
ti = get_technical_indicators()

st.title("📊 Live Market Data & Technical Analysis")

# Sidebar controls
with st.sidebar:
    st.header("📈 Chart Settings")
    
    # Coin selection
    coins = ['bitcoin', 'ethereum', 'dogecoin', 'cardano', 'polygon', 'solana', 'chainlink', 'avalanche-2']
    coin_names = ['Bitcoin (BTC)', 'Ethereum (ETH)', 'Dogecoin (DOGE)', 'Cardano (ADA)', 
                  'Polygon (MATIC)', 'Solana (SOL)', 'Chainlink (LINK)', 'Avalanche (AVAX)']
    
    selected_coin = st.selectbox(
        "Select Cryptocurrency",
        coins,
        format_func=lambda x: coin_names[coins.index(x)]
    )
    
    # Time period
    time_periods = {
        '7 days': 7,
        '30 days': 30,
        '90 days': 90,
        '180 days': 180,
        '365 days': 365
    }
    
    selected_period = st.selectbox("Time Period", list(time_periods.keys()))
    days = time_periods[selected_period]
    
    # Currency
    currency = st.selectbox("Currency", ['usd', 'inr'])
    
    # Chart type
    chart_type = st.selectbox("Chart Type", ['Line', 'Candlestick'])
    
    # Technical indicators
    st.subheader("Technical Indicators")
    show_sma = st.checkbox("Simple Moving Average", value=True)
    if show_sma:
        sma_period = st.slider("SMA Period", 5, 50, 20)
    
    show_ema = st.checkbox("Exponential Moving Average")
    if show_ema:
        ema_period = st.slider("EMA Period", 5, 50, 12)
    
    show_bollinger = st.checkbox("Bollinger Bands")
    show_rsi = st.checkbox("RSI", value=True)
    show_macd = st.checkbox("MACD")
    
    # Refresh button
    st.divider()
    if st.button("🔄 Refresh Data", help="Click to fetch fresh market data"):
        st.cache_data.clear()
        st.rerun()

# Main content
col1, col2, col3, col4 = st.columns(4)

# Fetch current data with enhanced error handling
try:
    with st.spinner("Fetching live market data..."):
        current_data = data_fetcher.get_current_price(selected_coin, currency)
        coin_info = data_fetcher.get_coin_info(selected_coin)
    
    with col1:
        st.metric(
            "Current Price",
            f"${current_data['current_price']:.2f}" if currency == 'usd' else f"₹{current_data['current_price']:.2f}",
            f"{current_data['price_change_percentage_24h']:.2f}%"
        )
    
    with col2:
        st.metric(
            "Market Cap",
            f"${current_data['market_cap']:,.0f}" if currency == 'usd' else f"₹{current_data['market_cap']:,.0f}"
        )
    
    with col3:
        st.metric(
            "24h Volume",
            f"${current_data['total_volume']:,.0f}" if currency == 'usd' else f"₹{current_data['total_volume']:,.0f}"
        )
    
    with col4:
        st.metric(
            "Market Cap Rank",
            f"#{coin_info.get('market_cap_rank', 'N/A')}"
        )

except Exception as e:
    error_msg = str(e)
    if "429" in error_msg or "Too Many Requests" in error_msg:
        st.error("⚠️ **API Rate Limit Exceeded**")
        st.info("""
        The CoinGecko API has temporary rate limits. This happens when too many requests are made in a short time.
        
        **What you can do:**
        - Wait 1-2 minutes and refresh the page
        - Try selecting a different cryptocurrency
        - The app will automatically retry with built-in delays
        """)
    else:
        st.error(f"Unable to fetch market data: {error_msg}")
        st.info("Please check your internet connection and try again in a moment.")
    st.stop()

# Fetch historical data
try:
    with st.spinner("Loading historical data..."):
        if chart_type == 'Candlestick':
            df = data_fetcher.get_ohlc_data(selected_coin, currency, days)
        else:
            df = data_fetcher.get_historical_data(selected_coin, currency, days)
    
    if df.empty:
        st.error("No historical data available for the selected cryptocurrency.")
        st.stop()
    
    # Calculate technical indicators
    if chart_type == 'Line':
        df = ti.calculate_all_indicators(df)
    
    # Create main chart
    if chart_type == 'Candlestick' and all(col in df.columns for col in ['open', 'high', 'low', 'close']):
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=('Price', 'RSI', 'MACD'),
            row_width=[0.2, 0.1, 0.1]
        )
        
        # Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=df['timestamp'],
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name='Price'
            ),
            row=1, col=1
        )
        
    else:
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=('Price', 'RSI', 'MACD'),
            row_width=[0.2, 0.1, 0.1]
        )
        
        # Line chart
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['price'],
                mode='lines',
                name='Price',
                line=dict(color='#FF6B6B', width=2)
            ),
            row=1, col=1
        )
        
        # Add moving averages
        if show_sma and f'SMA_{sma_period}' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df[f'SMA_{sma_period}'],
                    mode='lines',
                    name=f'SMA {sma_period}',
                    line=dict(color='orange', width=1)
                ),
                row=1, col=1
            )
        
        if show_ema and f'EMA_{ema_period}' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df[f'EMA_{ema_period}'],
                    mode='lines',
                    name=f'EMA {ema_period}',
                    line=dict(color='purple', width=1)
                ),
                row=1, col=1
            )
        
        # Add Bollinger Bands
        if show_bollinger and all(col in df.columns for col in ['BB_upper', 'BB_middle', 'BB_lower']):
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['BB_upper'],
                    mode='lines',
                    name='BB Upper',
                    line=dict(color='gray', width=1, dash='dash'),
                    showlegend=False
                ),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['BB_lower'],
                    mode='lines',
                    name='BB Lower',
                    line=dict(color='gray', width=1, dash='dash'),
                    fill='tonexty',
                    fillcolor='rgba(128,128,128,0.1)',
                    showlegend=False
                ),
                row=1, col=1
            )
    
    # Add RSI
    if show_rsi and 'RSI' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['RSI'],
                mode='lines',
                name='RSI',
                line=dict(color='blue', width=1)
            ),
            row=2, col=1
        )
        
        # Add RSI overbought/oversold lines
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    
    # Add MACD
    if show_macd and all(col in df.columns for col in ['MACD', 'MACD_signal', 'MACD_histogram']):
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['MACD'],
                mode='lines',
                name='MACD',
                line=dict(color='blue', width=1)
            ),
            row=3, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['MACD_signal'],
                mode='lines',
                name='MACD Signal',
                line=dict(color='red', width=1)
            ),
            row=3, col=1
        )
        
        fig.add_trace(
            go.Bar(
                x=df['timestamp'],
                y=df['MACD_histogram'],
                name='MACD Histogram',
                marker_color='gray',
                opacity=0.6
            ),
            row=3, col=1
        )
    
    # Update layout
    fig.update_layout(
        title=f"{coin_names[coins.index(selected_coin)]} - {selected_period} Chart",
        xaxis_title="Date",
        height=800,
        showlegend=True,
        xaxis_rangeslider_visible=False
    )
    
    # Update y-axis titles
    fig.update_yaxes(title_text="Price (USD)" if currency == 'usd' else "Price (INR)", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1, range=[0, 100])
    fig.update_yaxes(title_text="MACD", row=3, col=1)
    
    st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"Error creating chart: {str(e)}")

# Trading signals and market sentiment
if chart_type == 'Line' and not df.empty:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Trading Signals")
        
        try:
            signals = ti.get_trading_signals(df)
            
            if signals:
                for signal in signals[:5]:  # Show top 5 signals
                    signal_color = "🟢" if signal['type'] == 'BUY' else "🔴"
                    st.write(f"{signal_color} **{signal['type']}** - {signal['indicator']}")
                    st.write(f"   {signal['reason']} ({signal['strength']})")
                    st.write("")
            else:
                st.info("No strong trading signals detected at the moment.")
                
        except Exception as e:
            st.error(f"Error generating signals: {str(e)}")
    
    with col2:
        st.subheader("📊 Market Sentiment")
        
        try:
            sentiment, sentiment_score = ti.get_market_sentiment(df)
            
            # Sentiment gauge
            sentiment_color = "green" if sentiment_score > 60 else "red" if sentiment_score < 40 else "orange"
            
            st.markdown(f"""
            <div style="text-align: center; padding: 20px; background-color: {sentiment_color}20; border-radius: 10px;">
                <h3 style="color: {sentiment_color}; margin: 0;">{sentiment}</h3>
                <p style="font-size: 24px; margin: 10px 0;">{sentiment_score:.1f}/100</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Key insights
            st.write("**Key Insights:**")
            insights = ti.get_key_insights() if hasattr(ti, 'get_key_insights') else []
            for insight in insights[:3]:
                st.write(f"• {insight}")
                
        except Exception as e:
            st.error(f"Error calculating sentiment: {str(e)}")

# Additional information
st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    st.subheader("ℹ️ Coin Information")
    if coin_info:
        st.write(f"**Name:** {coin_info.get('name', 'N/A')}")
        st.write(f"**Symbol:** {coin_info.get('symbol', 'N/A')}")
        st.write(f"**Market Cap Rank:** #{coin_info.get('market_cap_rank', 'N/A')}")
        st.write(f"**All-Time High:** ${coin_info.get('ath', 0):.2f}")
        st.write(f"**All-Time Low:** ${coin_info.get('atl', 0):.6f}")

with col2:
    st.subheader("📈 Quick Stats")
    if not df.empty:
        st.write(f"**Data Points:** {len(df)}")
        st.write(f"**Price Range:** ${df['price'].min():.2f} - ${df['price'].max():.2f}")
        st.write(f"**Average Price:** ${df['price'].mean():.2f}")
        st.write(f"**Volatility:** {df['price'].std():.2f}")

# Footer
st.markdown("---")
st.markdown("💡 **Tip:** Use the sidebar to customize your chart and indicators. Technical analysis should be combined with fundamental analysis for better trading decisions.")
