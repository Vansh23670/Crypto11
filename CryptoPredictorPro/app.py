import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from utils.data_fetcher import CryptoDataFetcher
from utils.portfolio_manager import PortfolioManager
from utils.db_status import show_database_status
import time

# Page configuration
st.set_page_config(
    page_title="Crypto Trading Prediction App",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = PortfolioManager()

# Initialize data fetcher
@st.cache_resource
def get_data_fetcher():
    return CryptoDataFetcher()

data_fetcher = get_data_fetcher()

# Title and description
st.title("🚀 Crypto Trading Prediction & Strategy App")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("📊 Dashboard")
    
    # Database status
    show_database_status()
    
    # Coin selection
    coins = ['bitcoin', 'ethereum', 'dogecoin', 'cardano', 'polygon', 'solana']
    coin_names = ['Bitcoin (BTC)', 'Ethereum (ETH)', 'Dogecoin (DOGE)', 
                  'Cardano (ADA)', 'Polygon (MATIC)', 'Solana (SOL)']
    
    selected_coin = st.selectbox(
        "Select Cryptocurrency",
        coins,
        format_func=lambda x: coin_names[coins.index(x)]
    )
    
    # Currency selection
    currency = st.selectbox("Currency", ['usd', 'inr'])
    
    # Refresh button
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()

# Main dashboard
col1, col2, col3, col4 = st.columns(4)

# Fetch current price data
try:
    current_data = data_fetcher.get_current_price(selected_coin, currency)
    
    with col1:
        st.metric(
            label="Current Price",
            value=f"${current_data['current_price']:.2f}" if currency == 'usd' else f"₹{current_data['current_price']:.2f}",
            delta=f"{current_data['price_change_percentage_24h']:.2f}%"
        )
    
    with col2:
        st.metric(
            label="Market Cap",
            value=f"${current_data['market_cap']:,.0f}" if currency == 'usd' else f"₹{current_data['market_cap']:,.0f}"
        )
    
    with col3:
        st.metric(
            label="24h Volume",
            value=f"${current_data['total_volume']:,.0f}" if currency == 'usd' else f"₹{current_data['total_volume']:,.0f}"
        )
    
    with col4:
        portfolio_value = st.session_state.portfolio.get_portfolio_value()
        st.metric(
            label="Portfolio Value",
            value=f"${portfolio_value:.2f}" if currency == 'usd' else f"₹{portfolio_value:.2f}",
            delta=f"{st.session_state.portfolio.get_total_pnl():.2f}%"
        )

except Exception as e:
    error_msg = str(e)
    if "429" in error_msg or "Too Many Requests" in error_msg:
        st.error("⚠️ **API Rate Limit Exceeded**")
        st.info("The API has temporary rate limits. Please wait 1-2 minutes and click the Refresh Data button.")
    else:
        st.error(f"Unable to fetch market data: {error_msg}")
        st.info("Please check your internet connection and try refreshing.")

# Quick chart
st.subheader("📈 Price Chart (Last 7 Days)")

try:
    # Fetch historical data
    historical_data = data_fetcher.get_historical_data(selected_coin, currency, days=7)
    
    if not historical_data.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=historical_data['timestamp'],
            y=historical_data['price'],
            mode='lines',
            name='Price',
            line=dict(color='#FF6B6B', width=2)
        ))
        
        fig.update_layout(
            title=f"{coin_names[coins.index(selected_coin)]} Price Trend",
            xaxis_title="Date",
            yaxis_title=f"Price ({'USD' if currency == 'usd' else 'INR'})",
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No historical data available for the selected cryptocurrency.")

except Exception as e:
    st.error(f"Error fetching historical data: {str(e)}")

# Recent activity
st.subheader("📋 Recent Portfolio Activity")
recent_trades = st.session_state.portfolio.get_recent_trades()

if recent_trades:
    df = pd.DataFrame(recent_trades)
    st.dataframe(df, use_container_width=True)
else:
    st.info("No recent trading activity. Visit the Portfolio Tracker to start trading!")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>📊 Built with Streamlit • 🔄 Real-time data from CoinGecko API</p>
        <p>⚠️ This is for educational purposes only. Not financial advice.</p>
    </div>
""", unsafe_allow_html=True)

# Auto-refresh every 30 seconds
if st.sidebar.checkbox("Auto-refresh (30s)", value=False):
    time.sleep(30)
    st.rerun()
