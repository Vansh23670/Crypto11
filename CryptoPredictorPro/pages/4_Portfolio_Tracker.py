import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from utils.data_fetcher import CryptoDataFetcher
from utils.portfolio_manager import PortfolioManager

st.set_page_config(page_title="Portfolio Tracker", page_icon="💼", layout="wide")

# Initialize components
@st.cache_resource
def get_data_fetcher():
    return CryptoDataFetcher()

data_fetcher = get_data_fetcher()

# Get portfolio manager from session state
portfolio = st.session_state.portfolio if 'portfolio' in st.session_state else PortfolioManager()

st.title("💼 Portfolio Tracker & Paper Trading")

# Sidebar for trading
with st.sidebar:
    st.header("💰 Paper Trading")
    
    # Display current balance
    current_balance = portfolio.get_balance()
    st.metric("Cash Balance", f"${current_balance:.2f}")
    
    # Trading form
    st.subheader("📊 Execute Trade")
    
    # Trade type
    trade_type = st.radio("Trade Type", ["Buy", "Sell"])
    
    # Coin selection
    coins = ['bitcoin', 'ethereum', 'dogecoin', 'cardano', 'polygon', 'solana']
    coin_names = ['Bitcoin (BTC)', 'Ethereum (ETH)', 'Dogecoin (DOGE)', 
                  'Cardano (ADA)', 'Polygon (MATIC)', 'Solana (SOL)']
    
    selected_coin = st.selectbox(
        "Select Cryptocurrency",
        coins,
        format_func=lambda x: coin_names[coins.index(x)]
    )
    
    # Get current price
    try:
        current_price_data = data_fetcher.get_current_price(selected_coin, 'usd')
        current_price = current_price_data['current_price']
        
        st.info(f"Current Price: ${current_price:.2f}")
        
        # Trade amount
        if trade_type == "Buy":
            max_amount = current_balance / current_price
            amount = st.number_input(
                f"Amount to Buy (max: {max_amount:.6f})",
                min_value=0.000001,
                max_value=max_amount,
                value=min(1.0, max_amount),
                step=0.001,
                format="%.6f"
            )
            
            total_cost = amount * current_price
            st.write(f"Total Cost: ${total_cost:.2f}")
            
        else:  # Sell
            holdings = portfolio.get_holdings()
            if selected_coin in holdings:
                max_amount = holdings[selected_coin]['quantity']
                amount = st.number_input(
                    f"Amount to Sell (max: {max_amount:.6f})",
                    min_value=0.000001,
                    max_value=max_amount,
                    value=min(1.0, max_amount),
                    step=0.001,
                    format="%.6f"
                )
                
                total_value = amount * current_price
                st.write(f"Total Value: ${total_value:.2f}")
            else:
                st.warning("You don't own this cryptocurrency")
                amount = 0
        
        # Execute trade button
        if st.button(f"Execute {trade_type}", type="primary"):
            if trade_type == "Buy":
                success, message = portfolio.buy_crypto(
                    selected_coin,
                    coin_names[coins.index(selected_coin)],
                    amount,
                    current_price
                )
            else:
                success, message = portfolio.sell_crypto(
                    selected_coin,
                    coin_names[coins.index(selected_coin)],
                    amount,
                    current_price
                )
            
            if success:
                st.success(message)
                st.rerun()
            else:
                st.error(message)
        
    except Exception as e:
        st.error(f"Error fetching price: {str(e)}")
    
    # Portfolio actions
    st.subheader("⚙️ Portfolio Actions")
    
    if st.button("🔄 Refresh Prices"):
        st.cache_data.clear()
        st.rerun()
    
    if st.button("🗑️ Reset Portfolio"):
        portfolio.reset_portfolio()
        st.success("Portfolio reset!")
        st.rerun()

# Main content
# Portfolio overview
st.subheader("📊 Portfolio Overview")

# Get current prices for all holdings
holdings = portfolio.get_holdings()
current_prices = {}

if holdings:
    try:
        for coin_id in holdings.keys():
            price_data = data_fetcher.get_current_price(coin_id, 'usd')
            current_prices[coin_id] = price_data['current_price']
    except Exception as e:
        st.warning(f"Error fetching current prices: {str(e)}")

# Portfolio metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    portfolio_value = portfolio.get_portfolio_value(current_prices)
    st.metric("Total Portfolio Value", f"${portfolio_value:.2f}")

with col2:
    total_pnl = portfolio.get_total_pnl(current_prices)
    st.metric("Total P&L", f"{total_pnl:.2f}%")

with col3:
    cash_balance = portfolio.get_balance()
    st.metric("Cash Balance", f"${cash_balance:.2f}")

with col4:
    holdings_value = portfolio_value - cash_balance
    st.metric("Holdings Value", f"${holdings_value:.2f}")

# Portfolio breakdown
if holdings:
    st.subheader("📋 Portfolio Breakdown")
    
    breakdown = portfolio.get_portfolio_breakdown(current_prices)
    
    if breakdown:
        # Create portfolio pie chart
        fig = px.pie(
            values=[item['value'] for item in breakdown],
            names=[item['asset'] for item in breakdown],
            title="Portfolio Allocation"
        )
        
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(height=400)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Holdings table
            holdings_df = pd.DataFrame(breakdown)
            holdings_df = holdings_df[holdings_df['asset'] != 'Cash']  # Exclude cash from holdings table
            
            if not holdings_df.empty:
                st.dataframe(
                    holdings_df[['asset', 'quantity', 'avg_price', 'current_price', 'value', 'pnl']].rename(columns={
                        'asset': 'Asset',
                        'quantity': 'Quantity',
                        'avg_price': 'Avg Price ($)',
                        'current_price': 'Current Price ($)',
                        'value': 'Value ($)',
                        'pnl': 'P&L (%)'
                    }).round(2),
                    use_container_width=True
                )
            else:
                st.info("No cryptocurrency holdings yet. Start trading to build your portfolio!")
    
    # Performance chart
    st.subheader("📈 Portfolio Performance")
    
    # Create a simple performance visualization
    try:
        # Get historical data for portfolio holdings
        portfolio_history = []
        
        # For demonstration, we'll show the performance of the largest holding
        if holdings:
            largest_holding = max(holdings.items(), key=lambda x: x[1]['quantity'] * current_prices.get(x[0], 0))
            coin_id = largest_holding[0]
            
            # Get 30-day historical data
            historical_data = data_fetcher.get_historical_data(coin_id, 'usd', 30)
            
            if not historical_data.empty:
                # Calculate portfolio value over time (simplified)
                quantity = largest_holding[1]['quantity']
                portfolio_values = historical_data['price'] * quantity + cash_balance
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=historical_data['timestamp'],
                    y=portfolio_values,
                    mode='lines',
                    name='Portfolio Value',
                    line=dict(color='green', width=2)
                ))
                
                fig.update_layout(
                    title="Portfolio Value Over Time (Last 30 Days)",
                    xaxis_title="Date",
                    yaxis_title="Portfolio Value ($)",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    except Exception as e:
        st.info("Portfolio performance chart will be available after you build your portfolio.")

else:
    st.info("Your portfolio is empty. Start trading to build your portfolio!")

# Trading history
st.subheader("📋 Trading History")

recent_trades = portfolio.get_recent_trades(10)

if recent_trades:
    trades_df = pd.DataFrame(recent_trades)
    st.dataframe(trades_df, use_container_width=True)
else:
    st.info("No trading history yet. Execute your first trade to see it here!")

# Performance metrics
st.subheader("📊 Performance Metrics")

performance_metrics = portfolio.get_performance_metrics(current_prices)

if performance_metrics['total_trades'] > 0:
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Trades", f"{performance_metrics['total_trades']}")
    
    with col2:
        st.metric("Win Rate", f"{performance_metrics['win_rate']:.1f}%")
    
    with col3:
        st.metric("Avg Gain", f"{performance_metrics['avg_gain']:.2f}%")
    
    with col4:
        st.metric("Avg Loss", f"{performance_metrics['avg_loss']:.2f}%")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Winning Trades", f"{performance_metrics['winning_trades']}")
    
    with col2:
        st.metric("Losing Trades", f"{performance_metrics['losing_trades']}")
    
    # Performance analysis
    st.subheader("🔍 Performance Analysis")
    
    if performance_metrics['win_rate'] > 60:
        st.success("✅ **Excellent Win Rate**: You're making good trading decisions!")
    elif performance_metrics['win_rate'] > 40:
        st.warning("⚠️ **Average Win Rate**: Consider improving your strategy.")
    else:
        st.error("❌ **Poor Win Rate**: Review your trading strategy.")
    
    if performance_metrics['profit_factor'] > 1.5:
        st.success("✅ **Good Profit Factor**: Your wins are significantly larger than losses.")
    elif performance_metrics['profit_factor'] > 1:
        st.info("📊 **Positive Profit Factor**: You're profitable overall.")
    else:
        st.error("❌ **Poor Profit Factor**: Losses are outweighing gains.")

else:
    st.info("Performance metrics will be available after you execute some trades.")

# Market insights
st.subheader("🔍 Market Insights")

try:
    # Show current market data for popular coins
    market_data = []
    for coin_id in coins[:4]:  # Top 4 coins
        try:
            price_data = data_fetcher.get_current_price(coin_id, 'usd')
            market_data.append({
                'Coin': coin_names[coins.index(coin_id)],
                'Price': f"${price_data['current_price']:.2f}",
                '24h Change': f"{price_data['price_change_percentage_24h']:.2f}%",
                'Market Cap': f"${price_data['market_cap']:,.0f}",
                'Volume': f"${price_data['total_volume']:,.0f}"
            })
        except:
            continue
    
    if market_data:
        market_df = pd.DataFrame(market_data)
        st.dataframe(market_df, use_container_width=True)

except Exception as e:
    st.warning("Market insights temporarily unavailable.")

# Trading tips
st.subheader("💡 Trading Tips")

tips = [
    "🎯 **Set Clear Goals**: Define your risk tolerance and profit targets",
    "📊 **Use Technical Analysis**: Combine multiple indicators for better decisions",
    "💰 **Risk Management**: Never risk more than 2-5% of your portfolio on a single trade",
    "🕐 **Be Patient**: Wait for good setups rather than forcing trades",
    "📈 **Track Performance**: Keep a trading journal to learn from your trades",
    "🔍 **Stay Informed**: Follow market news and trends",
    "🎲 **Diversify**: Don't put all your money in one cryptocurrency",
    "🧠 **Control Emotions**: Stick to your strategy and avoid FOMO/fear"
]

# Display tips in columns
tip_cols = st.columns(2)
for i, tip in enumerate(tips):
    with tip_cols[i % 2]:
        st.markdown(tip)

# Footer
st.markdown("---")
st.markdown("""
⚠️ **Paper Trading Disclaimer**: This is a simulated trading environment using virtual money. 
Real trading involves additional risks including market volatility, slippage, fees, and emotional factors. 
Use this tool to practice and test strategies before risking real capital.
""")
