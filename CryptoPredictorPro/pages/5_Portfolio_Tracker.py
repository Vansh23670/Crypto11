import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import time
from utils.data_fetcher import CryptoDataFetcher
from utils.portfolio_manager import PortfolioManager
from utils.risk_manager import RiskManager
from utils.performance_metrics import PerformanceMetrics
import json

st.set_page_config(page_title="Portfolio Tracker", page_icon="💼", layout="wide")

# Initialize components
@st.cache_resource
def get_components():
    return (CryptoDataFetcher(), PortfolioManager(), RiskManager(), PerformanceMetrics())

data_fetcher, portfolio_manager, risk_manager, performance_metrics = get_components()

# Initialize portfolio in session state
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = portfolio_manager

st.title("💼 Portfolio Tracker & Management")
st.markdown("### Advanced Portfolio Analytics with Real-time P&L and Risk Assessment")

# Sidebar controls
with st.sidebar:
    st.header("💰 Portfolio Controls")
    
    # Portfolio overview
    total_value = st.session_state.portfolio.get_portfolio_value()
    total_pnl = st.session_state.portfolio.get_total_pnl()
    
    st.metric(
        "💼 Total Portfolio Value",
        f"${total_value:,.2f}",
        f"{total_pnl:+.2f}%"
    )
    
    # Quick trading section
    st.subheader("⚡ Quick Trade")
    
    # Coin selection for trading
    coins = ['bitcoin', 'ethereum', 'binancecoin', 'cardano', 'solana', 'polkadot', 
             'dogecoin', 'polygon', 'chainlink', 'litecoin', 'avalanche', 'uniswap']
    coin_names = ['Bitcoin (BTC)', 'Ethereum (ETH)', 'BNB (BNB)', 'Cardano (ADA)',
                  'Solana (SOL)', 'Polkadot (DOT)', 'Dogecoin (DOGE)', 'Polygon (MATIC)',
                  'Chainlink (LINK)', 'Litecoin (LTC)', 'Avalanche (AVAX)', 'Uniswap (UNI)']
    
    selected_coin = st.selectbox(
        "🪙 Select Coin",
        coins,
        format_func=lambda x: coin_names[coins.index(x)]
    )
    
    # Get current price
    try:
        current_data = data_fetcher.get_current_price(selected_coin, 'usd')
        current_price = current_data['current_price']
        st.write(f"**Current Price:** ${current_price:.4f}")
    except Exception as e:
        st.error(f"Unable to fetch price: {e}")
        current_price = 0
    
    # Trade type selection
    trade_type = st.radio("Trade Type", ["Buy", "Sell"])
    
    if trade_type == "Buy":
        # Buy section
        st.markdown("#### 💰 Buy Order")
        buy_amount = st.number_input("Amount (USD)", min_value=1.0, value=100.0, step=10.0)
        
        if current_price > 0:
            quantity = buy_amount / current_price
            st.write(f"**Quantity:** {quantity:.6f} {selected_coin.upper()}")
        
        if st.button("🟢 Execute Buy", type="primary", use_container_width=True):
            if current_price > 0:
                try:
                    st.session_state.portfolio.execute_trade(
                        symbol=selected_coin,
                        trade_type='buy',
                        amount=buy_amount,
                        price=current_price
                    )
                    st.success(f"✅ Successfully bought {quantity:.6f} {selected_coin.upper()} for ${buy_amount}")
                    st.rerun()
                except Exception as e:
                    st.error(f"Trade failed: {e}")
            else:
                st.error("Unable to execute trade - price data unavailable")
    
    else:
        # Sell section
        st.markdown("#### 💸 Sell Order")
        holdings = st.session_state.portfolio.get_holdings()
        coin_balance = holdings.get(selected_coin, 0)
        
        if coin_balance > 0:
            sell_percentage = st.slider("Sell Percentage", 0, 100, 25)
            sell_quantity = coin_balance * (sell_percentage / 100)
            sell_value = sell_quantity * current_price if current_price > 0 else 0
            
            st.write(f"**Sell Quantity:** {sell_quantity:.6f} {selected_coin.upper()}")
            st.write(f"**Estimated Value:** ${sell_value:.2f}")
            
            if st.button("🔴 Execute Sell", type="secondary", use_container_width=True):
                if current_price > 0:
                    try:
                        st.session_state.portfolio.execute_trade(
                            symbol=selected_coin,
                            trade_type='sell',
                            quantity=sell_quantity,
                            price=current_price
                        )
                        st.success(f"✅ Successfully sold {sell_quantity:.6f} {selected_coin.upper()} for ${sell_value:.2f}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Trade failed: {e}")
                else:
                    st.error("Unable to execute trade - price data unavailable")
        else:
            st.info(f"No {selected_coin.upper()} holdings to sell")
    
    # Portfolio actions
    st.subheader("📊 Portfolio Actions")
    
    if st.button("🔄 Refresh Portfolio", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    
    if st.button("📥 Export Portfolio", use_container_width=True):
        # This would export portfolio data
        st.info("Portfolio export functionality would be implemented here")
    
    # Auto-refresh toggle
    auto_refresh = st.checkbox("⚡ Auto-refresh (30s)")

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    # Portfolio overview metrics
    st.subheader("📊 Portfolio Overview")
    
    # Get portfolio data
    holdings = st.session_state.portfolio.get_holdings()
    cash_balance = st.session_state.portfolio.get_cash_balance()
    total_invested = st.session_state.portfolio.get_total_invested()
    
    col1_1, col1_2, col1_3, col1_4 = st.columns(4)
    
    with col1_1:
        st.metric(
            "💰 Cash Balance",
            f"${cash_balance:,.2f}",
            f"{(cash_balance / total_value * 100):.1f}% of portfolio" if total_value > 0 else "0%"
        )
    
    with col1_2:
        invested_value = total_value - cash_balance
        st.metric(
            "📈 Invested Value",
            f"${invested_value:,.2f}",
            f"{(invested_value / total_value * 100):.1f}% of portfolio" if total_value > 0 else "0%"
        )
    
    with col1_3:
        st.metric(
            "💵 Total Invested",
            f"${total_invested:,.2f}",
            "Historical cost basis"
        )
    
    with col1_4:
        unrealized_pnl = invested_value - total_invested if total_invested > 0 else 0
        unrealized_pnl_pct = (unrealized_pnl / total_invested * 100) if total_invested > 0 else 0
        st.metric(
            "📊 Unrealized P&L",
            f"${unrealized_pnl:+,.2f}",
            f"{unrealized_pnl_pct:+.2f}%"
        )
    
    # Holdings breakdown
    st.subheader("🪙 Current Holdings")
    
    if holdings:
        holdings_data = []
        total_holdings_value = 0
        
        for symbol, quantity in holdings.items():
            if quantity > 0:
                try:
                    price_data = data_fetcher.get_current_price(symbol, 'usd')
                    current_price = price_data['current_price']
                    current_value = quantity * current_price
                    total_holdings_value += current_value
                    
                    # Get cost basis
                    avg_cost = st.session_state.portfolio.get_average_cost(symbol)
                    cost_basis = quantity * avg_cost if avg_cost > 0 else 0
                    
                    # Calculate P&L
                    pnl = current_value - cost_basis if cost_basis > 0 else 0
                    pnl_pct = (pnl / cost_basis * 100) if cost_basis > 0 else 0
                    
                    holdings_data.append({
                        'Symbol': symbol.upper(),
                        'Quantity': f"{quantity:.6f}",
                        'Current Price': f"${current_price:.4f}",
                        'Current Value': f"${current_value:.2f}",
                        'Avg Cost': f"${avg_cost:.4f}" if avg_cost > 0 else "N/A",
                        'Cost Basis': f"${cost_basis:.2f}" if cost_basis > 0 else "N/A",
                        'P&L': f"${pnl:+.2f}" if cost_basis > 0 else "N/A",
                        'P&L %': f"{pnl_pct:+.2f}%" if cost_basis > 0 else "N/A",
                        '24h Change': f"{price_data.get('price_change_percentage_24h', 0):+.2f}%"
                    })
                    
                except Exception as e:
                    st.warning(f"Unable to fetch data for {symbol}: {e}")
        
        if holdings_data:
            df_holdings = pd.DataFrame(holdings_data)
            st.dataframe(df_holdings, use_container_width=True)
            
            # Portfolio allocation pie chart
            st.subheader("🥧 Portfolio Allocation")
            
            allocation_data = []
            for _, row in df_holdings.iterrows():
                value = float(row['Current Value'].replace('$', '').replace(',', ''))
                allocation_data.append({
                    'Symbol': row['Symbol'],
                    'Value': value,
                    'Percentage': (value / total_holdings_value * 100) if total_holdings_value > 0 else 0
                })
            
            # Add cash if significant
            if cash_balance > total_value * 0.01:  # Show cash if >1% of portfolio
                allocation_data.append({
                    'Symbol': 'CASH',
                    'Value': cash_balance,
                    'Percentage': (cash_balance / total_value * 100) if total_value > 0 else 0
                })
            
            df_allocation = pd.DataFrame(allocation_data)
            
            fig_pie = px.pie(
                df_allocation,
                values='Value',
                names='Symbol',
                title="Portfolio Allocation by Value",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            fig_pie.update_layout(height=400)
            st.plotly_chart(fig_pie, use_container_width=True)
    
    else:
        st.info("No cryptocurrency holdings found. Start trading to build your portfolio!")
    
    # Performance chart
    st.subheader("📈 Portfolio Performance")
    
    # Get portfolio history
    portfolio_history = st.session_state.portfolio.get_portfolio_history(days=30)
    
    if portfolio_history and len(portfolio_history) > 1:
        df_history = pd.DataFrame(portfolio_history)
        
        fig_performance = go.Figure()
        
        # Portfolio value line
        fig_performance.add_trace(go.Scatter(
            x=df_history['date'],
            y=df_history['total_value'],
            mode='lines',
            name='Portfolio Value',
            line=dict(color='#2E86AB', width=3),
            hovertemplate='Date: %{x}<br>Value: $%{y:,.2f}<extra></extra>'
        ))
        
        # Add benchmark comparison if available (e.g., Bitcoin)
        try:
            btc_data = data_fetcher.get_historical_data('bitcoin', 'usd', days=30)
            if not btc_data.empty and len(df_history) == len(btc_data):
                # Normalize Bitcoin to start at same value as portfolio
                btc_normalized = btc_data['price'] / btc_data['price'].iloc[0] * df_history['total_value'].iloc[0]
                
                fig_performance.add_trace(go.Scatter(
                    x=df_history['date'],
                    y=btc_normalized,
                    mode='lines',
                    name='Bitcoin (Normalized)',
                    line=dict(color='#FF6B6B', width=2, dash='dash'),
                    hovertemplate='Date: %{x}<br>BTC Value: $%{y:,.2f}<extra></extra>'
                ))
        except:
            pass
        
        fig_performance.update_layout(
            title="Portfolio Performance vs Bitcoin",
            xaxis_title="Date",
            yaxis_title="Value (USD)",
            height=400,
            showlegend=True,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_performance, use_container_width=True)
    
    else:
        st.info("Portfolio performance chart will appear after you start trading.")

with col2:
    # Risk metrics sidebar
    st.subheader("⚠️ Risk Metrics")
    
    if holdings:
        # Calculate portfolio risk metrics
        risk_metrics = risk_manager.calculate_portfolio_risk(holdings, data_fetcher)
        
        st.metric(
            "📊 Portfolio Beta",
            f"{risk_metrics.get('beta', 0):.2f}",
            "vs Market"
        )
        
        st.metric(
            "📈 Volatility",
            f"{risk_metrics.get('volatility', 0) * 100:.1f}%",
            "Annualized"
        )
        
        st.metric(
            "📉 Value at Risk",
            f"${risk_metrics.get('var_95', 0):,.2f}",
            "95% confidence (1-day)"
        )
        
        st.metric(
            "🎯 Sharpe Ratio",
            f"{risk_metrics.get('sharpe_ratio', 0):.2f}",
            "Risk-adjusted return"
        )
        
        # Risk distribution chart
        if 'risk_breakdown' in risk_metrics:
            fig_risk = px.bar(
                x=list(risk_metrics['risk_breakdown'].keys()),
                y=list(risk_metrics['risk_breakdown'].values()),
                title="Risk Contribution by Asset",
                labels={'x': 'Asset', 'y': 'Risk Contribution (%)'}
            )
            fig_risk.update_layout(height=300)
            st.plotly_chart(fig_risk, use_container_width=True)
    
    else:
        st.info("Risk metrics will appear after building your portfolio.")
    
    # Recent transactions
    st.subheader("📋 Recent Transactions")
    
    recent_trades = st.session_state.portfolio.get_recent_trades(limit=10)
    
    if recent_trades:
        for trade in recent_trades:
            trade_type = trade.get('type', 'unknown')
            symbol = trade.get('symbol', '').upper()
            amount = trade.get('amount', 0)
            price = trade.get('price', 0)
            timestamp = trade.get('timestamp', '')
            
            # Color code by trade type
            color = "#4CAF50" if trade_type == 'buy' else "#f44336"
            
            st.markdown(f"""
            <div style="border-left: 4px solid {color}; padding: 10px; margin: 5px 0; background: rgba(128,128,128,0.1);">
                <strong>{trade_type.upper()}</strong> {symbol}<br>
                Amount: ${amount:.2f}<br>
                Price: ${price:.4f}<br>
                <small>{timestamp}</small>
            </div>
            """, unsafe_allow_html=True)
    
    else:
        st.info("No recent transactions found.")
    
    # Portfolio insights
    st.subheader("💡 Portfolio Insights")
    
    insights = []
    
    if total_value > 0:
        cash_percentage = (cash_balance / total_value) * 100
        
        if cash_percentage > 50:
            insights.append("💰 High cash allocation - consider investing more")
        elif cash_percentage < 5:
            insights.append("⚠️ Low cash reserves - consider taking some profits")
        
        if len(holdings) == 1:
            insights.append("🎯 Single asset portfolio - consider diversification")
        elif len(holdings) > 10:
            insights.append("📊 Highly diversified - monitor correlation")
        
        # Calculate concentration risk
        if holdings_data:
            max_allocation = max([float(row['Current Value'].replace('$', '').replace(',', '')) 
                                for row in holdings_data]) / total_holdings_value * 100
            if max_allocation > 50:
                insights.append("⚠️ High concentration risk in single asset")
        
        if total_pnl > 20:
            insights.append("🎉 Strong performance - consider rebalancing")
        elif total_pnl < -20:
            insights.append("📉 Significant losses - review risk management")
    
    if not insights:
        insights.append("✅ Portfolio looks balanced")
    
    for insight in insights:
        st.write(insight)

# Transaction log section
st.subheader("📊 Complete Transaction History")

all_trades = st.session_state.portfolio.get_all_trades()

if all_trades:
    df_trades = pd.DataFrame(all_trades)
    
    # Add filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        trade_filter = st.selectbox("Filter by Type", ["All", "Buy", "Sell"])
    
    with col2:
        if 'symbol' in df_trades.columns:
            symbol_filter = st.selectbox("Filter by Symbol", ["All"] + list(df_trades['symbol'].unique()))
        else:
            symbol_filter = "All"
    
    with col3:
        days_filter = st.selectbox("Filter by Period", ["All Time", "Last 7 days", "Last 30 days", "Last 90 days"])
    
    # Apply filters
    filtered_df = df_trades.copy()
    
    if trade_filter != "All":
        filtered_df = filtered_df[filtered_df['type'] == trade_filter.lower()]
    
    if symbol_filter != "All":
        filtered_df = filtered_df[filtered_df['symbol'] == symbol_filter.lower()]
    
    if days_filter != "All Time":
        days_map = {"Last 7 days": 7, "Last 30 days": 30, "Last 90 days": 90}
        cutoff_date = datetime.now() - timedelta(days=days_map[days_filter])
        filtered_df = filtered_df[pd.to_datetime(filtered_df['timestamp']) >= cutoff_date]
    
    # Display filtered results
    if not filtered_df.empty:
        # Format the dataframe for display
        display_df = filtered_df.copy()
        if 'symbol' in display_df.columns:
            display_df['symbol'] = display_df['symbol'].str.upper()
        if 'type' in display_df.columns:
            display_df['type'] = display_df['type'].str.upper()
        
        st.dataframe(display_df, use_container_width=True)
        
        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_trades = len(filtered_df)
            st.metric("📊 Total Trades", total_trades)
        
        with col2:
            buy_trades = len(filtered_df[filtered_df['type'] == 'buy']) if 'type' in filtered_df.columns else 0
            st.metric("🟢 Buy Orders", buy_trades)
        
        with col3:
            sell_trades = len(filtered_df[filtered_df['type'] == 'sell']) if 'type' in filtered_df.columns else 0
            st.metric("🔴 Sell Orders", sell_trades)
        
        with col4:
            total_volume = filtered_df['amount'].sum() if 'amount' in filtered_df.columns else 0
            st.metric("💹 Total Volume", f"${total_volume:,.2f}")
    
    else:
        st.info("No transactions match the selected filters.")

else:
    st.info("No transaction history available. Start trading to see your history here!")

# Auto-refresh functionality
if auto_refresh:
    time.sleep(30)
    st.rerun()

# Footer
st.markdown("---")
st.markdown("💼 **Portfolio Tracker** • Real-time portfolio management and analytics • Educational purposes only")
