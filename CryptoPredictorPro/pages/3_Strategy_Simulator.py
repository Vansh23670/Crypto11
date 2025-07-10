import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from utils.data_fetcher import CryptoDataFetcher
from utils.strategy_simulator import StrategySimulator
from utils.technical_indicators import TechnicalIndicators

st.set_page_config(page_title="Strategy Simulator", page_icon="🎯", layout="wide")

# Initialize components
@st.cache_resource
def get_data_fetcher():
    return CryptoDataFetcher()

@st.cache_resource
def get_strategy_simulator():
    return StrategySimulator()

@st.cache_resource
def get_technical_indicators():
    return TechnicalIndicators()

data_fetcher = get_data_fetcher()
simulator = get_strategy_simulator()
ti = get_technical_indicators()

st.title("🎯 Trading Strategy Simulator & Backtesting")

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Strategy Settings")
    
    # Coin selection
    coins = ['bitcoin', 'ethereum', 'dogecoin', 'cardano', 'polygon', 'solana']
    coin_names = ['Bitcoin (BTC)', 'Ethereum (ETH)', 'Dogecoin (DOGE)', 
                  'Cardano (ADA)', 'Polygon (MATIC)', 'Solana (SOL)']
    
    selected_coin = st.selectbox(
        "Select Cryptocurrency",
        coins,
        format_func=lambda x: coin_names[coins.index(x)]
    )
    
    # Strategy selection
    strategy_name = st.selectbox(
        "Select Strategy",
        ['SMA Crossover', 'RSI Oversold/Overbought', 'MACD Signal', 'Bollinger Bands', 'Buy and Hold']
    )
    
    # Backtest period
    backtest_periods = {
        '30 days': 30,
        '90 days': 90,
        '180 days': 180,
        '365 days': 365
    }
    
    backtest_period = st.selectbox("Backtest Period", list(backtest_periods.keys()), index=2)
    backtest_days = backtest_periods[backtest_period]
    
    # Initial capital
    initial_capital = st.number_input("Initial Capital ($)", min_value=1000, max_value=100000, value=10000, step=1000)
    
    # Strategy-specific parameters
    st.subheader("📊 Strategy Parameters")
    
    if strategy_name == 'SMA Crossover':
        short_window = st.slider("Short SMA Period", 5, 50, 10)
        long_window = st.slider("Long SMA Period", 10, 100, 20)
        strategy_params = {'short_window': short_window, 'long_window': long_window}
    elif strategy_name == 'RSI Oversold/Overbought':
        rsi_oversold = st.slider("RSI Oversold Threshold", 10, 40, 30)
        rsi_overbought = st.slider("RSI Overbought Threshold", 60, 90, 70)
        strategy_params = {'rsi_oversold': rsi_oversold, 'rsi_overbought': rsi_overbought}
    else:
        strategy_params = {}
    
    # Currency
    currency = st.selectbox("Currency", ['usd', 'inr'])
    
    # Run simulation button
    run_simulation = st.button("🚀 Run Simulation", type="primary")

# Main content
if run_simulation:
    with st.spinner("Fetching data and running simulation..."):
        try:
            # Fetch historical data
            df = data_fetcher.get_historical_data(selected_coin, currency, backtest_days)
            
            if df.empty:
                st.error("No historical data available for the selected cryptocurrency.")
                st.stop()
            
            # Calculate technical indicators
            df = ti.calculate_all_indicators(df)
            
            # Run strategy simulation
            result = simulator.simulate_strategy(
                df, 
                strategy_name, 
                initial_capital, 
                **strategy_params
            )
            
            # Display results
            st.subheader("📊 Simulation Results")
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                final_value = result['portfolio_value'][-1] if result['portfolio_value'] else initial_capital
                st.metric(
                    "Final Portfolio Value",
                    f"${final_value:.2f}",
                    f"{result['returns']:.2f}%"
                )
            
            with col2:
                st.metric(
                    "Total Return",
                    f"{result['returns']:.2f}%"
                )
            
            with col3:
                st.metric(
                    "Total Trades",
                    f"{result['total_trades']}"
                )
            
            with col4:
                st.metric(
                    "Win Rate",
                    f"{result['win_rate']:.1f}%"
                )
            
            # Performance metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Max Drawdown",
                    f"{result['max_drawdown']:.2f}%"
                )
            
            with col2:
                st.metric(
                    "Sharpe Ratio",
                    f"{result['sharpe_ratio']:.2f}"
                )
            
            with col3:
                profit_factor = (result['returns'] / abs(result['max_drawdown'])) if result['max_drawdown'] != 0 else 0
                st.metric(
                    "Profit Factor",
                    f"{profit_factor:.2f}"
                )
            
            # Portfolio performance chart
            st.subheader("📈 Portfolio Performance")
            
            fig = go.Figure()
            
            # Portfolio value over time
            portfolio_dates = df['timestamp'][:len(result['portfolio_value'])]
            fig.add_trace(go.Scatter(
                x=portfolio_dates,
                y=result['portfolio_value'],
                mode='lines',
                name='Portfolio Value',
                line=dict(color='green', width=2)
            ))
            
            # Buy and hold comparison
            buy_hold_values = []
            initial_shares = initial_capital / df['price'].iloc[0]
            for price in df['price']:
                buy_hold_values.append(initial_shares * price)
            
            fig.add_trace(go.Scatter(
                x=df['timestamp'],
                y=buy_hold_values,
                mode='lines',
                name='Buy & Hold',
                line=dict(color='blue', width=2, dash='dash')
            ))
            
            fig.update_layout(
                title=f"{strategy_name} vs Buy & Hold Performance",
                xaxis_title="Date",
                yaxis_title="Portfolio Value ($)",
                height=400,
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Price chart with trading signals
            st.subheader("🎯 Price Chart with Trading Signals")
            
            fig2 = go.Figure()
            
            # Price line
            fig2.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df['price'],
                mode='lines',
                name='Price',
                line=dict(color='black', width=2)
            ))
            
            # Buy signals
            buy_signals = [trade for trade in result['trades'] if trade['type'] == 'BUY']
            if buy_signals:
                buy_dates = [trade['date'] for trade in buy_signals]
                buy_prices = [trade['price'] for trade in buy_signals]
                
                fig2.add_trace(go.Scatter(
                    x=buy_dates,
                    y=buy_prices,
                    mode='markers',
                    name='Buy Signals',
                    marker=dict(color='green', size=10, symbol='triangle-up')
                ))
            
            # Sell signals
            sell_signals = [trade for trade in result['trades'] if trade['type'] == 'SELL']
            if sell_signals:
                sell_dates = [trade['date'] for trade in sell_signals]
                sell_prices = [trade['price'] for trade in sell_signals]
                
                fig2.add_trace(go.Scatter(
                    x=sell_dates,
                    y=sell_prices,
                    mode='markers',
                    name='Sell Signals',
                    marker=dict(color='red', size=10, symbol='triangle-down')
                ))
            
            # Add moving averages if SMA strategy
            if strategy_name == 'SMA Crossover':
                if f'SMA_{strategy_params["short_window"]}' in df.columns:
                    fig2.add_trace(go.Scatter(
                        x=df['timestamp'],
                        y=df[f'SMA_{strategy_params["short_window"]}'],
                        mode='lines',
                        name=f'SMA {strategy_params["short_window"]}',
                        line=dict(color='orange', width=1)
                    ))
                
                if f'SMA_{strategy_params["long_window"]}' in df.columns:
                    fig2.add_trace(go.Scatter(
                        x=df['timestamp'],
                        y=df[f'SMA_{strategy_params["long_window"]}'],
                        mode='lines',
                        name=f'SMA {strategy_params["long_window"]}',
                        line=dict(color='purple', width=1)
                    ))
            
            fig2.update_layout(
                title=f"{coin_names[coins.index(selected_coin)]} - {strategy_name} Signals",
                xaxis_title="Date",
                yaxis_title="Price ($)",
                height=400,
                showlegend=True
            )
            
            st.plotly_chart(fig2, use_container_width=True)
            
            # Trading log
            st.subheader("📋 Trading Log")
            
            if result['trades']:
                trades_df = pd.DataFrame(result['trades'])
                trades_df['date'] = trades_df['date'].dt.strftime('%Y-%m-%d')
                trades_df['price'] = trades_df['price'].round(2)
                trades_df['shares'] = trades_df['shares'].round(6)
                trades_df['value'] = trades_df['value'].round(2)
                
                st.dataframe(
                    trades_df.rename(columns={
                        'date': 'Date',
                        'type': 'Type',
                        'price': 'Price ($)',
                        'shares': 'Shares',
                        'value': 'Value ($)'
                    }),
                    use_container_width=True
                )
            else:
                st.info("No trades executed during the simulation period.")
            
            # Strategy analysis
            st.subheader("🔍 Strategy Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Strategy Performance:**")
                
                if result['returns'] > 0:
                    st.success(f"✅ **Profitable Strategy**: {result['returns']:.2f}% return")
                else:
                    st.error(f"❌ **Loss-making Strategy**: {result['returns']:.2f}% return")
                
                if result['win_rate'] > 60:
                    st.success(f"✅ **Good Win Rate**: {result['win_rate']:.1f}%")
                elif result['win_rate'] > 40:
                    st.warning(f"⚠️ **Average Win Rate**: {result['win_rate']:.1f}%")
                else:
                    st.error(f"❌ **Poor Win Rate**: {result['win_rate']:.1f}%")
                
                if result['max_drawdown'] < 10:
                    st.success(f"✅ **Low Risk**: {result['max_drawdown']:.2f}% max drawdown")
                elif result['max_drawdown'] < 25:
                    st.warning(f"⚠️ **Medium Risk**: {result['max_drawdown']:.2f}% max drawdown")
                else:
                    st.error(f"❌ **High Risk**: {result['max_drawdown']:.2f}% max drawdown")
            
            with col2:
                st.markdown("**Recommendations:**")
                
                recommendations = []
                
                if result['returns'] > 10:
                    recommendations.append("🎯 Strong strategy performance")
                elif result['returns'] > 0:
                    recommendations.append("📈 Positive but modest returns")
                else:
                    recommendations.append("🔄 Consider strategy optimization")
                
                if result['total_trades'] < 5:
                    recommendations.append("📊 Limited trading activity - consider different parameters")
                elif result['total_trades'] > 50:
                    recommendations.append("⚡ High trading frequency - watch for fees")
                
                if result['sharpe_ratio'] > 1:
                    recommendations.append("✅ Good risk-adjusted returns")
                elif result['sharpe_ratio'] < 0:
                    recommendations.append("❌ Poor risk-adjusted returns")
                
                for rec in recommendations:
                    st.write(f"• {rec}")
            
        except Exception as e:
            st.error(f"Error running simulation: {str(e)}")
            st.info("Please try with different parameters or cryptocurrency.")

else:
    # Strategy comparison
    st.subheader("📊 Strategy Comparison")
    
    compare_strategies = st.button("Compare All Strategies")
    
    if compare_strategies:
        with st.spinner("Comparing strategies..."):
            try:
                # Fetch data for comparison
                df = data_fetcher.get_historical_data(selected_coin, 'usd', 180)  # 6 months
                
                if not df.empty:
                    df = ti.calculate_all_indicators(df)
                    
                    # Get comparison results
                    comparison = simulator.get_strategy_comparison(df, 10000)
                    
                    if comparison:
                        # Create comparison DataFrame
                        comp_df = pd.DataFrame(comparison).T
                        comp_df = comp_df.round(2)
                        
                        # Display comparison table
                        st.dataframe(
                            comp_df.rename(columns={
                                'returns': 'Return (%)',
                                'total_trades': 'Total Trades',
                                'win_rate': 'Win Rate (%)',
                                'max_drawdown': 'Max Drawdown (%)',
                                'sharpe_ratio': 'Sharpe Ratio',
                                'final_value': 'Final Value ($)'
                            }),
                            use_container_width=True
                        )
                        
                        # Best strategy
                        best_strategy = max(comparison.items(), key=lambda x: x[1]['returns'])
                        st.success(f"🏆 **Best Performing Strategy**: {best_strategy[0]} with {best_strategy[1]['returns']:.2f}% return")
                    
            except Exception as e:
                st.error(f"Error comparing strategies: {str(e)}")
    
    # Strategy explanations
    st.subheader("📚 Strategy Explanations")
    
    strategy_explanations = {
        'SMA Crossover': """
        **Simple Moving Average Crossover Strategy**
        
        • Buys when short-term SMA crosses above long-term SMA
        • Sells when short-term SMA crosses below long-term SMA
        • Good for trending markets
        • Parameters: Short window (5-20), Long window (20-50)
        """,
        
        'RSI Oversold/Overbought': """
        **RSI Overbought/Oversold Strategy**
        
        • Buys when RSI falls below oversold threshold (typically 30)
        • Sells when RSI rises above overbought threshold (typically 70)
        • Good for range-bound markets
        • Parameters: Oversold (20-40), Overbought (60-80)
        """,
        
        'MACD Signal': """
        **MACD Signal Strategy**
        
        • Buys when MACD line crosses above signal line
        • Sells when MACD line crosses below signal line
        • Good for identifying trend changes
        • Uses standard MACD parameters (12, 26, 9)
        """,
        
        'Bollinger Bands': """
        **Bollinger Bands Strategy**
        
        • Buys when price touches lower Bollinger Band
        • Sells when price touches upper Bollinger Band
        • Good for volatile markets
        • Uses standard parameters (20 period, 2 standard deviations)
        """,
        
        'Buy and Hold': """
        **Buy and Hold Strategy**
        
        • Buys at the beginning and holds until the end
        • Benchmark strategy for comparison
        • No active trading required
        • Good for long-term bull markets
        """
    }
    
    selected_strategy_info = st.selectbox("Select Strategy to Learn About", list(strategy_explanations.keys()))
    
    st.markdown(strategy_explanations[selected_strategy_info])

# Footer
st.markdown("---")
st.markdown("""
⚠️ **Important Notes:**
- Backtesting results don't guarantee future performance
- Real trading involves fees, slippage, and market impact
- Past performance is not indicative of future results
- Always test strategies with small amounts first
- Consider risk management and position sizing
""")
