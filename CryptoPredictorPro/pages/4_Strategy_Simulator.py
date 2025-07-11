import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import time
from utils.data_fetcher import CryptoDataFetcher
from utils.trading_strategies import TradingStrategies
from utils.technical_indicators import TechnicalIndicators
from utils.performance_metrics import PerformanceMetrics
import json

st.set_page_config(page_title="Strategy Simulator", page_icon="🎮", layout="wide")

# Initialize components
@st.cache_resource
def get_components():
    return (CryptoDataFetcher(), TradingStrategies(), TechnicalIndicators(), PerformanceMetrics())

data_fetcher, trading_strategies, technical_indicators, performance_metrics = get_components()

st.title("🎮 Advanced Strategy Simulator")
st.markdown("### Backtest Trading Strategies with High-Frequency Analysis & Risk Modeling")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Simulation Settings")
    
    # Coin selection
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
    
    # Backtesting period
    backtest_days = st.slider("📅 Backtesting Period (days)", 30, 365, 90)
    
    # Initial capital
    initial_capital = st.number_input("💰 Initial Capital ($)", 1000, 100000, 10000, step=1000)
    
    # Strategy selection
    st.subheader("📊 Trading Strategy")
    strategy_type = st.selectbox(
        "Strategy Type",
        [
            "SMA Crossover",
            "RSI Mean Reversion", 
            "MACD Momentum",
            "Bollinger Bands",
            "Momentum Trading",
            "Mean Reversion",
            "Breakout Strategy",
            "Scalping Strategy",
            "Arbitrage Scanner",
            "Buy & Hold"
        ]
    )
    
    # Strategy parameters
    st.subheader("🔧 Strategy Parameters")
    
    if strategy_type == "SMA Crossover":
        short_window = st.slider("Short MA Period", 5, 50, 20)
        long_window = st.slider("Long MA Period", 20, 200, 50)
        strategy_params = {'short_window': short_window, 'long_window': long_window}
    
    elif strategy_type == "RSI Mean Reversion":
        rsi_period = st.slider("RSI Period", 10, 30, 14)
        oversold = st.slider("Oversold Level", 20, 40, 30)
        overbought = st.slider("Overbought Level", 60, 80, 70)
        strategy_params = {'rsi_period': rsi_period, 'oversold': oversold, 'overbought': overbought}
    
    elif strategy_type == "MACD Momentum":
        fast_period = st.slider("Fast EMA", 8, 20, 12)
        slow_period = st.slider("Slow EMA", 20, 35, 26)
        signal_period = st.slider("Signal Period", 5, 15, 9)
        strategy_params = {'fast': fast_period, 'slow': slow_period, 'signal': signal_period}
    
    elif strategy_type == "Bollinger Bands":
        bb_period = st.slider("BB Period", 10, 30, 20)
        bb_std = st.slider("BB Standard Deviations", 1.5, 3.0, 2.0, step=0.1)
        strategy_params = {'period': bb_period, 'std_dev': bb_std}
    
    elif strategy_type == "Momentum Trading":
        momentum_period = st.slider("Momentum Period", 5, 30, 10)
        momentum_threshold = st.slider("Momentum Threshold (%)", 0.5, 5.0, 2.0, step=0.1)
        strategy_params = {'period': momentum_period, 'threshold': momentum_threshold / 100}
    
    elif strategy_type == "Scalping Strategy":
        scalp_threshold = st.slider("Price Move Threshold (%)", 0.1, 2.0, 0.5, step=0.1)
        hold_time = st.slider("Max Hold Time (hours)", 1, 24, 4)
        strategy_params = {'threshold': scalp_threshold / 100, 'hold_time': hold_time}
    
    else:
        strategy_params = {}
    
    # Risk management
    st.subheader("⚠️ Risk Management")
    use_stop_loss = st.checkbox("Use Stop Loss", value=True)
    stop_loss_pct = st.slider("Stop Loss (%)", 1, 20, 5) / 100 if use_stop_loss else None
    
    use_take_profit = st.checkbox("Use Take Profit", value=True)
    take_profit_pct = st.slider("Take Profit (%)", 2, 50, 10) / 100 if use_take_profit else None
    
    max_position_size = st.slider("Max Position Size (%)", 10, 100, 100) / 100
    
    # Transaction costs
    st.subheader("💸 Transaction Costs")
    trading_fee = st.slider("Trading Fee (%)", 0.0, 1.0, 0.1, step=0.01) / 100
    slippage = st.slider("Slippage (%)", 0.0, 0.5, 0.05, step=0.01) / 100
    
    # Run simulation button
    run_simulation = st.button("🚀 Run Backtest", type="primary")

# Main content area
if run_simulation or 'last_simulation' in st.session_state:
    
    if run_simulation:
        with st.spinner("📊 Fetching data and running simulation..."):
            try:
                # Fetch historical data
                historical_data = data_fetcher.get_historical_data(selected_coin, 'usd', days=backtest_days)
                
                if historical_data.empty:
                    st.error("Insufficient data for backtesting.")
                    st.stop()
                
                # Calculate technical indicators
                indicators = technical_indicators.calculate_all_indicators(historical_data)
                
                # Run strategy simulation
                simulation_results = trading_strategies.backtest_strategy(
                    strategy_type=strategy_type,
                    data=historical_data,
                    indicators=indicators,
                    initial_capital=initial_capital,
                    strategy_params=strategy_params,
                    stop_loss=stop_loss_pct,
                    take_profit=take_profit_pct,
                    trading_fee=trading_fee,
                    slippage=slippage,
                    max_position_size=max_position_size
                )
                
                # Calculate performance metrics
                performance_stats = performance_metrics.calculate_all_metrics(
                    returns=simulation_results['returns'],
                    prices=simulation_results['portfolio_values'],
                    benchmark_returns=historical_data['price'].pct_change(),
                    risk_free_rate=0.02  # 2% annual risk-free rate
                )
                
                # Store results in session state
                st.session_state.last_simulation = {
                    'results': simulation_results,
                    'performance': performance_stats,
                    'historical_data': historical_data,
                    'strategy_type': strategy_type,
                    'strategy_params': strategy_params,
                    'coin': selected_coin,
                    'initial_capital': initial_capital,
                    'timestamp': datetime.now()
                }
                
            except Exception as e:
                st.error(f"Simulation failed: {str(e)}")
                st.stop()
    
    # Display results
    if 'last_simulation' in st.session_state:
        sim_data = st.session_state.last_simulation
        results = sim_data['results']
        performance = sim_data['performance']
        
        # Performance Summary
        st.subheader("📈 Performance Summary")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        final_value = results['portfolio_values'][-1]
        total_return = ((final_value - sim_data['initial_capital']) / sim_data['initial_capital']) * 100
        
        with col1:
            st.metric(
                "💰 Final Portfolio Value",
                f"${final_value:,.2f}",
                f"{total_return:+.2f}%"
            )
        
        with col2:
            st.metric(
                "📊 Total Return",
                f"{total_return:+.2f}%",
                f"${final_value - sim_data['initial_capital']:+,.2f}"
            )
        
        with col3:
            sharpe_ratio = performance.get('sharpe_ratio', 0)
            st.metric(
                "⚡ Sharpe Ratio",
                f"{sharpe_ratio:.2f}",
                "Risk-adjusted return"
            )
        
        with col4:
            max_drawdown = performance.get('max_drawdown', 0) * 100
            st.metric(
                "📉 Max Drawdown",
                f"{max_drawdown:.2f}%",
                "Worst peak-to-trough"
            )
        
        with col5:
            win_rate = performance.get('win_rate', 0) * 100
            st.metric(
                "🎯 Win Rate",
                f"{win_rate:.1f}%",
                f"{results.get('total_trades', 0)} trades"
            )
        
        # Performance Chart
        st.subheader("📊 Portfolio Performance vs Buy & Hold")
        
        # Create performance comparison chart
        fig = go.Figure()
        
        # Portfolio value line
        dates = results.get('dates', sim_data['historical_data']['timestamp'])
        fig.add_trace(go.Scatter(
            x=dates,
            y=results['portfolio_values'],
            mode='lines',
            name=f'{sim_data["strategy_type"]} Strategy',
            line=dict(color='#2E86AB', width=3)
        ))
        
        # Buy & hold benchmark
        initial_price = sim_data['historical_data']['price'].iloc[0]
        buy_hold_values = (sim_data['historical_data']['price'] / initial_price) * sim_data['initial_capital']
        
        fig.add_trace(go.Scatter(
            x=sim_data['historical_data']['timestamp'],
            y=buy_hold_values,
            mode='lines',
            name='Buy & Hold',
            line=dict(color='#FF6B6B', width=2, dash='dash')
        ))
        
        # Add trade markers if available
        if 'trade_dates' in results and 'trade_types' in results:
            buy_trades = [i for i, t in enumerate(results['trade_types']) if t == 'buy']
            sell_trades = [i for i, t in enumerate(results['trade_types']) if t == 'sell']
            
            if buy_trades:
                fig.add_trace(go.Scatter(
                    x=[results['trade_dates'][i] for i in buy_trades],
                    y=[results['trade_prices'][i] for i in buy_trades],
                    mode='markers',
                    name='Buy Signals',
                    marker=dict(color='green', size=10, symbol='triangle-up')
                ))
            
            if sell_trades:
                fig.add_trace(go.Scatter(
                    x=[results['trade_dates'][i] for i in sell_trades],
                    y=[results['trade_prices'][i] for i in sell_trades],
                    mode='markers',
                    name='Sell Signals',
                    marker=dict(color='red', size=10, symbol='triangle-down')
                ))
        
        fig.update_layout(
            title=f"{sim_data['strategy_type']} Strategy Performance - {coin_names[coins.index(sim_data['coin'])]}",
            xaxis_title="Date",
            yaxis_title="Portfolio Value ($)",
            height=500,
            showlegend=True,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed Analytics
        st.subheader("🔬 Detailed Performance Analytics")
        
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Statistics", "📈 Returns", "⚠️ Risk Metrics", "💹 Trade Analysis"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📈 Return Metrics")
                
                stats_data = [
                    ["Total Return", f"{total_return:+.2f}%"],
                    ["Annualized Return", f"{performance.get('annualized_return', 0) * 100:.2f}%"],
                    ["Volatility", f"{performance.get('volatility', 0) * 100:.2f}%"],
                    ["Sharpe Ratio", f"{performance.get('sharpe_ratio', 0):.3f}"],
                    ["Sortino Ratio", f"{performance.get('sortino_ratio', 0):.3f}"],
                    ["Calmar Ratio", f"{performance.get('calmar_ratio', 0):.3f}"]
                ]
                
                for stat in stats_data:
                    st.write(f"**{stat[0]}:** {stat[1]}")
            
            with col2:
                st.markdown("#### ⚖️ Risk Metrics")
                
                risk_data = [
                    ["Maximum Drawdown", f"{performance.get('max_drawdown', 0) * 100:.2f}%"],
                    ["Value at Risk (95%)", f"{performance.get('var_95', 0) * 100:.2f}%"],
                    ["Expected Shortfall", f"{performance.get('expected_shortfall', 0) * 100:.2f}%"],
                    ["Beta", f"{performance.get('beta', 0):.3f}"],
                    ["Alpha", f"{performance.get('alpha', 0) * 100:.2f}%"],
                    ["Tracking Error", f"{performance.get('tracking_error', 0) * 100:.2f}%"]
                ]
                
                for risk in risk_data:
                    st.write(f"**{risk[0]}:** {risk[1]}")
        
        with tab2:
            st.markdown("#### 📊 Returns Distribution")
            
            if 'returns' in results:
                returns_df = pd.DataFrame({
                    'Daily Returns': results['returns']
                })
                
                # Returns histogram
                fig = px.histogram(
                    returns_df, 
                    x='Daily Returns', 
                    nbins=50,
                    title="Daily Returns Distribution",
                    labels={'Daily Returns': 'Daily Return (%)', 'count': 'Frequency'}
                )
                fig.add_vline(x=np.mean(results['returns']), line_dash="dash", line_color="red", 
                             annotation_text="Mean Return")
                st.plotly_chart(fig, use_container_width=True)
                
                # Rolling performance metrics
                st.markdown("#### 📈 Rolling Performance (30-day window)")
                
                if len(results['returns']) >= 30:
                    rolling_returns = pd.Series(results['returns']).rolling(30).mean() * 30
                    rolling_vol = pd.Series(results['returns']).rolling(30).std() * np.sqrt(30)
                    rolling_sharpe = rolling_returns / rolling_vol
                    
                    fig = make_subplots(
                        rows=3, cols=1,
                        subplot_titles=('Rolling Returns', 'Rolling Volatility', 'Rolling Sharpe Ratio'),
                        vertical_spacing=0.05
                    )
                    
                    fig.add_trace(
                        go.Scatter(x=dates[-len(rolling_returns):], y=rolling_returns, name='Rolling Returns'),
                        row=1, col=1
                    )
                    
                    fig.add_trace(
                        go.Scatter(x=dates[-len(rolling_vol):], y=rolling_vol, name='Rolling Volatility'),
                        row=2, col=1
                    )
                    
                    fig.add_trace(
                        go.Scatter(x=dates[-len(rolling_sharpe):], y=rolling_sharpe, name='Rolling Sharpe'),
                        row=3, col=1
                    )
                    
                    fig.update_layout(height=600, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.markdown("#### ⚠️ Risk Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Drawdown chart
                if 'drawdowns' in performance:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=dates,
                        y=performance['drawdowns'] * 100,
                        mode='lines',
                        fill='tonegative',
                        name='Drawdown',
                        line=dict(color='red')
                    ))
                    fig.update_layout(
                        title="Portfolio Drawdowns",
                        xaxis_title="Date",
                        yaxis_title="Drawdown (%)",
                        height=300
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Risk-Return scatter
                if len(results['returns']) > 0:
                    monthly_returns = pd.Series(results['returns']).groupby(
                        pd.Series(dates).dt.to_period('M')
                    ).sum()
                    
                    fig = px.scatter(
                        x=monthly_returns.rolling(3).std(),
                        y=monthly_returns.rolling(3).mean(),
                        title="Risk-Return Profile (3-month rolling)",
                        labels={'x': 'Risk (Volatility)', 'y': 'Return'}
                    )
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
        
        with tab4:
            st.markdown("#### 💹 Trade Analysis")
            
            if results.get('total_trades', 0) > 0:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("📊 Total Trades", results.get('total_trades', 0))
                    st.metric("✅ Winning Trades", results.get('winning_trades', 0))
                    st.metric("❌ Losing Trades", results.get('losing_trades', 0))
                
                with col2:
                    avg_win = results.get('avg_win', 0) * 100
                    avg_loss = results.get('avg_loss', 0) * 100
                    st.metric("📈 Avg Win", f"+{avg_win:.2f}%")
                    st.metric("📉 Avg Loss", f"{avg_loss:.2f}%")
                    
                    if avg_loss != 0:
                        profit_factor = abs(avg_win / avg_loss) * (results.get('winning_trades', 0) / max(results.get('losing_trades', 1), 1))
                        st.metric("⚖️ Profit Factor", f"{profit_factor:.2f}")
                
                with col3:
                    largest_win = results.get('largest_win', 0) * 100
                    largest_loss = results.get('largest_loss', 0) * 100
                    st.metric("🎯 Largest Win", f"+{largest_win:.2f}%")
                    st.metric("💥 Largest Loss", f"{largest_loss:.2f}%")
                    
                    avg_hold_time = results.get('avg_hold_time', 0)
                    st.metric("⏱️ Avg Hold Time", f"{avg_hold_time:.1f} days")
                
                # Trade log
                if 'trade_log' in results:
                    st.markdown("#### 📋 Recent Trades")
                    df_trades = pd.DataFrame(results['trade_log'][-10:])  # Last 10 trades
                    if not df_trades.empty:
                        st.dataframe(df_trades, use_container_width=True)
            else:
                st.info("No trades executed during the backtesting period.")
        
        # Strategy Optimization Suggestions
        st.subheader("🎯 Strategy Optimization Suggestions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📈 Performance Insights")
            
            insights = []
            
            if total_return > 0:
                insights.append("✅ Strategy outperformed holding cash")
            else:
                insights.append("❌ Strategy underperformed holding cash")
            
            buy_hold_return = ((buy_hold_values.iloc[-1] - buy_hold_values.iloc[0]) / buy_hold_values.iloc[0]) * 100
            if total_return > buy_hold_return:
                insights.append(f"✅ Strategy outperformed buy & hold by {total_return - buy_hold_return:.2f}%")
            else:
                insights.append(f"❌ Strategy underperformed buy & hold by {buy_hold_return - total_return:.2f}%")
            
            if performance.get('sharpe_ratio', 0) > 1:
                insights.append("✅ Good risk-adjusted returns (Sharpe > 1)")
            else:
                insights.append("⚠️ Poor risk-adjusted returns (Sharpe < 1)")
            
            if performance.get('max_drawdown', 0) < 0.2:
                insights.append("✅ Acceptable maximum drawdown (<20%)")
            else:
                insights.append("⚠️ High maximum drawdown (>20%)")
            
            for insight in insights:
                st.write(insight)
        
        with col2:
            st.markdown("#### 💡 Optimization Ideas")
            
            suggestions = []
            
            if performance.get('win_rate', 0) < 0.5:
                suggestions.append("🔧 Consider tightening entry criteria")
            
            if performance.get('max_drawdown', 0) > 0.15:
                suggestions.append("🛡️ Implement stricter risk management")
            
            if results.get('total_trades', 0) < 10:
                suggestions.append("📊 Strategy may be too conservative")
            elif results.get('total_trades', 0) > 100:
                suggestions.append("⚡ Strategy may be overtrading")
            
            if performance.get('sharpe_ratio', 0) < 0.5:
                suggestions.append("📈 Consider different entry/exit rules")
            
            suggestions.append("🔍 Test different parameter combinations")
            suggestions.append("📊 Consider market regime filters")
            
            for suggestion in suggestions:
                st.write(suggestion)

else:
    # Landing page
    st.info("👆 Configure your backtesting parameters in the sidebar and click 'Run Backtest' to start.")
    
    # Strategy descriptions
    st.subheader("📚 Available Trading Strategies")
    
    strategy_descriptions = {
        "SMA Crossover": "Classic trend-following strategy using moving average crossovers for entry/exit signals.",
        "RSI Mean Reversion": "Contrarian strategy buying oversold and selling overbought conditions based on RSI.",
        "MACD Momentum": "Momentum strategy using MACD line crossovers and signal line divergences.",
        "Bollinger Bands": "Range-bound strategy trading bounces off Bollinger Band boundaries.",
        "Momentum Trading": "Trend-following strategy based on price momentum and rate of change.",
        "Mean Reversion": "Statistical arbitrage strategy exploiting price deviations from mean.",
        "Breakout Strategy": "Momentum strategy capturing moves beyond key support/resistance levels.",
        "Scalping Strategy": "High-frequency strategy capturing small price movements.",
        "Arbitrage Scanner": "Market inefficiency strategy exploiting price differences.",
        "Buy & Hold": "Passive investment strategy for benchmark comparison."
    }
    
    for strategy, description in strategy_descriptions.items():
        st.write(f"**{strategy}:** {description}")

# Footer
st.markdown("---")
st.markdown("🎮 **Strategy Simulator** • Advanced backtesting with transaction costs & slippage modeling • For educational purposes only")
