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
from utils.performance_metrics import PerformanceMetrics
from utils.risk_manager import RiskManager
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Performance Analytics", page_icon="📊", layout="wide")

# Initialize components
@st.cache_resource
def get_components():
    return (CryptoDataFetcher(), PortfolioManager(), PerformanceMetrics(), RiskManager())

data_fetcher, portfolio_manager, performance_metrics, risk_manager = get_components()

# Initialize portfolio in session state
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = portfolio_manager

st.title("📊 Performance Analytics")
st.markdown("### Institutional-Grade Performance Attribution & Factor Analysis")

# Sidebar controls
with st.sidebar:
    st.header("📈 Analytics Settings")
    
    # Time period selection
    analysis_period = st.selectbox(
        "Analysis Period",
        ["1 Month", "3 Months", "6 Months", "1 Year", "All Time"],
        index=2
    )
    
    period_days = {
        "1 Month": 30,
        "3 Months": 90, 
        "6 Months": 180,
        "1 Year": 365,
        "All Time": 1000
    }
    
    days = period_days[analysis_period]
    
    # Benchmark selection
    benchmark = st.selectbox(
        "Benchmark",
        ["Bitcoin", "Ethereum", "S&P 500", "60/40 Portfolio"],
        index=0
    )
    
    # Performance metrics focus
    st.subheader("📊 Metrics Focus")
    show_returns = st.checkbox("Return Analysis", value=True)
    show_risk_metrics = st.checkbox("Risk Metrics", value=True)
    show_attribution = st.checkbox("Performance Attribution", value=True)
    show_factor_analysis = st.checkbox("Factor Analysis", value=True)
    
    # Risk-free rate for calculations
    risk_free_rate = st.slider("Risk-free Rate (%)", 0.0, 10.0, 2.0, step=0.1) / 100
    
    # Auto-refresh
    auto_refresh = st.checkbox("🔄 Auto-refresh (60s)")

# Get portfolio data
holdings = st.session_state.portfolio.get_holdings()
portfolio_value = st.session_state.portfolio.get_portfolio_value()

if not holdings:
    st.warning("⚠️ No portfolio holdings found. Performance analytics require an active portfolio.")
    st.stop()

# Calculate comprehensive performance metrics
with st.spinner("📊 Calculating performance metrics..."):
    try:
        # Get portfolio history
        portfolio_history = st.session_state.portfolio.get_portfolio_history(days=days)
        
        if not portfolio_history or len(portfolio_history) < 2:
            st.warning("Insufficient portfolio history for performance analysis.")
            st.stop()
        
        # Convert to DataFrame
        df_portfolio = pd.DataFrame(portfolio_history)
        df_portfolio['date'] = pd.to_datetime(df_portfolio['date'])
        df_portfolio = df_portfolio.sort_values('date')
        
        # Calculate portfolio returns
        df_portfolio['returns'] = df_portfolio['total_value'].pct_change()
        portfolio_returns = df_portfolio['returns'].dropna()
        
        # Get benchmark data
        if benchmark == "Bitcoin":
            benchmark_data = data_fetcher.get_historical_data('bitcoin', 'usd', days=days)
        elif benchmark == "Ethereum":
            benchmark_data = data_fetcher.get_historical_data('ethereum', 'usd', days=days)
        else:
            # For now, use Bitcoin as default benchmark
            benchmark_data = data_fetcher.get_historical_data('bitcoin', 'usd', days=days)
        
        if not benchmark_data.empty:
            benchmark_returns = benchmark_data['price'].pct_change().dropna()
            
            # Align dates
            min_length = min(len(portfolio_returns), len(benchmark_returns))
            portfolio_returns = portfolio_returns.iloc[-min_length:]
            benchmark_returns = benchmark_returns.iloc[-min_length:]
        else:
            benchmark_returns = None
        
        # Calculate comprehensive performance metrics
        perf_metrics = performance_metrics.calculate_comprehensive_metrics(
            portfolio_returns=portfolio_returns,
            benchmark_returns=benchmark_returns,
            risk_free_rate=risk_free_rate,
            portfolio_values=df_portfolio['total_value']
        )
        
    except Exception as e:
        st.error(f"Performance calculation error: {str(e)}")
        st.stop()

# Performance Overview Dashboard
st.subheader("🎯 Performance Overview")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    total_return = perf_metrics.get('total_return', 0) * 100
    st.metric(
        "📈 Total Return",
        f"{total_return:+.2f}%",
        f"{analysis_period}"
    )

with col2:
    annualized_return = perf_metrics.get('annualized_return', 0) * 100
    st.metric(
        "📊 Annualized Return",
        f"{annualized_return:+.2f}%",
        "Risk-adjusted"
    )

with col3:
    sharpe_ratio = perf_metrics.get('sharpe_ratio', 0)
    st.metric(
        "⚡ Sharpe Ratio",
        f"{sharpe_ratio:.2f}",
        "Risk/Return efficiency"
    )

with col4:
    max_drawdown = perf_metrics.get('max_drawdown', 0) * 100
    st.metric(
        "📉 Max Drawdown",
        f"{max_drawdown:.1f}%",
        "Worst decline"
    )

with col5:
    alpha = perf_metrics.get('alpha', 0) * 100
    st.metric(
        "🎯 Alpha",
        f"{alpha:+.2f}%",
        f"vs {benchmark}"
    )

# Performance Analysis Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Returns Analysis", 
    "⚠️ Risk Metrics", 
    "🎯 Attribution", 
    "📊 Factor Analysis", 
    "🏆 Benchmarking"
])

with tab1:
    if show_returns:
        st.markdown("#### 📈 Return Analysis")
        
        # Performance chart
        fig_performance = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Cumulative Performance vs Benchmark',
                'Rolling 30-Day Returns',
                'Monthly Returns Heatmap',
                'Return Distribution'
            ),
            specs=[[{"colspan": 2}, None],
                   [{}, {}]],
            vertical_spacing=0.1
        )
        
        # Cumulative performance
        portfolio_cumulative = (1 + portfolio_returns).cumprod()
        
        fig_performance.add_trace(
            go.Scatter(
                x=df_portfolio['date'].iloc[-len(portfolio_cumulative):],
                y=portfolio_cumulative,
                mode='lines',
                name='Portfolio',
                line=dict(color='#2E86AB', width=3)
            ),
            row=1, col=1
        )
        
        if benchmark_returns is not None:
            benchmark_cumulative = (1 + benchmark_returns).cumprod()
            fig_performance.add_trace(
                go.Scatter(
                    x=df_portfolio['date'].iloc[-len(benchmark_cumulative):],
                    y=benchmark_cumulative,
                    mode='lines',
                    name=benchmark,
                    line=dict(color='#FF6B6B', width=2, dash='dash')
                ),
                row=1, col=1
            )
        
        # Rolling returns
        rolling_returns = portfolio_returns.rolling(30).mean() * 30
        fig_performance.add_trace(
            go.Scatter(
                x=df_portfolio['date'].iloc[-len(rolling_returns):],
                y=rolling_returns * 100,
                mode='lines',
                name='30-Day Rolling',
                line=dict(color='#4ECDC4')
            ),
            row=2, col=1
        )
        
        # Return distribution
        fig_performance.add_trace(
            go.Histogram(
                x=portfolio_returns * 100,
                nbinsx=30,
                name='Daily Returns',
                opacity=0.7
            ),
            row=2, col=2
        )
        
        fig_performance.update_layout(height=700, showlegend=True)
        st.plotly_chart(fig_performance, use_container_width=True)
        
        # Return statistics table
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### 📊 Return Statistics")
            
            return_stats = {
                'Mean Daily Return': f"{portfolio_returns.mean() * 100:.3f}%",
                'Median Daily Return': f"{portfolio_returns.median() * 100:.3f}%",
                'Best Day': f"{portfolio_returns.max() * 100:+.2f}%",
                'Worst Day': f"{portfolio_returns.min() * 100:+.2f}%",
                'Positive Days': f"{(portfolio_returns > 0).sum()} ({(portfolio_returns > 0).mean() * 100:.1f}%)",
                'Volatility (Daily)': f"{portfolio_returns.std() * 100:.2f}%"
            }
            
            for stat, value in return_stats.items():
                st.write(f"**{stat}:** {value}")
        
        with col2:
            st.markdown("##### 🎯 Performance Periods")
            
            # Calculate performance over different periods
            if len(portfolio_returns) >= 7:
                weekly_return = portfolio_returns.tail(7).sum() * 100
                st.write(f"**Last 7 Days:** {weekly_return:+.2f}%")
            
            if len(portfolio_returns) >= 30:
                monthly_return = portfolio_returns.tail(30).sum() * 100
                st.write(f"**Last 30 Days:** {monthly_return:+.2f}%")
            
            if len(portfolio_returns) >= 90:
                quarterly_return = portfolio_returns.tail(90).sum() * 100
                st.write(f"**Last 90 Days:** {quarterly_return:+.2f}%")
            
            # Year-to-date calculation
            current_year = datetime.now().year
            ytd_data = df_portfolio[df_portfolio['date'].dt.year == current_year]
            if len(ytd_data) > 1:
                ytd_return = ((ytd_data['total_value'].iloc[-1] / ytd_data['total_value'].iloc[0]) - 1) * 100
                st.write(f"**Year-to-Date:** {ytd_return:+.2f}%")

with tab2:
    if show_risk_metrics:
        st.markdown("#### ⚠️ Risk Metrics Analysis")
        
        # Risk metrics overview
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("##### 📊 Volatility Metrics")
            
            daily_vol = perf_metrics.get('volatility', 0) * 100
            monthly_vol = daily_vol * np.sqrt(30)
            annual_vol = daily_vol * np.sqrt(252)
            
            st.write(f"**Daily Volatility:** {daily_vol:.2f}%")
            st.write(f"**Monthly Volatility:** {monthly_vol:.2f}%")
            st.write(f"**Annual Volatility:** {annual_vol:.2f}%")
            
            # Volatility classification
            if annual_vol < 20:
                st.success("✅ Low volatility")
            elif annual_vol < 40:
                st.warning("⚠️ Medium volatility")
            else:
                st.error("🚨 High volatility")
        
        with col2:
            st.markdown("##### 📉 Downside Risk")
            
            downside_deviation = perf_metrics.get('downside_deviation', 0) * 100
            sortino_ratio = perf_metrics.get('sortino_ratio', 0)
            var_95 = perf_metrics.get('var_95', 0) * 100
            
            st.write(f"**Downside Deviation:** {downside_deviation:.2f}%")
            st.write(f"**Sortino Ratio:** {sortino_ratio:.2f}")
            st.write(f"**VaR (95%):** {var_95:.2f}%")
            
            # Downside risk assessment
            if sortino_ratio > 1:
                st.success("✅ Good downside protection")
            elif sortino_ratio > 0.5:
                st.warning("⚠️ Moderate downside risk")
            else:
                st.error("🚨 High downside risk")
        
        with col3:
            st.markdown("##### 🎯 Risk-Adjusted Returns")
            
            information_ratio = perf_metrics.get('information_ratio', 0)
            calmar_ratio = perf_metrics.get('calmar_ratio', 0)
            
            st.write(f"**Sharpe Ratio:** {sharpe_ratio:.2f}")
            st.write(f"**Information Ratio:** {information_ratio:.2f}")
            st.write(f"**Calmar Ratio:** {calmar_ratio:.2f}")
            
            # Risk-adjusted performance assessment
            if sharpe_ratio > 1:
                st.success("✅ Excellent risk-adjusted returns")
            elif sharpe_ratio > 0.5:
                st.warning("⚠️ Moderate risk-adjusted returns")
            else:
                st.error("🚨 Poor risk-adjusted returns")
        
        # Drawdown analysis
        st.markdown("##### 📉 Drawdown Analysis")
        
        if 'drawdown_series' in perf_metrics:
            drawdowns = perf_metrics['drawdown_series']
            
            fig_dd = go.Figure()
            
            fig_dd.add_trace(go.Scatter(
                x=df_portfolio['date'].iloc[-len(drawdowns):],
                y=drawdowns * 100,
                mode='lines',
                fill='tonegative',
                name='Drawdown',
                line=dict(color='red'),
                fillcolor='rgba(255,0,0,0.3)'
            ))
            
            fig_dd.update_layout(
                title="Portfolio Drawdown Over Time",
                xaxis_title="Date",
                yaxis_title="Drawdown (%)",
                height=400
            )
            
            st.plotly_chart(fig_dd, use_container_width=True)
            
            # Drawdown statistics
            col1, col2 = st.columns(2)
            
            with col1:
                avg_drawdown = drawdowns.mean() * 100
                st.metric("📊 Average Drawdown", f"{avg_drawdown:.2f}%")
                
                # Count of significant drawdowns
                significant_dd = (drawdowns < -0.05).sum()  # More than 5% drawdown
                st.metric("📉 Significant Drawdowns", f"{significant_dd}")
            
            with col2:
                recovery_time = perf_metrics.get('avg_recovery_time', 0)
                st.metric("⏱️ Avg Recovery Time", f"{recovery_time:.0f} days")
                
                # Current drawdown status
                current_dd = drawdowns.iloc[-1] * 100 if len(drawdowns) > 0 else 0
                st.metric("📍 Current Drawdown", f"{current_dd:.2f}%")

with tab3:
    if show_attribution:
        st.markdown("#### 🎯 Performance Attribution Analysis")
        
        # Calculate attribution by holding
        attribution_data = []
        total_portfolio_return = portfolio_returns.sum()
        
        for symbol, quantity in holdings.items():
            try:
                # Get individual asset performance
                asset_data = data_fetcher.get_historical_data(symbol, 'usd', days=days)
                if not asset_data.empty:
                    asset_returns = asset_data['price'].pct_change().dropna()
                    
                    # Calculate weight and contribution
                    current_price = data_fetcher.get_current_price(symbol, 'usd')['current_price']
                    position_value = quantity * current_price
                    weight = position_value / portfolio_value
                    
                    # Align returns with portfolio
                    min_length = min(len(asset_returns), len(portfolio_returns))
                    asset_returns_aligned = asset_returns.iloc[-min_length:]
                    
                    # Calculate contribution to portfolio return
                    asset_total_return = asset_returns_aligned.sum()
                    contribution = weight * asset_total_return
                    
                    attribution_data.append({
                        'Asset': symbol.upper(),
                        'Weight': weight * 100,
                        'Asset Return': asset_total_return * 100,
                        'Contribution': contribution * 100,
                        'Active Return': (asset_total_return - total_portfolio_return) * 100
                    })
            
            except Exception as e:
                st.warning(f"Attribution calculation failed for {symbol}: {e}")
        
        if attribution_data:
            df_attribution = pd.DataFrame(attribution_data)
            
            # Attribution visualization
            fig_attr = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Return Contribution by Asset', 'Asset Weights vs Returns'),
                specs=[[{"type": "bar"}, {"type": "scatter"}]]
            )
            
            # Contribution chart
            fig_attr.add_trace(
                go.Bar(
                    x=df_attribution['Asset'],
                    y=df_attribution['Contribution'],
                    name='Contribution',
                    marker_color=['green' if x > 0 else 'red' for x in df_attribution['Contribution']]
                ),
                row=1, col=1
            )
            
            # Weight vs return scatter
            fig_attr.add_trace(
                go.Scatter(
                    x=df_attribution['Weight'],
                    y=df_attribution['Asset Return'],
                    mode='markers+text',
                    text=df_attribution['Asset'],
                    textposition="middle right",
                    marker=dict(
                        size=df_attribution['Weight'],
                        color=df_attribution['Contribution'],
                        colorscale='RdYlGn',
                        showscale=True
                    ),
                    name='Assets'
                ),
                row=1, col=2
            )
            
            fig_attr.update_layout(height=500, showlegend=False)
            st.plotly_chart(fig_attr, use_container_width=True)
            
            # Attribution table
            st.markdown("##### 📊 Detailed Attribution")
            
            # Format the dataframe for display
            display_df = df_attribution.copy()
            display_df['Weight'] = display_df['Weight'].round(1).astype(str) + '%'
            display_df['Asset Return'] = display_df['Asset Return'].round(2).astype(str) + '%'
            display_df['Contribution'] = display_df['Contribution'].round(2).astype(str) + '%'
            display_df['Active Return'] = display_df['Active Return'].round(2).astype(str) + '%'
            
            st.dataframe(display_df, use_container_width=True)

with tab4:
    if show_factor_analysis:
        st.markdown("#### 📊 Factor Analysis")
        
        # Perform factor analysis if benchmark data available
        if benchmark_returns is not None:
            # Calculate beta and correlation
            correlation = np.corrcoef(portfolio_returns, benchmark_returns)[0, 1]
            beta = perf_metrics.get('beta', 0)
            r_squared = correlation ** 2
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("##### 📈 Market Sensitivity")
                
                st.metric("📊 Beta", f"{beta:.2f}", "Market sensitivity")
                st.metric("🔗 Correlation", f"{correlation:.2f}", f"with {benchmark}")
                st.metric("📈 R-squared", f"{r_squared:.2f}", "Explained variance")
                
                # Beta interpretation
                if beta > 1.2:
                    st.warning("⚠️ High market sensitivity")
                elif beta < 0.8:
                    st.info("ℹ️ Low market sensitivity")
                else:
                    st.success("✅ Moderate market sensitivity")
            
            with col2:
                st.markdown("##### 🎯 Factor Decomposition")
                
                # Decompose returns into systematic and idiosyncratic
                systematic_return = beta * benchmark_returns.sum()
                total_return = portfolio_returns.sum()
                idiosyncratic_return = total_return - systematic_return
                
                st.write(f"**Total Return:** {total_return * 100:.2f}%")
                st.write(f"**Systematic (Beta):** {systematic_return * 100:.2f}%")
                st.write(f"**Alpha (Idiosyncratic):** {idiosyncratic_return * 100:.2f}%")
                
                # Factor contribution pie chart
                fig_factors = px.pie(
                    values=[abs(systematic_return), abs(idiosyncratic_return)],
                    names=['Market Factor', 'Alpha'],
                    title="Return Attribution: Market vs Alpha"
                )
                fig_factors.update_layout(height=300)
                st.plotly_chart(fig_factors, use_container_width=True)
            
            # Rolling factor analysis
            st.markdown("##### 📈 Rolling Factor Analysis")
            
            if len(portfolio_returns) >= 60:
                window = 60  # 60-day rolling window
                rolling_beta = []
                rolling_alpha = []
                rolling_corr = []
                
                for i in range(window, len(portfolio_returns)):
                    port_window = portfolio_returns.iloc[i-window:i]
                    bench_window = benchmark_returns.iloc[i-window:i]
                    
                    # Calculate rolling metrics
                    covariance = np.cov(port_window, bench_window)[0, 1]
                    variance = np.var(bench_window)
                    beta_rolling = covariance / variance if variance != 0 else 0
                    
                    corr_rolling = np.corrcoef(port_window, bench_window)[0, 1]
                    alpha_rolling = port_window.mean() - beta_rolling * bench_window.mean()
                    
                    rolling_beta.append(beta_rolling)
                    rolling_alpha.append(alpha_rolling * 252)  # Annualized
                    rolling_corr.append(corr_rolling)
                
                # Plot rolling metrics
                dates_rolling = df_portfolio['date'].iloc[-len(rolling_beta):]
                
                fig_rolling = make_subplots(
                    rows=3, cols=1,
                    subplot_titles=('Rolling Beta (60-day)', 'Rolling Alpha (Annualized)', 'Rolling Correlation'),
                    vertical_spacing=0.1
                )
                
                fig_rolling.add_trace(
                    go.Scatter(x=dates_rolling, y=rolling_beta, name='Beta'),
                    row=1, col=1
                )
                
                fig_rolling.add_trace(
                    go.Scatter(x=dates_rolling, y=rolling_alpha, name='Alpha'),
                    row=2, col=1
                )
                
                fig_rolling.add_trace(
                    go.Scatter(x=dates_rolling, y=rolling_corr, name='Correlation'),
                    row=3, col=1
                )
                
                fig_rolling.update_layout(height=600, showlegend=False)
                st.plotly_chart(fig_rolling, use_container_width=True)

with tab5:
    st.markdown("#### 🏆 Benchmark Comparison")
    
    if benchmark_returns is not None:
        # Comprehensive benchmark comparison
        portfolio_total_return = (portfolio_returns.sum()) * 100
        benchmark_total_return = (benchmark_returns.sum()) * 100
        excess_return = portfolio_total_return - benchmark_total_return
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "📈 Portfolio Return",
                f"{portfolio_total_return:+.2f}%",
                f"{analysis_period}"
            )
        
        with col2:
            st.metric(
                f"📊 {benchmark} Return",
                f"{benchmark_total_return:+.2f}%",
                f"{analysis_period}"
            )
        
        with col3:
            st.metric(
                "🎯 Excess Return",
                f"{excess_return:+.2f}%",
                "Outperformance" if excess_return > 0 else "Underperformance"
            )
        
        # Performance comparison chart
        fig_comparison = go.Figure()
        
        portfolio_cumulative = (1 + portfolio_returns).cumprod()
        benchmark_cumulative = (1 + benchmark_returns).cumprod()
        
        fig_comparison.add_trace(go.Scatter(
            x=df_portfolio['date'].iloc[-len(portfolio_cumulative):],
            y=portfolio_cumulative,
            mode='lines',
            name='Portfolio',
            line=dict(color='#2E86AB', width=3)
        ))
        
        fig_comparison.add_trace(go.Scatter(
            x=df_portfolio['date'].iloc[-len(benchmark_cumulative):],
            y=benchmark_cumulative,
            mode='lines',
            name=benchmark,
            line=dict(color='#FF6B6B', width=2)
        ))
        
        # Add excess return
        excess_cumulative = portfolio_cumulative / benchmark_cumulative
        fig_comparison.add_trace(go.Scatter(
            x=df_portfolio['date'].iloc[-len(excess_cumulative):],
            y=excess_cumulative,
            mode='lines',
            name='Relative Performance',
            line=dict(color='#4ECDC4', width=2, dash='dot'),
            yaxis='y2'
        ))
        
        fig_comparison.update_layout(
            title=f"Portfolio vs {benchmark} Performance",
            xaxis_title="Date",
            yaxis_title="Cumulative Return",
            yaxis2=dict(
                title="Relative Performance",
                overlaying='y',
                side='right'
            ),
            height=500
        )
        
        st.plotly_chart(fig_comparison, use_container_width=True)
        
        # Risk-return comparison
        st.markdown("##### 📊 Risk-Return Profile")
        
        portfolio_vol = portfolio_returns.std() * np.sqrt(252) * 100
        benchmark_vol = benchmark_returns.std() * np.sqrt(252) * 100
        
        fig_risk_return = go.Figure()
        
        fig_risk_return.add_trace(go.Scatter(
            x=[portfolio_vol],
            y=[portfolio_total_return],
            mode='markers+text',
            text=['Portfolio'],
            textposition="top center",
            marker=dict(size=20, color='blue'),
            name='Portfolio'
        ))
        
        fig_risk_return.add_trace(go.Scatter(
            x=[benchmark_vol],
            y=[benchmark_total_return],
            mode='markers+text',
            text=[benchmark],
            textposition="top center",
            marker=dict(size=20, color='red'),
            name=benchmark
        ))
        
        fig_risk_return.update_layout(
            title="Risk-Return Profile",
            xaxis_title="Volatility (% Annual)",
            yaxis_title=f"Return ({analysis_period})",
            height=400
        )
        
        st.plotly_chart(fig_risk_return, use_container_width=True)

# Performance summary and insights
st.subheader("💡 Performance Insights & Recommendations")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 📈 Key Insights")
    
    insights = []
    
    # Return insights
    if total_return > 0:
        insights.append(f"✅ Positive return of {total_return:.1f}% over {analysis_period}")
    else:
        insights.append(f"❌ Negative return of {total_return:.1f}% over {analysis_period}")
    
    # Risk-adjusted performance
    if sharpe_ratio > 1:
        insights.append("✅ Excellent risk-adjusted performance (Sharpe > 1)")
    elif sharpe_ratio > 0.5:
        insights.append("⚠️ Moderate risk-adjusted performance")
    else:
        insights.append("❌ Poor risk-adjusted performance")
    
    # Volatility assessment
    annual_vol = perf_metrics.get('volatility', 0) * np.sqrt(252) * 100
    if annual_vol < 20:
        insights.append("✅ Low volatility portfolio")
    elif annual_vol > 50:
        insights.append("⚠️ High volatility - consider risk management")
    
    # Benchmark comparison
    if benchmark_returns is not None and excess_return > 0:
        insights.append(f"✅ Outperformed {benchmark} by {excess_return:.1f}%")
    elif benchmark_returns is not None:
        insights.append(f"❌ Underperformed {benchmark} by {abs(excess_return):.1f}%")
    
    for insight in insights:
        st.write(insight)

with col2:
    st.markdown("#### 💡 Recommendations")
    
    recommendations = []
    
    # Based on Sharpe ratio
    if sharpe_ratio < 0.5:
        recommendations.append("🔧 Consider rebalancing to improve risk-adjusted returns")
    
    # Based on drawdown
    if max_drawdown > 20:
        recommendations.append("🛡️ Implement stronger risk management (stop-losses)")
    
    # Based on volatility
    if annual_vol > 40:
        recommendations.append("📊 Consider diversification to reduce portfolio volatility")
    
    # Based on alpha
    if alpha < 0:
        recommendations.append("🎯 Review strategy - negative alpha vs benchmark")
    
    # General recommendations
    recommendations.append("📈 Continue monitoring performance metrics regularly")
    recommendations.append("🔄 Consider periodic rebalancing")
    
    for rec in recommendations:
        st.write(rec)

# Auto-refresh functionality
if auto_refresh:
    time.sleep(60)
    st.rerun()

# Footer
st.markdown("---")
st.markdown("📊 **Performance Analytics** • Institutional-grade portfolio analysis • Educational purposes only")
