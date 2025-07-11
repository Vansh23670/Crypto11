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
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Risk Management", page_icon="⚠️", layout="wide")

# Initialize components
@st.cache_resource
def get_components():
    return (CryptoDataFetcher(), PortfolioManager(), RiskManager(), PerformanceMetrics())

data_fetcher, portfolio_manager, risk_manager, performance_metrics = get_components()

# Initialize portfolio in session state
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = portfolio_manager

st.title("⚠️ Advanced Risk Management")
st.markdown("### Comprehensive Risk Analysis with VaR, Stress Testing & Dynamic Position Sizing")

# Sidebar controls
with st.sidebar:
    st.header("🔧 Risk Settings")
    
    # Risk tolerance settings
    risk_tolerance = st.selectbox(
        "Risk Tolerance",
        ["Conservative", "Moderate", "Aggressive"],
        index=1
    )
    
    # Portfolio allocation limits
    st.subheader("📊 Position Limits")
    max_single_position = st.slider("Max Single Position (%)", 5, 50, 20)
    max_sector_exposure = st.slider("Max Crypto Exposure (%)", 50, 100, 80)
    min_cash_reserve = st.slider("Min Cash Reserve (%)", 0, 50, 10)
    
    # Risk metrics timeframe
    st.subheader("📅 Analysis Period")
    analysis_days = st.selectbox("Historical Period", [30, 60, 90, 180, 365], index=2)
    
    # VaR settings
    st.subheader("📉 Value at Risk")
    var_confidence = st.selectbox("VaR Confidence Level", ["95%", "99%", "99.9%"], index=0)
    var_horizon = st.selectbox("VaR Horizon", ["1 Day", "1 Week", "1 Month"], index=0)
    
    # Stress testing
    st.subheader("🧪 Stress Testing")
    stress_scenario = st.selectbox(
        "Stress Scenario",
        ["Market Crash (-50%)", "Bear Market (-30%)", "Correction (-20%)", "Volatility Spike", "Custom"]
    )
    
    if stress_scenario == "Custom":
        custom_shock = st.slider("Custom Shock (%)", -80, 80, -25)

# Main content
try:
    # Get portfolio data
    holdings = st.session_state.portfolio.get_holdings()
    portfolio_value = st.session_state.portfolio.get_portfolio_value()
    cash_balance = st.session_state.portfolio.get_cash_balance()
    
    if not holdings:
        st.warning("⚠️ No portfolio holdings found. Add some positions to analyze risk metrics.")
        st.stop()
    
    # Calculate comprehensive risk metrics
    with st.spinner("🔄 Calculating risk metrics..."):
        risk_analysis = risk_manager.comprehensive_risk_analysis(
            holdings=holdings,
            data_fetcher=data_fetcher,
            analysis_days=analysis_days,
            confidence_level=float(var_confidence.replace('%', '')) / 100
        )
    
    # Risk Dashboard Overview
    st.subheader("🎯 Risk Dashboard")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        overall_risk = risk_analysis.get('overall_risk_score', 0)
        risk_color = "#4CAF50" if overall_risk < 0.3 else "#FF9800" if overall_risk < 0.7 else "#F44336"
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, {risk_color} 0%, {risk_color}aa 100%); 
                    padding: 1.5rem; border-radius: 10px; text-align: center; color: white;">
            <h3>🎯 Risk Score</h3>
            <h1>{overall_risk:.1f}/10</h1>
            <p>{'Low' if overall_risk < 3 else 'Medium' if overall_risk < 7 else 'High'} Risk</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        var_1d = risk_analysis.get('var_1d', 0)
        st.metric(
            "📉 Daily VaR",
            f"${var_1d:,.2f}",
            f"{(var_1d / portfolio_value * 100):.2f}% of portfolio"
        )
    
    with col3:
        portfolio_beta = risk_analysis.get('portfolio_beta', 0)
        st.metric(
            "📊 Portfolio Beta",
            f"{portfolio_beta:.2f}",
            "vs Market"
        )
    
    with col4:
        volatility = risk_analysis.get('volatility', 0) * 100
        st.metric(
            "📈 Volatility",
            f"{volatility:.1f}%",
            "Annualized"
        )
    
    with col5:
        max_drawdown = risk_analysis.get('max_drawdown', 0) * 100
        st.metric(
            "📉 Max Drawdown",
            f"{max_drawdown:.1f}%",
            "Historical worst"
        )
    
    # Risk Breakdown Analysis
    st.subheader("🔍 Risk Breakdown Analysis")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Position Risk", "📉 VaR Analysis", "🧪 Stress Testing", "🎯 Optimization"])
    
    with tab1:
        st.markdown("#### 📈 Individual Position Risk")
        
        # Calculate individual position risks
        position_risks = []
        total_portfolio_value = portfolio_value
        
        for symbol, quantity in holdings.items():
            try:
                # Get current price and historical data
                current_data = data_fetcher.get_current_price(symbol, 'usd')
                historical_data = data_fetcher.get_historical_data(symbol, 'usd', days=analysis_days)
                
                if not historical_data.empty:
                    current_price = current_data['current_price']
                    position_value = quantity * current_price
                    allocation = (position_value / total_portfolio_value) * 100
                    
                    # Calculate position-specific metrics
                    returns = historical_data['price'].pct_change().dropna()
                    position_vol = returns.std() * np.sqrt(365) * 100
                    position_var = np.percentile(returns, 5) * position_value
                    
                    # Risk-adjusted metrics
                    sharpe = (returns.mean() * 365) / (returns.std() * np.sqrt(365)) if returns.std() > 0 else 0
                    
                    position_risks.append({
                        'Symbol': symbol.upper(),
                        'Value': f"${position_value:,.2f}",
                        'Allocation': f"{allocation:.1f}%",
                        'Volatility': f"{position_vol:.1f}%",
                        'Daily VaR': f"${abs(position_var):,.2f}",
                        'Sharpe Ratio': f"{sharpe:.2f}",
                        'Risk Score': min(10, max(1, (allocation * position_vol) / 100))
                    })
            
            except Exception as e:
                st.warning(f"Unable to analyze {symbol}: {e}")
        
        if position_risks:
            df_risks = pd.DataFrame(position_risks)
            st.dataframe(df_risks, use_container_width=True)
            
            # Position risk visualization
            fig_positions = go.Figure()
            
            # Create bubble chart: x=allocation, y=volatility, size=value
            for _, row in df_risks.iterrows():
                allocation = float(row['Allocation'].replace('%', ''))
                volatility = float(row['Volatility'].replace('%', ''))
                value = float(row['Value'].replace('$', '').replace(',', ''))
                
                fig_positions.add_trace(go.Scatter(
                    x=[allocation],
                    y=[volatility],
                    mode='markers+text',
                    text=[row['Symbol']],
                    textposition="middle center",
                    marker=dict(
                        size=value/1000,  # Scale down for visualization
                        color=row['Risk Score'],
                        colorscale='RdYlGn_r',
                        showscale=True,
                        colorbar=dict(title="Risk Score")
                    ),
                    name=row['Symbol'],
                    hovertemplate=f"<b>{row['Symbol']}</b><br>" +
                                f"Allocation: {row['Allocation']}<br>" +
                                f"Volatility: {row['Volatility']}<br>" +
                                f"Value: {row['Value']}<br>" +
                                f"Risk Score: {row['Risk Score']:.1f}<extra></extra>"
                ))
            
            fig_positions.update_layout(
                title="Position Risk Analysis (Size = Value, Color = Risk Score)",
                xaxis_title="Portfolio Allocation (%)",
                yaxis_title="Annualized Volatility (%)",
                height=500,
                showlegend=False
            )
            
            st.plotly_chart(fig_positions, use_container_width=True)
    
    with tab2:
        st.markdown("#### 📉 Value at Risk Analysis")
        
        # VaR calculations
        var_metrics = risk_analysis.get('var_metrics', {})
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### 📊 VaR Summary")
            
            var_95_1d = var_metrics.get('var_95_1d', 0)
            var_99_1d = var_metrics.get('var_99_1d', 0)
            expected_shortfall = var_metrics.get('expected_shortfall', 0)
            
            st.write(f"**95% VaR (1-day):** ${var_95_1d:,.2f}")
            st.write(f"**99% VaR (1-day):** ${var_99_1d:,.2f}")
            st.write(f"**Expected Shortfall:** ${expected_shortfall:,.2f}")
            
            # VaR as percentage of portfolio
            var_95_pct = (var_95_1d / portfolio_value) * 100
            var_99_pct = (var_99_1d / portfolio_value) * 100
            
            st.write(f"**95% VaR (% of portfolio):** {var_95_pct:.2f}%")
            st.write(f"**99% VaR (% of portfolio):** {var_99_pct:.2f}%")
        
        with col2:
            st.markdown("##### 🎯 VaR Interpretation")
            
            if var_95_pct < 2:
                st.success("✅ Low risk - VaR under 2% of portfolio")
            elif var_95_pct < 5:
                st.warning("⚠️ Moderate risk - VaR between 2-5%")
            else:
                st.error("🚨 High risk - VaR above 5% of portfolio")
            
            st.info(f"""
            **VaR Explanation:**
            - 95% confidence: On 95% of days, losses will not exceed ${var_95_1d:,.2f}
            - 99% confidence: On 99% of days, losses will not exceed ${var_99_1d:,.2f}
            - Expected once every: {1/0.05:.0f} days (95%), {1/0.01:.0f} days (99%)
            """)
        
        # VaR historical simulation
        if 'var_simulation' in risk_analysis:
            st.markdown("##### 📈 VaR Historical Simulation")
            
            var_data = risk_analysis['var_simulation']
            
            fig_var = go.Figure()
            
            # Portfolio returns distribution
            fig_var.add_trace(go.Histogram(
                x=var_data['returns'],
                nbinsx=50,
                name='Portfolio Returns',
                opacity=0.7
            ))
            
            # VaR lines
            fig_var.add_vline(
                x=var_data['var_95'],
                line_dash="dash",
                line_color="orange",
                annotation_text="95% VaR"
            )
            
            fig_var.add_vline(
                x=var_data['var_99'],
                line_dash="dash",
                line_color="red",
                annotation_text="99% VaR"
            )
            
            fig_var.update_layout(
                title="Portfolio Returns Distribution with VaR Levels",
                xaxis_title="Daily Return (%)",
                yaxis_title="Frequency",
                height=400
            )
            
            st.plotly_chart(fig_var, use_container_width=True)
    
    with tab3:
        st.markdown("#### 🧪 Stress Testing & Scenario Analysis")
        
        # Define stress scenarios
        scenarios = {
            "Market Crash (-50%)": -0.50,
            "Bear Market (-30%)": -0.30,
            "Correction (-20%)": -0.20,
            "Volatility Spike": "volatility"
        }
        
        if stress_scenario == "Custom":
            shock_magnitude = custom_shock / 100
        else:
            shock_magnitude = scenarios.get(stress_scenario, -0.20)
        
        # Calculate stress test results
        stress_results = risk_manager.stress_test_portfolio(
            holdings=holdings,
            data_fetcher=data_fetcher,
            shock_magnitude=shock_magnitude if shock_magnitude != "volatility" else -0.30
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### 📊 Stress Test Results")
            
            current_value = stress_results.get('current_portfolio_value', portfolio_value)
            stressed_value = stress_results.get('stressed_portfolio_value', 0)
            total_loss = current_value - stressed_value
            loss_percentage = (total_loss / current_value) * 100 if current_value > 0 else 0
            
            st.metric(
                "💰 Current Portfolio Value",
                f"${current_value:,.2f}",
                "Baseline"
            )
            
            st.metric(
                "📉 Stressed Portfolio Value",
                f"${stressed_value:,.2f}",
                f"-${total_loss:,.2f}"
            )
            
            st.metric(
                "💥 Total Loss",
                f"{loss_percentage:.1f}%",
                f"${total_loss:,.2f}"
            )
            
            # Risk assessment
            if loss_percentage < 10:
                st.success("✅ Portfolio resilient to stress scenario")
            elif loss_percentage < 25:
                st.warning("⚠️ Moderate impact from stress scenario")
            else:
                st.error("🚨 Significant vulnerability to stress scenario")
        
        with col2:
            st.markdown("##### 📈 Position-Level Impact")
            
            if 'position_impacts' in stress_results:
                impact_data = []
                for symbol, impact in stress_results['position_impacts'].items():
                    impact_data.append({
                        'Symbol': symbol.upper(),
                        'Current Value': f"${impact['current_value']:,.2f}",
                        'Stressed Value': f"${impact['stressed_value']:,.2f}",
                        'Loss': f"${impact['loss']:,.2f}",
                        'Loss %': f"{impact['loss_pct']:.1f}%"
                    })
                
                df_impact = pd.DataFrame(impact_data)
                st.dataframe(df_impact, use_container_width=True)
        
        # Scenario comparison chart
        st.markdown("##### 🎭 Multiple Scenario Analysis")
        
        scenario_results = []
        scenario_names = ["Current", "Mild (-10%)", "Moderate (-20%)", "Severe (-35%)", "Extreme (-50%)"]
        shock_levels = [0, -0.10, -0.20, -0.35, -0.50]
        
        for name, shock in zip(scenario_names, shock_levels):
            if shock == 0:
                scenario_value = portfolio_value
            else:
                result = risk_manager.stress_test_portfolio(holdings, data_fetcher, shock)
                scenario_value = result.get('stressed_portfolio_value', 0)
            
            scenario_results.append({
                'Scenario': name,
                'Portfolio Value': scenario_value,
                'Loss %': ((portfolio_value - scenario_value) / portfolio_value * 100) if portfolio_value > 0 else 0
            })
        
        df_scenarios = pd.DataFrame(scenario_results)
        
        fig_scenarios = go.Figure()
        
        # Portfolio value bars
        colors = ['green', 'yellow', 'orange', 'red', 'darkred']
        fig_scenarios.add_trace(go.Bar(
            x=df_scenarios['Scenario'],
            y=df_scenarios['Portfolio Value'],
            marker_color=colors,
            text=[f"${x:,.0f}" for x in df_scenarios['Portfolio Value']],
            textposition='auto'
        ))
        
        fig_scenarios.update_layout(
            title="Portfolio Value Under Different Stress Scenarios",
            xaxis_title="Scenario",
            yaxis_title="Portfolio Value ($)",
            height=400
        )
        
        st.plotly_chart(fig_scenarios, use_container_width=True)
    
    with tab4:
        st.markdown("#### 🎯 Risk Optimization Recommendations")
        
        # Generate optimization recommendations
        optimization = risk_manager.generate_optimization_recommendations(
            holdings=holdings,
            portfolio_value=portfolio_value,
            risk_analysis=risk_analysis,
            max_position_pct=max_single_position,
            target_risk=risk_tolerance
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### ⚠️ Risk Alerts")
            
            alerts = optimization.get('alerts', [])
            if alerts:
                for alert in alerts:
                    alert_type = alert.get('type', 'info')
                    message = alert.get('message', '')
                    
                    if alert_type == 'warning':
                        st.warning(f"⚠️ {message}")
                    elif alert_type == 'error':
                        st.error(f"🚨 {message}")
                    else:
                        st.info(f"ℹ️ {message}")
            else:
                st.success("✅ No immediate risk alerts")
        
        with col2:
            st.markdown("##### 💡 Optimization Suggestions")
            
            suggestions = optimization.get('suggestions', [])
            if suggestions:
                for suggestion in suggestions:
                    st.write(f"• {suggestion}")
            else:
                st.info("Portfolio allocation appears optimal")
        
        # Optimal allocation recommendation
        if 'recommended_allocation' in optimization:
            st.markdown("##### 🎯 Recommended Portfolio Allocation")
            
            current_allocation = []
            recommended_allocation = []
            
            for symbol in holdings.keys():
                current_value = holdings[symbol] * data_fetcher.get_current_price(symbol, 'usd')['current_price']
                current_pct = (current_value / portfolio_value) * 100
                recommended_pct = optimization['recommended_allocation'].get(symbol, current_pct)
                
                current_allocation.append({'Symbol': symbol.upper(), 'Current %': current_pct})
                recommended_allocation.append({'Symbol': symbol.upper(), 'Recommended %': recommended_pct})
            
            # Allocation comparison chart
            fig_allocation = go.Figure()
            
            symbols = [item['Symbol'] for item in current_allocation]
            current_pcts = [item['Current %'] for item in current_allocation]
            recommended_pcts = [item['Recommended %'] for item in recommended_allocation]
            
            fig_allocation.add_trace(go.Bar(
                name='Current Allocation',
                x=symbols,
                y=current_pcts,
                marker_color='lightblue'
            ))
            
            fig_allocation.add_trace(go.Bar(
                name='Recommended Allocation',
                x=symbols,
                y=recommended_pcts,
                marker_color='darkblue'
            ))
            
            fig_allocation.update_layout(
                title="Current vs Recommended Portfolio Allocation",
                xaxis_title="Asset",
                yaxis_title="Allocation (%)",
                barmode='group',
                height=400
            )
            
            st.plotly_chart(fig_allocation, use_container_width=True)
        
        # Kelly Criterion position sizing
        st.markdown("##### 📊 Kelly Criterion Position Sizing")
        
        kelly_results = risk_manager.kelly_criterion_sizing(holdings, data_fetcher)
        
        if kelly_results:
            kelly_data = []
            for symbol, kelly_info in kelly_results.items():
                kelly_data.append({
                    'Symbol': symbol.upper(),
                    'Current Size': f"{kelly_info['current_size']:.1f}%",
                    'Kelly Optimal': f"{kelly_info['kelly_size']:.1f}%",
                    'Recommendation': kelly_info['recommendation']
                })
            
            df_kelly = pd.DataFrame(kelly_data)
            st.dataframe(df_kelly, use_container_width=True)
            
            st.info("""
            **Kelly Criterion** optimizes position sizes based on win rate and average win/loss ratio.
            It maximizes long-term growth while managing risk of ruin.
            """)

except Exception as e:
    st.error(f"Risk analysis error: {str(e)}")
    st.info("Please ensure you have an active portfolio with holdings to analyze.")

# Risk monitoring alerts
st.subheader("🔔 Risk Monitoring & Alerts")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 📈 Real-time Risk Alerts")
    
    # Check current risk levels
    if holdings:
        current_risk_level = risk_analysis.get('overall_risk_score', 0)
        
        if current_risk_level > 7:
            st.error("🚨 HIGH RISK ALERT: Consider reducing position sizes")
        elif current_risk_level > 5:
            st.warning("⚠️ MEDIUM RISK: Monitor positions closely")
        else:
            st.success("✅ RISK UNDER CONTROL: Portfolio within limits")
        
        # VaR alert
        var_pct = (risk_analysis.get('var_1d', 0) / portfolio_value) * 100 if portfolio_value > 0 else 0
        if var_pct > 5:
            st.warning(f"📉 VaR Alert: Daily VaR is {var_pct:.1f}% of portfolio")

with col2:
    st.markdown("#### 🎯 Risk Management Actions")
    
    st.write("**Recommended Actions:**")
    
    # Dynamic recommendations based on current state
    if holdings:
        total_exposure = sum([holdings[symbol] * data_fetcher.get_current_price(symbol, 'usd')['current_price'] 
                            for symbol in holdings.keys()])
        exposure_pct = (total_exposure / portfolio_value) * 100 if portfolio_value > 0 else 0
        
        if exposure_pct > 90:
            st.write("• 📉 Reduce crypto exposure, increase cash reserves")
        
        if len(holdings) < 3:
            st.write("• 🎯 Consider diversifying across more assets")
        
        st.write("• 📊 Set stop-loss orders for large positions")
        st.write("• 🔄 Regular portfolio rebalancing")
        st.write("• 📈 Monitor correlation between holdings")

# Footer
st.markdown("---")
st.markdown("⚠️ **Risk Management** • Advanced portfolio risk analytics • For educational purposes only")
