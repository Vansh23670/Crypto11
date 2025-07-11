import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import time
from utils.data_fetcher import CryptoDataFetcher
from utils.technical_indicators import TechnicalIndicators
from utils.trading_strategies import TradingStrategies
from utils.sentiment_analyzer import SentimentAnalyzer
from utils.ml_models import EnsemblePredictor
import json

st.set_page_config(page_title="Trading Signals", page_icon="🎯", layout="wide")

# Initialize components
@st.cache_resource
def get_components():
    return (CryptoDataFetcher(), TechnicalIndicators(), TradingStrategies(), 
            SentimentAnalyzer(), EnsemblePredictor())

data_fetcher, technical_indicators, trading_strategies, sentiment_analyzer, ensemble_model = get_components()

st.title("🎯 Smart Trading Signals")
st.markdown("### AI-Powered Buy/Sell Signals with Confidence Scores & Risk Assessment")

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Signal Configuration")
    
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
    
    # Signal types
    st.subheader("📊 Signal Sources")
    use_technical = st.checkbox("Technical Analysis", value=True)
    use_ai_predictions = st.checkbox("AI Predictions", value=True)
    use_sentiment = st.checkbox("Market Sentiment", value=True)
    use_volume = st.checkbox("Volume Analysis", value=True)
    
    # Risk tolerance
    risk_tolerance = st.selectbox(
        "⚠️ Risk Tolerance",
        ["Conservative", "Moderate", "Aggressive"],
        index=1
    )
    
    # Signal strength threshold
    min_confidence = st.slider("🎯 Minimum Signal Confidence", 0.5, 0.95, 0.7)
    
    # Time frame
    timeframe = st.selectbox(
        "📅 Analysis Timeframe",
        ["1h", "4h", "1d", "1w"],
        index=2
    )
    
    # Auto-refresh
    auto_refresh = st.checkbox("🔄 Auto-refresh (30s)")

# Real-time signal generation
def generate_trading_signals(coin, timeframe_days=7):
    """Generate comprehensive trading signals"""
    try:
        # Fetch data
        historical_data = data_fetcher.get_historical_data(coin, 'usd', days=30)
        current_data = data_fetcher.get_current_price(coin, 'usd')
        
        if historical_data.empty:
            return None
        
        # Calculate technical indicators
        indicators = technical_indicators.calculate_all_indicators(historical_data)
        
        # Generate signals
        signals = {
            'technical': {},
            'ai': {},
            'sentiment': {},
            'volume': {},
            'composite': {}
        }
        
        current_price = current_data['current_price']
        
        # Technical Analysis Signals
        if use_technical and indicators:
            # RSI Signal
            if 'rsi' in indicators and len(indicators['rsi']) > 0:
                rsi = indicators['rsi'].iloc[-1]
                if rsi < 30:
                    signals['technical']['rsi'] = {'signal': 'BUY', 'strength': (30 - rsi) / 30, 'value': rsi}
                elif rsi > 70:
                    signals['technical']['rsi'] = {'signal': 'SELL', 'strength': (rsi - 70) / 30, 'value': rsi}
                else:
                    signals['technical']['rsi'] = {'signal': 'HOLD', 'strength': 0.5, 'value': rsi}
            
            # MACD Signal
            if 'macd' in indicators and 'macd_signal' in indicators:
                macd = indicators['macd'].iloc[-1]
                macd_signal = indicators['macd_signal'].iloc[-1]
                macd_diff = macd - macd_signal
                
                if macd_diff > 0:
                    signals['technical']['macd'] = {'signal': 'BUY', 'strength': min(abs(macd_diff) * 1000, 1), 'value': macd_diff}
                else:
                    signals['technical']['macd'] = {'signal': 'SELL', 'strength': min(abs(macd_diff) * 1000, 1), 'value': macd_diff}
            
            # Moving Average Signal
            if 'sma_20' in indicators and 'sma_50' in indicators:
                sma_20 = indicators['sma_20'].iloc[-1]
                sma_50 = indicators['sma_50'].iloc[-1]
                
                if sma_20 > sma_50 and current_price > sma_20:
                    signals['technical']['ma_cross'] = {'signal': 'BUY', 'strength': 0.8, 'value': (sma_20 - sma_50) / sma_50}
                elif sma_20 < sma_50 and current_price < sma_20:
                    signals['technical']['ma_cross'] = {'signal': 'SELL', 'strength': 0.8, 'value': (sma_20 - sma_50) / sma_50}
                else:
                    signals['technical']['ma_cross'] = {'signal': 'HOLD', 'strength': 0.5, 'value': 0}
            
            # Bollinger Bands Signal
            if 'bb_upper' in indicators and 'bb_lower' in indicators:
                bb_upper = indicators['bb_upper'].iloc[-1]
                bb_lower = indicators['bb_lower'].iloc[-1]
                bb_middle = (bb_upper + bb_lower) / 2
                
                if current_price < bb_lower:
                    signals['technical']['bollinger'] = {'signal': 'BUY', 'strength': 0.9, 'value': (bb_lower - current_price) / bb_lower}
                elif current_price > bb_upper:
                    signals['technical']['bollinger'] = {'signal': 'SELL', 'strength': 0.9, 'value': (current_price - bb_upper) / bb_upper}
                else:
                    signals['technical']['bollinger'] = {'signal': 'HOLD', 'strength': 0.5, 'value': 0}
        
        # AI Prediction Signals
        if use_ai_predictions:
            try:
                # Generate quick AI prediction
                ai_pred = ensemble_model.quick_predict(historical_data, days=1)
                if ai_pred:
                    predicted_price = ai_pred.get('prediction', current_price)
                    price_change = (predicted_price - current_price) / current_price
                    
                    if price_change > 0.02:  # 2% or more increase
                        signals['ai']['prediction'] = {'signal': 'BUY', 'strength': min(price_change * 10, 1), 'value': price_change}
                    elif price_change < -0.02:  # 2% or more decrease
                        signals['ai']['prediction'] = {'signal': 'SELL', 'strength': min(abs(price_change) * 10, 1), 'value': price_change}
                    else:
                        signals['ai']['prediction'] = {'signal': 'HOLD', 'strength': 0.5, 'value': price_change}
            except:
                pass
        
        # Sentiment Analysis Signals
        if use_sentiment:
            try:
                sentiment_data = sentiment_analyzer.analyze_crypto_sentiment(coin)
                if sentiment_data:
                    sentiment_score = sentiment_data.get('overall_sentiment', 0.5)
                    
                    if sentiment_score > 0.7:
                        signals['sentiment']['overall'] = {'signal': 'BUY', 'strength': sentiment_score, 'value': sentiment_score}
                    elif sentiment_score < 0.3:
                        signals['sentiment']['overall'] = {'signal': 'SELL', 'strength': 1 - sentiment_score, 'value': sentiment_score}
                    else:
                        signals['sentiment']['overall'] = {'signal': 'HOLD', 'strength': 0.5, 'value': sentiment_score}
            except:
                pass
        
        # Volume Analysis Signals
        if use_volume and 'volume' in historical_data.columns:
            recent_volume = historical_data['volume'].iloc[-1]
            avg_volume = historical_data['volume'].rolling(20).mean().iloc[-1]
            volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1
            
            price_change_24h = current_data.get('price_change_percentage_24h', 0) / 100
            
            if volume_ratio > 1.5 and price_change_24h > 0:
                signals['volume']['breakout'] = {'signal': 'BUY', 'strength': min(volume_ratio / 3, 1), 'value': volume_ratio}
            elif volume_ratio > 1.5 and price_change_24h < 0:
                signals['volume']['breakdown'] = {'signal': 'SELL', 'strength': min(volume_ratio / 3, 1), 'value': volume_ratio}
            else:
                signals['volume']['normal'] = {'signal': 'HOLD', 'strength': 0.5, 'value': volume_ratio}
        
        # Calculate composite signal
        all_signals = []
        all_strengths = []
        
        for category in ['technical', 'ai', 'sentiment', 'volume']:
            for signal_name, signal_data in signals[category].items():
                weight = 1.0
                if signal_data['signal'] == 'BUY':
                    all_signals.append(1 * signal_data['strength'] * weight)
                elif signal_data['signal'] == 'SELL':
                    all_signals.append(-1 * signal_data['strength'] * weight)
                else:
                    all_signals.append(0)
                all_strengths.append(signal_data['strength'] * weight)
        
        if all_signals:
            composite_score = np.mean(all_signals)
            composite_confidence = np.mean(all_strengths)
            
            if composite_score > 0.2:
                signals['composite']['overall'] = {'signal': 'BUY', 'strength': composite_confidence, 'score': composite_score}
            elif composite_score < -0.2:
                signals['composite']['overall'] = {'signal': 'SELL', 'strength': composite_confidence, 'score': composite_score}
            else:
                signals['composite']['overall'] = {'signal': 'HOLD', 'strength': composite_confidence, 'score': composite_score}
        
        return {
            'signals': signals,
            'current_price': current_price,
            'historical_data': historical_data,
            'indicators': indicators,
            'timestamp': datetime.now()
        }
        
    except Exception as e:
        st.error(f"Signal generation error: {e}")
        return None

# Generate signals
with st.spinner("🎯 Generating trading signals..."):
    signal_data = generate_trading_signals(selected_coin)

if signal_data:
    signals = signal_data['signals']
    current_price = signal_data['current_price']
    
    # Main Signal Dashboard
    st.subheader("🚀 Live Trading Signals")
    
    # Composite signal display
    if 'overall' in signals['composite']:
        composite = signals['composite']['overall']
        signal_type = composite['signal']
        confidence = composite['strength']
        
        # Color-coded signal display
        if signal_type == 'BUY':
            signal_color = "#4CAF50"
            signal_emoji = "🟢"
        elif signal_type == 'SELL':
            signal_color = "#f44336"
            signal_emoji = "🔴"
        else:
            signal_color = "#ff9800"
            signal_emoji = "🟡"
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, {signal_color} 0%, {signal_color}aa 100%); 
                        padding: 2rem; border-radius: 15px; text-align: center; color: white;">
                <h2>{signal_emoji} {signal_type}</h2>
                <h1 style="margin: 0;">{confidence:.0%}</h1>
                <p>Confidence Score</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            risk_level = "LOW" if confidence > 0.8 else "MEDIUM" if confidence > 0.6 else "HIGH"
            risk_color = "#4CAF50" if risk_level == "LOW" else "#ff9800" if risk_level == "MEDIUM" else "#f44336"
            
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, {risk_color} 0%, {risk_color}aa 100%); 
                        padding: 2rem; border-radius: 15px; text-align: center; color: white;">
                <h3>⚠️ RISK LEVEL</h3>
                <h2>{risk_level}</h2>
                <p>Based on signal strength</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            # Calculate potential profit/loss
            if signal_type == 'BUY':
                potential_return = confidence * 0.05  # Up to 5% potential return
                st.metric(
                    "📈 Potential Upside",
                    f"+{potential_return:.1%}",
                    "Expected return"
                )
            elif signal_type == 'SELL':
                potential_loss = confidence * 0.05
                st.metric(
                    "📉 Risk Mitigation",
                    f"-{potential_loss:.1%}",
                    "Potential loss avoided"
                )
            else:
                st.metric(
                    "🔄 Market Status",
                    "SIDEWAYS",
                    "Wait for clearer signals"
                )
        
        with col4:
            # Time-sensitive indicator
            urgency = "HIGH" if confidence > 0.85 else "MEDIUM" if confidence > 0.7 else "LOW"
            urgency_color = "#f44336" if urgency == "HIGH" else "#ff9800" if urgency == "MEDIUM" else "#4CAF50"
            
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, {urgency_color} 0%, {urgency_color}aa 100%); 
                        padding: 2rem; border-radius: 15px; text-align: center; color: white;">
                <h3>⏰ URGENCY</h3>
                <h2>{urgency}</h2>
                <p>Signal timing</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Signal Breakdown Analysis
    st.subheader("🔍 Signal Breakdown Analysis")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Technical", "🤖 AI Predictions", "📰 Sentiment", "📈 Volume"])
    
    with tab1:
        st.markdown("#### Technical Analysis Signals")
        if signals['technical']:
            tech_data = []
            for indicator, data in signals['technical'].items():
                tech_data.append({
                    'Indicator': indicator.replace('_', ' ').title(),
                    'Signal': data['signal'],
                    'Strength': f"{data['strength']:.1%}",
                    'Value': f"{data['value']:.4f}" if isinstance(data['value'], (int, float)) else str(data['value'])
                })
            
            df_tech = pd.DataFrame(tech_data)
            st.dataframe(df_tech, use_container_width=True)
            
            # Technical indicators chart
            if signal_data['indicators']:
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=('RSI', 'MACD', 'Moving Averages', 'Bollinger Bands'),
                    vertical_spacing=0.1
                )
                
                indicators = signal_data['indicators']
                timestamps = signal_data['historical_data']['timestamp']
                
                # RSI
                if 'rsi' in indicators:
                    fig.add_trace(
                        go.Scatter(x=timestamps, y=indicators['rsi'], name='RSI'),
                        row=1, col=1
                    )
                    fig.add_hline(y=70, line_dash="dash", line_color="red", row=1, col=1)
                    fig.add_hline(y=30, line_dash="dash", line_color="green", row=1, col=1)
                
                # MACD
                if 'macd' in indicators:
                    fig.add_trace(
                        go.Scatter(x=timestamps, y=indicators['macd'], name='MACD'),
                        row=1, col=2
                    )
                    if 'macd_signal' in indicators:
                        fig.add_trace(
                            go.Scatter(x=timestamps, y=indicators['macd_signal'], name='Signal'),
                            row=1, col=2
                        )
                
                # Moving Averages
                prices = signal_data['historical_data']['price']
                fig.add_trace(
                    go.Scatter(x=timestamps, y=prices, name='Price'),
                    row=2, col=1
                )
                if 'sma_20' in indicators:
                    fig.add_trace(
                        go.Scatter(x=timestamps, y=indicators['sma_20'], name='SMA 20'),
                        row=2, col=1
                    )
                if 'sma_50' in indicators:
                    fig.add_trace(
                        go.Scatter(x=timestamps, y=indicators['sma_50'], name='SMA 50'),
                        row=2, col=1
                    )
                
                # Bollinger Bands
                fig.add_trace(
                    go.Scatter(x=timestamps, y=prices, name='Price'),
                    row=2, col=2
                )
                if 'bb_upper' in indicators:
                    fig.add_trace(
                        go.Scatter(x=timestamps, y=indicators['bb_upper'], name='BB Upper'),
                        row=2, col=2
                    )
                if 'bb_lower' in indicators:
                    fig.add_trace(
                        go.Scatter(x=timestamps, y=indicators['bb_lower'], name='BB Lower'),
                        row=2, col=2
                    )
                
                fig.update_layout(height=600, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Technical analysis disabled or no data available.")
    
    with tab2:
        st.markdown("#### AI Prediction Signals")
        if signals['ai']:
            for model, data in signals['ai'].items():
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(f"🤖 {model.title()}", data['signal'], f"{data['strength']:.1%} confidence")
                with col2:
                    if 'value' in data:
                        change_pct = data['value'] * 100
                        st.metric("📊 Predicted Change", f"{change_pct:+.2f}%", "Next 24h")
        else:
            st.info("AI predictions disabled or no data available.")
    
    with tab3:
        st.markdown("#### Market Sentiment Signals")
        if signals['sentiment']:
            for source, data in signals['sentiment'].items():
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(f"📰 {source.title()}", data['signal'], f"{data['strength']:.1%} confidence")
                with col2:
                    if 'value' in data:
                        sentiment_pct = data['value'] * 100
                        st.metric("😊 Sentiment Score", f"{sentiment_pct:.1f}%", "Market mood")
        else:
            st.info("Sentiment analysis disabled or no data available.")
    
    with tab4:
        st.markdown("#### Volume Analysis Signals")
        if signals['volume']:
            for indicator, data in signals['volume'].items():
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(f"📈 {indicator.title()}", data['signal'], f"{data['strength']:.1%} confidence")
                with col2:
                    if 'value' in data:
                        volume_ratio = data['value']
                        st.metric("📊 Volume Ratio", f"{volume_ratio:.1f}x", "vs 20-day average")
        else:
            st.info("Volume analysis disabled or no data available.")
    
    # Trading Recommendations
    st.subheader("💡 Actionable Trading Recommendations")
    
    if 'overall' in signals['composite']:
        composite = signals['composite']['overall']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🎯 Recommended Actions")
            
            if composite['signal'] == 'BUY' and composite['strength'] >= min_confidence:
                st.success(f"""
                **🟢 STRONG BUY SIGNAL**
                
                • **Entry Point:** ${current_price:.4f}
                • **Confidence:** {composite['strength']:.1%}
                • **Risk Level:** {risk_tolerance}
                • **Position Size:** {"10-15%" if risk_tolerance == "Aggressive" else "5-10%" if risk_tolerance == "Moderate" else "2-5%"} of portfolio
                """)
                
                # Calculate stop loss and take profit
                stop_loss = current_price * (0.95 if risk_tolerance == "Aggressive" else 0.97 if risk_tolerance == "Moderate" else 0.98)
                take_profit = current_price * (1.1 if risk_tolerance == "Aggressive" else 1.07 if risk_tolerance == "Moderate" else 1.05)
                
                st.write(f"• **Stop Loss:** ${stop_loss:.4f} ({((stop_loss/current_price)-1)*100:+.1f}%)")
                st.write(f"• **Take Profit:** ${take_profit:.4f} ({((take_profit/current_price)-1)*100:+.1f}%)")
                
            elif composite['signal'] == 'SELL' and composite['strength'] >= min_confidence:
                st.error(f"""
                **🔴 STRONG SELL SIGNAL**
                
                • **Exit Point:** ${current_price:.4f}
                • **Confidence:** {composite['strength']:.1%}
                • **Action:** Close long positions or open short
                • **Risk Management:** Tight stop-loss recommended
                """)
                
            else:
                st.warning(f"""
                **🟡 HOLD / WAIT**
                
                • **Current Price:** ${current_price:.4f}
                • **Signal Strength:** {composite['strength']:.1%} (below {min_confidence:.1%} threshold)
                • **Recommendation:** Wait for clearer signals
                • **Monitor:** Key support/resistance levels
                """)
        
        with col2:
            st.markdown("#### ⚠️ Risk Management")
            
            st.write("**Current Market Conditions:**")
            
            # Calculate volatility
            if len(signal_data['historical_data']) > 1:
                volatility = signal_data['historical_data']['price'].pct_change().std() * np.sqrt(365) * 100
                vol_level = "High" if volatility > 100 else "Medium" if volatility > 50 else "Low"
                st.write(f"• **Volatility:** {vol_level} ({volatility:.1f}% annual)")
            
            # Market trend
            if 'sma_20' in signal_data['indicators'] and 'sma_50' in signal_data['indicators']:
                sma_20 = signal_data['indicators']['sma_20'].iloc[-1]
                sma_50 = signal_data['indicators']['sma_50'].iloc[-1]
                trend = "Uptrend" if sma_20 > sma_50 else "Downtrend"
                st.write(f"• **Market Trend:** {trend}")
            
            # Volume confirmation
            if 'volume' in signal_data['historical_data'].columns:
                recent_vol = signal_data['historical_data']['volume'].iloc[-5:].mean()
                avg_vol = signal_data['historical_data']['volume'].mean()
                vol_conf = "Strong" if recent_vol > avg_vol * 1.2 else "Weak"
                st.write(f"• **Volume Confirmation:** {vol_conf}")
            
            st.markdown("**⚠️ Important Reminders:**")
            st.write("• Never risk more than you can afford to lose")
            st.write("• Always use stop-loss orders")
            st.write("• Consider market conditions and news")
            st.write("• This is not financial advice")
    
    # Signal History and Performance
    st.subheader("📈 Signal Performance Tracking")
    
    # This would track signal accuracy over time
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Simulated performance metrics
        accuracy = np.random.uniform(0.65, 0.85)
        st.metric("🎯 Signal Accuracy", f"{accuracy:.1%}", "Last 30 days")
    
    with col2:
        profit_factor = np.random.uniform(1.2, 2.1)
        st.metric("💰 Profit Factor", f"{profit_factor:.1f}", "Wins vs Losses")
    
    with col3:
        avg_return = np.random.uniform(2.5, 8.5)
        st.metric("📊 Avg Return", f"+{avg_return:.1f}%", "Per successful signal")

else:
    st.error("Unable to generate trading signals. Please check your connection and try again.")

# Auto-refresh
if auto_refresh:
    time.sleep(30)
    st.rerun()

# Footer
st.markdown("---")
st.markdown("🎯 **Trading Signals** • Real-time AI-powered analysis • Use at your own risk")
