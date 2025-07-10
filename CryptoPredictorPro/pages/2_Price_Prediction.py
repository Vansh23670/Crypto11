import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime, timedelta
from utils.data_fetcher import CryptoDataFetcher
from utils.ml_predictor import CryptoPricePredictor

st.set_page_config(page_title="Price Prediction", page_icon="🔮", layout="wide")

# Initialize components
@st.cache_resource
def get_data_fetcher():
    return CryptoDataFetcher()

@st.cache_resource
def get_predictor():
    return CryptoPricePredictor()

data_fetcher = get_data_fetcher()
predictor = get_predictor()

st.title("🔮 AI-Powered Price Prediction")

# Sidebar controls
with st.sidebar:
    st.header("🤖 Prediction Settings")
    
    # Coin selection
    coins = ['bitcoin', 'ethereum', 'dogecoin', 'cardano', 'polygon', 'solana']
    coin_names = ['Bitcoin (BTC)', 'Ethereum (ETH)', 'Dogecoin (DOGE)', 
                  'Cardano (ADA)', 'Polygon (MATIC)', 'Solana (SOL)']
    
    selected_coin = st.selectbox(
        "Select Cryptocurrency",
        coins,
        format_func=lambda x: coin_names[coins.index(x)]
    )
    
    # Training period
    training_periods = {
        '30 days': 30,
        '90 days': 90,
        '180 days': 180,
        '365 days': 365
    }
    
    training_period = st.selectbox("Training Data Period", list(training_periods.keys()), index=2)
    training_days = training_periods[training_period]
    
    # Prediction period
    prediction_days = st.slider("Prediction Period (days)", 1, 30, 7)
    
    # Currency
    currency = st.selectbox("Currency", ['usd', 'inr'])
    
    # Model selection
    model_type = st.selectbox("Prediction Model", ['Prophet (Recommended)', 'Simple Trend'])
    
    # Prediction button
    run_prediction = st.button("🚀 Generate Prediction", type="primary")

# Main content
if run_prediction:
    with st.spinner("Fetching historical data and training model..."):
        try:
            # Fetch historical data
            df = data_fetcher.get_historical_data(selected_coin, currency, training_days)
            
            if df.empty:
                st.error("No historical data available for the selected cryptocurrency.")
                st.stop()
            
            # Current price info
            current_data = data_fetcher.get_current_price(selected_coin, currency)
            
            # Display current metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Current Price",
                    f"${current_data['current_price']:.2f}" if currency == 'usd' else f"₹{current_data['current_price']:.2f}",
                    f"{current_data['price_change_percentage_24h']:.2f}%"
                )
            
            with col2:
                st.metric(
                    "Training Period",
                    f"{training_days} days"
                )
            
            with col3:
                st.metric(
                    "Prediction Period",
                    f"{prediction_days} days"
                )
            
            with col4:
                st.metric(
                    "Data Points",
                    f"{len(df)}"
                )
            
            # Make predictions
            if model_type == 'Prophet (Recommended)':
                # Train Prophet model
                success = predictor.train_model(df, coin_names[coins.index(selected_coin)])
                
                if success:
                    predictions = predictor.predict_price(prediction_days)
                    model_performance = predictor.get_model_performance()
                    trend = predictor.predict_trend(prediction_days)
                    insights = predictor.get_key_insights()
                else:
                    st.error("Failed to train the prediction model. Please try with a different coin or time period.")
                    st.stop()
            else:
                # Simple trend prediction
                predictions = predictor.simple_trend_prediction(df, prediction_days)
                model_performance = {}
                trend = "Simple Trend Analysis"
                insights = ["Simple trend-based prediction", "Lower accuracy than ML models"]
            
            # Display predictions
            st.subheader("📊 Price Predictions")
            
            # Create prediction chart
            fig = go.Figure()
            
            # Historical data
            fig.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df['price'],
                mode='lines',
                name='Historical Price',
                line=dict(color='blue', width=2)
            ))
            
            # Predictions
            if predictions:
                pred_dates = [pred['date'] for pred in predictions]
                pred_prices = [pred['predicted_price'] for pred in predictions]
                pred_upper = [pred['upper_bound'] for pred in predictions]
                pred_lower = [pred['lower_bound'] for pred in predictions]
                
                # Predicted price line
                fig.add_trace(go.Scatter(
                    x=pred_dates,
                    y=pred_prices,
                    mode='lines+markers',
                    name='Predicted Price',
                    line=dict(color='red', width=2, dash='dash')
                ))
                
                # Confidence interval
                fig.add_trace(go.Scatter(
                    x=pred_dates,
                    y=pred_upper,
                    mode='lines',
                    line=dict(color='rgba(255,0,0,0)'),
                    showlegend=False,
                    name='Upper Bound'
                ))
                
                fig.add_trace(go.Scatter(
                    x=pred_dates,
                    y=pred_lower,
                    mode='lines',
                    line=dict(color='rgba(255,0,0,0)'),
                    fill='tonexty',
                    fillcolor='rgba(255,0,0,0.2)',
                    name='Confidence Interval',
                    showlegend=True
                ))
            
            fig.update_layout(
                title=f"{coin_names[coins.index(selected_coin)]} Price Prediction",
                xaxis_title="Date",
                yaxis_title=f"Price ({'USD' if currency == 'usd' else 'INR'})",
                height=500,
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Prediction summary
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🎯 Prediction Summary")
                
                if predictions:
                    current_price = df['price'].iloc[-1]
                    final_prediction = predictions[-1]['predicted_price']
                    price_change = (final_prediction - current_price) / current_price * 100
                    
                    st.markdown(f"""
                    **Current Price:** ${current_price:.2f}
                    
                    **Predicted Price ({prediction_days} days):** ${final_prediction:.2f}
                    
                    **Expected Change:** {price_change:+.2f}%
                    
                    **Trend:** {trend}
                    
                    **Confidence:** {predictions[-1]['confidence']:.1f}%
                    """)
                    
                    # Price target visualization
                    if price_change > 0:
                        st.success(f"📈 **BULLISH**: Price expected to rise by {price_change:.2f}%")
                    else:
                        st.error(f"📉 **BEARISH**: Price expected to fall by {abs(price_change):.2f}%")
                
                else:
                    st.warning("No predictions available.")
            
            with col2:
                st.subheader("🔍 Model Performance")
                
                if model_performance:
                    st.markdown(f"""
                    **Model Accuracy Metrics:**
                    
                    - **Mean Absolute Error:** {model_performance.get('mae', 0):.2f}
                    - **Root Mean Square Error:** {model_performance.get('rmse', 0):.2f}
                    - **Mean Absolute Percentage Error:** {model_performance.get('mape', 0):.2f}%
                    - **Training Data Points:** {model_performance.get('data_points', 0)}
                    """)
                    
                    # Model quality indicator
                    mape = model_performance.get('mape', 100)
                    if mape < 10:
                        st.success("✅ **Excellent** model accuracy")
                    elif mape < 20:
                        st.info("✅ **Good** model accuracy")
                    elif mape < 30:
                        st.warning("⚠️ **Fair** model accuracy")
                    else:
                        st.error("❌ **Poor** model accuracy")
                else:
                    st.info("Model performance metrics not available for simple trend analysis.")
            
            # Key insights
            st.subheader("💡 Key Insights")
            
            insight_cols = st.columns(3)
            for i, insight in enumerate(insights[:3]):
                with insight_cols[i % 3]:
                    st.info(insight)
            
            # Detailed predictions table
            st.subheader("📋 Detailed Predictions")
            
            if predictions:
                pred_df = pd.DataFrame(predictions)
                pred_df['date'] = pred_df['date'].dt.strftime('%Y-%m-%d')
                pred_df['predicted_price'] = pred_df['predicted_price'].round(2)
                pred_df['lower_bound'] = pred_df['lower_bound'].round(2)
                pred_df['upper_bound'] = pred_df['upper_bound'].round(2)
                pred_df['confidence'] = pred_df['confidence'].round(1)
                
                st.dataframe(
                    pred_df.rename(columns={
                        'date': 'Date',
                        'predicted_price': 'Predicted Price',
                        'lower_bound': 'Lower Bound',
                        'upper_bound': 'Upper Bound',
                        'confidence': 'Confidence %'
                    }),
                    use_container_width=True
                )
            
        except Exception as e:
            st.error(f"Error generating predictions: {str(e)}")
            st.info("Please try with a different cryptocurrency or time period.")

else:
    # Welcome message
    st.markdown("""
    ## 🚀 Welcome to AI-Powered Price Prediction!
    
    This advanced prediction system uses machine learning models to forecast cryptocurrency prices:
    
    ### 🤖 Available Models:
    - **Prophet (Recommended)**: Advanced time series forecasting with seasonality detection
    - **Simple Trend**: Basic trend analysis for quick predictions
    
    ### 📊 Features:
    - ✅ Multi-day price predictions
    - ✅ Confidence intervals
    - ✅ Model performance metrics
    - ✅ Trend analysis
    - ✅ Key insights and recommendations
    
    ### 🎯 How to Use:
    1. Select a cryptocurrency from the sidebar
    2. Choose training data period (more data = better accuracy)
    3. Set prediction period (1-30 days)
    4. Click "Generate Prediction" to start
    
    ### ⚠️ Important Notes:
    - Predictions are based on historical data and patterns
    - Cryptocurrency markets are highly volatile and unpredictable
    - Use predictions as one factor in your trading decisions
    - Never invest more than you can afford to lose
    - This is for educational purposes only, not financial advice
    """)
    
    # Quick stats
    st.subheader("📈 Market Overview")
    
    try:
        # Show trending coins
        trending = data_fetcher.get_trending_coins()
        
        if trending:
            st.write("**🔥 Trending Cryptocurrencies:**")
            
            cols = st.columns(min(len(trending), 4))
            for i, coin in enumerate(trending[:4]):
                with cols[i]:
                    st.markdown(f"""
                    **{coin['name']}** ({coin['symbol']})
                    
                    Rank: #{coin['market_cap_rank']}
                    """)
    
    except Exception as e:
        st.warning("Could not fetch trending coins data.")

# Footer
st.markdown("---")
st.markdown("""
📊 **Disclaimer:** This prediction tool uses statistical models and machine learning algorithms to analyze historical price data. 
Cryptocurrency markets are highly volatile and influenced by many factors not captured in price data alone. 
Always conduct thorough research and consider multiple factors before making investment decisions.
""")
