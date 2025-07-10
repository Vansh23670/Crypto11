# Crypto Trading Prediction & Strategy App

## Overview

This is a comprehensive cryptocurrency trading and analysis web application built using **Streamlit**. The app provides real-time market data, technical analysis, AI-powered price predictions, trading strategy simulation, and portfolio tracking capabilities. It's designed as a paper trading platform that allows users to test strategies and track performance without real money.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit-based multi-page web application
- **UI Components**: Interactive charts using Plotly, sidebar controls, and real-time data displays
- **Session Management**: Streamlit's session state for maintaining user data across pages
- **Caching**: Streamlit's caching mechanisms (@st.cache_resource, @st.cache_data) for performance optimization

### Backend Architecture
- **Data Layer**: RESTful API integration with CoinGecko API for real-time cryptocurrency data
- **Database**: PostgreSQL for persistent data storage with automated fallback to session state
- **ML Pipeline**: Prophet-based time series forecasting for price predictions
- **Strategy Engine**: Custom backtesting engine for trading strategy simulation
- **Portfolio Management**: Database-backed portfolio tracking with persistent transaction history

### Key Design Decisions
1. **Streamlit Multi-page Architecture**: Chosen for rapid development and easy navigation between different functionalities
2. **CoinGecko API**: Selected for reliable, free cryptocurrency data without API key requirements
3. **Prophet Model**: Used for price prediction due to its simplicity and effectiveness with time series data
4. **PostgreSQL Database**: Integrated for persistent data storage with automatic failover to session state
5. **Hybrid Storage System**: Database-first approach with session state fallback for maximum reliability

## Key Components

### 1. Main Application (`app.py`)
- Entry point with basic dashboard functionality
- Portfolio manager initialization
- Coin selection and data refresh controls

### 2. Market Data Module (`pages/1_Market_Data.py`)
- Real-time price display with technical indicators
- Interactive charts with customizable time periods
- Support for multiple cryptocurrencies (Bitcoin, Ethereum, etc.)

### 3. Price Prediction Module (`pages/2_Price_Prediction.py`)
- AI-powered price forecasting using Prophet
- Configurable training periods and prediction horizons
- Visualization of predictions with confidence intervals

### 4. Strategy Simulator (`pages/3_Strategy_Simulator.py`)
- Backtesting engine for multiple trading strategies
- Performance metrics calculation (returns, win rate, Sharpe ratio)
- Strategy types: SMA Crossover, RSI, MACD, Bollinger Bands, Buy & Hold

### 5. Portfolio Tracker (`pages/4_Portfolio_Tracker.py`)
- Paper trading functionality with virtual $10,000 starting balance
- Transaction history and performance tracking
- Real-time portfolio valuation

### 6. Utility Classes
- **CryptoDataFetcher**: Handles API calls to CoinGecko
- **TechnicalIndicators**: Calculates various technical analysis indicators
- **CryptoPricePredictor**: Implements Prophet-based ML predictions
- **StrategySimulator**: Executes trading strategy backtests
- **PortfolioManager**: Manages virtual portfolio and transactions with database persistence
- **DatabaseManager**: Handles PostgreSQL operations for data persistence

## Data Flow

1. **Data Acquisition**: CoinGecko API → CryptoDataFetcher → Cached in Streamlit
2. **Technical Analysis**: Raw price data → TechnicalIndicators → Calculated indicators
3. **ML Predictions**: Historical data → Prophet model → Future price predictions
4. **Strategy Simulation**: Historical data + strategy rules → StrategySimulator → Performance metrics
5. **Portfolio Management**: User actions → PortfolioManager → PostgreSQL database (with session state fallback)
6. **Data Persistence**: User data → DatabaseManager → PostgreSQL tables for permanent storage

## External Dependencies

### APIs
- **CoinGecko API**: Primary data source for cryptocurrency prices and market data
- **No authentication required**: Uses free tier with rate limiting considerations

### Python Libraries
- **Streamlit**: Web framework and UI components
- **Plotly**: Interactive charting and visualization
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations
- **Prophet**: Time series forecasting
- **TA-Lib (ta)**: Technical analysis indicators
- **Requests**: HTTP client for API calls
- **Scikit-learn**: ML metrics and utilities

## Deployment Strategy

### Development Environment
- **Local Development**: Direct Streamlit execution (`streamlit run app.py`)
- **File Structure**: Multi-page app with utils modules
- **State Management**: Session-based storage (resets on page reload)

### Production Considerations
- **Hosting**: Streamlit Community Cloud or similar platform
- **Performance**: Caching strategies implemented for API calls and computations
- **Scalability**: Database-backed design supports horizontal scaling
- **Data Persistence**: PostgreSQL database provides permanent storage with session state fallback

### Potential Enhancements
1. **User Authentication**: Multi-user support with individual portfolios
2. **Real Trading Integration**: Connection to actual cryptocurrency exchanges
3. **Advanced ML Models**: LSTM networks for more sophisticated predictions
4. **Risk Management**: Position sizing and stop-loss mechanisms
5. **Advanced Analytics**: Performance attribution and risk metrics

## Technical Limitations

1. **Data Source**: Dependent on CoinGecko API availability and rate limits
2. **ML Model**: Prophet may not capture all market dynamics
3. **Real-time Updates**: Manual refresh required for latest data
4. **Limited Coins**: Currently supports 6-8 major cryptocurrencies
5. **Database Fallback**: Session state used when PostgreSQL is unavailable

The architecture prioritizes simplicity and rapid development while providing a solid foundation for a comprehensive crypto trading analysis platform.