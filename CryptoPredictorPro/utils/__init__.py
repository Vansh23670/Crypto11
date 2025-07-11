"""
CryptoPredictorPro Utilities Package

This package contains utility modules for the advanced cryptocurrency trading dashboard.
All modules provide comprehensive functionality for data fetching, ML modeling, 
technical analysis, and portfolio management.
"""

__version__ = "1.0.0"
__author__ = "CryptoPredictorPro Team"

# Import main utility classes for easy access
from .data_fetcher import CryptoDataFetcher
from .portfolio_manager import PortfolioManager
from .ml_models import EnsemblePredictor, ProphetPredictor, LSTMPredictor, XGBoostPredictor
from .technical_indicators import TechnicalIndicators
from .trading_strategies import TradingStrategies
from .risk_manager import RiskManager
from .performance_metrics import PerformanceMetrics
from .sentiment_analyzer import SentimentAnalyzer
from .database_manager import DatabaseManager
from .websocket_client import WebSocketClient

__all__ = [
    'CryptoDataFetcher',
    'PortfolioManager', 
    'EnsemblePredictor',
    'ProphetPredictor',
    'LSTMPredictor',
    'XGBoostPredictor',
    'TechnicalIndicators',
    'TradingStrategies',
    'RiskManager',
    'PerformanceMetrics',
    'SentimentAnalyzer',
    'DatabaseManager',
    'WebSocketClient'
]
