import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class TradingStrategies:
    """
    Advanced trading strategy implementation and backtesting engine.
    Supports multiple strategies with transaction costs and slippage modeling.
    """
    
    def __init__(self):
        self.strategies = {
            'SMA Crossover': self.sma_crossover_strategy,
            'RSI Mean Reversion': self.rsi_mean_reversion_strategy,
            'MACD Momentum': self.macd_momentum_strategy,
            'Bollinger Bands': self.bollinger_bands_strategy,
            'Momentum Trading': self.momentum_trading_strategy,
            'Mean Reversion': self.mean_reversion_strategy,
            'Breakout Strategy': self.breakout_strategy,
            'Scalping Strategy': self.scalping_strategy,
            'Arbitrage Scanner': self.arbitrage_strategy,
            'Buy & Hold': self.buy_hold_strategy
        }
    
    def backtest_strategy(self, strategy_type: str, data: pd.DataFrame, indicators: Dict,
                         initial_capital: float = 10000, strategy_params: Dict = None,
                         stop_loss: float = None, take_profit: float = None,
                         trading_fee: float = 0.001, slippage: float = 0.0005,
                         max_position_size: float = 1.0) -> Dict:
        """
        Comprehensive strategy backtesting with transaction costs and risk management.
        
        Args:
            strategy_type: Name of the strategy to backtest
            data: Historical price data
            indicators: Pre-calculated technical indicators
            initial_capital: Starting capital
            strategy_params: Strategy-specific parameters
            stop_loss: Stop loss percentage (e.g., 0.05 for 5%)
            take_profit: Take profit percentage
            trading_fee: Trading fee percentage
            slippage: Slippage percentage
            max_position_size: Maximum position size as fraction of capital
            
        Returns:
            Dictionary with backtest results
        """
        if strategy_type not in self.strategies:
            raise ValueError(f"Unknown strategy: {strategy_type}")
        
        if data.empty or 'price' not in data.columns:
            raise ValueError("Invalid data format")
        
        # Initialize backtest variables
        capital = initial_capital
        position = 0  # Current position size
        position_value = 0
        entry_price = 0
        portfolio_values = [initial_capital]
        returns = [0]
        trades = []
        signals = []
        
        # Get strategy signals
        strategy_func = self.strategies[strategy_type]
        strategy_signals = strategy_func(data, indicators, strategy_params or {})
        
        # Process each data point
        for i in range(1, len(data)):
            current_price = data['price'].iloc[i]
            current_date = data['timestamp'].iloc[i] if 'timestamp' in data.columns else i
            
            # Get signal for current bar
            signal = strategy_signals.get(i, 'HOLD')
            signals.append(signal)
            
            # Calculate current portfolio value
            current_portfolio_value = capital + (position * current_price)
            
            # Check stop loss and take profit
            if position > 0 and entry_price > 0:  # Long position
                if stop_loss and current_price <= entry_price * (1 - stop_loss):
                    signal = 'SELL'  # Stop loss triggered
                elif take_profit and current_price >= entry_price * (1 + take_profit):
                    signal = 'SELL'  # Take profit triggered
            elif position < 0 and entry_price > 0:  # Short position
                if stop_loss and current_price >= entry_price * (1 + stop_loss):
                    signal = 'COVER'  # Stop loss triggered
                elif take_profit and current_price <= entry_price * (1 - take_profit):
                    signal = 'COVER'  # Take profit triggered
            
            # Execute trades based on signals
            if signal == 'BUY' and position <= 0:
                # Calculate position size
                available_capital = capital if position == 0 else capital + (position * current_price)
                max_position_value = available_capital * max_position_size
                
                # Close short position if any
                if position < 0:
                    trade_value = abs(position) * current_price
                    total_cost = trade_value * (1 + trading_fee + slippage)
                    capital -= total_cost
                    
                    trades.append({
                        'date': current_date,
                        'type': 'cover',
                        'price': current_price * (1 + slippage),
                        'quantity': abs(position),
                        'value': trade_value,
                        'cost': total_cost - trade_value,
                        'capital_after': capital
                    })
                    position = 0
                
                # Open long position
                if capital > 0:
                    position_size = min(max_position_value / current_price, capital / current_price)
                    trade_value = position_size * current_price
                    total_cost = trade_value * (1 + trading_fee + slippage)
                    
                    if total_cost <= capital:
                        capital -= total_cost
                        position = position_size
                        entry_price = current_price * (1 + slippage)
                        
                        trades.append({
                            'date': current_date,
                            'type': 'buy',
                            'price': entry_price,
                            'quantity': position_size,
                            'value': trade_value,
                            'cost': total_cost - trade_value,
                            'capital_after': capital
                        })
            
            elif signal == 'SELL' and position >= 0:
                # Close long position
                if position > 0:
                    trade_value = position * current_price
                    total_proceeds = trade_value * (1 - trading_fee - slippage)
                    capital += total_proceeds
                    
                    trades.append({
                        'date': current_date,
                        'type': 'sell',
                        'price': current_price * (1 - slippage),
                        'quantity': position,
                        'value': trade_value,
                        'cost': trade_value - total_proceeds,
                        'capital_after': capital
                    })
                    position = 0
                    entry_price = 0
            
            elif signal == 'SHORT' and position >= 0:
                # Close long position and open short
                if position > 0:
                    trade_value = position * current_price
                    total_proceeds = trade_value * (1 - trading_fee - slippage)
                    capital += total_proceeds
                    position = 0
                
                # Open short position (simplified - in reality more complex)
                available_capital = capital
                max_position_value = available_capital * max_position_size
                position_size = max_position_value / current_price
                
                position = -position_size
                entry_price = current_price * (1 - slippage)
                
                trades.append({
                    'date': current_date,
                    'type': 'short',
                    'price': entry_price,
                    'quantity': position_size,
                    'value': position_size * current_price,
                    'cost': (position_size * current_price) * trading_fee,
                    'capital_after': capital
                })
            
            # Update portfolio value and returns
            current_portfolio_value = capital + (position * current_price)
            portfolio_values.append(current_portfolio_value)
            
            # Calculate return
            if portfolio_values[-2] > 0:
                daily_return = (current_portfolio_value - portfolio_values[-2]) / portfolio_values[-2]
            else:
                daily_return = 0
            returns.append(daily_return)
        
        # Close any remaining position at the end
        if position != 0:
            final_price = data['price'].iloc[-1]
            if position > 0:
                final_value = position * final_price * (1 - trading_fee - slippage)
                capital += final_value
            else:  # Short position
                final_cost = abs(position) * final_price * (1 + trading_fee + slippage)
                capital -= final_cost
            
            trades.append({
                'date': data['timestamp'].iloc[-1] if 'timestamp' in data.columns else len(data) - 1,
                'type': 'close',
                'price': final_price,
                'quantity': abs(position),
                'value': abs(position) * final_price,
                'cost': abs(position) * final_price * trading_fee,
                'capital_after': capital
            })
        
        # Calculate performance metrics
        final_value = capital + (position * data['price'].iloc[-1])
        total_return = (final_value - initial_capital) / initial_capital
        
        # Calculate trade statistics
        winning_trades = [t for t in trades if t['type'] in ['sell', 'cover'] and 
                         self._calculate_trade_pnl(t, trades) > 0]
        losing_trades = [t for t in trades if t['type'] in ['sell', 'cover'] and 
                        self._calculate_trade_pnl(t, trades) < 0]
        
        return {
            'portfolio_values': portfolio_values,
            'returns': returns[1:],  # Exclude first zero return
            'trades': trades,
            'signals': signals,
            'total_trades': len([t for t in trades if t['type'] in ['buy', 'short']]),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(winning_trades) / max(len(winning_trades) + len(losing_trades), 1),
            'total_return': total_return,
            'final_capital': final_value,
            'max_drawdown': self._calculate_max_drawdown(portfolio_values),
            'sharpe_ratio': self._calculate_sharpe_ratio(returns[1:]),
            'avg_win': np.mean([self._calculate_trade_pnl(t, trades) for t in winning_trades]) if winning_trades else 0,
            'avg_loss': np.mean([self._calculate_trade_pnl(t, trades) for t in losing_trades]) if losing_trades else 0,
            'largest_win': max([self._calculate_trade_pnl(t, trades) for t in winning_trades]) if winning_trades else 0,
            'largest_loss': min([self._calculate_trade_pnl(t, trades) for t in losing_trades]) if losing_trades else 0,
            'dates': data['timestamp'].tolist() if 'timestamp' in data.columns else list(range(len(data)))
        }
    
    def _calculate_trade_pnl(self, exit_trade: Dict, all_trades: List[Dict]) -> float:
        """Calculate P&L for a completed trade"""
        # Simplified P&L calculation
        # In practice, you'd need to match entry and exit trades
        return 0  # Placeholder
    
    def _calculate_max_drawdown(self, portfolio_values: List[float]) -> float:
        """Calculate maximum drawdown"""
        if not portfolio_values:
            return 0
        
        peak = portfolio_values[0]
        max_dd = 0
        
        for value in portfolio_values:
            if value > peak:
                peak = value
            
            drawdown = (peak - value) / peak
            max_dd = max(max_dd, drawdown)
        
        return max_dd
    
    def _calculate_sharpe_ratio(self, returns: List[float], risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        if not returns or len(returns) < 2:
            return 0
        
        returns_array = np.array(returns)
        excess_returns = returns_array - (risk_free_rate / 252)  # Daily risk-free rate
        
        if np.std(excess_returns) == 0:
            return 0
        
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
    
    # Strategy implementations
    
    def sma_crossover_strategy(self, data: pd.DataFrame, indicators: Dict, params: Dict) -> Dict[int, str]:
        """Simple Moving Average Crossover Strategy"""
        signals = {}
        short_window = params.get('short_window', 20)
        long_window = params.get('long_window', 50)
        
        if f'sma_{short_window}' not in indicators or f'sma_{long_window}' not in indicators:
            return signals
        
        short_ma = indicators[f'sma_{short_window}']
        long_ma = indicators[f'sma_{long_window}']
        
        for i in range(1, len(data)):
            if i >= long_window and not (np.isnan(short_ma.iloc[i]) or np.isnan(long_ma.iloc[i])):
                # Current values
                short_current = short_ma.iloc[i]
                long_current = long_ma.iloc[i]
                
                # Previous values
                short_prev = short_ma.iloc[i-1]
                long_prev = long_ma.iloc[i-1]
                
                # Golden cross (bullish signal)
                if short_prev <= long_prev and short_current > long_current:
                    signals[i] = 'BUY'
                # Death cross (bearish signal)
                elif short_prev >= long_prev and short_current < long_current:
                    signals[i] = 'SELL'
                else:
                    signals[i] = 'HOLD'
        
        return signals
    
    def rsi_mean_reversion_strategy(self, data: pd.DataFrame, indicators: Dict, params: Dict) -> Dict[int, str]:
        """RSI Mean Reversion Strategy"""
        signals = {}
        rsi_period = params.get('rsi_period', 14)
        oversold = params.get('oversold', 30)
        overbought = params.get('overbought', 70)
        
        if 'rsi' not in indicators:
            return signals
        
        rsi = indicators['rsi']
        
        for i in range(rsi_period, len(data)):
            if not np.isnan(rsi.iloc[i]):
                rsi_current = rsi.iloc[i]
                
                if rsi_current < oversold:
                    signals[i] = 'BUY'
                elif rsi_current > overbought:
                    signals[i] = 'SELL'
                else:
                    signals[i] = 'HOLD'
        
        return signals
    
    def macd_momentum_strategy(self, data: pd.DataFrame, indicators: Dict, params: Dict) -> Dict[int, str]:
        """MACD Momentum Strategy"""
        signals = {}
        
        if 'macd' not in indicators or 'macd_signal' not in indicators:
            return signals
        
        macd = indicators['macd']
        macd_signal = indicators['macd_signal']
        
        for i in range(1, len(data)):
            if not (np.isnan(macd.iloc[i]) or np.isnan(macd_signal.iloc[i]) or 
                    np.isnan(macd.iloc[i-1]) or np.isnan(macd_signal.iloc[i-1])):
                
                # Current values
                macd_current = macd.iloc[i]
                signal_current = macd_signal.iloc[i]
                
                # Previous values
                macd_prev = macd.iloc[i-1]
                signal_prev = macd_signal.iloc[i-1]
                
                # Bullish crossover
                if macd_prev <= signal_prev and macd_current > signal_current:
                    signals[i] = 'BUY'
                # Bearish crossover
                elif macd_prev >= signal_prev and macd_current < signal_current:
                    signals[i] = 'SELL'
                else:
                    signals[i] = 'HOLD'
        
        return signals
    
    def bollinger_bands_strategy(self, data: pd.DataFrame, indicators: Dict, params: Dict) -> Dict[int, str]:
        """Bollinger Bands Mean Reversion Strategy"""
        signals = {}
        
        if not all(k in indicators for k in ['bb_upper', 'bb_lower', 'bb_middle']):
            return signals
        
        bb_upper = indicators['bb_upper']
        bb_lower = indicators['bb_lower']
        bb_middle = indicators['bb_middle']
        
        for i in range(len(data)):
            if not any(np.isnan([bb_upper.iloc[i], bb_lower.iloc[i], bb_middle.iloc[i]])):
                price = data['price'].iloc[i]
                
                # Price touches lower band - buy signal
                if price <= bb_lower.iloc[i]:
                    signals[i] = 'BUY'
                # Price touches upper band - sell signal
                elif price >= bb_upper.iloc[i]:
                    signals[i] = 'SELL'
                # Price crosses middle line
                elif i > 0:
                    prev_price = data['price'].iloc[i-1]
                    if prev_price <= bb_middle.iloc[i-1] and price > bb_middle.iloc[i]:
                        signals[i] = 'BUY'
                    elif prev_price >= bb_middle.iloc[i-1] and price < bb_middle.iloc[i]:
                        signals[i] = 'SELL'
                    else:
                        signals[i] = 'HOLD'
                else:
                    signals[i] = 'HOLD'
        
        return signals
    
    def momentum_trading_strategy(self, data: pd.DataFrame, indicators: Dict, params: Dict) -> Dict[int, str]:
        """Momentum Trading Strategy"""
        signals = {}
        period = params.get('period', 10)
        threshold = params.get('threshold', 0.02)  # 2% threshold
        
        if 'momentum' not in indicators:
            # Calculate momentum if not available
            momentum = np.full(len(data), np.nan)
            for i in range(period, len(data)):
                momentum[i] = (data['price'].iloc[i] - data['price'].iloc[i-period]) / data['price'].iloc[i-period]
        else:
            momentum = indicators['momentum']
        
        for i in range(period, len(data)):
            if not np.isnan(momentum[i]):
                mom_value = momentum[i] if isinstance(momentum, np.ndarray) else momentum.iloc[i]
                
                if mom_value > threshold:
                    signals[i] = 'BUY'
                elif mom_value < -threshold:
                    signals[i] = 'SELL'
                else:
                    signals[i] = 'HOLD'
        
        return signals
    
    def mean_reversion_strategy(self, data: pd.DataFrame, indicators: Dict, params: Dict) -> Dict[int, str]:
        """Mean Reversion Strategy using price deviations from moving average"""
        signals = {}
        ma_period = params.get('ma_period', 20)
        std_threshold = params.get('std_threshold', 2)
        
        # Calculate moving average and standard deviation
        ma = data['price'].rolling(window=ma_period).mean()
        std = data['price'].rolling(window=ma_period).std()
        
        for i in range(ma_period, len(data)):
            if not (np.isnan(ma.iloc[i]) or np.isnan(std.iloc[i])):
                price = data['price'].iloc[i]
                deviation = (price - ma.iloc[i]) / std.iloc[i]
                
                # Price is significantly below mean - buy
                if deviation < -std_threshold:
                    signals[i] = 'BUY'
                # Price is significantly above mean - sell
                elif deviation > std_threshold:
                    signals[i] = 'SELL'
                else:
                    signals[i] = 'HOLD'
        
        return signals
    
    def breakout_strategy(self, data: pd.DataFrame, indicators: Dict, params: Dict) -> Dict[int, str]:
        """Breakout Strategy using price channels"""
        signals = {}
        period = params.get('period', 20)
        
        # Calculate price channels
        high_channel = data['price'].rolling(window=period).max()
        low_channel = data['price'].rolling(window=period).min()
        
        for i in range(period, len(data)):
            if not (np.isnan(high_channel.iloc[i-1]) or np.isnan(low_channel.iloc[i-1])):
                price = data['price'].iloc[i]
                prev_high = high_channel.iloc[i-1]
                prev_low = low_channel.iloc[i-1]
                
                # Upward breakout
                if price > prev_high:
                    signals[i] = 'BUY'
                # Downward breakout
                elif price < prev_low:
                    signals[i] = 'SELL'
                else:
                    signals[i] = 'HOLD'
        
        return signals
    
    def scalping_strategy(self, data: pd.DataFrame, indicators: Dict, params: Dict) -> Dict[int, str]:
        """High-frequency scalping strategy"""
        signals = {}
        threshold = params.get('threshold', 0.005)  # 0.5% threshold
        
        # Use short-term price movements
        for i in range(1, len(data)):
            price_change = (data['price'].iloc[i] - data['price'].iloc[i-1]) / data['price'].iloc[i-1]
            
            # Quick momentum signals
            if price_change > threshold:
                signals[i] = 'BUY'
            elif price_change < -threshold:
                signals[i] = 'SELL'
            else:
                signals[i] = 'HOLD'
        
        return signals
    
    def arbitrage_strategy(self, data: pd.DataFrame, indicators: Dict, params: Dict) -> Dict[int, str]:
        """Arbitrage opportunity scanner (simplified)"""
        signals = {}
        
        # This is a simplified arbitrage strategy
        # In practice, you'd compare prices across multiple exchanges
        spread_threshold = params.get('spread_threshold', 0.01)  # 1% spread
        
        if 'bb_upper' in indicators and 'bb_lower' in indicators:
            bb_upper = indicators['bb_upper']
            bb_lower = indicators['bb_lower']
            
            for i in range(len(data)):
                if not (np.isnan(bb_upper.iloc[i]) or np.isnan(bb_lower.iloc[i])):
                    price = data['price'].iloc[i]
                    spread = (bb_upper.iloc[i] - bb_lower.iloc[i]) / price
                    
                    # High spread indicates arbitrage opportunity
                    if spread > spread_threshold:
                        if price <= bb_lower.iloc[i]:
                            signals[i] = 'BUY'
                        elif price >= bb_upper.iloc[i]:
                            signals[i] = 'SELL'
                        else:
                            signals[i] = 'HOLD'
                    else:
                        signals[i] = 'HOLD'
        
        return signals
    
    def buy_hold_strategy(self, data: pd.DataFrame, indicators: Dict, params: Dict) -> Dict[int, str]:
        """Simple buy and hold strategy for benchmarking"""
        signals = {}
        
        # Buy at the beginning, hold until the end
        for i in range(len(data)):
            if i == 0:
                signals[i] = 'BUY'
            else:
                signals[i] = 'HOLD'
        
        return signals
    
    def optimize_strategy_parameters(self, strategy_type: str, data: pd.DataFrame, 
                                   indicators: Dict, param_ranges: Dict) -> Dict:
        """
        Optimize strategy parameters using grid search.
        
        Args:
            strategy_type: Strategy to optimize
            data: Historical data
            indicators: Technical indicators
            param_ranges: Dictionary of parameter ranges to test
            
        Returns:
            Best parameters and their performance
        """
        best_params = None
        best_performance = -np.inf
        results = []
        
        # Generate parameter combinations
        param_combinations = self._generate_param_combinations(param_ranges)
        
        for params in param_combinations:
            try:
                # Run backtest with these parameters
                result = self.backtest_strategy(
                    strategy_type=strategy_type,
                    data=data,
                    indicators=indicators,
                    strategy_params=params
                )
                
                # Use Sharpe ratio as optimization metric
                performance = result.get('sharpe_ratio', -np.inf)
                
                results.append({
                    'params': params,
                    'performance': performance,
                    'total_return': result.get('total_return', 0),
                    'max_drawdown': result.get('max_drawdown', 1),
                    'win_rate': result.get('win_rate', 0)
                })
                
                if performance > best_performance:
                    best_performance = performance
                    best_params = params
                    
            except Exception as e:
                # Skip this parameter combination if it fails
                continue
        
        return {
            'best_params': best_params,
            'best_performance': best_performance,
            'all_results': results
        }
    
    def _generate_param_combinations(self, param_ranges: Dict) -> List[Dict]:
        """Generate all combinations of parameters for optimization"""
        import itertools
        
        keys = list(param_ranges.keys())
        values = list(param_ranges.values())
        
        combinations = []
        for combo in itertools.product(*values):
            combinations.append(dict(zip(keys, combo)))
        
        return combinations
    
    def get_strategy_description(self, strategy_type: str) -> str:
        """Get description of a trading strategy"""
        descriptions = {
            'SMA Crossover': 'Classic trend-following strategy using moving average crossovers for entry/exit signals.',
            'RSI Mean Reversion': 'Contrarian strategy buying oversold and selling overbought conditions based on RSI.',
            'MACD Momentum': 'Momentum strategy using MACD line crossovers and signal line divergences.',
            'Bollinger Bands': 'Range-bound strategy trading bounces off Bollinger Band boundaries.',
            'Momentum Trading': 'Trend-following strategy based on price momentum and rate of change.',
            'Mean Reversion': 'Statistical arbitrage strategy exploiting price deviations from mean.',
            'Breakout Strategy': 'Momentum strategy capturing moves beyond key support/resistance levels.',
            'Scalping Strategy': 'High-frequency strategy capturing small price movements.',
            'Arbitrage Scanner': 'Market inefficiency strategy exploiting price differences.',
            'Buy & Hold': 'Passive investment strategy for benchmark comparison.'
        }
        
        return descriptions.get(strategy_type, 'No description available.')
    
    def get_default_parameters(self, strategy_type: str) -> Dict:
        """Get default parameters for a strategy"""
        defaults = {
            'SMA Crossover': {'short_window': 20, 'long_window': 50},
            'RSI Mean Reversion': {'rsi_period': 14, 'oversold': 30, 'overbought': 70},
            'MACD Momentum': {'fast': 12, 'slow': 26, 'signal': 9},
            'Bollinger Bands': {'period': 20, 'std_dev': 2},
            'Momentum Trading': {'period': 10, 'threshold': 0.02},
            'Mean Reversion': {'ma_period': 20, 'std_threshold': 2},
            'Breakout Strategy': {'period': 20},
            'Scalping Strategy': {'threshold': 0.005},
            'Arbitrage Scanner': {'spread_threshold': 0.01},
            'Buy & Hold': {}
        }
        
        return defaults.get(strategy_type, {})
