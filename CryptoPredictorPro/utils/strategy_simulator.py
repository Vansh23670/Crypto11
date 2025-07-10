import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st

class StrategySimulator:
    def __init__(self):
        self.strategies = {
            'SMA Crossover': self.sma_crossover_strategy,
            'RSI Oversold/Overbought': self.rsi_strategy,
            'MACD Signal': self.macd_strategy,
            'Bollinger Bands': self.bollinger_strategy,
            'Buy and Hold': self.buy_and_hold_strategy
        }
        
    def simulate_strategy(self, df, strategy_name, initial_capital=10000, **kwargs):
        """
        Simulate a trading strategy on historical data
        """
        if df.empty or strategy_name not in self.strategies:
            return {
                'trades': [],
                'portfolio_value': [initial_capital],
                'returns': 0,
                'total_trades': 0,
                'win_rate': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0
            }
        
        try:
            # Run the strategy
            strategy_func = self.strategies[strategy_name]
            result = strategy_func(df, initial_capital, **kwargs)
            
            # Calculate performance metrics
            result = self.calculate_performance_metrics(result, df)
            
            return result
            
        except Exception as e:
            st.error(f"Error simulating strategy: {str(e)}")
            return {
                'trades': [],
                'portfolio_value': [initial_capital],
                'returns': 0,
                'total_trades': 0,
                'win_rate': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0
            }
    
    def sma_crossover_strategy(self, df, initial_capital=10000, short_window=10, long_window=20):
        """
        Simple Moving Average Crossover Strategy
        """
        df = df.copy()
        df['SMA_short'] = df['price'].rolling(window=short_window).mean()
        df['SMA_long'] = df['price'].rolling(window=long_window).mean()
        
        # Generate signals
        df['signal'] = 0
        df['signal'][short_window:] = np.where(
            df['SMA_short'][short_window:] > df['SMA_long'][short_window:], 1, 0
        )
        df['position'] = df['signal'].diff()
        
        return self.execute_trades(df, initial_capital)
    
    def rsi_strategy(self, df, initial_capital=10000, rsi_oversold=30, rsi_overbought=70):
        """
        RSI Overbought/Oversold Strategy
        """
        df = df.copy()
        
        # Calculate RSI if not present
        if 'RSI' not in df.columns:
            from utils.technical_indicators import TechnicalIndicators
            ti = TechnicalIndicators()
            df = ti.calculate_rsi(df)
        
        # Generate signals
        df['signal'] = 0
        df['signal'] = np.where(df['RSI'] < rsi_oversold, 1, 0)  # Buy when oversold
        df['signal'] = np.where(df['RSI'] > rsi_overbought, -1, df['signal'])  # Sell when overbought
        
        # Convert to position changes
        df['position'] = df['signal'].diff()
        
        return self.execute_trades(df, initial_capital)
    
    def macd_strategy(self, df, initial_capital=10000):
        """
        MACD Signal Strategy
        """
        df = df.copy()
        
        # Calculate MACD if not present
        if 'MACD' not in df.columns:
            from utils.technical_indicators import TechnicalIndicators
            ti = TechnicalIndicators()
            df = ti.calculate_macd(df)
        
        # Generate signals
        df['signal'] = 0
        df['signal'] = np.where(df['MACD'] > df['MACD_signal'], 1, 0)
        df['position'] = df['signal'].diff()
        
        return self.execute_trades(df, initial_capital)
    
    def bollinger_strategy(self, df, initial_capital=10000):
        """
        Bollinger Bands Strategy
        """
        df = df.copy()
        
        # Calculate Bollinger Bands if not present
        if 'BB_upper' not in df.columns:
            from utils.technical_indicators import TechnicalIndicators
            ti = TechnicalIndicators()
            df = ti.calculate_bollinger_bands(df)
        
        # Generate signals
        df['signal'] = 0
        df['signal'] = np.where(df['price'] <= df['BB_lower'], 1, 0)  # Buy at lower band
        df['signal'] = np.where(df['price'] >= df['BB_upper'], -1, df['signal'])  # Sell at upper band
        
        df['position'] = df['signal'].diff()
        
        return self.execute_trades(df, initial_capital)
    
    def buy_and_hold_strategy(self, df, initial_capital=10000):
        """
        Buy and Hold Strategy
        """
        df = df.copy()
        
        # Buy at the beginning, hold till the end
        df['signal'] = 1
        df['position'] = 0
        df.iloc[0, df.columns.get_loc('position')] = 1  # Buy on first day
        
        return self.execute_trades(df, initial_capital)
    
    def execute_trades(self, df, initial_capital):
        """
        Execute trades based on signals and calculate portfolio value
        """
        portfolio_value = initial_capital
        cash = initial_capital
        shares = 0
        trades = []
        portfolio_values = []
        
        for i, row in df.iterrows():
            if pd.isna(row['position']) or row['position'] == 0:
                # No trade
                portfolio_value = cash + shares * row['price']
                portfolio_values.append(portfolio_value)
                continue
            
            if row['position'] == 1:  # Buy signal
                if cash > 0:
                    shares_to_buy = cash / row['price']
                    shares += shares_to_buy
                    cash = 0
                    
                    trades.append({
                        'date': row['timestamp'],
                        'type': 'BUY',
                        'price': row['price'],
                        'shares': shares_to_buy,
                        'value': shares_to_buy * row['price']
                    })
            
            elif row['position'] == -1:  # Sell signal
                if shares > 0:
                    cash = shares * row['price']
                    
                    trades.append({
                        'date': row['timestamp'],
                        'type': 'SELL',
                        'price': row['price'],
                        'shares': shares,
                        'value': cash
                    })
                    
                    shares = 0
            
            portfolio_value = cash + shares * row['price']
            portfolio_values.append(portfolio_value)
        
        return {
            'trades': trades,
            'portfolio_value': portfolio_values,
            'final_cash': cash,
            'final_shares': shares,
            'final_price': df['price'].iloc[-1] if not df.empty else 0
        }
    
    def calculate_performance_metrics(self, result, df):
        """
        Calculate performance metrics for the strategy
        """
        try:
            portfolio_values = result['portfolio_value']
            trades = result['trades']
            
            if not portfolio_values or len(portfolio_values) < 2:
                return {**result, 'returns': 0, 'total_trades': 0, 'win_rate': 0, 'max_drawdown': 0, 'sharpe_ratio': 0}
            
            # Calculate returns
            initial_value = portfolio_values[0]
            final_value = portfolio_values[-1]
            total_return = (final_value - initial_value) / initial_value * 100
            
            # Calculate daily returns
            daily_returns = []
            for i in range(1, len(portfolio_values)):
                daily_return = (portfolio_values[i] - portfolio_values[i-1]) / portfolio_values[i-1]
                daily_returns.append(daily_return)
            
            # Calculate Sharpe ratio (assuming 2% risk-free rate)
            if daily_returns:
                avg_daily_return = np.mean(daily_returns)
                std_daily_return = np.std(daily_returns)
                sharpe_ratio = (avg_daily_return - 0.02/252) / std_daily_return * np.sqrt(252) if std_daily_return > 0 else 0
            else:
                sharpe_ratio = 0
            
            # Calculate maximum drawdown
            running_max = np.maximum.accumulate(portfolio_values)
            drawdown = (np.array(portfolio_values) - running_max) / running_max * 100
            max_drawdown = np.min(drawdown)
            
            # Calculate win rate
            winning_trades = 0
            total_trades = len(trades)
            
            if total_trades > 0:
                buy_prices = {}
                for trade in trades:
                    if trade['type'] == 'BUY':
                        buy_prices[trade['date']] = trade['price']
                    elif trade['type'] == 'SELL':
                        # Find corresponding buy price
                        for buy_date, buy_price in buy_prices.items():
                            if buy_date <= trade['date']:
                                if trade['price'] > buy_price:
                                    winning_trades += 1
                                break
                
                win_rate = winning_trades / (total_trades // 2) * 100 if total_trades > 0 else 0
            else:
                win_rate = 0
            
            return {
                **result,
                'returns': total_return,
                'total_trades': total_trades,
                'win_rate': win_rate,
                'max_drawdown': abs(max_drawdown),
                'sharpe_ratio': sharpe_ratio,
                'daily_returns': daily_returns
            }
            
        except Exception as e:
            st.warning(f"Error calculating performance metrics: {str(e)}")
            return {
                **result,
                'returns': 0,
                'total_trades': 0,
                'win_rate': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0
            }
    
    def get_strategy_comparison(self, df, initial_capital=10000):
        """
        Compare performance of all strategies
        """
        comparison_results = {}
        
        for strategy_name in self.strategies.keys():
            try:
                result = self.simulate_strategy(df, strategy_name, initial_capital)
                comparison_results[strategy_name] = {
                    'returns': result['returns'],
                    'total_trades': result['total_trades'],
                    'win_rate': result['win_rate'],
                    'max_drawdown': result['max_drawdown'],
                    'sharpe_ratio': result['sharpe_ratio'],
                    'final_value': result['portfolio_value'][-1] if result['portfolio_value'] else initial_capital
                }
            except Exception as e:
                st.warning(f"Error running {strategy_name}: {str(e)}")
                comparison_results[strategy_name] = {
                    'returns': 0,
                    'total_trades': 0,
                    'win_rate': 0,
                    'max_drawdown': 0,
                    'sharpe_ratio': 0,
                    'final_value': initial_capital
                }
        
        return comparison_results
