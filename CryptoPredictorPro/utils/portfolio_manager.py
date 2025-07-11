import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime, timedelta
import uuid
import json
import streamlit as st
from .database_manager import DatabaseManager

class PortfolioManager:
    """
    Advanced portfolio management system with database persistence and real-time tracking.
    Supports paper trading, transaction history, and performance analytics.
    """
    
    def __init__(self, initial_balance: float = 10000.0):
        self.initial_balance = initial_balance
        self.db_manager = DatabaseManager()
        
        # Initialize session state if not exists
        if 'portfolio_data' not in st.session_state:
            st.session_state.portfolio_data = {
                'cash_balance': initial_balance,
                'holdings': {},  # {symbol: quantity}
                'transactions': [],
                'created_at': datetime.now(),
                'last_updated': datetime.now()
            }
        
        # Try to load from database
        self._load_from_database()
    
    def _load_from_database(self):
        """Load portfolio data from database if available"""
        try:
            if self.db_manager.is_connected():
                # Load portfolio data from database
                portfolio_data = self.db_manager.load_portfolio()
                if portfolio_data:
                    st.session_state.portfolio_data.update(portfolio_data)
        except Exception as e:
            # Database not available, use session state
            pass
    
    def _save_to_database(self):
        """Save portfolio data to database"""
        try:
            if self.db_manager.is_connected():
                self.db_manager.save_portfolio(st.session_state.portfolio_data)
        except Exception as e:
            # Database not available, data remains in session state
            pass
    
    def execute_trade(self, symbol: str, trade_type: str, amount: Union[float, None] = None, 
                     quantity: Union[float, None] = None, price: float = None) -> Dict:
        """
        Execute a trade (buy/sell) and update portfolio.
        
        Args:
            symbol: Cryptocurrency symbol (e.g., 'bitcoin')
            trade_type: 'buy' or 'sell'
            amount: USD amount to trade (for buy orders)
            quantity: Quantity to trade (for sell orders)
            price: Current price per unit
            
        Returns:
            Dictionary with trade execution details
        """
        if not price or price <= 0:
            raise ValueError("Invalid price provided")
        
        trade_type = trade_type.lower()
        if trade_type not in ['buy', 'sell']:
            raise ValueError("Trade type must be 'buy' or 'sell'")
        
        # Generate unique transaction ID
        transaction_id = str(uuid.uuid4())
        timestamp = datetime.now()
        
        if trade_type == 'buy':
            return self._execute_buy_order(symbol, amount, price, transaction_id, timestamp)
        else:
            return self._execute_sell_order(symbol, quantity, price, transaction_id, timestamp)
    
    def _execute_buy_order(self, symbol: str, amount: float, price: float, 
                          transaction_id: str, timestamp: datetime) -> Dict:
        """Execute a buy order"""
        if amount <= 0:
            raise ValueError("Buy amount must be positive")
        
        # Check if sufficient cash
        cash_balance = st.session_state.portfolio_data['cash_balance']
        if amount > cash_balance:
            raise ValueError(f"Insufficient funds. Available: ${cash_balance:.2f}, Required: ${amount:.2f}")
        
        # Calculate quantity (including fees)
        trading_fee = 0.001  # 0.1% trading fee
        fee_amount = amount * trading_fee
        net_amount = amount - fee_amount
        quantity = net_amount / price
        
        # Update holdings
        current_holdings = st.session_state.portfolio_data['holdings']
        current_holdings[symbol] = current_holdings.get(symbol, 0) + quantity
        
        # Update cash balance
        st.session_state.portfolio_data['cash_balance'] -= amount
        
        # Record transaction
        transaction = {
            'id': transaction_id,
            'timestamp': timestamp.isoformat(),
            'symbol': symbol,
            'type': 'buy',
            'quantity': quantity,
            'price': price,
            'amount': amount,
            'fee': fee_amount,
            'cash_balance_after': st.session_state.portfolio_data['cash_balance']
        }
        
        st.session_state.portfolio_data['transactions'].append(transaction)
        st.session_state.portfolio_data['last_updated'] = timestamp
        
        # Save to database
        self._save_to_database()
        
        return {
            'success': True,
            'transaction_id': transaction_id,
            'quantity_purchased': quantity,
            'total_cost': amount,
            'fee': fee_amount,
            'remaining_cash': st.session_state.portfolio_data['cash_balance']
        }
    
    def _execute_sell_order(self, symbol: str, quantity: float, price: float, 
                           transaction_id: str, timestamp: datetime) -> Dict:
        """Execute a sell order"""
        if quantity <= 0:
            raise ValueError("Sell quantity must be positive")
        
        # Check if sufficient holdings
        current_holdings = st.session_state.portfolio_data['holdings']
        available_quantity = current_holdings.get(symbol, 0)
        
        if quantity > available_quantity:
            raise ValueError(f"Insufficient holdings. Available: {available_quantity:.6f}, Required: {quantity:.6f}")
        
        # Calculate proceeds (including fees)
        gross_amount = quantity * price
        trading_fee = 0.001  # 0.1% trading fee
        fee_amount = gross_amount * trading_fee
        net_amount = gross_amount - fee_amount
        
        # Update holdings
        current_holdings[symbol] -= quantity
        if current_holdings[symbol] <= 1e-8:  # Remove dust
            del current_holdings[symbol]
        
        # Update cash balance
        st.session_state.portfolio_data['cash_balance'] += net_amount
        
        # Record transaction
        transaction = {
            'id': transaction_id,
            'timestamp': timestamp.isoformat(),
            'symbol': symbol,
            'type': 'sell',
            'quantity': quantity,
            'price': price,
            'amount': gross_amount,
            'fee': fee_amount,
            'net_proceeds': net_amount,
            'cash_balance_after': st.session_state.portfolio_data['cash_balance']
        }
        
        st.session_state.portfolio_data['transactions'].append(transaction)
        st.session_state.portfolio_data['last_updated'] = timestamp
        
        # Save to database
        self._save_to_database()
        
        return {
            'success': True,
            'transaction_id': transaction_id,
            'quantity_sold': quantity,
            'gross_proceeds': gross_amount,
            'net_proceeds': net_amount,
            'fee': fee_amount,
            'cash_balance': st.session_state.portfolio_data['cash_balance']
        }
    
    def get_portfolio_value(self, data_fetcher=None) -> float:
        """
        Calculate total portfolio value (cash + holdings value).
        
        Args:
            data_fetcher: CryptoDataFetcher instance for current prices
            
        Returns:
            Total portfolio value in USD
        """
        try:
            cash_balance = st.session_state.portfolio_data['cash_balance']
            holdings_value = 0
            
            if data_fetcher:
                holdings = st.session_state.portfolio_data['holdings']
                for symbol, quantity in holdings.items():
                    if quantity > 0:
                        try:
                            current_data = data_fetcher.get_current_price(symbol, 'usd')
                            current_price = current_data['current_price']
                            holdings_value += quantity * current_price
                        except Exception:
                            # If price fetch fails, skip this holding
                            continue
            
            return cash_balance + holdings_value
            
        except Exception:
            return self.initial_balance
    
    def get_cash_balance(self) -> float:
        """Get current cash balance"""
        return st.session_state.portfolio_data.get('cash_balance', self.initial_balance)
    
    def get_holdings(self) -> Dict[str, float]:
        """Get current cryptocurrency holdings"""
        return st.session_state.portfolio_data.get('holdings', {}).copy()
    
    def get_total_invested(self) -> float:
        """Calculate total amount invested (excluding current cash)"""
        transactions = st.session_state.portfolio_data.get('transactions', [])
        total_invested = 0
        
        for transaction in transactions:
            if transaction['type'] == 'buy':
                total_invested += transaction['amount']
            elif transaction['type'] == 'sell':
                total_invested -= transaction['net_proceeds']
        
        return max(0, total_invested)
    
    def get_total_pnl(self, data_fetcher=None) -> float:
        """
        Calculate total profit/loss percentage.
        
        Args:
            data_fetcher: CryptoDataFetcher instance for current prices
            
        Returns:
            P&L percentage
        """
        try:
            current_value = self.get_portfolio_value(data_fetcher)
            return ((current_value - self.initial_balance) / self.initial_balance) * 100
        except Exception:
            return 0.0
    
    def get_average_cost(self, symbol: str) -> float:
        """
        Calculate average cost basis for a specific cryptocurrency.
        
        Args:
            symbol: Cryptocurrency symbol
            
        Returns:
            Average cost per unit
        """
        transactions = st.session_state.portfolio_data.get('transactions', [])
        total_cost = 0
        total_quantity = 0
        
        for transaction in transactions:
            if transaction['symbol'] == symbol:
                if transaction['type'] == 'buy':
                    total_cost += transaction['amount']
                    total_quantity += transaction['quantity']
                elif transaction['type'] == 'sell':
                    # Reduce cost basis proportionally
                    if total_quantity > 0:
                        avg_cost = total_cost / total_quantity
                        reduction = transaction['quantity'] * avg_cost
                        total_cost -= reduction
                        total_quantity -= transaction['quantity']
        
        return total_cost / total_quantity if total_quantity > 0 else 0
    
    def get_recent_trades(self, limit: int = 10) -> List[Dict]:
        """
        Get recent trading transactions.
        
        Args:
            limit: Maximum number of trades to return
            
        Returns:
            List of recent transactions
        """
        transactions = st.session_state.portfolio_data.get('transactions', [])
        recent_transactions = sorted(transactions, key=lambda x: x['timestamp'], reverse=True)[:limit]
        
        # Format for display
        formatted_trades = []
        for trade in recent_transactions:
            formatted_trades.append({
                'timestamp': trade['timestamp'],
                'symbol': trade['symbol'].upper(),
                'type': trade['type'].upper(),
                'quantity': f"{trade['quantity']:.6f}",
                'price': f"${trade['price']:.4f}",
                'amount': f"${trade['amount']:.2f}",
                'fee': f"${trade['fee']:.2f}"
            })
        
        return formatted_trades
    
    def get_all_trades(self) -> List[Dict]:
        """Get all trading transactions"""
        return st.session_state.portfolio_data.get('transactions', [])
    
    def get_portfolio_history(self, days: int = 30) -> List[Dict]:
        """
        Get portfolio value history over time.
        
        Args:
            days: Number of days of history to return
            
        Returns:
            List of portfolio value snapshots
        """
        try:
            # Try to get from database
            if self.db_manager.is_connected():
                history = self.db_manager.get_portfolio_history(days)
                if history:
                    return history
        except Exception:
            pass
        
        # Fallback: simulate history based on transactions
        return self._simulate_portfolio_history(days)
    
    def _simulate_portfolio_history(self, days: int) -> List[Dict]:
        """Simulate portfolio history based on transaction data"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Simple simulation - just return current value for each day
        current_value = self.get_portfolio_value()
        history = []
        
        for i in range(days):
            date = start_date + timedelta(days=i)
            # Add some random variation for simulation
            variation = np.random.normal(1, 0.02)  # 2% daily volatility
            simulated_value = current_value * variation
            
            history.append({
                'date': date.isoformat(),
                'total_value': simulated_value,
                'cash_balance': st.session_state.portfolio_data['cash_balance'],
                'holdings_value': simulated_value - st.session_state.portfolio_data['cash_balance']
            })
        
        return history
    
    def get_position_details(self, symbol: str, data_fetcher=None) -> Dict:
        """
        Get detailed information about a specific position.
        
        Args:
            symbol: Cryptocurrency symbol
            data_fetcher: CryptoDataFetcher instance for current prices
            
        Returns:
            Dictionary with position details
        """
        holdings = st.session_state.portfolio_data.get('holdings', {})
        quantity = holdings.get(symbol, 0)
        
        if quantity == 0:
            return {'symbol': symbol, 'quantity': 0, 'value': 0, 'pnl': 0, 'pnl_pct': 0}
        
        avg_cost = self.get_average_cost(symbol)
        current_price = 0
        current_value = 0
        
        if data_fetcher:
            try:
                current_data = data_fetcher.get_current_price(symbol, 'usd')
                current_price = current_data['current_price']
                current_value = quantity * current_price
            except Exception:
                pass
        
        # Calculate P&L
        cost_basis = quantity * avg_cost
        unrealized_pnl = current_value - cost_basis
        unrealized_pnl_pct = (unrealized_pnl / cost_basis * 100) if cost_basis > 0 else 0
        
        return {
            'symbol': symbol,
            'quantity': quantity,
            'avg_cost': avg_cost,
            'current_price': current_price,
            'current_value': current_value,
            'cost_basis': cost_basis,
            'unrealized_pnl': unrealized_pnl,
            'unrealized_pnl_pct': unrealized_pnl_pct
        }
    
    def get_portfolio_allocation(self, data_fetcher=None) -> Dict[str, float]:
        """
        Get portfolio allocation by asset.
        
        Args:
            data_fetcher: CryptoDataFetcher instance for current prices
            
        Returns:
            Dictionary with asset allocations (percentages)
        """
        total_value = self.get_portfolio_value(data_fetcher)
        if total_value == 0:
            return {}
        
        allocation = {}
        holdings = st.session_state.portfolio_data.get('holdings', {})
        cash_balance = st.session_state.portfolio_data['cash_balance']
        
        # Cash allocation
        allocation['CASH'] = (cash_balance / total_value) * 100
        
        # Crypto allocations
        if data_fetcher:
            for symbol, quantity in holdings.items():
                if quantity > 0:
                    try:
                        current_data = data_fetcher.get_current_price(symbol, 'usd')
                        current_price = current_data['current_price']
                        position_value = quantity * current_price
                        allocation[symbol.upper()] = (position_value / total_value) * 100
                    except Exception:
                        continue
        
        return allocation
    
    def calculate_portfolio_metrics(self, data_fetcher=None) -> Dict:
        """
        Calculate comprehensive portfolio performance metrics.
        
        Args:
            data_fetcher: CryptoDataFetcher instance for current prices
            
        Returns:
            Dictionary with portfolio metrics
        """
        current_value = self.get_portfolio_value(data_fetcher)
        total_return = ((current_value - self.initial_balance) / self.initial_balance) * 100
        
        # Calculate other metrics
        holdings = st.session_state.portfolio_data.get('holdings', {})
        num_positions = len([h for h in holdings.values() if h > 0])
        
        # Transaction metrics
        transactions = st.session_state.portfolio_data.get('transactions', [])
        total_trades = len(transactions)
        total_fees = sum(t.get('fee', 0) for t in transactions)
        
        # Time metrics
        created_at = datetime.fromisoformat(st.session_state.portfolio_data.get('created_at', datetime.now().isoformat()))
        days_active = (datetime.now() - created_at).days
        
        return {
            'current_value': current_value,
            'initial_balance': self.initial_balance,
            'total_return': total_return,
            'total_return_abs': current_value - self.initial_balance,
            'num_positions': num_positions,
            'total_trades': total_trades,
            'total_fees': total_fees,
            'days_active': max(1, days_active),
            'avg_daily_return': total_return / max(1, days_active),
            'cash_percentage': (st.session_state.portfolio_data['cash_balance'] / current_value * 100) if current_value > 0 else 100
        }
    
    def reset_portfolio(self):
        """Reset portfolio to initial state"""
        st.session_state.portfolio_data = {
            'cash_balance': self.initial_balance,
            'holdings': {},
            'transactions': [],
            'created_at': datetime.now(),
            'last_updated': datetime.now()
        }
        
        # Save to database
        self._save_to_database()
    
    def export_portfolio_data(self) -> Dict:
        """Export all portfolio data for backup/analysis"""
        return st.session_state.portfolio_data.copy()
    
    def import_portfolio_data(self, data: Dict):
        """Import portfolio data from backup"""
        # Validate data structure
        required_keys = ['cash_balance', 'holdings', 'transactions']
        if all(key in data for key in required_keys):
            st.session_state.portfolio_data = data
            self._save_to_database()
        else:
            raise ValueError("Invalid portfolio data format")
    
    def get_portfolio_summary(self, data_fetcher=None) -> str:
        """Get a formatted portfolio summary string"""
        metrics = self.calculate_portfolio_metrics(data_fetcher)
        holdings = self.get_holdings()
        
        summary = f"""
        Portfolio Summary:
        ==================
        Total Value: ${metrics['current_value']:,.2f}
        Total Return: {metrics['total_return']:+.2f}% (${metrics['total_return_abs']:+,.2f})
        Cash Balance: ${self.get_cash_balance():,.2f} ({metrics['cash_percentage']:.1f}%)
        Active Positions: {metrics['num_positions']}
        Total Trades: {metrics['total_trades']}
        Days Active: {metrics['days_active']}
        
        Holdings:
        """
        
        for symbol, quantity in holdings.items():
            if quantity > 0:
                summary += f"  {symbol.upper()}: {quantity:.6f}\n"
        
        return summary
