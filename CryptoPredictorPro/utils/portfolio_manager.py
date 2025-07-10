import pandas as pd
import numpy as np
from datetime import datetime
import streamlit as st
from utils.database import DatabaseManager

class PortfolioManager:
    def __init__(self, initial_balance=10000):
        self.db = DatabaseManager()
        self.initial_balance = initial_balance
        
        # Initialize user in session state
        if 'user_id' not in st.session_state:
            user = self.db.get_or_create_user("default_user")
            if user:
                st.session_state.user_id = user['id']
                st.session_state.user_data = user
            else:
                # Fallback to session state if database fails
                st.session_state.user_id = None
                self._ensure_portfolio_data()
    
    def _ensure_portfolio_data(self):
        """Ensure portfolio_data exists in session state for fallback"""
        if 'portfolio_data' not in st.session_state:
            st.session_state.portfolio_data = {
                'balance': self.initial_balance,
                'initial_balance': self.initial_balance,
                'holdings': {},
                'transactions': [],
                'daily_values': []
            }
    
    def get_balance(self):
        """Get current cash balance"""
        if st.session_state.user_id:
            return self.db.get_user_balance(st.session_state.user_id)
        else:
            self._ensure_portfolio_data()
            return st.session_state.portfolio_data['balance']
    
    def get_holdings(self):
        """Get current holdings"""
        if st.session_state.user_id:
            return self.db.get_portfolio_holdings(st.session_state.user_id)
        else:
            self._ensure_portfolio_data()
            return st.session_state.portfolio_data['holdings']
    
    def get_transactions(self):
        """Get all transactions"""
        if st.session_state.user_id:
            return self.db.get_transactions(st.session_state.user_id)
        else:
            self._ensure_portfolio_data()
            return st.session_state.portfolio_data['transactions']
    
    def buy_crypto(self, coin_id, coin_name, quantity, price):
        """
        Buy cryptocurrency
        """
        try:
            total_cost = quantity * price
            current_balance = self.get_balance()
            
            if total_cost > current_balance:
                return False, "Insufficient balance"
            
            if st.session_state.user_id:
                # Database-backed operations
                # Update balance
                new_balance = current_balance - total_cost
                self.db.update_user_balance(st.session_state.user_id, new_balance)
                
                # Update holdings
                holdings = self.get_holdings()
                if coin_id in holdings:
                    current_holding = holdings[coin_id]
                    total_quantity = current_holding['quantity'] + quantity
                    total_value = (current_holding['quantity'] * current_holding['avg_price']) + total_cost
                    avg_price = total_value / total_quantity
                    
                    self.db.update_portfolio_holding(
                        st.session_state.user_id, coin_id, coin_name, total_quantity, avg_price
                    )
                else:
                    self.db.update_portfolio_holding(
                        st.session_state.user_id, coin_id, coin_name, quantity, price
                    )
                
                # Record transaction
                self.db.add_transaction(
                    st.session_state.user_id, 'BUY', coin_id, coin_name, quantity, price, total_cost
                )
                
            else:
                # Fallback to session state
                st.session_state.portfolio_data['balance'] -= total_cost
                
                if coin_id in st.session_state.portfolio_data['holdings']:
                    current_holding = st.session_state.portfolio_data['holdings'][coin_id]
                    total_quantity = current_holding['quantity'] + quantity
                    total_value = (current_holding['quantity'] * current_holding['avg_price']) + total_cost
                    avg_price = total_value / total_quantity
                    
                    st.session_state.portfolio_data['holdings'][coin_id] = {
                        'quantity': total_quantity,
                        'avg_price': avg_price,
                        'coin_name': coin_name
                    }
                else:
                    st.session_state.portfolio_data['holdings'][coin_id] = {
                        'quantity': quantity,
                        'avg_price': price,
                        'coin_name': coin_name
                    }
                
                transaction = {
                    'date': datetime.now(),
                    'type': 'BUY',
                    'coin_id': coin_id,
                    'coin_name': coin_name,
                    'quantity': quantity,
                    'price': price,
                    'total': total_cost
                }
                st.session_state.portfolio_data['transactions'].append(transaction)
            
            return True, "Purchase successful"
            
        except Exception as e:
            return False, f"Error buying crypto: {str(e)}"
    
    def sell_crypto(self, coin_id, coin_name, quantity, price):
        """
        Sell cryptocurrency
        """
        try:
            holdings = self.get_holdings()
            
            if coin_id not in holdings:
                return False, "No holdings found for this cryptocurrency"
            
            current_holding = holdings[coin_id]
            
            if quantity > current_holding['quantity']:
                return False, "Insufficient quantity to sell"
            
            total_value = quantity * price
            current_balance = self.get_balance()
            
            if st.session_state.user_id:
                # Database-backed operations
                # Update balance
                new_balance = current_balance + total_value
                self.db.update_user_balance(st.session_state.user_id, new_balance)
                
                # Update holdings
                new_quantity = current_holding['quantity'] - quantity
                
                if new_quantity == 0:
                    self.db.delete_portfolio_holding(st.session_state.user_id, coin_id)
                else:
                    self.db.update_portfolio_holding(
                        st.session_state.user_id, coin_id, coin_name, 
                        new_quantity, current_holding['avg_price']
                    )
                
                # Record transaction
                self.db.add_transaction(
                    st.session_state.user_id, 'SELL', coin_id, coin_name, quantity, price, total_value
                )
                
            else:
                # Fallback to session state
                st.session_state.portfolio_data['balance'] += total_value
                
                new_quantity = current_holding['quantity'] - quantity
                
                if new_quantity == 0:
                    del st.session_state.portfolio_data['holdings'][coin_id]
                else:
                    st.session_state.portfolio_data['holdings'][coin_id]['quantity'] = new_quantity
                
                transaction = {
                    'date': datetime.now(),
                    'type': 'SELL',
                    'coin_id': coin_id,
                    'coin_name': coin_name,
                    'quantity': quantity,
                    'price': price,
                    'total': total_value
                }
                st.session_state.portfolio_data['transactions'].append(transaction)
            
            return True, "Sale successful"
            
        except Exception as e:
            return False, f"Error selling crypto: {str(e)}"
    
    def get_portfolio_value(self, current_prices=None):
        """
        Calculate total portfolio value
        """
        try:
            if st.session_state.user_id:
                # Database-backed calculation
                total_value = self.get_balance()
                holdings = self.get_holdings()
                
                if current_prices:
                    for coin_id, holding in holdings.items():
                        if coin_id in current_prices:
                            total_value += holding['quantity'] * current_prices[coin_id]
                
                return total_value
            else:
                self._ensure_portfolio_data()
                total_value = st.session_state.portfolio_data['balance']
                
                if current_prices:
                    for coin_id, holding in st.session_state.portfolio_data['holdings'].items():
                        if coin_id in current_prices:
                            total_value += holding['quantity'] * current_prices[coin_id]
                
                return total_value
            
        except Exception as e:
            st.error(f"Error calculating portfolio value: {str(e)}")
            return self.initial_balance
    
    def get_portfolio_breakdown(self, current_prices=None):
        """
        Get detailed portfolio breakdown
        """
        breakdown = []
        
        try:
            if st.session_state.user_id:
                # Database-backed calculation
                cash_balance = self.get_balance()
                holdings = self.get_holdings()
            else:
                self._ensure_portfolio_data()
                cash_balance = st.session_state.portfolio_data['balance']
                holdings = st.session_state.portfolio_data['holdings']
            
            # Cash
            breakdown.append({
                'asset': 'Cash',
                'quantity': 1,
                'current_price': cash_balance,
                'value': cash_balance,
                'percentage': 0  # Will be calculated later
            })
            
            # Crypto holdings
            for coin_id, holding in holdings.items():
                current_price = current_prices.get(coin_id, holding['avg_price']) if current_prices else holding['avg_price']
                value = holding['quantity'] * current_price
                
                breakdown.append({
                    'asset': holding['coin_name'],
                    'quantity': holding['quantity'],
                    'avg_price': holding['avg_price'],
                    'current_price': current_price,
                    'value': value,
                    'pnl': ((current_price - holding['avg_price']) / holding['avg_price']) * 100,
                    'percentage': 0  # Will be calculated later
                })
            
            # Calculate percentages
            total_value = sum(item['value'] for item in breakdown)
            for item in breakdown:
                item['percentage'] = (item['value'] / total_value * 100) if total_value > 0 else 0
            
            return breakdown
            
        except Exception as e:
            st.error(f"Error calculating portfolio breakdown: {str(e)}")
            return []
    
    def get_total_pnl(self, current_prices=None):
        """
        Calculate total profit/loss percentage
        """
        try:
            if st.session_state.user_id:
                # Database-backed calculation
                initial_balance = self.initial_balance
            else:
                self._ensure_portfolio_data()
                initial_balance = st.session_state.portfolio_data['initial_balance']
            
            current_value = self.get_portfolio_value(current_prices)
            
            if initial_balance == 0:
                return 0
            
            pnl = ((current_value - initial_balance) / initial_balance) * 100
            return pnl
            
        except Exception as e:
            st.error(f"Error calculating total PnL: {str(e)}")
            return 0
    
    def get_recent_trades(self, limit=5):
        """
        Get recent trading activity
        """
        if st.session_state.user_id:
            transactions = self.db.get_transactions(st.session_state.user_id, limit)
        else:
            self._ensure_portfolio_data()
            transactions = st.session_state.portfolio_data.get('transactions', [])
            transactions = transactions[-limit:] if len(transactions) > limit else transactions
        
        # Format for display
        formatted_trades = []
        for trade in reversed(transactions):
            formatted_trades.append({
                'Date': trade['date'].strftime('%Y-%m-%d %H:%M'),
                'Type': trade['type'],
                'Asset': trade['coin_name'],
                'Quantity': f"{trade['quantity']:.6f}",
                'Price': f"${trade['price']:.2f}",
                'Total': f"${trade['total']:.2f}"
            })
        
        return formatted_trades
    
    def reset_portfolio(self):
        """
        Reset portfolio to initial state
        """
        if st.session_state.user_id:
            # Database-backed reset
            self.db.reset_user_portfolio(st.session_state.user_id)
        else:
            self._ensure_portfolio_data()
            initial_balance = st.session_state.portfolio_data['initial_balance']
            st.session_state.portfolio_data = {
                'balance': initial_balance,
                'initial_balance': initial_balance,
                'holdings': {},
                'transactions': [],
                'daily_values': []
            }
    
    def get_performance_metrics(self, current_prices=None):
        """
        Calculate performance metrics
        """
        try:
            if st.session_state.user_id:
                # Database-backed calculation
                transactions = self.get_transactions()
            else:
                self._ensure_portfolio_data()
                transactions = st.session_state.portfolio_data['transactions']
            
            if not transactions:
                return {
                    'total_trades': 0,
                    'winning_trades': 0,
                    'losing_trades': 0,
                    'win_rate': 0,
                    'avg_gain': 0,
                    'avg_loss': 0,
                    'profit_factor': 0
                }
            
            # Group transactions by coin
            coin_trades = {}
            for trade in transactions:
                coin_id = trade['coin_id']
                if coin_id not in coin_trades:
                    coin_trades[coin_id] = []
                coin_trades[coin_id].append(trade)
            
            # Calculate trade outcomes
            winning_trades = 0
            losing_trades = 0
            total_gain = 0
            total_loss = 0
            
            for coin_id, trades in coin_trades.items():
                # Match buy and sell orders
                buy_orders = [t for t in trades if t['type'] == 'BUY']
                sell_orders = [t for t in trades if t['type'] == 'SELL']
                
                for sell in sell_orders:
                    # Find corresponding buy order
                    for buy in buy_orders:
                        if buy['date'] <= sell['date']:
                            gain_loss = (sell['price'] - buy['price']) / buy['price'] * 100
                            
                            if gain_loss > 0:
                                winning_trades += 1
                                total_gain += gain_loss
                            else:
                                losing_trades += 1
                                total_loss += abs(gain_loss)
                            break
            
            total_trades = winning_trades + losing_trades
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            avg_gain = total_gain / winning_trades if winning_trades > 0 else 0
            avg_loss = total_loss / losing_trades if losing_trades > 0 else 0
            profit_factor = total_gain / total_loss if total_loss > 0 else 0
            
            return {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': win_rate,
                'avg_gain': avg_gain,
                'avg_loss': avg_loss,
                'profit_factor': profit_factor
            }
            
        except Exception as e:
            st.error(f"Error calculating performance metrics: {str(e)}")
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'avg_gain': 0,
                'avg_loss': 0,
                'profit_factor': 0
            }
