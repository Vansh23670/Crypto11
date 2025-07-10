import os
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError
import streamlit as st
from datetime import datetime
import json

class DatabaseManager:
    def __init__(self):
        self.database_url = os.getenv('DATABASE_URL')
        self.engine = None
        self.Session = None
        self.connect()
    
    def connect(self):
        """Initialize database connection"""
        try:
            if self.database_url:
                self.engine = create_engine(self.database_url)
                self.Session = sessionmaker(bind=self.engine)
                self.create_tables()
            else:
                st.error("Database URL not found. Please check your environment variables.")
        except Exception as e:
            st.error(f"Database connection error: {str(e)}")
    
    def create_tables(self):
        """Create necessary tables for the crypto trading app"""
        try:
            with self.engine.connect() as conn:
                # Users table
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS users (
                        id SERIAL PRIMARY KEY,
                        username VARCHAR(50) UNIQUE NOT NULL,
                        email VARCHAR(100) UNIQUE,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        initial_balance DECIMAL(15,2) DEFAULT 10000.00,
                        current_balance DECIMAL(15,2) DEFAULT 10000.00
                    )
                """))
                
                # Portfolio holdings table
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS portfolio_holdings (
                        id SERIAL PRIMARY KEY,
                        user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
                        coin_id VARCHAR(50) NOT NULL,
                        coin_name VARCHAR(100) NOT NULL,
                        quantity DECIMAL(20,8) NOT NULL,
                        avg_price DECIMAL(15,2) NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """))
                
                # Transactions table
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS transactions (
                        id SERIAL PRIMARY KEY,
                        user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
                        transaction_type VARCHAR(10) NOT NULL CHECK (transaction_type IN ('BUY', 'SELL')),
                        coin_id VARCHAR(50) NOT NULL,
                        coin_name VARCHAR(100) NOT NULL,
                        quantity DECIMAL(20,8) NOT NULL,
                        price DECIMAL(15,2) NOT NULL,
                        total_amount DECIMAL(15,2) NOT NULL,
                        transaction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """))
                
                # Trading strategies table
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS trading_strategies (
                        id SERIAL PRIMARY KEY,
                        user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
                        strategy_name VARCHAR(100) NOT NULL,
                        coin_id VARCHAR(50) NOT NULL,
                        parameters JSON,
                        backtest_period INTEGER,
                        initial_capital DECIMAL(15,2),
                        final_return DECIMAL(10,4),
                        win_rate DECIMAL(5,2),
                        max_drawdown DECIMAL(5,2),
                        sharpe_ratio DECIMAL(8,4),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """))
                
                # Price predictions table
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS price_predictions (
                        id SERIAL PRIMARY KEY,
                        user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
                        coin_id VARCHAR(50) NOT NULL,
                        prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        predicted_price DECIMAL(15,2),
                        prediction_period INTEGER,
                        confidence_score DECIMAL(5,2),
                        model_type VARCHAR(50),
                        actual_price DECIMAL(15,2),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """))
                
                # Market data cache table
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS market_data_cache (
                        id SERIAL PRIMARY KEY,
                        coin_id VARCHAR(50) NOT NULL,
                        price_data JSON,
                        cached_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        expires_at TIMESTAMP,
                        UNIQUE(coin_id)
                    )
                """))
                
                conn.commit()
                
        except SQLAlchemyError as e:
            st.error(f"Error creating tables: {str(e)}")
    
    def get_or_create_user(self, username="default_user", email=None):
        """Get or create a user"""
        try:
            with self.engine.connect() as conn:
                # Check if user exists
                result = conn.execute(
                    text("SELECT * FROM users WHERE username = :username"),
                    {"username": username}
                ).fetchone()
                
                if result:
                    return dict(result._mapping)
                else:
                    # Create new user
                    conn.execute(
                        text("""
                            INSERT INTO users (username, email) 
                            VALUES (:username, :email)
                        """),
                        {"username": username, "email": email}
                    )
                    conn.commit()
                    
                    # Get the created user
                    result = conn.execute(
                        text("SELECT * FROM users WHERE username = :username"),
                        {"username": username}
                    ).fetchone()
                    
                    return dict(result._mapping)
                    
        except SQLAlchemyError as e:
            st.error(f"Error getting/creating user: {str(e)}")
            return None
    
    def get_portfolio_holdings(self, user_id):
        """Get portfolio holdings for a user"""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(
                    text("""
                        SELECT coin_id, coin_name, quantity, avg_price 
                        FROM portfolio_holdings 
                        WHERE user_id = :user_id
                    """),
                    {"user_id": user_id}
                ).fetchall()
                
                holdings = {}
                for row in result:
                    row_dict = dict(row._mapping)
                    holdings[row_dict['coin_id']] = {
                        'quantity': float(row_dict['quantity']),
                        'avg_price': float(row_dict['avg_price']),
                        'coin_name': row_dict['coin_name']
                    }
                
                return holdings
                
        except SQLAlchemyError as e:
            st.error(f"Error getting portfolio holdings: {str(e)}")
            return {}
    
    def update_portfolio_holding(self, user_id, coin_id, coin_name, quantity, avg_price):
        """Update or insert portfolio holding"""
        try:
            with self.engine.connect() as conn:
                # Check if holding exists
                result = conn.execute(
                    text("""
                        SELECT * FROM portfolio_holdings 
                        WHERE user_id = :user_id AND coin_id = :coin_id
                    """),
                    {"user_id": user_id, "coin_id": coin_id}
                ).fetchone()
                
                if result:
                    # Update existing holding
                    conn.execute(
                        text("""
                            UPDATE portfolio_holdings 
                            SET quantity = :quantity, avg_price = :avg_price, updated_at = CURRENT_TIMESTAMP
                            WHERE user_id = :user_id AND coin_id = :coin_id
                        """),
                        {
                            "user_id": user_id,
                            "coin_id": coin_id,
                            "quantity": quantity,
                            "avg_price": avg_price
                        }
                    )
                else:
                    # Insert new holding
                    conn.execute(
                        text("""
                            INSERT INTO portfolio_holdings (user_id, coin_id, coin_name, quantity, avg_price)
                            VALUES (:user_id, :coin_id, :coin_name, :quantity, :avg_price)
                        """),
                        {
                            "user_id": user_id,
                            "coin_id": coin_id,
                            "coin_name": coin_name,
                            "quantity": quantity,
                            "avg_price": avg_price
                        }
                    )
                
                conn.commit()
                return True
                
        except SQLAlchemyError as e:
            st.error(f"Error updating portfolio holding: {str(e)}")
            return False
    
    def delete_portfolio_holding(self, user_id, coin_id):
        """Delete a portfolio holding"""
        try:
            with self.engine.connect() as conn:
                conn.execute(
                    text("""
                        DELETE FROM portfolio_holdings 
                        WHERE user_id = :user_id AND coin_id = :coin_id
                    """),
                    {"user_id": user_id, "coin_id": coin_id}
                )
                conn.commit()
                return True
                
        except SQLAlchemyError as e:
            st.error(f"Error deleting portfolio holding: {str(e)}")
            return False
    
    def add_transaction(self, user_id, transaction_type, coin_id, coin_name, quantity, price, total_amount):
        """Add a new transaction"""
        try:
            with self.engine.connect() as conn:
                conn.execute(
                    text("""
                        INSERT INTO transactions 
                        (user_id, transaction_type, coin_id, coin_name, quantity, price, total_amount)
                        VALUES (:user_id, :transaction_type, :coin_id, :coin_name, :quantity, :price, :total_amount)
                    """),
                    {
                        "user_id": user_id,
                        "transaction_type": transaction_type,
                        "coin_id": coin_id,
                        "coin_name": coin_name,
                        "quantity": quantity,
                        "price": price,
                        "total_amount": total_amount
                    }
                )
                conn.commit()
                return True
                
        except SQLAlchemyError as e:
            st.error(f"Error adding transaction: {str(e)}")
            return False
    
    def get_transactions(self, user_id, limit=None):
        """Get transaction history for a user"""
        try:
            with self.engine.connect() as conn:
                query = """
                    SELECT * FROM transactions 
                    WHERE user_id = :user_id 
                    ORDER BY transaction_date DESC
                """
                
                if limit:
                    query += f" LIMIT {limit}"
                
                result = conn.execute(
                    text(query),
                    {"user_id": user_id}
                ).fetchall()
                
                transactions = []
                for row in result:
                    row_dict = dict(row._mapping)
                    transactions.append({
                        'date': row_dict['transaction_date'],
                        'type': row_dict['transaction_type'],
                        'coin_id': row_dict['coin_id'],
                        'coin_name': row_dict['coin_name'],
                        'quantity': float(row_dict['quantity']),
                        'price': float(row_dict['price']),
                        'total': float(row_dict['total_amount'])
                    })
                
                return transactions
                
        except SQLAlchemyError as e:
            st.error(f"Error getting transactions: {str(e)}")
            return []
    
    def update_user_balance(self, user_id, new_balance):
        """Update user's cash balance"""
        try:
            with self.engine.connect() as conn:
                conn.execute(
                    text("""
                        UPDATE users 
                        SET current_balance = :new_balance 
                        WHERE id = :user_id
                    """),
                    {"user_id": user_id, "new_balance": new_balance}
                )
                conn.commit()
                return True
                
        except SQLAlchemyError as e:
            st.error(f"Error updating user balance: {str(e)}")
            return False
    
    def get_user_balance(self, user_id):
        """Get user's current balance"""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(
                    text("SELECT current_balance FROM users WHERE id = :user_id"),
                    {"user_id": user_id}
                ).fetchone()
                
                if result:
                    return float(result[0])
                return 10000.0  # Default balance
                
        except SQLAlchemyError as e:
            st.error(f"Error getting user balance: {str(e)}")
            return 10000.0
    
    def reset_user_portfolio(self, user_id):
        """Reset user's portfolio to initial state"""
        try:
            with self.engine.connect() as conn:
                # Delete all holdings
                conn.execute(
                    text("DELETE FROM portfolio_holdings WHERE user_id = :user_id"),
                    {"user_id": user_id}
                )
                
                # Delete all transactions
                conn.execute(
                    text("DELETE FROM transactions WHERE user_id = :user_id"),
                    {"user_id": user_id}
                )
                
                # Reset balance
                conn.execute(
                    text("""
                        UPDATE users 
                        SET current_balance = initial_balance 
                        WHERE id = :user_id
                    """),
                    {"user_id": user_id}
                )
                
                conn.commit()
                return True
                
        except SQLAlchemyError as e:
            st.error(f"Error resetting portfolio: {str(e)}")
            return False
    
    def save_strategy_result(self, user_id, strategy_name, coin_id, parameters, backtest_period, 
                           initial_capital, final_return, win_rate, max_drawdown, sharpe_ratio):
        """Save trading strategy backtest results"""
        try:
            with self.engine.connect() as conn:
                conn.execute(
                    text("""
                        INSERT INTO trading_strategies 
                        (user_id, strategy_name, coin_id, parameters, backtest_period, 
                         initial_capital, final_return, win_rate, max_drawdown, sharpe_ratio)
                        VALUES (:user_id, :strategy_name, :coin_id, :parameters, :backtest_period, 
                                :initial_capital, :final_return, :win_rate, :max_drawdown, :sharpe_ratio)
                    """),
                    {
                        "user_id": user_id,
                        "strategy_name": strategy_name,
                        "coin_id": coin_id,
                        "parameters": json.dumps(parameters),
                        "backtest_period": backtest_period,
                        "initial_capital": initial_capital,
                        "final_return": final_return,
                        "win_rate": win_rate,
                        "max_drawdown": max_drawdown,
                        "sharpe_ratio": sharpe_ratio
                    }
                )
                conn.commit()
                return True
                
        except SQLAlchemyError as e:
            st.error(f"Error saving strategy result: {str(e)}")
            return False
    
    def save_price_prediction(self, user_id, coin_id, predicted_price, prediction_period, 
                            confidence_score, model_type):
        """Save price prediction results"""
        try:
            with self.engine.connect() as conn:
                conn.execute(
                    text("""
                        INSERT INTO price_predictions 
                        (user_id, coin_id, predicted_price, prediction_period, confidence_score, model_type)
                        VALUES (:user_id, :coin_id, :predicted_price, :prediction_period, :confidence_score, :model_type)
                    """),
                    {
                        "user_id": user_id,
                        "coin_id": coin_id,
                        "predicted_price": predicted_price,
                        "prediction_period": prediction_period,
                        "confidence_score": confidence_score,
                        "model_type": model_type
                    }
                )
                conn.commit()
                return True
                
        except SQLAlchemyError as e:
            st.error(f"Error saving price prediction: {str(e)}")
            return False
    
    def check_connection(self):
        """Check if database connection is working"""
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
                return True
        except Exception as e:
            st.error(f"Database connection check failed: {str(e)}")
            return False