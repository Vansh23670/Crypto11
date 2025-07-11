import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any
import warnings
warnings.filterwarnings('ignore')

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    import psycopg2.pool
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False

try:
    from sqlalchemy import create_engine, text
    from sqlalchemy.orm import sessionmaker
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False

class DatabaseManager:
    """
    Advanced database manager for CryptoPredictorPro with PostgreSQL and TimescaleDB support.
    Handles portfolio data, transaction history, and time-series market data persistence.
    """
    
    def __init__(self):
        self.psycopg2_available = PSYCOPG2_AVAILABLE
        self.sqlalchemy_available = SQLALCHEMY_AVAILABLE
        
        # Database configuration from environment
        self.db_config = {
            'host': os.getenv('PGHOST', 'localhost'),
            'port': os.getenv('PGPORT', '5432'),
            'database': os.getenv('PGDATABASE', 'cryptopredictor'),
            'user': os.getenv('PGUSER', 'postgres'),
            'password': os.getenv('PGPASSWORD', 'password')
        }
        
        # Connection URL for SQLAlchemy
        self.database_url = os.getenv('DATABASE_URL', 
            f"postgresql://{self.db_config['user']}:{self.db_config['password']}@"
            f"{self.db_config['host']}:{self.db_config['port']}/{self.db_config['database']}")
        
        self.connection_pool = None
        self.engine = None
        self.session_factory = None
        
        # Initialize database connection
        self._initialize_connection()
        
        # Create tables if they don't exist
        if self.is_connected():
            self._create_tables()
    
    def _initialize_connection(self):
        """Initialize database connections"""
        try:
            if self.psycopg2_available:
                # Create connection pool
                self.connection_pool = psycopg2.pool.ThreadedConnectionPool(
                    minconn=1,
                    maxconn=10,
                    **self.db_config
                )
            
            if self.sqlalchemy_available:
                # Create SQLAlchemy engine
                self.engine = create_engine(
                    self.database_url,
                    pool_size=10,
                    max_overflow=20,
                    pool_pre_ping=True,
                    echo=False
                )
                self.session_factory = sessionmaker(bind=self.engine)
                
        except Exception as e:
            # Database connection failed - will use session state fallback
            self.connection_pool = None
            self.engine = None
    
    def is_connected(self) -> bool:
        """Check if database connection is available"""
        try:
            if self.connection_pool:
                conn = self.connection_pool.getconn()
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                cursor.close()
                self.connection_pool.putconn(conn)
                return True
            return False
        except:
            return False
    
    def _create_tables(self):
        """Create necessary database tables"""
        try:
            if not self.connection_pool:
                return
            
            conn = self.connection_pool.getconn()
            cursor = conn.cursor()
            
            # Create TimescaleDB extension if available
            try:
                cursor.execute("CREATE EXTENSION IF NOT EXISTS timescaledb;")
            except:
                pass  # TimescaleDB not available, continue with regular PostgreSQL
            
            # Portfolio table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS portfolios (
                    id SERIAL PRIMARY KEY,
                    user_id VARCHAR(100) DEFAULT 'default_user',
                    cash_balance DECIMAL(20, 8) NOT NULL DEFAULT 10000.0,
                    holdings JSONB DEFAULT '{}',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            # Transactions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS transactions (
                    id UUID PRIMARY KEY,
                    user_id VARCHAR(100) DEFAULT 'default_user',
                    symbol VARCHAR(50) NOT NULL,
                    transaction_type VARCHAR(10) NOT NULL,
                    quantity DECIMAL(20, 8) NOT NULL,
                    price DECIMAL(20, 8) NOT NULL,
                    amount DECIMAL(20, 8) NOT NULL,
                    fee DECIMAL(20, 8) DEFAULT 0,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata JSONB DEFAULT '{}'
                );
            """)
            
            # Portfolio history table (time-series)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS portfolio_history (
                    id SERIAL PRIMARY KEY,
                    user_id VARCHAR(100) DEFAULT 'default_user',
                    timestamp TIMESTAMP NOT NULL,
                    total_value DECIMAL(20, 8) NOT NULL,
                    cash_balance DECIMAL(20, 8) NOT NULL,
                    holdings_value DECIMAL(20, 8) NOT NULL,
                    daily_return DECIMAL(10, 6) DEFAULT 0,
                    holdings_snapshot JSONB DEFAULT '{}'
                );
            """)
            
            # Market data table (time-series)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS market_data (
                    id SERIAL PRIMARY KEY,
                    symbol VARCHAR(50) NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    price DECIMAL(20, 8) NOT NULL,
                    volume DECIMAL(20, 8) DEFAULT 0,
                    market_cap DECIMAL(20, 2) DEFAULT 0,
                    price_change_24h DECIMAL(10, 6) DEFAULT 0,
                    source VARCHAR(50) DEFAULT 'coingecko'
                );
            """)
            
            # Trading signals table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trading_signals (
                    id SERIAL PRIMARY KEY,
                    symbol VARCHAR(50) NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    signal_type VARCHAR(20) NOT NULL,
                    confidence DECIMAL(5, 4) NOT NULL,
                    price DECIMAL(20, 8) NOT NULL,
                    indicators JSONB DEFAULT '{}',
                    source VARCHAR(50) DEFAULT 'technical_analysis'
                );
            """)
            
            # Risk metrics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS risk_metrics (
                    id SERIAL PRIMARY KEY,
                    user_id VARCHAR(100) DEFAULT 'default_user',
                    timestamp TIMESTAMP NOT NULL,
                    portfolio_value DECIMAL(20, 8) NOT NULL,
                    var_95 DECIMAL(20, 8) DEFAULT 0,
                    volatility DECIMAL(10, 6) DEFAULT 0,
                    beta DECIMAL(10, 6) DEFAULT 1,
                    sharpe_ratio DECIMAL(10, 6) DEFAULT 0,
                    max_drawdown DECIMAL(10, 6) DEFAULT 0,
                    risk_score DECIMAL(4, 2) DEFAULT 5.0
                );
            """)
            
            # Create indexes for better performance
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_transactions_user_timestamp 
                ON transactions(user_id, timestamp DESC);
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_portfolio_history_user_timestamp 
                ON portfolio_history(user_id, timestamp DESC);
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_market_data_symbol_timestamp 
                ON market_data(symbol, timestamp DESC);
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_trading_signals_symbol_timestamp 
                ON trading_signals(symbol, timestamp DESC);
            """)
            
            # Try to create TimescaleDB hypertables
            try:
                cursor.execute("""
                    SELECT create_hypertable('portfolio_history', 'timestamp', 
                    if_not_exists => TRUE, chunk_time_interval => INTERVAL '1 day');
                """)
                
                cursor.execute("""
                    SELECT create_hypertable('market_data', 'timestamp', 
                    if_not_exists => TRUE, chunk_time_interval => INTERVAL '1 day');
                """)
                
                cursor.execute("""
                    SELECT create_hypertable('trading_signals', 'timestamp', 
                    if_not_exists => TRUE, chunk_time_interval => INTERVAL '1 day');
                """)
                
                cursor.execute("""
                    SELECT create_hypertable('risk_metrics', 'timestamp', 
                    if_not_exists => TRUE, chunk_time_interval => INTERVAL '1 day');
                """)
                
            except:
                # TimescaleDB not available, tables will work as regular PostgreSQL tables
                pass
            
            conn.commit()
            cursor.close()
            self.connection_pool.putconn(conn)
            
        except Exception as e:
            # Table creation failed, will use session state fallback
            pass
    
    def save_portfolio(self, portfolio_data: Dict, user_id: str = 'default_user') -> bool:
        """Save portfolio data to database"""
        try:
            if not self.connection_pool:
                return False
            
            conn = self.connection_pool.getconn()
            cursor = conn.cursor()
            
            # Update or insert portfolio
            cursor.execute("""
                INSERT INTO portfolios (user_id, cash_balance, holdings, updated_at)
                VALUES (%s, %s, %s, CURRENT_TIMESTAMP)
                ON CONFLICT (user_id) DO UPDATE SET
                    cash_balance = EXCLUDED.cash_balance,
                    holdings = EXCLUDED.holdings,
                    updated_at = CURRENT_TIMESTAMP;
            """, (
                user_id,
                portfolio_data.get('cash_balance', 10000),
                json.dumps(portfolio_data.get('holdings', {}))
            ))
            
            conn.commit()
            cursor.close()
            self.connection_pool.putconn(conn)
            return True
            
        except Exception:
            return False
    
    def load_portfolio(self, user_id: str = 'default_user') -> Optional[Dict]:
        """Load portfolio data from database"""
        try:
            if not self.connection_pool:
                return None
            
            conn = self.connection_pool.getconn()
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            cursor.execute("""
                SELECT cash_balance, holdings, created_at, updated_at
                FROM portfolios 
                WHERE user_id = %s 
                ORDER BY updated_at DESC 
                LIMIT 1;
            """, (user_id,))
            
            result = cursor.fetchone()
            cursor.close()
            self.connection_pool.putconn(conn)
            
            if result:
                return {
                    'cash_balance': float(result['cash_balance']),
                    'holdings': result['holdings'] or {},
                    'created_at': result['created_at'].isoformat(),
                    'last_updated': result['updated_at'].isoformat()
                }
            
            return None
            
        except Exception:
            return None
    
    def save_transaction(self, transaction: Dict, user_id: str = 'default_user') -> bool:
        """Save transaction to database"""
        try:
            if not self.connection_pool:
                return False
            
            conn = self.connection_pool.getconn()
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO transactions 
                (id, user_id, symbol, transaction_type, quantity, price, amount, fee, timestamp, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
            """, (
                transaction.get('id'),
                user_id,
                transaction.get('symbol'),
                transaction.get('type'),
                transaction.get('quantity'),
                transaction.get('price'),
                transaction.get('amount'),
                transaction.get('fee', 0),
                transaction.get('timestamp', datetime.now()),
                json.dumps(transaction.get('metadata', {}))
            ))
            
            conn.commit()
            cursor.close()
            self.connection_pool.putconn(conn)
            return True
            
        except Exception:
            return False
    
    def get_transaction_history(self, user_id: str = 'default_user', limit: int = 100) -> List[Dict]:
        """Get transaction history from database"""
        try:
            if not self.connection_pool:
                return []
            
            conn = self.connection_pool.getconn()
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            cursor.execute("""
                SELECT id, symbol, transaction_type, quantity, price, amount, fee, timestamp, metadata
                FROM transactions 
                WHERE user_id = %s 
                ORDER BY timestamp DESC 
                LIMIT %s;
            """, (user_id, limit))
            
            results = cursor.fetchall()
            cursor.close()
            self.connection_pool.putconn(conn)
            
            transactions = []
            for row in results:
                transactions.append({
                    'id': row['id'],
                    'symbol': row['symbol'],
                    'type': row['transaction_type'],
                    'quantity': float(row['quantity']),
                    'price': float(row['price']),
                    'amount': float(row['amount']),
                    'fee': float(row['fee']),
                    'timestamp': row['timestamp'].isoformat(),
                    'metadata': row['metadata'] or {}
                })
            
            return transactions
            
        except Exception:
            return []
    
    def save_portfolio_snapshot(self, snapshot: Dict, user_id: str = 'default_user') -> bool:
        """Save portfolio value snapshot for history tracking"""
        try:
            if not self.connection_pool:
                return False
            
            conn = self.connection_pool.getconn()
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO portfolio_history 
                (user_id, timestamp, total_value, cash_balance, holdings_value, daily_return, holdings_snapshot)
                VALUES (%s, %s, %s, %s, %s, %s, %s);
            """, (
                user_id,
                snapshot.get('timestamp', datetime.now()),
                snapshot.get('total_value'),
                snapshot.get('cash_balance'),
                snapshot.get('holdings_value'),
                snapshot.get('daily_return', 0),
                json.dumps(snapshot.get('holdings_snapshot', {}))
            ))
            
            conn.commit()
            cursor.close()
            self.connection_pool.putconn(conn)
            return True
            
        except Exception:
            return False
    
    def get_portfolio_history(self, days: int = 30, user_id: str = 'default_user') -> List[Dict]:
        """Get portfolio history from database"""
        try:
            if not self.connection_pool:
                return []
            
            conn = self.connection_pool.getconn()
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            start_date = datetime.now() - timedelta(days=days)
            
            cursor.execute("""
                SELECT timestamp, total_value, cash_balance, holdings_value, daily_return, holdings_snapshot
                FROM portfolio_history 
                WHERE user_id = %s AND timestamp >= %s
                ORDER BY timestamp ASC;
            """, (user_id, start_date))
            
            results = cursor.fetchall()
            cursor.close()
            self.connection_pool.putconn(conn)
            
            history = []
            for row in results:
                history.append({
                    'date': row['timestamp'].isoformat(),
                    'total_value': float(row['total_value']),
                    'cash_balance': float(row['cash_balance']),
                    'holdings_value': float(row['holdings_value']),
                    'daily_return': float(row['daily_return']),
                    'holdings_snapshot': row['holdings_snapshot'] or {}
                })
            
            return history
            
        except Exception:
            return []
    
    def save_market_data(self, symbol: str, data: Dict) -> bool:
        """Save market data to database"""
        try:
            if not self.connection_pool:
                return False
            
            conn = self.connection_pool.getconn()
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO market_data 
                (symbol, timestamp, price, volume, market_cap, price_change_24h, source)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (symbol, timestamp) DO UPDATE SET
                    price = EXCLUDED.price,
                    volume = EXCLUDED.volume,
                    market_cap = EXCLUDED.market_cap,
                    price_change_24h = EXCLUDED.price_change_24h;
            """, (
                symbol,
                data.get('timestamp', datetime.now()),
                data.get('price'),
                data.get('volume', 0),
                data.get('market_cap', 0),
                data.get('price_change_24h', 0),
                data.get('source', 'coingecko')
            ))
            
            conn.commit()
            cursor.close()
            self.connection_pool.putconn(conn)
            return True
            
        except Exception:
            return False
    
    def get_market_data(self, symbol: str, days: int = 30) -> pd.DataFrame:
        """Get market data from database"""
        try:
            if not self.connection_pool:
                return pd.DataFrame()
            
            conn = self.connection_pool.getconn()
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            start_date = datetime.now() - timedelta(days=days)
            
            cursor.execute("""
                SELECT timestamp, price, volume, market_cap, price_change_24h
                FROM market_data 
                WHERE symbol = %s AND timestamp >= %s
                ORDER BY timestamp ASC;
            """, (symbol, start_date))
            
            results = cursor.fetchall()
            cursor.close()
            self.connection_pool.putconn(conn)
            
            if results:
                data = []
                for row in results:
                    data.append({
                        'timestamp': row['timestamp'],
                        'price': float(row['price']),
                        'volume': float(row['volume']),
                        'market_cap': float(row['market_cap']),
                        'price_change_24h': float(row['price_change_24h'])
                    })
                
                return pd.DataFrame(data)
            
            return pd.DataFrame()
            
        except Exception:
            return pd.DataFrame()
    
    def save_trading_signal(self, signal: Dict) -> bool:
        """Save trading signal to database"""
        try:
            if not self.connection_pool:
                return False
            
            conn = self.connection_pool.getconn()
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO trading_signals 
                (symbol, timestamp, signal_type, confidence, price, indicators, source)
                VALUES (%s, %s, %s, %s, %s, %s, %s);
            """, (
                signal.get('symbol'),
                signal.get('timestamp', datetime.now()),
                signal.get('signal_type'),
                signal.get('confidence'),
                signal.get('price'),
                json.dumps(signal.get('indicators', {})),
                signal.get('source', 'technical_analysis')
            ))
            
            conn.commit()
            cursor.close()
            self.connection_pool.putconn(conn)
            return True
            
        except Exception:
            return False
    
    def get_trading_signals(self, symbol: str = None, days: int = 7) -> List[Dict]:
        """Get trading signals from database"""
        try:
            if not self.connection_pool:
                return []
            
            conn = self.connection_pool.getconn()
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            start_date = datetime.now() - timedelta(days=days)
            
            if symbol:
                cursor.execute("""
                    SELECT symbol, timestamp, signal_type, confidence, price, indicators, source
                    FROM trading_signals 
                    WHERE symbol = %s AND timestamp >= %s
                    ORDER BY timestamp DESC;
                """, (symbol, start_date))
            else:
                cursor.execute("""
                    SELECT symbol, timestamp, signal_type, confidence, price, indicators, source
                    FROM trading_signals 
                    WHERE timestamp >= %s
                    ORDER BY timestamp DESC;
                """, (start_date,))
            
            results = cursor.fetchall()
            cursor.close()
            self.connection_pool.putconn(conn)
            
            signals = []
            for row in results:
                signals.append({
                    'symbol': row['symbol'],
                    'timestamp': row['timestamp'].isoformat(),
                    'signal_type': row['signal_type'],
                    'confidence': float(row['confidence']),
                    'price': float(row['price']),
                    'indicators': row['indicators'] or {},
                    'source': row['source']
                })
            
            return signals
            
        except Exception:
            return []
    
    def save_risk_metrics(self, metrics: Dict, user_id: str = 'default_user') -> bool:
        """Save risk metrics to database"""
        try:
            if not self.connection_pool:
                return False
            
            conn = self.connection_pool.getconn()
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO risk_metrics 
                (user_id, timestamp, portfolio_value, var_95, volatility, beta, sharpe_ratio, max_drawdown, risk_score)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s);
            """, (
                user_id,
                metrics.get('timestamp', datetime.now()),
                metrics.get('portfolio_value'),
                metrics.get('var_95'),
                metrics.get('volatility'),
                metrics.get('beta'),
                metrics.get('sharpe_ratio'),
                metrics.get('max_drawdown'),
                metrics.get('risk_score')
            ))
            
            conn.commit()
            cursor.close()
            self.connection_pool.putconn(conn)
            return True
            
        except Exception:
            return False
    
    def get_risk_metrics_history(self, days: int = 30, user_id: str = 'default_user') -> List[Dict]:
        """Get risk metrics history from database"""
        try:
            if not self.connection_pool:
                return []
            
            conn = self.connection_pool.getconn()
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            start_date = datetime.now() - timedelta(days=days)
            
            cursor.execute("""
                SELECT timestamp, portfolio_value, var_95, volatility, beta, sharpe_ratio, max_drawdown, risk_score
                FROM risk_metrics 
                WHERE user_id = %s AND timestamp >= %s
                ORDER BY timestamp ASC;
            """, (user_id, start_date))
            
            results = cursor.fetchall()
            cursor.close()
            self.connection_pool.putconn(conn)
            
            metrics_history = []
            for row in results:
                metrics_history.append({
                    'timestamp': row['timestamp'].isoformat(),
                    'portfolio_value': float(row['portfolio_value']),
                    'var_95': float(row['var_95']),
                    'volatility': float(row['volatility']),
                    'beta': float(row['beta']),
                    'sharpe_ratio': float(row['sharpe_ratio']),
                    'max_drawdown': float(row['max_drawdown']),
                    'risk_score': float(row['risk_score'])
                })
            
            return metrics_history
            
        except Exception:
            return []
    
    def cleanup_old_data(self, days_to_keep: int = 365):
        """Cleanup old data from time-series tables"""
        try:
            if not self.connection_pool:
                return False
            
            conn = self.connection_pool.getconn()
            cursor = conn.cursor()
            
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            
            # Clean up old market data
            cursor.execute("""
                DELETE FROM market_data WHERE timestamp < %s;
            """, (cutoff_date,))
            
            # Clean up old trading signals
            cursor.execute("""
                DELETE FROM trading_signals WHERE timestamp < %s;
            """, (cutoff_date,))
            
            # Clean up old portfolio history (keep more recent data)
            portfolio_cutoff = datetime.now() - timedelta(days=days_to_keep // 2)
            cursor.execute("""
                DELETE FROM portfolio_history WHERE timestamp < %s;
            """, (portfolio_cutoff,))
            
            conn.commit()
            cursor.close()
            self.connection_pool.putconn(conn)
            return True
            
        except Exception:
            return False
    
    def get_database_stats(self) -> Dict:
        """Get database statistics and health information"""
        try:
            if not self.connection_pool:
                return {'status': 'disconnected'}
            
            conn = self.connection_pool.getconn()
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            stats = {'status': 'connected'}
            
            # Get table row counts
            tables = ['portfolios', 'transactions', 'portfolio_history', 'market_data', 'trading_signals', 'risk_metrics']
            
            for table in tables:
                try:
                    cursor.execute(f"SELECT COUNT(*) as count FROM {table};")
                    result = cursor.fetchone()
                    stats[f'{table}_count'] = result['count']
                except:
                    stats[f'{table}_count'] = 0
            
            # Get database size
            try:
                cursor.execute("""
                    SELECT pg_size_pretty(pg_database_size(current_database())) as database_size;
                """)
                result = cursor.fetchone()
                stats['database_size'] = result['database_size']
            except:
                stats['database_size'] = 'Unknown'
            
            # Check TimescaleDB
            try:
                cursor.execute("SELECT default_version FROM pg_extension WHERE extname='timescaledb';")
                result = cursor.fetchone()
                stats['timescaledb_version'] = result['default_version'] if result else None
            except:
                stats['timescaledb_version'] = None
            
            cursor.close()
            self.connection_pool.putconn(conn)
            
            return stats
            
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    def close_connections(self):
        """Close all database connections"""
        try:
            if self.connection_pool:
                self.connection_pool.closeall()
            
            if self.engine:
                self.engine.dispose()
                
        except Exception:
            pass

