import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import streamlit as st
from typing import Dict, List, Optional, Union
import warnings
warnings.filterwarnings('ignore')

class CryptoDataFetcher:
    """
    Advanced cryptocurrency data fetcher using CoinGecko API.
    Provides real-time prices, historical data, and market metrics.
    """
    
    def __init__(self):
        self.base_url = "https://api.coingecko.com/api/v3"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'CryptoPredictorPro/1.0'
        })
        self._rate_limit_delay = 1.0  # Seconds between requests
        self._last_request_time = 0
    
    def _rate_limit(self):
        """Implement rate limiting to respect API limits"""
        current_time = time.time()
        time_since_last = current_time - self._last_request_time
        if time_since_last < self._rate_limit_delay:
            time.sleep(self._rate_limit_delay - time_since_last)
        self._last_request_time = time.time()
    
    def _make_request(self, endpoint: str, params: Dict = None) -> Optional[Dict]:
        """Make API request with error handling and rate limiting"""
        self._rate_limit()
        
        try:
            url = f"{self.base_url}/{endpoint}"
            response = self.session.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                # Rate limit exceeded
                st.warning("API rate limit exceeded. Please wait...")
                time.sleep(60)  # Wait 1 minute
                return self._make_request(endpoint, params)  # Retry
            else:
                st.error(f"API request failed: {response.status_code}")
                return None
                
        except requests.exceptions.RequestException as e:
            st.error(f"Network error: {str(e)}")
            return None
    
    @st.cache_data(ttl=60)  # Cache for 1 minute
    def get_current_price(_self, coin_id: str, currency: str = 'usd') -> Dict:
        """
        Get current price and market data for a cryptocurrency.
        
        Args:
            coin_id: CoinGecko coin identifier (e.g., 'bitcoin', 'ethereum')
            currency: Target currency (default: 'usd')
            
        Returns:
            Dictionary containing current price and market data
        """
        params = {
            'ids': coin_id,
            'vs_currencies': currency,
            'include_market_cap': 'true',
            'include_24hr_vol': 'true',
            'include_24hr_change': 'true',
            'include_last_updated_at': 'true'
        }
        
        data = _self._make_request('simple/price', params)
        
        if data and coin_id in data:
            coin_data = data[coin_id]
            return {
                'current_price': coin_data.get(currency, 0),
                'market_cap': coin_data.get(f'market_cap', 0),
                'total_volume': coin_data.get(f'{currency}_24h_vol', 0),
                'price_change_percentage_24h': coin_data.get(f'{currency}_24h_change', 0),
                'last_updated': coin_data.get('last_updated_at', time.time())
            }
        else:
            raise Exception(f"Unable to fetch data for {coin_id}")
    
    @st.cache_data(ttl=300)  # Cache for 5 minutes
    def get_historical_data(_self, coin_id: str, currency: str = 'usd', days: int = 30) -> pd.DataFrame:
        """
        Get historical price data for a cryptocurrency.
        
        Args:
            coin_id: CoinGecko coin identifier
            currency: Target currency
            days: Number of days of historical data
            
        Returns:
            DataFrame with columns: timestamp, price, volume
        """
        # Determine data granularity based on days
        if days <= 1:
            granularity = 'hourly'
        elif days <= 90:
            granularity = 'daily'
        else:
            granularity = 'daily'
        
        endpoint = f'coins/{coin_id}/market_chart'
        params = {
            'vs_currency': currency,
            'days': days,
            'interval': granularity if days > 1 else 'hourly'
        }
        
        data = _self._make_request(endpoint, params)
        
        if data and 'prices' in data:
            # Convert to DataFrame
            prices = data['prices']
            volumes = data.get('total_volumes', [])
            
            df = pd.DataFrame(prices, columns=['timestamp', 'price'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Add volume data if available
            if volumes and len(volumes) == len(prices):
                volume_df = pd.DataFrame(volumes, columns=['timestamp', 'volume'])
                volume_df['timestamp'] = pd.to_datetime(volume_df['timestamp'], unit='ms')
                df = df.merge(volume_df, on='timestamp', how='left')
            else:
                df['volume'] = 0
            
            # Sort by timestamp
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            return df
        else:
            return pd.DataFrame()
    
    @st.cache_data(ttl=900)  # Cache for 15 minutes
    def get_market_overview(_self, currency: str = 'usd', limit: int = 100) -> pd.DataFrame:
        """
        Get market overview for top cryptocurrencies.
        
        Args:
            currency: Target currency
            limit: Number of coins to fetch
            
        Returns:
            DataFrame with market data for top cryptocurrencies
        """
        params = {
            'vs_currency': currency,
            'order': 'market_cap_desc',
            'per_page': limit,
            'page': 1,
            'sparkline': 'false',
            'price_change_percentage': '1h,24h,7d'
        }
        
        data = _self._make_request('coins/markets', params)
        
        if data:
            df = pd.DataFrame(data)
            
            # Select and rename relevant columns
            columns_map = {
                'id': 'coin_id',
                'symbol': 'symbol',
                'name': 'name',
                'current_price': 'price',
                'market_cap': 'market_cap',
                'market_cap_rank': 'rank',
                'total_volume': 'volume_24h',
                'price_change_percentage_1h_in_currency': 'change_1h',
                'price_change_percentage_24h_in_currency': 'change_24h',
                'price_change_percentage_7d_in_currency': 'change_7d',
                'circulating_supply': 'circulating_supply',
                'total_supply': 'total_supply'
            }
            
            # Select available columns
            available_columns = {k: v for k, v in columns_map.items() if k in df.columns}
            df_selected = df[list(available_columns.keys())].copy()
            df_selected.columns = list(available_columns.values())
            
            return df_selected
        else:
            return pd.DataFrame()
    
    def get_coin_info(self, coin_id: str) -> Dict:
        """
        Get detailed information about a cryptocurrency.
        
        Args:
            coin_id: CoinGecko coin identifier
            
        Returns:
            Dictionary with detailed coin information
        """
        endpoint = f'coins/{coin_id}'
        params = {
            'localization': 'false',
            'tickers': 'false',
            'market_data': 'true',
            'community_data': 'true',
            'developer_data': 'false'
        }
        
        data = self._make_request(endpoint, params)
        
        if data:
            market_data = data.get('market_data', {})
            return {
                'name': data.get('name', ''),
                'symbol': data.get('symbol', '').upper(),
                'description': data.get('description', {}).get('en', ''),
                'homepage': data.get('links', {}).get('homepage', []),
                'market_cap_rank': market_data.get('market_cap_rank', 0),
                'market_cap': market_data.get('market_cap', {}).get('usd', 0),
                'total_volume': market_data.get('total_volume', {}).get('usd', 0),
                'high_24h': market_data.get('high_24h', {}).get('usd', 0),
                'low_24h': market_data.get('low_24h', {}).get('usd', 0),
                'ath': market_data.get('ath', {}).get('usd', 0),
                'ath_date': market_data.get('ath_date', {}).get('usd', ''),
                'atl': market_data.get('atl', {}).get('usd', 0),
                'atl_date': market_data.get('atl_date', {}).get('usd', ''),
                'circulating_supply': market_data.get('circulating_supply', 0),
                'total_supply': market_data.get('total_supply', 0),
                'max_supply': market_data.get('max_supply', 0)
            }
        else:
            return {}
    
    def get_trending_coins(self) -> List[Dict]:
        """
        Get trending cryptocurrencies.
        
        Returns:
            List of trending coins data
        """
        data = self._make_request('search/trending')
        
        if data and 'coins' in data:
            trending = []
            for coin_data in data['coins']:
                coin = coin_data.get('item', {})
                trending.append({
                    'id': coin.get('id', ''),
                    'name': coin.get('name', ''),
                    'symbol': coin.get('symbol', ''),
                    'market_cap_rank': coin.get('market_cap_rank', 0),
                    'price_btc': coin.get('price_btc', 0)
                })
            return trending
        else:
            return []
    
    def get_global_market_data(self) -> Dict:
        """
        Get global cryptocurrency market data.
        
        Returns:
            Dictionary with global market statistics
        """
        data = self._make_request('global')
        
        if data and 'data' in data:
            global_data = data['data']
            return {
                'total_market_cap': global_data.get('total_market_cap', {}).get('usd', 0),
                'total_volume': global_data.get('total_volume', {}).get('usd', 0),
                'market_cap_percentage': global_data.get('market_cap_percentage', {}),
                'active_cryptocurrencies': global_data.get('active_cryptocurrencies', 0),
                'markets': global_data.get('markets', 0),
                'market_cap_change_percentage_24h': global_data.get('market_cap_change_percentage_24h_usd', 0)
            }
        else:
            return {}
    
    def search_coins(self, query: str) -> List[Dict]:
        """
        Search for cryptocurrencies by name or symbol.
        
        Args:
            query: Search query
            
        Returns:
            List of matching coins
        """
        params = {'query': query}
        data = self._make_request('search', params)
        
        if data and 'coins' in data:
            results = []
            for coin in data['coins']:
                results.append({
                    'id': coin.get('id', ''),
                    'name': coin.get('name', ''),
                    'symbol': coin.get('symbol', ''),
                    'market_cap_rank': coin.get('market_cap_rank', 0)
                })
            return results
        else:
            return []
    
    def get_price_alerts_data(self, coin_ids: List[str], currency: str = 'usd') -> Dict:
        """
        Get price data for multiple coins (useful for alerts).
        
        Args:
            coin_ids: List of coin identifiers
            currency: Target currency
            
        Returns:
            Dictionary with price data for each coin
        """
        if not coin_ids:
            return {}
        
        params = {
            'ids': ','.join(coin_ids),
            'vs_currencies': currency,
            'include_24hr_change': 'true'
        }
        
        data = self._make_request('simple/price', params)
        
        if data:
            results = {}
            for coin_id in coin_ids:
                if coin_id in data:
                    results[coin_id] = {
                        'price': data[coin_id].get(currency, 0),
                        'change_24h': data[coin_id].get(f'{currency}_24h_change', 0)
                    }
            return results
        else:
            return {}
    
    def get_ohlc_data(self, coin_id: str, currency: str = 'usd', days: int = 7) -> pd.DataFrame:
        """
        Get OHLC (Open, High, Low, Close) data for candlestick charts.
        
        Args:
            coin_id: CoinGecko coin identifier
            currency: Target currency
            days: Number of days (limited by API)
            
        Returns:
            DataFrame with OHLC data
        """
        # CoinGecko OHLC endpoint has limitations
        endpoint = f'coins/{coin_id}/ohlc'
        params = {
            'vs_currency': currency,
            'days': min(days, 365)  # API limitation
        }
        
        data = self._make_request(endpoint, params)
        
        if data:
            df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df.sort_values('timestamp').reset_index(drop=True)
        else:
            # Fallback: use regular price data to approximate OHLC
            return self._approximate_ohlc_from_prices(coin_id, currency, days)
    
    def _approximate_ohlc_from_prices(self, coin_id: str, currency: str, days: int) -> pd.DataFrame:
        """
        Approximate OHLC data from regular price data when OHLC is not available.
        """
        try:
            price_data = self.get_historical_data(coin_id, currency, days)
            if price_data.empty:
                return pd.DataFrame()
            
            # Group by day and calculate OHLC
            price_data['date'] = price_data['timestamp'].dt.date
            
            ohlc = price_data.groupby('date')['price'].agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last'
            }).reset_index()
            
            ohlc['timestamp'] = pd.to_datetime(ohlc['date'])
            ohlc = ohlc.drop('date', axis=1)
            
            return ohlc[['timestamp', 'open', 'high', 'low', 'close']]
            
        except Exception:
            return pd.DataFrame()
