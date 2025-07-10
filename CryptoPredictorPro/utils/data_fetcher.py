import requests
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
import time
import random

class CryptoDataFetcher:
    def __init__(self):
        self.base_url = "https://api.coingecko.com/api/v3"
        self.session = requests.Session()
        self.last_request_time = 0
        self.min_request_interval = 1.2  # Minimum 1.2 seconds between requests
        self.max_retries = 3
        
    def _rate_limit(self):
        """Implement rate limiting to respect API limits"""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        
        if time_since_last_request < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last_request
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def _make_request_with_retry(self, url, params=None, timeout=10):
        """Make HTTP request with retry logic for rate limiting"""
        for attempt in range(self.max_retries):
            try:
                self._rate_limit()
                response = self.session.get(url, params=params, timeout=timeout)
                
                # Handle rate limiting
                if response.status_code == 429:
                    wait_time = 2 ** attempt + random.uniform(0, 1)  # Exponential backoff
                    st.warning(f"Rate limit hit. Waiting {wait_time:.1f} seconds before retry...")
                    time.sleep(wait_time)
                    continue
                
                response.raise_for_status()
                return response
                
            except requests.RequestException as e:
                if attempt == self.max_retries - 1:
                    raise
                wait_time = 2 ** attempt + random.uniform(0, 1)
                st.warning(f"Request failed. Retrying in {wait_time:.1f} seconds...")
                time.sleep(wait_time)
        
        raise requests.RequestException("Max retries exceeded")
    
    def get_current_price(self, coin_id, currency='usd'):
        """
        Fetch current price data for a specific cryptocurrency
        """
        try:
            url = f"{self.base_url}/simple/price"
            params = {
                'ids': coin_id,
                'vs_currencies': currency,
                'include_market_cap': 'true',
                'include_24hr_vol': 'true',
                'include_24hr_change': 'true'
            }
            
            response = self._make_request_with_retry(url, params=params)
            data = response.json()
            
            if coin_id in data:
                result = {
                    'current_price': data[coin_id][currency],
                    'market_cap': data[coin_id].get(f'{currency}_market_cap', 0),
                    'total_volume': data[coin_id].get(f'{currency}_24h_vol', 0),
                    'price_change_percentage_24h': data[coin_id].get(f'{currency}_24h_change', 0)
                }
                
                # Cache the result for fallback
                if 'last_price_data' not in st.session_state:
                    st.session_state.last_price_data = {}
                st.session_state.last_price_data[coin_id] = result
                
                return result
            else:
                raise ValueError(f"No data found for {coin_id}")
                
        except requests.RequestException as e:
            st.error(f"API request failed: {str(e)}")
            # Return cached data if available
            if hasattr(st.session_state, 'last_price_data') and coin_id in st.session_state.last_price_data:
                st.warning("Using cached price data due to API issues")
                return st.session_state.last_price_data[coin_id]
            
            # If no cached data, raise the exception to let the user know
            raise
        except Exception as e:
            st.error(f"Error fetching current price: {str(e)}")
            raise
    
    @st.cache_data(ttl=300)  # Cache for 5 minutes
    def get_historical_data(_self, coin_id, currency='usd', days=30):
        """
        Fetch historical price data for a specific cryptocurrency
        """
        try:
            url = f"{_self.base_url}/coins/{coin_id}/market_chart"
            params = {
                'vs_currency': currency,
                'days': days,
                'interval': 'daily' if days > 1 else 'hourly'
            }
            
            response = _self._make_request_with_retry(url, params=params, timeout=15)
            data = response.json()
            
            # Convert to DataFrame
            prices = data['prices']
            df = pd.DataFrame(prices, columns=['timestamp', 'price'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Add volume data if available
            if 'total_volumes' in data:
                volumes = data['total_volumes']
                volume_df = pd.DataFrame(volumes, columns=['timestamp', 'volume'])
                volume_df['timestamp'] = pd.to_datetime(volume_df['timestamp'], unit='ms')
                df = df.merge(volume_df, on='timestamp', how='left')
            
            return df
            
        except requests.RequestException as e:
            st.error(f"API request failed: {str(e)}")
            return pd.DataFrame()
        except Exception as e:
            st.error(f"Error fetching historical data: {str(e)}")
            return pd.DataFrame()
    
    @st.cache_data(ttl=3600)  # Cache for 1 hour
    def get_coin_info(_self, coin_id):
        """
        Fetch detailed information about a cryptocurrency
        """
        try:
            url = f"{_self.base_url}/coins/{coin_id}"
            params = {
                'localization': 'false',
                'tickers': 'false',
                'market_data': 'true',
                'community_data': 'false',
                'developer_data': 'false',
                'sparkline': 'false'
            }
            
            response = _self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            return {
                'name': data.get('name', ''),
                'symbol': data.get('symbol', '').upper(),
                'description': data.get('description', {}).get('en', ''),
                'image': data.get('image', {}).get('large', ''),
                'market_cap_rank': data.get('market_cap_rank', 0),
                'current_price': data.get('market_data', {}).get('current_price', {}).get('usd', 0),
                'ath': data.get('market_data', {}).get('ath', {}).get('usd', 0),
                'atl': data.get('market_data', {}).get('atl', {}).get('usd', 0),
                'max_supply': data.get('market_data', {}).get('max_supply', 0),
                'circulating_supply': data.get('market_data', {}).get('circulating_supply', 0)
            }
            
        except requests.RequestException as e:
            st.error(f"Network error: {str(e)}")
            return {}
        except Exception as e:
            st.error(f"Error fetching coin info: {str(e)}")
            return {}
    
    def get_trending_coins(self):
        """
        Fetch trending cryptocurrencies
        """
        try:
            url = f"{self.base_url}/search/trending"
            
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            trending = []
            for coin in data.get('coins', []):
                trending.append({
                    'id': coin['item']['id'],
                    'name': coin['item']['name'],
                    'symbol': coin['item']['symbol'],
                    'market_cap_rank': coin['item']['market_cap_rank'],
                    'image': coin['item']['large']
                })
            
            return trending
            
        except requests.RequestException as e:
            st.error(f"Network error: {str(e)}")
            return []
        except Exception as e:
            st.error(f"Error fetching trending coins: {str(e)}")
            return []
    
    def get_ohlc_data(self, coin_id, currency='usd', days=30):
        """
        Fetch OHLC (Open, High, Low, Close) data for candlestick charts
        """
        try:
            url = f"{self.base_url}/coins/{coin_id}/ohlc"
            params = {
                'vs_currency': currency,
                'days': days
            }
            
            response = self.session.get(url, params=params, timeout=15)
            response.raise_for_status()
            
            data = response.json()
            
            # Convert to DataFrame
            df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            return df
            
        except requests.RequestException as e:
            st.error(f"Network error: {str(e)}")
            return pd.DataFrame()
        except Exception as e:
            st.error(f"Error fetching OHLC data: {str(e)}")
            return pd.DataFrame()
