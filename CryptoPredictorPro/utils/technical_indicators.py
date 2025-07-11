import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
import warnings
warnings.filterwarnings('ignore')

try:
    import talib as ta
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False

class TechnicalIndicators:
    """
    Comprehensive technical analysis indicators calculator.
    Supports 20+ indicators with fallback implementations when TA-Lib is not available.
    """
    
    def __init__(self):
        self.talib_available = TALIB_AVAILABLE
    
    def calculate_all_indicators(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Calculate all technical indicators for the given price data.
        
        Args:
            data: DataFrame with columns: timestamp, price, volume (optional)
            
        Returns:
            Dictionary containing all calculated indicators
        """
        if data.empty or 'price' not in data.columns:
            return {}
        
        indicators = {}
        prices = data['price'].values
        volumes = data['volume'].values if 'volume' in data.columns else None
        
        # Moving Averages
        indicators.update(self.calculate_moving_averages(prices))
        
        # Momentum Indicators
        indicators.update(self.calculate_momentum_indicators(prices))
        
        # Volatility Indicators
        indicators.update(self.calculate_volatility_indicators(prices))
        
        # Volume Indicators (if volume data available)
        if volumes is not None:
            indicators.update(self.calculate_volume_indicators(prices, volumes))
        
        # Trend Indicators
        indicators.update(self.calculate_trend_indicators(prices))
        
        # Oscillators
        indicators.update(self.calculate_oscillators(prices))
        
        # Convert arrays to pandas Series with proper index
        for key, value in indicators.items():
            if isinstance(value, np.ndarray):
                indicators[key] = pd.Series(value, index=data.index)
        
        return indicators
    
    def calculate_moving_averages(self, prices: np.ndarray) -> Dict[str, np.ndarray]:
        """Calculate various moving averages"""
        indicators = {}
        
        try:
            if self.talib_available:
                # TA-Lib implementations
                indicators['sma_5'] = ta.SMA(prices, timeperiod=5)
                indicators['sma_10'] = ta.SMA(prices, timeperiod=10)
                indicators['sma_20'] = ta.SMA(prices, timeperiod=20)
                indicators['sma_50'] = ta.SMA(prices, timeperiod=50)
                indicators['sma_200'] = ta.SMA(prices, timeperiod=200)
                
                indicators['ema_5'] = ta.EMA(prices, timeperiod=5)
                indicators['ema_10'] = ta.EMA(prices, timeperiod=10)
                indicators['ema_20'] = ta.EMA(prices, timeperiod=20)
                indicators['ema_50'] = ta.EMA(prices, timeperiod=50)
                
                indicators['wma_20'] = ta.WMA(prices, timeperiod=20)
                indicators['tema_20'] = ta.TEMA(prices, timeperiod=20)
            else:
                # Fallback implementations
                indicators['sma_5'] = self._simple_moving_average(prices, 5)
                indicators['sma_10'] = self._simple_moving_average(prices, 10)
                indicators['sma_20'] = self._simple_moving_average(prices, 20)
                indicators['sma_50'] = self._simple_moving_average(prices, 50)
                indicators['sma_200'] = self._simple_moving_average(prices, 200)
                
                indicators['ema_5'] = self._exponential_moving_average(prices, 5)
                indicators['ema_10'] = self._exponential_moving_average(prices, 10)
                indicators['ema_20'] = self._exponential_moving_average(prices, 20)
                indicators['ema_50'] = self._exponential_moving_average(prices, 50)
                
                indicators['wma_20'] = self._weighted_moving_average(prices, 20)
        
        except Exception as e:
            print(f"Error calculating moving averages: {e}")
        
        return indicators
    
    def calculate_momentum_indicators(self, prices: np.ndarray) -> Dict[str, np.ndarray]:
        """Calculate momentum-based indicators"""
        indicators = {}
        
        try:
            if self.talib_available:
                # RSI
                indicators['rsi'] = ta.RSI(prices, timeperiod=14)
                indicators['rsi_fast'] = ta.RSI(prices, timeperiod=7)
                indicators['rsi_slow'] = ta.RSI(prices, timeperiod=21)
                
                # MACD
                macd, macd_signal, macd_hist = ta.MACD(prices)
                indicators['macd'] = macd
                indicators['macd_signal'] = macd_signal
                indicators['macd_histogram'] = macd_hist
                
                # Stochastic
                stoch_k, stoch_d = ta.STOCH(prices, prices, prices)
                indicators['stoch_k'] = stoch_k
                indicators['stoch_d'] = stoch_d
                
                # Williams %R
                indicators['williams_r'] = ta.WILLR(prices, prices, prices, timeperiod=14)
                
                # Rate of Change
                indicators['roc'] = ta.ROC(prices, timeperiod=10)
                
                # Momentum
                indicators['momentum'] = ta.MOM(prices, timeperiod=10)
                
                # Commodity Channel Index
                indicators['cci'] = ta.CCI(prices, prices, prices, timeperiod=14)
                
            else:
                # Fallback implementations
                indicators['rsi'] = self._rsi(prices, 14)
                indicators['rsi_fast'] = self._rsi(prices, 7)
                indicators['rsi_slow'] = self._rsi(prices, 21)
                
                macd, macd_signal, macd_hist = self._macd(prices)
                indicators['macd'] = macd
                indicators['macd_signal'] = macd_signal
                indicators['macd_histogram'] = macd_hist
                
                indicators['stoch_k'], indicators['stoch_d'] = self._stochastic(prices, prices, prices)
                indicators['williams_r'] = self._williams_r(prices, prices, prices, 14)
                indicators['roc'] = self._rate_of_change(prices, 10)
                indicators['momentum'] = self._momentum(prices, 10)
        
        except Exception as e:
            print(f"Error calculating momentum indicators: {e}")
        
        return indicators
    
    def calculate_volatility_indicators(self, prices: np.ndarray) -> Dict[str, np.ndarray]:
        """Calculate volatility-based indicators"""
        indicators = {}
        
        try:
            if self.talib_available:
                # Bollinger Bands
                bb_upper, bb_middle, bb_lower = ta.BBANDS(prices, timeperiod=20, nbdevup=2, nbdevdn=2)
                indicators['bb_upper'] = bb_upper
                indicators['bb_middle'] = bb_middle
                indicators['bb_lower'] = bb_lower
                
                # Average True Range
                indicators['atr'] = ta.ATR(prices, prices, prices, timeperiod=14)
                
                # Keltner Channels (approximation using ATR)
                ema_20 = ta.EMA(prices, timeperiod=20)
                atr_10 = ta.ATR(prices, prices, prices, timeperiod=10)
                indicators['keltner_upper'] = ema_20 + (2 * atr_10)
                indicators['keltner_lower'] = ema_20 - (2 * atr_10)
                
            else:
                # Fallback implementations
                bb_upper, bb_middle, bb_lower = self._bollinger_bands(prices, 20, 2)
                indicators['bb_upper'] = bb_upper
                indicators['bb_middle'] = bb_middle
                indicators['bb_lower'] = bb_lower
                
                indicators['atr'] = self._average_true_range(prices, prices, prices, 14)
                
                # Keltner Channels
                ema_20 = self._exponential_moving_average(prices, 20)
                atr_10 = self._average_true_range(prices, prices, prices, 10)
                indicators['keltner_upper'] = ema_20 + (2 * atr_10)
                indicators['keltner_lower'] = ema_20 - (2 * atr_10)
            
            # Bollinger Band Width and %B
            if 'bb_upper' in indicators and 'bb_lower' in indicators:
                indicators['bb_width'] = (indicators['bb_upper'] - indicators['bb_lower']) / indicators['bb_middle']
                indicators['bb_percent'] = (prices - indicators['bb_lower']) / (indicators['bb_upper'] - indicators['bb_lower'])
        
        except Exception as e:
            print(f"Error calculating volatility indicators: {e}")
        
        return indicators
    
    def calculate_volume_indicators(self, prices: np.ndarray, volumes: np.ndarray) -> Dict[str, np.ndarray]:
        """Calculate volume-based indicators"""
        indicators = {}
        
        try:
            if self.talib_available:
                # On-Balance Volume
                indicators['obv'] = ta.OBV(prices, volumes)
                
                # Volume Weighted Average Price (approximation)
                indicators['vwap'] = self._vwap(prices, volumes)
                
                # Accumulation/Distribution Line
                indicators['ad_line'] = ta.AD(prices, prices, prices, volumes)
                
                # Chaikin Money Flow
                indicators['cmf'] = self._chaikin_money_flow(prices, prices, prices, volumes, 20)
                
                # Volume Rate of Change
                indicators['volume_roc'] = ta.ROC(volumes, timeperiod=10)
                
            else:
                # Fallback implementations
                indicators['obv'] = self._on_balance_volume(prices, volumes)
                indicators['vwap'] = self._vwap(prices, volumes)
                indicators['ad_line'] = self._accumulation_distribution(prices, prices, prices, volumes)
                indicators['cmf'] = self._chaikin_money_flow(prices, prices, prices, volumes, 20)
                indicators['volume_roc'] = self._rate_of_change(volumes, 10)
            
            # Volume Moving Averages
            indicators['volume_sma_20'] = self._simple_moving_average(volumes, 20)
            indicators['volume_ema_20'] = self._exponential_moving_average(volumes, 20)
        
        except Exception as e:
            print(f"Error calculating volume indicators: {e}")
        
        return indicators
    
    def calculate_trend_indicators(self, prices: np.ndarray) -> Dict[str, np.ndarray]:
        """Calculate trend-following indicators"""
        indicators = {}
        
        try:
            if self.talib_available:
                # Average Directional Index
                indicators['adx'] = ta.ADX(prices, prices, prices, timeperiod=14)
                indicators['plus_di'] = ta.PLUS_DI(prices, prices, prices, timeperiod=14)
                indicators['minus_di'] = ta.MINUS_DI(prices, prices, prices, timeperiod=14)
                
                # Parabolic SAR
                indicators['sar'] = ta.SAR(prices, prices)
                
                # Aroon
                aroon_down, aroon_up = ta.AROON(prices, prices, timeperiod=14)
                indicators['aroon_up'] = aroon_up
                indicators['aroon_down'] = aroon_down
                indicators['aroon_oscillator'] = aroon_up - aroon_down
                
            else:
                # Fallback implementations
                indicators['adx'] = self._adx(prices, prices, prices, 14)
                indicators['plus_di'] = self._plus_di(prices, prices, prices, 14)
                indicators['minus_di'] = self._minus_di(prices, prices, prices, 14)
                
                indicators['sar'] = self._parabolic_sar(prices, prices)
                
                aroon_up, aroon_down = self._aroon(prices, prices, 14)
                indicators['aroon_up'] = aroon_up
                indicators['aroon_down'] = aroon_down
                indicators['aroon_oscillator'] = aroon_up - aroon_down
        
        except Exception as e:
            print(f"Error calculating trend indicators: {e}")
        
        return indicators
    
    def calculate_oscillators(self, prices: np.ndarray) -> Dict[str, np.ndarray]:
        """Calculate oscillator indicators"""
        indicators = {}
        
        try:
            # Detrended Price Oscillator
            indicators['dpo'] = self._detrended_price_oscillator(prices, 20)
            
            # Price Oscillator
            indicators['price_oscillator'] = self._price_oscillator(prices, 12, 26)
            
            # Ultimate Oscillator
            if self.talib_available:
                indicators['ultimate_oscillator'] = ta.ULTOSC(prices, prices, prices)
            else:
                indicators['ultimate_oscillator'] = self._ultimate_oscillator(prices, prices, prices)
        
        except Exception as e:
            print(f"Error calculating oscillators: {e}")
        
        return indicators
    
    # Fallback implementations when TA-Lib is not available
    
    def _simple_moving_average(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Calculate Simple Moving Average"""
        if len(prices) < period:
            return np.full(len(prices), np.nan)
        
        sma = np.full(len(prices), np.nan)
        for i in range(period - 1, len(prices)):
            sma[i] = np.mean(prices[i - period + 1:i + 1])
        return sma
    
    def _exponential_moving_average(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Calculate Exponential Moving Average"""
        if len(prices) < period:
            return np.full(len(prices), np.nan)
        
        ema = np.full(len(prices), np.nan)
        alpha = 2.0 / (period + 1)
        
        # Initialize with first SMA
        ema[period - 1] = np.mean(prices[:period])
        
        for i in range(period, len(prices)):
            ema[i] = alpha * prices[i] + (1 - alpha) * ema[i - 1]
        
        return ema
    
    def _weighted_moving_average(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Calculate Weighted Moving Average"""
        if len(prices) < period:
            return np.full(len(prices), np.nan)
        
        wma = np.full(len(prices), np.nan)
        weights = np.arange(1, period + 1)
        
        for i in range(period - 1, len(prices)):
            wma[i] = np.sum(prices[i - period + 1:i + 1] * weights) / np.sum(weights)
        
        return wma
    
    def _rsi(self, prices: np.ndarray, period: int = 14) -> np.ndarray:
        """Calculate Relative Strength Index"""
        if len(prices) < period + 1:
            return np.full(len(prices), np.nan)
        
        delta = np.diff(prices)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.full(len(prices), np.nan)
        avg_loss = np.full(len(prices), np.nan)
        
        # Initial averages
        avg_gain[period] = np.mean(gain[:period])
        avg_loss[period] = np.mean(loss[:period])
        
        # Smoothed averages
        for i in range(period + 1, len(prices)):
            avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gain[i - 1]) / period
            avg_loss[i] = (avg_loss[i - 1] * (period - 1) + loss[i - 1]) / period
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _macd(self, prices: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
        """Calculate MACD"""
        ema_fast = self._exponential_moving_average(prices, fast)
        ema_slow = self._exponential_moving_average(prices, slow)
        
        macd = ema_fast - ema_slow
        macd_signal = self._exponential_moving_average(macd, signal)
        macd_histogram = macd - macd_signal
        
        return macd, macd_signal, macd_histogram
    
    def _bollinger_bands(self, prices: np.ndarray, period: int = 20, std_dev: float = 2) -> tuple:
        """Calculate Bollinger Bands"""
        sma = self._simple_moving_average(prices, period)
        
        std = np.full(len(prices), np.nan)
        for i in range(period - 1, len(prices)):
            std[i] = np.std(prices[i - period + 1:i + 1])
        
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        
        return upper, sma, lower
    
    def _stochastic(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, 
                   k_period: int = 14, d_period: int = 3) -> tuple:
        """Calculate Stochastic Oscillator"""
        if len(close) < k_period:
            return np.full(len(close), np.nan), np.full(len(close), np.nan)
        
        k_percent = np.full(len(close), np.nan)
        
        for i in range(k_period - 1, len(close)):
            lowest_low = np.min(low[i - k_period + 1:i + 1])
            highest_high = np.max(high[i - k_period + 1:i + 1])
            
            if highest_high != lowest_low:
                k_percent[i] = ((close[i] - lowest_low) / (highest_high - lowest_low)) * 100
            else:
                k_percent[i] = 50
        
        d_percent = self._simple_moving_average(k_percent, d_period)
        
        return k_percent, d_percent
    
    def _williams_r(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
        """Calculate Williams %R"""
        if len(close) < period:
            return np.full(len(close), np.nan)
        
        williams_r = np.full(len(close), np.nan)
        
        for i in range(period - 1, len(close)):
            highest_high = np.max(high[i - period + 1:i + 1])
            lowest_low = np.min(low[i - period + 1:i + 1])
            
            if highest_high != lowest_low:
                williams_r[i] = ((highest_high - close[i]) / (highest_high - lowest_low)) * -100
            else:
                williams_r[i] = -50
        
        return williams_r
    
    def _rate_of_change(self, prices: np.ndarray, period: int = 10) -> np.ndarray:
        """Calculate Rate of Change"""
        if len(prices) < period + 1:
            return np.full(len(prices), np.nan)
        
        roc = np.full(len(prices), np.nan)
        
        for i in range(period, len(prices)):
            if prices[i - period] != 0:
                roc[i] = ((prices[i] - prices[i - period]) / prices[i - period]) * 100
        
        return roc
    
    def _momentum(self, prices: np.ndarray, period: int = 10) -> np.ndarray:
        """Calculate Momentum"""
        if len(prices) < period + 1:
            return np.full(len(prices), np.nan)
        
        momentum = np.full(len(prices), np.nan)
        
        for i in range(period, len(prices)):
            momentum[i] = prices[i] - prices[i - period]
        
        return momentum
    
    def _average_true_range(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
        """Calculate Average True Range"""
        if len(close) < 2:
            return np.full(len(close), np.nan)
        
        true_range = np.full(len(close), np.nan)
        
        for i in range(1, len(close)):
            tr1 = high[i] - low[i]
            tr2 = abs(high[i] - close[i - 1])
            tr3 = abs(low[i] - close[i - 1])
            true_range[i] = max(tr1, tr2, tr3)
        
        atr = self._simple_moving_average(true_range, period)
        return atr
    
    def _on_balance_volume(self, prices: np.ndarray, volumes: np.ndarray) -> np.ndarray:
        """Calculate On-Balance Volume"""
        if len(prices) != len(volumes) or len(prices) < 2:
            return np.full(len(prices), np.nan)
        
        obv = np.full(len(prices), np.nan)
        obv[0] = volumes[0]
        
        for i in range(1, len(prices)):
            if prices[i] > prices[i - 1]:
                obv[i] = obv[i - 1] + volumes[i]
            elif prices[i] < prices[i - 1]:
                obv[i] = obv[i - 1] - volumes[i]
            else:
                obv[i] = obv[i - 1]
        
        return obv
    
    def _vwap(self, prices: np.ndarray, volumes: np.ndarray) -> np.ndarray:
        """Calculate Volume Weighted Average Price"""
        if len(prices) != len(volumes):
            return np.full(len(prices), np.nan)
        
        vwap = np.full(len(prices), np.nan)
        cumulative_volume = 0
        cumulative_price_volume = 0
        
        for i in range(len(prices)):
            cumulative_volume += volumes[i]
            cumulative_price_volume += prices[i] * volumes[i]
            
            if cumulative_volume > 0:
                vwap[i] = cumulative_price_volume / cumulative_volume
        
        return vwap
    
    def _accumulation_distribution(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, volumes: np.ndarray) -> np.ndarray:
        """Calculate Accumulation/Distribution Line"""
        if len(high) != len(volumes):
            return np.full(len(high), np.nan)
        
        ad_line = np.full(len(high), np.nan)
        ad_line[0] = 0
        
        for i in range(len(high)):
            if high[i] != low[i]:
                money_flow_multiplier = ((close[i] - low[i]) - (high[i] - close[i])) / (high[i] - low[i])
            else:
                money_flow_multiplier = 0
            
            money_flow_volume = money_flow_multiplier * volumes[i]
            
            if i == 0:
                ad_line[i] = money_flow_volume
            else:
                ad_line[i] = ad_line[i - 1] + money_flow_volume
        
        return ad_line
    
    def _chaikin_money_flow(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, 
                           volumes: np.ndarray, period: int = 20) -> np.ndarray:
        """Calculate Chaikin Money Flow"""
        if len(high) < period:
            return np.full(len(high), np.nan)
        
        money_flow_volume = np.full(len(high), np.nan)
        
        for i in range(len(high)):
            if high[i] != low[i]:
                money_flow_multiplier = ((close[i] - low[i]) - (high[i] - close[i])) / (high[i] - low[i])
            else:
                money_flow_multiplier = 0
            
            money_flow_volume[i] = money_flow_multiplier * volumes[i]
        
        cmf = np.full(len(high), np.nan)
        
        for i in range(period - 1, len(high)):
            sum_mfv = np.sum(money_flow_volume[i - period + 1:i + 1])
            sum_volume = np.sum(volumes[i - period + 1:i + 1])
            
            if sum_volume > 0:
                cmf[i] = sum_mfv / sum_volume
        
        return cmf
    
    def _adx(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
        """Calculate Average Directional Index (simplified implementation)"""
        if len(close) < period + 1:
            return np.full(len(close), np.nan)
        
        # This is a simplified ADX calculation
        true_range = self._average_true_range(high, low, close, 1)
        plus_dm = np.full(len(close), np.nan)
        minus_dm = np.full(len(close), np.nan)
        
        for i in range(1, len(close)):
            up_move = high[i] - high[i - 1]
            down_move = low[i - 1] - low[i]
            
            if up_move > down_move and up_move > 0:
                plus_dm[i] = up_move
            else:
                plus_dm[i] = 0
                
            if down_move > up_move and down_move > 0:
                minus_dm[i] = down_move
            else:
                minus_dm[i] = 0
        
        plus_di = (self._simple_moving_average(plus_dm, period) / self._simple_moving_average(true_range, period)) * 100
        minus_di = (self._simple_moving_average(minus_dm, period) / self._simple_moving_average(true_range, period)) * 100
        
        dx = np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100
        adx = self._simple_moving_average(dx, period)
        
        return adx
    
    def _plus_di(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
        """Calculate Plus Directional Indicator"""
        # Simplified implementation
        return np.full(len(close), 50)  # Placeholder
    
    def _minus_di(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
        """Calculate Minus Directional Indicator"""
        # Simplified implementation
        return np.full(len(close), 50)  # Placeholder
    
    def _parabolic_sar(self, high: np.ndarray, low: np.ndarray) -> np.ndarray:
        """Calculate Parabolic SAR (simplified implementation)"""
        if len(high) < 5:
            return np.full(len(high), np.nan)
        
        sar = np.full(len(high), np.nan)
        sar[0] = low[0]  # Start with first low
        
        for i in range(1, len(high)):
            # Simplified SAR calculation
            sar[i] = sar[i - 1] + 0.02 * (high[i - 1] - sar[i - 1])
        
        return sar
    
    def _aroon(self, high: np.ndarray, low: np.ndarray, period: int = 14) -> tuple:
        """Calculate Aroon Up and Aroon Down"""
        if len(high) < period:
            return np.full(len(high), np.nan), np.full(len(high), np.nan)
        
        aroon_up = np.full(len(high), np.nan)
        aroon_down = np.full(len(high), np.nan)
        
        for i in range(period - 1, len(high)):
            high_period = high[i - period + 1:i + 1]
            low_period = low[i - period + 1:i + 1]
            
            highest_idx = np.argmax(high_period)
            lowest_idx = np.argmin(low_period)
            
            aroon_up[i] = ((period - highest_idx) / period) * 100
            aroon_down[i] = ((period - lowest_idx) / period) * 100
        
        return aroon_up, aroon_down
    
    def _detrended_price_oscillator(self, prices: np.ndarray, period: int = 20) -> np.ndarray:
        """Calculate Detrended Price Oscillator"""
        sma = self._simple_moving_average(prices, period)
        shift = period // 2 + 1
        
        dpo = np.full(len(prices), np.nan)
        
        for i in range(shift, len(prices)):
            if i - shift >= 0 and not np.isnan(sma[i - shift]):
                dpo[i] = prices[i] - sma[i - shift]
        
        return dpo
    
    def _price_oscillator(self, prices: np.ndarray, fast: int = 12, slow: int = 26) -> np.ndarray:
        """Calculate Price Oscillator"""
        ema_fast = self._exponential_moving_average(prices, fast)
        ema_slow = self._exponential_moving_average(prices, slow)
        
        oscillator = ((ema_fast - ema_slow) / ema_slow) * 100
        
        return oscillator
    
    def _ultimate_oscillator(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
        """Calculate Ultimate Oscillator (simplified implementation)"""
        # This is a simplified implementation
        # In practice, you'd want to implement the full Ultimate Oscillator formula
        if len(close) < 28:
            return np.full(len(close), np.nan)
        
        # Placeholder implementation
        return np.full(len(close), 50)
    
    def get_signal_summary(self, indicators: Dict[str, pd.Series], current_price: float) -> Dict[str, str]:
        """
        Generate trading signal summary based on technical indicators
        
        Args:
            indicators: Dictionary of calculated indicators
            current_price: Current price for comparison
            
        Returns:
            Dictionary with signal interpretations
        """
        signals = {}
        
        try:
            # RSI signals
            if 'rsi' in indicators and not indicators['rsi'].empty:
                rsi_current = indicators['rsi'].iloc[-1]
                if not np.isnan(rsi_current):
                    if rsi_current < 30:
                        signals['RSI'] = 'BUY - Oversold'
                    elif rsi_current > 70:
                        signals['RSI'] = 'SELL - Overbought'
                    else:
                        signals['RSI'] = 'NEUTRAL'
            
            # MACD signals
            if 'macd' in indicators and 'macd_signal' in indicators:
                if not indicators['macd'].empty and not indicators['macd_signal'].empty:
                    macd_current = indicators['macd'].iloc[-1]
                    macd_signal_current = indicators['macd_signal'].iloc[-1]
                    
                    if not (np.isnan(macd_current) or np.isnan(macd_signal_current)):
                        if macd_current > macd_signal_current:
                            signals['MACD'] = 'BUY - Bullish crossover'
                        else:
                            signals['MACD'] = 'SELL - Bearish crossover'
            
            # Bollinger Bands signals
            if all(k in indicators for k in ['bb_upper', 'bb_lower', 'bb_middle']):
                bb_upper = indicators['bb_upper'].iloc[-1]
                bb_lower = indicators['bb_lower'].iloc[-1]
                bb_middle = indicators['bb_middle'].iloc[-1]
                
                if not any(np.isnan([bb_upper, bb_lower, bb_middle])):
                    if current_price <= bb_lower:
                        signals['Bollinger Bands'] = 'BUY - Price at lower band'
                    elif current_price >= bb_upper:
                        signals['Bollinger Bands'] = 'SELL - Price at upper band'
                    elif current_price > bb_middle:
                        signals['Bollinger Bands'] = 'BULLISH - Above middle line'
                    else:
                        signals['Bollinger Bands'] = 'BEARISH - Below middle line'
            
            # Moving Average signals
            if 'sma_20' in indicators and 'sma_50' in indicators:
                sma_20 = indicators['sma_20'].iloc[-1]
                sma_50 = indicators['sma_50'].iloc[-1]
                
                if not (np.isnan(sma_20) or np.isnan(sma_50)):
                    if sma_20 > sma_50:
                        signals['MA Cross'] = 'BULLISH - Short MA above Long MA'
                    else:
                        signals['MA Cross'] = 'BEARISH - Short MA below Long MA'
            
            # Stochastic signals
            if 'stoch_k' in indicators:
                stoch_k = indicators['stoch_k'].iloc[-1]
                if not np.isnan(stoch_k):
                    if stoch_k < 20:
                        signals['Stochastic'] = 'BUY - Oversold'
                    elif stoch_k > 80:
                        signals['Stochastic'] = 'SELL - Overbought'
                    else:
                        signals['Stochastic'] = 'NEUTRAL'
        
        except Exception as e:
            print(f"Error generating signal summary: {e}")
        
        return signals
