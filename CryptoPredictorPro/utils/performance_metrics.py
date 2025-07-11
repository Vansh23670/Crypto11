import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime, timedelta
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class PerformanceMetrics:
    """
    Institutional-grade performance metrics calculator.
    Provides comprehensive portfolio analytics including alpha, beta, 
    Sharpe ratio, Calmar ratio, and advanced risk-adjusted metrics.
    """
    
    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize performance metrics calculator.
        
        Args:
            risk_free_rate: Annual risk-free rate (default: 2%)
        """
        self.risk_free_rate = risk_free_rate
        self.trading_days_per_year = 252
    
    def calculate_all_metrics(self, returns: Union[pd.Series, np.ndarray], 
                            benchmark_returns: Optional[Union[pd.Series, np.ndarray]] = None,
                            portfolio_values: Optional[Union[pd.Series, np.ndarray]] = None) -> Dict:
        """
        Calculate comprehensive performance metrics.
        
        Args:
            returns: Portfolio returns series
            benchmark_returns: Benchmark returns for comparison
            portfolio_values: Portfolio value time series
            
        Returns:
            Dictionary with all calculated metrics
        """
        if isinstance(returns, pd.Series):
            returns = returns.dropna().values
        if isinstance(benchmark_returns, pd.Series):
            benchmark_returns = benchmark_returns.dropna().values
        if isinstance(portfolio_values, pd.Series):
            portfolio_values = portfolio_values.dropna().values
        
        returns = np.array(returns)
        
        if len(returns) < 2:
            return self._empty_metrics()
        
        metrics = {}
        
        # Basic return metrics
        metrics.update(self._calculate_return_metrics(returns))
        
        # Risk metrics
        metrics.update(self._calculate_risk_metrics(returns, portfolio_values))
        
        # Risk-adjusted performance metrics
        metrics.update(self._calculate_risk_adjusted_metrics(returns))
        
        # Benchmark comparison metrics
        if benchmark_returns is not None and len(benchmark_returns) > 0:
            metrics.update(self._calculate_benchmark_metrics(returns, benchmark_returns))
        
        # Drawdown analysis
        if portfolio_values is not None and len(portfolio_values) > 0:
            metrics.update(self._calculate_drawdown_metrics(portfolio_values))
        
        # Advanced metrics
        metrics.update(self._calculate_advanced_metrics(returns))
        
        return metrics
    
    def calculate_comprehensive_metrics(self, portfolio_returns: Union[pd.Series, np.ndarray],
                                      benchmark_returns: Optional[Union[pd.Series, np.ndarray]] = None,
                                      risk_free_rate: Optional[float] = None,
                                      portfolio_values: Optional[Union[pd.Series, np.ndarray]] = None) -> Dict:
        """
        Calculate comprehensive performance metrics with additional analysis.
        
        Args:
            portfolio_returns: Portfolio returns
            benchmark_returns: Benchmark returns
            risk_free_rate: Risk-free rate override
            portfolio_values: Portfolio values for drawdown analysis
            
        Returns:
            Comprehensive metrics dictionary
        """
        if risk_free_rate is not None:
            original_rf = self.risk_free_rate
            self.risk_free_rate = risk_free_rate
        
        try:
            # Convert to numpy arrays
            if isinstance(portfolio_returns, pd.Series):
                portfolio_returns = portfolio_returns.dropna().values
            if isinstance(benchmark_returns, pd.Series):
                benchmark_returns = benchmark_returns.dropna().values
            if isinstance(portfolio_values, pd.Series):
                portfolio_values = portfolio_values.dropna().values
            
            portfolio_returns = np.array(portfolio_returns)
            
            if len(portfolio_returns) < 2:
                return self._empty_metrics()
            
            # Calculate all standard metrics
            metrics = self.calculate_all_metrics(portfolio_returns, benchmark_returns, portfolio_values)
            
            # Additional comprehensive metrics
            metrics.update(self._calculate_higher_moments(portfolio_returns))
            metrics.update(self._calculate_tail_risk_metrics(portfolio_returns))
            metrics.update(self._calculate_performance_consistency(portfolio_returns))
            
            if benchmark_returns is not None and len(benchmark_returns) > 0:
                metrics.update(self._calculate_factor_analysis(portfolio_returns, benchmark_returns))
            
            # Rolling metrics analysis
            if len(portfolio_returns) >= 60:  # Minimum for rolling analysis
                metrics.update(self._calculate_rolling_metrics(portfolio_returns))
            
            return metrics
            
        finally:
            if risk_free_rate is not None:
                self.risk_free_rate = original_rf
    
    def _empty_metrics(self) -> Dict:
        """Return empty metrics dictionary for invalid inputs"""
        return {
            'total_return': 0,
            'annualized_return': 0,
            'volatility': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0,
            'calmar_ratio': 0,
            'error': 'Insufficient data for calculations'
        }
    
    def _calculate_return_metrics(self, returns: np.ndarray) -> Dict:
        """Calculate basic return metrics"""
        try:
            total_return = np.prod(1 + returns) - 1
            
            # Annualized return
            periods_per_year = self.trading_days_per_year
            if len(returns) < periods_per_year:
                periods_per_year = len(returns)  # Adjust for shorter periods
            
            annualized_return = (1 + total_return) ** (self.trading_days_per_year / len(returns)) - 1
            
            # Mean and compound returns
            arithmetic_mean = np.mean(returns)
            geometric_mean = np.prod(1 + returns) ** (1 / len(returns)) - 1
            
            return {
                'total_return': total_return,
                'annualized_return': annualized_return,
                'arithmetic_mean_return': arithmetic_mean,
                'geometric_mean_return': geometric_mean,
                'best_return': np.max(returns),
                'worst_return': np.min(returns),
                'positive_periods': np.sum(returns > 0),
                'negative_periods': np.sum(returns < 0),
                'win_rate': np.sum(returns > 0) / len(returns)
            }
            
        except Exception:
            return {
                'total_return': 0,
                'annualized_return': 0,
                'arithmetic_mean_return': 0,
                'geometric_mean_return': 0,
                'best_return': 0,
                'worst_return': 0,
                'positive_periods': 0,
                'negative_periods': 0,
                'win_rate': 0
            }
    
    def _calculate_risk_metrics(self, returns: np.ndarray, portfolio_values: Optional[np.ndarray] = None) -> Dict:
        """Calculate risk metrics"""
        try:
            # Volatility metrics
            volatility = np.std(returns)
            annualized_volatility = volatility * np.sqrt(self.trading_days_per_year)
            
            # Downside deviation
            downside_returns = returns[returns < 0]
            downside_deviation = np.std(downside_returns) if len(downside_returns) > 0 else 0
            annualized_downside_deviation = downside_deviation * np.sqrt(self.trading_days_per_year)
            
            # Semi-deviation (below mean)
            mean_return = np.mean(returns)
            below_mean_returns = returns[returns < mean_return] - mean_return
            semi_deviation = np.sqrt(np.mean(below_mean_returns ** 2)) if len(below_mean_returns) > 0 else 0
            
            # Value at Risk (VaR)
            var_95 = np.percentile(returns, 5)
            var_99 = np.percentile(returns, 1)
            
            # Conditional Value at Risk (Expected Shortfall)
            cvar_95 = np.mean(returns[returns <= var_95]) if np.any(returns <= var_95) else var_95
            cvar_99 = np.mean(returns[returns <= var_99]) if np.any(returns <= var_99) else var_99
            
            return {
                'volatility': annualized_volatility,
                'downside_deviation': annualized_downside_deviation,
                'semi_deviation': semi_deviation,
                'var_95': var_95,
                'var_99': var_99,
                'cvar_95': cvar_95,
                'cvar_99': cvar_99,
                'upside_capture': np.mean(returns[returns > 0]) if np.any(returns > 0) else 0,
                'downside_capture': np.mean(returns[returns < 0]) if np.any(returns < 0) else 0
            }
            
        except Exception:
            return {
                'volatility': 0,
                'downside_deviation': 0,
                'semi_deviation': 0,
                'var_95': 0,
                'var_99': 0,
                'cvar_95': 0,
                'cvar_99': 0,
                'upside_capture': 0,
                'downside_capture': 0
            }
    
    def _calculate_risk_adjusted_metrics(self, returns: np.ndarray) -> Dict:
        """Calculate risk-adjusted performance metrics"""
        try:
            daily_rf_rate = self.risk_free_rate / self.trading_days_per_year
            excess_returns = returns - daily_rf_rate
            
            # Sharpe Ratio
            sharpe_ratio = np.mean(excess_returns) / np.std(returns) if np.std(returns) > 0 else 0
            annualized_sharpe = sharpe_ratio * np.sqrt(self.trading_days_per_year)
            
            # Sortino Ratio
            downside_returns = returns[returns < daily_rf_rate]
            downside_std = np.std(downside_returns) if len(downside_returns) > 0 else np.std(returns)
            sortino_ratio = np.mean(excess_returns) / downside_std if downside_std > 0 else 0
            annualized_sortino = sortino_ratio * np.sqrt(self.trading_days_per_year)
            
            # Calmar Ratio (will be calculated in drawdown metrics if portfolio values available)
            calmar_ratio = 0
            
            # Information Ratio (excess return per unit of tracking error)
            # For now, use Sharpe as approximation
            information_ratio = sharpe_ratio
            
            # Treynor Ratio (need beta, will calculate as 0 for now)
            treynor_ratio = 0
            
            return {
                'sharpe_ratio': annualized_sharpe,
                'sortino_ratio': annualized_sortino,
                'calmar_ratio': calmar_ratio,
                'information_ratio': information_ratio,
                'treynor_ratio': treynor_ratio
            }
            
        except Exception:
            return {
                'sharpe_ratio': 0,
                'sortino_ratio': 0,
                'calmar_ratio': 0,
                'information_ratio': 0,
                'treynor_ratio': 0
            }
    
    def _calculate_benchmark_metrics(self, returns: np.ndarray, benchmark_returns: np.ndarray) -> Dict:
        """Calculate benchmark comparison metrics"""
        try:
            # Align return series
            min_length = min(len(returns), len(benchmark_returns))
            returns = returns[-min_length:]
            benchmark_returns = benchmark_returns[-min_length:]
            
            if min_length < 2:
                return {'beta': 1, 'alpha': 0, 'correlation': 0, 'tracking_error': 0}
            
            # Beta calculation
            covariance = np.cov(returns, benchmark_returns)[0, 1]
            benchmark_variance = np.var(benchmark_returns)
            beta = covariance / benchmark_variance if benchmark_variance > 0 else 1
            
            # Alpha calculation (Jensen's Alpha)
            daily_rf_rate = self.risk_free_rate / self.trading_days_per_year
            portfolio_excess = np.mean(returns) - daily_rf_rate
            benchmark_excess = np.mean(benchmark_returns) - daily_rf_rate
            alpha = portfolio_excess - (beta * benchmark_excess)
            annualized_alpha = alpha * self.trading_days_per_year
            
            # Correlation
            correlation = np.corrcoef(returns, benchmark_returns)[0, 1]
            
            # Tracking Error
            tracking_error = np.std(returns - benchmark_returns) * np.sqrt(self.trading_days_per_year)
            
            # R-squared
            r_squared = correlation ** 2
            
            # Up/Down Capture Ratios
            up_periods = benchmark_returns > 0
            down_periods = benchmark_returns < 0
            
            if np.any(up_periods):
                up_capture = np.mean(returns[up_periods]) / np.mean(benchmark_returns[up_periods])
            else:
                up_capture = 1
            
            if np.any(down_periods):
                down_capture = np.mean(returns[down_periods]) / np.mean(benchmark_returns[down_periods])
            else:
                down_capture = 1
            
            # Information Ratio
            excess_returns = returns - benchmark_returns
            information_ratio = np.mean(excess_returns) / np.std(excess_returns) if np.std(excess_returns) > 0 else 0
            annualized_ir = information_ratio * np.sqrt(self.trading_days_per_year)
            
            return {
                'beta': beta,
                'alpha': annualized_alpha,
                'correlation': correlation,
                'r_squared': r_squared,
                'tracking_error': tracking_error,
                'up_capture_ratio': up_capture,
                'down_capture_ratio': down_capture,
                'information_ratio': annualized_ir
            }
            
        except Exception:
            return {
                'beta': 1,
                'alpha': 0,
                'correlation': 0,
                'r_squared': 0,
                'tracking_error': 0,
                'up_capture_ratio': 1,
                'down_capture_ratio': 1,
                'information_ratio': 0
            }
    
    def _calculate_drawdown_metrics(self, portfolio_values: np.ndarray) -> Dict:
        """Calculate drawdown analysis metrics"""
        try:
            if len(portfolio_values) < 2:
                return {'max_drawdown': 0, 'calmar_ratio': 0}
            
            # Calculate running maximum (peak)
            peak = np.maximum.accumulate(portfolio_values)
            
            # Calculate drawdown
            drawdown = (portfolio_values - peak) / peak
            
            # Maximum drawdown
            max_drawdown = np.min(drawdown)
            
            # Average drawdown
            avg_drawdown = np.mean(drawdown[drawdown < 0]) if np.any(drawdown < 0) else 0
            
            # Drawdown duration analysis
            in_drawdown = drawdown < 0
            if np.any(in_drawdown):
                # Find drawdown periods
                drawdown_starts = np.where(np.diff(in_drawdown.astype(int)) == 1)[0] + 1
                drawdown_ends = np.where(np.diff(in_drawdown.astype(int)) == -1)[0] + 1
                
                # Handle edge cases
                if in_drawdown[0]:
                    drawdown_starts = np.concatenate([[0], drawdown_starts])
                if in_drawdown[-1]:
                    drawdown_ends = np.concatenate([drawdown_ends, [len(in_drawdown)]])
                
                if len(drawdown_starts) > 0 and len(drawdown_ends) > 0:
                    durations = drawdown_ends[:len(drawdown_starts)] - drawdown_starts
                    avg_recovery_time = np.mean(durations)
                    max_recovery_time = np.max(durations)
                else:
                    avg_recovery_time = 0
                    max_recovery_time = 0
            else:
                avg_recovery_time = 0
                max_recovery_time = 0
            
            # Calmar Ratio (annualized return / max drawdown)
            total_return = (portfolio_values[-1] / portfolio_values[0]) - 1
            periods = len(portfolio_values)
            annualized_return = (1 + total_return) ** (self.trading_days_per_year / periods) - 1
            calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown < 0 else 0
            
            # Pain Index (average drawdown over entire period)
            pain_index = np.mean(np.abs(drawdown))
            
            # Ulcer Index (RMS of drawdowns)
            ulcer_index = np.sqrt(np.mean(drawdown ** 2))
            
            return {
                'max_drawdown': abs(max_drawdown),
                'avg_drawdown': abs(avg_drawdown),
                'calmar_ratio': calmar_ratio,
                'pain_index': pain_index,
                'ulcer_index': ulcer_index,
                'avg_recovery_time': avg_recovery_time,
                'max_recovery_time': max_recovery_time,
                'drawdown_series': drawdown
            }
            
        except Exception:
            return {
                'max_drawdown': 0,
                'avg_drawdown': 0,
                'calmar_ratio': 0,
                'pain_index': 0,
                'ulcer_index': 0,
                'avg_recovery_time': 0,
                'max_recovery_time': 0
            }
    
    def _calculate_advanced_metrics(self, returns: np.ndarray) -> Dict:
        """Calculate advanced performance metrics"""
        try:
            # Omega Ratio (probability weighted ratio of gains vs losses)
            threshold = 0  # Can be customized
            gains = returns[returns > threshold] - threshold
            losses = threshold - returns[returns <= threshold]
            
            omega_ratio = np.sum(gains) / np.sum(losses) if np.sum(losses) > 0 else np.inf
            
            # Gain-to-Pain Ratio
            positive_returns = returns[returns > 0]
            negative_returns = returns[returns < 0]
            
            gain_to_pain = np.sum(positive_returns) / abs(np.sum(negative_returns)) if np.sum(negative_returns) < 0 else np.inf
            
            # Tail Ratio (95th percentile return / 5th percentile return)
            percentile_95 = np.percentile(returns, 95)
            percentile_5 = np.percentile(returns, 5)
            tail_ratio = percentile_95 / abs(percentile_5) if percentile_5 < 0 else percentile_95
            
            # Common Sense Ratio
            tail_ratio_cs = tail_ratio if tail_ratio > 0 else 1
            
            # Kelly Criterion (optimal position size)
            if len(positive_returns) > 0 and len(negative_returns) > 0:
                win_rate = len(positive_returns) / len(returns)
                avg_win = np.mean(positive_returns)
                avg_loss = abs(np.mean(negative_returns))
                
                if avg_loss > 0:
                    kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
                    kelly_fraction = max(0, min(1, kelly_fraction))  # Bound between 0 and 1
                else:
                    kelly_fraction = 0
            else:
                kelly_fraction = 0
            
            return {
                'omega_ratio': omega_ratio,
                'gain_to_pain_ratio': gain_to_pain,
                'tail_ratio': tail_ratio,
                'kelly_fraction': kelly_fraction
            }
            
        except Exception:
            return {
                'omega_ratio': 1,
                'gain_to_pain_ratio': 1,
                'tail_ratio': 1,
                'kelly_fraction': 0
            }
    
    def _calculate_higher_moments(self, returns: np.ndarray) -> Dict:
        """Calculate higher moment statistics"""
        try:
            # Skewness (asymmetry)
            skewness = stats.skew(returns)
            
            # Kurtosis (tail heaviness)
            kurtosis = stats.kurtosis(returns)
            
            # Jarque-Bera test for normality
            jb_stat, jb_pvalue = stats.jarque_bera(returns)
            
            return {
                'skewness': skewness,
                'kurtosis': kurtosis,
                'jarque_bera_stat': jb_stat,
                'jarque_bera_pvalue': jb_pvalue,
                'is_normal_distribution': jb_pvalue > 0.05
            }
            
        except Exception:
            return {
                'skewness': 0,
                'kurtosis': 0,
                'jarque_bera_stat': 0,
                'jarque_bera_pvalue': 1,
                'is_normal_distribution': True
            }
    
    def _calculate_tail_risk_metrics(self, returns: np.ndarray) -> Dict:
        """Calculate tail risk metrics"""
        try:
            # Expected Shortfall at different confidence levels
            var_90 = np.percentile(returns, 10)
            var_95 = np.percentile(returns, 5)
            var_99 = np.percentile(returns, 1)
            
            es_90 = np.mean(returns[returns <= var_90]) if np.any(returns <= var_90) else var_90
            es_95 = np.mean(returns[returns <= var_95]) if np.any(returns <= var_95) else var_95
            es_99 = np.mean(returns[returns <= var_99]) if np.any(returns <= var_99) else var_99
            
            # Maximum Consecutive Losses
            consecutive_losses = 0
            max_consecutive_losses = 0
            
            for ret in returns:
                if ret < 0:
                    consecutive_losses += 1
                    max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
                else:
                    consecutive_losses = 0
            
            return {
                'expected_shortfall_90': es_90,
                'expected_shortfall_95': es_95,
                'expected_shortfall_99': es_99,
                'max_consecutive_losses': max_consecutive_losses
            }
            
        except Exception:
            return {
                'expected_shortfall_90': 0,
                'expected_shortfall_95': 0,
                'expected_shortfall_99': 0,
                'max_consecutive_losses': 0
            }
    
    def _calculate_performance_consistency(self, returns: np.ndarray) -> Dict:
        """Calculate performance consistency metrics"""
        try:
            # Rolling period analysis (if enough data)
            if len(returns) >= 60:  # At least 60 periods
                window = 30  # 30-period rolling window
                rolling_returns = pd.Series(returns).rolling(window).sum()
                rolling_volatility = pd.Series(returns).rolling(window).std()
                
                consistency_ratio = np.std(rolling_returns.dropna()) / np.mean(np.abs(rolling_returns.dropna())) if len(rolling_returns.dropna()) > 0 else 0
                volatility_consistency = np.std(rolling_volatility.dropna()) / np.mean(rolling_volatility.dropna()) if len(rolling_volatility.dropna()) > 0 else 0
            else:
                consistency_ratio = 0
                volatility_consistency = 0
            
            # Percentage of positive periods
            positive_periods_pct = np.sum(returns > 0) / len(returns)
            
            # Standard deviation of returns
            return_stability = 1 / (1 + np.std(returns))  # Higher value = more stable
            
            return {
                'consistency_ratio': consistency_ratio,
                'volatility_consistency': volatility_consistency,
                'positive_periods_percentage': positive_periods_pct,
                'return_stability': return_stability
            }
            
        except Exception:
            return {
                'consistency_ratio': 0,
                'volatility_consistency': 0,
                'positive_periods_percentage': 0.5,
                'return_stability': 0.5
            }
    
    def _calculate_factor_analysis(self, returns: np.ndarray, benchmark_returns: np.ndarray) -> Dict:
        """Calculate factor analysis metrics"""
        try:
            min_length = min(len(returns), len(benchmark_returns))
            returns = returns[-min_length:]
            benchmark_returns = benchmark_returns[-min_length:]
            
            if min_length < 10:
                return {'factor_loading': 0, 'idiosyncratic_risk': 0}
            
            # Simple factor model: return = alpha + beta * market_return + error
            # Calculate beta (factor loading)
            beta = np.cov(returns, benchmark_returns)[0, 1] / np.var(benchmark_returns)
            
            # Calculate alpha
            alpha = np.mean(returns) - beta * np.mean(benchmark_returns)
            
            # Calculate idiosyncratic risk (tracking error)
            predicted_returns = alpha + beta * benchmark_returns
            residuals = returns - predicted_returns
            idiosyncratic_risk = np.std(residuals)
            
            # R-squared (explained variance)
            total_variance = np.var(returns)
            explained_variance = np.var(predicted_returns)
            r_squared = explained_variance / total_variance if total_variance > 0 else 0
            
            return {
                'factor_loading': beta,
                'factor_alpha': alpha,
                'idiosyncratic_risk': idiosyncratic_risk,
                'factor_r_squared': r_squared
            }
            
        except Exception:
            return {
                'factor_loading': 0,
                'factor_alpha': 0,
                'idiosyncratic_risk': 0,
                'factor_r_squared': 0
            }
    
    def _calculate_rolling_metrics(self, returns: np.ndarray, window: int = 60) -> Dict:
        """Calculate rolling performance metrics"""
        try:
            if len(returns) < window * 2:
                return {'rolling_sharpe_mean': 0, 'rolling_sharpe_std': 0}
            
            returns_series = pd.Series(returns)
            
            # Rolling Sharpe ratio
            daily_rf = self.risk_free_rate / self.trading_days_per_year
            rolling_excess_returns = returns_series.rolling(window).mean() - daily_rf
            rolling_volatility = returns_series.rolling(window).std()
            rolling_sharpe = rolling_excess_returns / rolling_volatility
            rolling_sharpe = rolling_sharpe.dropna()
            
            if len(rolling_sharpe) > 0:
                rolling_sharpe_mean = np.mean(rolling_sharpe)
                rolling_sharpe_std = np.std(rolling_sharpe)
                rolling_sharpe_min = np.min(rolling_sharpe)
                rolling_sharpe_max = np.max(rolling_sharpe)
            else:
                rolling_sharpe_mean = rolling_sharpe_std = rolling_sharpe_min = rolling_sharpe_max = 0
            
            # Rolling maximum drawdown
            portfolio_values = np.cumprod(1 + returns)
            rolling_max = pd.Series(portfolio_values).rolling(window).max()
            rolling_current = pd.Series(portfolio_values)
            rolling_dd = (rolling_current - rolling_max) / rolling_max
            rolling_max_dd = rolling_dd.rolling(window).min().dropna()
            
            if len(rolling_max_dd) > 0:
                avg_rolling_max_dd = np.mean(rolling_max_dd)
            else:
                avg_rolling_max_dd = 0
            
            return {
                'rolling_sharpe_mean': rolling_sharpe_mean,
                'rolling_sharpe_std': rolling_sharpe_std,
                'rolling_sharpe_min': rolling_sharpe_min,
                'rolling_sharpe_max': rolling_sharpe_max,
                'avg_rolling_max_drawdown': abs(avg_rolling_max_dd)
            }
            
        except Exception:
            return {
                'rolling_sharpe_mean': 0,
                'rolling_sharpe_std': 0,
                'rolling_sharpe_min': 0,
                'rolling_sharpe_max': 0,
                'avg_rolling_max_drawdown': 0
            }

