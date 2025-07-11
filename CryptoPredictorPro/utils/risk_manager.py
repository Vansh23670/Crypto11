import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class RiskManager:
    """
    Advanced risk management system with VaR calculations, stress testing,
    and dynamic position sizing using modern portfolio theory.
    """
    
    def __init__(self):
        self.confidence_levels = [0.95, 0.99, 0.999]
        self.risk_free_rate = 0.02  # 2% annual risk-free rate
    
    def comprehensive_risk_analysis(self, holdings: Dict[str, float], data_fetcher,
                                  analysis_days: int = 90, confidence_level: float = 0.95) -> Dict:
        """
        Perform comprehensive risk analysis on portfolio holdings.
        
        Args:
            holdings: Dictionary of {symbol: quantity}
            data_fetcher: CryptoDataFetcher instance
            analysis_days: Number of days for historical analysis
            confidence_level: Confidence level for VaR calculations
            
        Returns:
            Dictionary with comprehensive risk metrics
        """
        try:
            # Get portfolio data
            portfolio_data = self._get_portfolio_data(holdings, data_fetcher, analysis_days)
            
            if not portfolio_data:
                return {'overall_risk_score': 5, 'error': 'No data available'}
            
            # Calculate individual asset risks
            asset_risks = {}
            portfolio_returns = []
            
            for symbol, data in portfolio_data.items():
                if 'returns' in data and len(data['returns']) > 0:
                    asset_risks[symbol] = self._calculate_asset_risk_metrics(data['returns'], confidence_level)
            
            # Calculate portfolio-level metrics
            portfolio_metrics = self._calculate_portfolio_risk_metrics(
                portfolio_data, holdings, confidence_level
            )
            
            # Calculate overall risk score (1-10 scale)
            overall_risk_score = self._calculate_overall_risk_score(portfolio_metrics, asset_risks)
            
            return {
                'overall_risk_score': overall_risk_score,
                'portfolio_beta': portfolio_metrics.get('beta', 1.0),
                'volatility': portfolio_metrics.get('volatility', 0.0),
                'var_1d': portfolio_metrics.get('var_1d', 0.0),
                'var_1w': portfolio_metrics.get('var_1w', 0.0),
                'var_1m': portfolio_metrics.get('var_1m', 0.0),
                'max_drawdown': portfolio_metrics.get('max_drawdown', 0.0),
                'sharpe_ratio': portfolio_metrics.get('sharpe_ratio', 0.0),
                'sortino_ratio': portfolio_metrics.get('sortino_ratio', 0.0),
                'var_metrics': portfolio_metrics.get('var_metrics', {}),
                'correlation_matrix': portfolio_metrics.get('correlation_matrix', {}),
                'risk_breakdown': self._calculate_risk_contribution(holdings, asset_risks),
                'asset_risks': asset_risks
            }
            
        except Exception as e:
            return {'overall_risk_score': 8, 'error': str(e)}
    
    def _get_portfolio_data(self, holdings: Dict[str, float], data_fetcher, 
                           analysis_days: int) -> Dict:
        """Get historical data for all holdings"""
        portfolio_data = {}
        
        for symbol, quantity in holdings.items():
            if quantity > 0:
                try:
                    # Get historical data
                    historical_data = data_fetcher.get_historical_data(symbol, 'usd', days=analysis_days)
                    
                    if not historical_data.empty and len(historical_data) > 1:
                        # Calculate returns
                        returns = historical_data['price'].pct_change().dropna()
                        
                        # Get current price and value
                        current_price = data_fetcher.get_current_price(symbol, 'usd')['current_price']
                        position_value = quantity * current_price
                        
                        portfolio_data[symbol] = {
                            'returns': returns.values,
                            'prices': historical_data['price'].values,
                            'timestamps': historical_data['timestamp'].values,
                            'current_price': current_price,
                            'position_value': position_value,
                            'quantity': quantity
                        }
                        
                except Exception as e:
                    continue
        
        return portfolio_data
    
    def _calculate_asset_risk_metrics(self, returns: np.ndarray, confidence_level: float) -> Dict:
        """Calculate risk metrics for individual asset"""
        if len(returns) < 2:
            return {}
        
        # Basic statistics
        mean_return = np.mean(returns)
        volatility = np.std(returns)
        
        # Value at Risk
        var_95 = np.percentile(returns, (1 - 0.95) * 100)
        var_99 = np.percentile(returns, (1 - 0.99) * 100)
        
        # Expected Shortfall (Conditional VaR)
        var_threshold = np.percentile(returns, (1 - confidence_level) * 100)
        expected_shortfall = np.mean(returns[returns <= var_threshold])
        
        # Maximum Drawdown
        cumulative_returns = np.cumprod(1 + returns)
        peak = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - peak) / peak
        max_drawdown = np.min(drawdown)
        
        # Downside deviation
        downside_returns = returns[returns < 0]
        downside_deviation = np.std(downside_returns) if len(downside_returns) > 0 else 0
        
        # Skewness and Kurtosis
        skewness = stats.skew(returns)
        kurtosis = stats.kurtosis(returns)
        
        return {
            'mean_return': mean_return,
            'volatility': volatility,
            'var_95': var_95,
            'var_99': var_99,
            'expected_shortfall': expected_shortfall,
            'max_drawdown': max_drawdown,
            'downside_deviation': downside_deviation,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'sharpe_ratio': (mean_return - self.risk_free_rate/252) / volatility if volatility > 0 else 0,
            'sortino_ratio': (mean_return - self.risk_free_rate/252) / downside_deviation if downside_deviation > 0 else 0
        }
    
    def _calculate_portfolio_risk_metrics(self, portfolio_data: Dict, holdings: Dict, 
                                        confidence_level: float) -> Dict:
        """Calculate portfolio-level risk metrics"""
        if not portfolio_data:
            return {}
        
        # Calculate portfolio weights
        total_value = sum(data['position_value'] for data in portfolio_data.values())
        weights = {symbol: data['position_value'] / total_value 
                  for symbol, data in portfolio_data.items()}
        
        # Calculate portfolio returns
        portfolio_returns = self._calculate_weighted_portfolio_returns(portfolio_data, weights)
        
        if len(portfolio_returns) < 2:
            return {}
        
        # Portfolio risk metrics
        portfolio_volatility = np.std(portfolio_returns)
        portfolio_mean_return = np.mean(portfolio_returns)
        
        # Portfolio VaR
        var_1d = np.percentile(portfolio_returns, (1 - confidence_level) * 100)
        var_1w = var_1d * np.sqrt(7)  # Scale to weekly
        var_1m = var_1d * np.sqrt(30)  # Scale to monthly
        
        # Convert to dollar amounts
        var_1d_dollar = abs(var_1d * total_value)
        var_1w_dollar = abs(var_1w * total_value)
        var_1m_dollar = abs(var_1m * total_value)
        
        # Expected Shortfall
        var_threshold = np.percentile(portfolio_returns, (1 - confidence_level) * 100)
        expected_shortfall = np.mean(portfolio_returns[portfolio_returns <= var_threshold])
        
        # Portfolio Beta (vs Bitcoin as market proxy)
        portfolio_beta = self._calculate_portfolio_beta(portfolio_data, weights)
        
        # Sharpe and Sortino ratios
        excess_returns = portfolio_returns - (self.risk_free_rate / 252)
        sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) if np.std(excess_returns) > 0 else 0
        
        downside_returns = excess_returns[excess_returns < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 0 else np.std(excess_returns)
        sortino_ratio = np.mean(excess_returns) / downside_std if downside_std > 0 else 0
        
        # Maximum Drawdown
        cumulative_returns = np.cumprod(1 + portfolio_returns)
        peak = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - peak) / peak
        max_drawdown = abs(np.min(drawdown))
        
        # Correlation matrix
        correlation_matrix = self._calculate_correlation_matrix(portfolio_data)
        
        return {
            'volatility': portfolio_volatility,
            'var_1d': var_1d_dollar,
            'var_1w': var_1w_dollar,
            'var_1m': var_1m_dollar,
            'expected_shortfall': abs(expected_shortfall * total_value),
            'beta': portfolio_beta,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'correlation_matrix': correlation_matrix,
            'var_metrics': {
                'var_95_1d': var_1d_dollar,
                'var_99_1d': abs(np.percentile(portfolio_returns, 1) * total_value),
                'expected_shortfall': abs(expected_shortfall * total_value)
            },
            'portfolio_returns': portfolio_returns
        }
    
    def _calculate_weighted_portfolio_returns(self, portfolio_data: Dict, weights: Dict) -> np.ndarray:
        """Calculate weighted portfolio returns"""
        if not portfolio_data or not weights:
            return np.array([])
        
        # Find the minimum length among all return series
        min_length = min(len(data['returns']) for data in portfolio_data.values())
        
        if min_length == 0:
            return np.array([])
        
        # Calculate weighted returns
        portfolio_returns = np.zeros(min_length)
        
        for symbol, data in portfolio_data.items():
            if symbol in weights:
                # Take the last min_length returns to align all series
                asset_returns = data['returns'][-min_length:]
                portfolio_returns += weights[symbol] * asset_returns
        
        return portfolio_returns
    
    def _calculate_portfolio_beta(self, portfolio_data: Dict, weights: Dict) -> float:
        """Calculate portfolio beta vs market (Bitcoin)"""
        try:
            # Use Bitcoin as market proxy if available
            if 'bitcoin' in portfolio_data:
                market_returns = portfolio_data['bitcoin']['returns']
                portfolio_returns = self._calculate_weighted_portfolio_returns(portfolio_data, weights)
                
                if len(market_returns) > 0 and len(portfolio_returns) > 0:
                    # Align lengths
                    min_length = min(len(market_returns), len(portfolio_returns))
                    market_returns = market_returns[-min_length:]
                    portfolio_returns = portfolio_returns[-min_length:]
                    
                    # Calculate beta
                    covariance = np.cov(portfolio_returns, market_returns)[0, 1]
                    market_variance = np.var(market_returns)
                    
                    if market_variance > 0:
                        return covariance / market_variance
            
            return 1.0  # Default beta
            
        except Exception:
            return 1.0
    
    def _calculate_correlation_matrix(self, portfolio_data: Dict) -> Dict:
        """Calculate correlation matrix between assets"""
        symbols = list(portfolio_data.keys())
        if len(symbols) < 2:
            return {}
        
        # Find minimum length
        min_length = min(len(data['returns']) for data in portfolio_data.values())
        
        # Create returns matrix
        returns_matrix = np.zeros((len(symbols), min_length))
        for i, symbol in enumerate(symbols):
            returns_matrix[i] = portfolio_data[symbol]['returns'][-min_length:]
        
        # Calculate correlation matrix
        correlation_matrix = np.corrcoef(returns_matrix)
        
        # Convert to dictionary
        corr_dict = {}
        for i, symbol1 in enumerate(symbols):
            corr_dict[symbol1] = {}
            for j, symbol2 in enumerate(symbols):
                corr_dict[symbol1][symbol2] = correlation_matrix[i, j]
        
        return corr_dict
    
    def _calculate_overall_risk_score(self, portfolio_metrics: Dict, asset_risks: Dict) -> float:
        """Calculate overall portfolio risk score (1-10 scale)"""
        score = 5.0  # Start with medium risk
        
        try:
            # Volatility factor (0-3 points)
            volatility = portfolio_metrics.get('volatility', 0)
            if volatility > 0.05:  # > 5% daily volatility
                score += 3
            elif volatility > 0.03:  # > 3% daily volatility
                score += 2
            elif volatility > 0.02:  # > 2% daily volatility
                score += 1
            
            # Max drawdown factor (0-2 points)
            max_drawdown = portfolio_metrics.get('max_drawdown', 0)
            if max_drawdown > 0.3:  # > 30% drawdown
                score += 2
            elif max_drawdown > 0.2:  # > 20% drawdown
                score += 1
            
            # Sharpe ratio factor (-2 to 0 points)
            sharpe_ratio = portfolio_metrics.get('sharpe_ratio', 0)
            if sharpe_ratio > 1:
                score -= 2
            elif sharpe_ratio > 0.5:
                score -= 1
            
            # Diversification factor (0-1 point)
            if len(asset_risks) < 3:
                score += 1  # Lack of diversification
            
            # Ensure score is within bounds
            score = max(1, min(10, score))
            
        except Exception:
            score = 5.0
        
        return score
    
    def _calculate_risk_contribution(self, holdings: Dict, asset_risks: Dict) -> Dict[str, float]:
        """Calculate risk contribution by asset"""
        if not asset_risks:
            return {}
        
        # Calculate total portfolio value
        total_value = sum(holdings.values())
        if total_value == 0:
            return {}
        
        # Risk contribution based on position size and volatility
        risk_contributions = {}
        total_risk = 0
        
        for symbol in holdings.keys():
            if symbol in asset_risks:
                weight = holdings[symbol] / total_value
                volatility = asset_risks[symbol].get('volatility', 0)
                contribution = weight * volatility
                risk_contributions[symbol] = contribution
                total_risk += contribution
        
        # Normalize to percentages
        if total_risk > 0:
            for symbol in risk_contributions:
                risk_contributions[symbol] = (risk_contributions[symbol] / total_risk) * 100
        
        return risk_contributions
    
    def stress_test_portfolio(self, holdings: Dict[str, float], data_fetcher, 
                            shock_magnitude: float = -0.30) -> Dict:
        """
        Perform stress testing on portfolio with various shock scenarios.
        
        Args:
            holdings: Portfolio holdings
            data_fetcher: CryptoDataFetcher instance
            shock_magnitude: Magnitude of price shock (e.g., -0.30 for 30% decline)
            
        Returns:
            Dictionary with stress test results
        """
        try:
            # Get current portfolio value
            current_portfolio_value = 0
            position_impacts = {}
            
            for symbol, quantity in holdings.items():
                if quantity > 0:
                    try:
                        current_price = data_fetcher.get_current_price(symbol, 'usd')['current_price']
                        current_value = quantity * current_price
                        current_portfolio_value += current_value
                        
                        # Apply stress shock
                        stressed_price = current_price * (1 + shock_magnitude)
                        stressed_value = quantity * stressed_price
                        loss = current_value - stressed_value
                        loss_pct = (loss / current_value) * 100 if current_value > 0 else 0
                        
                        position_impacts[symbol] = {
                            'current_value': current_value,
                            'stressed_value': stressed_value,
                            'loss': loss,
                            'loss_pct': loss_pct
                        }
                        
                    except Exception:
                        continue
            
            # Calculate total stressed value
            stressed_portfolio_value = sum(impact['stressed_value'] for impact in position_impacts.values())
            
            return {
                'current_portfolio_value': current_portfolio_value,
                'stressed_portfolio_value': stressed_portfolio_value,
                'total_loss': current_portfolio_value - stressed_portfolio_value,
                'loss_percentage': ((current_portfolio_value - stressed_portfolio_value) / current_portfolio_value * 100) if current_portfolio_value > 0 else 0,
                'position_impacts': position_impacts,
                'shock_magnitude': shock_magnitude * 100  # Convert to percentage
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def calculate_portfolio_risk(self, holdings: Dict[str, float], data_fetcher) -> Dict:
        """
        Calculate basic portfolio risk metrics for sidebar display.
        
        Args:
            holdings: Portfolio holdings
            data_fetcher: CryptoDataFetcher instance
            
        Returns:
            Dictionary with basic risk metrics
        """
        try:
            # Get portfolio data (shorter period for speed)
            portfolio_data = self._get_portfolio_data(holdings, data_fetcher, analysis_days=30)
            
            if not portfolio_data:
                return {'beta': 1.0, 'volatility': 0.0, 'var_95': 0.0, 'sharpe_ratio': 0.0}
            
            # Calculate weights
            total_value = sum(data['position_value'] for data in portfolio_data.values())
            weights = {symbol: data['position_value'] / total_value 
                      for symbol, data in portfolio_data.items()}
            
            # Calculate portfolio returns
            portfolio_returns = self._calculate_weighted_portfolio_returns(portfolio_data, weights)
            
            if len(portfolio_returns) < 2:
                return {'beta': 1.0, 'volatility': 0.0, 'var_95': 0.0, 'sharpe_ratio': 0.0}
            
            # Basic metrics
            volatility = np.std(portfolio_returns)
            var_95 = abs(np.percentile(portfolio_returns, 5) * total_value)
            beta = self._calculate_portfolio_beta(portfolio_data, weights)
            
            # Sharpe ratio
            excess_returns = portfolio_returns - (self.risk_free_rate / 252)
            sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) if np.std(excess_returns) > 0 else 0
            
            # Risk breakdown by asset
            risk_breakdown = {}
            for symbol, data in portfolio_data.items():
                weight = weights[symbol]
                asset_vol = np.std(data['returns']) if len(data['returns']) > 0 else 0
                risk_breakdown[symbol] = weight * asset_vol * 100
            
            return {
                'beta': beta,
                'volatility': volatility,
                'var_95': var_95,
                'sharpe_ratio': sharpe_ratio,
                'risk_breakdown': risk_breakdown
            }
            
        except Exception as e:
            return {'beta': 1.0, 'volatility': 0.0, 'var_95': 0.0, 'sharpe_ratio': 0.0, 'error': str(e)}
    
    def generate_optimization_recommendations(self, holdings: Dict, portfolio_value: float,
                                            risk_analysis: Dict, max_position_pct: float,
                                            target_risk: str) -> Dict:
        """
        Generate portfolio optimization recommendations based on risk analysis.
        
        Args:
            holdings: Current portfolio holdings
            portfolio_value: Total portfolio value
            risk_analysis: Risk analysis results
            max_position_pct: Maximum position size percentage
            target_risk: Target risk level ('Conservative', 'Moderate', 'Aggressive')
            
        Returns:
            Dictionary with optimization recommendations
        """
        alerts = []
        suggestions = []
        recommended_allocation = {}
        
        try:
            # Risk level mapping
            risk_thresholds = {
                'Conservative': {'max_vol': 0.02, 'max_var': 0.05, 'max_position': 15},
                'Moderate': {'max_vol': 0.04, 'max_var': 0.10, 'max_position': 25},
                'Aggressive': {'max_vol': 0.06, 'max_var': 0.15, 'max_position': 40}
            }
            
            thresholds = risk_thresholds.get(target_risk, risk_thresholds['Moderate'])
            
            # Check overall risk
            overall_risk = risk_analysis.get('overall_risk_score', 5)
            if overall_risk > 7:
                alerts.append({
                    'type': 'error',
                    'message': 'Portfolio risk is very high - consider reducing exposure'
                })
            elif overall_risk > 5:
                alerts.append({
                    'type': 'warning',
                    'message': 'Portfolio risk is elevated - monitor positions closely'
                })
            
            # Check volatility
            volatility = risk_analysis.get('volatility', 0)
            if volatility > thresholds['max_vol']:
                alerts.append({
                    'type': 'warning',
                    'message': f'Portfolio volatility ({volatility:.1%}) exceeds {target_risk.lower()} target'
                })
                suggestions.append('Consider diversifying into less volatile assets')
            
            # Check position sizes
            for symbol, quantity in holdings.items():
                if quantity > 0:
                    try:
                        # Calculate position percentage (simplified)
                        position_pct = (quantity / portfolio_value) * 100  # Approximation
                        
                        if position_pct > max_position_pct:
                            alerts.append({
                                'type': 'warning',
                                'message': f'{symbol.upper()} position ({position_pct:.1f}%) exceeds maximum ({max_position_pct}%)'
                            })
                            suggestions.append(f'Reduce {symbol.upper()} position to below {max_position_pct}%')
                            recommended_allocation[symbol] = max_position_pct
                        else:
                            recommended_allocation[symbol] = position_pct
                    except Exception:
                        continue
            
            # Check diversification
            num_positions = len([q for q in holdings.values() if q > 0])
            if num_positions < 3:
                suggestions.append('Consider adding more positions for better diversification')
            elif num_positions > 10:
                suggestions.append('Consider consolidating positions to reduce complexity')
            
            # VaR check
            var_95 = risk_analysis.get('var_1d', 0)
            var_pct = (var_95 / portfolio_value) if portfolio_value > 0 else 0
            if var_pct > thresholds['max_var']:
                alerts.append({
                    'type': 'warning',
                    'message': f'Daily VaR ({var_pct:.1%}) is high for {target_risk.lower()} risk profile'
                })
            
            # Generate general suggestions
            if not suggestions:
                suggestions.append('Portfolio allocation appears reasonable for current risk profile')
            
            return {
                'alerts': alerts,
                'suggestions': suggestions,
                'recommended_allocation': recommended_allocation,
                'target_risk_level': target_risk,
                'current_risk_score': overall_risk
            }
            
        except Exception as e:
            return {
                'alerts': [{'type': 'error', 'message': f'Optimization analysis failed: {str(e)}'}],
                'suggestions': ['Unable to generate recommendations due to data issues'],
                'recommended_allocation': {}
            }
    
    def kelly_criterion_sizing(self, holdings: Dict, data_fetcher) -> Dict:
        """
        Calculate optimal position sizes using Kelly Criterion.
        
        Args:
            holdings: Current portfolio holdings
            data_fetcher: CryptoDataFetcher instance
            
        Returns:
            Dictionary with Kelly optimal position sizes
        """
        kelly_results = {}
        
        try:
            for symbol, quantity in holdings.items():
                if quantity > 0:
                    try:
                        # Get historical data for win rate calculation
                        historical_data = data_fetcher.get_historical_data(symbol, 'usd', days=90)
                        
                        if len(historical_data) > 1:
                            returns = historical_data['price'].pct_change().dropna()
                            
                            # Calculate win rate and average win/loss
                            wins = returns[returns > 0]
                            losses = returns[returns < 0]
                            
                            win_rate = len(wins) / len(returns) if len(returns) > 0 else 0.5
                            avg_win = np.mean(wins) if len(wins) > 0 else 0
                            avg_loss = abs(np.mean(losses)) if len(losses) > 0 else 0
                            
                            # Kelly formula: f = (bp - q) / b
                            # where b = avg_win/avg_loss, p = win_rate, q = 1-win_rate
                            if avg_loss > 0:
                                b = avg_win / avg_loss
                                p = win_rate
                                q = 1 - win_rate
                                kelly_fraction = (b * p - q) / b
                                
                                # Cap Kelly fraction to reasonable limits (max 25%)
                                kelly_fraction = max(0, min(0.25, kelly_fraction))
                                kelly_percentage = kelly_fraction * 100
                                
                                # Current position size (approximation)
                                current_size = 10  # Placeholder - would need portfolio value calculation
                                
                                # Generate recommendation
                                if kelly_percentage > current_size * 1.2:
                                    recommendation = 'INCREASE'
                                elif kelly_percentage < current_size * 0.8:
                                    recommendation = 'DECREASE'
                                else:
                                    recommendation = 'HOLD'
                                
                                kelly_results[symbol] = {
                                    'current_size': current_size,
                                    'kelly_size': kelly_percentage,
                                    'win_rate': win_rate * 100,
                                    'avg_win': avg_win * 100,
                                    'avg_loss': avg_loss * 100,
                                    'recommendation': recommendation
                                }
                    except Exception:
                        continue
            
            return kelly_results
            
        except Exception:
            return {}
