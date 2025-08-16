"""
Helper functions for financial calculations and data validation.
"""

from typing import Union, Optional, Tuple, Dict, Any
import numpy as np
import pandas as pd
from datetime import datetime, date


def validate_date(
    date_str: str, 
    date_format: str = '%Y-%m-%d'
) -> bool:
    """
    Validate if a date string is in the specified format.

    Args:
        date_str: Date string to validate
        date_format: Expected date format (default: 'YYYY-MM-DD')

    Returns:
        bool: True if date is valid, False otherwise
    """
    try:
        datetime.strptime(date_str, date_format)
        return True
    except ValueError:
        return False


def calculate_annualized_return(
    returns: Union[pd.Series, np.ndarray], 
    periods_per_year: int = 252
) -> float:
    """
    Calculate the annualized return from a series of returns.

    Args:
        returns: Series or array of periodic returns
        periods_per_year: Number of periods in a year (default: 252 for daily returns)

    Returns:
        float: Annualized return
    """
    if isinstance(returns, pd.Series):
        returns = returns.values
    
    if len(returns) == 0:
        return 0.0
    
    # Calculate cumulative return
    cum_return = np.prod(1 + returns) - 1
    
    # Annualize
    n_periods = len(returns)
    if n_periods == 0:
        return 0.0
    
    years = n_periods / periods_per_year
    if years <= 0:
        return 0.0
    
    annualized_return = (1 + cum_return) ** (1 / years) - 1
    return annualized_return


def calculate_annualized_volatility(
    returns: Union[pd.Series, np.ndarray], 
    periods_per_year: int = 252
) -> float:
    """
    Calculate the annualized volatility from a series of returns.

    Args:
        returns: Series or array of periodic returns
        periods_per_year: Number of periods in a year (default: 252 for daily returns)

    Returns:
        float: Annualized volatility
    """
    if isinstance(returns, pd.Series):
        returns = returns.values
    
    if len(returns) < 2:
        return 0.0
    
    return np.std(returns, ddof=1) * np.sqrt(periods_per_year)


def calculate_sharpe_ratio(
    returns: Union[pd.Series, np.ndarray], 
    risk_free_rate: float = 0.0, 
    periods_per_year: int = 252
) -> float:
    """
    Calculate the Sharpe ratio.

    Args:
        returns: Series or array of periodic returns
        risk_free_rate: Annual risk-free rate (default: 0.0)
        periods_per_year: Number of periods in a year (default: 252 for daily returns)

    Returns:
        float: Sharpe ratio
    """
    if isinstance(returns, pd.Series):
        returns = returns.values
    
    if len(returns) < 2:
        return 0.0
    
    # Calculate excess returns
    excess_returns = returns - (risk_free_rate / periods_per_year)
    
    # Calculate annualized return and volatility
    ann_return = calculate_annualized_return(returns, periods_per_year)
    ann_vol = calculate_annualized_volatility(returns, periods_per_year)
    
    # Handle zero volatility case
    if ann_vol == 0:
        return 0.0
    
    return (ann_return - risk_free_rate) / ann_vol


def calculate_sortino_ratio(
    returns: Union[pd.Series, np.ndarray], 
    risk_free_rate: float = 0.0, 
    periods_per_year: int = 252,
    target_return: float = 0.0
) -> float:
    """
    Calculate the Sortino ratio.

    Args:
        returns: Series or array of periodic returns
        risk_free_rate: Annual risk-free rate (default: 0.0)
        periods_per_year: Number of periods in a year (default: 252 for daily returns)
        target_return: Target return for downside deviation (default: 0.0)

    Returns:
        float: Sortino ratio
    """
    if isinstance(returns, pd.Series):
        returns = returns.values
    
    if len(returns) < 2:
        return 0.0
    
    # Calculate excess returns
    excess_returns = returns - (risk_free_rate / periods_per_year)
    
    # Calculate annualized return
    ann_return = calculate_annualized_return(returns, periods_per_year)
    
    # Calculate downside deviation
    downside_returns = returns[returns < target_return] - target_return
    
    if len(downside_returns) == 0:
        return float('inf') if ann_return > risk_free_rate else 0.0
    
    downside_deviation = np.std(downside_returns, ddof=1) * np.sqrt(periods_per_year)
    
    # Handle zero downside deviation
    if downside_deviation == 0:
        return float('inf') if ann_return > risk_free_rate else 0.0
    
    return (ann_return - risk_free_rate) / downside_deviation


def calculate_max_drawdown(
    returns: Union[pd.Series, np.ndarray],
    return_duration: bool = False
) -> Union[float, Tuple[float, int]]:
    """
    Calculate the maximum drawdown and optionally its duration.

    Args:
        returns: Series or array of periodic returns
        return_duration: Whether to also return the duration of the max drawdown

    Returns:
        float or tuple: Maximum drawdown (and optionally duration in periods)
    """
    if isinstance(returns, pd.Series):
        cum_returns = (1 + returns).cumprod()
    else:
        cum_returns = np.cumprod(1 + np.asarray(returns))
    
    if len(cum_returns) == 0:
        return (0.0, 0) if return_duration else 0.0
    
    # Calculate running maximum
    running_max = np.maximum.accumulate(cum_returns)
    
    # Calculate drawdown series
    drawdowns = (cum_returns - running_max) / running_max
    
    # Find maximum drawdown
    max_dd = drawdowns.min()
    
    if not return_duration:
        return max_dd
    
    # Calculate duration of maximum drawdown
    if np.isnan(max_dd):
        return max_dd, 0
    
    # Find the start and end points of the maximum drawdown
    max_dd_idx = np.argmin(drawdowns)
    peak_before = np.argmax(cum_returns[:max_dd_idx + 1])
    
    duration = max_dd_idx - peak_before
    
    return max_dd, duration


def calculate_beta(
    asset_returns: Union[pd.Series, np.ndarray],
    market_returns: Union[pd.Series, np.ndarray],
    risk_free_rate: float = 0.0
) -> float:
    """
    Calculate the beta of an asset relative to the market.

    Args:
        asset_returns: Returns of the asset
        market_returns: Returns of the market/benchmark
        risk_free_rate: Risk-free rate (default: 0.0)

    Returns:
        float: Beta coefficient
    """
    if isinstance(asset_returns, pd.Series):
        asset_returns = asset_returns.values
    if isinstance(market_returns, pd.Series):
        market_returns = market_returns.values
    
    # Ensure same length
    min_len = min(len(asset_returns), len(market_returns))
    if min_len < 2:
        return 0.0
    
    asset_returns = asset_returns[:min_len]
    market_returns = market_returns[:min_len]
    
    # Calculate covariance and variance
    cov_matrix = np.cov(asset_returns, market_returns, ddof=1)
    market_variance = np.var(market_returns, ddof=1)
    
    # Handle division by zero
    if market_variance == 0:
        return 0.0
    
    beta = cov_matrix[0, 1] / market_variance
    return beta


def calculate_alpha(
    asset_returns: Union[pd.Series, np.ndarray],
    market_returns: Union[pd.Series, np.ndarray],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Jensen's alpha of an asset relative to the market.

    Args:
        asset_returns: Returns of the asset
        market_returns: Returns of the market/benchmark
        risk_free_rate: Annual risk-free rate (default: 0.0)
        periods_per_year: Number of periods in a year (default: 252 for daily returns)

    Returns:
        float: Alpha (annualized)
    """
    if isinstance(asset_returns, pd.Series):
        asset_returns = asset_returns.values
    if isinstance(market_returns, pd.Series):
        market_returns = market_returns.values
    
    # Ensure same length
    min_len = min(len(asset_returns), len(market_returns))
    if min_len < 2:
        return 0.0
    
    asset_returns = asset_returns[:min_len]
    market_returns = market_returns[:min_len]
    
    # Calculate beta
    beta = calculate_beta(asset_returns, market_returns, risk_free_rate)
    
    # Calculate average returns
    asset_avg_return = np.mean(asset_returns) * periods_per_year
    market_avg_return = np.mean(market_returns) * periods_per_year
    
    # Calculate alpha (annualized)
    alpha = asset_avg_return - (risk_free_rate + beta * (market_avg_return - risk_free_rate))
    
    return alpha


def calculate_tracking_error(
    portfolio_returns: Union[pd.Series, np.ndarray],
    benchmark_returns: Union[pd.Series, np.ndarray],
    periods_per_year: int = 252
) -> float:
    """
    Calculate the tracking error between a portfolio and its benchmark.

    Args:
        portfolio_returns: Returns of the portfolio
        benchmark_returns: Returns of the benchmark
        periods_per_year: Number of periods in a year (default: 252 for daily returns)

    Returns:
        float: Annualized tracking error
    """
    if isinstance(portfolio_returns, pd.Series):
        portfolio_returns = portfolio_returns.values
    if isinstance(benchmark_returns, pd.Series):
        benchmark_returns = benchmark_returns.values
    
    # Ensure same length
    min_len = min(len(portfolio_returns), len(benchmark_returns))
    if min_len < 2:
        return 0.0
    
    portfolio_returns = portfolio_returns[:min_len]
    benchmark_returns = benchmark_returns[:min_len]
    
    # Calculate active returns
    active_returns = portfolio_returns - benchmark_returns
    
    # Calculate tracking error (annualized standard deviation of active returns)
    tracking_error = np.std(active_returns, ddof=1) * np.sqrt(periods_per_year)
    
    return tracking_error
