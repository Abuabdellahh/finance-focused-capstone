"""
Portfolio analysis and optimization.

This module provides functionality for portfolio analysis, optimization, and risk metrics calculation.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt


def calculate_portfolio_metrics(
    returns: pd.DataFrame,
    weights: Optional[np.ndarray] = None,
    risk_free_rate: float = 0.0,
    freq: int = 252
) -> Dict[str, float]:
    """
    Calculate key portfolio metrics.

    Args:
        returns: DataFrame of asset returns (each column is an asset)
        weights: Portfolio weights for each asset. If None, equal weights are used.
        risk_free_rate: Annual risk-free rate (default: 0.0)
        freq: Number of trading days in a year (default: 252)

    Returns:
        Dictionary containing portfolio metrics
    """
    if weights is None:
        weights = np.ones(returns.shape[1]) / returns.shape[1]
    
    if isinstance(weights, list):
        weights = np.array(weights)
    
    # Ensure weights sum to 1
    weights = weights / np.sum(weights)
    
    # Portfolio return (annualized)
    port_return = np.sum(returns.mean() * weights) * freq
    
    # Portfolio volatility (annualized)
    port_vol = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * freq, weights)))
    
    # Sharpe ratio
    sharpe_ratio = (port_return - risk_free_rate) / port_vol if port_vol > 0 else 0
    
    # Sortino ratio
    downside_returns = returns[returns < 0]
    downside_vol = np.sqrt(np.dot(weights.T, np.dot(downside_returns.cov() * freq, weights)))
    sortino_ratio = (port_return - risk_free_rate) / downside_vol if downside_vol > 0 else 0
    
    # Maximum Drawdown
    cum_returns = (1 + returns).cumprod()
    rolling_max = cum_returns.cummax()
    drawdowns = (cum_returns - rolling_max) / rolling_max
    max_drawdown = drawdowns.min()
    
    return {
        'return': port_return,
        'volatility': port_vol,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'max_drawdown': max_drawdown,
        'weights': dict(zip(returns.columns, weights))
    }


def optimize_portfolio(
    returns: pd.DataFrame,
    objective: str = 'sharpe',
    risk_free_rate: float = 0.0,
    freq: int = 252,
    constraints: Optional[List[dict]] = None,
    bounds: Optional[Tuple[float, float]] = (0, 1)
) -> Dict:
    """
    Optimize portfolio weights to maximize the specified objective function.

    Args:
        returns: DataFrame of asset returns
        objective: Optimization objective ('sharpe', 'sortino', 'min_vol', or 'max_return')
        risk_free_rate: Annual risk-free rate (default: 0.0)
        freq: Number of trading days in a year (default: 252)
        constraints: List of constraints for the optimization
        bounds: Bounds for asset weights (default: (0, 1))

    Returns:
        Dictionary containing optimization results
    """
    n_assets = returns.shape[1]
    
    # Default constraints: weights sum to 1
    if constraints is None:
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
    
    # Bounds for weights (default: no short selling)
    if bounds is not None:
        bounds = tuple(bounds for _ in range(n_assets))
    
    # Define objective function based on the specified metric
    if objective == 'sharpe':
        def objective_function(weights):
            metrics = calculate_portfolio_metrics(
                returns, weights, risk_free_rate, freq
            )
            return -metrics['sharpe_ratio']
    
    elif objective == 'sortino':
        def objective_function(weights):
            metrics = calculate_portfolio_metrics(
                returns, weights, risk_free_rate, freq
            )
            return -metrics['sortino_ratio']
    
    elif objective == 'min_vol':
        def objective_function(weights):
            metrics = calculate_portfolio_metrics(
                returns, weights, risk_free_rate, freq
            )
            return metrics['volatility']
    
    elif objective == 'max_return':
        def objective_function(weights):
            metrics = calculate_portfolio_metrics(
                returns, weights, risk_free_rate, freq
            )
            return -metrics['return']
    
    else:
        raise ValueError(f"Unsupported objective: {objective}")
    
    # Initial guess (equal weights)
    init_weights = np.ones(n_assets) / n_assets
    
    # Run optimization
    result = minimize(
        objective_function,
        init_weights,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    
    # Calculate metrics with optimal weights
    optimal_weights = result['x']
    metrics = calculate_portfolio_metrics(
        returns, optimal_weights, risk_free_rate, freq
    )
    
    return {
        'optimal_weights': optimal_weights,
        'metrics': metrics,
        'success': result['success'],
        'message': result['message']
    }


def calculate_efficient_frontier(
    returns: pd.DataFrame,
    n_points: int = 20,
    risk_free_rate: float = 0.0,
    freq: int = 252
) -> pd.DataFrame:
    """
    Calculate the efficient frontier for a set of assets.

    Args:
        returns: DataFrame of asset returns
        n_points: Number of points on the efficient frontier
        risk_free_rate: Annual risk-free rate (default: 0.0)
        freq: Number of trading days in a year (default: 252)

    Returns:
        DataFrame containing points on the efficient frontier
    """
    n_assets = returns.shape[1]
    
    # Calculate individual asset metrics
    asset_returns = returns.mean() * freq
    cov_matrix = returns.cov() * freq
    
    # Find minimum and maximum return portfolios
    min_return = asset_returns.min()
    max_return = asset_returns.max()
    
    # Generate target returns
    target_returns = np.linspace(min_return, max_return, n_points)
    
    # Store results
    frontier = []
    
    for target in target_returns:
        # Define constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Weights sum to 1
            {'type': 'eq', 'fun': lambda x, target=target: 
                np.sum(x * asset_returns) - target}  # Target return
        ]
        
        # Find minimum variance portfolio for target return
        result = minimize(
            lambda w: np.sqrt(np.dot(w.T, np.dot(cov_matrix, w))),
            x0=np.ones(n_assets) / n_assets,
            method='SLSQP',
            bounds=tuple((0, 1) for _ in range(n_assets)),
            constraints=constraints
        )
        
        if result.success:
            weights = result.x
            metrics = calculate_portfolio_metrics(
                returns, weights, risk_free_rate, freq
            )
            
            frontier.append({
                'return': metrics['return'],
                'volatility': metrics['volatility'],
                'sharpe_ratio': metrics['sharpe_ratio'],
                'weights': weights
            })
    
    return pd.DataFrame(frontier)


def calculate_risk_metrics(returns: pd.DataFrame, freq: int = 252) -> Dict:
    """
    Calculate various risk metrics for a portfolio.

    Args:
        returns: Series or DataFrame of portfolio returns
        freq: Number of trading days in a year (default: 252)

    Returns:
        Dictionary of risk metrics
    """
    if isinstance(returns, pd.DataFrame):
        returns = returns.iloc[:, 0]  # Use first column if DataFrame
    
    # Basic metrics
    annual_return = np.mean(returns) * freq
    annual_vol = np.std(returns) * np.sqrt(freq)
    
    # Drawdown calculations
    cum_returns = (1 + returns).cumprod()
    rolling_max = cum_returns.cummax()
    drawdowns = (cum_returns - rolling_max) / rolling_max
    max_drawdown = drawdowns.min()
    
    # Value at Risk (VaR) - Historical method at 95% confidence
    var_95 = np.percentile(returns, 5)
    
    # Expected Shortfall (CVaR) - Average of losses beyond VaR
    cvar_95 = returns[returns <= var_95].mean()
    
    # Skewness and Kurtosis
    skewness = returns.skew()
    kurtosis = returns.kurtosis()
    
    return {
        'annual_return': annual_return,
        'annual_volatility': annual_vol,
        'sharpe_ratio': annual_return / annual_vol if annual_vol > 0 else 0,
        'max_drawdown': max_drawdown,
        'var_95': var_95,
        'cvar_95': cvar_95,
        'skewness': skewness,
        'kurtosis': kurtosis,
        'sortino_ratio': annual_return / (np.std(returns[returns < 0]) * np.sqrt(freq)) 
                         if len(returns[returns < 0]) > 0 else 0
    }
