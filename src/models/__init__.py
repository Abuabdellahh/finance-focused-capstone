"""
Financial models for the finance-focused capstone project.

This module contains implementations of various financial models and analyses.
"""

from .portfolio import (
    calculate_portfolio_metrics,
    optimize_portfolio,
    calculate_efficient_frontier,
    calculate_risk_metrics
)

__all__ = [
    'calculate_portfolio_metrics',
    'optimize_portfolio',
    'calculate_efficient_frontier',
    'calculate_risk_metrics'
]
