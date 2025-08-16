"""
Data processing module for the finance-focused capstone project.

This module contains functionality for loading and processing financial data.
"""

from .loader import load_financial_data
from .processor import process_financial_data

__all__ = ['load_financial_data', 'process_financial_data']
