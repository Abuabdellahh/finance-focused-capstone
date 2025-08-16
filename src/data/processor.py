"""
Data processing utilities for financial data.

This module provides functions to clean, transform, and process financial data.
"""

from typing import Dict, List, Optional, Union
import pandas as pd
import numpy as np
from pathlib import Path


def process_financial_data(
    data: pd.DataFrame,
    tickers: Optional[Union[str, List[str]]] = None,
    fill_method: str = 'ffill',
    min_data_points: int = 10,
    output_dir: Optional[Union[str, Path]] = None,
    save_processed: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    Process raw financial data by cleaning, handling missing values, and calculating returns.

    Args:
        data: Raw financial data from load_financial_data
        tickers: List of tickers to process. If None, will try to infer from data.
        fill_method: Method to fill missing values ('ffill', 'bfill', 'interpolate')
        min_data_points: Minimum number of data points required for each ticker
        output_dir: Directory to save processed data
        save_processed: Whether to save processed data to disk

    Returns:
        Dictionary of processed DataFrames for each ticker
    """
    if tickers is None:
        # Try to infer tickers from the data structure
        if isinstance(data.columns, pd.MultiIndex):
            tickers = list(set([col[0] for col in data.columns if col[0] != '']))
        else:
            tickers = ['SINGLE_TICKER']
    
    if isinstance(tickers, str):
        tickers = [tickers]
    
    processed_data = {}
    
    for ticker in tickers:
        try:
            # Handle multi-index columns (OHLCV data)
            if isinstance(data.columns, pd.MultiIndex):
                df = data[ticker].copy()
            else:
                df = data.copy()
            
            # Basic cleaning
            df = _clean_data(df, fill_method, min_data_points)
            
            # Calculate returns and other financial metrics
            df = _calculate_financial_metrics(df)
            
            processed_data[ticker] = df
            
            # Save processed data
            if save_processed:
                if output_dir is None:
                    output_dir = Path(__file__).parent.parent.parent / 'data' / 'processed'
                else:
                    output_dir = Path(output_dir)
                
                output_dir.mkdir(parents=True, exist_ok=True)
                output_file = output_dir / f"{ticker}_processed.parquet"
                df.to_parquet(output_file)
                print(f"Processed data saved to {output_file}")
                
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
    
    return processed_data


def _clean_data(
    df: pd.DataFrame,
    fill_method: str = 'ffill',
    min_data_points: int = 10
) -> pd.DataFrame:
    """Clean and preprocess financial data."""
    if df.empty:
        return df
    
    # Make a copy to avoid SettingWithCopyWarning
    df = df.copy()
    
    # Ensure we have a datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    
    # Sort by date
    df = df.sort_index()
    
    # Handle missing values
    if fill_method == 'ffill':
        df = df.ffill()
    elif fill_method == 'bfill':
        df = df.bfill()
    elif fill_method == 'interpolate':
        df = df.interpolate()
    
    # Drop remaining NA values
    df = df.dropna(thresh=min_data_points)
    
    return df


def _calculate_financial_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate financial metrics like returns, volatility, etc."""
    if df.empty:
        return df
    
    # Make a copy to avoid SettingWithCopyWarning
    df = df.copy()
    
    # Ensure we have required columns
    required_cols = {'Open', 'High', 'Low', 'Close', 'Volume'}
    available_cols = set(df.columns)
    
    # Only calculate metrics for available columns
    if 'Close' in available_cols:
        # Calculate daily returns
        df['Daily_Return'] = df['Close'].pct_change()
        
        # Calculate cumulative returns
        df['Cumulative_Return'] = (1 + df['Daily_Return']).cumprod() - 1
        
        # Calculate moving averages
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
    
    if 'Volume' in available_cols:
        # Calculate volume-based metrics
        df['Volume_MA_20'] = df['Volume'].rolling(window=20).mean()
    
    return df
