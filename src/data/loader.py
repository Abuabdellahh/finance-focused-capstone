"""
Data loading utilities for financial data.

This module provides functions to load financial data from various sources.
"""

import os
import pandas as pd
from pathlib import Path
from typing import Union, Optional, Dict, Any
import yfinance as yf


def load_financial_data(
    tickers: Union[str, list],
    start_date: str,
    end_date: str,
    data_dir: Optional[Union[str, Path]] = None,
    use_cache: bool = True
) -> pd.DataFrame:
    """
    Load financial data from Yahoo Finance or local cache.

    Args:
        tickers: A single ticker or list of tickers to load
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        data_dir: Directory to save/load cached data
        use_cache: Whether to use cached data if available

    Returns:
        DataFrame containing the financial data
    """
    if isinstance(tickers, str):
        tickers = [tickers]
    
    if data_dir is None:
        data_dir = Path(__file__).parent.parent.parent / 'data' / 'raw'
    else:
        data_dir = Path(data_dir)
    
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate a unique cache filename
    cache_file = data_dir / f"{'_'.join(sorted(tickers))}_{start_date}_{end_date}.parquet"
    
    # Try to load from cache if available
    if use_cache and cache_file.exists():
        try:
            return pd.read_parquet(cache_file)
        except Exception as e:
            print(f"Error loading cached data: {e}")
    
    # Download data from Yahoo Finance
    try:
        print(f"Downloading data for {tickers} from {start_date} to {end_date}...")
        data = yf.download(
            tickers,
            start=start_date,
            end=end_date,
            group_by='ticker',
            progress=False
        )
        
        # Save to cache
        if use_cache:
            try:
                data.to_parquet(cache_file)
                print(f"Data saved to {cache_file}")
            except Exception as e:
                print(f"Warning: Could not save data to cache: {e}")
        
        return data
    
    except Exception as e:
        print(f"Error downloading data: {e}")
        raise
