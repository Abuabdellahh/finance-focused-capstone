"""
Main script for the finance-focused capstone project.

This script demonstrates the usage of the financial analysis tools
for portfolio management and risk analysis.
"""

import os
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple

# Import project modules
from src.data import load_financial_data, process_financial_data
from src.models.portfolio import (
    calculate_portfolio_metrics,
    optimize_portfolio,
    calculate_efficient_frontier,
    calculate_risk_metrics
)
from src.visualization.plots import (
    plot_price_series,
    plot_returns_distribution,
    plot_efficient_frontier,
    plot_drawdown,
    plot_rolling_metrics,
    plot_correlation_heatmap
)
from src.utils.helpers import (
    calculate_annualized_return,
    calculate_annualized_volatility,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_max_drawdown
)


def analyze_portfolio(
    tickers: List[str],
    start_date: str,
    end_date: str,
    risk_free_rate: float = 0.02,
    output_dir: Optional[Union[str, Path]] = None,
    save_plots: bool = True
) -> Dict[str, any]:
    """
    Perform a comprehensive analysis of a portfolio.

    Args:
        tickers: List of ticker symbols to analyze
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        risk_free_rate: Annual risk-free rate (default: 0.02)
        output_dir: Directory to save output files (default: 'output' in project root)
        save_plots: Whether to save the generated plots

    Returns:
        Dictionary containing analysis results
    """
    # Set up output directory
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / 'output'
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"Loading data for {', '.join(tickers)} from {start_date} to {end_date}...")
    raw_data = load_financial_data(tickers, start_date, end_date)
    
    if raw_data.empty:
        raise ValueError("No data returned for the given tickers and date range.")
    
    # Process data
    print("Processing data...")
    processed_data = process_financial_data(raw_data)
    
    if not processed_data:
        raise ValueError("No data was processed successfully.")
    
    # Extract returns for each asset
    returns_dict = {}
    for ticker, df in processed_data.items():
        if 'Daily_Return' in df.columns:
            returns_dict[ticker] = df['Daily_Return'].dropna()
    
    if not returns_dict:
        raise ValueError("No return data available for analysis.")
    
    # Create a DataFrame of returns
    returns_df = pd.DataFrame(returns_dict)
    
    # Calculate portfolio metrics with equal weights
    print("Calculating portfolio metrics...")
    equal_weights = np.ones(len(returns_df.columns)) / len(returns_df.columns)
    portfolio_metrics = calculate_portfolio_metrics(
        returns_df,
        weights=equal_weights,
        risk_free_rate=risk_free_rate
    )
    
    # Optimize portfolio
    print("Optimizing portfolio...")
    optimization_result = optimize_portfolio(
        returns_df,
        objective='sharpe',
        risk_free_rate=risk_free_rate
    )
    
    # Calculate efficient frontier
    print("Calculating efficient frontier...")
    frontier = calculate_efficient_frontier(
        returns_df,
        n_points=50,
        risk_free_rate=risk_free_rate
    )
    
    # Generate visualizations
    print("Generating visualizations...")
    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)
    
    # Plot price series for each asset
    for ticker, df in processed_data.items():
        if 'Close' in df.columns:
            plot_price_series(
                df,
                column='Close',
                title=f'{ticker} Price Series',
                save_path=str(plots_dir / f'{ticker}_price_series.png') if save_plots else None
            )
    
    # Plot returns distribution for each asset
    for ticker, returns in returns_dict.items():
        plot_returns_distribution(
            returns,
            title=f'{ticker} Returns Distribution',
            save_path=str(plots_dir / f'{ticker}_returns_dist.png') if save_plots else None
        )
    
    # Plot efficient frontier
    if not frontier.empty:
        plot_efficient_frontier(
            frontier,
            title='Efficient Frontier',
            save_path=str(plots_dir / 'efficient_frontier.png') if save_plots else None
        )
    
    # Plot correlation heatmap
    plot_correlation_heatmap(
        returns_df,
        title='Asset Returns Correlation',
        save_path=str(plots_dir / 'correlation_heatmap.png') if save_plots else None
    )
    
    # Plot rolling metrics for the portfolio
    portfolio_returns = (returns_df * equal_weights).sum(axis=1)
    plot_rolling_metrics(
        portfolio_returns,
        title='Portfolio Rolling Metrics',
        save_path=str(plots_dir / 'rolling_metrics.png') if save_plots else None
    )
    
    # Plot drawdown for the portfolio
    plot_drawdown(
        portfolio_returns,
        title='Portfolio Drawdown',
        save_path=str(plots_dir / 'drawdown.png') if save_plots else None
    )
    
    # Prepare results
    results = {
        'tickers': tickers,
        'start_date': start_date,
        'end_date': end_date,
        'risk_free_rate': risk_free_rate,
        'equal_weight_metrics': portfolio_metrics,
        'optimization_result': optimization_result,
        'efficient_frontier': frontier.to_dict('list') if not frontier.empty else {}
    }
    
    # Save results to JSON
    results_path = output_dir / 'analysis_results.json'
    pd.Series(results).to_json(results_path, indent=2)
    print(f"Analysis complete. Results saved to {results_path}")
    
    return results


def main():
    """Main function to run the financial analysis."""
    parser = argparse.ArgumentParser(description='Financial Portfolio Analysis Tool')
    
    # Required arguments
    parser.add_argument(
        '--tickers',
        nargs='+',
        required=True,
        help='List of ticker symbols to analyze (e.g., AAPL MSFT GOOGL)'
    )
    
    # Optional arguments
    parser.add_argument(
        '--start-date',
        default='2018-01-01',
        help='Start date in YYYY-MM-DD format (default: 2018-01-01)'
    )
    
    parser.add_argument(
        '--end-date',
        default=pd.Timestamp.today().strftime('%Y-%m-%d'),
        help=f'End date in YYYY-MM-DD format (default: today)'
    )
    
    parser.add_argument(
        '--risk-free-rate',
        type=float,
        default=0.02,
        help='Annual risk-free rate (default: 0.02)'
    )
    
    parser.add_argument(
        '--output-dir',
        default='output',
        help='Directory to save output files (default: output)'
    )
    
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Disable saving plots'
    )
    
    args = parser.parse_args()
    
    try:
        # Run the analysis
        results = analyze_portfolio(
            tickers=args.tickers,
            start_date=args.start_date,
            end_date=args.end_date,
            risk_free_rate=args.risk_free_rate,
            output_dir=args.output_dir,
            save_plots=not args.no_plots
        )
        
        # Print a summary of the results
        print("\n=== Analysis Summary ===")
        print(f"Tickers: {', '.join(results['tickers'])}")
        print(f"Period: {results['start_date']} to {results['end_date']}")
        print(f"Risk-free rate: {results['risk_free_rate']:.2%}")
        
        print("\n=== Equal Weight Portfolio ===")
        metrics = results['equal_weight_metrics']
        print(f"Return (annualized): {metrics['return']:.2%}")
        print(f"Volatility (annualized): {metrics['volatility']:.2%}")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"Sortino Ratio: {metrics['sortino_ratio']:.2f}")
        print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
        
        print("\n=== Optimized Portfolio ===")
        opt_metrics = results['optimization_result']['metrics']
        print(f"Return (annualized): {opt_metrics['return']:.2%}")
        print(f"Volatility (annualized): {opt_metrics['volatility']:.2%}")
        print(f"Sharpe Ratio: {opt_metrics['sharpe_ratio']:.2f}")
        print(f"Optimal Weights:")
        for ticker, weight in opt_metrics['weights'].items():
            print(f"  {ticker}: {weight:.2%}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
