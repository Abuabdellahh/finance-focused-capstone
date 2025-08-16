"""
Plotting functions for financial data visualization.
"""

from typing import Optional, List, Dict, Any, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter

# Set the style for all plots
plt.style.use('seaborn')
sns.set_palette('deep')


def plot_price_series(
    data: pd.DataFrame,
    column: str = 'Close',
    title: str = 'Price Series',
    ylabel: str = 'Price',
    figsize: tuple = (12, 6),
    save_path: Optional[str] = None,
    **kwargs
) -> plt.Figure:
    """
    Plot price series from a DataFrame.

    Args:
        data: DataFrame with datetime index and price data
        column: Column name to plot (default: 'Close')
        title: Plot title
        ylabel: Y-axis label
        figsize: Figure size (width, height)
        save_path: If provided, save the figure to this path
        **kwargs: Additional arguments to pass to pandas plot

    Returns:
        Matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot the data
    if isinstance(data, pd.DataFrame):
        data[column].plot(ax=ax, **kwargs)
    else:  # Series
        data.plot(ax=ax, **kwargs)
    
    # Format the plot
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Format y-axis as currency if the values look like prices
    if data[column].max() > 1:
        ax.yaxis.set_major_formatter(
            FuncFormatter(lambda x, p: f'${x:,.2f}')
        )
    
    plt.tight_layout()
    
    # Save the figure if path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    return fig


def plot_returns_distribution(
    returns: pd.Series,
    title: str = 'Returns Distribution',
    bins: int = 50,
    figsize: tuple = (12, 6),
    kde: bool = True,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot the distribution of returns.

    Args:
        returns: Series of returns
        title: Plot title
        bins: Number of bins for the histogram
        figsize: Figure size (width, height)
        kde: Whether to plot KDE (Kernel Density Estimate)
        save_path: If provided, save the figure to this path

    Returns:
        Matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot histogram of returns
    sns.histplot(
        returns,
        bins=bins,
        kde=kde,
        ax=ax,
        stat='density',
        kde_kws={'linewidth': 2}
    )
    
    # Add mean and median lines
    mean = returns.mean()
    median = returns.median()
    
    ax.axvline(mean, color='r', linestyle='--', linewidth=2, label=f'Mean: {mean:.2%}')
    ax.axvline(median, color='g', linestyle='--', linewidth=2, label=f'Median: {median:.2%}')
    
    # Format the plot
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Returns')
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Format x-axis as percentage
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.1%}'))
    
    plt.tight_layout()
    
    # Save the figure if path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    return fig


def plot_efficient_frontier(
    frontier: pd.DataFrame,
    title: str = 'Efficient Frontier',
    show_portfolios: bool = True,
    show_max_sharpe: bool = True,
    show_min_vol: bool = True,
    figsize: tuple = (10, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot the efficient frontier.

    Args:
        frontier: DataFrame with columns 'volatility' and 'return'
        title: Plot title
        show_portfolios: Whether to show individual portfolios
        show_max_sharpe: Whether to highlight max Sharpe ratio portfolio
        show_min_vol: Whether to highlight minimum volatility portfolio
        figsize: Figure size (width, height)
        save_path: If provided, save the figure to this path

    Returns:
        Matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot the efficient frontier
    ax.plot(
        frontier['volatility'],
        frontier['return'],
        'b-',
        linewidth=2,
        label='Efficient Frontier'
    )
    
    # Highlight max Sharpe ratio portfolio
    if show_max_sharpe and 'sharpe_ratio' in frontier.columns:
        idx = frontier['sharpe_ratio'].idxmax()
        ax.scatter(
            frontier.loc[idx, 'volatility'],
            frontier.loc[idx, 'return'],
            color='red',
            s=100,
            marker='*',
            label='Max Sharpe Ratio'
        )
    
    # Highlight minimum volatility portfolio
    if show_min_vol and 'volatility' in frontier.columns:
        idx = frontier['volatility'].idxmin()
        ax.scatter(
            frontier.loc[idx, 'volatility'],
            frontier.loc[idx, 'return'],
            color='green',
            s=100,
            marker='^',
            label='Min Volatility'
        )
    
    # Format the plot
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Volatility (Annualized)')
    ax.set_ylabel('Return (Annualized)')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.1%}'))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.1%}'))
    
    plt.tight_layout()
    
    # Save the figure if path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    return fig


def plot_drawdown(
    returns: pd.Series,
    title: str = 'Drawdown',
    figsize: tuple = (12, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot the drawdown of a strategy or asset.

    Args:
        returns: Series of returns
        title: Plot title
        figsize: Figure size (width, height)
        save_path: If provided, save the figure to this path

    Returns:
        Matplotlib Figure object
    """
    # Calculate cumulative returns
    cum_returns = (1 + returns).cumprod()
    
    # Calculate running maximum
    running_max = cum_returns.cummax()
    
    # Calculate drawdown
    drawdown = (cum_returns - running_max) / running_max
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot drawdown
    drawdown.plot(ax=ax, color='red', linewidth=1.5)
    
    # Fill the area under the drawdown curve
    ax.fill_between(
        drawdown.index,
        drawdown.values,
        0,
        color='red',
        alpha=0.2
    )
    
    # Format the plot
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylabel('Drawdown')
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.1%}'))
    
    # Add horizontal line at maximum drawdown
    max_dd = drawdown.min()
    ax.axhline(
        max_dd,
        color='black',
        linestyle='--',
        linewidth=1,
        label=f'Max Drawdown: {max_dd:.2%}'
    )
    
    ax.legend()
    plt.tight_layout()
    
    # Save the figure if path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    return fig


def plot_rolling_metrics(
    returns: pd.Series,
    window: int = 21,
    metrics: List[str] = None,
    figsize: tuple = (14, 10),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot rolling performance metrics.

    Args:
        returns: Series of returns
        window: Rolling window size in periods
        metrics: List of metrics to plot. Options: 'volatility', 'sharpe', 'sortino', 'drawdown'
        figsize: Figure size (width, height)
        save_path: If provided, save the figure to this path

    Returns:
        Matplotlib Figure object
    """
    if metrics is None:
        metrics = ['volatility', 'sharpe', 'sortino', 'drawdown']
    
    n_plots = len(metrics)
    fig, axes = plt.subplots(n_plots, 1, figsize=figsize, sharex=True)
    
    if n_plots == 1:
        axes = [axes]
    
    # Calculate metrics
    for i, metric in enumerate(metrics):
        if metric == 'volatility':
            # Annualized rolling volatility
            rolling_vol = returns.rolling(window).std() * np.sqrt(252)
            axes[i].plot(rolling_vol, label='Rolling Volatility (Annualized)')
            axes[i].set_ylabel('Volatility')
            
        elif metric == 'sharpe':
            # Annualized rolling Sharpe ratio (assuming risk-free rate = 0)
            rolling_mean = returns.rolling(window).mean() * 252
            rolling_vol = returns.rolling(window).std() * np.sqrt(252)
            rolling_sharpe = rolling_mean / rolling_vol.replace(0, np.nan)
            axes[i].plot(rolling_sharpe, label='Rolling Sharpe Ratio (Annualized)')
            axes[i].axhline(0, color='black', linestyle='--', linewidth=1)
            axes[i].set_ylabel('Sharpe Ratio')
            
        elif metric == 'sortino':
            # Annualized rolling Sortino ratio (assuming risk-free rate = 0 and target return = 0)
            rolling_mean = returns.rolling(window).mean() * 252
            downside_returns = returns[returns < 0]
            if not downside_returns.empty:
                rolling_downside_vol = returns.rolling(window).apply(
                    lambda x: x[x < 0].std() * np.sqrt(252), raw=True
                )
                rolling_sortino = rolling_mean / rolling_downside_vol.replace(0, np.nan)
                axes[i].plot(rolling_sortino, label='Rolling Sortino Ratio (Annualized)')
                axes[i].axhline(0, color='black', linestyle='--', linewidth=1)
                axes[i].set_ylabel('Sortino Ratio')
        
        elif metric == 'drawdown':
            # Rolling maximum drawdown
            cum_returns = (1 + returns).cumprod()
            rolling_max = cum_returns.rolling(window, min_periods=1).max()
            rolling_dd = (cum_returns - rolling_max) / rolling_max
            axes[i].plot(rolling_dd, label=f'Rolling {window}-Day Drawdown', color='red')
            axes[i].set_ylabel('Drawdown')
        
        # Format subplot
        axes[i].grid(True, linestyle='--', alpha=0.7)
        axes[i].legend(loc='upper left')
        
        # Format y-axis as percentage for volatility and drawdown
        if metric in ['volatility', 'drawdown']:
            axes[i].yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.1%}'))
    
    # Set common x-label
    axes[-1].set_xlabel('Date')
    
    plt.suptitle('Rolling Performance Metrics', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save the figure if path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    return fig


def plot_correlation_heatmap(
    data: pd.DataFrame,
    title: str = 'Correlation Heatmap',
    figsize: tuple = (10, 8),
    cmap: str = 'coolwarm',
    annot: bool = True,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot a correlation heatmap for a DataFrame.

    Args:
        data: DataFrame with numeric columns for correlation
        title: Plot title
        figsize: Figure size (width, height)
        cmap: Colormap for the heatmap
        annot: Whether to annotate the heatmap with correlation values
        save_path: If provided, save the figure to this path

    Returns:
        Matplotlib Figure object
    """
    # Calculate correlation matrix
    corr = data.corr()
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    sns.heatmap(
        corr,
        cmap=cmap,
        annot=annot,
        fmt='.2f',
        linewidths=0.5,
        cbar_kws={'label': 'Correlation'},
        ax=ax
    )
    
    # Format the plot
    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save the figure if path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    return fig
