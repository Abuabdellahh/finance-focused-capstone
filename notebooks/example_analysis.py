# %% [markdown]
# Finance Portfolio Analysis Example

# %% [markdown]
## 1. Import Required Libraries

# %%
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join('..', '..', 'src')))

# Import our finance library
from data.loader import load_financial_data
from data.processor import process_financial_data
from models.portfolio import (
    calculate_portfolio_metrics,
    optimize_portfolio,
    calculate_efficient_frontier,
    calculate_risk_metrics
)
from visualization.plots import (
    plot_price_series,
    plot_returns_distribution,
    plot_efficient_frontier,
    plot_drawdown,
    plot_rolling_metrics,
    plot_correlation_heatmap
)

# Set plot style
plt.style.use('seaborn')
sns.set_palette('deep')

# %% [markdown]
## 2. Load and Prepare Data

# %%
# Define the tickers we want to analyze
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']

# Define the date range
start_date = '2018-01-01'
end_date = '2023-12-31'

# Load the data
print(f"Loading data for {tickers} from {start_date} to {end_date}...")
raw_data = load_financial_data(tickers, start_date, end_date)

# Display the first few rows of the data
print("\nRaw data shape:", raw_data.shape)
raw_data.head()

# %% [markdown]
## 3. Process the Data

# %%
# Process the data
print("Processing data...")
processed_data = process_financial_data(raw_data)

# Display the processed data for the first ticker
ticker = tickers[0]
print(f"\nProcessed data for {ticker}:")
processed_data[ticker].head()

# %% [markdown]
## 4. Calculate and Visualize Returns

# %%
# Extract returns for each asset
returns_dict = {}
for ticker, df in processed_data.items():
    if 'Daily_Return' in df.columns:
        returns_dict[ticker] = df['Daily_Return'].dropna()

# Create a DataFrame of returns
returns_df = pd.DataFrame(returns_dict)

# Plot the cumulative returns
cumulative_returns = (1 + returns_df).cumprod()
plt.figure(figsize=(12, 6))
cumulative_returns.plot()
plt.title('Cumulative Returns')
plt.ylabel('Growth of $1 Investment')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# %% [markdown]
## 5. Calculate Portfolio Metrics

# %%
# Calculate portfolio metrics with equal weights
equal_weights = np.ones(len(returns_df.columns)) / len(returns_df.columns)
portfolio_metrics = calculate_portfolio_metrics(
    returns_df,
    weights=equal_weights,
    risk_free_rate=0.02  # 2% risk-free rate
)

print("Portfolio Metrics (Equal Weights):")
for metric, value in portfolio_metrics.items():
    if metric != 'weights':
        if isinstance(value, float):
            print(f"{metric.replace('_', ' ').title()}: {value:.4f}")

print("\nWeights:")
for ticker, weight in zip(returns_df.columns, equal_weights):
    print(f"  {ticker}: {weight:.2%}")

# %% [markdown]
## 6. Optimize the Portfolio

# %%
# Optimize the portfolio for maximum Sharpe ratio
optimization_result = optimize_portfolio(
    returns_df,
    objective='sharpe',
    risk_free_rate=0.02
)

# Display the optimization results
print("Optimization Results:")
print(f"Success: {optimization_result['success']}")
print(f"Message: {optimization_result['message']}")

# Display the optimal weights and metrics
optimal_weights = optimization_result['optimal_weights']
metrics = optimization_result['metrics']

print("\nOptimal Weights:")
for ticker, weight in zip(returns_df.columns, optimal_weights):
    print(f"  {ticker}: {weight:.2%}")

print("\nPortfolio Metrics (Optimal Weights):")
for metric, value in metrics.items():
    if metric != 'weights' and isinstance(value, (int, float)):
        print(f"{metric.replace('_', ' ').title()}: {value:.4f}")

# %% [markdown]
## 7. Calculate and Plot the Efficient Frontier

# %%
# Calculate the efficient frontier
frontier = calculate_efficient_frontier(
    returns_df,
    n_points=50,
    risk_free_rate=0.02
)

# Plot the efficient frontier
plt.figure(figsize=(10, 6))
plt.plot(
    frontier['volatility'],
    frontier['return'],
    'b-',
    linewidth=2,
    label='Efficient Frontier'
)

# Mark the optimal portfolios
max_sharpe_idx = frontier['sharpe_ratio'].idxmax()
min_vol_idx = frontier['volatility'].idxmin()

plt.scatter(
    frontier.loc[max_sharpe_idx, 'volatility'],
    frontier.loc[max_sharpe_idx, 'return'],
    color='red',
    s=100,
    marker='*',
    label='Max Sharpe Ratio'
)

plt.scatter(
    frontier.loc[min_vol_idx, 'volatility'],
    frontier.loc[min_vol_idx, 'return'],
    color='green',
    s=100,
    marker='^',
    label='Min Volatility'
)

# Format the plot
plt.title('Efficient Frontier', fontsize=14, fontweight='bold')
plt.xlabel('Volatility (Annualized)')
plt.ylabel('Return (Annualized)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

# Format axes as percentages
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter('{0:.1%}'.format))
plt.gca().xaxis.set_major_formatter(plt.FuncFormatter('{0:.1%}'.format))

plt.tight_layout()
plt.show()

# %% [markdown]
## 8. Analyze Risk Metrics

# %%
# Calculate risk metrics for the optimal portfolio
optimal_portfolio_returns = (returns_df * optimal_weights).sum(axis=1)
risk_metrics = calculate_risk_metrics(optimal_portfolio_returns)

print("Risk Metrics for Optimal Portfolio:")
for metric, value in risk_metrics.items():
    if isinstance(value, (int, float)):
        if abs(value) >= 0.01:
            print(f"{metric.replace('_', ' ').title()}: {value:.4f}")
        else:
            print(f"{metric.replace('_', ' ').title()}: {value:.4e}")

# Plot drawdown
plt.figure(figsize=(12, 6))
plot_drawdown(optimal_portfolio_returns, title='Optimal Portfolio Drawdown')
plt.show()

# Plot rolling metrics
plt.figure(figsize=(14, 10))
plot_rolling_metrics(
    optimal_portfolio_returns,
    window=63,  # 3 months (assuming 21 trading days per month)
    metrics=['volatility', 'sharpe', 'sortino', 'drawdown']
)
plt.suptitle('Optimal Portfolio Rolling Metrics (3-Month Window)', y=1.02)
plt.tight_layout()
plt.show()

# %% [markdown]
## 9. Visualize Correlations

# %%
# Plot correlation heatmap
plt.figure(figsize=(10, 8))
plot_correlation_heatmap(
    returns_df,
    title='Asset Returns Correlation',
    annot=True
)
plt.tight_layout()
plt.show()

# %% [markdown]
## 10. Save Results

# %%
# Create output directory
output_dir = '../../output'
os.makedirs(output_dir, exist_ok=True)

# Save portfolio metrics to CSV
portfolio_metrics_df = pd.DataFrame({
    'Metric': [k.replace('_', ' ').title() for k, v in metrics.items() 
               if not isinstance(v, dict) and not isinstance(v, list)],
    'Value': [v for k, v in metrics.items() 
              if not isinstance(v, dict) and not isinstance(v, list)]
})
portfolio_metrics_df.to_csv(os.path.join(output_dir, 'portfolio_metrics.csv'), index=False)

# Save optimal weights to CSV
weights_df = pd.DataFrame({
    'Ticker': returns_df.columns,
    'Weight': optimal_weights
})
weights_df.to_csv(os.path.join(output_dir, 'optimal_weights.csv'), index=False)

# Save the efficient frontier to CSV
frontier.to_csv(os.path.join(output_dir, 'efficient_frontier.csv'), index=False)

print(f"Results saved to {output_dir}")

# %% [markdown]
## Conclusion

# %% [markdown]
This example demonstrated how to use the finance library to:
1. Load and process financial data
2. Calculate portfolio metrics
3. Optimize portfolio weights
4. Analyze risk and return trade-offs
5. Visualize results

You can modify the tickers, date range, and other parameters to analyze different portfolios.
