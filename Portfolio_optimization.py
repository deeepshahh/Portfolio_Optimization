import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pypfopt import EfficientFrontier, risk_models, expected_returns, objective_functions
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt.expected_returns import capm_return
from scipy.stats import norm

# 1. Data Collection
def get_data(tickers, start_date, end_date):
    print("Fetching data...")
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    return data

# 2. Calculate Log Returns
def calculate_log_returns(data):
    log_returns = np.log(data / data.shift(1)).dropna()
    return log_returns

# 3. Covariance Matrix Shrinkage
def calculate_shrinked_covariance(log_returns):
    print("Calculating shrinked covariance matrix...")
    S = CovarianceShrinkage(log_returns).ledoit_wolf()
    return S

# 4. Portfolio Optimization with Regularization (L2)
def portfolio_optimization(log_returns, S, l2_reg=0.01, target_return=None):
    mu = expected_returns.mean_historical_return(log_returns)
    
    ef = EfficientFrontier(mu, S)
    ef.add_objective(objective_functions.L2_reg, gamma=l2_reg)
    
    if target_return:
        ef.efficient_return(target_return=target_return)
    else:
        ef.max_sharpe()
    
    weights = ef.clean_weights()
    performance = ef.portfolio_performance(verbose=True)
    
    return weights, performance, ef

# 5. Calculate Advanced Risk Metrics
def calculate_risk_metrics(log_returns, weights):
    portfolio_returns = log_returns.dot(weights)
    VaR_95 = np.percentile(portfolio_returns, 5)
    CVaR_95 = portfolio_returns[portfolio_returns <= VaR_95].mean()
    annualized_volatility = np.std(portfolio_returns) * np.sqrt(252)
    max_drawdown = (portfolio_returns.cumsum().apply(np.exp).cummax() - portfolio_returns.cumsum().apply(np.exp)).max()
    sortino_ratio = portfolio_returns.mean() / np.sqrt(np.mean(np.minimum(0, portfolio_returns)**2)) * np.sqrt(252)
    
    return {
        "Value at Risk (95%)": VaR_95,
        "Conditional VaR (95%)": CVaR_95,
        "Annualized Volatility": annualized_volatility,
        "Max Drawdown": max_drawdown,
        "Sortino Ratio": sortino_ratio
    }

# 6. Plot Risk Metrics Over Time
def plot_rolling_metrics(log_returns, weights):
    portfolio_returns = log_returns.dot(weights)
    rolling_volatility = portfolio_returns.rolling(window=252).std() * np.sqrt(252)
    rolling_sharpe = portfolio_returns.rolling(window=252).mean() / portfolio_returns.rolling(window=252).std() * np.sqrt(252)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    ax1.plot(rolling_volatility)
    ax1.set_title('Rolling Volatility (Annualized)')
    ax1.set_ylabel('Volatility')
    
    ax2.plot(rolling_sharpe)
    ax2.set_title('Rolling Sharpe Ratio')
    ax2.set_ylabel('Sharpe Ratio')
    
    plt.tight_layout()
    plt.show()

# 7. Drawdown Chart
def plot_drawdown(portfolio_returns):
    cumulative_returns = portfolio_returns.cumsum().apply(np.exp)
    running_max = cumulative_returns.cummax()
    drawdown = (running_max - cumulative_returns) / running_max
    
    plt.figure(figsize=(10, 6))
    plt.plot(drawdown)
    plt.title('Drawdown from Peak')
    plt.ylabel('Drawdown')
    plt.xlabel('Date')
    plt.show()

# Main Execution
def main():
    tickers = input("Enter the tickers of the assets separated by a comma: ").split(',')
    start_date = input("Enter the start date (YYYY-MM-DD): ")
    end_date = input("Enter the end date (YYYY-MM-DD): ")
    
    data = get_data(tickers, start_date, end_date)
    log_returns = calculate_log_returns(data)
    S = calculate_shrinked_covariance(log_returns)
    
    weights, performance, ef = portfolio_optimization(log_returns, S, l2_reg=0.01, target_return=None)
    
    print("Optimal Weights: ", weights)
    print("Portfolio Performance: ", performance)
    
    risk_metrics = calculate_risk_metrics(log_returns, np.array(list(weights.values())))
    print("Advanced Risk Metrics: ", risk_metrics)
    
    plot_rolling_metrics(log_returns, np.array(list(weights.values())))
    plot_drawdown(log_returns.dot(np.array(list(weights.values()))))

if __name__ == "__main__":
    main()
