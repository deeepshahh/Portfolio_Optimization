import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from scipy.optimize import minimize
from fpdf import FPDF
import streamlit as st

# Data Collection Function
def fetch_data(tickers, start_date, end_date):
    """
    Fetch adjusted closing prices for a list of tickers from Yahoo Finance.
    
    Parameters:
    tickers (list): List of ticker symbols.
    start_date (str): Start date in 'YYYY-MM-DD' format.
    end_date (str): End date in 'YYYY-MM-DD' format.
    
    Returns:
    pd.DataFrame: DataFrame containing adjusted closing prices.
    """
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    return data

# Risk Metrics Calculation
def calculate_var(returns, alpha=0.05):
    """
    Calculate Value at Risk (VaR) for a given set of returns.
    
    Parameters:
    returns (pd.Series or pd.DataFrame): Asset returns.
    alpha (float): Confidence level for VaR, default is 0.05.
    
    Returns:
    float: Value at Risk.
    """
    if len(returns) == 0:
        return np.nan
    var = np.percentile(returns, 100 * alpha)
    return var

def calculate_es(returns, alpha=0.05):
    """
    Calculate Expected Shortfall (ES) or Conditional VaR for a given set of returns.
    
    Parameters:
    returns (pd.Series or pd.DataFrame): Asset returns.
    alpha (float): Confidence level for ES, default is 0.05.
    
    Returns:
    float: Expected Shortfall.
    """
    if len(returns) == 0:
        return np.nan
    var = calculate_var(returns, alpha)
    es = returns[returns <= var].mean()
    return es

# Monte Carlo Simulation for Efficient Frontier
def monte_carlo_simulation(mean_returns, cov_matrix, num_simulations=10000):
    """
    Perform Monte Carlo simulation to generate portfolios.

    Parameters:
    mean_returns (pd.Series): Expected returns of assets.
    cov_matrix (pd.DataFrame): Covariance matrix of asset returns.
    num_simulations (int): Number of simulated portfolios, default is 10,000.
    
    Returns:
    np.ndarray: Array containing portfolio returns, volatility, and Sharpe ratios.
    """
    np.random.seed(42)
    num_assets = len(mean_returns)
    results = np.zeros((3, num_simulations))

    for i in range(num_simulations):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)

        portfolio_return = np.sum(mean_returns * weights)
        portfolio_stddev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

        results[0, i] = portfolio_return
        results[1, i] = portfolio_stddev
        results[2, i] = (portfolio_return - 0.01) / portfolio_stddev  # Sharpe Ratio

    return results

# Portfolio Optimization Using Scipy
def optimize_portfolio(mean_returns, cov_matrix):
    """
    Optimize portfolio to maximize Sharpe ratio using mean-variance optimization.
    
    Parameters:
    mean_returns (pd.Series): Expected returns of assets.
    cov_matrix (pd.DataFrame): Covariance matrix of asset returns.
    
    Returns:
    dict: Optimal portfolio weights, returns, volatility, and Sharpe ratio.
    """
    num_assets = len(mean_returns)

    def neg_sharpe_ratio(weights):
        portfolio_return = np.sum(mean_returns * weights)
        portfolio_stddev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return - (portfolio_return - 0.01) / portfolio_stddev

    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    initial_guess = num_assets * [1. / num_assets,]

    optimized = minimize(neg_sharpe_ratio, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)
    optimal_weights = optimized.x

    portfolio_return = np.sum(mean_returns * optimal_weights)
    portfolio_stddev = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights)))
    sharpe_ratio = (portfolio_return - 0.01) / portfolio_stddev

    return {
        "weights": optimal_weights,
        "return": portfolio_return,
        "volatility": portfolio_stddev,
        "sharpe_ratio": sharpe_ratio
    }

# Plotting the Efficient Frontier
def plot_efficient_frontier(mean_returns, cov_matrix):
    """
    Plot the efficient frontier using Monte Carlo simulation.
    
    Parameters:
    mean_returns (pd.Series): Expected returns of assets.
    cov_matrix (pd.DataFrame): Covariance matrix of asset returns.
    """
    results = monte_carlo_simulation(mean_returns, cov_matrix, 10000)
    fig = px.scatter(x=results[1], y=results[0], color=results[2], 
                     labels={'x': 'Volatility', 'y': 'Return', 'color': 'Sharpe Ratio'},
                     title='Efficient Frontier')
    st.plotly_chart(fig)

# Generate a PDF Report
def create_pdf_report(tickers, optimized_weights, port_return, port_std, sharpe_ratio, var, es):
    """
    Create a PDF report summarizing the portfolio optimization results.
    
    Parameters:
    tickers (list): List of ticker symbols.
    optimized_weights (list): Optimal portfolio weights.
    port_return (float): Expected portfolio return.
    port_std (float): Portfolio volatility.
    sharpe_ratio (float): Sharpe ratio of the portfolio.
    var (float): Value at Risk of the portfolio.
    es (float): Expected Shortfall of the portfolio.
    """
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    pdf.cell(200, 10, txt="Portfolio Optimization Report", ln=True, align='C')
    
    pdf.cell(200, 10, txt=f"Tickers: {', '.join(tickers)}", ln=True)
    pdf.cell(200, 10, txt=f"Weights: {', '.join([f'{w:.2%}' for w in optimized_weights])}", ln=True)
    
    pdf.cell(200, 10, txt=f"Expected Return: {port_return:.2%}", ln=True)
    pdf.cell(200, 10, txt=f"Portfolio Volatility: {port_std:.2%}", ln=True)
    pdf.cell(200, 10, txt=f"Sharpe Ratio: {sharpe_ratio:.2f}", ln=True)
    pdf.cell(200, 10, txt=f"Value at Risk (VaR): {var:.2%}", ln=True)
    pdf.cell(200, 10, txt=f"Expected Shortfall (ES): {es:.2%}", ln=True)
    
    pdf.output("Portfolio_Optimization_Report.pdf")

# Streamlit Interface
def run_streamlit_app():
    st.title("Portfolio Optimization Tool")
    
    tickers = st.text_input("Enter ticker symbols separated by commas (e.g., AAPL, MSFT, GOOG)").split(',')
    start_date = st.date_input("Select start date")
    end_date = st.date_input("Select end date")
    
    if st.button("Optimize Portfolio"):
        # Fetch data
        data = fetch_data(tickers, start_date, end_date)
        log_returns = np.log(data / data.shift(1)).dropna()
        mean_returns = log_returns.mean()
        cov_matrix = log_returns.cov()
        
        # Optimize portfolio
        optimal_portfolio = optimize_portfolio(mean_returns, cov_matrix)
        
        # Risk metrics
        var = calculate_var(log_returns.mean(axis=1))
        es = calculate_es(log_returns.mean(axis=1))
        
        # Display results
        st.write("Optimal Weights:", optimal_portfolio['weights'])
        st.write("Expected Return:", optimal_portfolio['return'])
        st.write("Volatility:", optimal_portfolio['volatility'])
        st.write("Sharpe Ratio:", optimal_portfolio['sharpe_ratio'])
        st.write("Value at Risk (VaR):", var)
        st.write("Expected Shortfall (ES):", es)
        
        # Plot efficient frontier
        plot_efficient_frontier(mean_returns, cov_matrix)
        
        # Create PDF report
        create_pdf_report(tickers, optimal_portfolio['weights'], optimal_portfolio['return'],
                          optimal_portfolio['volatility'], optimal_portfolio['sharpe_ratio'], var, es)
        st.success("PDF Report Created: Portfolio_Optimization_Report.pdf")

# Entry Point
if __name__ == "__main__":
    run_streamlit_app()
