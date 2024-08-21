import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.optimize import minimize
from fpdf import FPDF
import streamlit as st

def fetch_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    if data.empty:
        st.error("No data retrieved. Please check the tickers and date range.")
        st.stop()
    return data

def calculate_var(returns, alpha=0.05):
    var = np.percentile(returns, 100 * alpha)
    return var

def calculate_es(returns, alpha=0.05):
    var = calculate_var(returns, alpha)
    es = returns[returns <= var].mean()
    return es

def monte_carlo_simulation(mean_returns, cov_matrix, num_simulations=10000):
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
        results[2, i] = (portfolio_return - 0.01) / portfolio_stddev

    return results

def optimize_portfolio(mean_returns, cov_matrix):
    num_assets = len(mean_returns)

    def neg_sharpe_ratio(weights):
        portfolio_return = np.sum(mean_returns * weights)
        portfolio_stddev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return - (portfolio_return - 0.01) / portfolio_stddev

    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    initial_guess = num_assets * [1. / num_assets,]

    try:
        optimized = minimize(neg_sharpe_ratio, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)
        optimal_weights = optimized.x
    except ValueError:
        st.error("Optimization failed due to insufficient data or invalid inputs.")
        st.stop()

    portfolio_return = np.sum(mean_returns * optimal_weights)
    portfolio_stddev = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights)))
    sharpe_ratio = (portfolio_return - 0.01) / portfolio_stddev

    return {
        "weights": optimal_weights,
        "return": portfolio_return,
        "volatility": portfolio_stddev,
        "sharpe_ratio": sharpe_ratio
    }

def plot_efficient_frontier(mean_returns, cov_matrix):
    results = monte_carlo_simulation(mean_returns, cov_matrix, 10000)
    fig = px.scatter(x=results[1], y=results[0], color=results[2], 
                     labels={'x': 'Volatility', 'y': 'Return', 'color': 'Sharpe Ratio'},
                     title='Efficient Frontier')
    st.plotly_chart(fig)

def create_pdf_report(tickers, optimized_weights, port_return, port_std, sharpe_ratio, var, es):
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

def run_streamlit_app():
    st.title("Portfolio Optimization Tool")
    
    tickers = st.text_input("Enter ticker symbols separated by commas (e.g., AAPL, MSFT, GOOG)").split(',')
    start_date = st.date_input("Select start date")
    end_date = st.date_input("Select end date")
    
    if st.button("Optimize Portfolio"):
        data = fetch_data(tickers, start_date, end_date)
        log_returns = np.log(data / data.shift(1)).dropna()
        
        if log_returns.empty:
            st.error("Log returns could not be calculated. Please adjust the date range or ticker symbols.")
            st.stop()
        
        mean_returns = log_returns.mean()
        cov_matrix = log_returns.cov()
        
        optimal_portfolio = optimize_portfolio(mean_returns, cov_matrix)
        
        var = calculate_var(log_returns.mean(axis=1))
        es = calculate_es(log_returns.mean(axis=1))
        
        st.write("Optimal Weights:", optimal_portfolio['weights'])
        st.write("Expected Return:", optimal_portfolio['return'])
        st.write("Volatility:", optimal_portfolio['volatility'])
        st.write("Sharpe Ratio:", optimal_portfolio['sharpe_ratio'])
        st.write("Value at Risk (VaR):", var)
        st.write("Expected Shortfall (ES):", es)
        
        plot_efficient_frontier(mean_returns, cov_matrix)
        
        create_pdf_report(tickers, optimal_portfolio['weights'], optimal_portfolio['return'],
                          optimal_portfolio['volatility'], optimal_portfolio['sharpe_ratio'], var, es)
        st.success("PDF Report Created: Portfolio_Optimization_Report.pdf")

if __name__ == "__main__":
    run_streamlit_app()
