import yfinance as yf
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
from scipy.optimize import minimize
from fpdf import FPDF
from io import BytesIO
import base64

# Risk and Monte Carlo Simulation functions
def calculate_var(returns, alpha=0.05):
    if len(returns) == 0:
        return np.nan
    var = np.percentile(returns, 100 * alpha)
    return var

def calculate_es(returns, alpha=0.05):
    if len(returns) == 0:
        return np.nan
    var = calculate_var(returns, alpha)
    es = returns[returns <= var].mean()
    return es

def monte_carlo_simulation(mean_returns, cov_matrix, num_simulations, num_days):
    np.random.seed(42)
    num_assets = len(mean_returns)
    results = np.zeros((num_simulations, num_days))

    for i in range(num_simulations):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)

        daily_returns = np.random.multivariate_normal(mean_returns, cov_matrix, num_days)
        portfolio_returns = np.dot(daily_returns, weights)
        results[i, :] = portfolio_returns

    return results

# Risk Parity Optimization
def risk_parity_optimization(cov_matrix):
    def risk_contribution(weights, cov_matrix):
        portfolio_var = np.dot(weights.T, np.dot(cov_matrix, weights))
        marginal_contrib = np.dot(cov_matrix, weights)
        risk_contrib = weights * marginal_contrib / portfolio_var
        return risk_contrib

    num_assets = len(cov_matrix)
    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
    bounds = tuple((0, 1) for asset in range(num_assets))
    initial_guess = num_assets * [1. / num_assets,]

    result = minimize(lambda weights: np.sum((risk_contribution(weights, cov_matrix) - 1/num_assets)**2),
                      initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x

# Data Retrieval and Processing
def get_historical_data(tickers, start, end):
    data = yf.download(tickers, start=start, end=end)['Adj Close']
    return data

def calculate_annual_returns(data):
    returns = data.pct_change().mean() * 252
    return returns

def calculate_annual_covariance(data):
    cov_matrix = data.pct_change().cov() * 252
    return cov_matrix

# Mean-Variance Optimization
def mean_variance_optimization(returns, cov_matrix):
    def portfolio_performance(weights, returns, cov_matrix):
        portfolio_return = np.sum(weights * returns)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return portfolio_return, portfolio_volatility

    def negative_sharpe_ratio(weights, returns, cov_matrix, risk_free_rate=0.01):
        portfolio_return, portfolio_volatility = portfolio_performance(weights, returns, cov_matrix)
        return - (portfolio_return - risk_free_rate) / portfolio_volatility

    num_assets = len(returns)
    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
    bounds = tuple((0, 1) for asset in range(num_assets))
    initial_guess = num_assets * [1. / num_assets,]

    result = minimize(negative_sharpe_ratio, initial_guess, args=(returns, cov_matrix), method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x

# Plotting functions
def plot_portfolio(data, weights):
    portfolio = np.dot(data, weights)
    fig, ax = plt.subplots()
    ax.plot(portfolio, label='Optimized Portfolio')
    ax.legend()
    return fig

def plot_interactive_portfolio(data, weights):
    portfolio = np.dot(data, weights)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=portfolio, mode='lines', name='Optimized Portfolio'))
    return fig

# Download PDF function
def download_pdf(fig):
    buffer = BytesIO()
    fig.savefig(buffer, format='pdf')
    buffer.seek(0)
    b64 = base64.b64encode(buffer.read()).decode()
    href = f'<a href="data:application/pdf;base64,{b64}" download="portfolio_optimization.pdf">Download PDF Report</a>'
    return href

# Main Streamlit app
def main():
    st.title("Portfolio Optimization Tool")

    st.sidebar.header("User Input Parameters")

    portfolio_type = st.sidebar.selectbox("Select Portfolio Type", ["Multi-Asset Portfolio", "Equity Portfolio"])

    tickers = st.sidebar.text_area("Enter tickers separated by commas")
    tickers = [ticker.strip() for ticker in tickers.split(",")]

    start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
    end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2023-01-01"))

    num_simulations = st.sidebar.number_input("Number of Simulations", min_value=100, max_value=10000, value=1000)
    num_days = st.sidebar.number_input("Number of Days for Monte Carlo Simulation", min_value=1, max_value=252, value=252)

    if st.sidebar.button("Optimize Portfolio"):
        data = get_historical_data(tickers, start=start_date, end=end_date)
        returns = calculate_annual_returns(data)
        cov_matrix = calculate_annual_covariance(data)

        st.subheader("Optimized Portfolio Weights (Mean-Variance Optimization)")
        mv_weights = mean_variance_optimization(returns, cov_matrix)
        st.write(dict(zip(tickers, mv_weights)))

        st.subheader("Optimized Portfolio Weights (Risk Parity)")
        rp_weights = risk_parity_optimization(cov_matrix)
        st.write(dict(zip(tickers, rp_weights)))

        mv_fig = plot_portfolio(data, mv_weights)
        st.pyplot(mv_fig)
        rp_fig = plot_portfolio(data, rp_weights)
        st.pyplot(rp_fig)

        monte_carlo_results = monte_carlo_simulation(returns, cov_matrix, num_simulations, num_days)
        mc_fig = plt.figure(figsize=(10, 6))
        plt.plot(np.mean(monte_carlo_results, axis=0))
        plt.title("Monte Carlo Simulation")
        plt.xlabel("Days")
        plt.ylabel("Portfolio Value")
        st.pyplot(mc_fig)

        mv_interactive_fig = plot_interactive_portfolio(data, mv_weights)
        st.plotly_chart(mv_interactive_fig)
        rp_interactive_fig = plot_interactive_portfolio(data, rp_weights)
        st.plotly_chart(rp_interactive_fig)

        mv_pdf_href = download_pdf(mv_fig)
        st.markdown(mv_pdf_href, unsafe_allow_html=True)
        rp_pdf_href = download_pdf(rp_fig)
        st.markdown(rp_pdf_href, unsafe_allow_html=True)
        mc_pdf_href = download_pdf(mc_fig)
        st.markdown(mc_pdf_href, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
