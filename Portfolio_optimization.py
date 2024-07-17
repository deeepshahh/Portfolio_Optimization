import yfinance as yf
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from scipy.optimize import minimize
from fpdf import FPDF
from joblib import Parallel, delayed

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

def calculate_cvar(returns, alpha=0.05):
    if len(returns) == 0:
        return np.nan
    var = calculate_var(returns, alpha)
    cvar = np.mean([x for x in returns if x < var])
    return cvar

def calculate_max_drawdown(returns):
    cumulative_returns = (1 + returns).cumprod()
    peak = cumulative_returns.cummax()
    drawdown = (cumulative_returns - peak) / peak
    max_drawdown = drawdown.min()
    return max_drawdown

def monte_carlo_simulation(mean_returns, cov_matrix, num_simulations, num_assets):
    np.random.seed(42)
    results = np.zeros((4, num_simulations))

    def simulate_once():
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        portfolio_return = np.sum(mean_returns * weights)
        portfolio_stddev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe_ratio = (portfolio_return - 0.01) / portfolio_stddev
        max_drawdown = calculate_max_drawdown(np.dot(data.pct_change(), weights))
        return portfolio_return, portfolio_stddev, sharpe_ratio, max_drawdown

    results = Parallel(n_jobs=-1)(delayed(simulate_once)() for _ in range(num_simulations))
    return np.array(results).T

def plot_efficient_frontier(mean_returns, cov_matrix, data):
    num_assets = len(mean_returns)
    results = monte_carlo_simulation(mean_returns, cov_matrix, 10000, num_assets)
    fig = px.scatter(x=results[1], y=results[0], color=results[2], 
                     labels={'x': 'Volatility', 'y': 'Return', 'color': 'Sharpe Ratio'},
                     title='Efficient Frontier')
    st.plotly_chart(fig)

def create_pdf_report(tickers, optimized_weights, port_return, port_std, sharpe_ratio, var, es, cvar, max_drawdown):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    pdf.cell(200, 10, txt="Portfolio Optimization Report", ln=True, align='C')
    
    pdf.cell(200, 10, txt=f"Tickers: {', '.join(tickers)}", ln=True)
    pdf.cell(200, 10, txt=f"Weights: {', '.join([f'{w:.2%}' for w in optimized_weights])}", ln=True)
    
    pdf.cell(200, 10, txt=f"Expected Annual Return: {port_return:.2%}", ln=True)
    pdf.cell(200, 10, txt=f"Annual Volatility (Standard Deviation): {port_std:.2%}", ln=True)
    pdf.cell(200, 10, txt=f"Sharpe Ratio: {sharpe_ratio:.2f}", ln=True)
    pdf.cell(200, 10, txt=f"Value at Risk (VaR) at 95% confidence level: {var:.2%}", ln=True)
    pdf.cell(200, 10, txt=f"Expected Shortfall (ES) at 95% confidence level: {es:.2%}", ln=True)
    pdf.cell(200, 10, txt=f"Conditional Value at Risk (CVaR) at 95% confidence level: {cvar:.2%}", ln=True)
    pdf.cell(200, 10, txt=f"Maximum Drawdown: {max_drawdown:.2%}", ln=True)
    
    pdf_buffer = pdf.output(dest='S').encode('latin1')
    return pdf_buffer

def enforce_asset_class_constraints(weights, asset_classes, min_allocation, max_allocation):
    if weights is None:
        return []

    constraints = []
    unique_classes = np.unique(asset_classes)

    for asset_class in unique_classes:
        class_indices = [i for i, ac in enumerate(asset_classes) if ac == asset_class]
        constraints.append({
            'type': 'ineq',
            'fun': lambda w, idx=class_indices, min_allocation=min_allocation: np.sum(w[idx]) - min_allocation,
        })
        constraints.append({
            'type': 'ineq',
            'fun': lambda w, idx=class_indices, max_allocation=max_allocation: max_allocation - np.sum(w[idx]),
        })

    return constraints

def optimize_portfolio(mean_returns, cov_matrix, additional_constraints=None):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix)
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
    bounds = tuple((0, 1) for _ in range(num_assets))
    
    if additional_constraints:
        constraints.extend(additional_constraints)
    
    # Add a constraint to ensure no single asset has more than 30% of the portfolio
    max_allocation_per_asset = 0.3
    constraints.append({'type': 'ineq', 'fun': lambda weights: max_allocation_per_asset - np.max(weights)})

    # Objective function to maximize return and minimize risk
    def objective_function(weights, mean_returns, cov_matrix):
        return -1 * (np.sum(weights * mean_returns) - 0.01) / np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    result = minimize(
        objective_function,
        num_assets * [1. / num_assets,],
        args=args, method='SLSQP', bounds=bounds, constraints=constraints
    )
    return result

def main():
    st.title("Portfolio Optimization Tool")

    tickers = st.text_input("Enter ticker symbols separated by commas (e.g., AAPL,MSFT,GOOGL,AMZN,TSLA):", "AAPL,MSFT,GOOGL,AMZN,TSLA").split(',')
    data = yf.download(tickers, start="2020-01-01")['Adj Close']
    
    if data.isnull().values.any():
        st.write("Data contains null values. Please check the ticker symbols and try again.")
        return
    
    mean_returns = data.pct_change().mean()
    cov_matrix = data.pct_change().cov()
    
    apply_constraints = st.checkbox("Apply asset class constraints?")
    additional_constraints = None
    
    if apply_constraints:
        asset_classes = st.text_input("Enter asset classes separated by a comma (e.g., Tech,Tech,Tech,Tech,Tech):", "Tech,Tech,Tech,Tech,Tech").split(',')
        min_allocation = float(st.text_input("Enter minimum allocation for each class:", "0.1"))
        max_allocation = float(st.text_input("Enter maximum allocation for each class:", "0.5"))
        additional_constraints = enforce_asset_class_constraints(None, asset_classes, min_allocation, max_allocation)
    
    result = optimize_portfolio(mean_returns, cov_matrix, additional_constraints=additional_constraints)
    
    if not result.success:
        st.write("Optimization failed. Please check your inputs and try again.")
        return
    
    optimized_weights = result.x
    st.write("Optimized Portfolio Weights:")
    weights_df = pd.DataFrame({'Ticker': tickers, 'Weight': optimized_weights})
    st.write(weights_df)
    
    st.write("Portfolio Performance:")
    port_return, port_std = np.sum(mean_returns * optimized_weights), np.sqrt(np.dot(optimized_weights.T, np.dot(cov_matrix, optimized_weights)))
    sharpe_ratio = (port_return - 0.01) / port_std
    st.write(f"Expected Annual Return: {port_return:.2%}")
    st.write(f"Annual Volatility (Standard Deviation): {port_std:.2%}")
    st.write(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    
    # Covariance matrix heatmap
    st.write("Covariance Matrix:")
    plt.figure(figsize=(10, 7))
    sns.heatmap(cov_matrix, annot=True, cmap='coolwarm', xticklabels=tickers, yticklabels=tickers)
    st.pyplot(plt)
    
    # Historical closing prices
    st.write("Historical Adjusted Closing Prices:")
    st.line_chart(data)
    
    # Portfolio weights pie chart
    st.write("Portfolio Weights Distribution:")
    fig, ax = plt.subplots()
    ax.pie(optimized_weights, labels=tickers, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    st.pyplot(fig)
    
    # Value at Risk (VaR)
    returns = data.pct_change().dropna().dot(optimized_weights)
    var = calculate_var(returns)
    st.write(f"Value at Risk (VaR) at 95% confidence level: {var:.2%}")
    
    # Expected Shortfall (ES)
    es = calculate_es(returns)
    st.write(f"Expected Shortfall (ES) at 95% confidence level: {es:.2%}")
    
    # Conditional Value at Risk (CVaR)
    cvar = calculate_cvar(returns)
    st.write(f"Conditional Value at Risk (CVaR) at 95% confidence level: {cvar:.2%}")
    
    # Maximum Drawdown
    max_drawdown = calculate_max_drawdown(returns)
    st.write(f"Maximum Drawdown: {max_drawdown:.2%}")
    
    # Monte Carlo simulation
    st.write("Monte Carlo Simulation:")
    num_simulations = st.number_input("Number of Simulations:", 1000, 100000, 10000)
    simulation_results = monte_carlo_simulation(mean_returns, cov_matrix, num_simulations, len(tickers))
    simulation_df = pd.DataFrame({'Return': simulation_results[0], 'Volatility': simulation_results[1], 'Sharpe Ratio': simulation_results[2], 'Max Drawdown': simulation_results[3]})
    fig = px.scatter(simulation_df, x='Volatility', y='Return', color='Sharpe Ratio', title='Monte Carlo Simulation Results')
    st.plotly_chart(fig)
    
    # PDF report generation
    st.write("Generate PDF Report:")
    generate_pdf = st.button("Generate PDF")
    if generate_pdf:
        pdf_buffer = create_pdf_report(tickers, optimized_weights, port_return, port_std, sharpe_ratio, var, es, cvar, max_drawdown)
        st.download_button("Download PDF", data=pdf_buffer, file_name="Portfolio_Optimization_Report.pdf", mime='application/pdf')

if __name__ == "__main__":
    main()
