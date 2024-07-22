import yfinance as yf
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from scipy.optimize import minimize
from fpdf import FPDF

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

def monte_carlo_simulation(mean_returns, cov_matrix, num_simulations):
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
    
    pdf.cell(200, 10, txt=f"Expected Annual Return: {port_return:.2%}", ln=True)
    pdf.cell(200, 10, txt=f"Annual Volatility (Standard Deviation): {port_std:.2%}", ln=True)
    pdf.cell(200, 10, txt=f"Sharpe Ratio: {sharpe_ratio:.2f}", ln=True)
    pdf.cell(200, 10, txt=f"Value at Risk (VaR) at 95% confidence level: {var:.2%}", ln=True)
    pdf.cell(200, 10, txt=f"Expected Shortfall (ES) at 95% confidence level: {es:.2%}", ln=True)
    
    pdf_buffer = pdf.output(dest='S').encode('latin1')
    return pdf_buffer

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
    st.title("Portfolio Optimization Tool (Indian Stocks)")

    tickers = st.text_input("Enter ticker symbols separated by commas (e.g., HDFCBANK.NS,RELIANCE.NS,TCS.NS):", "HDFCBANK.NS,RELIANCE.NS,TCS.NS").split(',')
    investment_amount = st.number_input("Enter the investment amount (in INR):", min_value=0.0, value=100000.0)
    investment_date = st.date_input("Enter the investment date:", value=pd.to_datetime("2020-01-01"))

    data = yf.download(tickers, start=investment_date)['Adj Close']

    if data.isnull().values.any():
        st.write("Data contains null values. Attempting to fill missing data.")
        data = data.fillna(method='ffill').fillna(method='bfill')
    
    if data.isnull().values.any():
        st.write("Data still contains null values. Please check the ticker symbols and try again.")
        return

    mean_returns = data.pct_change().mean()
    cov_matrix = data.pct_change().cov()

    result = optimize_portfolio(mean_returns, cov_matrix)
    
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
    
    st.write("Historical Adjusted Closing Prices:")
    st.line_chart(data)
    
    st.write("Portfolio Weights Distribution:")
    fig, ax = plt.subplots()
    ax.pie(optimized_weights, labels=tickers, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    st.pyplot(fig)
    
    # Covariance matrix heatmap
    st.write("Covariance Matrix:")
    plt.figure(figsize=(10, 7))
    sns.heatmap(cov_matrix, annot=True, cmap='coolwarm', xticklabels=tickers, yticklabels=tickers)
    st.pyplot(plt)
    
    plot_efficient_frontier(mean_returns, cov_matrix)
    
    # Calculate VaR and ES for the optimized portfolio
    portfolio_returns = data.pct_change().dot(optimized_weights)
    var_95 = calculate_var(portfolio_returns, alpha=0.05)
    es_95 = calculate_es(portfolio_returns, alpha=0.05)
    st.write(f"Value at Risk (VaR) at 95% confidence level: {var_95:.2%}")
    st.write(f"Expected Shortfall (ES) at 95% confidence level: {es_95:.2%}")
    
    # Create PDF report
    pdf_buffer = create_pdf_report(tickers, optimized_weights, port_return, port_std, sharpe_ratio, var_95, es_95)
    st.download_button(label="Download Report", data=pdf_buffer, file_name="Portfolio_Optimization_Report.pdf", mime="application/pdf")

if __name__ == "__main__":
    main()
