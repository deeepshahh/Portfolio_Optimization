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

def enforce_asset_class_constraints(weights, asset_classes, min_allocation, max_allocation):
    if weights is None:
        return []

    constraints = []
    unique_classes = np.unique(asset_classes)

    for asset_class in unique_classes:
        class_indices = [i for i, ac in enumerate(asset_classes) if ac == asset_class]
        constraints.append({
            'type': 'ineq',
            'fun': lambda w: np.sum(w[class_indices]) - float(min_allocation),
        })
        constraints.append({
            'type': 'ineq',
            'fun': lambda w: float(max_allocation) - np.sum(w[class_indices]),
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

def load_data(tickers, start_date):
    data = yf.download(tickers, start=start_date)['Adj Close']
    if data.isnull().values.any():
        data = data.fillna(method='ffill').fillna(method='bfill')
        if data.isnull().values.any():
            st.write("Data contains null values. Please check the ticker symbols and try again.")
            return None
    return data

def main():
    st.title("Portfolio Optimizer")

    portfolio_type = st.selectbox("Select Portfolio Type:", ["Multi Asset Portfolio", "Equity Portfolio"])

    if portfolio_type == "Multi Asset Portfolio":
        asset_classes = st.multiselect("Select Asset Classes:", ["Equity", "Commodities", "Currency", "Debt Market"])
        if "Equity" in asset_classes:
            equity_tickers = st.text_input("Enter equity ticker symbols separated by commas:", "AAPL,MSFT,GOOGL,AMZN")
        if "Commodities" in asset_classes:
            commodities = st.text_input("Enter commodities symbols separated by commas:", "GC=F,SI=F")
        if "Currency" in asset_classes:
            currencies = st.text_input("Enter currency pairs separated by commas:", "USDINR=X,EURUSD=X")
        if "Debt Market" in asset_classes:
            debt_market = st.text_input("Enter debt market symbols separated by commas:", "SHY,IEF")

        start_date = st.date_input("Select start date for historical data:", pd.to_datetime("2020-01-01"))
        data = load_data(equity_tickers + commodities + currencies + debt_market, start_date)
        if data is None:
            return

        mean_returns = data.pct_change().mean()
        cov_matrix = data.pct_change().cov()

        apply_constraints = st.checkbox("Apply asset class constraints?")
        additional_constraints = None

        if apply_constraints:
            asset_classes = st.text_input("Enter asset classes separated by a comma:", "Equity,Equity,Equity,Equity").split(',')
            min_allocation = list(map(float, st.text_input("Enter minimum allocation for each class separated by a comma:", "0.1, 0.1, 0.1").split(',')))
            max_allocation = list(map(float, st.text_input("Enter maximum allocation for each class separated by a comma:", "0.5, 0.3, 0.2").split(',')))
            additional_constraints = enforce_asset_class_constraints(None, asset_classes, min_allocation, max_allocation)

        result = optimize_portfolio(mean_returns, cov_matrix, additional_constraints=additional_constraints)

        if not result.success:
            st.write("Optimization failed. Please check your inputs and try again.")
            return

        optimized_weights = result.x
        st.write("Optimized Portfolio Weights:")
        weights_df = pd.DataFrame({'Ticker': data.columns, 'Weight': optimized_weights})
        st.write(weights_df)

        st.write("Portfolio Performance:")
        port_return, port_std = np.sum(mean_returns * optimized_weights), np.sqrt(np.dot(optimized_weights.T, np.dot(cov_matrix, optimized_weights)))
        sharpe_ratio = (port_return - 0.01) / port_std
        st.write(f"Expected Annual Return: {port_return:.2%}")
        st.write(f"Annual Volatility (Standard Deviation): {port_std:.2%}")
        st.write(f"Sharpe Ratio: {sharpe_ratio:.2f}")

        st.write("Covariance Matrix:")
        plt.figure(figsize=(10, 7))
        sns.heatmap(cov_matrix, annot=True, cmap='coolwarm', xticklabels=data.columns, yticklabels=data.columns)
        st.pyplot(plt)

        st.write("Historical Adjusted Closing Prices:")
        st.line_chart(data)

        st.write("Portfolio Weights Distribution:")
        fig, ax = plt.subplots()
        ax.pie(optimized_weights, labels=data.columns, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
        st.pyplot(fig)

        returns = data.pct_change().dropna().dot(optimized_weights)
        var = calculate_var(returns)
        st.write(f"Value at Risk (VaR) at 95% confidence level: {var:.2%}")

        es = calculate_es(returns)
        st.write(f"Expected Shortfall (ES) at 95% confidence level: {es:.2%}")

        pdf_report = create_pdf_report(data.columns, optimized_weights, port_return, port_std, sharpe_ratio, var, es)
        st.download_button(label="Download Report", data=pdf_report, file_name='Portfolio_Report.pdf', mime='application/pdf')

        st.write("Efficient Frontier:")
        plot_efficient_frontier(mean_returns, cov_matrix)

    elif portfolio_type == "Equity Portfolio":
        equity_option = st.selectbox("Do you want to optimize:", ["Your chosen stocks", "Indexes"])
        if equity_option == "Your chosen stocks":
            tickers = st.text_input("Enter ticker symbols separated by commas:", "AAPL,MSFT,GOOGL,AMZN")
            start_date = st.date_input("Select start date for historical data:", pd.to_datetime("2020-01-01"))
            data = load_data(tickers, start_date)
            if data is None:
                return

            mean_returns = data.pct_change().mean()
            cov_matrix = data.pct_change().cov()

            result = optimize_portfolio(mean_returns, cov_matrix)

            if not result.success:
                st.write("Optimization failed. Please check your inputs and try again.")
                return

            optimized_weights = result.x
            st.write("Optimized Portfolio Weights:")
            weights_df = pd.DataFrame({'Ticker': data.columns, 'Weight': optimized_weights})
            st.write(weights_df)

            st.write("Portfolio Performance:")
            port_return, port_std = np.sum(mean_returns * optimized_weights), np.sqrt(np.dot(optimized_weights.T, np.dot(cov_matrix, optimized_weights)))
            sharpe_ratio = (port_return - 0.01) / port_std
            st.write(f"Expected Annual Return: {port_return:.2%}")
            st.write(f"Annual Volatility (Standard Deviation): {port_std:.2%}")
            st.write(f"Sharpe Ratio: {sharpe_ratio:.2f}")

            st.write("Covariance Matrix:")
            plt.figure(figsize=(10, 7))
            sns.heatmap(cov_matrix, annot=True, cmap='coolwarm', xticklabels=data.columns, yticklabels=data.columns)
            st.pyplot(plt)

            st.write("Historical Adjusted Closing Prices:")
            st.line_chart(data)

            st.write("Portfolio Weights Distribution:")
            fig, ax = plt.subplots()
            ax.pie(optimized_weights, labels=data.columns, autopct='%1.1f%%', startangle=90)
            ax.axis('equal')
            st.pyplot(fig)

            returns = data.pct_change().dropna().dot(optimized_weights)
            var = calculate_var(returns)
            st.write(f"Value at Risk (VaR) at 95% confidence level: {var:.2%}")

            es = calculate_es(returns)
            st.write(f"Expected Shortfall (ES) at 95% confidence level: {es:.2%}")

            pdf_report = create_pdf_report(data.columns, optimized_weights, port_return, port_std, sharpe_ratio, var, es)
            st.download_button(label="Download Report", data=pdf_report, file_name='Portfolio_Report.pdf', mime='application/pdf')

            st.write("Efficient Frontier:")
            plot_efficient_frontier(mean_returns, cov_matrix)

        elif equity_option == "Indexes":
            indexes = st.text_input("Enter index symbols separated by commas:", "^GSPC,^DJI,^IXIC,^BSESN")
            start_date = st.date_input("Select start date for historical data:", pd.to_datetime("2020-01-01"))
            data = load_data(indexes, start_date)
            if data is None:
                return

            mean_returns = data.pct_change().mean()
            cov_matrix = data.pct_change().cov()

            result = optimize_portfolio(mean_returns, cov_matrix)

            if not result.success:
                st.write("Optimization failed. Please check your inputs and try again.")
                return

            optimized_weights = result.x
            st.write("Optimized Portfolio Weights:")
            weights_df = pd.DataFrame({'Index': data.columns, 'Weight': optimized_weights})
            st.write(weights_df)

            st.write("Portfolio Performance:")
            port_return, port_std = np.sum(mean_returns * optimized_weights), np.sqrt(np.dot(optimized_weights.T, np.dot(cov_matrix, optimized_weights)))
            sharpe_ratio = (port_return - 0.01) / port_std
            st.write(f"Expected Annual Return: {port_return:.2%}")
            st.write(f"Annual Volatility (Standard Deviation): {port_std:.2%}")
            st.write(f"Sharpe Ratio: {sharpe_ratio:.2f}")

            st.write("Covariance Matrix:")
            plt.figure(figsize=(10, 7))
            sns.heatmap(cov_matrix, annot=True, cmap='coolwarm', xticklabels=data.columns, yticklabels=data.columns)
            st.pyplot(plt)

            st.write("Historical Adjusted Closing Prices:")
            st.line_chart(data)

            st.write("Portfolio Weights Distribution:")
            fig, ax = plt.subplots()
            ax.pie(optimized_weights, labels=data.columns, autopct='%1.1f%%', startangle=90)
            ax.axis('equal')
            st.pyplot(fig)

            returns = data.pct_change().dropna().dot(optimized_weights)
            var = calculate_var(returns)
            st.write(f"Value at Risk (VaR) at 95% confidence level: {var:.2%}")

            es = calculate_es(returns)
            st.write(f"Expected Shortfall (ES) at 95% confidence level: {es:.2%}")

            pdf_report = create_pdf_report(data.columns, optimized_weights, port_return, port_std, sharpe_ratio, var, es)
            st.download_button(label="Download Report", data=pdf_report, file_name='Portfolio_Report.pdf', mime='application/pdf')

            st.write("Efficient Frontier:")
            plot_efficient_frontier(mean_returns, cov_matrix)

if __name__ == "__main__":
    main()
