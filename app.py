import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.optimize import minimize
import math
from scipy.stats import skew, kurtosis

# Description using a mix of st.write and st.latex
st.write("## Portfolio Replication and Optimization")

with st.expander("Explanation"):
    st.write("This application explores the replication of ETFs via portfolio optimization by visualizing popular risk and return metrics for an \"in-sample\" replication window to an \"out of sample\" future window.")
    st.write("The objective function \( f(x) \) aims to minimize the Tracking Error as:")
    st.latex(r"f(x) = \sum_{t=1}^{T} \left( R_{\text{ETF}, t} - R_{\text{Portfolio}, t} \right)^2")

    st.write("Where:")
    st.latex(r"R_{t} = \frac{P_{t} - P_{t-1}}{P_{t-1}}")

    st.write("Rebasing is done using the formula:")
    st.latex(r"\text{Rebased}_{t} = \text{Price}_{t} \times \frac{\text{Base Price}}{\text{Price}_{\text{initial}}}")

    st.write("Keep in mind optimization of the discrete allocation is a function of several parameters available to the user on the side panel. Moreover, we explore the future out of sample performance of the optimized portfolio.")

# Extended list of largest stocks in various ETFs
ETF_STOCKS = {
    'SPY': ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'NVDA', 'BRK-B', 'JPM', 'TSLA', 'UNH'],
    'QQQ': ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'NVDA', 'ADBE', 'PYPL', 'INTC'],
    'DIA': ['UNH', 'GS', 'HD', 'CRM', 'MSFT', 'V', 'TRV', 'JPM', 'MMM', 'BA']
}

# Function to compute Sharpe and Market Sharpe Ratio
def compute_sharpe_and_mkt_sharpe(portfolio_returns, etf_returns):
    # Calculate the excess returns over ETF returns
    excess_returns = portfolio_returns - etf_returns

    # Compute the Sharpe Ratio
    sharpe_ratio = (np.mean(excess_returns)*252) / (np.std(excess_returns)*np.sqrt(252))
    
    market_sharpe = (np.mean(etf_returns)*252) / (np.std(etf_returns)*np.sqrt(252))

    return sharpe_ratio, market_sharpe

def fetch_data(stock, start, end):
    stock_data = yf.download(stock, start=start, end=end)
    return stock_data['Adj Close']

def optimize_portfolio(selected_stocks, target_etf, capital_allocation, start_date, end_date):
    stocks_data = {stock: fetch_data(stock, start_date, end_date) for stock in selected_stocks}
    target_data = fetch_data(target_etf, start_date, end_date)

    df = pd.DataFrame(stocks_data)
    df['Target_ETF'] = target_data
    df.dropna(inplace=True)

    if df.empty or df.shape[0] < 2:
        return None, None, "Insufficient data"

    stock_prices = df.iloc[-1, :-1]

    def objective(shares):
        portfolio_value = sum(stock_prices * shares)
        portfolio_return = df[selected_stocks].apply(lambda x: x * shares, axis=1).sum(axis=1)
        tracking_error = ((df['Target_ETF'] - portfolio_return) ** 2).mean()
        return tracking_error

    initial_shares = [1 for _ in selected_stocks]
    bounds = [(0, math.floor(capital_allocation / price)) for price in stock_prices]
    result = minimize(objective, initial_shares, bounds=bounds, method='SLSQP')
    optimal_shares = np.round(result.x).astype(int)

    # Calculate the value of the optimized portfolio with original optimal shares
    portfolio_value = sum(stock_prices * optimal_shares)

    # Calculate an adjustment factor to match the portfolio value with capital allocation
    adjustment_factor = capital_allocation / portfolio_value
    
    # Adjust the optimal shares according to the adjustment factor
    adjusted_optimal_shares = np.round(optimal_shares * adjustment_factor).astype(int)

    return adjusted_optimal_shares, df, None

def compute_var_cvar(returns, capital, alpha=0.01, days=252):
    VaR = returns.quantile(alpha) * capital * np.sqrt(days)  # Scale to one year
    CVaR = returns[returns <= returns.quantile(alpha)].mean() * capital * np.sqrt(days)  # Scale to one year
    return VaR, CVaR

# Sidebar for user inputs
st.sidebar.header('User Inputs')
target_etf = st.sidebar.selectbox('Select Target ETF', list(ETF_STOCKS.keys()))
selected_stocks = st.sidebar.multiselect('Select Stocks', ETF_STOCKS[target_etf])
capital_allocation = st.sidebar.number_input('Capital Allocation ($)', min_value=1000, max_value=1000000, value=10000)
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime('2020-01-01'))
end_date = st.sidebar.date_input("Replication End Date", value=pd.to_datetime('2021-01-01'))
st.sidebar.title("A BytePotion App")
st.sidebar.image("bytepotion.png", width=200)  # Replace with your image URL or file path
st.sidebar.write("This app explores replication through the lens of portfolio optimization and common risk and return metrics.")
st.sidebar.write("https://bytepotion.com")
st.sidebar.title("Author")
st.sidebar.image("roman2.png", width=200)  # Replace with your image URL or file path
st.sidebar.write("Roman Paolucci")
st.sidebar.write("MSOR Graduate Student @ Columbia University")
st.sidebar.write("roman.paolucci@columbia.edu")

# Optimize portfolio
if st.button('Optimize Portfolio'):
    if len(selected_stocks) == 0:
        st.error("No stocks are selected to replicate the ETF")
    else:
        optimal_shares, df, error_message = optimize_portfolio(selected_stocks, target_etf, capital_allocation, start_date, end_date)

        if error_message:
            st.error(error_message)
        else:
            st.write('Optimal Shares:', {stock: shares for stock, shares in zip(selected_stocks, optimal_shares)})

            # Replication period calculations and visualizations
            portfolio = df[selected_stocks].apply(lambda x: x * optimal_shares, axis=1).sum(axis=1)
            portfolio_value = sum(df[selected_stocks].iloc[-1] * optimal_shares)
            
            rebase_factor = capital_allocation / portfolio_value  # Rebase using the end-of-period portfolio value
            portfolio_rebased = portfolio * rebase_factor
            etf_rebased = df['Target_ETF'] * (capital_allocation / df['Target_ETF'].iloc[0])

            plot_df = pd.DataFrame({'ETF': etf_rebased, 'Optimized Portfolio': portfolio_rebased})
            fig1 = px.line(plot_df)
            fig1.update_layout(title="Performance during Replication Period")
            st.plotly_chart(fig1)

            correlation_replication = np.corrcoef(portfolio_rebased, etf_rebased)[0, 1]
            st.write(f"Correlation during Replication Period: {correlation_replication:.2f}")

            returns_etf = etf_rebased.pct_change().dropna()
            returns_portfolio = portfolio_rebased.pct_change().dropna()

            # Note: Use compute_var_cvar with days=252 for annualized VaR/CVaR
            VaR_etf, CVaR_etf = compute_var_cvar(returns_etf, capital_allocation, days=252)
            VaR_portfolio, CVaR_portfolio = compute_var_cvar(returns_portfolio, capital_allocation, days=252)
            
            # Compute Sharpe Ratio and Market Sharpe Ratio for replication period
            sharpe_replication, mkt_sharpe_replication = compute_sharpe_and_mkt_sharpe(returns_portfolio, returns_etf)

            # VaR and CVaR table along with Sharpe and Market Sharpe Ratios
            var_cvar_sharpe_df_replication = pd.DataFrame({
                'VaR': [VaR_etf, VaR_portfolio],
                'CVaR': [CVaR_etf, CVaR_portfolio],
                'Sharpe (annum)': [mkt_sharpe_replication, sharpe_replication] 
            }, index=['ETF', 'Optimized Portfolio'])

            st.write('Metrics During Replication Period')
            st.table(var_cvar_sharpe_df_replication)
            
            # Out-of-sample calculations and visualizations
            current_end_date = pd.to_datetime('today').strftime('%Y-%m-%d')
            new_stock_data = {stock: fetch_data(stock, end_date, current_end_date) for stock in selected_stocks}
            new_target_data = fetch_data(target_etf, end_date, current_end_date)
            
            df_new = pd.DataFrame(new_stock_data)
            df_new['Target_ETF'] = new_target_data
            df_new.dropna(inplace=True)
            
            portfolio_out_of_sample = df_new[selected_stocks].apply(lambda x: x * optimal_shares, axis=1).sum(axis=1)
            portfolio_out_of_sample_rebased = portfolio_out_of_sample * (capital_allocation / portfolio_out_of_sample.iloc[0])
            new_data_rebased = new_target_data * (capital_allocation / new_target_data.iloc[0])
            
            plot_df_new = pd.DataFrame({'ETF': new_data_rebased, 'Optimized Portfolio': portfolio_out_of_sample_rebased})
            fig2 = px.line(plot_df_new)
            fig2.update_layout(title="Performance after Replication Period")
            st.plotly_chart(fig2)

            correlation_out_of_sample = np.corrcoef(new_data_rebased.dropna(), portfolio_out_of_sample_rebased.dropna())[0, 1]
            st.write(f"Correlation after Replication Period: {correlation_out_of_sample:.2f}")

            returns_etf_new = new_data_rebased.pct_change().dropna()
            returns_portfolio_new = portfolio_out_of_sample_rebased.pct_change().dropna()

            VaR_etf_new, CVaR_etf_new = compute_var_cvar(returns_etf_new, capital_allocation, days=252)
            VaR_portfolio_new, CVaR_portfolio_new = compute_var_cvar(returns_portfolio_new, capital_allocation, days=252)

            # Compute Sharpe Ratio and Market Sharpe Ratio for out-of-sample period
            sharpe_out_of_sample, mkt_sharpe_out_of_sample = compute_sharpe_and_mkt_sharpe(returns_portfolio_new, returns_etf_new)

            # VaR and CVaR table along with Sharpe and Market Sharpe Ratios for out-of-sample period
            var_cvar_sharpe_df_out_of_sample = pd.DataFrame({
                'VaR': [VaR_etf_new, VaR_portfolio_new],
                'CVaR': [CVaR_etf_new, CVaR_portfolio_new],
                'Sharpe (annum)': [mkt_sharpe_out_of_sample, sharpe_out_of_sample]
            }, index=['ETF', 'Optimized Portfolio'])

            st.write('Metrics after Replication Period')
            st.table(var_cvar_sharpe_df_out_of_sample)
            


