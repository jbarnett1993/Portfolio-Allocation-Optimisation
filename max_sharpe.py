from __future__ import division
import numpy as np
import pandas as pd
import datetime as dt 
import cvxpy as cp
from datetime import datetime
from scipy.optimize import minimize
from dateutil.relativedelta import relativedelta
import tia.bbg.datamgr as dm
from dateutil.relativedelta import relativedelta
import numpy as np
from numpy.linalg import inv,pinv
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# read the excel file and get the positions and long/short indicators
df = pd.read_excel('fund_positions.xlsx', sheet_name='Sheet1')
positions = df['Positions']

# get historical prices and returns
mgr = dm.BbgDataManager()
sids = mgr[positions]
mgr.sid_result_mode = 'frame'
start_date = dt.datetime(2024, 1 ,1) 
end_date   = dt.datetime(2024, 5 , 14)


# start_date = (datetime.today() - relativedelta(years=1)).strftime('%Y-%m-%d')
# end_date = datetime.today().strftime('%Y-%m-%d')


df = sids.get_historical(['PX_LAST'], start_date, end_date)
df.to_csv("sids with prices.csv")

df_returns = df.pct_change()


df_returns.dropna(inplace=True)
df_returns.to_csv("returns.csv")


covariance_matrix = df_returns.cov()

def portfolio_annualized_sharpe(weights, returns):

    portfolio_returns = np.dot(returns, weights)
    portfolio_total_return = np.prod(1 + portfolio_returns) - 1
    nb_years = (returns.index[-1] - returns.index[0]).days / 365
    portfolio_std_dev = np.std(portfolio_returns)
    annual_r = (portfolio_total_return + 1) ** (1 / nb_years) - 1
    annual_std = portfolio_std_dev * np.sqrt(252)
    if annual_std == 0:
        return np.nan
    return annual_r / annual_std


# Optimization constraints and bounds
constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]  # Sum of weights must be 1
# bounds = [(-0.20, 0.20) for _ in range(len(positions))]  # Long/short constraints

# Objective function (to be minimized)
def objective_function(weights):
    return -portfolio_annualized_sharpe(weights, df_returns)



# Initial guess for weights
initial_weights = np.array([1. / len(positions)] * len(positions))

sharpe_ratio = portfolio_annualized_sharpe(initial_weights, df_returns)
# bounds = [(-20,20) for _ in range(len(positions))]
# Optimization
optimal_weights = minimize(objective_function, initial_weights, method='SLSQP', constraints=constraints)

# Display optimized weights
optimized_weights = pd.Series(optimal_weights.x, index=df_returns.columns)
portfolio_returns = np.dot(df_returns, optimized_weights)


weighted_returns = df_returns.multiply(optimized_weights, axis=1)
daily_weighted_returns = weighted_returns.sum(axis=1)
cumulative_returns = (1 + daily_weighted_returns).cumprod()
total_port_return = (1+daily_weighted_returns).prod()-1
print(total_port_return)



sharpe_ratio = portfolio_annualized_sharpe(optimized_weights, df_returns)
print(sharpe_ratio)
print(optimized_weights)

#  Drawdown Calculation
rolling_max = cumulative_returns.cummax()
drawdowns = (cumulative_returns - rolling_max) / rolling_max

# Set the overall figure size
plt.figure(figsize=(10, 12))

# Set common title
plt.suptitle('max_sharpe Performance', fontsize=16)

# Plot Cumulative Returns
plt.subplot(3, 1, 1)  # 3 rows, 1 column, 1st subplot
cumulative_plot = plt.plot(cumulative_returns, label='Cumulative Return', linewidth=2)
plt.title('Cumulative Return', fontsize=14)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Cumulative Return', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True)

# Adjust x-axis date formatting
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %d %Y'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())

# Plot Daily Returns
plt.subplot(3, 1, 2)  # 3 rows, 1 column, 2nd subplot
daily_plot = plt.bar(daily_weighted_returns.index, daily_weighted_returns, label='Daily Return', linewidth=2)
plt.title('Daily Return', fontsize=14)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Daily Return', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True)

# Plot Drawdown
plt.subplot(3, 1, 3)  # 3 rows, 1 column, 3rd subplot
drawdown_plot = plt.plot(drawdowns, label='Drawdown', linewidth=2, color='red')
plt.title('Drawdown', fontsize=14)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Drawdown', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True)

# Adjust layout to prevent overlap
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# Show plot with adjustments
plt.show()