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

# read the excel file and get the positions and long/short indicators
df = pd.read_excel('fund_positions.xlsx', sheet_name='Sheet1')
# print(df)
positions = df['Positions']
long_short = df['long/short']

print(df)
# get historical prices and returns
mgr = dm.BbgDataManager()
sids = mgr[positions]
mgr.sid_result_mode = 'frame'
start_date = dt.datetime(2022, 12, 30)
end_date   = dt.datetime(2023, 12, 29)

df = sids.get_historical(['PX_LAST'], start_date, end_date)
print(df)
df = df.pct_change()

print(df)
df.dropna(inplace=True)
print(df)