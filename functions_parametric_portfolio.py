import pandas as pd
import numpy as np

import statsmodels.api as sm
from pandas.tseries.offsets import *
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
from functions_asset_pricing import *

def BPR_portfolio(data, stock_name, date_name, return_name, frequency, industry_name, rolling_window, gamma, weight_winsorize, chars, vars_keep):
# frequency
    if frequency == 'daily':
        date_frequency = Day(rolling_window)
        unit_date = Day(1)
    
    if frequency == 'monthly':
        date_frequency = MonthEnd(rolling_window)
        unit_date = MonthEnd(1)

# estimator names
    theta_name = []
    for char in chars:
        theta_name = ['theta_' + char] + theta_name

# keep variables used in weights calculation
    df = data[[stock_name, date_name, return_name]+chars+vars_keep]

    dates = sorted(pd.unique(df[date_name]))
    dates = dates[rolling_window:]

# create benchmark weight
# one unit date lagged
    benchmark_weight = pd.DataFrame(1/df.groupby(date_name)[stock_name].count()).rename(columns={stock_name:'w_benchmark'})
    benchmark_weight.index = benchmark_weight.index + unit_date
    df = pd.merge(df, benchmark_weight, how='left', left_on=date_name, right_index=True)

    results = pd.DataFrame()

# no industry
# rolling date optimization
    for date in dates:
        df_temp = df[(df[date_name]<=date) & (df[date_name]>=(date-date_frequency-unit_date))]
        df_train = df_temp[df_temp[date_name] < date]
        df_test = df_temp[df_temp[date_name] == date]
    
# function for 3 chars
# utility
        fun = lambda x: -1/rolling_window * np.power(
        1 + np.sum(df_train[return_name]*(df_train['w_benchmark']+df_train['w_benchmark']*(
            x[0]*df_train[chars[0]]+x[1]*df_train[chars[1]]+x[2]*df_train[chars[2]]+x[3]*df_train[chars[3]]))),
                                                    1-gamma)/(1-gamma)
# constraint        
        cons = ({'type':'ineq','fun':lambda x:x[0] + 10},
                {'type':'ineq','fun':lambda x:10 - x[0]},
                {'type':'ineq','fun':lambda x:x[1] + 10},
                {'type':'ineq','fun':lambda x:10 - x[1]},
                {'type':'ineq','fun':lambda x:x[2] + 10},
                {'type':'ineq','fun':lambda x:10 - x[2]},
                {'type':'ineq','fun':lambda x:x[3] + 10},
                {'type':'ineq','fun':lambda x:10 - x[3]},
                {'type':'ineq','fun':lambda x:x[4] + 10},
                {'type':'ineq','fun':lambda x:10 - x[4]},
               )
        initials = [0,0,0,0,0]
    
# optimization
        res = minimize(fun, initials, method='SLSQP', constraints=cons)

# save results
        result_temp = pd.DataFrame(res.x,index=theta_name).T
        result_temp[date_name] = date
        results = results.append(result_temp)

# merge optimization results
    df = pd.merge(df, results, how='left', on=date_name)

# calcualte parametric portfolio weight
# initialize the value of weight
    df['w_raw'] = 0

    for char in chars:
        df['w_raw'] = df['w_raw']+ df[char]*df['theta_'+char]
    
    df['w_raw'] = df['w_benchmark'] + df['w_benchmark']*df['w_raw']
#df['w_raw'] = df['w_benchmark'] + df['w_raw']

# weight winsorize
# df = get_winsorize(data=df, var_name='w_raw', date_name='date', 
#                   left_ratio=0.05, right_ratio=0.95)
    if weight_winsorize==True:
        df = get_winsorize_all(data=df, var_name='w_raw',
                   left_ratio=0.10, right_ratio=0.88)

# normalize the sum of weights to 1
    weights_normalize = df.groupby(date_name).apply(lambda x: x['w_raw']/np.sum(x['w_raw'])).reset_index(drop=False).rename(columns={'w_raw':'w_raw_normalize'})
    weights_normalize.index = weights_normalize['level_1']
    df = pd.merge(df, weights_normalize[['w_raw_normalize']], how='left', left_index=True, right_index=True)
    
    return df

def get_portfolio_chars(df,chars,date_name,rolling_window):
    for char in chars:
        port_chars_temp = pd.DataFrame(df.groupby(date_name).apply(lambda x: np.sum(x[char]*x['w_raw_normalize']))).rename(columns={0:char})  
    
        if char == chars[0]:
            port_chars = port_chars_temp
            continue
        
        port_chars = pd.merge(port_chars, port_chars_temp, how='left', left_index=True, right_index=True)

    port_chars = port_chars[port_chars.index.isin(port_chars.index[rolling_window:])]
    
    return port_chars

def get_weights_properties(df, date_name, port_name, port_column_name, benchmark_portfolio, benchmark_portfolio_name, chars):
    table = pd.DataFrame()

    for char in chars:
        table.at['theta '+char, port_column_name] = np.mean(df.groupby(date_name)['theta_'+char].mean().dropna())
    
    table.at['|w|*100', port_column_name] = np.mean(
        df.groupby(date_name)[port_name].apply(lambda x: np.mean(np.abs(x))).dropna())*100

    table.at['max w*100', port_column_name] = np.mean(df.groupby(date_name)[port_name].max().dropna())*100

    table.at['min w*100', port_column_name] = np.mean(df.groupby(date_name)[port_name].min().dropna())*100
    
    if benchmark_portfolio==True:
        table.at['|w|*100', benchmark_portfolio_name] = 1/np.mean(df.groupby(date_name)[port_name].count())*100
        
    table = round(table,3)
        
    return table

def get_portfolio(df, weight_name, date_name, stock_name, return_name, rolling_window):
# the import of this function is the output of function BPR_portfolio
# calculate portfolio return
    port_return = pd.DataFrame(df.groupby(date_name).apply(lambda x: np.sum(x[weight_name]*x[return_name])))

# drop the training period
    port_return = port_return[port_return.index.isin(port_return.index[rolling_window:])]

# rename return column
    port_return = port_return.rename(columns={0:'ret'})

# initilaize cumulative return
    port_return['ret_cum'] = 1

    dates = pd.unique(port_return.index)

# calculate cumulative return
    for i in range(1, len(port_return)):
        port_return.loc[dates[i], 'ret_cum'] = port_return.loc[dates[i-1], 'ret_cum'] * (1 + port_return.loc[dates[i], 'ret'])

# calculate 1/N portfolio returns
    df = pd.merge(df, pd.DataFrame(df.groupby(date_name)[stock_name].count()).rename(columns={stock_name:'count'}),
         how='left', left_on=date_name, right_index=True)

    market_return = pd.DataFrame(df.groupby(date_name).apply(lambda x: np.sum(1/x['count']*x['ret']))).rename(columns={0:'ret_market'})

    port_return = pd.merge(port_return, market_return, how='left', left_on=date_name, right_index=True)

    port_return['ret_market_cum'] = 1

# calculate cumulative 1/N portfolio returns
    for i in range(1, len(port_return)):
        port_return.loc[dates[i], 'ret_market_cum'] = port_return.loc[dates[i-1], 'ret_market_cum'] * (1 + port_return.loc[dates[i], 'ret_market'])
    
    return port_return

def portfolio_evaluation(port, port_name, benchmark_name, rf_name, port_column_name, benchmark_portfolio, benchmark_portfolio_name):
# the import of this function is the output of function get_portfolio
    table = pd.DataFrame()

# Average returns
    table.at['Average returns', port_column_name] = np.mean(port[port_name])

# Std. returns
    table.at['Std. returns', port_column_name] = np.std(port[port_name], ddof=1)

# Sharpe ratio
    table.at['Sharpe ratio', port_column_name] = (np.mean(port[port_name])-np.mean(port[rf_name]))/np.std(port[port_name], ddof=1)

# Alpha 
    reg = LinearRegression(fit_intercept=True).fit(port[benchmark_name].values.reshape(len(port),1), 
                                               port[port_name].values.reshape(len(port),1))
    table.at['Alpha', port_column_name] = reg.intercept_[0]

# Beta
    table.at['Beta', port_column_name] = reg.coef_[0][0]

# Std. CAPM residuals
    table.at['Std. CAPM residuals', port_column_name] = np.std(port[port_name].values.reshape(len(port),1) -
       reg.predict(port[benchmark_name].values.reshape(len(port),1)))

# Information ratio
    excess_portfolio_returns = port[port_name] - port[benchmark_name]
    table.at['Information ratio', port_column_name] = np.mean(excess_portfolio_returns) / np.std(excess_portfolio_returns, ddof=1)
    
# observations
    table.at['Observations', port_column_name] = len(port)
    
# market portfolios
    if benchmark_portfolio==True:
        table.at['Average returns', benchmark_portfolio_name] = np.mean(port[benchmark_name])
        table.at['Std. returns', benchmark_portfolio_name] = np.std(port[benchmark_name], ddof=1)
        table.at['Sharpe ratio', benchmark_portfolio_name] = (np.mean(port[benchmark_name])-np.mean(port[rf_name]))/np.std(port[benchmark_name], ddof=1)
        table.at['Observations', benchmark_portfolio_name] = len(port)
    table = round(table,3)
    
    return table

def get_weights_short(df, weight_name, date_name):
    df['zero'] = 0

# calculate weights
    weights = df.groupby(date_name).apply(lambda x: x[[weight_name,'zero']].max(axis=1) /np.sum(x[[weight_name,'zero']].max(axis=1))).reset_index(drop=False)
    weights.index = weights['level_1']
    weights = weights.rename(columns={0:'w_short'})

    df = pd.merge(df, weights[['w_short']], how='left', left_index=True, right_index=True)

# normalize weights sum to 1
    weights = df.groupby(date_name).apply(lambda x: x['w_short']/np.sum(x['w_short'])).reset_index(drop=False)
    weights.index = weights['level_1']
    weights = weights.rename(columns={'w_short':'w_short_normalize'})

    df = pd.merge(df, weights[['w_short_normalize']], how='left', left_index=True, right_index=True)

    del df['zero']
    
    return df

def get_portfolio_transaction_cost(df, port, weight_name, stock_name, date_name, me_name, port_name):
    
# normalize market equity from 0 to 1
    me_normalize = df.groupby(date_name).apply(lambda x: x[me_name]/np.max(x[me_name]) ).reset_index(drop=False)
    me_normalize.index = me_normalize['level_1']
    me_normalize = me_normalize.rename(columns={me_name:'me_normalize'})
    df = pd.merge(df, me_normalize[['me_normalize']], how='left', left_index=True, right_index=True)
    
# get weight lag
    df = get_char_lag(data=df, stock_name=stock_name, date_name=date_name, var_name=weight_name)
    df[weight_name+'_lag'] = df[weight_name+'_lag'].fillna(0)

# calculate turnover
    df[weight_name + '_change'] = np.abs(df[weight_name] - df[weight_name+'_lag']) 
    T = pd.DataFrame(df.groupby(date_name)[weight_name + '_change'].sum()).rename(columns={weight_name + '_change':'T'})

    df = pd.merge(df, T, how='left', left_on=date_name, right_index=True)

# calculate individual stock transaction cost
    df['tc'] = (0.002 - 0.001*df['me_normalize'])*df['T']*df[weight_name + '_change']

# calcualte portfolio transaction cost
    tc = pd.DataFrame(df.groupby(date_name)['tc'].sum())
    port = pd.merge(port, tc, how='left', left_index=True, right_index=True)

# calculate return with transaction cost    
    port[port_name + '_tc'] = port[port_name] - port['tc']
    
    port[port_name + '_tc' + '_cum'] = 1
    dates = sorted(pd.unique(port.index))
    
# calculate cumulative return
    for i in range(1, len(port)):
        port.loc[dates[i], port_name + '_tc' + '_cum'] = port.loc[dates[i-1], port_name + '_tc' + '_cum'] * (1 + port.loc[dates[i], port_name + '_tc'])
    
    return df, port