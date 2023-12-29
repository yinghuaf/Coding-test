import pandas as pd
import numpy as np
import statsmodels.api as sm

def get_rank_char(data, var_name, date_name):
    # from -0.5 to 0.5
    temp_char = data[data[var_name].notna()].groupby(date_name).apply(lambda x: x[var_name].rank(ascending=True)).reset_index()
    temp_char.index = temp_char['level_1']
    
    # make sure the existence of denominator
    temp = temp_char.groupby(date_name).count()
    temp = temp[temp[var_name]>1].index
    temp_char = temp_char[temp_char[date_name].isin(temp)]

    temp_char = temp_char.groupby(date_name).apply(lambda x: (-1 + 2/(len(x)-1)*(x[var_name]-1))/2 ).reset_index()
    temp_char.index = temp_char['level_1']
    temp_char = temp_char.rename(columns={var_name:'rank_'+var_name})    

    data = pd.merge(data, temp_char[['rank_'+var_name]], how='left', left_index=True, right_index=True)

    #fill rank nan by 0
    data['rank_'+var_name] = data['rank_'+var_name].fillna(0)
    
    return data

def get_char_lag(data, stock_name, date_name, var_name):

    data = data.sort_values(by=[stock_name,date_name]).reset_index(drop=True)
    
    temp_char = data.groupby([stock_name]).apply(lambda x: x[var_name].shift(1)).reset_index().rename(columns={var_name:var_name+'_lag'})
    #temp_char.index = temp_char['level_1']

    data = pd.merge(data, temp_char[[var_name+'_lag']], how='left', left_index=True, right_index=True)
         
    #fill rank nan by 0
    # data[var_name + '_lag'] = data[var_name+'_lag'].fillna(0)
    
    return data


def get_winsorize(data, var_name, date_name, left_ratio, right_ratio):
    dates = pd.unique(data[date_name])

    for date in dates:
        data.loc[(data[var_name][data[date_name] == date] <= data[var_name][data[date_name] == date].quantile(
            left_ratio)) & (data[date_name] == date), var_name] = data[var_name][data[date_name] == date].quantile(
            left_ratio)
        data.loc[(data[var_name][data[date_name] == date] >= data[var_name][data[date_name] == date].quantile(
            right_ratio)) & (data[date_name] == date), var_name] = data[var_name][data[date_name] == date].quantile(
            right_ratio)

    return data


def get_winsorize_all(data, var_name, left_ratio, right_ratio):
    data.loc[data[var_name] <= data[var_name].quantile(left_ratio), var_name] = data[var_name].quantile(left_ratio)
    data.loc[data[var_name] >= data[var_name].quantile(right_ratio), var_name] = data[var_name].quantile(right_ratio)

    return data


def get_OLS_beta(x):
    x1 = x['ret']
    x2 = x['mktrf']
    results = sm.OLS(x1, sm.add_constant(x2), missing='drop').fit()
    return results.params[1], results.resid.var()


def get_beta(data, stock_name, date_name, market_return, stock_return, rolling_length):
    # get excess market return
    data['mktrf'] = data[market_return] - data['rf']

    # ger excess stock return
    data['ret'] = data[stock_return] - data['rf']

    stock_list = pd.unique(data[stock_name])
    beta_rolling = pd.DataFrame()

    for stock_temp in stock_list:
        # print(stock_temp)
        beta_rolling_temp = pd.DataFrame()

        temp = data[[date_name, 'ret', 'mktrf']][data[stock_name] == stock_temp]
        temp = temp.reset_index(drop=True)
        temp_dates = sorted(temp[date_name])

        for i in range(len(temp_dates)):
            if (i + rolling_length + 1) > len(temp_dates):
                continue
            else:
                temp_date1 = temp_dates[i]
                temp_date2 = temp_dates[i + rolling_length]
                beta_rolling_temp.at[temp_date2, 'beta'], beta_rolling_temp.at[temp_date2, 'rvar_capm'] = get_OLS_beta(
                    temp[(temp[date_name] > temp_date1) & (temp[date_name] <= temp_date2)])

        beta_rolling_temp[stock_name] = stock_temp

        beta_rolling = beta_rolling.append(beta_rolling_temp)

    beta_rolling = beta_rolling.reset_index(drop=False)
    beta_rolling = beta_rolling.rename(columns={'index': date_name})

    data = pd.merge(data, beta_rolling, how='left', on=[date_name, stock_name])

    return data


def get_mom12m(data, date_name, stock_name, ret_name):
    data = data.sort_values(by=[stock_name, date_name])

    data['mom12m'] = 1

    for i in range(1, 12):
        data['mom12m'] = data['mom12m'] * (1 + data.groupby(stock_name)[ret_name].shift(i))

    data['mom12m'] = data['mom12m'] - 1
    return data