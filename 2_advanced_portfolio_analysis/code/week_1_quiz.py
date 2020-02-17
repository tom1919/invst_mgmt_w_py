# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 04:12:31 2020

@author: tommy
"""

#%% load libs 
import pandas as pd
import helper_functions as hf
import statsmodels.api as sm

p = hf.get_path() + 'invst_mgmt_w_py/2_advanced_portfolio_analysis/'

#%% load data

ind49 = pd.read_csv(p + 'data/ind49_m_vw_rets.csv', header=0, index_col = 0,
                    na_values=-99.99) / 100
ind49.index = pd.to_datetime(ind49.index, format = '%Y%m').to_period('M')
ind49 = ind49.loc['1991':,:]
ind49.columns = ind49.columns.str.strip()

rets = pd.read_csv(p + "data/F-F_Research_Data_Factors_m.csv",
                   header=0, index_col=0, na_values=-99.99)/100
rets.index = pd.to_datetime(rets.index, format="%Y%m").to_period('M')
rets = rets.loc['1991':, :]

#%% Q 1-2

'''
CAPM model:
    ret of port - rf = alpha + beta(ret of mkt - rf) + e
        where rf = risk free rate
'''

rets['Constant'] = 1
x_var = rets.loc[:, ['Mkt-RF', 'Constant']]

# Q1: beta = 0.5295
beer = ind49.loc[:, 'Beer'] - rets.RF
lm = sm.OLS(beer, x_var).fit()
lm.summary()

# Q2: beta = 1.5546
steel = ind49.Steel - rets.RF
sm.OLS(steel, x_var).fit().summary()

#%% Q 3-4

x_var = rets.loc['2013':'2018', ['Mkt-RF', 'Constant']]

# Q1 beta = .5860
beer = ind49.loc['2013':'2018', 'Beer'] - rets.loc['2013':'2018', 'RF']
sm.OLS(beer, x_var).fit().summary()

# Q2 beta = 1.4169
steel = ind49.loc['2013':'2018', 'Steel'] - rets.loc['2013':'2018', 'RF']
sm.OLS(steel, x_var).fit().summary()

#%% Q 5-6

y_vars = ind49.loc['1991': '1993':]
x_var = rets.loc['1991': '1993', ['Mkt-RF', 'Constant']]


y_vars2 = y_vars - rets.loc['1991':'1993', 'RF'].values.reshape((36,1))

betas = []
ind_names = []
for col in y_vars2.columns:
    fit = sm.OLS(y_vars2[col], x_var).fit()
    
    betas.append(fit.params[0])
    ind_names.append(col)

ind_betas = pd.DataFrame({'ind': ind_names, 'beta': betas})

# Q5 hightest beta = Hlth
# Q6 lowest beta = Gold

#%% Q 7-10

'''
SMB: small minus big. long small cap stocks and short large cap stocks
HML: high minus low. long high value stocks and short growth stocks
'''

x_var = rets.loc['1991': '2018', :].drop('RF', axis=1)
y_vars = ind49.loc['1991': '2018':]
y_vars2 = y_vars - rets.loc['1991':'2018', 'RF'].values.reshape((336,1))

betas = []
ind_names = []
for col in y_vars2.columns:
    fit = sm.OLS(y_vars2[col], x_var).fit()
    
    betas.append(fit.params)
    ind_names.append(col)

beta_df = pd.concat(betas, axis=1)
beta_df.columns = ind_names
beta_df = beta_df.transpose()

# Q7 highest small cap tilt: FabPr
# Q8 highest larg cap tilt: Beer
# Q9 highest value tilt Txtls
# Q10 highest growth tilt Softw












