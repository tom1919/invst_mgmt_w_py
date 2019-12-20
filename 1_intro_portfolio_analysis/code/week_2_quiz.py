# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#%% import libs
import pandas as pd
import os 
import numpy as np
os.getcwd()
#os.chdir('C:/Users/tommy/Downloads/invst_mgmt_w_py/1_intro_portfolio_analysis')
#%% functions


def skewness(r):
    """
    Alternative to scipy.stats.skew()
    Computes the skewness of the supplied Series or DataFrame
    Returns a float or a Series
    """
    demeaned_r = r - r.mean()
    # use the population standard deviation, so set dof=0
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r**3).mean()
    return exp/sigma_r**3


def kurtosis(r):
    """
    Alternative to scipy.stats.kurtosis()
    Computes the kurtosis of the supplied Series or DataFrame
    Returns a float or a Series
    """
    demeaned_r = r - r.mean()
    # use the population standard deviation, so set dof=0
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r**4).mean()
    return exp/sigma_r**4

from scipy.stats import norm
def var_gaussian(r, level=5, modified=False):
    """
    Returns the Parametric Gauusian VaR of a Series or DataFrame
    If "modified" is True, then the modified VaR is returned,
    using the Cornish-Fisher modification
    """
    # compute the Z score assuming it was Gaussian
    z = norm.ppf(level/100)
    if modified:
        # modify the Z score based on observed skewness and kurtosis
        s = skewness(r)
        k = kurtosis(r)
        z = (z +
                (z**2 - 1)*s/6 +
                (z**3 -3*z)*(k-3)/24 -
                (2*z**3 - 5*z)*(s**2)/36
            )
    return -(r.mean() + z*r.std(ddof=0))

def var_historic(r, level=5):
    """
    Returns the historic Value at Risk at a specified level
    i.e. returns the number such that "level" percent of the returns
    fall below that number, and the (100-level) percent are above
    """
    if isinstance(r, pd.DataFrame):
        return r.aggregate(var_historic, level=level)
    elif isinstance(r, pd.Series):
        return -np.percentile(r, level)
    else:
        raise TypeError("Expected r to be a Series or DataFrame")
        
def portfolio_return(weights, returns):
    """
    Computes the return on a portfolio from constituent returns and weights
    weights are a numpy array or Nx1 matrix and returns are a numpy array or Nx1 matrix
    """
    return weights.T @ returns

def portfolio_vol(weights, covmat):
    """
    Computes the vol of a portfolio from a covariance matrix and constituent weights
    weights are a numpy array or N x 1 maxtrix and covmat is an N x N matrix
    """
    return (weights.T @ covmat @ weights)**0.5


from scipy.optimize import minimize

def msr(riskfree_rate, er, cov):
    """
    Returns the weights of the portfolio that gives you the maximum sharpe ratio
    given the riskfree rate and expected returns and a covariance matrix
    """
    n = er.shape[0]
    init_guess = np.repeat(1/n, n)
    bounds = ((0.0, 1.0),) * n # an N-tuple of 2-tuples!
    # construct the constraints
    weights_sum_to_1 = {'type': 'eq',
                        'fun': lambda weights: np.sum(weights) - 1
    }
    def neg_sharpe(weights, riskfree_rate, er, cov):
        """
        Returns the negative of the sharpe ratio
        of the given portfolio
        """
        r = portfolio_return(weights, er)
        vol = portfolio_vol(weights, cov)
        return -(r - riskfree_rate)/vol
    
    weights = minimize(neg_sharpe, init_guess,
                       args=(riskfree_rate, er, cov), method='SLSQP',
                       options={'disp': False},
                       constraints=(weights_sum_to_1,),
                       bounds=bounds)
    return weights.x

def annualize_rets(r, periods_per_year):
    """
    Annualizes a set of returns
    We should infer the periods per year
    but that is currently left as an exercise
    to the reader :-)
    """
    compounded_growth = (1+r).prod()
    n_periods = r.shape[0]
    return compounded_growth**(periods_per_year/n_periods)-1

def annualize_vol(r, periods_per_year):
    """
    Annualizes the vol of a set of returns
    We should infer the periods per year
    but that is currently left as an exercise
    to the reader :-)
    """
    return r.std()*(periods_per_year**0.5)

def gmv(cov):
    """
    Returns the weights of the Global Minimum Volatility portfolio
    given a covariance matrix
    """
    n = cov.shape[0]
    return msr(0, np.repeat(1, n), cov)

#%% read in data

df = pd.read_csv('./data/original/edhec-hedgefundindices.csv', index_col = 0)

df.index = pd.to_datetime(df.index, format="%d/%m/%Y").to_period('M')

ind = pd.read_csv("./data/original/ind30_m_vw_rets.csv", 
                    header=0, index_col=0)/ 100
ind.index = pd.to_datetime(ind.index, format="%Y%m").to_period('M')
ind.columns = ind.columns.str.strip()

ind_cols = ["Books", "Steel", "Oil",  "Mines"]
rets = annualize_rets(ind.loc['2013':'2017', ind_cols], 12)
cov = ind.loc["2013":"2017", ind_cols].cov()
#%% quiz answers

# 1. 
var_gaussian(df.loc['2000-01':, 'Distressed Securities'], level = 1)
# 3.1

# 2.
var_gaussian(df.loc['2000-01':, 'Distressed Securities'], level = 1, 
             modified = True)
# 4.96

# 3.
var_historic(df.loc['2000-01':, 'Distressed Securities'], level = 1)
# 4.3

# 4.
# 25 

# 5. 
msr_w = msr(.1, rets, cov)
df_w = pd.DataFrame({'ind':rets.index, 'msr_w': msr_w})
df_w.sort_values(by = 'msr_w').tail()
# 100

# 6.
df_w.sort_values(by = 'msr_w').tail()
# Steel

# 7. 
df_w.sort_values('msr_w', ascending=False)
# 1

# 8. 
df_w['gmv'] = gmv(cov)
df_w.sort_values('gmv')
# 47.7

# 9.
df_w.sort_values('gmv')
# Books

# 10.
df_w.sort_values('gmv')
# 3

# 11.
cov18 = ind.loc['2018':, ind_cols].cov()

port_var = msr_w.T @ cov18 @ msr_w 

port_var = np.dot(np.dot(msr_w.T, cov18), msr_w)

port_sd = port_var ** .5

# annualized portfolio vol
port_sd * (12**0.5)
# 21.98

# 12.
port_var2 = df_w.gmv.values.T @ cov18 @ df_w.gmv.values


port_sd2 = port_var2 ** .5
port_sd2 * (12**0.5) 
# 18.9

#%% .























































