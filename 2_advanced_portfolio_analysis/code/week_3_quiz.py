# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 18:47:31 2020

@author: tommy
"""

#%% load libs

import pandas as pd
import numpy as np
import helper_functions as hf
from numpy.linalg import inv
import os
os.chdir(hf.get_path() + 'invst_mgmt_w_py/2_advanced_portfolio_analysis/code/' )
import  edhec_risk_kit_206 as erk

os.chdir(hf.get_path() + 'invst_mgmt_w_py/2_advanced_portfolio_analysis/' )
         
#%%
def implied_returns(delta, sigma, w):
    """
    Obtain the implied expected returns by reverse engineering the weights
    Inputs:
    delta: Risk Aversion Coefficient (scalar)
    sigma: Variance-Covariance Matrix (N x N) as DataFrame
        w: Portfolio weights (N x 1) as Series
    Returns an N x 1 vector of Returns as Series
    """
    ir = delta * sigma.dot(w).squeeze() # to get a series from a 1-column dataframe
    ir.name = 'Implied Returns'
    return ir

def proportional_prior(sigma, tau, p):
    """
    Returns the He-Litterman simplified Omega
    Inputs:
    sigma: N x N Covariance Matrix as DataFrame
    tau: a scalar
    p: a K x N DataFrame linking Q and Assets
    returns a P x P DataFrame, a Matrix representing Prior Uncertainties
    """
    helit_omega = p.dot(tau * sigma).dot(p.T)
    # Make a diag matrix from the diag elements of Omega
    return pd.DataFrame(np.diag(np.diag(helit_omega.values)),index=p.index, columns=p.index)

def bl(w_prior, sigma_prior, p, q,
                omega=None,
                delta=2.5, tau=.02):
    """
    # Computes the posterior expected returns based on 
    # the original black litterman reference model
    #
    # W.prior must be an N x 1 vector of weights, a Series
    # Sigma.prior is an N x N covariance matrix, a DataFrame
    # P must be a K x N matrix linking Q and the Assets, a DataFrame
    # Q must be an K x 1 vector of views, a Series
    # Omega must be a K x K matrix a DataFrame, or None
    # if Omega is None, we assume it is
    #    proportional to variance of the prior
    # delta and tau are scalars
    """
    if omega is None:
        omega = proportional_prior(sigma_prior, tau, p)
    # Force w.prior and Q to be column vectors
    # How many assets do we have?
    N = w_prior.shape[0]
    # And how many views?
    K = q.shape[0]
    # First, reverse-engineer the weights to get pi
    pi = implied_returns(delta, sigma_prior,  w_prior)
    # Adjust (scale) Sigma by the uncertainty scaling factor
    sigma_prior_scaled = tau * sigma_prior  
    # posterior estimate of the mean, use the "Master Formula"
    # we use the versions that do not require
    # Omega to be inverted (see previous section)
    # this is easier to read if we use '@' for matrixmult instead of .dot()
    #     mu_bl = pi + sigma_prior_scaled @ p.T @ inv(p @ sigma_prior_scaled @ p.T + omega) @ (q - p @ pi)
    mu_bl = pi + sigma_prior_scaled.dot(p.T).dot(inv(p.dot(sigma_prior_scaled).dot(p.T) + omega).dot(q - p.dot(pi).values))
    # posterior estimate of uncertainty of mu.bl
#     sigma_bl = sigma_prior + sigma_prior_scaled - sigma_prior_scaled @ p.T @ inv(p @ sigma_prior_scaled @ p.T + omega) @ p @ sigma_prior_scaled
    sigma_bl = sigma_prior + sigma_prior_scaled - sigma_prior_scaled.dot(p.T).dot(inv(p.dot(sigma_prior_scaled).dot(p.T) + omega)).dot(p).dot(sigma_prior_scaled)
    return (mu_bl, sigma_bl)


def inverse(d):
    """
    Invert the dataframe by inverting the underlying matrix
    """
    return pd.DataFrame(inv(d.values), index=d.columns, columns=d.index)

def w_msr(sigma, mu, scale=True):
    """
    Optimal (Tangent/Max Sharpe Ratio) Portfolio weights
    by using the Markowitz Optimization Procedure
    Mu is the vector of Excess expected Returns
    Sigma must be an N x N matrix as a DataFrame and Mu a column vector as a Series
    This implements page 188 Equation 5.2.28 of
    "The econometrics of financial markets" Campbell, Lo and Mackinlay.
    """
    w = inverse(sigma).dot(mu)
    if scale:
        w = w/sum(w) # fix: this assumes all w is +ve
    return w

def w_star(delta, sigma, mu):
    return (inverse(sigma).dot(mu))/delta
#%% Q 1-3

inds = ['Hlth', 'Fin', 'Whlsl', 'Rtail', 'Food']

ind_rets =  erk.get_ind_file('returns', 'vw', n_inds=49).loc['2013':'2018', inds]        
 
cap_wts = erk.get_ind_market_caps(n_inds=49, weights=True).loc['2013':'2018', inds]
cap_wts = cap_wts / cap_wts.sum()

cor_mtx = ind_rets.corr()
vol = ind_rets.std() * np.sqrt(12)
sigma_prior = vol.dot(vol) * cor_mtx

w_eq = implied_returns(2.5, sigma_prior,  cap_wts)


'''
1.  Rtail
2. Rtail
3. Hlth
'''

#%% Q 4-5

q = pd.Series([.03])
p = pd.DataFrame([0] * len(inds), index = inds).T
w_r = cap_wts.loc['Rtail'] / (cap_wts.loc['Rtail'] + cap_wts.loc['Whlsl'])
w_w = cap_wts.loc['Whlsl'] / (cap_wts.loc['Rtail'] + cap_wts.loc['Whlsl'])

p.iloc[0]['Hlth'] = 1
p['Rtail'] = -w_r
p['Whlsl'] = -w_w

'''
4. -0.1513
5. -0.84869
'''


#%% Q 6-9

delta = 2.5
tau = 0.05 # from Footnote 8

# Find the Black Litterman Expected Returns
bl_mu, bl_sigma = bl(w_eq, sigma_prior, p, q, tau = tau)
(bl_mu*100).round(1)

wstar = w_star(delta=2.5, sigma=bl_sigma, mu=bl_mu)
# display w*
(wstar*100).round(1)

wts = w_msr(delta * bl_sigma, bl_mu)

'''
6. Food
7. Rtail (my code wrong)
8. Whlsl  (my code wrong)
9. q and not p

'''

#%%

q = pd.Series([.05])

bl_mu, bl_sigma = bl(w_eq, sigma_prior, p, q, tau = tau)
(bl_mu*100).round(1)

wstar = w_star(delta=2.5, sigma=bl_sigma, mu=bl_mu)
(wstar*100).round(1)

'''
10. Rtail (my code wrong)
11. hlth
'''












