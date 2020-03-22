# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 18:47:31 2020

@author: tommy
"""

#%% load libs

import pandas as pd
import helper_functions as hf
import os
os.chdir(hf.get_path() + 'invst_mgmt_w_py/2_advanced_portfolio_analysis/code/' )
import  edhec_risk_kit_206 as erk

os.chdir(hf.get_path() + 'invst_mgmt_w_py/2_advanced_portfolio_analysis/' )
         
#%% read in data

ind_rets =  erk.get_ind_file('returns', 'vw', n_inds=49).loc['2014':'2018',:]        
cap_wts = erk.get_ind_market_caps(n_inds=49, weights=True).loc['2014':'2018',:]

#%% Q 1-4

cw_rc = erk.risk_contribution(cap_wts.iloc[0,:], ind_rets.cov())
cw_rc = cw_rc.sort_values()

ew_rc = erk.risk_contribution(erk.weight_ew(ind_rets), ind_rets.cov())
ew_rc = ew_rc.sort_values()

# =============================================================================
# 1. Banks
# 2. 10.40
# 3. Steel
# 4. 3.09
# =============================================================================

#%% Q 5-8

erc_wt = erk.equal_risk_contributions(ind_rets.cov())
pd.DataFrame({'ind': ind_rets.columns, 'wt': erc_wt}).sort_values('wt')

# =============================================================================
# 5. Util
# 6. 5.21
# 7. Steel
# 8. 1.28 
# =============================================================================

#%% Q 9-10

cw_rc[len(cw_rc)-1] - cw_rc[0]
ew_rc[len(ew_rc)-1] - ew_rc[0]

# =============================================================================
# 9. 10.396
# 10. 2.502
# =============================================================================

