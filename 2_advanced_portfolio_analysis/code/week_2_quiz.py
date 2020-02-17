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
         
         
#%% Q 1-4

ind_rets =  erk.get_ind_file('returns', 'vw', n_inds=30)
ind_rets = ind_rets.loc['1997':'2018', :]    

mkt_caps = erk.get_ind_market_caps(n_inds=30, weights=False)
mkt_caps = mkt_caps.loc['1997':'2018', :]  

ewr = erk.backtest_ws(ind_rets, estimation_window=36)
cwr = erk.backtest_ws(ind_rets, weighting=erk.weight_cw, cap_weights=mkt_caps
                      , estimation_window=36)
btr = pd.DataFrame({"EW": ewr, "CW": cwr})
(1+btr).cumprod().plot(figsize=(12,5), title="30 Industries - CapWeighted vs Equally Weighted")
erk.summary_stats(btr.dropna())

# =============================================================================
# 1. 6.45
# 2. 15.13
# 3. 7.76
# 4. 15.87
# =============================================================================


#%%

ewtr = erk.backtest_ws(ind_rets, estimation_window=36, cap_weights=mkt_caps
                      , max_cw_mult=2, microcap_threshold=.01)

btr = pd.DataFrame({"EW": ewtr, "CW": cwr})
(1+btr).cumprod().plot(figsize=(12,5), title="30 Industries - CapWeighted vs Equally Weighted")
erk.summary_stats(btr.dropna())

erk.tracking_error(ewr, cwr),erk.tracking_error(ewtr, cwr)

# =============================================================================
# 5. 7.76
# 6. 15.87 ??
# 7. 18.74
# 8. 18.74 ??
# =============================================================================

#%%

mv_s_r = erk.backtest_ws(ind_rets, estimation_window=36, weighting=erk.weight_gmv
                         , cov_estimator=erk.sample_cov)

mv_sh_r = erk.backtest_ws(ind_rets, estimation_window=36, 
                          weighting=erk.weight_gmv, 
                          cov_estimator=erk.shrinkage_cov, delta=0.25)
btr = pd.DataFrame({"EW": ewr, "CW": cwr, "GMV-Sample": mv_s_r,
                    'GMV-Shrink 0.25': mv_sh_r})


(1+btr).cumprod().plot(figsize=(12,6), title="Industry Portfolios")
erk.summary_stats(btr.dropna())


# =============================================================================
# 9. 6.63
# 10. 11.74
# 11. 6.85
# 12. 11.46
# =============================================================================







