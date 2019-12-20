# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 15:49:04 2019

@author: tommy
"""


#%%
import os
os.chdir('C:\\Users\\tommy\\Dropbox\\7. projects\\invst_mgmt_w_py\\1_intro_portfolio_analysis/code')
import edhec_risk_kit as er
import pandas as pd

#%%



rets = er.get_ffme_returns2()

ann_rets = er.annualize_rets(rets.loc[:,'SmallCap'],12)
ann_vol = er.annualize_vol(rets.loc[:,'SmallCap'],12)
ann_rets2 = er.annualize_rets(rets.loc[:,'LargeCap'],12)
ann_vol2 = er.annualize_vol(rets.loc[:,'LargeCap'],12)
ann_ret3 = er.annualize_rets(rets.loc['1999-01':'2015-12','SmallCap'],12)
ann_vol3 = er.annualize_vol(rets.loc['1999-01':'2015-12','SmallCap'],12)
ann_ret4 = er.annualize_rets(rets.loc['1999-01':'2015-12','LargeCap'],12)
ann_vol4 = er.annualize_vol(rets.loc['1999-01':'2015-12','LargeCap'],12)


dd = er.drawdown(rets.loc['1999-01':'2015-12':,'SmallCap'])
min(dd.Drawdown)
dd[dd.Drawdown == min(dd.Drawdown)]

dd2 = er.drawdown(rets.loc['1999-01':'2015-12':,'LargeCap'])
min(dd2.Drawdown)
dd2[dd2.Drawdown == min(dd2.Drawdown)]

#%%


hfi = er.get_hfi_returns()
sd = er.semideviation(hfi.loc['2009-01':,:])
sd.sort_values()


er.skewness(hfi.loc['2009-01':,:]).sort_values()
er.kurtosis(hfi.loc['2000-01':,:]).sort_values()

#%%

# =============================================================================
# 1. 15.2
# 2. 33.7
# 3. 9.8
# 4. 19.5
# 5. 11.4
# 6. 22.9
# 7. 6.3
# 8. 17.2
# 9. 62.5
# 10. 2009-02
# 11. 55.3
# 12. 2009-02
# 13. Short Selling
# 14. Fixed Income Arbitrage
# 15. Equity Market Neutral
# 16. Fixed Income Arbitrage
# =============================================================================
