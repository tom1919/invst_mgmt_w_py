#This library calculates Financial Metrics

import numpy as np #for numerical array data
import pandas as pd #for tabular data
import math
import statsmodels.api as sm
from scipy import stats
import matplotlib.pyplot as plt

def calc_annualized_mean_return_metrics(data, listOfFactors, freq='monthly'):
    if(freq=='monthly'):
        out = pd.DataFrame((data[listOfFactors].mean() +1)**12-1, columns=['Annual Mean Return'])
        out['Annual SD'] = (12**.5)*data[listOfFactors].std()
        out['t stat'] = data[listOfFactors].mean()/(data[listOfFactors].std()/(data.shape[0]**.5))
        return out
    elif(freq=='daily'):
        out = pd.DataFrame((data[listOfFactors].mean() +1)**252-1, columns=['Annual Mean Return'])
        out['Annual SD'] = (252**.5)*data[listOfFactors].std()
        out['t stat'] = data[listOfFactors].mean()/(data[listOfFactors].std()/(data.shape[0]**.5))
        return out
    else:
        print('Incorrect Freq Specification')
        return 0

def max_drawdown(data, dictOfPeriods, listOfFactors, interestRate, dateCol, regimeCol, timeStep='monthly'):
    '''This will be a max drawdown code'''
    #Step 1: Define the output matrix
    originalCols = list(dictOfPeriods.keys())
    newCols = []
    for col in originalCols:
        newCols.append(col)
        newCols.append(col + ' TTR')
    out = pd.DataFrame(np.zeros((len(listOfFactors),2*len(list(dictOfPeriods.keys())))),
                        columns=newCols, index=listOfFactors)

    #Step 2: Calculate current drawdown at each time step for each factor
    dataCumProd = data.copy()
    dataCumProd[listOfFactors] = dataCumProd[listOfFactors]+1
    dataCumProd[listOfFactors] = dataCumProd[listOfFactors].cumprod()

    dataCumMax = dataCumProd.copy()
    dataCumMax[listOfFactors] = dataCumMax[listOfFactors].cummax()

    dataCurrDrawdown = data.copy()

    dataCurrDrawdown[listOfFactors] = dataCumProd[listOfFactors]/dataCumMax[listOfFactors]


    #Calculate Current Drawdown
    for periodName in dictOfPeriods.keys():
        dateSet = dictOfPeriods[periodName]
        data2 = dataCurrDrawdown.copy()
        data2 = data2[(data2[dateCol] >= dateSet['startDate']) & (data2[dateCol] <= dateSet['endDate'])]
        if(dateSet['Regime'] is not None):
            data2 = data2[data2[regimeCol] == dateSet['Regime']]

        for factor in listOfFactors:
            out.loc[factor,periodName] = data2[factor].min()


    for factor in listOfFactors:
        for periodName in dictOfPeriods.keys():
            val = out.loc[factor, periodName]
            indOfMaxDrawdown = dataCurrDrawdown[dataCurrDrawdown[factor] == val].index[0]
            data2 = dataCurrDrawdown.loc[indOfMaxDrawdown:,:].copy()
            try:
                indOfRecoveryDrawdown = data2[data2[factor] >= 1].index[0]
            except:
                indOfRecoveryDrawdown = np.Inf
            #Calculate Time to Recovery
            out.loc[factor, periodName+' TTR'] = indOfRecoveryDrawdown - indOfMaxDrawdown

    #So far, the drawdown is expressed as proportion of the maximum, we want to express it as proportion below the maximum
    for periodName in dictOfPeriods.keys():
        out[periodName] = 1 - out[periodName]

    return out

def calc_metrics(data, dictOfPeriods, listOfFactors, interestRate, dateCol, regimeCol, method='sharpe', timeStep='monthly'):
    '''calc_sharpe_ratio_by_regime returns the sharpe ratio, broken out by time period and regime for set of series
    Sharpe Ratio and Log Return are annualized.  Mean return is not
    '''
    #Step 1: check if the time period is correctly specified
    if(timeStep not in ['monthly', 'daily', 'yearly']):
        print('Incorrect argument for timeStep')
        return 0
    #Step 2: Run max_drawdown code if specified
    if(method=='max_drawdown'):
        return max_drawdown(data, dictOfPeriods, listOfFactors, interestRate, dateCol, regimeCol, timeStep=timeStep)

    #Step 3: Otherwise, run normal analysis
    out = pd.DataFrame(np.zeros((len(listOfFactors),len(list(dictOfPeriods.keys())))),
                        columns=list(dictOfPeriods.keys()), index=listOfFactors)    
    for periodName in dictOfPeriods.keys():
        '''Pull Out the Set of Dates'''
        dateSet = dictOfPeriods[periodName]
        '''Filter The Data'''
        data2 = data.copy()
        data2 = data2[(data2[dateCol] >= dateSet['startDate']) & (data2[dateCol] <= dateSet['endDate'])]
        if(dateSet['Regime'] is not None):
            data2 = data2[data2[regimeCol] == dateSet['Regime']]
        '''Calculate Sharpe Ratio for set of factors'''
        for factor in listOfFactors:
            if(method=='mean'):
                '''Calculate the mean'''
                m = data2[factor].mean()
                out.loc[factor, periodName] = m
            if(method=='sharpe'):
                '''Calculate the sharpe ratio'''
                sharpe = np.mean(data2[factor] - data2[interestRate])/data2[factor].std()
                '''Convert to Annual if desired'''
                if(timeStep=='monthly'):
                    out.loc[factor, periodName] = np.sqrt(12)*sharpe
                elif(timeStep=='daily'):
                    out.loc[factor, periodName] = np.sqrt(252)*sharpe
                elif(timeStep=='yearly'):
                    out.loc[factor, periodName] = sharpe

            if(method == 'logReturn'):
                '''Calculate the annualized log return'''
                newData = data2[factor].copy()
                newData = newData+1
                m = np.log(newData).mean()
                if(timeStep=='monthly'):
                    out.loc[factor, periodName] = 12*m
                elif(timeStep=='daily'):
                    out.loc[factor, periodName] = 252*m
                elif(timeStep=='yearly'):
                    out.loc[factor, periodName] = m
            if(method=='skew'):
                '''Calculate the skew'''
                skew = data2[factor].skew()
                out.loc[factor, periodName] = skew
            if(method=='kurtosis'):
                '''Calculate the kurtosis'''
                kurt = data2[factor].kurtosis()
                out.loc[factor, periodName] = kurt
    return out

def calc_contagion_measure(data, listOfFactors, listOfRegimeCols):
    out = np.zeros((1,1+2*len(listOfRegimeCols)))
    data2 = data.copy()
    out[0,0] = np.linalg.norm(data2[listOfFactors].corr())/len(listOfFactors)
    count = 1
    l = ['Unconditional']
    for col in listOfRegimeCols:
        conditionalData = data2[data2[col] == 1].copy()
        out[0,count] = np.linalg.norm(conditionalData[listOfFactors].corr())/len(listOfFactors)
        l = l + [col+' = 1']
        count = count + 1
        conditionalData = data2[data2[col] == -1].copy()
        out[0,count] = np.linalg.norm(conditionalData[listOfFactors].corr())/len(listOfFactors)
        count = count + 1
        l = l + [col+' = -1']
    
    out = pd.DataFrame(out, columns=l)
    return out

def calc_metrics_with_nan(data, dictOfPeriods, listOfFactors, interestRate, dateCol, regimeCol, method='sharpe', timeStep='monthly'):
    '''calc_sharpe_ratio_by_regime returns the sharpe ratio, broken out by time period and regime for set of series
    Sharpe Ratio and Log Return are annualized.  Mean return is not
    '''
    if(timeStep not in ['monthly', 'daily', 'yearly']):
        print('Incorrect argument for timeStep')
        return 0

    out = pd.DataFrame(np.zeros((len(listOfFactors),len(list(dictOfPeriods.keys())))),
                        columns=list(dictOfPeriods.keys()), index=listOfFactors)    
    for periodName in dictOfPeriods.keys():
        '''Pull Out the Set of Dates'''
        dateSet = dictOfPeriods[periodName]
        '''Filter The Data'''
        data2 = data.copy()
        data2 = data2[(data2[dateCol] >= dateSet['startDate']) & (data2[dateCol] <= dateSet['endDate'])]
        if(dateSet['Regime'] is not None):
            data2 = data2[data2[regimeCol] == dateSet['Regime']]
        '''Calculate Metric for set of factors'''
        for factor in listOfFactors:
            data3 = data2[[factor, interestRate]].copy()
            data3.dropna(inplace=True)
            if(method=='mean'):
                '''Calculate the mean'''
                m = data3[factor].mean()
                out.loc[factor, periodName] = m
            if(method=='sharpe'):
                '''Calculate the sharpe ratio'''
                sharpe = np.mean(data3[factor] - data3[interestRate])/data3[factor].std()
                '''Convert to Annual if desired'''
                if(timeStep=='monthly'):
                    out.loc[factor, periodName] = np.sqrt(12)*sharpe
                elif(timeStep=='daily'):
                    out.loc[factor, periodName] = np.sqrt(252)*sharpe
                elif(timeStep=='yearly'):
                    out.loc[factor, periodName] = sharpe
            if(method == 'logReturn'):
                '''Calculate the annualized log return'''
                newData = data3[factor].copy()
                newData = newData+1
                m = np.log(newData).mean()
                if(timeStep=='monthly'):
                    out.loc[factor, periodName] = 12*m
                elif(timeStep=='daily'):
                    out.loc[factor, periodName] = 252*m
                elif(timeStep=='yearly'):
                    out.loc[factor, periodName] = m
            if(method=='skew'):
                '''Calculate the skew'''
                skew = data3[factor].skew()
                out.loc[factor, periodName] = skew
            if(method=='kurtosis'):
                '''Calculate the kurtosis'''
                kurt = data3[factor].kurtosis()
                out.loc[factor, periodName] = kurt
    return out

def compute_individual_factor_ts_momentum(data, listOfFactors, lookbackWindow=[12, 1], dateCol='Date'):
    '''compute_factor_ts_momentum takes a data set, and computes the leg of each factor for TS momentum
    INPUTS:
        data: pandas df, columns should include listOfFactors
        listOfFactors: list, set of factors to include
        lookbackWindow: set of months to use.  Time period to consider it t-lookbackWindow[0], t-lokbackWindow[1]
    OUTPUTS:
        out: pandas df, columns should be the leg (i.e 1 or -1) for that factor in the TS momentum portfolio
    '''
    vals = np.zeros((data.shape[0] - lookbackWindow[0], len(listOfFactors)))
    #Fill in Date Column
    for i in range(lookbackWindow[0], data.shape[0]):
        #Compute cumulative return over lookback window
        new = data.loc[i-lookbackWindow[0]:i-lookbackWindow[1], listOfFactors].copy()
        ret = new + 1
        ret = ret.product()
        ret = ret - 1
        #Compute Time Series Momentum
        #Get list of Factors with Positive Return over lookback period
        pos = list(ret[ret > 0].index)
        #Get list of Factors with Negative Return over lookback period
        neg = list(ret[ret < 0].index)
        #Now, Mark 1 if in positive leg, otherwise negative
        for j in range(len(listOfFactors)):
            if listOfFactors[j] in pos:
                vals[i-lookbackWindow[0],j] = 1
            else:
                vals[i-lookbackWindow[0],j] = -1
        
    #Now, create the output dataframe
    cols = list()
    for col in listOfFactors:
        cols.append('TSSign'+col)
    out = pd.DataFrame(vals, columns=cols)
    out['Date'] = data.loc[lookbackWindow[0]:data.shape[0],dateCol].values
    cols = ['Date'] + cols
    out = out[cols].copy()
    return out

def calc_conditional_metrics(data, listOfFactors, splitCol, interestRate, namingDict=None, dateCol='Date', method='mean'):
    '''calc_conditional_metrics takes a data set, and computes the conditional financial metrics given the split column
    INPUTS:
        data: pandas df, columns should include listOfFactors and TSSign(theFactor)
        listOfFactors: list, set of factors to include
        interestCol: string, interest rate column
        splitCol: column to be used for conditioning
        namingDict: optional dictionary obejct that changes the names of the columns.  
            Keys are the values in splitCol, values are the names you would like to put into the columns
            
        dateCol: optional string, names the date column
        method: optional string, names the metric to calculate
    OUTPUTS:
        out: pandas df'''
    #Step 1: Store the unique values in the split column
    splitVals = list(data[splitCol].unique())
    splitVals = ['Unconditional'] + splitVals
    vals = np.zeros((len(listOfFactors),len(splitVals)))
    #Step 2: Loop over all of the factors
    for i in range(len(listOfFactors)):
        factor = listOfFactors[i]
        #Loop over the different splits, calculate the metric for this factor
        for j in range(len(splitVals)):
            if(splitVals[j] == 'Unconditional'):
                specificData = data[[factor, interestRate]].copy()
            else:
                specificData = data[[factor, interestRate]][data[splitCol] == splitVals[j]].copy()
            #Calculate the metric
            if(method=='mean'):
                vals[i,j] = (specificData[factor].mean()+1)**12-1            
            if(method=='sharpe'):
                vals[i,j] = np.sqrt(12)*np.mean(specificData[factor] - specificData[interestRate])/specificData[factor].std()
            if(method == 'logReturn'):
                vals[i,j] = 12*np.log(specificData[factor]).mean()      
            if(method == 'skew'):
                vals[i,j] = specificData[factor].skew()
            if(method == 'kurtosis'):
                vals[i,j] = specificData[factor].kurtosis()
            
    #Step 3: Change the column names if specified
    if(namingDict is None):
        out = pd.DataFrame(vals, columns=splitVals,
                      index=listOfFactors)
    else:
        cols = ['Unconditional']
        for j in range(1,len(splitVals)):
            cols.append(namingDict[splitVals[j]])
        out = pd.DataFrame(vals, columns=cols, index=listOfFactors)
    return out

def calc_participation(data, listOfFactors, referenceCol):
    '''cal_participation calculates the upside / downside participation and the particiption ratio difference
    INPUTS:
        data: pandas df, must contain columns listOfFactors and referenceCol
        listOfFactors: list, elements are strings, indicating which columns should be included in the calculation
        referenceCol: string, indicating which column is used as the benchmark
    OUTPUTS:
        out: pandas df, columns are {Upside Ratio, Downside Ratio, PRD}, index is the factors in listOfFactors
    '''
    #Step 1: Breakt into upside and downside
    upsideData = data[data[referenceCol] > 0].copy()
    downsideData = data[data[referenceCol] < 0].copy()

    out = pd.DataFrame(upsideData[listOfFactors].mean()/upsideData[referenceCol].mean(), columns=['Upside Participation'])
    out['Downside Participation'] = downsideData[listOfFactors].mean()/downsideData[referenceCol].mean()
    out['PRD'] = out['Upside Participation'] - out['Downside Participation']

    return out

def hurst_ratio_calc(data, listOfFactors, plotFactor = None):
    '''hurst_ratio_cal calculates the hurst ratio for listOfFactors
    INPUTS:
        data: pandas df, must contain listOfFactors
        listOfFactors: list, elements are strings, names of columns you wish to copy
    OUTPUTS:
        out: df, columns are listOfFactors, rows are (R/S)_t'''
    #Part 1: Calculate the rescaled range series
    #Step 1: Calculate Y_t
    data.reset_index(inplace=True, drop=True)
    Yt = data[listOfFactors].copy()
    Yt = Yt - data[listOfFactors].mean()
    #Step 2: Calculate Z_t
    Zt = Yt.cumsum()
    #Step 3: Calculate R_t
    Rt = Zt.cummax() - Zt.cummin()
    #Step 4: Calculate St
    St = np.zeros((data.shape[0], len(listOfFactors)))
    for i in range(St.shape[0]):
        St[i,:] = data.loc[0:i,listOfFactors].std()
    St = pd.DataFrame(St, columns=listOfFactors)
    #Step 5: Calculate (R/S)_t
    rangeSeries = Rt/St
    #Part 2: Run Regressions using the powers of t
    rangeSeries['t'] = rangeSeries.index
    rangeSeries = rangeSeries[['t'] + listOfFactors]
    rangeSeries = rangeSeries.loc[1:,:]
    logSeries = rangeSeries.applymap(math.log)/math.log(2)
    
    out = np.zeros((len(listOfFactors), 2))
    inds = [2**x for x in range(4,math.floor(math.log(logSeries.shape[0],2))+1)]
    filtered = logSeries.loc[inds, :].copy()
    for i in range(len(listOfFactors)):
        #Run Regression
        l = ['t']
        X = filtered[l]
        y = filtered[listOfFactors[i]]
        X2 = sm.add_constant(X)
        est = sm.OLS(y, X2)
        est2 = est.fit()
        #if(listOfFactors[i]=='RMW'):
            #print(est2.summary())
        #Store to pandas df
        out[i,0] = est2.params.t
        out[i,1] = est2.bse.t

        if(plotFactor == listOfFactors[i]):
            #Make a plot
            plt.plot(X,y, 'o')
            predicted = X*est2.params.t + est2.params.const
            plt.plot(X,predicted)
            plt.title('Plot of Line of Best Fit verses log((R/S)_t) vs log(t) for ' + plotFactor)
            plt.legend(['log((R/S)_t)', 'Line of Best Fit'])
            plt.xlabel('Log(t)')
            plt.ylabel('(R/S)_t')
            plt.show()

    if(plotFactor is None):
        out = pd.DataFrame(out, columns=['Hurst Coef', 'Std Error'], index=listOfFactors)
        return out

def hurst_ratio_over_time(data, listOfFactors, dateCol, breakNum = 1025):
    '''hurst_ratio_over_time calcualtes the hurst ratio over time
    INPUTS:
        data: pandas df, must contain listOFFactors in it's columns
        listOfFactors: list, elements are strings
        breakNum: int, number of days to include
        dateCol: string, names the date column in data
    OUTPUTS:
        out: pandas df, contains the listOFactors and the date
    '''
    nPeriods = int(np.ceil(data.shape[0]/breakNum))
    out = np.zeros((nPeriods,len(listOfFactors)))
    dates = []

    for i in range(nPeriods):
        currPeriodData = data[listOfFactors].loc[i*breakNum:min((i+1)*breakNum-1,data.shape[0]-1),listOfFactors].copy()
        hurstRatioForPeriod = hurst_ratio_calc(currPeriodData, listOfFactors)
        out[i,:] = hurstRatioForPeriod['Hurst Coef']
        dates.append(data.loc[min((i+1)*breakNum-1,data.shape[0]-1),dateCol])

    out = pd.DataFrame(out, columns=listOfFactors)
    out['Date'] = dates
    out = out[['Date'] + listOfFactors]

    return out











