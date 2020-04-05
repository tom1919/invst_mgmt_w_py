from __future__ import division
from matplotlib import pyplot as plt
from numpy.linalg import inv,pinv
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import math



def compute_simple_factor_momentums(data, listOfFactors, lookbackWindow=[12, 1], dateCol='Date'):
    '''compute_simple_factor_momentums takes a data set, and computes the time series and cross sectional factor momentum
    INPUTS:
        data: pandas df, columns should include listOfFactors
        listOfFactors: list, set of factors to include
        lookbackWindow: set of months to use.  Time period to consider it t-lookbackWindow[0], t-lokbackWindow[1]
        dateCol: string, names the date column
    OUTPUTS:
        out: pandas df, should be TSMOM, CSMOM, and Date
            TSMOM: Time series momentum, split into positive, negative and net
            CSMOM: Cross sectional momentum, split into positive, negative and net
            Date: Date Col
    '''
    vals = np.zeros((data.shape[0] - lookbackWindow[0], 6))
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
        #Now, compute return on the positive leg of the time series momentum
        if(len(pos) != 0):
            posRet = data.loc[i, pos].mean()
        else:
            posRet = 0
        if(len(neg) != 0):
            negRet = data.loc[i, neg].mean()
        else:
            negRet = 0
            
        vals[i-lookbackWindow[0],0] = posRet
        vals[i-lookbackWindow[0],1] = negRet
        vals[i-lookbackWindow[0],2] = posRet - negRet
        #Do the same thing with cross sectional factor momentum
        crossPos = list(ret[ret > ret.median()].index)
        crossNeg = list(ret[ret < ret.median()].index)
        crossPosRet = data.loc[i, crossPos].mean()
        crossNegRet = data.loc[i, crossNeg].mean()
        vals[i-lookbackWindow[0],3] = crossPosRet
        vals[i-lookbackWindow[0],4] = crossNegRet
        vals[i-lookbackWindow[0],5] = crossPosRet - crossNegRet
        
    out = pd.DataFrame(vals, columns=['TSMOMPos', 'TSMOMNeg','TSMOMNet',
                                     'CSMOMPos', 'CSMOMNeg', 'CSMOMNet'])
    out['Date'] = data.loc[lookbackWindow[0]:data.shape[0],dateCol].values
    out = out[['Date', 'TSMOMPos', 'TSMOMNeg','TSMOMNet','CSMOMPos', 'CSMOMNeg', 'CSMOMNet']].copy()
    return out

def compute_factor_momentum(data, listOfFactors, lookbackWindow=[12, 1], dateCol='Date', typeOfMOM='TS', method='equal', volHistory=[12,1], topN=False):
    '''compute_factor_momentum takes a data set, and computes the time series and cross sectional factor momentum
    INPUTS:
        data: pandas df, columns should include listOfFactors
        listOfFactors: list, set of factors to include
        lookbackWindow: set of months to use.  Time period to consider it t-lookbackWindow[0], t-lokbackWindow[1].  Second argument must be geq 1
        typeOfMOM: string, acceptable inputs are TS or CS.  For time series factor momentum or cross sectional factor momentum
        method: string, acceptable inputs are 'equal' or 'rps' for equal weight or risk parity (simplified), weighting scheme
        volHistory: string, number of data points to use in volatility calculation
    OUTPUTS:
        out: pandas df, should be TSMOM, CSMOM, and Date
            TSMOM: Time series momentum, split into positive, negative and net
            CSMOM: Cross sectional momentum, split into positive, negative and net
            Date: Date Col
    '''
    #Perform basic input checks
    if(dateCol not in list(data.columns)):
        print(dateCol + ' Columm not in data')
        return 0
    if(typeOfMOM not in ['TS','CS']):
        print('Incorrect value for typeOfMOM')
        return 0
    if(method not in ['equal', 'srp', 'rp']):
        print('Incorrect value for method')
        return 0

    data2 = data.copy()
    data2.fillna(0, inplace=True)  
    indexOfBeginnings = data2[listOfFactors].ne(0).idxmax()
    listOfCurrentAssets = list(indexOfBeginnings[indexOfBeginnings==0].index)
    listOfCurrentAssets = [asset for asset in listOfCurrentAssets if asset in listOfFactors]

    vals = np.zeros((data2.shape[0] - lookbackWindow[0], 3))
    #Fill in Date Column
    for i in range(lookbackWindow[0], data2.shape[0]):
        #Check if you need to append the asset list
        if((i in indexOfBeginnings.values) & (i != lookbackWindow[0])):
            newAssetsList = list(indexOfBeginnings[indexOfBeginnings==i].index)
            newAssetsList = [asset for asset in newAssetsList if asset in listOfFactors]
            listOfCurrentAssets = listOfCurrentAssets + newAssetsList


        #Compute cumulative return over lookback window
        new = data2.loc[i-lookbackWindow[0]:i-lookbackWindow[1], listOfCurrentAssets].copy()
        ret = new + 1
        ret = ret.product()
        ret = ret - 1
        ret.sort_values(ascending=False,inplace=True)
        #Check Method to define set of factors to use
        if(typeOfMOM == 'TS'):
            #Get list of Factors with Positive Return over lookback period
            pos = list(ret[ret > 0].index)
            #Get list of Factors with Negative Return over lookback period
            neg = list(ret[ret < 0].index)

        elif(typeOfMOM == 'CS'):
            if(topN==False):
                #Get list of factors with above median return over lookback period
                pos = list(ret[ret > ret.median()].index)
                #Get list of factor with below median return over lookback period
                neg = list(ret[ret < ret.median()].index)

            elif(topN >=1):
                pos = list(ret.index)
                l = ret.shape[0]
                pos = pos[:min(l,topN)]
                #Get list of factor with below median return over lookback period
                neg = list(ret.index)
                neg = neg[-min(l,topN):]

            elif((topN <1) & (topN > 0)):
                pos = list(ret.index)
                l = math.ceil(len(pos)*topN)
                pos = pos[:l]
                #Get list of factor with below median return over lookback period
                neg = list(ret.index)
                l = math.ceil(len(neg)*topN)
                neg = neg[-l:]

            else:
                print('Invalid Value for variable topN')
                return 0


        #Now, compute return on the positive leg of the time series momentum
        if(len(pos) != 0):
            if(method== 'equal'):
                posRet = data2.loc[i, pos].mean()
            elif(method == 'srp'):
                posVol = data.loc[max(i-volHistory[0],0):max(i-volHistory[1],0), pos].std()
                weights = 1/posVol
                weights = weights/weights.sum()
                posRet = data.loc[i, pos].transpose().dot(weights)
            elif(method=='rp'):
                covarMatrix = data.loc[max(i-volHistory[0],0):max(i-volHistory[1],0), pos].cov()
                weights = calc_risk_parity_weights(covarMatrix)
                posRet = data.loc[i, pos].transpose().dot(weights)
        else:
            posRet = 0

        if(len(neg) != 0):
            if(method== 'equal'):
                negRet = data2.loc[i, neg].mean()
            elif(method == 'srp'):
                negVol = data2.loc[max(i-volHistory[0],0):max(i-volHistory[1],0), neg].std()
                weights = 1/negVol
                weights = weights/weights.sum()
                negRet = data2.loc[i, neg].transpose().dot(weights)

            elif(method=='rp'):
                covarMatrix = data2.loc[max(i-volHistory[0],0):max(i-volHistory[1],0), neg].cov()
                weights = calc_risk_parity_weights(covarMatrix)
                negRet = data2.loc[i, neg].transpose().dot(weights)
        else:
            negRet = 0
        
        vals[i-lookbackWindow[0],0] = posRet
        vals[i-lookbackWindow[0],1] = negRet
        vals[i-lookbackWindow[0],2] = posRet - negRet

    #Determine the column names
    if(typeOfMOM == 'TS'):
        cols = ['TSMOMPos', 'TSMOMNeg','TSMOMNet']
    elif(typeOfMOM == 'CS'):
        cols = ['CSMOMPos', 'CSMOMNeg', 'CSMOMNet']
        
    out = pd.DataFrame(vals, columns=cols)
    out['Date'] = data2.loc[lookbackWindow[0]:data2.shape[0],dateCol].values
    cols = ['Date'] + cols
    out = out[cols].copy()
    return out


def compute_portfolio_returns(data, listOfFactors, dateCol, weightingScheme='equal', volHistory=[12,1]):
    '''compute_portfolio_returns takes a set of factors, and a weightingScheme and returns a set of returns
    INPUTS:
        data: pandas df, columns should include the date column, and the listOfFactors
        dateCol: string, names the date column
        weightingScheme: string, must be one of ['equal','srp','cap']
        volHistory: two entry list, defines the lookback window for the calculating volatility
    OUTPUTS:
        out: pandas df, has the date column dateCol, and a second column corresponding to the series of returns specified
    '''
    #Perform Sanity Checks on the Inputs
    if (dateCol not in data.columns):
        print('Date column not in data frame')
        return 0
    if(weightingScheme not in ['equal','srp','rp','cap']):
        print('Incorrect weighting scheme, must be either equal, srp, or cap')
        return 0
    if(volHistory[1] < 1):
        print('bad lookback window, second argumnet must be at least 1')

    #Begin function
    data = data.sort_values(dateCol)
    if(weightingScheme=='equal'):
        out = data.copy()
        out['Equal Weight Returns'] = out[listOfFactors].mean(axis=1)
        l = [dateCol] + ['Equal Weight Returns']
        return out[l]

    #Simple Risk Parity
    elif(weightingScheme=='srp'):
        vals = np.zeros((data.shape[0] - volHistory[0], 1))
        for i in range(volHistory[0], data.shape[0]):
            Vol = data.loc[max(i-volHistory[0],0):max(i-volHistory[1],0), listOfFactors].std()
            weights = 1/Vol
            weights = weights/weights.sum()
            ret = data.loc[i, listOfFactors].transpose().dot(weights)
            vals[i-volHistory[0]] = ret
        out = pd.DataFrame(vals, columns=['Simple Risk Parity Return'])
        out[dateCol] = data.loc[volHistory[0]:data.shape[0],dateCol].values
        return out[[dateCol,'Simple Risk Parity Return']]

    #Full Risk Parity
    elif(weightingScheme=='rp'):
        vals = np.zeros((data.shape[0] - volHistory[0], 1))
        for i in range(volHistory[0], data.shape[0]):
            covarMatrix = data.loc[max(i-volHistory[0],0):max(i-volHistory[1],0), listOfFactors].cov()
            weights = calc_risk_parity_weights(covarMatrix)
            ret = data.loc[i, listOfFactors].transpose().dot(weights)
            vals[i-volHistory[0]] = ret
        out = pd.DataFrame(vals, columns=['Risk Parity Return'])
        out[dateCol] = data.loc[volHistory[0]:data.shape[0],dateCol].values
        return out[[dateCol,'Risk Parity Return']]

    elif(weightingScheme=='cap'):
        #This means you are cap weighted
        weights = pd.DataFrame(1/len(listOfFactors), index=listOfFactors, columns=['Weights'])
        vals = np.zeros((data.shape[0], 1))
        for i in range(data.shape[0]):
            #Compute Returns
            vals[i] = data.loc[i, listOfFactors].dot(weights)
            #Drift Weights
            R = pd.DataFrame(data.loc[i, listOfFactors] + 1)
            R.columns = ['Weights']
            weights = weights*R
            weights = weights/weights.sum()
        out = pd.DataFrame(vals, columns=['Cap Weighted Return'])
        out[dateCol] = data[dateCol]
        return out[[dateCol,'Cap Weighted Return']]


def create_index_with_unbalanced_data(data, listOfAssets, dateCol, weightingScheme='cap', fillVal=0):
    '''create_index_with_unbalanced_data creates an unbalanced '''
    if(weightingScheme=='equal_ignore_missing'):
        out = pd.DataFrame(data[listOfAssets].mean(axis=1), columns=['Returns'])
        out[dateCol] = data[dateCol]
        return out[[dateCol, 'Returns']]
    elif(weightingScheme=='equal_fill_missing'):
        data2 = data.copy()
        data2.fillna(fillVal, inplace=True)  
        indexOfBeginnings = data2[listOfAssets].ne(fillVal).idxmax()
        listOfCurrentAssets = list(indexOfBeginnings[indexOfBeginnings==0].index)
        dfWeights = pd.DataFrame(np.ones((len(listOfCurrentAssets),1)), columns=['Weights'])
        dfWeights = dfWeights/dfWeights.shape[0]
        dfWeights.index = listOfCurrentAssets
        vals = np.zeros((data2.shape[0],2))
        vals[0,0] = data2.loc[0,list(dfWeights.index)].dot(dfWeights)
        vals[0,1] = 1/np.power(dfWeights,2).sum()
        #print(dfWeights)
        for i in range(1,data2.shape[0]):
            #Check if you need to add anyone new thing into the mix
            if(i in indexOfBeginnings.values):
                newAssetsList = list(indexOfBeginnings[indexOfBeginnings==i].index)
                listOfCurrentAssets = listOfCurrentAssets + newAssetsList
                dfWeights = pd.DataFrame(np.ones((len(listOfCurrentAssets),1)), columns=['Weights'])
                dfWeights.index = listOfCurrentAssets
                dfWeights = dfWeights/dfWeights.shape[0]
                #print(dfWeights)
            vals[i,0] = data2.loc[i,list(dfWeights.index)].dot(dfWeights)
            vals[i,1] = 1/np.power(dfWeights,2).sum()
        out = pd.DataFrame(vals, columns=['Returns', 'Effective Asset Number'])
        out[dateCol]=data2[dateCol]
        return out[[dateCol,'Returns', 'Effective Asset Number']]
        
    elif(weightingScheme=='cap'):
        data2 = data.copy()
        data2.fillna(fillVal, inplace=True)  
        indexOfBeginnings = data2[listOfAssets].ne(fillVal).idxmax()
        initialAssets = list(indexOfBeginnings[indexOfBeginnings==0].index)
        dfWeights = pd.DataFrame(np.ones((len(initialAssets),1)), columns=['Weights'])
        dfWeights = dfWeights/dfWeights.shape[0]
        dfWeights.index = initialAssets
        vals = np.zeros((data2.shape[0],2))
        #Calculate the first return and update the effective manager number
        vals[0,0] = data2.loc[0,list(dfWeights.index)].dot(dfWeights)
        vals[0,1] = 1/np.power(dfWeights,2).sum()
        #Drift the weights
        drift = pd.DataFrame(1 + data2.loc[0,initialAssets])
        drift.columns=['Weights']
        dfWeights = dfWeights*drift
        dfWeights = dfWeights/dfWeights.sum()
        listOfCurrentAssets = initialAssets
        #print(dfWeights)
        for i in range(1,data2.shape[0]):
            #Check to see if a new index has entered the mix/if so, adjust the weights
            if(i in indexOfBeginnings.values):
                newAssetsList = list(indexOfBeginnings[indexOfBeginnings==i].index)
                newWeights = pd.DataFrame(np.ones((len(newAssetsList),1)))
                newWeights.columns=['Weights']
                newWeights.index=newAssetsList
                newWeights = newWeights*len(newAssetsList)/(len(newAssetsList)+dfWeights.shape[0])
                dfWeights = dfWeights*dfWeights.shape[0]/(len(newAssetsList)+dfWeights.shape[0])
                dfWeights = dfWeights.append(newWeights)
                listOfCurrentAssets = listOfCurrentAssets + newAssetsList
            #Then, calculate the returns, update the effective manager number
            vals[i,0] = data2.loc[i,list(dfWeights.index)].dot(dfWeights)
            vals[i,1] = 1/np.power(dfWeights,2).sum()
            #Finally, drift the weights
            drift = pd.DataFrame(1 + data2.loc[i,listOfCurrentAssets])
            drift.columns=['Weights']
            dfWeights = dfWeights*drift
            dfWeights = dfWeights/dfWeights.sum()
        
        #Now, put everything into a nice package
        out = pd.DataFrame(vals, columns=['Returns', 'Effective Asset Number'])
        out[dateCol] = data2[dateCol]
        return out[[dateCol,'Returns','Effective Asset Number']]

#Leverage Functions
def dynamic_leverage(data, baseCol, colsToLever, borrowCostCol, dateCol='Date', lookbackWindow=[12,1]):
    '''dynamic leverage scales the returns of colsToLever to match the volatility of baseCol
    INPUTS:
        data: pandas df, needs to contain baseCol, colsToLever and dateCol amoung it's columns
        baseCol: string, names the column to be used as scaling
        colsToLever: columns to lever
        dateCol: date column
        lookbackWindow: array with 2 values, defines the lookback window to adjust the std deviation
    OUTPUTS:
        out: pandas df, contains the dateCol, baseCol, the original unlevered returns, and the leveraged returns, and the leverage ratios
    '''
    vals = np.zeros((data.shape[0] - lookbackWindow[0], 2*len(colsToLever)+1))
    basePlusCols = [baseCol] + colsToLever
    for i in range(lookbackWindow[0], data.shape[0]):
        #Calculate Leverage Using the Lookback Window
        hist = data.loc[i-lookbackWindow[0]:i-lookbackWindow[1], basePlusCols].copy()
        vols = hist.std()
        leverage = vols[baseCol] / vols
        #Store Leverage in the vals array
        vals[i-lookbackWindow[0],0] = vols[baseCol]
        vals[i-lookbackWindow[0],1:len(colsToLever)+1] = leverage[colsToLever]
        vals[i-lookbackWindow[0],len(colsToLever)+1:] = leverage[colsToLever]*data.loc[i,colsToLever] - (leverage[colsToLever]-1)*data.loc[i,borrowCostCol]
    
    leverageNames = [name + ' Leverege Ratio' for name in colsToLever]
    leveredNames = [name + ' DL Return' for name in colsToLever]
    cols = [baseCol + ' Lookback Vol'] + leverageNames + leveredNames
    out = pd.DataFrame(vals, columns=cols)
    out['Date'] = data.loc[lookbackWindow[0]:data.shape[0],dateCol].values
    cols = ['Date'] + cols
    out = out[cols].copy()
    return out

def static_leverage(data, colsToLever, leverage, borrowCostCol, dateCol='Date'):
    '''static_leverage leverages the columns in data according to some a priori leverage ratio'''
    data2 = data[[dateCol]].copy()
    for col in colsToLever:
        newColName = col + ' Levered ' + str(leverage) + 'x'
        data2[newColName] = data[col]*leverage - (leverage-1)*data[borrowCostCol]
    return data2


#The code below calculates the Full Risk Parity Nonsense
# risk budgeting optimization
def calculate_portfolio_var(w,V):
    # function that calculates portfolio risk
    w = np.matrix(w)
    return (w*V*w.T)[0,0]

def calculate_risk_contribution(w,V):
    # function that calculates asset contribution to total risk
    w = np.matrix(w)
    sigma = np.sqrt(calculate_portfolio_var(w,V))
    # Marginal Risk Contribution
    MRC = V*w.T
    # Risk Contribution
    RC = np.multiply(MRC,w.T)/sigma
    return RC

def risk_budget_objective(x,pars):
    # calculate portfolio risk
    V = pars[0]# covariance table
    x_t = pars[1] # risk target in percent of portfolio risk
    sig_p =  np.sqrt(calculate_portfolio_var(x,V)) # portfolio sigma
    risk_target = np.asmatrix(np.multiply(sig_p,x_t))
    asset_RC = calculate_risk_contribution(x,V)
    J = sum(np.square(asset_RC-risk_target.T))[0,0] # sum of squared error
    return J

def total_weight_constraint(x):
    return np.sum(x)-1.0

def long_only_constraint(x):
    return x

def calc_risk_parity_weights(covarMatrix, targetScale = .5):
    '''def_calc_risk_parity_weights, calculates risk parity weights
    INPUTS:
        coarMatrix: pd DataFrame, covariance matrix
    OUTPUTS:
        dfWeights: pd Dataframe, weight vector
    '''
    #Scale Covariance Matrix to Avoid Rounding Errors
    normVal = np.linalg.norm(covarMatrix)
    covarMatrixScaled = targetScale/normVal*covarMatrix

    n = covarMatrixScaled.shape[0]
    x_t = [1/n for x in range(n)] # your risk budget percent of total portfolio risk (equal risk)
    w0 = x_t
    V = np.matrix(covarMatrixScaled)
    cons = ({'type': 'eq', 'fun': total_weight_constraint},
    {'type': 'ineq', 'fun': long_only_constraint})
    res= minimize(risk_budget_objective, w0, args=[V,x_t], method='SLSQP',constraints=cons, options={'disp': False})
    w_rb = np.asmatrix(res.x)
    dfWeights = pd.DataFrame(w_rb, columns=list(covarMatrixScaled.columns))
    dfWeights = dfWeights.transpose()
    dfWeights.columns = ['Weights']
    return dfWeights

def covariance_matrix_projection_from_factors(dataAssets, listOfAssets, dataFactors, listOfFactors, 
                                              dateColAssets='Date',dateColFactors='Date'):
    '''covariance_matrix_projection_from_factors takes a pandas df data, listOfAssets and listOfFactors
    returns a covariance matrix using the projection of factor covariance into asset space
    INPUTS:
        dataAssets: pandas df, must contain listOfAssets and dateColAssets
        listOfAssets: list, elements are strings naming the columns that are asset returns
        dataFactors: pandas df, must contain listOfFactors, and dateColFactors
        listOfFactors: pandas df, elements are strings naming the columns that are asset returns'''
    #Step 1: Calculate the Factor Covariance Matrix
    factorCovar = dataFactors[listOfFactors].cov()
    combined = dataAssets.merge(dataFactors, left_on=dateColAssets, right_on=dateColFactors)
    betas = np.zeros((len(listOfAssets),len(listOfFactors)))
    #Step 2: Calculate the factor loadings for the assets
    for i in range(len(listOfAssets)):
        reg = LinearRegression().fit(combined[listOfFactors], combined[[listOfAssets[i]]])
        betas[i,:] = reg.coef_
    betas = pd.DataFrame(betas, columns=listOfFactors)
    betas.index=listOfAssets
    #Step 3: Take B*V*B.T to project into asset space
    return betas.dot(factorCovar).dot(betas.T)









