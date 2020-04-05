from __future__ import division
from matplotlib import pyplot as plt
from numpy.linalg import inv,pinv
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import math


def compute_returns(data, listOfAssets, typeOfStrat, lookbackWindow=[12, 1], dateCol='Date', weightScheme='Equal', volHistory=[12,1], topN=False, supressTurnover=True):
    '''compute_returns takes in a pandas df data, and a listOfAssets and computes the series of returns
    INPUTS:
        data: pandas df, columns must include listOfAssets and dateCol
        listOfAssets: list, elements are strings defining the list of assets
        typeOfStrat: string, defines the type of return series being created.  Acceptable verions are INDEX, TS, CS
            Index: indicates the series in an index
            TS: Time series momentum
            CS: Cross sectional momentum
        lookbackWindow: list, 2 elements long, defines the lookback window used for momentum calculation
        dateCol: string, indicates the dateCol
        weightScheme: string, defines the weighting scheme, supported values are equal, srp, rp, cap
        volHistory: list, 2 elements long, defines the lookback window for computing a covariance matrix for risk parity weights
        topN: int or float, defines number of assets to include in CS MOM
        supressTurnover: Boolean, True 
    Outputs:
        out: pandas df, n columns, first column is dateCol, if typeOfStrat = 'Index' then second column is index returns, otherwise it has 4 columns that are the returns of Pos, Neg and Net
        outTurnover: pandas df, 3 columns
        weightsDict: dictionary, keys are dates, values are dictionaries with 2 pandas df corresponding to the weights before and after the end of the holding period
    '''

    #Step 1: Perform Basic Data Checks
    if(dateCol not in list(data.columns)):
        print(dateCol + ' Columm not in data')
        return 0
    if(typeOfStrat not in ['Index','TS','CS']):
        print('Incorrect value for typeOfStrat')
        return 0
    if(weightScheme not in ['Cap','Equal', 'SRP', 'RP']):
        print('Incorrect value for weightScheme')
        return 0
    if((typeOfStrat in ['TS','CS']) & (weightScheme == 'Cap')):
        print('Can''t create cap weighted momentum strategy')
        return 0
    if(topN < 0):
        print('Invalid value for variable topN')
        return 0

    data2 = data.copy()
    data2.fillna(0, inplace=True) 
    data2.reset_index(drop=True, inplace=True)
    #Step 2: Initialize Active Set and set of returns
    indexOfBeginnings = data2[listOfAssets].ne(0).idxmax()

    #Step 3: Initialize size of set of values to calculate
    if(typeOfStrat in ['TS', 'CS']):
        vals = np.zeros((data2.shape[0] - lookbackWindow[0], 3))
    else:
        vals = np.zeros((data2.shape[0] - lookbackWindow[0], 1))

    #Step 4: Initialize dfWeights and the dictionary of returns, and compute the first day's returns, and store them
    dictOfWeights = dict()
    if((typeOfStrat=='Index') & (weightScheme=='Cap')):
        activeAssets, newAssets = active_set(data2, listOfAssets, typeOfStrat, topN, indexOfBeginnings, lookbackWindow[0],lookbackWindow)
        dfWeights = compute_weights(data2, listOfAssets, activeAssets, lookbackWindow[0], lookbackWindow, 'Equal', volHistory)
        dictOfWeights[data2.loc[lookbackWindow[0],dateCol]] = {'StartWeights':dfWeights}
        ret, dfWeights = drift_weights(data2, lookbackWindow[0], listOfAssets, dfWeights)
        vals[0,0] = ret
        dictOfWeights[data2.loc[lookbackWindow[0],dateCol]]['EndWeights'] = dfWeights
    elif(typeOfStrat == 'Index'):
        #Define dfWeights
        activeAssets, newAssets = active_set(data2, listOfAssets, typeOfStrat, topN, indexOfBeginnings, lookbackWindow[0],lookbackWindow)
        dfWeights = compute_weights(data2, listOfAssets, activeAssets, lookbackWindow[0], lookbackWindow, weightScheme, volHistory)
        dictOfWeights[data2.loc[lookbackWindow[0],dateCol]] = {'StartWeights':dfWeights}
        ret, dfWeights = drift_weights(data2, lookbackWindow[0], listOfAssets, dfWeights)
        vals[0,0] = ret
        dictOfWeights[data2.loc[lookbackWindow[0],dateCol]]['EndWeights'] = dfWeights
    else: #Compute initial weights
        pos, neg = active_set(data2, listOfAssets, typeOfStrat, topN, indexOfBeginnings, lookbackWindow[0],lookbackWindow)
        dfWeightsPos = compute_weights(data2, listOfAssets, pos, lookbackWindow[0], lookbackWindow, weightScheme, volHistory)
        dfWeightsNeg = compute_weights(data2, listOfAssets, neg, lookbackWindow[0], lookbackWindow, weightScheme, volHistory)
        dfWeightsNet = dfWeightsPos - dfWeightsNeg
        
        #Compute positive basket returns
        dictOfWeights[data2.loc[lookbackWindow[0],dateCol]] = {'StartWeightsPos':dfWeightsPos}
        retPos, dfWeightsPos = drift_weights(data2, lookbackWindow[0], listOfAssets, dfWeightsPos)
        vals[0,0] = retPos
        dictOfWeights[data2.loc[lookbackWindow[0],dateCol]]['EndWeightsPos'] = dfWeightsPos

        #Compute negative basket returns
        dictOfWeights[data2.loc[lookbackWindow[0],dateCol]]['StartWeightsNeg'] = dfWeightsNeg
        retNeg, dfWeightsNeg = drift_weights(data2, lookbackWindow[0], listOfAssets, dfWeightsNeg)
        vals[0,1] = retNeg
        dictOfWeights[data2.loc[lookbackWindow[0],dateCol]]['EndWeightsNeg'] = dfWeightsNeg


        #Compute Net Basket Returns
        dictOfWeights[data2.loc[lookbackWindow[0],dateCol]]['StartWeightsNet'] = dfWeightsNet
        retNet, dfWeightsNet = drift_weights(data2, lookbackWindow[0], listOfAssets, dfWeightsNet)
        vals[0,2] = retNet
        dictOfWeights[data2.loc[lookbackWindow[0],dateCol]]['EndWeightsNet'] = dfWeightsNet

    #Step 5: Loop over the dates, and compute the returns, and store the weights

    turnoverVals = np.zeros((vals.shape[0]-1, vals.shape[1]))

    for i in range(lookbackWindow[0]+1, data2.shape[0]):
        #Compute New Weights, calc returns, and later compute turnover
        if((typeOfStrat=='Index') & (weightScheme=='Cap')):
            #Compute new weights
            activeAssets, newAssets = active_set(data2, listOfAssets, typeOfStrat, topN, indexOfBeginnings, i, lookbackWindow)
            dfWeights = dictOfWeights[data2.loc[i-1,dateCol]]['EndWeights']
            dfWeights = 1.0*(len(activeAssets)-len(newAssets))/len(activeAssets)*dfWeights
            dfWeights.loc[newAssets,:] = 1/len(activeAssets)
            #Compute Returns
            dictOfWeights[data2.loc[i,dateCol]] = {'StartWeights':dfWeights}
            ret, dfWeights = drift_weights(data2, i, listOfAssets, dfWeights)
            vals[i-lookbackWindow[0],0] = ret
            dictOfWeights[data2.loc[i,dateCol]]['EndWeights'] = dfWeights

            #Compute Turnover
            turnoverVals[i - 1 - lookbackWindow[0],0] = calc_turnover(dictOfWeights[data2.loc[i-1,dateCol]]['EndWeights'], dictOfWeights[data2.loc[i,dateCol]]['StartWeights'])

        elif(typeOfStrat=='Index'):
            activeAssets, newAssets = active_set(data2, listOfAssets, typeOfStrat, topN, indexOfBeginnings, i, lookbackWindow)
            dfWeights = compute_weights(data2, listOfAssets, activeAssets, lookbackWindow[0], lookbackWindow, weightScheme, volHistory)

            #Compute Returns
            dictOfWeights[data2.loc[i,dateCol]] = {'StartWeights':dfWeights}
            ret, dfWeights = drift_weights(data2, i, listOfAssets, dfWeights)
            vals[i-lookbackWindow[0],0] = ret
            dictOfWeights[data2.loc[i,dateCol]]['EndWeights'] = dfWeights

            #Compute Turnover
            turnoverVals[i - 1 - lookbackWindow[0],0] = calc_turnover(dictOfWeights[data2.loc[i-1,dateCol]]['EndWeights'], dictOfWeights[data2.loc[i,dateCol]]['StartWeights'])

        else: #Compute momentum strategy weights
            pos, neg = active_set(data2, listOfAssets, typeOfStrat, topN, indexOfBeginnings, i, lookbackWindow)
            dfWeightsPos = compute_weights(data2, listOfAssets, pos, i, lookbackWindow, weightScheme, volHistory)
            dfWeightsNeg = compute_weights(data2, listOfAssets, neg, i, lookbackWindow, weightScheme, volHistory)
            dfWeightsNet = dfWeightsPos - dfWeightsNeg

            #Compute Returns
            #Compute positive basket returns
            dictOfWeights[data2.loc[i,dateCol]] = {'StartWeightsPos':dfWeightsPos}
            retPos, dfWeightsPos =drift_weights(data2, i, listOfAssets, dfWeightsPos)
            vals[i-lookbackWindow[0],0] = retPos
            dictOfWeights[data2.loc[i,dateCol]]['EndWeightsPos'] = dfWeightsPos

            #Compute negative basket returns
            dictOfWeights[data2.loc[i,dateCol]]['StartWeightsNeg'] = dfWeightsNeg
            retNeg, dfWeightsNeg = drift_weights(data2, i, listOfAssets, dfWeightsNeg)
            vals[i-lookbackWindow[0],1] = retNeg
            dictOfWeights[data2.loc[i,dateCol]]['EndWeightsNeg'] = dfWeightsNeg

            #Compute Net Basket Returns
            dictOfWeights[data2.loc[i,dateCol]]['StartWeightsNet'] = dfWeightsNet
            retNet, dfWeightsNet = drift_weights(data2, i, listOfAssets, dfWeightsNet)
            vals[i-lookbackWindow[0],2] = retNet
            dictOfWeights[data2.loc[i,dateCol]]['EndWeightsNet'] = dfWeightsNet

            #Compute turnover
            turnoverVals[i - 1 - lookbackWindow[0],0] = calc_turnover(dictOfWeights[data2.loc[i-1,dateCol]]['EndWeightsPos'], dictOfWeights[data2.loc[i,dateCol]]['StartWeightsPos'])
            turnoverVals[i - 1 - lookbackWindow[0],1] = calc_turnover(dictOfWeights[data2.loc[i-1,dateCol]]['EndWeightsNeg'], dictOfWeights[data2.loc[i,dateCol]]['StartWeightsNeg'])
            turnoverVals[i - 1 - lookbackWindow[0],2] = calc_turnover(dictOfWeights[data2.loc[i-1,dateCol]]['EndWeightsNet'], dictOfWeights[data2.loc[i,dateCol]]['StartWeightsNet'])


    #Define the output
    #Turnover Output
    if(typeOfStrat == 'TS'):
        cols = ['TSMOMPos', 'TSMOMNeg','TSMOMNet']
    elif(typeOfStrat == 'CS'):
        cols = ['CSMOMPos', 'CSMOMNeg', 'CSMOMNet']
    elif(weightScheme == 'Cap'):
        cols = ['Buy and Hold Index']
    else:
        cols = ['Index ' + weightScheme + ' Weighting']    
    out = pd.DataFrame(vals, columns=cols)

    out[dateCol] = data2.loc[lookbackWindow[0]:data2.shape[0],dateCol].values
    cols = [dateCol] + cols
    out = out[cols].copy()

    if(supressTurnover==True):
        return out

    #Turnover Output
    if(typeOfStrat == 'TS'):
        turnoverCols = ['TSMOMPos Turnover', 'TSMOMNeg Turnover','TSMOMNet Turnover']
    elif(typeOfStrat == 'CS'):
        turnoverCols = ['CSMOMPos Turnover', 'CSMOMNeg Turnover', 'CSMOMNet Turnover']
    elif(weightScheme == 'Cap'):
        turnoverCols = ['Buy and Hold Index']
    else:
        turnoverCols = ['Index ' + weightScheme + ' Weighting']  

    turnoverOut = pd.DataFrame(turnoverVals, columns=turnoverCols)

    turnoverOut[dateCol] = data2.loc[lookbackWindow[0]+1:data2.shape[0],dateCol].values
    turnoverCols = [dateCol] + turnoverCols
    turnoverOut = turnoverOut[turnoverCols].copy()    

    return out, turnoverOut, dictOfWeights

def unpack_weights_dict(dictOfWeights, leg):
    '''unpack_weights_dict unpacks the dictOfWeights object from compute returns
    INTPUTS:
        dictOfWeights: Output from compute_returns, keys are timestamps, values are dictionaries containing the weights at the beginning and end of the month
        leg: string, Pos, Neg or Net, indicates which portfolio to unpack
    OUTPUTS:
        out: pandas df, columns include Date, and the weights of the factors
    '''
    keys = list(dictOfWeights.keys())
    out = np.zeros((len(list(dictOfWeights.keys())),dictOfWeights[keys[0]]['StartWeightsPos'].shape[0]))
    for i in range(len(keys)):
        if(leg=='Pos'):
            out[i,:] = dictOfWeights[keys[i]]['StartWeightsPos'].transpose()
        elif(leg=='Neg'):
            out[i,:] = dictOfWeights[keys[i]]['StartWeightsNeg'].transpose()
        else:
            out[i,:] = dictOfWeights[keys[i]]['StartWeightsNet'].transpose()
    out = pd.DataFrame(out, columns=list(dictOfWeights[keys[0]]['StartWeightsPos'].index))
    out['Date'] = keys
    return out[['Date'] + list(dictOfWeights[keys[0]]['StartWeightsPos'].index)]
    


#Define Helper Functions for compute_returns
def active_set(data, listOfAssets, typeOfStrat, topN, indexOfBeginnings, i, lookbackWindow):
    '''active_set calculates the active set of variables to be used
    INPUTS:
        data, listOfAssets, typeOfStrat, topN, lookbackWindow are same inputes in compute_returns function
        indexOfBeginnings: pandas df, index is the names of the factors, columns are the index of pandas df when they become active
        i: int, represents the index of the original data frame we are processing
    Output:
        if it's an index, if gives you the full set of active assets and full assets, we have activeAssets and newAssets
        if it's anything else, you give the positive and negative
    '''
    #Define the set of assets that are active at this point
    activeAssets = list(indexOfBeginnings[indexOfBeginnings<=i].index)
    activeAssets = [asset for asset in activeAssets if asset in listOfAssets]
    newAssets = list(indexOfBeginnings[indexOfBeginnings==i].index)
    newAssets = [asset for asset in newAssets if asset in listOfAssets]
    #If it is an index, return activeAssets
    if(typeOfStrat=='Index'):
        return activeAssets, newAssets

    #If not, compute TS or CS active sets
    new = data.loc[i-lookbackWindow[0]:i-lookbackWindow[1], activeAssets].copy()
    ret = new + 1
    ret = ret.product()
    ret = ret - 1
    ret.sort_values(ascending=False,inplace=True)
    #Check Method to define set of factors to use
    if(typeOfStrat == 'TS'):
        #Get list of Factors with Positive Return over lookback period
        pos = list(ret[ret > 0].index)
        #Get list of Factors with Negative Return over lookback period
        neg = list(ret[ret < 0].index)

    elif(typeOfStrat == 'CS'):
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

    return pos, neg
    
def compute_weights(data, listOfAssets, assetSet, i, lookbackWindow, weightScheme, volHistory):
    '''compute_weights_mom_strat computes a pandas df, index are listOfAssets and column is weights
    INPUTS:
        data, listOfAssets, lookbackWindow, weightScheme, volHistory are the same as in compute_returns
        i: int, index of the date whose returns you are computing
        assetSet: defines the active set of assets
    Outputs:
        dfWeights: pandas df, index listOfAssets, column is weights
    '''
    dfWeights = pd.DataFrame(np.zeros((len(listOfAssets),1)), columns=['Weights'], index=listOfAssets)
    #Check if you have a null set
    if(len(assetSet) == 0):
        return dfWeights
    #Otherwise, calculate the weights normally
    #Equal Weights
    if(weightScheme=='Equal'):
        dfWeights.loc[assetSet,:] = 1/len(assetSet)
        return dfWeights
    #SRP: Inverse volatility weights
    elif(weightScheme=='SRP'):
        Vol = data.loc[max(i-volHistory[0],0):max(i-volHistory[1],0), assetSet].std()
        weights = 1/Vol
        weights = weights/weights.sum()
        weights = pd.DataFrame(weights)
        dfWeights.loc[assetSet,:] = weights.loc[assetSet,:].values
        return dfWeights
    elif(weightScheme == 'RP'):
        covarMatrix = data.loc[max(i-volHistory[0],0):max(i-volHistory[1],0), assetSet].cov()
        weights = calc_risk_parity_weights(covarMatrix)
        dfWeights.loc[assetSet,:] = weights.loc[assetSet,:]
        return dfWeights

def drift_weights(data, i, listOfAssets, dfWeights):
    '''drift_weights takes in a data set, and computes the 1 period return, and drifts the weights
    INPUTS:
        data: pandas df, must constain list of assets in it's columns
        i: int, index of the date whose returns you are computing
        listOfAssets: list, elements are strings representing the assets
        dfWeights: pandas df, index is listOfAssets, column name is weights
    Outputs:
        ret: float, 1 period return
        dfWeights: new weights
    '''
    previousGrossExposure = np.sum(abs(dfWeights.values))
    ret = data.loc[i,listOfAssets].dot(dfWeights)
    R = pd.DataFrame(data.loc[i, listOfAssets] + 1)
    R.columns = ['Weights']
    dfWeightsDrifted = dfWeights*R

    currentGrossExposure = np.sum(abs(dfWeightsDrifted.values))
    dfWeightsDrifted = (previousGrossExposure/currentGrossExposure)*dfWeightsDrifted
    return ret, dfWeightsDrifted

def calc_turnover(df1, df2):
    '''calc_turnover calculates turnover between two data frames'''
    if(list(df1.index) != list(df2.index)):
        print('Dataframes contain different assets')
        return 0
    else:
        return np.sum(abs((df1.values-df2.values)))/(2*np.sum(abs(df1.values)))

         


#The code below calculates the Risk parity Weights
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









