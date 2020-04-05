#This is a library that runs the regime-filtering analysis


import numpy as np #for numerical array data
import pandas as pd #for tabular data
from scipy.optimize import minimize
import matplotlib.pyplot as plt #for plotting purposes

import cvxpy as cp


def filter_time_series(data, nameCol, dateCol, paramLambda):
    '''filter_time_series filters a time series based on Mulvey' non-parametric trend filtering algorithm
    INPUTS:
        data: pandas df, should contain columns nameCol, and dateCol
        nameCol: string, name of column to be used in trend filtering
        dateCol: string, name of column that indexes the dates
        paramLambda: lambda, hyper-parameter for trend filtering algorithm
    Outputs:
        out: pandas df, should contain two columns dateCol, regime
            dateCol: column that indexes the date
            regime: column that specifies the regime
    '''
    #Define the returns vector
    ret = data.as_matrix(columns=[nameCol])
    n = np.size(ret)
    x_ret = ret.reshape(n)

    #Define the D matrix
    Dfull = np.diag([1]*n) - np.diag([1]*(n-1),1)
    D = Dfull[0:(n-1),]

    #set up CVX problem
    beta = cp.Variable(n)
    lambd = cp.Parameter(nonneg=True)
    x_ret = ret.reshape(n)

    def tf_obj(x,beta,lambd):
        return cp.norm(x-beta,2)**2 + lambd*cp.norm(cp.matmul(D, beta),1)

    problem = cp.Problem(cp.Minimize(tf_obj(x_ret, beta, lambd)))

    lambd.value = paramLambda
    problem.solve()

    out = pd.DataFrame(data[dateCol], columns=[dateCol])
    out['Regime'] = np.sign(beta.value)

    return out

def parseBetaSeries(data, regimeCol):
    '''parseBetaSeries takes an df object with a regime column, returns a list which indicates the parts in the series for the crash regime
    Inputs:
        data: pd dataframe object, must contain the regimeCol
        regimeCol: string, names the regimeCol in the dataset
    Outputs:
        l: list object, that looks like [start_0, end_0, start_1, end_1...]'''
    
    #Step 1: create an array -2 and 0s where -2 corresponds to a negative beta value
    betas = data[regimeCol]
    betas = betas - 1
    nonZeroArray = betas.nonzero()
    nonZeroArray = nonZeroArray[0]

    #Step 1.1: get the length of the original array
    n = len(betas)

    #Step 2: compute the start and end point of the negative regimes
    l = list()

    if (len(nonZeroArray) == 0): #Check if there are no regimes return a empty list
        return l
    else: #if there are regimes, process them
        start = nonZeroArray[0]
        previous = start
        for i in range(1,len(nonZeroArray)):
            if(previous + 1 != nonZeroArray[i]): #i.e you have hit the end of a range
                l.append(1.0*start/n)
                l.append(1.0*previous/n)
                start = nonZeroArray[i]
                previous = nonZeroArray[i]
            else: #you haven't hit the end of a range
                previous = nonZeroArray[i]
        #add the last range
        l.append(1.0*start/n)
        l.append(1.0*previous/n)
        return l

def regime_switch(betas):
    '''returns list of starting points of each regime'''
    n = len(betas)
    init_points = [0]
    curr_reg = (betas[0]>0)
    for i in range(n):
        if (betas[i]>0) == (not curr_reg):
            curr_reg = not curr_reg
            init_points.append(i)
    init_points.append(n)
    return init_points


def plot_returns_regime(data, factorName, regimeCol, flag='Total Return', date='Date', ymaxvar=None, pathToSavePlot = False):
    '''plot_returns returns a plot of the returns
    INPUTS:
        factorName: string, name of column to be plotted
        data: pd dataframe, where the data is housed
        regimeCol: string, name of the regime column in the pandas df: data
        flag: string, Either Total Return or Monthly Return
        date: string, column name corresponding to the date variable
        pathToSavePlot: if specified, saves the plot at the specific path instead of showing the plot
    Outputs:
        a plot'''
    #Clean Inputs:
    if(date not in data.columns):
        print ('date column not in the pandas df')
        return
    if(factorName not in data.columns):
        print ('column ' + factorName + ' not in pandas df')
        return
    #If the inputs are clean, create the plot

    #Step 1: Parse the regime list
    data = data.sort_values(date).copy()
    data.reset_index(drop=True, inplace=True)
    data[date] = pd.to_datetime(data[date])

    regimelist = regime_switch(data[regimeCol])
    curr_reg = np.sign(data[regimeCol][0])

    #Now create plot
    if (flag == 'Total Return'):
        data['TotalReturns'] = data[factorName] + 1
        data['TotalReturns'] = data['TotalReturns'].cumprod()
        if(ymaxvar is None):
            ymaxvar = data.loc[data.shape[0]-1,'TotalReturns']

        fig, ax = plt.subplots()
        for i in range(len(regimelist)-1):
            if curr_reg == 1:
                ax.axhspan(0, ymaxvar, xmin=regimelist[i]/regimelist[-1], xmax=regimelist[i+1]/regimelist[-1], 
                       facecolor='white', alpha=0.3)
            else:
                ax.axhspan(0, ymaxvar, xmin=regimelist[i]/regimelist[-1], xmax=regimelist[i+1]/regimelist[-1], 
                       facecolor='grey', alpha=0.5)
            curr_reg = -1 * curr_reg

        plt.semilogy(data[date], data['TotalReturns'], color='black')
        plt.title(factorName + ' Total Return Over Time')
        plt.ylabel(factorName)
        plt.xlabel('Date')
        plt.xlim(data[date].iloc[0],data[date].iloc[-1])
        if(pathToSavePlot):
            path = pathToSavePlot + 'RegimeGraph.png'
            plt.savefig(path)
        else:
            plt.show()

    elif (flag == 'Relative Return'):
        ymaxvar = max(0,max(data[factorName]))
        yminvar = min(0,min(data[factorName]))
        lim = max(ymaxvar, -yminvar)

        fig, ax = plt.subplots()
        for i in range(len(regimelist)-1):
            if curr_reg == 1:
                ax.axhspan(-lim, lim, xmin=regimelist[i]/regimelist[-1], xmax=regimelist[i+1]/regimelist[-1], 
                       facecolor='white', alpha=0.3)
            else:
                ax.axhspan(-lim, lim, xmin=regimelist[i]/regimelist[-1], xmax=regimelist[i+1]/regimelist[-1], 
                       facecolor='grey', alpha=0.5)
            curr_reg = -1 * curr_reg

        plt.plot(data[date], data[factorName])
        plt.title(factorName + ' Returns Over Time')
        plt.ylabel(factorName)
        plt.xlabel('Date')
        plt.xlim(data[date].iloc[0],data[date].iloc[-1])
        if(pathToSavePlot):
            path = pathToSavePlot + 'RegimeGraph.png'
            plt.savefig(path)
        else:
            plt.show()
    else:
        print ('flag variable must be either Total Return or Relative Return')


def plot_returns_by_environment(data, factorName, macroCol, date='Date', pathToSavePlot = False):
    '''plot_returns_by_enviornment plots two time series, each defined by holding the asset in a specific macro environment
    INPUTS:
        factorName: string, name of column to be plotted
        data: pd dataframe, where the data is housed
        macroCol: string, name of the regime column in the pandas df.  Should contain 2 unique values.  1) if the return if the macro environment is true, -1 if it is false
        flag: string, Either Total Return or Monthly Return
        date: string, column name corresponding to the date variable
        ymaxvar: optional argument, sets max of the plot
        pathToSavePlot: if specified, saves the plot at the specific path instead of showing the plot
    Outputs:
        a plot'''
    #Clean Inputs:
    if(date not in data.columns):
        print ('date column not in the pandas df')
        return
    if(factorName not in data.columns):
        print ('column ' + factorName + ' not in pandas df')
        return
    #If the inputs are clean, create the plot

    #Step 1: Parse the regime list

    data = data.sort_values(date).copy()
    data.reset_index(drop=True, inplace=True)
    data[date] = pd.to_datetime(data[date])

    regimelist = regime_switch(data[macroCol])
    curr_reg = np.sign(data[macroCol][0])

    #Now create the two time series
    data['AboveMedian'] = data[factorName]*((data[macroCol]+1)/2)
    data['BelowMedian'] = -1*data[factorName]*((data[macroCol]-1)/2)
    data['AboveMedian'] = data['AboveMedian']+1
    data['BelowMedian'] = data['BelowMedian']+1

    data['AboveMedian'] = data['AboveMedian'].cumprod()
    data['BelowMedian'] = data['BelowMedian'].cumprod()

    ymaxvar = max(data['AboveMedian'].max(), data['BelowMedian'].max())
    fig, ax = plt.subplots()
    for i in range(len(regimelist)-1):
        if curr_reg == 1:
            ax.axhspan(0, ymaxvar, xmin=regimelist[i]/regimelist[-1], xmax=regimelist[i+1]/regimelist[-1], 
                   facecolor='white', alpha=0.3)
        else:
            ax.axhspan(0, ymaxvar, xmin=regimelist[i]/regimelist[-1], xmax=regimelist[i+1]/regimelist[-1], 
                facecolor='grey', alpha=0.5)
        curr_reg = -1 * curr_reg

    plt.semilogy(data[date], data['AboveMedian'], color='black')
    plt.semilogy(data[date], data['BelowMedian'], color='yellow')

    plt.title(factorName + ' Total Return Over Time by Regime')
    plt.ylabel(factorName)
    plt.xlabel('Date')
    plt.xlim(data[date].iloc[0],data[date].iloc[-1])
    plt.legend([macroCol+' Up', macroCol+' Down'])


    if(pathToSavePlot):
        path = pathToSavePlot + 'RegimeGraph.png'
        plt.savefig(path)
    else:
        plt.show()



