#Out of sample cross validation stuff

import numpy as np #for numerical array data
import pandas as pd #for tabular data
from scipy.optimize import minimize
import matplotlib.pyplot as plt #for plotting purposes

import cvxpy as cp
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV


def linear_regression(x,y):
    lin_reg = LinearRegression(fit_intercept=True)
    lin_reg.fit(x,y)
    return lin_reg.intercept_, lin_reg.coef_

def cv_LASSO(x,y,n_folds = 10,lambda_max=0.25):
    alpha_max = lambda_max /(2*x.shape[0])
    LASSO_test = Lasso(random_state=7777, fit_intercept=True)
    alphas = np.linspace(1e-6, alpha_max, 100)

    tuned_parameters = [{'alpha': alphas}]
    
    clf = GridSearchCV(LASSO_test, tuned_parameters, cv=n_folds, refit=False) 
    clf.fit(x,y)

    alpha_best = alphas[np.argmax(clf.cv_results_['mean_test_score'])]
    LASSO_best = Lasso(alpha=alpha_best, fit_intercept=True)
    LASSO_best.fit(x,y)
    return LASSO_best.intercept_, LASSO_best.coef_, alpha_best*2*x.shape[0]

def best_subset(x,y,l_0):
    # Mixed Integer Programming in feature selection
    M = 1000
    n_factor = x.shape[1]
    z = cp.Variable(n_factor, boolean=True)
    beta = cp.Variable(n_factor)
    alpha = cp.Variable(1)

    def MIP_obj(x,y,b,a):
        return cp.norm(y-cp.matmul(x,b)-a,2)

    best_subset = cp.Problem(cp.Minimize(MIP_obj(x, y, beta, alpha)), 
                             [cp.sum(z)<=l_0, beta+M*z>=0, M*z>=beta])
    best_subset.solve()
    return alpha.value, beta.value

def oos_error(a,b,x,y,method):
    # a, b are one-dimensional array
    error = np.linalg.norm(y-x.dot(b)-a)
    print('The out-of-sample error of ' + method + ' is: {0:3f}\n'.format(error))
    

def error_analysis(x,y,x2,y2):
    a1, b1 = linear_regression(x,y)
    a2, b2 = best_subset(x,y,2)
    a3, b3 = best_subset(x,y,3)
    a4, b4 = best_subset(x,y,4)
    a5, b5 = best_subset(x,y,5)
    a6, b6, __ = cv_LASSO(x,y)

    oos_error(a1,b1,x2,y2,'OLS')
    oos_error(a2,b2,x2,y2,'best 2-subset')
    oos_error(a3,b3,x2,y2,'best 3-subset')
    oos_error(a4,b4,x2,y2,'best 4-subset')
    oos_error(a5,b5,x2,y2,'best 5-subset')
    oos_error(a6,b6,x2,y2,'LASSO')
    return a6, b6

def error_analysis_complete(data,factornames, assetnames, oos_ratio=0.9, decimal=1):
    '''error_analysis_complete takes in a df object, returns an out of sample test and returns the results
    INPUTS:
        data: pandas df, contains the factors in factornames and assets in assetnames
        factornames: list, elements are strings, constains names of factors
        assetnames: list, elements are strings, constaims names of assets
        oos_ratio: amount of data used to test out of sample, kept in order of data.  So if oos_ratio=.9 last 10% of data held for out of sample test
    Outputs:
        printed output
    '''
    X = data[factornames].copy()
    X = X.values
    Y = data[assetnames].copy()
    Y = Y.values

    n_time, n_factor = X.shape
    _, n_asset = Y.shape
    
    n_is = int(n_time*oos_ratio)
    X_test = X[:n_is,:]
    X_oos = X[n_is:,:]
    
    betas_LASSO = np.zeros((n_factor+1,n_asset))
    
    for i in range(n_asset):
        print('For asset: ' + assetnames[i] + '\n')
        y_test = Y[:n_is,i]
        y_oos = Y[n_is:,i]
        a,b = error_analysis(X_test,y_test,X_oos,y_oos)
        betas_LASSO[0,i] = a*100
        betas_LASSO[1:,i] = b
        
    print ('Best CV Lasso for Each Asset')
    return pd.DataFrame(np.round_(betas_LASSO, decimals=decimal), index=['Intercept (%)']+factornames, 
                        columns=assetnames, dtype=None, copy=False)