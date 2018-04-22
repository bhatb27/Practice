# -*- coding: utf-8 -*-
"""
Created on Sat Apr 21 18:25:29 2018

@author: Karthik Bhat
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.svm import SVR

Dataset = pd.read_csv('C:\\Study\\Machine Learning A-Z\\Part 2 - Regression\\Section 5 - Multiple Linear Regression\\50_Startups.csv')


X = Dataset.iloc[:,:-1].values
y = Dataset.iloc[:,4].values

X = pd.get_dummies(Dataset, columns=['State'], drop_first=True)
X = X.drop(['Profit'], axis=1)
X = X.iloc[:,:].values
X = np.append(np.ones([50,1]).astype('int'), values=X, axis=1)

reg = SVR(kernel='linear')
 
sel = RFE(reg,2,step=1)
 
selector = sel.fit(X,y)
selector.support_
selector.ranking_

pred = selector.predict(X)
dif=np.sum(np.sqrt(np.square(pred2-y)))

import statsmodels.formula.api as sm
X_OLS = X[:,(0,1,3)]
X_sel = sm.OLS(endog = y,exog = X_OLS).fit()
X_sel.summary()
pred2 = X_sel.predict(X_OLS)

from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR_fit = LR.fit(X_OLS,y)
pred1 = LR_fit.predict(X_OLS)


