# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 03:41:00 2018

@author: Karthik Bhat
"""

import pandas as pd
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

dataset = pd.read_csv('C:\\Study\\Machine Learning A-Z\\Part 2 - Regression\\Section 7 - Support Vector Regression (SVR)\\Position_Salaries.csv')

X = dataset.iloc[:,1:2].values
y = dataset.iloc[:, -1:].values
stdX = StandardScaler()
stdY = StandardScaler()
X = stdX.fit_transform(X)
y = stdY.fit_transform(y)

reg = SVR(kernel='rbf')
reg.fit(X,y)
stdY.inverse_transform(reg.predict(stdX.transform(6.5)))


plt.scatter(X,y)
plt.plot(X,reg.predict(X))
plt.show()

