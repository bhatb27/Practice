# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 20:32:16 2018

@author: Karthik Bhat
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

dataset = pd.read_csv('C:\\Study\\Machine Learning A-Z\\Part 2 - Regression\\Section 9 - Random Forest Regression\\Position_Salaries.csv')


abc = dataset[2,1:3]
X = dataset.iloc[:,1:2].values
print (dataset)
ztest = dataset.iloc[:,2].values

y = dataset.iloc[:,-1:].values

reg = RandomForestRegressor(n_estimators=10)
reg.fit(X,y)

reg.predict(7.3)

X_grid = np.arange(min(X),max(X),0.01)
X_grid = X_grid.reshape((len(X_grid), 1))

plt.scatter(X,y)
plt.plot(X_grid,reg.predict(X_grid))
plt.show()







