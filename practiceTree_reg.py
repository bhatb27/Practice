# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 18:30:55 2018

@author: Karthik Bhat
"""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

dataset = pd.read_csv('C:\\Study\\Machine Learning A-Z\\Part 2 - Regression\\Section 8 - Decision Tree Regression\\Position_Salaries.csv')

X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:,-1:].values

regressor = DecisionTreeRegressor()

reg = regressor.fit(X,y)

X_grid = np.arange(min(X),max(X),0.01)
X_grid = X_grid.reshape((len(X_grid),1))

plt.scatter(X,y)
plt.plot(X_grid,reg.predict(X_grid),color='red')
plt.show()




