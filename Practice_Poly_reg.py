# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 12:10:00 2018

@author: Karthik Bhat
"""
#Importing Libraries:
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

Import_Data = pd.read_csv('C:\\Study\\Machine Learning A-Z\\Part 2 - Regression\\Section 6 - Polynomial Regression\\Position_Salaries.csv')

X = Import_Data.iloc[:,1:2].values
y = Import_Data.iloc[:,2].values 

#Linear Regression:
LR = LinearRegression()
LR.fit(X,y)


#Poly Regression:
PR = PolynomialFeatures(degree = 4)
X_poly = PR.fit_transform(X)
LR2 = LinearRegression()
LR2.fit(X_poly,y)

#Comparing with graphs:

plt.scatter(X,y, color = 'red')
plt.plot(X,LR.predict(X),color='blue')
plt.show()

plt.scatter(X,y, color = 'red')
plt.plot(X,LR2.predict(X_poly),'blue')
plt.show()








