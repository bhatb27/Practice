# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 14:51:00 2018

@author: Karthik Bhat
"""

#Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Importing dataset
dataset = pd.read_csv('C:\\Study\\Machine Learning A-Z\\Part 2 - Regression\\Section 4 - Simple Linear Regression\\Salary_Data.csv')
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,1].values

from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 1/3, random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

#predicting
y_pred = regressor.predict(x_test)

plt.scatter(x_train,y_train,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title('Salary Vs Exp')
plt.xlabel('exp')
plt.ylabel('salary')
plt.show()