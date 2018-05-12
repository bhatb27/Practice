# -*- coding: utf-8 -*-
"""
Created on Wed May  2 12:10:10 2018

@author: Karthik Bhat
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix

dataset = pd.read_csv('C:\\Study\\Machine Learning A-Z\\Part 3 - Classification\\Section 14 - Logistic Regression\\Social_Network_Ads.csv')

X = dataset.iloc[:,2:4].values
y = dataset.iloc[:,-1:].values

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.25,random_state = 0)

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

clasi = LogisticRegression()
clasi.fit(X_train,y_train)

y_pred = clasi.predict(X_test)

cm = confusion_matrix(y_test,y_pred)



plt.scatter()




