# -*- coding: utf-8 -*-
"""
Created on Sun May 13 19:27:47 2018

@author: Karthik Bhat
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('C:\\Study\\Machine Learning A-Z\\Part 3 - Classification\\Section 20 - Random Forest Classification\\Social_Network_Ads.csv')
X = dataset.iloc[:,2:4].values
y = dataset.iloc[:,-1:].values

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y)

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=350, criterion="entropy")
classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)









