# -*- coding: utf-8 -*-
"""
Created on Mon May  7 01:09:12 2018

@author: Karthik Bhat
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('C:\\Study\\Machine Learning A-Z\\Part 3 - Classification\\Section 16 - Support Vector Machine (SVM)\\Social_Network_Ads.csv')
X = dataset.iloc[:,[2,3]].values
y = dataset.iloc[:,-1:].values

from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y)

#feature scaling is required for SVM
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#SVM
from sklearn.svm import SVC
classifier = SVC(kernel="linear")
classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)







