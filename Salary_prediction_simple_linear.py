#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 21:32:52 2020

@author: vishvmac
"""

# Tutorial 1 (Linear regression)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the Dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:,0].values
Y = dataset.iloc[:,1].values



regressor=LinearRegression()

X=X.reshape(-1,1)
regressor.fit(X,Y)

# Splitting the dataset into training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.33, random_state = 0)

# Fitting multiole linear regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the test set results
y_pred = regressor.predict(X_test)

# Visulizing the training ser results
plt.scatter(X_train,y_train, color ='blue')
plt.plot(X_train,regressor.predict(X_train), color = 'red' )
plt.title('salary vs experience(training set)')
plt.xlabel('years of experience')
plt.ylabel('salary')
plt.show()
