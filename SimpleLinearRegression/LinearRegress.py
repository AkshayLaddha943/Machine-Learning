# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 10:17:49 2020

@author: Admin
"""


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

Dataframe_2  = pd.read_csv("D:\\Machine Learning_Algoritms\\SimpleLinearRegression\\Salary_Data.csv", encoding = 'latin1')
X1 = Dataframe_2.iloc[:, :-1]
y1 = Dataframe_2.iloc[:, -1]

from sklearn.model_selection import train_test_split
X_train1,X_test1,y_train1,y_test1 = train_test_split(X1, y1, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train1 = sc_X.fit_transform(X_train1)
X_test1 = sc_X.transform(X_test1)

from sklearn.linear_model import LinearRegression 
regressors = LinearRegression()
regressors.fit(X_train1, y_train1)



y_pred = regressors.predict(X_test1)


plt.scatter(X_test1, y_test1, color = 'red')
plt.plot(X_test1, regressors.predict(X_test1), color = 'blue')
plt.title('Age vs Salary')
plt.xlabel('Age')
plt.ylabel('Salary')
plt.plot()