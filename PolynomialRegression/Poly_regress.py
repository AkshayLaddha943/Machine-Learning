# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 22:06:37 2020

@author: Admin
"""


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

Dataframe_2  = pd.read_csv("D:\\Machine Learning_Algoritms\\PolynomialRegression\\Position_Salaries.csv", encoding = 'latin1')
X1 = Dataframe_2.iloc[:, 1:2]
y1 = Dataframe_2.iloc[:, 2]

# Splitting the dataset into the Training set and Test set
"""from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""

# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X1, y1)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 5)
X_poly = poly_reg.fit_transform(X1)
poly_reg.fit(X_poly, y1)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y1)

# Visualising the Linear Regression results
plt.scatter(X1, y1, color = 'red')
plt.plot(X1, lin_reg.predict(X1), color = 'blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualising the Polynomial Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualising the Polynomial Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
