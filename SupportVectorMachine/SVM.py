# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 10:44:19 2020

@author: Admin
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

Dataframe  = pd.read_csv("D:\\Machine Learning_Algoritms\\SupportVectorMachine\\cell_samples.csv", encoding = 'latin1')
Dataframe = Dataframe.drop(['ID'], axis = 1)

X = Dataframe.iloc[:, :-1].values
y = Dataframe.iloc[:, -1].values

Dataframe.groupby('Class').size()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.25, random_state = 0)



from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = np.nan, strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 5:6])
X[:, 5:6] = imputer.transform(X[:, 5:6])


from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn import metrics
Cm = metrics.confusion_matrix(y_test,y_pred)
Cm_2 = metrics.classification_report(y_test, y_pred)
Cm_3 = metrics.accuracy_score(y_test, y_pred)

from sklearn.datasets.samples_generator import make_blobs
from matplotlib.colors import ListedColormap
X_train, y_train = make_blobs(n_samples=90, centers=2,
                  random_state=0, cluster_std=0.60)
for i, j in enumerate(np.unique(y_train)):
        plt.scatter(X_train[y_train == j, 1],X_train[y_train == j ,0],  c = ListedColormap(('red', 'green'))(i) ,s = 50,cmap = 'autumn', label = j)

plt.scatter(X_train[:, 0], X_train[:, 1], c = y_train)
# Create the hyperplane
w = classifier.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(-0.5, 5.0)
yy = a * xx - (classifier.intercept_[0]) / w[1]

plt.plot(xx, yy)
plt.show()
