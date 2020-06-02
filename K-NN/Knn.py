# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 19:39:34 2020

@author: Admin
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


Dataframe  = pd.read_csv("D:\\Machine Learning_Algoritms\\K-NN\\Iris.csv", encoding = 'latin1')
Dataframe = Dataframe.drop(['Id'], axis = 1)
X = Dataframe.iloc[:, :-1].values
y = Dataframe.iloc[:, -1].values

#One-hot_encoding of y
#from sklearn import preprocessing
#y = preprocessing.label_binarize(y, classes=[0, 1, 2])

Dataframe.shape

Dataframe.describe()

Dataframe.groupby('Species').size()

#Encoding the dependent variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
label_X = LabelEncoder()
label_X.fit_transform(y)
onehotencoder = OneHotEncoder(categorical_features=[0])
y = onehotencoder.fit_transform(y).toarray()


#Splitting dataset into Training and Testing set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.25, random_state = 0)

from pandas.plotting import parallel_coordinates
plt.figure(figsize = (15,10))
parallel_coordinates(Dataframe, "Species")
plt.title("Parallel Coordinates Plot")
plt.xlabel("Independent")
plt.ylabel("dependent")
plt.legend(loc = 1, prop = {'size' : 15})
plt.show()

import seaborn as sns
plt.figure()
sns.set(style = 'ticks')
sns.pairplot(Dataframe, hue = "Species", size = 3)
plt.show()

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski',p = 2)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn import metrics
Cm = metrics.confusion_matrix(y_test,y_pred)
Cm_2 = metrics.classification_report(y_test, y_pred)
Cm_3 = metrics.accuracy_score(y_test, y_pred)

error_rate = []
for k in range(1,40):
    knn = KNeighborsClassifier(n_neighbors = k, metric = 'minkowski', p = 2)
    knn.fit(X_train, y_train)
    pred = knn.predict(X_test)
    error_rate.append(np.mean(pred != y_test))

plt.figure(figsize = (10,6))
plt.plot(range(1,40), error_rate, linestyle = 'dashed', color = 'blue', marker = 'o')
plt.title("True vs Predicted")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

classifier_2 = KNeighborsClassifier(n_neighbors = 28, metric = 'minkowski',p = 2)
classifier_2.fit(X_train, y_train)

y_pred_2 = classifier_2.predict(X_test)

Cm_3 = metrics.classification_report(y_test, y_pred_2)
Cm_4 = metrics.accuracy_score(y_test, y_pred_2)

from sklearn.metrics import roc_curve, auc
y_pred_proba = classifier.predict_proba(X_train)[:,1]
fpr, tpr, threshold = roc_curve(y_train,  y_pred_proba)
auc_logit = auc(y_train, y_pred_proba)
plt.figure(figsize =(5,5), dpi = 100)
plt.plot(fpr,tpr,linestyle = '-' )
plt.xtitle("False Positive Rate")
plt.ytitle("True Positive Rate")

from sklearn.datasets.samples_generator import make_blobs
from matplotlib.colors import ListedColormap
X_train, y_train = make_blobs(n_samples=100, centers=2,
                  random_state=0, cluster_std=0.60)
for i, j in enumerate(np.unique(y_train)):
        plt.scatter(X_train[y_train == j, 0],X_train[y_train == j ,1],  c = ListedColormap(('red', 'green'))(i) ,s = 50,cmap = 'autumn', label = j)