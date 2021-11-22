# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 18:24:53 2020

@author: Admin
"""

import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

tf.__version__

dataframe = pd.read_csv("D:\\Machine Learning_Algoritms\\ANN\\Churn_Modelling.csv", encoding = "latin1")
X = dataframe.iloc[:, 3:-1].values
Y = dataframe.iloc[: , -1].values


label_encod = LabelEncoder()
X[: , 1] = label_encod.fit_transform(X[: , 1])

label_encod = LabelEncoder()
X[: , 2] = label_encod.fit_transform(X[: , 2])

onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

sc_X = StandardScaler()
x_train = sc_X.fit_transform(x_train)
x_test = sc_X.transform(x_test)


ann = tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units = 10 , activation = 'relu'))
ann.add(tf.keras.layers.Dense(units = 10 , activation = 'relu'))
ann.add(tf.keras.layers.Dense(units  = 1 , activation = 'sigmoid'))

ann.compile(optimizer = 'adam' , loss = 'binary_crossentropy' , metrics = ['accuracy'])

ann.fit(x_train , y_train, batch_size = 32 ,  epochs = 70)
y_pred = ann.predict(sc_X.transform(x_test))
y_pred = np.round(y_pred)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
#from sklearn.metrics import classification_report
#Cm_3 = classification_report(y_test, y_pred)
CM = confusion_matrix(y_test, y_pred)
AC = accuracy_score(y_test, y_pred)
print (CM)
print ((AC) * 100, '%')