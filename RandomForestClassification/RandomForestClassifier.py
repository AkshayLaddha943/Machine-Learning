# -*- coding: utf-8 -*-
"""
Created on Tue May 12 17:04:24 2020

@author: Admin
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

Dataframe = pd.read_csv("D:\\Machine Learning_Algoritms\\RandomForestClassification\\StudentsPerformance.csv", encoding = 'latin1')
X = Dataframe.drop('test preparation course' , axis = 1)
Y = Dataframe['test preparation course']

Dataframe.shape
Dataframe.isnull().sum()

Dataframe['test preparation course'].value_counts()

pd.crosstab(Dataframe.gender,Dataframe.lunch).plot(kind='bar')
plt.title('Exam')
plt.xlabel('gender')
plt.ylabel('lunch')
plt.show()

X1 = pd.get_dummies(X['gender'])
X = pd.concat([X, X1], axis = 1)
X.head()
X = X.drop('gender', axis = 1)

Dataframe['race/ethnicity'].value_counts()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size = 0.2, random_state = 0)


X_train = X_train.drop('race/ethnicity', axis = 1)
X_train = X_train.drop('parental level of education', axis = 1)
X_train = X_train.drop('lunch', axis = 1)

X_test = X_test.drop('race/ethnicity', axis = 1)
X_test = X_test.drop('parental level of education', axis = 1)
X_test = X_test.drop('lunch', axis = 1)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

from sklearn.ensemble import RandomForestClassifier
RFC = RandomForestClassifier(n_estimators = 10, random_state = 0)
RFC.fit(X_train, y_train)

y_pred = RFC.predict(X_test)
y_pred_2 = RFC.predict(X_train)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
Cm_3 = classification_report(y_train, y_pred_2)
CM = confusion_matrix(y_train, y_pred_2)
AC = accuracy_score(y_train, y_pred_2)
print (CM)
print ((AC) * 100, '%')


#N = Dataframe[Dataframe.gender == "female"]
#C = Dataframe[Dataframe.gender == "male"]
# scatter plot
#plt.scatter(N.lunch,N.lunch, color="red",label="kotu",alpha= 0.3)
#plt.scatter(C.lunch,C.lunch, color="green",label="iyi",alpha= 0.3)
#plt.xlabel("radius_mean")
#plt.ylabel("texture_mean")
#plt.legend()
#plt.show()