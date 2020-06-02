# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 00:01:04 2020

@author: Admin
"""


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

Dataframe  = pd.read_csv("D:\\Machine Learning_Algoritms\\LogisticRegression\\clean_diabetes.csv", encoding = 'latin1')
X = Dataframe.iloc[:, :-1]
y = Dataframe.iloc[:, -1]

Dataframe['Outcome'].value_counts()

sns.countplot(x = 'Outcome', data = Dataframe)
plt.show()

count_1 = len(Dataframe[Dataframe['Outcome'] == 1])
count_2 = len(Dataframe[Dataframe['Outcome'] == 0])
total_1 = count_1/ (count_1 + count_2)
print(total_1 *100) 
total_2 = count_2/ (count_1 + count_2)
print(total_2 * 100)



pd.crosstab(Dataframe.Glucose, Dataframe.Outcome).plot(kind ='bar')
plt.title('Health')
plt.xlabel('Glucose')
plt.ylabel('Outcome')
plt.show()


pd.crosstab(Dataframe.BMI, Dataframe.Outcome).plot(kind ='bar')
plt.title('Health')
plt.xlabel('Glucose')
plt.ylabel('Outcome')
plt.show()

pd.crosstab(Dataframe.Age, Dataframe.Outcome).plot(kind ='bar')
plt.title('Health')
plt.xlabel('Glucose')
plt.ylabel('Outcome')
plt.show()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.25, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#Fitting Logistic Regression to training dataset
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

#Predicting the test set results
y_pred = classifier.predict(X_test)

#Creating a confusion matrix
from sklearn.metrics import confusion_matrix
Cm  = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots()
sns.heatmap(pd.DataFrame(Cm), annot=True)
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

from sklearn.metrics import classification_report
Cm_2 = classification_report(y_test, y_pred)

import statsmodels.api as sm
logit_model=sm.Logit(y,X)
result=logit_model.fit()
result.summary2()

from sklearn import metrics
y_pred_proba = classifier.predict_proba(X_test)[:,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()

