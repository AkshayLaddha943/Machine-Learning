# -*- coding: utf-8 -*-
"""
Created on Fri May  8 17:31:07 2020

@author: Admin
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

Dataframe  = pd.read_csv("D:\\Machine Learning_Algoritms\\DecisionTreeClassification\\train.csv", encoding = 'latin1')

X = Dataframe.iloc[:, :-1]
y = Dataframe.iloc[:, -1]

ax = X["Age"].hist(bins=15, alpha=0.8)
ax.set(xlabel='Age', ylabel='Count')
plt.show()

X['Age'].mean(skipna = True)

X['Age']  = X['Age'].replace(to_replace = np.nan, value = int(29.699))

X.isnull().sum().any()

y.isnull().sum()

sns.countplot(y = 'Embarked', data = Dataframe)
plt.show()

y = y.replace(to_replace = np.nan, value = 'S')

y.isnull().sum().any()

pd.crosstab(Dataframe.Cabin,Dataframe.Embarked).plot(kind='line')
plt.title('titanic test')
plt.xlabel('Cabin')
plt.ylabel('Embarked')
plt.show()

X = X.drop('Cabin' , axis = 1)
X = X.drop('Name' , axis = 1)

pd.crosstab(Dataframe.Ticket,Dataframe.Embarked).plot(kind='bar')
plt.title('titanic test')
plt.xlabel('Fare')
plt.ylabel('Embarked')
plt.show()

X = X.drop('Ticket' , axis = 1)

X1 = pd.get_dummies(X['Sex'])
X = pd.concat([X, X1], axis = 1)
X.head()
X.drop('Sex', axis = 1)

sns.barplot('Survived', 'Embarked', data = Dataframe)
plt.show()


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)


X_train = X_train.drop('Sex', axis = 1)

X_test = X_test.drop('Sex', axis = 1)

from sklearn.tree import DecisionTreeClassifier
DTC = DecisionTreeClassifier(random_state = 0, criterion = 'entropy' , max_depth = 3)
DTC.fit(X_train, y_train)

y_pred = DTC.predict(X_test)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
CM = confusion_matrix(y_test, y_pred)
AC = accuracy_score(y_test, y_pred)
print (CM)
print ((AC) * 100)

#pip install graphviz


#from sklearn import tree
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  
from IPython.display import Image
import pydotplus
#tree1_view = tree.export_graphviz(DTC, out_file=None, feature_names = X_train.columns.values, rotate=True) 
dot_data = StringIO()
export_graphviz(DTC,out_file=dot_data)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())

from sklearn.tree import DecisionTreeClassifier
DTC = DecisionTreeClassifier(criterion = "gini",random_state = 0)
DTC.fit(X_train, y_train)

y_pred = DTC.predict(X_test)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
CM = confusion_matrix(y_test, y_pred)
AC = accuracy_score(y_test, y_pred)
print (CM)
print ((AC) * 100)

criteria = ["gini", "entropy"]
accuracy = []
for a in criteria:
    DTC = DecisionTreeClassifier(criterion = a, random_state = 0)
    DTC.fit(X_train, y_train)
    y_pred = DTC.predict(X_test)
    accuracy.append(accuracy_score(y_test, y_pred))
    
accuracy

#from sklearn.metrics import roc_curve, auc
#DTC.fit(X3, Y3)
#y_test = Y3_test
#X_test = X3_test
 
# Determine the false positive and true positive rates
#y_proba = DTC.predict_proba(X_test)[:,1]
#FPR, TPR, _ = roc_curve(y_test,y_proba)
from sklearn import preprocessing
def multiclass_roc_auc_score(y_test, y_pred, average="macro"):
       lb = preprocessing.LabelBinarizer()
       lb.fit(y_test)
       y_test = lb.transform(y_test)
       y_pred = lb.transform(y_pred)
       return multiclass_roc_auc_score(y_test, y_pred, average=average)
# Calculate the AUC
Y1 = pd.get_dummies(y_train)
y_train = pd.concat([y, Y1], axis = 1)
y_train.head()
y_train = y_train.drop('Embarked', axis = 1)

y_train['C']  = y_train['C'].replace(to_replace = np.nan, value = 0)
y_train['Q']  = y_train['Q'].replace(to_replace = np.nan, value = 0)
y_train['S']  = y_train['S'].replace(to_replace = np.nan, value = 1)


y_test = pd.concat([y, Y1], axis = 1)
y_test.head()
y_test = y_test.drop('Embarked', axis = 1)

y_test['C']  = y_test['C'].replace(to_replace = np.nan, value = 0)
y_test['Q']  = y_test['Q'].replace(to_replace = np.nan, value = 0)
y_test['S']  = y_test['S'].replace(to_replace = np.nan, value = 1)

y_pred = pd.concat([y, Y1], axis = 1)
y_pred.head()
y_pred = y_pred.drop('Embarked', axis = 1)

y_pred['C']  = y_pred['C'].replace(to_replace = np.nan, value = 0)
y_pred['Q']  = y_pred['Q'].replace(to_replace = np.nan, value = 0)
y_pred['S']  = y_pred['S'].replace(to_replace = np.nan, value = 1)

CM = confusion_matrix(y_test, y_pred)
AC = accuracy_score(y_test, y_pred)
print (CM)
print ((AC))

roc_auc = auc(y_test, DTC.predict_proba(y_pred))
#print ('ROC AUC: %0.3f' % roc_auc )

plt.figure(figsize = (10,10))
plt.plot(y_test,DTC.predict_proba(y_pred), label='ROC curve (area = %0.3f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([-0.05, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (Test Sample Performance)')
plt.legend(loc="lower right")
plt.show()