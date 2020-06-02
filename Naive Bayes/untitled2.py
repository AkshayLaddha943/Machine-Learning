# -*- coding: utf-8 -*-
"""
Created on Wed May  6 16:45:09 2020

@author: Admin
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

Dataframe  = pd.read_csv("D:\\Machine Learning_Algoritms\\Naive Bayes\\adult.csv", encoding = 'latin1')
Dataframe[Dataframe == '?'] = np.nan
x_data=Dataframe.drop('income',axis=1)
#X = Dataframe.iloc[:, :-1].values
y = Dataframe.iloc[:, -1].values
#Dataframe  = Dataframe.replace(to_replace = "?", value = 'nan')

Dataframe.isnull().sum()

#convert null data to string for encoding
x_data[x_data.workclass.isnull()]
x_data["workclass"].fillna("", inplace = True) 
x_data["occupation"].fillna("", inplace = True)
x_data["native.country"].fillna("", inplace = True)

#encode categorical column using .cat.codes
x_data["workclass"]=x_data["workclass"].astype("category").cat.codes
x_data["occupation"]=x_data["occupation"].astype("category").cat.codes
x_data["native.country"]=x_data["native.country"].astype("category").cat.codes
x_data["education"]=x_data["education"].astype("category").cat.codes
x_data["marital.status"]=x_data["marital.status"].astype("category").cat.codes
x_data["relationship"]=x_data["relationship"].astype("category").cat.codes
x_data["race"]=x_data["race"].astype("category").cat.codes
x_data["sex"]=x_data["sex"].astype("category").cat.codes

#Note:missing data is now assigned with 0 after encoding
#apply mean on columns with missing data
from sklearn.preprocessing import Imputer
imp = Imputer(missing_values=0, strategy='mean', axis=0)
x_data['workclass']=imp.fit_transform(x_data[['workclass']])
x_data['occupation']=imp.fit_transform(x_data[['occupation']])
x_data['native.country']=imp.fit_transform(x_data[['native.country']])

#for col in  Dataframe[['workclass', 'occupation', 'native.country']]:
    #Mode = Dataframe[col].mode()
    #if col == 'nan':
#Dataframe['workclass'] = Dataframe['workclass'].fillna("Private", inplace=True)
