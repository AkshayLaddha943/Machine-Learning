# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 18:00:04 2020

@author: Admin
"""

#!pip install apyori

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

Dataframe = pd.read_csv("D:\\Machine Learning_Algoritms\\Apriori\\GroceryStoreDataSet.csv", encoding = 'latin1',names = ['products'], header = None)
num_records = len(Dataframe)
print(num_records)

transactions = []
for i in range(0,num_records):
    transactions.append([str(Dataframe.values[i,j]) for j in range(0,3)])
    
Dataframe = list(Dataframe["products"].apply(lambda x:x.split(',')))
    
from mlxtend.preprocessing import TransactionEncoder
te = TransactionEncoder()
te_data = te.fit_transform(Dataframe)

Dataframe = pd.DataFrame(te_data, columns = te.columns_)

count = Dataframe.loc[:,:].sum()
reverse_count = count.sort_values(0, ascending = False).head(11)
reverse_count = reverse_count.to_frame()
reverse_count = reverse_count.reset_index()
#reverse_count = reverse_count.rename(columns = {“index”: “items” ,0: “count”})

plt.style.available

plt.rcParams['figure.figsize'] = (10, 6)
plt.style.use('dark_background')
ax = reverse_count.plot.barh()
plt.title("Popular items")
plt.gca().invert_yaxis()

from mlxtend.frequent_patterns import apriori

df1 = apriori(Dataframe,min_support=0.03, use_colnames=True)

from mlxtend.frequent_patterns import association_rules

rules = association_rules(df1, metric = 'lift', min_threshold = 1)
rules