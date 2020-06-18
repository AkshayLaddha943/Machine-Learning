# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 01:35:11 2020

@author: Admin
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df = pd.read_csv('D:\\Machine Learning_Algoritms\\Youtube-Spam-Check\\Youtube01-Psy.csv', encoding ='latin1')
X = df.drop('CLASS' , axis = 1)
Y = df['CLASS']

pd.crosstab(df.AUTHOR,df.CLASS).plot(kind='bar')
plt.title('relation')
plt.xlabel('AUTHOR')
plt.ylabel('CLASS')
plt.show()

df['CLASS'].value_counts()

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()

dv = vectorizer.fit_transform(df['CONTENT'])

dshuf = df.sample(frac = 1)

dtrain = dshuf[:300]
dtest = dshuf[300:]
dtrain_att = vectorizer.fit_transform(dtrain['CONTENT']) #fit bag-of-words model to training set
dtest_att = vectorizer.transform(dtest['CONTENT']) 

dtrain_label = dtrain['CLASS']
dtest_label = dtest['CLASS']

from sklearn.ensemble import RandomForestClassifier
RFC = RandomForestClassifier(n_estimators = 80, random_state = 0)
RFC.fit(dtrain_att, dtrain_label)

RFC.score(dtrain_att, dtrain_label)

from sklearn.metrics import confusion_matrix
y_pred = RFC.predict(dtrain_att)
confusion_matrix(y_pred, dtrain_label)

from sklearn.model_selection import cross_val_score
scores = cross_val_score(RFC, dtrain_att, dtrain_label, cv=5)
scores.mean()

df = pd.concat([pd.read_csv('D:\\Machine Learning_Algoritms\\Youtube-Spam-Check\\Youtube01-Psy.csv', encoding ='latin1'),
                pd.read_csv('D:\\Machine Learning_Algoritms\\Youtube-Spam-Check\\Youtube02-KatyPerry.csv', encoding ='latin1'),
                pd.read_csv('D:\\Machine Learning_Algoritms\\Youtube-Spam-Check\\Youtube03-LMFAO.csv', encoding ='latin1'),
                pd.read_csv('D:\\Machine Learning_Algoritms\\Youtube-Spam-Check\\Youtube04-Eminem.csv', encoding ='latin1'),
                pd.read_csv('D:\\Machine Learning_Algoritms\\Youtube-Spam-Check\\Youtube05-Shakira.csv', encoding ='latin1')])

len(df)

df['CLASS'].value_counts()

dshuf = df.sample(frac=1)
d_content = dshuf['CONTENT']
d_label = dshuf['CLASS']

from sklearn.pipeline import Pipeline,make_pipeline
pl = Pipeline([
        ('bag of words: ', CountVectorizer()),
        ('Random Forest Classifier:', RandomForestClassifier())])

make_pipeline(CountVectorizer(), RandomForestClassifier())

pl.fit(d_content[:1500],d_label[:1500])

pl.score(d_content[:1500],d_label[:1500])

scores = cross_val_score(pl, d_content, d_label, cv=5)
scores.mean()

from sklearn.feature_extraction.text import TfidfTransformer
pl_2 = make_pipeline(CountVectorizer(),
                     TfidfTransformer(norm=None),
                     RandomForestClassifier())

scores = cross_val_score(pl_2, d_content, d_label, cv=5)
scores.mean()



