# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 14:21:27 2020

@author: Admin
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re                                  #specifies a string of patterns that matches it                      
import nltk                                #to remove all the redundant stopwords (sice, they are not helpful to predict a review)
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from nltk.stem.porter import PorterStemmer #Stemming refers to taking only the root of the word that indicates enough about the sentence

dataframe = pd.read_csv("D:\\Machine Learning_Algoritms\\NLP\\Restaurant_Reviews.csv", encoding = "latin1")

#Cleaning the texts
corpus = []

for i in range(0, 1000):                                             #Since, we have 1000 reviews in our dataset
    review = re.sub('[^ a-zA-Z]', ' ', dataframe['Review'][i])       # Replace all the strings which do not consists of alphabets with blank space
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
    review = ' '.join(review)
    corpus.append(review)
    
#Creating Bag of words model
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataframe.iloc[:, -1].values

#Splitting the dataset into training and testing
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Training with Naive Bayes Model
model = GaussianNB()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

#Training with Logistic Regression
model_2 = LogisticRegression(random_state = 0)
model_2.fit(x_train, y_train)
y_pred_2 = model_2.predict(x_test)

#Training with SVM
model_3 = SVC(kernel = 'linear',random_state = 0)
model_3.fit(x_train, y_train)
y_pred_3 = model_3.predict(x_test)

#Confusion Matrix
cm = confusion_matrix(y_test, y_pred_3)
ac = accuracy_score(y_test, y_pred_3)
print(cm)
print(ac * 100)