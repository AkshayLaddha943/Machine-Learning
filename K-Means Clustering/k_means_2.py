# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 08:09:23 2020

@author: Admin
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 14:23:04 2020

@author: Admin
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

Dataframe = pd.read_csv("D:\\Machine Learning_Algoritms\\K-Means Clustering\\cars.csv", encoding = 'latin1', na_values = ' ')
Dataframe.columns = [ "mpg", "cylinders","cubicinches","hp", "weightlbs","time-to-60","year","brand"]

Dataframe["weightlbs"] = Dataframe["weightlbs"].replace(to_replace = np.nan, value = Dataframe["weightlbs"].mean(skipna = True))
Dataframe.isnull().sum()
Dataframe["cubicinches"] = Dataframe["cubicinches"].replace(to_replace = np.nan, value = Dataframe["cubicinches"].mean(skipna = True))
Dataframe.isnull().sum()

#Dataframe = Dataframe.to_numpy()


x = Dataframe


x_array = x.values

from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
x_array[: , -1] = labelencoder_X.fit_transform(x_array[: , -1])


from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(x_array)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11), wcss)
plt.title("Cluster count")
plt.xlabel('cluster range')
plt.ylabel('WCSS')
plt.show()

#Clusters = 3

kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 42)
y_means = kmeans.fit_predict(x_array)

#Visualizing results
plt.scatter(x_array[y_means == 0, 0], x_array[y_means == 0, 1], s = 70, c = 'red', label = 'cluster1')
plt.scatter(x_array[y_means == 1, 0], x_array[y_means == 1, 1], s = 70, c = 'blue', label = 'cluster2')
plt.scatter(x_array[y_means == 2, 0], x_array[y_means == 2, 1], s = 70, c = 'green', label = 'cluster3')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], s = 200, c = 'yellow',label = 'centroid')
plt.title("K-MEANS")
plt.xlabel('Rest factors')
plt.ylabel('Brand')
plt.show()