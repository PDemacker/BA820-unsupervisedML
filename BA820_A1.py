#imports
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.cluster.hierarchy import fcluster

from sklearn import metrics 
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import silhouette_samples, silhouette_score

import scikitplot as skplt


# get the data set
stocks = pd.read_csv('stock-fundamentals.csv')
stocks.head(3)

# let's take a look at the data set
stocks.info()

# drop the unnessary columns
stocks.drop(columns=['Quarter end', 'ticker', 'quarter_end'], inplace=True)

# replace missing values with the mean and lower case column names
stocks = stocks.fillna(stocks.mean())
stocks.columns = stocks.columns.str.lower()

print(stocks)

# double checking if there are any missing values 
print(stocks.isna().sum().sum())

# first, I am going to scale the data given the varying units of measurement
sc = StandardScaler()
sm = sc.fit_transform(stocks)

sm = pd.DataFrame(sm, columns=stocks.columns)

# confirm the changes
sm.head(3)

# Hierarchical Clustering - first attempt
# going to do euclidean, cosine, jaccard, cityblock distance
diste = pdist(sm.values)
distc = pdist(sm.values, metric="cosine")
distj = pdist(sm.values, metric="jaccard")
distm = pdist(sm.values, metric="cityblock")

# put all on the same linkage to compare
hclust_e = linkage(diste)
hclust_c = linkage(distc)
hclust_j = linkage(distj)
hclust_m = linkage(distm)

# plots
LINKS = [hclust_e, hclust_c, hclust_j,hclust_m]
TITLE = ['Euclidean', 'Cosine', 'Jaccard', 'Manhattan']

plt.figure(figsize=(15, 5))

# loop and build our plot
for i, m in enumerate(LINKS):
  plt.subplot(1, 4, i+1)
  plt.title(TITLE[i])
  dendrogram(m,
             leaf_rotation=90,
             orientation="left")
  
plt.show()

#cosine - more "clusters" visible
METHODS = ['single', 'complete', 'average', 'ward']
plt.figure(figsize=(20,5))


# loop and build our plot for the different methods
for i, m in enumerate(METHODS):
  plt.subplot(1, 4, i+1)
  plt.title(m)
  dendrogram(linkage(distc, method=m), 
             leaf_rotation=90)
  
plt.show()


# using cosine + complete
# the labels with 7 clusters
labs = fcluster(linkage(distc, method="complete"), 7, criterion="maxclust")

# confirm
np.unique(labs)

# add a cluster column to the stocks data set
stocks['cluster'] = labs
print(stocks.head(3))

print(stocks.cluster.value_counts(dropna=False, sort=False))

# cluster solution
clus_profile = stocks.groupby("cluster").mean()

clus_profile.T

# heatmap plot of the clusters with normalized data
scp = StandardScaler()
cluster_scaled = scp.fit_transform(clus_profile)

cluster_scaled = pd.DataFrame(cluster_scaled, 
                              index=clus_profile.index, 
                              columns=clus_profile.columns)

sns.heatmap(cluster_scaled, cmap="Blues", center=0)
plt.show()

# findings
## It's easy to see that some clusters have higher averages in various metrics than others

# another method: KMeans 
xs = sc.fit_transform(stocks)
X = pd.DataFrame(xs, index=stocks.index, columns=stocks.columns)

# Kmeans
KS = range(2, 30)

# storage
inertia = []
silo = []

for k in KS:
  km = KMeans(k)
  km.fit(X)
  labs = km.predict(X)
  inertia.append(km.inertia_)
  silo.append(silhouette_score(X, labs))

#plot 
plt.figure(figsize=(15,5))


plt.subplot(1, 2, 1)
plt.title("Inertia")
sns.lineplot(KS, inertia)

plt.subplot(1, 2, 2)
plt.title("Silohouette Score")
sns.lineplot(KS, silo)

plt.show()