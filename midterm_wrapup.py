# -*- coding: utf-8 -*-
"""Midterm_wrapup.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1KbGnejJo1bcjBRGXl7dFinkdXcJL6948
"""

# installs
! pip install scikit-learn
! pip install umap-learn
! pip install scikit-plot

#imports
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm
import plotly.express as px

#scipy
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.cluster.hierarchy import fcluster
from scipy.spatial.distance import pdist, squareform

#sklearn
from sklearn import metrics 
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.manifold import MDS
from sklearn.manifold import TSNE
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors


from mlxtend.frequent_patterns import apriori, association_rules

import scikitplot as skplot

# auth into GCP Big Query

# COLAB Only
from google.colab import auth
auth.authenticate_user()
print('Authenticated')

# for non-Colab
# see resources, as long as token with env var setup properly, below should work

# get the data
SQL = "SELECT * from `questrom.datasets.mtcars`"
YOUR_BILLING_PROJECT = "ba-775pd"

df = pd.read_gbq(SQL, YOUR_BILLING_PROJECT)

"""# Examples"""

# lets drop the model column and use it as the index
df.index = df.model
df.drop(columns="model", inplace=True)

# keep just the continous variables
df2 = pd.concat((df.loc[:, "mpg"], df.loc[:, "disp":"qsec"]), axis=1)

# or
# drop the unnessary columns, I identified in the step ahead
df.drop(columns=['Quarter end', 'ticker', 'quarter_end', 'Split factor','Shares split adjusted'], inplace=True)

# replace missing values with the mean and lower case column names
df = df.fillna(df.mean())
df.columns = df.columns.str.lower()

# double checking if there are any missing values 
print(df.isna().sum().sum())

"""# Scaling the Data
only if the data varies a lot
"""

# first, I am going to scale the data given the varying units of measurement
sc = StandardScaler() #standardize features
nl = Normalizer() #rescales each sample
sm = sc.fit_transform(df)

sm = pd.DataFrame(sm, columns=df.columns)

# confirm the changes
sm.head(3)

"""# Hierarchical Clustering"""

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

#chose the one with the best clusterts - more "clusters" visible
METHODS = ['single', 'complete', 'average', 'ward']
plt.figure(figsize=(20,5))


# loop and build our plot for the different methods
for i, m in enumerate(METHODS):
  plt.subplot(1, 4, i+1)
  plt.title(m)
  dendrogram(linkage(distc, method=m), 
             leaf_rotation=90)

# Example: using cosine + complete
# the labels with 7 clusters
labs = fcluster(linkage(distc, method="complete"), 7, criterion="maxclust")

# confirm
np.unique(labs)

# add a cluster column to the stocks data set
df['cluster'] = labs
print(df.head(3))

#let see if the data in the cluster is evenly distributed
print(df.cluster.value_counts(dropna=False, sort=False))

# cluster solution
clus_profile = df.groupby("cluster").mean()

clus_profile.T

# heatmap plot of the clusters with normalized data
scp = StandardScaler()
cluster_scaled = scp.fit_transform(clus_profile)

cluster_scaled = pd.DataFrame(cluster_scaled, 
                              index=clus_profile.index, 
                              columns=clus_profile.columns)

sns.heatmap(cluster_scaled, cmap="Blues", center=0)
plt.show()

"""# KMeans"""

# another method: KMeans 
xs = sc.fit_transform(df)
X = pd.DataFrame(xs, index=df.index, columns=df.columns)

# Kmeans for 2 to 30 clusters
KS = range(2, 30)

# storage
inertia = []
silo = []

# wenn mehrere Clustergößen getestet werden sollen
for k in KS:
  km = KMeans(k)
  km.fit(X)
  labs = km.predict(X)
  inertia.append(km.inertia_)
  silo.append(silhouette_score(X, labs))

print(silo)

#plot 
plt.figure(figsize=(15,5))


plt.subplot(1, 2, 1)
plt.title("Inertia")
sns.lineplot(KS, inertia)

plt.subplot(1, 2, 2)
plt.title("Silohouette Score")
sns.lineplot(KS, silo)

plt.show()

for i, s in enumerate(silo[:10]):
  print(i+2,s)

# looks like 5 is a good point
# get the model
k5 = KMeans(5)
k5_labs = k5.fit_predict(X)

# metrics
k5_silo = silhouette_score(X, k5_labs)
k5_ssamps = silhouette_samples(X, k5_labs)
np.unique(k5_labs)

# lets compare via silo

skplot.metrics.plot_silhouette(X, labs, title="HClust", figsize=(15,5))
plt.show()

skplot.metrics.plot_silhouette(X, k5_labs, title="KMeans - 5", figsize=(15,5))
plt.show()

"""# PCA"""

# Let try another method -PCA

pca = PCA()
pcs = pca.fit_transform(df)

# shape confirmation (rows/features) are identical SHAPES
pcs.shape == df.shape

# first, lets get the explained variance
# elbow plot

varexp = pca.explained_variance_ratio_
sns.lineplot(range(1, len(varexp)+1), varexp)

# cumulative variance

plt.title("Cumulative Explained Variance")
plt.plot(range(1, len(varexp)+1), np.cumsum(varexp))
plt.axhline(.987)

# quick function to construct the barplot easily
def ev_plot(ev):
  y = list(ev)
  x = list(range(1,len(ev)+1))
  return x, y

# another approach for selection is to use explained variance > 1
ev = pca.explained_variance_

x, y = ev_plot(pca.explained_variance_)
sns.barplot(x, y)

plt.title("Explained Variance - Eigenvalue")
plt.bar(x=x, height=y)
plt.axhline(y=1, ls="--")
plt.show()

# component, feature
comps = pca.components_
type(comps)
comps.shape


# build column labels

COLS = ["PC" + str(i) for i in range(1, len(comps)+1)]
loadings = pd.DataFrame(comps.T, columns=COLS, index=judges.columns)
loadings

# help with hacking on matplotlib
# https://matplotlib.org/3.1.1/gallery/images_contours_and_fields/image_annotated_heatmap.html

sns.heatmap(loadings)
plt.show()

pca.explained_variance_ratio_

#make a new dataframe
# remember, these are what we might use if our task was to learn a model

j = pd.DataFrame(comps, columns=['pc1', 'pc2'], index=judges.index)
j.head()

## notice that I am NOT putting these back onto the original
## you can of course, but the point is that these are now our new features for any other downstream tasks

# viz
sns.scatterplot(x="pc1", y="pc2", data=j)
plt.show()

pca = PCA()
pca.fit(judges)
pcs = pca.transform(judges)