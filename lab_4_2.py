import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn import preprocessing
from sklearn import metrics


data_init = pd.read_csv("clustering_data.csv")

# Initial data plot
plt.scatter(data_init.annual_income, data_init.home_index, marker='.')
plt.title('Initial Data')
plt.xlabel('Annual Income')
plt.ylabel('Home Index')
plt.show()

# Data preparation
min_max_scaler = preprocessing.MinMaxScaler()
data_scaled = min_max_scaler.fit_transform(list(zip(data_init.annual_income, data_init.home_index)))


# --------Agglomerative Algorithm---------------

# Agglomerative processing with default parameter n_clusters
cluster = AgglomerativeClustering()
cluster.fit(data_scaled)
# Plotting default results
plt.scatter(data_scaled[:, 0], data_scaled[:, 1], c=cluster.labels_, marker='.')
plt.title('Agglomerative default clustering')
plt.xlabel('Income scaled')
plt.ylabel('Ratio Index scaled')
plt.show()

# The plot reveals that best parameter for n_clusters=3

# Cluster quality evaluation by Silhouette method
sil = []
for n_clust in range(2, 11):
    cluster = AgglomerativeClustering(n_clusters=n_clust)
    cluster.fit(data_scaled)
    labels = cluster.labels_
    sil.append(metrics.silhouette_score(data_scaled, labels, metric='euclidean'))
plt.plot(range(2, 11), sil)
plt.title('Silhouette method (Agglomerative)')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Silhouette Scores')
plt.show()


# Agglomerative processing with optimal parameter n_clusters=3
cluster = AgglomerativeClustering(n_clusters=3)
cluster.fit(data_scaled)
# Plotting default results
plt.scatter(data_scaled[:, 0], data_scaled[:, 1], c=cluster.labels_, marker='.')
plt.xlabel('Income scaled')
plt.ylabel('Ratio Index scaled')
plt.title('Agglomerative with optimal clusters (n_clusters=3)')
plt.show()


# --------DBSCAN Algorithm---------------

# DBSCAN processing with optimal parameters eps=0.06 and min_samples=15. They have been determined by hand
# by visual analysis which outputs 3 clusters
dbscan = DBSCAN(eps=0.06, min_samples=15)
dbscan.fit(data_scaled)
# Plotting default results
plt.scatter(data_scaled[:, 0], data_scaled[:, 1], c=dbscan.labels_, marker='.')
plt.title('DBSCAN optimized clustering')
plt.xlabel('Income scaled')
plt.ylabel('Ratio Index scaled')
plt.show()
