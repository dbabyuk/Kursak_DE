import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
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


# --------KMeans Algorithm---------------

# Kmeans processing with default parameter n_clusters
kmeans = KMeans()
kmeans.fit(data_scaled)
Y_predicted = kmeans.predict(data_scaled)
# Plotting default results
plt.scatter(data_scaled[:, 0], data_scaled[:, 1], c=Y_predicted, marker='.')
plt.title('KMeans default clustering')
plt.xlabel('Income scaled')
plt.ylabel('Ratio Index scaled')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], s=100, marker='o', facecolors='none', edgecolors='r')
plt.show()

# Cluster quality evaluation by Elbow method
wcss = []
ari = []
for n_clust in range(1, 10):
    kmeans = KMeans(n_clusters=n_clust)
    kmeans.fit(data_scaled)
    wcss.append(kmeans.inertia_)
    ari.append(kmeans.labels_)
plt.plot(range(1, 10), wcss)
plt.title('Elbow method (KMeans)')
plt.xlabel('Number of clusters (k)')
plt.ylabel('WCSS')
plt.show()
# The plot reveals that best parameter for n_clusters=3

# Cluster quality evaluation by Silhouette method
sil = []
for n_clust in range(2, 11):
    kmeans = KMeans(n_clusters=n_clust)
    kmeans.fit(data_scaled)
    wcss.append(kmeans.inertia_)
    labels = kmeans.labels_
    sil.append(metrics.silhouette_score(data_scaled, labels, metric='euclidean'))
plt.plot(range(2, 11), sil)
plt.title('Silhouette method (KMeans)')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Silhouette Scores')
plt.show()


# Kmeans processing with optimal parameter n_clusters=3
kmeans = KMeans(n_clusters=3)
kmeans.fit(data_scaled)
Y_predicted = kmeans.predict(data_scaled)
# Plotting default results
plt.scatter(data_scaled[:, 0], data_scaled[:, 1], c=Y_predicted, marker='.')
plt.xlabel('Income scaled')
plt.ylabel('Ratio Index scaled')
plt.title('KMeans with optimal clusters (n_clusters=3)')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], s=100, marker='o', facecolors='none', edgecolors='r')
plt.show()


# --------KMedoids Algorithm---------------

# Kmedoids processing with default parameter n_clusters
kmedoids = KMedoids()
kmedoids.fit(data_scaled)
Y_predicted = kmedoids.predict(data_scaled)
# Plotting default results
plt.scatter(data_scaled[:, 0], data_scaled[:, 1], c=Y_predicted, marker='.')
plt.title('Kmedoids default clustering')
plt.xlabel('Income scaled')
plt.ylabel('Ratio Index scaled')
centers = kmedoids.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], s=100, marker='o', facecolors='none', edgecolors='r')
plt.show()

# Cluster quality evaluation by Elbow method
wcss = []
ari = []
for n_clust in range(1, 10):
    kmedoids = KMedoids(n_clusters=n_clust)
    kmedoids.fit(data_scaled)
    wcss.append(kmedoids.inertia_)
    ari.append(kmedoids.labels_)
plt.plot(range(1, 10), wcss)
plt.title('Elbow method (KMedoids)')
plt.xlabel('Number of clusters (k)')
plt.ylabel('WCSS')
plt.show()
# The plot reveals that best parameter for n_clusters=3

# Cluster quality evaluation by Silhouette method
sil = []
for n_clust in range(2, 11):
    kmedoids = KMedoids(n_clusters=n_clust)
    kmedoids.fit(data_scaled)
    wcss.append(kmedoids.inertia_)
    labels = kmedoids.labels_
    sil.append(metrics.silhouette_score(data_scaled, labels, metric='euclidean'))
plt.plot(range(2, 11), sil)
plt.title('Silhouette method (KMedoids)')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Silhouette Scores')
plt.show()


# KMedoids processing with optimal parameter n_clusters=3
kmedoids = KMedoids(n_clusters=3)
kmedoids.fit(data_scaled)
Y_predicted = kmeans.predict(data_scaled)
# Plotting default results
plt.scatter(data_scaled[:, 0], data_scaled[:, 1], c=Y_predicted, marker='.')
plt.xlabel('Income scaled')
plt.ylabel('Ratio Index scaled')
plt.title('Kmedoids with optimal clusters (n_clusters=3)')
centers = kmedoids.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], s=100, marker='o', facecolors='none', edgecolors='r')
plt.show()
