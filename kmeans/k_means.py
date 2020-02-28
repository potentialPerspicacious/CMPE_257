from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import random
import numpy as np
from sklearn.metrics import accuracy_score

n_samples = 2000
n_features = 2
n_centers = 5
cluster_sd = 0.8
x, y = make_blobs(n_samples=n_samples, n_features=n_features, centers=n_centers, cluster_std=cluster_sd, random_state=22)

plt.scatter(x[:, 0], x[:, 1], c='blue')
plt.show()

kmeans = KMeans(n_clusters=5, random_state=22)
kmeans.fit(x)
y_pred = kmeans.predict(x)
blobCenters = kmeans.cluster_centers_
print("The centroids are: ", blobCenters)
print("Accuracy:", accuracy_score(y, y_pred))
plt.scatter(x[:, 0], x[:, 1], c=y_pred, s=50, cmap='rainbow')
plt.scatter(blobCenters[:, 0], blobCenters[:, 1], c='black')
plt.show()

kmeans_10 = KMeans(n_clusters=10, random_state=2)
kmeans_10.fit(x)
y10_pred = kmeans_10.predict(x)
blobCenters_10 = kmeans_10.cluster_centers_
print("The centroids of 10 cluster kmeans are: ", blobCenters_10)
print("Accuracy:", accuracy_score(y, y10_pred))
plt.scatter(x[:, 0], x[:, 1], c=y10_pred, s=50, cmap='rainbow')
plt.scatter(blobCenters_10[:, 0], blobCenters_10[:, 1], c='black')
plt.show()

#Deterministic noise
random.shuffle(y[:200])
kmeans_N10 = KMeans(n_clusters=10, random_state=2)
kmeans_N10.fit(x)
yN10_pred = kmeans_N10.predict(x)
blobCenters_N10 = kmeans_N10.cluster_centers_
print("The centroids of 10 cluster kmeans are: ", blobCenters_N10)
print("Accuracy:", accuracy_score(y, yN10_pred))
plt.scatter(x[:, 0], x[:, 1], c=yN10_pred, s=50, cmap='rainbow')
plt.scatter(blobCenters_N10[:, 0], blobCenters_N10[:, 1], c='black')
plt.show()

#Stocastic Noise
x_noise = np.random.uniform(-5, 0, (200, 2))
y_noise = np.random.uniform(0, 4, (200,))
x = np.vstack((x, x_noise))
y = np.hstack((y, y_noise))
kmeans_SN10 = KMeans(n_clusters=10, random_state=2)
kmeans_SN10.fit(x)
ySN10_pred = kmeans_SN10.predict(x)
blobCenters_SN10 = kmeans_SN10.cluster_centers_
print("The centroids of 10 cluster kmeans are: ", blobCenters_SN10)
print("Accuracy:", accuracy_score(y, ySN10_pred))
plt.scatter(x[:, 0], x[:, 1], c=ySN10_pred, s=50, cmap='rainbow')
plt.scatter(blobCenters_SN10[:, 0], blobCenters_SN10[:, 1], c='black')
plt.show()


