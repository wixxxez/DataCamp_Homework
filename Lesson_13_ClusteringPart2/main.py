from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt
import Agglomerative
import DBSCAN_Cluster
import Kmeans
X, y = make_blobs(n_samples = 500, n_features = 2, centers = 5,
                           cluster_std = 0.6, random_state = 0)
colors = np.array([plt.cm.Spectral(val)
          for val in np.linspace(0, 1, len(set(y)))])
plt.figure(figsize=(8,6))


plt.scatter(X[:,0], X[:,1], c= colors[y], s= 20)
#plt.figure()
#sch.dendrogram(sch.linkage(X, method='ward'))
def startAgglomerative():

    agglomerative_clusters = Agglomerative.AglomerativeClusters()
    agglomerative_clusters.fit(X, "complete")
    agglomerative_clusters.Visualize()
    agglomerative_clusters.fit(X,"single")
    agglomerative_clusters.Visualize()

def DBSCAN_clustering():
    DBSCAN_Cluster.Dbscan_Cluster().fit(X)

def KMeans_clusters():
    kmeans = Kmeans.Kmeans()
    kmeans.fit(X)
    kmeans.Visualize(y)


#startAgglomerative()
#DBSCAN_clustering()
KMeans_clusters()
plt.show()

