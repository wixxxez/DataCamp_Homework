from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt
import Agglomerative
import DBSCAN_Cluster
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV


class Kmeans():

    def fit(self, X):
        grid = {
            "n_clusters": [3, 4, 5]
        }
        clf = KMeans(n_clusters=5)
        clf.fit(X);
        # rf_cv = GridSearchCV(estimator=KMeans(), param_grid=grid, cv=5)
        # rf_cv.fit(X)
        # print(rf_cv.best_params_)
        self.X = X
        self.clf = clf

    def Visualize(self,y):
        X = self.X
        clf = self.clf
        plt.figure()
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
        centroids = clf.cluster_centers_
        plt.scatter(centroids[:, 0], centroids[:, 1], c="red")