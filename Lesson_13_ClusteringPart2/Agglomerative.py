from sklearn.datasets import make_blobs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch



class AglomerativeClusters():

    def fit(self,X, linkage, affinity = None):
        self.X = X
        self.linkage = linkage
        model = AgglomerativeClustering(n_clusters=5, linkage=linkage)
        model.fit(X)
        self.labels = model.labels_

    def Visualize(self):
        X= self.X
        labels = self.labels
        plt.figure(figsize=(8, 6))
        plt.title(self.linkage)
        plt.scatter(X[labels == 0, 0], X[labels == 0, 1], s=20, marker='o', c='red')
        plt.scatter(X[labels == 1, 0], X[labels == 1, 1], s=20, marker='o', color='orange')
        plt.scatter(X[labels == 2, 0], X[labels == 2, 1], s=20, marker='o', color='green')
        plt.scatter(X[labels == 3, 0], X[labels == 3, 1], s=20, marker='o', color='purple')
        plt.scatter(X[labels == 4, 0], X[labels == 4, 1], s=20, marker='o', color='yellow')

