from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import numpy as np

class Dbscan_Cluster():

    def fit(self, X):
        for eps in [0.3, 0.5, 0.7, 1, 1.2]:
            dbscan = DBSCAN(eps=eps, min_samples=2)
            dbscan.fit(X)
            self.predicted = dbscan.labels_
            self.eps = eps
            self.X = X
            self.Visualie()


    def Visualie(self):
        plt.figure(figsize=(8, 6))
        plt.title("DBSCAN with epsilon ={} ".format(self.eps))
        colors = np.array(['green', 'orange', 'brown', 'purple', 'yellow'])
        colors = np.r_[colors, np.array(['black'] * 100)]
        predicted = self.predicted
        plt.scatter(self.X[:, 0], self.X[:, 1], c=colors[predicted], s=40, edgecolor='black', label='negative', alpha=0.7)