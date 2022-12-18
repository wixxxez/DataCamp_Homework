from sklearn.cluster import KMeans
import  pandas as pd
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
import numpy as np
import Plotter
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

Xc_2, y_true_= make_classification(n_samples=200,
                                    n_features=2,
                                    n_informative=2,
                                    n_redundant=0,
                                    random_state=0,
                                    n_clusters_per_class=1,
                                    class_sep = 0.8)

x = Xc_2[:, 0]
y = Xc_2[:, 1]

scaler = MinMaxScaler()
clf = KMeans(n_clusters=4)
clf.fit(Xc_2);
print(y_true_)
plt.figure()
plt.scatter(x,y)

y_p = clf.predict(Xc_2)

print("Train accuracy: ")
print(accuracy_score(y_true_,y_p))

plt.figure()
plt.scatter(x,y, c=y_p, cmap='viridis')
centroids = clf.cluster_centers_
plt.scatter(centroids[:,0],centroids[:, 1],c="red")
plt.show()



