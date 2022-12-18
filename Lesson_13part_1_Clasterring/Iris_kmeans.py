from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

data = load_iris()

x = data.data
y_true = data.target

x_train , x_test, y_train, y_test = train_test_split(x, y_true)
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
features_dict = {k: v for k, v in enumerate(data.target_names)}
clf = KMeans(n_clusters=3)
clf.fit(x_train)

plt.figure()
plt.scatter(x[:,2], x[:,3], c=data.target, cmap='viridis')

y_pred = clf.predict(x_test)

print(1-accuracy_score(y_train, clf.predict(x_train)))
print(1-accuracy_score(y_test,y_pred))

plt.figure()
plt.scatter(x_test[:,2], x_test[:,3], c=y_pred, cmap='viridis' )
plt.scatter(clf.cluster_centers_[:,2], clf.cluster_centers_[:,3], c='red')
plt.show()