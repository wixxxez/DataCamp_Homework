import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
import graphviz
from sklearn.model_selection import train_test_split
import os
import PlotService
from sklearn.datasets import make_blobs

from mlxtend.plotting import plot_decision_regions
import  DecisionBoundary
import Dataset
from sklearn.tree import plot_tree
os.environ["PATH"] += os.pathsep + r'C:\Program Files\Graphviz\bin'
plot = PlotService.DecisionTreeVisualizeService()


class RandomForestBreastCancer():

    def __init__(self,
                 X_train, X_test,
                 y_train, y_test,
                 labels, features
                 ):

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.labels = labels
        self.features = features
    def predict(self, max_depth):
        X_train =self.X_train
        X_test = self.X_test
        y_train =self.y_train
        y_test = self.y_test
        labels = self.labels
        features =self.features
        max_depth = max_depth
        clf = RandomForestClassifier(
            criterion=  'entropy',
            random_state=20,
            max_depth=max_depth,
        #     max_leaf_nodes=4,
        ).fit(X_train, y_train)


        print("train accuracy= {:.3%}".format(clf.score (X_train, y_train)))
        print("test accuracy= {:.3%}".format(clf.score (X_test, y_test)))
        if(len(features) == 2):
            DecisionBoundary.plot_labeled_decision_regions(X_test,y_test,  clf)

        else:
            plt.figure()

            feature_values = {i: np.mean(X_train[:, i]) for i in range(2, 30)}

            plot_decision_regions(X_test, y_test.to_numpy(), clf=clf,
                                  filler_feature_values=feature_values, )
            plt.title("Decision Boundary")

        fig = plt.figure(figsize=(15, 10))
        plot_tree(clf.estimators_[0],
                  feature_names=features,
                  class_names=labels,
                  filled=True, impurity=True,
                  rounded=True)
        plt.show()

dataset = Dataset.DataSet()
X,y,labels,features = dataset.LoadAllFeatures()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)
d = RandomForestBreastCancer(
    X_train,X_test,
    y_train,y_test,
    labels, features
)
d.predict(7)