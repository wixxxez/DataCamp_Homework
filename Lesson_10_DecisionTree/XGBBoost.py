import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from xgboost import XGBClassifier
from sklearn import tree
import graphviz
from sklearn.model_selection import train_test_split
import os
import PlotService
from sklearn.datasets import make_blobs

from mlxtend.plotting import plot_decision_regions
import  DecisionBoundary
import Dataset
os.environ["PATH"] += os.pathsep + r'C:\Program Files\Graphviz\bin'
plot = PlotService.DecisionTreeVisualizeService()


class GradientBoostBreastCancer():

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
    def predict(self):
        X_train =self.X_train
        X_test = self.X_test
        y_train =self.y_train
        y_test = self.y_test
        labels = self.labels
        features =self.features

        clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        clf.fit(X_train, y_train)


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
        plt.figure()


        plt.show()

dataset = Dataset.DataSet()
X,y,labels,features = dataset.LoadAllFeatures()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)
d = GradientBoostBreastCancer(
    X_train,X_test,
    y_train,y_test,
    labels, features
)
d.predict()