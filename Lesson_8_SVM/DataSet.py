import pandas as pd
from sklearn.datasets import load_iris

class IrisDataSet():

    def __init__(self):
        iris = load_iris()

        self.X, self.y, labels, feature_names = iris.data, iris.target, iris.target_names, iris['feature_names']
        df_iris = pd.DataFrame(self.X, columns=feature_names)
        self.target_names = labels
        df_iris['label'] = self.y
        features_dict = {k: v for k, v in enumerate(labels)}
        df_iris['label_names'] = df_iris.label.apply(lambda x: features_dict[x])
        self.df_iris = df_iris

    def Load2Features(self):

        return self.df_iris[['petal length (cm)', 'petal width (cm)']]

    def getDf(self):
        return self.df_iris;

    def getX(self):
        return self.X;

    def getY(self):
        return self.y;

    def getTargetNames(self):
        labels_dict = {k: v for k, v in enumerate(self.target_names)}
        return labels_dict;