import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
class DataSet:

    def __init__(self):
        cancer = load_breast_cancer()
        self.cancer = cancer
        df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
        df['target'] = cancer.target
        self.df = df

    def LoadAllFeatures(self):
        X = self.cancer.data
        X = X.reshape(-1, 30)
        y = pd.Series(self.cancer.target)
        labels, features = self.cancer.target_names, self.cancer.feature_names

        return X, y ,labels, features

    def Load2Features(self, features = ['mean radius', 'mean concave points']):
        X = self.df[['mean radius', 'mean concave points']]
        y = pd.Series(self.cancer.target)
        labels, features = self.cancer.target_names, features

        return X,y,labels,features
