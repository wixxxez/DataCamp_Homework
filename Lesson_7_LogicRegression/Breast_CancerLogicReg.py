import numpy as np
from sklearn.linear_model import LogisticRegression
# import load_breast_cancer and get the X_cancer, y_cancer
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import Plotter
import  seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
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

X_cancer,y_cancer,labels, features = DataSet().Load2Features()
#  split to train and test using random_state = 0
X_train, X_test, y_train, y_test =  train_test_split(X_cancer,y_cancer)
#  train LogisticRegression classifier  for max_iter= 10000
clf = LogisticRegression(C=0.1).fit(X_train,y_train);
print('\nBreast cancer dataset')
print ('X_cancer.shape= {}'.format(X_cancer.shape))
print('Accuracy of Logistic regression classifier on training set: {:.2f}'
     .format(clf.score(X_train, y_train)))
print('Accuracy of Logistic regression classifier on test set: {:.2f}'
     .format(clf.score(X_test, y_test)))


plt.figure()
Plotter.plot_labeled_decision_regions(X_cancer,y_cancer,clf)
plt.show()
