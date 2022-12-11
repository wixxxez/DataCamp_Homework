import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import DataSet
import Plotter
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from mlxtend.plotting import plot_decision_regions


class SVMLinear:

    def setXY(self, X,Y):

        self.X = X;
        self.Y = Y;

    def model(self):
        self.clf = LinearSVC(C=1, max_iter=10000).fit(self.X, self.Y)

    def printScore(self):
        print("train accuracy= {:.3%}".format(self.clf.score(self.X, self.Y)))
        print('b = {}\nw = {}'.format(self.clf.intercept_, self.clf.coef_))

    def plotBoundary(self, labels_dict):
        plt.figure()
        plot_decision_regions(self.X.to_numpy(), self.Y, self.clf, legend=2)
        plt.show()

class SVMRBF():


    def setXY(self, X,Y):

        self.X = X;
        self.Y = Y;

    def model(self):
        self.clf = SVC(C=1, gamma=1).fit(self.X, self.Y)

    def printScore(self):
        print("train accuracy= {:.3%}".format(self.clf.score(self.X, self.Y)))
        print('b = {}\n'.format(self.clf.intercept_))

    def plotBoundary(self, labels_dict):
        plt.figure()

        plot_decision_regions(self.X.to_numpy() ,self.Y, self.clf,legend=2)
        plt.show()

class SVMPolynom():
    def setXY(self, X,Y):

        self.X = X;
        self.Y = Y;

    def model(self):
        self.clf = SVC(kernel= 'poly').fit(self.X, self.Y)

    def printScore(self):
        print("train accuracy= {:.3%}".format(self.clf.score(self.X, self.Y)))
        print('b = {}\n'.format(self.clf.intercept_))

    def plotBoundary(self, labels_dict):
        plt.figure()

        plot_decision_regions(self.X.to_numpy() ,self.Y, self.clf,legend=2)
        plt.show()

