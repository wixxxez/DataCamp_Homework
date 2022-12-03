import pandas as pd
import numpy as np
import KNNService
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import MinMaxScaler
from mlxtend.plotting import plot_decision_regions


class RegressorTrain:

    def setXYtrain(self,x,y):

        self.x_train = x;
        self.y_train = y;

    def setXYtest(self,x,y):

        self.x_test = x
        self.y_test = y
    def TrainNoScaledData(self):
        k_range = range(1, 25, 2)
        scores_train = []
        scores_test = []
        accuracy = []
        for k in k_range:
            knn = KNeighborsRegressor(n_neighbors=k)
            knn.fit(self.x_train, self.y_train)
            scores_train.append(knn.score(self.x_train, self.y_train))
            scores_test.append(knn.score(self.x_train, self.y_test))
            accuracy.append([knn.score(self.x_test, self.y_test), k])
        self.selfScoresTrainNoScaled = scores_train
        self.selfScoresTestNoScaled = scores_test
        self.k_range = k_range
        self.NoScaledKnn = knn;
        return max(accuracy);

    def TrainScaledData(self):

        scaler = MinMaxScaler()
        x_train_scaled = scaler.fit_transform(self.x_train, self.y_train)
        x_test_scaled = scaler.transform(self.x_test)
        k_range = range(1, 25, 2)
        scores_train = []
        scores_test = []
        accuracy = []
        for k in k_range:
            knn = KNeighborsRegressor(n_neighbors=k)
            knn.fit(x_train_scaled, self.y_train)
            scores_train.append(knn.score(x_train_scaled, self.y_train))
            scores_test.append(knn.score(x_test_scaled, self.y_test))
            accuracy.append([knn.score(x_test_scaled, self.y_test), k])

        self.selfScoresTrainScaled = scores_train
        self.selfScoresTestScaled = scores_test
        self.k_range = k_range
        self.ScaledKnn = knn
        return max(accuracy);

    def visualizeScaled(self,plotter):

        plotter.knnRegressorScoreVisualization(self.selfScoresTrainScaled,self.selfScoresTestScaled,self.k_range);

    def visualizeNoScaled(self, plotter):

        plotter.knnRegressorScoreVisualization(self.selfScoresTrainNoScaled,self.selfScoresTestNoScaled,self.k_range);

    def getNoScalledKnn(self):

        return self.NoScaledKnn;

    def getScalledKnn(self):

        return self.ScaledKnn;