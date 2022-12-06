import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso

class LassoRegression:

    def __init__(self):

        self.x_train = None
        self.x_test = None
        self.y_test = None
        self.y_train = None
        self.model = Lasso()

    def setTestTrainData(self, xTrain, xTest, yTrain, yTest):

        self.x_train = xTrain
        self.x_test = xTest
        self.y_test = yTest
        self.y_train = yTrain

    def fitRegression(self):

        return self.model.fit(self.x_train, self.y_train)

    def predict(self):

        return self.model.predict(self.x_test)
    def getTrainScoreRegression(self):

        return self.model.score(self.x_train, self.y_train)

    def getTestScoreRegression(self):

        return self.model.score(self.x_test, self.y_test);

    def getCoef(self):

        return 'b: {}, \nw= {}'.format(self.model.intercept_, self.model.coef_)
