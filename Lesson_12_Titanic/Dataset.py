import pandas as pd
from sklearn.model_selection import  train_test_split

class Dataset():

    def __init__(self):

        self.train = pd.read_csv('data/train.csv').set_index('PassengerId')
        self.test = pd.read_csv('data/test.csv').set_index('PassengerId')

    def getTrainData(self):

        return self.train

    def getTestData(self):

        return  self.test;

    def getXTrain(self):

        featuresList = [ 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
       'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']

        X = self.train[featuresList];

        return X;

    def getYTrain(self):

        return self.train[['Survived']];

    def getSplitedData(self, X,y):

        return  train_test_split(X,y);
    def GetFeatures(self):
        features = ['Pclass',
                    'Embarked',
                    'Fare',
                    'Sex',
                    'Age',
                    'Parch',
                    'SibSp',
                    'Name',
                    'Cabin',
                    'Ticket',
                    ]

        return features;

