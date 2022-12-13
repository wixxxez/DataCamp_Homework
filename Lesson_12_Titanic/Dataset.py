import pandas as pd


class Dataset():

    def __init__(self):

        self.train = pd.read_csv('data/train.csv')
        self.test = pd.read_csv('data/test.csv')

    def getTrainData(self):

        return self.train

    def getTestData(self):

        return  self.test;