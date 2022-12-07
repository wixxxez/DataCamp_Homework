from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
class DataSet:

    def __init__(self, datasetfunc, useOneF):

        dataset = datasetfunc()
        self.df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
        x = dataset.data
        if(useOneF):
            x = x[:, np.newaxis, 2]
            self.x = np.array(x).reshape(-1, 1)
        else:
            self.x = np.array(x).reshape(-1, 10)
        #x = self.df[['bmi']]
        y = dataset.target
        #



        self.y = np.array(y).reshape(-1, 1)

    def getTrainTestSplit(self):

        return train_test_split(self.x,self.y);

    def getScaledData(self, x_train, x_test):

        scaler = StandardScaler()

        return scaler.fit_transform(x_train), scaler.transform(x_test)

    def getDF(self):

        return self.df

    def setX(self,x):

        self.x = x ;

