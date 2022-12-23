from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.covariance import EllipticEnvelope
#from pyemma import msm # not available on Kaggle Kernel
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.cluster import DBSCAN
import pandas as pd
import DataSet
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
outliers_fraction = 0.01

class EllipticEnvelopeForData():

    def __init__(self, df):

        self.df = df

    def train(self):
        df = self.df
        envelope = EllipticEnvelope(contamination=outliers_fraction)
        X_train = df.values.reshape(-1, 1)
        envelope.fit(X_train)
        df = pd.DataFrame(df)
        df['deviation'] = envelope.decision_function(X_train)
        df['anomaly'] = envelope.predict(X_train)

        return df

class IsolationForestForData():

    def __init__(self, df):

        self.df = df

    def train(self):
        df = self.df
        data = df[['value', 'hours', 'daylight', 'DayOfTheWeek', 'WeekDay']]
        min_max_scaler = preprocessing.StandardScaler()
        np_scaled = min_max_scaler.fit_transform(data)
        data = pd.DataFrame(np_scaled)
        # train isolation forest
        model = IsolationForest(contamination=outliers_fraction)
        model.fit(data)
        # add the data to the main
        df['anomaly'] = pd.Series(model.predict(data))
        df['anomaly'] = df['anomaly'].map({1: 0, -1: 1})

        return df

    def visualize_data_by_time(self, df):
        fig, ax = plt.subplots()

        a = df.loc[df['anomaly'] == 1, ['time_epoch', 'value']]  # anomaly

        ax.plot(df['time_epoch'], df['value'], color='blue')
        ax.scatter(a['time_epoch'], a['value'], color='red')

    def visualize_data_by_temp(self,df):
        a = df.loc[df['anomaly'] == 0, 'value']
        b = df.loc[df['anomaly'] == 1, 'value']

        fig, axs = plt.subplots()
        axs.hist([a, b], bins=32, stacked=True, color=['blue', 'red'], label=['normal', 'anomaly'])
        plt.legend()
class DBSCANForData():

    def __init__(self, df):

        self.df = df

    def train(self):
        df = self.df
        dbscan = DBSCAN(eps=outliers_fraction+0.05)

        X_train = df.values.reshape(-1, 1)
        dbscan.fit(X_train)
        df = pd.DataFrame(df)

        df['anomaly'] = dbscan.fit_predict(X_train)

        return df
    def visualize_data_by_time(self, df):
        fig, ax = plt.subplots()

        a = df.loc[df['anomaly'] == 1, ['time_epoch', 'value']]  # anomaly

        ax.plot(df['time_epoch'], df['value'], color='blue')
        ax.scatter(a['time_epoch'], a['value'], color='red')

    def visualize_data_by_temp(self,df):
        a = df.loc[df['anomaly'] == 0, 'value']
        b = df.loc[df['anomaly'] == 1, 'value']

        fig, axs = plt.subplots()
        axs.hist([a, b], bins=32, stacked=True, color=['blue', 'red'], label=['normal', 'anomaly'])
        plt.legend()
class SVMForData():

    def __init__(self, df):

        self.df = df

    def train(self):
        df = self.df
        data = df[['value', 'hours', 'daylight', 'DayOfTheWeek', 'WeekDay']]
        min_max_scaler = preprocessing.StandardScaler()
        np_scaled = min_max_scaler.fit_transform(data)
        # train one class SVM
        model = OneClassSVM(nu=0.95 * outliers_fraction)  # nu=0.95 * outliers_fraction  + 0.05
        data = pd.DataFrame(np_scaled)
        model.fit(data)
        # add the data to the main
        df['anomaly'] = pd.Series(model.predict(data))
        df['anomaly'] = df['anomaly'].map({1: 0, -1: 1})

        return df

    def visualize_data_by_time(self, df):
        fig, ax = plt.subplots()

        a = df.loc[df['anomaly'] == 1, ['time_epoch', 'value']]  # anomaly

        ax.plot(df['time_epoch'], df['value'], color='blue')
        ax.scatter(a['time_epoch'], a['value'], color='red')

    def visualize_data_by_temp(self,df):
        a = df.loc[df['anomaly'] == 0, 'value']
        b = df.loc[df['anomaly'] == 1, 'value']

        fig, axs = plt.subplots()
        axs.hist([a, b], bins=32, stacked=True, color=['blue', 'red'], label=['normal', 'anomaly'])
        plt.legend()
def EnvelopeAnomalys():
    dt = DataSet.DataSet()
    df0, df1, df2, df3 = dt.getDatasetsForEachCategory()
    main_df = dt.getData()
    df_with_anomaly = []
    for i in [df0, df1, df2, df3]:
        model = EllipticEnvelopeForData(i)
        df = model.train()
        df_with_anomaly.append(df)

    plotter = DataSet.DataVisualization()

    plotter.visualize_each_category_with_anomalys(df_with_anomaly[0], df_with_anomaly[1], df_with_anomaly[2],
                                                  df_with_anomaly[3])
    plotter.visualize_all_category_by_time(df_with_anomaly[0], df_with_anomaly[1], df_with_anomaly[2],
                                           df_with_anomaly[3], main_df)
    plt.show()

def IsolationForestAnomalys():

    dt = DataSet.DataSet()
    df = dt.getData()
    iso_forest = IsolationForestForData(df)
    df = iso_forest.train()
    iso_forest.visualize_data_by_temp(df)
    iso_forest.visualize_data_by_time(df)
    plt.show()

def OneSVMAnomalys():
    dt = DataSet.DataSet()
    df = dt.getData()
    svm = SVMForData(df)
    svm.train()
    svm.visualize_data_by_time(df)
    svm.visualize_data_by_temp(df)
    plt.show()

def DBSCANAnomalys():

    dt = DataSet.DataSet()
    df0, df1, df2, df3 = dt.getDatasetsForEachCategory()
    main_df = dt.getData()
    df_with_anomaly = []
    for i in [df0, df1, df2, df3]:
        model = DBSCANForData(i)
        df = model.train()
        df_with_anomaly.append(df)

    plotter = DataSet.DataVisualization()

    plotter.visualize_each_category_with_anomalys(df_with_anomaly[0], df_with_anomaly[1], df_with_anomaly[2],
                                                  df_with_anomaly[3])
    plotter.visualize_all_category_by_time(df_with_anomaly[0], df_with_anomaly[1], df_with_anomaly[2],
                                           df_with_anomaly[3], main_df)
    plt.show()




