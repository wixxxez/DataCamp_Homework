
import pandas as pd
import numpy as np

import matplotlib
import seaborn
import matplotlib.dates as md
from matplotlib import pyplot as plt


import DataPreprocessing
class DataVisualization():

    def visualie_df(self, df):


        df.plot(x='timestamp', y='value')

    def visualize_categories(self, df):
        a = df.loc[df['categories'] == 0, 'value']
        b = df.loc[df['categories'] == 1, 'value']
        c = df.loc[df['categories'] == 2, 'value']
        d = df.loc[df['categories'] == 3, 'value']

        fig, ax = plt.subplots()
        a_heights, a_bins = np.histogram(a)
        b_heights, b_bins = np.histogram(b, bins=a_bins)
        c_heights, c_bins = np.histogram(c, bins=a_bins)
        d_heights, d_bins = np.histogram(d, bins=a_bins)

        width = (a_bins[1] - a_bins[0]) / 6

        ax.bar(a_bins[:-1], a_heights * 100 / a.count(), width=width, facecolor='blue', label='WeekEndNight')
        ax.bar(b_bins[:-1] + width, (b_heights * 100 / b.count()), width=width, facecolor='green', label='WeekEndLight')
        ax.bar(c_bins[:-1] + width * 2, (c_heights * 100 / c.count()), width=width, facecolor='red',
               label='WeekDayNight')
        ax.bar(d_bins[:-1] + width * 3, (d_heights * 100 / d.count()), width=width, facecolor='black',
               label='WeekDayLight')

        plt.legend()

    def visualize_dataset_for_each_category(self, df_class0, df_class1, df_class2, df_class3):
        fig, axs = plt.subplots(2, 2)
        df_class0.hist(ax=axs[0, 0], bins=32)
        df_class1.hist(ax=axs[0, 1], bins=32)
        df_class2.hist(ax=axs[1, 0], bins=32)
        df_class3.hist(ax=axs[1, 1], bins=32)

    def visualize_each_category_with_anomalys(self, df0, df1, df2,df3):
        a0 = df0.loc[df0['anomaly'] == 1, 'value']
        b0 = df0.loc[df0['anomaly'] == -1, 'value']

        a1 = df1.loc[df1['anomaly'] == 1, 'value']
        b1 = df1.loc[df1['anomaly'] == -1, 'value']

        a2 = df2.loc[df2['anomaly'] == 1, 'value']
        b2 = df2.loc[df2['anomaly'] == -1, 'value']

        a3 = df3.loc[df3['anomaly'] == 1, 'value']
        b3 = df3.loc[df3['anomaly'] == -1, 'value']

        fig, axs = plt.subplots(2, 2)
        axs[0, 0].hist([a0, b0], bins=32, stacked=True, color=['blue', 'red'], label=['normal', 'anomaly'])
        axs[0, 1].hist([a1, b1], bins=32, stacked=True, color=['blue', 'red'], label=['normal', 'anomaly'])
        axs[1, 0].hist([a2, b2], bins=32, stacked=True, color=['blue', 'red'], label=['normal', 'anomaly'])
        axs[1, 1].hist([a3, b3], bins=32, stacked=True, color=['blue', 'red'], label=['normal', 'anomaly'])
        axs[0, 0].set_title("WeekEndNight")
        axs[0, 1].set_title("WeekEndLight")
        axs[1, 0].set_title("WeekDayNight")
        axs[1, 1].set_title("WeekDayLight")
        plt.legend()

    def visualize_all_category_by_time(self,df_class0, df_class1, df_class2, df_class3, df):
        df_class = pd.concat([df_class0, df_class1, df_class2, df_class3])
        df['anomaly22'] = df_class['anomaly']
        df['anomaly22'] = np.array(df['anomaly22'] == -1).astype(int)
        # visualisation of anomaly throughout time (viz 1)
        fig, ax = plt.subplots()

        a = df.loc[df['anomaly22'] == 1, ('time_epoch', 'value')]  # anomaly

        ax.plot(df['time_epoch'], df['value'], color='blue')
        ax.scatter(a['time_epoch'], a['value'], color='red')
class DataSet:

    def __init__(self):

        self.load_data();
    def load_data(self):
        df = pd.read_csv("data/realKnownCause/realKnownCause/ambient_temperature_system_failure.csv")

        df = DataPreprocessing.DataPreprocessing().transform(df)
        self.df = df

    def getData(self):
        return self.df

    def getDatasetsForEachCategory(self):
        df = self.getData()
        df_class0 = df.loc[df['categories'] == 0, 'value']
        df_class1 = df.loc[df['categories'] == 1, 'value']
        df_class2 = df.loc[df['categories'] == 2, 'value']
        df_class3 = df.loc[df['categories'] == 3, 'value']

        return df_class0,df_class1,df_class2,df_class3


def show_data():
    dt = DataSet()

    df = dt.getData()
    df1, df2, df3, df4 = dt.getDatasetsForEachCategory()
    plotter = DataVisualization()
    plotter.visualie_df(df)
    plotter.visualize_categories(df)
    plotter.visualize_dataset_for_each_category(df1, df2, df3, df4)
    plt.show()


