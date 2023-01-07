import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import Dataset



def printInfo(df):

    print("DF info: ")
    print(df.info());
    print("DF shape: ")
    print(df.shape)
    print("DF Columns: ")
    print(df.columns)

def VisualizeCorrMatrix(df):

    plt.figure()
    correlation = df.corr();
    heatmap = sns.heatmap(correlation, annot= True)
    heatmap.set(xlabel = "X values", ylabel="Survive",title= "CorrelationMatrix")

def VisualizeDependency(df, x, y):
    #plt.figure()
    sns.barplot(x=x, y=y, data=df)
    plt.title("Barplot {}".format(x))
    plt.get_current_fig_manager().canvas.set_window_title("Barplot {}".format(x))
    plt.figure()
    sns.countplot(x=x, hue=y, data=df)
    plt.title("Countplot {}".format(x))
    plt.get_current_fig_manager().set_window_title("Countplot {}".format(x))

def VisualizeY(df,y):
    plt.figure()
    sns.barplot(y=y, data=df)


def checkNAN(df):

    return df.isnull().sum()


dt = Dataset.Dataset()
X = dt.getTrainData()





