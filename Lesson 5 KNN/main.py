import pandas as pd
import numpy as np
import KNNService
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import MinMaxScaler
from mlxtend.plotting import plot_decision_regions
import KNNRegresorService
def Iris():

    dataset = KNNService.IrisDataSet()
    x = dataset.getX()
    y = dataset.getY()

    x_train, x_test, y_train, y_test = dataset.split_on_test_and_train(x,y)

    trainService = KNNService.TrainKNN()
    trainService.set_train_XY(x_train,y_train)
    trainService.set_test_XY(x_test,y_test)
    NoScaledData = trainService.TrainNoScaledData();
    accuracyService = KNNService.AccuracyService(NoScaledData[1])

    #print("Best K for no scaled data is:",  accuracyService.getBestK());
    plotter = KNNService.ShowPlot()
    df = accuracyService.getDF()
    plotter.print_df(df,"k","accuracy","Iris: Best K for no scaled data")

    #---------------------------#
    ScaledData = trainService.TrainWithScaledData()
    accuracyService_scaled = KNNService.AccuracyService(ScaledData[1])
    #print("Best K for scaled data is: ", accuracyService_scaled.getBestK())
    df_scaled = accuracyService_scaled.getDF();
    plotter.print_df(df_scaled, "k", "accuracy", "Iris: Best K for scaled data")
    k_best = accuracyService_scaled.getBestK()[1]  # 'compute the best k'
    score_best = accuracyService_scaled.getBestK()[0]*100  # 'compute the best score'
    print('The best k = {} , score = {}'.format(k_best, score_best))
    plotter.show()

Iris();

def SyntheticDataset():
    X_D2, y_D2 = make_blobs(n_samples = 300, n_features = 2, centers = 8,
                           cluster_std = 1.3, random_state = 4)
    y_D2 = y_D2 % 2

    x_train, x_test, y_train, y_test = KNNService.DataSet().split_on_test_and_train(X_D2,y_D2);
    knn = KNNRegresorService.RegressorTrain()
    knn.setXYtest(x_test,y_test)
    knn.setXYtrain(x_train,y_train)
    result = knn.TrainScaledData()
    k_best = result[1] # 'compute the best k'
    score_best = result[0]*100  # 'compute the best score'
    print('The best k = {} , score = {}'.format(k_best, score_best))
    plotter = KNNService.ShowPlot()
    knn.visualizeScaled(plotter);
    plt.figure()
    plot_decision_regions(X_D2,y_D2,clf=knn.getScalledKnn(),legend=2,)
    plt.show()


SyntheticDataset()