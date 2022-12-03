import pandas as pd
import numpy as np
import KNNService

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

    print("Best K for no scaled data is:",  accuracyService.getBestK());
    plotter = KNNService.ShowPlot()
    df = accuracyService.getDF()
    plotter.print_df(df,"k","accuracy","Iris: Best K for no scaled data")

    #---------------------------#
    ScaledData = trainService.TrainWithScaledData()
    accuracyService_scaled = KNNService.AccuracyService(ScaledData[1])
    print("Best K for scaled data is: ", accuracyService_scaled.getBestK())
    df_scaled = accuracyService_scaled.getDF();
    plotter.print_df(df_scaled, "k", "accuracy", "Iris: Best K for scaled data")
    plotter.show()

Iris();
