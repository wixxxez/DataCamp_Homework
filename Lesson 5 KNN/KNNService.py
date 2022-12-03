import pandas as pd
import numpy as np
import sklearn.preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class DataSet:

    def split_on_test_and_train(self,x,y):

        return train_test_split(x, y)
    def getDf(self):
        pass;
    def getX(self):
        pass;
    def getY(self):
        pass;
class IrisDataSet(DataSet):

    def __init__(self):
        iris = load_iris()

        self.X, self.y, labels, feature_names = iris.data, iris.target, iris.target_names, iris['feature_names']
        df_iris = pd.DataFrame(self.X, columns=feature_names)
        df_iris['label'] = self.y
        features_dict = {k: v for k, v in enumerate(labels)}
        df_iris['label_names'] = df_iris.label.apply(lambda x: features_dict[x])
        self.df_iris = df_iris

    def getDf(self):
        return self.df_iris;

    def getX(self):
        return self.X;

    def getY(self):
        return self.y;


class TrainKNN():

    def set_test_XY(self,x,y):

        self.X_test = x;
        self.Y_test = y

    def set_train_XY(self, x ,y):

        self.X_train = x;
        self.Y_train = y;

    def TrainNoScaledData(self):
        k_list = list(range(1, 50, 2))
        accuracy = []
        for k in k_list:
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(self.X_train, self.Y_train)
            y_pred = knn.predict(self.X_test)
            accuracy.append([accuracy_score(self.Y_test, y_pred), k]);

        return [y_pred , accuracy]

    def TrainWithScaledData(self):

        k_list = list(range(1, 50, 2))
        accuracy = []
        scaler = sklearn.preprocessing.MinMaxScaler();

        x_train_scaled =scaler.fit_transform(self.X_train)
        x_test_scaled = scaler.transform(self.X_test);
        for k in k_list:
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(x_train_scaled, self.Y_train)
            y_pred = knn.predict(x_test_scaled)
            accuracy.append([accuracy_score(self.Y_test, y_pred), k]);

        return [y_pred, accuracy];

class AccuracyService():

    def __init__(self, accuracy):
        self.accuracy = accuracy

    def getBestK(self):

        return self.accuracy[self.accuracy.index(max(self.accuracy))]

    def getDF(self):
        return pd.DataFrame(self.accuracy, columns=['accuracy', 'k'])

class ShowPlot:

    def print_df(self,df,x,y,title):

        df.plot(
            x=x,
            y=y,
            title=title
        )

    def show(self):

        plt.show();

    def plot_decision_boundary(self,x,y,model,**model_params):

        plt.figure()
        x_reduce = x[:,:2]

        model.fit(x,y);
        step_size = 0.02;
        xMin, xMax = x_reduce[:, 0].min()-1, x_reduce[:,0].max()+1
        yMin, yMax = x_reduce[:,1].min()-1, x_reduce[:, 1].max()+1

        xx,yy = np.meshgrid(np.arange(xMin,xMax,step_size), np.arange(yMin,yMax,step_size));

        model_predicted = model.predict(np.c_[xx.ravel(), yy.ravel()]);

    def knnRegressorScoreVisualization(self, scores_train, scores_test, k_range ):
        fig, ((ax1, ax2)) = plt.subplots(nrows=2, ncols=1, sharey=True)
        plt.sca(ax1)
        plt.ylabel('accuracy')
        ax1.scatter(k_range, scores_train)
        plt.sca(ax2)
        plt.xlabel('k')
        plt.ylabel('accuracy')
        ax2.scatter(k_range, scores_test)
        plt.xticks([0, 5, 10, 15, 20]);
        fig.subplots_adjust(hspace=.4)  # set the vertical space
        ax1.set_title('train set');
        ax2.set_title('test set');