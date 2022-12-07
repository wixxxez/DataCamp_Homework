from sklearn.datasets import load_diabetes
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression as linr
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
import Dataset
import LinearRegression
import Ridge
import Lasso
import Polynom
import Factory
def Start(factory: Factory.Factory, polynomial: bool = False, polynomial_degrees  : int = 2, use_one_feature  : bool = False):

    dataset = Dataset.DataSet(load_diabetes, use_one_feature)
    x_train, x_test, y_train, y_test = dataset.getTrainTestSplit()

    x_t = x_test

    x_train, x_test = dataset.getScaledData(x_train, x_test)
    if (polynomial):
        polynom = Polynom.Polynomial()
        x_train, x_test = polynom.getScaledData(x_train, x_test, polynomial_degrees);  # activate polinom

    regressor = factory.factory()
    regressor.setTestTrainData(x_train, x_test, y_train, y_test)
    regressor.fitRegression();

    print('R2 train score =', regressor.getTrainScoreRegression())
    print('R2 test score =', regressor.getTestScoreRegression())
    print(regressor.getCoef())
    y_pred = regressor.predict()
    if(use_one_feature):

        viz_values(x_t,y_test, y_pred)
    else:
        plt.figure()
        plt.scatter(y_test, y_pred, color='black')
        plt.plot(y_test, y_test, color='blue', linewidth=3)
        plt.xlabel('True values')
        plt.ylabel('Predicted values')
    plt.show()

def viz_values(x,y, y_pred):

    plt.figure()
    plt.scatter(x,y, color = "black")
    plt.plot(x,y_pred, color = 'blue' , linewidth = 3)
    plt.xlabel('feature')
    plt.ylabel('target')

print("Lasso")
Start(Factory.LassFactory(),polynomial=False,polynomial_degrees=5,use_one_feature=False)

print("Ridge")
Start(Factory.RidgeFactory(),polynomial=False,polynomial_degrees=5, use_one_feature=False)

print("LinReg + polynomial")
Start(Factory.RidgeFactory(),polynomial=True,polynomial_degrees=5, use_one_feature=False)
