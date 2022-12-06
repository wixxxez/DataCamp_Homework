from sklearn.datasets import load_diabetes
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
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
dataset = Dataset.DataSet(load_diabetes)
x_train, x_test, y_train, y_test = dataset.getTrainTestSplit()
scaler = StandardScaler()
Polynom = Polynom.Polynomial()
x_train, x_test = dataset.getScaledData(x_train,x_test)
x_train, x_test = Polynom.getScaledData(x_train,x_test,2); # activate polinom
#regressor = LinearRegression.LinearRegress()
regressor = Ridge.RidgeRegression()
#regressor = Lasso.LassoRegression()
regressor.setTestTrainData(x_train,x_test,y_train,y_test)
regressor.fitRegression();
print ('R2 train score =',regressor.getTrainScoreRegression())
print ('R2 test score =', regressor.getTestScoreRegression())
print (regressor.getCoef())
y_pred = regressor.predict()


