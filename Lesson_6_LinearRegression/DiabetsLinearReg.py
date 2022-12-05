from sklearn.datasets import load_diabetes
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

dataset = load_diabetes()
dataframe = pd.DataFrame (data = dataset. data, columns = dataset. feature_names)
dataframe ["relation"] = dataset. target
correlation = dataframe.corr ()
heatmap = sns.heatmap(correlation, annot = True)
heatmap.set (xlabel = 'IRIS values on x axis',ylabel = 'IRIS values on y axis\t', title = "Correlation matrix of IRIS dataset\n")
plt.show()