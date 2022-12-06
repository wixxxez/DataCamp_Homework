from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics as mt

X,y = load_boston(return_X_y=True);

df = pd.DataFrame (X, columns= ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'])
X = df[[ 'RM', 'LSTAT']]
X = np.array(X).reshape(-1,2)
y = np.array(y).reshape(-1,1)

print(X.shape)
print(y.shape)
x_train, x_test,y_train,y_test = train_test_split(X,y)

linear_model = LinearRegression()
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

linear_model.fit(x_train_scaled, y_train)
print ('Linear Regression')
print ('R2 train score =',linear_model.score(x_train_scaled, y_train))
print ('R2 test score =', linear_model.score(x_test_scaled, y_test))
print ('b: {}, \nw= {}'.format(linear_model.intercept_, linear_model.coef_))
print("Score: ", linear_model.score(x_test_scaled, y_test))
y_pred = linear_model.predict(x_test_scaled)

print("1) The model explains,", np.round(mt.explained_variance_score(y_test, y_pred) * 100, 2),
      "% variance of the target w.r.t features is")
print("2) The Mean Absolute Error of model is:", np.round(mt.mean_absolute_error(y_test, y_pred), 2))
print("3) The R-Square score of the model is ", np.round(mt.r2_score(y_test, y_pred), 2))

plt.scatter(y_pred,y_test)
X = X.reshape(-1,1)
plt.plot([min(X.tolist()), max(X.tolist())], [min(y_pred), max(y_pred)], color='red')
plt.ylabel('value of house/1000($)')
plt.xlabel('RM + LSTAT + PTRATIO')

plt.show()


