from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


X,y = load_boston(return_X_y=True);

df = pd.DataFrame (X, columns= ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'])
X = df[['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']]
X = np.array(X).reshape(-1,13)
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

plt.scatter(y_pred,y_test)
X = X.reshape(-1,1)
plt.plot([min(X.tolist()), max(X.tolist())], [min(y_pred), max(y_pred)], color='red')
plt.ylabel('value of house/1000($)')
plt.xlabel('RM + LSTAT + PTRATIO')
plt.xlim([5,45])
plt.show()


