from scipy.io import loadmat
from sklearn.model_selection import train_test_split
import  os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
import h5py
X, y = load_breast_cancer(return_X_y= True)
print ('X.shape= ', X.shape)
print ('y.shape= ', y.shape)
cwd= os.getcwd() # current working directory
path = os.path.join(cwd,'data')
# YOUR_CODE.  Preproces data, train classifier and evaluate the perfromance on train and test sets
# START_CODE
x_train, x_test , y_train, y_test = train_test_split(X, y);
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)
clf = MLPClassifier(
    solver= 'lbfgs',
    hidden_layer_sizes= (75,25),
    random_state = 0,
    alpha=5
).fit(x_train_scaled,y_train)

print("train accuracy= {:.3%}".format(clf.score(x_train_scaled,y_train)))
print("test accuracy= {:.3%}".format(clf.score(x_test_scaled,y_test)))
# END_CODE

def load_dataset():
    fn = os.path.join(path, 'train_signs.h5')
    train_dataset = h5py.File(fn, "r")
    X_train = np.array(train_dataset["train_set_x"][:])  # your train set features
    y_train = np.array(train_dataset["train_set_y"][:])  # your train set labels

    fn = os.path.join(path, 'test_signs.h5')
    test_dataset = h5py.File(fn, "r")
    X_test = np.array(test_dataset["test_set_x"][:])  # your test set features
    y_test = np.array(test_dataset["test_set_y"][:])  # your test set labels

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    y_train = y_train.reshape((1, y_train.shape[0]))
    y_test = y_test.reshape((1, y_test.shape[0]))

    return X_train, y_train, X_test, y_test, classes

X_train, y_train, X_test, y_test, classes = load_dataset()
y_train = y_train.ravel()
y_test = y_test.ravel()
print ('X_train.shape=', X_train.shape)
print ('X_test.shape=', X_test.shape)
print ('y_train.shape=', y_train.shape)
print ('y_test.shape=', y_test.shape)
def display_samples_in_grid(X, n_rows, n_cols= None, y = None ):
    if n_cols is None:
        n_cols= n_rows
    indices = np.random.randint(0, len(X),n_rows*n_cols)
    for i in range (n_rows):
        for j in range (n_cols):
            index = n_rows*i+j
            ax = plt.subplot(n_rows,n_cols,index+1)
            plt.imshow(X[indices[index]])
            if not (y is None):
                plt.title(y[indices[index]])
            plt.axis('off')

    plt.tight_layout(h_pad=1)

plt.figure()

display_samples_in_grid(X_train, n_rows=4, y= y_train)

plt.gcf().canvas.set_window_title('Train set')

scaler = StandardScaler()
X_train = X_train.reshape(1080,64*64*3)
x2t = X_test
X_test = X_test.reshape(120, 64*64*3)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print ("number of training examples = " + str(X_train_scaled.shape[1]))
print ("number of test examples = " + str(X_test_scaled.shape[1]))
print ("X_train_scaled shape: " + str(X_train_scaled.shape))

print ("X_test_scaled shape: " + str(X_test_scaled.shape))
# YOUR_CODE.  Train classifier and evaluate the perfromance on train and test sets
# START_CODE
clf= MLPClassifier(
    solver= 'lbfgs',
    #hidden_layer_sizes=(100,100),
    random_state=0,
    #alpha=3

).fit(X_train_scaled, y_train)
print("train accuracy= {:.3%}".format(clf.score(X_train_scaled,y_train)))
print("test accuracy= {:.3%}".format(clf.score(X_test_scaled, y_test)))
# END_CODE
plt.figure()

predicted = clf.predict(X_test_scaled)

display_samples_in_grid(x2t, n_rows=4, y= predicted)
plt.gcf().canvas.set_window_title('Test set prediction')


plt.show()