from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
import h5py # common package to interact with a dataset that is stored on an H5 file.
import scipy
# from PIL import Image
from scipy import ndimage
import os
import Plotter

cwd= os.getcwd() # current working directory
path = os.path.join(cwd,'')
def load_dataset():
    file_name = os.path.join(path, 'train_catvnoncat.h5')
    train_dataset = h5py.File(file_name, "r")
    X_train = np.array(train_dataset["train_set_x"][:])  # your train set features
    Y_train = np.array(train_dataset["train_set_y"][:])  # your train set labels

    file_name = os.path.join(path, 'test_catvnoncat.h5')
    test_dataset = h5py.File(file_name, "r")
    X_test = np.array(test_dataset["test_set_x"][:])  # your test set features
    Y_test = np.array(test_dataset["test_set_y"][:])  # your test set labels

    classes = ['non-cat', 'cat']

    Y_train = Y_train.reshape(-1, 1)
    Y_test = Y_test.reshape(-1, 1)

    return X_train, Y_train, X_test, Y_test, classes

X_train,Y_train, X_test, Y_test, classes = load_dataset()
print ('X_train.shape= ',X_train.shape)
print ('X_test.shape= ',X_test.shape)
print ('Y_train.shape= ',Y_train.shape)
print ('Y_test.shape= ',Y_test.shape)

m_train = X_train.shape[0]
num_px = X_test.shape[1]
m_test = X_test.shape[0]


X_train_flatten = X_train.reshape(m_train, num_px*num_px*3)
X_test_flatten =  X_test.reshape(m_test, num_px*num_px*3)


print ("train_set_x_flatten shape: {}".format(X_train_flatten.shape))
print ("test_set_x_flatten shape: {}".format(X_test_flatten.shape))
print ("sanity check after reshaping: {}".format(X_train_flatten[0, :5]))
X_train_scaled = X_train_flatten/255.
X_test_scaled = X_test_flatten/255.

y_train = np.squeeze(Y_train) # LogisticRegression requires 1d input for y
clf = LogisticRegression(C=0.01).fit(X_train_scaled, y_train)

print("train accuracy= {:.3%}".format(clf.score (X_train_scaled, y_train)))
y_test = np.squeeze(Y_test)
print("test accuracy= {:.3%}".format(clf.score (X_test_scaled, y_test)))
