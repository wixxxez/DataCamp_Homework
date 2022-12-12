import os
import numpy as np
import h5py
from sklearn.preprocessing import  StandardScaler
cwd= os.getcwd() # current working directory
path = os.path.join(cwd,'data')

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

def scale_data(X_train, X_test):
    scaler = StandardScaler()
    X_train = X_train.reshape(1080, 64 * 64 * 3)
    X_test = X_test.reshape(120, 64 * 64 * 3)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled
