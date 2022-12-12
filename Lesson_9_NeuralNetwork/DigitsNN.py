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
from DisplayService import display_samples_in_grid
from DataSet import load_dataset
from DataSet import scale_data

def start_CancerNN():
    # YOUR_CODE.  Preproces data, train classifier and evaluate the perfromance on train and test sets
    # START_CODE
    X, y = load_breast_cancer(return_X_y=True)
    x_train, x_test, y_train, y_test = train_test_split(X, y);
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    clf = MLPClassifier(
        solver='lbfgs',
        hidden_layer_sizes=(75, 25),
        random_state=0,
        alpha=5
    ).fit(x_train_scaled, y_train)

    print("train accuracy= {:.3%}".format(clf.score(x_train_scaled, y_train)))
    print("test accuracy= {:.3%}".format(clf.score(x_test_scaled, y_test)))
    # END_CODE



def start_SignsNN():
    X_train, y_train, X_test, y_test, classes = load_dataset()
    y_train = y_train.ravel()
    y_test = y_test.ravel()
    print('X_train.shape=', X_train.shape)
    print('X_test.shape=', X_test.shape)
    print('y_train.shape=', y_train.shape)
    print('y_test.shape=', y_test.shape)

    plt.figure()

    display_samples_in_grid(X_train, n_rows=4, y=y_train)

    plt.gcf().canvas.set_window_title('Train set')
    x2t = X_test

    X_train_scaled , X_test_scaled = scale_data(X_train, X_test)

    print("number of training examples = " + str(X_train_scaled.shape[1]))
    print("number of test examples = " + str(X_test_scaled.shape[1]))
    print("X_train_scaled shape: " + str(X_train_scaled.shape))
    print("X_test_scaled shape: " + str(X_test_scaled.shape))
    # YOUR_CODE.  Train classifier and evaluate the perfromance on train and test sets
    # START_CODE
    clf = MLPClassifier(
        solver='lbfgs',
        # hidden_layer_sizes=(100,100),
        random_state=0,
        # alpha=3

    ).fit(X_train_scaled, y_train)
    print("train accuracy= {:.3%}".format(clf.score(X_train_scaled, y_train)))
    print("test accuracy= {:.3%}".format(clf.score(X_test_scaled, y_test)))
    # END_CODE
    plt.figure()

    predicted = clf.predict(X_test_scaled)

    display_samples_in_grid(x2t, n_rows=4, y=predicted)
    plt.gcf().canvas.set_window_title('Test set prediction')

    plt.show()

start_CancerNN()
start_SignsNN()