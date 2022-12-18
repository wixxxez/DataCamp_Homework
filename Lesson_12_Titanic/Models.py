import  pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import Dataset
import DataPreprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import MinMaxScaler

dataset = Dataset.Dataset()
X = dataset.getXTrain()
Y = dataset.getYTrain()
X = DataPreprocessing.preprocessing(X)

x_train, x_test, y_train, y_test = dataset.getSplitedData(X, Y)

def DecisionTree(x_train, y_train,x_test,y_test):
    print("Decision Tree: ")
    for max_depth in range(3,50,2):
        clf = DecisionTreeClassifier(
            criterion='entropy',
            random_state=20,
            max_depth=max_depth,
        ).fit(x_train,y_train);
        print("Depth: ", max_depth)
        print("train accuracy= {:.3%}".format(clf.score (x_train, y_train)))
        print("test accuracy= {:.3%}".format(clf.score (x_test, y_test)))

def GradientBoost(x_train, y_train, x_test, y_test):


    learning = [0.01 ]
    for learning_rate in learning:
        for max_depth in range(3,15,2):
            clf = GradientBoostingClassifier(
                learning_rate=learning_rate,
                max_depth=max_depth,
                n_estimators=150,
                validation_fraction=0.1
                #     max_leaf_nodes=4,
            ).fit(x_train, y_train.to_numpy().ravel())
            print("Depth: ", max_depth)
            print("Learning rate: ", learning_rate)
            print("train accuracy= {:.3%}".format(clf.score(x_train, y_train)))
            print("test accuracy= {:.3%}".format(clf.score(x_test, y_test)))

def RandomForest(x_train, y_train, x_test = None, y_test = None):

    bestdepth = [7,8,9,10]
    models = []
    model_accuracy = []
    for i in range(10):

        for max_depth in bestdepth:
            clf = RandomForestClassifier(
                criterion='entropy',
                random_state=10,
                max_depth=max_depth,
                n_estimators=150,
                ccp_alpha=0.01
            ).fit(x_train, y_train.to_numpy().ravel())
            print("Depth: ", max_depth)
            #print("Learning rate: ", learning_rate)
            print("train accuracy= {:.3%}".format(clf.score(x_train, y_train)))
            score = clf.score(x_train, y_train)
            #print("test accuracy= {:.3%}".format(score))
            models.append(clf)
            model_accuracy.append(score)

    print(max(model_accuracy), model_accuracy.index(max(model_accuracy)))

    return models[model_accuracy.index(max(model_accuracy))]
def XGBoost(x_train, y_train, x_test, y_test):

    clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    clf.fit(x_train, y_train)
    print("train accuracy= {:.3%}".format(clf.score(x_train, y_train)))
    print("test accuracy= {:.3%}".format(clf.score(x_test, y_test)))

def Predict(clf, x):
    x_v = DataPreprocessing.preprocessing(x)

    return clf.predict(x_v)

def createSubmission(x,y):
    x_validate["Survived"] = y
    sumbission = x_validate["Survived"]
    sumbission.to_csv('submission.csv')

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
#DecisionTree(x_train, y_train, x_test, y_test);
#GradientBoost(x_train, y_train, x_test, y_test)

clf = RandomForest(X, Y)
print(clf)
x_validate = dataset.getTestData()
y_pred = Predict(clf, x_validate)
createSubmission(x_validate,y_pred)

#XGBoost(x_train, y_train, x_test, y_test)