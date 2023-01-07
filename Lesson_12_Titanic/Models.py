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
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import  LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
import seaborn as sns
from hyperopt import Trials
from sklearn.ensemble import BaggingClassifier

import optuna
dataset = Dataset.Dataset()
X = dataset.getXTrain()
Y = dataset.getYTrain()
X = DataPreprocessing.preprocessing(X)
print(X.head(40))
x_train, x_test, y_train, y_test = dataset.getSplitedData(X, Y)
from sklearn.model_selection import GridSearchCV
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

def GradientBoost(x_train, y_train, x_test = None, y_test = None):
    models = []
    model_accuracy = []
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
            score = clf.score(x_train, y_train)
            print("train accuracy= {:.3%}".format(score))
            #print("test accuracy= {:.3%}".format(clf.score(x_test, y_test)))
            models.append(clf)
            model_accuracy.append(score)
    return models[model_accuracy.index(max(model_accuracy))]
def RandomForest(x_train, y_train, x_test = None, y_test = None):


            param = {'bootstrap': True,
                 'ccp_alpha': 0.0,
                 'class_weight': None,
                 'criterion': 'gini',
                 'max_depth': None,
                 'max_features': 'auto',
                 'max_leaf_nodes': None,
                 'max_samples': None,
                 'min_impurity_decrease': 0.0,
                 'min_samples_leaf': 1,
                 'min_samples_split': 2,
                 'min_weight_fraction_leaf': 0.0,
                 'n_estimators': 100,
                 'n_jobs': None,
                 'oob_score': False,
                 'random_state': None,
                 'verbose': 0,
                 'warm_start': False}
            clf = RandomForestClassifier(
                **param
            ).fit(x_train, y_train.to_numpy().ravel())
            print("Depth: ", 9)
            #print("Learning rate: ", learning_rate)
            print("train accuracy= {:.3%}".format(clf.score(x_train, y_train)))
            score = clf.score(x_train, y_train)
            #print("test accuracy= {:.3%}".format(score))




            return clf
def XGBoost(x_train, y_train, x_test, y_test):

    clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    clf.fit(x_train, y_train)
    print("train accuracy= {:.3%}".format(clf.score(x_train, y_train)))
    print("test accuracy= {:.3%}".format(clf.score(x_test, y_test)))

def Predict(clf, x_v):


    return clf.predict(x_v)

def createSubmission(x,y):
    x_validate["Survived"] = y
    sumbission = x_validate["Survived"]
    sumbission.to_csv('submissionf.csv')

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
#DecisionTree(x_train, y_train, x_test, y_test);
#GradientBoost(x_train, y_train, x_test, y_test)
X = scaler.fit_transform(X)
x_validate = dataset.getTestData()

def returnScore(params):

    clf = RandomForestClassifier(**params).fit(x_train,y_train)

    return clf.score(x_test, y_test)


def objectives(trial):
    param = {

        "max_depth": trial.suggest_int("max_depth", 2, 15),
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
    }

    return returnScore(param)

def optunaRandomForest():
    opt = optuna.create_study(direction="maximize")
    opt.optimize(objectives, n_trials=300)

    trial = opt.best_trial
    clf_best = opt.best_trial.params
    print(clf_best)
    print(trial.value)

    clf = RandomForestClassifier(**clf_best)
    clf.fit(X, Y)
    print(clf.score(x_test, y_test))
    x_v = DataPreprocessing.preprocessing(x_validate)
    x_v = scaler.transform(x_v)

    createSubmission(x_validate, clf.predict(x_v))
estimator_range = [2,4,6,8,10,12,14,16]
models = []
scores = []
for n in estimator_range:

    clf = BaggingClassifier(n_estimators=16, random_state=22)
    clf.fit(X,Y)
    models.append(clf)
    scores.append(accuracy_score(y_true=Y, y_pred=clf.predict(X)))
    createSubmission(x_validate, clf.predict(scaler.transform(DataPreprocessing.preprocessing(x_validate))))
plt.figure(figsize=(9,6))
plt.plot(estimator_range, scores)

# Adjust labels and font (to make visable)
plt.xlabel("n_estimators", fontsize = 18)
plt.ylabel("score", fontsize = 18)
plt.tick_params(labelsize = 16)

# Visualize plot
plt.show()