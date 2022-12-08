import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree


from mlxtend.plotting import plot_decision_regions

def plot_labeled_decision_regions(X, y, models):

    if not isinstance(X, pd.DataFrame):
        raise Exception('''X has to be a pandas DataFrame with two numerical features.''')
    if not isinstance(y, pd.Series):
        raise Exception('''y has to be a pandas Series corresponding to the labels.''')
    fig, ax = plt.subplots( figsize=(10.0, 5),)
    plot_decision_regions(X.values, y.values, models, legend=2, ax=ax)
    ax.set_title(models.__class__.__name__)
    ax.set_xlabel(X.columns[0])

    ax.set_ylabel(X.columns[1])
    ax.set_ylim(X.values[:, 1].min(), X.values[:, 1].max())
    ax.set_xlim(X.values[:, 0].min(), X.values[:, 0].max())
    plt.tight_layout()


