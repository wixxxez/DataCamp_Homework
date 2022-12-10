import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mlxtend.plotting import plot_decision_regions


def plot_decision_boundary(clf, X_train, y_train, X_test=None, y_test=None, title=None, precision=0.01,
                           plot_symbol_size=50, ax=None, is_extended=True):
    '''
    Draws the binary decision boundary for X that is nor required additional features and transformation (like polynomial)
    '''
    # Create color maps - required by pcolormesh
    from matplotlib.colors import ListedColormap
    colors_for_points = np.array(['grey', 'orange'])  # neg/pos
    colors_for_areas = np.array(['grey', 'orange'])  # neg/pos  # alpha is applied later
    cmap_light = ListedColormap(colors_for_areas)

    mesh_step_size = precision  # .01  # step size in the mesh
    if X_test is None or y_test is None:
        show_test = False
        X = X_train
    else:
        show_test = True
        X = np.concatenate([X_train, X_test], axis=0)
    x1_min, x1_max = X[:, 0].min() - .1, X[:, 0].max() + 0.1
    x2_min, x2_max = X[:, 1].min() - .1, X[:, 1].max() + 0.1
    # Create grids of pairs
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, mesh_step_size),
                           np.arange(x2_min, x2_max, mesh_step_size))
    # Flatten all samples
    target_samples_grid = (np.c_[xx1.ravel(), xx2.ravel()])

    print(
        'Call prediction for all grid values (precision of drawing = {},\n you may configure to speed up e.g. precision=0.05)'.format(
            precision))
    Z = clf.predict(target_samples_grid)

    # Reshape the result to original meshgrid shape
    Z = Z.reshape(xx1.shape)

    if ax:
        plt.sca(ax)

    # Plot all meshgrid prediction
    plt.pcolormesh(xx1, xx2, Z, cmap=cmap_light, alpha=0.2)

    # Plot train set
    plt.scatter(X_train[:, 0], X_train[:, 1], s=plot_symbol_size,
                c=colors_for_points[y_train.ravel()], edgecolor='black', alpha=0.6)
    # Plot test set
    if show_test:
        plt.scatter(X_test[:, 0], X_test[:, 1], marker='^', s=plot_symbol_size,
                    c=colors_for_points[y_test.ravel()], edgecolor='black', alpha=0.6)
    if is_extended:
        # Create legend
        import matplotlib.patches as mpatches  # use to assign lavels for colored points
        patch0 = mpatches.Patch(color=colors_for_points[0], label='negative')
        patch1 = mpatches.Patch(color=colors_for_points[1], label='positive')
        plt.legend(handles=[patch0, patch1])
    plt.title(title)
    if is_extended:
        plt.xlabel('feature 1')
        plt.ylabel('feature 2')
    else:
        plt.tick_params(
            top=False,
            bottom=False,
            left=False,
            labelleft=False,
            labelbottom=False
        )


def plot_data_logistic_regression(X, y, legend_loc=1, title=None):
    '''
    :param X: 2 dimensional ndarray
    :param y:  1 dimensional ndarray. Use y.ravel() if necessary
    :return:
    '''

    positive_indices = (y == 1)
    negative_indices = (y == 0)
    #     import matplotlib as mpl
    colors_for_points = ['grey', 'orange']  # neg/pos

    plt.scatter(X[negative_indices][:, 0], X[negative_indices][:, 1], s=40, c=colors_for_points[0], edgecolor='black',
                label='negative', alpha=0.7)
    plt.scatter(X[positive_indices][:, 0], X[positive_indices][:, 1], s=40, c=colors_for_points[1], edgecolor='black',
                label='positive', alpha=0.7)
    plt.title(title)
    plt.legend(loc=legend_loc)

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



def plot_decision_boundary_universal(clf, X_train, y_train, X_test=None, y_test=None, title=None, precision=0.1,
                                     plot_symbol_size=50, ax=None, is_extended=True, labels=None, features=None):
    '''
    expected to be universal for binary and multiclass classification but not tested for binary
    '''
    # Create color maps - required by pcolormesh
    from matplotlib.colors import ListedColormap
    colors_for_areas = colors_for_points = np.array(['green', 'grey', 'orange', 'brown'])
    cmap_light = ListedColormap(colors_for_areas)

    mesh_step_size = precision  # .01  # step size in the mesh
    if X_test is None or y_test is None:
        show_test = False
        X = X_train
    else:
        show_test = True
        X = np.concatenate([X_train, X_test], axis=0)
    x1_min, x1_max = X[:, 0].min() - .1, X[:, 0].max() + 0.1
    x2_min, x2_max = X[:, 1].min() - .1, X[:, 1].max() + 0.1
    # Create grids of pairs
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, mesh_step_size),
                           np.arange(x2_min, x2_max, mesh_step_size))
    # Flatten all samples
    target_samples_grid = (np.c_[xx1.ravel(), xx2.ravel()])
    if precision < 0.05:
        print(
            'Calling to predict for all grid values (precision of drawing = {},\n you may configure to speed up e.g. precision=0.05)'.format(
                precision))

    Z = clf.predict(target_samples_grid)

    # Reshape the result to original meshgrid shape
    Z = Z.reshape(xx1.shape)

    if ax:
        plt.sca(ax)

    # Plot all meshgrid prediction
    plt.pcolormesh(xx1, xx2, Z, cmap=cmap_light, alpha=0.2)

    # Plot train set
    plt.scatter(X_train[:, 0], X_train[:, 1], s=plot_symbol_size,
                c=colors_for_points[y_train.ravel()], edgecolor='black', alpha=0.6)
    # Plot test set
    if show_test:
        plt.scatter(X_test[:, 0], X_test[:, 1], marker='^', s=plot_symbol_size,
                    c=colors_for_points[y_test.ravel()], edgecolor='black', alpha=0.6)
    if is_extended:

        # Create legend
        if labels is None:
            labels = ['negative', 'positive']  # assume this is for binary or for muticlass with labels
        import matplotlib.patches as mpatches  # use to assign lavels for colored points
        patches = [mpatches.Patch(color=colors_for_points[i], label=labels[i]) for i in range(len(labels))]
        plt.legend(handles=patches)
        if features is None:
            plt.xlabel('feature 1')
            plt.ylabel('feature 2')
        else:
            plt.xlabel(features[0])
            plt.ylabel(features[1])

    else:
        plt.tick_params(
            top=False,
            bottom=False,
            left=False,
            labelleft=False,
            labelbottom=False
        )
    plt.title(title)

