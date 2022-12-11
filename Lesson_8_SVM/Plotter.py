import matplotlib.pyplot as plt
import numpy as np

def plot_decision_boundary(clf, X_train, y_train, X_test=None, y_test=None, title=None, precision=0.05,
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


def plot_decision_boundary_poly(clf, X_train, y_train, degree, X_test=None, y_test=None, title=None, precision=0.05,
                                plot_symbol_size=50, ax=None, is_extended=True):
    '''
    Draws the binary decision boundary for X that is nor required additional features and transformation (like polynomial)
    '''
    from sklearn.preprocessing import PolynomialFeatures
    poly = PolynomialFeatures(degree=degree, include_bias=False)

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
    target_samples_grid_poly = poly.fit_transform(target_samples_grid)
    print(
        'Call prediction for all grid values (precision of drawing = {},\n you may configure to speed up e.g. precision=0.05)'.format(
            precision))
    Z = clf.predict(target_samples_grid_poly)
    print('Computing prediction completed.')
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


def plot_data_logistic_regression(X, y, legend_loc=None, title=None):
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


def plot_multi_class_logistic_regression(X, y, dict_names=None, colors=None, title=None):
    '''
    Draw the multi class samples of 2 features
    :param X: X 2 ndarray (m,2),
    :param y: vector (m,)
    :param dict_names: dict of values of y and names
    :return: None
    '''
    if not colors:
        colors_for_points = ['green', 'grey', 'orange', 'brown']
    else:
        colors_for_points = colors

    y_unique = list(set(y))

    for i in range(len(y_unique)):
        ind = y == y_unique[i]  # vector

        if dict_names:
            plt.scatter(X[ind, 0], X[ind, 1], c=colors_for_points[i], s=40, label=dict_names[y_unique[i]],
                        edgecolor='black', alpha=.7)
        else:
            plt.scatter(X[ind, 0], X[ind, 1], s=40, c=colors_for_points[i], edgecolor='black', alpha=0.7)
    if title:
        plt.title(title)

    if dict_names:
        plt.legend(frameon=True)


def draw_linear_decision_boundaries_multiclass(clf, X, ):
    colors = ['green', 'grey', 'orange', 'brown']
    x_line = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
    for w, b, color in zip(clf.coef_, clf.intercept_, colors):
        # Since class prediction with a linear model uses the formula y = w_0 x_0 + w_1 x_1 + b,
        # and the decision boundary is defined as being all points with y = 0, to plot x_1 as a
        # function of x_0 we just solve w_0 x_0 + w_1 x_1 + b = 0 for x_1:
        y_line = -(x_line * w[0] + b) / w[1]
        ind = (X[:, 0].min() < x_line) & (x_line < X[:, 0].max()) & (X[:, 1].min() < y_line) & (y_line < X[:, 1].max())
        plt.plot(x_line[ind], y_line[ind], '-', c=color, alpha=.8)