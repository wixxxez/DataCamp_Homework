from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import graphviz
from sklearn.model_selection import train_test_split
import os
from sklearn.datasets import make_blobs

class DecisionTreeVisualizeService():

    def plot_decision_boundary(self, clf, X_train, y_train, X_test=None, y_test=None, title=None, precision=0.05,
                               plot_symbol_size=50, ax=None, is_extended=True, title_size=None):
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

        fontdict = {'fontsize': title_size} if title_size else None

        plt.title(title, fontdict)
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

    def plotDecisonBoundary(self, X_train , y_train):

        plt.figure(figsize=(10, 3))
        max_depths = [3, 5, 10]

        y_train = y_train % 2  # make it binary since make_blobs  with centers = 8 creates y in [0..7]
        for i, max_depth in enumerate(max_depths):
            ax = plt.subplot(1, len(max_depths), i + 1)
            clf = DecisionTreeClassifier(
                criterion='entropy',
                random_state=20,
                max_depth=max_depth,
                #     max_leaf_nodes=4,
            ).fit(X_train, y_train)
            accuracy = clf.score(X_train, y_train)
            #     print("train accuracy= {:.3%}".format(accuracy))
            self.plot_decision_boundary(
                clf,
                X_train,
                y_train,
                precision=0.05,
                ax=ax,
                title='max_depth={}. accuracy = {:.3%}'.format(max_depth, accuracy),
            )

        plt.tight_layout(w_pad=-2)

    def getTreeGraph(self, clf, features, labels):
        graph_viz = tree.export_graphviz(clf, out_file=None, feature_names=features, class_names=labels, filled=True)
        graph = graphviz.Source(graph_viz)
        graph.view(cleanup =True) # cleanup (bool) – Delete the source file after rendering.