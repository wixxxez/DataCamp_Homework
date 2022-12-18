import matplotlib.pyplot as plt
import  numpy as np

colors = np.array(['green', 'orange', 'purple', 'brown','red'])

def plot_centroids(centroids):
    try:
        for i, c in enumerate (centroids.to_numpy()):
            plt.plot(c[0], c[1], marker = 'x', color= colors[i], markersize=20, linewidth= 25)
    except AttributeError:
        for i, c in enumerate (centroids):
            plt.plot(c[0], c[1], marker = 'x', color= colors[i], markersize=20, linewidth= 25)

def draw_state(x,y, centroids = None, closest_centroids = None):
    plt.figure ()
    plot_points(x,y, closest_centroids)
    plot_centroids(centroids)

def plot_points(x,y, closest_centroids = None):

    if closest_centroids is None:
        plt.scatter(x,y)
    else:
        plt.scatter(x,y, c= colors[closest_centroids])
