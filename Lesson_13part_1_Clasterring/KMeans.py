import pandas as pd
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
import numpy as np
import Plotter
'''
Steps of implementatiion
Initialize  𝐾  centroids

Assign every point to closest centroid

Compute new centroids as means of samples assigned to corresponding centroid

Implement cost  𝐽=1𝑚∑𝑖(𝑑(𝑖))2  where  𝑑(𝑖)  is distance from sample  𝑥(𝑖)  to assigned centroid  𝑐(𝑖) 
Iterate setps 2,3 till cost is stabile

Select the best clustering (with the least cost) from 100 results computed with random centroid initializations

Visualize results (clusters, centroids) for  𝐾=4 
Note:

You may need develop couple of additional/intermediate functions
You may track cost changes to make sure the algorithm is working properly

'''
Xc_2,_= make_classification(n_samples=200,
                                    n_features=2,
                                    n_informative=2,
                                    n_redundant=0,
                                    random_state=0,
                                    n_clusters_per_class=1,
                                    class_sep = 0.8)



def init_centroids(amount, dset):

    df = pd.DataFrame(dset)
    return df.sample((amount));

def cost(a,b):

    return np.square(np.sum((a-b)**2))

def centroid_assignation(df, centroids):

    x, y = df['x'].to_numpy(), df['y'].to_numpy()
    dset = pd.DataFrame(x, y)
    k = centroids.shape[0]
    n = dset.shape[0]
    assignation = []
    assign_errors = []



    for obs in range(n):
        # Estimate error
        all_errors = np.array([])
        for centroid in range(k):
            err = cost(centroids.iloc[centroid, :], dset.iloc[obs,:])
            all_errors = np.append(all_errors, err)

        # Get the nearest centroid and the error
        nearest_centroid =  np.where(all_errors==np.amin(all_errors))[0].tolist()[0]
        nearest_centroid_error = np.amin(all_errors)

        # Add values to corresponding lists
        assignation.append(nearest_centroid)
        assign_errors.append(nearest_centroid_error)

    return assignation, assign_errors


def kmeans(dset, k=2, tol=1e-4):


    working_dset = pd.DataFrame(dset, columns=['x', 'y'])
    # We define some variables to hold the error, the
    # stopping signal and a counter for the iterations
    err = []
    goahead = True
    j = 0

    # Step 2: Initiate clusters by defining centroids
    centroids = init_centroids(k, dset)
    CRR = centroids
    iterat = 0
    #Plotter.draw_state(working_dset['x'],working_dset['y'], centroids)
    while (goahead):
        # Step 3 and 4 - Assign centroids and calculate error
        working_dset['centroid'], j_err = centroid_assignation(working_dset, centroids)
        err.append(sum(j_err))
        colnames = ['x', 'y']
        # Step 5 - Update centroid position
        centroids = working_dset.groupby('centroid').agg('mean').reset_index(drop = True)
        centroids_draw = centroids.copy()
        centroids1 = centroids['x'].to_numpy()
        centroids2 = centroids['y'].to_numpy()
        centroids = pd.DataFrame(centroids1, centroids2)

        # Step 6 - Restart the iteration
        if j > 0:
            # Break point
            if err[j - 1] - err[j] <= tol:
                goahead = False
        j += 1

        iterat+=1;
    working_dset['centroid'], j_err = centroid_assignation(working_dset, centroids)
    centroids = working_dset.groupby('centroid').agg('mean').reset_index(drop=True)
    #Plotter.draw_state(working_dset['x'], working_dset['y'], centroids=centroids_draw,
    #                   closest_centroids=working_dset['centroid'])
    return working_dset, j_err, centroids, centroids_draw,CRR,err

#df,j, centr = kmeans(Xc_2, 4)
err = []
min = 99999
for i in range(100):
    working_dset, my_errs, centroids, cd, CRR, err = kmeans(Xc_2, 4)
    if min > sum(my_errs):
        working_dset_min = working_dset
        my_errs_min = my_errs
        centroids_min = centroids
        centroids_draw = cd
        CRR_MIN = CRR
        err_min = err
    print(i)

plt.figure()
print("Lowest cost")
print(sum(my_errs_min))
plt.plot(err_min)

Plotter.draw_state(working_dset['x'],working_dset['y'], CRR)
Plotter.draw_state(working_dset['x'], working_dset['y'], centroids=centroids_draw,closest_centroids=working_dset['centroid'])
plt.show()