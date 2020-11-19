"""
Implementations basic unsupervised classification algorithms, largely based on scipy.
"""
import numpy as np
from hylite import HyData
from hylite.analyse.sam import spectral_angles
from scipy.cluster.hierarchy import average, dendrogram, fcluster
import matplotlib.pyplot as plt
from hylite.filter import PCA

def cluster_hierarchical( data, nclasses, distance='SAM', vb=False, labels=None, **kwds ):
    """
    Clusters a HyData instance using a hierarchical clustering algorithm and the
    specified distance metric. Note that this can be very slow for large datasets, but
    can work well for spectral libraries or small images.

    *Arguments*:
     - data = a HyData instance to cluster.
     - nclasses = the number of classes to extract (level to slice the tree at).
     - distance = the distance metric to use. Options are 'SAM' (spectral angle) and 'pca' (use top n principal components).
     - vb = True if a preview plot of the dendrogram should be created.
     - labels = labels for the dendrogram, or None (default).

    *Keywords*:
     - nbands = the number of mnf bands to use if distance = MNF. Default is 10.

    *Returns*:
     - labels = an integer array of class labels.
     - Z = the linkage matrix returned by scipy.
    """

    assert isinstance( data, HyData ), "Error - dataset is not a HyData instance."


    # calculate distance matrix
    if 'sam' in distance.lower(): # calculate spectral angle distance matrix
        X = data.get_raveled()
        M = spectral_angles( X, X )
    elif 'pca' in distance.lower(): # calculate euclidian distance between PCA bands
        # calculate PCA
        pca = PCA(data, kwds.get("pca", 10))
        X = pca.get_raveled()

    else:
        assert False, "%s is an unknown distance type." % distance

    # do clustering
    Z = average(M)
    if vb:
        dendrogram(Z, labels=labels)
        plt.show()

    # cut tree
    C = fcluster( Z, nclasses, criterion='maxclust')

    if data.is_image():
        C = C.reshape( data.xdim(), data.ydim() )

    return C, Z