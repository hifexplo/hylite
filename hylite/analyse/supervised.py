"""
A list of utility functions for creating test and training datasets from labelled hyperspectral data. Note that we avoid implementing
specific supervised classification algorithms, as scikit-learn already does an excellent job of this. Hence, the following
functions are simply designed to easily extract features and labels that are compatible with scikit-learn.
"""

import numpy as np
from hylite import HyData


def get_feature_vectors(data, labels, ignore=[]):
    """
    Returns a feature vector and associated labels from a HyData instance.

    *Arguments*:
     - data = the dataset (HyData instance) to extract features from.
     - labels = a list of boolean point or pixel masks where True values should be associated with
                that label. Generated label indices will range from 0 to len(labels). Alternatively, if labels is
                a HyData instance (e.g. a classification image), then labels will be extacted from this.
     - ignore = a list of labels to ignore (if labels is a HyData instance). E.g. [ 0 ] will ignore pixels labelled as background.
    *Returns*:
     - F = a list containing a feature array for each class in labels.
     - c = a list of the number of features for each class.
    """

    # build boolean masks from HyData instance if necessary
    if isinstance(labels, HyData):
        # extract unique labels
        ll = np.unique(labels.data)

        # remove ignored labels
        for n in ignore:
            ll = np.delete(ll, np.where(ll == n))

        # sort increasing
        ll = np.sort(ll)

        # build masks
        masks = [labels.data[..., 0] == n for n in ll]

        # return features
        return get_feature_vectors(data, masks)

    # check labels do not overlap...
    assert np.max(np.sum(labels, axis=0)) == 1, "Error - class labels overlap..."

    # reshape image data
    data = data.get_raveled()

    # get features
    F = []
    c = []
    for i, lab in enumerate(labels):
        mask = lab.reshape(data.shape[0]).astype(np.bool)
        F.append(data[mask])
        c.append(np.sum(mask))

    return F, c


def balance( F, n=1.0):
    """
    Samples a balanced feature vector from a list of features, as returned by
    get_feature_vectors( ... ).

    *Arguments*:
     - F = a list containing an array of features for each class.
     - n = the number of features to extract. Default is None (extract as mean features as possible).
           If a float between 0 and 1 is passed then it is treated as a fraction of the maximum number of features.
           If an integer is passed then this number of features will be extracted (or max(counts)).

    *Returns*:
     - X = a balanced feature feature vector with dimensions N_samples x M_features.
     - y = an array of length N_samples containing labels for each feature (ranging from 0 - n_classes).
    """

    c = [f.shape[0] for f in F]
    if n > 0 and n <= 1:
        n = int(n * np.min(c))
    else:
        n = int(n)
        assert n < np.min(c), "Error - unsufficient training data (%d < %d)" % (n, np.min(c))

    # balance dataset
    X = []
    y = []
    for i, f in enumerate(F):
        idx = np.random.choice(f.shape[0], n, replace=False)
        X.append(f[idx])
        y.append(np.full((n), i))

    # shuffle (just because)
    X = np.vstack(X)
    y = np.hstack(y)
    idx = np.random.choice(y.shape[0], y.shape[0], replace=False)
    return X[idx, :], y[idx]


def split(X, y, frac=0.5):
    """
    Randomly split a labeled feature set into testing and training sets.

    *Arguments*:
     - X = the feature set to split.
     - y = the label set to split.
     - frac = the fraction of train vs test datasets. Default is 0.5 (50%).

    *Returns*:
     - train_X, train_y, test_X, test_y = training and testing features and labels.
    """

    # extract training dataset
    n_train = int(X.shape[0] * frac)
    mask = np.full(X.shape[0], False)
    mask[np.random.choice(X.shape[0], n_train, replace=False)] = True
    return X[mask, :], y[mask], X[np.logical_not(mask), :], y[np.logical_not(mask)]