"""
A collection of functions for performing histogram equalisation and other similar colour-matching routines.
"""

import numpy as np

def hist_eq(adj, ref):
    """
    Perform histogram normalisation on two datasets, as describe at: https://stackoverflow.com/questions/32655686/histogram-matching-of-two-images-in-python-2-x.

    *Arguments*:
     - adj = the source data to transform
     - ref = data following the distribution to be matched too.
    *Returns*
     - an array of the same shape as source that contains the transformed data.
    """
    olddtype = adj.dtype
    oldshape = adj.shape
    adj = adj.ravel()
    ref = ref.ravel()
    src_msk = np.isfinite(adj)
    tmp_msk = np.isfinite(ref)

    if np.isfinite(adj).any() and np.isfinite(ref).any():  # we need at least some finite values
        s_values, bin_idx, s_counts = np.unique(adj[src_msk], return_inverse=True,
                                                return_counts=True)
        t_values, t_counts = np.unique(ref[tmp_msk], return_counts=True)
        s_quantiles = np.cumsum(s_counts).astype(np.float64)
        s_quantiles /= s_quantiles[-1]
        t_quantiles = np.cumsum(t_counts).astype(np.float64)
        t_quantiles /= t_quantiles[-1]
        interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)
        interp_t_values = interp_t_values.astype(olddtype)

        # update output and return
        adj[src_msk] = interp_t_values[bin_idx]

    return adj.reshape(oldshape)


def norm_eq(adj, adj_s, ref_s, per_band=True, inplace=False):
    """
    Match image colour by assuming normally distributed data, and so
    transform adj so it has the same mean and standard deviation as ref.

    (loosely based on https://www.cs.tau.ac.il/~turkel/imagepapers/ColorTransfer.pdf )

    *Arguments*
     - adj = the source data to transform.
     - adj_s = subset of adj to calculate statistics from (e.g. matching pixels).
     - ref_s = reference subset to match statistics too (e.g. matching pixels from reference image).
     - per_band = True (default) if the correction should be applied to each band separately. This can
                  potentially introduce spectral artefacts so use with care. If False, the whole dataset is
                  scaled/recentered at once, so the spectra will not be distorted.
     - inplace = True if data should be modified in place rather than being copied. Default is False.
    """

    assert isinstance(adj, np.ndarray), "Adj must be numpy array."
    assert isinstance(adj_s, np.ndarray), "Ref must be numpy array."
    assert isinstance(ref_s, np.ndarray), "Ref must be numpy array."

    # copy?
    if not inplace:
        adj = adj.copy()

    # calculate means and standard deviation of each channel
    if per_band:
        a = tuple(range(len(adj_s.shape) - 1))  # apply per band
    else:
        a = tuple(range(len(adj_s.shape)))  # apply to whole dataset

    Mu1 = np.nanmean(adj_s, axis=a)
    Mu2 = np.nanmean(ref_s, axis=a)
    Sd1 = np.nanstd(adj_s, axis=a)
    Sd2 = np.nanstd(ref_s, axis=a)

    adj -= Mu1  # center on mean
    adj *= Sd2 / Sd1  # match standard deviations (basically adjust contrast)
    adj += Mu2  # recenter onto target mean

    return adj