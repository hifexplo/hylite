"""
PCA and MNF methods for dimensionality reduction.
"""

import numpy as np

import spectral
from hylite import HyData

def PCA(hydata, bands=20, band_range=None, step=5, mask : np.ndarray = None):
    """
    Apply a PCA dimensionality reduction to the hyperspectral dataset using singular vector decomposition (SVD).

    Args:
        data: the dataset (HyData object) to apply PCA to.
        output_bands: number of bands to return (i.e. how many dimensions to retain). Default is 20.
        bands: the spectral range to perform the PCA over. If (int,int) is passed then the values are treated as
                    min/max band IDs, if (float,float) is passed then values are treated as wavelenghts (in nm). If None is
                    passed (default) then the PCA is computed using all bands. Note that wavelengths can only be passed
                    if image is a hyImage object.
        step: subsample the dataset during SVD for performance reason. step = 1 will include all pixels in the calculation,
              step = n includes every nth pixel only. Default is 5 (as most images contain more than enough pixels to
              accurately estimate variance etc.).
        mask: A mask containing pixels to ignore during the fitting of this PCA. Default is None (consider all pixels).
    Returns:
        A tuple containing:

        - bands = A HyData instance containing the PCA components, ordered from highest to lowest variance.
        - factors = the factors (vector) each band is multiplied with to give the corresponding PCA band.
        - means = the means for each band that are subtracted before applying the PCA transform.

        Additional info (including loadings and the per-band means) are stored in the header file of the returned HyData
        instance.
    """

    # get numpy array
    wav = None
    decomp = False
    if isinstance(hydata, HyData):
        wav = hydata.get_wavelengths()

        if hydata.is_int():
            hydata.decompress()  # MNF doesn't work very well with ints....
            decomp = True  # so we can compress again afterwards

        data = hydata.data.copy()
    else:
        data = hydata.copy()

    # get band range
    if band_range is None:  # default to all bands
        minb = 0
        maxb = data.shape[-1]
    else:
        if isinstance(band_range[0], int) and isinstance(band_range[1], int):
            minb, maxb = band_range
        else:
            assert isinstance(hydata, HyData), "Error - no wavelength information found."
            minb = hydata.get_band_index(band_range[0])
            maxb = hydata.get_band_index(band_range[1])

    # prepare feature vectors
    if mask is not None:
        X = data[mask, : ]
    else:
        X = data[..., :].reshape(-1, data.shape[-1])

    # print(minb,maxb)
    X = X[::step, minb:maxb]  # subsample
    X = X[np.isfinite(np.sum(X, axis=1)), :]  # drop vectors containing nans
    X = X[np.sum(X, axis=1) > 0, :]  # drop vectors containing all zeros

    # calculate mean and center
    mean = np.mean(X, axis=0)
    X = X - mean[None, :]

    # calculate covariance
    cov = np.dot(X.T, X) / (X.shape[0] - 1)

    # and eigens (sorted from biggest to smallest)
    eigval, eigvec = np.linalg.eig(cov)
    idx = np.argsort(np.abs(eigval))[::-1]
    eigvec = eigvec[:, idx]
    eigval = np.abs(eigval[idx])

    # project data
    data = data[..., minb:maxb] - mean
    #out = np.zeros_like(data)
    #for b in range(min(bands, data.shape[-1])):
    #    out[..., b] = np.dot(data, eigvec[:, b])
    out = np.dot(data, eigvec)

    # compute variance percentage of each eigenvalue
    eigval /= np.sum(eigval)  # sum to 1

    # compress?
    if decomp:
        hydata.compress()

    # prepare output
    bands = min(bands, out.shape[-1])
    if isinstance(hydata, HyData):
        outobj = hydata.copy(data=False)
        outobj.data = out[..., 0:bands]
    else:
        outobj = HyData( out[..., 0:bands] )
    outobj.set_wavelengths(np.cumsum(eigval[0:bands]))  # wavelengths are % of explained variance
    outobj.header['mean'] = mean
    for n in range(bands):
        outobj.header['L_%d'%n] = eigvec[:, n]
    return outobj, eigvec[:, :bands ].T, mean


def MNF(hydata, bands=20, band_range=None, noise='diff', noise_thresh=50, denoise=False, mask : np.ndarray =None):
    """
    Apply a minimum noise filter to a hyperspectral image.

    Args:
        hydata: A HyData instance containing the source dataset (e.g. image or point cloud).
        bands: the number of bands to keep after MNF for dimensionality reduction or denoising. Default is 20.
        band_range: the spectral range to perform the MNF over. If (int,int) is passed then the values are treated as
                    min/max band IDs, if (float,float) is passed then values are treated as wavelenghts (in nm). If None is
                    passed (default) then the MNF is computed using all bands. Note that wavelengths can only be passed
                    if image is a hyImage object.
        noise: The noise model to use. If None (default) then the 'band noise' parameter will be retrieved from the headerinfo,
                and if this does not exist then, this it is crudely estimated by comparing adjacent pixels / points.
                This estimation can be forced by setting noise to 'diff'.
        noise_thresh: The threshold percentile used when estimating noise with 'diff'.
        denoise: True if a MNF denoised image should be returned (rather than the MNF bands). Default is False.
        mask: A boolean mask containing pixels to ignore during the fitting of this PCA. Default is None (consider all pixels).
    Returns:
        A tuple containing:

        - mnf = a HyData instance containing the MNF bands or denoised image.
        - factors = A 2D numpy array containing the factors applied to the input datset. Useful
                     for plotting/interpreting the regions each MNF band is sensitive too.
        - means = the means for each band that are subtracted before applying the MNF transform.
    """

    # prepare data for MNF
    wav = hydata.get_wavelengths()
    decomp = False
    if hydata.is_int():
        hydata.decompress() #MNF doesn't work very well with ints....
        decomp=True #so we can compress again afterwards
    data = hydata.data

    # get range of bands to include in calculation
    if band_range is None:  # default to all bands
        minb = 0
        maxb = data.shape[-1]
    else:
        minb = hydata.get_band_index( band_range[0] )
        maxb = hydata.get_band_index( band_range[1] )

    assert minb < maxb, "Error - invalid range... band_range[0] > band_range[1]??"
    assert minb < data.shape[-1], "Error - band_range[0] out of range."
    if maxb == -1 or maxb > data.shape[-1]: maxb = data.shape[-1]

    # remove invalid bands
    valid_bands = []
    for b in range(minb,maxb):
        if np.isfinite(data[..., b]).any() \
                and not (np.nanmax(data[..., b]) == 0).all():
            valid_bands.append(b)

    #remove invalid bands
    data = np.array(data[..., valid_bands])

    # warn if bands have negative values...
    if np.nanmin(data) < 0.0:
        print("Warning - image contains negative pixels. This can cause unstable behaviour...")

    # calculate signal stats (as in spectral.calc_stats(...) but allowing for nans)
    if mask is not None:
        X = data[mask, : ].T
    else:
        X = data.reshape(-1, data.shape[-1]).T  # reshape to 1D list of pixels for each band

    X = X[:, np.isfinite(np.sum(X, axis=0))]  # drop columns containing nans
    X = X[:, np.sum(X, axis=0) > 0 ] #drop columns containing all zeros
    dmean = np.mean(X, axis = 1)
    cov = np.cov(X)
    n = X.shape[1]
    signal = spectral.GaussianStats(dmean, cov, n)

    if (noise is None) and 'band noise' in hydata.header:  # get noise from header?
        assert False, "Error - this is not yet implemented! [ hopefully soon! ]"
        noise = np.array( hydata.header.get_list('band noise') )
    if (noise is not None) and (noise != 'diff'): # we have noise info - use it
        assert False, "Error - this is not yet implemented! [ hopefully soon! ]"
        assert noise.shape[0] == hydata.band_count(), "Error - noise provided for %d bands but data has %d" % (noise.shape[0], hydata.band_count())
        mean = noise[valid_bands] # easy
        cov = np.full( (mean.shape[0], mean.shape[0]), 0. ) # assume noise is de-corellated
        noise = spectral.GaussianStats(mean, cov, n)
    else: # we need to guesstimate the noise model from the data
        if len(data.shape) == 3:  # estimate noise by subtracting adjacent pixels
            if data.shape[0] == 1:  # special case - 1-D image
                deltas = data[:, :-1, :] - data[:, 1:, :]
            elif data.shape[1] == 1:
                deltas = data[:-1, :, :] - data[1:, :, :]
            else:
                deltas = data[:-1, :-1, :] - data[1:, 1:, :]
        elif len(data.shape) == 2:  # point cloud data. N.B. this assumes that adjacent points are close in space!
            deltas = data[:-1, :] - data[1:, :]

        X = deltas.reshape(-1, deltas.shape[-1]).T
        X = X[:, np.isfinite(np.sum(X, axis=0))]  # drop columns containing nans
        X = X[:, np.sum(X, axis=0) > 0]  # drop columns containing all zeros
        X = X[:, np.sum(X, axis=0) < np.nanpercentile( np.sum(X,axis=0), noise_thresh ) ] # drop high noise data (these relate to edges)
        assert np.isfinite(X).all(), "Error - the nans snuck into our data!"
        mean = np.mean(X, axis=1)
        cov = np.cov(X)
        #n = X.shape[1]

        noise = spectral.GaussianStats(mean, cov, n)
    mnfr = spectral.mnf(signal, noise)

    # reduce bands
    reduced = mnfr.reduce(data, num=bands)

    #apply sign correction so there are less positive pixels than negative ones (sign is aribrary, helps maintain
    #consistency for plotting etc. by having low-valued background with some high-value regions (<50%)
    sign = np.nanmedian(reduced / np.abs(reduced)) #n.b. this will always be 1.0 or -1.0
    assert np.isfinite(sign), "Weird error - no non-nan values in MNF result?"
    reduced *= sign

    # calculate factors for return
    factors = sign * mnfr.get_reduction_transform(num=bands)._A

    # denoise and export
    if denoise:
        out = hydata.copy(data=False)
        out.data = mnfr.denoise(data, num=bands)
        out.set_wavelengths(hydata.get_wavelengths()[valid_bands])
    else:
        out = hydata.copy(data=False)
        out.header.drop_all_bands()  # drop band specific attributes
        out.data = reduced.reshape(hydata.data.shape[:-1] + (bands,))
        out.push_to_header()

    # compress input dataset (so we don't change it)
    if decomp:
        hydata.compress()

    # put factors in the header
    for i in range(factors.shape[0]):
        out.header['mnf weights %d' % i] = factors[i, :]

    return out, factors, dmean


def from_loadings(data, L, m=None):
    """
    Transform a dataset using a precomputed loading vector.  This allows PCA
    transforms to be computed on one dataset and then applied to another.

    Args:
       data: A dataset (HyData instance or numpy array) with b bands in the last axis.
       L: the loadings vector of shape (k,b), such that data is projected into a
                 k-dimensional space.
       m: the mean of each dimension. Default is None ( do not apply mean offset ).
    Returns:
       a hydata instance or numpy array containing the transformed data.
    """
    # get relevant data
    if isinstance(data, HyData):
        X = data.X()
        outshape = data.data.shape[:-1] + (L.shape[0],)
    else:
        X = data.reshape((-1, data.shape[-1]))
        outshape = data.shape[:-1] + (L.shape[0],)

    # project data
    if m is not None:
        X = X - m
    out = np.dot(X, L.T)

    # add mean
    #if m is not None:
    #    out += m

    # reshape
    out = out.reshape(outshape)

    # return HyData or numpy array
    if isinstance(data, HyData):
        O = data.copy(data=False)
        O.data = out
        return O
    else:
        return out
