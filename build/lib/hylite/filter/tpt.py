"""
Use spectra derivatives to extract and analyse turning points (maxima and minima).
"""

import numpy as np
from tqdm import tqdm
from scipy import signal


import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

def build_kernel(sigma, res):
    """
    Construct a kernel (normal distribution with specified standard deviation)
    """
    kx = np.arange(-5 * sigma, 5 * sigma, step=res, dtype=np.float32)
    ky = ((1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((kx / sigma) ** 2))).astype(np.float32)
    ky /= np.trapz(ky, kx)  # increase values slightly to account for numerical integration errors
    return ky


def TPT(data, sigma=10., window=21, n=2, thresh=0, domain=None, weighted=True, maxima=True, minima=True, vb=True):
    """
    Extract the turning points from each spectra in dataset, and build an associated kernel density estimates that
    can be used to calculate the probability of observing given features.

    Args:
        data: a HyData instance containing the spectra to filter.
        sigma: Standard deviation of the KDE kernel. Either a number, or a function such that sigma(w) returns the
               standard deviation for a specific wavelength (e.g. if a larger kernel should be used for
               LWIR than VNIR data). Default is 10 nm.
        window: the size of the window (in indices) used during savgol smoothing and derivative calculation. Default is 21.
        n: the order of the polynomial to use for savgol smoothing. Larger number preserve smaller features but
           are slower to compute (and more prone to noise). Default is 2.
        thresh: a depth/prominence threshold to ignore small maxima or minima. Default is 0.
        domain: None (default) if the KDE should be evaluated over the same wavelengths as the input data. Alternatively, a
                tuple can be passed containing (min_wav, max_wav, [resolution (optional)]).
        weighted: True if the KDE should be weighted based on feature depth (i.e. more prominent features get extra
                  weight in the KDE as they are ~ 'more likely to be observed'). Default is True.
        maxima: True the filter should include maxima. Default is True.
        minima: True if the filter should include minima. Default is True.
        vb: True progress should be printed to the console. Default is True.

    Returns:
        A tuple containing:

         - tpt = a HyData instance containg kernel density estimates of maxima and minima in the each pixel/point. Maxima
                 are given positive weight while minima are given negative weight.
         - Tpos = an array containing a list of turning point positions (wavelength) for each point/pixel.
         - Tdepth = an array containing a list of turning point depths for each point/pixel. Maxima are positive, minima negative.
    """

    # init wavelength domain
    if domain is not None:
        assert isinstance(domain, tuple) and (len(domain) == 3 or len(
            domain) == 2), "Domain must be a tuple of length two or three containing (start_wavelength, end_wavelength, resolution )."
        if (len(domain) == 3):
            w = np.arange(domain[0], domain[1], step=domain[2])
            res = domain[2]
        data = data.export_bands((domain[0], domain[1]))  # subset data
        if (len(domain) == 2):
            w = data.get_wavelengths()
            res = w[1] - w[0]
    else:
        w = data.get_wavelengths()
        res = w[1] - w[0]

    # get flattened reflectance array
    R = data.X()
    assert R.shape[-1] > window, "Error - window > band count. Please reduce window size (or increase band count)."

    # calculate derivatives
    mask = np.isfinite(R).all(axis=-1)
    dy = np.full( R.data.shape, np.nan )
    dy[mask,:] = signal.savgol_filter(R[mask, :], deriv=1, window_length=window, polyorder=n, axis=-1 )

    # init output array
    out = np.zeros((dy.shape[0], w.shape[0]))
    Tpos = [[] for i in range(dy.shape[0])]
    Tdepth = [[] for i in range(dy.shape[0])]
    K = {}
    loop = range(dy.shape[0])
    if vb:
        loop = tqdm(loop, leave=False, desc="Finding turning points")
    for p in loop:
        # find turning points
        minx = np.argwhere((dy[p, 1:] > 0) & (dy[p, :-1] < 0))[:, 0]
        maxx = np.argwhere((dy[p, 1:] < 0) & (dy[p, :-1] > 0))[:, 0]

        # add minima to output
        if minima:
            for i in minx:
                # match feature with adjacent maxima
                left = 0
                right = dy.shape[-1] - 1
                if (maxx < i).any():
                    left = maxx[maxx < i][np.argmin(np.abs(i - maxx[maxx < i]))]
                if (maxx > i).any():
                    right = maxx[maxx > i][np.argmin(np.abs(maxx[maxx > i] - i))]

                # calculate depth and associated weight
                d = max(R[p, left], R[p, right]) - R[p, i]  # compute simple maxima - minima depth
                weight = 1.0
                if weighted:
                    weight = d / max(R[p, left],
                                     R[p, right])  # weight is the fractional decrease associated with this minima

                # calculate index in output array
                if domain is not None:
                    idx = int((data.get_wavelengths()[i] - w[0]) / res)  # compute idx in ouput array
                else:
                    idx = i  # no difference

                # calculate/get kernel
                if callable(sigma):
                    s = sigma(w[idx])  # get sigma for this wavelength
                else:
                    s = sigma  # use fixed sigma
                if s not in K:  # build kernel if not already done
                    K[s] = build_kernel(s, res)

                    # add to output
                if d > thresh:
                    # store position
                    Tpos[p].append(w[idx])
                    Tdepth[p].append(-d)

                    # apply kernel
                    k0 = int(idx - (K[s].shape[0] / 2))  # index of start of kernel (pos - 5 sigma)
                    k1 = k0 + K[s].shape[0]  # index of end of kernel (pos + 5 sigma)
                    if k1 < out.shape[
                        -1] and k0 > 0:  # ignore features close to edges (these are probably dodgy anyway)
                        out[p, k0:k1] -= weight * K[s]  # add kernel to likelihood function

        # add maxima to output
        if maxima:
            for i in maxx:
                # match feature with adjacent maxima
                left = 0
                right = dy.shape[-1] - 1
                if (minx < i).any():
                    left = minx[minx < i][np.argmin(np.abs(i - minx[minx < i]))]
                if (minx > i).any():
                    right = minx[minx > i][np.argmin(np.abs(minx[minx > i] - i))]

                # calculate height
                d = R[p, i] - min(R[p, left], R[p, right])  # compute simple max - min height
                weight = 1.0
                if weighted:
                    weight = d / min(R[p, left],
                                     R[p, right])  # weight is the fractional increaes associated with this maxima

                # calculate index in output array
                if domain is not None:
                    idx = int((data.get_wavelengths()[i] - w[0]) / res)  # compute idx in ouput array
                else:
                    idx = i  # no difference

                # calculate/get kernel
                if callable(sigma):
                    s = sigma(w[idx])  # get sigma for this wavelength
                else:
                    s = sigma  # use fixed sigma
                if s not in K:  # build kernel if not already done
                    K[s] = build_kernel(s, res)

                    # add kernel to output
                if d > thresh:
                    # store position
                    Tpos[p].append(w[idx])
                    Tdepth[p].append(d)

                    # calculate position (index)
                    k0 = int(idx - (K[s].shape[0] / 2))  # index of start of kernel (pos - 5 sigma)
                    k1 = k0 + K[s].shape[0]  # index of end of kernel (pos + 5 sigma)
                    if k1 < out.shape[
                        -1] and k0 > 0:  # ignore features close to edges (these are probably dodgy anyway)
                        out[p, k0:k1] += weight * K[s]  # add kernel to likelihood function

    # construct outputs
    O = data.copy(data=False)
    if data.is_image():  # reshape outputs
        Tpos = np.array(Tpos).reshape(data.data.shape[:-1])
        Tdepth = np.array(Tdepth).reshape(data.data.shape[:-1])
        O.data = out.reshape(data.data.shape[:-1] + (out.shape[-1],))
    else:
        O.data = out
    O.set_wavelengths(w)
    return O, Tpos, Tdepth


def TPT2MWL(pos, depth, wmin=0, wmax=-1, data=None, vb=True):
    """
    Convert the results from a turning point filter (TPT) into a minimum wavelength array. This will return
    the deepest feature within the specified range.

    Args:
        pos: an array containing per-spectra turning point positions, as returned by TPT(...).
        depth: an array containing per-spectra turning point depths, as returned by TPT(...).
        wmin: the start of the wavelength range of interest. Default is 0 (all wavelengths).
        wmax: the end of the wavelength range of interest. Default is -1 (all wavelengths).
        data: a HyData instance to use as a template for outputs. If not None (default) then this will return a HyData instance.
        vb: True if a progress bar should be created. Default is True.

    Returns:
        a numpy array or HyData instance (if data is not none), containing 4 bands (pos, width, depth, strength).
    """

    # reshape for loop
    shape = np.array(pos).shape
    pos = np.array(pos).ravel()
    depth = np.array(depth).ravel()
    out = np.full((len(pos), 4), np.nan, dtype=np.float32)

    # set wmax if need be
    if wmax == -1:
        wmax = np.inf

    # loop through spectra
    loop = range(len(pos))
    if vb:
        loop = tqdm(loop, leave=False, desc="Filtering turning points")
    for i in loop:
        if len(pos[i]) > 0:
            p = np.array(pos[i])
            d = np.array(depth[i])
            mask = (d < 0) & (p > wmin) & (p < wmax)
            if mask.any():
                idx = np.argmin(d[mask])
                out[i, 0] = p[mask][idx]
                out[i, 1] = 0  # how to estimate width?
                out[i, 2] = -d[mask][idx]
                out[i, 3] = out[i, 2]

    # return array for mwl map
    out = out.reshape(shape + (4,))
    if data is None:  # return a numpy array directly
        return out
    else:  # make a copy of data and return HyData instance
        data = data.copy(data=False)
        data.data = out
        return data