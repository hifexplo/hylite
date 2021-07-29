"""
Use spectra derivatives to extract and analyse turning points (maxima and minima).
"""

import numpy as np
from tqdm import tqdm

def build_kernel( sigma, res ):
    """
    Construct a kernel (normal distribution with specified standard deviation)
    """
    kx = np.arange(-5*sigma,5*sigma,step=res,dtype=np.float32)
    ky = ((1 / (sigma * np.sqrt( 2 * np.pi ) )) * np.exp( -0.5*( (kx / sigma )**2) )).astype(np.float32)
    ky /= np.trapz(ky,kx) # increase values slightly to account for numerical integration errors
    return ky


def TPT(data, sigma=10., window=21, thresh=0, domain=None, weighted=True, maxima=True, minima=True, vb=True):
    """
    Extract the turning points from each spectra in dataset, and build an associated kernel density estimates that
    can be used to calculate the probability of observing given features.

    *Arguments:*
     - sigma = Standard deviation of the KDE kernel. Either a number, or a function such that sigma(w) returns the
               standard deviation for a specific wavelength (e.g. if a larger kernel should be used for
               LWIR than VNIR data). Default is 10 nm.
     - window = the size of the window (in indices) used during savgol smoothing and derivative calculation. Default is 21.
     - thresh = a depth/prominence threshold to ignore small maxima or minima. Default is 0.
     - domain = None (default) if the KDE should be evaluated over the same wavelengths as the input data. Alternatively, a
                tuple can be passed containing (min_wav, max_wav, [resolution (optional)]).
     - weighted = True if the KDE should be weighted based on feature depth (i.e. more prominent features get extra
                  weight in the KDE as they are ~ 'more likely to be observed'). Default is True.
     - maxima = True the filter should include maxima. Default is True.
     - minima = True if the filter should include minima. Default is True.
     - vb = True progress should be printed to the console. Default is True.

    *Returns*:
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

    # calculate derivatives
    dy = data.smooth_savgol(window, 2, deriv=1)
    dy = dy.X()  # convert dataset to vector form

    # init output array
    out = np.zeros((dy.shape[0], w.shape[0]))
    Tpos = [[] for i in range(dy.shape[0])]
    Tdepth = [[] for i in range(dy.shape[0])]
    K = {}
    loop = range(dy.shape[0])
    if vb:
        loop = tqdm(loop)
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