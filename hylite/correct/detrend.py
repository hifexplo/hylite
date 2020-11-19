from hylite import HyLibrary
import numpy as np
from scipy.spatial.qhull import ConvexHull
from tqdm import tqdm

def polynomial(data, degree = 1, method='div'):

    """
    Detrend an image data array using a polynomial fit and np.polyfit( ... ).

    *Arguments*:
     - data = numpy array of the format image[x][y][b].
     - degree = the degree of the polynomial to fit. Default is 2.
     - method = 'divide' or 'subtract'. Default is 'divide'.

    *Returns*:
     - corr = the corrected (detrended) data.
     - trend = the trend that was removed.
    """

    #calculate trend
    y = np.array(data.reshape( -1, data.shape[-1] )) #reshape array so each pixel is a column
    y[np.logical_not(np.isfinite(y))] = 0 #kill NaNs
    _x = np.arange(data.shape[-1])
    fit = np.polynomial.polynomial.polyfit(_x, y.T, degree) # fit polynomial
    t = np.polynomial.polynomial.polyval(_x, fit) # evaluate it

    #apply correction
    if 'div' in method:
        y /= t
    elif 'sub' in method.lower():
        y -= t
        y += np.min(y) #map to positive

    return y.reshape(data.shape), t.reshape(data.shape)

def hull(spectra, div=True):
    """
    Detrend a 1D spectra by performing a hull correction. Note that this performs the correction in-situ.

    *Arguments*:
     - div = True if the spectra should be divided by it's hull (default). False if the hull should be subtracted.
    *Returns*:
     - corr = hull corrected spectra.
     - trend = the (hull) trend that was subtracted to give the corrected spectra

    Returns an unchanged spectra and trend = [0,0,...] if the spectra contains nans or infs.
    """

    # calculate convex hull
    hull = ConvexHull(np.array([np.hstack([0, np.arange(len(spectra)), len(spectra) - 1]),
                                np.hstack([0, spectra, 0])]).T)

    # remove unwanted simplices (e.g. along sides and base)
    mask = (hull.simplices != 0).all(axis=1) & (hull.simplices != len(spectra) + 1).all(axis=1)

    # build piecewise equations
    x = np.arange(len(spectra), dtype=np.float32)
    if not mask.any():  # edge case - convex hull is one simplex between first and last points!
        y = spectra[0] + (spectra[-1] - spectra[0]) / x[-1]
    else:
        grad = -hull.equations[mask, 0]
        itc = -hull.equations[mask, 2]
        dom = [(min(x[s[0]], x[s[1]]), max(x[s[0]], x[s[1]])) for s in (hull.simplices[mask] - 1)]
        cl = [(x >= d[0]) & (x <= d[1]) for d in dom]

        # evaluate piecewise functions
        fn = [(lambda x, m=grad[i], c=itc[i]: m * x + c) for i in range(len(grad))]
        y = np.piecewise(x, cl, fn)

    # return
    if div:
        return spectra / y, y
    else:
        return 1 + spectra - y, y


def get_hull_corrected(data, band_range=None, method='div', vb=True):
    """
    Apply a hull correction to an entire HyData instance (HyImage, HyCloud or HyLibrary). Returns a corrected copy of
    the input dataset.

    *Arguments*:
     - band_range = Tuple containing the (min,max) band indices or wavelengths to run the correction between. If None
                     (default) then the correction is run of the entire range.
     - method = Trend removal method: 'divide' or 'subtract'. Default is 'divide'.
     - vb = True if this should print output.
    """

    # create copy containing the bands of interest
    if band_range is None:
        band_range = (0, -1)
    else:
        band_range = (data.get_band_index(band_range[0]), data.get_band_index(band_range[1]))
    corrected = data.export_bands(band_range)

    # convert integer data to floating point (we need floats for the hull correction)
    comp = False
    if corrected.is_int():
        corrected.decompress()
        comp = True

    method = 'div' in method  # convert method to bool (for performance)

    # get valid pixels
    D = corrected.get_raveled()
    nan = corrected.header.get_data_ignore_value()
    valid = (np.isfinite(D) & (D != nan)).all(axis=1)  # drop nans/no-data values
    valid = valid & (D != D[:, 0][:, None]).any(axis=1)  # drop flat spectra (e.g. all zeros)

    if len(valid > 0): # if some valid points exist, do correction
        X = D[valid]
        upper = []
        lower = []
        loop = range(X.shape[0])
        if vb:
            loop = tqdm(loop, leave=False, desc='Applying hull correction')
        for p in loop:
            X[p, :], fac = hull(X[p, :], div=method)

            # special case - also apply this correction to upper/lower spectra of the HyData instance
            if isinstance(corrected, HyLibrary):
                if corrected.upper is not None:
                    # also apply correction to bounds
                    if method:
                        upper.append(corrected.upper[valid][p, :] / fac)
                    else:
                        upper.append(corrected.upper[valid][p, :] - fac)
                if corrected.lower is not None:
                    # also apply correction to bounds
                    if method:
                        lower.append(corrected.lower[valid][p, :] / fac)
                    else:
                        lower.append(corrected.lower[valid][p, :] - fac)

                        # copy data back into original array
        D[valid] = X
        corrected.set_raveled(D)
        if len(upper) > 0:
            corrected.upper[valid] = np.array(upper)
        if len(lower) > 0:
            corrected.lower[valid] = np.array(lower)

    # convert back to integer if need be
    if comp:
        corrected.compress()

    return corrected