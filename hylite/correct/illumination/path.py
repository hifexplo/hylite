"""
Classes containing different occlusion models (e.g. estimators for shadows, sky view factor etc.)
"""
import hylite
import matplotlib.pyplot as plt
import numpy as np

from hylite.correct import get_hull_corrected
from hylite.multiprocessing import parallel_chunks


def estimate_path_radiance(image, depth, thresh=1):
    """
    Apply the dark object subtraction (DOS) method to estimate path radiance in the provided image.

    *Arguments*:
     - image = the hyperspectral image (HyImage instance) to estimate path radiance for.
     - depth = A 2-D (width,height) numpy array of pixel depths in meters. This can be easily computed using
               a HyScene instance.
     - thresh = the percentile threshold to use when selecting dark pixels. Default is 1%.

    *Returns*:
     - spectra = a numpy array containing the estimated path radiance spectra (in radiance per meter of depth).
     - path = a HyImage instance containing the estimated path radiance per pixel (computed by multiplying
              the spectra by the depth).
    """

    # identify dark pixels
    r = np.nanmean(image.data, axis=-1)  # calculate mean brightness

    assert r.shape == depth.shape, "Error: depth array and HSI have different shapes: %s != %s" % (
    str(r.shape), str(depth.shape))

    r[r == 0] = np.nan  # remove background / true zeros as this can bias the percentile clip
    r[np.logical_not(np.isfinite(depth))] = np.nan  # remove areas without depth info
    r[depth == 0] = np.nan  # remove areas without depth info

    # extract dark pixels and get median
    thresh = np.nanpercentile(r, thresh)  # threshold for darkest pixels
    darkref = image.data[r <= thresh, :]
    darkdepth = depth[r <= thresh]  # corresponding depths

    # compute path radiance estimate
    S = darkref / darkdepth[..., None]  # compute estimated path-radiance per meter
    S = np.nanpercentile(S, 50, axis=0)  # take median of all estimates

    # compute per pixel path radiance
    P = image.copy(data=False)
    P.data = depth[..., None] * S[None, None, :]

    return S, P

def correct_path_absorption(data, band_range=(0, -1), thresh=99, atabs = 1126., vb=True):
    """
    Fit and remove a known atmospheric water feature to remove atmospheric path absorbtions from reflectance spectra.
    See  Lorenz et al., 2018 for more details.

    Reference:
    https://doi.org/10.3390/rs10020176

    *Arguments*:
     - image = a hyperspectral image to correct
     - band_range = a range of bands to do this over. Default is (0,-1), which applies the correction to all bands.
     - thresh = the percentile to apply when identifying the smallest absorbtion in any range based on hull corrected
                spectra. Lower values will remove more absorption (potentially including features of interest).
     - atabs = wavelength position at which a known control feature is situated that defines the intensity of correction
                - for atmospheric effects, this is set to default to 1126 nm
     - vb = True if a progress bar should be created during hull correction steps.

    *Returns*:
     - a HyData instance containing the corrected spectra.
    """
    assert isinstance(atabs, float), "Absorption wavelength must be float"
    # subset dataset
    out = data.export_bands(band_range)
    nanmask = np.logical_not(np.isfinite(out.data))
    out.data[nanmask] = 0  # replace nans with 0

    # get depth of water feature at 1126 nm for all pixels
    atm_depth = (out.get_band(atabs - 50.) + out.get_band(atabs + 50.)) / 2 - out.get_band(atabs)

    # kick out data points with over-/undersaturated spectra
    atm_temp = atm_depth.copy()
    atm_temp[np.logical_or(np.nanmax(out.data, axis=-1) >= 1, np.nanmax(out.data, axis=-1) <= 0)] = 0
    # extract pixels that are affected most by the features
    highratio = out.data[atm_temp > np.percentile(atm_temp, 90)]
    # hull correct those
    hull = get_hull_corrected(hylite.HyData(highratio), vb=vb)
    # extract the always consistent absorptions
    hull_max = np.nanpercentile(hull.data, thresh, axis=0)

    vmin = hull_max[out.get_band_index(atabs)]
    # apply adjustment and return
    if out.is_image():
        nmin = -atm_depth[..., None]
        out.data -= ((hull_max[None, None, :] - vmin) * (-nmin) / (1 - vmin) + nmin)
    else:
        nmin = -atm_depth[..., None]
        out.data -= ((hull_max[None, :] - vmin) * (-nmin) / (1 - vmin) + nmin)
    out.data[nanmask] = np.nan  # add nans back in
    out.data = np.clip(out.data, 0, 1)

    return out