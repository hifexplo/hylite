"""
Denoising, feature-extraction and dimensionality reduction filters.
"""

from .dimension_reduction import MNF, PCA, from_loadings
from .segment import *
from .sample import *
from .tpt import *

import numbers
import numpy as np

def boost_saturation( image, bands, flip=True, sat=0.8, val=None, clip=(2,98),per_band=False ):
    """
    Create a saturation boosted composite image.

    Args:
        image (hylite.HyImage): the image containing the data
        bands: the bands to map to r, g, b.
        flip: true if the image should be inverted prior to boosting saturation
              (good for interpreting absorbtion features). Default is True.
        sat: the (constant) saturation value to use. Default is 0.8.
        val: the (constant) brightness value to use. Default is None (do not fix brightness).
        clip: the percentile clip range to use to map to 0 - 1. Default is the 2nd and 98th percentile (2,98).
        per_band: True if this clipping should be done per band or over the whole RGB composite. Default is False
    Returns:
        a HyImage instance containing the saturation enhanced result
    """

    from matplotlib.colors import rgb_to_hsv, hsv_to_rgb

    # subset
    rgb = image.export_bands(bands)

    # clip
    _ = rgb.percent_clip(clip[0], clip[1], per_band=per_band)

    # invert?
    if flip:
        rgb.data = 1 - rgb.data

    hsv = rgb_to_hsv(rgb.data) # map to hsv space

    if sat is not None: # boost sat
        hsv[...,1] = sat
    if val is not None: # boost brightness
        hsv[..., 2] = val

    rgb.data = hsv_to_rgb(hsv) # map back to RGB space

    return rgb

def combine(image_list, method="median", warp=False):
    """
    Combines a list of images to improve the signal to noise ratio/remove issues associated with dead pixels etc.

    Args:
        image_list: a list of hyImage objects for which data will be averaged. These must be identical sizes.
        method: The method used to combine the images. Can be "mean", "median", "min", "max" or a
                percentile between 0 and 100. Default is "median"
        warp: should the images be warped to optimise coregistration using optical flow? Slow... default is False. Always
              matches images to the first one in image_list. The middle band is used for matching.
    Returns:
        A tuple containing:

         - average = a numpy array containing the averaged image data
         - std = a numpy array containing the standard deviation of the image data.
    """
    from hylite import HyImage

    # resize to fit (sometimes some images are 1-2 pixels too long for line-scanners on a tripod)
    minx = min([i.xdim() for i in image_list])
    miny = min([i.ydim() for i in image_list])
    arr = [i.data[0:minx, 0:miny, :] for i in image_list]

    # build co-aligned data array to average
    if warp:
        import cv2 # import this here to avoid errors if opencv is not installed properly
        alg = cv2.optflow.createOptFlow_DeepFlow()
        X, Y = np.meshgrid(range(arr[0].shape[1]), range(arr[0].shape[0]))
        match_idx = int(image_list[0].band_count() / 2)
        bnd1 = HyImage.to_grey(arr[0][:, :, match_idx])  # we use the middle band to do alignment

        for i, image in enumerate(arr[1:]):
            print("Warping image %d" % (i + 1))

            # extract matching band
            bnd2 = HyImage.to_grey(image[:, :, match_idx])

            # calculate optical flow
            flow = alg.calc(bnd1, bnd2, None)

            # transform to pixel map
            map = np.dstack([X, Y]).astype(np.float32)
            map[:, :, 0] += flow[:, :, 0]
            map[:, :, 1] += flow[:, :, 1]

            # remap bands
            for b in range(image.shape[-1]):
                image[:, :, b] = cv2.remap(image[:, :, b], map, None, cv2.INTER_LINEAR)

    # calculate average and standard deviation
    std = np.nanstd(arr, axis=0)
    if "mean" in method.lower():
        out = np.nanmean(arr, axis=0)
    elif "median" in method.lower():
        out = np.nanmedian(arr, axis=0)
    elif "min" in method.lower():
        out = np.nanmin(arr, axis=0)
    elif "max" in method.lower():
        out = np.nanmax(arr, axis=0)
    elif isinstance(method, numbers.Number) and 0 <= method <= 100:  # percentile
        out = np.nanpercentile(arr, axis=0)

    # return result
    oimg = image_list[0].copy()
    oimg.data = out

    simg = image_list[0].copy()
    simg.data = std
    return oimg, simg