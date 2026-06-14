"""
Image compositing and display transforms.
"""
import numbers
import numpy as np
from hylite._deps import require

def boost_saturation(image, bands, flip=True, sat=0.8, val=None, clip=(2, 98), per_band=False):
    """
    Create a saturation boosted composite image.

    Args:
        image (`hylite.hyimage.HyImage`): the image containing the data
        bands: the bands to map to r, g, b.
        flip: true if the image should be inverted prior to boosting saturation
              (good for interpreting absorption features). Default is True.
        sat: the (constant) saturation value to use. Default is 0.8.
        val: the (constant) brightness value to use. Default is None (do not fix brightness).
        clip: the percentile clip range to use to map to 0 - 1. Default is the 2nd and 98th percentile (2,98).
        per_band: True if this clipping should be done per band or over the whole RGB composite. Default is False
    Returns:
        a `hylite.hyimage.HyImage` instance containing the saturation enhanced result
    """

    rgb_to_hsv = require("matplotlib.colors").rgb_to_hsv
    hsv_to_rgb = require("matplotlib.colors").hsv_to_rgb

    rgb = image.export_bands(bands)
    _ = rgb.percent_clip(clip[0], clip[1], per_band=per_band)

    if flip:
        rgb.data = 1 - rgb.data

    hsv = rgb_to_hsv(rgb.data)

    if sat is not None:
        hsv[..., 1] = sat
    if val is not None:
        hsv[..., 2] = val

    rgb.data = hsv_to_rgb(hsv)
    return rgb

def overlay(image_list, method="median", warp=False):
    """
    Combine a list of images to improve signal-to-noise or mitigate dead pixels.

    Args:
        image_list: a list of `hylite.hyimage.HyImage` objects for which data will be averaged. These must be identical sizes.
        method: The method used to combine the images. Can be "mean", "median", "min", "max" or a
                percentile between 0 and 100. Default is "median"
        warp: should the images be warped to optimise coregistration using optical flow? Slow... default is False. Always
              matches images to the first one in image_list. The middle band is used for matching.
    Returns:
        A tuple containing:

         - average = a `hylite.hyimage.HyImage` containing the combined image data
         - std = a `hylite.hyimage.HyImage` containing the standard deviation of the image data.
    """
    from hylite import HyImage

    minx = min([i.xdim() for i in image_list])
    miny = min([i.ydim() for i in image_list])
    arr = [i.data[0:minx, 0:miny, :] for i in image_list]

    if warp:
        cv2 = require("cv2")
        alg = cv2.optflow.createOptFlow_DeepFlow()
        X, Y = np.meshgrid(range(arr[0].shape[1]), range(arr[0].shape[0]))
        match_idx = int(image_list[0].band_count() / 2)
        bnd1 = HyImage.to_grey(arr[0][:, :, match_idx])

        for i, image in enumerate(arr[1:]):
            print("Warping image %d" % (i + 1))
            bnd2 = HyImage.to_grey(image[:, :, match_idx])
            flow = alg.calc(bnd1, bnd2, None)
            map = np.dstack([X, Y]).astype(np.float32)
            map[:, :, 0] += flow[:, :, 0]
            map[:, :, 1] += flow[:, :, 1]
            for b in range(image.shape[-1]):
                image[:, :, b] = cv2.remap(image[:, :, b], map, None, cv2.INTER_LINEAR)

    std = np.nanstd(arr, axis=0)
    if "mean" in method.lower():
        out = np.nanmean(arr, axis=0)
    elif "median" in method.lower():
        out = np.nanmedian(arr, axis=0)
    elif "min" in method.lower():
        out = np.nanmin(arr, axis=0)
    elif "max" in method.lower():
        out = np.nanmax(arr, axis=0)
    elif isinstance(method, numbers.Number) and 0 <= method <= 100:
        out = np.nanpercentile(arr, axis=0)
    else:
        assert False, "Error - %s is an unknown overlay method." % method

    oimg = image_list[0].copy()
    oimg.data = out

    simg = image_list[0].copy()
    simg.data = std
    return oimg, simg
