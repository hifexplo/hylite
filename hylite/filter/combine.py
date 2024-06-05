"""
Utility functions for combining / averaging multiple Datasets together.
"""

import numbers
import numpy as np
from hylite import io as io


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
        bnd1 = io.HyImage.to_grey(arr[0][:, :, match_idx])  # we use the middle band to do alignment

        for i, image in enumerate(arr[1:]):
            print("Warping image %d" % (i + 1))

            # extract matching band
            bnd2 = io.HyImage.to_grey(image[:, :, match_idx])

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
    return out, std