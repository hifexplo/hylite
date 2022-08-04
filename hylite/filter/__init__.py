"""
Denoising, feature-extraction and dimensionality reduction filters.
"""

from .dimension_reduction import MNF, PCA, from_loadings
from .segment import *
from .sample import *
from .tpt import *


def boost_saturation( image, bands, flip=True, sat=0.8, val=None, clip=(2,98),per_band=False ):
    """
    Create a saturation boosted composite image.

    *Arguments*:
     - the image containing the data
     - bands = the bands to map to r, g, b.
     - flip = true if the image should be inverted prior to boosting saturation
              (good for interpreting absorbtion features). Default is True.
     - sat = the (constant) saturation value to use. Default is 0.8.
     - val = the (constant) brightness value to use. Default is None (do not fix brightness).
     - clip = the percentile clip range to use to map to 0 - 1. Default is the 2nd and 98th percentile (2,98).
     - per_band = True if this clipping should be done per band or over the whole RGB composite. Default is False
    *Returns*
     - a HyImage instance containing the saturation enhanced result
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