"""
A package for applying radiometric and topographic corrections to hyperspectral datasets to convert measured at-sensor
radiance to reflectance. Data detrending methods (e.g. hull corrections) are also included here.
"""

from .detrend import get_hull_corrected
from .panel import Panel
from .equalize import norm_eq, hist_eq

