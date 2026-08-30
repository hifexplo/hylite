"""
A package for applying radiometric and topographic corrections to hyperspectral datasets to convert measured at-sensor
radiance to reflectance. Data detrending methods (e.g. hull corrections) are also included here.
"""

# lazy imports
def __getattr__(name):
    if name == "get_hull_corrected":
        from .detrend import get_hull_corrected
        return get_hull_corrected
    if name == "Panel":
        from .panel import Panel
        return Panel
    if name == "norm_eq":
        from .equalize import norm_eq
        return norm_eq
    if name == "hist_eq":
        from .equalize import hist_eq
        return hist_eq
    raise AttributeError("module %r has no attribute %r" % (__name__, name))
