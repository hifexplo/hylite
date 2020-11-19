"""
A package for applying radiometric and topographic corrections to hyperspectral datasets to convert measured at-sensor
radiance to reflectance. Data detrending methods (e.g. hull corrections) are also included here.
"""

from .detrend import hull, polynomial, get_hull_corrected
from .panel import ELC
from .panel import Panel
from .topography import estimate_sun_vec
from .topography import estimate_incidence
from .topography import estimate_ambient
from .topography import correct_topo
from .equalize import norm_eq, hist_eq
