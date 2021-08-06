import numpy as np
import warnings

#disable annoying warnings
np.warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ignore all warnings
def warn(*args, **kwargs):
    pass
warnings.warn = warn

###########################################
## Define useful preset band combinations
###########################################
RGB = (680.0,550.0,505.0)
"""
Wavelengths for red [680.0], green [550.0] and blue [505.0]- useful for plotting.
Note that we use the upper end of blue as this is the first band of rikola data.
"""

VNIR = (800.0, 550.0, 505.0)
"""Useful preview for VNIR data using (infrared [1972.0], green [644.0], blue [1450.0])."""

SWIR = (2200.0,2250.0,2350.0)
"""Useful preview for SWIR data (2200.0, 2250.0, 2350.0) sensitive to clay, mica, carbonate and amphibole absorbtions."""

BROAD = (1972.0,644.0,1450.0) #useful preview for data that covers visible VNIR and SWIR range
"""Useful preview that covers VNIR and SWIR range (1972.0,644.0,1450.0) ."""

MWIR = (3000., 3400., 3800. )
"""Useful preview for MWIR range (3000., 3400., 3800. )."""

LWIR = TIR = (10101.01, 9174.31, 8547.01)
"""Useful preview for TIR range (10101.01, 9174.31, 8547.01)."""

band_select_threshold = 10.
"""Maximum distance (in nanometers) to use when matching wavelengths with band indices. See HyData.get_band_index(...) for more detail."""

#import basic data classes
from .hyheader import HyHeader
from .hydata import HyData
from .hyimage import HyImage
from .hycloud import HyCloud
from .hylibrary import HyLibrary
from .hycollection import HyCollection
from .hyscene import HyScene
from .hyfeature import HyFeature, MultiFeature, MixedFeature
