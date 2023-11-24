"""
An open-source python toolbox for hyperspectral data preprocessing, correction, projection and analysis.

-----------

### Tutorials

A variety of interactive notebook tutorials are available for *hylite*:

Basic:
- [Introduction tutorial using GoogleColab](https://drive.google.com/drive/folders/1hkr4gtP1OY_PIK7cynl3dWd3sYi_9s5F?usp=drive_link)
- [Another introduction from the DRT workshop](https://tinyurl.com/drt2022)
- [Yet another introduction from the VGC conference](https://drive.google.com/drive/folders/1_gDRMrccNG3OMyIPYy0mkpkbN6nn92OW?usp=sharing)

Advanced:
- [Building corrected hyperclouds](https://tinyurl.com/Maamorilik01)
- [Minimum wavelength mapping](https://tinyurl.com/Maamorilik02)
- [Visualising hyperclouds](https://tinyurl.com/Maamorilik03)

----------

### Publications

If you use hylite for your work then please cite:

* Thiele, S. T., Lorenz, S., et al., (2021). Multi-scale, multi-sensor data integration for automated 3-D geological
mapping using hylite. *Ore Geology Reviews*. https://doi.org/10.1016/j.oregeorev.2021.104252

Other relevant papers include:

* Thiele, S.T., Bnoulkacem, Z., Lorenz, S., Bordenave, A., Menegoni, N., Madriz, Y., Dujoncquoy, E., Gloaguen, R. and Kenter, J., 2021.
Mineralogical Mapping with Accurately Corrected Shortwave Infrared Hyperspectral Data Acquired Obliquely from UAVs.
*Remote Sensing*, 14(1), p.5. https://doi.org/10.3390/rs14010005

* Thiele, S. T., Lorenz, S., Kirsch, M., & Gloaguen, R. (2021).
A Novel and Open-Source Illumination Correction for Hyperspectral Digital Outcrop Models. *IEEE Transactions on
Geoscience and Remote Sensing*. https://doi.org/10.1109/TGRS.2021.3098725

* Lorenz, S., Thiele, S.T., Kirsch, M., Unger, G., Zimmermann, R., Guarnieri, P., Baker, N.,
Sørensen, E.V., Rosa, D. and Gloaguen, R., 2022. Three-Dimensional, Km-Scale Hyperspectral Data of Well-Exposed Zn–Pb
Mineralization at Black Angel Mountain, Greenland. *Data*, 7(8), p.104. https://doi.org/10.3390/data7080104

* Guarnieri, P., Thiele, S.T., Baker, N., Sørensen, E.V., Kirsch, M., Lorenz, S., Rosa, D., Unger, G. and Zimmermann, R., 2022.
Unravelling the Deformation of Paleoproterozoic Marbles and Zn-Pb Ore Bodies by Combining 3D-Photogeology and
Hyperspectral Data (Black Angel Mine, Central West Greenland). *Minerals*, 12(7), p.800. https://doi.org/10.3390/min12070800

* Kirsch, M., Mavroudi, M., Thiele, S., Lorenz, S., Tusa, L., Booysen, R., Herrmann, E., Fatihi, A., Möckel, R., Dittrich, T. and Gloaguen, R.,
2023. Underground hyperspectral outcrop scanning for automated mine‐face mapping: The lithium deposit of Zinnwald/Cínovec.
The Photogrammetric Record, 38(183), pp.408-429. https://doi.org/10.1111/phor.12457

-------

# Documentation

Almost all of the modules, classes and functions in *hylite* have docstrings. These can be viewed in a notebook or
python console using the help(...) function or by typing "?" after a class or function name. Searchable documentation
is also available online.
"""

# to generate docs with pdoc run:  pdoc --html hylite --output-dir docs --force

# disable numpy multithreading
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

#disable annoying warnings
#np.warnings.filterwarnings('ignore')
#warnings.filterwarnings("ignore", category=DeprecationWarning)
# ignore all warnings
#def _warn(*args, **kwargs):
#    pass
#warnings.warn = _warn

import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", category=DeprecationWarning)

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
