"""
Deprecated PCA/MNF dimensionality reduction filters.

Image compositing and spectral resampling have moved to `hylite.transform`.
"""

import warnings
warnings.filterwarnings("once", category=DeprecationWarning, module=__name__)
warnings.warn(
    "hylite.filter is deprecated and will be removed in a future release; "
    "use hylite.transform instead.",
    DeprecationWarning,
)

from .dimension_reduction import MNF, PCA, from_loadings
