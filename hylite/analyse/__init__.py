"""
This package implements common analysis methods for hyperspectral datasets, including:
    - supervised and unsupervised classification.
    - minimum wavelength mapping.
    - endmember extraction and unmixing.
    - commonly used band ratios (e.g. NDVI).
"""

from .indices import *
from .mwl import *
from .sam import *
from .dtree import *