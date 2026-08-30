"""
This package implements common analysis methods for hyperspectral datasets, including:

- minimum wavelength mapping.
- endmember extraction and unmixing.
- fourier analysis, filtering and compression.
- commonly used band ratios (e.g. NDVI).
"""

import importlib
import os

import numpy as np

from .indices import *
from .sam import *


def __getattr__(name):
    for submodule in ("mwl", "unmixing", "dtree"):
        try:
            mod = importlib.import_module(".%s" % submodule, __name__)
        except ImportError:
            continue
        if hasattr(mod, name):
            return getattr(mod, name)
    raise AttributeError("module %r has no attribute %r" % (__name__, name))


def saveLegend(red: str, green: str, blue: str, path: str):
    """
    Create and save a legend image for a ternary image.

    Args:
        - red: String containing the description / label for the red channel.
        - green: String containing the description / label for the green channel.
        - blue: String containing the description / label for the blue channel.
        - path: The output path to save the image to.
    """
    from hylite._deps import require

    plt = require("matplotlib.pyplot")
    tr = [np.nan, np.nan, np.nan]
    plt.figure(figsize=(4, 0.5))
    plt.fill(tr, tr, label=red, color='r')
    plt.fill(tr, tr, label=green, color='g')
    plt.fill(tr, tr, label=blue, color='b')
    plt.legend(ncol=3, loc='center')
    plt.axis('off')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=350)
    plt.close()
