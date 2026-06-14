"""
A collection of functions for applying common transforms to hyperspectral data cubes, hyperclouds and/or spectral libraries.

Dimensionality reduction (PCA/MNF) lives in `hylite.transform.reduction` and requires scikit-learn.
"""
import numpy as np
import warnings
from hylite import HyData, HyImage

def convertToAbsorbance( data : HyData, method: str = 'kubelka-munk', band_range = None) -> HyImage:
    """
    Converts Reflectance to Pseudo-Absorbance using methods such as Kubelka-Munk.
    Applies to an entire `hylite.hydata.HyData` instance (`hylite.hyimage.HyImage`, `hylite.hycloud.HyCloud`
    or `hylite.hylibrary.HyLibrary`).

    Kubelka-Munk transformation:
        Based on Escobedo-Morales et al. 2019, 'Automated method for the determination of
        the band gap energy of pure and mixed powder samples using diffuse reflectance spectroscopy',
        DOI: 10.1016/j.heliyon.2019.e01505

        This function is developed by Andréa de Lima Ribeiro (Orchid Id: 0000-0003-0096-3627)

    Args:
        data: a numpy array or `hylite.hydata.HyData` instance to detrend.
        method: string specifying the conversion method. Currently only 'kubelka-munk' is supported.
        band_range: Tuple containing the (min,max) band indices or wavelengths to run the correction between. If None
        (default) then the correction is run over the entire range. Only works if data is a `hylite.hydata.HyData` instance.

    Returns:
        `hylite.hyimage.HyImage`: Pseudo-absorbance dataset
    """

    if isinstance(data, HyData):
        # create copy containing the bands of interest
        if band_range is None:
            band_range = (0, -1)
        else:
            band_range = (data.get_band_index(band_range[0]), data.get_band_index(band_range[1]))
        corrected = data.export_bands(band_range)

        # selected range check
        arr = corrected.data
        valid = arr[np.isfinite(arr)]
        valid = valid[ valid >= 0.0 ] # make the negative values as nan
        min_val = valid.min()
        max_val = valid.max()

        # check if reflectance values are within 0–1 or 0–100
        in_01 = (min_val >= 0.0) and (max_val <= 1.0)
        in_0100 = (min_val >= 0.0) and (max_val <= 100.0)

        if not (in_01 or in_0100):
            warnings.warn(f"Reflectance values out of expected range: "
                          f"min={min_val:.3f}, max={max_val:.3f}. Expected 0–1 or 0–100.")
            raise ValueError("Invalid reflectance range. Function stopped.")

        # if reflectance values between 0–100, convert to 0–1 for the formula
        if in_0100 and not in_01:
            arr = arr / 100.0

        # Kubelka–Munk transformation
        if method.lower() == 'kubelka-munk':
            eps = 1e-12 #factor added to avoid division by 0 or by negative numbers
            arr = np.clip(arr, eps, 1.0)  # reflectance should be (0,1]
            km = ((1.0 - arr) ** 2) / (2.0 * arr) # Kubelka-Munk transformation

        corrected.data = km
        return corrected

    else:
        raise TypeError("data must be a HyData instance (HyImage/HyCloud/HyLibrary).")

# expose these functions at the module level for convenience
from .filter import boost_saturation, overlay
from .sample import Resample, ASTER, SENTINEL

# lazily import skilearn-dependent classes and complain if someone tries to use them without sklearn installed
_REDUCTION_NAMES = frozenset({"PCA", "MNF", "NoiseWhitener"})
def __getattr__(name):
    """Lazy import sklearn-dependent reduction classes for backward compatibility."""
    if name in _REDUCTION_NAMES:
        from . import reduction
        return getattr(reduction, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

def __dir__():
    return sorted(list(globals().keys()) + list(_REDUCTION_NAMES))
