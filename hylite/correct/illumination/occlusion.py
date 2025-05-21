"""
Classes containing different occlusion models (e.g. estimators for shadows, sky view factor etc.)
"""

import numpy as np
from hylite.analyse import band_ratio


def calcBandRatioOcc(rad, **kwds):
    """
    Calculate the fraction of incoming light blocked by this occlusion (0 - 1) based on a band ratio.

    Args:
        radiance: Radiance data (HyData instance) to calculate band ratio with.
        **kwds: Keywords can include:

             - num = Numerator of band ratio, as defined by hylite.analyse.band_ratio. Default is (400,450) nm.
             - den = Denominator of band ratio, as defined by hylite.analyse.band_ratio. Default is (550,600) nm.
             - a = the threshold for complete shadow (values > this are full shadow). Default is 1.20.
             - b = the threshold for complete sun (values < this are full sun). Default is 1.3.
    Returns:
        A HyData instance containing the occlusion factor (0-1) for each point.
    """
    br = band_ratio(rad, kwds.get('num', (400., 450.)), kwds.get('num', (550., 600.)))
    out = br.data[..., 0] - kwds.get('a', 1.20)
    out /= kwds.get('b', 1.3) - kwds.get('a', 1.2)
    out = np.clip(out, 0, 1)
    return out

def calcProjectedOcc(self, geom, sunvec, s=3 ):
    """
    Calculate projected shadows based on the scene geometry and sun direction vector.

    Args:
        geom: a HyCloud instance describing scene geometry.
        sunvec: a (3,) numpy array containing the downward pointing illumination direction.
        s: the size of points in the cloud.
    """
    assert False, "Error: this function has not been implemented yet."

