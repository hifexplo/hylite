"""
Classes containing different occlusion models (e.g. estimators for shadows, sky view factor etc.)
"""

from abc import ABCMeta, abstractmethod
import numpy as np
import hylite
from hylite.correct.illumination.source import SourceModel
from hylite.analyse import band_ratio

"""
Classes containing different occlusion models (e.g. estimators for shadows, sky view factor etc.)
"""
from abc import ABCMeta, abstractmethod

"""
Classes containing different occlusion models (e.g. estimators for shadows, sky view factor etc.)
"""
from abc import ABCMeta, abstractmethod


class OccModel(object):
    """
    Base class for modelling occlusions (e.g. shadows) within a scene based on a specific light source (e.g. sunlight, skylight).
    """

    def __init__(self, geometry):
        """
        Create a new occlusion model.

        *Arguments*:
         - geometry = a HyData instance containing the 3-D scene geometry. This can either be a HyImage instance where the
                      first three bands correspond to xyz, or a HyCloud instance.
        """
        self.geometry = geometry
        self.data = None
        assert isinstance(self.geometry, hylite.HyScene) \
               or isinstance(self.geometry, hylite.HyCloud), \
            "Error, self.geometry must be an instance of HyScene or HyCloud."

    def evaluate(self, source=None, **kwds):
        """
        Calculate the fraction of incoming light blocked by this occlusion (0 - 1) as a HyData instance.

        *Arguments*:
         - source = the light source to calculate occlusions from.
        *Optional Keywords*:
         - see child class for details.
        *Returns*:
         - A HyCloud or HyImage instance containing the occlusion factor (0-1) for each point. This dataset
           is also stored as self.data.
        """

        if source is not None:
            assert isinstance(source, SourceModel), "Error, source must be an instance of SourceModel (or None)."

        # apply to HyScene (image data)
        if isinstance(self.geometry, hylite.HyScene):
            # gather relevant data
            shape = (self.geometry.image.xdim(), self.geometry.image.ydim())
            xyz = self.geometry.get_xyz().reshape((shape[0] * shape[1], 3)).copy()
            klm = self.geometry.get_normals().reshape((shape[0] * shape[1], 3)).copy()
            rad = self.geometry.image.data.reshape((shape[0] * shape[1], -1)).copy()
            rad = hylite.HyData(rad)
            rad.set_wavelengths(self.geometry.image.get_wavelengths())

            # compute occlusions
            self.data = hylite.HyImage(self.compute(source, xyz, klm, rad, **kwds).reshape(shape + (1,)))

        # apply to HyCloud data
        elif isinstance(self.geometry, hylite.HyCloud):
            # gather relevant data
            xyz = self.geometry.xyz.copy()
            klm = self.geometry.normals.copy()
            rad = None
            if hasattr(self.geometry, 'data'):
                rad = self.geometry.copy()
            self.data = self.geometry.copy( data=False )
            self.data.data = self.compute(source, xyz, klm, rad, **kwds)
        else:
            assert False, "Error, self.geometry must be an instance of HyScene or HyCloud."

        # return
        return self.data

    @abstractmethod
    def compute(self, source, xyz, klm, rad, **kwds):
        """
        *Arguments*:
         - xyz = a (n,3) array of point or pixel coordinates in 3-D space.
         - klm = a (n,3) array of point or pixel normal vectors.
         - rad = a (n,3) array of point or pixel radiance, or None if not passed (default).

        *Keywords*:
         - keywords can be supplied as needed by implementing children classes.

        *Returns*:
            A numpy array of shape (n,) that contains 0 for fully shaded pixels and 1 for fully illuminated ones.
        """
        pass

    def quick_plot(self, **kwds):
        """
        Plot this illumination mask. If not previously calculate, then self.evaluate() will be called.

        *Optional Keywords*:
         - keywords are passed to self.data.quick_plot( ... ).
        *Returns*:
         - fig, ax = the plot figure.
        """

        if self.data is None:
            try:
                self.evaluate()
            except:
                assert False, "Error: call evaluate(...) first to compute occlusions before plotting."

        kwds['band'] = 0
        kwds['cmap'] = 'gray_r'

        return self.data.quick_plot(**kwds)

    def isEvaluated(self):
        """
        Return true if this occlusion model has been evaluated.
        """
        return self.data is not None

class SelfOcc(OccModel):
    """
    Model self occlusions using a band ratio.
    """

    def compute(self, source, xyz, klm, rad, **kwds):
        """
        Calculate self-occlusions using the angle of incidience (occlude if > 90 deg).
        """
        a = np.dot(klm, -source.illuVec)  # calculate angle of incidence
        return (a < 0).astype(np.float)  # return True for self-occ areas


class BandOcc(OccModel):
    """
    Model occlusion using a band ratio to identify shadows that are significantly "bluer" than their surroundings.
    """

    def compute(self, source, xyz, klm, rad, **kwds):
        """
        Calculate the fraction of incoming light blocked by this occlusion (0 - 1) as a HyData instance.

        *Keywords*:
         - radiance = Radiance data to calculate band ratio with.
         - num = Numerator of band ratio, as defined by hylite.analyse.band_ratio. Default is (400,450) nm.
         - den = Denominator of band ratio, as defined by hylite.analyse.band_ratio. Default is (550,600) nm.
         - a = the threshold for complete shadow (values > this are full shadow). Default is 1.20.
         - b = the threshold for complete sun (values < this are full sun). Default is 1.3.
        *Returns*:
         - A HyCloud instance containing the occlusion factor (0-1) for each point.
        """
        br = band_ratio(rad, kwds.get('num', (400., 450.)), kwds.get('num', (550., 600.)))
        out = br.data[..., 0] - kwds.get('a', 1.20)
        out /= kwds.get('b', 1.3) - kwds.get('a', 1.2)
        out = np.clip(out, 0, 1)
        return out

class SkyOcc(OccModel):
    """
    Model occlusion of the sky by loading precalculated sky view factors. These could be calculated using e.g. CloudCompare.

    N.B. this simply assumes that self.geometry.data[...,0] gives an array of sky view factors. Hence when creating this
    type of occlusion, the 'geometry' argument should be a HyCloud or HyScene instance with band 0 = sky view factors
    (ranging from 0 [ no view of sky ] to 1 [ full view of sky ].
    """
    def compute(self, source, xyz, klm, rad, **kwds):
        """
        Calculate the fraction of incoming light blocked by this occlusion (0 - 1) as a HyData instance.

        *Returns*:
         - A numpy array containg occlusion factors for each point.
        """
        if isinstance(self.geometry, hylite.HyCloud):
            return 1. - self.geometry.data
        elif isinstance(self.geometry, hylite.HyScene):
            return 1. - self.geometry.cloud.render(self.geometry.camera, 0, fill_holes=True).data
        else:
            assert False, "Error, unknown geometry type."

        return (a < 0).astype(np.float)  # return True for self-occ areas
