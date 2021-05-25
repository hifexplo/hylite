"""
Classes containing different occlusion models (e.g. estimators for shadows, sky view factor etc.)
"""
from abc import ABCMeta, abstractmethod

"""
Classes containing different occlusion models (e.g. estimators for shadows, sky view factor etc.)
"""
from abc import ABCMeta, abstractmethod
import numpy as np
import hylite
from hylite.correct.illumination.source import SourceModel

class OccModel(object):
    """
    Base class for modelling occlusions (e.g. shadows) within a scene based on a specific light source (e.g. sunlight, skylight).
    """

    def __init__(self, geometry, source=None):
        """
        Create a new occlusion model.

        *Arguments*:
         - geometry = a HyData instance containing the 3-D scene geometry. This can either be a HyImage instance where the
                      first three bands correspond to xyz, or a HyCloud instance.
         - source = a SourceModel containing the light source geometry (direction).
        """
        self.geometry = geometry
        self.source = source
        self.occ = None
        assert isinstance(self.geometry, hylite.HyScene) \
               or isinstance(self.geometry, hylite.HyCloud), \
            "Error, self.geometry must be an instance of HyScene or HyCloud."
        assert isinstance(source, SourceModel)

    def evaluate(self, **kwds):
        """
        Calculate the fraction of incoming light blocked by this occlusion (0 - 1) as a HyData instance.

        *Optional Keywords*:
         - see child class for details.
        *Returns*:
         - A HyCloud or HyImage instance containing the occlusion factor (0-1) for each point.
        """

        # apply to HyScene (image data)
        if isinstance(self.geometry, hylite.HyScene):
            shape = (self.geometry.image.xdim(), self.geometry.image.ydim())
            xyz = self.geometry.get_xyz().reshape((shape[0] * shape[1], 3)).copy()
            klm = self.geometry.get_normals().reshape((shape[0] * shape[1], 3)).copy()
            rad = self.geometry.image.data.reshape((shape[0] * shape[1], -1)).copy()
            self.occ = hylite.HyImage(self.compute(xyz, klm, **kwds).reshape(shape + (1,)))

        # apply to HyCloud data
        elif isinstance(self.geometry, hylite.HyCloud):
            xyz = self.geometry.xyz.copy()
            klm = self.geometry.normals.copy()
            rad = self.geometry.copy()

            self.occ = self.geometry.cloud.copy()
            self.occ.data = self.compute(xyz, klm, **kwds)[..., 0]
        else:
            assert False, "Error, self.geometry must be an instance of HyScene or HyCloud."

        # return
        return self.occ

    @abstractmethod
    def compute(self, xyz, klm, **kwds):
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
         - cam = a camera instance if self.geometry as a HyCloud instance.
        *Returns*:
         - fig, ax = the plot figure.
        """

        if self.occ is None:
            try:
                self.evaluate()
            except:
                assert False, "Error: call evaluate(...) first to compute occlusions before plotting."

        if isinstance(self.geometry, hylite.HyCloud):
            cam = kwds.get("cam", self.geometry.get_camera(0))
            return self.occ.quick_plot(0, cam)
        else:
            return self.occ.quick_plot(0)


class selfOcc(OccModel):
    """
    Model self occlusions using a band ratio.
    """

    def compute(self, xyz, klm, **kwds):
        """
        Calculate self-occlusions using the angle of incidience (occlude if > 90 deg).
        """
        a = np.dot(klm, -self.source.illuVec)  # calculate angle of incidence
        return (a > 0).astype(np.float)  # return True for self-occ areas
