from abc import ABCMeta, abstractmethod

import numpy as np
import matplotlib.pyplot as plt

import hylite
from hylite import HyCloud, HyImage, HyScene
from hylite.correct.illumination.source import SourceModel
from hylite.correct.illumination.occlusion import OccModel

def estimate_incidence(normals, sunvec):
    """
    Utility function to estimate the cosine of incidence angles based on normals and calculated sun position.

    *Arguments*:
     - normals = either: (1) a HyImage with band 0 = nx, band 1 = ny and band 2 = nz, (2) HyCloud instance containing
                 normals, or (3) mx3 numpy array of normal vectors.
     - sunvec = a numpy array containing the sun illumination vector (as calculated by estimate_sun_vec(...)).

    *Returns*:
     - list of incidence angles matching the shape of input data (but with a single band only).
    """

    # extract normal vectors
    if isinstance(normals, hylite.HyCloud):
        N = normals.normals[:, :3]
        outshape = normals.point_count()
    elif isinstance(normals, hylite.HyImage):
        N = normals.get_raveled()[:, :3]
        outshape = (normals.xdim(), normals.ydim())
    else:
        N = normals.reshape((-1, 3))
        outshape = normals.shape[:-1]

    # normalize normals (just to be safe)
    N = N / np.linalg.norm(N, axis=1)[:, None]

    # calculate cosine of angles used in correction
    cosInc = np.dot(-N, sunvec)  # cos incidence angle

    # return in same shape as original data
    return cosInc.reshape(outshape)


class ReflModel(object):
    """
    Model the fraction of downwelling light reflected into the sensor from each pixel based on scene geometry and/or
    statistical reflection models. Generally this takes the form of a physical reflection model ( e.g., Lambert, Oren-Nayer)
    that can be statistically adjusted using an empirical correction (cfac, minnaert) if necessary. Note that the physical models
    will calculate this as a constant (wavelength independent) value, but that statistical adjustments expand this to a per-band
    reflectance factor. Hence ReflModel.data will be a HyData instance containing either 1 band, or the number of bands that were
    used during empirical correction.
    """

    def __init__(self, geometry):
        """
        Create a new reflectance model.

        *Arguments*:
         - geometry = a HyData instance containing the 3-D geometry (xyz) and associated normal vectors (klm). This
                      can either be a HyCloud instance with normals, or a HyScene instance.
        """
        assert isinstance(geometry, HyImage) or \
               isinstance(geometry, HyScene) or \
               isinstance(geometry,
                          HyCloud), "Error - scene geometry must either be a HyImage, HyCloud or HyScene instance."
        self.geometry = geometry
        self.data = None  # modelled reflectance factors will be put here
        self.prior = None  # if fitting is used, this stores initial estimates

    def quick_plot(self, **kwds):
        """
        Plot this model (must be evaluated first).

        *Keywords*:
         - keyword arguments are passed to self.data.quick_plot( ... ).
        """
        assert self.data is not None, "Error - please compute reflectance model using self.evaluate(...) first."
        kwds['band'] = 0
        kwds['cmap'] = kwds.get('cmap', 'gray')
        return self.data.quick_plot(**kwds)

    def X(self):
        """
        Return a ravelled form of the underlying reflectance values. Equivalent to HyData.X().
        """
        assert self.data is not None, "Error - please compute reflectance model using self.evaluate(...) first."
        return self.data.X()

    def evaluate(self, source, viewpos=None, occ=None):
        """
        Calculate the per-point and per band reflected light fraction (alpha).

        *Arguments*
         - source = the illumination source to compute reflectance from.
         - viewpos = numpy array with the viewing position to evaluate reflectance from (for BRDF models). Default is None.
         - occ = an occlusion model to apply. Default is None. If a list is passed, these will all be applied.
        *Returns*:
         - A HyData instance containing a single band with the per-point or per-pixel reflectance factor.
        """

        # gather necessary information
        xyz = None  # raveled position vector
        klm = None  # raveled normal vector
        if isinstance(self.geometry, HyCloud):
            # create output object
            outshape = (self.geometry.point_count(),1)
            out = self.geometry.copy(data=False)

            # get geometry data
            klm = self.geometry.normals.reshape((-1, 3))
            xyz = self.geometry.xyz.reshape((-1, 3))
        elif isinstance( self.geometry, HyImage):
            # create output object
            outshape = (self.geometry.data.shape[:-1] + (1,))
            out = self.geometry.copy(data=False)

            # get geometry data
            klm = self.geometry.X()[:, :3]
            xyz = self.geometry.X()[:, 3:6]
        elif isinstance(self.geometry, HyScene):
            # create output object
            outshape = (self.geometry.image.data.shape[:-1] + (1,))
            out = self.geometry.image.copy(data=False)

            # get geometry data
            klm = self.geometry.normals.reshape((-1, 3))
            xyz = self.geometry.xyz.reshape((-1, 3))
        else:
            assert False, "Error - scene geometry must either be a HyImage, HyCloud or HyScene instance."

        # get incidence source direction and viewing direction.
        i = None
        v = None
        if source is not None:
            assert isinstance(source, SourceModel), "Error - illu must be a SourceModel."
            i = source.illuVec
        if viewpos is not None:
            assert isinstance(viewpos, np.ndarray), "Error - view position must be a numpy array."
            v = xyz - viewpos  # calculate view vectors
            v /= np.linalg.norm(v, axis=-1)[..., None]  # normalize

        # calculate reflection ignoring occlusions
        alpha = self.calculateReflection(i, klm, v)

        # apply occlusions
        if occ is not None:
            if not isinstance(occ, list):
                occ = [occ]
            for o in occ:
                assert o.data != None, "Error - please evaluate all occlusion models before using."
                _o = o.data.data.ravel()
                assert _o.shape == alpha.shape, "Error - occlusion and alpha shapes do not match; %s != %s." % (
                _o.shape, alpha.shape)
                alpha *= (1 - _o)  # apply occlusion mask

        # reshape and return
        out.data = alpha.reshape(outshape)
        self.data = out  # store results
        return out  # return

    def isEvaluated(self):
        """
        Return true if this occlusion model has been evaluated.
        """
        return self.data is not None

    @abstractmethod
    def calculateReflection(self, I, N, V):
        """
        Function implemented in child classes to calculate e.g. Lambert or Oren-Nayar reflection.

        *Arguments*:
         - I = a (3,) numpy array containing the downward pointing illumination direction.
         - N = a (n,3) array containing a list of point/pixel normal vectors (orientations).
         - V = a (n,3) array containing the viewing direction (normalised) for each point/pixel.
        *Returns*:
         - Should return a numpy array of shape (n,) containing the reflection fractions (0 - 1) at each point/pixel.
        """
        pass


########################################
## Reflectance model implementations
#######################################
class IdealRefl(ReflModel):
    """
    A perfectly reflective material (reflects all downwelling light to the sensor).
    """

    def calculateReflection(self, I, N, V):
        """
        Simply returns 100% reflectance for each point/pixel.
        """
        return N[:, 0] / N[:, 0]  # return list of ones and nans.


class LambertRefl(ReflModel):
    """
    A perfectly reflective material (reflects all downwelling light to the sensor).
    """

    def calculateReflection(self, I, N, V):
        """
        Return the cosine of the incidence angle as per lamberts law.
        """

        assert I is not None, "Error: Illumination direction needs to be specified for Lambert Reflection."
        assert N is not None, "Error: Normal vectors need to be defined for Lambert Reflection."

        # calculate alpha = cos( incidence angle ) = I . N
        a = estimate_incidence(N, I)
        a[a < 0] = 0  # remove backfaces
        return a


class OrenNayar(ReflModel):
    """
    Reflection from a rough surface (specified by roughness parameter, which is the standard deviation (in radians) of
    surface orientation at each point/pixel). Roughness defaults to 18 degrees (sqrt[0.1] radians).
    """

    r2 = 0.1

    def setRoughness(self, r):
        self.r2 = r ** 2

    def getRoughness(self):
        return np.sqrt(self.r2)

    def setRoughnessSquared(self, r2):
        self.r2 = r2

    def getRoughnessSquared(self):
        return self.r2

    def calculateReflection(self, I, N, V):
        """
        Return the cosine of the incidence angle as per lamberts law.
        """
        # calculate roughness terms
        A = 1.0 - (0.5 * self.r2) / (self.r2 + 0.33)
        B = (0.45 * self.r2) / (self.r2 + 0.09)

        LdotN = estimate_incidence(N, I)  # I . N [ = cos (incidence angle) ]
        VdotN = np.sum(V * N, axis=-1)  # V . N [ = cos( viewing angle ) ]

        # remove backfaces
        irradiance = LdotN.copy()
        irradiance[irradiance < 0] = 0

        # convert cosines to radians
        angleViewNormal = np.arccos(VdotN);
        angleLightNormal = np.arccos(LdotN);

        a = (V - N * VdotN[:, None]) / np.linalg.norm((V - N * VdotN[:, None]), axis=-1)[:, None]
        b = (I - N * LdotN[:, None]) / np.linalg.norm((I - N * LdotN[:, None]), axis=-1)[:, None]

        angleDiff = np.sum(a * b, axis=-1)
        angleDiff[angleDiff < 0] = 0

        alpha = np.nanmax(np.dstack([angleViewNormal, angleLightNormal]), axis=-1)[0, :]
        beta = np.nanmin(np.dstack([angleViewNormal, angleLightNormal]), axis=-1)[0, :]

        # return
        return irradiance * (A + B * angleDiff * np.sin(alpha) * np.tan(beta))