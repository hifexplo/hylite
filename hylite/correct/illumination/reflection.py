from abc import ABCMeta, abstractmethod

import numpy as np
import matplotlib.pyplot as plt

import hylite
from hylite import HyCloud, HyScene
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

        assert isinstance(geometry, HyScene) or \
               isinstance(geometry,
                          HyCloud), "Error - scene geometry must either be a HyImage, HyCloud or HyScene instance."
        self.geometry = geometry
        self.data = None  # modelled reflectance factors will be put here
        self.prior = None  # if fitting is used, this stores initial estimates

    def fit(self, radiance, method='cfac'):
        """
        Adjust this reflectance model using c-factor or minnaert corrections.

        *Arguments*:
         - radiance = a HyData instance containing point or pixel radiance spectra to fit reflectance too.
         - method = the correction method to apply. Options are 'cfac' or 'minnaert'. Default is 'cfac'.

        *Returns:
         - A HyData instance as returned by evaluate( ... ), but adjusted using the cfac or minnaert method. This adjustment
           is calculated per-band, so the returned HyData instance will contain the same number of bands as the passed
           radiance data. The self.data variable will also be updated accordingly, and the uncorrected model stored as self.prior.
        """

        assert self.data is not None, "Error - please compute reflectance model using self.compute(...) before fitting step."

        # get initial reflectance estimates and check input shape is appropriate
        if self.prior is None:
            self.prior = self.data  # define prior

        alpha = self.prior.X()[...,0]

        self.r25, self.r50, self.r75 = np.nanpercentile(alpha, (
        25, 50, 75))  # reference value to compare adjusted reflectance with

        # get radiance vector and deal with invalid pixels
        X = radiance.X().copy()
        assert alpha.shape[0] == X.shape[0], "Error - radiance data has incorrect shape."

        nans = np.logical_or(np.logical_not(np.isfinite(X).any(axis=0)), (X == 0).all(
            axis=0))  # replace any bands that are all nan with 1.0 (hack that avoids issues when calculating regressions)
        X[:, nans] = 1.0  # regressions will now give 0 slope ( == 0 correction ) for nan bands
        X[
            X <= 0] = 1e-6  # get rid of zeros and negative numbers [ should be impossible and causes errors for log in minnaert]

        # calculate direct illumination mask
        i_mask = alpha > 0.01  # non-illuminated pixels; remove from regression.

        # calculate mask to use for regressions
        mask = np.isfinite(X).all(axis=1) & (X != 0).any(axis=1) & np.isfinite(alpha) & i_mask
        assert mask.any(), "Error - all pixels are invalid. Check shadow mask and remove bands that are all nan or 0."

        # regress against observed reflectance and calculate correction
        if 'cfac' in method.lower():
            (self.intercept, self.slope), resid = np.polynomial.polynomial.polyfit(alpha[mask], X[mask, :], 1,
                                                                                   full=True)
            self.cfac = self.intercept / self.slope
            m = (alpha[:, None] + self.cfac[None, :]) / (1 + self.cfac[None, :])
        elif 'minn' in method.lower():
            (self.intercept, self.slope), resid = np.polynomial.polynomial.polyfit(np.log(alpha[mask]),  # x
                                                                                   np.log(X[mask, :]), 1,
                                                                                   full=True)  # y
            m = np.power((1.0 / alpha[:, None]), -self.slope[None, :])
        else:
            assert False, "Error - %s is an unknown correction method." % method

        self.residual = np.sqrt((resid[0] / X.shape[0]))

        # store adjusted reflectance factor
        self.data = radiance.copy()
        self.data.data = m.reshape(radiance.data.shape)

        # add nan bands back in
        mask = np.logical_not(np.isfinite(radiance.data).any(axis=(0, 1)))
        self.data.data[..., mask] = np.nan
        self.residual[mask] = np.nan
        self.intercept[mask] = np.nan
        self.slope[mask] = np.nan

        # return it
        return self.data

    def quick_plot(self, **kwds):
        """
        Plot this model (must be evaluated first).

        *Keywords*:
         - keyword arguments are passed to self.data.quick_plot( ... ).
        """
        assert self.data is not None, "Error - please compute reflectance model using self.compute(...) first."
        kwds['band'] = 0
        kwds['cmap'] = kwds.get('cmap', 'gray')
        return self.data.quick_plot(**kwds)

    def plot_fit(self):
        assert (self.data is not None) and (
                    self.prior is not None), "Error - no model has been fitted. Do so using self.fit(...)."
        fig, ax = plt.subplots(2, 2, figsize=(18, 8))
        x = self.data.get_wavelengths()

        # plot prior
        ax[0, 0].set_title("Prior reflectance factor")
        self.prior.quick_plot(0, cmap='gray', vmin=0, vmax=1.1, ax=ax[0, 0])

        # plot adjusted
        n = self.data.band_count()
        b = [x[int(n * i)] for i in (.25, .5, .75)]
        ax[0, 1].set_title("Adjusted reflectance factor (%d, %d, %d nm)" % tuple(b))
        self.data.quick_plot(b, vmin=0, vmax=1.1, ax=ax[0, 1])

        ax[1, 0]
        # plot regression coeff
        ax[1, 0].set_title("Regression coefficients")
        ax[1, 0].plot(x, self.intercept, color='b', alpha=0.4, label='Intercept')
        ax[1, 0].plot(x, self.slope, color='g', alpha=0.4, label='Slope')

        # plot bounds
        mask = np.isfinite(self.intercept)
        mask = np.hstack([mask, mask[::-1]])
        ymin = self.intercept - self.residual
        ymax = self.intercept + self.residual
        ax[1, 0].fill(np.hstack([x, x[::-1]])[mask], np.hstack([ymin, ymax[::-1]])[mask], color='b', alpha=0.2)

        ymin = self.slope - self.residual
        ymax = self.slope + self.residual
        ax[1, 0].fill(np.hstack([x, x[::-1]])[mask], np.hstack([ymin, ymax[::-1]])[mask], color='g', alpha=0.2)
        ax[1, 0].legend()

        ax[1, 1].set_title("Adjustment fraction")
        self.data.plot_spectra(ax=ax[1, 1])
        ax[1, 1].axhline(self.r25, alpha=0.2, label='Original (quartile)')
        ax[1, 1].axhline(self.r75, alpha=0.2)
        ax[1, 1].axhline(self.r50, alpha=0.7, label='Original (median)')
        [ax[1, 1].axvline(_x, color=c) for _x, c in zip(b, ['r', 'g', 'b'])]
        ax[1, 1].set_ylabel("Reflected fraction")
        return fig, ax

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
            outshape = (self.geometry.point_count(),)
            out = self.geometry.copy(data=False)

            # get geometry data
            klm = self.geometry.normals.reshape((-1, 3))
            xyz = self.geometry.xyz.reshape((-1, 3))
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