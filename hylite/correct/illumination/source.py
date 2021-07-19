"""
Simple class for encapsulating light source properties.
"""
import hylite
from hylite.correct.illumination import sph2cart
import numpy as np

class SourceModel(object):
    """
    Class for encapsulating light source propertes (spectra, position, etc.)
    """
    def __init__(self, direction=None, spectra=1.0, ):
        """
        Create a new light source.

        *Arguments*:
         - direction = the (downward pointing) illumination vector. Can be none if illumination is omnidirectional.
         - spectra = a n-d numpy array defining the radiance of the light source in each band. E.g. downwelling illumination
                     calculated using calibration panels. Default is 1.0 (pure white).
        """

        self.spectra = spectra
        self.illuVec = direction
        if self.illuVec is not None:
            if len(self.illuVec) == 3:
                if self.illuVec[2] > 0: # z-component should always point downwards
                    self.illuVec *= -1
            elif len(self.illuVec) == 2:
                self.illuVec = sph2cart( self.illuVec[0] + 180, self.illuVec[1] ) # n.b. +180 flips direction from vector point at sun to vector
            else:
                assert False, "Error: direction must be a (3,) cartesian vector or a (2,) vector with azimuth, elevation (in degrees)."

    def r(self):
        """
        Return the radiance spectra for this source.sdfdfdfdfd
        """
        return self.spectra

    def d(self):
        """
        Return the direction vector for this source, or None if the source is omnidirectional.
        """
        return self.illuVec

    def fit_to_radiance(self, rad, occ, panel, refl=None, skyview=None, hori_adj=0.0):
        """
        Use this SourceModel's position and the provided radiance data, occlusion model(s) and calibration panels to
        separate skylight and sunlight spectra using the method defined in:

        Thiele, S. T., Lorenz S., Kirsch, M., Gloaguen, R., “A novel and open-source illumination correction
        for hyperspectral digital outcrop models.” Transactions on Geoscience and Remote Sensing (2021).

        N.B. this returns two new SourceModel instances, and does not modify the original (but uses its
        direction to model the sunlight direction).

        *Arguments*
            - rad = measured radiance (HyImage or HyCloud instance).
            - occ = an OccModel (or list of OccModel) instances used to extract fully shaded and fully lit radiance spectra.
            - panel = a calibration panel to use to estimate downwelling (skylight + sunlight) illumination. This must have a
                      defined normal vector (see panel.get_normal(...) for details on how to estimate this, or use
                      panel.set_normal(...) to set it explicitely).
        *Recommended Arguments*:

        These are not essential, but should be defined to ensure results are as accurate as possible.

            - refl = a HyImage, HyCloud or ReflModel instance (e.g. OrenNayar or LambertRefl instance) that defines
                     the proportion of reflected sunlight throughout scene. Underlying data shape must match the data
                     contained in `rad`. Default is None (assume all points/pixels reflected fraction = 1.0).
            - skyview = Either HyData instance containing sky view factors in the first band or a SkyOcc instance. Default
                        is None (assume sky view = 1.0 across the scene).
            - hori_adj = An adjustment factor to apply when calculating the sky view fraction of the calibration panel (see
                         Panel.get_skyview(...) for details). Default is 0.0.
        """

        # ravel radiance data
        assert isinstance(rad, hylite.HyData), 'Error - rad must be a HyCloud or HyImage instance'
        rad = rad.X()

        # get sky view factor data (if provided, otherwise assume = 1.0)
        s = np.full(rad.shape[0], 1.0)
        if skyview is not None:
            s = skyview.X()[:, 0]  # get sky view factors

        # get reflectance factors (if provided, otherwise assume = 1.0)
        a = np.full(rad.shape[0], 1.0)
        if refl is not None:
            a = refl.X()[:, 0]  # get sky view factors

        # get combined illumination factors (by multiplication)
        i = np.full(rad.shape[0], 1.0)
        if not isinstance(occ, list):  # wrap in list
            occ = [occ]
        for o in occ:
            i *= o.data.X()[:, 0]

        # masks of fully illuminated (illu) and fully shaded (shad) spectra
        shad = i >= 0.95
        illu = i <= 0.05

        # get panel properties
        assert panel.normal is not None, "Error - panel must have defined normal vector. Use panel.set_normal(...) to set it."
        pa = panel.get_alpha(self.illuVec)
        ps = panel.get_skyview(hori_adj)

        # estimate skylight spectra
        ar = s[:, None] / rad
        sigma = (np.nanmedian(ar[shad], axis=0) - np.nanmedian(ar, axis=0)) / np.nanmedian(a[:, None] / rad, axis=0)
        S_est = (panel.get_mean_radiance() / panel.get_reflectance()) * (1 / (ps + sigma * pa))
        I_est = (1 / pa) * ((panel.get_mean_radiance() / panel.get_reflectance()) - ps * S_est)

        # occasionally sun can become negative... force this to 0
        S_est[ S_est < 0 ] = 0.0
        I_est[ I_est < 0 ] = 0.0

        # put into SourceModel instances and return
        sky = SourceModel(None, S_est)
        sun = SourceModel(self.illuVec, I_est)
        return sun, sky