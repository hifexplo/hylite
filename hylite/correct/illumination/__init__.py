from datetime import datetime
import numpy as np
import pytz
import hylite
import warnings
import datetime

#####################
## Utility functions
######################
def sph2cart(az, el, r=1.0):
    """
    Convert spherical coordiantes to cartesian ones.
    """

    az = np.deg2rad(az)
    el = np.deg2rad(el)
    return np.array( [
        np.sin(az) * np.cos(el),
        np.cos(az) * np.cos(el),
        -np.sin(el) ]) * r


def cart2sph(x, y, z):
    """
    Convert cartesian coordinates to spherical trend, plunge and radius.
    """

    hxy = np.hypot(x, y)
    r = np.hypot(hxy, z)
    az = np.arctan2(x, y)
    el = np.arcsin(z / r)

    while az < 0:
        az += (2*np.pi)
    return np.array([np.rad2deg(az), -np.rad2deg(el), r])

def estimate_sun_vec(lat, lon, time):
    """
    Calculate the sun illumination vector at the specified position and time.
    *Arguments*:
     - lat = the latitude of the position to calculate the sun vector at (in decimal degrees).
     - lon = the longitude of the position to calculate the sun vector at (in decimal degrees).
     - time = the time the dataset was acquired. This, and the position defined by the "pos"
              argument will be used to calculate the sun direction. Should be an instance of datetime.datetime,
              or a tuple containing (timestring, formatstring, pytz timezone).
              E.g. time = ("19/04/2019 12:28","%d/%m/%Y %H:%M", 'Europe/Madrid')
    *Returns*:
     - sunvec = the sun illumination direction (i.e. from the sun to the observer) in cartesian coords
     - azimuth = the azimuth of the sun (bearing towards sun)
     - elevation = the elevation of the sun (angle above horizon)
    """

    # get time
    if isinstance(time, tuple):  # parse time from strings
        tz = time[2]
        time = datetime.strptime(time[0], time[1])
        tz = pytz.timezone(tz)
        time = tz.localize(time)

    # time = time.astimezone(pytz.utc) #convert to UTC

    assert isinstance(time, datetime), "Error - time must be a datetime.datetime instance"
    assert not time.tzinfo is None, "Error - time zone must be specified (e.g. using tz='timezone')."

    # calculate illumination vector from time/position
    import astral.sun
    pos = astral.Observer(lat, lon, 0)
    azimuth = astral.sun.azimuth(pos, time)
    elevation = astral.sun.elevation(pos, time)

    sunvec = sph2cart(azimuth + 180, elevation)  # n.b. +180 flips direction from vector point at sun to vector
    # pointing away from the sun.

    return sunvec, azimuth, elevation

##############################
## Define generic illumination model
###############################

from .occlusion import *
from .reflection import *
from .source import *
from .transmittance import *


class IlluModel(object):
    """
    Combine source, reflection, occlusion and transmittance models to simulate
    the illumination within a scene. A single IlluModel can include multiple
    light sources.
    """

    def __init__(self):
        """
        Create a new (empty) illumination model.
        """
        self.names = []
        self.sources = []  # source models
        self.refl = []  # reflection models
        self.trans = None  # transmittance model
        self.data = None

    def addSource(self, name, source, refl):
        """
        Add a light source (e.g. sun, sky) to this illumination model.
        """

        # check variables
        assert isinstance(name, str), "Error - source name must be a string."
        assert isinstance(source, SourceModel), "Error - source must be a SourceModel instance."
        assert isinstance(refl, ReflModel), "Error - refl must be a ReflModel instance."
        assert refl.isEvaluated(), "Error - reflection model must be evaluated. Call refl.evaluate(...) first!"

        # add to lists
        self.names.append(name)
        self.sources.append(source)
        self.refl.append(refl)

    def setTrans(self, trans):
        """
        Set the target -> sensor transmittance model.
        """
        assert isinstance(trans, TransModel), "Error - trans must be a ReflModel instance."
        assert trans.isEvaluated(), "Error - transmittance model must be evaluated. Call trans.evaluate(...) first!"
        self.trans = trans

    def evaluate(self):
        """
        Evaluate the per-pixel illumination spectra. This (theoretically) gives what the camera would see if
        material reflectance is constant and equal to 1.0. Hence it can be used to convert radiance to reflectance.

        *Arguments*:
         - viewpos = the viewing position.
         - radiance = a
        *Returns*:
         - a HyData instance containing the illumination spectra per pixel. This is also stored in self.data.
        """

        # if there is no light, you can't see!
        if len(self.refl) == 0:
            print("Warning - no reflectance models or illumination sources. Returning 0 (no light).")
            return 0

        # calculate and check number of bands
        nbands = 1  # default to only one band
        for s, r in zip(self.sources, self.refl):
            if isinstance(s.spectra, np.ndarray):
                assert nbands == 1 \
                       or nbands == s.spectra.shape[0], "Error - light sources have different number of bands."
                nbands = s.spectra.shape[0]
            else:
                nbands = max(nbands, r.data.data.shape[
                    -1])  # we could also find number of bands if refl model is fitted to data

        # loop through light sources and accumulate light
        self.data = self.refl[0].data.copy(data=False)
        self.data.data = np.zeros(self.refl[0].data.data.shape[:-1] + (nbands,))  # init output array
        for s, r in zip(self.sources, self.refl):
            if isinstance(s.spectra, float):  # constance illumination
                self.data.data[..., :] += r.data.data * s.spectra
            else:
                self.data.data += r.data.data * s.spectra  # accumulate illuminatino

        # apply transmittance model
        if self.trans is not None:
            self.data.data *= self.trans.data.data

        return self.data

    def fit(self, radiance, method='cfac'):
        """
        Adjust this illumination model using c-factor or minnaert corrections. This takes the self.data array (which
        must be populated by calling self.evaluate( ... ) and computes correlation with the specified radiance. This
        correlation is used to adjust the illumination estimates (per band) using the cfactor or minnaert methods.

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

    def isEvaluated(self):
        return self.data is not None


####################################################
## Functions for constructing illumination models
####################################################


def buildIlluModel_ELC( sundir, sunpanel, refl, occ=None ):
    """
    Build an illumination model containing only sunlight based on calibration panel spectra (using the ELC method).

    *Arguments*:
      - sundir = a (3,) numpy array containing the downwelling sunlight direction.
      - sunpanel =  a list of Panel instance containing one or more fully illuminated calibration panel(s) OR a numpy
                    array that specifies the downwelling sunlight spectra.
      - refl = the reflection model (ReflModel instance) to use for this illumination model.
      - occ = the OccModel used to map shaded or shadowed pixels. Default is None.
    *Returns*:
     - sunIllu = an IlluModel instance describing direct illumination from the sun.
    """
    pass

def buildIlluModel_Joint( sundir, sunpanel, shadepanel, refl, occSun, occSky):
    """
    Build a joint illumination model (sunlight + skylight) from a pair of fully illuminated
    and fully shaded illumination panels, following the method outlined in:

    **Thiele, S.T., et al., "A novel and open-source illumination correction for
    hyperspectral digital outcrop models", IEEE Transactions on Geoscience & Remote Sensing (2021)**

    *Arguments*:
      - sundir = a (3,) numpy array containing the downwelling sunlight direction.
      - sun =  a Panel instance containing the fully illuminated calibration panel OR a numpy
                    array that specifies the downwelling sunlight spectra.
      - shade = a Panel instance containing the fully shaded calibration panel OR a numpy
                    array that specifies the downwelling skylight spectra.
      - refl = the reflection model (ReflModel instance) to use for the sunlight part of this illumination model.
      - occSun = the OccModel used to map sunlight occlusions.
      - occSky = the OccModel used to map sky view factor.
    *Returns*:
     - sunIllu = an IlluModel instance describing direct illumination from the sun.
     - skyIllu = an IlluModel instance describing indiriect illumination from the sky.
    """
    pass

def estimateIlluModel_Joint( radiance, sunpanel, refl, occSun, occSky ):
    """
    Estimate a joint illumination model (sunlight + skylight) from a fully illuminated (sun) panel and an
    occlusion model specifying shade pixels following the method outlined in:

    **Thiele, S.T., et al., "A novel and open-source illumination correction for
    hyperspectral digital outcrop models", IEEE Transactions on Geoscience & Remote Sensing (2021)**

    *Arguments*:
      - sundir = a (3,) numpy array containing the downwelling sunlight direction.
      - radiance =  a HyData instance containing per pixel or per-point radiance.
      - refl = the reflection model to use for the sunlight part of this illumination model. Default is
               OrenNayar.
      - occSun = the OccModel used to identify fully shaded / shadowed pixels.
      - occSky = the OccModel used to map sky view factor.
    *Returns*:
     - sunIllu = an IlluModel instance describing direct illumination from the sun.
     - skyIllu = an IlluModel instance describing indiriect illumination from the sky.
    """
    pass

##########################
## Correction functions
##########################
def rad2refl(self, radiance, illu):
    """
    Convert from radiance to reflectance given the specified illumination models and viewing direction.

    *Arguments*:
     - radiance = a HyData instance containing at-sensor radiance spectra to correct.
     - illu = a list of IlluModel instances describing scene illumination.
     - trans = a transmittance model describing atmospheric path radiance effects. Default is None (no path radiance).
    """
    pass

def refl2rad(self, reflectance, illu):
    """
    Convert from reflectance to radiance given the specified illumination model and viewing direction.

     *Arguments*:
     - reflectance = a HyData instance containing material reflectance spectra.
     - illu = a list of IlluModel instances describing scene illumination.
     - camera = Camera instance containing the viewing position and direction. Can be None (default) for some illumination models.
     - trans = a transmittance model describing atmospheric path radiance effects. Default is None (no path radiance).
    """
    pass
