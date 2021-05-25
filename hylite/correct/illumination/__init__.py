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
## Generic illumination model
###############################

from .occlusion import *
from .reflection import *
from .source import *
from .transmittance import *

class IlluModel( object ):
    """
    Combine source and reflection models (as well as optional occlusion and transmittance models) to
    model the illumination effect of a single light source (e.g. the sun or sky). For modelling multiple light
    sources IlluModels can be added together to calculate overall scene illumination.
    """

    def __init__(self, sourceModel, reflModel, transModel=None):
        """
        Create a new illumination model.

        *Arguments*:
         - sourceModel = the light source properties.
         - reflModel = the material reflectance model (e.g. Oren-Nayar, Lambert, etc.)
         - occModel = the occlusion model (shadows, sky view factor etc.). Default is None.
         - transModel = the atmospheric transmittance model for modelling path radiance. Default is None.
        """

        self.source = sourceModel
        self.refl = reflModel
        self.transModel = transModel

    def get_illumination(self, viewpos, radiance):
        """
        Return the per-pixel illumination spectra as viewed through the specified camera. This (theoretically)
        gives what the camera would see if material reflectance is constant and equal to 1.0. Hence it can be used
        to convert radiance to reflectance.

        *Arguments*:
         - viewpos = the viewing position.
         - radiance = a
        *Returns*:
         - a HyImage instance containing the illumination spectra per pixel.
        """

        # get illumination spectra
        I = self.source.r()

        # get reflectivity model
        a = self.refl.evaluate() # n.b. this can either be shaped (x,y,b) or (x,y,1)

        # get transmittance factors
        t = np.ones_like(a)
        if self.transModel is not None:
            t = self.transModel.evaluate()
        pass


####################################################
## Functions for constructing illumination models
####################################################
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
def rad2refl(self, radiance, illu, trans=None):
    """
    Convert from radiance to reflectance given the specified illumination models and viewing direction.

    *Arguments*:
     - radiance = a HyData instance containing at-sensor radiance spectra to correct.
     - illu = a list of IlluModel instances describing scene illumination.
     - trans = a transmittance model describing atmospheric path radiance effects. Default is None (no path radiance).
    """
    pass

def refl2rad(self, reflectance, illu, trans):
    """
    Convert from reflectance to radiance given the specified illumination model and viewing direction.

     *Arguments*:
     - reflectance = a HyData instance containing material reflectance spectra.
     - illu = a list of IlluModel instances describing scene illumination.
     - camera = Camera instance containing the viewing position and direction. Can be None (default) for some illumination models.
     - trans = a transmittance model describing atmospheric path radiance effects. Default is None (no path radiance).
    """
    pass
