import numpy as np
from datetime import datetime
import pytz
import hylite

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

def estimate_incidence( normals, sunvec ):
    """
    Estimate the cosign of incidence angles based on normals and calculated sun position.

    *Arguments*:
     - normals = either: (1) a HyImage with band 0 = nx, band 1 = ny and band 2 = nz, (2) HyCloud instance containing
                 normals, or (3) mx3 numpy array of normal vectors.
     - sunvec = a numpy array containing the sun illumination vector (as calculated by estimate_sun_vec(...)).

    *Returns*:
     - list of incidence angles matching the shape of input data (but with a single band only).
    """

    # extract normal vectors
    if isinstance( normals, hylite.HyCloud):
        N = normals.normals[:,:3]
        outshape = normals.point_count()
    elif isinstance( normals, hylite.HyImage ):
        N = normals.get_raveled()[:, :3]
        outshape = (normals.xdim(), normals.ydim())
    else:
        N = normals.reshape((-1, 3))
        outshape = (normals.shape[0], normals.shape[1])

    # normalize normals (just to be safe)
    N = N / np.linalg.norm(N, axis=1)[:, None]

    # calculate cosign of angles used in correction
    cosInc = np.dot(-N, sunvec)  # cos incidence angle

    # return in same shape as original data
    return cosInc.reshape( outshape )

def estimate_ambient(data, cosInc, shadow_mask=None):
    """
    Estimate ambient light conditions based on the difference between illuminated data (cosinc > 0) vs shadowed/occluded
    data (cosinc < 0).

    *Arguments*:
     - data = a hyperspectral dataset (HyData incidence)
     - cosinc = the cosign of the incidence angle for each pixel/point.
     - shadow_thresh = boolean array with True for pixels that are shadows. Or None if no shadow mask applied.
    """

    # extract pixel vectors
    if isinstance(data, hylite.HyData):
        X = data.get_raveled()
    else:
        assert isinstance(data,np.ndarray),"Error - unknown data type."
        X = data
    cosInc = cosInc.reshape(X.shape[:-1])
    assert X.shape[0] == cosInc.shape[0], "Error - X and cosinc must contain the same number of data points."

    # compute diffuse component using lamberts cosign law
    #d = 1.0 / cosInc

    # calculate shadow mask
    if shadow_mask is None:
        obsc = cosInc < -0.1 #calculate non illuminated pixels
    else:
        # split the pixels into illuminated vs non illuminated
        obsc = np.logical_or(cosInc < -0.1, shadow_mask.reshape(X.shape[0]))
    illu = np.logical_not(obsc)

    assert len(illu) > 100, "Error - not enough illuminated pixels to estimate ambient light."
    assert len(obsc) > 100, "Error - not enough shaded pixels to estimate ambient light."

    #also check that obscured pixels are darker than illuminated pixels...
    if not (np.nanmedian(illu, axis=0) >= np.nanmedian(obsc, axis=0)).all():
        print("Warning - shaded pixels are brighter than illuminated pixels in some bands? This will result in negative light"
              "intensity....")

    a = np.nanmedian((cosInc[illu, None] / X[illu, :]), axis=0) / (
                np.nanmedian(1.0 / X[obsc], axis=0) - np.nanmedian(1.0 / X[illu], axis=0))

    return a

def correct_topo(data, cosInc, method="cfac", **kwds):
    """
    Apply topographic correction to a HyImage or HyCloud dataset.

    *Arguments*:
     - data = a HyImage or HyCloud instance. If a HyCloud instance then normals must be specified. If a HyImage instance
              then the normals keyword must be passed.
     - cosinc = the cosign of the incidence angle for each pixel/point.
     - method = the topographic correction method to use (string). Options are:
        - "cos" = correction using the cosign method.
        - "icos" = correction using the improved cosign method.
        - "percent" = correction using the percent method.
        - "cfac" = correction using the c-factor method (default).
        - "minnaert" = correction using the minnaert method.
        - "minnaert_slope" = correction using the minnaert method including slope.
        - "ambient" = correction using the ambient method.
    *General Keywords*:
     - target = the target orientation to normalise to. Default is "normal", which normalises reflectance as though the surface
                at each data point is illuminated normal to its surface. Use "horizontal" to normalise to a horizontal plane instead.
     - thresh = the maximum multiplication factor for topographic correction (to avoid unrealisticly large values). Default is 10.0.
     - sunvec = the sun vector to use if target normalisation is "horizontal". Not used for "normal" normalisation.
     - inplace = True if the correction should be made in-place (i.e. modifie the passed data array). Default is True.

    *Method specific keywords*:

    "ambient":
     - ambient = ambient light intensity per band for method="ambient", as returned by estimate_ambient(...).
                 If not provided this is calculated automatically.
     - shadow_mask = array containing true if a pixel is a shadow, false otherwise.

     "minnaert_slope":
      - slope = an array of slope values, in degrees (required).

    *Returns*:
     - (m, c) = linear correction factors, such that i_corrected = m*i_initial + c
     - illum_mask = boolean array containing False for datapoints/pixels that are not directly illuminated.

    """

    # check type
    assert isinstance(data,
                      hylite.HyData), "Error - data must be an instance of HyData or it's derivatives (HyCloud or HyImage)."

    # extract pixel vectors
    X = data.get_raveled()

    # replace any bands that are all nan with 1.0 (hack that avoids issues when calculating regressions)
    nans = np.logical_not( np.isfinite(X).any(axis=0) ) & (X == 0).all(axis=0)
    X[ :, nans ] = 1.0

    # extract cosInc
    cosInc = cosInc.reshape(X.shape[0])


    target = kwds.get("target", "normal")
    if "horiz" in target.lower():  # normalize to horizontal plane
        sunvec = kwds.get("sunvec")
        cosZen = np.cos(
            np.deg2rad(90 - cart2sph(sunvec[0], sunvec[1], sunvec[2])[1]))  # cos(zenith) = cos(90 - elevation)
    else:
        cosZen = 1.0  # sun is at "zenith" (perpendicular to surface ori)

    # calculate ambient spectra?
    if "ambient" in method.lower():
        ambient = kwds.get("ambient", estimate_ambient(X, cosInc))

    # calculate direct illumination mask
    i_mask = cosInc > 0.01 #no direct illumination for shaded pixels
    cosInc[ cosInc < 0.01 ] = 0.01 # backface culling

    #get shadow mask (and adjust illumination mask accordingly)
    # shadow correction
    shadow_mask = kwds.get("shadow_mask", np.full(X.shape[0], False))
    if shadow_mask is None: shadow_mask = np.full(X.shape[0], False)
    shadow_mask = shadow_mask.reshape(X.shape[0])
    i_mask = np.logical_and(i_mask, np.logical_not(shadow_mask))  # remove shadows from illumination mask

    # calculate correction factors
    m = np.zeros_like(X)
    c = np.zeros_like(X)
    if 'cfac' in method.lower():  # cfactor
        mask = np.isfinite(X).all(axis=1) & (X != 0).any(axis=1) & np.isfinite(cosInc) & i_mask
        assert mask.any(), "Error - all pixels are invalid. Check shadow mask and remove bands that are all nan or 0."
        #mask = mask & np.logical_not(shadow_mask) # also remove shadows
        intercept, slope = np.polynomial.polynomial.polyfit(cosInc[mask], X[mask, :], 1)
        cfac = intercept / slope
        m = (cosZen + cfac[None, :]) / (cosInc[:, None] + cfac[None, :])
    elif 'minnaert' in method.lower() and 'slope' in method.lower(): # minnaert with slope
        mask = np.isfinite(X).all(axis=1) & (X != 0).any(axis=1) & np.isfinite(cosInc) & i_mask
        assert "slope" in kwds, "Error - 'slope' must be specified for the minnaert slope correction."
        cosDip = np.cos( np.deg2rad(kwds.get("slope")) ).reshape(X.shape[0])
        intercept, k = np.polynomial.polynomial.polyfit(np.log(cosInc[mask] / cosZen),  # x
                                                        np.log(X[mask, :]), 1)  # y
        m += cosDip[:, None] * np.power((cosZen / (cosInc*cosDip)[:, None]), k[None, :])
    elif 'minnaert' in method.lower():  # minnaert
        mask = np.isfinite(X).all(axis=1) & (X != 0).any(axis=1) & np.isfinite(cosInc) & i_mask
        intercept, k = np.polynomial.polynomial.polyfit( np.log( cosInc[mask] / cosZen ), # x
                                                                       np.log(X[mask, :]), 1) # y
        m += np.power((cosZen / cosInc[:, None]),k[None, :])
    elif "icos" in method.lower():  # improved cosign correction
        # improved cosign
        cos_i_mean = np.nanmean(cosInc[i_mask])
        m += ((cos_i_mean - cosInc) / cos_i_mean)[:, None]
        c += X  # refl = refl + refl * fac
    elif "cos" in method.lower():  # Cosign correction
        m += (cosZen / cosInc)[:, None]
    elif "percent" in method.lower():  # percent
        m += (2 / (cosInc + 1))[:, None]
    elif "ambient" in method.lower():
        cosInc = cosInc.copy() #it's rude to unexpectedly modify user data...
        cosInc[ cosInc < 0.01 ] = 0 #no direct illumination for shaded pixels
        cosInc[ shadow_mask ] = 0 #and no direct illumination for shadow pixels
        m = 1/(cosInc[:,None]+ambient[None,:]) #calculate correction factor
    else:
        assert False, "Error - unknown correction method '%s'" % method

    # reshape datasets
    m = m.reshape(data.data.shape)
    c = c.reshape(data.data.shape)
    i_mask = i_mask.reshape(data.data.shape[:-1])

    # apply threshold
    thresh = kwds.get("thresh", 10.0)
    m[m > thresh] = thresh

    # apply correction
    if kwds.get('inplace', True):
        data.data = c + m * data.data

    # return
    return m, c, i_mask
