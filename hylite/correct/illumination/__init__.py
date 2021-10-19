import datetime
import numpy as np
import pytz
from scipy import stats
import datetime
import matplotlib.pyplot as plt
import numpy as np

#####################
## Utility functions
######################
from hylite.correct import Panel

from .occlusion import *
from .reflection import *
from .transmittance import *

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
        time = datetime.datetime.strptime(time[0], time[1])
        tz = pytz.timezone(tz)
        time = tz.localize(time)

    # time = time.astimezone(pytz.utc) #convert to UTC

    assert isinstance(time, datetime.datetime), "Error - time must be a datetime.datetime instance"
    assert not time.tzinfo is None, "Error - time zone must be specified (e.g. using tz='timezone')."

    # calculate illumination vector from time/position
    import astral.sun
    pos = astral.Observer(lat, lon, 0)
    azimuth = astral.sun.azimuth(pos, time)
    elevation = astral.sun.elevation(pos, time)

    sunvec = sph2cart(azimuth + 180, elevation)  # n.b. +180 flips direction from vector point at sun to vector
    # pointing away from the sun.

    return sunvec, azimuth, elevation

def estimate_skyview( normal ):
    """
    Return a crude estimate of sky-view factor by assuming geometry is an inclined plane (and that the sky is invisible
    below the horizon level. Useful for e.g. estimating the skyview factor for calibration panels based on their orientation.

    *Arguments*:
     - normal = the panel normal orientation as a (3,) numpy array.
    *Returns*:
     - the skyview factor (0 - 1).
    """
    return ( np.pi - np.arccos( np.dot( np.array([0,0,1]), normal ) ) ) / np.pi # sky view factor for inclined plane

def _regress(x, y, split=True, clip=(10, 90)):
    """ Loop through all bands in specified arrays and fit a line. Used for statistical adjustments (e.g. c-factor)"""

    assert x.shape == y.shape, "Error - x and y have incompatible shapes (%s != %s)." % (x.shape, y.shape)

    if not split:  # easy!
        x = x.ravel()
        y = y.ravel()

        # calculate percentile clip
        xmn, xmx = np.nanpercentile(x, clip)
        ymn, ymx = np.nanpercentile(y, clip)

        # calculate valid mask
        mask = np.isfinite(x) & np.isfinite(y) & (x > xmn) & (y > ymn) & (x < xmx) & (y < ymx)
        if not mask.any():  # no data to regress... return boring line.
            i = 0.0
            s = 0.0
            r = np.array([[np.nan]])
        else:
            (i, s), r = np.polynomial.polynomial.polyfit(x[mask], y[mask], 1, full=True)

            # plt.plot(x,y,alpha=0.2)
            # sx = np.nanmin(x[mask])
            # ex = np.nanmax(x[mask])
            # plt.plot( [sx,ex], [s*sx+i, s*ex+i])
            # plt.show()
            # assert False
        return s, i, np.sqrt((r[0][0] / np.sum(mask)))
    else:
        out = []

        # loop through bands
        for b in range(x.shape[1]):
            out.append(_regress(x[:, b], y[:, b], split=False))  # do regression

        # return output
        out = np.array(out)
        return out[:, 1], out[:, 0], out[:, 2]

##############################
## Define generic illumination model
###############################


class IlluModel(object):
    """
    Combine source, reflection, occlusion and transmittance information to simulate
    the illumination within a scene.

    Fundamentally, an illumination model represents one or all parts of the following
    equation:

    r = (aS + oI)R + P

    Where,
     - r = the measured pixel radiance.
     - a = the skyview factor associated with this pixel.
     - S = the downwelling skylight spectra (assumed to be constant across the scene).
     - o = the illumination factor for this pixel; will combine occlusion and e.g. lambert or oren-nayar reflection.
     - I = the incident sunlight spectra in the scene.
     - R = the material reflectance (0 - 1) of the pixel. Typically this is what we are trying to solve for.
     - P = path radiance effects that add or remove light due to interactions with air or particles between the
           sensor and the target.
    """

    def __init__(self, I, S=0, P=0, skv=0.6, rf=1.0, oc=0):
        """
        Create an illumination model. All arguments should be (...,1) shape arrays.

        *Arguments*:
         - I = the incident sunlight spectra. This must be provided.
         - P = the path radiance spectra. Defaults to 0 if unknown (i.e. ignore path radiance). If this
               is a 1-D array then it is assumed to be constant across the scene. If it is an (...,1) shaped
               array then it will be applied separately per-pixel or per-point, to account for e.g. variations
               in distance to target.
         - S = the downwelling skylight spectra. Defaults to 0 if unknown (i.e. ignore skylight).
         - skv = skyview factors. Will be ignored unless a skylight spectra is defined. Default is 0.6
                     (if a skylight spectra is defined but skyview factors are unknown).
         - rf = reflectance factors as determined by e.g. a lambertian or oren-nayar reflectance model. Default
                is a perfect reflection (1.0).
         - o = occlusion factors that reduce the amount of incident light recieved across the scene due to e.g.
               shadows. Reflection factors specified by the rf argument are multiplied by (1 - oc). Default is 0
               (no occlusion).

        *Returns*:
         - an IlluModel instance.
        """
        # store data
        self.I = I  # n.b. this copies arrays (for safety), and also wraps constants in a numpy object.
        self.P = P
        self.S = S
        self.skv = skv
        self.rf = rf
        self.oc = oc

        # initialise fitted attributes
        self.rboost = np.array(0)  # radiance boost [ c-factor adjustment in y ]
        self.iboost = np.array(0)  # illumination boost [ c-factor adjustment in x ]

        # and output attributes
        self.r = None
        self.R = None

        # make all attributes numpy arrays
        for attr in ['I', 'P', 'S', 'skv', 'rf', 'oc']:
            val = getattr(self, attr)
            if isinstance(val, hylite.HyData):  # hydata instance
                setattr(self, attr, np.array(val.data).squeeze().astype(np.float32))  # copy out data array
            else:
                setattr(self, attr, np.array(val).squeeze().astype(np.float32))

    def getRadiance(self, R=1.0):
        """
        Evaluate the measured radiance for a known reflectance, based on the equation:

        r = (aS + oI)R + P

        *Arguments*:
         - R = the reflectance(s) to evaluate. If a scalar value is passed (must be between 0 and 1) then
               a single combined illumination spectra will be returned. If a (...,1) array of values are
               provided then a (...,n) array containing the illumination spectra at each point or pixel will
               be returned.
        *Returns*:
         - an array containing the illumination spectra in its last axis.
        """
        if isinstance(R, hylite.HyData):
            R = R.data.squeeze()
        else:
            R = np.array(R).squeeze()
        return (self.skv[..., None] * self.S + ((1 - self.oc) * self.rf)[..., None]) * R[..., None] * self.I + self.P

    def getReflectance(self, r):
        """
        Evaluate the reflectance from measured radiance, based on the equation:

        R = (r - P) / (aS + oI)

        *Arguments*:
         - r = a (..., n) array containing the measured radiance values.
        *Returns*:
         - an array containing the reflectance spectra in its last axis.
        """
        if isinstance(r, hylite.HyData):
            self.r = r.data.squeeze()
        else:
            self.r = np.array(r).squeeze()

        self.R = np.clip((self.r - self.P + self.rboost) / (
                    self.skv[..., None] * self.S + ((1 - self.oc) * self.rf)[..., None] * self.I + self.iboost), 0, 1)
        return self.R

    # def fit(self, radiance, shift='x'):
    #     """
    #     Compute c-factor offsets. This assumes a linear relationship between illumination and measured radiance, and
    #     increases/decreases either (1) the illumination component [y-shift] or (2) the measured radiance [x-shift] to
    #     force the regression line to pass through (0,0). Reflectance outliers are then detected and can be masked or
    #     corrected.
    #
    #     *Arguments*:
    #      - radiance = radiance data to fit to. Must have the same shape as self.data.
    #      - shift = apply the correction in the y-direction (adjust measured radiance to simulate the influence of path
    #                radiance) or in the x-direction (adjust modelled illumination to account for unknown light source). Default
    #                is 'x' (this is the typical c-factor correction).
    #     """
    #     #assert self.data is not None, "Error - please compute reflectance model using self.compute(...) before fitting step."
    #     if self.data is None:
    #         self.evaluate() # evaluate if need be
    #     self.r = radiance # store radiance
    #
    #     # get data
    #     x = self.data.X()
    #     y = radiance.X()
    #
    #     # remove any mischevous negative radiances... (these are noise)
    #     x[x < 0] = np.nan
    #
    #     # compute slope and intercepts of regression
    #     s, i, r = _regress(x, y)  # fit linear regressions
    #     if 'x' in shift: # typical c-factor adjustment (add/subtract illumination)
    #         self.cfac = np.array(i) / np.array(s)  # calculate cfacor (x-intercept)
    #         mn = np.nanmin(x, axis=0)  # calculate floor cfactor to avoid negative values
    #         self.cfac[self.cfac < mn] = mn[self.cfac < mn]  # apply floor and store
    #     elif 'y' in shift: # alternative adjustment ("r-factor"); add/subtract irradiance.
    #         self.rfac = -np.array(s)  # calculate alternative c-factor (y-intercept) and store
    #     else:
    #         assert False, 'Error - %s should be either "x" or "y"' % shift
    #
    # def plot_fit(self, bands=None, n=100, nb=5, **kwds):
    #     """
    #     Plot the relationship between illumination and measured radiance.
    #
    #     *Arguments*:
    #      - radiance = the radiance data (HyImage or HyCloud) to compare too. Shape must match internal self.data array.
    #      - bands = the band (integer or float), band range (tuple) or bands (list) to include on the regression plot. Default
    #                is None (use all bands).
    #      - n = plot every nth point (only) to speed up plotting. Default is 100. This value does not affect the regressions.
    #      - nb = only calculate / plot every nb'th band if bands is a (min,max) tuple. Default is 5.
    #     *Keywords*:
    #      - keywords are passed to plt.scatter(...).
    #     """
    #
    #     assert self.data is not None, "Error - please compute reflectance model using self.compute(...) before plotting."
    #     assert self.r is not None, "Error - please fit reflectance model using self.fit(...) before plotting."
    #
    #     # get relevant bands
    #     if bands is None:
    #         bands = (0, -1)
    #     if isinstance(bands, float) or isinstance(bands, int):
    #         idx = self.r.get_band_index(bands)
    #         x = self.data.X()[:, idx][:, None]
    #         y = self.r.X()[:, idx][:, None]
    #         w = self.r.get_wavelengths()[idx]
    #     elif isinstance(bands, tuple) and len(bands) == 2:
    #         mn,mx = [self.r.get_band_index(b) for b in bands]
    #         idx = np.array(range(self.r.band_count()))[mn:mx]  # band indices
    #         x = self.data.X()[:, idx][:, ::nb]
    #         y = self.r.X()[:, idx][:, ::nb]
    #         w = self.r.get_wavelengths()[idx][::nb]
    #     elif isinstance(bands, list) or isinstance(bands, tuple):
    #         idx = np.array([self.r.get_band_index(b) for b in bands])
    #         nb = 1
    #         x = self.data.X()[:, idx][::nb]
    #         y = self.r.X()[:, idx][::nb]
    #         w = self.r.get_wavelengths()[idx][::nb]
    #     else:
    #         assert False, "Error - %s is an unknown band type. Should be int, float, list or tuple." % type(bands)
    #
    #     # remove any mischevous negative radiances... (these are noise)
    #     x[x < 0] = np.nan
    #
    #     # build plot
    #     if self.cfac is None and self.rfac is None:  # no adjustment applied
    #         fig, ax = plt.subplots(2, 1, figsize=(10, 10))  # only two axes needed if no shift applied
    #         ax = [ax[0], ax[0], ax[1]]
    #     else:
    #         fig, ax = plt.subplots(3, 1, figsize=(10, 10))
    #
    #     ###################
    #     # (a) plot points
    #     ###################
    #     cmap = plt.get_cmap('rainbow')
    #     kwds['s'] = kwds.get('s', 3.0)
    #     kwds['alpha'] = kwds.get('alpha', 0.1)
    #     for b in range(x.shape[-1]):
    #         kwds['color'] = cmap((w[b] - np.min(w)) / np.ptp(w))
    #         ax[0].scatter(x[::n, b], y[::n, b], **kwds)
    #     ax[0].set_xlabel("Modelled radiance")
    #     ax[0].set_ylabel("Measured radiance")
    #     ax[0].set_title("a. Radiance")
    #
    #     ###################
    #     # (b) plot c-factor
    #     ###################
    #     if self.cfac is not None:
    #         ax[1].plot(w, self.cfac[idx][::nb])
    #         ax[1].axhline(0, color='k', lw=2)
    #         ax[1].set_ylabel("Illumination boost")
    #         ax[1].set_xlabel("Wavelength (nm)")
    #         ax[1].set_title("b. Adjustment (c-factor adjustment)")
    #
    #         # apply adjustment for plotting
    #         x += self.cfac[idx][::nb]
    #
    #     elif self.rfac is not None:  # n.b. cfac and rfac should never both be set
    #         ax[1].plot(w, self.rfac[idx][::nb])
    #         ax[1].axhline(0, color='k', lw=2)
    #         ax[1].set_ylabel("Irradiance boost")
    #         ax[1].set_xlabel("Wavelength (nm)")
    #         ax[1].set_title("b. Adjustment (alternate c-factor adjustment)")
    #
    #         # apply adjustment for plotting
    #         y += self.rfac[idx][::nb]
    #
    #     #############################################
    #     # (b) plot reflectance vs at target radiance
    #     #############################################
    #     y = y / x  # convert to reflectance
    #
    #     cmap = plt.get_cmap('rainbow')
    #     kwds['s'] = kwds.get('s', 3.0)
    #     kwds['alpha'] = kwds.get('alpha', 0.1)
    #     for b in range(x.shape[-1]):
    #         kwds['color'] = cmap((w[b] - np.min(w)) / np.ptp(w))
    #         ax[2].scatter(x[::n, b], y[::n, b], **kwds)
    #
    #     p5, p10, p50, p90, p95 = np.nanpercentile(y, (5, 10, 50, 90, 95))
    #     ax[2].axhline(p10, color='k', ls='--')
    #     ax[2].axhline(p90, color='k', ls='--')
    #     ax[2].axhline(p50, color='k', lw=2)
    #
    #     ax[2].set_ylim(p5, p95)
    #     ax[2].set_ylabel("Reflectance")
    #     ax[2].set_xlabel("Modelled Radiance")
    #     ax[2].set_title("c. Reflectance")
    #
    #     # add a wavelength colorbar
    #     sc = ax[0].scatter([np.nan, np.nan], [np.nan, np.nan], c=[np.min(w), np.max(w)],
    #                        vmin=np.min(w), vmax=np.max(w), cmap='rainbow')  # build colormap
    #     cbar = fig.colorbar(sc, orientation='horizontal', shrink=0.5, pad=0.25)  # plot
    #     cbar.set_label('Wavelength (nm)')
    #
    #     fig.tight_layout()
    #     return fig, ax


class ELC(object):
    """
    Class that gathers one or more Panels and computes calculates an empirical line correction. This does not
    adequately describe or correct for scene illumination, but can be useful as a quick correction step.
    """

    def __init__(self, panels):

        """
        Constructor that takes a list of Panel objects (one for each target used for the correction) and computes
        an empirical line correction.

        *Arguments*:
          - panels = a list of Panel objects defining the reflectance and radiance of each panel in the scene.
        """

        if not isinstance(panels, list):
            panels = [panels]

        self.wav = np.array(panels[0].get_wavelengths())
        for p in panels:
            assert isinstance(p, Panel), "Error - ELC panels must be instances of hylite.correct.Panel"
            assert (self.wav == np.array(
                p.get_wavelengths())).all(), 'Error - ELC panels must cover the same wavelengths'

        # compute ELC
        self.slope = np.zeros(self.wav.shape)
        self.intercept = np.zeros(self.wav.shape)
        if len(panels) == 1:  # only one panel - assume intercept = 0
            self.slope = panels[0].get_mean_radiance() / panels[0].get_reflectance()
        else:
            # calculate regression for each band
            for b, w in enumerate(self.wav):
                _y = np.array([p.get_mean_radiance()[b] for p in panels])
                _x = np.array([p.get_reflectance()[b] for p in panels])
                self.slope[b], self.intercept[b], _, _, _ = stats.linregress(_x, _y)

    def get_wavelengths(self):
        """
        Get the wavelengths for which this ELC has been calculated.
        """
        return self.wav

    def get_bad_bands(self, **kwds):

        """
        Find bands in which signal-noise ratios are amplified above a threshold (due to large correction slope).

        *Keywords*:
         - thresh = the threshold slope. Defaults to the 85th percentile.

        *Returns*:
         - a boolean numpy array containing True for bad bands and False otherwise.
        """

        thresh = kwds.get("thresh", np.nanpercentile(self.slope, 85))
        return self.slope > thresh

    def apply(self, data, **kwds):

        """
        Apply this empirical line calibration to the specified image.

        *Arguments*:
         - data = a HyData instance to correct

        *Keywords*:
         - thresh = the threshold slope. Defaults to the 90th percentile.

        *Returns*:
         - a mask containing true where the corrected values are considered reasonable - see get_bad_bands(...) for more
           details. Note that this returns the np.logical_not( self.get_bad_bands(...) ).
        """

        assert data.band_count() == len(self.slope), "Error - data has %d bands but ELC has %d" % (
        data.band_count(), len(self.slope))
        data.data -= self.intercept # subtract path radiance
        data.data /= self.slope # remove illumination source

        return np.logical_not(self.get_bad_bands(**kwds))

    def quick_plot(self, ax=None, **kwds):

        """
        Plots the correction factors (slope and intercept) computed for this ELC.

        *Arguments*:
         - ax = the axes to plot on. If None (default) then a new axes is created.
        *Keywords*:
         - thresh = the threshold to separate good vs bad correction values (see get_bad_bands(...)). Default is the
                    85th percentile of slope values.
        *Returns*:
         -fig, ax = the figure and axes objects containing the plot.

        """

        if ax is None:
            fig, ax = plt.subplots(figsize=(15, 10))

        # plot slope
        _x = self.get_wavelengths()
        _y1 = self.slope
        _y2 = [kwds.get("thresh", np.nanpercentile(self.slope, 85))] * len(_y1)
        ax.plot(_x, _y1, color='k', lw=1)
        ax.plot(_x, _y2, color='gray', lw=2)
        ax.fill_between(_x, _y1, [0] * len(_x), where=_y1 > _y2, facecolor='red', interpolate=True, alpha=0.3)
        ax.fill_between(_x, _y1, [0] * len(_x), where=_y1 < _y2, facecolor='green', interpolate=True, alpha=0.3)
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("ELC slope")
        if not (self.intercept == 0).all():
            ax2 = ax.twinx()
            ax2.plot(_x, self.intercept, color='b')
            ax2.set_ylabel("ELC intercept")
            ax2.spines['right'].set_color('blue')
            ax2.yaxis.label.set_color('blue')
            ax2.tick_params(axis='y', colors='blue')

        return ax.get_figure(), ax