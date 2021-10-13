import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import matplotlib
import hylite
from hylite.correct.detrend import hull, polynomial
from hylite.hyfeature import HyFeature, MixedFeature
from hylite.multiprocessing import parallel_chunks

def minimum_wavelength(data, minw, maxw, method='gaussian', trend='hull', n=1, ftol=1e-2, order=3, threads=1, constraints=True, vb=True):
    """
    Perform minimum wavelength mapping to map the position of absorbtion features.

    *Arguments*:
     - data = the hyperspectral data (e.g. image or point cloud) to perform minimum wavelength mapping on.
     - minw = the lower limit of the range of wavelengths to search (in nanometers).
     - maxw = the upper limit of the range of wavelengths to search (in nanometers).
     - method = the method/model used to quantify the feature. Options are:
                 - "minmax" - performs a continuum removal and extracts feature depth as max - min. If n > 1 it detects local minima using scipy.signal.
                 - "gaussian" - fits one or more gaussians to the detrended spectra. This is the default.
                 - "lorentz" - fits one or more lorentzians equation to the detrended spectra.
                 - "tpt" - uses hylite.filter.tpt.TPT( ... ) to identify turning points in smoothed spectra and so identify absorbtions.
                           This only works for a single feature, but is quite fast and insensitive to asymetric features (but is limited
                           to the spectral resolution of the input data). Additionally, it can be run on data without doing detrending first.
     - trend = the method used to detrend the spectra. Can be 'poly' (fast) or 'hull' (slow) or None. Default is 'poly'.
     - n = the number of features to fit. Note that this is not compatible with method = 'tpt'.
     - ftol = the convergence tolerance to use during least squares fitting. Default is 1e-2.
     - order = the order of local minima detection. Default is 3. Smaller numbers return smaller local minima but are more
           sensitive to noise. For method = 'tpt' order gives the number of polynomial terms used for savgol filtering.
     - threads = the number of threads to use for the computation. Default is 1 (no multithreading).
     - constraints = Use constrained solver to constrain search bounds and avoid spurious results. Default is True (slower
                     but generally more accurate).
     - vb = True if graphical progress updates should be created.

    *Returns*:
     A HyImage (single-feature fitting) or list of HyImages (multi-feature fitting) with four bands each, feature
     position, depth, width and strength. Multi-features arrays will be returned in an arbitrary order, but can be
     sorted using sortMultiMWL( ... ).
    """

    # handle multi-threading
    if threads > 1:
        result = parallel_chunks( minimum_wavelength,
                                  data=data,
                                  minw=minw,
                                  maxw=maxw,
                                  method=method,
                                  trend=trend,
                                  n=n,
                                  ftol=ftol,
                                  order=order,
                                  threads=-1, # flag that this is being run multi-threaded
                                  constraints=constraints,
                                  vb=vb,
                                  nthreads=threads)
        if result.band_count() > 4: # special case - this is a stacked image returned by multifitting
            mwl = []
            if not (result.band_count() - 1) % 3 == 0:
                print("Warning: weird shit is happening?")
                return result
            for i in range(0,n):
                out = result.copy(data=False)
                out.header.drop_all_bands()
                out.data = result.data[..., [i,n+i,n*2+i,-1]]
                out.set_band_names(['pos', 'width', 'depth', 'strength'])
                out.push_to_header()
                mwl.append(out)
            return mwl
        else:
            return result

    # convert wavelengths to band ids
    minidx = data.get_band_index(minw)
    maxidx = data.get_band_index(maxw)

    # subset spectra from this range
    subset = np.array(data.data[..., minidx:maxidx]).copy()  # copy subset
    subset = subset.reshape(-1, subset.shape[-1])  # reshape to list of pixels/points

    wav = data.get_wavelengths()[minidx:maxidx]

    # get vector of valid spectra (non-nan and non-flat)
    mask = np.isfinite(subset).all(axis=1)  # drop nans
    mask = mask & (subset != subset[:, 0][:, None]).any(axis=1)  # drop flat spectra (e.g. all zeros)
    X = subset[mask]

    # special case - no valid data points! This happens fairly often with multiprocessing...
    if not mask.any():
        out = []
        for i in range(0, n): # make empty mwl maps of the right size/shape
            img = data.copy(data=False)
            img.header.drop_all_bands()
            img.data = np.full(data.data.shape[:-1] + (n*3 + 1,), np.nan)
            img.push_to_header()
            out.append(img)
        if n > 1 and threads != -1:
            return out # return multiple empty dataset
        else:
            return out[0] # return single dataset

    # detrend
    if not trend is None:
        if 'poly' in trend.lower():  # polynomial (vectorised == fast))
            X, _ = polynomial(X)
        elif 'hull' in trend.lower():  # convex hull (per pixel == slow)
            loop = range(X.shape[0])
            if vb: loop = tqdm(loop, desc="Hull correction", leave=False)
            for _n in loop:
                X[_n], _ = hull(X[_n, :])
        else:
            assert False, "Error: Unknown detrend method. Should be 'hull' or 'poly'."

    X = X / np.nanmax(X, axis=-1)[..., None]  # set max to 1.0
    # N.B. flat spectra/spectra with no feature should now be a horizontal line with value == 1 (plus noise). Features
    #     will thus appear as < 1.0.

    # remove any mystery nans... (hack!)
    if not np.isfinite(X).all():
        X[ np.logical_not( np.isfinite(X) ) ] = 1.0
        print( "Warning: data has mystery nans after hull correction. Beware!" )

    # fit features
    # special case - apply mwl using TPT
    if 'tpt' in method.lower():
        assert n == 1, "Error - TPT fitting only works for a single feature. Try using several MWL ranges in" \
                       "separate function calls to fit multiple features."

        # put filtered and detrended spectra in a HyData instance
        from hylite.filter import TPT, TPT2MWL
        D = hylite.HyData( X )

        D.set_wavelengths( wav )

        # apply TPT filter
        tpt, pos, depth = TPT( D, window=11, n=order, vb=vb )

        # convert to MWL map and reshape
        mwl = np.full((data.get_raveled().shape[0], n * 3 + 1), np.nan, dtype=np.float32)
        mwl[mask, :] = TPT2MWL( pos, depth, data=D, vb=vb).data
        mwl = mwl.reshape((*data.data.shape[:-1], mwl.shape[-1]))
    else:
        pos, width, depth, strength = HyFeature.fit( wav, X, method=method, n=n, ftol=ftol, order=order, vb=vb )

        # assemble and reshape
        mwl = np.full( (data.get_raveled().shape[0], n*3 + 1), np.nan, dtype=np.float32)
        mwl[mask,:] = np.vstack( [pos.T, width.T, depth.T, strength.T ] ).T
        mwl = mwl.reshape((*data.data.shape[:-1], mwl.shape[-1]))

    # convert to HyData instance and return :-)
    if mwl.shape[-1] > 4 and threads != -1: # split into multiple images
        out = []
        if not (mwl.shape[-1]-1) % 3 == 0:
            assert False, "Warning: weird shit is happening? MWL image has %d bands" % mwl.shape[-1]
        for i in range(0,n):
            img = data.copy(data=False)
            img.header.drop_all_bands()
            img.data = mwl[..., [i,n+i,n*2+i,-1]]
            img.set_band_names(['pos', 'width', 'depth', 'strength'])
            img.push_to_header()
            out.append(img)
        return out
    else: #return single image
        out = data.copy(data=False)
        out.header.drop_all_bands()
        out.data = mwl
        if mwl.shape[-1] == 4:
            out.set_band_names(['pos', 'width', 'depth', 'strength'])
        return out

class mwl_legend(object):
    """
    A utility class storing data needed to create a legend for mwl plots.
    """

    def __init__(self, minh, maxh, minc, maxc, mode='val', **kwds):
        """
        Create an mwl_legend instance.

        *Arguments*:
          - minh = the value (wavelength) mapped to hue of 0
          - maxh = the value (wavelength) mapped to a hue of 1
          - minc = the value (typically depth/strength) mapped to brightness/saturation of 0.
          - maxc = the value (typically depth/strength) mapped to brightness/saturation fo 1.
          - mode = specifies if minc and maxc refer to brightness ('val', default) or saturation ('sat').

        *Keywords*:
          - xlab = a custom name for the hue (x) label. Default is 'Wavelength (nm)'
          - ylab = a custom name for the second axis. Default is 'Strength'.
        """

        self.minh = minh
        self.maxh = maxh
        self.minc = minc
        self.maxc = maxc
        self.xlab = kwds.get("xlab", 'Wavelength (nm)')
        self.ylab = kwds.get("ylab", 'Strength')
        self.mode = mode

    def plot(self, ax, pos='top left', s=(0.3, 0.2)):
        """
        Add this legend to the specified figure.

        *Arguments*:
         - ax = the axes to overlay the legend on.
         - pos = the position of the legend. Can be:
                    - a string: 'top left' (default), 'bottom left', 'top right' or 'bottom right', or;
                    - a tuple (x,y) coordinates of the top left of the legend in figure coordinates.
         - s = the (width, height) of the legend in figure coordinates. Default is (0.2, 0.1).
        *Returns*:
         - ax = the axis that was added to the plot.
        """

        # create axis
        pad = 0.025
        tickLeft = True
        tickBottom = True
        origin = 'upper'
        if isinstance(pos, str):
            if 'top' in pos.lower():
                y0 = 1.0 - s[1] - pad
            elif 'bottom' in pos.lower() or 'lower' in pos.lower():
                y0 = pad
                tickBottom = False
                origin = 'lower'
            else:
                assert False, "Error - unknown position %s" % pos

            if 'left' in pos.lower():
                x0 = pad
                tickLeft = False
            elif 'right' in pos.lower():
                x0 = 1.0 - s[0] - pad
            elif 'middle' in pos.lower() or 'centre' in pos.lower():
                x0 = 0.5 - (s[0] * 0.5)
            else:
                assert False, "Error - unknown position %s" % pos
        else:
            x0, y0 = pos

        ax = ax.inset_axes([x0, y0, s[0], s[1]])
        if not tickLeft:
            ax.yaxis.tick_right()
            ax.yaxis.set_label_position("right")
        if not tickBottom:
            ax.xaxis.tick_top()
            ax.xaxis.set_label_position("top")

        # fill axes using imshow
        if 'swir' in str(self.minh): # again - this is dirty dirty hack... todo - write proper stop-based colour map
            extent =( 2150.0, 2380.0, self.minc, self.maxc)

            # calculate hue
            h = np.linspace(2150., 2380., num=int(2000 * s[0]))
            _x = np.array([0, 2150,
                           2175, 2190, 2220,  # white mica
                           2245, 2254, 2261,  # chlorite/biotite/epidote
                           2325, 2330, 2345,  # carbonate
                           2360, 3000])

            _y = np.array([0.1, 0.1,
                           0.15, 0.3, 0.44,  # white mica
                           0.48, 0.5, 0.54,  # chlorite/biotite/epidote
                           0.82, 0.85, 0.94,  # carbonate
                           1.0, 1.0])
            wav = np.linspace(2150.0, 2380.0, 255)
            lookup = np.interp(wav, _x, 1 - _y)
            idx = (((h - 2150.0) / (2380.0 - 2150.0)) * lookup.shape[0]).astype(np.int)
            idx[idx < 0] = 0
            idx[idx > 254] = 254
            x = lookup[idx]

        else:
            extent = (self.minh, self.maxh, self.minc, self.maxc)
            x = np.linspace(0, 1, num=int(2000 * s[0]))

        y = np.linspace(0, 1, num=int(2000 * s[1]))
        xx, yy = np.meshgrid(x, y)
        zz = np.full(xx.shape, 0.8)
        if 'val' in self.mode.lower():
            ax.imshow(matplotlib.colors.hsv_to_rgb(np.dstack([xx, zz, yy])),
                      origin=origin, aspect='auto', extent=extent)
        else:
            assert 'sat' in self.mode.lower(), "Error - %s is an invalid mode. Should be 'sat' or 'val'." % self.mode
            ax.imshow(matplotlib.colors.hsv_to_rgb(np.dstack([xx, yy, zz])),
                      origin=origin, aspect='auto', extent=extent)

        ax.set_xlabel(self.xlab)
        ax.set_ylabel(self.ylab)
        ax.set_yticks([])
        return ax

def colourise_mwl(mwl, strength=True, mode='p-d', **kwds):
    """
    Takes a HyData instance containing minimum wavelength bands (pos, depth, width and strength) and creates
    a RGB composite such that hue ~ pos, sat ~ width and val ~ strength or depth.

    *Arguments*:
     - mwl = the HyData instance containing minimum wavelength data.
     - strength = True if brightness should be proportional to strength. Otherwise brightness is proportional to depth
                    (sensitive to noise). Default is True.
     - mode = the mapping from position (p), width (w) and depth and (d) to hsv. Default is 'pwd', though other options
              are 'p-d' (constant saturation of 80%), 'pdw' and 'pd-' (constant brightness of 85%).
    *Keywords*:
      - hue_map = Wavelengths (xxx.x, yyy.y) or percentiles (x,y) to use when converting wavelength to hue.
                    Default is (0,100). Alternatively use 'SWIR' for customised colour stretch optimised for
                    clay, mica and chlorite absorbtion features.
      - sat_map = Widths (xxx.x, yyy.y) or percentiles (x,y) to use when converting width to saturation.
                    Default is (0,100).
      - val_map = Strengths/depths (xxx.x, yyy.y) or percentiles (x,y) to use when converting strength to brightness.
                    Default is (0,75), as these tend to be dominated by low/zero values.

    *Returns*:
     - either an RGB HyImage object (if mwl is an image) or the original HyCloud with defined rgb bands.
     - cmap = a mwl_legend instance for plotting colour maps.
    """

    # extract data
    assert mwl.band_count() == 4, "Error - HyData instance does not contain minimum wavelength data?"

    h = mwl.get_raveled()[..., 0].copy()  # pos
    s = mwl.get_raveled()[..., 1].copy()  # width
    v = mwl.get_raveled()[..., 2].copy()  # strength

    # multiply depth by strength to suppress deep minima from noisy spectra
    if strength and np.isfinite( mwl.get_raveled()[...,3]).any():
        v *= mwl.get_raveled()[...,3]
        v = np.sqrt(np.abs(v)) # as we basically squared the answer, take the sqrt to keep mapping similar to depth

    # normalise to range 0 - 1
    stretch = ['hue_map', 'sat_map', 'val_map']
    ranges = []  # store data ranges for colour map
    for i, b in enumerate([h, s, v]):
        if 'swir' in str(kwds.get(stretch[i], '')).lower():
            ranges.append((2150.0, 2380.0))
            continue

        mn, mx = kwds.get(stretch[i], (0, 100))
        if i == 2:  # use different default stretch for value (as these tend to be heavily skewed)
            mn, mx = kwds.get(stretch[i], (0, 75))
        if isinstance(mn, int):  # convert percentiles to values
            mn = np.nanpercentile(b, mn)
        if isinstance(mx, int):  # convert percentiles to values
            mx = np.nanpercentile(b, mx)

        ranges.append((mn, mx))  # store data ranges for colour map

        # apply stretch
        b -= mn
        b /= (mx - mn)
        b[b < 0] = 0
        b[b > 1] = 1

    # flip width (so low width = high saturation)
    s = 1 - s

    # special case - width is all the same
    if not np.isfinite(s).any():
        s = np.full(s.shape[0], 0.8)  # fill mostly saturated

    # remove nans
    mask = np.logical_not(np.isfinite(h) & np.isfinite(s) & np.isfinite(v))
    h[mask] = 0
    v[mask] = 0
    s[mask] = 0

    # map wavelength using custom map if specified
    # N.B. this is a filthy hack.... todo - write proper stop-based colour map sometime?
    if 'swir' in str(kwds.get('hue_map', '')).lower():
        _x = np.array([0, 2150,
                       2175, 2190, 2220,  # white mica
                       2245, 2254, 2261,  # chlorite/biotite/epidote
                       2325, 2330, 2345,  # carbonate
                       2360, 3000])

        _y = np.array([0.1, 0.1,
                       0.15, 0.3, 0.44,  # white mica
                       0.48, 0.5, 0.54,  # chlorite/biotite/epidote
                       0.82, 0.85, 0.94,  # carbonate
                       1.0, 1.0])
        wav = np.linspace(2150.0, 2380.0, 255)
        lookup = np.interp(wav, _x, 1 - _y)
        idx = (((h - 2150.0) / (2380.0 - 2150.0)) * lookup.shape[0]).astype(np.int)
        idx[idx < 0] = 0
        idx[idx > 254] = 254
        h = lookup[idx]

    # convert to rgb based on mapping mode
    if 'pwd' in mode.lower():  # pos, width, depth (default)
        rgb = matplotlib.colors.hsv_to_rgb(np.array([h, s, v]).T)
    elif 'pdw' in mode.lower():  # pos, depth, width
        s[mask] = 0.5
        rgb = matplotlib.colors.hsv_to_rgb(np.array([h, v, s]).T)
    elif 'p-d' in mode.lower():  # pos, const, depth
        rgb = matplotlib.colors.hsv_to_rgb(np.array([h, np.full(len(h), 0.8), v]).T)
    elif 'pd-' in mode.lower():  # pos, depth, const
        rgb = matplotlib.colors.hsv_to_rgb(np.array([h, v, np.full(len(h), 0.8)]).T)

    # create a colourbar object
    if 'pdw' in mode.lower() or 'pd-' in mode.lower():
        cbar = mwl_legend(ranges[0][0], ranges[0][1], ranges[2][0], ranges[2][1], mode='sat')
    else:
        cbar = mwl_legend(ranges[0][0], ranges[0][1], ranges[2][0], ranges[2][1], mode='val')
    if 'swir' in str(kwds.get('hue_map', '')).lower():
        cbar.minh = 'swir'
        cbar.maxh = 'swir'

    if mwl.is_image():
        out = mwl.copy(data=False)
        out.data = rgb.reshape((mwl.data.shape[0], mwl.data.shape[1], 3))
        out.set_band_names([0, 1, 2])
        out.set_wavelengths(None)
        return out, cbar
    else:  # HyCloud - set rgb and return
        # store
        out = mwl.copy(data=False)
        out.data = rgb
        out.set_band_names(["r", "g", "b"])

        # set rgb
        rgb[rgb <= 0.01] = 0.01  # avoid 0 as these are masked by plotting functions
        out.rgb = (rgb * 255).astype(np.uint8)
        return out, cbar

def plotmws(data, minw, maxw, trend='poly', step=2000, **kwds):
    """
    Plot detrended spectra over a small spectral range to assist with minimum wavelength mapping.

    *Arguments*:
     - data = the HyData instance to analyse.
     - minw = the lower limit of the range of wavelengths to search (in nanometers).
     - maxw = the upper limit of the range of wavelengths to search (in nanometers).
     - trend = the method used to detrend the spectra. Can be 'poly' (fast) or 'hull' (slow) or None. Default is 'poly'.
     - step = step used to skip pixels (rather than plotting everything). Default is 1000.
    *Keywords*:
     - keywords are passed to plt.plot(...)
    """

    # convert wavelengths to band ids
    minidx = data.get_band_index(minw)
    maxidx = data.get_band_index(maxw)

    # subset spectra from this range
    subset = np.array(data.data[..., minidx:maxidx])  # copy subset
    subset = subset.reshape(-1, subset.shape[-1])  # reshape to list of data points/pixels
    subset = subset[::step, :]  # throw out most of the pixels
    wav = data.get_wavelengths()[minidx:maxidx]

    # detrend
    if not trend is None:
        if 'poly' in trend.lower():  # polynomial (vectorised == fast))
            subset, _ = polynomial(subset)
        elif 'hull' in trend.lower():  # convex hull (per pixel == slow)
            for _n in tqdm(range(subset.shape[0]), desc="Hull correction", leave=False):
                subset[_n], _ = hull(subset[_n])
        else:
            assert False, "Error: Unknown detrend method. Should be 'hull' or 'poly'."
    else:  # trend is none, but still set max to 1.0
        with np.errstate(all='ignore'):
            subset = subset / np.nanmax(subset, axis=-1)[..., None]

    # do quick minmax interpolation to estimate initial values (can be vectorised == fast)
    m = wav[np.argmin(subset, axis=-1)]
    with np.errstate(all='ignore'):
        d = 1.0 - np.nanmin(subset, axis=-1)  # np.ptp(subset, axis=2)

    # calculate color based on minima position
    if "color" in kwds:
        calc_color = False
    else:
        calc_color = True
        cmap_name = kwds.get("cmap", "rainbow")
        if "cmap" in kwds:
            del kwds["cmap"]

        cmap = plt.get_cmap(cmap_name)
        cols = cmap((m - minw) / (maxw - minw))  # color from minima position

        # calculate brightness
        with np.errstate(invalid='ignore'):  # ignore nan errors
            b = (np.vstack([d, d, d, d] - np.nanmin(d)) / np.nanpercentile(d, 95)).T
            b[b > 1.0] = 1.0  # ensure valid range
            b[b < 0] = 0.0  # ensure valid range
            b[np.isnan(b)] = 0.0

            cols *= b  # apply brightness to colors

        cols[..., 3] = kwds.get('alpha', 0.1)
        if 'alpha' in kwds:
            del kwds['alpha']

    # plot
    fig, ax = plt.subplots(2, 2, figsize=(15, 10),
                           gridspec_kw={'width_ratios': [3, 1],
                                        'height_ratios': [3, 1]})

    ax[0, 0].set_title("Stacked spectra")

    kwds['alpha'] = kwds.get('alpha', 0.2)  # set default alpha
    for _n in range(subset.shape[0]):  # need to loop ... slow
        if np.isfinite(subset[_n]).all():  # skip nans

            # calculate color
            if calc_color:
                kwds['color'] = cols[_n]  # set color
            # plot
            ax[0, 0].plot(wav, subset[_n, :], **kwds)
    ax[0, 0].axhline(1.0, color='k')  # trend line

    mask = np.logical_and(np.isfinite(d), np.isfinite(m))

    ax[1, 0].set_title("Minimum wavelength")
    ax[1, 0].hist(m[mask].ravel(), bins=wav, color='k', alpha=0.75,
                  density=True, weights=d[mask].ravel(), histtype='step')
    n, bins, patches = ax[1, 0].hist(m[mask].ravel(), bins=wav, color='k', alpha=0.5,
                                     density=True, weights=d[mask].ravel())
    if calc_color:
        for i, p in enumerate(patches):
            p.set_facecolor(cmap(i / float(len(patches))))

    ax[0, 1].set_title("Feature depth")
    ax[0, 1].hist(d[mask].ravel(), bins=100, orientation='horizontal', color='k',
                  alpha=0.75, density=True,
                  range=(np.nanmin(d), np.nanpercentile(d, 95)),
                  histtype='step')
    n, bins, patches = ax[0, 1].hist(d[mask].ravel(), bins=100, orientation='horizontal', color='k',
                                     alpha=0.5, density=True,
                                     range=(np.nanmin(d), np.nanpercentile(d, 95)))
    ax[0, 1].invert_yaxis()
    if calc_color:
        for i, p in enumerate(patches):
            p.set_facecolor(np.array([1.0, 1.0, 1.0]) * (i / float(len(patches))))

    ax[1, 1].set_axis_off()
    fig.show()

    return fig, ax

def has_feature_between(mwl, wmin, wmax, depth_cutoff=0.05):
    """
    Calculate if a feature has been modelled between the specified wavelength ranges.

    *Arguments*:
     - mwl = a list of minimum wavelength maps as returned by minimum_wavelength(...) with multiple feature fitting.
     - wmin = the lower bound of the wavelength range.
     - wmax = the upper bound of the wavelength range.
     - depth_cutoff = the minimum depth required for a feature to count as existing.

    *Returns*:
     - a numpy array  populated with False (no feature) or True (feature).
    """

    return np.sum(
        [((m.data[..., 0] >= wmin) & (m.data[..., 0] <= wmax) & (m.data[..., 2] >= depth_cutoff)) for m in mwl],
        axis=0) > 0

def closestFeature(mwl, position, valid_range=None, depth_cutoff=0.05):
    """
    Returns the closest feature to the specified position from a list of minimum wavelength maps, as returned by
    multi-mwl.
    *Arguments*:
     - mwl = a list of minimum HyImage or HyCloud instances containing minimum wavelength data (pos,width,depth,strength).
     - position = the 'ideal' feature position to compare with (e.g. 2200.0 for AlOH)
     - valid_range = A tuple defining the minimum and maximum acceptible wavelengths, or None (default). Values outside
                     of the valid range will be set to nan.
     - depth_cutoff = Features with depths below this value will be discared (and set to nan).
    *Returns*
     - a single HyData instance containing the closest minima.

    """

    # get data arrays
    if mwl[0].is_image():
        pos = np.dstack([m.data[..., 0] for m in mwl])
        width = np.dstack([m.data[..., 1] for m in mwl])
        depth = np.dstack([m.data[..., 2] for m in mwl])
        strength = np.dstack([m.data[..., 3] for m in mwl])
    else:
        pos = np.vstack([m.data[..., 0] for m in mwl]).T
        width = np.vstack([m.data[..., 1] for m in mwl]).T
        depth = np.vstack([m.data[..., 2] for m in mwl]).T
        strength = np.vstack([m.data[..., 3] for m in mwl]).T

    # find differences
    diff = np.abs(pos - position)
    idx = np.argmin(diff, axis=-1)
    pos = np.take_along_axis(pos, idx[..., None], axis=-1)
    width = np.take_along_axis(width, idx[..., None], axis=-1)
    depth = np.take_along_axis(depth, idx[..., None], axis=-1)
    strength = np.take_along_axis(strength, idx[..., None], axis=-1)

    msk = depth < depth_cutoff
    if valid_range is not None:
        msk = np.logical_or(msk, np.logical_or(pos < valid_range[0], pos > valid_range[1]))
    pos[msk] = np.nan
    width[msk] = np.nan
    depth[msk] = np.nan
    strength[msk] = np.nan

    out = mwl[0].copy(data=False)

    if mwl[0].is_image():
        out.data = np.dstack([pos, width, depth, strength])
    else:
        out.data = np.hstack([pos, width, depth, strength])
    return out

def sortMultiMWL( mwl, mode='pos'):
    """
    Sort a list of minimum wavelength maps, as returned by minimum_wavelength using multi-feature fitting, based on feature
    position or depth.

    *Arguments*:
     - mwl = a list of minimum HyImage or HyCloud instances containing minimum wavelength data (pos,width,depth,strength).
     - mode = the mode to sort by. Options are 'pos', to sort in acending order by position, or 'depth', to sort in
              decending order by feature depth.

    *Returns*:
     - a list of sorted HyData instances.
    """

    # get data arrays
    if mwl[0].is_image():
        pos = np.dstack([m.data[..., 0] for m in mwl])
        width = np.dstack([m.data[..., 1] for m in mwl])
        depth = np.dstack([m.data[..., 2] for m in mwl])
        strength = np.dstack([m.data[..., 3] for m in mwl])
    else:
        pos = np.vstack([m.data[..., 0] for m in mwl]).T
        width = np.vstack([m.data[..., 1] for m in mwl]).T
        depth = np.vstack([m.data[..., 2] for m in mwl]).T
        strength = np.vstack([m.data[..., 3] for m in mwl]).T

    if 'pos' in mode.lower():
        idx = np.argsort(pos,axis=-1)
    elif 'width' in mode.lower():
        idx = np.argsort(width, axis=-1)
    elif 'depth' in mode.lower():
        idx = np.argsort(depth, axis=-1)[..., ::-1]
    else:
        assert False, "Error - sorting mode %s unrecognised. Use 'pos', 'width' or 'depth'." % mode

    # apply sorting
    pos = np.take_along_axis(pos, idx, axis=-1)
    width = np.take_along_axis(width, idx, axis=-1)
    depth = np.take_along_axis(depth, idx, axis=-1)
    strength = np.take_along_axis(strength, idx, axis=-1)

    # assemble into HyImage
    mwls = []
    for i in range(len(mwl)):
        _mwl = mwl[i].copy()
        if mwl[0].is_image():
            _mwl.data = np.dstack( [pos[...,i], width[...,i], depth[...,i], strength[...,i]] )
        else:
            _mwl.data = np.vstack( [pos[...,i], width[...,i], depth[...,i], strength[...,i]] ).T
        mwls.append( _mwl )
    return mwls

def getMixedFeature(idx, mwl, source=None):
    """
    Return a MixedFeature index corresponding to the specified point or pixel index. Useful for plotting fitted results
    vs actual spectra.

    *Arguments*:
     - idx = the index of the point or pixel to be retrieved.
     - mwl = a list of minimum wavelength datasets (as returned by multi-mwl fitting).
     - source = the source dataset, for plotting original spectra. Default is None.

    *Returns*: a MixedFeature instance containing the modelled minimum wavelength data at this point.
    """

    # extract fitted feature info
    if isinstance(idx, int):
        idx = (idx,)
    if not isinstance(idx, tuple):
        idx = tuple(idx)

    pos = [m.data[(*idx, 0)] for m in mwl]
    width = [m.data[(*idx, 1)] for m in mwl]
    depth = [m.data[(*idx, 2)] for m in mwl]

    if isinstance(idx, int):
        idx = (idx)
    if not source is None:  # we have source data
        wav = source.get_wavelengths()
        refl = source.data[idx]
        refl, corr = hull(refl)
        feats = [HyFeature('fit', p, w, d) for p, w, d in zip(pos, width, depth)]
        return MixedFeature('mix', feats, data=np.array([wav, refl]), color='r')
    else:  # we have no source data
        # domain = np.max(pos) - np.min(pos)
        # wav = np.linspace( np.min(pos) - domain*0.5, np.max(pos)+domain*0.5, 1000 )
        feats = [HyFeature('fit', p, w, d) for p, w, d in zip(pos, width, depth)]
        return MixedFeature('mix', feats, color='r')

