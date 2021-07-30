import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from scipy import ndimage, signal

import hylite
from hylite import HyHeader
import hylite.reference.features as ref
from hylite.hyfeature import HyFeature, MultiFeature, MixedFeature
from matplotlib.ticker import AutoMinorLocator

class HyData(object):

    """
    A generic class for encapsulating hyperspectral (points and images), and associated metadata such as georeferencing, bands etc.

    This class is based around a data numpy array containing the hyperspectral data in a representation such that the last dimension
    corresponds to individual bands (e.g. data[pointID, band] or data[px,py,band]). Note that the data array can be empty!
    """

    @classmethod
    def to_grey(cls, data):
        """
        Return a copy of the specified data array as uint8 greyscale. Useful for OpenCV operations, compression or mapping to RGB.
        """

        return np.uint8(255 * (data - np.nanmin(data)) / (np.nanmax(data) - np.nanmin(data)))

    #####################################
    ##  Instance methods
    #####################################
    def __init__(self, data, **kwds):
        """
        Create an image object from a data array.

        *Arguments*:
         - data = a numpy array such that the last dimension
                corresponds to individual bands (e.g. data[pointID, band] or data[px,py,band])

        *Keywords*:
         - header = associated header file. Default is None (create a new header).
        """

        #copy reference to data. Note that this can be None!
        self.data = data
        if not data is None:
            self.dtype = data.dtype
        else:
            self.dtype = None

        # header data
        self.set_header(kwds.get('header', None))

    def __getitem__(self, key):
        """
        Expose underlying data array when using [ ] operators
        """
        return self.data.__getitem__(key)

    def __setitem__(self, key, value):
        """
        Expose underlying data array when using [ ] operators
        """
        self.data.__setitem__(key, value)

    def copy(self, data=True):
        """
        Make a deep copy of this image instance.
        *Arguments*:
         - data = True if a copy of the data should be made, otherwise only copy header.
        *Returns*
          - a new HyData instance.
        """

        if not data or self.data is None:
            return HyData( None, header=self.header.copy())
        else:
            return HyData( self.data.copy(), header=self.header.copy())

    def set_header(self, header):
        """
        Loads associated header data into self.header.

        Arguments:
         - header = a HyHeader object or None.
        """

        #no header - create one
        if header is None:
            self.header = HyHeader()
            return

        #set the header
        self.header = header

    def push_to_header(self):
        """
        Update header data to match this hyperspectral data.
        """
        self.header['samples'] = str(self.samples())
        self.header['bands'] = str(self.band_count())
        self.header['lines'] = str(self.lines())


    #############################################
    ##Expose important parts of the header file
    #############################################
    def has_wavelengths(self):
        """
        True if the header data contains wavelength information.
        """
        return self.header.has_wavelengths()

    def get_wavelengths(self):
        """
        Get the wavelength that corresponds with each band of this image, as stored in the .hdr file.
        """
        return self.header.get_wavelengths()

    def has_band_names(self):
        """
        True if the header data contains band names.
        """
        return self.header.has_band_names()

    def get_band_names(self):
        """
        Return band names as defined in the header file.
        """
        return self.header.get_band_names()

    def has_fwhm(self):
        """
        True if the header data contains band names.
        """
        return self.header.has_fwhm()

    def get_fwhm(self):
        """
        Return band names as defined in the header file.
        """
        return self.header.get_fwhm()

    def set_wavelengths(self, wavelengths):
        """
        Set the wavelengths associated with this hyperspectral data
        """
        if not wavelengths is None:
            assert len(wavelengths) == self.band_count(), "Error - wavelengths must be specified for each band."
        self.header.set_wavelengths(wavelengths)

    def set_band_names(self, names ):
        """
        Set the band names associated with this hyperspectral data
        """
        if not names is None:
            assert len(names) == self.band_count(), "Error - band names must be specified for each band."
        self.header.set_band_names( names )

    def set_fwhm(self, fwhm):
        """
        Set the band widths associated with this hyperspectral data
        """
        if not fwhm is None:
            assert len(fwhm) == self.band_count(), "Error - wavelengths must be specified for each band."
        self.header.set_fwhm(fwhm)

    def is_image(self):
        """
        Return true if this dataset is an image (i.e. data array has dimension [x,y,b]).
        """

        return len(self.data.shape) == 3

    def is_point(self):
        """
        Return true if this dataset is an point cloud or related dataset (i.e. data array has dimension [idx,b]). Note
        that this will return true for spectral libraries and other 'cloud like' datasets.
        """

        return len(self.data.shape) == 2

    def is_classification(self):
        return 'classification' in self.header['file type'].lower()

    ###################################
    ## Data dimensions and properties
    ###################################
    def band_count(self):
        """
        Return the number of bands in this dataset.
        """
        if self.data is None: return 0
        return self.data.shape[-1]

    def samples(self):
        """
        Return number of samples in this dataset. For 2D data (images) this is the image height (number of pixels in a line
        scanner). For 1D data (point clouds) this is 1 (each point is like an individual sample).
        """
        if self.data is None:
            return 0
        return self.data.shape[0]

    def lines(self):
        """
        Return number of lines in this dataset. For 2D data (images) this is the image height. For 1D data (point clouds)
        this is 1.
        """

        if self.data is None:
            return 0
        if len(self.data.shape) > 2:
            return self.data.shape[1]
        else:
            return 1

    def is_int(self):
        """
        Return true if this dataset contains data with integer precision.
        """
        return np.issubdtype( self.data.dtype, np.integer )

    def is_float(self):
        """
        Return true if this dataset contains data with floating point precision.
        """
        return np.issubdtype( self.data.dtype, np.floating )

    #############################
    ## masking and band removal
    #############################
    def export_bands(self, bands ):
        """
        Export a specified band range to a new HyData instance.

        *Arguments*:
         - bands = either:
                     (1) a tuple containing the (min,max) wavelength to extract. If range is a tuple, -1 can be used to specify the
                         first or last band index.
                     (2) a list of bands or boolean mask such that image.data[:,:,range] is exported to the new image.
        """

        # wrap individual integers or floats in a list
        if isinstance(bands, int) or isinstance(bands, float):
            bands = [bands]  # wrap in list

        # calculate bands to remove
        mask = np.full( self.band_count(), True )
        if isinstance(bands, np.ndarray) and bands.dtype == np.bool:
            # mask is a numpy array containing True for bands to keep, so flip it to get bands to remove.
            mask = np.logical_not(bands)
        elif isinstance(bands, tuple) and len(bands) == 2:
            # get indices of bands to keep and flag these as False in mask.
            if bands[0] == -1: bands = (0, bands[1])
            if bands[1] == -1: bands = (bands[0], self.band_count())
            mn = self.get_band_index(bands[0])
            mx = self.get_band_index(bands[1])

            # calculate mask
            mask[ mn:mx ] = False
        else:
            # check bands are all indices and flag these bands as False in mask.
            bands = list(bands)
            for i, b in enumerate(bands):
                if isinstance(b,float) or isinstance(b,str):
                    bands[i] = self.get_band_index(b)

            mask[bands] = False

        # check that we're leaving at least one band....
        assert (mask==False).any(), "Error - cannot export image with no bands."

        # quick exit if we are exporting all bands
        if (mask==False).all(): # no bands are masked
            return self.copy()

        # copy this dataset
        subset = self.copy()
        subset.header.drop_bands(mask) # apply mask to header data
        subset.data = np.zeros( self.data[..., np.logical_not(mask) ].shape, dtype=self.data.dtype )
        subset.data[...] = self.data[..., np.logical_not(mask)]

        # special case for spectral libraries
        if hasattr(self, 'upper'):
            if self.upper is not None:
                subset.upper = self.upper[..., np.logical_not(mask)]
        if hasattr(self, 'lower'):
            if self.lower is not None:
                subset.lower = self.lower[..., np.logical_not(mask)]

        return subset

    def resample(self, wavelengths):
        """
        Returns a new image resampled to the specified list of wavelengths. Note that this
        simply uses a nearest neighbour resampling, so chooses the closest band matching each
        wavelength. No averaging will be performed using this method - for more advanced resampling see
        hylite.filter.sample.

        Also note that to avoid confusion, the original wavelengths will be preserved rather than
        overwritten. Also note that bands will not be duplicated, so the number of bands returned MAY NOT equal
        the number of wavelengths provided!

        *Arguments*:
         - wavelengths = the wavelengths (list of floats) to resample to. MUST be in ascending order.
        *Returns*:
         - a resampled HyData instance
        """

        out = []
        for w in wavelengths:
            out.append( self.get_band_index(w) )
        out = self.export_bands(out)
        return out

    def delete_nan_bands(self, inplace=True):
        """
        Remove bands in this image that contain only nans.

        *Arguments*:
         - inplace = True if this operation should be applied to the data in situ. Default is True.

        *Returns*:
         - an image copy with the nan bands removed IF inplaces is False. Otherwise the image is modified inplace.
        """

        if len(self.data.shape) == 3: #hyImage
            cpy = self.export_bands(np.isfinite(self.data).any(axis=(0, 1)))  # remove bands that are all nan
        else: #hycloud
            assert len(self.data.shape) == 2, "Weird error?"
            cpy = self.export_bands(np.isfinite(self.data).any(axis=0))  # remove bands that are all nan
        if inplace:
            self.header = cpy.header
            self.data = cpy.data
        else:
            return cpy

    def set_as_nan(self, value):
        """
        Sets data with the specified value to NaN. Useful for handling no-data values.

        *Arguments*:
         - value = the value to (permanently) replace with np.nan.
        """

        if self.is_int():
            nan = int(self.header.get("data ignore value", 0))
            self.data[ self.data == value ] = nan
            self.header["data ignore value"] = str(nan)
        else:
            self.data[self.data == value] = np.nan

    def mask_bands(self, mn, mx=None, val=np.nan):
        """
        Masks a specified range of bands, useful for removing water features etc.

        *Arguments*:
         - min = the start of the band mask (as per get_band_index(...)).
         - max = the end of the band mask (as per get_band_index( ... )). Can be None to mask individual bands. Default is
                 None.
         - val = the value to set masked bands to. Default is np.nan. Set to None to keep values but flag bands in band band list.
        """

        if mx is None:
            self.data[..., self.get_band_index(mn)] = np.nan
            return
        elif mx == -1:
            mx = self.band_count()

        # update bad band list
        bbl = np.full( self.band_count(), True )
        if self.header.has_bbl():
            bbl = self.header.get_bbl()
        bbl[..., self.get_band_index(mn): self.get_band_index(mx)] = False
        self.header.set_bbl( bbl )

        if not val is None:
            self.data[..., self.get_band_index(mn): self.get_band_index(mx)] = val

    def mask_water_features(self, **kwds):
        """
        Removes typical water features. By default this removes bands between:
          - 960 - 990 nm
          - 1320 - 1500 nm
          - 1780 - 2050 nm
          - 2400 - 2500 nm

        Custom wavelengths can be set using the mask keyword.

        *Keywords*:
         - mask = mask custom bands. This should be a list of tuple band indices or wavelengths containing the
           minimum and maximum wavelenght/index of each region to mask.
        """

        default = [(960.0, 990.0), (1320.0, 1500.0), (1780.0, 2050.0), (2400.0, 2502.0)]
        bands = kwds.get("mask", default)

        # mask bands
        for mn, mx in bands:
            try:
                self.mask_bands(mn, mx)
            except:
                pass  # ignore errors associated with out of range etc.

        # delete/export

    #########################
    ## band getters/setters
    #########################
    def get_band(self, b):
        """
        Gets an individual band from this dataset. If an integer is passed it is treated as a band index. If a string is passed it is
        treated as a band name. If a float is passed then the closest band to this wavelength is retrieved.

        *Arguments*:
         - b = the band to get. Integers are treated as indices, strings as band names and floats as wavelengths.
        *Returns*:
         - a sliced np.array exposing the band. Note that this is NOT a copy.
        """

        return self.data[...,self.get_band_index(b)]

    def get_band_grey(self, b):
        """
        Returns the specified band as a uint8 greyscale image compatable with opencv.
        """

        return HyData.to_grey( self.get_band(b) )

    def get_raveled(self):
        """
        Get the data array as a 2D array of points/pixels. NOTE: this is just a view of the original data array, so any
        operations changes made to it will affect the original image. Useful for fast transformations!

        *Returns*
         - pixels = a list such that pixel[n][band] gives the spectra of the nth pixel.
        """

        return self.data.reshape(-1, self.data.shape[-1])

    def X(self):
        """
        A shorthand way of writing get_raveled(), as X is conventionally used for a vector of spectra.
        """
        return self.get_raveled()

    def set_raveled(self, pix, shape=None):
        """
        Fills the image/dataset from a list of pixels of the format returned by get_pixel_list(...). Note that this does not
        copy the list, but simply stores a view of it in this image.

        *Arguments*:
         - pix = a list such that pixel[n][band] gives the spectra of the nth pixel.
         - shape = the reshaped data dimensions. Defaults to the shape of the current dataset, except with auto-shape for the last dimension.
        """
        if shape is None:
            shape = list( self.data.shape )
            shape[-1] = -1
        self.data = pix.reshape(shape)

    def get_band_index(self, w, **kwds):
        """
        Get the band index that corresponds with the given wavelength or band name.

        *Arguments*:
         - w = the wavelength, index or band name search for. Note that if w is an integer it is treated
               as a band index and simply returned. If it is a string then the index of the matching band name
               is returned. If it is a wavelength then the closest band to this wavelength is returned.
        *Keywords*:
         - thresh = the threshold (in nanometers) within which a band must fall to be valid. Default is
                    hylite.band_select_threshold (which defaults to 10 nm). If a wavelength is passed and a
                    band exists within this distance, then it is returned. Otherwise an error is thrown).
        *Returns*:
         - the matching band index.
        """

        thresh = kwds.get("thresh", hylite.band_select_threshold)

        if isinstance(w, int):  # already a valid band index
            assert -self.band_count() <= w <= self.band_count(), "Error - band index %d is out of range (image has %d bands)." % (w, self.band_count())
            if w < 0: #convert negative indices to positive ones
                return self.band_count() + w
            else:
                return w
        elif isinstance(w, str): # treat w as band name
            assert w in self.get_band_names(), "Error - could not find band with name %s" % w
            return int(self.get_band_names().index(w))
        elif isinstance(w, float): # otherwise treat w as wavelength
            wavelengths = self.get_wavelengths()
            diff = np.abs( wavelengths - w)
            assert np.nanmin(diff) <= thresh, "Error - no bands exist within %d nm of wavelength %f. Try increasing the 'thresh' keyword?" % (thresh, w)
            return int(np.argmin(diff))
        else:
            assert False, "Error - %s is an unknown band descriptor type." % type(w)

    def contiguous_chunks(self, p=75, min_size=0):
        """
        Extract contiguous chunks of spectra, splitting a (1) completely nan bands or (2) large steps in wavelength.

        *Arguments*:
         - p = the percentile used to define a large change in wavelength. Default is 90. A "gap" is considered to be
               a change in wavelength greater than double this percentile.
         - min_size = the minimum number of bands required to consider a chunk valid. Default is 0 (return all chunks).
        *Returns*:
         - chunks = copies of the orignal data array that contain continuous spectra. At least one pixel/point
                    in each slice of these bans is guaranteed to be finite.
         - wav = array containing the wavelengths corresponding to each band of each chunk.
        """

        # find gaps in wavelength and/or completely nan bands and/or data ignore values
        finite = np.isfinite(self.data).any(axis=tuple(range(len(self.data.shape) - 1)))  # False = nans
        finite = finite & (self.data != float(self.header.get("data ignore value", 0))).any(
            axis=tuple(range(len(self.data.shape) - 1)))
        assert len(self.get_wavelengths()) == len(
            finite), "Error - hyperspectral dataset has %d bands but %d wavelengths." % (
        len(finite), len(self.get_wavelengths()))
        x = self.get_wavelengths()[finite]
        dx = np.abs(np.diff(x))
        maxstep = 2. * np.percentile(dx, p)
        if not (dx >= maxstep).any():  # no gaps - just return contiguous block!
            assert len(x) > min_size, "Error - total band count < min_size."
            msk = [self.get_band_index(b) for b in x]
            return [self.data[..., msk]], [x]
        else:
            break_start = list(np.argwhere(dx > maxstep)[:, 0])
            break_end = list((-np.argwhere(np.abs(np.diff(x[::-1])) > maxstep)[:, 0])[::-1])
            break_start.append(-1)  # add end of dataset so we don't miss last chunk
            break_end.append(-1)  # add end of dataset so we don't miss last chunk
            assert len(break_start) == len(
                break_end), r"Error - weird shit is happening? [ useful error messages ftw ¯\_(ツ)_/¯ ]"
            idx0 = 0
            chunks = []
            wav = []
            for i in range(len(break_start)):  # build chunks
                W = x[idx0:break_start[i]]
                msk = [self.get_band_index(b) for b in W]
                if W.shape[-1] > min_size:
                    wav.append(W)
                    chunks.append(self.data[..., msk])
                idx0 = break_end[i]  # skip forwards
            return chunks, wav

    ##################################
    ## Smoothing algorithms
    ###################################
    def smooth_median(self, window=3):
        """
        Applies running median filter on data.

        *Arguments*:
         - window = size of running window, must be int.

        *Returns*: Nothing - overwrites data with smoothed result.
        """

        assert isinstance(window, int), "Error - running window size must be integer."
        if len(self.data.shape) == 3:  # image data
            self.data = ndimage.median_filter(self.data, size=(1, 1, window))
        elif len(self.data.shape) == 2:  # point cloud data
            self.data = ndimage.median_filter(self.data, size=(1, window))
        else:
            assert False, "Error: Run_median does not work on %d-d data." % len(self.data.shape)

    def smooth_savgol(self, window=5, poly=2, **kwds):
        """
        Applies Savitzky-Golay-filter on data.

        *Arguments*:
         - window = size of running window, must be an odd integer.
         - poly = degree of polynom, must be int.
         - deriv = the order of derivative to evaluate (e.g. for stationary point analysis). Default is 0 [smooth
                   but don't compute derivative].
        *Keywords*: Keywords are passed to scipy.signal.savgol_filter(...).
        *Returns*: A copy of the input dataset with smoothed spectra.
        """

        assert isinstance(window, int), "Error - running window size must be integer."

        # extract contiguous chunks
        C, w = self.contiguous_chunks(min_size=window)

        # do smoothing
        kwds['window_length'] = window
        kwds['polyorder'] = poly
        kwds['axis'] = -1
        for X in C:
            mask = np.isfinite(X).all(axis=-1)  # remove nans
            X[mask, :] = signal.savgol_filter(X[mask, :], **kwds)

        # return copy
        out = self.copy(data=False)
        if self.is_image():
            out.data = np.dstack(C)
            out.set_wavelengths(np.hstack(w))
        else:
            out.data = np.vstack(C)
            out.set_wavelengths(np.hstack(w))
        return out


    ###################################
    # PLOTTING AND OTHER VISUALISATIONS
    ###################################
    # noinspection PyDefaultArgument
    def plot_spectra(self, ax=None, band_range=None, labels=None, indices=[], colours='blue', **kwds):
        """
        Plots a summary of all the spectra in this dataset.

        *Arguments*:
         - ax = an axis to plot to. If None (default), a new axis is created.
         - band_range = tuple containing the (min,max) band index (int) or wavelength (float) to plot.
         - labels = Labels for spectral features such that labels[0] = [feat1,feat2,..] and labels[1] = [name1,name2,...]
                    can be passed. Pass None (default) to disable labels.
         - indices = specific data point to plot. Should be a list containing index tuples, or an empty list if no pixels
                    should be plotted (Default).
         - colours = a matplotlib colour string or list of colours corresponding to each index spectra. Default is 'blue'.
        *Keywords*
         - quantiles = True if summary quantiles of all pixels should be plotted. Default is True.
         - median = True if the median spectra of all pixels should be plotted. Default is True.
         - other keywords are passed to plt.plot( ... ).
        """

        if ax is None:
            fig, ax = plt.subplots(figsize=(18, 6))

        # extract relevant range
        subset = self
        if band_range is not None:
            subset = self.export_bands(band_range)

        # ensure wavelengths are appropriate
        if not subset.has_wavelengths() or len(subset.get_wavelengths()) != subset.band_count():
            subset.set_wavelengths( np.arange( subset.band_count() ) )

        # calculate and plot percentiles
        quantiles = kwds.get("quantiles", True)
        median = kwds.get("median", True)
        if "quantiles" in kwds: del kwds['quantiles']  # remove keyword
        if 'median' in kwds: del kwds['median']  # remove keyword
        C, x = subset.contiguous_chunks()
        for C, x in zip(C, x):
            if quantiles or median:  # calculate percentiles
                percent = np.nanpercentile(C, axis=tuple(range(len(self.data.shape) - 1)),
                                           q=[5, 25, 50, 75, 95])  # calculate percentiles
                q5, q25, q50, q75, q95 = percent

            if median:  # plot median line
                ax.plot(x, q50, color='k', label='median', **kwds)
            if quantiles:  # plot percentile areas
                for lower, upper in zip([q5, q25], [q95, q75]):
                    _y = np.hstack([lower, upper[::-1]])
                    _x = np.hstack([x, x[::-1]])
                    ax.fill(_x[np.isfinite(_y)], _y[np.isfinite(_y)], color='grey', alpha=0.25)

            # plot specific spectra
            if isinstance(indices, tuple): indices = [indices]
            for i,idx in enumerate(indices):
                if isinstance(colours, list):
                    ax.plot(x, C[idx], color=colours[i], **kwds)
                else:
                    ax.plot(x, C[idx], color=colours, **kwds)

        # plot labels
        if not labels is None:  # plot labels?

            # parse string labels
            if isinstance(labels, str):
                if 'silicate' in labels.lower():  # plot common minerals theme
                    labels = ref.Themes.CLAY
                elif 'carbonates' in labels.lower():  # plot carbonate minerals
                    labels = ref.Themes.CARBONATE
                elif 'ree' in labels.lower():  # plot REE features
                    labels = None  # todo - add REE features

            i = 0
            for l in labels:
                # plot label
                if ax.get_xlim()[0] < l.pos < ax.get_xlim()[1]:
                    if isinstance(l, MultiFeature) or isinstance(l, MixedFeature):
                        l.quick_plot(ax=ax, method='fill+line', sublabel=i)
                        i += l.count()
                    else:
                        l.quick_plot(ax=ax, method='fill+line', label=i)
                        i += 1

        # add grid x-ticks
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.grid(which='major', axis='x', alpha=0.75)
        ax.grid(which='minor', axis='x', alpha=0.2)
        ax.set_xlabel("Wavelength (%s)" % self.header.get('wavelength units', 'nm'))
        #ax.set_ylabel("Reflectance") # n.b. not all images contain reflectance data...
        return ax.get_figure(), ax

    ###################################
    ##DATA TRANSFORMS AND COMPRESSION
    ###################################

    def compress(self):
        """
        Convert data array to int16 to save memory.
        """

        #no need to compress...
        if self.data.dtype == np.uint16: return

        assert np.nanmin(self.data) >= 0, "Error - to compress data range must be 0 - 1 but min is %s." % np.nanmin(self.data)
        assert np.nanmax(self.data) <= 1.0, "Error - to compress data range must be 0 - 1 but max is %s." % np.nanmax(self.data)

        #map to range 1 - 65535
        self.data = 65535 * (self.data)

        #map nans to zero
        self.data[ np.logical_not( np.isfinite(self.data)) ] = 0

        #convert data
        self.data = self.data.astype(np.uint16)

        #store min/max in header
        self.header["data ignore value"] = str(0)
        self.header['reflectance scale factor'] = str(65535)

    def decompress(self):
        """
        Expand data array to floats to get actual values
        """
        if (np.nanmax(self.data) <= 1.0) and (np.nanmin(self.data) >= 0.0):
            return # datset is already decompressed

        # get min/max data
        sf = float(self.header.get("reflectance scale factor", 65535))
        nan = float(self.header.get("data ignore value", -1))

        # expand data array to float32
        self.data = self.data.astype(np.float32)

        # set nans
        self.data[ np.isnan(self.data) | (self.data == nan) ] = np.nan

        # transform data back to original range
        self.data = self.data / sf #scale to desired range

    def normalise(self, minv=None, maxv=None):
        """
        Normalizes individual data points to account for variations in illumination and overall reflectivity. This can be done
        in two ways: if minv and maxv are both none, each pixel vector will be normalized to length 1. Otherwise, if minv
        and maxv are specified, every data point is normalised to the average of the bands between minv and maxv.

        *Returns*:
         - the normalising factor used for each data point.
        """

        # convert to float
        dtype = self.data.dtype
        self.data = self.data.astype(np.float32)

        # normalise pixel vectors
        if minv is None and maxv is None:

            # get valid bands
            valid = []
            for b in range(self.band_count()):
                if np.isfinite(self.get_band(b)).any() and not (self.get_band(b) == 0).all():
                    valid.append(b)

            # easy!
            nf = np.linalg.norm(self.data[..., valid], axis=-1).astype(np.float32)
            self.data /= nf[..., None]

        #normalise to band average
        else:
            # get minimum band to normalize over
            minidx = self.get_band_index( minv )
            maxidx = self.get_band_index( maxv )

            if minidx > maxidx:
                t = minidx
                minidx = maxidx
                maxidx = t

            # normalize
            nf = np.nanmean(self.data[..., minidx:maxidx], axis=-1).astype(np.float32)
            with np.errstate(all='ignore'):  # ignore div 0 errors etc.
                self.data /= nf[..., None]

        #convert back to original datatype
        if np.issubdtype( dtype, np.integer ):
            self.compress() #convert back to integer

        return nf

    def correct_spectral_shift(self, position):
        """
        Corrects potential spectral sensor shifts by shifting the offset (right) part of the spectrum
        :param position: Wavelength or band position of the first offset value - e.g. FENIX: band 714 or wavelength 976., respectively.
        :return: None - changes data in place
        """
        assert isinstance(position, int) or isinstance(position, float), "Error - shift position must be int (band number) or float (wavelength)."
        if isinstance(position, float):
            position = self.get_band_index(position)

        self.data[..., position:] += (self.data[..., (position - 1)] - self.data[..., position])[..., None]