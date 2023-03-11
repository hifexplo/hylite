"""
A base class for all types of hyperspectral data. Inherited by HyCloud, HyImage and HyLibrary.
"""

import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from scipy import ndimage, signal

import hylite
from hylite import HyHeader
import hylite.reference.features as ref
from hylite.hyfeature import HyFeature, MultiFeature, MixedFeature
from matplotlib.ticker import AutoMinorLocator

from tqdm import tqdm

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
    def __init__(self, data, header=None, **kwds):
        """
        Create an image object from a data array.

        Args:
            data (ndarray): array such that the last dimension corresponds to individual bands (e.g. data[pointID, band] or data[px,py,band])
            header (hylite.HyHeader): associated header file. Default is None (create a new header).
        """

        #copy reference to data. Note that this can be None!
        self.data = data
        if not data is None:
            self.dtype = data.dtype
        else:
            self.dtype = None

        # header data
        self.set_header(header)

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
        Args:
            data (bool): True if a copy of the data should be made, otherwise only copy header.

        Returns:
            a new HyData instance.
        """

        if not data or self.data is None:
            return HyData( None, header=self.header.copy())
        else:
            return HyData( self.data.copy(), header=self.header.copy())

    def set_header(self, header):
        """
        Loads associated header data into self.header.

        Args:
            header (hylite.HyHeader): a HyHeader object or None.
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
        if self.has_wavelengths():
            return self.header.get_wavelengths() # return wavelengths
        else:
            return np.arange(self.band_count()) # return band indices as proxy

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
        if self.data is None: # for point clouds data can be none
            return False
        return len(self.data.shape) == 3

    def is_point(self):
        """
        Return true if this dataset is an point cloud or related dataset (i.e. data array has dimension [idx,b]). Note
        that this will return true for spectral libraries and other 'cloud like' datasets.
        """
        if self.data is None: # for point clouds data can be none
            return True
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

        Args:
            bands (tuple, list): either:
                     (1) a tuple containing the (min,max) wavelength to extract. If range is a tuple, -1 can be used to specify the
                         last band index.
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
            mask[ mn:(mx+1) ] = False
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

    def delete_nan_bands(self, inplace=True):
        """
        Remove bands in this image that contain only nans.

        Args:
            inplace (bool): True if this operation should be applied to the data in situ. Default is True.

        Returns:
            an image copy with the nan bands removed IF inplaces is False. Otherwise the image is modified inplace.
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

    def set_as_nan(self, value, strict=True):
        """
        Sets data with the specified value to NaN. Useful for handling no-data values.

        Args:
            value (float, int): the value to (permanently) replace with np.nan.
            strict (bool): True if all bands must have this value to set it as nan. Default is True. If False, all occurences of
                    value will be replaced with nan.
        """

        if strict:
            if self.is_int():
                nan = int(self.header.get("data ignore value", 0))
                self.data[ (self.data == value).all(axis=-1) ] = nan
                self.header["data ignore value"] = str(nan)
            else:
                self.data[ (self.data == value).all(axis=-1) ] = np.nan
        else:
            if self.is_int():
                nan = int(self.header.get("data ignore value", 0))
                self.data[ self.data == value ] = nan
                self.header["data ignore value"] = str(nan)
            else:
                self.data[self.data == value] = np.nan

    def mask_bands(self, mn, mx=None, val=np.nan):
        """
        Masks a specified range of bands, useful for removing water features etc.

        Args:
            min (float): the start of the band mask (as per get_band_index(...)).
            max (float): the end of the band mask (as per get_band_index( ... )). Can be None to mask individual bands. Default is
                 None.
            val (float, int): the value to set masked bands to. Default is np.nan. Set to None to keep values but flag bands in band band list.
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

    def mask_water_features(self, mask=None):
        """
        Removes typical water features. By default this removes bands between:

          - 960 - 990 nm
          - 1320 - 1500 nm
          - 1780 - 2050 nm
          - 2400 - 2500 nm

        Custom wavelengths can be set using the mask keyword.

        Args:
            mask (list,tuple): mask custom bands. This should be a list of tuple band indices or wavelengths containing the
                    minimum and maximum wavelenght/index of each region to mask.
        """

        default = [(960.0, 990.0), (1320.0, 1500.0), (1780.0, 2050.0), (2400.0, 2502.0)]
        if mask is None:
            mask = default

        # mask bands
        for mn, mx in mask:
            try:
                self.mask_bands(mn, mx)
            except:
                pass  # ignore errors associated with out of range etc.

    #########################
    ## band getters/setters
    #########################
    def get_band(self, b):
        """
        Gets an individual band from this dataset. If an integer is passed it is treated as a band index. If a string is passed it is
        treated as a band name. If a float is passed then the closest band to this wavelength is retrieved.

        Args:
            b (int,float,str): the band to get. Integers are treated as indices, strings as band names and floats as wavelengths.

        Returns:
            a sliced np.array exposing the band. Note that this is NOT a copy.
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

        Returns:
            pixels (ndarray): an array such that pixel[n][band] gives the spectra of the nth pixel.
        """

        return self.data.reshape(-1, self.data.shape[-1])

    def X(self, onlyFinite = False):
        """
        A shorthand way of writing get_raveled(), as X is conventionally used for a vector of spectra.

        Args:
            onlyFinite (bool): True if data points containing nan bands should be removed from the feature vector. Default is False.
        """
        X = self.get_raveled()
        if onlyFinite:
            return X[ np.isfinite(X).all(axis=-1) ]
        else:
            return X

    def set_raveled(self, pix, shape=None, onlyFinite = False, strict=True ):
        """
        Fills the image/dataset from a list of pixels of the format returned by get_pixel_list(...). Note that this does not
        copy the list, but simply stores a view of it in this image.

        Args:
            pix (list, ndarray): a list such that pixel[n][band] gives the spectra of the nth pixel.
            shape (tuple): the reshaped data dimensions. Defaults to the shape of the current dataset, except with auto-shape for the last dimension.
            onlyFinite (bool): True if pix contains only pixel values corresponding to non-nan pixels in self.data (as returned by self.X( True ) ).
            strict (bool): True if set_raveled should not change the number of bands in this image. Default is True.
        """
        if shape is None:
            shape = list( self.data.shape )
            shape[-1] = -1

        if strict: # number of bands cannot change
            assert self.data.shape[-1] == pix.shape[-1], \
                "Error: image and pix array have different number of bands. To allow changes to band count please specify strict=False."
            if onlyFinite:
                self.data[ np.isfinite(self.data).all(axis=-1), : ] = pix
            else:
                self.data = pix.reshape(shape)
        else:
            newdata = np.full( self.data.shape[:-1] + (pix.shape[-1],), np.nan, )
            if onlyFinite:
                if onlyFinite:
                    newdata[np.isfinite(self.data).all(axis=-1), :] = pix
                else:
                    newdata = pix.reshape( self.data.shape[:-1] + (pix.shape[-1],) )
            self.data = newdata

    def get_band_index(self, w, **kwds):
        """
        Get the band index that corresponds with the given wavelength or band name.

        Args:
            w (float, int, str): the wavelength, index or band name search for. Note that if w is an integer it is treated
               as a band index and simply returned. If it is a string then the index of the matching band name
               is returned. If it is a wavelength then the closest band to this wavelength is returned.
            **kwds: This function takes one keyword:

                 - thresh = the threshold (in nanometers) within which a band must fall to be valid. Default is
                            hylite.band_select_threshold (which defaults to 10 nm). If a wavelength is passed and a
                            band exists within this distance, then it is returned. Otherwise an error is thrown).

        Returns:
            the matching band index.
        """

        thresh = kwds.get("thresh", hylite.band_select_threshold)

        if np.issubdtype( type(w), np.integer ):  # already a valid band index
            assert -self.band_count() <= w <= self.band_count(), "Error - band index %d is out of range (image has %d bands)." % (w, self.band_count())
            if w < 0: #convert negative indices to positive ones
                return self.band_count() + w
            else:
                return w
        elif isinstance(w, str): # treat w as band name
            assert w in self.get_band_names(), "Error - could not find band with name %s" % w
            return int(list(self.get_band_names()).index(w))
        elif np.issubdtype( type(w), np.floating ): # otherwise treat w as wavelength
            wavelengths = self.get_wavelengths()
            diff = np.abs( wavelengths - w)
            assert np.nanmin(diff) <= thresh, "Error - no bands exist within %d nm of wavelength %f. Try increasing the 'thresh' keyword?" % (thresh, w)
            return int(np.argmin(diff))
        else:
            assert False, "Error - %s is an unknown band descriptor type." % type(w)

    def resample(self, w, agg=True, bw=None, vb=True, partial=False, **kwds):
        """
        Return a copy of this dataset resampled onto the specified wavelength array. Note that this does not
        do any interpolation, but rather selects the nearest band(s) and averages them if need be. Hence it is useful
        for reducing spectral resolution to match e.g. another dataset, but should NOT be used
         to sample bands at higher spectral resolution.

        Args:
            w (ndarray): Either a 1-D numpy array containing the wavelengths to sample onto (see bw for setting the band width), or
               a (n,2) array containing (start, end) wavelength ranges for each band (in this case bw will be ignored).
            agg (bool): True if bands between each entry in w (i.e. w - bw -> w + bw) should be averaged (i.e. spectral binning).
                  Default is True. If False then the closest value will be used (i.e. spectral subsampling).
            bw (float): the width of each band. If None (default) then this is calculated as w[1] - w[0] (i.e. assume regular
                 spacing). This has no effect if agg is False.
            partial (bool): True if partial overlap between the source and target wavelengths is allowed. Non-overlapping areas will be
                      replaced with nan. Default is False.
            vb (bool): True if a progress bar should be created. Default is True.
            **kwds: if provided, the thresh keyword will be passed to get_band_index(...) to control tolerances when selecting the
                closest bands. See documentation for get_band_index(...) for more details.

        Returns:
            a copy of this HyData instance resampled onto the new wavelength array.
        """

        out = self.copy(data=False)  # create output array
        out.data = np.full(self.data.shape[:-1] + (len(w),), np.nan)
        out_wav = []
        w = np.array(w)  # ensure this is an array
        if bw is None and len(w.shape) == 1:
            bw = abs(w[1] - w[0])
        loop = range(len(w))
        if vb:
            loop = tqdm(loop, desc='Resampling bands', leave=False)
        for i in loop:
            idx0 = None
            idx1 = None
            if agg:  # binning
                if len(w.shape) == 1: # W is 1-D
                    out_wav.append(w[i])  # use specified wavelength.
                    try:
                        idx0 = self.get_band_index(w[i] - bw * 0.5, **kwds)
                        idx1 = self.get_band_index(w[i] + bw * 0.5, **kwds)
                    except AssertionError:
                        if partial:
                            pass
                        else:
                            assert False, "Error - source wavelength array (%.1f - %.1f) does not cover target array (%.1f - %.1f)." % (
                                self.get_wavelengths()[0], self.get_wavelengths()[-1], w[0], w[-1])

                else: # W is 1-D, containing (start,end) pairs.
                    out_wav.append((w[i, 0] + w[i, 1]) / 2)  # use midpoint as wavelength
                    try:
                        idx0 = self.get_band_index(w[i,0], **kwds)
                        idx1 = self.get_band_index(w[i,1], **kwds)
                    except AssertionError:
                        if partial:
                            pass
                        else:
                            assert False, "Error - source wavelength array (%.1f - %.1f) does not cover target array (%.1f - %.1f)." % (
                                self.get_wavelengths()[0], self.get_wavelengths()[-1], w[0], w[-1])
                if idx0 is not None and idx1 is not None:
                    out.data[..., i] = np.nanmean(self.data[..., idx0:(idx1 + 1)], axis=-1)
            else:
                assert len(w.shape) == 1, "Error - if agg is False then w must be 1d, not %s" % str(w.shape)
                try:
                    idx = self.get_band_index(w[i], **kwds)
                    out_wav.append(self.get_wavelengths()[idx])  # use real wavelength
                    out.data[..., i] = self.data[..., idx]  # nearest-neighbour resampling
                except AssertionError:
                    if partial:
                        out_wav.append(w[i])  # use passed wavelength
                    else:
                        assert False, "Error - source wavelength array (%.1f - %.1f) does not cover target array (%.1f - %.1f)." % (
                        self.get_wavelengths()[0], self.get_wavelengths()[-1], w[0], w[-1])
        out.set_wavelengths(out_wav)
        return out

    def contiguous_chunks(self, p=75, min_size=0):
        """
        Extract contiguous chunks of spectra, splitting a (1) completely nan bands or (2) large steps in wavelength.

        Args:
            p (int): the percentile used to define a large change in wavelength. Default is 90. A "gap" is considered to be
               a change in wavelength greater than double this percentile.
            min_size (int): the minimum number of bands required to consider a chunk valid. Default is 0 (return all chunks).

        Returns:
            Tuple containing

            chunks (ndarray): copies of the orignal data array that contain continuous spectra. At least one pixel/point
                    in each slice of these bans is guaranteed to be finite.
            wav (ndarray): array containing the wavelengths corresponding to each band of each chunk.
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

        Args:
            window (int): size of running window, must be int.

        Returns:
            Nothing - overwrites data with smoothed result.
        """

        assert isinstance(window, int), "Error - running window size must be integer."
        if len(self.data.shape) == 3:  # image data
            self.data = ndimage.median_filter(self.data, size=(1, 1, window))
        elif len(self.data.shape) == 2:  # point cloud data
            self.data = ndimage.median_filter(self.data, size=(1, window))
        else:
            assert False, "Error: Run_median does not work on %d-d data." % len(self.data.shape)

    def smooth_savgol(self, window=5, poly=2, chunk=False, **kwds):
        """
        Applies Savitzky-Golay-filter on data.

        Args:
            window (int): size of running window, must be an odd integer.
            poly (int): degree of polynom, must be int.
            chunk (bool): True if the data should be split into chunks (removing e.g. nan bands) before filtering. Use with care!
                   Default is False.
            **kwds: Keywords are passed to scipy.signal.savgol_filter(...).

        Returns:
            A copy of the input dataset with smoothed spectra.
        """

        assert isinstance(window, int), "Error - running window size must be integer."

        # extract contiguous chunks
        if chunk:
            C, w = self.contiguous_chunks(min_size=window)
        else:
            C = [self.data.copy()]
            w = [self.get_wavelengths().copy()]

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

    def fill_gaps(self):
        """
        Fill spectral gaps using nearest neighbour interpolation (in the spectral direction). This operation is applied in-place.
        """
        from scipy import ndimage as nd
        for i in range(self.data.shape[0]):
            if self.is_image():
                for j in range(self.data.shape[1]):
                    invalid = np.logical_not(np.isfinite(self.data[i, j, :]))
                    if invalid.any():
                        ind = nd.distance_transform_edt(invalid, return_distances=False, return_indices=True)
                        self.data[i, j, :] = self.data[i, j, tuple(ind)]
            else:
                invalid = np.logical_not(np.isfinite(self.data[i, :]))
                if invalid.any():
                    ind = nd.distance_transform_edt(invalid, return_distances=False, return_indices=True)
                    self.data[i, :] = self.data[i, tuple(ind) ]

    ###################################
    # PLOTTING AND OTHER VISUALISATIONS
    ###################################
    # noinspection PyDefaultArgument
    def plot_spectra(self, ax=None, band_range=None, labels=None, indices=[], colours='blue', **kwds):
        """
        Plots a summary of all the spectra in this dataset.

        Args:
            ax: an axis to plot to. If None (default), a new axis is created.
            band_range (tuple): tuple containing the (min,max) band index (int) or wavelength (float) to plot.
            labels (list): Labels for spectral features such that labels[0] = [feat1,feat2,..] and labels[1] = [name1,name2,...]
                    can be passed. Pass None (default) to disable labels.
            indices (list): specific data point to plot. Should be a list containing index tuples, or an empty list if no pixels
                    should be plotted (Default).
            colours (list,str): a matplotlib colour string or list of colours corresponding to each index spectra. Default is 'blue'.
            **kwds: keywords are passed to plt.plot( ... ), except the following options:

                 - quantiles = True if summary quantiles of all pixels should be plotted. Default is True.
                 - median = True if the median spectra of all pixels should be plotted. Default is True.

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
        if np.nanmin(self.data) < 0:
            print( "Warning - negative values cannot be converted to uint and will be lost. "
                   "Minimum value in dataset is: %s." % np.nanmin(self.data) )

        #map to range 1 - 65535
        sf = 65535 / np.nanmax(self.data)
        self.data *= sf
        self.data[self.data < 0] = 0

        #convert data to uint16
        self.data = self.data.astype(np.uint16)

        #store min/max in header
        self.header["data ignore value"] = str(0)
        self.header['reflectance scale factor'] = sf

    def decompress(self):
        """
        Expand data array to floats to get actual values
        """
        # no need to decompress...
        if self.data.dtype != np.uint16: return

        # get min/max data
        sf = float(self.header.get("reflectance scale factor", 65535))
        nan = float(self.header.get("data ignore value", -1))

        # update header accordingly (to avoid save / load issues in the future!)
        self.header["reflectance scale factor"] = 1.0
        if 'data ignore value' in self.header:
            del self.header['data ignore value']

        # expand data array to float32
        self.data = self.data.astype(np.float32)
        self.data /= sf

        # set nans
        self.set_as_nan(nan)

    def normalise(self, minv=None, maxv=None):
        """
        Normalizes individual data points to account for variations in illumination and overall reflectivity. This can be done
        in two ways: if minv and maxv are both none, each pixel vector will be normalized to length 1. Otherwise, if minv
        and maxv are specified, every data point is normalised to the average of the bands between minv and maxv.

        Returns:
            the normalising factor used for each data point.
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

    def percent_clip(self, minv=2, maxv=98, per_band=False, clip=True):
        """
        Scale self.data such that the specified percentiles become 0 and 1 respectively. Note that this
        normalisation is applied in situ.

        Args:
            minv (int): the lower percentile. Default is 2.
            maxv (int): the upper percentile. Default is 98.
            per_band (bool): apply scaling to bands independently. Default is False.
            clip (bool): True if values < minv or > maxv should be clipped to 0 or 1. Default is True.

        Returns:
            vmin, vmax (float): the percentile clip thresholds used for the normalisation
        """

        # calculate percentile thresholds
        if per_band:
            minv, maxv = np.nanpercentile(self.data, (minv, maxv), tuple(np.arange(len(self.data.shape) - 1)))
        else:
            minv, maxv = np.nanpercentile(self.data, (minv, maxv))

        # ensure dtype is a float
        if np.issubdtype(self.data.dtype, np.integer):
            self.data = self.data.astype(np.float32)

        # apply normalisation
        self.data = (self.data - minv) / (maxv-minv)

        # apply clipping
        if clip:
            self.data = np.clip( self.data, 0, 1 )

        return minv, maxv

    def correct_spectral_shift(self, position):
        """
        Corrects potential spectral sensor shifts by shifting the offset (right) part of the spectrum.

        Args:
            position (float,int): Wavelength or band position of the first offset value - e.g. FENIX: band 714 or wavelength 976., respectively.

        Returns:
            None - changes data in place
        """
        assert isinstance(position, int) or isinstance(position, float), "Error - shift position must be int (band number) or float (wavelength)."
        if isinstance(position, float):
            position = self.get_band_index(position)

        self.data[..., position:] += (self.data[..., (position - 1)] - self.data[..., position])[..., None]