from hylite import HyData, HyLibrary
import numpy as np

"""
Utility functions for sampling from hyperspectral datasets, including extracting averaged spectra (spectra libraries) 
and resampling images to match band widths of other sensors (e.g. ASTER bands).
"""

def getClassLibrary( data, classification, skip0=True ):
    """
    Create a spectral library from a classified image by extracting all spectra associated with each class.

    *Arguments*:
     - data = a HyImage or HyCloud instance containing spectral data.
     - classification = a HyImage or HyCloud instance containing classifications per point/pixel.
     - skip0 = True if class 0 (typically background) should be ignored. Default is True.
    """

    assert isinstance(data, HyData), "Error - data must be a HyData instance not %s" % type(data)
    assert isinstance(classification, HyData), "Error - classification must be a HyData instance not %s" % type(classification)
    if not classification.is_classification():
        print( "Warning - classification is not a classification dataset (at least according to its header)?")

    spectra = data.get_raveled()
    cls = classification.get_raveled()[:,0]
    names = classification.header.get_class_names()
    if len(names) == 0:
        names = [str(i) for i in range(np.max(cls))]

    refl = []
    upper = []
    lower = []
    n = []
    mn = 0
    if skip0:
        mn = 1
    for i in range(mn,len(names)):
        msk = cls==i
        if msk.any():
            l, m, u = np.nanpercentile(spectra[msk,:], (25, 50, 75), axis=0)
            lower.append(l)
            refl.append(m)
            upper.append(u)
            n.append(names[i])

    return HyLibrary(n, np.vstack(refl),
                     lower=np.vstack(lower),
                     upper=np.vstack(upper),
                     wav=data.get_wavelengths())

class Resample( object ):
    """
    A wrapper class for spectral resampling of hyperspectral images.
    """

    def __init__( self, bands ):
        """
        Create a spectral resampler.

        *Arguments*:
         - a list of wavelength tuples specifying the minimum and maximum spectra of each band in this resampling scheme.
        """

        # check valid
        assert len(bands) > 0, "Error - no valid bands specified"
        for b in bands:
            assert len(b) == 2, "Error - bands must be specified as a tuple of length 2."
        self.bands = bands

    def get_band(self, data, n ):
        """
        Get the n'th band under this resampling scheme by averaging hyperspectral bands between the
        specified range.

        *Arguments*:
         - data = the dataset to extract information from.
         - n = the resampled band index to extract. NOTE THAT BAND INDICES START AT 1 FOR COMPATIBILITY WITH
               STANDARD SATELLITE NOTATION!
        """
        assert n >= 1 and (n-1) < len(self.bands), "Error - Band %d is not defined in this resampling scheme." % n
        idx0 = data.get_band_index( self.bands[n-1][0], thresh=np.inf )
        idx1 = data.get_band_index( self.bands[n-1][1], thresh=np.inf )

        if idx1 != idx0:
            return np.nanmean( data.data[...,idx0:idx1], axis=-1 )
        else:
            # is data within sampling range at all?
            minw,maxw = data.get_wavelengths()[[0,-1]]
            if (minw > self.bands[n-1][0]) and (maxw > self.bands[n-1][0]) \
                or (minw < self.bands[n-1][0]) and (maxw < self.bands[n-1][0]):
                return np.full( data.data.shape[:-1], np.nan ) # no data
            else:
                return data.data[..., idx0] # return this band

    def print_bands(self):
        """
        Quickly print the wavelengths associated with each band of this resampler.
        """

        for i,b in enumerate(self.bands):
            print("Band %d: %.1f - %.1f nm" % (i+1, b[0], b[1]))

    def apply(self, data):
        """
        Apply this resampling to a HyData instance and return a new instance with appropriately averaged bands.

        *Arguments*:
         - data = the HyData instance to apply this sampling scheme to.
        *Returns*: A copy of the original HyData instance with the bands averaged as defined. Corresponding wavelengths will be set to the middle of each averaged region.
        """

        bands = [self.get_band(data, n + 1) for n in range(len(self.bands))]
        out = data.copy(data=False)
        if data.is_image():
            out.data = np.dstack(bands)
        else:
            out.data = np.hstack(bands)
        out.set_wavelengths([np.mean(self.bands[n]) for n in range(len(self.bands))])
        return out

# create instances for common satellites
# based on https://www.indexdatabase.de/db/bs.php

ASTER = Resample( [
    (520.0, 600.0),
    (630.0, 690.0),
    (760.0, 860.0),
    (1600.0, 1700.0),
    (2145.0, 2185.0),
    (2185.0, 2225.0),
    (2235.0, 2365.0),
    (2295.0, 2365.0),
    (2360.0, 2430.0),
    (8125.0, 8475.0),
    (8475.0, 8825.0),
    (8925.0, 9275.0),
    (10250.0, 10950.0),
    (10950.0, 11650.0) ] )
""" Static resampling class for sampling hyperspectral data onto ASTER bands"""

SENTINEL = Resample( [
    (433.0, 453.0),
    (458.0, 522.0),
    (543.0, 577.0),
    (650.0, 680.0),
    (698.0, 712.0),
    (733.0, 747.0),
    (773.0, 793.0),
    (785.0, 899.0),
    (855.0, 875.0),
    (935.0, 955.0),
    (1360.0, 1390.0),
    (1565.0, 1655.0),
    (2100.0, 2280.0) ] )
""" Static resampling class for sampling hyperspectral data onto Sentinel bands"""
