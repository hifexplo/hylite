import numpy as np

# principal component analysis
from hylite.filter.mnf import plotMNF
from hylite import HyData

def PCA( hydata, output_bands = 20, band_range=None, step=5 ):

    """
    Apply a PCA dimensionality reduction to the hyperspectral dataset using singular vector decomposition (SVD).

    *Arguments*:
     - data = the dataset (HyData object) to apply PCA to.
     - output_bands = number of bands to return (i.e. how many dimensions to retain). Default is 20.
     - band_range = the spectral range to perform the PCA over. If (int,int) is passed then the values are treated as
                    min/max band IDs, if (float,float) is passed then values are treated as wavelenghts (in nm). If None is
                    passed (default) then the PCA is computed using all bands. Note that wavelengths can only be passed
                    if image is a hyImage object.
     - step = subsample the dataset during SVD for performance reason. step = 1 will include all pixels in the calculation,
              step = n includes every nth pixel only. Default is 5 (as most images contain more than enough pixels to
              accurately estimate variance etc.).
    *Returns*:
     - bands = Bands transformed into PCA space, ordered from highest to lowest variance.
     - factors = the factors (vector) each band is multiplied with to give the corresponding PCA band.
     - wav = a list of wavelengths the transform was applied to (handy for plotting), or None if wavelength info is not avaliable.
    """

    # get numpy array
    wav = None
    decomp = False
    if isinstance(hydata, HyData):
        wav = hydata.get_wavelengths()

        if hydata.is_int():
            hydata.decompress()  # MNF doesn't work very well with ints....
            decomp = True  # so we can compress again afterwards

        data = hydata.data.copy()
    else:
        data = hydata.copy()

    #get band range
    if band_range is None:  # default to all bands
        minb = 0
        maxb = data.shape[-1]
    else:
        if isinstance(band_range[0],int) and isinstance(band_range[1],int):
            minb, maxb = band_range
        else:
            assert isinstance(hydata, HyData), "Error - no wavelength information found."
            minb = hydata.get_band_index(band_range[0])
            maxb = hydata.get_band_index(band_range[1])

    #prepare feature vectors
    X = data[...,:].reshape(-1, data.shape[-1] )
    #print(minb,maxb)
    X = X[::step , minb:maxb ] #subsample
    X = X[np.isfinite(np.sum(X, axis=1)), :]  # drop vectors containing nans
    X = X[np.sum(X, axis=1) > 0, :]  # drop vectors containing all zeros

    #calculate mean and center
    mean = np.mean(X,axis=0)
    X = X - mean[None, :]

    #calcualte covariance
    cov = np.dot(X.T, X) / (X.shape[0] - 1)

    #and eigens (sorted from biggest to smallest)
    eigval, eigvec = np.linalg.eig(cov)
    idx = np.argsort(eigval)[::-1]
    eigvec = eigvec[:, idx]

    #project data
    data = data[...,minb:maxb] - mean
    out = np.zeros_like(data)
    for b in range( min(output_bands, data.shape[-1]) ):
        out[..., b] = np.dot( data, eigvec[:, b] )

    #filter wavelengths for return
    if not wav is None:
        wav = wav[minb:maxb]
    #compress?
    if decomp:
        hydata.compress()

    #prepare output
    outobj = hydata.copy(data=False)
    outobj.header.drop_all_bands()  # drop band specific attributes
    outobj.data = out[..., 0:output_bands]
    outobj.push_to_header()
    outobj.set_wavelengths( list(range(0,output_bands)) )
    return outobj, eigvec.T, wav


def plotPCA(n, R, factors, wavelength, flip=False, **kwds):

    """
    Utility function for plotting PCA components and their associated band weights calculate on images. Note that
    this method is identical to plotMNF(...).

    *Arguments*:
     - n = the nth PCA component will be plotted
     - R = array containing the PCA (as returned by PCA(...))
     - factors = the list of principal compoenent weights, as returned by PCA(...)
     - wavelengths = wavelengths corresponding to each band/factor used to calculate the minimum noise fractions,
                    as returned by PCA( ... ).
     - flip = True if the sign of the principal components/weights should be flipped. Default is False.
    *Keywords*:
     - keywords are passed to plot.imshow(...).
    *Returns*:
     - fig, ax = the figure and list of associated axes.
    """

    return plotMNF(n, R, factors, wavelength, flip, **kwds)