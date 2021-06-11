import numpy as np
import spectral
from matplotlib import pyplot as plt

# minimum noise filter
def MNF(hydata, output_bands=20, denoise_bands=40, band_range=None, inplace=False):

    """
    Apply a minimum noise filter to a hyperspectral image.

    *Arguments*:
     - hydata = A HyData instance containing the source dataset (e.g. image or point cloud).
     - output_bands = the number of bands to keep after MNF (dimensionality reduction). Default is 20.
     - denoise_bands = number of high-noise bands to treat as noise for denoising.
     - band_range = the spectral range to perform the MNF over. If (int,int) is passed then the values are treated as
                    min/max band IDs, if (float,float) is passed then values are treated as wavelenghts (in nm). If None is
                    passed (default) then the MNF is computed using all bands. Note that wavelengths can only be passed
                    if image is a hyImage object.
     - inplace = True if the original image should be denoised based on the MNF transform. Default is False.
    *Returns*:
     - mnf = a HyData instance containing the MNF bands.Note that only bands 0:*out_bands* will be kept in this dataset.
     - factors = A 2D numpy array containing the factors applied to the input datset. Useful
                 for plotting/interpreting the regions each MNF band is sensitive too.
    """

    # prepare data for MNF
    wav = hydata.get_wavelengths()
    decomp = False
    if hydata.is_int():
        hydata.decompress() #MNF doesn't work very well with ints....
        decomp=True #so we can compress again afterwards
    data = hydata.data

    # get range of bands to include in calculation
    if band_range is None:  # default to all bands
        minb = 0
        maxb = data.shape[-1]
    else:
        minb = hydata.get_band_index( band_range[0] )
        maxb = hydata.get_band_index( band_range[1] )

    assert minb < maxb, "Error - invalid range... band_range[0] > band_range[1]??"
    assert minb < data.shape[-1], "Error - band_range[0] out of range."
    if maxb == -1 or maxb > data.shape[-1]: maxb = data.shape[-1]

    # remove invalid bands
    valid_bands = []
    for b in range(minb,maxb):
        if np.isfinite(data[..., b]).any() \
                and not (np.nanmax(data[..., b]) == 0).all():
            valid_bands.append(b)

    #remove invalid bands
    data = np.array(data[..., valid_bands])

    # warn if bands have negative values...
    if np.nanmin(data) < 0.0:
        print("Warning - image contains negative pixels. This can cause unstable behaviour...")

    # calculate signal stats (as in spectral.calc_stats(...) but allowing for nans)
    X = data.reshape(-1, data.shape[-1]).T  # reshape to 1D list of pixels for each band
    X = X[:, np.isfinite(np.sum(X, axis=0))]  # drop columns containing nans
    X = X[:, np.sum(X, axis=0) > 0 ] #drop columns containing all zeros
    mean = np.mean(X, axis = 1)
    cov = np.cov(X)
    n = X.shape[1]
    signal = spectral.GaussianStats(mean, cov, n)

    # calculate noise as per spectral.noise_from_diffs (but allowing for nans)
    if len(data.shape) == 3: # image data
        deltas = data[:-1, :-1, :] - data[1:, 1:, :] #estimate noise by subtracting adjacent pixels
    elif len(data.shape) == 2: #point cloud data
        deltas = data[:-1, :] - data[1:, :]  # estimate noise by subtracting adjacent points

    X = deltas.reshape(-1, deltas.shape[-1]).T
    X = X[:, np.isfinite(np.sum(X, axis=0))]  # drop columns containing nans
    X = X[:, np.sum(X, axis=0) > 0]  # drop columns containing all zeros
    X = X[:, np.sum(X, axis=0) < np.nanpercentile( np.sum(X,axis=0), 50) ] #drop high noise data (these relate to edges)
    mean = np.mean(X, axis=1)
    cov = np.cov(X)
    n = X.shape[1]
    noise = spectral.GaussianStats(mean, cov, n)

    mnfr = spectral.mnf(signal, noise)

    # reduce bands
    reduced = mnfr.reduce(data, num=output_bands)

    #apply sign correction so there are less positive pixels than negative ones (sign is aribrary, helps maintain
    #consistency for plotting etc. by having low-valued background with some high-value regions (<50%)
    sign = np.nanmedian(reduced / np.abs(reduced)) #n.b. this will always be 1.0 or -1.0
    assert np.isfinite(sign), "Weird error - no non-nan values in MNF result?"
    reduced *= sign

    # denoise and export
    denoise = mnfr.denoise(data, num=denoise_bands)

    #update original image bands?
    if inplace:
        data[..., valid_bands] = denoise

    #calculate factors (for determining "important" bands)
    # noinspection PyProtectedMember
    factors = sign*mnfr.get_reduction_transform(num=output_bands)._A
    if not wav is None:
        wav = wav[valid_bands]

    #compress input dataset (so we don't change it)
    if decomp:
        hydata.compress()

    #prepare output
    out = hydata.copy(data=False)
    out.header.drop_all_bands()  # drop band specific attributes
    out.data = reduced
    out.push_to_header()

    return out, factors

def plotMNF(data, n, factors, wavelengths=None, flip=False, **kwds):

    """
    Utility function for plotting minimum noise fractions and their associated band weights.

    *Arguments*:
     - data = a HyData instance containing the MNF.
     - n = the nth mininimum noise fraction will be plotted.
     - factors = the factors array returned by MNF( ... ).
     - wavelength = the wavelengths corresponding to each factor. If None (default) indices are used instead.
     - flip = True if the sign of the minimum noise fraction and associated weights should be flipped. Default is False.
    *Keywords*:
     - cam = a camera object if data is a HyCloud instance. By default the header file will be searched for cameras.
     - other keywords are passed to HyData.quick_plot( ... ).
    *Returns*:
     - fig, ax = the figure and list of associated axes.
    """

    sign = 1
    if flip:
        sign = -1

    assert data.is_image() or data.is_point(), "Error - MNF data instance must be a HyImage or HyCloud."

    # create plot of mnf band
    data.data[..., n] *= sign # flip sign if needed
    kwds['vmin'] = kwds.get('vmin', np.nanpercentile(data.data[..., n], 1))
    kwds['vmax'] = kwds.get('vmax', np.nanpercentile(data.data[..., n], 99))
    if data.is_image(): # plot image
        aspx = data.aspx()
        fig, ax = plt.subplots(1, 2, figsize=(18 * 1.15, 18 * aspx),
                               gridspec_kw={'width_ratios': [10, 1], 'wspace': -0.11})
        data.quick_plot(n, ax=ax[0], **kwds)
    else: # plot point cloud
        kwds['cam'] = kwds.get("cam", data.header.get_camera())
        cam = kwds['cam']
        assert cam is not None, "Error - no valid camera object found. Try passing 'cam' as a keyword."
        aspx = cam.dims[1] / cam.dims[0]
        fig, ax = plt.subplots(1, 2, figsize=(18 * 1.15, 18 * aspx),
                               gridspec_kw={'width_ratios': [10, 1], 'wspace': -0.11})
        data.quick_plot(band=n, ax=ax[0],**kwds)
    ax[0].set_xticks([])
    ax[0].set_yticks([])

    data.data[..., n] *= sign # flip sign back

    # plot component weights
    if wavelengths is None:
        wavelengths = np.arange( factors[n].shape )
    assert wavelengths.shape[0] == factors[n].shape[0], "Error - number of wavelengths (%d) != number of factors (%d) " \
                                                        % (wavelengths.shape[0], factors[n].shape[0])
    ax[1].fill(np.hstack([[0], factors[n]*sign, [0]]),
               np.hstack([wavelengths[0], wavelengths, wavelengths[-1]]),
               color='k', alpha=0.2)
    ax[1].plot(factors[n]*sign, wavelengths, color='k')
    ax[1].axvline(0, color='k')
    ax[1].set_title("Band weights")
    ax[1].set_xticks([])
    ax[1].yaxis.tick_right()
    fig.subplots_adjust(wspace=None, hspace=None)
    fig.show()

    return fig, ax