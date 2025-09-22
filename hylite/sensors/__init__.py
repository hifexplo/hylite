"""
Static classes for sensor-specific processing such as lens-correction, adjustments for sensor shift and conversions
from digital numbers to radiance.
"""
import matplotlib.pyplot as plt
from .sensor import Sensor
from .fx import FX10
from .fx import FX17
from .fx import FX50
from .owl import OWL
from .fenix import Fenix
from .rikola import Rikola, Rikola_HSC2, Rikola_RSC1
from .fx import *
from .telopsNano import TelopsNano

# noinspection PyDefaultArgument
def QAQC(image, method, dim=0, fit="minmax", checklines=[]):
    """
    Estimate the spectral quality of a sensor according to reference measurements. Mask image first if required.

    Args:
        image (hylite.HyImage): the image containing data from the sensor..
        method (str): "LDPE" for SWIR using reference LDPE foil, "FT" for VNIR using fluorescence tube.
        dim (int): dimensionality of the evaluation (0 = overall average, 1 = row-wise, 2 = full frame).
        fit (str): method for peak fitting. For details, check hylite.analyse.mapping.minimum_wavelength( ... ).
        checklines (list): define custom features to check for (list of ints or floats).
    """

    image.data = image.data.astype(np.float32)

    # define indicative feature lines
    if method == "FT":
        checklines = [404.66, 435.83, 546.08, 611.08]
        image.data = np.nanmax(image.data) - image.data
    elif method == "LDPE":
        checklines = [1728., 1764., 2310., 2350.]

    # calculate and plot accuracy assessment depending on defined dimensionality
    if dim == 0:
        image.data = np.mean(image.data, axis=(0, 1))
        for line in checklines:
            lim = image.get_fwhm()[image.get_band_index(float(line))]
            mini, _, _ = image.minimum_wavelength(line - 10., line + 10., method=fit)
            # plot warning if error exceeds FWHM
            if (line - mini) > lim:
                print("\x1b[31m" + "Spectral accuracy at " + str(line) + " nm:   " + "{:.4f}".format(
                    line - mini) + " nm - WARNING: OVER FWHM" + '\x1b[0m')
            else:
                print("\x1b[32m" + "Spectral accuracy at " + str(line) + " nm:   " + "{:.4f}".format(
                    line - mini) + " nm" + '\x1b[0m')

    elif dim == 1:
        image.data = np.mean(image.data, axis=1)
        fig, axs = plt.subplots(2, 2, figsize=(15, 6))
        fig.subplots_adjust(hspace=.5, wspace=.3)
        axs = axs.ravel()
        i = 0
        for line in checklines:
            mini, _, _ = image.minimum_wavelength(line - 10., line + 10., method=fit)

            # color plot based on fwhm
            lim = image.get_fwhm()[image.get_band_index(float(line))]
            if any(x > lim for x in (line - mini)):
                col = "orangered"
            else:
                col = "limegreen"
            axs[i].plot(line - mini, col)
            axs[i].set_title(str(line) + ' nm')
            axs[i].axhline(y=lim, color="grey", linestyle="--")
            axs[i].text(0.9, 0.9, 'FWHM', fontsize=10, color="grey", va='center', ha='center', backgroundcolor='w',
                        transform=axs[i].transAxes)
            i += 1
        for ax in axs.flat:
            ax.set(xlabel='swath pixels', ylabel='spectral accuracy [nm]')

        plt.show()

    elif dim == 2:
        from matplotlib import colors
        # color plot based on fwhm
        fig, axs = plt.subplots(2, 2, figsize=(15, 6))
        fig.subplots_adjust(hspace=.5, wspace=.1)
        axs = axs.ravel()
        i = 0
        for line in checklines:
            mini, _, _ = image.minimum_wavelength(line - 10., line + 10., method=fit)
            lim = image.get_fwhm()[image.get_band_index(float(line))]
            cmap = "RdYlGn_r"
            norm = colors.Normalize(vmin=0, vmax=lim * 2)
            im = axs[i].imshow(line - mini, cmap=cmap, norm=norm)
            axs[i].set_title(str(line) + ' nm')
            cbar = fig.colorbar(im, ax=axs[i], cmap=cmap, norm=norm)
            cbar.ax.locator_params(nbins=3)
            cbar.ax.set_yticklabels(['within limits', 'FWHM', 'exceeding FWHM'])
            i += 1

        plt.show()