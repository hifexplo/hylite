"""
Classes containing different occlusion models (e.g. estimators for shadows, sky view factor etc.)
"""
from abc import ABCMeta, abstractmethod

class TransModel(object):
    """
    Base class for modelling occlusions (e.g. shadows) within a scene based on a specific light source (e.g. sunlight, skylight).
    """

    def __init__(self, geometry, pos ):
        """
        Create a new occlusion model.

        *Arguments*:
         - geometry = a HyCloud instance containing the 3-D scene geometry.
         - pos = a (3,) numpy array containing the viewing position.
        """
        self.geometry = geometry
        self.pos = pos

    @abstractmethod
    def fit(self, radiance ):
        """
        Estimate path radiance effects from measured radiance (or some other method). This should be implemented in child
        classes.
        """
        pass

    @abstractmethod
    def evaluate(self):
        """
        Calculate the fraction of light subtracted (absorbed) or added (emmitted/scattered) due to path radiance effects.

        *Returns*:
         - A HyCloud instance containing two bands (m and c), such that at sensor radiance = m * reflected radiance + c.
        """
        pass




    # def correct_atmosphere(self, ref_feature=1125., cdepth=90, indices=[], maxp=98, nthreads=1, vb=True):
    #     """
    #     Correct for atmospheric absorbtions that are not captured by the calibration panel due to e.g., large distance
    #     between sensor/calib panel and target using the method described
    #     by Lorenz et al., 2018, https://doi.org/10.3390/rs10020176
    #
    #     *Arguments*:
    #      - ref_feature = The reference absorbtion feature to quantify atmospheric influence with (and hence calculate the
    #                      magnitude of the correction. Cf. Lorenz et al., 2018 for more details. Default is the water
    #                      absorbion feature at 1225 nm.
    #      - cdepth = The percentile depth cutoff used to select pixels with large atmospheric effects. Default is 90, meaning
    #                the atmospheric absorbtion spectra is characterised using the farthest 10% of pixels.
    #      - indices = individual pixel indices to include in output plots (if vb is True). See HyImage.quick_plot(...) for details.
    #      - maxp = Post-hull correction threshold used to distinguish atmospheric effects present in all pixels from mineralogical
    #               features that only exist in some pixels. Default is 98.
    #      - nthreads = number of threads used to compute the hull corrections. Default is 1.
    #      - vb = True if comparison plots should be created showing the correction spectra and adjusted spectra. Default is True.
    #     """
    #
    #     # get reference pixels with "largest" distance (and hence atmospheric effects)
    #     depth = self.get_depth()
    #     depth[np.isinf(depth)] = np.nan
    #     refpx = np.argwhere(depth > np.nanpercentile(depth, cdepth))
    #
    #     # plot these to check that they cover most of the "geological" variation in the scene
    #     if vb:
    #         fig, ax = plt.subplots(1, 3, figsize=(25, 8))
    #         self.image.quick_plot(hylite.RGB, ax=ax[0])
    #         ax[0].set_title("Pixels used to estimate atmospheric spectra.")
    #         ax[0].scatter(refpx[..., 0], refpx[..., 1], s=1, alpha=0.1)
    #
    #     ## extract these pixels
    #     refpx = hylite.HyData(self.image.data[refpx[..., 0], refpx[..., 1], :],
    #                           wavelengths=self.image.get_wavelengths())
    #
    #     ## apply hull correction and extract atmospheric signal (this will be the "max" of the hc spectra)
    #     hc = get_hull_corrected(refpx)
    #     atmos = np.nanpercentile(hc.data, maxp, axis=0)
    #
    #     # plot results before correction
    #     if vb:
    #         ax[1].set_title("Uncorrected spectra + estimated atmospheric spectra")
    #         ax[1].set_ylabel("Uncorrected pixel reflectance")
    #         self.image.plot_spectra(indices=indices, ax=ax[1])
    #         ax2 = ax[1].twinx()
    #         ax2.plot( self.image.get_wavelengths(), atmos, color='b', alpha=0.5)
    #         ax2.set_ylabel("Estimated atmospheric absorbtions (%)")
    #         ax2.spines['right'].set_color('blue')
    #         ax2.yaxis.label.set_color('blue')
    #         ax2.tick_params(axis='y',colors='blue')
    #         ax[1].axvline(ref_feature, color='b', lw=4, alpha=0.5) # plot reference feature
    #
    #     # calculate correction factor and scale to match depth of reference water feature
    #     if ref_feature == 1125.:
    #         refdepth = parallel_chunks(get_hull_corrected, self.image, (1050., 1250.), nthreads=nthreads) # just hull-correct relevant part of spectra (faster)
    #     else:
    #         refdepth = parallel_chunks(get_hull_corrected, self.image, (0, -1), nthreads=nthreads) # hull correct whole spectra....
    #
    #     r1025 = refdepth.get_band(ref_feature)  # observed reflectance at r1025 feature
    #     f = r1025 / atmos[self.image.get_band_index(ref_feature)] # adjustment factor such that correction removes reference feature
    #     fac = atmos[None, None, :] * f[:, :, None] # resulting per-pixel atmospheric correction factor
    #
    #     # apply correction
    #     self.image.data /= fac
    #
    #     # plot results
    #     if vb:
    #         ax[2].set_title("Corrected spectra")
    #         self.image.plot_spectra(indices=indices, ax=ax[2])
    #         ax[2].axvline(ref_feature, color='b', lw=4, alpha=0.5)
    #         fig.show()