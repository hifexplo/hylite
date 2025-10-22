"""
A collection of functions for applying common transforms to hyperspectral data cubes, hyperclouds and/or spectral libraries.
Requires scikit-learn to be installed.
"""
try:
    import sklearn
except:
    assert False, "Please install scikit-learn using `pip install scikit-learn` to use these transforms."

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA as sPCA
from sklearn.covariance import EmpiricalCovariance
import hylite
import matplotlib.pyplot as plt

class NoiseWhitener(BaseEstimator, TransformerMixin):
    """
    A scikit-learn compatible transformer that performs spatial noise whitening
    for use in Minimum Noise Fraction (MNF) transforms on hyperspectral cubes.
    """

    def __init__(self, noise_estimate=None, neighbor_axis=0, subsample=5, noiseMethod='spectral'):
        """
        Estimate or apply a noise model for hyperspectral data.

        This function either uses a provided noise estimate or computes one from spatial or spectral
        differences in the data. The resulting noise statistics can be used for subsequent 
        denoising, covariance estimation, or dimensionality reduction tasks.

        Args:
            noise_estimate: Optional array of shape (H, W, B) containing per-pixel noise estimates. 
                            If None, noise is estimated from spatial or spectral differences.
            neighbor_axis: Integer specifying which spatial axis to use for differencing when 
                        estimating noise (0 = rows, 1 = columns). Default is 0.
            subsample: Integer factor to subsample noise samples during covariance estimation. 
                    Reduces computation time and memory usage for large datasets.
            noiseMethod: String specifying the differencing method to use. 
                        'spectral' computes band-wise differences, while 
                        'spatial' computes spatial differences (image data only).

        Returns:
            A numpy array representing the estimated noise covariance or adjusted noise model.
        """

        self.noise_estimate = noise_estimate
        self.neighbor_axis = neighbor_axis
        self.Wn_ = None
        self.subsample = subsample
        self.noiseMethod = noiseMethod
        self.wavelengths = None
        self.estimate = None
        
    def fit(self, X, y=None):
        """
        Fit the noise whitening matrix.
        X : ndarray of shape (H, W, B)
        """
        if isinstance(X, hylite.HyData):
            self.wavelengths = X.get_wavelengths()
            X = X.data
        
        # estimate noise if not provided
        if self.noise_estimate is None:
            if 'spatial' in self.noiseMethod.lower():
                assert X.ndim == 3, "Spatial differencing can only be used for hyperspectral image data."
                if self.neighbor_axis == 0:
                    noise = np.abs(X[1:, :, :] - X[:-1, :, :])
                elif self.neighbor_axis == 1:
                    noise = np.abs(X[:, 1:, :] - X[:, :-1, :])
            else:
                noise = np.abs(X[..., 1:] - X[..., :-1]) # compute forward difference between adjacent bands
                noise += np.abs(X[...,::-1][..., 1:] - X[...,::-1][..., :-1])[...,::-1] # compute backward difference between adjacent bands
                noise /= 2
                noise = np.concatenate([ noise[..., 0][...,None], noise ], axis=-1) # add first band difference using padding
        else:
            noise = self.noise_estimate

        # Flatten noise to (n_samples, n_bands)
        if X.ndim == 3: # images
            noise = noise.reshape(-1, noise.shape[-1])

        # remove nans and subsample
        noise = noise[~np.isnan(noise).any(axis=1), :][::self.subsample, :]
        self.estimate = noise.mean(axis=0)

        # Estimate noise covariance and compute whitening matrix
        cov_noise = EmpiricalCovariance().fit(noise).covariance_
        eigvals, eigvecs = np.linalg.eigh(cov_noise)
        eigvals = np.clip(eigvals, a_min=1e-12, a_max=None) # avoid negative eigenvals as this breaks everything!
        self.Wn_ = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

        return self

    def quick_plot(self, ax=None, **kwargs):
        """
        Plot the estimated noise spectrum as a function of wavelength or band index.

        This function visualizes the fitted noise estimate per spectral band, using either
        provided wavelength information or band indices. It can plot on an existing Matplotlib
        axis or create a new figure if none is supplied.

        Args:
            ax: Optional Matplotlib Axes object to plot on. If None, a new figure and axis 
                are created automatically.
            **kwargs: Additional keyword arguments passed to `ax.plot()` (e.g., color, linestyle, label).

        Returns:
            fig: The Matplotlib Figure object containing the plot.
            ax: The Matplotlib Axes object used for plotting.
        """

        if ax is None:
            fig,ax = plt.subplots(1,1,figsize=(10,5))
        assert self.estimate is not None, "Noise is not fitted yet"
        if self.wavelengths is None:
            wav = np.arange( len(self.estimate) )
        else:
            wav = self.wavelengths
        assert len(wav) == len(self.estimate), "Fitted noise does not match wavelengths? Weird."

        ax.plot(wav, self.estimate, **kwargs)
        return ax.get_figure(), ax
    
    def transform(self, X):
        """Apply whitening to given numpy array."""
        shape = X.shape
        X_flat = X.reshape(-1, shape[-1])
        X_white = X_flat @ self.Wn_.T
        return X_white.reshape(shape)

    def inverse_transform(self, X_white):
        """Inverse noise whitening."""
        shape = X_white.shape
        X_flat = X_white.reshape(-1, shape[-1])
        X_orig = X_flat @ np.linalg.pinv(self.Wn_).T
        return X_orig.reshape(shape)

class MNF(BaseEstimator, TransformerMixin):
    """
    A flexible wrapper for scikit-learn's PCA implementation that (1) allows the integration of a noise whitener to
    perform MNF transforms, and (2) works with either HyData instances or numpy arrays.
    """

    def __init__(self, n_components=None, normalise=False, subsample=5, noise=None):
        """
        Initialize a Principal Component Analysis (PCA) or Minimum Noise Fraction (MNF) transformer.

        This class sets up a dimensionality reduction model that can operate in either PCA or MNF mode,
        depending on whether a noise estimator is provided. MNF uses noise whitening to maximize the
        signal-to-noise ratio across components, while PCA identifies directions of maximum variance.

        Reference:
            Green et al. (1988), "Transformation for ordering multispectral data in terms of image quality
            with implications for noise removal," *IEEE Transactions on Geoscience and Remote Sensing*, 26(1), 65–74.

        Args:
            n_components: Integer or None specifying the number of components to retain. 
                        If None, all components are preserved.
            normalise: Boolean flag indicating whether to normalize PCA components to unit variance. 
                    Default is False.
            subsample: Integer subsampling factor applied to input data to accelerate model fitting. 
                    Only every `subsample`-th sample is used. Default is 5.
            noise: Optional NoiseWhitener instance. If provided, an MNF transform is performed; 
                otherwise, a standard PCA is used.

        Attributes:
            _pca: The fitted sklearn.decomposition.PCA model or None if not yet fitted.
            n_components: The number of components to retain after transformation.
            normalise: Whether PCA components are normalized to unit variance.
            subsample: The subsampling factor used during fitting.
            noise: The NoiseWhitener instance used for MNF, or None for PCA mode.
        """
        self.n_components = n_components # number of components to keep after transform.
        self.normalise = normalise # if True, PCA components will be normalised to have a variance of 1.
        self.subsample = subsample # subsampling factor. Default is 5.
        self._pca = None # store the results here
        self.noise = noise # a NoiseWhitener instance. If passed, an MNF will be performed. If not, a PCA.
        self.wavelengths = None # wavelengths will be stored here if used with a HyData instance

    def fit(self, X, y=None):
        """Fit PCA on flattened spectral data."""
        if isinstance(X, hylite.HyData):
            self.wavelengths = X.get_wavelengths()
            X = X.data
        shape = X.shape
        X_flat = X.reshape(-1, shape[-1])
        X_flat = X_flat[::self.subsample, :]  # Subsample for fitting
        if self.noise is not None:
            X_flat = self.noise.transform( X_flat ) # transform before fitting PCA
        X_flat = X_flat[~np.isnan(X_flat).any(axis=1), :]  # Remove NaNs for fitting
        self._pca = sPCA(n_components=self.n_components, whiten=self.normalise)
        self._pca.fit(X_flat)
        return self

    def transform(self, X):
        """Apply PCA band-space transformation, returning a HyData instance or numpy array matching X."""
        out = None
        if isinstance(X, hylite.HyData):
            out = X.copy(data=False)
            X = X.data

        shape = X.shape
        X_flat = X.reshape(-1, shape[-1])
        nan_mask = np.isnan(X_flat).any(axis=1) # Flag NaNs
        if self.noise is not None:
            X_flat = self.noise.transform( X_flat ) # transform before applying PCA
        Xt = self._pca.transform( np.nan_to_num( X_flat) ) # get transformed X
        Xt[nan_mask, :] = np.nan  # Restore NaNs
        Xt = Xt.reshape(shape[:-1] + (-1,)) # reshape
        if out is not None:
            out.data = Xt # set data
            out.set_wavelengths( np.cumsum( self._pca.explained_variance_ratio_ ) ) # set cumulative explained variance as wavelengths
            return out # return HyData instance of same type as input
        return Xt # keep as numpy

    def inverse_transform(self, Xt):
        """Inverse MNF or PCA to reconstruct cube, returning a HyData instance or numpy array matching Xt."""
        out = None
        if isinstance(Xt, hylite.HyData):
            out = Xt.copy(data=False)
            Xt = Xt.data

        shape = Xt.shape
        X_flat = Xt.reshape(-1, shape[-1])
        nan_mask = np.isnan(X_flat).any(axis=1) # Flag NaNs
        X_recon = self._pca.inverse_transform( np.nan_to_num(X_flat) ) # undo PCA transform
        if self.noise is not None: # also undo noise whitening
            X_recon = self.noise.inverse_transform( X_recon )
        X_recon[nan_mask, :] = np.nan  # Restore NaNs
        X_recon = X_recon.reshape( shape[:-1] + (-1,))
        if out is not None:
            out.data = X_recon # set data
            if self.wavelengths is not None:
                out.set_wavelengths( self.wavelengths )
            else:
                out.set_wavelengths( np.arange( X_recon.shape[-1]))
            return out # return HyData instance of same type as input
        return X_recon # keep as numpy
    
class PCA( MNF ):
    """
    A class for performing PCA operations. This uses the MNF class above, but hides some of it's functionality for clarity.
    """ 
    def __init__(self, n_components=None, normalise=False, subsample=5):
        """
        Initialize a Principal Component Analysis (PCA) or Minimum Noise Fraction (MNF) transformer.

        This class sets up a dimensionality reduction model that can operate in either PCA or MNF mode.
        MNF uses a noise estimator to maximize signal-to-noise ratio across components, while PCA
        identifies directions of maximum variance without noise weighting.

        Args:
            n_components: int or None, optional
                Number of components to retain after the transform. If None, all components are kept.
            normalise: bool, default=False
                If True, PCA components are normalized to unit variance.
            subsample: int, default=5
                Subsampling factor applied to input data to speed up fitting. Only every `subsample`-th 
                sample is used.

        Attributes:
            _pca: sklearn.decomposition.PCA or None
                Placeholder for the fitted PCA model. Can be used to access the underlying sklearn object.
            n_components: int or None
                Number of components to retain (as specified in Args).
            normalise: bool
                Whether components are normalized to unit variance.
            subsample: int
                Subsampling factor used during fitting.
        """
        self.n_components = n_components # number of components to keep after transform.
        self.normalise = normalise # if True, PCA components will be normalised to have a variance of 1.
        self.subsample = subsample # subsampling factor. Default is 5.
        self._pca = None # store the results here
        self.noise = None # Noise whitener is kept as None for PCA analyses
        self.wavelengths = None # wavelengths will be stored here if used with a HyData instance