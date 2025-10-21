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
        Parameters
        ----------
        noise_estimate : np.ndarray, optional
            Optional noise array of shape (H, W, B). If None, noise is estimated from spatial differences.
        neighbor_axis : int
            Which spatial axis to use for differencing if noise is estimated (0=row, 1=column).
        subsample : int
            An integer factor to subsample the noise samples for covariance estimation. This can help
            reduce computation time and memory usage when dealing with large datasets.
        noiseMethod : str
            Direction to compute noise differences. 'spectral' for band-wise differences, 'spatial' for spatial differences.
            Note that 'spatial' can only be used for image data.
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
        Quick plot of the estimated noise spectrum per band.

        Plots the fitted noise estimate against the corresponding wavelengths (or band indices if 
        wavelengths are not provided). Can plot on an existing Matplotlib axis or create a new one.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            An existing matplotlib axes object to plot on. If None, a new figure and axes are created.
        **kwargs : dict
            Additional keyword arguments passed to `ax.plot()` (e.g., color, linestyle, label).

        Returns
        -------
        fig : matplotlib.figure.Figure
            The Matplotlib figure object containing the plot.
        ax : matplotlib.axes.Axes
            The axes object used for the plot.
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
        Initialize the MNF or PCA transformer.

        Parameters
        ----------
        n_components : int or None, optional
            The number of principal components to retain after the transform. If None, all components
            are kept.
        normalise : bool, default=False
            If True, the PCA components will be normalised to have unit variance.
        subsample : int, default=5
            Subsampling factor applied to the input data to speed up fitting. Only every `subsample`-th
            sample is used when computing the transform.
        noise : NoiseWhitener instance or None, optional
            If provided, a Minimum Noise Fraction (MNF) transform will be performed using this noise
            estimator. If None, a standard PCA is performed.

        Attributes
        ----------
        _pca : sklearn.decomposition.PCA or None
            Placeholder for the fitted PCA or MNF model.
        n_components : int or None
            Number of components to retain (as above).
        normalise : bool
            Whether components will be normalised (as above).
        subsample : int
            Subsampling factor (as above).
        noise : NoiseWhitener instance or None
            Noise estimator for MNF (as above).
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
        Initialize the MNF or PCA transformer.

        Parameters
        ----------
        n_components : int or None, optional
            The number of principal components to retain after the transform. If None, all components
            are kept.
        normalise : bool, default=False
            If True, the PCA components will be normalised to have unit variance.
        subsample : int, default=5
            Subsampling factor applied to the input data to speed up fitting. Only every `subsample`-th
            sample is used when computing the transform.
        Attributes
        ----------
        _pca : sklearn.decomposition.PCA or None
            Placeholder for the fitted PCA or MNF model.
        n_components : int or None
            Number of components to retain (as above).
        normalise : bool
            Whether components will be normalised (as above).
        subsample : int
            Subsampling factor (as above).
        noise : NoiseWhitener instance or None
            Noise estimator for MNF (as above).
        """
        self.n_components = n_components # number of components to keep after transform.
        self.normalise = normalise # if True, PCA components will be normalised to have a variance of 1.
        self.subsample = subsample # subsampling factor. Default is 5.
        self._pca = None # store the results here
        self.noise = None # Noise whitener is kept as None for PCA analyses
        self.wavelengths = None # wavelengths will be stored here if used with a HyData instance