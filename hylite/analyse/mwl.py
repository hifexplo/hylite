import os
import numpy as np
import matplotlib
from hylite.correct.detrend import get_hull_corrected
from gfit import initialise, gfit,evaluate
from hylite import HyCollection, HyCloud, HyImage

import matplotlib.pyplot as plt
import matplotlib as mpl

class MWL(HyCollection):
    """
    A convenient class for manipulating and storing minimum wavelength mapping results.
    """

    def bind(self, model, nfeatures, x, X, sym=False):
        """
        Band a new MWL mapping results. Essentially treat this as the constructor.

        *Arguments*:
         - model = the underlying HyData instance containing feature parameters.
         - nfeatures = the number of features stored in the underlying model.
         - x = Wavelengths that the model should be evaluated at.
         - X = the underlying data that the model was fitted to. Useful for plotting / debugging. Default is None.
         - sym = True if symmetric features are stored (3-parameters). Default is False (4-parameters).
        """

        # store attributes
        self.model = model  # store underlying feature model
        self.x = x # model domain
        self.X = X # fitted data
        self.n = nfeatures  # store number of features
        self.sym = sym # store symmetry
        if sym: # set stride
            self.stride = 3
        else:
            self.stride = 4

    def getAttributes(self):
        """
        Return a list of available attributes in this HyScene. We must override the HyCollection implementation to remove
        functions associated with HyScene.
        """
        return list(set(dir(self)) - set(dir(HyCollection)) - set(dir(MWL)) - set(['header', 'root', 'name']))

    def _getDirectory(self, root=None, name=None):
        """
        Return the directory files associated with the HyScene are stored in. We override this to change the file extension
        associated with HyScene objects.

         *Arguments*:
         - root = the directory to store this HyCollection in. Defaults to the root directory specified when
                  this HyCollection was initialised, but this can be overriden for e.g. saving in a new location.
         - name = the name to use for the HyCollection in the file dictionary. If None (default) then this instance's
                  name will be used, but this can be overriden for e.g. saving in a new location.
        """
        p = os.path.splitext( super()._getDirectory(root,name) )[0]
        return p + ".mwl"

    def __getitem__(self, n):
        """
        Slice this MWL object to return specific features or feature parameters.

        Options are:
         self[n] = get the n'th feature (as a HyData instance). See self.sortByDepth(..) and self.sortByPos(..) to
                   change feature order.
         self[n,b] = return a numpy array containing a specific property of the n'th feature. b can be an index 0-3
                     for symmetric and 0-4 for asymmetric features, or string ('depth', 'pos', 'width', 'width2').
        """

        # return MWL as HyData instance
        if isinstance(n, int):
            assert n < self.n, "Error - MWL has only %d features (not %d)" % (self.n, n + 1)
            out = self.model.copy(data=False)
            out.data = self.model[...,
                       n * self.stride:(n + 1) * self.stride]  # return bands associated with the features
            return out
        else:  # return slice of MWL data as numpy array.
            assert len(n) == 2, "Error - %s is an invalid key." % (n)
            #  parse parameter descriptor
            if isinstance(n[1], str):
                if 'depth' in n[1]:
                    b = 0
                if 'pos' in n[1]:
                    b = 1
                elif 'width2' in n[1]:
                    b = 3
                    assert not self.sym, "Error - symmetric mwl features have no width2."
                elif 'width' in n[1]:
                    b = 2
            elif isinstance(n[1], slice):
                b = n[1]  # no change needed
            else:
                assert isinstance(n[1], int), "Error - key should be integer"
                b = n[1]
                if self.sym:
                    assert b < 2, "Error - %d is an invalid (symmetric) mwl band index. Should be 0 - 2."
                else:
                    assert b < 3, "Error - %d is an invalid mwl band index. Should be 0 - 3."

            if isinstance(n[0], slice):  # we want to slice only depth or only
                assert not isinstance(b, slice), "Error - invalid slice key %s" % (n)
                return self.model[..., b::self.stride]
            else:
                assert n[0] < self.n, "Error - MWL has only %d features (not %d)" % (self.n, n[0] + 1)
                return self.model[..., n[0] * self.stride + b]

    def getFeature(self, n):
        """
        Return a HyData instance containing bands associated with th nth minimum wavelength feature.
        """
        return self[n]

    def sortByDepth(self):
        """
        Sort features such that they are stored/returned from deepest to shallowest.
        """
        # sort by depth
        depth = self.model.X()[:, ::self.stride]
        idx = np.argsort(-depth, axis=-1)

        # expand index
        idx_full = np.zeros((depth.shape[0], self.model.band_count()), dtype=int)
        for c in range(self.stride):
            for n in range(idx.shape[-1]):
                idx_full[:, n * self.stride + c] = idx[:, n] * self.stride + c

        # take values to rearranged form
        out = np.take_along_axis(self.model.X(), idx_full, axis=-1).reshape(self.model.data.shape)
        self.model.set_raveled(out)

    def sortByPos(self):
        """
        Sort features such that they are stored/returned from lowest to highest wavelength.
        """
        # sort by depth
        pos = self.model.X()[:, 1::self.stride]
        idx = np.argsort(pos, axis=-1)

        # expand index
        idx_full = np.zeros((pos.shape[0], self.model.band_count()), dtype=int)
        for c in range(self.stride):
            for n in range(idx.shape[-1]):
                idx_full[:, n * self.stride + c] = idx[:, n] * self.stride + c

        # take values to rearranged form
        out = np.take_along_axis(self.model.X(), idx_full, axis=-1).reshape(self.model.data.shape)
        self.model.set_raveled(out)

    def deepest(self, wmin=0, wmax=-1):
        """
        Returns a HyData instance containing the deepest feature within the specified range.

        *Arguments*:
         - wmin = the lower bound of the wavelength range. Default is 0 (accept all positions)
         - wmax = the upper bound of the wavelength range. Default is -1 (accept all positions)
        *Returns*:
         - a MWL instance containing the deepest features within the range, or nans if no feature exists.
        """

        if wmin==0:
            wmin = np.nanmin(self[:,'pos'])
        if wmax==-1:
            wmax = np.nanmax(self[:,'pos'])

        # get valid positions
        valid_pos = (self[:, 'pos'] > wmin) & (np.array(self[:, 'pos']) < wmax)

        # get depths and filter to only include valid positions
        depth = self[:, 'depth'].copy()
        depth[np.logical_not(valid_pos)] = 0
        depth = depth.reshape((-1, self.n))

        # find deepest feature and expand index
        idx = np.argmax(depth, axis=-1)
        idx_full = np.zeros((depth.shape[0], self.stride), dtype=int)
        for c in range(self.stride):
            idx_full[:, c] = idx * self.stride + c

        # return deepest feature
        out = self.model.copy(data=False)
        out.data = np.take_along_axis(self.model.X(), idx_full, axis=-1).reshape(
            self.model.data.shape[:-1] + (-1,)).copy()
        out.data[out.data[..., 0] == 0, :] = np.nan  # remove 0 depths
        out.data[np.logical_not(valid_pos.any(axis=-1)), :] = np.nan  # remove invalid positions
        return out

    def closest(self, position, valid_range=None, depth_cutoff=0.05):
        """
        Returns a HyData instance containing the closest feature within the specified range.

        *Arguments*:
         - position = the 'ideal' feature position to compare with (e.g. 2200.0 for AlOH)
         - valid_range = A tuple defining the minimum and maximum acceptible wavelengths, or None (default). Values outside
                         of the valid range will be set to nan.
         - depth_cutoff = Features with depths below this value will be discared (and set to nan).
        *Returns*
         - a single HyData instance containing the closest minima.
        """
        # get valid positions
        valid_pos = np.isfinite(self[:, 'pos'])
        if valid_range is not None:
            valid_pos = (self[:, 'pos'] > valid_range[0]) & (np.array(self[:, 'pos']) < valid_range[1])

        # get deviations and filter to only include valid positions
        dp = self[:, 'pos'].copy() - position
        dp[np.logical_not(valid_pos)] = np.inf
        dp = dp.reshape((-1, self.n))

        # find closest feature and expand index
        idx = np.argmin(dp, axis=-1)
        idx_full = np.zeros((dp.shape[0], self.stride), dtype=int)
        for c in range(self.stride):
            idx_full[:, c] = idx * self.stride + c

        # return closest feature
        out = self.model.copy(data=False)
        out.data = np.take_along_axis(self.model.X(), idx_full, axis=-1).reshape(
            self.model.data.shape[:-1] + (-1,)).copy()
        out.data[out.data[..., 0] == 0, :] = np.nan  # remove 0 depths
        out.data[np.logical_not(valid_pos.any(axis=-1)), :] = np.nan  # remove invalid positions
        return out

    def feature_between(self, wmin, wmax, depth_cutoff=0.05):
        """
        Return True if the entries in each pixel/point include a feature within the specified wavelength range,
        and False otherwise. Useful for e.g. decision tree classifications.

        *Arguments*:
         - wmin = the lower bound of the wavelength range.
         - wmax = the upper bound of the wavelength range.
         - depth_cutoff = the minimum depth required for a feature to count as existing.

        *Returns*:
         - a numpy array  populated with False (no feature) or True (feature).
        """
        valid_pos = (self[:, 'pos'] > wmin) & (np.array(self[:, 'pos']) < wmax)
        valid_depth = (self[:, 'depth'] > depth_cutoff)
        return (valid_pos & valid_depth).any(axis=-1)

    def getHyFeature(self, idx, source ):
        """
        Return a HyFeature instance based on the specified point or pixel index. Useful for plotting fitted results
        vs actual spectra.

        *Arguments*:
         - idx = the index of the point or pixel to be retrieved.
         - source = the source dataset, for plotting original spectra. Default is None.
        *Returns*: a HyFeature instance containing the modelled minimum wavelength data at this point.
        """
        pass

    def evaluate(self):
        """
        Evaluate this model and return the result as a HyData instance.

        *Returns*
         - A HyData instance containing the estimated spectra based on the fitted features.
        """
        out = self.model.copy(data=False)
        out.data = 1. - evaluate(self.x, self.model.data)
        out.set_wavelengths(self.x)
        return out

    def residual(self, sum=False):
        """
        Evaluate and return the residuals to the fitted minimum wavelength model.

        *Returns*
         - A HyData instance containing the residuals in band 0.
        """
        out = self.model.copy(data=False)
        out.data = np.sum(np.abs(self.X.data - self.evaluate().data), axis=-1)[..., None]
        return out

    def classify(self, n, nf=2):
        """
        Identify clusters in feature position space to classify this MWL map.
        This uses the hierarchichal method scipy.cluster.hierarchy.fclusterdata.

        See this publication for more details: https://doi.org/10.3390/min11020136

        *Arguments*:
         - n = the number of classes to use.
         - nf = the number of feature positions to use. Default is 2. Must not exceed the number of features fitted.

        *Returns*:
         - labels =  a HyData instance containing integer class labels in band 0.
         - centroids = a list containing the index of each class centroid (in the dataset).
        """

        assert nf <= self.n, "Error - MWL map has only %d features (<%d)." % (self.n, nf)
        import scipy.cluster.hierarchy as shc
        X = self[:, 'pos'][..., 0:nf]

        # remove nans
        mask = np.isfinite(X).all(axis=-1)
        X = X[mask, :]

        L = shc.fclusterdata(X, n, criterion='maxclust', method='ward')  # class labels
        C = [np.median(X[L == i, :], axis=0) for i in range(1, n + 1)]  # class centroids
        Cn = [np.unravel_index(np.argmin(np.linalg.norm(X - c, axis=1)), self[:, 'pos'].shape[:-1]) for c in
              C]  # center pixels

        out = self.model.copy(data=False)
        out.data = np.full(self[:, 'pos'].shape[:-1] + (1,), np.nan)
        out.data[mask, 0] = L
        return out, Cn


    ####################################
    ## Plotting functions
    ####################################
    def plot_features(self, ax=None, **kwds):
        """
        Plot all features in this minimum wavelength model on a scatter plot.

        *Arguments*:
         - ax = a different axes to plot this figure on. Default is None (creates a new axis).
        *Keywords*
         - n = the number of classes to use for classification (see self.classify()),
             or a list of class ids as returned by classify.
         - cmap = the colour map to use (string). Default is 'tab10'.
         - point_size = the size of the points to plot.
         - point_alpha = the transparency of the points to plot.
         - legend = True if a legend should be plotted. Default is True.
        *Returns*:
         - fig,ax = the figure that was plotted.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))
        fig = ax.get_figure()

        ax.set_title('Spectral feature summary')

        # get data
        symbols = ['s', 'v', 'o', '.']
        names = ['Primary', 'Secondary', 'Tertiary', 'Other']
        p = self[:, 'pos']
        d = self[:, 'depth']

        # compute colours
        n = kwds.get('n', 5)
        if isinstance(n, int):
            L, _ = self.classify(n)
            n = L.X()
        c = mpl.cm.get_cmap(kwds.get('cmap', 'tab10'))(n.ravel() / np.nanmax(n))

        for i, f in enumerate(range(self.n)):
            if i >= len(symbols):
                i = -1
            ax.scatter(p[..., f].ravel()[0], d[..., f].ravel()[0], c='k', marker=symbols[i],
                       label='%s' % names[i], zorder=-1)  # plot single point for legend
            ax.scatter(p[..., f].ravel(), d[..., f].ravel(), c=c,
                       marker=symbols[i],
                       s=kwds.get("point_size", 20), alpha=kwds.get("point_alpha", 0.5), lw=0)

            ax.set_title("Spectral feature summary")
            ax.set_xlabel("Feature position (nm)")
            ax.set_ylabel("Hull-corrected depth")
        if kwds.get('legend', True):
            ax.legend()

        return fig, ax

    def biplot(self, f1=0, f2=1, ax=None, **kwds):
        """
        Plot all features in this minimum wavelength model on a scatter plot. Note that calling
        this function will sort the features in this MWL instance by depth.

        *Arguments*:
         - f1 = the index of the first feature to plot (sorted by depth). Default is 0 (deepest feature).
         - f2 = the index of the first feature to plot (sorted by depth). Default is 1.
         - ax = a different axes to plot this figure on. Default is None (creates a new axis).
        *Keywords*
         - n = the number of classes to use for classification (see self.classify()),
             or a list of class ids as returned by classify.
         - cmap = the colour map to use (string). Default is 'tab10'.
         - point_size = the size of the points to plot.
         - point_alpha = the transparency of the points to plot.
         - legend = True if a legend should be plotted. Default is True.
        *Returns*:
         - fig,ax = the figure that was plotted.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))
        fig = ax.get_figure()

        ax.set_title('Spectral feature summary')

        # get feature information
        self.sortByDepth()
        p = self[:, 'pos']

        # compute colours
        n = kwds.get('n', 5)
        if isinstance(n, int):
            L, _ = self.classify(n)
            n = L.X()
        c = mpl.cm.get_cmap(kwds.get('cmap', 'tab10'))(n.ravel() / np.nanmax(n))

        # draw biplot
        ax.scatter(p[..., 0].ravel(), p[..., 1].ravel(), c=c, lw=0, s=kwds.get("point_size", 20),
                   alpha=kwds.get("point_alpha", 0.5))  # plot

        ax.set_title("Deepest feature bi-plot")
        ax.set_xlabel("Deepest feature position (nm)")
        ax.set_ylabel("Secondary feature position (nm)")

        return fig, ax

    def quick_plot(self, **kwds):
        """
        Plot an overview visualisation that is useful for QAQC of MWL mapping results.

        *Keywords*:
         - mode = the plotting mode. 'class' (default) plots a classification. 'resid' plots the average residuals.
         - residual_clip = the (mn,mx) percentile colour stretch to apply to the residuals.
         - n = the number of clusters do create during absorbtion-feature clustering, or a list with precalculated
                    cluster labels.
         - nf = the number of features to use for this clustering. Default is 2.
         - offset = the vertical offset between spectra in the spectral summaries. Default is 0.25
         - cmap = the colour map to use (string). Default is 'tab10'.
         - point_size = the size of the points to plot.
         - point_alpha = the transparency of the points to plot.
         - legend = True if a legend should be plotted. Default is True.
         - cam = a camera instance if the underlying dataset is a pointcloud. Default is 'ortho'.
         - s = specify point size if the underlying dataset is a pointcloud. Default is 1.
        *Returns*:
         - fig,ax = the plot that was created.
        """

        # todo; write another function for single-feature maps?
        assert self.n >= 2, "Error - quick_plot(...) can only be used for MWL instances with > 2 features."

        fig = plt.figure(constrained_layout=True, figsize=kwds.get('figsize', (15, 10)))
        gs = fig.add_gridspec(3, 2)
        ax1 = fig.add_subplot(gs[0, :])
        ax2a = fig.add_subplot(gs[1, 0])
        ax2b = fig.add_subplot(gs[1, 1])
        ax3a = fig.add_subplot(gs[2, 0])
        ax3b = fig.add_subplot(gs[2, 1])

        # compute residuals
        E = self.residual()
        mn, mx = np.nanpercentile(E.data, kwds.get("residual_clip", (2, 98)))

        # compute clustering
        n = kwds.get('n', 5)
        if isinstance(n, int):  # compute clustering if need be
            L, Cn = self.classify(n, kwds.get('nf', 2))
            kwds['n'] = L.X().ravel()

        # get colormap
        cmap = mpl.cm.get_cmap(kwds.get('cmap', 'tab10'))

        # plot overview image
        ax1.set_xticks([])
        ax1.set_yticks([])
        if 'class' in kwds.get('mode', 'class') or 'biplot' in kwds.get('mode', 'class'):  # plot class
            if isinstance(self.model, HyImage):
                L.quick_plot(0, cmap=cmap, ax=ax1, vmin=0, vmax=np.nanmax(L.X()))  # plot
            elif isinstance(self.model, HyCloud):
                L.quick_plot(0, cam=kwds.get('cam', 'ortho'), s=kwds.get('s', 1), cmap=cmap,
                             ax=ax1, vmin=0, vmax=np.nanmax(L.X()))
            ax1.set_title("Classification")
        elif 'resid' in kwds.get('mode', 'resid'):  # plot residual
            if isinstance(self.model, HyImage):
                E.quick_plot(0, cmap='gray', vmin=mn, vmax=mx, ax=ax1)
            elif isinstance(self.model, HyCloud):
                E.quick_plot(0, cam=kwds.get('cam', 'ortho'), s=kwds.get('s', 1),
                             cmap='gray', vmin=mn, vmax=mx, ax=ax1)
            ax1.set_title("Residuals")
        else:
            assert False, 'Error = %s is an unknown plotting mode.' % kwds['mode']

        # position / depth plots
        self.plot_features(ax=ax2a, **kwds)

        # biplot
        self.biplot(ax=ax2b, **kwds)

        # biggest residual spectra
        ax3a.set_title('Spectral fitting quality')
        offs = 0
        mm = self.evaluate()
        for p, c, n in zip((50, 75, 95, 99), ['lightgreen', 'skyblue', 'orange', 'red'],
                           ['typical', 'dodgy', 'bad', 'worst']):
            t = np.nanpercentile(E.X(), p)
            idx = np.argmin(np.abs(np.nan_to_num(E.X()[:, 0] - t, nan=np.inf)))
            ax3a.plot(self.x, self.X.X()[idx, :] + offs, color=c, lw=2, label=n)
            ax3a.plot(self.x, mm.X()[idx, :] + offs, color=c, ls=':')
            offs += kwds.get('offset', 0.25)
        ax3a.legend()
        ax3a.set_xlabel("Wavelength (nm)")
        ax3a.set_ylabel("Hull-corrected reflectance")

        # plot cluster centroids
        offs = 0
        for i, idx in enumerate(Cn):
            c = cmap((i + 1) / np.nanmax(L.X()))
            if isinstance(self.model, HyImage):  # N.B doesn't work on clouds.
                ax1.scatter(idx[0], idx[1], color=c, marker='o', edgecolors='k', lw=1)

            # stack spectra and plot
            y = self.X.data[idx]
            ax3b.plot(self.x, y + offs, color=c, lw=1.7, label=n, alpha=0.8)
            y = mm.data[idx]
            ax3b.plot(self.x, y + offs, color=c, ls='--', alpha=0.8)

            offs += kwds.get('offset', 0.25)

        ax3b.set_title("Cluster centroids")
        ax3b.set_xlabel("Wavelength (nm)")
        ax3b.set_ylabel("Hull-corrected reflectance")

        return fig, [ax1, ax2a, ax2b, ax3a, ax3b]

    def plot_spectra(self, indices, ax=None, **kwds):
        """
        Plot the fitted features (and underlying data) for the specified indices.

        *Arguments*:
         - indices = a list containing the indices to plot.
         - ax = a matplotlib axes to plot to, or None (default) to create a new one.

        *Keywords*:
         - offset = the vertical offset between successive spectra. Default is 0.25.
        """

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))
        fig = ax.get_figure()
        ax.set_title('Fitted spectra')
        offs = 0

        for i, idx in enumerate(indices):

            # evaluate model and plot it
            if np.issubdtype(type(idx), int):
                y = 1 - evaluate(self.x, self.model[idx, :], self.sym)
            else:
                y = 1 - evaluate(self.x, self.model[idx[0], idx[1], :], self.sym)
            ax.plot(self.x, y + offs, color=plt.cm.tab10(i), ls='--', lw=2, label='Spectra %s' % (str(idx)))

            # plot data (if it is defined)
            try:
                if np.issubdtype(type(idx), int):
                    y = self.X[idx, :]
                else:
                    y = self.X[idx[0], idx[1], :]
                ax.plot(self.x, y + offs, color=plt.cm.tab10(i), lw=2, ls='-')
            except:
                continue

            offs += kwds.get('offset', 0.25)
        ax.legend()
        ax.set_xlabel("Wavelength (nm)")

        return fig, ax



def minimum_wavelength(data, minw, maxw, method='gaussian', trend='hull', n=1,
                       sym=False, minima=True, k=4, nthreads=1, vb=True, **kwds):
    """
    Perform minimum wavelength mapping to map the position of absorbtion features.

    *Arguments*:
     - data = the hyperspectral data (e.g. image or point cloud) to perform minimum wavelength mapping on.
     - minw = the lower limit of the range of wavelengths to search (in nanometers).
     - maxw = the upper limit of the range of wavelengths to search (in nanometers).
     - method = the method/model used to quantify the feature. Options are:
                 - "minmax" - Identifies the n most prominent local minima to approximately resolve absorbtion feature positions. Fast but inaccurate.
                 - "poly" - Applies the minmax method but then interpolates feature position between bands using a polynomial function. TODO.
                 - "gaussian" - fits n gaussian absorbtion features to the detrended spectra. Slow but most accurate.
     - trend = the method used to detrend the spectra. Can be 'hull' or None. Default is 'hull'.
     - n = the number of features to fit. Note that this is not compatible with method = 'tpt'.
     - minima = True if features should fit minima. If False then 'maximum wavelength mapping' is performed.
     - sym = True if symmetric gaussian fitting should be used. Default is False.
     - k = the number of adjacent measurements to look at during detection of local minima. Default is 10. Larger numbers ignore smaller features as noise.
     - nthreads = the number of threads to use for the computation. Default is 1 (no multithreading).
     - vb = True if graphical progress updates should be created.

    *Keywords*: Keywords are passed to gfit.gfit( ... ).
    *Returns*: A MWL (n>1) or HyData instance containing the minimum wavelength mapping results.
    """

    # get relevant bands and detrend
    hc = data.export_bands((minw, maxw))
    if not minima: # flip if need be
        hc.data = -hc.data
    if trend is not None:
        if 'hull' in trend:
            hc = get_hull_corrected(hc,vb=vb)
        else:
            assert False, "Error - %s is an unsupported detrending method." % trend

    # setup output
    if sym:
        out = np.full((np.prod(data.data.shape[:-1]), n * 3), np.nan)
    else:
        out = np.full((np.prod(data.data.shape[:-1]), n * 4), np.nan)

    # remove invalid y-values from input
    S = hc.X()
    mask = np.isfinite(S).all(axis=-1)  # drop nans
    mask = mask & (S != S[:, 0][:, None]).any(axis=1)  # drop flat spectra (e.g. all zeros)
    X = S[mask]

    if mask.any():  # if no valid spectra, skip to end

        # flip hull corrected spectra as gfit only fits maxima
        X = 1 - X

        # get initial values
        x = hc.get_wavelengths()
        x0 = initialise(x, X, n, sym=sym, d=k, nthreads=nthreads)  # get initial values [ minmax method ]
        if 'minmax' in method:
            out[mask, :] = x0  # just use x0 as output
        elif 'gauss' in method:
            out[mask, :] = gfit(x, X, x0, n, sym=sym, nthreads=nthreads, vb=vb, **kwds)  # fit gaussians
        elif 'poly' in method:
            assert False, "Error - polynomial mwl mapping is not yet implemented."  # todo
        else:
            assert False, "Error - %s is an unsupported fitting method." % method

    # reshape outputs and add to HyData instance
    mwld = data.copy(data=False)
    mwld.data = out.reshape(data.data.shape[:-1] + (out.shape[-1],))

    # setup mwl collection and return
    mwl = MWL('M', '')
    mwl.bind( mwld, n, x=hc.get_wavelengths(), sym=sym, X=hc )
    return mwl

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

def colourise_mwl(mwl, mode='p-d', **kwds):
    """
    Takes a HyData instance containing minimum wavelength bands (pos, depth, width and strength) and creates
    a RGB composite such that hue ~ pos, sat ~ width and val ~ strength or depth.

    *Arguments*:
     - mwl = the HyData instance containing minimum wavelength data.
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
    assert (mwl.band_count() == 4) or (mwl.band_count() == 3), "Error - HyData instance does not contain minimum wavelength data?"

    h = mwl.get_raveled()[..., 1].copy()  # pos
    s = mwl.get_raveled()[..., 2].copy()  # width
    v = mwl.get_raveled()[..., 0].copy()  # depth

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

    # add nans back in
    rgb[ mask, : ] = np.nan

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

