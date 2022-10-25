import os
import numpy as np
import matplotlib
from hylite.correct.detrend import get_hull_corrected
from gfit import initialise, gfit,evaluate
from hylite import HyCollection, HyCloud, HyImage, HyData

import matplotlib.pyplot as plt
import matplotlib as mpl

class MWL(HyCollection):
    """
    A convenient class for manipulating and storing minimum wavelength mapping results.
    """

    def __init__(self, name, root, header=None):
        super().__init__(name, root, header)
        self.ext = '.mwl'

    def bind(self, model, nfeatures, x, X, sym=False):
        """
        Band a new MWL mapping results. Essentially treat this as the constructor.

        Args:
            model: the underlying HyData instance containing feature parameters.
            nfeatures: the number of features stored in the underlying model.
            x: Wavelengths that the model should be evaluated at.
            X: the underlying data that the model was fitted to. Useful for plotting / debugging. Default is None.
            sym: True if symmetric features are stored (3-parameters). Default is False (4-parameters).
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

    def __getitem__(self, n):
        """
        Slice this MWL object to return specific features or feature parameters.

        Options are:
         self[n]: get the n'th feature (as a HyData instance). See self.sortByDepth(..) and self.sortByPos(..) to
                   change feature order.
         self[n,b]: return a numpy array containing a specific property of the n'th feature. b can be an index 0-3
                     for symmetric and 0-4 for asymmetric features, or string ('depth', 'pos', 'width', 'width2').
        """

        # return MWL as HyData instance
        if isinstance(n, int):
            assert n < self.n, "Error - MWL has only %d features (not %d)" % (self.n, n + 1)
            out = self.model.copy(data=False)
            out.data = self.model.data[...,
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
                return self.model.data[..., b::self.stride]
            else:
                assert n[0] < self.n, "Error - MWL has only %d features (not %d)" % (self.n, n[0] + 1)
                return self.model.data[..., n[0] * self.stride + b]

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

        Args:
            wmin: the lower bound of the wavelength range. Default is 0 (accept all positions)
            wmax: the upper bound of the wavelength range. Default is -1 (accept all positions)
        Returns:
            a MWL instance containing the deepest features within the range, or nans if no feature exists.
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

        Args:
            position: the 'ideal' feature position to compare with (e.g. 2200.0 for AlOH)
            valid_range: A tuple defining the minimum and maximum acceptible wavelengths, or None (default). Values outside
                         of the valid range will be set to nan.
            depth_cutoff: Features with depths below this value will be discared (and set to nan).
        Returns:
            a single HyData instance containing the closest minima.
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

        Args:
            wmin: the lower bound of the wavelength range.
            wmax: the upper bound of the wavelength range.
            depth_cutoff: the minimum depth required for a feature to count as existing.

        Returns:
            a numpy array  populated with False (no feature) or True (feature).
        """
        valid_pos = (self[:, 'pos'] > wmin) & (np.array(self[:, 'pos']) < wmax)
        valid_depth = (self[:, 'depth'] > depth_cutoff)
        return (valid_pos & valid_depth).any(axis=-1)

    def getHyFeature(self, idx, source ):
        """
        Return a HyFeature instance based on the specified point or pixel index. Useful for plotting fitted results
        vs actual spectra.

        Args:
            idx (int, tuple): the index of the point or pixel to be retrieved.
            source (HyData): the source dataset, for plotting original spectra. Default is None.

        Returns:
            a HyFeature instance containing the modelled minimum wavelength data at this point.
        """
        pass

    def evaluate(self):
        """
        Evaluate this model and return the result as a HyData instance.

        Returns:
            A HyData instance containing the estimated spectra based on the fitted features.
        """
        out = self.model.copy(data=False)
        out.data = 1. - evaluate(self.x, self.model.data, sym=self.sym)
        out.set_wavelengths(self.x)
        return out

    def residual(self, sum=False):
        """
        Evaluate and return the residuals to the fitted minimum wavelength model.

        Returns:
            A HyData instance containing the residuals in band 0.
        """
        out = self.model.copy(data=False)
        out.data = np.sum(np.abs(self.X.data - self.evaluate().data), axis=-1)[..., None]
        return out

    def classify(self, n, nf=2, step=1):
        """
        Identify clusters in feature position space to classify this MWL map.
        This uses the hierarchichal method scipy.cluster.hierarchy.fclusterdata.

        See this publication for more details: https://doi.org/10.3390/min11020136

        Args:
            n: the number of classes to use.
            nf: the number of feature positions to use. Default is 2. Must not exceed the number of features fitted.
            step: the step to subsample points (used to avoid really slow plotting). Computes on all points by default.
        Returns:
            A tuple containing:

            - labels =  a HyData instance (or numpy array if step > 1) containing integer class labels in band 0.
            - centroids = a list containing the index of each class centroid (in the dataset).
         """

        assert nf <= self.n, "Error - MWL map has only %d features (<%d)." % (self.n, nf)
        import scipy.cluster.hierarchy as shc
        X = self[:, 'pos'][..., 0:nf]
        idx = np.array(np.meshgrid(*[np.arange(i) for i in self.X.data.shape[:-1]])).T

        # generate mask that subsamples and removes nans
        mask = np.full(X.shape[:-1], False)
        mask.ravel()[::step] = True  # subsample
        mask = mask & np.isfinite(X).all(axis=-1)  # remove nans
        X = X[mask, :]
        idx = idx[mask, :]

        # do clustering and get class centroids (in MWL space)
        L = shc.fclusterdata(X, n, criterion='maxclust', method='ward')
        C = np.array([np.median(X[L == i, :], axis=0) for i in range(1, n + 1)])

        # get index of centroids
        Cn = np.array([idx[np.argmin(np.linalg.norm(X - c, axis=1)), :] for c in C])  # center indices

        # return results
        if step == 1:
            out = self.model.copy(data=False)
            out.data = np.full(self[:, 'pos'].shape[:-1] + (1,), np.nan)
            out.data[mask, 0] = L
        else:
            out = np.full( mask.shape, np.nan )
            out[mask] = L

        return out, Cn


    ####################################
    ## Plotting functions
    ####################################
    def plot_features(self, ax=None, **kwds):
        """
        Plot all features in this minimum wavelength model on a scatter plot.

        Args:
            ax: a different axes to plot this figure on. Default is None (creates a new axis).
            **kwds: Keywords can include:

                 - step = the step to subsample points (to avoid really slow plotting). Defaults to a step that gives 1000 points.
                 - n = the number of classes to use for classification (see self.classify()),
                     or a list of class ids as returned by classify.
                 - cmap = the colour map to use (string). Default is 'tab10'.
                 - point_size = the size of the points to plot.
                 - point_alpha = the transparency of the points to plot.
                 - legend = True if a legend should be plotted. Default is True.
        Returns:
            fig,ax = the figure that was plotted.
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

        # compute step
        step = int(kwds.get("step", np.prod(p.shape[:-1]) / 5000.))
        step = max(step, 1)  # step cannot be < 1

        # compute colours
        n = kwds.get('n', 5)
        if isinstance(n, int):
            L, _ = self.classify(n, step=step)  # self.classify(n, step=step)
            if step == 1:
                n = L.X()
            else:
                n = L

        c = mpl.cm.get_cmap(kwds.get('cmap', 'tab10'))(n.ravel() / np.nanmax(n))

        for i, f in enumerate(range(self.n)):
            if i >= len(symbols):
                i = -1
            ax.scatter(p[..., f].ravel()[0], d[..., f].ravel()[0], c='k', marker=symbols[i],
                       label='%s' % names[i], zorder=-1)  # plot single point for legend
            #print(c.shape, n.shape, n.ravel().shape, '   ', p[...,f].ravel().shape, p[...,f].ravel()[::step].shape)
            ax.scatter(p[..., f].ravel()[::step], d[..., f].ravel()[::step], c=c[::step],
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

        Args:
            f1: the index of the first feature to plot (sorted by depth). Default is 0 (deepest feature).
            f2: the index of the first feature to plot (sorted by depth). Default is 1.
            ax: a different axes to plot this figure on. Default is None (creates a new axis).
            **kwds: Keywords can include:

                - step = the step to subsample points (to avoid really slow plotting). Defaults to a step that gives 1000 points.
                - n = the number of classes to use for classification (see self.classify()), or a list of class ids as returned by classify.
                - cmap = the colour map to use (string). Default is 'tab10'.
                - point_size = the size of the points to plot.
                - point_alpha = the transparency of the points to plot.
                - legend = True if a legend should be plotted. Default is True.
        Returns:
            fig,ax = the figure that was plotted.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))
        fig = ax.get_figure()

        ax.set_title('Spectral feature summary')

        # get feature information
        self.sortByDepth()
        p = self[:, 'pos']

        # compute step
        step = int(kwds.get("step", np.prod(p.shape[:-1]) / 5000.))
        step = max(step, 1)  # step cannot be < 1

        # compute colours
        n = kwds.get('n', 5)
        if isinstance(n, int):
            L, _ = self.classify(n, step=step)  # self.classify(n, step=step)
            if step == 1:
                n = L.X()
            else:
                n = L

        c = mpl.cm.get_cmap(kwds.get('cmap', 'tab10'))(n.ravel() / np.nanmax(n))

        # draw biplot
        ax.scatter(p[..., 0].ravel()[::step], p[..., 1].ravel()[::step], c=c[::step], lw=0, s=kwds.get("point_size", 20),
                   alpha=kwds.get("point_alpha", 0.5))  # plot

        ax.set_title("Deepest feature biplot")
        ax.set_xlabel("Deepest feature position (nm)")
        ax.set_ylabel("Secondary feature position (nm)")

        return fig, ax

    def quick_plot(self, **kwds):
        """
        Plot an overview visualisation that is useful for QAQC of MWL mapping results.

        Args:
            **kwds: Keywords can include:

                 - image = the image preview to plot. Can be a HyData instance, or 'resid' (default) to plots the average residuals,
                           or 'class' to a classification [slow!].
                 - bands = the bands of image to plot (if image is provided).
                 - vmin = the vmin value for plotting 'image' (if image is provided).
                 - vmax = the vmax value for plotting 'image' (if image is provided).
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
        Returns:
            fig,ax = the plot that was created.
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

        # compute step
        step = int(kwds.get("step", np.prod(self.model.data.shape[:-1]) / 5000.))
        step = max(step, 1)  # step cannot be < 1
        if isinstance(kwds.get('image','resid'), str) and 'class' in kwds.get('image', 'resid'):
            step = 1 # step must be one if we want to plot the classification
        kwds["step"] = step  # store for later functions

        # do clustering
        n = kwds.get('n', 5)
        if isinstance(n, int):
            if step == 1: # compute full clustering... (slow)
                if E.X().shape[0] > 5000: # warn about speed
                    print("Computing clusters... this can be slow.")
                L, Cn = self.classify(n, kwds.get('nf', 2))
                kwds['n'] = L.X().ravel()
            else:
                L, Cn = self.classify(n, kwds.get('nf', 2), step=step)
                kwds['n'] = L.ravel()

        # get colormap
        cmap = mpl.cm.get_cmap(kwds.get('cmap', 'tab10'))

        # plot overview image
        ax1.set_xticks([])
        ax1.set_yticks([])
        image = kwds.get('image', 'resid')
        if isinstance(image, HyData): # image has been provided
            bands = kwds.get('bands', (0,1,2))
            vmin = kwds.get('vmin', 0)
            vmax = kwds.get('vmax', 99)
            if isinstance(image, HyImage):
                image.quick_plot(bands, ax=ax1, vmin=vmin, vmax=vmax)  # plot
            elif isinstance(image, HyCloud):
                image.quick_plot(bands, cam=kwds.get('cam', 'ortho'), s=kwds.get('s', 1),
                             ax=ax1, vmin=vmin, vmax=vmax)
        elif 'class' in image:  # plot classification
            assert step == 1, "Error - to plot classification, step must = 1."
            if isinstance(self.model, HyImage):
                L.quick_plot(0, cmap=cmap, ax=ax1, vmin=0, vmax=np.nanmax(kwds['n']))  # plot
            elif isinstance(self.model, HyCloud):
                L.quick_plot(0, cam=kwds.get('cam', 'ortho'), s=kwds.get('s', 1), cmap=cmap,
                             ax=ax1, vmin=0, vmax=np.nanmax(kwds['n']))
            ax1.set_title("Classification")
        elif 'resid' in image:  # plot residuals
            if isinstance(self.model, HyImage):
                E.quick_plot(0, cmap='gray', vmin=mn, vmax=mx, ax=ax1)
            elif isinstance(self.model, HyCloud):
                E.quick_plot(0, cam=kwds.get('cam', 'ortho'), s=kwds.get('s', 1),
                             cmap='gray', vmin=mn, vmax=mx, ax=ax1)
            ax1.set_title("Residuals")
        else:
            assert False, 'Error = %s is an unknown image type.' % image

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
            c = cmap((i + 1) / np.nanmax(kwds['n']))

            # plot spectra on main image?
            if isinstance(self.model, HyImage):
                ax1.scatter(idx[0], idx[1], color=c, marker='o', edgecolors='k', lw=1)

            # stack spectra and plot
            y = self.X.data[ tuple(idx) ]
            ax3b.plot(self.x, y+offs, color=c, lw=1.7, label=n, alpha=0.8)
            y = mm.data[ tuple(idx)]
            ax3b.plot(self.x, y + offs, color=c, ls='--', alpha=0.8)

            offs += kwds.get('offset', 0.25)

        ax3b.set_title("Cluster centroids")
        ax3b.set_xlabel("Wavelength (nm)")
        ax3b.set_ylabel("Hull-corrected reflectance")

        return fig, [ax1, ax2a, ax2b, ax3a, ax3b]

    def plot_spectra(self, indices, ax=None, **kwds):
        """
        Plot the fitted features (and underlying data) for the specified indices.

        Args:
            indices: a list containing the indices to plot.
            ax: a matplotlib axes to plot to, or None (default) to create a new one.
            **kwds: Keywords can include:

                 - offset = the vertical offset between successive spectra. Default is 0.25.
                 - leg = True if a legend should be plotted. Default is True.
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

        if kwds.get('leg', True):
            ax.legend()
        ax.set_xlabel("Wavelength (nm)")
        return fig, ax

def minimum_wavelength(data, minw, maxw, method='gaussian', trend='hull', n=1,
                       sym=False, minima=True, k=4, nthreads=1, vb=True, **kwds):
    """
    Perform minimum wavelength mapping to map the position of absorbtion features.

    Args:
        data: the hyperspectral data (e.g. image or point cloud) to perform minimum wavelength mapping on.
        minw: the lower limit of the range of wavelengths to search (in nanometers).
        maxw: the upper limit of the range of wavelengths to search (in nanometers).
        method: the method/model used to quantify the feature. Options are:

         - "minmax" - Identifies the n most prominent local minima to approximately resolve absorbtion feature positions. Fast but inaccurate.
         - "poly" - Applies the minmax method but then interpolates feature position between bands using a polynomial function. TODO.
         - "gaussian" - fits n gaussian absorbtion features to the detrended spectra. Slow but most accurate.

        trend: the method used to detrend the spectra. Can be 'hull' or None. Default is 'hull'.
        n: the number of features to fit. Note that this is not compatible with method = 'tpt'.
        minima: True if features should fit minima. If False then 'maximum wavelength mapping' is performed.
        sym: True if symmetric gaussian fitting should be used. Default is False.
        k: the number of adjacent measurements to look at during detection of local minima. Default is 10. Larger numbers ignore smaller features as noise.
        nthreads: the number of threads to use for the computation. Default is 1 (no multithreading).
        vb: True if graphical progress updates should be created.
        **kwds: Keywords are passed to gfit.gfit( ... ).

    Returns:
        A MWL (n>1) or HyData instance containing the minimum wavelength mapping results.
    """

    # get relevant bands and detrend
    hc = data.export_bands((minw, maxw))
    if not minima: # flip if need be
        hc.data = np.nanmax(hc.data)-hc.data
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

    def __init__(self, minh, maxh, minc, maxc, mode='val', cmap='rainbow', **kwds):
        """
        Create an mwl_legend instance.

        Args:
            minh: the value (wavelength) mapped to hue of 0
            maxh: the value (wavelength) mapped to a hue of 1
            minc: the value (typically depth/strength) mapped to brightness/saturation of 0.
            maxc: the value (typically depth/strength) mapped to brightness/saturation fo 1.
            mode: specifies if minc and maxc refer to brightness ('val', default) or saturation ('sat').
            cmap: the colormapping to use to determine hue.
            **kwds: Keywords can include:

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
        self.cmap = cmap

    def plot(self, ax, pos='top left', s=(0.3, 0.2)):
        """
        Add this legend to the specified figure.

        Args:
            ax: the axes to overlay the legend on.
            pos: the position of the legend. Can be:

                - a string: 'top left' (default), 'bottom left', 'top right' or 'bottom right', or;
                - a tuple (x,y) coordinates of the top left of the legend in figure coordinates.
            s: the (width, height) of the legend in figure coordinates. Default is (0.2, 0.1).
        Returns:
            the axis that was added to the plot.
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
        if 'swir' in str(self.cmap):  # again - this is dirty dirty hack...
            extent = (2150.0, 2380.0, self.minc, self.maxc)

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
        elif 'rainbow' in self.cmap:
            extent = (self.minh, self.maxh, self.minc, self.maxc)
            x = np.linspace(0, 1, num=int(2000 * s[0]))
        else:
            extent = (self.minh, self.maxh, self.minc, self.maxc)
            x = np.linspace(0, 1, num=int(2000 * s[0]))
            cm = mpl.cm.get_cmap(self.cmap)
            rgba = cm(x)  # compute color from cmap
            x = mpl.colors.rgb_to_hsv(rgba[:, :3])[:, 0]  # update hue value accordingly

        # build image grid
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


def colourise_mwl(mwl, mode='p-d', cmap='rainbow', **kwds):
    """
    Takes a HyData instance containing minimum wavelength bands (pos, depth, width and strength) and creates
    a RGB composite such that hue ~ pos, sat ~ width and val ~ strength or depth.

    Args:
        mwl: the HyData instance containing minimum wavelength data.
        mode: the mapping from position (p), width (w) and depth and (d) to hsv. Default is 'pwd', though other options
              are 'p-d' (constant saturation of 80%), 'pdw' and 'pd-' (constant brightness of 85%).
        cmap: the colour mapping to use. Default is to map position to hue ('rainbow'), but any matplotlib
              colormap string can be provided here. Note that colours output by this colormap will be scaled in brightness
              and saturation according to the mode argument. Alternatively use 'swir' for customised
              rainbow-like colour stretch optimised for clay, mica and chlorite absorbtion features in the SWIR.
        **kwds: Keywords can include:

              - hue_map = Wavelengths (xxx.x, yyy.y) or percentiles (x,y) to use when converting wavelength to hue.
                            Default is (0,100).
              - sat_map = Widths (xxx.x, yyy.y) or percentiles (x,y) to use when converting width to saturation.
                            Default is (0,100).
              - val_map = Strengths/depths (xxx.x, yyy.y) or percentiles (x,y) to use when converting strength to brightness.
                            Default is (0,75), as these tend to be dominated by low/zero values.

    Returns:
        A tuple containing:

        - either an RGB HyImage object (if mwl is an image) or the original HyCloud with defined rgb bands.
        - cmap = a mwl_legend instance for plotting colour maps.
    """

    # extract data
    assert (mwl.band_count() == 4) or (
                mwl.band_count() == 3), "Error - HyData instance does not contain minimum wavelength data?"

    h = mwl.X()[..., 1].copy()  # pos
    s = mwl.X()[..., 2].copy()  # width
    v = mwl.X()[..., 0].copy()  # depth

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

    # calculate hue value from colormap
    if 'rainbow' in cmap.lower():
        h = h  # no change to hue
    elif 'swir' in cmap.lower():  # use SWIR customised colormap
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
    else:  # use matplotlib colormap
        cm = mpl.cm.get_cmap(cmap)
        rgba = cm(h)
        h = mpl.colors.rgb_to_hsv(rgba[:, :3])[:, 0]  # update hue value accordingly

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
    rgb[mask, :] = np.nan

    # create a colourbar object
    if 'pdw' in mode.lower() or 'pd-' in mode.lower():
        cbar = mwl_legend(ranges[0][0], ranges[0][1],
                          ranges[2][0], ranges[2][1], mode='sat', cmap=cmap)
    else:
        cbar = mwl_legend(ranges[0][0], ranges[0][1],
                          ranges[2][0], ranges[2][1], mode='val', cmap=cmap)
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


def plot_ternary(F1, F2, F3, bounds, weights=[1., 1., 1.], subsample=1, depth_thresh=0.01, ax=None, **kwds):
    """
    Plot a ternary diagram comparing the depths and positions of three minimum wavelength features.

    Args:
        F1 (HyData): A HyData instance containing feature depth and position as band 0 and 1 respectively.
        F2 (HyData): A HyData instance containing feature depth and position as band 0 and 1 respectively.
        F3 (HyData): A HyData instance containing feature depth and position as band 0 and 1 respectively.
        bounds: List containing min and max position values for each feature [ (min,max), (min,max), (min,max)].
        weights: weights applied to each depth. Default is [1,1,1].
        subsample: skip entries in the data features for large datasets. Default is 1 (don't skip any).
        depth_thresh: the minimum depth to be considered a valid feature. Default is 0.01.
        ax: an axes to plot to. If None (default) then a new axes is created.
        **kwds: Keywords can include:

             - r = the diameter of the ternary diagram (default is 0.8)
             - figsize = the figure size as a (width,height) tuple. Default is (10,10)
             - w = the width of edge plots. Default is 0.2.
             - gs = the number of grid lines. Default is 5.
             - invalid = invalid region (don't plot position if abundance < this value). Default is 0.2.
             - title = the title of the plot.
             - labels = the names of each of the features (F1, F2 and F3) used for labelling.
             - label_offset = the space between each label and the relevant vertex.
             - s = the point size. Default is 4.
             - a = point alpha. Default is 0.1.
    """
    # put features in a list
    features = [F1, F2, F3]

    # get axes
    if ax is None:
        fig, ax = plt.subplots(figsize=kwds.get('figsize', (10, 10)))
    else:
        fig = ax.get_figure()
    ax.set_aspect('equal')

    # compute coordinates of border
    r = kwds.get('r', 0.8)  # diameter of ternary diagram
    w = kwds.get('w', 0.2)  # width of edge plots
    gs = kwds.get('gs', 5)  # number of grid points
    invalid = kwds.get('invalid', 0.2)  # invalid region (don't plot position if abundance < this value)
    title = kwds.get('title', 'Absorption features')
    label_offset = kwds.get('label_offset', 0.2)
    labels = kwds.get('labels', ['F1', 'F2', 'F3'])
    edge_bounds = bounds
    colors = ['r', 'g', 'b']  # colours for the labels. These should not be changed.
    psize = kwds.get('s', 4)
    palpha = kwds.get('a', 0.1)

    # plot triangle and outline of ternary diagram
    X = np.array([[0, np.cos(np.deg2rad(30)), -np.cos(np.deg2rad(30))],
                  [1, -np.sin(np.deg2rad(30)), -np.sin(np.deg2rad(30))]]).T
    X *= r  # apply scale factor

    def baryToCC(a, b, c):
        # normalise
        sm = a + b + c
        for v in [a, b, c]:
            v /= sm

        # compute coordinates
        C, A, B = X
        x = A[0] * a + B[0] * b + C[0] * c
        y = A[1] * a + B[1] * b + C[1] * c
        return x, y

    def edgeToCC(edge, abundance, y):
        C, A, B = X
        if 'a' in edge.lower():
            xx = A - C
            o = C  # edge a starts at C and ends at A
        if 'b' in edge.lower():
            xx = B - A
            o = A  # edge b starts at A and ends at B
        if 'c' in edge.lower():
            xx = C - B
            o = B  # edge c starts at B and ends at C

        yy = xx[::-1] * np.array([-1, 1])  # y is perpendicular to x
        yy = yy * np.linalg.norm(yy) * w  # normalised to correct length

        x = o[0] + xx[0] * abundance + yy[0] * y
        y = o[1] + xx[1] * abundance + yy[1] * y
        return x, y

    # plot triangle
    ax.plot(X[[0, 1, 2, 0], 0], X[[0, 1, 2, 0], 1], color='k', lw=2, zorder=1)

    # plot ternary grid
    for e, c in zip('abc', 'gbr'):
        V = []
        ee = 'abc'.replace(e, '')
        for a in np.linspace(0, 1, gs):
            x0, y0 = edgeToCC(ee[0], a, 0)
            x1, y1 = edgeToCC(ee[1], 1 - a, 0)

            # plot edge
            ax.plot([x0, x1], [y0, y1], color=c, zorder=0, alpha=0.4, ls=':')

    # plot edges
    for e, c in zip('abc', 'rgb'):
        ax.plot(*edgeToCC(e, np.array([0, 0, 1, 1]), np.array([0, 1, 1, 0])), color=c, lw=2)

    # plot edge grid
    for e, c in zip('abc', 'rgb'):
        for a in np.linspace(0, 1, gs):
            ax.plot(*edgeToCC(e, np.array([0, 1]), np.array([a, a])), color='k', zorder=0, alpha=0.4, ls=':')
            ax.plot(*edgeToCC(e, np.array([a, a]), np.array([0, 1])), color=c, zorder=0, alpha=0.4, ls=':')

    # plot invalid areas
    for e in 'abc':
        ax.fill(*edgeToCC(e, np.array([0, 0, invalid, invalid]), np.array([0, 1, 1, 0])), color='k', zorder=-1, lw=0,
                alpha=0.1)

    # plot component labels
    for i, e in enumerate('abc'):
        co = np.array(edgeToCC(e, 1, 0))
        a = np.zeros(3)
        a[i] = 1 - label_offset
        off = co - np.array(baryToCC(*a))
        ax.text(*(co + off), labels[i], fontsize='xx-large', ha='center', va='top', color=colors[i])

    # plot edge labels
    for i, (e, a) in enumerate(zip('abc', [-60, 0, 60])):
        mn = edgeToCC(e, 0.05, 0.15)
        ax.text(*mn, str(edge_bounds[i][0]), ha='center', va='center', rotation=a)
        mx = edgeToCC(e, 0.05, 0.85)
        ax.text(*mx, str(edge_bounds[i][1]), ha='center', va='center', rotation=a)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_axis_off()

    # plot data
    depth = np.nan_to_num(np.array([D.data[..., 0].ravel() for D in features]))
    depth *= np.array(weights)[:, None]
    depth /= np.sum(depth, axis=0)[None, :]  # normalise to sum to one
    pos = np.array([D.data[..., 1].ravel() for D in features])
    mask = (depth > depth_thresh).any(axis=0)  # don't plot points with only shallow features
    pos = pos[:, mask][:, ::subsample]
    depth = depth[:, mask][:, ::subsample]

    # compute colours
    rgb = (depth / np.sum(depth, axis=0)).T
    ax.scatter(*baryToCC(depth[0, :], depth[1, :], depth[2, :]),
               color=rgb, alpha=palpha, s=psize)

    for i, e in enumerate('abc'):
        y = pos[i, :] - edge_bounds[i][0]
        y = np.clip(y / (edge_bounds[i][1] - edge_bounds[i][0]), 0, 1)
        mask = depth[i, :] > invalid
        ax.scatter(*edgeToCC(e, depth[i, mask], y[mask]),
                   color=rgb[mask], alpha=palpha, s=psize)

    ax.set_title(title, size='xx-large')
    return fig, ax