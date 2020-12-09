import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.signal import argrelmin
from tqdm import tqdm

class HyFeature(object):
    """
    Utility class for representing and fitting individual or multiple absorption features.
    """

    def __init__(self, name, pos, width, depth=1, data=None, color='g'):
        """
        Create a new feature:

        *Arguments*:
         - name = a name for this feature.
         - pos = the position of this feature (in nm).
         - width = the width of this feature (in nm).
         - data = a real spectra associated with this feature (e.g. for feature fitting or from reference libraries).
                  Should be a numpy array such that data[0,:] gives wavelength and data[1,:] gives reflectance.
        """

        self.name = name
        self.pos = pos
        self.width = width
        self.depth = depth
        self.color = color
        self.data = data
        self.mae = -1
        self.strength = -1
        self.components = None
        self.endmembers = None

    def get_start(self):
        """
        Get start of feature.

        Return:
         returns feature position - 0.5 * feature width.
        """

        return self.pos - self.width * 0.5

    def get_end(self):
        """
        Get approximate end of feature

        Return:
        returns feature position - 0.5 * feature width.
        """

        return self.pos + self.width * 0.5


    ######################
    ## Feature models
    ######################

    @classmethod
    def lorentzian(cls, x, pos, width, depth, offset=1.0):
        """
        Static function for evaluating a Lorentzian feature model

        *Arguments*:
         - x = wavelengths (nanometres) to evaluate the feature over
         - pos = the position of the features (nanometres)
         - depth = the depth of the feature (max to min)
         - width = parameter controlling the width of the feature.
                   Scaled such that pos - width / 2 -> pos + width / 2 contains ~99% of an equivalent
                   gaussian feature.
         - offset = the vertical offset of the feature (i.e. value where no absorption exists). Default is 1.0.
        """

        width = width / 6  # conversion so that width contains ~99% of (gaussian) feature
        return offset - (depth * width ** 2 / width) * width / ((x - pos) ** 2 + width ** 2)

    @classmethod
    def gaussian(cls, x, pos, width, depth, offset=1.0):
        """
        Static function for evaluating a Gaussian feature model

        *Arguments*:
         - x = wavelengths (nanometres) to evaluate the feature over
         - pos = the position of the features (nanometres)
         - depth = the depth of the feature (max to min)
         - width = parameter controlling the width of the feature (= standard deviation / 6).
                   Scaled such that pos - width / 2 -> pos + width / 2 contains ~99% of the feature.
         - offset = the vertical offset of the feature (i.e. value where no absorption exists). Default is 1.0.
        """

        width = width / 6  # conversion so that width contains ~99% of (gaussian) feature
        return offset - depth * np.exp(-(x - pos) ** 2 / (2 * width ** 2))

    @classmethod
    def multi_lorentz(cls, x, pos, width, depth, offset=1.0):
        """
        Static function for evaluating a multi-Lorentzian feature model

        *Arguments*:
         - x = wavelengths (nanometres) to evaluate the feature over
         - pos = a list of positions for each individual lorentzian function (nanometres)
         - depth = a list of depths for each individual lorentzian function (max to min)
         - width = a list of widths for each individual lorentzian function.
         - offset = the vertical offset of the functions. Default is 1.0.
        """

        y = np.zeros_like(x)
        for p, w, d in zip(pos, width, depth):
            y += cls.lorentzian(x, p, w, d, 0)
        return y + offset

    @classmethod
    def multi_gauss(cls, x, pos, width, depth, offset=1.0):
        """
        Static function for evaluating a multi-gaussian feature model

        *Arguments*:
         - x = wavelengths (nanometres) to evaluate the feature over
         - pos = a list of positions for each individual gaussian function (nanometres)
         - depth = a list of depths for each individual gaussian function (max to min)
         - width = a list of widths for each individual gaussian function.
         - offset = the vertical offset of the functions. Default is 1.0.
        """

        y = np.zeros_like(x)
        for p, w, d in zip(pos, width, depth):
            y += cls.gaussian(x, p, w, d, 0)
        return y + offset

    ############################
    ## Feature fitting
    ############################

    @classmethod
    def _lsq(cls, params, func, x, y, n=1):
        """
        Calculate error for least squares optimization
        """
        return np.nan_to_num( y - func(x, *params), nan=99999999999999,
                                                    posinf=99999999999999,
                                                    neginf=99999999999999 )

    @classmethod
    def _lsq_multi(cls, params, func, x, y, n):
        """
        Calculate error for least squares optimization of multi-gauss or multi-lorentz
        """
        return np.nan_to_num( y - func(x, params[0:n], params[n:(2*n)], params[(2*n):]), nan=99999999999999,
                                                              posinf=99999999999999,
                                                              neginf=99999999999999 )

    @classmethod
    def fit(cls, wav, refl, method='lorentz', n=1, vb=True, ftol=1e-4, order=3):
        """
        Fit a hyperspectral feature(s) to a (detrended) spectra.

        *Arguments*:
         - wav = the wavelength of the spectral subset to fit to.
         - refl = the reflectance spectra to fit to.
         - method = the spectra type to fit. Options are: 'minmax' (quick but rough), 'lorentz' or 'gauss'.
         - n = the number of features to fit. Default is 1.
         - verbose = True if a progress bar should be created when fitting to multiple spectra (as this can be slow).
         - ftol = the stopping criterion for the least squares optimization. Default is 1e-4.
         - order = the order of local minima detection. Default is 3. Smaller numbers return smaller local minima but are more
                   sensitive to noise.
        *Returns*: a HyData, or list of HyData instances (n>1) describing each features:
         - pos = the optimised feature position ( number or array depending on shape of refl)
         - width = the optimised feature width
         - depth = the optimised feature depth
         - strength = the feature strength (reduction in variance compared to no feature)
        """

        # get list of spectra and check it is the correct shape
        X = np.array(refl)
        if len(X.shape) == 1:
            vb = False
            X = X[None, :]

        assert len(X.shape) == 2, "Error - refl must be an Nxm array of N spectra over m wavelenghts."
        assert X.shape[1] == len(wav), "Error - inconsistent lengths; reflectance data must match provided wavelengths."
        assert np.isfinite(X).all(), "Error - input spectra contain nans"
        X /= max(1.0, np.max(X))  # ensure max of refl is 1

        # calculate initial guesses
        if n == 1:
            idx = np.argmin(X, axis=1)
            pos = wav[idx]
            depth = 1.0 - X[range(X.shape[0]), idx]
            width = 0.5 * (wav[-1] - wav[0])  # TODO; is there a better way to estimate width?
            X0 = np.vstack([pos, [width] * len(pos), depth, depth]).T
        else:
            idx, minima = argrelmin( X, axis=1, order=order ) # get all local minima
            width = 0.5 * (wav[-1] - wav[0])  # TODO; is there a better way to estimate width?
            midp =  int( len(wav) / 2 ) # index of middle of search domain (used as default for pos).
            X0 = np.zeros( (X.shape[0], 3*n + 1) )
            loop = range(X.shape[0])
            if vb: loop = tqdm(loop, desc="Extracting local minima", leave=False)
            for i in loop:

                # get indices of local minima
                _idx = minima[ idx==i ]
                if _idx.shape[0] == 0:
                    continue # no minima
                if _idx.shape[0] < n: # ensure we have an index for each feature we want to fit
                    _idx = np.hstack([_idx,[midp] * (n-len(_idx))])

                # get depths
                d = 1 - X[i, _idx]

                # too many features?
                if _idx.shape[0] > n:
                    srt = np.argsort(d)[::-1][0:n] # keep deepest features
                    d = d[ srt ]
                    _idx = _idx[ srt ]

                # sort by depth
                #srt = np.argsort(d)[::-1]
                #d = d[srt]
                #_idx = _idx[srt]

                # get position and build width prior
                p = wav[ _idx ]
                w = [ width ] * _idx.shape[0]

                # store
                X0[i] = np.hstack( [p,w,d,0] )

        # quick and dirty and done already!
        if 'minmax' in method.lower():
            out = X0

        else:  # loop through all spectra (sloooooooow!)
            X0 = X0[:,:-1] # drop last value (faked strengths) from X0
            # choose model and associated optimisation function
            if 'lorentz' in method.lower():
                if n == 1:
                    fmod = cls.lorentzian
                    lsq = cls._lsq
                else:
                    fmod = cls.multi_lorentz
                    lsq = cls._lsq_multi
            elif 'gauss' in method.lower():
                if n == 1:
                    fmod = cls.gaussian
                    lsq = cls._lsq
                else:
                    fmod = cls.multi_gauss
                    lsq = cls._lsq_multi
            else:
                assert False, "Error: %s is an unknown method" % method

            # calculate bounds constraints
            if n == 1:
                mn = [wav[0] - 1, (wav[1] - wav[0]) * 5, 0]  # min pos, width, depth
                mx = [wav[-1] + 1, (wav[-1] - wav[0]) * 2, 1]  # max pos, width, depth
            else:
                mn = np.array([wav[0] - 1] * n + [(wav[1] - wav[0]) * 5] * n + [0] * n) # min pos, width, depth
                mx = np.array([wav[-1] + 1] * n  + [(wav[-1] - wav[0]) * 2] * n + [1] * n ) # max pos, width, depth
            bnds = [mn,mx]

            x0 = X0[0, :]  # prior x0 for first feature, after this we test x0 from previous spectra
            out = np.zeros((X.shape[0], n*3 + 1)) # output array

            loop = range(X.shape[0])
            if vb:
                loop = tqdm(loop, desc="Fitting features", leave=False)
            for i in loop:
                # check if opt values from previous spectra are a better initial guess
                # (as the spectra are probably very similar!).
                #if np.sum(lsq(X0[i], fmod, wav, X[i],n)**2) < np.sum(lsq(x0, fmod, wav, X[i],n) ** 2):
                #    x0 = X0[i]
                x0 = X0[i]

                # check x0 is feasible
                if not ((x0 > bnds[0]).all() and (x0 < bnds[1]).all()):
                    continue # skip

                # do optimisation
                fit = least_squares(lsq, x0=x0, args=(fmod, wav, X[i], n), bounds=bnds, ftol=ftol)

                #if n > 1: # multi-feature - sort by depth
                    #idx = np.argsort( fit.x[2*n : 3*n] )[::-1]
                    #out[i, :] = (*fit.x[0:n][idx], *fit.x[n:(2 * n)][idx], *fit.x[2*n:3*n][idx],
                    #             max(0, np.std(1 - refl) - np.std(fit.fun)))
                #else:
                #    out[i,:] = (*fit.x, max(0, np.std(1 - refl) - np.std(fit.fun)))

                # store output
                out[i, :] = (*fit.x, max(0, np.std(1 - refl) - np.std(fit.fun)))
                #x0 = fit.x

        # resolve out into pos, width, depth and strength
        if out.shape[0] == 1: # run on a single spectra - return HyFeature instances
            if out.shape[1] == 4: # single feature
                feat = cls('est', out[0, 0], out[0, 1], out[0, 2], data=np.array([wav, X[0, :]]), color='r')
                feat.strength = out[0, 3]
                return feat
            else:
                feat = []
                for i in range(n):
                    feat.append(cls('est', out[0, 0+i], out[0, n+i], out[0, (2*n+i)], data=np.array([wav, X[0, :]]), color='r'))
                mf = MixedFeature('mix', feat, data=np.array([wav, X[0, :]]), color='r' )
                mf.strength = out[0,-1]
                return mf
        else:
            # resolve out into pos, width, depth and strength
            pos = out[:, 0:n]
            width = out[:, n:(n*2)]
            depth = out[:, (n*2):(n*3)]
            strength = out[:, -1]

            return pos, width, depth, strength

    # noinspection PyDefaultArgument
    def quick_plot(self, method='gauss', ax=None, label='top', lab_kwds={}, **kwds):
        """
        Quickly plot this feature.

        *Arguments*:
         - method = the method used to represent this feature. Options are:
                        - 'gauss' = represent using a gaussian function
                        - 'lorentz' = represent using a lorentzian function
                        - 'range' = draw vertical lines at pos - width / 2 and pos + width / 2.
                        - 'fill' = fill a rectangle in the region dominated by the feature with 'color' specifed in kwds.
                        - 'line' = plot a (vertical) line at the position of this feature.
                        - 'all' = plot with all of the above methods.
         - ax = an axis to add the plot to. If None (default) a new axis is created.
         - label = Label this feature (using it's name?). Options are None (no label), 'top', 'middle' or 'lower'. Or,
                   if an integer is passed, odd integers will be plotted as 'top' and even integers as 'lower'.
         - lab_kwds = Dictionary of keywords to pass to plt.text( ... ) for controlling labels.

        *Keywords*: Keywords are passed to ax.axvline(...) if method=='range' or ax.plot(...) otherwise.

        *Returns*:
         - fig = the figure that was plotted to
         - ax = the axis that was plotted to
        """

        if ax is None:
            fig, ax = plt.subplots()

        # plot reference spectra and get _x for plotting
        if self.data is not None:
            _x = self.data[0, : ]
            ax.plot(_x, self.data[1, :], color='k', **kwds)
        else:
            _x = np.linspace(self.pos - self.width, self.pos + self.width)

        # set color
        if 'c' in kwds:
            kwds['color'] = kwds['c']
            del kwds['c']
        kwds['color'] = kwds.get('color', self.color)

        # get _x for plotting
        if 'range' in method.lower() or 'all' in method.lower():
            ax.axvline(self.pos - self.width / 2, **kwds)
            ax.axvline(self.pos + self.width / 2, **kwds)
        if 'line' in method.lower() or 'all' in method.lower():
            ax.axvline(self.pos, color='k', alpha=0.4)
        if 'gauss' in method.lower() or 'all' in method.lower():
            if self.components is None: # plot single feature
                _y = HyFeature.gaussian(_x, self.pos, self.width, self.depth)
            else:
                _y = HyFeature.multi_gauss(_x, [c.pos for c in self.components],
                                               [c.width for c in self.components],
                                               [c.depth for c in self.components] )
            ax.plot(_x, _y, **kwds)
        if 'lorentz' in method.lower() or 'all' in method.lower():
            if self.components is None:  # plot single feature
                _y = HyFeature.lorentzian(_x, self.pos, self.width, self.depth)
            else:
                _y = HyFeature.multi_lorentz(_x, [c.pos for c in self.components],
                                                 [c.width for c in self.components],
                                                 [c.depth for c in self.components] )
            ax.plot(_x, _y, **kwds)
        if 'fill' in method.lower() or 'all' in method.lower():
            kwds['alpha'] = kwds.get('alpha', 0.25)
            ax.axvspan(self.pos - self.width / 2, self.pos + self.width / 2, **kwds)

        # label
        if not label is None:

            # calculate label position
            rnge = ax.get_ylim()[1] - ax.get_ylim()[0]
            if isinstance(label, int):
                if label % 2 == 0:
                    label = 'top'  # even
                else:
                    label = 'low'  # odd
            if 'top' in label.lower():
                _y = ax.get_ylim()[1] - 0.05 * rnge
                va = lab_kwds.get('va', 'top')
            elif 'mid' in label.lower():
                _y = ax.get_ylim()[0] + 0.5 * rnge
                va = lab_kwds.get('va', 'center')
            elif 'low' in label.lower():
                _y = ax.get_ylim()[0] + 0.05 * rnge
                va = lab_kwds.get('va', 'bottom')
            else:
                assert False, "Error - invalid label position '%s'" % label.lower()

            # plot label
            lab_kwds['rotation'] = lab_kwds.get('rotation', 90)
            lab_kwds['alpha'] = lab_kwds.get('alpha', 0.5)
            ha = lab_kwds.get('ha', 'center')
            if 'ha' in lab_kwds: del lab_kwds['ha']
            if 'va' in lab_kwds: del lab_kwds['va']
            lab_kwds['bbox'] = lab_kwds.get('bbox', dict(boxstyle="round",
                                                         ec=(0.2, 0.2, 0.2),
                                                         fc=(1., 1., 1.),
                                                         ))
            ax.text(self.pos, _y, self.name, va=va, ha=ha, **lab_kwds)

        return ax.get_figure(), ax

class MultiFeature(HyFeature):
    """
    A spectral feature with variable position due to a solid solution between known end-members.
    """

    def __init__(self, name, endmembers):
        """
        Create this multifeature from known end-members.

        *Arguments*:
         - endmembers = a list of HyFeature objects representing each end-member.
        """

        # init this feature so that it ~ covers all of its 'sub-features'
        minw = min([e.pos - e.width / 2 for e in endmembers])
        maxw = max([e.pos + e.width / 2 for e in endmembers])
        depth = np.mean([e.depth for e in endmembers])
        super().__init__(name, pos=(minw + maxw) / 2, width=maxw - minw, depth=depth, color=endmembers[0].color)

        # store endmemebers
        self.endmembers = endmembers

    def count(self):
        return len(self.endmembers)

    def quick_plot(self, method='fill+line', ax=None, suplabel=None, sublabel=('alternate', {}), **kwds):
        """
         Quickly plot this feature.

         *Arguments*:
          - method = the method used to represent this feature. Options are:
                         - 'gauss' = represent using a gaussian function at each endmember.
                         - 'lorentz' = represent using a lorentzian function at each endmember.
                         - 'range' = draw vertical lines at pos - width / 2 and pos + width / 2.
                         - 'fill' = fill a rectangle in the region dominated by the feature with 'color' specifed in kwds.
                         - 'line' = plot a (vertical) line at the position of each feature.
                         - 'all' = plot with all of the above methods.

                      default is 'fill+line'.

          - ax = an axis to add the plot to. If None (default) a new axis is created.
          - suplabel = Label positions for this feature. Default is None (no labels). Options are 'top', 'middle' or 'lower'.
          - sublabel = Label positions for endmembers. Options are None (no labels), 'top', 'middle', 'lower' or 'alternate'. Or, if an integer
                    is passed then it will be used to initialise an alternating pattern (even = top, odd = lower).
          - lab_kwds = Dictionary of keywords to pass to plt.text( ... ) for controlling labels.

         *Keywords*: Keywords are passed to ax.axvline(...) if method=='range' or ax.plot(...) otherwise.

         *Returns*:
          - fig = the figure that was plotted to
          - ax = the axis that was plotted to
         """

        if ax is None:
            fig, ax = plt.subplots()

        # plot
        if 'range' in method.lower() or 'all' in method.lower():
            super().quick_plot(method='range', ax=ax, label=None, **kwds)
        if 'line' in method.lower() or 'all' in method.lower():
            for e in self.endmembers:  # plot line for each end-member
                e.quick_plot(method='line', ax=ax, label=None, **kwds)
        if 'gauss' in method.lower() or 'all' in method.lower():
            for e in self.endmembers:  # plot gaussian for each end-member
                e.quick_plot(method='gauss', ax=ax, label=None, **kwds)
                if isinstance(sublabel, int): sublabel += 1
        if 'lorentz' in method.lower() or 'all' in method.lower():
            for e in self.endmembers:  # plot lorentzian for each end-member
                e.quick_plot(method='lorentz', ax=ax, label=None, **kwds)
                if isinstance(sublabel, int): sublabel += 1
        if 'fill' in method.lower() or 'all' in method.lower():
            super().quick_plot(method='fill', ax=ax, label=None, **kwds)

        # and do labels
        if not suplabel is None:
            if not isinstance(suplabel, tuple): suplabel = (suplabel, {})
            super().quick_plot(method='label', ax=ax, label=suplabel[0], lab_kwds=suplabel[1])
        if not sublabel is None:
            if not isinstance(sublabel, tuple): sublabel = (sublabel, {})
            if isinstance(sublabel[0], str) and 'alt' in sublabel[0].lower():
                sublabel = (1, sublabel[1])  # alternate labelling
            for e in self.endmembers:
                e.quick_plot(method='label', ax=ax, label=sublabel[0], lab_kwds=sublabel[1])
                sublabel = (sublabel[0] + 1, sublabel[1])
        return ax.get_figure(), ax

class MixedFeature(HyFeature):
    """
    A spectral feature resulting from a mixture of known sub-features.
    """

    def __init__(self, name, components, **kwds):
        """
        Create this mixed features from known components.

        *Arguments*:
         - components = a list of HyFeature objects representing each end-member.
        *Keywords*:
         - keywords are passed to HyFeature.init()
        """

        # init this feature so that it ~ covers all of its 'sub-features'
        minw = min([e.pos - e.width / 2 for e in components])
        maxw = max([e.pos + e.width / 2 for e in components])
        depth = np.mean([e.depth for e in components])

        if not 'color' in kwds:
            kwds['color'] = components[0].color
        super().__init__(name, pos=(minw + maxw) / 2, width=maxw - minw, depth=depth, **kwds)

        # store components
        self.components = components

    def count(self):
        return len(self.components)