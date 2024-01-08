"""
Fit and visualise individual hyperspectral features.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.signal import argrelmin
from tqdm import tqdm

from gfit import gfit, initialise, evaluate

class HyFeature(object):
    """
    Utility class for representing and fitting individual or multiple absorption features.
    """

    def __init__(self, name, pos, width, depth=1, data=None, color='g'):
        """
        Args:
            name (str) a name for this feature.
            pos (float): the position of this feature (in nm).
            width (float): the width of this feature (in nm).
            data (ndarray): a real spectra associated with this feature (e.g. for feature fitting or from reference libraries).
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

        Returns:
            the feature position - 0.5 * feature width.
        """

        return self.pos - self.width * 0.5

    def get_end(self):
        """
        Get approximate end of feature

        Returns:
        returns feature position - 0.5 * feature width.
        """

        return self.pos + self.width * 0.5


    ######################
    ## Feature models
    ######################
    @classmethod
    def gaussian(cls, _x, pos, width, depth):
        """
        Static function for evaluating a gaussian feature model

        Args:
            x (ndarray): wavelengths (nanometres) to evaluate the feature over
            pos (float): position for the gaussian function (nanometres)
            width (float): width for the gaussian function.
            depth (float): depth for the gaussian function (max to min)
            offset (float): the vertical offset of the functions. Default is 1.0.
        """
        return 1 - depth * np.exp( -(_x - pos)**2 / width )

    @classmethod
    def multi_gauss(cls, x, pos, width, depth, asym=None):
        """
        Static function for evaluating a multi-gaussian feature model

        Args:
            x (ndarray): wavelengths (nanometres) to evaluate the feature over
            pos (list): a list of positions for each individual gaussian function (nanometres)
            width (list): a list of widths for each individual gaussian function.
            depth (list): a list of depths for each individual gaussian function (max to min)
            asym (list): a list of feature asymmetries. The right-hand width will be calculated as:
                         w2 = asym * width. Default is 1.0.
        """
        if asym is None:
            asym = np.ones( len(width) )
        M = np.hstack( [[depth[i], pos[i], width[i], width[i]*asym[i]] for i in range(len(depth))] )
        y = evaluate( x, M, sym=False )
        return 1 - y

    # noinspection PyDefaultArgument
    def quick_plot(self, method='gauss', ax=None, label='top', lab_kwds={}, **kwds):
        """
        Quickly plot this feature.

        Args:
            method (str): the method used to represent this feature. Options are:

                        - 'gauss' = represent using a gaussian function
                        - 'range' = draw vertical lines at pos - width / 2 and pos + width / 2.
                        - 'fill' = fill a rectangle in the region dominated by the feature with 'color' specifed in kwds.
                        - 'line' = plot a (vertical) line at the position of this feature.
                        - 'all' = plot with all of the above methods.

            ax: an axis to add the plot to. If None (default) a new axis is created.
            label (float): Label this feature (using it's name?). Options are None (no label), 'top', 'middle' or 'lower'. Or,
                   if an integer is passed, odd integers will be plotted as 'top' and even integers as 'lower'.
            lab_kwds (dict): Dictionary of keywords to pass to plt.text( ... ) for controlling labels.
            **kwds: Keywords are passed to ax.axvline(...) if method=='range' or ax.plot(...) otherwise.

        Returns:
            Tuple containing

            - fig: the figure that was plotted to.
            - ax: the axis that was plotted to.
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
        Args:
            endmembers (list): a list of HyFeature objects representing each end-member.
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

         Args:
            method (str): the method used to represent this feature. Default is 'fill+line'. Options are:

                         - 'gauss' = represent using a gaussian function at each endmember.
                         - 'range' = draw vertical lines at pos - width / 2 and pos + width / 2.
                         - 'fill' = fill a rectangle in the region dominated by the feature with 'color' specifed in kwds.
                         - 'line' = plot a (vertical) line at the position of each feature.
                         - 'all' = plot with all of the above methods.

            ax: an axis to add the plot to. If None (default) a new axis is created.
            suplabel (str): Label positions for this feature. Default is None (no labels). Options are 'top', 'middle' or 'lower'.
            sublabel (str): Label positions for endmembers. Options are None (no labels), 'top', 'middle', 'lower' or 'alternate'. Or, if an integer
                    is passed then it will be used to initialise an alternating pattern (even = top, odd = lower).
            lab_kwds (dict): Dictionary of keywords to pass to plt.text( ... ) for controlling labels.
            **kwds: Keywords are passed to ax.axvline(...) if method=='range' or ax.plot(...) otherwise.

         Returns:
            Tuple containing

            - fig: the figure that was plotted to
            - ax: the axis that was plotted to
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
        Args:
            components: a list of HyFeature objects representing each end-member.
            **kwds: keywords are passed to HyFeature.init()
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