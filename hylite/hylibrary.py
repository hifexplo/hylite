from hylite.hydata import HyData
from hylite.hyfeature import HyFeature, MultiFeature, MixedFeature
import hylite.reference.features as ref
import numpy as np
import matplotlib.pyplot as plt

class HyLibrary(HyData):
    """
    A class for loading and managing spectral libraries and associated metadata.
    """

    def __init__(self, lab, refl, upper=None, lower=None,
                 wav=None, pos=None, header=None):

        """
        Create a hyperspectral library.

        *Arguments*:
         - lab = list of sample labels (one label per sample).
         - refl = 2D numpy array containing (mean) spectral reflectance measurements for each sample such that:
                    refl[ sample_index, : ] = [ refl1, refl2, refl3, .... ].
         - upper = a 2D numpy array containing upper error bound of spectra. Default is None.
         - lower = a 2D numpy array containing lower error bound of spectra. Default is None.
         - wav = list of wavelengths in the spectra for each sample. If None this must be defined in the header file passed.
         - pos = a (n x 3) numpy array of positions for each sample. Default is None.
         - header = a io.HyHeader instance containing additional metadata to associate with this library.
        """

        # checks
        assert isinstance(refl, np.ndarray), "Error - reflectance data must be a numpy array."
        assert len(refl.shape) == 2, "Error - reflectance data must by a 2D array indexed as [samples,wavelengths]."

        # init HyData object with reflectance data
        super().__init__(refl, header=header)
        self.header['file type'] = 'Hylite Library'  # set file type

        # store upper/lower data
        self.upper = upper
        self.lower = lower
        if upper is not None:
            assert upper.shape[0] == self.data.shape[0], \
            "Error - upper bound is different length (%d) to reflectance data (%d)." % (upper.shape[0], self.data.shape[0])
        if lower is not None:
            assert lower.shape[0] == self.data.shape[0], \
            "Error - lower bound is different length (%d) to reflectance data (%d)." % (lower.shape[0], self.data.shape[0])

        # store wavelength data
        if wav is None:
            assert self.header.has_wavelengths(), "Error - no wavelength data specified"
        else:
            assert isinstance(wav, list) or isinstance(wav, np.ndarray), "Error - wavelengths must be a list."
            assert refl.shape[1] == len(wav), "Error - wavelengths must be specified for each reflectance band."
            self.header['wavelength'] = wav.copy()  # store wavelength data

        # store sample labels
        self.set_sample_names( lab )

        # store position data
        self.xyz = None
        if not pos is None:
            assert isinstance(pos, np.ndarray), "Error - positions must be a numpy array."
            assert pos.shape[0] == self.data.shape[0], "Error - positions must be specified for every point."
            assert pos.shape[1] == 3, "Error - positions must be 3D, but pos.shape = %s." % pos.shape
            self.xyz = pos.copy()

    def copy(self,data=True):
        """
        Make a deep copy of this image instance.
        *Arguments*:
         - data = True if a copy of the data should be made, otherwise only copy header.
        *Returns*
          - a new HyLibrary instance.
        """
        header = self.header.copy()

        xyz=None
        if not self.xyz is None:
            xyz = self.xyz.copy()

        data = self.data.copy()
        return HyLibrary(self.get_sample_names(), data, upper=self.upper, lower=self.lower,
                         wav=self.get_wavelengths(), pos=xyz, header=header)

    def merge(self, library2):
        """
        Return a copy of this library merged with another.
        """

        assert (self.get_wavelengths() == library2.get_wavelengths()).all(), "Error - libraries must have same wavelength info..."

        # copy/merge data
        out = self.copy()
        out.data = np.vstack( [self.data, library2.data] )
        if (self.upper is not None) and (self.lower is not None):
            out.upper = np.vstack( [self.upper, library2.upper] )
        if (self.lower is not None) and (self.lower is not None):
            out.lower = np.vstack([self.lower, library2.lower])
        if (self.xyz is not None) and (library2.xyz is not None):
            out.xyz = np.vstack( [self.xyz, library2.xyz] )
        if self.has_sample_names() and library2.has_sample_names():
            out.set_sample_names ( list(self.get_sample_names())+list(library2.get_sample_names()))

        return out

    def sample_count(self):
        """
        Number of samples in this library.
        """
        return self.data.shape[0]

    def has_sample_names(self):
        """
        Has this library got associated sample names?
        """
        return 'sample names' in self.header

    def get_sample_names(self):
        """
        Get list of sample names.
        """
        assert self.has_sample_names(), "Error - sample names are not defined."
        if isinstance(self.header['sample names'], str):  # sample names not yet parsed to list
            self.header['sample names'] = self.header['sample names'].split(',')
        return self.header['sample names']

    def set_sample_names(self, names):
        """
        Set sample names
        """
        assert isinstance(names, list) or isinstance(names, np.ndarray), "Error - sample labels must be a list."
        assert self.data.shape[0] == len(names), "Error - sample names must be specified for each sample."
        self.header['sample names'] = np.array(names)

    def has_sample_labels(self):
        return 'class names' in self.header

    def set_sample_labels(self, labels):
        """
        Set sample names
        """
        assert isinstance(labels, list) or isinstance(labels, np.ndarray), "Error - sample labels must be a list."
        assert self.data.shape[0] == len(labels), "Error - sample labels must be specified for each sample."
        self.header['class names'] = np.array(labels)

    def get_sample_labels(self):
        """
        Get list of sample names.
        """
        assert self.has_sample_names(), "Error - sample names are not defined."
        if isinstance(self.header['class names'], str):  # sample names not yet parsed to list
            self.header['class names'] = self.header['class names'].split(',')
        return self.header['class names']

    # noinspection PyChainedComparisons
    def get_sample_index(self, name):
        """
        Get the index associated with a sample name.

        *Arguments*:
         - the desired sample name (string). If an integer passed it is returned (assumed to be an index).
        """
        if np.issubdtype( type(name), np.integer):  # already an int - easy!
            idx = name
            # noinspection PyChainedComparisons
            assert 0 <= idx < len(self.get_sample_names()), "Error - %d is an invalid index." % idx
            return idx
        else:
            assert name in self.get_sample_names(), "Error - invalid sample name %s" % name
            return list(self.get_sample_names()).index(name)

    def get_sample_spectra(self, sample ):
        """
        Get the reflectance spectra of the specified sample.

        *Arguments*:
         - sample = the sample index or name to extract.
        """

        idx = self.get_sample_index(sample)
        return self.data[ idx, : ]

    def has_positions(self):
        """
        Do samples have defined positions?
        """
        return not self.xyz is None

    def quick_plot(self,band_range=None,  samples=None, ax=None, labels=ref.Themes.DIAGNOSTIC, pad=None, **kwds):
        """
        Plots individual spectra in this library.

        *Arguments*:
         - ax = an axis to plot to. If None (default), a new axis is created.
         - band_range = tuple containing the (min,max) band index (int) or wavelength (float) to plot.
         - samples = a list of sample names or indices to plot, or None (plot all).
         - labels = list of HyFeature instances to plot on spectra. Default is reference.features.DIAGNOSTIC
         - pad = the spacing to add between individual spectra. Default is 5% of the range of reflectance values.
        *Keywords*
         - keywords are passed to plt.plot( ... ).
        """

        if labels is None: # allow None labels (= don't plot labels)
            labels = []

        # parse band range
        minb = 0
        maxb = self.band_count() - 1
        if not band_range is None:
            minb = self.get_band_index(band_range[0])
            maxb = self.get_band_index(band_range[1])

        # parse samples to plot
        sidx = list(range(self.sample_count()))  # plot all samples by default
        if not samples is None:
            sidx = [self.get_sample_index(s) for s in samples]

        # create axes if need be
        if ax is None:
            height = max(min(len(sidx), 100), 8)
            fig, ax = plt.subplots(figsize=(18, height))

        # calculate pad if need be
        if pad is None:
            pad = 0.05 * (np.nanmax(self.data[sidx, minb:maxb]) - np.nanmin(self.data[sidx, minb:maxb]))

        # group based on sample labels
        color = [0 for idx in sidx]
        if self.has_sample_labels():
            lab = self.get_sample_labels()
            sidx2=[]
            color=[]
            n = 0
            for i,idx in enumerate(sidx):
                if idx in sidx2: continue
                sidx2.append(idx)
                color.append(n)
                for idx2 in sidx:
                    if idx2 in sidx2: continue
                    if lab[idx] == lab[idx2]:
                        sidx2.append(idx2)
                        color.append(n)
                n+=1
            sidx = sidx2

        # plot spectra
        wav = self.get_wavelengths()[minb:maxb]
        baseline = np.zeros(len(wav))
        yticks = []
        color_map = {0 : 'g', 1 : 'b' }
        for i, idx in enumerate(sidx):

            # extract spectra slice
            refl = self.data[idx, minb:maxb]

            # calculate offset
            if self.lower is None:
                offset = np.abs(np.nanmax(np.abs(refl - baseline))) + pad
            else:
                lower = self.lower[idx, minb:maxb]
                offset = np.abs(np.nanmax(np.abs(lower - baseline))) + pad

            # plot
            if not 'c' in kwds or 'color' in kwds:
                kwds['color'] = color_map[color[i]%2]

            # plot spectra
            _y = offset + refl  # calculate y
            plt.plot(wav, _y, **kwds)
            baseline = _y
            yticks.append(_y[0])

            # plot error bounds?
            if (self.upper is not None) and (self.lower is not None):
                upper = self.upper[idx, minb:maxb]
                _y = np.hstack( [offset+lower, (offset+upper)[::-1] ] )
                _x = np.hstack( [wav, wav[::-1]] )
                plt.fill(_x,_y,color='grey',alpha=0.25)
                baseline = offset+upper # move baseline up to upper error bound

        #ax.autoscale(False)
        i = 0
        if not isinstance(labels, list): labels = [labels]
        for l in labels:

            # plot label
            if ax.get_xlim()[0] < l.pos < ax.get_xlim()[1]:
                if isinstance(l, MultiFeature) or isinstance(l, MixedFeature):
                    l.quick_plot(ax=ax, method='fill+line', sublabel=i)
                    i+=l.count()
                else:
                    l.quick_plot(ax=ax, method='fill+line', label=i)
                    i+=1

        # sort out y-ticks
        ax.set_yticks(yticks)
        ax.set_yticklabels([self.get_sample_names()[i] for i in sidx])

        # add major x-ticks
        mn, mx = ax.get_xlim()
        ticks = range(int(mn/100)*100, int(mx/100)*100,100)
        ax.set_xticks( ticks)
        ax.set_xticklabels(ticks, rotation=45)

        # add minor x-ticks
        order = int(np.log10(ax.get_xlim()[1] - ax.get_xlim()[0])) - 1
        mn = np.around(ax.get_xlim()[0], -order)
        mx = np.around(ax.get_xlim()[1], -order)
        ax.set_xticks(np.arange(mn, mx, 10 ** order)[1::], minor=True)
        ax.set_xticklabels([], minor=True)
        ax.grid(which='major', axis='x', alpha=0.75)
        ax.grid(which='minor', axis='x', alpha=0.2)

        return ax.get_figure(), ax