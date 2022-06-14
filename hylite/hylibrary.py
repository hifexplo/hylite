from hylite.hydata import HyData
from hylite.hyfeature import HyFeature, MultiFeature, MixedFeature
import hylite.reference.features as ref
import numpy as np
import matplotlib.pyplot as plt


def from_indices(data, indices, s=4, names=None, ):
    """
    Extract a spectral library by sampling and averaging pixels within the specified distance of sample points.
    *Arguments*:
     - data = a HyData instance containing the spectral data.
     - indices = a list of sample indices to extract spectra from
     - s = the number of adjacent points to include in each sample. For HyImage data this will be a square patch of sxs
           pixels. For HyCloud data s is used to take adjacent points from the points list (which is assumed to be somewhat ordered).
     - names = a list containing names for each sample, or None to generate (numeric) sample ids.
    *Returns*: a HyLibrary instance
    """

    if names is None:
        names = ["Sample_%d" % i for i in np.arange(1, len(indices) + 1)]

    # extract spectra
    S = []
    for idx, name in zip(indices, names):
        if isinstance(idx, int):
            X = data.data[idx - s: idx + 1, :]
        elif len(idx) == 2:  # image
            X = data.data[idx[0] - s: idx[0] + s, idx[1] - s: idx[1] + s, :].reshape(-1, data.band_count())
        elif len(idx) == 1:
            X = data.data[idx[0] - s: idx[0] + s, :]
        S.append(HyLibrary(X[None, :, :], lab=[name], wav=data.get_wavelengths()))

    # merge
    out = S[0]
    for i in range(1, len(S)):
        out += S[i]

    # return
    return out


def from_classification(data, labels, names=None, ignore=[0], subsample='all'):
    """
    Extract a spectral library from a labelled dataset.

    *Arguments*:
     - data = a HyData instance containing the hyperspectral data.
     - labels = a HyData instance or numpy array containing class labels.
     - names = a dictionary with numerical labels (integers) in labels.data as keys and corresponding string names
               as values. If None, class names will be pulled from the header, and if these don't exist then integer
               names will be used.
     - ignore = a list of class IDs (integer values) to ignore. By default all nans and 0's will be ignored.
     - method = method used for reducing the number of spectra in this library. Options are 'all' (retain all
                measurements), an integer of the number of points to (randomly) sample, or a list containing
                percentiles to sample. Using the percentile method, samples will be ranked by median brightness
                (albedo) and the spectra corresponding to the desired percentiles kept in the library. Default is 'all'.
    *Returns*: a HyLibrary instance.
    """

    X = data.X()

    if isinstance(labels, HyData):
        L = labels.X().ravel()
    else:
        assert isinstance(labels, np.ndarray), "Error - labels must be array or HyData, not %s" % str(type(labels))
        L = labels.ravel()
    assert X.shape[0] == L.shape[0], "Error - number of spectra (%d) does not match number of labels (%d)?" % (
    X.shape[0], L.shape[0])

    ids = np.unique(L)  # get unique classes

    # build name dictionary if not provided
    cls_names = {}
    if names is None:
        classes = np.arange(0, np.nanmax(ids))
        if isinstance(labels, HyData) and 'class names' in labels.header:
            classes = labels.header['class names']
            if isinstance(classes, str):
                classes = classes.split(',')
        for i, n in enumerate(classes):
            cls_names[i] = n

    Y = []  # store columns of spectral library here
    samples = []  # sample names
    for i, n in enumerate(ids):  # loop through classes
        if n in ignore:
            continue  # skip ignored classes

        _S = X[L == n, :]  # get samples
        if isinstance(subsample, str) and 'all' in subsample:
            S = _S  # keep all samples
        if isinstance(subsample, int):  # random selection
            if _S.shape[0] < subsample:
                S = _S
            else:
                S = _S[np.random.choice(subsample, subsample), :]  # select random spectra
        elif isinstance(subsample, list) or isinstance(subsample, tuple) or isinstance(subsample, np.ndarray):
            # get percentiles
            a = np.nanmean(_S, axis=-1)
            idx = np.argsort(a)
            S = []
            for p in subsample:
                assert isinstance(p, int), "Error - percentiles must be integers between 0 and 100, not %s." % p
                assert 0 <= p and 100 >= p, "Error - percentiles must be integers between 0 and 100, not %s." % p
                p = int(p / 100. * len(idx))  # convert percentile to index
                S.append(_S[idx[p], :])
            S = np.array(S)
        # store
        Y.append(S)

        samples.append(cls_names.get(n, 'S_%s' % (i + 1)))

    out = np.full((len(Y), np.max([len(y) for y in Y]), X.shape[-1]), np.nan)  # build output array
    for i, s in enumerate(Y):
        out[i, :s.shape[0], :] = s

    return HyLibrary(out, samples, wav=data.get_wavelengths())

class HyLibrary(HyData):
    """
    A class for loading and managing spectral libraries and associated metadata.
    """

    def __init__(self, data, lab=None, wav=None, header=None):

        """
        Create a hyperspectral library.

        *Arguments*:
         - data = a numpy array containing spectral measurements and indexed as [sample, measurement, band]. If different
                  numbers of measurements are available per sample then some of the measurement axes can be set as nan. If
                  a 2D array is passed (sample, band) then it will be expanded to (sample, 1, band).
         - lab = list of sample labels (one label per sample). Default is None (labels will be integers from 0 - n).
         - wav = list of wavelengths in the spectra for each sample. If None this must be defined in the header file passed.
         - header = a io.HyHeader instance containing additional metadata to associate with this library.
        """

        # checks
        assert isinstance(data, np.ndarray), "Error - reflectance data must be a numpy array."
        if len(data.shape) == 2:
            data = data[:,None,:] # expand to correct shape
        assert len(data.shape) == 3, "Error - reflectance data must by a 2D array indexed as [sample,measurement,bands]."

        # init HyData object with reflectance data
        super().__init__(data, header=header)
        self.header['file type'] = 'Hylite Library'  # set file type

        # store wavelength data
        if wav is None:
            assert self.header.has_wavelengths(), "Error - no wavelength data specified"
        else:
            assert isinstance(wav, list) or isinstance(wav, np.ndarray), "Error - wavelengths must be a list."
            assert data.shape[-1] == len(wav), "Error - wavelengths must be specified for each reflectance band."
            self.header['wavelength'] = wav.copy()  # store wavelength data

        # store sample labels
        if lab is not None:
            self.set_sample_names( lab )

    def copy(self,data=True):
        """
        Make a deep copy of this HyLibrary instance.
        *Arguments*:
         - data = True if a copy of the data should be made, otherwise only copy header.
        *Returns*
          - a new HyLibrary instance.
        """
        header = self.header.copy()
        if data:
            arr = self.data.copy()
        else:
            arr = self.data
        names = None
        wav = None
        if self.has_sample_names():
            names = self.get_sample_names()
        if data==True and self.has_wavelengths():
            wav = self.get_wavelengths()
        out = HyLibrary(arr, lab=names, wav=wav, header=header)
        if not data:
            out.data = None # drop data array to protect it (we don't want shallow copies)
        return out

    def as_image(self, shallow=False):
        """
        Convert this library to a HyImage dataset for e.g. visualisation using HyImage.quick_plot(...).

        *Arguments*:
         - shallow = True if the underlying data array should be shared. Default is False as copies are dangerous.
        *Returns*:
         - a HyImage instance of this dataset.
        """
        from hylite.hyimage import HyImage # N.B. this import must be here to avoid circular import
        if shallow:
            return HyImage( self.data, header=self.header )
        else:
            return HyImage( self.data.copy(), header=self.header.copy() )

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
        if self.has_sample_names() and library2.has_sample_names():
            out.set_sample_names ( list(self.get_sample_names())+list(library2.get_sample_names()))

        return out

    def sample_count(self):
        """
        Number of samples/spectra in this library.
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
        if self.has_sample_names():
            return self.header.get_list('sample names') # names are defined
        else:
            # no names defined, so make some
            self.header['sample names'] = ['S%d' % i for i in range(self.sample_count()) ]
            return self.get_sample_names()

    def set_sample_names(self, names):
        """
        Set sample names
        """
        assert isinstance(names, list) or isinstance(names, np.ndarray), "Error - sample labels must be a list."
        assert self.data.shape[0] == len(names), "Error - sample names must be specified for each sample."
        self.header['sample names'] = np.array(names)

    def has_groups(self):
        """
        Returns true if any groups are present in this dataset.
        """
        return len(self.get_groups()) > 0

    def get_groups(self):
        """
        Return a list of groups that have been defined for the HyLibrary.
        """
        groups = []
        for k,v in self.header.items():
            k = k.split(' ')
            if k[0].strip() == 'group':
                groups.append( k[1].strip() )
        return groups

    def get_group_ids(self, name ):
        """
        Return a list containing the indices of spectra associated with the specified group.  This can be useful for
        slicing the self.data matrix.

        *Arguments*:
         - name = a string containing the name of the group.
        *Returns*:
         - a list containing integer indices of the spectra in this group.
        """
        assert 'group %s'%name in self.header, "Error - no group named %s exists." % name
        return self.header.get_list('group %s'%name)

    def get_group(self, name, shallow=False):
        """
        Return a HyLibrary instance containing only the spectra associated with the specified group.

        *Arguments*:
         - name = a string containing the name of the group.
         - shallow = True if the returned HyLibrary instance should be a shallow copy (share the same underlying data
                      array and header file) as this instance so changes propagate. Default is False - shallow copies
                      are powerful but dangerous!.
        *Returns*:
         - a HyLibrary instance containing the requested group of spectra.
        """
        ids = self.get_group_ids(name)
        ids = [ self.get_sample_index(i) for i in ids ]
        if shallow:
            return HyLibrary( self.data[ids,:], lab=self.get_sample_names()[ids], wav=self.get_wavelengths() )
        else:
            return HyLibrary( self.data[ids,:].copy(),  lab=self.get_sample_names()[ids], wav=self.get_wavelengths() )

    def add_group(self, name, indices):
        """
        Associate the specified spectra with some arbitrary grouping key for easy subsequent access. Note that if
        a group of this name already exists then it will be overwritten / updated.

        *Arguments*:
         - name = the name of this group.
         - indices = a list containing the indices (integer) or sample names (string) associated with each of the
                     spectra in this group.
        """
        name = name.strip()
        assert name not in self.get_sample_names(), "Error - %s is already a sample name." % name
        self.header['group %s'%name] = indices # add group to header file

    def __getitem__(self, n):
        """
        Slice this library to extract groups or label names. Keys can be either string defining sample or group names,
        integers (treated as indices of the sample array) or lists/arrays of the above.
        """
        if isinstance(n, list) or isinstance(n, tuple) or isinstance(n, np.ndarray):
            if len(n) == 0:
                assert False, 'Error - %s is an invalid spectra key.' % str(n)
            out = self[n[0]] # create ouput with first entry
            for i in range(1, len(n)):
                out += self[n[i]] # append each entry
            return out # done, return
        else:
            if isinstance(n, int) or isinstance(n,float): # subset using index
                n = self.get_sample_names()[int(n)] # get corresponding name
            if n in self.get_sample_names(): # this is a sample name
                idx = self.get_sample_index(n)
                arr = self.data[ [idx], :, : ]
            elif isinstance(n, str):
                return self.get_group( n ) # return group (easy)
            else:
                assert False, "Error - %s is an invalid spectra key." % str(n)
            return HyLibrary(arr.copy(), lab= [n], wav=self.get_wavelengths())

    def __add__(self, other):

        if (other.band_count() != self.band_count()):
            print("Warning - wavelength arrays do not match. Resampling.")
            other = other.resample( self.get_wavelengths(), vb=False )

        # merge samples
        S = {}
        maxs = -np.inf
        for n in self.get_sample_names():
            arr = self[n].data[0, ...]
            if n in S:
                S[n].append(arr)
            else:
                S[n] = [arr]
            maxs = max(maxs, np.sum([a.shape[0] for a in S[n]]))
        for n in other.get_sample_names():
            arr = other[n].data[0, ...]
            if n in S:
                S[n].append(arr)
            else:
                S[n] = [arr]
            maxs = max(maxs, np.sum([a.shape[0] for a in S[n]]))

        # stack into output array
        arr = np.full( (len(S), maxs, self.band_count() ), np.nan )
        names = []
        for i, (k, v) in enumerate(S.items()):
            names.append(k)
            sample = np.vstack( v )
            arr[i, :sample.shape[0], : ] = sample

        # return
        return HyLibrary( arr, lab=names, wav=self.get_wavelengths())

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

    def collapse(self):
        """
        Returns a copy of this HyLibrary with any groups collapsed into individual samples.
        """
        groups = self.get_groups()
        assert len(groups) > 0, "Error - library has no groups to collapse."

        # get each group and collapse into samples
        samples = []
        for g in groups:
            samples.append( np.vstack( self.get_group(g).data ) ) # stack group

        # build new data array
        arr = np.full( (len(groups), np.max([s.shape[0] for s in samples]), self.band_count()), np.nan)
        for i,v in enumerate(samples):
            arr[i, 0:v.shape[0], : ] = v

        # return output
        return HyLibrary( arr, lab=groups, wav=self.get_wavelengths() )

    def squash(self):
        """
        Returns a copy of this HyLibrary instance with multiple measurements averaged to give a single spectra per sample.
        """

        data = np.nanmedian(self.data, axis=1)
        out = self.copy(data=False)
        out.data = data[:, None, :]
        return out


    def quick_plot(self, band_range=None, ax=None, labels=None, pad=None, collapse=False, hc=False, **kwds):
        """
        Plots individual spectra in this library.

        *Arguments*:
         - ax = an axis to plot to. If None (default), a new axis is created.
         - band_range = tuple containing the (min,max) band index (int) or wavelength (float) to plot.
         - labels = list of HyFeature instances to plot on spectra. Default is reference.features.DIAGNOSTIC
         - pad = the spacing to add between individual spectra. Default is 5% of the range of reflectance values.
         - collapse = True if groups should be plotted rather than individual samples. Default is False.
         - hc = True if the plotted spectra should be hull corrected first. Default is False.
        *Keywords*
         - clip = a tuple with the min, med and max percentiles to plot. Default is (5,50,95).
         - figsize = a figsize for the figure to create (if ax is None).
         - other keywords are passed to plt.plot( ... ).
        """

        if collapse:
            self = self.collapse() # this seems dodgy somehow?

        if labels is None:  # allow None labels (= don't plot labels)
            labels = []

        # parse band range
        minb = 0
        maxb = self.band_count() - 1
        if not band_range is None:
            minb = self.get_band_index(band_range[0])
            maxb = self.get_band_index(band_range[1])

        # hull correct?
        if hc:
            from hylite.correct import get_hull_corrected
            self = get_hull_corrected(self, band_range=(minb, maxb), vb=False)
            minb = 0 # spectra has already been subset
            maxb = self.band_count() - 1

        # create axes if need be
        if ax is None:
            height = max(min(self.data.shape[0], 100), 8)
            fig, ax = plt.subplots(figsize=kwds.pop('figsize', (18, height) ))

        # calculate pad if need be
        if pad is None:
            pad = 0.05 * (np.nanmax(self.data[..., minb:maxb]) - np.nanmin(self.data[..., minb:maxb]))

        # plot spectra
        wav = self.get_wavelengths()[minb:maxb]
        baseline = np.zeros(len(wav))
        yticks = []
        color_map = kwds.pop('color', ['g', 'b'])
        if not isinstance(color_map, list) or isinstance(color_map, np.ndarray):
            color_map = list(color_map)

        clip = kwds.pop('clip', (5,50,95))  # get percentiles keyword for plotting
        for i in range(self.data.shape[0]):

            # extract spectra slice
            refl = self.data[i, :, minb:maxb]
            y0 = y1 = None
            if refl.shape[0] > 1:  # plot percentiles
                if isinstance(clip, int):
                    y = np.nanpercentile(refl, clip, axis=0)
                elif len(clip) == 1:
                    y = np.nanpercentile(refl, clip[0], axis=0)
                elif len(clip) == 3:
                    y0, y, y1 = np.nanpercentile(refl, clip, axis=0)
                else:
                    assert False, "%s is an invalid clip. Must be either (min,med,max) or an integer." % str(clip)
            else:
                y = refl[0, :]


            # calculate offset
            if y0 is None:
                offset = np.abs(np.nanmax(np.abs(y - baseline))) + pad
            else:
                offset = np.abs(np.nanmax(np.abs(y0 - baseline))) + pad

            # calculate color
            kwds['color'] = color_map[i % len(color_map)]

            # plot spectra
            _y = offset + y  # calculate y
            ax.plot(wav, _y, **kwds)
            baseline = _y
            yticks.append(_y[0])

            # plot range
            if (y0 is not None) and (y1 is not None):
                _y = np.hstack([offset + y0, (offset + y1)[::-1]])
                _x = np.hstack([wav, wav[::-1]])
                ax.fill(_x, _y, color='grey', alpha=0.25)
                baseline = offset + y1  # move baseline up to upper error bound

        # ax.autoscale(False)
        i = 0
        if not isinstance(labels, list): labels = [labels]
        for l in labels:

            # plot label
            if ax.get_xlim()[0] < l.pos < ax.get_xlim()[1]:
                if isinstance(l, MultiFeature) or isinstance(l, MixedFeature):
                    l.quick_plot(ax=ax, method='fill+line', sublabel=i)
                    i += l.count()
                else:
                    l.quick_plot(ax=ax, method='fill+line', label=i)
                    i += 1

        # sort out y-ticks
        ax.set_yticks(yticks)
        ax.set_yticklabels(self.get_sample_names())

        # add major x-ticks
        mn, mx = ax.get_xlim()
        ticks = range(int(mn / 100) * 100, int(mx / 100) * 100, 100)
        ax.set_xticks(ticks)
        ax.set_xticklabels(ticks, rotation=45)

        # add minor x-ticks
        order = int(np.log10(ax.get_xlim()[1] - ax.get_xlim()[0])) - 1
        mn = np.around(ax.get_xlim()[0], -order)
        mx = np.around(ax.get_xlim()[1], -order)
        ax.set_xticks(np.arange(mn, mx, 10 ** order)[1::], minor=True)
        ax.set_xticklabels([], minor=True)
        ax.grid(which='major', axis='x', alpha=0.75)
        ax.grid(which='minor', axis='x', alpha=0.2)
        ax.set_xlim(np.min(wav), np.max(wav))
        return ax.get_figure(), ax