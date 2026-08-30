"""Fourier representation of hyperspectral datasets (`hylite.hydata.HyData`). These can be used
(1) for denoising and smoothing,
(2) as a lossy compression technique, and
(3) for the extraction of hyperspectral feature positions and depths."""

import io
import json
import os
import re
import zipfile
from math import erf as _math_erf
import numpy as np
from hylite import HyData, HyHeader
from hylite._deps import require
from hylite.analyse.mwl import MWL
from hylite.hycloud import HyCloud
from hylite.hyimage import HyImage
from hylite.hylibrary import HyLibrary

HYFOURIER_EXTENSION = '.fdr'
FOURIER_ARCHIVE_EXTENSION = '.fda'
_HEADER_ARRAY_KEYS = ('wavelength', 'fourier frequencies')
_FOURIER_HEADER_PREFIX = 'fourier '

class HyFourier:
    def __init__(
        self,
        source: HyData,
        padding: str = "cosine",
        min_freq: float = 0.0,
        max_freq: float = 0.5,
        interpolate=True,
        min_finite=0.8,
        vb=False,
    ):
        """
        Apply an FFT to all spectra in the source data and store the resulting coefficients (in self.data).
        If min_freq and max_freq are specified then the coefficients are truncated to the specified range
        (i.e. a band-pass filter is applied).

        Args:
            source (`hylite.hydata.HyData`): the source data to transform.
            padding : str: the type of padding to use. Options are "cosine" or "reflect".
            min_freq : float: the minimum frequency to include in the transform, as a fraction of the Nyquist frequency.
            max_freq : float: the maximum frequency to include in the transform, as a fraction of the Nyquist frequency.
        """
        if not isinstance(source, HyData):
            raise TypeError('Expected a HyData instance.')
        if source.data is None:
            raise ValueError('HyData has no data array.')
        if padding not in ('cosine', 'reflect'):
            raise ValueError('padding must be "cosine" or "reflect"')

        # copy metadata and flatten spectra to (n_spectra, n_bands)
        self.header = source.header.copy()
        self.padding = padding
        self.min_freq = float(min_freq)
        self.max_freq = float(max_freq)
        self.original_shape = source.data.shape
        self.spatial_shape = source.data.shape[:-1]
        self.n_spectra = int(np.prod(self.spatial_shape))
        self.wav = np.asarray(source.get_wavelengths(), dtype=np.float64)
        self.wav_range = (float(self.wav[0]), float(self.wav[-1]))
        self._cloud_xyz = None
        self._cloud_rgb = None
        self._cloud_normals = None
        if isinstance(source, HyCloud):
            self._cloud_xyz = np.asarray(source.xyz, dtype=np.float64)
            if source.has_rgb():
                self._cloud_rgb = np.asarray(source.rgb)
            if source.has_normals():
                self._cloud_normals = np.asarray(source.normals)

        # interpolate NaNs and mark spectra with enough valid bands
        spectra = np.asarray(source.data, dtype=np.float32).reshape(self.n_spectra, -1)
        if interpolate:
            spectra = np.stack([_fillNans(spec) for spec in spectra])
        self._valid = np.isfinite(spectra).all(axis=1) & (np.isfinite(spectra).mean(axis=1) >= min_finite)
        if not self._valid.any():
            raise ValueError('No valid spectra found in the source data.')

        # band-limit FFT and quantise coefficients to int16 for storage
        valid_coeffs, self._geometry = _bandLimitFftBatch(
            spectra[self._valid], min_freq, max_freq, padding=padding,
        )
        n_coeff = valid_coeffs.shape[1]
        self._scale = float(max(np.max(np.abs(valid_coeffs.real)), np.max(np.abs(valid_coeffs.imag)), 1e-15))
        self.data = np.zeros((self.n_spectra, n_coeff, 2), dtype=np.int16)
        self.data[self._valid, :, 0] = np.clip(
            np.round(valid_coeffs.real / self._scale * 32767), -32768, 32767,
        )
        self.data[self._valid, :, 1] = np.clip(
            np.round(valid_coeffs.imag / self._scale * 32767), -32768, 32767,
        )
        self._extrema = None
        self._extrema_key = None
        self._extrema_sidecar = None
        self._kde_sidecar = None

        # record transform parameters in the ENVI-style header
        self.header['description'] = 'Band-limited FFT coefficients (int16 real, imag)'
        self.header['fourier mode'] = '1d'
        self.header['fourier coeff scale'] = self._scale
        self.header['fourier min frequency'] = self.min_freq
        self.header['fourier max frequency'] = self.max_freq
        self.header['fourier padding'] = self.padding
        self.header['fourier spatial shape'] = list(self.spatial_shape)
        self.header['fourier M'] = self._geometry['M']
        self.header['fourier M min'] = self._geometry['M_min']
        self.header['fourier n samples'] = self._geometry['n_samples']
        self.header['fourier n work'] = self._geometry['n_work']
        self.header['fourier pad'] = self._geometry['pad']
        self.header['fourier frequencies'] = np.arange(-self._geometry['M'], self._geometry['M'] + 1, dtype=np.int32)
        self.header['wavelength'] = self.wav
        self.header['band names'] = [str(int(k)) for k in self.header['fourier frequencies']]
        self.header['fourier source type'] = _sourceTypeName(source)

    @property
    def coeffs(self):
        # dequantise int16 storage back to complex64 coefficients
        s = np.float32(self._scale / 32767.0)
        return (self.data[..., 0].astype(np.float32) + 1j * self.data[..., 1].astype(np.float32)) * s

    def _computeCoeffs(self, min_freq, max_freq):
        """Re-run the band-limited FFT for a non-default frequency pass."""
        spectra = np.asarray(self.toHyData().data, dtype=np.float32).reshape(self.n_spectra, -1)
        spectra = np.stack([_fillNans(spec) for spec in spectra])
        out = np.zeros((self.n_spectra, self.data.shape[1]), dtype=np.complex64)
        valid_coeffs, geometry = _bandLimitFftBatch(
            spectra[self._valid], min_freq, max_freq, padding=self.padding,
        )
        out[self._valid] = valid_coeffs
        return out, geometry

    def precomputeExtrema(self, min_freq=None, max_freq=None, kde_sigma=10.0,
                          kde_minw=None, kde_maxW=None, vb=True):
        """
        Compute minima/maxima for all spectra and cache them in packed sidecar arrays on
        this instance. Also builds an in-memory KDE sidecar used by :meth:`search`.

        Min/max sidecars (wavelength and prominence only, float16) are written to the archive
        by :meth:`save`. Inflection points and KDE arrays are kept in RAM only (rebuilt on
        demand after load).

        Args:
            min_freq: optional lower band-pass bound (fraction of Nyquist).
            max_freq: optional upper band-pass bound (fraction of Nyquist).
            kde_sigma: Gaussian kernel width (nm) for the cached KDE sidecar. Default is 10.
            kde_minw: minimum wavelength for KDE features (default: dataset minimum).
            kde_maxW: maximum wavelength for KDE features (default: dataset maximum).
            vb: show a progress bar during root finding. Default is True.

        Returns:
            The extrema sidecar dictionary.
        """
        min_freq = self.min_freq if min_freq is None else float(min_freq)
        max_freq = self.max_freq if max_freq is None else float(max_freq)

        if (min_freq, max_freq) == (self.min_freq, self.max_freq):
            coeffs, geometry = self.coeffs, self._geometry
        else:
            coeffs, geometry = self._computeCoeffs(min_freq, max_freq)

        records = _computeExtremaBatch(self, coeffs, geometry, vb=vb)
        self._extrema = records
        self._extrema_key = (min_freq, max_freq) if (min_freq, max_freq) == (self.min_freq, self.max_freq) else None
        self._extrema_sidecar = _packExtremaSidecar(records, self.n_spectra, min_freq, max_freq)
        self._getKDE(
            sigma=kde_sigma, minw=kde_minw, maxW=kde_maxW,
            min_freq=min_freq, max_freq=max_freq,
        )
        return self._extrema_sidecar

    def _getKDE(self, sigma=10.0, minw=None, maxW=None, min_prominence=0.0,
                minima=True, maxima=True, min_freq=None, max_freq=None, vb=False):
        """Return a cached KDE sidecar, building extrema and KDE arrays if needed."""
        minw = self.wav_range[0] if minw is None else float(minw)
        maxW = self.wav_range[1] if maxW is None else float(maxW)
        min_freq = self.min_freq if min_freq is None else float(min_freq)
        max_freq = self.max_freq if max_freq is None else float(max_freq)
        key = (minw, maxW, float(sigma), float(min_prominence), min_freq, max_freq, bool(minima), bool(maxima))
        if self._kde_sidecar is not None and self._kde_sidecar.get('key') == key:
            return self._kde_sidecar
        self._ensureExtrema(min_freq=min_freq, max_freq=max_freq, vb=vb)
        self._kde_sidecar = _packKdeSidecar(
            self._extrema_sidecar, sigma=sigma, minw=minw, maxW=maxW,
            min_prominence=min_prominence, minima=minima, maxima=maxima,
            min_freq=min_freq, max_freq=max_freq,
        )
        return self._kde_sidecar

    def _loadExtremaFromBlob(self, ext_blob):
        """Restore CSR extrema cache and record list from archive ext_* arrays."""
        self._extrema_sidecar = _extremaSidecarFromNpz(ext_blob)
        self._extrema = _recordsFromExtremaSidecar(self._extrema_sidecar)
        key = (self._extrema_sidecar['min_freq'], self._extrema_sidecar['max_freq'])
        self._extrema_key = key if key == (self.min_freq, self.max_freq) else None

    def _archivePayload(self):
        """Build the dict passed to np.savez_compressed for a .fdr archive."""
        self.header['fourier coeff scale'] = self._scale
        payload = {
            'header_json': np.frombuffer(
                json.dumps(_jsonifyHeader(self.header)).encode('utf-8'), dtype=np.uint8,
            ),
            'data': self.data,
            'wav': self.wav,
            'valid': self._valid,
            'spatial_shape': np.array(self.spatial_shape),
            'original_shape': np.array(self.original_shape),
            'padding': np.array(self.padding),
            'min_freq': np.array(self.min_freq),
            'max_freq': np.array(self.max_freq),
            'n_samples': np.array(self._geometry['n_samples'], dtype=np.int32),
            'n_work': np.array(self._geometry['n_work'], dtype=np.int32),
            'pad': np.array(self._geometry['pad'], dtype=np.int32),
            'M': np.array(self._geometry['M'], dtype=np.int32),
            'M_min': np.array(self._geometry['M_min'], dtype=np.int32),
        }
        if self._extrema_sidecar is not None:
            for key, value in self._extrema_sidecar.items():
                payload['ext_' + key] = value
        return payload

    @classmethod
    def _fromArchiveBlob(cls, blob):
        """Construct a HyFourier instance from a loaded archive blob."""
        # split coefficient arrays from optional CSR extrema sidecar (ext_* keys)
        main, ext = {}, {}
        for key in blob.files:
            if key.startswith('ext_'):
                ext[key[4:]] = blob[key]
            else:
                main[key] = blob[key]
        ext_blob = ext or None

        header = _headerFromJson(json.loads(main['header_json'].tobytes().decode('utf-8')))
        obj = cls.__new__(cls)
        obj.header = header
        obj.data = main['data'].astype(np.int16)
        obj._scale = float(header['fourier coeff scale'])
        obj.wav = main['wav']
        obj.spatial_shape = tuple(main['spatial_shape'].tolist())
        obj.original_shape = tuple(main['original_shape'].tolist())
        obj.padding = str(np.asarray(main['padding']).item())
        obj.min_freq = float(np.asarray(main['min_freq']).item())
        obj.max_freq = float(np.asarray(main['max_freq']).item())
        obj.n_spectra = obj.data.shape[0]
        obj.wav_range = (float(obj.wav[0]), float(obj.wav[-1]))
        obj._valid = main['valid'].astype(bool) if 'valid' in main else np.any(obj.data, axis=(1, 2))
        obj._geometry = {
            'n_samples': int(main['n_samples']),
            'n_work': int(main['n_work']),
            'pad': int(main['pad']),
            'M': int(main['M']),
            'M_min': int(main['M_min']),
        }
        obj._extrema = None
        obj._extrema_key = None
        obj._extrema_sidecar = None
        obj._kde_sidecar = None
        obj._cloud_xyz = None
        obj._cloud_rgb = None
        obj._cloud_normals = None
        if ext_blob is not None:
            obj._loadExtremaFromBlob(ext_blob)
        return obj

    def _ensureExtrema(self, min_freq=None, max_freq=None, vb=False):
        """Return cached extrema records and keep the packed sidecar in sync."""
        min_freq = self.min_freq if min_freq is None else float(min_freq)
        max_freq = self.max_freq if max_freq is None else float(max_freq)
        key = (min_freq, max_freq)
        if self._extrema is not None and self._extrema_key == key:
            return self._extrema
        if self._extrema_sidecar is not None and (
            self._extrema_sidecar['min_freq'], self._extrema_sidecar['max_freq'],
        ) == key:
            self._extrema = _recordsFromExtremaSidecar(self._extrema_sidecar)
            if key == (self.min_freq, self.max_freq):
                self._extrema_key = key
            return self._extrema
        if key == (self.min_freq, self.max_freq):
            coeffs, geometry = self.coeffs, self._geometry
        else:
            coeffs, geometry = self._computeCoeffs(min_freq, max_freq)
        self._extrema = _computeExtremaBatch(self, coeffs, geometry, vb=vb)
        if key == (self.min_freq, self.max_freq):
            self._extrema_key = key
        self._extrema_sidecar = _packExtremaSidecar(self._extrema, self.n_spectra, min_freq, max_freq)
        self._kde_sidecar = None
        return self._extrema

    def get_wavelengths(self):
        """Return the wavelengths associated with this dataset (from the header file)."""
        return np.asarray(self.wav, dtype=np.float64)

    def getSpectra(self, name=None, wav=None):
        """
        Apply the inverse Fourier transform and return reconstructed spectra.

        Args:
            name: optional sample label(s) as returned by :meth:`search`. When omitted, all
                spectra are returned in the original `hylite.hydata.HyData` subtype. When given, a
                `hylite.hylibrary.HyLibrary` containing the matching spectra is returned.
            wav: optional wavelength grid for evaluation (default: stored wavelengths).

        Returns:
            A `hylite.hydata.HyData` or `hylite.hylibrary.HyLibrary` instance.
        """
        if name is None:
            return self.toHyData(wav)
        if isinstance(name, str):
            names = [name]
        elif isinstance(name, (list, tuple)):
            names = [str(n) for n in name]
        else:
            raise TypeError('name must be a string, list of strings, or None.')
        indices, _ = self._resolveSampleNames(names)
        return self._toHyDataAtIndices(indices, names, wav)

    def getSpectraByName(self, name, wav=None):
        """
        Reconstruct spectra whose display labels match a near-exact name filter.

        Matching accepts any of the label forms returned by :meth:`search`, with
        optional archive and group prefixes omitted, e.g.

        - ``[topaz] splib07b_Topaz_HS184.3B_ASDNGb_AREF``
        - ``splib07b_Topaz_HS184.3B_ASDNGb_AREF``

        An optional archive prefix ``(key)`` is ignored on :class:`HyFourier`.

        Args:
            name: sample label(s) as a string or list of strings.
            wav: optional wavelength grid for evaluation (default: stored wavelengths).

        Returns:
            A `hylite.hylibrary.HyLibrary` containing all matching spectra.

        Raises:
            ValueError: if no valid spectra match the name(s).
        """
        queries = _normalizeSampleNameQueries(name)
        indices, labels = self._matchingSampleNames(queries)
        if not indices:
            raise ValueError('No spectra match name(s) %r.' % (name,))
        return self._toHyDataAtIndices(indices, labels, wav)

    def toHyData(self, wav=None):
        """Evaluate spectra and return the original `hylite.hydata.HyData` subtype (lossy reconstruction)."""
        if wav is None:
            wav = self.get_wavelengths()
        wav = np.asarray(wav, dtype=np.float32)
        g = self._geometry
        t = np.interp(wav, self.wav, np.arange(self.wav.size)).astype(np.float32) + np.float32(g['pad'])
        k = np.arange(-g['M'], g['M'] + 1, dtype=np.float32)
        zpow = _trigPowers(t, k, g['n_work'])
        flat = np.full((self.n_spectra, wav.size), np.nan, dtype=np.float32)
        flat[self._valid] = _evalTrig(self.coeffs[self._valid], zpow)
        data = flat.reshape(self.spatial_shape + (wav.size,))
        header = _reconstructionHeader(self.header, wav)
        kind = str(self.header.get('fourier source type', 'HyData'))

        if kind == 'HyLibrary':
            names = None
            if 'sample names' in header:
                names = list(header.get_list('sample names', str))
            return HyLibrary(data, lab=names, wav=wav, header=header)
        if kind == 'HyImage':
            return HyImage(
                data, header=header,
                projection=header.get('projection', None),
                affine=header.get('affine', [0, 1, 0, 0, 0, 1]),
            )
        if kind == 'HyCloud':
            if self._cloud_xyz is None:
                raise ValueError(
                    'Cannot reconstruct HyCloud without point geometry; '
                    'cloud xyz was not stored on this HyFourier instance.'
                )
            kw = dict(header=header)
            if self._cloud_rgb is not None:
                kw['rgb'] = self._cloud_rgb
            if self._cloud_normals is not None:
                kw['normals'] = self._cloud_normals
            return HyCloud(self._cloud_xyz, bands=data.reshape(self.n_spectra, -1), **kw)
        return HyData(data, header=header)

    def _resolveSampleNames(self, names):
        """Map display names from :meth:`search` to flat spectrum indices."""
        all_names = _sampleNames(self.header, self.n_spectra, self.original_shape, self.spatial_shape)
        name_to_idx = {label: i for i, label in enumerate(all_names)}
        indices = []
        for label in names:
            if label not in name_to_idx:
                raise ValueError('Unknown sample name %r.' % label)
            indices.append(name_to_idx[label])
        return indices, all_names

    def _matchingSampleNames(self, queries):
        """Return flat indices and labels matching near-exact sample-name queries."""
        all_names = _sampleNames(self.header, self.n_spectra, self.original_shape, self.spatial_shape)
        indices = []
        labels = []
        seen = set()
        for query in queries:
            _, inner_query = _parseOptionalArchivePrefix(query)
            for i, label in enumerate(all_names):
                if i in seen or not self._valid[i]:
                    continue
                if _displayNameMatchesQuery(label, inner_query):
                    indices.append(i)
                    labels.append(label)
                    seen.add(i)
        return indices, labels

    def _toHyDataAtIndices(self, indices, labels, wav=None):
        """Reconstruct a subset of spectra as a `hylite.hylibrary.HyLibrary`."""
        if wav is None:
            wav = self.get_wavelengths()
        wav = np.asarray(wav, dtype=np.float32)
        g = self._geometry
        t = np.interp(wav, self.wav, np.arange(self.wav.size)).astype(np.float32) + np.float32(g['pad'])
        k = np.arange(-g['M'], g['M'] + 1, dtype=np.float32)
        zpow = _trigPowers(t, k, g['n_work'])
        indices = np.asarray(indices, dtype=int)
        flat = np.full((len(indices), wav.size), np.nan, dtype=np.float32)
        valid = self._valid[indices]
        if valid.any():
            flat[valid] = _evalTrig(self.coeffs[indices[valid]], zpow)
        data = flat.reshape(len(indices), 1, wav.size)
        return HyLibrary(data, lab=list(labels), wav=wav, header=_reconstructionHeader(self.header, wav))

    def _features(self, minw, maxW, kind='min', format='MWL', min_freq=None, max_freq=None, n_features=1, vb=False):
        """Extract minima, maxima, or inflection features in list or MWL format."""
        if minw is None:
            minw = self.wav_range[0]
        if maxW is None:
            maxW = self.wav_range[1]

        # inflection is not stored in sidecars; recompute when requested
        if kind == 'inflection':
            min_freq = self.min_freq if min_freq is None else float(min_freq)
            max_freq = self.max_freq if max_freq is None else float(max_freq)
            if (min_freq, max_freq) == (self.min_freq, self.max_freq):
                coeffs, geometry = self.coeffs, self._geometry
            else:
                coeffs, geometry = self._computeCoeffs(min_freq, max_freq)
            extrema = _computeExtremaBatch(self, coeffs, geometry, vb=vb)
        elif min_freq is not None or max_freq is not None:
            min_freq = self.min_freq if min_freq is None else min_freq
            max_freq = self.max_freq if max_freq is None else max_freq
            coeffs, geometry = self._computeCoeffs(min_freq, max_freq)
            extrema = _computeExtremaBatch(self, coeffs, geometry, vb=vb)
        else:
            extrema = self._ensureExtrema(vb=vb)

        record_key = {'min': 'minima', 'max': 'maxima', 'inflection': 'inflection'}[kind]
        if format.upper() != 'MWL':
            # return raw feature dicts per spectrum
            out = []
            for record in extrema:
                if record is None:
                    out.append([])
                    continue
                out.append([
                    f for f in record[record_key]
                    if not f.get('fake', False) and minw <= f['wavelength'] <= maxW
                ])
            return out

        # pack the n most prominent features into an MWL-compatible array
        stride = 4
        packed = np.full((self.n_spectra, n_features * stride), np.nan, dtype=np.float64)
        for row, record in enumerate(extrema):
            if record is None:
                continue
            feats = sorted(
                [f for f in record[record_key] if not f.get('fake', False) and minw <= f['wavelength'] <= maxW],
                key=lambda f: -f['prominence'],
            )[:n_features]
            for j, feat in enumerate(feats):
                base = j * stride
                packed[row, base:base + stride] = (
                    feat['prominence'], feat['wavelength'], feat['left_width'], feat['right_width'],
                )

        ref = self.toHyData()
        mwld = ref.copy(data=False)
        mwld.data = packed.reshape(self.spatial_shape + (n_features * stride,))
        mwl = MWL('M', '')
        mwl.bind(mwld, n_features, x=self.get_wavelengths(), sym=False, X=ref)
        return mwl

    def minima(self, minw=None, maxW=None, format='MWL', min_freq=None, max_freq=None, n_features=1, vb=False):
        """
        Get the position, depth, left-hand and right-hand width of minima in the specified spectral range.
        If None, the min and max of the wavelength array is used. Optionally applies a band-pass before
        identifying turning points (to e.g., remove long-wavelength signal).

        Args:
         minw : the minimum wavelength of the search range.
         maxw : the maximum wavelength of the search range.
         format : desired output format. Can be 'MWL' (default) to return an MWL array, or 'list'.
        """
        return self._features(minw, maxW, kind='min', format=format, min_freq=min_freq, max_freq=max_freq, n_features=n_features, vb=vb)

    def maxima(self, minw=None, maxW=None, format='MWL', min_freq=None, max_freq=None, n_features=1, vb=False):
        """
        Get the position, depth, left-hand and right-hand width of maxima in the specified spectral range.
        If None, the min and max of the wavelength array is used. Optionally applies a band-pass before
        identifying turning points (to e.g., remove long-wavelength signal).
        """
        return self._features(minw, maxW, kind='max', format=format, min_freq=min_freq, max_freq=max_freq, n_features=n_features, vb=vb)

    def inflection(self, minw=None, maxW=None, format='MWL', min_freq=None, max_freq=None, n_features=1, vb=False):
        """
        Get the position, depth, left-hand and right-hand width of inflection points in the specified
        spectral range. If None, the min and max of the wavelength array is used. Optionally applies a band-pass before
        identifying turning points (to e.g., remove long-wavelength signal).
        """
        return self._features(minw, maxW, kind='inflection', format=format, min_freq=min_freq, max_freq=max_freq, n_features=n_features, vb=vb)

    def kde(
        self,
        minw=None,
        maxW=None,
        minima=True,
        maxima=True,
        sigma=10.0,
        min_prominence=0.0,
        index=None,
        grid=None,
        min_freq=None,
        max_freq=None,
        vb=False,
    ):
        """
        Build Gaussian kernel-density representations of spectral extrema.

        By default returns a list (one entry per spectrum) of lists of Gaussian dictionaries with keys
        `mu` (centre wavelength), `sigma` (standard deviation in nm), `weight` (feature prominence),
        and `kind` (`'minimum'` or `'maximum'`). Pass `index` to return Gaussians for a single spectrum.

        If `grid` is True the KDE is evaluated on `self.get_wavelengths()`; if `grid` is a numpy array
        it is evaluated on those wavelengths instead. Grid output has shape `(n_spectra, n_grid)` or
        `(n_grid,)` when `index` is set.

        Args:
            minw: minimum wavelength of the search range (default: dataset minimum).
            maxW: maximum wavelength of the search range (default: dataset maximum).
            minima: include absorption features (minima). Default is True.
            maxima: include reflectance peaks (maxima). Default is True.
            sigma: standard deviation of each Gaussian kernel in nm. Default is 10.
            min_prominence: ignore extrema with prominence below this threshold. Default is 0.
            index: if set, return Gaussians (or grid values) for this spectrum index only.
            grid: None to return Gaussian parameters; True or a wavelength array to evaluate the KDE.
            min_freq: optional lower band-pass bound (fraction of Nyquist) before extrema extraction.
            max_freq: optional upper band-pass bound (fraction of Nyquist) before extrema extraction.
            vb: print progress during extrema extraction. Default is False.
        """
        if minw is None:
            minw = self.wav_range[0]
        if maxW is None:
            maxW = self.wav_range[1]
        sigma = float(sigma)
        if sigma <= 0:
            raise ValueError('sigma must be positive.')

        min_freq = self.min_freq if min_freq is None else float(min_freq)
        max_freq = self.max_freq if max_freq is None else float(max_freq)
        if not minima and not maxima:
            raise ValueError('At least one of minima or maxima must be True.')
        sidecar = self._getKDE(
            sigma=sigma, minw=minw, maxW=maxW, min_prominence=min_prominence,
            minima=minima, maxima=maxima, min_freq=min_freq, max_freq=max_freq, vb=vb,
        )

        if index is not None:
            idx = int(index)
            if idx < 0 or idx >= self.n_spectra:
                raise IndexError('Spectrum index %d out of range [0, %d).' % (idx, self.n_spectra))

        if grid is None:
            gaussians = _kdeSidecarToGaussians(sidecar, index=index)
            return gaussians

        wav = self.get_wavelengths() if grid is True else np.asarray(grid, dtype=np.float64)
        if wav.ndim != 1 or wav.size == 0:
            raise ValueError('grid must be True or a one-dimensional wavelength array.')
        return _evalKdeGrid(sidecar, wav, index=index)

    def search(
        self,
        query,
        confidence=10.0,
        n_result=10,
        minw=None,
        maxW=None,
        min_freq=None,
        max_freq=None,
        vb=False,
    ):
        """
        Rank spectra using a naive-Bayes feature matcher and/or sample-name filter.

        Query syntax (whitespace-separated tokens are combined with AND):

        - `2200` — absorption (minimum) near 2200 nm
        - `^2300` — reflectance peak (maximum) near 2300 nm
        - `!2200` — absence of an absorption (minimum) near 2200 nm
        - `!^2300` — absence of a reflectance peak (maximum) near 2300 nm
        - `2160-2200` — feature anywhere in the wavelength range
        - `Kaolinite` — case-insensitive substring match on sample names
        - `beck quartz` — sample name must contain **all** name tokens (score 1.0
          when every token matches, lower when only some match)
        - tokens can be combined, e.g. `2200 Kaolinite`
        - `kaolinite | dolomite` — OR between ``|``-separated sub-queries (results
          interleaved by sub-query rank)

        `confidence` sets the half-width (nm) used when integrating over a point query
        (default 10 nm, i.e. ±10 nm). Range queries integrate over the stated interval.

        Args:
            query: search string describing observed/excluded features and/or sample names.
            confidence: positional uncertainty for point queries, in nm. Default is 10.
            n_result: maximum number of ranked results to return. Default is 10.
            minw: minimum wavelength for extrema extraction (default: dataset minimum).
            maxW: maximum wavelength for extrema extraction (default: dataset maximum).
            min_freq: optional lower band-pass bound before extrema extraction.
            max_freq: optional upper band-pass bound before extrema extraction.
            vb: print progress during extrema extraction. Default is False.

        Returns:
            A tuple `(names, scores)` of ranked sample labels and non-normalised
            likelihoods in `[0, 1]` (product of per-feature match probabilities).
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError('query must be a non-empty string.')
        sub_queries = _split_or_queries(query)
        if len(sub_queries) <= 1:
            return _hyfourier_search_single(
                self,
                query,
                confidence=confidence,
                n_result=n_result,
                minw=minw,
                maxW=maxW,
                min_freq=min_freq,
                max_freq=max_freq,
                vb=vb,
            )
        query_results = [
            _hyfourier_search_single(
                self,
                sub_query,
                confidence=confidence,
                n_result=n_result,
                minw=minw,
                maxW=maxW,
                min_freq=min_freq,
                max_freq=max_freq,
                vb=vb,
            )
            for sub_query in sub_queries
        ]
        return _merge_or_search_results(query_results, n_result=n_result)

    def save(self, path):
        """
        Save header metadata, Fourier coefficients, and precomputed min/max sidecars to a
        single compressed `.fdr` archive (zlib/deflate, same compression as `.npz`).
        KDE and inflection data are not written to disk.
        """
        archive_path = _parsePath(path)
        # np.savez appends .npz to paths; write raw zip bytes to keep the .fdr extension
        buf = io.BytesIO()
        np.savez_compressed(buf, **self._archivePayload())
        with open(archive_path, 'wb') as fh:
            fh.write(buf.getvalue())

    @classmethod
    def load(cls, path):
        """Load a `.fdr` archive written by :meth:`save`."""
        archive_path = _parsePath(path)
        if not os.path.isfile(archive_path):
            raise FileNotFoundError('No HyFourier archive found at %r' % archive_path)
        with open(archive_path, 'rb') as fh:
            return cls._fromArchiveBlob(np.load(fh, allow_pickle=False))

class FourierArchive:
    """Named collection of :class:`HyFourier` instances stored in a single `.fda` file."""

    def __init__(self, entries=None):
        self._entries = {}
        if entries:
            for name, value in entries.items():
                self[name] = value

    def __len__(self):
        return len(self._entries)

    def __iter__(self):
        return iter(self._entries)

    def __contains__(self, name):
        return name in self._entries

    def __getitem__(self, name):
        return self._entries[name]

    def __setitem__(self, name, value):
        _validateArchiveName(name)
        if not isinstance(value, HyFourier):
            raise TypeError('FourierArchive values must be HyFourier instances.')
        self._entries[str(name)] = value

    def __delitem__(self, name):
        del self._entries[name]

    def keys(self):
        """Return archive entry names."""
        return self._entries.keys()

    def values(self):
        """Return :class:`HyFourier` instances."""
        return self._entries.values()

    def items(self):
        """Return `(name, HyFourier)` pairs."""
        return self._entries.items()

    def get(self, name, default=None):
        """Return a named :class:`HyFourier`, or `default` if missing."""
        return self._entries.get(name, default)

    def save(self, path):
        """Write all entries to a compressed `.fda` archive (zip of `.fdr` payloads)."""
        archive_path = _fdaPath(path)
        manifest = {'names': list(self._entries.keys())}
        with zipfile.ZipFile(archive_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr('manifest.json', json.dumps(manifest))
            for name, hyfourier in self._entries.items():
                buf = io.BytesIO()
                np.savez_compressed(buf, **hyfourier._archivePayload())
                zf.writestr('%s.fdr' % name, buf.getvalue())

    @classmethod
    def load(cls, path):
        """Load a `.fda` archive written by :meth:`save`."""
        archive_path = _fdaPath(path)
        if not os.path.isfile(archive_path):
            raise FileNotFoundError('No FourierArchive found at %r' % archive_path)
        with open(archive_path, 'rb') as fh:
            return cls.load_from_buffer(fh)

    @classmethod
    def load_bytes(cls, data):
        """
        Load a `.fda` archive from raw bytes (e.g. fetched in a browser via Pyodide).

        Args:
            data: `bytes`, `bytearray`, or `memoryview` containing a saved archive.

        Returns:
            A :class:`FourierArchive` instance.
        """
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise TypeError('data must be bytes-like.')
        return cls.load_from_buffer(data)

    @classmethod
    def load_from_buffer(cls, buffer):
        """
        Load a `.fda` archive from a readable binary buffer.

        Args:
            buffer: `bytes`/`bytearray`/`memoryview`, or a binary file-like object
                (e.g. `io.BytesIO`) positioned at the start of the archive.

        Returns:
            A :class:`FourierArchive` instance.
        """
        if isinstance(buffer, (bytes, bytearray, memoryview)):
            buffer = io.BytesIO(bytes(buffer))
        entries = {}
        with zipfile.ZipFile(buffer, 'r') as zf:
            manifest = json.loads(zf.read('manifest.json').decode('utf-8'))
            for name in manifest['names']:
                blob = np.load(io.BytesIO(zf.read('%s.fdr' % name)), allow_pickle=False)
                entries[name] = HyFourier._fromArchiveBlob(blob)
        return cls(entries)

    def search(
        self,
        query,
        confidence=10.0,
        n_result=10,
        minw=None,
        maxW=None,
        min_freq=None,
        max_freq=None,
        vb=False,
    ):
        """
        Search all archive entries and return a merged, globally ranked result list.

        Sample names are prefixed with the archive key, e.g.
        `(beck) [topaz] splib07b_Topaz_HS184.3B_BECKb_AREF`. Name-token searches
        match against that archive-qualified form (so ``usgs`` matches
        ``usgs_minerals:beck``, etc.).

        See :meth:`HyFourier.search` for query syntax (including ``|`` OR sub-queries)
        and other arguments.

        Returns:
            A tuple `(names, scores)` of ranked archive-qualified labels and
            non-normalised likelihoods in `[0, 1]`.
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError('query must be a non-empty string.')
        sub_queries = _split_or_queries(query)
        if len(sub_queries) <= 1:
            return _fourier_archive_search_merged(
                self,
                query,
                confidence=confidence,
                n_result=n_result,
                minw=minw,
                maxW=maxW,
                min_freq=min_freq,
                max_freq=max_freq,
                vb=vb,
            )
        query_results = [
            _fourier_archive_search_merged(
                self,
                sub_query,
                confidence=confidence,
                n_result=n_result,
                minw=minw,
                maxW=maxW,
                min_freq=min_freq,
                max_freq=max_freq,
                vb=vb,
            )
            for sub_query in sub_queries
        ]
        return _merge_or_search_results(query_results, n_result=n_result)

    def getSpectra(self, name, wav=None):
        """
        Reconstruct spectra for an archive-qualified sample name.

        Args:
            name: label as returned by :meth:`search`, e.g.
                `(beck) [topaz] splib07b_Topaz_HS184.3B_BECKb_AREF`, or a list of such names.
            wav: optional wavelength grid for evaluation (default: stored wavelengths).

        Returns:
            A `hylite.hylibrary.HyLibrary` instance (merged when `name` is a list).
        """
        if isinstance(name, str):
            key, sample_name = _parseArchiveSampleName(name)
            return self._entries[key].getSpectra(sample_name, wav=wav)
        if isinstance(name, (list, tuple)):
            out = self.getSpectra(name[0], wav=wav)
            for entry in name[1:]:
                out += self.getSpectra(entry, wav=wav)
            return out
        raise TypeError('name must be a string or list of archive-qualified sample names.')

    def getSpectraByName(self, name, wav=None):
        """
        Reconstruct the first spectrum matching `name` across archive entries.

        Entries are checked in insertion order; the first :class:`HyFourier`
        library containing a match is used.

        Matching accepts any of the label forms returned by :meth:`search`, with
        optional prefixes omitted, e.g.

        - ``(beck) [topaz] splib07b_Topaz_HS184.3B_BECKb_AREF``
        - ``[topaz] splib07b_Topaz_HS184.3B_BECKb_AREF``
        - ``splib07b_Topaz_HS184.3B_BECKb_AREF``

        When an archive key prefix ``(key)`` is given, only that entry is searched.

        Args:
            name: sample label(s) as a string or list of strings.
            wav: optional wavelength grid for evaluation (default: stored wavelengths).

        Returns:
            A `hylite.hylibrary.HyLibrary` containing the first matching spectrum.

        Raises:
            ValueError: if no valid spectra match the name(s) in any entry.
        """
        queries = _normalizeSampleNameQueries(name)
        for query in queries:
            query_key, _ = _parseOptionalArchivePrefix(query)
            if query_key is not None:
                if query_key not in self._entries:
                    continue
                entries = [(query_key, self._entries[query_key])]
            else:
                entries = self._entries.items()
            for key, hyfourier in entries:
                all_names = _sampleNames(
                    hyfourier.header, hyfourier.n_spectra,
                    hyfourier.original_shape, hyfourier.spatial_shape,
                )
                for i, label in enumerate(all_names):
                    if hyfourier._valid[i] and _archiveDisplayNameMatchesQuery(key, label, query):
                        return hyfourier._toHyDataAtIndices([i], [label], wav)
        raise ValueError('No spectra match name(s) %r in any archive entry.' % (name,))


def _validateArchiveName(name):
    """Reject archive keys that are unsafe as zip member names."""
    name = str(name)
    if not name or name in ('.', '..') or '/' in name or '\\' in name:
        raise ValueError('Invalid FourierArchive name: %r' % name)


def _fdaPath(path):
    """Normalise a user path to the single-file .fda archive name."""
    base, ext = os.path.splitext(path)
    if ext.lower() == FOURIER_ARCHIVE_EXTENSION:
        return path
    return base + FOURIER_ARCHIVE_EXTENSION


def _sourceTypeName(source):
    """Return the most specific `hylite.hydata.HyData` subclass name for `source`."""
    if isinstance(source, HyLibrary):
        return 'HyLibrary'
    if isinstance(source, HyImage):
        return 'HyImage'
    if isinstance(source, HyCloud):
        return 'HyCloud'
    return 'HyData'


def _reconstructionHeader(header, wav):
    """Copy header metadata for reconstructed spectra, omitting Fourier archive keys."""
    wav = np.asarray(wav, dtype=np.float64)
    out = HyHeader()
    for key, value in header.items():
        if str(key).startswith(_FOURIER_HEADER_PREFIX):
            continue
        if key == 'description' and 'FFT coefficients' in str(value):
            continue
        out[key] = value
    out['wavelength'] = wav
    out['bands'] = str(wav.size)
    return out

def _parsePath(path):
    """Normalise a user path to the single-file .fdr archive name."""
    base, ext = os.path.splitext(path)
    if ext.lower() == HYFOURIER_EXTENSION:
        return path
    return base + HYFOURIER_EXTENSION


def _jsonifyHeader(header):
    """Convert a `hylite.hyheader.HyHeader` to JSON-serialisable Python values."""
    out = {}
    for key, value in header.items():
        if isinstance(value, np.ndarray):
            out[key] = value.tolist()
        elif isinstance(value, (np.integer, np.floating)):
            out[key] = value.item()
        else:
            out[key] = value
    return out


def _headerFromJson(data):
    """Rebuild a `hylite.hyheader.HyHeader` from JSON-deserialised header data."""
    header = HyHeader()
    for key, value in data.items():
        if key in _HEADER_ARRAY_KEYS and isinstance(value, list):
            header[key] = np.asarray(value)
        else:
            header[key] = value
    return header

def _newtonRefineBatch(t, fc, fpc, k, N, n_iter=6):
    """Refine bracketed roots on the Fourier series and its derivative."""
    t = np.asarray(t, dtype=np.float32)
    fc = np.asarray(fc, dtype=np.complex64)
    fpc = np.asarray(fpc, dtype=np.complex64)
    for _ in range(n_iter):
        zpow = _trigPowers(t, k, N)
        f = np.real((fc * zpow).sum(axis=-1))
        fp = np.real((fpc * zpow).sum(axis=-1))
        t = t - f / np.where(np.abs(fp) < 1e-12, 1.0, fp)
    return t


def _evalTrigAt(t, coeff_rows, k, N):
    """Evaluate the Fourier series at one or more parameter values."""
    zpow = _trigPowers(np.asarray(t, dtype=np.float32), k, N)
    return np.real((coeff_rows * zpow).sum(axis=-1)).astype(np.float64)


def _indexToWavelength(t, pad, ns, wav):
    """Map Fourier-series parameter values to wavelengths by linear interpolation."""
    idx = np.clip(np.asarray(t, dtype=np.float64) - pad, 0, ns - 1)
    return np.interp(idx, np.arange(ns, dtype=np.float64), wav)


def _dedupePoints(points):
    """Merge extrema closer than 0.5 nm, keeping the stronger derivative."""
    if not points:
        return points
    kind_order = {'maximum': 0, 'minimum': 1, 'inflection': 2}
    points = sorted(points, key=lambda p: (p['wavelength'], kind_order.get(p['kind'], 3)))
    kept = [points[0]]
    for p in points[1:]:
        if p['wavelength'] - kept[-1]['wavelength'] < 0.5:
            if abs(p.get('second_deriv', p.get('third_deriv', 0.0))) > abs(
                kept[-1].get('second_deriv', kept[-1].get('third_deriv', 0.0))
            ):
                kept[-1] = p
        else:
            kept.append(p)
    return kept


def _prominence(point, pool, pkind):
    """Compute feature prominence from neighbouring turning points."""
    if pkind == 'minimum':
        left = [p for p in pool if p['kind'] == 'maximum' and p['wavelength'] < point['wavelength']]
        right = [p for p in pool if p['kind'] == 'maximum' and p['wavelength'] > point['wavelength']]
        if not left or not right:
            return 0.0
        return 0.5 * (left[-1]['value'] + right[0]['value']) - point['value']
    if pkind == 'maximum':
        left = [p for p in pool if p['kind'] == 'minimum' and p['wavelength'] < point['wavelength']]
        right = [p for p in pool if p['kind'] == 'minimum' and p['wavelength'] > point['wavelength']]
        if not left or not right:
            return 0.0
        return point['value'] - 0.5 * (left[-1]['value'] + right[0]['value'])
    return abs(point.get('third_deriv', 0.0))


def _computeExtremaBatch(hyfourier, coeffs, geometry, vb=False):
    """Find minima, maxima, and inflection points via batched derivative root-finding."""
    M = geometry['M']
    N = geometry['n_work']
    pad = geometry['pad']
    ns = geometry['n_samples']
    wlo, whi = hyfourier.wav_range
    wav = hyfourier.wav

    # coarse grid for batched sign-change detection across all valid spectra
    k = np.arange(-M, M + 1, dtype=np.float32)
    factor = (2j * np.pi * k / np.float32(N)).astype(np.complex64)
    n_grid = max(2 * ns, 8)
    t_grid = np.linspace(pad, pad + ns - 1, n_grid, dtype=np.float32)
    pw = _trigPowers(t_grid, k, N)

    valid_rows = np.where(hyfourier._valid)[0]
    cache = [None] * coeffs.shape[0]
    if valid_rows.size == 0:
        return cache

    vc = np.asarray(coeffs[valid_rows], dtype=np.complex64)
    d1c = vc * factor
    d2c = vc * factor ** 2
    d3c = vc * factor ** 3
    d1 = _evalTrig(d1c, pw)
    d2 = _evalTrig(d2c, pw)

    sc1 = (d1[:, :-1] * d1[:, 1:]) < 0
    sc2 = (d2[:, :-1] * d2[:, 1:]) < 0
    row1, col1 = np.where(sc1)
    row2, col2 = np.where(sc2)

    extrema_points = [[] for _ in range(vc.shape[0])]
    infl_points = [[] for _ in range(vc.shape[0])]

    # vectorised Newton refinement on all first-derivative brackets (minima/maxima)
    if row1.size:
        t1 = _newtonRefineBatch(
            0.5 * (t_grid[col1] + t_grid[col1 + 1]), d1c[row1], d2c[row1], k, N,
        )
        d1v = _evalTrigAt(t1, d1c[row1], k, N)
        d2v = _evalTrigAt(t1, d2c[row1], k, N)
        vals = _evalTrigAt(t1, vc[row1], k, N)
        wl = _indexToWavelength(t1, pad, ns, wav)
        keep = (np.abs(d1v) <= 1e-4) & (d2v != 0) & (wl >= wlo) & (wl <= whi)
        for vi, w, val, d2val, t in zip(row1[keep], wl[keep], vals[keep], d2v[keep], t1[keep]):
            extrema_points[int(vi)].append({
                'wavelength': float(w), 'index': float(t), 'value': float(val),
                'second_deriv': float(d2val),
                'kind': 'minimum' if d2val > 0 else 'maximum', 'fake': False,
            })

    # vectorised Newton refinement on second-derivative brackets (inflection)
    if row2.size:
        t2 = _newtonRefineBatch(
            0.5 * (t_grid[col2] + t_grid[col2 + 1]), d2c[row2], d3c[row2], k, N,
        )
        d2v = _evalTrigAt(t2, d2c[row2], k, N)
        d3v = _evalTrigAt(t2, d3c[row2], k, N)
        vals = _evalTrigAt(t2, vc[row2], k, N)
        wl = _indexToWavelength(t2, pad, ns, wav)
        keep = (np.abs(d2v) <= 1e-4) & (wl >= wlo) & (wl <= whi)
        for vi, w, val, d3val, t in zip(row2[keep], wl[keep], vals[keep], d3v[keep], t2[keep]):
            infl_points[int(vi)].append({
                'wavelength': float(w), 'index': float(t), 'value': float(val),
                'third_deriv': float(d3val), 'kind': 'inflection', 'fake': False,
            })

    t0, t1b = float(pad), float(pad + ns - 1)

    # per-spectrum assembly: dedupe, boundary fakes, prominence, MWL-style records
    loop = np.arange(vc.shape[0])
    if vb:
        tqdm = require("tqdm").tqdm
        loop = tqdm(loop, desc='Root finding', leave=False)

    for vi in loop:
        i = int(valid_rows[vi])
        c = vc[vi]
        points = _dedupePoints(extrema_points[vi])
        infl = _dedupePoints(infl_points[vi])
        v0 = float(_evalTrigAt(t0, c[None], k, N)[0])
        v1 = float(_evalTrigAt(t1b, c[None], k, N)[0])
        fakes = [
            {'wavelength': wlo, 'value': v0, 'kind': 'maximum', 'fake': True},
            {'wavelength': wlo, 'value': v0, 'kind': 'minimum', 'fake': True},
            {'wavelength': whi, 'value': v1, 'kind': 'maximum', 'fake': True},
            {'wavelength': whi, 'value': v1, 'kind': 'minimum', 'fake': True},
        ]
        all_tp = sorted(
            points + fakes,
            key=lambda p: (p['wavelength'], {'maximum': 0, 'minimum': 1}.get(p['kind'], 2)),
        )
        minima, maxima = [], []
        for j, point in enumerate(all_tp):
            left = all_tp[j - 1] if j > 0 else None
            right = all_tp[j + 1] if j < len(all_tp) - 1 else None
            record = {
                'wavelength': point['wavelength'],
                'index': point.get('index', 0.0),
                'value': point['value'],
                'left_width': float(point['wavelength'] - left['wavelength']) if left else 0.0,
                'right_width': float(right['wavelength'] - point['wavelength']) if right else 0.0,
                'fake': point.get('fake', False),
            }
            if point['kind'] == 'minimum':
                record['prominence'] = _prominence(point, all_tp, 'minimum')
                minima.append(record)
            elif point['kind'] == 'maximum':
                record['prominence'] = _prominence(point, all_tp, 'maximum')
                maxima.append(record)
        cache[i] = {
            'minima': minima,
            'maxima': maxima,
            'inflection': [{
                'wavelength': p['wavelength'],
                'index': p['index'],
                'value': p['value'],
                'left_width': 0.0,
                'right_width': 0.0,
                'prominence': _prominence(p, infl, 'inflection'),
                'fake': False,
            } for p in infl],
        }
    return cache


def _packExtremaSidecar(records, n_spectra, min_freq, max_freq):
    """Pack min/max records into CSR flat arrays (inflection and widths are not persisted)."""
    buckets = {
        kind: {'w': [], 'p': [], 'f': [], 'off': [0]}
        for kind in ('min', 'max')
    }
    for i in range(n_spectra):
        record = records[i] if i < len(records) else None
        if record is not None:
            for kind, key in (('min', 'minima'), ('max', 'maxima')):
                bucket = buckets[kind]
                for feat in record[key]:
                    bucket['w'].append(feat['wavelength'])
                    bucket['p'].append(feat['prominence'])
                    bucket['f'].append(feat.get('fake', False))
        for kind in buckets:
            buckets[kind]['off'].append(len(buckets[kind]['w']))

    sidecar = {'min_freq': float(min_freq), 'max_freq': float(max_freq)}
    for kind in ('min', 'max'):
        bucket = buckets[kind]
        sidecar.update({
            '%s_offsets' % kind: np.asarray(bucket['off'], dtype=np.int32),
            '%s_wavelength' % kind: np.asarray(bucket['w'], dtype=np.float16),
            '%s_prominence' % kind: np.asarray(bucket['p'], dtype=np.float16),
            '%s_fake' % kind: np.asarray(bucket['f'], dtype=np.uint8),
        })
    return sidecar


def _extremaNSpectra(sidecar):
    """Return the number of spectra stored in a CSR extrema sidecar."""
    return int(sidecar['min_offsets'].shape[0] - 1)


def _extremaKindSlice(sidecar, i, kind):
    """Return one spectrum's min or max features as sliced CSR arrays."""
    offsets = sidecar['%s_offsets' % kind]
    lo, hi = int(offsets[i]), int(offsets[i + 1])
    return {
        'wavelength': sidecar['%s_wavelength' % kind][lo:hi],
        'prominence': sidecar['%s_prominence' % kind][lo:hi],
        'fake': sidecar['%s_fake' % kind][lo:hi].astype(bool),
    }


def _extremaSidecarFromNpz(blob):
    """Load CSR extrema sidecar arrays from an archive ext_* group."""
    if isinstance(blob, dict):
        return {k: v for k, v in blob.items()}
    return {k: blob[k] for k in blob.files}


def _recordsFromExtremaSidecar(sidecar):
    """Rebuild per-spectrum extrema record dicts from a CSR sidecar."""
    n = _extremaNSpectra(sidecar)
    records = [None] * n
    for i in range(n):
        minima, maxima = [], []
        for kind, bucket in (('min', minima), ('max', maxima)):
            sl = _extremaKindSlice(sidecar, i, kind)
            for j in range(len(sl['wavelength'])):
                bucket.append({
                    'wavelength': float(sl['wavelength'][j]),
                    'prominence': float(sl['prominence'][j]),
                    'fake': bool(sl['fake'][j]),
                    'index': 0.0,
                    'value': 0.0,
                    'left_width': 0.0,
                    'right_width': 0.0,
                })
        records[i] = {'minima': minima, 'maxima': maxima, 'inflection': []}
    return records


def _packKdeSidecar(ext_sidecar, sigma, minw, maxW, min_prominence, minima, maxima, min_freq, max_freq):
    """Build in-memory KDE arrays from CSR extrema (not written to disk)."""
    n = _extremaNSpectra(ext_sidecar)
    entries = []
    for i in range(n):
        row = []
        for kind_code, kind_name, enabled in ((0, 'min', minima), (1, 'max', maxima)):
            if not enabled:
                continue
            sl = _extremaKindSlice(ext_sidecar, i, kind_name)
            for j in range(len(sl['wavelength'])):
                if sl['fake'][j]:
                    continue
                prom = float(sl['prominence'][j])
                mu = float(sl['wavelength'][j])
                if prom < min_prominence or mu < minw or mu > maxW:
                    continue
                row.append((mu, prom, kind_code))
        entries.append(row)

    cap = max((len(r) for r in entries), default=1)
    mu = np.zeros((n, cap), dtype=np.float64)
    weight = np.zeros((n, cap), dtype=np.float64)
    kind = np.full((n, cap), -1, dtype=np.int8)
    n_kde = np.zeros(n, dtype=np.int32)
    for i, row in enumerate(entries):
        n_kde[i] = min(len(row), cap)
        for j, (m, w, kcode) in enumerate(row[:cap]):
            mu[i, j] = m
            weight[i, j] = w
            kind[i, j] = kcode

    key = (float(minw), float(maxW), float(sigma), float(min_prominence),
           float(min_freq), float(max_freq), bool(minima), bool(maxima))
    return {
        'mu': mu, 'weight': weight, 'kind': kind, 'n_kde': n_kde,
        'sigma': float(sigma), 'key': key,
    }


def _kdeSidecarToGaussians(sidecar, index=None):
    """Convert a packed KDE sidecar to Gaussian feature dictionaries."""
    kind_names = ('minimum', 'maximum')
    sigma = sidecar['sigma']
    cap = sidecar['mu'].shape[1]

    def row_to_list(i):
        out = []
        for j in range(int(sidecar['n_kde'][i])):
            if j >= cap or sidecar['kind'][i, j] < 0:
                continue
            out.append({
                'mu': float(sidecar['mu'][i, j]),
                'sigma': sigma,
                'weight': float(sidecar['weight'][i, j]),
                'kind': kind_names[int(sidecar['kind'][i, j])],
            })
        return out

    if index is not None:
        return row_to_list(int(index))
    return [row_to_list(i) for i in range(sidecar['mu'].shape[0])]


def _evalKdeGrid(sidecar, wav, index=None):
    """Evaluate cached KDE Gaussians on a wavelength grid."""
    wav = np.asarray(wav, dtype=np.float64)
    sigma = sidecar['sigma']
    inv_s = 1.0 / (sigma * np.sqrt(2.0 * np.pi))
    mu = sidecar['mu']
    weight = sidecar['weight']
    n_kde = sidecar['n_kde']
    cap = mu.shape[1]

    def eval_row(i):
        out = np.zeros(wav.shape[0], dtype=np.float64)
        for j in range(min(int(n_kde[i]), cap)):
            z = (wav - mu[i, j]) / sigma
            out += weight[i, j] * np.exp(-0.5 * z * z) * inv_s
        return out

    if index is not None:
        return eval_row(int(index))
    return np.stack([eval_row(i) for i in range(mu.shape[0])], axis=0)


def _gaussianWindowLikelihoodBatch(kde_sidecar, lo, hi, kind):
    """Integrate weighted Gaussian kernels over [lo, hi] for all spectra at once."""
    kind_code = 0 if kind == 'minimum' else 1
    mu = kde_sidecar['mu']
    weight = kde_sidecar['weight']
    kind_arr = kde_sidecar['kind']
    sigma = kde_sidecar['sigma']
    n_kde = kde_sidecar['n_kde']
    cap = mu.shape[1]
    j_idx = np.arange(cap)[None, :]
    active = (j_idx < n_kde[:, None]) & (kind_arr == kind_code)
    cdf_hi = _gaussianCdf(hi, mu, sigma)
    cdf_lo = _gaussianCdf(lo, mu, sigma)
    contrib = weight * (cdf_hi - cdf_lo) * active
    return contrib.sum(axis=1)


def _fillNans(spectrum):
    """Linearly interpolate missing values in a 1D spectrum."""
    s = np.asarray(spectrum, dtype=np.float32)
    finite = np.isfinite(s)
    if finite.all():
        return s
    idx = np.arange(s.size, dtype=np.float32)
    if finite.sum() < 3:
        return np.full(s.size, np.nan, dtype=np.float32)
    out = s.copy()
    out[~finite] = np.interp(idx[~finite], idx[finite], s[finite])
    return out


def _bandLimitFftBatch(spectra, min_freq, max_freq, padding='reflect'):
    """Band-limit spectra with FFT, returning complex coefficients and geometry metadata."""
    y = np.asarray(spectra, dtype=np.float32)
    n = y.shape[1]
    M_max = max(1, int(np.floor(max_freq * (n // 2))))
    M_min = int(np.floor(min_freq * (n // 2)))
    pad = min(M_max, max(0, n - 2))

    # pad ends to reduce Gibbs ringing before the FFT
    if pad > 0:
        if padding == 'reflect':
            y = np.pad(y, ((0, 0), (pad, pad)), mode='reflect')
        elif padding == 'cosine':
            y = np.pad(y, ((0, 0), (pad, pad)), mode='reflect')
            ramp = np.float32(0.5) * (np.float32(1.0) - np.cos(np.linspace(0.0, np.pi, pad, dtype=np.float32)))
            y[:, :pad] *= ramp
            y[:, -pad:] *= ramp[::-1]
        else:
            raise ValueError('padding must be "cosine" or "reflect"')

    n_work = y.shape[1]
    X = np.fft.fft(y, axis=1)
    # zero bins outside the requested normalised frequency band
    freq_bins = np.zeros(n_work, dtype=bool)
    for k in range(-M_max, M_max + 1):
        if abs(k) >= M_min:
            freq_bins[k % n_work] = True
    X_lp = X * freq_bins

    # extract symmetric coefficient vector and normalise like np.fft.ifft
    n_coeff = 2 * M_max + 1
    out = np.zeros((y.shape[0], n_coeff), dtype=np.complex64)
    for k in range(-M_max, M_max + 1):
        if abs(k) >= M_min:
            out[:, k + M_max] = X_lp[:, k % n_work] / n_work

    geometry = {'n_samples': n, 'n_work': n_work, 'pad': pad, 'M': M_max, 'M_min': M_min}
    return out, geometry


def _trigPowers(t, k, N):
    """Return exp(2 pi i k t / N) for vectorised Fourier-series evaluation."""
    t = np.asarray(t, dtype=np.float32)
    k = np.asarray(k, dtype=np.float32)
    N = np.float32(N)
    return np.exp(2j * np.pi * t[..., None] * k[None, :] / N).astype(np.complex64)


def _evalTrig(coeffs, zpow):
    """Evaluate the real part of a Fourier series at precomputed trig powers."""
    coeffs = np.asarray(coeffs, dtype=np.complex64)
    zpow = np.asarray(zpow, dtype=np.complex64)
    if coeffs.ndim == 1:
        return np.real((coeffs * zpow).sum(axis=-1)).astype(np.float32)
    return np.real(coeffs @ zpow.T).astype(np.float32)


_erf = np.vectorize(_math_erf, otypes=[np.float64])


def _gaussianCdf(x, mu, sigma):
    """Evaluate the Gaussian CDF used by windowed search likelihoods."""
    z = (np.asarray(x, dtype=np.float64) - mu) / (sigma * np.sqrt(2.0))
    return 0.5 * (1.0 + _erf(z))


def _formatArchiveSampleName(archive_key, sample_name):
    """Build an archive-qualified label: `(key) sample_name`."""
    return '(%s) %s' % (archive_key, sample_name)


def _parseArchiveSampleName(name):
    """Split `(key) sample_name` into archive key and inner sample label."""
    if not isinstance(name, str):
        raise TypeError('Archive sample name must be a string.')
    match = re.match(r'^\(([^)]+)\)\s+(.*)$', name.strip())
    if not match:
        raise ValueError(
            'Expected archive-qualified name like "(key) sample", got %r.' % name
        )
    return match.group(1), match.group(2)


def _sampleNames(header, n_spectra, original_shape, spatial_shape):
    """Build per-spectrum labels for :meth:`HyFourier.search` from header metadata and layout."""
    original_shape = tuple(original_shape)
    spatial_shape = tuple(spatial_shape)

    if header is not None and 'sample names' in header:
        raw = header.get_list('sample names', str)
        names = [str(v) for v in np.atleast_1d(raw).tolist()]
        names = _alignSampleNames(names, n_spectra, spatial_shape)
    else:
        names = _indexedSampleNames(n_spectra, original_shape, spatial_shape, header)

    if header is not None:
        names = _applyGroupPrefixes(names, header)
    return names


def _alignSampleNames(names, n_spectra, spatial_shape):
    """Match header sample-name count to flattened spectrum count."""
    if len(names) == n_spectra:
        return list(names)
    if spatial_shape and len(names) == spatial_shape[0]:
        per_sample = int(np.prod(spatial_shape[1:])) if len(spatial_shape) > 1 else 1
        if len(names) * per_sample == n_spectra:
            return [name for name in names for _ in range(per_sample)]
    if len(names) >= n_spectra:
        return list(names[:n_spectra])
    return list(names) + ['S%d' % i for i in range(len(names), n_spectra)]


def _indexedSampleNames(n_spectra, original_shape, spatial_shape, header):
    """Fallback labels from data shape when no sample names are stored."""
    if len(original_shape) == 2:
        return ['S%d' % i for i in range(n_spectra)]
    if len(original_shape) == 3 and _imageSpatialLayout(spatial_shape, header):
        return ['(%d,%d)' % np.unravel_index(i, spatial_shape) for i in range(n_spectra)]
    return ['S%d' % i for i in range(n_spectra)]


def _imageSpatialLayout(spatial_shape, header):
    """True when flattened spectra correspond to a 2D image grid."""
    if len(spatial_shape) != 2 or spatial_shape[1] <= 1:
        return False
    if header is not None and 'library' in str(header.get('file type', '')).lower():
        return False
    return True


def _applyGroupPrefixes(names, header):
    """Prepend `[group]` to spectra listed under `group <name>` header keys."""
    names = list(names)
    for key in header:
        if not str(key).startswith('group '):
            continue
        group_name = str(key)[6:].strip()
        ids = header.get_list(key, str)
        if ids is None:
            continue
        prefix = '[%s] ' % group_name
        for value in np.atleast_1d(ids).ravel():
            idx = _resolveGroupIndex(value, names)
            if idx is not None and 0 <= idx < len(names):
                names[idx] = prefix + names[idx]
    return names


def _resolveGroupIndex(value, names):
    """Map a group entry (index or sample name) to a flat spectrum index."""
    if isinstance(value, (str, np.str_)) and value in names:
        return names.index(value)
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None


def _normalizeSampleNameQueries(name):
    """Normalise sample-name query string(s) for near-exact matching."""
    if isinstance(name, str):
        queries = [name.strip()]
    elif isinstance(name, (list, tuple)):
        queries = [str(n).strip() for n in name]
    else:
        raise TypeError('name must be a string or list of strings.')
    if not queries or not any(q for q in queries):
        raise ValueError('name must be a non-empty string or list of strings.')
    return queries


def _parseOptionalArchivePrefix(name):
    """Split an optional ``(key)`` prefix from a sample label query."""
    name = str(name).strip()
    match = re.match(r'^\(([^)]+)\)\s+(.*)$', name)
    if match:
        return match.group(1), match.group(2).strip()
    return None, name


def _stripGroupPrefix(name):
    """Remove a leading ``[group]`` prefix from a sample label."""
    name = str(name).strip()
    match = re.match(r'^\[([^\]]+)\]\s+(.*)$', name)
    if match:
        return match.group(2).strip()
    return name


def _displayNameMatchesQuery(display_name, query):
    """Return True when `query` near-exactly matches a display or bare sample label."""
    display_name = str(display_name).strip()
    query = str(query).strip()
    if not query:
        return False
    if display_name.lower() == query.lower():
        return True
    bare_display = _stripGroupPrefix(display_name)
    if bare_display.lower() == query.lower():
        return True
    bare_query = _stripGroupPrefix(query)
    if bare_query.lower() != query.lower():
        return bare_display.lower() == bare_query.lower()
    return False


def _archiveDisplayNameMatchesQuery(archive_key, display_name, query):
    """Return True when `query` near-exactly matches an archive-qualified label."""
    query = str(query).strip()
    query_key, inner_query = _parseOptionalArchivePrefix(query)
    if query_key is not None and query_key != archive_key:
        return False
    qualified = _formatArchiveSampleName(archive_key, display_name)
    if qualified.lower() == query.lower():
        return True
    return _displayNameMatchesQuery(display_name, inner_query)


def _split_or_queries(query):
    """Split a search string on ``|`` into OR sub-queries."""
    parts = [part.strip() for part in str(query).split('|')]
    return [part for part in parts if part]


def _merge_or_search_results(query_results, n_result=None):
    """Interleave ranked results from OR sub-queries, preserving each branch's order."""
    best_rank = {}
    for names, scores in query_results:
        for rank, name in enumerate(names):
            score = float(scores[rank])
            if name not in best_rank or rank < best_rank[name][0]:
                best_rank[name] = (rank, score)

    emitted = set()
    merged_names = []
    merged_scores = []
    max_len = max((len(names) for names, _ in query_results), default=0)

    for rank_index in range(max_len):
        for names, _scores in query_results:
            if rank_index >= len(names):
                continue
            name = names[rank_index]
            if name in emitted:
                continue
            emitted.add(name)
            _rank, score = best_rank[name]
            merged_names.append(name)
            merged_scores.append(score)

    merged_scores = np.asarray(merged_scores, dtype=np.float64)
    if n_result is not None and len(merged_names) > int(n_result):
        merged_names = merged_names[: int(n_result)]
        merged_scores = merged_scores[: int(n_result)]
    return merged_names, merged_scores


def _searchableSampleName(sample_name, archive_key=None):
    """Build the case-folded string used for name-token substring matching."""
    sample_name = str(sample_name).strip()
    if archive_key is not None:
        return _formatArchiveSampleName(archive_key, sample_name).lower()
    return sample_name.lower()


def _nameMatch(name, patterns, archive_key=None):
    """Return the fraction of name tokens matched as case-insensitive substrings."""
    if not patterns:
        return 0.0
    name_l = _searchableSampleName(name, archive_key)
    matched = sum(1 for pattern in patterns if pattern in name_l)
    return matched / len(patterns)


def _hyfourier_search_single(
    hyfourier,
    query,
    confidence=10.0,
    n_result=10,
    minw=None,
    maxW=None,
    min_freq=None,
    max_freq=None,
    vb=False,
    archive_key=None,
):
    """Run one AND-combined search on a single :class:`HyFourier` instance."""
    confidence = float(confidence)
    if confidence <= 0:
        raise ValueError('confidence must be positive.')

    features, name_patterns = _parseSearchQuery(query)
    if not features and not name_patterns:
        raise ValueError('Could not parse any features or names from query: %r' % query)

    names = _sampleNames(
        hyfourier.header,
        hyfourier.n_spectra,
        hyfourier.original_shape,
        hyfourier.spatial_shape,
    )
    minw = hyfourier.wav_range[0] if minw is None else float(minw)
    maxW = hyfourier.wav_range[1] if maxW is None else float(maxW)
    min_freq = hyfourier.min_freq if min_freq is None else float(min_freq)
    max_freq = hyfourier.max_freq if max_freq is None else float(max_freq)
    kde_sidecar = hyfourier._getKDE(
        sigma=confidence, minw=minw, maxW=maxW, min_freq=min_freq, max_freq=max_freq, vb=vb,
    )

    p = np.ones(hyfourier.n_spectra, dtype=np.float64)
    p[~hyfourier._valid] = 0.0
    if name_patterns:
        for i, name in enumerate(names):
            if hyfourier._valid[i]:
                p[i] *= _nameMatch(name, name_patterns, archive_key=archive_key)
            else:
                p[i] = 0.0

    for feat in features:
        if 'point' in feat:
            lo = feat['point'] - confidence
            hi = feat['point'] + confidence
        else:
            lo, hi = feat['lo'], feat['hi']
        lik = _gaussianWindowLikelihoodBatch(kde_sidecar, lo, hi, feat['kind'])
        if feat['exclude']:
            lik = 1.0 - lik
        p *= lik

    p[~np.isfinite(p)] = 0.0

    n_result = min(int(n_result), hyfourier.n_spectra)
    idx = np.argsort(p)[::-1][:n_result]
    return [names[i] for i in idx], p[idx]


def _fourier_archive_search_merged(
    archive,
    query,
    confidence=10.0,
    n_result=10,
    minw=None,
    maxW=None,
    min_freq=None,
    max_freq=None,
    vb=False,
):
    """Search all archive entries for one AND-combined sub-query."""
    merged_names = []
    merged_scores = []
    for key, hyfourier in archive._entries.items():
        names, scores = _hyfourier_search_single(
            hyfourier,
            query,
            confidence=confidence,
            n_result=n_result,
            minw=minw,
            maxW=maxW,
            min_freq=min_freq,
            max_freq=max_freq,
            vb=vb,
            archive_key=key,
        )
        for name, score in zip(names, scores):
            merged_names.append(_formatArchiveSampleName(key, name))
            merged_scores.append(float(score))
    if not merged_names:
        return [], np.asarray([], dtype=np.float64)
    merged_scores = np.asarray(merged_scores, dtype=np.float64)
    order = np.argsort(merged_scores)[::-1][: int(n_result)]
    return [merged_names[i] for i in order], merged_scores[order]


def _parseSearchQuery(query):
    """Parse a search string into feature constraints and name patterns."""
    features = []
    names = []
    range_re = re.compile(r'^\d+(?:\.\d+)?-\d+(?:\.\d+)?$')
    number_re = re.compile(r'^\d+(?:\.\d+)?$')

    for token in query.strip().split():
        # optional ! (exclude) and ^ (peak) prefixes before numeric or name tokens
        exclude = False
        if token.startswith('!'):
            exclude = True
            token = token[1:]
        peak = False
        if token.startswith('^'):
            peak = True
            token = token[1:]
        if not token:
            continue

        kind = 'maximum' if peak else 'minimum'
        if range_re.match(token):
            # wavelength interval query, e.g. 2160-2200
            parts = token.split('-')
            lo, hi = float(parts[0]), float(parts[1])
            if lo > hi:
                lo, hi = hi, lo
            features.append({'kind': kind, 'exclude': exclude, 'lo': lo, 'hi': hi})
        elif number_re.match(token):
            # single-wavelength query; search() expands point ± confidence
            w = float(token)
            features.append({
                'kind': kind, 'exclude': exclude,
                'lo': w - 0.0, 'hi': w + 0.0, 'point': w,
            })
        else:
            names.append(token.lower())

    return features, names
