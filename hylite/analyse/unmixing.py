"""
Functions for linear unmixing using endmember spectra.

`unmix(..., method='nnls')` uses SciPy only. `method='fcls'` and `endmembers()`
require `pysptools` (FCLS uses CVXOPT via pysptools).
"""
import math
import random
import numpy as np
from hylite import HyLibrary, HyData
from hylite._deps import require

_nfindr_patched = False

# monkey patch to fix NDFINDR function in pysptools on newer numpy versions
def _patch_nfindr():
    """Replace pysptools NFINDR with a numpy-compatible implementation."""
    global _nfindr_patched
    if _nfindr_patched:
        return

    pysptools = require("pysptools")
    from pysptools.eea import eea

    def NFINDR(data, q, transform=None, maxit=None, ATGP_init=False):
        """N-FINDR endmembers induction algorithm."""
        nsamples, nvariables = data.shape

        if maxit is None:
            maxit = 3 * q

        if transform is None:
            transform = eea._PCA_transform(data, q - 1)

        TestMatrix = np.zeros((q, q), dtype=np.float32)
        TestMatrix[0, :] = 1
        IDX = np.zeros(q, dtype=np.int64)

        if ATGP_init:
            induced_em, idx = eea.ATGP(transform, q)
            IDX = np.array(idx, dtype=np.int64)
            for i in range(q):
                TestMatrix[1:q, i] = induced_em[i]
        else:
            for i in range(q):
                idx = int(math.floor(random.random() * nsamples))
                TestMatrix[1:q, i] = transform[idx]
                IDX[i] = idx

        actualVolume = 0
        it = 0
        v1 = -1.0
        v2 = actualVolume

        while it <= maxit and v2 > v1:
            for k in range(q):
                for i in range(nsamples):
                    TestMatrix[1:q, k] = transform[i]
                    volume = abs(np.linalg.det(TestMatrix))
                    if volume > actualVolume:
                        actualVolume = volume
                        IDX[k] = i
                TestMatrix[1:q, k] = transform[IDX[k]]
            it += 1
            v1 = v2
            v2 = actualVolume

        E = np.zeros((len(IDX), nvariables), dtype=np.float32)
        Et = np.zeros((len(IDX), q - 1), dtype=np.float32)
        for j in range(len(IDX)):
            E[j] = data[IDX[j]]
            Et[j] = transform[IDX[j]]

        return E, Et, IDX, it

    import pysptools.eea.nfindr as _nfindr_mod
    _nfindr_mod.NFINDR = NFINDR
    _nfindr_patched = True

# linear mixing based on abundance and endmember arrays
def mix(abundances: HyData, endmembers: np.ndarray):
    """
    Generate synthetic spectra by linearly mixing an abundance and endmember
    matrix.

    Args:
        abundances: a `hylite.hydata.HyData` instance (e.g. image or cloud) with bands representing abundances (typically from 0 to 1).
        endmembers: A numpy array of shape (nendmembers, bands), or `hylite.hylibrary.HyLibrary` instance
                    containing these same endmembers.
    Returns:
        A `hylite.hydata.HyData` instance with the same type as abundances, but containing the forward modelled (linearly mixed) spectra.
    """

    out = abundances.copy()
    A = abundances.X(onlyFinite=True)

    if isinstance(endmembers, HyLibrary):
        E = np.mean(endmembers.data, axis=1)

    X = A @ E

    out.set_raveled(X, onlyFinite=True, strict=False)
    if isinstance(endmembers, HyLibrary):
        out.set_wavelengths(endmembers.get_wavelengths())
    return out


def _nnls(spectra, endmembers):
    """
    Non-negative least squares abundance estimation.

    Args:
        spectra: (N, p) pixel spectra.
        endmembers: (q, p) endmember signatures.

    Returns:
        (N, q) abundance estimates.
    """
    nnls = require("scipy.optimize").nnls
    n, _ = spectra.shape
    q, _ = endmembers.shape
    gram = endmembers @ endmembers.T
    abundances = np.zeros((n, q), dtype=np.float32)
    for i in range(n):
        abundances[i] = nnls(gram, endmembers @ spectra[i])[0]
    return abundances

# linear unmixing to estimate abundances given measured spectra and endmembers
def unmix(data: HyData, endmembers: np.ndarray, method: str = 'nnls'):
    """
    Perform least squares unmixing to estimate linear combinations
    of the specified endmembers that best reproduce the observed data.

    Args:
        data: a `hylite.hydata.HyData` instance (e.g. image or cloud) to unmix.
        endmembers: A numpy array of shape (nendmembers, bands), or `hylite.hylibrary.HyLibrary` instance
                    containing these same endmembers.
        method: the unmixing constraints. Options are non-negative ('nnls', default; SciPy
                only) or fully constrained ('fcls'; requires `pysptools`).
    Returns:
        A `hylite.hydata.HyData` instance with the same type as data, but containing the estimated endmember abundances.
    """

    out = data.copy()
    X = data.X(onlyFinite=True)

    if isinstance(endmembers, HyLibrary):
        E = np.mean(endmembers.data, axis=1)
    else:
        E = endmembers
    assert X.shape[-1] == E.shape[-1], "Endmembers have %d bands, data has %d." % (E.shape[-1], X.shape[-1])

    if 'nnls' in method.lower():
        A = _nnls(X, E)
    elif 'fcls' in method.lower():
        amaps = require("pysptools.abundance_maps.amaps")
        A = amaps.FCLS(X, E)
    else:
        assert False, "%s is an unknown unmixing method. Should be 'nnls' or 'fcls'." % method

    out.set_raveled(A, onlyFinite=True, strict=False)
    if isinstance(endmembers, HyLibrary) and endmembers.has_band_names():
        out.set_band_names(endmembers.get_band_names())
    else:
        out.set_band_names(["EM%d" % (i + 1) for i in range(E.shape[0])])
    return out


def endmembers(data: HyData, n: int, method: str = 'nfindr', **kwds):
    """
    Use endmember identification methods implemented in pysptools to find candidate
    "pure" pixels to use as endmembers. Note that these should always be manually vetted
    as they represent, in essence, outliers.

    Args:
        data: a `hylite.hydata.HyData` instance (e.g. image or cloud) to unmix.
        n: The number of endmembers to find.
        method: The endmember identification method. These are fully documented in the
                pysptools documentation, and can be one of: 'nfindr', '' ...
    Returns:
        library: A `hylite.hylibrary.HyLibrary` containing the identified endmembers.
        indices: A numpy array containing the coordinates of the selected endmembers in the input data.
    """

    require("pysptools")

    ix = np.indices(data.data.shape[:-1], )
    mask = np.isfinite(data.data).all(axis=-1)
    X = data.data[mask, :]
    ix = ix[:, mask].T

    np.int = int  # hack needed to deal with some numpy versions... (and avoid a pysptools bug)
    if 'atgp' in method.lower():
        from pysptools.eea.eea import ATGP
        em, im = ATGP(X, n)
    elif 'fippi' in method.lower():
        from pysptools.eea.eea import FIPPI
        em, im = FIPPI(X, q=n, **kwds)
    elif 'nfindr' in method.lower():
        _patch_nfindr()
        from pysptools.eea import NFINDR
        N = NFINDR()
        em = N.extract(X[:, None, :], q=n, **kwds)
        im = np.array(N.get_idx())[:, 1]
    elif 'ppi' in method.lower():
        from pysptools.eea import PPI
        P = PPI()
        em = P.extract(X[:, None, :], q=n, **kwds)
        im = np.array(P.get_idx())[:, 1]
    else:
        assert False, "%s is an unknown endmember selection method. Options are 'atgp', 'fippi', 'nfindr' and 'ppi'." % method

    im = np.array([ix[i, :] for i in im]).squeeze()

    out = HyLibrary(np.array(em), lab=['EM%d' % (i + 1) for i in range(len(em))], wav=data.get_wavelengths())
    return out, im
