"""
Functions for linear unmixing using endmember spectra.

Note that some of these rely on functions implemented in `pysptools`, which thus
may need to be installed for them to work.
"""
import numpy as np
from hylite import HyLibrary, HyData

def mix( abundances : HyData, endmembers : np.ndarray ):
    """
    Generate synthetic spectra by linearly mixing an abundance and endmember
    matrix. 

    Args:
        abundances: a HyData instance (e.g. image or cloud) with bands representing abundances (typically from 0 to 1).
        endmembers: A numpy array of shape (nendmembers, bands), or HyLibrary instance
                    containing these same endmembers.
    Returns:
        A HyData instance with the same type as abundances, but containing the forward modelled (linearly mixed) spectra.
    """

    # create output object and get data vector to unmix (without nans)
    out = abundances.copy()
    A = abundances.X(onlyFinite=True)

    # get endmembers as numpy array
    if isinstance( endmembers, HyLibrary):
        E = np.mean( endmembers.data, axis=1 )

    # do forward mixing
    X = A @ E

    # put into output object and return
    out.set_raveled( X, onlyFinite=True, strict=False)
    if isinstance( endmembers, HyLibrary):
        out.set_wavelengths( endmembers.get_wavelengths() )
    return out

def unmix( data : HyData, endmembers : np.ndarray, method : str = 'nnls' ):
    """
    Perform least squares unmixing to estimate linear combinations
    of the specified endmembers that best reproduce the observed data.

    Args:
        data: a HyData instance (e.g. image or cloud) to unmix.
        endmembers: A numpy array of shape (nendmembers, bands), or HyLibrary instance
                    containing these same endmembers.
        method: the unmixing constraints. Options are non-negative ('nnls', default) or fully
                constrained ('fcls'). 
    Returns:
        A HyData instance with the same type as data, but containing the estimated endmember abundances.
    """

    # create output object and get data vector to unmix (without nans)
    out = data.copy()
    X = data.X(onlyFinite=True)
    
    # get endmembers as numpy array
    if isinstance( endmembers, HyLibrary):
        E = np.mean( endmembers.data, axis=1 )
    else:
        E = endmembers # data is a numpy array?
    assert X.shape[-1] == E.shape[-1], "Endmembers have %d bands, data has %d." % (E.shape[-1], X.shape[-1])

    # do unmixing (using pysptools)
    # todo - probably we could do this ourselves using scipy.optimize? 
    try:
        from pysptools.abundance_maps.amaps import NNLS, FCLS
    except:
        assert False, "pysptools is not install. Try `pip install pysptools`."
    if 'nnls' in method.lower():
        A = NNLS(X, E)
    elif 'fcls' in method.lower():
        A = FCLS(X, E)
    else:
        assert False, "%s is an unknown unmixing method. Should be 'nnls' or 'fcls'."%method
    
    # put into output object and return
    out.set_raveled( A, onlyFinite=True, strict=False)
    if isinstance( endmembers, HyLibrary) and endmembers.has_band_names():
        out.set_band_names( endmembers.get_band_names() )
    else:
        out.set_band_names(["EM%d"%(i+1) for i in range(E.shape[0])])
    return out

def endmembers( data : HyData, n : int, method : str = 'nfindr', **kwds):
    """
    Use endmember identification methods implemented in pysptools to find candidate 
    "pure" pixels to use as endmembers. Note that these should always be manually vetted
    as they represent, in essence, outliers.

    Args:
        data: a HyData instance (e.g. image or cloud) to unmix.
        n: The number of endmembers to find.
        method: The endmember identification method. These are fully documented in the 
                pysptools documentation, and can be one of: 'nfindr', '' ...
    Returns:
        library: A HyLibrary containing the identified endmembers. 
        indices: A numpy array containing the coordinates of the selected endmembers in the input data.
    """
    
    # create data and corresponding index array (and drop nans)
    ix = np.indices( data.data.shape[:-1], )
    mask = np.isfinite( data.data ).all(axis=-1)
    X = data.data[mask, : ] # get data vector
    ix = ix[:, mask].T # get original indices
    
    # find endmembers
    try:
        import pysptools
    except:
        assert False, "pysptools is not install. Try `pip install pysptools`."
    np.int = int # hack needed to deal with some numpy versions... (and avoid a pysptools bug)
    if 'atgp' in method.lower():
        from pysptools.eea.eea import ATGP
        em, im = ATGP(X, n)
    elif 'fippi' in method.lower():
        from pysptools.eea.eea import FIPPI
        em, im = FIPPI(X, q=n, **kwds)
    elif 'nfindr' in method.lower():
        from pysptools.eea import NFINDR
        N = NFINDR()
        em = N.extract(X[:,None,:], q=n, **kwds)
        im = np.array(N.get_idx())[:,1]
    elif 'ppi' in method.lower():
        from pysptools.eea import PPI
        P = PPI()
        em = P.extract(X[:,None,:], q=n, **kwds)
        im = np.array(P.get_idx())[:,1]
    else:
        assert False, "%s is an unknown endmember selection method. Options are 'atgp', 'fippi', 'nfindr' and 'ppi'."%method
    
    # sort out indices
    im = np.array([ix[i,:] for i in im]).squeeze()
    
    # build output library
    out = HyLibrary( np.array(em), lab=['EM%d'%(i+1) for i in range(len(em))], wav=data.get_wavelengths())
    return out, im

