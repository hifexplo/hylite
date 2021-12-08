import numpy as np

from hylite import io, HyData


def saveMultiMWL( path, mwl ):
    """
    A utility function for compressing multi-mwl results into a single image/cloud and saving them.

    *Arguments*:
     - path = the path to save to.
     - mwl = a list of MWL HyData instances, as returned by minimum_wavelength when n > 1.
    """

    # check mwl is a list
    assert isinstance(mwl, list), "Error - mwl must be a list. Perhaps you're not passing a multiMWL result?"
    # stack mwl data into single HyData instance
    out = mwl[0].copy(data=False)
    if mwl[0].is_image():
        out.data = np.dstack( [m.data for m in mwl])
    else:
        out.data = np.hstack( [m.data for m in mwl])

    # add flag to header that this is a MultiMWL
    out.header['multimwl'] = 'true'

    # save
    io.save( path, out )


def loadMultiMWL( path ):
    """
    A utility function for loading multi-mwl dataset as saved with saveMultiMWL(...) and converting them to a
    list of minimum wavelength maps.

    *Arguments*:
     - path = the file to load. Must point to a multiMWL dataset saved with saveMultiMWL(...). Can also be a HyData instance
              that needs to be split into a list (used by io.load to avoid infinite recursion).

    *Returns*: A list of mwl datasets.
    """

    # load stacked dataset
    if isinstance(path, str):
        return io.load( path ) # io.load will handle loading and pass a HyData instance back to us
    elif isinstance(path, HyData):
        _mwl = path
    else:
        assert False, "Error - %s is an unsupported type for path" % type(path)

    # check header flag
    if not 'true' in _mwl.header.get('multimwl', 'false').lower():
        print("Warning - dataset may not be a MultiMWL dataset (the isMultiMWL flag is not in the header). Proceed with care!")

    # split
    mwl = []
    assert _mwl.band_count() % 4 == 0, "Error - invalid number of bands for a multi-mwl dataset?"
    n = int(_mwl.band_count() / 4)
    for i in range(0, n):
        out = _mwl.copy(data=False)
        out.header.drop_all_bands()
        out.data = _mwl.data[..., [4*i, 4*i+1, 4*i+2, 4*i+3]]
        out.set_band_names(['pos', 'width', 'depth', 'strength'])
        out.push_to_header()
        mwl.append(out)
    if len(mwl) == 1:
        return mwl[0] # drop list and just return mwl object
    return mwl #otherwise, return list