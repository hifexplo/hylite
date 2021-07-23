import numpy as np
import os

from hylite.project import PMap

def savePMap(path, pmap):
    """
    Save a PMap instance using numpy.

    *Arguments*:
     - path = the path to save to.
     - pmap = the PMap instance to save.
    """
    pnt,pix,z = pmap.get_flat()
    dims = np.array([pmap.xdim, pmap.ydim, pmap.npoints])
    np.savez_compressed( path, dims=dims, points=pnt, pixels=pix, depths=z )

def loadPMap(path):
    """
    Load a PMap instance using numpy.

    *Arguments*:
     - path = the file path to load from.

    *Returns*:
     - a PMap instance loaded from the file.
    """

    # check extension
    if not os.path.exists(path):
        path += ".npz" # try adding npz extension to see if that helps

    assert os.path.exists(path), "Error - file not found: %s" % path

    # load data
    data = np.load( path )

    # check attributes
    if 'dims' not in data:
        data.close()
        assert False, "Error - npz does not contain a 'dims' attribute."
    if 'points' not in data:
        data.close()
        assert False, "Error - npz does not contain a 'points' attribute."
    if 'pixels' not in data:
        data.close()
        assert False, "Error - npz does not contain a 'pixels' attribute."
    if 'depths' not in data:
        data.close()
        assert False, "Error - npz does not contain a 'depths' attribute."

    # extract attrubites
    xdim, ydim, npoints = data[ "dims" ]
    points = data[ "points" ]
    pixels = data[ "pixels" ]
    depths = data[ "depths" ]

    # close file
    data.close()

    # create new PMap and populate with data
    pm = PMap( xdim, ydim, npoints )
    pm.set_flat( points, pixels, depths )
    return pm