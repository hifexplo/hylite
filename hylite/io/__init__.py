
"""
Import and export hyperspectral data. For hyperspectral images this is mostly done using GDAL,
while for point clouds and hyperspectral libraries a variety of different methods are included.
"""
import os
from .headers import *
from .images import *
from .clouds import *
from .libraries import *
from .pmaps import *

from hylite import HyData, HyImage, HyCloud, HyLibrary, HyCollection
from hylite.project import PMap

def save(path, data, **kwds):
    """
    A generic function for saving HyData instances such as HyImage, HyLibrary and HyCloud. The appropriate file format
    will be chosen automatically.

    *Arguments*:
     - path = the path to save the file too.
     - data = the data to save. This must be an instance of HyImage, HyLibrary or HyCloud.

    *Keywords*:
     - vmin = the data value that = 0 when saving RGB images.
     - vmax = the data value that = 255 when saving RGB images. Must be > vmin.
    """

    if isinstance(data, HyImage):

        # special case - save ternary image to png or jpg or bmp
        ext = os.path.splitext(path)[1].lower()
        if 'jpg' in ext or 'bmp' in ext or 'png' in ext or 'pdf' in ext:
            if data.band_count() == 1 or data.band_count() == 3 or data.band_count == 4:
                from matplotlib.pyplot import imsave
                rgb = np.transpose( data.data, (1,0,2) )
                if not ((data.is_int() and np.max(rgb) <= 255)): # handle normalisation
                    rgb = rgb - kwds.get("vmin", 0)
                    rgb /= (kwds.get("vmax", np.max(rgb) ) - kwds.get("vmin", 0) )
                    rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8) # convert to 8 bit image
                imsave( path, rgb ) # save the image
                return
        else: # save hyperspectral image
            try:
                from osgeo import gdal  # is gdal installed?
                save_func = saveWithGDAL
            except ModuleNotFoundError:  # no gdal, use SPy
                save_func = saveWithSPy
            ext = 'dat'
    elif isinstance(data, HyHeader):
        save_func = saveHeader
        ext = 'hdr'
    elif isinstance(data, HyCloud):
        save_func = saveCloudPLY
        ext = 'ply'
    elif isinstance(data, HyLibrary):
        save_func = saveLibraryCSV
        ext = 'csv'
    elif isinstance(data, PMap ):
        save_func = savePMap
        ext = 'npz'
    elif isinstance(data, HyCollection):
        save_func = saveCollection
        ext = 'hyc'
    elif isinstance(data, np.ndarray):
        save_func = np.save
        ext = 'npy'
    else:
        assert False, "Error - data type %s is unsupported by hylite.io.save." % type(data)

    # check path file extension
    if 'hdr' in os.path.splitext(path)[1]: # auto strip .hdr extensions if provided
        path = os.path.splitext(path)[0]
    if ext not in os.path.splitext(path)[1]: # add type-specific extension if needed
        path += '.%s'%ext

    # save!
    save_func( path, data )

def load(path):
    """
    A generic function for loading hyperspectral images, point clouds and libraries. The appropriate load function
    will be chosen based on the file extension.

    *Arguments*:
     - path = the path of the file to load.

    *Returns*:
     - a HyData instance containing the loaded dataset.
    """

    assert os.path.exists( path ), "Error: file %s does not exist." % path

    # load file formats with no associated header
    if 'npz' in os.path.splitext( path )[1].lower():
        return loadPMap(path)
    elif 'npy' in os.path.splitext( path )[1].lower():
        return np.load( path ) # load numpy

    # file (should/could) have header - look for it
    header, data = matchHeader( path )
    assert os.path.exists(data), "Error - file %s does not exist." % data
    ext = os.path.splitext(data)[1].lower()
    if ext == '':
        assert os.path.isfile(data), "Error - %s is a directory not a file." % data

    if 'ply' in ext: # point or hypercloud
        return loadCloudPLY(path)
    elif 'las' in ext: # point or hypercloud
        return loadCloudLAS(path)
    elif 'csv' in ext: # spectral library
        return loadLibraryCSV(path)
    elif 'sed' in ext: # spectral library
        return loadLibrarySED(path)
    elif 'tsg' in ext: # spectral library
        return loadLibraryTSG(path)
    elif 'hyc' in ext: # load hylite collection
        return loadCollection(path)
    else: # image
        # load conventional images with PIL
        if 'png' in ext or 'jpg' in ext or 'bmp' in ext:
            # load image with matplotlib
            from matplotlib.pyplot import imread
            return HyImage(np.transpose(imread(path), (1, 0, 2)))
        try:
            from osgeo import gdal # is gdal installed?
            return loadWithGDAL(path)
        except ModuleNotFoundError: # no gdal, use SPy
            return loadWithSPy(path)

##############################################
## save and load data collections
##############################################
# save collection
def saveCollection(path, collection):
    # strip file extension (if it exists)
    path = os.path.splitext(path)[0]

    # strip collection name (if it was included)
    if os.path.basename(path) == collection.name:
        path = os.path.dirname(path)

    # generate file paths
    dirmap = collection.get_file_dictionary(root=path)

    # save files
    for p, o in dirmap.items():
        os.makedirs(os.path.dirname(p), exist_ok=True)
        save(p, o)  # save each path and item [ n.b. this includes the header file! :-) ]


def loadCollection(path):
    # load header and find directory path
    header, directory = matchHeader(path)

    # parse name and root
    root = os.path.dirname(directory)
    name = os.path.basename(os.path.splitext(directory)[0])
    C = HyCollection(name, root, header=loadHeader(header))

    return C