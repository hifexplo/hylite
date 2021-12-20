
"""
Import and export hyperspectral data. For hyperspectral images this is mostly done using GDAL,
while for point clouds and hyperspectral libraries a variety of different methods are included.
"""
from .headers import *
from .images import *
from .clouds import *
from .libraries import *
from .pmaps import *
from .cameras import saveCameraTXT, loadCameraTXT

from hylite import HyImage, HyCloud, HyLibrary, HyCollection, HyScene, HyData
from hylite.project import PMap, Camera, Pushbroom
from hylite.analyse.mwl import MWL
import shutil

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
                    vmin = kwds.get("vmin", np.nanpercentile(rgb, 1 ) )
                    vmax = kwds.get("vmax", np.nanpercentile(rgb, 99) )
                    rgb = (rgb - vmin) / (vmax-vmin)
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
    elif isinstance(data, Camera ):
        save_func = saveCameraTXT
        ext = 'cam'
    elif isinstance(data, Pushbroom):
        save_func = saveCameraTXT
        ext = 'brm'
    elif isinstance(data, HyCollection):
        save_func = saveCollection
        ext = 'hyc'
        if isinstance(data, HyScene): # special type of HyCollection, should have different extension
            ext = 'hys'
        if isinstance(data, MWL): # special type of HyCollection, should have different extension
            ext = 'mwl'
        if os.path.splitext(path)[0]+"."+ext != data._getDirectory(): # we're moving to a new home! Copy folder
            if os.path.exists(data._getDirectory()): # if it exists...
                shutil.copytree( data._getDirectory(), os.path.splitext(path)[0]+"."+ext)
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
    assert os.path.exists(data), "Error - data file %s does not exist." % path
    ext = os.path.splitext(data)[1].lower()
    if ext == '':
        assert os.path.isfile(data), "Error - %s is a directory not a file." % data

    # load other file types
    if 'ply' in ext: # point or hypercloud
        out = loadCloudPLY(path) # load dataset
    elif 'las' in ext: # point or hypercloud
        out =  loadCloudLAS(path)
    elif 'csv' in ext: # spectral library
        out = loadLibraryCSV(path)
    elif 'sed' in ext: # spectral library
        out = loadLibrarySED(path)
    elif 'tsg' in ext: # spectral library
        out = loadLibraryTSG(path)
    elif 'hyc' in ext or 'hys' in ext or 'mwl' in ext: # load hylite collection, hyscene or mwl map
        out = loadCollection(path)
    elif 'cam' in ext or 'brm' in ext: # load pushbroom and normal cameras
        out = loadCameraTXT(path)
    else: # image
        # load conventional images with PIL
        if 'png' in ext or 'jpg' in ext or 'bmp' in ext:
            # load image with matplotlib
            from matplotlib.pyplot import imread
            out = HyImage(np.transpose(imread(path), (1, 0, 2)))
        else:
            try:
                from osgeo import gdal # is gdal installed?
                out = loadWithGDAL(path)
            except ModuleNotFoundError: # no gdal, use SPy
                out = loadWithSPy(path)

    return out  # return dataset

##############################################
## save and load data collections
##############################################
# save collection
def saveCollection(path, collection):
    # generate file paths
    dirmap = collection.get_file_dictionary(root=os.path.dirname(path),
                                            name=os.path.splitext(os.path.basename(path))[0])
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

    if 'hyc' in os.path.splitext(directory)[1]:
        C = HyCollection(name, root, header=loadHeader(header))
    elif 'hys' in os.path.splitext(directory)[1]:
        C = HyScene(name, root, header=loadHeader(header))
    elif 'mwl' in os.path.splitext(directory)[1]:
        C = MWL(name, root, header=loadHeader(header))
    else:
        print(header, directory )
        assert False, "Error - %s is an invalid collection." % directory
    return C