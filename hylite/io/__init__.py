
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
from distutils.dir_util import copy_tree
import os

# check if gdal is installed
try:
    from osgeo import gdal
    usegdal = True
except ModuleNotFoundError:
    usegdal = False

def save(path, data, **kwds):
    """
    A generic function for saving HyData instances such as HyImage, HyLibrary and HyCloud. The appropriate file format
    will be chosen automatically.

    Args:
        path (str): the path to save the file too.
        data (HyData or ndarray): the data to save. This must be an instance of HyImage, HyLibrary or HyCloud.
        **kwds: Keywords can include:

             - vmin = the data value that = 0 when saving RGB images.
             - vmax = the data value that = 255 when saving RGB images. Must be > vmin.
    """

    if isinstance(data, HyImage):

        # special case - save ternary image to png or jpg or bmp
        ext = os.path.splitext(path)[1].lower()
        if 'jpg' in ext or 'bmp' in ext or 'png' in ext or 'pdf' in ext:
            if data.band_count() == 1 or data.band_count() == 3 or data.band_count == 4:
                rgb = np.transpose( data.data, (1,0,2) )
                if not ((data.is_int() and np.max(rgb) <= 255)): # handle normalisation
                    vmin = kwds.get("vmin", np.nanpercentile(rgb, 1 ) )
                    vmax = kwds.get("vmax", np.nanpercentile(rgb, 99) )
                    rgb = (rgb - vmin) / (vmax-vmin)
                    rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8) # convert to 8 bit image
                #from matplotlib.pyplot import imsave
                # imsave( path, rgb )
                from skimage import io as skio
                skio.imsave( path, rgb ) # save the image
                return
        elif ((data.band_count() == 3) or (data.band_count() == 4)) and (data.data.dtype == np.uint8):
            # save 3 and 4 band uint8 arrays as png files
            # from matplotlib.pyplot import imsave
            # imsave( os.path.splitext(path)[0]+".png", data.data)  # save the image
            from skimage import io as skio
            skio.imsave(os.path.splitext(path)[0]+".png", np.transpose( data.data, (1,0,2) ))  # save the image
            save( os.path.splitext(path)[0] + ".hdr", data.header ) # save header
            return
        else: # save hyperspectral image
            if usegdal:
                from osgeo import gdal  # is gdal installed?
                save_func = saveWithGDAL
            else:  # no gdal, use SPy
                save_func = saveWithSPy
            if 'lib' in ext: # special case - we are actually saving a HyLibrary (as an image)
                ext = 'lib'
            else:
                ext = 'dat'
    elif isinstance(data, HyHeader):
        save_func = saveHeader
        ext = 'hdr'
    elif isinstance(data, HyCloud):
        save_func = saveCloudPLY
        ext = 'ply'
    elif isinstance(data, HyLibrary):
        save_func = saveLibraryLIB
        ext = 'lib'
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
        save_func = _saveCollection
        ext = data.ext[1:]
        outdir = os.path.join(data.root, os.path.splitext(data.name)[0])
        if os.path.splitext(path)[0] != outdir:
            if os.path.exists( outdir+"."+ext): # if it exists...
                #if sys.version_info[1] >= 8: # python 3.8 or greater
                #    shutil.copytree( outdir+"."+ext, os.path.splitext(path)[0]+"."+ext, dirs_exist_ok=True)
                #else:
                #    shutil.copytree( outdir+"."+ext, os.path.splitext(path)[0]+"."+ext ) # will fail if directory already exists unfortunately.
                copy_tree(outdir+"."+ext, os.path.splitext(path)[0]+"."+ext)

    elif isinstance(data, np.ndarray) or isinstance(data, list):
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
    os.makedirs( os.path.dirname(path), exist_ok=True)  # make output directory
    save_func( path, data )

def load(path):
    """
    A generic function for loading hyperspectral images, point clouds and libraries. The appropriate load function
    will be chosen based on the file extension.

    Args:
        path (str): the path of the file to load.

    Returns:
        The loaded data.
    """

    assert os.path.exists( path ), "Error: file %s does not exist." % path

    # load file formats with no associated header
    if 'npz' in os.path.splitext( path )[1].lower():
        return loadPMap(path)
    elif 'npy' in os.path.splitext( path )[1].lower():
        return np.load( path ) # load numpy

    # file (should/could) have header - look for it
    header, data = matchHeader( path )
    assert os.path.exists(str(data)), "Error - data file %s does not exist." % path
    ext = os.path.splitext(data)[1].lower()
    if ext == '':
        assert os.path.isfile(data), "Error - %s is a directory not a file." % data

    # load other file types
    if 'ply' in ext: # point or hypercloud
        out = loadCloudPLY(path) # load dataset
    elif 'las' in ext: # point or hypercloud
        out =  loadCloudLAS(path)
    elif 'csv' in ext: # (flat) spectral library
        out = loadLibraryCSV(path)
    elif 'txt' in ext: # (flat) spectral library
        out = loadLibraryTXT(path)
    elif 'sed' in ext: # (flat) spectral library
        out = loadLibrarySED(path)
    elif 'tsg' in ext: # (flat) spectral library
        out = loadLibraryTSG(path)
    elif 'hyc' in ext or 'hys' in ext or 'mwl' in ext: # load hylite collection, hyscene or mwl map
        out = _loadCollection(path)
    elif 'cam' in ext or 'brm' in ext: # load pushbroom and normal cameras
        out = loadCameraTXT(path)
    else: # image
        # load conventional images with PIL
        if 'png' in ext or 'jpg' in ext or 'bmp' in ext:
            # load image with matplotlib
            #from matplotlib.pyplot import imread
            #im = imread(path)
            from skimage import io as skio
            im = skio.imread(data)
            if len(im.shape) == 2:
                im = im[:,:,None] # add last dimension if greyscale image is loaded
            out = HyImage(np.transpose(im, (1, 0, 2)))
            if header is not None:
                out.header = loadHeader(header)
        else:
            if usegdal:
                from osgeo import gdal # is gdal installed?
                out = loadWithGDAL(path)
            else: # no gdal, use SPy
                out = loadWithSPy(path)

        # special case - loading spectral library; convert image to HyData
        if 'lib' in ext:
            out = HyLibrary(out.data, header=out.header)
    return out  # return dataset

##############################################
## save and load data collections
##############################################
# save collection
def _saveCollection(path, collection):
    # generate file paths
    dirmap = collection.get_file_dictionary(root=os.path.dirname(path),
                                            name=os.path.splitext(os.path.basename(path))[0])
    # save files
    os.makedirs(collection.getDirectory(), exist_ok=True) # make output directory
    for p, o in dirmap.items():
        os.makedirs(os.path.dirname(p), exist_ok=True)
        save(p, o)  # save each path and item [ n.b. this includes the header file! :-) ]

def _loadCollection(path):
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
        # print(header, directory )
        assert False, "Error - %s is an invalid collection." % directory
    return C