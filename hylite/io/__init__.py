
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

from hylite import HyData, HyImage, HyCloud, HyLibrary
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
                if not (data.is_int() and np.max(rgb) <= 255): # handle normalisation
                    rgb = rgb.data - kwds.get("vmin", 0)
                    rgb /= (kwds.get("vmax", np.max(rgb.data) ) - kwds.get("vmin", 0) )
                    rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8) # convert to 8 bit image
                imsave( path, rgb ) # save the image
                return
        else: # save hyperspectral image
            try:
                from osgeo import gdal  # is gdal installed?
                save_func = saveWithGDAL
            except ModuleNotFoundError:  # no gdal, use SPy
                save_func = saveWithSPy
    elif isinstance(data, HyCloud):
        save_func = saveCloudPLY
    elif isinstance(data, HyLibrary):
        save_func = saveLibraryCSV
    elif isinstance(data, PMap ):
        save_func = savePMap
    else:
        assert False, "Error - data must be an instance of HyImage, HyCloud or HyLibrary."

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

    # load file formats with no associated header
    if 'npz' in os.path.splitext( path )[1].lower():
        return loadPMap(path)

    # file (should/could) have header - look for it
    header, data = matchHeader( path )
    ext = ''
    if data is not None:
        ext = os.path.splitext(data)[1].lower()

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

