
"""
Import and export hyperspectral data. For hyperspectral images this is mostly done using GDAL,
while for point clouds and hyperspectral libraries a variety of different methods are included.
"""

from .headers import *
from .images import *
from .clouds import *
from .libraries import *

from hylite import HyData, HyImage, HyCloud, HyLibrary

def save(path, data):
    """
    A generic function for saving HyData instances such as HyImage, HyLibrary and HyCloud. The appropriate file format
    will be chosen automatically.

    *Arguments*:
     - path = the path to save the file too.
     - data = the data to save. This must be an instance of HyImage, HyLibrary or HyCloud.
    """

    if isinstance(data, HyImage):
        save_func = saveWithGDAL
    elif isinstance(data, HyCloud):
        save_func = saveCloudPLY
    elif isinstance(data, HyLibrary):
        save_func = saveLibraryCSV
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

    header, data = matchHeader( path )
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
    else: # image - load with gdal
        return loadWithGDAL(path)
