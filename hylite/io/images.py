"""
Read common image formats, including ENVI format hyperspectral data.
"""

import sys, os
import numpy as np
from hylite.hyimage import HyImage, HyData
from .headers import matchHeader, makeDirs, loadHeader, saveHeader

# spectral python throws depreciation warnings - ignore these!
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def loadWithGDAL(path, dtype=np.float32, mask_zero = True, to_nm=False):
    """
    Load an image using gdal.

    Args:
        path: file path to the image to load
        mask_zero: True if zero values should be masked (replaced with nan). Default is true.
    Returns:
        a hyImage object
    """

    # find GDAL
    try:
        import osgeo.gdal as gdal
        gdal.PushErrorHandler('CPLQuietErrorHandler') # ignore GDAL warnings
    except:
        assert False, "Error - please install GDAL before using loadWithGDAL(...)"

    #parse file format
    _, ext = os.path.splitext(path)
    # load envi file or its variants
    if len(ext) == 0 or 'hdr' in ext.lower() or \
            'dat' in ext.lower() or \
            'img' in ext.lower() or \
            'lib' in ext.lower():
        header, image = matchHeader(path)
    elif 'tif' in ext.lower() or 'png' in ext.lower() or 'jpg' in ext.lower(): #standard image formats
        image = path
        header = None
    else:
        print( 'Warning - %s is an unknown/unsupported file format. Trying to load anyway....' % ext)
        header, image = matchHeader(path)
    # load header
    if not header is None:
        header = loadHeader(header, to_nm=to_nm)

    #load image
    assert os.path.exists(image), "Error - %s does not exist." % image
    try:
        raster = gdal.Open(image)  # open image
        data = raster.ReadAsArray().T
    except:
        assert False, "Error - %s could not be read by GDAL." % image

    #create image object
    assert data is not None, "Error - GDAL could not retrieve valid image data from %s" % path
    pj = raster.GetProjection()
    gt = raster.GetGeoTransform()
    img = HyImage(data, projection=pj, affine=gt, header=header, dtype=dtype)

    if mask_zero and (img.dtype == np.float32 or img.dtype == np.float64):
            img.data[img.data == 0] = np.nan #note to self: np.nan is float...

    return img

def loadWithSPy( path, dtype=np.float32, mask_zero = True, to_nm=False):
    """
    Load an image using spectral python. This works for most envi images, but doesn not load
    georeferencing information (in which case loadWithGDAL(...) should be used).

    Args:
        path: file path to the image to load
        mask_zero: True if zero values should be masked (replaced with nan). Default is true.
    Returns:
        a hyImage object
    """
    assert os.path.exists(path), "Error - %s does not exist." % path
    try: 
        import spectral
    except:
        assert False, "Error - please install spectral python using `pip install spectral` before using loadWithSPy(...)"
        
    # parse file format
    _, ext = os.path.splitext(path)
    if len(ext) == 0 or 'hdr' in ext.lower() or \
            'dat' in ext.lower() or \
            'img' in ext.lower() or \
            'lib' in ext.lower():
        header, image = matchHeader(path)

        # load image with SPy
        assert os.path.exists(image), "Error - %s does not exist." % image
        try: # try loading envi file first
            img = spectral.envi.open(header, image) # this must be an envi file
        except:
            img = spectral.open_image(header) # load unknown image type

        data = np.transpose( np.array(img.load()), (1,0,2) )
        if (data.dtype == np.float32) or (data.dtype == np.float64):
            # Spy still divides float values by scale factor. We don't want it to, so undo this.
            data *= img.scale_factor

        # load header
        if not header is None:
            header = loadHeader(header, to_nm=to_nm)
    elif 'tif' in ext.lower() or 'png' in ext.lower() or 'jpg' in ext.lower():  # standard image formats
        # load with matplotlib
        import matplotlib.image as mpimg
        data = mpimg.imread(path)
        header = None
    else:
        print('Warning - %s is an unknown/unsupported file format. Trying to load anyway...'%ext)
        #assert False, "Error - %s is an unknown/unsupported file format." % ext

    # create image object
    assert data is not None, "Error - GDAL could not retrieve valid image data from %s" % path
    img = HyImage(data, projection=None, affine=None, header=header, dtype=dtype)

    # spectral python automatically applies reflectance scale factor, so we must set this to 1.0 to avoid future nightmares...
    if np.nanmax(img.data) < 1.0:
        img.header['reflectance scale factor'] = 1.0

    if mask_zero and img.dtype == float:
        img.data[img.data == 0] = np.nan  # note to self: np.nan is float...

    return img

def loadSubset( path, *, bands=None, pixels=None, dtype=np.float32):
    """
    Load either specific bands (bands!=None) or pixels (pixels != None) from an ENVI file using spy to facilitate e.g. out-of-core
    processing routines.

    Args:
        path: a path to the hyperspectral image to read.
        bands: a list of hyperspectral band indices or wavelengths to extract, or None.
        pixels: a list of [(x1,y1),(x2,y2)] pixels to extract spectra for, or None. Either bands or pixels must be defined (but not both).
        dtype: the output data type. Default is float32.
    """
    assert os.path.exists(path), "Error - %s does not exist." % path
    assert (pixels is not None) or (bands is not None), "Error - either pixels OR bands must be specified"
    assert not ((pixels is not None) and (bands is not None)), "Error - pixels AND bands cannot both be specified"

    # parse file format
    _, ext = os.path.splitext(path)
    if len(ext) == 0 or 'hdr' in ext.lower() or \
            'dat' in ext.lower() or \
            'img' in ext.lower() or \
            'lib' in ext.lower():
        header, image = matchHeader(path)

        # load header and convert bands to band indices
        imageheader = loadHeader(header)
        if bands is not None:
            bands = [ HyImage( np.zeros((3,3,imageheader.band_count())),
                        header=imageheader, wav=imageheader.get_wavelengths() ).get_band_index(b) for b in bands ]

        # load image with SPy  TODO - replace this with numpy loading if possible?
        assert os.path.exists(image), "Error - %s does not exist." % image
        try: 
            import spectral
        except:
            assert False, "Error - please install spectral python using `pip install spectral` before using saveWithSPy(...)"
        try:  # try loading envi file first
            img = spectral.envi.open(header, image)  # this must be an envi file
        except:
            img = spectral.open_image(header)  # load unknown image type

        if bands is not None:  # get bands and put in HyImage
            data = np.dstack( [ img.read_band( b ).T for b in bands ] )
            out = HyImage( data, projection=None, affine=None, header=imageheader, dtype=dtype)
            out.set_wavelengths( imageheader.get_wavelengths()[bands] )
            if out.has_band_names():
                out.set_band_names( imageheader.get_band_names()[bands])
        if pixels is not None:  # get pixels and put in HyCloud
            data = np.array( [ img.read_pixel( *p[::-1] ) for p in pixels ] )
            out = HyData( data )
            out.header=imageheader
        return out

def loadWithNumpy( path, dtype=np.float32, mask_zero=True, to_nm=False ):
    # parse file format
    _, ext = os.path.splitext(path)
    if len(ext) == 0 or 'hdr' in ext.lower() or \
            'dat' in ext.lower() or \
            'img' in ext.lower() or \
            'lib' in ext.lower():
        header, image = matchHeader(path)

        # load header
        assert os.path.exists(image), "Error - %s does not exist." % image
        assert os.path.exists(header), "Error - %s does not exist." % header
        header = loadHeader(header, to_nm=to_nm)
        samples = int(header['samples']) # read relevant bits of header file
        lines = int(header['lines'])
        bands = int(header['bands'])
        data_type = int(header['data type'])
        interleave = header.get('interleave', 'bil').lower()

        # get byte offset
        offset = int(header.get('header offset', 0))

        # ENVI data type mapping to NumPy
        dtype_map = {
            1: np.uint8,
            2: np.int16,
            3: np.int32,
            4: np.float32,
            5: np.float64,
            12: np.uint16,
            13: np.uint32,
            14: np.int64,
            15: np.uint64
        }
    
        if data_type not in dtype_map:
            raise ValueError(f"Unsupported data type: {data_type}")
        dtype = dtype_map[data_type]
    
        # Load binary data
        data = np.fromfile(image, dtype=dtype)
        if offset > 0:
            offset = int(offset / data.dtype.itemsize) # convert from bytes to index
            data = data[offset:] # slice off header bytes
        expected_size = lines * bands * samples
        if data.size != expected_size:
            raise ValueError(f"Expected {expected_size} elements, got {data.size}. Check lines, bands and samples entries in header file.")

        if interleave == 'bil':
            data = data.reshape((lines, bands, samples))
            data = np.transpose(data, (0, 2, 1))  # (lines, samples, bands)
        elif interleave == 'bsq':
            data = data.reshape((bands, lines, samples))
            data = np.transpose(data, (1, 2, 0))  # (lines, samples, bands)
        elif interleave == 'bip':
            data = data.reshape((lines, samples, bands))  # already in correct order
        else:
            raise ValueError(f"Unsupported interleave format: {interleave}")

        # flip from row-column to x-y layout
        data = np.transpose( data, (1,0,2) )
        return HyImage(data, header=header)
    elif 'tif' in ext.lower() or 'png' in ext.lower() or 'jpg' in ext.lower():  # standard image formats
        # load with matplotlib
        import matplotlib.image as mpimg
        data = mpimg.imread(path)
        return HyImage(data)
    else:
        assert False, "Error - %s is an unknown/unsupported file format." % ext

def saveWithNumpy( path, image, writeHeader=True, interleave='BSQ'):
    # make sure extension is proper
    path, ext = os.path.splitext(path)
    if "hdr" in str.lower(ext) or ext == '':
        ext = ".dat"

    interleave = interleave.lower()
    if interleave not in ['bil', 'bip', 'bsq']:
        raise ValueError("Interleave must be one of: 'bil', 'bip', or 'bsq'")

    image.push_to_header() # update header flags
    
    # Map NumPy dtype to ENVI data type code
    dtype = str(image.data.dtype)
    numpy_to_envi = {
        'uint8': 1,
        'int16': 2,
        'int32': 3,
        'float32': 4,
        'float64': 5,
        'uint16': 12,
        'uint32': 13,
        'int64': 14,
        'uint64': 15
    }
    if dtype not in numpy_to_envi:
        raise ValueError(f"Unsupported dtype for ENVI: {dtype}")
    envi_data_type = numpy_to_envi[dtype]

    # Reorder data based on interleave
    array = np.transpose( image.data, (1,0,2) ) # from x-y to i-j ordering
    if interleave == 'bil':
        out_data = np.transpose(array, (0, 2, 1))  # (lines, bands, samples)
    elif interleave == 'bsq':
        out_data = np.transpose(array, (2, 0, 1))  # (bands, lines, samples)
    elif interleave == 'bip':
        out_data = array  # (lines, samples, bands)
    out_data.tofile(path+ext)

    # save header file
    header = image.header.copy()
    header['data type'] = envi_data_type
    header['interleave'] = interleave
    header['byte order'] = '0'
    header_path = path + '.hdr'
    saveHeader( header_path, header)

# noinspection PyUnusedLocal
def saveWithGDAL(path, image, writeHeader=True, interleave='BSQ'):
    """
    Write this image to a file.

    Args:
        path: the path to save to.
        image: the image to write.
        writeHeader: true if a .hdr file will be written. Default is true.
        interleave: data interleaving for ENVI files. Default is 'BSQ', other options are 'BIL' and 'BIP'.
    """

    # find GDAL
    try:
        import osgeo.gdal as gdal
        gdal.PushErrorHandler('CPLQuietErrorHandler') # ignore GDAL warnings
    except:
        assert False, "Error - please install GDAL before using saveWithGDAL(...)"

    # make directories if need be
    makeDirs( path )

    path, ext = os.path.splitext(path)

    if "hdr" in str.lower(ext):
        ext = ".dat"

    #get image driver
    driver = 'ENVI'
    if '.tif' in str.lower(ext):
        driver = 'GTiff'

    #todo - add support for png and jpg??

    #set byte order
    if 'little' in sys.byteorder:
        image.header['byte order'] = 0
    else:
        image.header['byte order'] = 1

    #parse data type from image array
    data = image.data
    dtype = gdal.GDT_Float32
    image.header["data type"] = 4
    image.header["interleave"] = str.lower(interleave)
    if image.data.dtype == np.intc or image.data.dtype == np.int32:
        dtype = gdal.GDT_Int32
        image.header["data type"] = 3
    if image.data.dtype == np.int16:
        dtype = gdal.GDT_Int16
        image.header["data type"] = 2
    if image.data.dtype == np.uint8:
        data = np.array(image.data, np.dtype('b'))
        dtype = gdal.GDT_Byte
        image.header["data type"] = 1
    if image.data.dtype == np.uint or image.data.dtype == np.uint32:
        dtype = gdal.GDT_UInt32
        image.header["data type"] = 13
    if image.data.dtype == np.uint16:
        dtype = gdal.GDT_UInt16
        image.header["data type"] = 12

    #write
    if driver == 'GTiff':
        output = gdal.GetDriverByName(driver).Create( path + ext, image.xdim(), image.ydim(), image.band_count(), dtype)
    else:
        output = gdal.GetDriverByName(driver).Create( path + ext, image.xdim(), image.ydim(), image.band_count(), dtype, ['INTERLEAVE=%s'%interleave] )
    
    # check we got a valid file
    if output is None:
        assert False, "Could not create output file at %s"%(path + ext)
    assert image.band_count() > 0, "Image has incorrect data shape (%s) and could not be saved."%image.data.shape

    #write bands
    for i in range(image.band_count()):
         rb = output.GetRasterBand(i+1)
         rb.WriteArray(data[:, :, i].T)
         rb = None #close band
    output = None #close file

    if writeHeader and not image.header is None: #write .hdr file
        image.push_to_header()
        saveHeader(path + ".hdr", image.header)

    # save geotransform/project information
    if (image.affine is not None) or (image.projection is not None):
        output = gdal.Open(path + ext, gdal.GA_Update)
        if output is None:
            print("Warning - could not save geotransform information for %s"%(path + ext)) # this happens sometimes -- not sure why.
        else:
            output.SetGeoTransform(image.affine)
            if not image.projection is None:
                output.SetProjection(image.projection.ExportToPrettyWkt())
            output = None  # close file

def saveWithSPy( path, image, writeHeader=True, interleave='BSQ'):
    try: 
        import spectral
    except:
        assert False, "Error - please install spectral python using `pip install spectral` before using saveWithSPy(...)"
        
    # make directories if need be
    makeDirs(path)

    # make sure extension is proper
    path, ext = os.path.splitext(path)
    if "hdr" in str.lower(ext) or ext == '':
        ext = ".dat"

    # set byte order
    if 'little' in sys.byteorder:
        image.header['byte order'] = 0
        byteorder = 0
    else:
        image.header['byte order'] = 1
        byteorder = 1

    image.push_to_header()
    spectral.envi.save_image( path + ".hdr", np.transpose(image.data,(1,0,2)),
                                    dtype=image.data.dtype, force=True,
                                    ext=ext, byteorder=byteorder, metadata=image.header)
