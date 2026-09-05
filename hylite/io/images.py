"""
Read common image formats, including ENVI format hyperspectral data.
"""

import sys, os
import numpy as np
from hylite.hyimage import HyImage, HyData
from .headers import matchHeader, makeDirs, loadHeader, saveHeader
from hylite._deps import require

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
        a `hylite.hyimage.HyImage` object
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
        a `hylite.hyimage.HyImage` object
    """
    assert os.path.exists(path), "Error - %s does not exist." % path
    spectral = require("spectral")

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
    Load either specific bands (bands!=None) or pixels (pixels != None) from an ENVI file to facilitate e.g. out-of-core
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
    return loadWithNumpy(path, dtype=dtype, bands=bands, pixels=pixels)

def loadWithNumpy( path, dtype=None, mask_zero=True, to_nm=False, bands=None, pixels=None, step=1, average=False, memmap=False ):
    """
    Load an ENVI image with NumPy. Optionally read only some bands, some pixels, or a spatially / spectrally
    reduced subset (stride or block-average) without materialising the full cube.

    Args:
        path: file path to the image (or its .hdr) to load.
        dtype: output array dtype, or None to keep the on-disk type (average always accumulates in float32).
        mask_zero: True if zero values should be masked (replaced with nan) in floating-point data. Default is True.
        to_nm: if True, convert header wavelengths to nanometres.
        bands: optional band subset — a list of indices, wavelengths or band names, or a slice. If set, this
            replaces any spectral `step`. Cannot be combined with `average=True`.
        pixels: optional list of `(x, y)` pixel coordinates (hylite / sample, line). Returns a `HyData` of spectra
            instead of a `HyImage`. `step` and `average` are ignored when pixels are specified.
        step: spatial / spectral factor. An int `n` applies to x and y. A `(sx, sy)` tuple applies to samples and
            lines. A `(sx, sy, sb)` tuple also reduces bands. Default is 1 (no reduction).
        average: if False (default), `step` is a stride (keep every n-th sample). If True, `step` is a block size
            and each output value is the nan-mean of the corresponding block (streamed in native interleave).
        memmap: if True, return a transposed view of the on-disk memmap (no copy). Incompatible with `average`,
            `mask_zero`, pixel loads, and a `dtype` that differs from the file.
    Returns:
        a `hylite.HyImage` (image load) or `hylite.HyData` (pixel load).
    """
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
        n_bands = int(header['bands'])
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
        np_dtype = dtype_map[data_type]
        if memmap and (average or pixels is not None):
            raise ValueError("memmap=True cannot be combined with average=True or pixels=")
        if memmap and mask_zero:
            raise ValueError("memmap=True requires mask_zero=False")
        if average and bands is not None:
            raise ValueError("average=True cannot be combined with a band list")

        # parse spatial / spectral factor (pixels ignore step / average)
        if np.isscalar(step):
            step_x = step_y = int(step)
            step_b = 1
        else:
            step = tuple(int(v) for v in step)
            if len(step) == 2:
                step_x, step_y = step
                step_b = 1
            elif len(step) == 3:
                step_x, step_y, step_b = step
            else:
                raise ValueError("step must be an int or a (x, y) or (x, y, band) tuple")
        if min(step_x, step_y, step_b) < 1:
            raise ValueError("step values must be >= 1")

        # resolve band selectors to integer indices (None = all, possibly strided)
        if average:
            band_idx = None
        elif bands is None:
            band_idx = None if step_b == 1 else np.arange(0, n_bands, step_b)
        elif isinstance(bands, slice):
            band_idx = np.arange(n_bands)[bands]
        else:
            if isinstance(bands, (int, float, str)) or np.isscalar(bands):
                bands = [bands]
            ref = HyImage(np.zeros((1, 1, n_bands)), header=header.copy())
            band_idx = np.array([ref.get_band_index(b) for b in bands], dtype=int)
        if band_idx is not None and len(band_idx) == 0:
            raise ValueError("Band selection is empty.")

        # map file (trust a declared header offset; only guess an undeclared prefix when offset is 0)
        itemsize = np.dtype(np_dtype).itemsize
        expected_bytes = lines * n_bands * samples * itemsize
        file_size = os.path.getsize(image)
        if offset == 0 and file_size > expected_bytes:
            offset = file_size - expected_bytes
        if file_size - offset < expected_bytes:
            raise ValueError(f"Expected {expected_bytes} bytes, got {file_size - offset}. Check lines, bands and samples entries in header file.")

        if interleave == 'bil':
            native = (lines, n_bands, samples)
        elif interleave == 'bsq':
            native = (n_bands, lines, samples)
        elif interleave == 'bip':
            native = (lines, samples, n_bands)
        else:
            raise ValueError(f"Unsupported interleave format: {interleave}")
        mmap = np.memmap(image, dtype=np_dtype, mode='r', offset=offset, shape=native)

        # trim or average header metadata to the bands we will actually load
        header = header.copy()
        if average and step_b > 1:
            out_b = n_bands // step_b
            n_keep = out_b * step_b
            if header.has_wavelengths():
                header.set_wavelengths(header.get_wavelengths()[:n_keep].reshape(out_b, step_b).mean(axis=-1))
            if header.has_band_names():
                names = np.asarray(header.get_band_names())[:n_keep].reshape(out_b, step_b)[:, 0]
                header.set_band_names(names)
            if header.has_fwhm():
                header.set_fwhm(header.get_fwhm()[:n_keep].reshape(out_b, step_b).mean(axis=-1))
            if header.has_bbl():
                header.set_bbl(header.get_bbl()[:n_keep].reshape(out_b, step_b).all(axis=-1))
        elif band_idx is not None:
            if header.has_wavelengths():
                header.set_wavelengths(header.get_wavelengths()[band_idx])
            if header.has_band_names():
                header.set_band_names(np.asarray(header.get_band_names())[band_idx])
            if header.has_fwhm():
                header.set_fwhm(header.get_fwhm()[band_idx])
            if header.has_bbl():
                header.set_bbl(header.get_bbl()[band_idx])

        # extract spectra at (x, y) = (sample, line) and return HyData
        if pixels is not None:
            xs = np.asarray([p[0] for p in pixels], dtype=int)
            ys = np.asarray([p[1] for p in pixels], dtype=int)
            if np.any(xs < 0) or np.any(xs >= samples) or np.any(ys < 0) or np.any(ys >= lines):
                raise IndexError("Pixel coordinates are out of bounds.")
            if interleave == 'bil':
                data = np.array(mmap[ys, :, xs], copy=True, order='C')
            elif interleave == 'bsq':
                data = np.array(mmap[:, ys, xs].T, copy=True, order='C')
            else:
                data = np.array(mmap[ys, xs, :], copy=True, order='C')
            if band_idx is not None:
                data = data[:, band_idx]
            if dtype is not None:
                data = np.asarray(data, dtype=dtype)
            out = HyData(data, header=header)
            out.push_to_header()
            if mask_zero and (out.dtype == np.float32 or out.dtype == np.float64):
                out.data[out.data == 0] = np.nan
            return out

        # block-average in native interleave (one output line / band-group at a time)
        if average:
            out_x = samples // step_x
            out_y = lines // step_y
            out_b = n_bands // step_b
            if min(out_x, out_y, out_b) < 1:
                raise ValueError("step exceeds image dimensions")
            ignore = header.get('data ignore value', None)
            if ignore is not None and str(ignore).strip() != '':
                ignore = float(ignore)
            else:
                ignore = None
            data = np.empty((out_x, out_y, out_b), dtype=np.float32)
            chunk = 32  # output lines (or band-groups for BSQ) per streamed read
            if interleave == 'bil':
                for iy0 in range(0, out_y, chunk):
                    iy1 = min(iy0 + chunk, out_y)
                    n = iy1 - iy0
                    block = np.array(mmap[iy0 * step_y:iy1 * step_y, :out_b * step_b, :out_x * step_x],
                                     dtype=np.float32, copy=True)
                    if ignore is not None:
                        block[block == ignore] = np.nan
                    if mask_zero:
                        block[block == 0] = np.nan
                    block = block.reshape(n, step_y, out_b, step_b, out_x, step_x)
                    data[:, iy0:iy1, :] = np.nanmean(block, axis=(1, 3, 5)).transpose(2, 0, 1)
            elif interleave == 'bsq':
                for ib0 in range(0, out_b, chunk):
                    ib1 = min(ib0 + chunk, out_b)
                    n = ib1 - ib0
                    block = np.array(mmap[ib0 * step_b:ib1 * step_b, :out_y * step_y, :out_x * step_x],
                                     dtype=np.float32, copy=True)
                    if ignore is not None:
                        block[block == ignore] = np.nan
                    if mask_zero:
                        block[block == 0] = np.nan
                    block = block.reshape(n, step_b, out_y, step_y, out_x, step_x)
                    data[:, :, ib0:ib1] = np.nanmean(block, axis=(1, 3, 5)).transpose(2, 1, 0)
            else:
                for iy0 in range(0, out_y, chunk):
                    iy1 = min(iy0 + chunk, out_y)
                    n = iy1 - iy0
                    block = np.array(mmap[iy0 * step_y:iy1 * step_y, :out_x * step_x, :out_b * step_b],
                                     dtype=np.float32, copy=True)
                    if ignore is not None:
                        block[block == ignore] = np.nan
                    if mask_zero:
                        block[block == 0] = np.nan
                    block = block.reshape(n, step_y, out_x, step_x, out_b, step_b)
                    data[:, iy0:iy1, :] = np.nanmean(block, axis=(1, 3, 5)).transpose(1, 0, 2)
            if dtype is not None and np.dtype(dtype) != data.dtype:
                data = np.asarray(data, dtype=dtype)
            img = HyImage(data, header=header)
            img.push_to_header()
            return img

        # slice in native interleave so unused lines / samples / bands are not copied
        regular_band_step = bands is None and step_b > 1 and band_idx is not None
        if interleave == 'bil':
            if regular_band_step:
                view = mmap[::step_y, ::step_b, ::step_x]
            else:
                view = mmap[::step_y, :, ::step_x]
                if band_idx is not None:
                    view = view[:, band_idx, :]
            axes = (2, 0, 1)  # (x, y, band)
        elif interleave == 'bsq':
            if regular_band_step:
                view = mmap[::step_b, ::step_y, ::step_x]
            else:
                view = mmap[:, ::step_y, ::step_x]
                if band_idx is not None:
                    view = view[band_idx, :, :]
            axes = (2, 1, 0)
        else:
            if regular_band_step:
                view = mmap[::step_y, ::step_x, ::step_b]
            else:
                view = mmap[::step_y, ::step_x, :]
                if band_idx is not None:
                    view = view[:, :, band_idx]
            axes = (1, 0, 2)
        if memmap:
            if dtype is not None and np.dtype(dtype) != view.dtype:
                raise ValueError("memmap=True requires dtype=None or the on-disk dtype")
            data = np.transpose(view, axes)
        else:
            data = np.array(np.transpose(view, axes), copy=True, order='C')
            if dtype is not None:
                data = np.asarray(data, dtype=dtype)
        img = HyImage(data, header=header)
        img.push_to_header()
        if (not memmap) and mask_zero and (img.dtype == np.float32 or img.dtype == np.float64):
            img.data[img.data == 0] = np.nan
        return img
    elif 'tif' in ext.lower() or 'png' in ext.lower() or 'jpg' in ext.lower():  # standard image formats
        if bands is not None or pixels is not None or step != 1 or average or memmap:
            raise ValueError("bands, pixels, step, average and memmap are only supported for ENVI images.")
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
    data = image.data
    dtype = str(data.dtype)
    if dtype in ('bool', 'bool_'):  # ENVI has no bool; write as byte 0/1
        data = data.astype(np.uint8)
        dtype = 'uint8'
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
    array = np.transpose( data, (1,0,2) ) # from x-y to i-j ordering
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
    header['header offset'] = '0'  # tofile writes a raw cube with no prefix
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
    if image.data.dtype == np.uint8 or image.data.dtype == np.bool_ or image.data.dtype == bool:
        data = np.array(image.data, np.uint8)
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
    spectral = require("spectral")

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
