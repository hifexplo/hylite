from hylite import HyData, HyImage, HyCloud, HyHeader
import numpy as np
from pathlib import Path
from hylite import io
import os

# functions for creating test datasets
def genHeader():
    return  io.loadHeader( str( Path(__file__).parent.parent / "test_data" / "image.hdr") )

# create test cloud
def genCloud( npoints = 1000, nbands=10, data=True, normals=True, rgb=True):
    header = genHeader()

    xyz = np.random.rand(npoints, 3)
    if rgb:
        rgb = np.random.rand(npoints, 3)
    if normals:
        normals = np.random.rand(npoints, 3)
        normals = normals / np.linalg.norm(normals, axis=1)[:,None]
    if data:
        data = np.random.rand(npoints, nbands)  # point cloud

    cloud = HyCloud( xyz, rgb = rgb, normals=normals, bands=data, header=header )
    cloud.set_wavelengths( np.linspace(500.0,1000.0,nbands) )
    cloud.set_band_names( ["Band %d" % (i+1) for i in range(nbands)])
    cloud.set_fwhm( [1.0 for i in range(nbands)])
    cloud.push_to_header() # push info on dims

    return cloud

def genImage( dimx = 1464, dimy=401, nbands=10 ):
    header = genHeader()
    data = np.random.rand(dimx, dimy, nbands)  # image
    data = HyImage( data, header = header )
    data.set_wavelengths( np.linspace(500.0,1000.0,nbands) )
    data.set_band_names( ["Band %d" % (i+1) for i in range(nbands)])
    data.set_fwhm( [1.0 for i in range(nbands)])
    data.push_to_header() # push info on dims
    return data