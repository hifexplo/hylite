import os
import numpy as np
from hylite import HyCloud
from .headers import matchHeader, makeDirs, saveHeader, loadHeader
from .images import loadWithGDAL


def saveCloudCSV(path, cloud, delimeter=' ', fmt='%.3f', writeHeader=True):

    """
    Saves a HyperCloud object to csv format.

    *Arguments*:
     - path = the path to save with. If an extension is provided (e.g. .las) then it will be retained. Otherwise .csv is used.
     - cloud = the hypercloud to write
     - delimeter = the delimeter to use to separate values. Default is ",".
     - fmt = numeric value formatting (see documentation for np.savetxt(...) for more details).
     - write = True if first row should contain field names (including scalar fields). Default is true.
    """

    # make directories if need be
    makeDirs( path )

    # add file extention to path if need be
    pth, ext = os.path.splitext(path)
    if ext is '': path += ".csv"

    # build header and data arrays
    d = delimeter
    format = [fmt, fmt, fmt]
    data = [cloud.xyz]
    header = 'x%sy%sz' % (d, d)

    #add rgb (as integers)
    if cloud.has_rgb():
        header += '%sr%sg%sb' % (d, d, d)
        format += ['%d', '%d', '%d']
        if cloud.rgb.dtype == np.uint8:
            data.append(cloud.rgb)
        elif cloud.rgb.dtype == np.float or cloud.rgb.dtype == np.float32:
            data.append((cloud.rgb * 255).astype(np.int))
        else:
            print("Warning - unknown data type for RGB colours (%s)" % cloud.rgb.dtype)

    #add normals
    if cloud.has_normals():
        header += '%sNx%sNy%sNz' % (d, d, d)
        format += [fmt, fmt, fmt]
        data.append(cloud.normals)

    #add scalar fields
    if cloud.has_bands():

        #calculate band identifiers
        if cloud.has_wavelengths():
            names = cloud.get_wavelengths()
        elif cloud.has_band_names():
            names = cloud.get_band_names()
        else:
            names = [str(x) for x in range(cloud.band_count())]

        for n in names:
            header += '%s%s' % (d, n)
            format += [fmt]
        data.append(cloud.data)

    #strip header?
    if not writeHeader:
        header = ''
    # write csv with numpy
    np.savetxt(path, np.hstack(data), fmt=format, delimiter=delimeter, header=header, newline='\n')

    #write header file
    if not cloud.header is None:
        hdrpth, _ = os.path.splitext( path )
        cloud.push_to_header()
        saveHeader( hdrpth + ".hdr", cloud.header )

def loadCloudCSV(path, delimiter=' ', order='xyzrgbklm'):

    """
    Loads a point cloud from a csv (or other formatted text) file.

    *Arguments*:
     - path = the file path
     - delimeter = the delimeter used (e.g. ' ', ',', ';'). Default is ' '.
     - order = A string defining the order of data in the csv. Each character maps to the following:
                -'x' = point x coordinate
                -'y' = point y coordinate
                -'z' = point z coordinate
                -'r' = point r coordinate
                -'g' = point g coordinate
                -'b' = point b coordinate
                -'k' = point normal (x)
                -'l' = point normal (y)
                -'m' = point normal (z).

              Default is 'xyzrgbklm'.
    """

    # look for/load header file if one exists
    header, data = matchHeader(path)
    if not header is None:
        header = loadHeader(header) # load header file

    # load points
    points = np.genfromtxt(data, delimiter=delimiter, skip_header=1)
    assert points.shape[1] >= 3, "Error - dataset must at-least contain x,y,z point coordinates."

    # parse order string
    order = order.lower()
    xyz = [order.find('x'), order.find('y'), order.find('z')]
    assert not -1 in xyz, "Error - order must include x,y,z data."
    rgb = [order.find('r'), order.find('g'), order.find('b')]
    norm = [order.find('k'), order.find('l'), order.find('m')]
    notS = []  # columns that have already been parsed (so are not scalar fields)

    # get xyz
    notS += xyz
    xyz = points[:, xyz]

    # try get rgb and normals
    if max(rgb) < points.shape[1] and not -1 in rgb:
        notS += rgb
        rgb = points[:, rgb]

        # if RGB is 0-255 store as uint8 to minimise memory usage
        if (rgb > 1.0).any():
            rgb = rgb.astype(np.uint8)
    else:
        rgb = None
    if max(norm) < points.shape[1] and not -1 in norm:
        notS += norm
        norm = points[:, norm].astype(np.float32) #store norm as float32 rather than float64 (default)
    else:
        norm = None

        # get remaining data as scalar field
    scal = [i for i in range(points.shape[1]) if i not in notS]
    if len(scal) > 0:
        scal = points[:, scal]
    else:
        scal = None

    # return point cloud
    return HyCloud(xyz, rgb=rgb, normals=norm, bands=scal, header=header)

def saveCloudLAS(path, cloud):
    """
    Write a point cloud object to .las. Note that LAS file formats cannot handle scalar field data.

    *Arguments*:
     - path = the .las file to save to.
     - cloud = a HyCloud instance containing data to save

    """

    try:
        import laspy
    except:
        assert False, "Please install laspy (pip install laspy) to export to LAS."

    # make directories if need be
    makeDirs( path )

    header = laspy.header.Header(point_format=2)
    outfile = laspy.file.File(path, mode='w', header=header)

    # setup header
    outfile.header.offset = [np.min(cloud.xyz[:, 0]),
                             np.min(cloud.xyz[:, 1]),
                             np.min(cloud.xyz[:, 2])]
    outfile.header.scale = [1.0, 1.0, 1.0]
    outfile.header.max = [np.max(cloud.xyz[:,0]), np.max(cloud.xyz[:,1]), np.max(cloud.xyz[:,2])]
    # export point coordinates
    outfile.X = cloud.xyz[:, 0]
    outfile.Y = cloud.xyz[:, 1]
    outfile.Z = cloud.xyz[:, 2]

    # export point colours
    if cloud.has_rgb():
        if cloud.rgb.dtype == np.uint8: #RGB is 0-255
            outfile.red = cloud.rgb[:, 0]
            outfile.green = cloud.rgb[:, 1]
            outfile.blue = cloud.rgb[:, 2]
        elif cloud.rgb.dtype is np.float or cloud.rgb.dtype is np.float32: #RGB is 0-1
            outfile.red = (cloud.rgb[:, 0] * 255).astype(np.int)
            outfile.green = (cloud.rgb[:, 1] * 255).astype(np.int)
            outfile.blue = (cloud.rgb[:, 2] * 255).astype(np.int)
        else:
            print("Warning - unknown data type for RGB colours (%s)" % cloud.rgb.dtype)

    outfile.close()

    #write header file
    if not cloud.header is None:
        hdrpth, _ = os.path.splitext( path )
        cloud.push_to_header()
        saveHeader(hdrpth + ".hdr",cloud.header)

def loadCloudLAS(path):
    """
    Loads a LAS file from the specified path.
    """

    try:
        import laspy
    except:
        assert False, "Please install laspy (pip install laspy) to load LAS."

    # look for/load header file if one exists
    header, data = matchHeader(path)
    if not header is None:
        header = loadHeader(header) # load header file

    f = laspy.file.File(data, mode='r')

    # get data
    xyz = np.array([f.x, f.y, f.z], dtype=f.x.dtype).T

    # try to get colour info
    try:
        rgb = np.array([f.get_red(), f.get_green(), f.get_blue()], dtype=f.get_blue().dtype)
    except:
        rgb = None

    # construct cloud
    return HyCloud(xyz, rgb=rgb, header=header)

def saveCloudPLY(path, cloud, sfmt=None):
    """
    Write a point cloud and associated RGB and scalar fields to .ply.

    *Arguments*:
     - path = the .ply file to save to.
     - cloud = a HyCloud instance containing data to save
     - sfmt = the format for scalar field data. Can be 'u1', 'u2' or 'f4'.
                 - 'u1' uses one byte per point per scalar field (255 possible values). This results in smaller file size.
                 - 'u2' uses two bytes per point per scalar field (65535 possible values).
                 - 'f4' uses four bytes per point per scalar field (float32 precision). This results in large files.
              Default (None) is to use u2 for integer data and f4 for float data.
    """

    # make directories if need be
    makeDirs( path )

    try:
        from plyfile import PlyData, PlyElement
    except:
        assert False, "Please install plyfile (pip install plyfile) to export to PLY."

    # calculate format?
    if cloud.has_bands():
        if sfmt is None:
            if cloud.is_int():
                sfmt='u2'
            else:
                sfmt='f4'

        #estimate size of file
        ps = 4
        if 'u2' in sfmt.lower(): ps = 2
        elif 'u4' in sfmt.lower(): ps = 4
        ps = (3 * 4 + 3 + cloud.band_count() * ps)
        file_size = cloud.point_count() * ps
        if file_size > 10e9:
            print("Warning: writing large point cloud file (%.1f Gb)." % (file_size / 1e9))

    # create structured data arrays
    vertex = np.array(list(zip(cloud.xyz[:, 0], cloud.xyz[:, 1], cloud.xyz[:, 2])),
                      dtype=[('x', 'double'), ('y', 'double'), ('z', 'double')])

    # create ply elements to write to file
    ply = [PlyElement.describe(vertex, 'vertices')]

    if cloud.has_rgb():

        # map to correct data type
        if cloud.rgb.dtype == np.uint8:
            irgb = cloud.rgb
        elif np.nanmax(cloud.rgb) <= 255 and np.nanmin(cloud.rgb) >= 0:
            irgb = cloud.rgb.astype(np.uint8)
        else:
            assert np.nanmin(cloud.rgb) >= 0 and np.nanmax(
                cloud.rgb) <= 1, "Error: rgb must be int (0-255) or float (0-1)."

            irgb = (cloud.rgb * 255).astype(np.uint8)

        # convert to structured arrays
        irgb = np.array(list(zip(irgb[:, 0], irgb[:, 1], irgb[:, 2])),
                        dtype=[('r', 'u1'), ('g', 'u1'), ('b', 'u1')])
        ply.append(PlyElement.describe(irgb, 'color'))  # create ply elements

    if cloud.has_normals():
        # convert to structured arrays
        norm = np.array(list(zip(cloud.normals[:, 0], cloud.normals[:, 1], cloud.normals[:, 2])),
                        dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
        ply.append(PlyElement.describe(norm, 'normals'))  # create ply elements

    if cloud.has_bands():

        # map scalar fields to required type
        if 'u1' in sfmt.lower():
            if cloud.data.dtype == np.uint8:
                data = cloud.data  # no need to do any mapping :)
            else:
                # map scalar field to range 0 - 1
                data = cloud.data.astype(np.float32)
                data -= np.nanmin(data)
                data = data / np.nanmax(data)

                # and compress to uint8
                data = (data * 255).astype(np.uint8)

        elif 'u2' in sfmt.lower():
            if cloud.data.dtype == np.uint16:
                data = cloud.data  # no need to do any mapping :)
            else:
                cloud.compress()
                data = cloud.data
        elif 'f4' in sfmt.lower():
            if cloud.data.dtype == np.float32:
                data = cloud.data  # no need to do any mapping :)
            elif cloud.data.dtype == np.float:
                data = cloud.data.astype(np.float32)
            else:
                cloud.decompress()
                data = cloud.data.astype(np.float32)
        else:
            assert False, "Error - %s is an invalid data format." % sfmt
        # build data arrays
        for b in range(cloud.band_count()):

            #generate band names
            n = str(b)
            if cloud.has_wavelengths():
                n = str(cloud.get_wavelengths()[b])
            elif cloud.has_band_names():
                n = str(cloud.get_band_names()[b])

            #remove spaces from n
            n = n.strip()
            n.replace(' ', '_')

            #name already includes 'scalar'?
            if 'scalar' in n:
                ply.append(PlyElement.describe(data[:, b].astype([('%s' % n, sfmt)]), '%s' % n))

            #otherwise prepend it (so CloudCompare recognises this as a scalar field).
            else:
                ply.append(PlyElement.describe(data[:, b].astype([('scalar_%s' % n, sfmt)]), 'scalar_%s' % n))

    # write
    PlyData(ply).write(path)

    #write header file
    if not cloud.header is None:
        hdrpth, _ = os.path.splitext( path )
        cloud.push_to_header()
        saveHeader(hdrpth + ".hdr",cloud.header)

def loadCloudPLY(path):
    """
    Loads a PLY file from the specified path.
    """

    try:
        from plyfile import PlyData, PlyElement
    except:
        assert False, "Please install plyfile (pip install plyfile) to load PLY."

    # look for/load header file if one exists
    header, data = matchHeader(path)
    if not header is None:
        header = loadHeader(header) # load header file

    # load file
    data = PlyData.read(data)

    # extract data
    xyz = None
    rgb = None
    norm = None
    scalar = []
    scalar_names = []
    for e in data.elements:
        if 'vert' in e.name.lower():  # vertex data
            xyz = np.array([e['x'], e['y'], e['z']]).T

            if len(e.properties) > 3:  # vertices have more than just position
                names = e.data.dtype.names

                # colour?
                if 'red' in names and 'green' in names and 'blue' in names:
                    rgb = np.array([e['red'], e['green'], e['blue']], dtype=e['red'].dtype).T

                # normals?
                if 'nx' in names and 'ny' in names and 'nz' in names:
                    norm = np.array([e['nx'], e['ny'], e['nz']], dtype=e['nx'].dtype).T

                # load others as scalar
                mask = ['red', 'green', 'blue', 'nx', 'ny', 'nz', 'x', 'y', 'z']
                for n in names:
                    if not n in mask:
                        scalar_names.append(n)
                        scalar.append(e[n])

        elif 'color' in e.name.lower():  # rgb data
            rgb = np.array([e['r'], e['g'], e['b']], dtype=e['r'].dtype).T
        elif 'normal' in e.name.lower():  # normal data
            norm = np.array([e['x'], e['y'], e['z']], dtype=e['z'].dtype).T
        else:  # scalar data
            scalar_names.append(e.properties[0].name.strip())
            scalar.append(np.array(e[e.properties[0].name], dtype=e[e.properties[0].name].dtype))

    if len(scalar) == 0:
        scalar = None
        scalar_names = None
    else:
        scalar = np.vstack(scalar).T

    assert (not xyz is None) and (xyz.shape[0] > 0), "Error - cloud has not points?"

    # return HyCloud
    return HyCloud(xyz, rgb=rgb, normals=norm, bands=scalar, band_names=scalar_names, header=header)

def loadCloudDEM(pathDEM, pathRGB=None):
    """
    Combines a DEM file and an Orthophoto from the specified path to a cloud.
    """

    # load files
    DEMdata = loadWithGDAL(pathDEM)
    if not pathRGB is None:
        RGBdata = loadWithGDAL(pathRGB)

    # create list of pixel coordinates
    pixgrid = np.meshgrid(np.arange(0,DEMdata.xdim(),1),np.arange(0,DEMdata.ydim(),1))
    pixlist = np.vstack(((pixgrid[0]).flatten(),(pixgrid[1]).flatten()))

    # transform pixel coordinates to world xy coordinates
    fx=[]
    fy=[]
    for i in range(pixlist.shape[1]):
        tx,ty= DEMdata.pix_to_world(int(pixlist[0,i]),int(pixlist[1,i]))
        fx.append(tx)
        fy.append(ty)

    # add DEM data as z information and create xyz cloud list
    fz = DEMdata.data.flatten()
    xyz = np.array([fx, fy, fz]).T

    # if orthophoto is specified, extract RGB values for each DEM point and colorize cloud
    if not pathRGB is None:
        rgb=[]
        for i in range(len(xyz)):
            wx,wy = RGBdata.world_to_pix(xyz[i,0],xyz[i,1])
            rgb.append(RGBdata.data[wx,wy,:])

    # otherwise fill rgb info with zeros
    else:
        rgb = np.zeros((len(xyz),3))

    # return HyCloud
    return HyCloud(xyz, rgb=rgb)

