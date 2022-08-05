"""
Store and manipulate point clouds and hyperclouds.
"""


import numpy as np
import scipy.spatial as spatial
from scipy.spatial import KDTree
from tqdm import tqdm

import hylite
from hylite.hydata import HyData
from hylite.hyimage import HyImage
from hylite.project import proj_persp, proj_pano, rasterize, Camera



class HyCloud( HyData ):
    """
    A class for point cloud data, including (but not limited to) hyperclouds.
    """

    def __init__(self, xyz, **kwds):
        """
        Create a HyCloud from a data array.

        Args:
            xyz (ndarray): a numpy array such that data[i] = [ x, y, z ]
            **kwds: Optional keywords include:

                 - normals = a Nx3 numpy array containing point normals. Default is None.
                 - rgb = a Nx3 numpy array containing point colors. Default is None.
                 - bands = a Nxm numpy array containing additional scalar bands (e.g. hyperspectral bands). This will become the
                            data array of this cloud.
                 - band_names = a Nxm list of names corresponding to each scalar band.
                 - wavelengths = a Nxm list of hyperspectral wavelengths corresponding to each scalar band. Should be -1 for non
                                 hyperspectral bands.


        """
        assert xyz.shape[1] == 3, "Error - data array must be a list of 3D vectors."

        #pass any arguments to constructor of HyData (header file, band names)
        super().__init__( kwds.get('bands', None) , **kwds )

        self.xyz = xyz
        self.normals = kwds.get('normals', None)
        self.rgb = kwds.get('rgb', None)

        #update header stuff
        if "band names" in kwds:
            self.header['band names'] = list(kwds.get("band_names"))
        if "wavelengths" in kwds:
            self.header['wavelength'] = np.array( kwds.get("wavelengths") )
        self.header["file type"] = "Hypercloud"

    def copy(self, data=True):
        """
        Make a (deep) copy of this point cloud.

        Args:
            data (bool): True if scalar bands should be copied as well (or just geometry). Default is True. If set to false then
                  points, colors and normals are copied, but not data.
        """

        xyz = self.xyz.copy()
        rgb = None
        normals = None
        bands = None
        header = self.header.copy()
        if self.has_rgb(): rgb = self.rgb.copy()
        if self.has_normals(): normals = self.normals.copy()
        if self.has_bands() and data: bands = self.data.copy()

        #if we don't copy data, remove band names and wavelengths from header
        if not data:
            header.set_band_names(None)
            header.set_wavelengths(None)

        return HyCloud(xyz, rgb=rgb, normals=normals, bands=bands, header=header)

    ############################
    ##General cloud properties
    ############################
    def point_count(self):
        """
        Number of points in this cloud.
        """

        return self.xyz.shape[0]

    def has_rgb(self):
        """
        Do points have defined colours?
        """

        return not self.rgb is None

    def has_normals(self):
        """
        Do points have defined normals?
        """

        return not self.normals is None

    def flip_normals(self):
        """
        Flip cloud normals
        """

        if self.has_normals():
            self.normals *= -1

    def has_bands(self):
        """
        Do points have defined scalar bands?
        """

        return not self.data is None

    ############################
    ##Data management
    ############################

    def delete_bands(self):
        """
        Delete all scalar band info associated with this cloud.
        """

        self.data = None
        if "wavelength" in self.header:
            del self.header['wavelength']
        if "band names" in self.header:
            del self.header["band names"]

    def set_bands(self, data, wavelengths=None, band_names=None, copy=False):
        """
        Set the scalar bands that are associated with this point cloud. Any exisiting bands will be overwritten.

        Args:
            data (ndarray): the new scalar bands
            wavelengths (ndarray): associated wavelength data. Default is None.
            band_names (list): associated band names. Default is None.
            copy (bool): True if the data array should be copied (rather than just copying the reference). Default is False.
        """

        assert data.shape[0] == self.point_count(), "Error - scalar bands must be defined for all points."
        if copy:
            self.data = data.copy()
        else:
            self.data = data

        if not band_names is None:
            self.set_band_names(band_names)
        if not wavelengths is None:
            self.set_wavelengths(wavelengths)

    def add_bands(self, data, wavelengths=None, band_names=None, copy=False):
        """
        Append the specified bands to any existing ones.

        Args:
            data (ndarray): the new scalar bands
            wavelengths (ndarray): associated wavelength data. Default is None.
            band_names (list): associated band names. Default is None.
            copy (bool): True if the data array should be copied (rather than just copying the reference). Default is False.
        """

        assert data.shape[0] == self.point_count(), "Error - scalar bands must be defined for all points."

        if self.data is None: # no band-data exists, simply set
            self.set_bands( data, wavelengths, band_names, copy )
        else:

            if copy:
                data = data.copy()
            else:
                data = data

            self.data = np.hstack([self.data, data]).astype(self.dtype)
            if self.has_band_names():
                self.set_band_names( list(self.get_band_names()) + list(band_names))
            if self.has_wavelengths():
                self.set_wavelengths(np.hstack( [self.get_wavelengths(), wavelengths] ))

    def filter_points(self, band, val, trim=True):
        """
        Remove points based on their scalar field values.

        Args:
            band (int,float,str): index (int), wavelength (float) or name (string) of the band to filter with.
            val (int,float,tuple,bool): the filter to use. Can be of the following forms:

                        - numeric = points with this value (e.g. 0, np.nan) are deleted.
                        - tuple = all points outside (trim = True) or inside (trim=False) the range of values are deleted.
                        - boolean ndarray = all points scored as 'True' are deleted.

            trim (bool): True if points outside a tuple val should be deleted. False if points falling within the range defined by val
                 should be deleted.
        """

        b = self.get_band_index(band)

        if isinstance(val, tuple) or isinstance(val, list):
            assert len(val) == 2, "Error - filter range (val) must have length 2."
            if trim:
                msk = np.logical_or(self.data[..., b] <= val[0], self.data[..., b] >= val[1])
            else:
                msk = np.logical_and(self.data[..., b] >= val[0], self.data[..., b] <= val[1])
        elif isinstance(val, np.ndarray):
            assert val.dtype == np.bool, "Error - only boolean arrays can be used as point masks."
            msk = val
        else:
            msk = self.data[..., b] == val

        # convert to indices
        msk = np.argwhere(msk)

        # do deletes (n.b. we don't slice here so that the memory is actually freed)
        self.data = np.delete(self.data, msk, axis=0)
        self.xyz = np.delete(self.xyz, msk, axis=0)
        if self.has_rgb():
            self.rgb = np.delete(self.rgb, msk, axis=0)
        if self.has_normals():
            self.normals = np.delete(self.normals, msk, axis=0)

    def point_neighbourhood_operation(self, radius, function, vb=False, args=()):
        """
        Apply the specified function to each point neighbourhood (points within the specified radius).

        Args:
            radius (float): the radius around each point defining the neighbourhood.
            function: the operator to call. This should have the following form: function( radius, current_id, neighbour_ids, *args ).
            vb (bool): True if a progress bar should be created. Default is True.
            args (tuple): remaininh arguments are passed to function(...).

        Returns:
            a list of values corresponding to the returned value of function for each point in this cloud (or None if the
            function has no return value).
        """

        tree = KDTree(self.xyz, leafsize=10)  # build kdtree
        loop = range(self.xyz.shape[0])
        out = []
        if vb:
            loop = tqdm(loop, leave=False)
        for n in loop:
            # get neighbours
            N = tree.query_ball_point(self.xyz[n, :], r=radius)

            # run operation
            result = function( radius, n, N, *args )
            if result is None: # no return
                out = None
            elif out is not None:
                out.append( result )
        return np.array(out)

    def despeckle(self, radius, bands, vb=True):
        """
        Despeckle scalar fields or rgb bands associated with this point cloud using a median filter.

        Args:
            radius (float): the radius to use to define point neighbourhoods for median calculation
            bands (str,list,float,tuple): the bands to apply the median filter to. Options are: 'rgb' to smooth colours, a list of integer
                   or float band wavelengths, or a tuple of length 2.
            vb (bool): True if a progress bar should be created, as this can be a slow operation.
        """

        # build appropriate smoothing function
        if isinstance(bands, str) and 'rgb' in bands:  # smooth RGB
            def median(radius, current_id, neighbour_ids):
                self.rgb[current_id, :] = np.nanmedian(
                    np.vstack([self.rgb[neighbour_ids, :], self.rgb[[current_id], :]]), axis=0)
        elif isinstance(bands, int) or isinstance(bands, float):  # smooth single band
            idx = self.get_band_index(bands)

            def median(radius, current_id, neighbour_ids):
                self.data[current_id, idx] = np.nanmedian(
                    np.hstack([self.data[neighbour_ids, idx], self.data[[current_id], idx]]), axis=0)
        elif isinstance(bands, tuple) and len(bands) == 2:  # (min,max) tuple
            idx0 = self.get_band_index(bands[0])
            idx1 = self.get_band_index(bands[1])

            def median(radius, current_id, neighbour_ids):
                self.data[current_id, idx0:idx1] = np.nanmedian(
                    np.vstack([self.data[neighbour_ids, idx0:idx1], self.data[[current_id], idx0:idx1]]), axis=0)
        elif len(bands) > 2:  # list of bands
            bands = np.array(bands)
            if bands.dtype == 'bool':
                assert len(
                    bands) == self.band_count(), "Error - band mask has invalid shape (%s) for dataset with %d bands." % (
                    bands.shape, self.band_count())
            else:
                bands = [self.get_band_index(int(b)) for b in bands]

            def median(radius, current_id, neighbour_ids):
                subset = np.vstack([self.data[neighbour_ids, :], self.data[[current_id], :]])
                self.data[current_id, bands] = np.nanmedian(subset[:, bands], axis=0)
        else:
            assert False, 'Error - %s is an invalid band descriptor' % bands

        # apply it
        self.point_neighbourhood_operation(radius, median, vb)

    def compute_normals(self, radius, vb=True):
        """
        Compute surface normals by fitting a plane to points within  the specified distance of each point in the cloud.
        Note that this can be slow... (especially for large values of radius).

        Args:
            radius (float): the search distance for points to use in the plane fitting.
            vb (bool): True if a progress bar should be created. Default is true.
        """

        # reset normals
        self.normals = np.zeros(self.xyz.shape)

        # function for computing normal on each neighbourhood
        def cmpN( r, n, N ):
            if len(N) > 3:
                # fit plane to points
                patch = self.xyz[N, :]
                patch -= np.mean(patch, axis=0)[None, :]  # convert to barycentric coords
                u, s, vh = np.linalg.svd(patch)  # fit plane using SVD
                self.normals[n, :] = vh[2, :]

        # compute normals for each point
        self.point_neighbourhood_operation( radius, cmpN, vb=vb )

        # orient normals upwards
        self.normals[self.normals[:, 2] < 0, :] *= -1

    ############################
    ## Plotting functions
    ############################
    # noinspection PyDefaultArgument
    def render(self, cam='ortho', bands=['rgb'], **kwds ):

        """
        Renders this point cloud to a HyImage using the specified camera.

        Args:
            cam (hylite.project.Camera, str): the camera to render with. Either a Camera instance or 'ortho' to render an orthographic top-down view (default).
            step (int): the image pixel angular step (in x) for panoramic images. Default is None == square pixels.
            bands (list,str, tuple): List defining the bands to include in the output image. Elements should be one of:

                - 'rgb' = rgb
                - 'xyz' = point position
                - 'klm' = normal vectors
                - numeric = index (int), wavelength (float) of a scalar field
                - tuple of length 2 = slice of scalar fields (e.g. (0,-1) would return all bands).
                - tuple of length > 2 or list: list of band indices (int) or wavelengths (float).

                Default is ['rgb'].

            **kwds: Keyword arguments can be any of:

                 - s = the point size (in pixels). Must be an integer. Default is 1.
                 - step = skip through n points for quicker plotting (default is 1 = draw all points).
                 - fill_holes = True if 1-pixel holes should be filled. Default is False.
                 - blur = Gaussian kernel size (in pixels) to blur/smooth final image with. Default is 0 (no blur). Must be odd.
                 - despeckle = Median kernel size (in pixels) to blur/smooth final image with. Default is 0 (no blur). Must be odd.
                 - res = the pixel size (in world coordinates) used to georeference the resulting raster (if creating an orthophoto).
                         Default is one thousandth of the maximum dimension (in x or y).
                 - epsg = an epsg code used to georeference the resulting render (if creating an orthophoto). Default is 32629.
                 - depth = include the depth buffer in the output image. Default is False.

        Returns:
            a HyImage object.
        """

        # get keywords
        s = kwds.get("s",1)
        step = kwds.get("step",1)
        fill_holes = kwds.get("fill_holes", False)
        blur = kwds.get("blur", 0)
        despeckle = kwds.get("despeckle", 0)
        EPSG = kwds.get("EPSG", "EPSG:32629")
        depth = kwds.get("depth", False)

        # project points
        assert isinstance(step, int) and step >= 1, "Error - step argument must be an integer equal to or larger than 1."
        assert isinstance(EPSG, str) and "EPSG" in EPSG, "Error - EPSG must be given as string in format 'EPSG:32629' (example) "
        if isinstance(cam, str):
            # project to nadir orthophoto
            if 'ortho' in cam:
                # find world coordinate size and position
                cloudxmin = np.amin(self.xyz[::step, 0])
                cloudxmax = np.amax(self.xyz[::step, 0])
                cloudymin = np.amin(self.xyz[::step, 1])
                cloudymax = np.amax(self.xyz[::step, 1])

                # compute sensible resolution
                res = kwds.get("res", None)
                if res is None: # use default value
                    res = max(cloudxmax - cloudxmin, cloudymax - cloudymin) / 1000


                cloudxsize = int(cloudxmax - cloudxmin) / res
                cloudysize = int(cloudymax - cloudymin) / res

                # project data
                C = np.array([cloudxmin,cloudymax,0])
                pp = np.abs(self.xyz[::step,:].copy() - C[None, :])/res
                vis = np.ones(pp.shape[0], dtype=bool)
                dims=(int(cloudxsize+1/res), int(cloudysize+1/res))
            else:
                assert False, "Error - unknown camera_type. Should be 'ortho', 'perspective' or 'panorama'."
        else:
            if 'persp' in cam.proj.lower():
                dims = cam.dims
                pp, vis = proj_persp(self.xyz[::step,:], C=cam.pos, a=cam.ori,
                                     fov=cam.fov, dims=dims)
            elif 'pano' in cam.proj.lower():
                dims = cam.dims
                pp, vis = proj_pano(self.xyz[::step,:], C=cam.pos, a=cam.ori,
                                    fov=cam.fov, dims=dims, step=cam.step)
            else:
                assert False, "Error - unknown camera_type. Should be 'ortho', 'perspective' or 'panorama'."

        # gather data
        data = []
        wav = []
        nam = []

        # special case: individual band; wrap in list
        if isinstance(bands, int) or isinstance(bands, float) or isinstance(bands, str):
            bands = [bands]

        # special case: tuple of two bands; treat as slice
        elif isinstance(bands, tuple) and len(bands) == 2:
            # noinspection PyUnusedLocal
            bands = [bands]

        #loop through bands tuple/list and extract data indices/slices
        for e in bands:

            #extract based on string
            if isinstance(e, str):
                for c in e.lower():
                    if  c == 'r':
                        assert self.has_rgb(), "Error - RGB information not found."
                        data.append(self.rgb[::step,0])
                        nam.append('r')
                        wav.append( hylite.RGB[0] )
                    elif c == 'g':
                        assert self.has_rgb(), "Error - RGB information not found."
                        data.append(self.rgb[::step, 1])
                        nam.append('g')
                        wav.append( hylite.RGB[1] )
                    elif c == 'b':
                        assert self.has_rgb(), "Error - RGB information not found."
                        data.append(self.rgb[::step, 2])
                        nam.append('b')
                        wav.append( hylite.RGB[2] )
                    elif c == 'x':
                        data.append(self.xyz[::step, 0])
                        nam.append('x')
                        wav.append(-1)
                    elif c == 'y':
                        data.append(self.xyz[::step, 1])
                        nam.append('y')
                        wav.append(-1)
                    elif c == 'z':
                        data.append(self.xyz[::step, 2])
                        nam.append('z')
                        wav.append(-1)
                    elif c == 'k':
                        assert self.has_normals(), "Error - normals not found."
                        data.append(self.normals[::step, 0])
                        nam.append('k')
                        wav.append(-1)
                    elif c == 'l':
                        assert self.has_normals(), "Error - normals not found."
                        data.append(self.normals[::step, 1])
                        nam.append('l')
                        wav.append(-1)
                    elif c == 'm':
                        assert self.has_normals(), "Error - normals not found."
                        data.append(self.normals[::step, 2])
                        nam.append('m')
                        wav.append(-1)
            #extract slice
            elif isinstance(e, tuple):
                assert self.has_bands(), "Error -  band data not found."
                assert len(e) == 2, "Error - band slices must be tuples of length two."
                idx0 = self.get_band_index(e[0])
                idx1 = self.get_band_index(e[1])

                data.append(self.data[::step, idx0 : idx1])
                if self.has_band_names():
                    nam += [ self.get_band_names()[b] for b in range(idx0, idx1) ]
                else:
                    nam += [ str(b) for b in range(idx0, idx1) ]

                if self.has_wavelengths():
                    wav += [ self.get_wavelengths()[b] for b in range(idx0, idx1)]
                else:
                    wav += [ float(b) for b in range(idx0, idx1) ]
            #extract band based on index or wavelength
            elif isinstance(e,float) or isinstance(e,int):
                b = self.get_band_index( e )
                data.append(self.data[::step, b])
                if self.has_band_names():
                    nam.append(self.get_band_names()[b])
                else:
                    nam.append(str(b))
                if self.has_wavelengths():
                    wav.append(self.get_wavelengths()[b])
                else:
                    wav.append(float(b))
            else:
                assert False, "Unrecognised band descriptor %s" % b

        # rasterise and make HyImage
        assert len(data) > 0, "Error - no bands found to project?"
        _vals, _d = rasterize(pp, vis, data, dims, s)
        if depth: # add depth (z) buffer as last channel
            img = HyImage(np.dstack([_vals, _d]), header=self.header.copy() )
            img.set_band_names(nam + ['depth'])
            img.set_wavelengths(wav + [-1.0])
        else:
            img = HyImage( _vals, header=self.header.copy() )
            img.set_band_names(nam)
            img.set_wavelengths(wav)

        # if orthophoto is saved, define geotransform and project
        if isinstance(cam, str):
            if 'ortho' in cam:
                img.affine = ([cloudxmin, res, 0,  cloudymax, 0, -res])
                try:
                    img.set_projection_EPSG(EPSG)
                except:
                    #print("Warning - could not set orthoimage projection. Check GDAL installation.")
                    pass # silently continue if no GDAL available

        # postprocessing
        if fill_holes:
            img.fill_holes()
        if blur == True:
            blur = 3
        if despeckle == True:
            despeckle = 5
        if blur > 2:
            img.blur(int(blur))
        if despeckle > 2:
            img.despeckle(int(despeckle))
        return img

    def quick_plot(self, band='rgb', cam='ortho', s=1, step=1, fill_holes=False, blur=False, despeckle=False, res=None, **kwds):

        """
        Renders this point cloud using the specified camera.

        Args:
            band (str,float,int,tuple): the bands to plot. 'rgb' will plot colour, 'norm' will plot normals, 'xyz' will plot
                   coordinates. Or an index or tuple of 3-indices (mapped to rgb) can be passed to plot scalar fields.
            cam (hylite.project.Camera, str): the camera to render with. Default is 'ortho' (top down).
            s (int): point size (in pixels; must be an integer).
            step (int): skip through n points for quicker plotting (default is 1 = draw all points).
            fill_holes (bool): True if 1-pixel holes should be filled. Default is False.
            blur (bool): True if a 3x3 gaussian blur kernel is used to smooth the scene. Default is False.
            despeckle (bool): True if a 5x5 median filter should be used to denoise rendered image before plotting. Default is False.
            res (float): the resolution to plot in 'ortho' mode. Default is one thousandth of the maximum dimension (in x or y).
            **kwds: other keywords are passed to HyImage.quick_plot( ... ).

        Returns:
            Tuple containing

            - fig: the plot figure
            - ax: the plot axis
        """

        # render image
        img = self.render(cam, band, s=s, step=step, fill_holes=fill_holes, blur=blur, despeckle=despeckle, res=res)

        # plot image
        if img.band_count() >= 3:  # we have enough bands to map to rgb
            if 'rgb' in band: # edge case - rgb values should map from 0 to 1!
                kwds['vmin'] = kwds.get('vmin', 0.)
                if np.issubdtype(self.rgb.dtype, np.integer):
                    kwds['vmax'] = kwds.get('vmax', 255.)
                else:
                    kwds['vmax'] = kwds.get('vmax', 1.0)
                return img.quick_plot((0, 1, 2), **kwds)
            else:
                return img.quick_plot((0, 1, 2), **kwds)
        else:
            return img.quick_plot(0, **kwds)

    def colourise(self, bands, stretch=(1, 99)):

        """
        Map the specified bands to this clouds RGB using the specified percentile stretch.

        Args:
            bands (list, tuple): a list of 3 bands to map to r, g and b (respectively). Can be indices (integer) or wavelengths (float).
            stretch (tuple): a tuple containing the (min,max) percentiles (integer) of values (float) to use for the colour stretch. Default is (1,99).
        """

        # for greyscale mapping
        if isinstance(bands, int) or isinstance(bands, float):
            bands = [bands, bands, bands]
        if len(bands) == 1:
            bands = [bands[0], bands[0], bands[0]]
        assert len(bands) == 3, "Error - bands must have length three. No more. No less. Three."

        # convert to band indexes
        bands = list(bands)
        for i, b in enumerate(bands):
            bands[i] = self.get_band_index(b)

        self.rgb = self.data[:, bands].astype(np.float)

        # normalise bands
        for i, b in enumerate(bands):
            if 'data ignore value' in self.header:  # deal with data ignore values for integer arrays...
                self.rgb[self.rgb[:, i] == int(self.header['data ignore value'])] = np.nan  # make nans

            # calculate stretch values for normalisation
            minv, maxv = stretch
            if isinstance(minv, int) and isinstance(maxv, int):
                minv, maxv = np.nanpercentile(self.rgb[:, i], (minv,maxv))
            elif isinstance(minv, int):
                minv = np.nanpercentile(self.rgb[:, i], minv)
            elif isinstance(maxv, int):
                minv = np.nanpercentile(self.rgb[:, i], maxv)

            # normalise to range 0 - 1
            self.rgb[:, i] = (self.rgb[:, i] - minv) / (maxv - minv)

        # convert to uint8 (0 - 255)
        self.rgb[self.rgb < 0] = 0
        self.rgb[self.rgb > 1] = 1
        self.rgb = (self.rgb * 255.0).astype(np.uint8)

    def project(self, image, cam, bands=None, band_names=None, occ_tol=1.0, vb=True, ignore_zeros=True, trim=True):
        """
        Projects an image into this point cloud such that each a new scalar field is defined for each point.

        Args:
            image (hylite.HyImage): the image to project (as a HyImage object) or list of images.
            cam (hylite.project.Camera): the camera to project through, or a list of cameras (of the same length as image).
            bands (list,float): the bands to project (a scalar field is created for each band). Default (None) projects
                   all bands.
            band_names (list): names of the bands. Default is None (bands are given integer names).
            occ_tol (float): the occlusion tolerance (in point cloud coordinate system). Points within this distance of the z-buffer will be attributed with
                     projected data. Set to 0 to do no occlusion (i.e. all points along each camera ray will be attributed).
            vb (bool): True if progress bar should be created when projecting multiple images. Default is True.
            ignore_zeros (bool): True if zero data is not projected (but instead treated as "transparent"). Default is True.
            trim (bool) True if points that did not get any data projected onto them should be deleted from the cloud. Default is True.
        """

        # wrap in lists
        if isinstance(image, HyImage):
            image = [image]
        if isinstance(cam, Camera):
            cam = [cam]

        assert len(image) == len(cam), "Error - a camera position is required for every image."

        #calculate output data type
        if self.data is None:
            dtype = image[0].data.dtype
        else:
            dtype = self.data.dtype

        ###########################################
        # Extract header info from first image
        ###########################################
        if bands is None:
            bid = np.arange(0, image[0].band_count())  # all bands
        else:
            bid = [image[0].get_band_index(b) for b in bands]

        #extract band names from first image
        if band_names is None:
            if image[0].has_wavelengths():
                band_names = np.array(image[0].get_wavelengths())[bid]
            else:
                band_names = bid
        band_names = [str(b) for b in band_names] #ensure band names are string

        #extract wavelengths from first image
        wavelengths = None
        if image[0].has_wavelengths():
            wavelengths = np.array(image[0].get_wavelengths())[bid]

        # calculate value to use as nan
        nan = image[0].header.get('data ignore value', np.nan)

        #check size of data array.... (and warn if it's biiiiiiig)
        size = image[0].data.dtype.itemsize * self.point_count() * len(bid) / 1e9
        if size > 10:
            print("Warning - generating large hypercloud (%.1f Gb). You might run out of RAM and die?" % size, flush=True)

        # initialise data array
        if np.issubdtype( image[0].data.dtype, np.integer ):
            data = np.zeros((self.point_count(), len(bid)), dtype=np.uint32) #sum values as integer
        else:
            data = np.zeros((self.point_count(), len(bid)), dtype=np.float32) #we need floating point precision
        count = np.zeros((self.point_count(), len(bid)), dtype=np.uint32)

        # loop through images
        if vb and len(image) > 1:
            loop = tqdm(range(len(image)),leave=False)
        else:
            loop = range(len(image))
        for i in loop:

            #get camera
            _cam = cam[i]

            #get image
            _image = image[i]

            #update band wavelengths (if need be)
            #n.b. this is needed as the band indices may change if wavelengths are specified...
            if not bands is None:
                bid = [_image.get_band_index(b) for b in bands]

            # project points
            if 'persp' in _cam.proj.lower():
                pp, vis = proj_persp(self.xyz, C=_cam.pos, a=_cam.ori,
                                     fov=_cam.fov, dims=_image.data.shape)
            elif 'pano' in _cam.proj.lower():
                pp, vis = proj_pano(self.xyz, C=_cam.pos, a=_cam.ori,
                                    fov=_cam.fov, dims=_image.data.shape, step=_cam.step)
            else:
                assert False, "Error - unknown camera_type. Should be 'perspective' or 'panorama'."

            # no visible points...
            if not vis.any(): print("Warning: no visible points in image %d" % i, flush=True)

            # rasterise to make depth map
            _vals, _d = rasterize(pp, vis, self.xyz, _image.data.shape, 1)

            # mask occluded points
            if occ_tol != 0:  # occlusion is enabled
                vis[vis == True] = np.abs(
                    _d[pp[vis, 0].astype(np.int), pp[vis, 1].astype(np.int)] - pp[vis, 2]) <= occ_tol

            # extract pixels
            pixels = _image.data[pp[vis, 0].astype(np.int), pp[vis, 1].astype(np.int)][:, bid]
            valid = np.zeros(pixels.shape, dtype=count.dtype) + 1  # valid pixels

            #deal with nans/missing data
            for nanpixels in [np.isnan(pixels), pixels == nan ]: #nan == 0 for integer data...
                valid[nanpixels] = 0
                pixels[nanpixels] = 0

            #ignore zeros also?
            if ignore_zeros and nan != 0:
                valid[pixels == 0] = 0

            #accumulate
            data[vis, :] += pixels
            count[vis, :] += valid

        # calculate averages
        if np.issubdtype(data.dtype, np.integer): #use floor division
            data = data // count #data / count #np.divide(data, count[...,None], dtype = data.dtype)\
        else: #use floating point division
            data = data / count

        # store data to the scalar field array
        if not band_names is None:
            assert data.shape[-1] == len(band_names), \
                "Error - data has %d bands but %d names?" % (data.shape[-1],len(band_names))
        if not wavelengths is None:
            assert data.shape[-1] == len(wavelengths), \
                "Error - data has %d bands but %d wavelengths?" % (data.shape[-1], len(wavelengths))

        self.add_bands(data.astype(dtype), band_names, wavelengths )

        # trim
        if trim:
            self.filter_points( 0, np.sum(count,axis=-1) == 0 ) #remove zero points