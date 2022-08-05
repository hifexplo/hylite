"""
Combine 2D image data and 3D point cloud information in one data structure that facilitates transfer of information
between images and point clouds.
"""

from hylite import HyCollection
from hylite.project import PMap, Pushbroom, Camera, proj_persp, proj_pano, project_pushbroom, push_to_cloud, push_to_image
from tqdm import tqdm
from scipy.ndimage import grey_dilation
import numpy as np

class HyScene( HyCollection ):
    """
    A special type of HyCollection and contains a projection map for transferring information between a point cloud and
    a hyperspectral image.
    """
    def __init__(self, name, root, header=None ):
        """
        Args:
            name (str): unique name for this scene
            root (str): path to folder where scene will be saved to disk
            header (str): header information for this scene, or None.
        """

        super().__init__( name, root, header )
        self.ext = '.hys'

    def getAttributes(self):
        """
        Return a list of available attributes in this HyScene. We must override the HyCollection implementation to remove
        functions associated with HyScene.
        """
        return list(set(dir(self)) - set(dir(HyCollection)) - set(dir(HyScene)) - set(['header', 'root', 'name', 'ext']))

    def construct(self, image, cloud, camera, s=1, occ_tol = 10, maxf=0, bf=True, vb = True, **kwds):
        """
        Construct a mapping between the specified image and cloud based on the camera position / orientation / track.

        Args:
            image (HyImage): data to map onto the point cloud.
            cloud (HyCloud): data/geometry to map onto the image.
            camera (hylite.project.Camera or hylite.project.Pushbroom):  Object describing the projection geometry.
            s (int): a dilation to apply when mapping point data to the image (to fill gaps/holes). Default is 1 (do not apply a dilation).
                If s is an integer then an (s,s) dilation filter is applied. Alternatively, s can be a tuple such that s=(n,m)
                defining the 2-D dimensions of the dilation (useful for e.g. pushbroom data).
            occ_tol (float): the distance between a point and the z-buffer before it becomes occluded. Default is 10. Set to 0 to
                    disable occlusion.
            maxf (int): the maximum acceptible pixel footprint. Pixels containing > than this number of points will be excluded
                   from the dataset. Set as 0 to disable (default).
            bf (bool): True if backface culling (using cloud normal vectors) should be applied during projection. Default is True.
            **kwds: Keywords are passed to project_pushbroom for pushbroom type cameras.
        """

        # check dimensions match
        assert camera.dims[0] == image.xdim() and camera.dims[
            1] == image.ydim(), "Error - image and camera dimensions do not match."

        # store reference to associated datasets and properties
        self.image = image
        self.cloud = cloud
        self.camera = camera
        self.occ_tol = occ_tol
        self.maxf = maxf

        # build projection map
        self.pmap = PMap(camera.dims[0], camera.dims[1], cloud.point_count(), cloud=cloud, image=image)
        normals = None
        if bf:
            normals = cloud.normals
        if isinstance(self.camera, Camera): # conventional camera
            if 'persp' in camera.proj:
                pp, vis = proj_persp(cloud.xyz, C=camera.pos, a=camera.ori,
                                     fov=camera.fov, dims=camera.dims, normals=normals)
                self.pmap.set_ppc( pp, vis )
            elif 'pano' in camera.proj:
                pp, vis = proj_pano(cloud.xyz, C=camera.pos, a=camera.ori,
                                    fov=camera.fov, dims=camera.dims, step=camera.step, normals=normals)
                self.pmap.set_ppc(pp, vis)
            else:
                assert False, "Error, %s is an incompatible projection type" % camera.proj
        elif isinstance(self.camera, Pushbroom):
                self.pmap = project_pushbroom( self.image, self.cloud, self.camera, vb=vb, **kwds )
        else:
            assert False, "Error, %s is an unknown camera type" % type(self.camera)


        # filter occlusions
        if vb:
            prg = tqdm(total=6, leave=False)
            prg.set_description("Filtering occlusions")
        if self.occ_tol > 0:
            self.pmap.filter_occlusions( self.occ_tol )

        # filter footprint
        if maxf > 0:
            if vb:
                prg.set_description("Filtering by footprint")
                prg.update(1)
            self.pmap.filter_footprint( self.maxf )

        # do projections
        if vb:
            prg.set_description("Converting geometry")
            prg.update(2)
        if cloud.has_normals():
            xyzklm = push_to_image( self.pmap, 'xyzklm', method='average').data
            self.xyz = xyzklm[...,[0,1,2]]
            self.normals = xyzklm[..., [3, 4, 5]]
        else:
            self.xyz = push_to_image( self.pmap, 'xyz', method='average').data
            self.normals = None

        # build depth map
        if vb:
            prg.set_description("Building depth map")
            prg.update(3)
        self.depth = self.pmap.get_pixel_depths()

        # build viewvector map
        if vb:
            prg.set_description("Computing view vectors")
            prg.update(5)
        if isinstance( self.camera, Camera ): # fixed view position
            self.view = self.xyz - self.camera.pos
        else: # camera positions changes according to pushbroom track
            self.view = self.xyz - self.camera.cp
        self.view = self.view / np.linalg.norm(self.view, axis=-1)[...,None] # normalise

        # apply dilations
        if isinstance(s, tuple) or s > 1:
            if vb:
                prg.set_description("Applying dilations")
                prg.update(6)
            if isinstance(s,int):
                s = (s,s)
            elif isinstance(s, tuple):
                assert len(s) == 2, "Error - s must be a tuple of shape (n,m)."
            else:
                assert False, "Error, %s is an invalid type for s." % type(s)

            # loop through attributes and apply dilation
            for a,v in zip(['xyz','normals','depth', 'view'], [self.xyz, self.normals, self.depth, self.view]):
                nanmask = np.logical_not(np.isfinite(v))
                v[nanmask] = -np.inf  # replace nans with -infinity
                if (len(v.shape) == 2):
                    k = np.ones(s)
                elif (len(v.shape) == 3):
                    k = np.ones( s + (1,))
                dil = grey_dilation(v, footprint=k)
                v[nanmask] = dil[nanmask]  # where we have new finite values from dilation, add them in
                v[np.isinf(v)] = np.nan  # add nans back again



        if vb:
            prg.set_description("Complete")
            prg.close()

    def get_pixel_normal(self, px, py):
        """
        Get the average normal vector of all points in the specified pixel.
        """
        assert self.cloud.has_normals(), "Error - no normals are available (on point cloud)."
        return self.normals[px,py,:]

    def get_point_normal(self, index):
        """
        Get the normal vector of the specified point.
        """
        assert self.cloud.has_normals(), "Error - no normals are available (on point cloud)."
        return self.cloud.normals[index,:]

    def get_xyz(self):
        """
        Get per-pixel positions in world coords (as a numpy array)
        """
        return self.xyz

    def get_normals(self):
        """
        Get per-pixel normals (as a numpy array).
        """
        return self.normals

    def get_depth(self):
        """
        Get per-pixel depth array.
        """
        return self.depth

    def get_GSD(self):
        """
        Get per-pixel ground sampling distance (pixel size). Assumes square pixels.

        Returns:
            gsd (numpy array): sampling distance (GSD) in meters.
        """
        # calculate pixel pitch in degrees
        if isinstance(self.camera, Camera): # normal camera
            pitch = np.deg2rad(self.camera.dims[1] / self.camera.fov)
        else: # pushbroom camera
            pitch = np.deg2rad( 1 / self.camera.xfov )

        # calculate GSD and return
        return 2 * self.depth * np.tan(pitch / 2)

    def get_obliquity(self):
        """
        Get array obliquity angles (degrees) between camera look direction and surface normals. Note that this angle
        will be 0 if the look direction is perpendicular to the surface.
        """
        return np.dot(self.view, self.normals)

    def get_view_dir(self):
        """
        Get per-pixel viewing direction vector (normalised to length = 1).
        """
        return self.view

    def get_slope(self):
        """
        Get array of slope angles for each pixel (based on the surface normal vectors).
        """
        return np.rad2deg(np.arccos(np.abs(self.normals[..., 2])))

    def push_to_cloud(self, bands, method='best', image=None, cloud=None):
        """
        Push attributes from this scenes image to this scene's cloud. Is a wrapper around hylite.project.push_to_cloud(...).

        Args:
            bands (int, float, tuple, list): either:
                     (1) a index (int), wavelength (float) of a (single) image band to export.
                     (2) a tuple containing the (min,max) wavelength to extract. If range is a tuple, -1 can be used to specify the
                         first or last band index.
                     (3) a list of bands or boolean mask such that image.data[:,:,range] is exported.
            method (string): The method used to condense data from multiple pixels onto each point. Options are:

                             - 'closest': use the closest pixel to each point.
                             - 'distance': average with inverse distance weighting.
                             - 'count' : average weighted inverse to the number of points in each pixel.
                             - 'best' : use the pixel that is mapped to the fewest points (only). Default.
                             - 'average' : average with all pixels weighted equally.

            image (hylite.HyImage): an alternative image to use (defaults to self.image) Shapes must match self.pmap.
            cloud (hylite.HyCloud): an alternative cloud to use (defaults to self.cloud). Shapes must match self.pmap.

        Returns:
            A HyCloud instance containing the back-projected data.
        """
        if cloud is None:
            cloud = self.cloud
        if image is None:
            image = self.image

        return push_to_cloud( self.pmap, bands, method, image=image, cloud=cloud)

    def push_to_image(self, bands, method='closest', image=None, cloud=None):
        """
        Push attributes from this scenes cloud to this scenes image. Is a wrapper around hylite.project.push_to_image(...).

        Args:
            bands (int,float,str,tuple,list): List defining the bands to include in the output dataset. Elements should be one of:
                  - numeric = index (int), wavelength (float) of an image band
                  - bands = a list of image band indices (int) or wavelengths (float). Inherent properties of point clouds
                       can also be expected by passing any combination of the following:

                        - 'rgb' = red, green and blue per-point colour values
                        - 'klm' = point normals
                        - 'xyz' = point coordinates

                  - iterable of length > 2: list of bands (float or integer) to export.
            method (str): The method used to condense data from multiple points onto each pixel. Options are:

                         - 'closest': use the closest point to each pixel (default is this is fastest).
                         - 'average' : average with all pixels weighted equally. Slow.

            image (hylite.HyImage): an alternative image to use (defaults to self.image) Shapes must match self.pmap.
            cloud (hylite.HyCloud): an alternative cloud to use (defaults to self.cloud). Shapes must match self.pmap.

        Returns:
            A HyImage instance containing the projected data.
        """
        if cloud is None:
            cloud = self.cloud
        if image is None:
            image = self.image

        return push_to_image( self.pmap, bands, method, image=image, cloud=cloud )

    # def match_colour_to(self, reference, uniform=True, method='norm', inplace=True):
    #
    #     """
    #     Identifies matching pixels between two hyperspectral scenes and uses them to minimise
    #     colour differences using a linear model (aka by adjusting the brightness/contrast of this scene
    #     to match the brightness/contrast of the reference scene). WARNING: by default this modifies this scene's
    #     image IN PLACE.
    #
    #     *Arguments*:
    #     - reference = the scene to match colours to.
    #     - uniform = True if a single brightness contrast adjustment is applied to all bands (avoids introducing spectral
    #              artefacts). If False, different corrections area applied to each band - use with CARE! Default is True.
    #     - method = The colour matching method to use. Current options are:
    #                 - 'norm' = centre-means and scale to match standard deviation. Only compares points known to match.
    #                 - 'hist' = histogram equalisation. Applies to all pixels in scene - use with care!
    #                Default is 'norm'.
    #     - inplace = True if the correction should be applied to self.image in-place. If False, no correction is
    #               applied, and the correction weights (cfac and mfac) returned for future use. Default is True.
    #     *Returns*:
    #     - The corrected image as a HyImage object. If inplace=True (default) then this will be the same as self.image.
    #     """
    #
    #     image = self.image
    #     if not inplace:
    #         image = image.copy()
    #
    #     if 'norm' in method.lower():
    #         # get matching pixels
    #         px1, px2 = self.intersect_pixels(reference)
    #         assert px1.shape[0] > 0, "Error - no overlap between images."
    #         if px1.shape[0] < 1000:
    #             print("Warning: images only have %d overlapping pixels,"
    #                " which may result in poor colour matching." % px1.shape[0])
    #
    #         # extract data to create vector of matching values
    #         px1 = image.data[px1[:, 0], px1[:, 1], :]
    #         px2 = reference.image.data[px2[:, 0], px2[:, 1], :]
    #
    #         # apply correction
    #         image.data = norm_eq( image.data, px1, px2, per_band=not uniform, inplace=True)
    #
    #     elif 'hist' in method.lower():
    #         if uniform: # apply to whole dataset
    #             image.data = hist_eq(image.data, reference.image.data)
    #         else: # apply per band
    #             for b in range(self.image.band_count()):
    #                 image.data[:, :, b] = hist_eq(image.data[:, :, b], reference.image.data[:, :, b])
    #     else:
    #         assert False, "Error - %s is an unrecognised colour correction method." % method
    #
    #     return image
    #
    # ###################################
    # ##PLOTTING AND EXPORT FUNCTIONS
    # ###################################
    # def _gather_bands(self, bands):
    #     """
    #     Utility function used by push_to_image( ... ) and push_to_cloud( ... ).
    #     """
    #
    #     # extract wavelength and band name info
    #     wav = []
    #     nam = []
    #
    #     # loop through bands tuple/list and extract data indices/slices
    #     for e in bands:
    #         # extract from point cloud based on string
    #         if isinstance(e, str):
    #             for c in e.lower():
    #                 if c == 'r':
    #                     assert self.cloud.has_rgb(), "Error - RGB information not found."
    #                     nam.append('r')
    #                     wav.append(hylite.RGB[0])
    #                 elif c == 'g':
    #                     assert self.cloud.has_rgb(), "Error - RGB information not found."
    #                     nam.append('g')
    #                     wav.append(hylite.RGB[1])
    #                 elif c == 'b':
    #                     assert self.cloud.has_rgb(), "Error - RGB information not found."
    #                     nam.append('b')
    #                     wav.append(hylite.RGB[2])
    #                 elif c == 'x':
    #                     nam.append('x')
    #                     wav.append(-1)
    #                 elif c == 'y':
    #                     nam.append('y')
    #                     wav.append(-1)
    #                 elif c == 'z':
    #                     nam.append('z')
    #                     wav.append(-1)
    #                 elif c == 'k':
    #                     assert self.cloud.has_normals(), "Error - normals not found."
    #                     nam.append('k')
    #                     wav.append(-1)
    #                 elif c == 'l':
    #                     assert self.cloud.has_normals(), "Error - normals not found."
    #                     nam.append('l')
    #                     wav.append(-1)
    #                 elif c == 'm':
    #                     assert self.cloud.has_normals(), "Error - normals not found."
    #                     nam.append('m')
    #                     wav.append(-1)
    #         # extract slice (from image)
    #         elif isinstance(e, tuple):
    #             assert len(e) == 2, "Error - band slices must be tuples of length two."
    #             idx0 = self.image.get_band_index(e[0])
    #             idx1 = self.image.get_band_index(e[1])
    #             if self.image.has_band_names():
    #                 nam += [self.image.get_band_names()[b] for b in range(idx0, idx1)]
    #             else:
    #                 nam += [str(b) for b in range(idx0, idx1)]
    #             if self.image.has_wavelengths():
    #                 wav += [self.image.get_wavelengths()[b] for b in range(idx0, idx1)]
    #             else:
    #                 wav += [float(b) for b in range(idx0, idx1)]
    #         # extract band based on index or wavelength
    #         elif isinstance(e, float) or isinstance(e, int):
    #             b = self.image.get_band_index(e)
    #             if self.image.has_band_names():
    #                 nam.append(self.image.get_band_names()[b])
    #             else:
    #                 nam.append(str(b))
    #             if self.image.has_wavelengths():
    #                 wav.append(self.image.get_wavelengths()[b])
    #             else:
    #                 wav.append(float(b))
    #         else:
    #             assert False, "Unrecognised band descriptor %s" % b
    #
    #     return wav, nam
    #
    # def push_to_image(self, bands, fill_holes=False, blur=0):
    #     """
    #     Export data from associated cloud and image to a (new) HyImage object.
    #
    #     *Arguments*:
    #      - bands = a list of image band indices (int) or wavelengths (float). Inherent properties of point clouds
    #                can also be expected by passing any of the following:
    #                 - 'rgb' = red, green and blue per-point colour values
    #                 - 'klm' = point normals
    #                 - 'xyz' = point coordinates
    #      - fill_holes = post-processing option to fill single-pixel holes with maximum value from adjacent pixels. Default is False.
    #      - blur = size of gaussian kernel to apply to image in post-processing. Default is 0 (no blur).
    #     *Returns*:
    #      - a HyImage object containing the requested data.
    #     """
    #
    #     # special case: individual band; wrap in list
    #     if isinstance(bands, int) or isinstance(bands, float) or isinstance(bands, str):
    #         bands = [bands]
    #
    #     # special case: tuple of two bands; treat as slice
    #     if isinstance(bands, tuple) and len(bands) == 2:
    #         bands = [bands]
    #
    #     # gather bands and extract wavelength and name info
    #     wav, nam = self._gather_bands(bands)
    #
    #     # rasterise and make HyImage
    #     img = np.full((self.image.xdim(), self.image.ydim(), len(wav)), np.nan)
    #     for _x in range(self.image.xdim()):
    #         for _y in range(self.image.ydim()):
    #             if not self.valid[_x, _y]:
    #                 continue
    #             pID = self.get_point_index(_x, _y)
    #             n = 0
    #             for e in bands:
    #                 if isinstance(e, str):  # extract from point cloud based on string
    #                     for c in e.lower():
    #                         if c == 'r':
    #                             img[_x, _y, n] = self.cloud.rgb[pID, 0]
    #                         elif c == 'g':
    #                             img[_x, _y, n] = self.cloud.rgb[pID, 1]
    #                         elif c == 'b':
    #                             img[_x, _y, n] = self.cloud.rgb[pID, 2]
    #                         elif c == 'x':
    #                             img[_x, _y, n] = self.cloud.xyz[pID, 0]
    #                         elif c == 'y':
    #                             img[_x, _y, n] = self.cloud.xyz[pID, 1]
    #                         elif c == 'z':
    #                             img[_x, _y, n] = self.cloud.xyz[pID, 2]
    #                         elif c == 'k':
    #                             img[_x, _y, n] = self.normals[_x, _y, 0]
    #                         elif c == 'l':
    #                             img[_x, _y, n] = self.normals[_x, _y, 1]
    #                         elif c == 'm':
    #                             img[_x, _y, n] = self.normals[_x, _y, 2]
    #                         n += 1
    #                     continue
    #                 elif isinstance(e, tuple):  # extract slice (from image)
    #                     assert len(e) == 2, "Error - band slices must be tuples of length two."
    #                     idx0 = self.image.get_band_index(e[0])
    #                     idx1 = self.image.get_band_index(e[1])
    #                     slc = self.image.data[_x, _y, idx0:idx1]
    #                     img[_x, _y, n:n + len(slc)] = slc
    #                     n += len(slc)
    #                     continue
    #                 elif isinstance(e, float) or isinstance(e, int):  # extract band based on index or wavelength
    #                     b = self.image.get_band_index(e)
    #                     img[_x, _y, n] = self.image.data[_x, _y, b]
    #                     n += 1
    #                     continue
    #                 else:
    #                     assert False, "Unrecognised band descriptor %s" % b
    #
    #     # build HyImage
    #     img = HyImage(img, header=self.image.header.copy())
    #     img.set_band_names(nam)
    #     img.set_wavelengths(wav)
    #
    #     # postprocessing
    #     if fill_holes:
    #         img.fill_holes()
    #     if blur > 2:
    #         img.blur(int(blur))
    #
    #     return img
    #
    # def push_to_cloud(self, bands):
    #     """
    #     Export data from associated image and cloud to a (new) HyCloud object.
    #
    #     *Arguments*:
    #      - bands = a list of image band indices (int) or wavelengths (float). Inherent properties of point clouds
    #                can also be expected by passing any of the following:
    #                 - 'rgb' = red, green and blue per-point colour values
    #                 - 'klm' = point normals
    #                 - 'xyz' = point coordinates
    #     *Returns*:
    #      - a HyImage object containing the requested data.
    #     """
    #
    #     # special case: individual band; wrap in list
    #     if isinstance(bands, int) or isinstance(bands, float) or isinstance(bands, str):
    #         bands = [bands]
    #
    #     # special case: tuple of two bands; treat as slice
    #     if isinstance(bands, tuple) and len(bands) == 2:
    #         bands = [bands]
    #
    #     # gather bands and extract wavelength and name info
    #     wav, nam = self._gather_bands(bands)
    #
    #     # loop through points in cloud and add data
    #     data = np.full((self.cloud.point_count(), len(wav)), np.nan)
    #     valid = np.full(self.cloud.point_count(), False, dtype=np.bool)
    #     for i in range(self.cloud.point_count()):
    #         # is point visible?
    #         _x, _y = self.get_pixel(i)
    #         if _x is None:
    #             continue
    #
    #         valid[i] = True  # yes - this point has data
    #
    #         # gather data
    #         n = 0
    #         for e in bands:
    #             if isinstance(e, str):  # extract from point cloud based on string
    #                 for c in e.lower():
    #                     if c == 'r':
    #                         data[i, n] = self.cloud.rgb[i, 0]
    #                     elif c == 'g':
    #                         data[i, n] = self.cloud.rgb[i, 1]
    #                     elif c == 'b':
    #                         data[i, n] = self.cloud.rgb[i, 2]
    #                     elif c == 'x':
    #                         data[i, n] = self.cloud.xyz[i, 0]
    #                     elif c == 'y':
    #                         data[i, n] = self.cloud.xyz[i, 1]
    #                     elif c == 'z':
    #                         data[i, n] = self.cloud.xyz[i, 2]
    #                     elif c == 'k':
    #                         data[i, n] = self.cloud.normals[i, 0]
    #                     elif c == 'l':
    #                         data[i, n] = self.cloud.normals[i, 1]
    #                     elif c == 'm':
    #                         data[i, n] = self.cloud.normals[i, 2]
    #                     n += 1
    #                 continue
    #             elif isinstance(e, tuple):  # extract slice (from image)
    #                 assert len(e) == 2, "Error - band slices must be tuples of length two."
    #                 idx0 = self.image.get_band_index(e[0])
    #                 idx1 = self.image.get_band_index(e[1])
    #                 slc = self.image.data[_x, _y, idx0:idx1]
    #                 data[i, n:(n + len(slc))] = slc
    #                 n += len(slc)
    #                 continue
    #             elif isinstance(e, float) or isinstance(e, int):  # extract band based on index or wavelength
    #                 b = self.image.get_band_index(e)
    #                 data[i, n] = self.image.data[_x, _y, b]
    #                 n += 1
    #                 continue
    #             else:
    #                 assert False, "Unrecognised band descriptor %s" % b
    #
    #     # build HyCloud
    #     cloud = self.cloud.copy(data=False)
    #     cloud.data = data
    #     cloud.filter_points(0, np.logical_not(valid))  # remove points with no data
    #     cloud.set_band_names(nam)
    #     cloud.set_wavelengths(wav)
    #
    #     return cloud
    #
    # def quick_plot(self, band=0, ax=None, bfac=0.0, cfac=0.0,
    #                **kwds):
    #     """
    #     Plot a projected data using matplotlib.imshow(...).
    #
    #     *Arguments*:
    #      - band = the band name (string), index (integer) or wavelength (float) to plot. Default is 0. If a tuple is passed then
    #               each band in the tuple (string or index) will be mapped to rgb.
    #      - bands = List defining the bands to include in the output image. Elements should be one of:
    #           - 'rgb' = rgb
    #           - 'xyz' = point position
    #           - 'klm' = normal vectors
    #           - numeric = index (int), wavelength (float) of an image band
    #           - tuple of length 3: wavelengths or band indices to map to rgb.
    #      - ax = an axis object to plot to. If none, plt.imshow( ... ) is used.
    #      - bfac = a brightness adjustment to apply to RGB mappings (-1 to 1)
    #      - cfac = a contrast adjustment to apply to RGB mappings (-1 to 1)
    #     *Keywords*:
    #      - keywords are passed to matplotlib.imshow( ... ).
    #     """
    #
    #     # plot hyImage data
    #     if not isinstance(band, str):
    #         kwds["mask"] = np.logical_not( np.isfinite(self.depth) ) #np.logical_not(self.valid)  # mask out pixels with no valid point mappings
    #         return self.image.quick_plot(band, ax, bfac, cfac, **kwds)  # render
    #     else:
    #         img = self.push_to_image(band)
    #         if (len(band) == 3):
    #             # do some normalizations
    #             mn = kwds.get("vmin", np.nanmin(img.data[img.data != 0]))
    #             mx = kwds.get("vmax", np.nanmax(img.data[img.data != 0]))
    #             img.data = (img.data - mn) / (mx - mn)
    #             if 'x' in band or 'y' in band or 'z' in band:  # coordinates need per-band mapping
    #                 for i in range(3):
    #                     img.data[..., i] = (img.data[..., i] - np.nanmin(img.data[..., i])) / (
    #                             np.nanmax(img.data[..., i]) - np.nanmin(img.data[..., i]))
    #             # plot it image
    #             return img.quick_plot((0, 1, 2), ax=ax, bfac=bfac, cfac=cfac, **kwds)
    #         else:
    #             return img.quick_plot(0, ax=ax, bfac=bfac, cfac=cfac, **kwds)
    #
    # @classmethod
    # def build_hypercloud(cls, scenes, bands, blending_mode='average', trim=True, vb=True, inplace=True, export_footprint = False):
    #     """
    #     Combine multiple HyScene objects into a hypercloud. Warning - this modifies the cloud associated with the HyScenes in-place.
    #     Returns processed cloud and (optional) footprint map indicating the number of involved scenes per pixel.
    #
    #     *Arguments*:
    #      - scenes = a list of scenes to combine. These scenes must all reference the same point cloud!
    #      - bands = either:
    #              (1) a tuple containing the (min,max) wavelength to map. If range is a tuple, -1 can be used to specify the
    #                  last band index.
    #              (2) a list of bands or boolean mask such that image.data[:,:,range] is exported to the hypercloud.
    #      - blending_mode = the mode used to blend points that can be assigned values from multiple values. Options are:
    #            - 'average' (default) = calculate the average of all pixels that map to the point
    #            - 'weighted' = calculate the average of all pixels, weighted by inverse footprint size.
    #            - 'gsd' = chose the pixel with the smallest gsd (i.e. the pixel with the closest camera)
    #            - 'obl' = chose the pixel that is least oblique (i.e. most perpendicular to the surface)
    #      - trim = True if the point cloud should be trimmed after doing project. Default is True.
    #      - vb = True if output should be written to the console. Default is True.
    #      - inplace = True if the reference cloud should be modified in place (to save RAM). Default is True.
    #      - export_footprint = True if footprint map indicating the number of involved scenes per pixel.
    #     """
    #
    #     if vb: print("Preparing data....", end='')
    #
    #     # get cloud
    #     cloud = scenes[0].cloud
    #     if not inplace:
    #         cloud = cloud.copy()
    #
    #     # remove everything except desired bands from scenes and calculate range of values
    #     images = []
    #     for i, s in enumerate(scenes):
    #         images.append(s.image.export_bands(bands))  # get bands of interest
    #         cloud.header.set_camera( s.camera, i ) # add camera to cloud header
    #
    #     # do scenes all have wavelength data?
    #     has_wav = np.array([i.has_wavelengths() for i in images]).all()
    #     wavelengths = None
    #     band_names = None
    #     if has_wav:  # match wavelengths between scenes
    #         wavelengths = []  # list of wavelengths
    #         indices = []  # list of corresponding band indices for each scene
    #         for w in images[0].get_wavelengths():
    #             idx = []
    #             for i, img in enumerate(images):
    #                 if not w in img.get_wavelengths():  # bail!
    #                     assert False, "Error - Scene %d does have data for data for band %d." % (i, w)
    #                 else:
    #                     idx.append(img.get_band_index(w))
    #             indices.append(idx)
    #             wavelengths.append(w)  # this wavelength is in all scenes
    #
    #         wavelengths = np.array(wavelengths)
    #         band_names = ["%.2f" % w for w in wavelengths]
    #         indices = np.array(indices)
    #
    #         assert wavelengths.shape[0] > 0, "Error - images do not have any matching bands."
    #     else:  # no - check all images have the same number of bands...
    #         assert (np.array([i.band_count() for i in images]) == images[0].band_count()).all(), \
    #             "Error - scenes with now wavelength information must have (exactly) the same band count."
    #         # no wavelengths, create band list and set band names
    #         band_list = np.array([i for i in range(scenes[0].image.band_count())])
    #         # noinspection PyUnusedLocal
    #         indices = np.array([band_list for s in scenes]).T  # bands to export
    #         # noinspection PyUnusedLocal
    #         band_names = ["%s" for b in band_list]
    #
    #     # handle bbl
    #     if np.array([img.header.has_bbl() for img in images]).all():
    #         bbl = np.full(wavelengths.shape[0], True)
    #         for i, idx in enumerate(indices):  # list of indices for each band
    #             bb = [images[n].header.get_bbl()[idx[n]] for n in range(len(images))]
    #             bbl[i] = np.array(bb).all()  # if any data is bad, this becomes a bad band
    #         cloud.header.set_bbl(bbl)  # store
    #
    #     if vb: print(" Done.")
    #
    #     # create data array
    #     data = np.zeros((cloud.point_count(), len(band_names)),
    #                     dtype=np.float32)  # point values will be accumulated here
    #     point_count = np.zeros(cloud.point_count(), dtype=np.float32)  # number of pixels used to calculate each point
    #
    #     # loop through points
    #     if vb:
    #         loop = tqdm(range(data.shape[0]), desc='Projecting data', leave=False)
    #     else:
    #         loop = range(data.shape[0])
    #     if 'average' in blending_mode.lower():
    #         for n in loop:
    #             for i, s in enumerate(scenes):
    #                 px, py = s.get_pixel(n)
    #                 if px is not None:  # point exists in scene
    #                     if np.isfinite(images[i].data[px, py, indices[:, i]]).all():  # is pixel finite?
    #                         point_count[n] += 1  # increment pixel count
    #                         data[n, :] += images[i].data[px, py, indices[:, i]]  # accumulate point value
    #         # divide by point count to calculate average
    #         data = data / point_count[:, None]
    #     elif 'gsd' in blending_mode.lower():
    #         for n in loop:
    #             best = np.inf
    #             for i, s in enumerate(scenes):
    #                 px, py = s.get_pixel(n)
    #                 if px is not None:  # point exists in scene
    #                     if np.isfinite(images[i].data[px, py, indices[:, i]]).all():  # is pixel finite?
    #                         point_count[n] += 1  # increment pixel count
    #                         if s.depth[px, py] < best:  # is this the best GSD so far?
    #                             best = s.depth[px, py]  # store best GSD
    #                             data[n, :] = images[i].data[px, py, indices[:, i]]  # set point value
    #     elif 'obl' in blending_mode.lower():
    #         for n in loop:
    #             best = np.inf
    #             for i, s in enumerate(scenes):
    #                 px, py = s.get_pixel(n)
    #                 if px is not None:  # point exists in scene
    #                     if np.isfinite(images[i].data[px, py, indices[:, i]]).all():  # is pixel finite?
    #                         point_count[n] += 1  # increment pixel count
    #                         if s.obliquity[px, py] < best:  # is this the best GSD so far?
    #                             best = s.obliquity[px, py]  # store best GSD
    #                             data[n, :] = images[i].data[px, py, indices[:, i]]  # set point value
    #     elif 'weight' in blending_mode.lower():
    #         for n in loop:
    #             for i, s in enumerate(scenes):
    #                 px, py = s.get_pixel(n)
    #                 if px is not None:  # point exists in scene
    #                     if np.isfinite(images[i].data[px, py, indices[:, i]]).all():  # is pixel finite?
    #                         w = 1 / s.depth[px, py] # calculate weight and increment to count
    #                         point_count[n] += w  #
    #                         data[n, :] += w * images[i].data[px, py, indices[:, i]]  # accumulate point value
    #         # divide by sum of weights to get weighted average
    #         data = data / point_count[:, None]
    #     else:
    #         assert False, "Error - unrecognised blending mode %s" % blending_mode
    #
    #     # add bands to point cloud
    #     cloud.set_bands(data, band_names=band_names, wavelengths=wavelengths)
    #
    #     # export footprint?
    #     if export_footprint:
    #         footprint = hylite.HyCloud(cloud.xyz, bands=point_count[:, None])
    #         if trim:
    #             footprint.filter_points(0, point_count == 0)  # remove zero points
    #
    #     # trim cloud?
    #     if trim:
    #         cloud.filter_points(0, point_count == 0)  # remove zero points
    #
    #     # return
    #     if export_footprint:
    #         return cloud, footprint
    #     else:
    #         return cloud



