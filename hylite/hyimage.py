"""
Store and manipulate hyperspectral image data.
"""

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import path
from roipoly import MultiRoi
import imageio
import cv2
import scipy as sp
from scipy import ndimage
import hylite
from hylite.hydata import HyData
from hylite.hylibrary import HyLibrary



class HyImage( HyData ):
    """
    A class for hyperspectral image data. These can be individual scenes or hyperspectral orthoimages.
    """

    def __init__(self, data, **kwds):
        """
        Args:
            data (ndarray): a numpy array such that data[x][y][band] gives each pixel value.
            **kwds:

                affine = an affine transform of the format returned by GDAL.GetGeoTransform().
                projection = string defining the project. Default is None.
                sensor = sensor name. Default is "unknown".
                header = path to associated header file. Default is None.

        """

        #call constructor for HyData
        super().__init__(data, **kwds)

        # special case - if dataset only has oneband, slice it so it still has
        # the format data[x,y,b].
        if not self.data is None:
            if len(self.data.shape) == 1:
                self.data = self.data[None, None, :] # single pixel image
            if len(self.data.shape) == 2:
                self.data = self.data[:, :, None] # single band iamge

        #load any additional project information (specific to images)
        self.set_projection(kwds.get("projection",None))
        self.affine = kwds.get("affine",[0,1,0,0,0,1])

        #special header formatting
        self.header['file type'] = 'ENVI Standard'

    def copy(self,data=True):
        """
        Make a deep copy of this image instance.

        Args:
            data (bool): True if a copy of the data should be made, otherwise only copy header.

        Returns:
            a new HyImage instance.
        """
        if not data:
            return HyImage(None, header=self.header.copy(), projection=self.projection, affine=self.affine)
        else:
            return HyImage( self.data.copy(), header=self.header.copy(), projection=self.projection, affine=self.affine)

    def T(self):
        """
        Return a transposed view of the data matrix (corresponding with the [y,x] indexing used by matplotlib, opencv etc.
        """
        return np.transpose(self.data, (1,0,2))

    def xdim(self):
        """
        Return number of pixels in x (first dimension of data array)
        """
        return self.data.shape[0]

    def ydim(self):
        """
        Return number of pixels in y (second dimension of data array)
        """
        return self.data.shape[1]

    def aspx(self):
        """
        Return the aspect ratio of this image (width/height).
        """
        return self.ydim() / self.xdim()

    def get_extent(self):
        """
        Returns the width and height of this image in world coordinates.

        Returns:
            tuple with (width, height).
        """
        return self.xdim * self.pixel_size[0], self.ydim * self.pixel_size[1]

    def set_projection(self,proj):
        """
        Set this project to an existing osgeo.osr.SpatialReference or GDAL georeference string.

        Args:
            proj (str, osgeo.osr.SpatialReference): the project to use as osgeo.osr.SpatialReference or GDAL georeference string.
        """
        if proj is None:
            self.projection = None
        else:
            try:
                from osgeo.osr import SpatialReference
            except:
                assert False, "Error - GDAL must be installed to work with spatial projections in hylite."
            if isinstance(proj, SpatialReference):
                self.projection = proj
            elif isinstance(proj, str):
                self.projection = SpatialReference(proj)
            else:
                print("Invalid project %s" % proj)
                raise

    def set_projection_EPSG(self,EPSG):
        """
        Sets this image project using an EPSG code.

        Args:
            EPSG (str): string EPSG code that can be passed to SpatialReference.SetFromUserInput(...).
        """

        try:
            from osgeo.osr import SpatialReference
        except:
            assert False, "Error - GDAL must be installed to work with spatial projections in hylite."

        self.projection = SpatialReference()
        self.projection.SetFromUserInput(EPSG)

    def get_projection_EPSG(self):
        """
        Gets a string describing this projections EPSG code (if it is an EPSG project).

        Returns:
            an EPSG code string of the format "EPSG:XXXX".
        """
        if self.projection is None:
            return None
        else:
            return "%s:%s" % (self.projection.GetAttrValue("AUTHORITY",0),self.projection.GetAttrValue("AUTHORITY",1))

    def pix_to_world(self, px, py, proj=None):
        """
        Take pixel coordinates and return world coordinates

        Args:
            px (int): the pixel x-coord.
            py (int): the pixel y-coord.
            proj (str, osr.SpatialReference): the coordinate system to use. Default (None) uses the same system as this image. Otherwise
                   an osr.SpatialReference can be passed (HyImage.project), or an EPSG string (e.g. get_projection_EPSG(...)).
        Returns:
            the world coordinates in the coordinate system defined by get_projection_EPSG(...).
        """

        try:
            from osgeo import osr
            import osgeo.gdal as gdal
            from osgeo import ogr
        except:
            assert False, "Error - GDAL must be installed to work with spatial projections in hylite."

        # parse project
        if proj is None:
            proj = self.projection
        elif isinstance(proj, str) or isinstance(proj, int):
            epsg = proj
            if isinstance(epsg, str):
                try:
                    epsg = int(str.split(':')[1])
                except:
                    assert False, "Error - %s is an invalid EPSG code." % proj
            proj = osr.SpatialReference()
            proj.ImportFromEPSG(epsg)

        # check we have all the required info
        assert isinstance(proj, osr.SpatialReference), "Error - invalid spatial reference %s" % proj
        assert (not self.affine is None) and (
            not self.projection is None), "Error - project information is undefined."

        #project to world coordinates in this images project/world coords
        x,y = gdal.ApplyGeoTransform(self.affine, px, py)

        #project to target coords (if different)
        if not proj.IsSameGeogCS(self.projection):
            P = ogr.Geometry(ogr.wkbPoint)
            if proj.EPSGTreatsAsNorthingEasting():
                P.AddPoint(x, y)
            else:
                P.AddPoint(y, x)
            P.AssignSpatialReference(self.projection)  # tell the point what coordinates it's in
            P.TransformTo(proj)  # reproject it to the out spatial reference
            x, y = P.GetX(), P.GetY()

            #do we need to transpose?
            if proj.EPSGTreatsAsLatLong():
                x,y=y,x #we want lon,lat not lat,lon
        return x, y

    def world_to_pix(self, x, y, proj = None):
        """
        Take world coordinates and return pixel coordinates

        Args:
            x (float): the world x-coord.
            y (float): the world y-coord.
            proj (str, osr.SpatialReference): the coordinate system of the input coordinates. Default (None) uses the same system as this image. Otherwise
                   an osr.SpatialReference can be passed (HyImage.project), or an EPSG string (e.g. get_projection_EPSG(...)).

        Returns:
            the pixel coordinates based on the affine transform stored in self.affine.
        """

        try:
            from osgeo import osr
            import osgeo.gdal as gdal
            from osgeo import ogr
        except:
            assert False, "Error - GDAL must be installed to work with spatial projections in hylite."

        # parse project
        if proj is None:
            proj = self.projection
        elif isinstance(proj, str) or isinstance(proj, int):
            epsg = proj
            if isinstance(epsg, str):
                try:
                    epsg = int(str.split(':')[1])
                except:
                    assert False, "Error - %s is an invalid EPSG code." % proj
            proj = osr.SpatialReference()
            proj.ImportFromEPSG(epsg)


        # check we have all the required info
        assert isinstance(proj, osr.SpatialReference), "Error - invalid spatial reference %s" % proj
        assert (not self.affine is None) and (not self.projection is None), "Error - project information is undefined."

        # project to this images CS (if different)
        if not proj.IsSameGeogCS(self.projection):
            P = ogr.Geometry(ogr.wkbPoint)
            if proj.EPSGTreatsAsNorthingEasting():
                P.AddPoint(x, y)
            else:
                P.AddPoint(y, x)
            P.AssignSpatialReference(proj)  # tell the point what coordinates it's in
            P.AddPoint(x, y)
            P.TransformTo(self.projection)  # reproject it to the out spatial reference
            x, y = P.GetX(), P.GetY()
            if self.projection.EPSGTreatsAsLatLong(): # do we need to transpose?
                x, y = y, x  # we want lon,lat not lat,lon

        inv = gdal.InvGeoTransform(self.affine)
        assert not inv is None, "Error - could not invert affine transform?"

        #apply
        return gdal.ApplyGeoTransform(inv, x, y)

    def flip(self, axis='x'):
        """
        Flip the image on the x or y axis.

        Args:
            axis (str): 'x' or 'y' or both 'xy'.
        """

        if 'x' in axis.lower():
            self.data = np.flip(self.data,axis=0)
        if 'y' in axis.lower():
            self.data = np.flip(self.data,axis=1)

    def rot90(self):
        """
        Rotate this image by 90 degrees by transposing the underlying data array. Combine with flip('x') or flip('y')
        to achieve positive/negative rotations.
        """
        self.data = np.transpose( self.data, (1,0,2) )
        self.push_to_header()

    #####################################
    ##IMAGE FILTERING
    #####################################
    def fill_holes(self):
        """
        Replaces nan pixel with an average of their neighbours, thus removing 1-pixel large holes from an image. Note that
        for performance reasons this assumes that holes line up across bands. Note that this is not vectorized so very slow...
        """

        # perform greyscale dilation
        dilate = self.data.copy()
        mask = np.logical_not(np.isfinite(dilate))
        dilate[mask] = 0
        for b in range(self.band_count()):
            dilate[:, :, b] = sp.ndimage.grey_dilation(dilate[:, :, b], size=(3, 3))

        # map back to holes in dataset
        self.data[mask] = dilate[mask]
        #self.data[self.data == 0] = np.nan  # replace remaining 0's with nans

    def blur(self, n=3):
        """
        Applies a gaussian kernel of size n to the image using OpenCV.

        Args:
            n (int): the dimensions of the gaussian kernel to convolve. Default is 3. Increase for more blurry results.
        """

        nanmask = np.isnan(self.data)
        assert isinstance(n, int) and n >= 3, "Error - invalid kernel. N must be an integer > 3. "
        kernel = np.ones((n, n), np.float32) / (n ** 2)
        self.data = cv2.filter2D(self.data, -1, kernel)
        self.data[nanmask] = np.nan  # remove mask

    def erode(self, size=3, iterations=1):
        """
        Apply an erode filter to this image to expand background (nan) pixels. Refer to open-cv's erode
        function for more details.

        Args:
            size (int): the size of the erode filter. Default is a 3x3 kernel.
            iterations (int): the number of erode iterations. Default is 1.
        """

        # erode
        kernel = np.ones((size, size), np.uint8)
        if self.is_float():
            mask = np.isfinite(self.data).any(axis=-1)
            mask = cv2.erode(mask.astype(np.uint8), kernel, iterations=iterations)
            self.data[mask == 0, :] = np.nan
        else:
            mask = (self.data != 0).any( axis=-1 )
            mask = cv2.erode(mask.astype(np.uint8), kernel, iterations=iterations)
            self.data[mask == 0, :] = 0

    def resize(self, newdims, interpolation=cv2.INTER_LINEAR):
        """
        Resize this image with opencv.

        Args:
            newdims (tuple): the new image dimensions.
            interpolation (int): opencv interpolation method. Default is cv2.INTER_LINEAR.
        """

        self.data = cv2.resize(self.data, (newdims[1],newdims[0]), interpolation=interpolation)

    def despeckle(self, size=5):
        """
        Despeckle each band of this image (independently) using a median filter.

        Args:
            size (int): the size of the median filter kernel. Default is 5. Must be an odd number.
        """

        assert (size % 2) == 1, "Error - size must be an odd integer"

        if self.is_float():
            self.data = cv2.medianBlur( self.data.astype(np.float32), size )
        else:
            self.data = cv2.medianBlur( self.data, size )

    #####################################
    ##FEATURES AND FEATURE MATCHING
    ######################################
    def get_keypoints(self, band, eq=False, mask=True, method='sift', cfac=0.0,bfac=0.0, **kwds):
        """
        Get feature descriptors from the specified band.

        Args:
            band (int,float,str,tuple): the band index (int) or wavelength (float) to extract features from. Alternatively, a tuple can be passed
                    containing a range of bands (min : max) to average before feature matching.
            eq (bool): True if the image should be histogram equalized first. Default is False.
            mask (bool): True if 0 value pixels should be masked. Default is True.
            method (str): the feature detector to use. Options are 'SIFT' and 'ORB' (faster but less accurate). Default is 'SIFT'.
            cfac (float): contrast adjustment to apply to hyperspectral bands before matching. Default is 0.0.
            bfac (float): brightness adjustment to apply to hyperspectral bands before matching. Default is 0.0.
            **kwds: keyword arguments are passed to the opencv feature detector. For SIFT these are:

                - contrastThreshold: default is 0.01.
                - edgeThreshold: default is 10.
                - sigma: default is 1.0

                For ORB these are:

                - nfeatures = the number of features to detect. Default is 5000.

            Returns:
                Tuple containing

                    - k (ndarray): the keypoints detected
                    - d (ndarray): corresponding feature descriptors
         """

        # get image
        if isinstance(band, int) or isinstance(band, float): #single band
            image = self.data[:, :, self.get_band_index(band)]
        elif isinstance(band,tuple): #range of bands (averaged)
            idx0 = self.get_band_index(band[0])
            idx1 = self.get_band_index(band[1])

            #deal with out of range errors
            if idx0 is None:
                idx0 = 0
            if idx1 is None:
                idx1 = self.band_count()

            #average bands
            image = np.nanmean(self.data[:,:,idx0:idx1],axis=2)
        else:
            assert False, "Error, unrecognised band %s" % band

        #normalise image to range 0 - 1
        image -= np.nanmin(image)
        image = image / np.nanmax(image)

        #apply brightness/contrast adjustment
        image = (1.0+cfac)*image + bfac
        image[image > 1.0] = 1.0
        image[image < 0.0] = 0.0

        #convert image to uint8 for opencv
        image = np.uint8(255 * image)
        if eq:
            image = cv2.equalizeHist(image)

        if mask:
            mask = np.zeros(image.shape, dtype=np.uint8)
            mask[image != 0] = 255  # include only non-zero pixels
        else:
            mask = None

        if 'sift' in method.lower():  # SIFT

            # setup default keywords
            kwds["contrastThreshold"] = kwds.get("contrastThreshold", 0.01)
            kwds["edgeThreshold"] = kwds.get("edgeThreshold", 10)
            kwds["sigma"] = kwds.get("sigma", 1.0)

            # make feature detector
            #alg = cv2.xfeatures2d.SIFT_create(**kwds)
            alg = cv2.SIFT_create()
        elif 'orb' in method.lower():  # orb
            kwds['nfeatures'] = kwds.get('nfeatures', 5000)
            alg = cv2.ORB_create(scoreType=cv2.ORB_FAST_SCORE, **kwds)
        else:
            assert False, "Error - %s is not a recognised feature detector." % method

        # detect keypoints
        kp = alg.detect(image, mask)

        # extract and return feature vectors
        return alg.compute(image, kp)

    @classmethod
    def match_keypoints(cls, kp1, kp2, d1, d2, method='SIFT', dist=0.7, tree = 5, check = 100, min_count=5):
        """
        Compares keypoint feature vectors from two images and returns matching pairs.

        Args:
            kp1 (ndarray): keypoints from the first image
            kp2 (ndarray): keypoints from the second image
            d1 (ndarray): descriptors for the keypoints from the first image
            d2 (ndarray): descriptors for the keypoints from the second image
            method (str): the method used to calculate the feature descriptors. Should be 'sift' or 'orb'. Default is 'sift'.
            dist (float): minimum match distance (0 to 1), default is 0.7
            tree (int): not sure what this does? Default is 5. See open-cv docs.
            check (int): ditto. Default is 100.
            min_count (int): the minimum number of matches to consider a valid matching operation. If fewer matches are found,
                       then the function returns None, None. Default is 5.
        """

        if 'sift' in method.lower():
            algorithm = cv2.NORM_INF
        elif 'orb' in method.lower():
            algorithm = cv2.NORM_HAMMING
        else:
            assert False, "Error - unknown matching algorithm %s" % method

        #calculate flann matches
        index_params = dict(algorithm=algorithm, trees=tree)
        search_params = dict(checks=check)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(d1, d2, k=2)

        # store all the good matches as per Lowe's ratio test.
        good = []
        for m, n in matches:
            if m.distance < dist * n.distance:
                good.append(m)

        if len(good) < min_count:
            return None, None
        else:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            return src_pts, dst_pts

    ############################
    ## Visualisation methods
    ############################
    def quick_plot(self, band=0, ax=None, bfac=0.0, cfac=0.0, samples=False, tscale=False, rot=False, flipX=False, flipY=False,
                   **kwds):
        """
        Plot a band using matplotlib.imshow(...).

        Args:
            band (str,int,float,tuple): the band name (string), index (integer) or wavelength (float) to plot. Default is 0. If a tuple is passed then
                  each band in the tuple (string or index) will be mapped to rgb. Bands with negative wavelengths or indices will be inverted before plotting.
            ax: an axis object to plot to. If none, plt.imshow( ... ) is used.
            bfac (float): a brightness adjustment to apply to RGB mappings (-1 to 1)
            cfac (float): a contrast adjustment to apply to RGB mappings (-1 to 1)
            samples (bool): True if sample points (defined in the header file) should be plotted. Default is False. Otherwise, a list of
                     [ (x,y), ... ] points can be passed.
            tscale (bool): True if each band (for ternary images) should be scaled independently. Default is False.
                    When using scaling, vmin and vmax can be used to set the clipping percentiles (integers) or
                    (constant) values (float).
            rot (bool): if True, the x and y axis will be flipped (90 degree rotation) before plotting. Default is False.
            flipX (bool): if True, the x axis will be flipped before plotting (after applying rotations).
            flipY (bool): if True, the y axis will be flippe before plotting (after applying rotations).
            **kwds: keywords are passed to matplotlib.imshow( ... ), except for the following:

                 - mask = a 2D boolean mask containing true if pixels should be drawn and false otherwise.
                 - path = a file path to save the image too (at matching resolution; use fig.savefig(..) if you want to save the figure).
                 - ticks = True if x- and y- ticks should be plotted. Default is False.
                 - ps, pc = the size and color of sample points to plot. Can be constant or list.
                 - figsize = a figsize for the figure to create (if ax is None).

        Returns:
            Tuple containing

            - fig: matplotlib figure object
            - ax:  matplotlib axes object. If a colorbar is created, (band is an integer or a float), then this will be stored in ax.cbar.
        """

        #create new axes?
        if ax is None:
            fig, ax = plt.subplots(figsize=kwds.pop('figsize', (18,18*self.ydim()/self.xdim()) ))

        # deal with ticks
        if not kwds.pop('ticks', False ):
            ax.set_xticks([])
            ax.set_yticks([])

        #map individual band using colourmap
        if isinstance(band, str) or isinstance(band, int) or isinstance(band, float):
            #get band
            if isinstance(band, str):
                data = self.data[:, :, self.get_band_index(band)]
            else:
                data = self.data[:, :, self.get_band_index(np.abs(band))]
            if not isinstance(band, str) and band < 0:
                data = np.nanmax(data) - data # flip

            # convert integer vmin and vmax values to percentiles
            if 'vmin' in kwds:
                if isinstance(kwds['vmin'], int):
                    kwds['vmin'] = np.nanpercentile( data, kwds['vmin'] )
            if 'vmax' in kwds:
                if isinstance(kwds['vmax'], int):
                    kwds['vmax'] = np.nanpercentile( data, kwds['vmax'] )

            #mask nans (and apply custom mask)
            mask = np.isnan(data)
            if not np.isnan(self.header.get_data_ignore_value()):
                mask = mask + data == self.header.get_data_ignore_value()
            if 'mask' in kwds:
                mask = mask + kwds.get('mask')
                del kwds['mask']
            data = np.ma.array(data, mask = mask > 0 )

            # apply rotations and flipping
            if rot:
                data = data.T
            if flipX:
                data = data[::-1, :]
            if flipY:
                data = data[:, ::-1]

            # save?
            if 'path' in kwds:
                path = kwds.pop('path')
                from matplotlib.pyplot import imsave
                if not os.path.exists(os.path.dirname(path)):
                    os.makedirs(os.path.dirname(path)) # ensure output directory exists
                imsave(path, data.T, **kwds)  # save the image

            ax.cbar = ax.imshow(data.T, interpolation=kwds.pop('interpolation', 'none'), **kwds) # change default interpolation to None

        #map 3 bands to RGB
        elif isinstance(band, tuple) or isinstance(band, list):
            #get band indices and range
            rgb = []
            for b in band:
                if isinstance(b, str):
                    rgb.append(self.get_band_index(b))
                else:
                    rgb.append(self.get_band_index(np.abs(b)))

            #slice image (as copy) and map to 0 - 1
            img = np.array(self.data[:, :, rgb]).copy()
            if np.isnan(img).all():
                print("Warning - image contains no data.")
                return ax.get_figure(), ax

            # invert if needed
            for i,b in enumerate(band):
                if not isinstance(b, str) and (b < 0):
                    img[..., i] = np.nanmax(img[..., i]) - img[..., i]

            # do scaling
            if tscale: # scale bands independently
                for b in range(3):
                    mn = kwds.get("vmin", float(np.nanmin(img)))
                    mx = kwds.get("vmax", float(np.nanmax(img)))
                    if isinstance (mn, int):
                        assert mn >= 0 and mn <= 100, "Error - integer vmin values must be a percentile."
                        mn = float(np.nanpercentile(img[...,b], mn ))
                    if isinstance (mx, int):
                        assert mx >= 0 and mx <= 100, "Error - integer vmax values must be a percentile."
                        mx = float(np.nanpercentile(img[...,b], mx ))
                    img[...,b] = (img[..., b] - mn) / (mx - mn)
            else: # scale bands together
                mn = kwds.get("vmin", float(np.nanmin(img)))
                mx = kwds.get("vmax", float(np.nanmax(img)))
                if isinstance(mn, int):
                    assert mn >= 0 and mn <= 100, "Error - integer vmin values must be a percentile."
                    mn = float(np.nanpercentile(img, mn))
                if isinstance(mx, int):
                    assert mx >= 0 and mx <= 100, "Error - integer vmax values must be a percentile."
                    mx = float(np.nanpercentile(img, mx))
                img = (img - mn) / (mx - mn)

            #apply brightness/contrast mapping
            img = np.clip((1.0 + cfac) * img + bfac, 0, 1.0 )

            #apply masking so background is white
            img[np.logical_not( np.isfinite( img ) )] = 1.0
            if 'mask' in kwds:
                img[kwds.pop("mask"),:] = 1.0

            # apply rotations and flipping
            if rot:
                img = np.transpose( img, (1,0,2) )
            if flipX:
                img = img[::-1, :, :]
            if flipY:
                img = img[:, ::-1, :]

            # save?
            if 'path' in kwds:
                path = kwds.pop('path')
                from matplotlib.pyplot import imsave
                if not os.path.exists(os.path.dirname(path)):
                    os.makedirs(os.path.dirname(path)) # ensure output directory exists
                imsave(path, np.transpose( np.clip( img*255, 0, 255).astype(np.uint8), (1, 0, 2)))  # save the image

            # plot samples?
            ps = kwds.pop('ps', 5)
            pc = kwds.pop('pc', 'r')
            if samples:
                if isinstance(samples, list) or isinstance(samples, np.ndarray):
                    ax.scatter([s[0] for s in samples], [s[1] for s in samples], s=ps, c=pc)
                else:
                    for n in self.header.get_class_names():
                        points = np.array(self.header.get_sample_points(n))
                        ax.scatter(points[:, 0], points[:, 1], s=ps, c=pc)

            #plot
            ax.imshow(np.transpose(img, (1,0,2)), interpolation=kwds.pop('interpolation', 'none'), **kwds)
            ax.cbar = None  # no colorbar

        return ax.get_figure(), ax

    def createGIF(self, path, bands=None, figsize=(10,10), fps=10, **kwds):
        """
        Create and save an animated gif that loops through the bands of the image.

        Args:
            path (str): the path to save the .gif
            bands (tuple): Tuple containing the range of band indices to draw. Default is the whole range.
            figsize (tuple): the size of the image to draw. Default is (10,10).
            fps (int): the framerate (frames per second) of the gif. Default is 10.
            **kwds: keywords are passed directly to matplotlib.imshow. Use this to specify cmap etc.
        """

        frames = []
        if bands is None:
            bands = (0,self.band_count())
        else:
            assert 0 < bands[0] < self.band_count(), "Error - invalid range."
            assert 0 < bands[1] < self.band_count(), "Error - invalid range."
            assert bands[1] > bands[0], "Error - invalid range."

        #plot frames
        for i in range(bands[0],bands[1]):
            fig, ax = plt.subplots(figsize=figsize)
            ax.imshow(self.data[:, :, i], **kwds)
            fig.canvas.draw()
            frames.append(np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8'))
            frames[-1] = np.reshape(frames[-1], (fig.canvas.get_width_height()[1], fig.canvas.get_width_height()[0], 3))
            plt.close(fig)

        #save gif
        imageio.mimsave( os.path.splitext(path)[0] + ".gif", frames, fps=fps)

    ## masking
    def mask(self, mask=None, flag=np.nan, invert=False, crop=False, bands=None):
        """
         Apply a mask to an image, flagging masked pixels with the specified value. Note that this applies the mask to the
         image in-situ.

         Args:
            flag (float): the value to use for masked pixels. Default is np.nan
            mask (ndarray): a numpy array defining the mask polygon of the format [[x1,y1],[x2,y2],...]. If None is passed then
                    pickPolygon( ... ) is used to interactively define a polygon. If a file path is passed then the polygon
                    will be loaded using np.load( ... ). Alternatively if mask.shape == image.shape[0,1] then it is treated as a
                    binary image mask (must be boolean) and True values will be masked across all bands. Default is None.
            invert (bool): if True, pixels within the polygon will be masked. If False, pixels outside the polygon are masked. Default is False.
            crop (bool): True if rows/columns containing only zeros should be removed. Default is False.
            bands (tuple): the bands of the image to plot if no mask is specified. If None, the middle band is used.

         Returns:
            Tuple containing

            - mask (ndarray): a boolean array with True where pixels are masked and False elsewhere.
            - poly (ndarray): the mask polygon array in the format described above. Useful if the polygon was interactively defined.
         """

        if mask is None:  # pick mask interactively
            if bands is None:
                bands = int(self.band_count() / 2)

            regions = self.pickPolygons(region_names=["mask"], bands=bands)

            # the user bailed without picking a mask?
            if len(regions) == 0:
                print("Warning - no mask picked/applied.")
                return

            # extract polygon mask
            mask = regions[0]

        # convert polygon mask to binary mask
        if mask.shape[1] == 2:

            # build meshgrid with pixel coords
            xx, yy = np.meshgrid(np.arange(self.xdim()), np.arange(self.ydim()))
            xx = xx.flatten()
            yy = yy.flatten()
            points = np.vstack([xx, yy]).T  # coordinates of each pixel

            # calculate per-pixel mask
            mask = path.Path(mask).contains_points(points)
            mask = mask.reshape((self.ydim(), self.xdim())).T

            # flip as we want to mask (==True) outside points (unless invert is true)
            if not invert:
                mask = np.logical_not(mask)

        # apply binary image mask
        assert mask.shape[0] == self.data.shape[0] and mask.shape[1] == self.data.shape[1], \
            "Error - mask shape %s does not match image shape %s" % (mask.shape, self.data.shape)
        for b in range(self.band_count()):
            self.data[:, :, b][mask] = flag

        # crop image
        if crop:
            # calculate non-masked pixels
            valid = np.logical_not(mask)

            # integrate along axes
            xdata = np.sum(valid, axis=1) > 0.0
            ydata = np.sum(valid, axis=0) > 0.0

            # calculate domain containing valid pixels
            xmin = np.argmax(xdata)
            xmax = xdata.shape[0] - np.argmax(xdata[::-1])
            ymin = np.argmax(ydata)
            ymax = ydata.shape[0] - np.argmax(ydata[::-1])

            # crop
            self.data = self.data[xmin:xmax, ymin:ymax, :]

        return mask

    def crop_to_data(self):
        """
        Remove padding of nan or zero pixels from image. Note that this is performed in place.
        """

        valid = np.isfinite(self.data).any(axis=-1) & (self.data != 0).any(axis=-1)
        ymin, ymax = np.percentile(np.argwhere(np.sum(valid, axis=0) != 0), (0, 100))
        xmin, xmax = np.percentile(np.argwhere(np.sum(valid, axis=1) != 0), (0, 100))
        self.data = self.data[int(xmin):int(xmax), int(ymin):int(ymax), :]  # do clipping

    ##################################################
    ## Interactive tools for picking regions/pixels
    ##################################################
    def pickPolygons(self, region_names, bands=0):
        """
        Creates a matplotlib gui for selecting polygon regions in an image.

        Args:
            names (list, str): a list containing the names of the regions to pick. If a string is passed only one name is used.
            bands (tuple): the bands of the image to plot.
        """

        if isinstance(region_names, str):
            region_names = [region_names]

        assert isinstance(region_names, list), "Error - names must be a list or a string."

        # set matplotlib backend
        backend = matplotlib.get_backend()
        matplotlib.use('Qt5Agg')  # need this backend for ROIPoly to work

        # plot image and extract roi's
        fig, ax = self.quick_plot(bands)
        roi = MultiRoi(roi_names=region_names)
        plt.close(fig)  # close figure

        # extract regions
        regions = []
        for name, r in roi.rois.items():
            # store region
            x = r.x
            y = r.y
            regions.append(np.vstack([x, y]).T)

        # restore matplotlib backend (if possible)
        try:
            matplotlib.use(backend)
        except:
            print("Warning: could not reset matplotlib backend. Plots will remain interactive...")
            pass

        return regions

    def pickPoints(self, n=-1, bands=hylite.RGB, integer=True, title="Pick Points", **kwds):
        """
        Creates a matplotlib gui for picking pixels from an image.

        Args:
            n (int): the number of pixels to pick, or -1 if the user can select as many as they wish. Default is -1.
            bands (tuple): the bands of the image to plot. Default is HyImage.RGB
            integer (bool): True if points coordinates should be cast to integers (for use as indices). Default is True.
            title (str): The title of the point picking window.
            **kwds: Keywords are passed to HyImage.quick_plot( ... ).

        Returns:
            A list containing the picked point coordinates [ (x1,y1), (x2,y2), ... ].
        """

        # set matplotlib backend
        backend = matplotlib.get_backend()
        matplotlib.use('Qt5Agg')  # need this backend for ROIPoly to work

        # create figure
        fig, ax = self.quick_plot( bands, **kwds )
        ax.set_title(title)

        # get points
        points = fig.ginput( n )

        if integer:
            points = [ (int(p[0]), int(p[1])) for p in points ]

        # restore matplotlib backend (if possible)
        try:
            matplotlib.use(backend)
        except:
            print("Warning: could not reset matplotlib backend. Plots will remain interactive...")
            pass

        return points

    def pickSamples(self, names=None, store=True, **kwds):
        """
        Pick sample probe points and store these in the image header file.

        Args:
            names (str, list): the name of the sample to pick, or a list of names to pick multiple.
            store (bool): True if sample should be stored in the image header file (for later access). Default is True.
            **kwds: Keywords are passed to HyImage.quick_plot( ... )

        Returns:
            a list containing a list of points for each sample.
        """

        if isinstance(names, str):
            names = [names]

        # pick points
        points = []
        for s in names:
            pnts = self.pickPoints(title="%s" % s, **kwds)
            if store:
                self.header['sample %s' % s] = pnts # store in header
            points.append(pnts)
        # add class to header file
        if store:
            cls_names = self.header.get_class_names()
            if cls_names is None:
                cls_names = []
            self.header['class names'] = cls_names + names

        return points





