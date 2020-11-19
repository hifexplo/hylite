from tqdm import tqdm
import matplotlib.pyplot as plt

import hylite
from hylite import HyImage
from hylite.correct.topography import sph2cart
from hylite.correct import estimate_sun_vec, estimate_incidence, estimate_ambient, correct_topo, ELC, norm_eq, hist_eq
from hylite.project.basic import *

class HyScene(object):
    """
    A class that combines a hyperspectral image with known camera pose and a 3D point cloud into a 2.5D scene. The scene
    can then be topographically corrected, calibrated or aligned with other scenes and/or projected to make a hypercloud.
    """

    def __init__(self, image, cloud, camera, s=1, occ_tol=10, vb=True):
        """
        Create a HyScene instance using the defined image, camera and point cloud data.

        *Arguments*:
         - image = a HyImage instance map onto the point cloud.
         - cloud = a HyCloud instance containing point cloud data.
         - camera = hylite.project.Camera object containing camera properties to use for the mapping.
         - s = the point size to use when rendering points and building the image <-> cloud mappings. Default is 1 (map to
               a single pixel)
         - occ_tol = the distance between a point and the z-buffer before it becomes occluded. Default is 10. Set to 0 to
                    disable occlusion.
         - vb = true if a progress bar should be displayed (as this function can be slooooow). Default is true.
        """

        if image is None and cloud is None and camera is None:
            return  # dummy constructor used for loading data from file

        if s > 0: s -= 1  # reduce s by one due to how numpy does indexing.

        # store reference to associated datasets and properties
        self.image = image
        self.cloud = cloud
        self.camera = camera
        self.s = 1
        self.occ_tol = occ_tol

        assert camera.dims[0] == image.xdim() and camera.dims[
            1] == image.ydim(), "Error - image and camera dimensions do not match."

        # project image onto cloud and build mappings for projecting pixels onto the cloud and vice-versa.
        self.image_to_cloud = {}  # map image pixels to point IDs
        self.cloud_to_image = {}  # map point IDs to image pixels
        self.point_depth = {}  # map point IDs to depths
        self.depth = np.full((image.data.shape[0], image.data.shape[1]),
                             np.inf)  # also store depth buffer as this is handy
        self.normals = np.full((image.data.shape[0], image.data.shape[1], 3), np.nan)  # and point normals
        self.obliquity = np.full((image.data.shape[0], image.data.shape[1]), np.nan)  # and incidence rays
        # project point cloud using camera
        if 'persp' in camera.proj.lower():
            pp, vis = proj_persp(cloud.xyz, C=camera.pos, a=camera.ori,
                                 fov=camera.fov, dims=camera.dims)
        elif 'pano' in camera.proj.lower():
            pp, vis = proj_pano(cloud.xyz, C=camera.pos, a=camera.ori,
                                fov=camera.fov, dims=camera.dims, step=camera.step, normals=cloud.normals)
        else:
            assert False, "Error - unknown camera_type. Should be 'perspective' or 'panorama'."

        assert vis.any(), "Error, project contains no visible points."

        # cull invisible points (but remember ids)
        ids = np.arange(0, cloud.point_count())[vis]
        pp = pp[vis, :]

        # loop through points, rasterise and build mapping
        if vb:
            pp = tqdm(pp, leave=False)  # initialise progress bar
            pp.set_description("Mapping points")
        for i, p in enumerate(pp):
            x = int(p[0])
            y = int(p[1])
            z = p[2]

            # is there image data in this pixel? If not... ignore it.
            if np.isnan(self.image.data[x, y, :]).all():
                continue  # nope...
            if 'data ignore value' in self.image.header:
                if (self.image.data[x, y, :] == self.image.header.get('data ignore value')).all():
                    continue

            # success - link point with this pixel :)
            self.cloud_to_image[ids[i]] = (x, y)
            self.point_depth[ids[i]] = z

            # link pixel to point
            if (x, y) in self.image_to_cloud:
                self.image_to_cloud[(x, y)].append(ids[i])  # pixel already contains some points
            else:  # no points in pixel yet
                self.image_to_cloud[(x, y)] = [ids[i]]

            # update depth buffers
            if z < self.depth[x, y]:  # in front of depth buffer?
                self.depth[x - s:x + s + 1, y - s:y + s + 1] = z

        if vb: pp.clear(nolock=False)  # remove progress bar

        # remove occluded points
        if occ_tol > 0:
            to_del = []
            if vb:
                loop = tqdm(self.cloud_to_image.items(), leave=False)  # initialise progress bar
                loop.set_description("Filtering occluded points")
            else:
                loop = self.cloud_to_image.items()
            for id, xy in loop:
                x, y = xy
                if abs(self.point_depth[id] - self.depth[x, y]) > occ_tol:  # point is occluded
                    # remove from maps
                    self.image_to_cloud[(x, y)].remove(id)
                    if len(self.image_to_cloud[(x, y)]) == 0:
                        del self.image_to_cloud[(x, y)]
                    to_del.append(id)
            for idx in to_del:
                del self.cloud_to_image[idx]

        # compute normals and obliquity
        self.valid = np.full((self.image.xdim(), self.image.ydim()), False,
                             dtype=np.bool)  # also store pixels with valid mappings
        if cloud.has_normals():
            if vb:
                loop = tqdm(range(self.normals.shape[0]), leave=False)
                loop.set_description("Averaging normal vectors...")
            else:
                loop = range(self.normals.shape[0])
            for _x in loop:
                for _y in range(self.normals.shape[1]):
                    if (_x, _y) in self.image_to_cloud:

                        # valid pixel
                        self.valid[_x][_y] = True

                        # calculate point normal
                        N = np.mean(self.cloud.normals[self.image_to_cloud[(_x, _y)], :], axis=0)
                        self.normals[_x, _y, :] = N

                        # calculate obliquity
                        if self.camera.is_perspective():
                            O = np.arccos(np.dot(pix_to_ray_persp(_x, _y, self.camera.fov, self.camera.dims), N))
                        else:
                            O = np.arccos(
                                np.dot(pix_to_ray_pano(_x, _y, self.camera.fov, self.camera.step, self.camera.dims), N))
                        self.obliquity[_x, _y] = np.abs(np.rad2deg(O))

            # normalise normals
            self.normals = self.normals[:, :, :] / np.linalg.norm(self.normals, axis=2)[:, :, None]

    def visible_point_count(self):
        """
        How many points are visible in this scene (i.e. how many points are mapped to pixels?)
        """

        return len(self.cloud_to_image.keys())

    def valid_pixel_count(self):
        """
        How many pixels of the original image have points in them (i.e. have a valid mapping?)
        """

        return len(self.image_to_cloud.keys())

    def valid_pixels(self):
        """
        Return a 2D numpy array of the same size as image scored with true if the pixel contains points.
        """
        return self.valid

    def get_point_index(self, px, py):
        """
        Get the index of the closest point in the specified pixel

        *Arguments*;
         - px = the x index of the pixel.
         - py = the y index of the pixel.
        *Returns*:
         - the point index or None if no points are in the specified pixel.
        """

        if (px, py) in self.image_to_cloud:
            return self.image_to_cloud[(px, py)][0]
        else:
            return None

    def get_point_indices(self, px, py):
        """
        Get the indices and depths of all points in the specified pixel.

        *Arguments*;
         - px = the x index of the pixel.
         - py = the y index of the pixel.
        *Returns*:
         - a list of (index, depth) tuples for each point in this pixel, or [ ] if no points are present.
        """

        if (px, py) in self.image_to_cloud:
            return self.image_to_cloud[(px, py)]
        else:
            return []

    def get_pixel_depth(self, px, py):
        """
        Get depth to the nearest pixel at the specifed coordinates.
        """

        return self.get_depth()[px, py]

    def get_point_depth(self, pointID):
        """
        Get the depth to the specified point. Returns np.nan if the point is not visible.
        """

        if pointID in self.point_depth:
            return self.point_depth[pointID]
        else:
            return np.nan

    def get_pixel_normal(self, px, py):
        """
        Get the average normal vector of all points in the specified pixel.
        """

        return self.get_normals()[px, py]

    def get_point_normal(self, index):
        """
        Get the normal vector of the specified point.
        """

        return self.cloud.normals[index]

    def get_pixel(self, idx):
        """
        Get the pixel coordinates and depth of the specified point.

        *Arguments*:
         - idx = the point index
        *Returns*:
         - px, py. Or (None, None) if the point is not visible in the image
        """

        if idx in self.cloud_to_image:
            return self.cloud_to_image[idx]
        else:
            return None, None

    def get_normals(self):
        """
        Get per-pixel normals array.
        """

        return self.normals

    def get_depth(self):
        """
        Get per-pixel depth array.
        """

        return self.depth

    def get_GSD(self):
        """
        Get per-pixel ground sampling distance (pixel size).

        *Return*:
         - gsd_x = numpy array containing the sampling distance in x
         - gsd_y = numpy array containing the sampling distance in y
        """

        # calculate pixel pitch in degrees
        pitch_y = np.deg2rad(self.camera.dims[1] / self.camera.fov)
        if self.camera.is_panoramic():
            pitch_x = np.deg2rad(self.camera.step)
        else:
            pitch_x = pitch_y  # assume square pixels

        # calculate GSD
        gsdx = 2 * self.depth * np.tan(pitch_x / 2)
        gsdy = 2 * self.depth * np.tan(pitch_y / 2)
        return gsdx, gsdy

    def get_obliquity(self):
        """
        Get array obliquity angles (degrees) between camera look direction and surface normals. Note that this angle
        will be 0 if the look direction is perpendicular to the surface.
        """

        return self.obliquity

    def get_slope(self):
        """
        Get array of slope angles for each pixel (based on the surface normal vectors).
        """

        return np.rad2deg(np.arccos(np.abs(self.normals[..., 2])))

    def intersect(self, scene2):
        """
        Get point indices that exist (are visible in) this scene and another.

        *Arguments*:
         - scene2 = a hyScene instance that references the same cloud but with a different image/viewpoint.
        *Returns*:
         - indices = a list of point indices that are visible in both scenes.
        """

        assert self.cloud == scene2.cloud, "Error - scene2 must reference the same point cloud as this one."
        keys_a = set(self.cloud_to_image.keys())
        keys_b = set(scene2.cloud_to_image.keys())
        return list(keys_a & keys_b)

    def union(self, scenes):
        """
        Returns point that are visible in either this scene or scene2 (or both).

        *Arguments*:
         - scenes = the scene (or list of scenes) to compare with
        *Returns*:
         - indices = a list of point indices that are visible in either or both scenes.
        """

        if not isinstance(scenes, list):
            scenes = [scenes]
        assert np.array(
            [s.cloud.point_count() == self.cloud.point_count() for s in scenes]).all(), "Error - clouds do not match."
        sets = [set(s.cloud_to_image.keys()) for s in scenes]
        keys_a = set(self.cloud_to_image.keys())
        return keys_a.union(*sets)

    def intersect_pixels(self, scene2):
        """
        Identifies matching pixels between two scenes.

        *Arguments*:
         - scene2 = the scene to match against this one

        *Returns*:
         - px1 = a numpy array of (x,y) pixel coordinates in this scene.
         - px2 = a numpy array of corresponding (x,y) pixel coordinates in scene 2.
        """

        matches = {}  # key = scene 1 pixel (x,y), value = scene 2 pixel (x,y)
        overlap = self.intersect(scene2)  # get points visible in both
        for idx in overlap:
            matches[self.get_pixel(idx)] = scene2.get_pixel(idx)

        return np.array(list(matches.keys())), np.array(list(matches.values()))

    #################################################
    ##Expose topographic correction functionality
    #################################################

    def calculate_sunvec(self, lat, lon, time, tz="Etc/GMT-1", fmt="%d/%m/%Y %H:%M"):
        """
        Estimate the illumination vector from position and time, as calculateded and with
        hylite.correct.correct.estimate_sun_vec( ... ).

        *Arguments*:
         - lat = the latitude at which to calculate the illumination vector.
         - lon = the longitude at which to calculate the illumination vector.
         - time = string describing the time and date to calculate illumination vector on, in a format
                  described by fmt.
         - tz = the timezone name (string), as recognised by pytz.
         - fmt = the format string used to parse the time/date. Default is "%d/%m/%Y %H:%M".
        *Returns*:
         - a numpy array containing the sunvector.
        """

        time = (time, fmt, tz)
        return estimate_sun_vec(lat, lon, time)

    def get_lighting(self, sunvec):
        """
        Calculate the Lambert shading based on the specified sun vector and associated cloud normals.

        *Arguments*:
         - sunvec = the illumination vector, as calculated with calculate_sunvec(...).
        *Returns*:
         - a 2D numpy array containing the shading factors.
        """

        # calculate cos incidence angles (~direct illumination)
        return estimate_incidence(self.get_normals(), sunvec)

    def estimate_ambient(self, sunvec, shadow_mask=None):
        """
        Estimates the ambient light spectra (and intensity relative to direct light) as described in
        hylite.correct.correct.estimate_ambient( ... ).

        *Arguments*:
         - sunvec = the illumination vector (pointing downwards).

        *Returns*:
         - array containing the estimated ambient light intensity fo reach band in
           the image associated with this project.
        """

        return estimate_ambient(self.image, self.get_lighting(sunvec), shadow_mask)

    def correct(self, atmos=True, topo=True, **kwds):
        """
        Apply topographic and atmospheric corrections to this scene using information on sun orientation and
        calibration targets in the image header.

        *Arguments*:
         - atmos = True if an atmospheric correction should be applied using calibration targets (ELC). Default is True.
         - topo = True if a topographic correction should be applied. Default is True.

        *Keywords*:
         - method = the method of topographic correction to apply (see hylite.correct.correct_topo(...)). Default is 'cfac'.
         - topo_kwds = a dictionary of keywords to pass to hylite.correct.correct_topo(...).
         - low_thresh = pixels darker than this percentile will be removed from the corrected scene (e.g. shadows). Default
                        is 0 (keep all pixels).
         - high_thresh = pixels brighter than this percentile will be removed from the corrected scene (e.g. overcorrected
                         highlights. Default is 100 (keep all pixels).
         - vb = True if outputs (progress text and plots to check quality of corrections) should be generated. Default is True.
         - name = a scene name for plots generated by this function.
         - bands = bands to use for plot of corrected scene. Default is hylite.RGB.
         - bbl_thresh = the threshold to use to define band bands during ELC correction. Default is the 85th percentile. Set to None
                        to disable ELC thresholding (i.e. retain all bands, regardless of noise amplification).
        *Returns*: True if correction was succesfully applied. False if information was missing in header (e.g. calibration data).
        """

        # get kwds
        vb = kwds.get('vb',True)
        high_thresh = kwds.get('high_thresh', 100)
        low_thresh = kwds.get('low_thresh', 0)
        topo_kwds = kwds.get('topo_kwds', {})
        method = kwds.get('method', 'cfac')
        name = kwds.get('name', "Correction")

        #############################
        # gather required information
        ##############################
        if atmos:
            # get calibration panels
            names = self.image.header.get_panel_names()
            if len(names) == 0:
                return False # no calibration panels

            # calculate ELC correction (but don't apply it yet)
            panels = [self.image.header.get_panel(n) for n in names]
            elc = ELC(panels)

            # store elc info in header file
            bbl_thresh = kwds.get('bbl_thresh', np.nanpercentile(elc.slope, 85) )
            if bbl_thresh is not None:
                self.image.header.set_bbl( np.logical_not(elc.get_bad_bands(thresh=bbl_thresh)))
            self.image.header['elc slope'] = elc.slope
            self.image.header['ecl intercept'] = elc.intercept

            # plot for reference
            if vb:
                fig, ax = elc.quick_plot()
                ax.set_title("%s: Empirical line correction." % name)
                fig.show()

        if topo:

            # get sun vector
            if 'sunvec' in topo_kwds:
                sunvec = topo_kwds['sunvec']
            else:
                if not ('sun azimuth' in self.image.header and 'sun elevation' in self.image.header):
                    return False  # no sun vector
                az = float(self.image.header['sun azimuth'])
                el = float(self.image.header['sun elevation'])
                sunvec = sph2cart(az+180, el) #n.b. +180 converts vector to sun to vector from sun

            # calculate cos illumination angle (lambertian shading)
            assert self.cloud.has_normals(), "Error - cannot correct topography as cloud has no normals."
            cosI = self.get_lighting(sunvec)

            # plot for reference
            if vb:
                fig, ax = plt.subplots(figsize=(18, 8))
                ax.imshow(cosI.T, vmin=0, vmax=1, cmap='gray')
                ax.set_title("%s: Lambertian shading" % name)
                ax.set_xticks([])
                ax.set_yticks([])
                fig.show()

        #######################
        # apply corrections
        #######################
        if topo:
            topo_kwds['sunvec'] = sunvec
            if 'ambient' in method:  # special case - ambient applies a combined atmospheric and topo correction
                # apply atmospheric correction first
                if atmos:
                    elc.apply(self.image) # todo - remove this once 'ambient' does topo correction

                ambient = estimate_ambient(self.image, cosI, shadow_mask=None)
                correct_topo(self.image, cosInc=cosI, method=method, ambient=ambient, **topo_kwds)

                if vb:
                    plt.figure(figsize=(10,3))
                    plt.plot( self.image.get_wavelengths(), ambient, color='b' )
                    plt.title("%s: Estimated ambient spectra" % name )
                    plt.show()

            else:
                # apply atmospheric correction first
                if atmos:
                    elc.apply(self.image)

                # apply topographic correction
                correct_topo(self.image, cosInc=cosI, method=method, **topo_kwds)
        elif atmos:  # apply only ELC (atmospheric correction)
            elc.apply(self.image)


        # apply high/low threshold postprocessing
        brightness = np.nansum(self.image.data, axis=2)
        vmax = np.nanpercentile(brightness, high_thresh)
        vmin = np.nanpercentile(brightness, low_thresh)
        mask = self.image.mask(np.logical_or(brightness > vmax, brightness < vmin))

        if vb: # plot corrected scene
            fig, ax = self.quick_plot(kwds.get('bands',hylite.RGB), vmin=0.0, vmax=1.0)
            ax.set_title("%s: Final result" % name)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.show()

        return True # success!!

    def correct_topography(self, sunvec, method='ambient', thresh=10.0, **kwds):
         """
         Calculates the illumination based on normal vectors stored in this project and the specified sunvector, and
         then applices the specified topographic correction to the image associated with this project.

         See hylite.correct.correct.correct_topo(...) for a more detailed description.

         *Arguments*:
          - sunvec = the illumination vector (pointing downwards), as calculated by calculate_sunvec( ... ).
          - method = the correction method to apply (cf. hylite.correct.correct.correct_topo ).

         *Keywords*:
          - keyword arguments are passed to hylite.correct.correct.correct_topo( ... ).

         *Returns*:
          - m, c = linear correction factors, such that i_corrected = m*i_initial + c
          - illum_mask = boolean array containing False for datapoints/pixels that are not directly illuminated.
         """

         cosI = self.get_lighting(sunvec)
         kwds['sunvec'] = kwds.get('sunvec', sunvec)
         return correct_topo(self.image, cosInc=cosI, method=method, thresh=thresh, **kwds)

    def match_colour_to(self, reference, uniform=True, method='norm', inplace=True):

        """
        Identifies matching pixels between two hyperspectral scenes and uses them to minimise
        colour differences using a linear model (aka by adjusting the brightness/contrast of this scene
        to match the brightness/contrast of the reference scene). WARNING: by default this modifies this scene's
        image IN PLACE.

        *Arguments*:
        - reference = the scene to match colours to.
        - uniform = True if a single brightness contrast adjustment is applied to all bands (avoids introducing spectral
                 artefacts). If False, different corrections area applied to each band - use with CARE! Default is True.
        - method = The colour matching method to use. Current options are:
                    - 'norm' = centre-means and scale to match standard deviation. Only compares points known to match.
                    - 'hist' = histogram equalisation. Applies to all pixels in scene - use with care!
                   Default is 'norm'.
        - inplace = True if the correction should be applied to self.image in-place. If False, no correction is
                  applied, and the correction weights (cfac and mfac) returned for future use. Default is True.
        *Returns*:
        - The corrected image as a HyImage object. If inplace=True (default) then this will be the same as self.image.
        """

        image = self.image
        if not inplace:
            image = image.copy()

        if 'norm' in method.lower():
            # get matching pixels
            px1, px2 = self.intersect_pixels(reference)
            assert px1.shape[0] > 0, "Error - no overlap between images."
            if px1.shape[0] < 1000:
                print("Warning: images only have %d overlapping pixels,"
                   " which may result in poor colour matching." % px1.shape[0])

            # extract data to create vector of matching values
            px1 = image.data[px1[:, 0], px1[:, 1], :]
            px2 = reference.image.data[px2[:, 0], px2[:, 1], :]

            # apply correction
            image.data = norm_eq( image.data, px1, px2, per_band=not uniform, inplace=True)

        elif 'hist' in method.lower():
            if uniform: # apply to whole dataset
                image.data = hist_eq(image.data, reference.image.data)
            else: # apply per band
                for b in range(self.image.band_count()):
                    image.data[:, :, b] = hist_eq(image.data[:, :, b], reference.image.data[:, :, b])
        else:
            assert False, "Error - %s is an unrecognised colour correction method." % method

        return image

    ###################################
    ##PLOTTING AND EXPORT FUNCTIONS
    ###################################
    def _gather_bands(self, bands):
        """
        Utility function used by push_to_image( ... ) and push_to_cloud( ... ).
        """

        # extract wavelength and band name info
        wav = []
        nam = []

        # loop through bands tuple/list and extract data indices/slices
        for e in bands:
            # extract from point cloud based on string
            if isinstance(e, str):
                for c in e.lower():
                    if c == 'r':
                        assert self.cloud.has_rgb(), "Error - RGB information not found."
                        nam.append('r')
                        wav.append(hylite.RGB[0])
                    elif c == 'g':
                        assert self.cloud.has_rgb(), "Error - RGB information not found."
                        nam.append('g')
                        wav.append(hylite.RGB[1])
                    elif c == 'b':
                        assert self.cloud.has_rgb(), "Error - RGB information not found."
                        nam.append('b')
                        wav.append(hylite.RGB[2])
                    elif c == 'x':
                        nam.append('x')
                        wav.append(-1)
                    elif c == 'y':
                        nam.append('y')
                        wav.append(-1)
                    elif c == 'z':
                        nam.append('z')
                        wav.append(-1)
                    elif c == 'k':
                        assert self.cloud.has_normals(), "Error - normals not found."
                        nam.append('k')
                        wav.append(-1)
                    elif c == 'l':
                        assert self.cloud.has_normals(), "Error - normals not found."
                        nam.append('l')
                        wav.append(-1)
                    elif c == 'm':
                        assert self.cloud.has_normals(), "Error - normals not found."
                        nam.append('m')
                        wav.append(-1)
            # extract slice (from image)
            elif isinstance(e, tuple):
                assert len(e) == 2, "Error - band slices must be tuples of length two."
                idx0 = self.image.get_band_index(e[0])
                idx1 = self.image.get_band_index(e[1])
                if self.image.has_band_names():
                    nam += [self.image.get_band_names()[b] for b in range(idx0, idx1)]
                else:
                    nam += [str(b) for b in range(idx0, idx1)]
                if self.image.has_wavelengths():
                    wav += [self.image.get_wavelengths()[b] for b in range(idx0, idx1)]
                else:
                    wav += [float(b) for b in range(idx0, idx1)]
            # extract band based on index or wavelength
            elif isinstance(e, float) or isinstance(e, int):
                b = self.image.get_band_index(e)
                if self.image.has_band_names():
                    nam.append(self.image.get_band_names()[b])
                else:
                    nam.append(str(b))
                if self.image.has_wavelengths():
                    wav.append(self.image.get_wavelengths()[b])
                else:
                    wav.append(float(b))
            else:
                assert False, "Unrecognised band descriptor %s" % b

        return wav, nam

    def push_to_image(self, bands, fill_holes=False, blur=0):
        """
        Export data from associated cloud and image to a (new) HyImage object.

        *Arguments*:
         - bands = a list of image band indices (int) or wavelengths (float). Inherent properties of point clouds
                   can also be expected by passing any of the following:
                    - 'rgb' = red, green and blue per-point colour values
                    - 'klm' = point normals
                    - 'xyz' = point coordinates
         - fill_holes = post-processing option to fill single-pixel holes with maximum value from adjacent pixels. Default is False.
         - blur = size of gaussian kernel to apply to image in post-processing. Default is 0 (no blur).
        *Returns*:
         - a HyImage object containing the requested data.
        """

        # special case: individual band; wrap in list
        if isinstance(bands, int) or isinstance(bands, float) or isinstance(bands, str):
            bands = [bands]

        # special case: tuple of two bands; treat as slice
        if isinstance(bands, tuple) and len(bands) == 2:
            bands = [bands]

        # gather bands and extract wavelength and name info
        wav, nam = self._gather_bands(bands)

        # rasterise and make HyImage
        img = np.full((self.image.xdim(), self.image.ydim(), len(wav)), np.nan)
        for _x in range(self.image.xdim()):
            for _y in range(self.image.ydim()):
                if not self.valid[_x, _y]:
                    continue
                pID = self.get_point_index(_x, _y)
                n = 0
                for e in bands:
                    if isinstance(e, str):  # extract from point cloud based on string
                        for c in e.lower():
                            if c == 'r':
                                img[_x, _y, n] = self.cloud.rgb[pID, 0]
                            elif c == 'g':
                                img[_x, _y, n] = self.cloud.rgb[pID, 1]
                            elif c == 'b':
                                img[_x, _y, n] = self.cloud.rgb[pID, 2]
                            elif c == 'x':
                                img[_x, _y, n] = self.cloud.xyz[pID, 0]
                            elif c == 'y':
                                img[_x, _y, n] = self.cloud.xyz[pID, 1]
                            elif c == 'z':
                                img[_x, _y, n] = self.cloud.xyz[pID, 2]
                            elif c == 'k':
                                img[_x, _y, n] = self.normals[_x, _y, 0]
                            elif c == 'l':
                                img[_x, _y, n] = self.normals[_x, _y, 1]
                            elif c == 'm':
                                img[_x, _y, n] = self.normals[_x, _y, 2]
                            n += 1
                        continue
                    elif isinstance(e, tuple):  # extract slice (from image)
                        assert len(e) == 2, "Error - band slices must be tuples of length two."
                        idx0 = self.image.get_band_index(e[0])
                        idx1 = self.image.get_band_index(e[1])
                        slc = self.image.data[_x, _y, idx0:idx1]
                        img[_x, _y, n:n + len(slc)] = slc
                        n += len(slc)
                        continue
                    elif isinstance(e, float) or isinstance(e, int):  # extract band based on index or wavelength
                        b = self.image.get_band_index(e)
                        img[_x, _y, n] = self.image.data[_x, _y, b]
                        n += 1
                        continue
                    else:
                        assert False, "Unrecognised band descriptor %s" % b

        # build HyImage
        img = HyImage(img, header=self.image.header.copy())
        img.set_band_names(nam)
        img.set_wavelengths(wav)

        # postprocessing
        if fill_holes:
            img.fill_holes()
        if blur > 2:
            img.blur(int(blur))

        return img

    def push_to_cloud(self, bands):
        """
        Export data from associated image and cloud to a (new) HyCloud object.

        *Arguments*:
         - bands = a list of image band indices (int) or wavelengths (float). Inherent properties of point clouds
                   can also be expected by passing any of the following:
                    - 'rgb' = red, green and blue per-point colour values
                    - 'klm' = point normals
                    - 'xyz' = point coordinates
        *Returns*:
         - a HyImage object containing the requested data.
        """

        # special case: individual band; wrap in list
        if isinstance(bands, int) or isinstance(bands, float) or isinstance(bands, str):
            bands = [bands]

        # special case: tuple of two bands; treat as slice
        if isinstance(bands, tuple) and len(bands) == 2:
            bands = [bands]

        # gather bands and extract wavelength and name info
        wav, nam = self._gather_bands(bands)

        # loop through points in cloud and add data
        data = np.full((self.cloud.point_count(), len(wav)), np.nan)
        valid = np.full(self.cloud.point_count(), False, dtype=np.bool)
        for i in range(self.cloud.point_count()):
            # is point visible?
            _x, _y = self.get_pixel(i)
            if _x is None:
                continue

            valid[i] = True  # yes - this point has data

            # gather data
            n = 0
            for e in bands:
                if isinstance(e, str):  # extract from point cloud based on string
                    for c in e.lower():
                        if c == 'r':
                            data[i, n] = self.cloud.rgb[i, 0]
                        elif c == 'g':
                            data[i, n] = self.cloud.rgb[i, 1]
                        elif c == 'b':
                            data[i, n] = self.cloud.rgb[i, 2]
                        elif c == 'x':
                            data[i, n] = self.cloud.xyz[i, 0]
                        elif c == 'y':
                            data[i, n] = self.cloud.xyz[i, 1]
                        elif c == 'z':
                            data[i, n] = self.cloud.xyz[i, 2]
                        elif c == 'k':
                            data[i, n] = self.cloud.normals[i, 0]
                        elif c == 'l':
                            data[i, n] = self.cloud.normals[i, 1]
                        elif c == 'm':
                            data[i, n] = self.cloud.normals[i, 2]
                        n += 1
                    continue
                elif isinstance(e, tuple):  # extract slice (from image)
                    assert len(e) == 2, "Error - band slices must be tuples of length two."
                    idx0 = self.image.get_band_index(e[0])
                    idx1 = self.image.get_band_index(e[1])
                    slc = self.image.data[_x, _y, idx0:idx1]
                    data[i, n:(n + len(slc))] = slc
                    n += len(slc)
                    continue
                elif isinstance(e, float) or isinstance(e, int):  # extract band based on index or wavelength
                    b = self.image.get_band_index(e)
                    data[i, n] = self.image.data[_x, _y, b]
                    n += 1
                    continue
                else:
                    assert False, "Unrecognised band descriptor %s" % b

        # build HyCloud
        cloud = self.cloud.copy(data=False)
        cloud.data = data
        cloud.filter_points(0, np.logical_not(valid))  # remove points with no data
        cloud.set_band_names(nam)
        cloud.set_wavelengths(wav)

        return cloud

    def quick_plot(self, band=0, ax=None, bfac=0.0, cfac=0.0,
                   **kwds):
        """
        Plot a projected data using matplotlib.imshow(...).

        *Arguments*:
         - band = the band name (string), index (integer) or wavelength (float) to plot. Default is 0. If a tuple is passed then
                  each band in the tuple (string or index) will be mapped to rgb.
         - bands = List defining the bands to include in the output image. Elements should be one of:
              - 'rgb' = rgb
              - 'xyz' = point position
              - 'klm' = normal vectors
              - numeric = index (int), wavelength (float) of an image band
              - tuple of length 3: wavelengths or band indices to map to rgb.
         - ax = an axis object to plot to. If none, plt.imshow( ... ) is used.
         - bfac = a brightness adjustment to apply to RGB mappings (-1 to 1)
         - cfac = a contrast adjustment to apply to RGB mappings (-1 to 1)
        *Keywords*:
         - keywords are passed to matplotlib.imshow( ... ).
        """

        # plot hyImage data
        if not isinstance(band, str):
            kwds["mask"] = np.logical_not( np.isfinite(self.depth) ) #np.logical_not(self.valid)  # mask out pixels with no valid point mappings
            return self.image.quick_plot(band, ax, bfac, cfac, **kwds)  # render
        else:
            img = self.push_to_image(band)
            if (len(band) == 3):
                # do some normalizations
                mn = kwds.get("vmin", np.nanmin(img.data[img.data != 0]))
                mx = kwds.get("vmax", np.nanmax(img.data[img.data != 0]))
                img.data = (img.data - mn) / (mx - mn)
                if 'x' in band or 'y' in band or 'z' in band:  # coordinates need per-band mapping
                    for i in range(3):
                        img.data[..., i] = (img.data[..., i] - np.nanmin(img.data[..., i])) / (
                                np.nanmax(img.data[..., i]) - np.nanmin(img.data[..., i]))
                # plot it image
                return img.quick_plot((0, 1, 2), ax=ax, bfac=bfac, cfac=cfac, **kwds)
            else:
                return img.quick_plot(0, ax=ax, bfac=bfac, cfac=cfac, **kwds)

    @classmethod
    def build_hypercloud(cls, scenes, bands, blending_mode='average', trim=True, vb=True, inplace=True, export_footprint = False):
        """
        Combine multiple HyScene objects into a hypercloud. Warning - this modifies the cloud associated with the HyScenes in-place.
        Returns processed cloud and (optional) footprint map indicating the number of involved scenes per pixel.

        *Arguments*:
         - scenes = a list of scenes to combine. These scenes must all reference the same point cloud!
         - bands = either:
                 (1) a tuple containing the (min,max) wavelength to map. If range is a tuple, -1 can be used to specify the
                     last band index.
                 (2) a list of bands or boolean mask such that image.data[:,:,range] is exported to the hypercloud.
         - blending_mode = the mode used to blend points that can be assigned values from multiple values. Options are:
               - 'average' (default) = calculate the average of all pixels that map to the point
               - 'weighted' = calculate the average of all pixels, weighted by inverse footprint size.
               - 'gsd' = chose the pixel with the smallest gsd (i.e. the pixel with the closest camera)
               - 'obl' = chose the pixel that is least oblique (i.e. most perpendicular to the surface)
         - trim = True if the point cloud should be trimmed after doing project. Default is True.
         - vb = True if output should be written to the console. Default is True.
         - inplace = True if the reference cloud should be modified in place (to save RAM). Default is True.
         - export_footprint = True if footprint map indicating the number of involved scenes per pixel.
        """

        if vb: print("Preparing data....", end='')

        # get cloud
        cloud = scenes[0].cloud
        if not inplace:
            cloud = cloud.copy()

        # remove everything except desired bands from scenes and calculate range of values
        images = []
        for i, s in enumerate(scenes):
            images.append(s.image.export_bands(bands))  # get bands of interest
            cloud.header.set_camera( s.camera, i ) # add camera to cloud header

        # do scenes all have wavelength data?
        has_wav = np.array([i.has_wavelengths() for i in images]).all()
        wavelengths = None
        band_names = None
        if has_wav:  # match wavelengths between scenes
            wavelengths = []  # list of wavelengths
            indices = []  # list of corresponding band indices for each scene
            for w in images[0].get_wavelengths():
                idx = []
                for i, img in enumerate(images):
                    if not w in img.get_wavelengths():  # bail!
                        assert False, "Error - Scene %d does have data for data for band %d." % (i, w)
                    else:
                        idx.append(img.get_band_index(w))
                indices.append(idx)
                wavelengths.append(w)  # this wavelength is in all scenes

            wavelengths = np.array(wavelengths)
            band_names = ["%.2f" % w for w in wavelengths]
            indices = np.array(indices)

            assert wavelengths.shape[0] > 0, "Error - images do not have any matching bands."
        else:  # no - check all images have the same number of bands...
            assert (np.array([i.band_count() for i in images]) == images[0].band_count()).all(), \
                "Error - scenes with now wavelength information must have (exactly) the same band count."
            # no wavelengths, create band list and set band names
            band_list = np.array([i for i in range(scenes[0].image.band_count())])
            # noinspection PyUnusedLocal
            indices = np.array([band_list for s in scenes]).T  # bands to export
            # noinspection PyUnusedLocal
            band_names = ["%s" for b in band_list]

        # handle bbl
        if np.array([img.header.has_bbl() for img in images]).all():
            bbl = np.full(wavelengths.shape[0], True)
            for i, idx in enumerate(indices):  # list of indices for each band
                bb = [images[n].header.get_bbl()[idx[n]] for n in range(len(images))]
                bbl[i] = np.array(bb).all()  # if any data is bad, this becomes a bad band
            cloud.header.set_bbl(bbl)  # store

        if vb: print(" Done.")

        # create data array
        data = np.zeros((cloud.point_count(), len(band_names)),
                        dtype=np.float32)  # point values will be accumulated here
        point_count = np.zeros(cloud.point_count(), dtype=np.float32)  # number of pixels used to calculate each point

        # loop through points
        if vb:
            loop = tqdm(range(data.shape[0]), desc='Projecting data', leave=False)
        else:
            loop = range(data.shape[0])
        if 'average' in blending_mode.lower():
            for n in loop:
                for i, s in enumerate(scenes):
                    px, py = s.get_pixel(n)
                    if px is not None:  # point exists in scene
                        if np.isfinite(images[i].data[px, py, indices[:, i]]).all():  # is pixel finite?
                            point_count[n] += 1  # increment pixel count
                            data[n, :] += images[i].data[px, py, indices[:, i]]  # accumulate point value
            # divide by point count to calculate average
            data = data / point_count[:, None]
        elif 'gsd' in blending_mode.lower():
            for n in loop:
                best = np.inf
                for i, s in enumerate(scenes):
                    px, py = s.get_pixel(n)
                    if px is not None:  # point exists in scene
                        if np.isfinite(images[i].data[px, py, indices[:, i]]).all():  # is pixel finite?
                            point_count[n] += 1  # increment pixel count
                            if s.depth[px, py] < best:  # is this the best GSD so far?
                                best = s.depth[px, py]  # store best GSD
                                data[n, :] = images[i].data[px, py, indices[:, i]]  # set point value
        elif 'obl' in blending_mode.lower():
            for n in loop:
                best = np.inf
                for i, s in enumerate(scenes):
                    px, py = s.get_pixel(n)
                    if px is not None:  # point exists in scene
                        if np.isfinite(images[i].data[px, py, indices[:, i]]).all():  # is pixel finite?
                            point_count[n] += 1  # increment pixel count
                            if s.obliquity[px, py] < best:  # is this the best GSD so far?
                                best = s.obliquity[px, py]  # store best GSD
                                data[n, :] = images[i].data[px, py, indices[:, i]]  # set point value
        elif 'weight' in blending_mode.lower():
            for n in loop:
                for i, s in enumerate(scenes):
                    px, py = s.get_pixel(n)
                    if px is not None:  # point exists in scene
                        if np.isfinite(images[i].data[px, py, indices[:, i]]).all():  # is pixel finite?
                            w = 1 / s.depth[px, py] # calculate weight and increment to count
                            point_count[n] += w  #
                            data[n, :] += w * images[i].data[px, py, indices[:, i]]  # accumulate point value
            # divide by sum of weights to get weighted average
            data = data / point_count[:, None]
        else:
            assert False, "Error - unrecognised blending mode %s" % blending_mode

        # add bands to point cloud
        cloud.set_bands(data, band_names=band_names, wavelengths=wavelengths)

        # export footprint?
        if export_footprint:
            footprint = hylite.HyCloud(cloud.xyz, bands=point_count[:, None])
            if trim:
                footprint.filter_points(0, point_count == 0)  # remove zero points

        # trim cloud?
        if trim:
            cloud.filter_points(0, point_count == 0)  # remove zero points

        # return
        if export_footprint:
            return cloud, footprint
        else:
            return cloud
