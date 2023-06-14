"""
Projection maps store lookup tables (python dictionaries) that link PointIDs in a point cloud with pixelIDs in an image.
They store many : many relationships and can store arbitrarily complicated projections ( perspective, panoramic,
pushbroom etc.). PMaps only store the mapping function; see HyScene for functionality that pushes data between different
data types. PMaps can be saved using io.save( ... ) to avoid expensive computation multiple times in a processing
workflow.
"""

import os
import numpy as np
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix
import hylite
from tempfile import mkdtemp
import shutil
from tqdm import tqdm

class PMap(object):
    """
    A class for storing lookup tables that map between 3-D point clouds and 2-D images.
    """

    def __init__(self, xdim, ydim, npoints, cloud=None, image=None):
        """
        Create a new (empty) PMap object.

        Args:
             xdim: the width of the associated image in pixels. Used for ravelling indices.
             ydim: the height of the associated image in pixels. Used for ravelling indices.
             points: the number of points that are in this map. Used for ravelling indices.
             cloud: a link to the source cloud, default is None.
             image: a link to the source image, default is None.
        """
        self.xdim = xdim  # width of the associated image
        self.ydim = ydim  # height of the associated image
        self.npoints = npoints
        self.cloud = cloud # store ref to cloud if provided
        self.image = image # store ref to image if provided
        self.points = [] # points with non-zero references in this matrix
        self.pixels = [] # pixels with non-zero references in this matrix
        self.data = coo_matrix( (npoints, xdim*ydim), dtype=np.float32 ) # initialise mapping matrix

    def _ridx(self, pixel):
        """
        Ravel a pixel index tuple to an integer. If an integer is passed to this function
        it will be returned directly.
        """
        if isinstance(pixel, tuple):
            assert len(pixel) == 2, "Error - (x,y) tuples must have length 2."
            pixel = np.ravel_multi_index(pixel, dims=(self.xdim, self.ydim), order='F')
        assert np.issubdtype(type(pixel), np.integer), "Error - non-integer pixel ID (%s = %s)?" % (
        pixel, type(pixel))  # always wear protection
        return pixel

    def _uidx(self, pixel):
        """
        Unravel an integer pixel index to a (x,y) tuple. If a tuple is passed to this function
        it will be returned directly.
        """
        if np.issubdtype(type(pixel), np.integer):
            pixel = np.unravel_index(pixel, (self.xdim, self.ydim), order='F')
        elif isinstance(pixel, tuple):
            pass
        else:
            assert np.issubdtype(type(pixel), np.integer), "Error - non-integer pixel ID (%s = %s)?" % (
            pixel, type(pixel))
        return pixel

    def coo(self):
        """
        Convert this pmap's internal sparse matrix to coo format.
        """
        if not isinstance(self.data, coo_matrix):
            self.data = self.data.tocoo()
    def csc(self):
        """
        Convert this pmap's internal sparse matrix to compressed column format.
        """
        if not isinstance(self.data, csc_matrix):
            self.data = self.data.tocsc()
    def csr(self):
        """
        Convert this pmap's internal sparse matrix to compressed row format.
        """
        if not isinstance(self.data, csr_matrix):
            self.data = self.data.tocsr()

    def get_flat(self):
        """
        Return three flat arrays that contain all the links in this projection map.

        Returns:
            A tuple containing:

            - points = an (n,) list of point ids.
            - pixels = an (n,) list of pixel ids corresponding to the points above.
            - z = an (n,) list of distances between each point and corresponding pixel.
        """
        self.coo()
        return self.data.row, self.data.col, 1/self.data.data # return

    def set_flat(self, points, pixels, z ):
        """
        Adds links to this pmap from flattened arrays as returned by get_flat( ... ).

        Args:
            points: an (n,) list of point ids.
            pixels: an (n,) list of pixel ids corresponding to the points above.
            z: an (n,) list of distances between each point and corresponding pixel.
        """
        # easy!
        self.data = coo_matrix( (1/z, (points,pixels)), shape=(self.npoints, self.xdim*self.ydim), dtype=np.float32 )

        # also store unique points and pixels
        self.points = np.unique( points )
        self.pixels = np.unique( pixels )

    def set_ppc(self, pp, vis):
        """
        Adds links to the pmap based on a projected point coordinates array and a visibility list, as returned by
        e.g. proj_persp and proj_pano.

        Args:
            pp: projected point coordinates as returned by e.g. proj_pano.
            vis: point visibilities, as returned by e.g. proj_pano.
        """
        # convert to indices
        pid = np.argwhere(vis)[:, 0]
        pp = pp[vis]

        # convert pixel indices to flat indices
        pix = np.ravel_multi_index(pp[:,[0,1]].astype(int).T, dims=(self.xdim, self.ydim), order='F')

        # set data values
        self.set_flat( pid, pix, pp[:,2] )

    def size(self):
        """
        How many relationships are stored in this?
        """
        return len( self.data.data )

    def point_count(self):
        """
        How many points are included in this mapping (i.e. how many points are mapped to pixels?)
        """
        return len(self.points)

    def pixel_count(self):
        """
        How many pixels are included in this mapping?
        """
        return len(self.pixels)

    def get_depth(self, pixel, point):
        """
        Get the distance between a pixel and point pair. Returns
        None if there is no mapping between the pixel and the point.
        """
        pixel = self._ridx(pixel)
        if isinstance(self.data, coo_matrix): # need to change from coo coordiantes
            self.csc()
        return 1 / self.data[ point, pixel ] # n.b. note that matrix entries are 1 / z

    def get_point_index(self, pixel):
        """
        Get the index of the closest point in the specified pixel

        Args:
            pixel: the index of the pixel (integer or (x,y) tuple).
        Returns:
            A tuple containing:

             - point = the point index or None if no points are in the specified pixel.
             - depth = the distance to this point.
        """
        self.csc() # convert to column format
        C = self.data[:, self._ridx(pixel)] # get column
        return C.nonzero()[0][ np.argmax( C.data ) ], 1 / np.max( C.data ) # return closest point
                                                                       # n.b. note that matrix entries are 1 / z

    def get_point_indices(self, pixel):
        """
        Get the indices and depths of all points in the specified pixel.

        Args:
            pixel: the index of the pixel (integer or (x,y) tuple).
        Returns:
            A tuple containing:
                 - points = a list of point indices, or [ ] if no points are present.
                 - depths = a list of associated depths, or [ ] if no points are present.
        """
        self.csc() # convert to column format
        C = self.data[:, self._ridx(pixel)] # get column
        return C.nonzero()[0], 1 / C.data # return indices and depths

    def get_pixel_index(self, point):
        """
        Get the index of the closest pixel to the specified point.

        Args:
            point: the point index
        Returns:
            A tuple containing:

             - (px, py) = the pixel coordinates, or None if no mapping exists
             - depth = the distance to this pixel.
        """
        self.csr() # convert to row format
        R = self.data[point, :] # get row
        return self._uidx( R.nonzero()[1][ np.argmax( R.data ) ] ), 1 / np.max( R.data ) # closest pixel and depth

    def get_pixel_indices(self, point):
        """
        Get a list of pixel coordinates associated with the specified point.

        Args:
            point: the point index

        Returns:
            A tuple containing:

                 - pixels = a list of (n,2) containing pixel coordinates.
                 - depths = a list of (n,) containing associated distances.
        """
        self.csr() # convert to row format
        R = self.data[point,:] # get row
        return np.array( np.unravel_index(R.nonzero()[1], (self.xdim, self.ydim), order='F')).T, R.data

    def get_pixel_depths(self):
        """
        Return a (xdim,ydim) array containing the depth to the closest point in each pixel. Pixels with no points
        will be given 0 values.
        """
        out = (1 / np.max(self.data, axis=0).toarray()).astype(np.float32)
        out[np.logical_not(np.isfinite(out))] = 0
        return out.reshape(self.xdim,self.ydim,order='F')

    def get_point_depths(self):
        """
        Return a (npoints,) array containing the depth to the closest pixel from each point. Points with no
        pixels will be given 0 values.
        """
        out = (1 / np.max(self.data, axis=1).toarray()).astype(np.float32)
        out[np.logical_not(np.isfinite(out))] = 0
        return out

    def points_per_pixel(self):
        """
        Calculate how many points are in each pixel.

        Returns:
            a HyImage instance containing point counts per pixel.
        """
        self.csr() # use row-compressed form
        W = (self.data > 0).astype(np.float32)  # convert to binary adjacency matrix
        npnt = np.array(W.sum(axis=0)).ravel()  # get number of points per pixel
        return hylite.HyImage( npnt.reshape( (self.xdim, self.ydim, 1 ), order='F' ) )

    def pixels_per_point(self):
        """
        Calculates how many pixels project to each point.

        Returns:
            a copy of self.cloud, but with a scalar field containing pixel counts per point. If self.cloud is not defined
           then a numpy array of point counts will be returned.
        """
        self.csc() # use column compressed form
        W = (self.data > 0).astype(np.float32)  # convert to binary adjacency matrix
        npix = np.array(W.sum(axis=1)).ravel()  # get number of points per pixel

        if self.cloud is not None: # return a cloud
            out = self.cloud.copy( data = False )
            out.data = npix[:,None]
            return out
        else:
            return npix # return a numpy array

    def intersect(self, map2):
        """
        Get point indices that exist (are visible in) this scene and another.

        Args:
            map2: a PMap instance that references the same cloud but with a different image/viewpoint.
        Returns:
            a list of point indices that are visible in both scenes.
        """
        S1 = set( self.points )
        S2 = set( map2.points )
        return list( S1 & S2 )

    def union(self, maps):
        """
        Returns points that are included in one or more of the passed PMaps.

        Args:
            maps: a list of pmap instances to compare with (or just one).
        Returns:
            a list of point indices that are visible in either or both scenes.
        """
        if not isinstance(maps, list):
            maps = [maps]
        S_n = [set(s.points) for s in maps]
        S1 = set(self.points)
        return S1.union(*S_n)

    def intersect_pixels(self, map2):
        """
        Identifies matching pixels between two scenes.

        Args:
            map2: the scene to match against this one

        Returns:
            A tuple containing:

             - px1 = a numpy array of (x,y) pixel coordinates in this scene.
             - px2 = a numpy array of corresponding (x,y) pixel coordinates in scene 2.
        """
        px1 = []
        px2 = []
        overlap = self.intersect(map2)  # get points visible in both
        for idx in overlap:
            for _px1 in self.get_pixel_indices(idx)[0]:
                for _px2 in map2.get_pixel_indices(idx)[0]:
                    px1.append(_px1)
                    px2.append(_px2)
        return np.array(px1), np.array(px2)

    def remove_nan_pixels(self, image=None):
        """
        Removes mappings to nan pixels from linkage matrix.

        Args:
            image: the image containing nan pixels that should be removed. Default is self.image.
        """
        self.csc()  # change to column format

        if image is None:
            image = self.image

        # build list of nan columns
        isnan = np.logical_not(np.isfinite(image.data).all(axis=-1)).ravel(order='F')
        f = np.ones(isnan.shape[0])
        f[isnan] = 0

        # zero elements in these
        self.data = self.data.multiply(f)

        # remove zero elements
        self.data.eliminate_zeros()

    def filter_footprint(self, thresh=50):
        """
        Filter projections in a PMap instance and remove pixels that have a
        on-ground footprint above the specified threshold. This operation is
        applied in-place to conserve memory.

        Args:
            thresh: the maximum allowable pixel footprint (in points). Pixels containing > than
                    this number of points will be removed from the projection map.
        """

        # calculate footprint
        W = (self.data > 0).astype(np.float32)  # convert to binary adjacency matrix
        n = np.array(W.sum(axis=0)).ravel()  # get number of points per pixel

        # if isinstance(thresh, int): # calculate threshold as percentile if need be
        #    thresh = np.percentile( n[ n > 0], thresh )

        # convert to coo format
        self.coo()

        # rebuild mapping matrix
        mask = n[self.data.col] < thresh  # valid points
        self.data = coo_matrix((self.data.data[mask], (self.data.row[mask], self.data.col[mask])),
                       shape=(self.npoints, self.xdim * self.ydim), dtype=np.float32)

    def filter_occlusions(self, occ_tol=5.):
        """
        Filter projections in a PMap instance and remove points that are likely to be
        occluded. This operation is applied in-place to conserve memory.

        Args:
            occ_tol: the tolerance of the occlusion culling. Points within this distance of the
                     closest point in each pixel will be retained.
        """

        zz = np.max(self.data, axis=0).power(-1)  ## calculate closest point in each pixel
        zz.data += occ_tol
        zz = zz.tocsc()
        self.coo()

        # rebuild mapping matrix
        mask = self.data.data > (1 / zz[0, self.data.col].toarray()[0, :])  ## which points to include
        self.data = coo_matrix((self.data.data[mask], (self.data.row[mask], self.data.col[mask])),
                       shape=(self.npoints, self.xdim * self.ydim), dtype=np.float32)


def _gather_bands(data, bands):
    """
    Utility function used by push_to_image( ... ) and push_to_cloud( ... ) to slice data from a HyData instance.

    Returns:
        A tuple containing:

             - data = a data array containing the requested bands (hopefully).
             - wav = the wavelengths of the extracted bands (or -1 for non-spectral attributes).
             - names = the names of the extracted bands.
    """

    # extract wavelength and band name info
    dat = []
    wav = []
    nam = []

    # loop through bands tuple/list and extract data indices/slices
    for e in bands:
        # extract from point cloud based on string
        if isinstance(e, str):
            for c in e.lower():
                if c == 'r':
                    assert data.has_rgb(), "Error - RGB information not found."
                    dat.append(data.rgb[..., 0])
                    nam.append('r')
                    wav.append(hylite.RGB[0])
                elif c == 'g':
                    assert data.has_rgb(), "Error - RGB information not found."
                    dat.append(data.rgb[..., 1])
                    nam.append('g')
                    wav.append(hylite.RGB[1])
                elif c == 'b':
                    assert data.has_rgb(), "Error - RGB information not found."
                    dat.append(data.rgb[..., 2])
                    nam.append('b')
                    wav.append(hylite.RGB[2])
                elif c == 'x':
                    dat.append(data.xyz[..., 0])
                    nam.append('x')
                    wav.append(-1)
                elif c == 'y':
                    dat.append(data.xyz[..., 1])
                    nam.append('y')
                    wav.append(-1)
                elif c == 'z':
                    dat.append(data.xyz[..., 2])
                    nam.append('z')
                    wav.append(-1)
                elif c == 'k':
                    assert data.has_normals(), "Error - normals not found."
                    dat.append(data.normals[..., 0])
                    nam.append('k')
                    wav.append(-1)
                elif c == 'l':
                    assert data.has_normals(), "Error - normals not found."
                    dat.append(data.normals[..., 1])
                    nam.append('l')
                    wav.append(-1)
                elif c == 'm':
                    assert data.has_normals(), "Error - normals not found."
                    dat.append(data.normals[..., 2])
                    nam.append('m')
                    wav.append(-1)
        # extract slice
        elif isinstance(e, tuple):
            assert len(e) == 2, "Error - band slices must be tuples of length two."
            idx0 = data.get_band_index(e[0])
            idx1 = data.get_band_index(e[1])
            dat += [data.data[..., b] for b in range(idx0, idx1)]
            if data.has_band_names():
                nam += [data.get_band_names()[b] for b in range(idx0, idx1)]
            else:
                nam += [str(b) for b in range(idx0, idx1)]
            if data.has_wavelengths():
                wav += [data.get_wavelengths()[b] for b in range(idx0, idx1)]
            else:
                wav += [float(b) for b in range(idx0, idx1)]
        # extract band based on index or wavelength
        elif isinstance(e, float) or isinstance(e, int):
            b = data.get_band_index(e)
            dat.append(data[..., b])
            if data.has_band_names():
                nam.append(data.get_band_names()[b])
            else:
                nam.append(str(b))
            if data.has_wavelengths():
                wav.append(data.get_wavelengths()[b])
            else:
                wav.append(float(b))
        else:
            assert False, "Unrecognised band descriptor %s" % b

    if data.is_image():
        dat = np.dstack(dat)  # stack
    else:
        dat = np.vstack(dat).T
    return dat, wav, nam

def push_to_cloud(pmap, bands=(0, -1), method='best', image=None, cloud=None ):
    """
    Push the specified bands from an image onto a hypercloud using a (precalculated) PMap instance.

    Args:
        pmap: a pmap instance. the pmap.image and pmap.cloud references must also be defined.
        bands: List defining the bands to include in the output dataset. Elements should be one of:

            - numeric = index (int), wavelength (float) of an image band
            - tuple of length 2: start and end bands (float or integer) to export.
            - iterable of length > 2: list of bands (float or integer) to export.
        method: The method used to condense data from multiple pixels onto each point. Options are:

            - 'closest': use the closest pixel to each point.
            - 'distance': average with inverse distance weighting.
            - 'count' : average weighted inverse to the number of points in each pixel.
            - 'best' : use the pixel that is mapped to the fewest points (only). Default.
            - 'average' : average with all pixels weighted equally.

        image: the image to project (if different to pmap.image). Must have matching dimensions. Default is pmap.image.
        cloud: the cloud to project (if different to pmap.cloud). Must have matching dimensions. Default is pmap.cloud.

    Returns:
        A HyCloud instance containing the back-projected data.
    """

    if image is None:
        image = pmap.image
    if cloud is None:
        cloud = pmap.cloud

    # get image array of data to copy across
    data = image.export_bands(bands)

    # flatten it
    X = data.data.reshape(-1, data.data.shape[-1], order='F')  # n.b. we don't used data.X() to ensure 'C' order

    # convert pmap to csc format
    pmap.csc()  # thunk; would csr format be faster??
    pmap.remove_nan_pixels(image=image) # drop nan pixels

    # build weights matrix
    if 'closest' in method.lower():
        # get closest pixels
        closest = np.argmax(pmap.data, axis=1)  # find closest pixel to each point (largest 1/z along cols)

        # assemble sparse matrix with closest pixels scored as 1
        rows = np.argwhere(closest > 0)[:, 0]
        cols = np.array(closest[rows])[:, 0]
        vals = np.ones(rows.shape)
        W = csc_matrix((vals, (rows, cols)), pmap.data.shape, dtype=np.float32)

        # sum of weights is easy
        n = np.ones(pmap.npoints, dtype=np.float32)
    elif 'average' in method.lower():
        W = (pmap.data > 0).astype(np.float32)  # weights matrix [ full of ones ]
        n = np.array(W.sum(axis=1))[:, 0]  # sum of weights (for normalising later)
    elif 'count' in method.lower() or 'best' in method.lower():
        W = (pmap.data > 0).astype(np.float32)  # convert to binary adjacency matrix
        npoints = np.array(W.sum(axis=0))[0, :]  # get number of points per pixel
        W = W.multiply(1 / npoints)  # fill non-zero values with 1 / points in relevant pixel
        if 'best' in method.lower():  # filter W so we keep only the best points
            best = np.argmax(W, axis=1)  # assemble sparse matrix with best pixels scored as 1
            rows = np.argwhere(best > 0)[:, 0]
            cols = np.array(best[rows])[:, 0]
            vals = np.ones(rows.shape)
            W = csc_matrix((vals, (rows, cols)), pmap.data.shape, dtype=np.float32)

        n = np.array(W.sum(axis=1))[:, 0]  # sum of weights (for normalising later)
    elif 'distance' in method.lower():
        W = pmap.data  # easy!
        n = np.array(W.sum(axis=1))[:, 0]  # sum of weights (for normalising later)
    else:
        assert False, "Error - %s is an invalid method." % method

    # calculate output
    #V = W.dot(X) / n[:, None]
    V = W@X / n[:, None]

    # build output cloud
    out = cloud.copy(data=False)
    out.data = V
    if data.has_wavelengths():
        out.set_wavelengths(data.get_wavelengths())

    return out


def push_to_image(pmap, bands='xyz', method='closest', image=None, cloud=None):
    """
    Project the specified data from a point cloud onto an image using a (precalculated) PMap instance. If multiple points map
    to a single pixel then the results are averaged.

    Args:
        pmap: a pmap instance. the pmap.image and pmap.cloud references must also be defined.
        bands: List defining the bands to include in the output dataset. Elements should be one of:

            - numeric = index (int), wavelength (float) of an image band
            - bands = a list of image band indices (int) or wavelengths (float). Inherent properties of point clouds
               can also be expected by passing any combination of the following:
                - 'rgb' = red, green and blue per-point colour values
                - 'klm' = point normals
                - 'xyz' = point coordinates
            - iterable of length > 2: list of bands (float or integer) to export.

        method: The method used to condense data from multiple points onto each pixel. Options are:

            - 'closest': use the closest point to each pixel (default is this is fastest).
            - 'average' : average with all pixels weighted equally. Slow.

        image: the image to project (if different to pmap.image). Must have matching dimensions. Default is pmap.image.
        cloud: the cloud to project (if different to pmap.cloud). Must have matching dimensions. Default is pmap.cloud.
    Returns:
        A HyImage instance containing the projected data.
    """

    if image is None:
        image = pmap.image
    if cloud is None:
        cloud = pmap.cloud

    # special case: individual band; wrap in list
    if isinstance(bands, int) or isinstance(bands, float) or isinstance(bands, str):
        bands = [bands]

        # special case: tuple of two bands; treat as slice
        if isinstance(bands, tuple) and len(bands) == 2:
            bands = [bands]

    # gather data to project
    dat, wav, nam = _gather_bands(cloud, bands)  # extract point normals, positions and sky view

    # convert pmap to csr format
    # pmap.csr()

    # build weights matrix
    if 'closest' in method.lower():
        closest = np.array(np.argmax(pmap.data, axis=0))[0,
                  :]  # find closest point to each pixel (largest 1/z along rows)

        # assemble sparse matrix with closest points scored as 1
        cols = np.argwhere(closest > 0)[:, 0]
        rows = np.array(closest[cols])
        vals = np.ones(rows.shape)
        W = csc_matrix((vals, (rows, cols)), pmap.data.shape, dtype=np.float32)
        #V = W.T.dot(dat)  # project closest poits
        V = W.T@dat  # project closest poits
    elif 'average' in method.lower():
        W = (pmap.data > 0).astype(np.float32)  # weights matrix [ full of ones ]
        n = np.array(W.sum(axis=0))[0, :]  # sum of weights
        #V = W.T.dot(dat) / n[:, None]  # calculate average
        V = W.T@dat / n[:, None]  # calculate average
    else:
        assert False, "Error - %s is an invalid method for cloud_to_image." % method

    # build output image
    out = image.copy(data=False)
    out.data = np.reshape(V, (image.xdim(), image.ydim(), -1), order='F')
    out.data[out.data == 0] = np.nan  # replace zeros with nans
    out.set_wavelengths(wav)
    out.set_band_names(nam)

    return out

# functions for blending scenes together

def push_geomattr(scene, method='best'):
    """
    Return a HyCloud instance containing the geometric attributes of a given projection. This will have
    two or three bands as follows:

     - distance from the sensor to each point
     - effective GSD (number of points per source pixel)
     - obliquity (angle in degrees between surface normal and view vector), if the point cloud has normal vectors.

    These are useful for QAQC of projected results or doing weighted blending.

    Args:
        scene: the HyScene instance to compute geometric attributes for.
        method: the projection method used for handling duplicate pixels. See scene.push_to_cloud for details.
    Returns:
        a HyCloud instance containing the three geometric attributes.
    """
    # get projection map and relevant attributes (in image space)
    pmap = scene.pmap
    attr = [scene.depth,  # point depth
            pmap.points_per_pixel().data[..., 0]]  # gsd
    bands = ['depth', 'gsd']
    if hasattr(scene, "normals"):
        # dot product of view product and surface normal vector
        attr.append(np.rad2deg(np.arccos(
            -(scene.view * scene.normals).sum(-1))))
        bands.append('obliquity')

    # replace zeros with nans
    for a in attr:
        a[a == 0] = np.nan
    image = hylite.HyImage(np.dstack(attr))
    attr = scene.push_to_cloud((0, -1), method=method, image=image)
    attr.set_band_names(bands)
    return attr


def get_blend_weights(scenes, method, ascloud=True):
    """
    Compute weights for doing scene blending based on geometric attributes computed using
    push_geomattr( ... ).

    Args:
        scenes: a list of scenes to compute blend weights for.
        method: the weighting method. Options are:

             - 'equal' = all values are weighted equally.
             - 'distance' = closer points are given higher weight.
             - 'gsd' = lower ground-sampling values are given higher weight.
             - 'obliquity' = less oblique points are given higher weight.

        ascloud: True if weights should be returned as a HyCloud instance. Otherwise a numpy array is returned.
    Returns:
        a numpy array containing the normalised blending weight for each (point,scene).
    """

    weights = np.ones((scenes[0].cloud.point_count(), len(scenes)))
    if 'equal' not in method.lower():
        for i, s in enumerate(scenes):
            # compute geometric features
            geom = push_geomattr(s)
            if 'dist' in method.lower():
                weights[:, i] = geom.data[:, 0]
            elif 'gsd' in method.lower():
                weights[:, i] = geom.data[:, 1]
            elif 'obl' in method.lower():
                weights[:, i] = geom.data[:, 2]
            else:
                assert False, "Error - %s is an unknown weighting method" % method
        # invert and normalise
        mn, mx = np.nanpercentile(weights, (0, 100))
        weights = mx - weights  # lower values should get higher weight, so flip
    weights /= np.nansum(weights, axis=-1)[..., None]  # sum to 1

    if ascloud:
        weights = hylite.HyCloud(scenes[0].cloud.xyz, bands=np.nan_to_num(weights, nan=0, posinf=0))
        weights.set_band_names([s.name for s in scenes])
    return weights


def blend_scenes(scenes, weights, bands=(0, -1), chunksize=25, trim=False, ooc=True, vb=True):
    """
    Blend together collections of scenes that reference the same underlying cloud to create a fused
    hypercloud.

    Args:
        scenes: a list of scenes to fuse. Data will be extracted from scene.image and pushed to the cloud using
                scene.pmap.
        weights: a numpy array or HyCloud instance containings weighting factors with shape (points,scenes). See
                 get_blend_weights(...) for more details.
        bands: tuple specifying the (min,max) bands of scenes.image to export. Default is all bands (0,-1).
        chunksize: the number of bands to project in any given iteration. Larger numbers are faster but require
                   more RAM.
        trim: True if points with no data mapped to them should be removed.
        ooc: True if hypercloud bands should be created out-of-core and then assembled. This is slower but less RAM
             intensive for large hyperclouds. If this is True, scene.free() will also be called to unload hyperspectral images
             after they have been processed; so please save HyScenes before using!
        vb: True if progress bars should be created.

    Returns:
        a HyCloud instance contain
    """

    # check / format input
    if isinstance(weights, hylite.HyCloud):
        weights = weights.data
    weights /= np.sum(weights, axis=-1)[:, None]  # normalise

    assert len(scenes) == weights.shape[1], "Error: %d scenes != %d weights" % (len(scenes), weights.shape[1])
    assert scenes[0].cloud.point_count() == weights.shape[0], "Error: scene has %d points but weights have %d" % (
    scenes[0].cloud.point_count(), weights.shape[0])

    # make a temp directory
    tmp = mkdtemp()
    err = None

    try:  # wrap everything here in a try, so we can delete the temp folder if issues arise
        # gather data needed for projection
        bands = [scenes[0].image.get_band_index(b) for b in bands]  # convert bands to band indices
        if len(bands) == 2:
            bands = range(bands[0], bands[1] + 1)  # bands is a range of bands
        bands = sorted(set(bands))  # remove duplicates and sort
        wav = [scenes[0].image.get_wavelengths()[b] for b in bands]

        nbands = len(bands)
        mask = np.full(scenes[0].cloud.point_count(), False)  # these become True as valid data is added
        out = scenes[0].cloud.copy(data=False)  # create output cloud

        # loop through scenes and project bands
        for s in scenes:

            assert s.cloud.point_count() == mask.shape[0], "Error: scene %s has %d points, not %d" % (scene.name,
                                                                                                      scene.cloud.point_count(),
                                                                                                      mask.shape[0])
            # make output directory
            pth = os.path.join(tmp, s.name)
            os.makedirs(pth, exist_ok=True)

            # remove zeros
            s.image.set_as_nan(0)

            # push bands to disk
            loop = bands
            if vb:
                loop = tqdm(loop, desc='Projecting scene %s' % s.name, leave=False)
            chunk = []
            for i, b in enumerate(loop):
                chunk.append(b)  # append band to projection queue
                if (len(chunk) == chunksize) or (i == (len(loop) - 1)):
                    data = s.push_to_cloud(chunk).data  # project bands
                    for n in range(data.shape[-1]):  # save them as individual numpy files
                        mask = np.logical_or(mask, np.isfinite(data[:, n]))
                        np.save(os.path.join(pth, 'b%d.npy' % chunk[n]), data[:, n])
                    chunk = []
            if ooc:
                s.free()

        # create output array (n.b. this can be a biiiig malloc)
        out.data = np.zeros((out.point_count(), nbands), dtype=np.float32)

        # combine data
        loop = bands
        if vb:
            loop = tqdm(loop, desc='Blending bands', leave=False)
        for n, b in enumerate(loop):
            sm = np.zeros(mask.shape[0])
            for i, s in enumerate(scenes):
                pth = os.path.join(tmp, s.name)
                out.data[:, n] += np.nan_to_num(np.load(os.path.join(pth, 'b%d.npy' % b)) * weights[:, i])
    except KeyboardInterrupt as inst:
        print("Operation cancelled: cleaning up after KeyboardInterrupt.")
        err = inst
    except Exception as inst:
        print("An unexpected error occured: cleaning up before termination.")
        err = inst  # keep the error to throw later

    # delete temp directory
    shutil.rmtree(tmp)

    # if there was a problem, raise it now!
    if err is not None:
        raise err

    # cleanup point cloud (trim points with no data) and add wavelengths
    if trim:
        out.filter_points(0, np.logical_not(mask))
    out.set_wavelengths(wav)

    return out  # done!
