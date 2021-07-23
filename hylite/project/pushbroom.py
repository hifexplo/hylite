"""
Functions for projecting between pushbroom sensors (with known orientation/position from IMU data) and a 3D point cloud.
"""
import hylite
from hylite.project import rasterize, PMap, push_to_cloud
from scipy import spatial
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy as sp
from scipy.optimize import minimize


class Pushbroom(object):
    """
    An extension of the Camera class to handle pushbroom sensors.

    A note on coordinates: hylite always considers the x-axis of an image to be in the cross-track direction, and
    the y-axis to be in the along-track (movement) direction. This is consistent with most sensor manufacturers, but
    should not be confused with the coordinates used for panoramic sensors (where x = lines at each rotation, y=cross track
    direction). Issues plotting can be resolved using the rot, flipX and flipY arguments for image.quick_plot( ... ).
    """

    def __init__(self, pos, ori, xfov, lfov, dims ):
        """
        Initialise this pushbroom camera instance.

        *Arguments*:
         - pos = a list containing the position of the sensor at each frame.
         - ori = a list containing the orientation of the sensor (roll, pitch, yaw in degrees) at each frame.
         - xfov = the across-track field of view of one pixel (e.g. 0.01 degree).
         - lfov = the along-track field of view of one pixel (e.g. 0.01 degree).
         - dims = the dimensions of the resulting image. Should be (pixels_in_sensor, number_of_frames).
        """

        # store track and fov info
        self.cp = np.array(pos)  # camera position track
        self.co = np.array(ori)  # camera orientation track
        self.xfov = xfov
        self.lfov = lfov
        self.px = 2 * np.tan(0.5 * np.deg2rad(self.xfov))  # pixel pitch (across track)
        self.pl = 2 * np.tan(0.5 * np.deg2rad(self.lfov))  # pixel pitch (along track)

        # calculate and store rotation matrices
        self.R = [spatial.transform.Rotation.from_euler('xyz', [a[0] - 180, a[1], 90 - a[2]], degrees=True) for a in
                  self.co]
        # N.B. 90 degree rotation of yaw transforms to geographic coordinates (000 = North). 180 degree rotation
        # transforms camera direction from pointing up to down.

        # check and store dims
        assert dims[1] == len(pos), "Error - dims has %d lines, but only %d positions provided." % (dims[1], len(pos))
        assert dims[1] == len(ori), "Error - dims has %d lines, but only %d orientations provided." % (
        dims[1], len(pos))
        self.dims = dims

    def fudge(self, t_all, t_known, cp_known, co_known, method='quadratic'):
        """
        Apply a fudge-factor to fit IMU data to sparse points with known positions / orientations (e.g.
        from SfM results using a co-aligned camera). This (1) matches the two position datasets using the
        two time values (these should be floating point values like GPS seconds), (2) calculates the residual
        at these points, (3) interpolates these residuals between the known points and (4) subtracts them from
        the IMU track.

        *Arguments*:
         - t_all = timestamps associated with the orientation and position data in this pushbroom track.
         - t_known = an array of timestamps of shape (n,) associated with the known positions/orientations.
         - cp_known = an array of shape (n,3) containing known positions to fudge to.
         - co_known = an array of of shape (n,3) containing known orientations to fudge to.
         - method = the interpolation method to use, as defined in scipy.interpolate.interp1d. Default is 'quadratic'.

        *Returns*:
         - a fudged copy of this track.
        """

        # check shapes
        assert t_all.shape[0] == self.co.shape[0], "Error - timestamps t_all has %d entries, but %d are needed." % (t_all.shape[0], self.co.shape[0])
        assert t_known.shape[0] == cp_known.shape[0], "Error - timestamp shape (%d) != cp_known shape (%d)." % (t_known.shape[0], cp_known.shape[0])
        assert t_known.shape[0] == co_known.shape[0], "Error - timestamp shape (%d) != co_known shape (%d)." % (t_known.shape[0], co_known.shape[0])

        # get indices of main track that correspond to known frames
        S = [np.argmin(np.abs(t_all - _t)) for _t in t_known]
        assert len(np.unique(S)) == len(S), "Error - duplicate matches are present; tracks probably do not overlap properly?"

        # compute errors
        e_p = cp_known - self.cp[S, :] # error in position
        e_o = co_known - self.co[S, :] # error in orientation

        # apply fudge to remove these (and interpolate fudge factor between known points)
        cp_adj = self.cp + sp.interpolate.interp1d(t_known, e_p, bounds_error=False, axis=0, kind='quadratic', fill_value=0)(t_all)
        co_adj = self.co + sp.interpolate.interp1d(t_known, e_o, bounds_error=False, axis=0, kind='quadratic', fill_value=0)(t_all)

        # return a copy of this track
        return Pushbroom( cp_adj, co_adj, self.xfov, self.lfov, self.dims )

    def rebuild_R(self):
        """
        Rebuild the internal rotation matrix from the self.co arrays.
        """
        self.R = [spatial.transform.Rotation.from_euler('xyz', [a[0] - 180, a[1], 90 - a[2]], degrees=True) for a in
                  self.co]

    def apply_boresight(self, roll, pitch, yaw ):
        """
        Add constant values (boresight) to the roll, pitch and yaw angles.

        *Returns*: a new Pushbroom instance with the adjusted values.
        """
        # appy boresight
        co_adj = self.co + np.array( [roll, pitch, yaw ] )
        return Pushbroom( self.cp, co_adj, self.xfov, self.lfov, self.dims )

    def get_R(self, i=None):
        """
        Return scipy.spatial.Rotation objects that store camera orientation data.

        *Arguments*:
         i = the frame index to get a rotation object for. If None (Default) a list of all Rotations is returned.
        """
        if i is not None:
            return self.R[i]
        else:
            return self.R

    def get_axis(self, axis, i=None):
        """
        Get the local camera x (0), y (1) or z (2) axis vector from the rotation matrices.

        *Arguments*:
         - axis = 0 (x), 1 (y) or 2 (z).
         - i = the frame index to get vector for. If None (Default) a list of all vectors is returned.
        """
        if i is not None:
            assert i >= 0 and i < 3, "Error - i must be 0, 1 or 2."
            return self.R[i].as_matrix()[:, axis]
        else:
            return np.array([_R.as_matrix()[:, axis] for _R in self.R])

    def get_x(self, i=None):
        """
        Get the Camera's along-track (movement) vector.

        *Arguments*:
         i = the frame index to get vector for. If None (Default) a list of all vectors is returned.
        """
        return self.get_axis(0, i)

    def get_y(self, i=None):
        """
        Get the Camera's cross-track vector.

        *Arguments*:
         i = the frame index to get vector for. If None (Default) a list of all vectors is returned.
        """
        return self.get_axis(1, i)

    def get_z(self, i=None):
        """
        Get the Camera's view vector.

        *Arguments*:
         i = the frame index to get vector for. If None (Default) a list of all vectors is returned.
        """
        return self.get_axis(2, i)

    def project_to_frame(self, cloud, i, flip=True):
        """
        Project the point cloud onto an (instantaneous) frame from this pushbroom camera.

        *Arguments*:
         - cloud = the point cloud to project. Must have points stored in cloud.xyz (HyCloud) or be a (n,3) array.
         - i = the camera frame (along track index) to use.
         - flip = True if pixel coordinates should be flipped.
        *Returns*:
         - a (3,) array of projected points with coordinates that are:
             0. xtrack = the position (in pixels) of the points across track (pixels on the sensor)
             1. ltrack = the position (in pixels) of the points along track. Values between 0 and 1 will pass through the sensor slit.
             2. depth = the depth along the view direction of each point (in meters).
        """

        # project into camera coordinates
        if isinstance(cloud, np.ndarray):
            xyz = np.dot(cloud - self.cp[i], self.R[i].as_matrix())
        else:
            xyz = np.dot(cloud.xyz - self.cp[i], self.R[i].as_matrix())

        # calculate along-track coordinate (perspective projection perpendicular to flight line)
        ltrack = 0.5 + (
                    xyz[:, 0] / xyz[:, 2]) / self.pl  # N.B. +0.5 moves from pixel-center coords to pixel-edge coords

        # calculate across-track coordinate (perspective projection onto sensor array)
        if flip:
            xtrack = (self.dims[0] / 2) - ((xyz[:, 1] / xyz[:, 2]) / self.px)
        else:
            xtrack = (self.dims[0] / 2) + ((xyz[:, 1] / xyz[:, 2]) / self.px)

        return np.array([xtrack, ltrack, xyz[:, 2]]).T

    def crop(self, start, end, image=None):
        """
        Clip this tracks IMU data to the specified frames. Note that this is not reversable.

        *Arguments*:
         - start = the start frame.
         - end = the end frame.
         - image = a HyImage instance to clip also, if provided (default is None). Clipping will
                   be applied to the y-direction.
        """
        assert self.co.shape[0] > end and start > 0, "Error - invalid range %d - %d (data shape is %s)" % (start,end,self.co.shape)
        assert start < end, "Error - start (%d) cannot be after end (%d)." % (start,end)
        self.co = self.co[start:end, :]
        self.cp = self.cp[start:end, :]
        self.R = self.R[start:end]
        self.dims = (self.dims[0], self.co.shape[0] )
        if image is not None:
            image.data = image.data[ :, start:end, : ]

    def plot_waterfall(self, image = None, bands=(0,1,2), flipY=True, flipX=False, rot=True ):
        """
        Plot IMU data and (if provided) the raw image frames.

        *Arguments*:
         - image = the waterfall image to plot. Default is None (no plot).
         - bands = the bands of the waterfall image to plot. Default is (0,1,2).
         - flipY = flip the Y axis of the image. Default is True.
         - flipX = flip the X axis of the image. Default is False.
         - rot = rotate the image by 90 degrees. Default is True.
        *Returns*:
         - fig,ax = the figure and a list of associated axes.
        """
        if image is None:
            fig, ax = plt.subplots(6, 1, figsize=(18, 12))
            ax = [ax[0],ax[0],ax[1],ax[2],ax[3],ax[4],ax[5]]
        else:
            fig, ax = plt.subplots(7, 1, figsize=(18, 8))
            image.quick_plot(bands, ax=ax[0], rot=rot, flipY=flipY, flipX=flipX)
            ax[0].set_aspect('auto')  # override aspect equal, as this is probably wrong anywa
            ax[0].set_title("Waterfall")
            ax[0].set_yticks([])
        # plot IMU data
        for i,(y,l,c) in enumerate(zip(
            [self.cp[:,0], self.cp[:,1], self.cp[:,2], self.co[:,0], self.co[:,1], self.co[:,2]],
            ['X','Y','Z','Roll','Pitch','Yaw'],
            ['r','g','b','r','g','b'])):

            ax[i+1].set_ylabel(l)
            ax[i+1].plot(y,color=c)

        # clean up axes
        for a in ax:
            a.set_xticks(range(0, self.co.shape[0], 250))
            a.set_xlim(0, self.co.shape[0])
            a.grid()
        [ax[i].set_xticklabels([]) for i in range(6)]
        fig.tight_layout()
        return fig, ax

    def plot_curtain(self, scale=10, alpha=0.1, ax=None, ortho=None):
        """
        Create a curtain plot for visualising this camera track.

        *Arguments*:
         - scale = the scale of the camera basis vectors (curtain). This is in transformed coordinates.
         - alpha = the alpha value to use for the curtain. Default is 0.1.
         - ax = an external axis to plot too.
         - ortho = an orthoimage to use to project to pixel coordinates. Image must have a defined affine transform.
        """

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            ax.set_aspect('equal')

        # get camera basis
        X = self.get_x()
        Y = self.get_y()
        Z = self.get_z()

        # build curtain
        S = []  # store camera pos for plotting
        XV = []
        YV = []
        ZV = []

        for i in range(len(self.cp)):

            # get camera position at this frame
            px = self.cp[i, 0]
            py = self.cp[i, 1]

            if ortho is not None:
                assert ortho.affine is not None, "Error - image must have an affine transform set."
                px, py = ortho.world_to_pix(px, py)

            S.append([px, py])

            if ortho is None:
                ZV += [[px, px + Z[i, 0] * scale], [py, py + Z[i, 1] * scale]]
                YV += [[px, px + Y[i, 0] * scale], [py, py + Y[i, 1] * scale]]
                XV += [[px, px + X[i, 0] * scale], [py, py + X[i, 1] * scale]]
            else:
                # we need to flip the y-axis as imshow puts the origin in the top-left.
                ZV += [[px, px + Z[i, 0] * scale], [py, py - Z[i, 1] * scale]]
                YV += [[px, px + Y[i, 0] * scale], [py, py - Y[i, 1] * scale]]
                XV += [[px, px + X[i, 0] * scale], [py, py - X[i, 1] * scale]]

        ax.plot(*ZV, color='r', lw=1, alpha=alpha, zorder=3)
        ax.plot(*YV, color='g', lw=1, alpha=alpha, zorder=2)
        ax.plot(*XV, color='b', lw=1, alpha=alpha, zorder=1)

        # plot camera track
        S = np.array(S)
        ax.cmap = ax.scatter(S[:, 0], S[:, 1], c=np.arange(0, S.shape[0]), cmap='jet', zorder=4)  # also store colorbar
        return ax.get_figure(), ax

    def plot_pose(self, i):
        """
        Plot the camera axes and scan line in world coordinates. Useful for checking camera orientation at a
        given frame (and debugging!).

        *Arguments*:
         - i = plot the i'th camera.
        *Returns*:
         - fig,ax = the matplotlib plot.
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import axes3d
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        # get local axes from rotation matrix
        M = self.get_R(i).as_matrix()
        x = M[:, 0]
        y = M[:, 1]
        z = M[:, 2]

        # plot them in 3D
        fig = plt.figure(figsize=(10, 10))
        ax = fig.gca(projection='3d')

        ax.quiver([0], [0], [0],
                  [x[0]], [x[1]], [x[2]],
                  length=1.0, normalize=False, colors='b', label='x (flight direction)')
        ax.quiver([0], [0], [0],
                  [y[0]], [y[1]], [y[2]],
                  length=1.0, normalize=False, colors='g', label='y (pixel direction)')
        ax.quiver([0], [0], [0],
                  [z[0]], [z[1]], [z[2]],
                  length=1.0, normalize=False, colors='r', label='z (view direction)')

        # add plot showing scan line
        V0 = np.array([0, 0, 0])
        V1 = z - 0.4 * y
        V2 = z + 0.4 * y
        xx = [V0[0], V1[0], V2[0], V0[0]]
        yy = [V0[1], V1[1], V2[1], V0[1]]
        zz = [V0[2], V1[2], V2[2], V0[2]]
        verts = [list(zip(xx, yy, zz))]
        ax.add_collection3d(Poly3DCollection(verts, alpha=0.2))

        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)

        ax.set_xlabel("Easting")
        ax.set_ylabel("Northing")
        ax.set_zlabel("Elevation")

        ax.legend()

        return fig, ax

    def plot_strip(self, line, width=100, cloud=None, image=None, s=2, aspect='auto'):
        """
        Plot a projected strip for comparision between an image and a point cloud.

        *Arguments*:
         - line = the line ID to plot.
         - width = how many pixels to plot either side of the line. Default is 10.
         - cloud = the cloud to plot. Can be None.
         - image = the (linescanner) image to plot. Can be None.
         - s = size of rendered points in pixels. Default is 2.
         - aspect = the aspect ratio of the renders / image plot. Default is 'equal'. Change to
                    'auto' to stretch data to figure size.
        """

        # build plot
        if cloud is None and image is None:
            assert False, "At least one dataset (cloud or image) must be passed for plotting."
        elif cloud is None or image is None:
            fig, ax = plt.subplots(1, 1, figsize=(15, 5))
            iax = cax = ax
        else:
            fig, ax = plt.subplots(2, 1, figsize=(15, 5))
            cax = ax[0]
            iax = ax[1]

        # plot cloud
        if cloud is not None:
            # project onto frame
            pp = self.project_to_frame(cloud, line)
            pp[:, 1] += width / 2  # offset origin to left side of image

            # calculate point visibility (including points within width)
            vis = (pp[:, 0] > 0) & (pp[:, 0] < self.dims[0]) & (pp[:, 1] > -(width / 2)) & (pp[:, 1] < width / 2)
            # pp[:, 0] += width / 2 # offset so we see either side of scan line

            # calculate point visibility (including points within width)
            vis = (pp[:, 0] > 0) & (pp[:, 0] < self.dims[0]) & (pp[:, 1] > 0) & (pp[:, 1] < width)

            # rasterise
            grd, z = rasterize(pp, vis, cloud.rgb, dims=(self.dims[0], width), s=s)

            # plot
            cax.imshow(np.transpose(grd, (1, 0, 2)) / 255.)
            cax.set_yticks([width / 2])
            cax.set_yticklabels(["line %d" % line])
            cax.set_aspect(aspect)
            cax.set_xticks([])
            cax.set_title("Point cloud RGB")

        if image is not None:
            image.quick_plot((0, 1, 2), ax=ax[1])
            iax.set_ylim(line - width, line + width)
            iax.set_yticks([line])
            iax.set_yticklabels(["line %d" % line])
            iax.set_aspect(aspect)
            iax.set_title("Scanner frame")

        fig.tight_layout()
        fig.show()


def project_pushbroom(image, cloud, cam, chunk=500, step=100, near_clip=10., vb=True):
    """
    Map an image acquired using a moving pushbroom scanner onto a point cloud using known
    camera position and orientations for each line of the image.

    *Arguments*:
     - image = a HyImage instance containing the data to project. This is only used to determine image dimensions.
     - cloud = the destination point cloud to project data onto.
     - cam = a pushbroom camera instance.
     - chunk = The size of chunks used to optimise the projection step. Points are culled based on the first and
               last line of each chunk prior to processing to reduce the number of projections that need to be
               performed. To reduce errors at chunk margins these chunks are padded by 50%. Default is 500.
     - step = the step to use in the masking step for each chunk. Default is 100. Reducing this will ensure no points
              are missed, but at large performance cost.
     - near_clip = the z-depth of the near clipping plane. Default is 10.0 m.
     - vb = True if a progress bar should be displayed. Default is True.


    *Returns*:
     - a dictionary containing (xpixel,ystart,yend) tuples that define the range of pixels this point was projected
       onto during acquisition. Two y-values are provided as point can be projected onto multiple pixels in the
       cross-track direction during a single frame of acquisition (due to sensor movement).
    """

    # check dims match
    assert image.xdim() == cam.dims[0] \
           and image.ydim() == cam.dims[1], \
        "Error - image and camera dimensions do not match. Try rotating the image using image.rot90()."

    # store point IDs (so these don't get ruined by clipping)
    pointIDs = np.arange(0, cloud.point_count(), dtype=np.uint32)

    # build an adjacency matrix in flat form
    points = []
    pixels = []
    depths = []

    # loop through each chunk
    for c in range(0, image.ydim(), chunk):
        #######################################################
        ## Clip point cloud to chunk using coarse projections
        #######################################################
        # calculate start of frame projection and facing
        P0 = cam.project_to_frame(cloud, c)
        F0 = P0[:, 1] > 0  # true for points in front of scan line
        mask = np.full(cloud.point_count(), False)  # mask of points visible in chunk
        loop = range(c + step, c + chunk + step, step)
        if vb:
            loop = tqdm(loop, leave=False, desc="Masking chunk %d/%d" % (c / chunk + 1, image.ydim() / chunk))
        for i in loop:
            if i >= image.ydim():  # reached end of image
                continue  # use continue here rather than break to use up the rest of the progress bar so it is removed

            P1 = cam.project_to_frame(cloud, i)
            F1 = P1[:, 1] > 0  # true for points in front of scan line
            vis = (F0 != F1)  # points visible in this frame
            vis = vis & np.logical_or((P0[:, 0] > 0) & (P0[:, 0] < cam.dims[0]),
                                      (P1[:, 0] > 0) & (P1[:, 0] < cam.dims[0]))
            mask[vis] = True  # these points should be included in final projection
            F0 = F1
            P0 = P1

        # subset points to those visible in this chunk (only)
        xyz = cloud.xyz[mask]  # get points
        pIDs = pointIDs[mask]  # store IDs in the original point cloud

        #######################################################
        ## Do per-line projections for subset point cloud
        #######################################################
        # calculate start of frame projection and facing (with subsetted cloud)
        P0 = cam.project_to_frame(xyz, c)
        F0 = P0[:, 1] > 0  # true for points in front of scan line

        loop = range(c + 1, c + chunk + 1)
        if vb:
            loop = tqdm(loop, leave=False,
                        desc="Projecting chunk %d/%d (%d points)" % (c / chunk + 1, image.ydim() / chunk, xyz.shape[0]))
        for i in loop:
            if i >= image.ydim():  # reached end of image
                continue  # use continue here rather than break to use up the rest of the progress bar so it is removed

            # calculate end of frame projection and facing
            P1 = cam.project_to_frame(xyz, i)
            F1 = P1[:, 1] > 0  # true for points in front of scan line

            # get points crossed by pushbroom
            vis = (F0 != F1)  # N.B. this culls most of the points in the cloud!

            # remove points out of field of view (but co-planar with line)
            vis = vis & np.logical_or((P0[:, 0] > 0) & (P0[:, 0] < cam.dims[0]),
                                      (P1[:, 0] > 0) & (P1[:, 0] < cam.dims[0])) \
                  & np.logical_or(P0[:, 2] > near_clip, P1[:, 2] > near_clip)

            # cull invisible points
            _PP = np.vstack([np.clip(P0[vis, 0], 0, image.xdim() - 1),
                             np.clip(P1[vis, 0], 0, image.xdim() - 1)]).T  # cast to indices
            _PP = np.sort(_PP, axis=1).astype(np.uint)  # sort so _PP[i,0]:_pp[i,1] gives the range of indices.
            _PP[:, 1] += 1  # add one to final index for slicing
            _Z = np.max([P0[vis, 2], P1[vis, 2]], axis=0)
            for n, pid in enumerate(pIDs[vis]):
                # add to link matrix
                px = range(int(i * image.xdim() + _PP[n, 0]), int(i * image.xdim() + _PP[n, 1]))
                pixels += px
                points += [pid] * len(px)
                depths += [_Z[n]] * len(px)

            # update left clipping plane
            F0 = F1
            P0 = P1

    # build a projection map and return it
    pmap = PMap(image.xdim(), image.ydim(), cloud.point_count(), cloud=cloud, image=image)
    pmap.set_flat(points, pixels, 1. / np.array(depths))

    return pmap

def get_corr_coef(pmap, bands=(0, 1, 2)):
    """
    Calculate the pearsons correlation coefficient between projected RGB colours (pmap.image[...,bands])
    and pmap.cloud.rgb.  Used to optimise boresight values against point
    cloud colours (slow, but worthwhile).

    *Arguments*:
     - pmap = the pmap instance containing the projection.
     - bands = indices for the red, green and blue bands of pmap.image. Default is (0,1,2).

    *Returns*:
     - the corellation of the red, green and blue bands.
    """
    assert pmap.cloud.rgb is not None, "Error - cloud must contain independent RGB information."
    rgb_ref = pmap.cloud.rgb  # 'true' RGB from SfM
    rgb_adj = push_to_cloud(pmap, bands, method='count').data  # calculate projected RGB

    # filter to known points
    mask = np.isfinite(rgb_adj.data).all(axis=-1)

    # return the corellation of the red, green and blue bands.
    return [sp.stats.pearsonr(rgb_ref[mask, i], rgb_adj[mask, i])[0] for i in range(3)]


def optimize_boresight(track, cloud, image, bands=(0, 1, 2), n=100, iv=np.array([0, 0, 0]), scale=3.,
                       coarse_eps=0.5, fine_eps=0.05, vb=True, gf=True):
    """
    Applies a least-squares solver to find boresight values that result in the best correlation
    between projected RGB and RGB stored on a photogrammetric point cloud. This can be very slow,
    so use the subsample argument to significantly subsample the point cloud. It is also a good idea
    to calculate these values using a small but high-quality subset of a swath; the optimised values
    can subsequently be applied to the whole dataset

    *Arguments*:
     - cloud = the cloud containing the geometry to project onto and associated RGB values.
     - image = the image containing RGB data.
     - n = the subsampling factor for the pointcloud (uses cloud.xyz[::n,:]). Default is 100.
     - iv = the initial value for the boresight estimation. Default is [0,0,0].
     - bands = indices for the red, green and blue bands of image. Default is (0,1,2).
     - scale = The size of the search space. Default is ±3 degrees from the initial value.
     - coarse_eps = the perturbation step size for the first solution.
     - fine_eps = the perturbation step size for subsequent refinement. Set to 0 to disable.
     - vb = True if output of the least-squares solver should be printed at each state.
     - gf = True if a plot should be generated showing the search trace. Default is True
    *Returns*:
     - track = an updated track with the boresight applied.
     - boresight = a (3,) array with optimised boresight adjustments (roll, pitch, yaw).
     - trace = the search path of the optimisation, with shape (niter,4). The last column contains
                the cost function at each iteration point.
    """

    # check cloud and build subsampled copy
    assert cloud.rgb is not None, "Error - cloud must contain independent RGB information."
    cloud = hylite.HyCloud(cloud.xyz[::n, :], rgb=cloud.rgb[::n, :])

    # setup cost function
    trace = []

    def _opt(X, track, cloud, image):
        # update track
        track = track.apply_boresight(*X)

        # build projection map
        pmap = project_pushbroom(image, cloud, track, chunk=200, step=50, vb=False)

        # return cost
        c = -np.sum(get_corr_coef(pmap, bands=bands))
        trace.append((X[0], X[1], X[2], -c))  # add to trace
        if vb:
            print("\r%d: boresight = [%.3f,%.3f,%.3f]; correlation=%.4f." % (len(trace), X[0], X[1], X[2], -c / 3),
                  end='  ')

        return c

    # calculate bounds
    bounds = [(v - scale, v + scale) for v in iv]

    # run coarse optimisation
    res = minimize(_opt, np.array(iv), args=(track, cloud, image),
                   bounds=bounds, method='TNC', options=dict(eps=coarse_eps))
    if vb:
        print("Complete [%s]." % res.message)

    tr = np.array(trace)
    s = np.argmax(
        tr[:, -1])  # get best value from iterations (with TNC this isn't always the optimal solution for some reason?)

    # run fine optimisation
    if fine_eps > 0:
        if vb:
            print("Running fine adjustment...")
        bounds = [(v - scale, v + scale) for v in iv]
        res = minimize(_opt, tr[s, :3], args=(track, cloud, image),
                       bounds=bounds, method='TNC', options=dict(eps=fine_eps))
        if vb:
            print("Complete [%s]." % res.message)
        tr = np.array(trace)

    tr[:, -1] /= 3  # convert sum of corellation coeffs to average (is more interpretable)
    s = np.argmax(
        tr[:, -1])  # get best value from iterations (with TNC this isn't always the optimal solution for some reason?)

    # plot?
    if gf:
        plt.plot(tr[:, 0], tr[:, 3], color='r', label='Roll')
        plt.plot(tr[:, 1], tr[:, 3], color='g', label='Pitch')
        plt.plot(tr[:, 2], tr[:, 3], color='b', label='Yaw')
        plt.scatter(tr[s, 0], tr[s, 3], color='k', zorder=10)
        plt.scatter(tr[s, 1], tr[s, 3], color='k', zorder=10)
        plt.scatter(tr[s, 2], tr[s, 3], color='k', zorder=10)
        plt.title("Iter %d: boresight = [%.3f,%.3f,%.3f]" % (tr.shape[0], tr[s, 0], tr[s, 1], tr[s, 2]))
        plt.legend()
        plt.xlabel('Value')
        plt.ylabel('Correlation')
        plt.show()

    # apply to track
    return track.apply_boresight(*tr[s, :3]), tr[s, :3], tr