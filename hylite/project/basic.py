import numpy as np
from scipy import spatial
import cv2

def proj_persp( xyz, C, a, fov, dims, normals=None):
    """
    Project 3d point xyz based on a pinhole camera model.

    *Arguments*:
     - xyz = a nx3 numpy array with a position vector (x,y,z) in each column.
     - C = the position of the camera.
     - a = the three camera rotatation euler angles (appled around x, then y, then z == pitch, roll, yaw). In DEGREES.
     - fov = the vertical field of view.
     - dims = the image dimensions (width, height)
     - normals = per-point normals. Used to do backface culling if specified. Default is None (not used).
    """

    #transform origin to camera position
    xyz = xyz - C[None,:]

    # backface culling
    vis = np.full(xyz.shape[0],True,dtype=np.bool)
    if not normals is None:
        ndv = np.einsum('ij, ij->i', normals, xyz) #calculate dot product between each normal and vector from camera to point
                                                   #(which is the point position vector in this coordinate system)
        vis = ndv < 0 #normal is pointing towards the camera (in opposite direction to view)

    #apply camera rotation to get to coordinate system
    #  where y is camera up and z distance along the view axis
    R = spatial.transform.Rotation.from_euler('XYZ',-a,degrees=True).as_matrix()
    xyz = np.dot(xyz, R)

    #calculate image plane width/height in projected coords
    h = 2 * np.tan( np.deg2rad( fov / 2 ) )
    w = h * dims[0] / float(dims[1])

    #do project and re-map to image coordinates such that (0,0) = lower left, (1,1) = top right)
    px = ((w/2.0) + (xyz[:,0] / -xyz[:,2])) / w
    py = ((h/2.0) + (xyz[:,1] / xyz[:,2])) / h
    pz = -xyz[:,2] #return positive depths for convenience

    #calculate mask showing 'visible' points based on image dimensions
    vis = vis & (pz > 0) & (px > 0) & (px < 1) & (py > 0) & (py < 1)

    #convert to pixels
    px *= dims[0]
    py *= dims[1]

    return np.array([px,py,pz]).T, vis

def proj_pano(xyz, C, a, fov, dims, step=None, normals=None):
    """
    Project 3d point xyz based on a pinhole camera model.

    *Arguments*:
     - xyz = a nx3 numpy array with a position vector (x,y,z) in each column.
     - C = the position of the camera.
     - a = the three camera rotatation euler angles (appled around x, then y, then z == pitch, roll, yaw). In DEGREES.
     - fov = the vertical field of view.
     - dims = the image dimensions (width, height)
     - step = the angular step (degrees) between pixels in x.  If None, this is calculated to ensure square pixels.
     - normals = per-point normals. Used to do backface culling if specified. Default is None (not used).
    """

    # transform origin to camera position
    xyz = xyz - C[None, :]

    # backface culling
    vis = np.full(xyz.shape[0], True, dtype=np.bool)
    if not normals is None:
        ndv = np.einsum('ij, ij->i', normals,
                        xyz)  # calculate dot product between each normal and vector from camera to point
        # (which is the point position vector in this coordinate system)
        vis = ndv < 0  # normal is pointing towards the camera (in opposite direction to view)

    # apply camera rotation to get to coordinate system
    #  where y is camera up and z distance along the view axis
    R = spatial.transform.Rotation.from_euler('XYZ', -a, degrees=True).as_matrix()
    xyz = np.dot(xyz, R)

    # calculate image height (in projected coords) for vertical perspective project
    h = 2 * np.tan(np.deg2rad(fov / 2))

    # calculate position in spherical coords
    pz = np.sqrt(xyz[:, 0] ** 2 + xyz[:, 2] ** 2)  # distance from axis of rotation (world units)
    py = ((h / 2.0) + (xyz[:, 1] / -pz)) / h  # perspective project vertically (0 - 1; pin-hole camera model)
    px = np.arctan2(xyz[:, 0], -xyz[:, 2])  # rotation left/right of view axis (in radians)

    # convert x to pixels (slightly tricky)
    if step is None:
        step = fov / dims[1]  # angular pixel size in y = angular pixel size in x, assuming pixels are square...
    px = px / np.deg2rad(step) + (dims[0] / 2)

    # convert y to pixels (easy)
    py *= dims[1]

    # calculate mask showing 'visible' points based on image dimensions
    vis = vis & (px > 0) & (px < dims[0]) & (py > 0) & (py < dims[1])

    return np.array([px, py, pz]).T, vis

def proj_ortho( xyz, C, V, s=1.0 ):
    """
    Project points onto a plane (orthographic projection).

    *Arguments*:
    - xyz = a nx3 numpy array with a position vector (x,y,z) in each column.
    - C = the position of the viewing plane origin (will become pixel 0,0).
    - V = the [x,y,z] viewing/projection vector (normal to the projection plane).
    - s = the scale factor to transform world coordinates into image coordinates. Default is 1.0 (keep world coords).
    *Returns*:
     - px,py,pz = projected point coordinates (x,y,depth)
     - vis = array containing True if each point is visible and False for points behind the viewing plane.
    """

    xyz = xyz - C[None, :]  # center origin on camera
    pz = np.dot(xyz, V)  # calculate depths (distances from plane)
    xyz -= (V[None, :] * pz[:, None])  # project onto plane (by removing depth)
    return s*np.array([ xyz[:,0], xyz[:,1], pz]).T, pz > 0

def rasterize(points, vis, vals, dims, s=1):
    """
    Rasterizes projected points onto an image grid.

    *Arguments*:
     - points = the points (px,py,depth) to rasterise,
                as returned by proj_persp or proj_pano.
     - vis = a boolean array that is True if the points are visible (as returned by
                proj_persp or proj_pano).
     - vals = Nxm array of (m) additional point values (e.g. position, normals, colour) to add to rasterize. Alternatively
              a list of Nxm numpy arrays can be passed (these will be concatentated).
     - dims = the size of the image grid.
     - s = the size of the points in pixels. Default is 1.
    *Returns*:
     - raster = rasterised image with m bands corresponding to vals.
     - depth = the depth buffer.
    """

    #vals is a list of numpy arrays - stack them
    if isinstance(vals, list):
        if len(vals[0].shape) == 1: #list of 1-d arrays; need to v-stack
            vals = np.vstack(vals).T
        else:                       #list of n-d arrays; need to h-stack
            vals = np.hstack(vals)

    #shoulder case - vals is a 1D numpy array, so we need to add second axis
    if len(np.array(vals).shape) == 1:
        vals = vals[None, :].T

    # create image arrays
    out = np.full((dims[0], dims[1], vals.shape[1]), np.nan)
    depth = np.full((dims[0], dims[1]), np.inf)

    # cull invisible points
    points = points[vis, :]
    vals = vals[vis, :]

    # loop through points
    assert points.shape[1] == 3, "Error - points array should have shape (N,3)."
    assert isinstance(s, int), "Error - size (s) must be an integer."
    if s > 0: s -= 1 #reduce s by one due to how numpy does indexing.
    for i, p in enumerate(points):
        x = int(p[0])
        y = int(p[1])
        z = p[2]

        # double check point in the image
        if x < 0 or x >= dims[0] or y < 0 or y > dims[1]:
            continue  # skip

        if z < depth[x, y]:  # in front of depth buffer?
            out[(x - s):(x + s + 1), (y - s):(y + s + 1), :] = vals[None, None, i]
            depth[(x - s):(x + s + 1), (y - s):(y + s + 1)] = z

    return out, depth

def pix_to_ray_persp(x, y, fov, dims):
    """
    Transform pixel coordinates to a unit direction (ray) in camera coordinates using a
    perspective pinhole camera model.

    *Arguments*:
     - x = the pixel x-coordinate. Cannot be array (only works for single pixels).
     - y = the pixel y-coordinate. Cannot be array (only works for single pixels).
     - fov = the camera's vertical field of view (degrees)
     - dims = the dimensions of the image the pixels are contained in.

    *Returns*:
     - numpy array with the components of the light ray in camera coordinates (z == view direction, y == up)
    """

    aspx = dims[0] / float(dims[1])
    h = 2 * np.tan(np.deg2rad(fov / 2))
    Px = (2 * (x / dims[0]) - 1) * h / 2 * aspx
    Py = (2 * (y / dims[1]) - 1) * h / 2
    ray = np.array([Px, -Py, -1])
    return ray / np.linalg.norm(ray)  # return normalized ray

def pix_to_ray_pano(x, y, fov, step, dims):
    """
    Transform pixel coordinates to a unit direction (ray) in camera coordinates using a
    panoramic camera model.

    *Arguments*:
     - px = the pixel x-coordinate. Cannot be array (only works for single pixels).
     - py = the pixel y-coordinate. Cannot be array (only works for single pixels).
     - fov = the camera's vertical field of view (degrees)
     - step = the angular step between pixels in the x-direction.
     - dims = the dimensions of the image the pixels are contained in.

    *Returns*:
     - numpy array with the components of the light ray in camera coordinates (z == view direction, y == up)
    """

    # calculate image plane properties
    h = 2 * np.tan(np.deg2rad(fov / 2))

    # calculate ray in y-z plane
    Py = (2 * (y / dims[1]) - 1) * h / 2  # transform y pixels to camera coords
    ray = np.array([0, -Py, -1])  # construct ray vector
    ray = ray / np.linalg.norm(ray)

    # rotate it around y based on x
    alpha = (x - (dims[0] / 2.0)) * step  # map to angle coords (-180 to 180 degrees)
    R = spatial.transform.Rotation.from_euler('Y', alpha, degrees=True).as_matrix()

    return np.dot(ray, R)  # return rotated ray

def pano_to_persp(x, y, fov, step, dims):
    """
    Transforms pixels in a panoramic image to pixel coordinates as though they had been captured using a
    perspective camera with the same dimensions.

    *Arguments*:
     - x = array of pixel x-coordinates. These will be projected onto the image plane.
     - y = array of pixel y-coordinate.
     - fov = the camera's vertical field of view (degrees)
     - step = the angular step between pixels in the x-direction.
     - dims = the dimensions of the image the pixels are contained in.

    *Returns*:
     - px, py = numpy array with the projected pixel coordinates.
    """
    # are px and py lists?
    if isinstance(x, np.ndarray) or isinstance(x, list):

        # do housekeeping to make sure arrays/lists are proper and good :)
        assert (isinstance(y, np.ndarray) or isinstance(y, list)), "Error both x and y must be lists or arrays."
        if isinstance(x, np.ndarray):
            x = x.ravel()
            y = y.ravel()

        assert len(x) == len(y), "Error, px and py must have identical dimensions."

        out = []
        for i in range(len(x)):
            out.append(pano_to_persp(x[i], y[i], fov, step, dims).ravel())
        return np.array(out)

    else:  # no - individual pixels :)

        # convert pixel to ray
        xyz = pix_to_ray_pano(x, y, fov, step, dims)

        # project ray to pixel using pinhole model
        pp, vis = proj_persp(xyz, C=np.array([0, 0, 0]),
                             a=np.array([0, 0, 0]),
                             fov=fov, dims=dims)

        return pp[:, 0:2]  # return pixel coords

def pnp(kxyz, kxy, fov, dims, ransac=True, **kwds):
    """
    Solves the pnp problem to locate a camera given keypoints that are known in world and pixel coordinates using opencv.

    *Arguments*:
     - kxyz = Nx3 array of keypoint positions in world coordinates.
     - kxy = Nx2 array of corresponding keypoint positions in pixel coordinates.
     - fov = the (vertical) field of view of the camera.
     - dims = the dimensions of the image in pixels
     - ransac = true if ransac should be used to filter outliers. Default is True.

    *Keywords*:
     - optical_centre = the pixel coordinates (cx,cy) of the optical centre to use. By default the
         middle pixel of the image is used.
     - other keyword arguments are passed to cv2.solvePnP(...) or cv2.solvePnPRansac(...).

       Additionally a custom optical center can be passed as (cx,cy)
    *Returns*:
     - p = the camera position in world coordinates
     - r = the camera orientation (as XYZ euler angle).
     - inl = list of Ransac inlier indices used to estimate the position, or None if ransac == False.
    """
    # normalize keypoints so that origin is at mean
    mean = np.mean(kxyz, axis=0)
    kxyz = kxyz - mean

    # flip kxy coords to match opencv coord system
    # kxy[:, 1] = dims[1] - kxy[:, 1]
    # kxy[:, 0] = dims[0] - kxy[:, 0]

    # compute camera matrix
    tanfov = np.tan(np.deg2rad(fov / 2))
    aspx = dims[0] / dims[1]
    fx = dims[0] / (2 * tanfov * aspx)
    fy = dims[1] / (2 * tanfov)
    cx, cy = kwds.get("optical_centre", (0.5 * dims[0] - 0.5, 0.5 * dims[1] - 0.5))
    if 'optical_centre' in kwds: del kwds['optical_centre']  # remove keyword
    cameraMatrix = np.array([[fx, 0, cx],
                             [0, fy, cy],
                             [0, 0, 1]])

    # distortion free lens
    dist_coef = np.zeros(4)

    # use opencv to solve pnp problem and hence camera position
    if ransac:
        suc, rot, pos, inl = cv2.solvePnPRansac(objectPoints=kxyz[:, :3].copy(),  # points in world coords
                                                imagePoints=kxy[:, :2].copy(),  # points in img coords
                                                cameraMatrix=cameraMatrix,  # camera matrix
                                                distCoeffs=dist_coef,
                                                **kwds)
    else:
        inl = None
        suc, rot, pos = cv2.solvePnP(objectPoints=kxyz[:, :3].copy(),  # points in world coords
                                     imagePoints=kxy[:, :2].copy(),  # points in img coords
                                     cameraMatrix=cameraMatrix,  # camera matrix
                                     distCoeffs=dist_coef,
                                     **kwds)

    assert suc, "Error - no solution found to pnp problem."

    # get first solution
    if not inl is None:
        inl = inl[:, 0]
    pos = np.array(pos[:, 0])
    rot = np.array(rot[:, 0])

    # apply rotation position vector
    p = -np.dot(pos, spatial.transform.Rotation.from_rotvec(rot).as_matrix())

    # convert rot from axis-angle to euler
    r = spatial.transform.Rotation.from_rotvec(rot).as_euler('xyz', degrees=True)

    # apply dodgy correction to r (some coordinate system shift)
    r = np.array([r[0] - 180, -r[1], -r[2]])

    return p + mean, r, inl
