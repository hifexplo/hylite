"""
Functions for coregistering image and/or point cloud data.
"""

from hylite import HyImage
from hylite.project.basic import *
from hylite.project.camera import Camera
import cv2
import numpy as np
import hylite

def deepWarp(image,target):

    """
    Uses deep optical flow to warp one image to match another.

    Args:
        source (ndarray): the image to deform. Must be numpy array.
        target (ndarray): the target image to fit the source image too. Must be numpy array.

    Returns:
        A tuple containing the:

         - warped numpy array.
         - displacement map of the warp.

    """

    #convert images to greyscale uint8
    image = np.uint8(255 * (image - np.nanmin(image)) /
                         (np.nanmax(image) - np.nanmin(image)))
    target = np.uint8(255 * (target - np.nanmin(target)) /
                         (np.nanmax(target) - np.nanmin(target)))

    #calculate deep optical flow
    alg = cv2.optflow.createOptFlow_DeepFlow()
    flow = alg.calc(image,target,None)

    #distort image to fit target
    X, Y = np.meshgrid(range(image.data.shape[1]), range(image.data.shape[0]))
    dmap = np.dstack([X, Y]).astype(np.float32)
    dmap[:,:,0] += flow[:,:,0]
    dmap[:,:,1] += flow[:,:,1]

    return cv2.remap(image, dmap, None,cv2.INTER_LINEAR), dmap

def align_to_cloud_manual( cloud, cam, points, pixels, **kwds ):
    """
    Solve for camera location given a list of >4 manually chosen pixel -> point pairs.

    Args:
        cloud (hylite.HyCloud): the point cloud (HyCloud instance) to match to
        cam (hylite.project.Camera): a Camera instance containing the camera parameters (fov, etc.). The cam.pos and cam.ori properties will be ignored so can be set to anything.
        points (list): a list of keypoint ids with length > 4. The position of each keypoint should thus be given by cloud.xyz[ points[i], : ].
        pixels (list): a list of corresponding pixel coordinates (after projection), such that pixels[i] = (px,py).
        **kwds:  keywords are passed to hylite.project.pnp( ... ).

    Returns:
        A tuple containing:

         - a Camera instance containing the PnP solution.
         - err = the mean absolute error between projected keypoints (using the PnP solution) and the corresponding
                 positions given in the pixels array.
    """

    assert len(points) == len(pixels), "Error - %d pointIDs != %d corresponding pixels" % (len(points), len(pixels))
    assert len(points) >= 4, "Error - at least four pixel/point pairs are needed to solve PnP problem."

    # get world coordinate array
    points = np.array(points).astype(np.uint)
    kxyz = cloud.xyz[ points, : ]

    # get pixel coordinates array
    kxy = np.array( pixels ).astype(float) - 0.5 # must be float for OpenCV. - 0.5 transforms to pixel centers.

    # solve pnp problem
    if 'pano' in cam.proj.lower():
        kxy_pp = pano_to_persp(kxy[:,0], kxy[:, 1], cam.fov, cam.step, cam.dims)
        p_est, r_est, inl = pnp(kxyz, kxy_pp, cam.fov, cam.dims, ransac=True, **kwds)
    else:
        p_est, r_est, inl = pnp(kxyz, kxy, cam.fov, cam.dims, ransac=True, **kwds)

    # put estimate in a camera object
    est = Camera(p_est, r_est, cam.proj, cam.fov, cam.dims, cam.step)

    # calculate final correspondances
    if 'pano' in cam.proj.lower():
        pp, vis = proj_pano(kxyz[inl, :], C=p_est, a=r_est,
                            fov=cam.fov, dims=cam.dims, step=cam.step)
    else:
        pp, vis = proj_persp(kxyz[inl, :], C=p_est, a=r_est,
                             fov=cam.fov, dims=cam.dims)

    err = np.linalg.norm(pp[:, 0:2] - kxy[inl, :], axis=1)
    err = np.mean(err)

    return est, err


def refine_alignment(image, cloud, cam, bands=hylite.RGB, s=2,
                     maxdist=25, histeq=True, method='sift',
                     recurse=1, feat_args={}, match_args={}, vb=True, **kwds):
    if 'pano' in cam.proj.lower():
        assert not cam.step is None, "Error - angluar step ('step') must be defined for panoramic cameras."

    # generate rendered view to get matches from
    if vb: print("Projecting scene... ", end='')
    render = cloud.render(cam, fill_holes=True, s=s, bands='rgb')
    xyz = cloud.render(cam, fill_holes=True, s=s, bands='xyz')
    if vb: print("Done.")

    # apply masking
    image = image.export_bands(bands)
    image.set_as_nan(0)
    render.set_as_nan(0)
    mask = np.logical_or(np.isnan(render.data).all(axis=-1), np.isnan(image.data).all(axis=-1))
    mask = cv2.erode(mask.astype(np.uint8), np.ones((int(maxdist * 2) + 1, int(maxdist * 2) + 1), np.uint8))
    mask = mask == 1
    image.data[mask] = np.nan
    render.data[mask] = np.nan

    if histeq:
        from hylite.correct import hist_eq
        render.data = hist_eq(render.data, image.data)

    # extract matches
    if vb: print("Gathering matches..", end='')
    k1, d1 = image.get_keypoints((0, 1, 2), method=method, **feat_args)
    k2, d2 = render.get_keypoints((0, 1, 2), method=method, **feat_args)
    src, dst = hylite.HyImage.match_keypoints(k1, k2, d1, d2, method=method, dist=match_args.pop("dist", 0.9),
                                              **match_args)

    # filter matches by distance
    r = np.linalg.norm(src - dst, axis=-1)
    mask = r < maxdist
    src = src[mask][:, ::-1]  # flip from (y,x) to (x,y) notation here
    dst = dst[mask][:, ::-1]
    if vb: print("Found %d matches." % len(src))
    assert len(src) != 0, "Error - no matches found."

    # plot matches
    if vb:
        fig, ax = image.quick_plot((0, 1, 2), vmin=0, vmax=90)
        ax.scatter(src[:, 0], src[:, 1], color='cyan', marker='+')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(0, image.xdim())
        ax.set_ylim(image.ydim(), 0)
        fig.show()

    # get 3D location of tie points
    k3d = []
    for px, py in dst:
        p = xyz.data[int(px), int(py), :]  # extract 3d position
        k3d.append(p)
    k3d = np.array(k3d)

    # occasionaly keypoints have undefined positions... remove these ones.
    mask = np.isfinite(np.sum(k3d, axis=1))
    assert sum(mask) > 3, "Error - at least 4 keypoints with finite positions are needed...."
    k3d = k3d[mask, :]
    src = src[mask, :]

    if 'pano' in cam.proj.lower():
        pp = pano_to_persp(src[:, 0], src[:, 1], cam.fov, cam.step, image.data.shape)
        p_est, r_est, inl = pnp(k3d, pp, cam.fov, image.data.shape, **kwds)
    else:
        p_est, r_est, inl = pnp(k3d, src, cam.fov, image.data.shape, **kwds)

    # put estimate in a camera object
    est = Camera(p_est, r_est, cam.proj, cam.fov, image.data.shape, cam.step)

    if recurse > 1:
        return refine_alignment(image, cloud, est, bands=bands, s=s,
                                maxdist=maxdist, histeq=histeq, method=method,
                                recurse=recurse - 1, feat_args=feat_args, match_args=match_args, vb=vb, **kwds)

    # plot updated view
    if vb:
        print("Projecting updated preview...")
        render = cloud.render(est, fill_holes=True, s=s, bands='rgb')
        render.set_as_nan(0)
        fig, ax = render.quick_plot((0, 1, 2), vmin=0, vmax=90)
        ax.scatter(dst[:, 0], dst[:, 1], color='r', marker='+')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(0, image.xdim())
        ax.set_ylim(image.ydim(), 0)
        fig.show()

    return est, src, k3d

def align_to_cloud(image, cloud, cam, bands=hylite.RGB,
                   method='sift', recurse=2, s=2, sf=3.0, cfac=0.0, bfac=0.0,
                   vb=True, gf=True, **kwds):
    """
    Aligns a hyperspectral image to a point cloud based on an approximate camera position and
    view direction. This inital position must be reasonable to ensure good tie-points can be
    created and used to solve for the camera position.

    Args:
        image (hylite.HyImage): a HyImage object to match.
        cloud (hylite.HyCloud): a georeferenced HyCloud to match to.
        cam (hylite.project.Camera): a Camera object containing camera data (fov, step, dims) and initial position/orientation estimate.
        bands (tuple): a tuple containing the 3 hyperspectral bands to match agains the RGB bands of the point cloud. Default is
                       io.HyImage.RGB.
        method (str): the matching method to use. Can be 'sift' (default) or 'orb'.
        recurse (int): The number of rounds of matching/alignment to perform. Default is 2.
        s (int): the point size to use for rendering. Default is 2. Must be integer.
        sf (int): increase resolution of rendered images to enhance point matching. Default is 3.
        cfac (float): contrast adjustment to apply to hyperspectral bands before matching. Default is 0.0.
        bfac (float): brightness adjustment to apply to hyperspectral bands before matching. Default is 0.0.
        vb (bool): True if messages should be written to the console. Default is True.
        gf (bool): True if a plot showing matching keypoints and residuals should be created. Useful for checking
            matching accuracy, troubleshooting and evaluating the accuracy of the reconstructed camera position.
            Default is True.
        **kwds: keyword arguments are passed to the opencv feature detecting and matching algorithms.  For matching,
            selectivity can be adjusted using the dist parameter:

            - dist = the similarity threshold for identifying matches. Default is 0.7.

            For keypoint detection using SIFT, available parameters are:

            - contrastThreshold: default is 0.01.
            - edgeThreshold: default is 10.
            - sigma: default is 2.0

            For keypoint detection using ORB, available parameters are:

            - nfeatures = the number of features to detect. Default is 5000.

            Remaining keywords are passed to hylite.project.pnp( ... ).

    Returns:
        A tuple containing:

         - cam_est = a camera object with optimised positions.
         - keypoints = the keypoints used to determine this, such that keypoints[i] = [px,py,x,y,z].
         - err = average distance (in pixels) between inlier points.
    """

    if 'pano' in cam.proj.lower():
        assert not cam.step is None, "Error - angluar step ('step') must be defined for panoramic cameras."

    # generate rendered view to get matches from
    if vb: print("Projecting scene... ", end='')
    ortho_cam = cam.clone()
    ortho_cam.dims = (int(image.data.shape[0] * sf), int(image.data.shape[1] * sf)) #increase resolution by scale factor
    if ortho_cam.is_panoramic():
        ortho_cam.step = cam.step / sf
    ortho = cloud.render(ortho_cam, fill_holes=True, blur=3,
                         s=s, bands='rgbxyz')
    if vb: print("Done.")

    # extract matches
    if vb: print("Gathering matches..", end='')
    k_ortho = None
    k_hyper = None
    for b in range(3):

        # extract points
        if 'sift' in method:
            sigma = kwds.pop('sigma',2.0)
            contrastThreshold = kwds.pop('contrastThreshold',0.01)
            edgeThreshold = kwds.pop('edgeThreshold', 10)
            k1, d1 = ortho.get_keypoints(band=b, method=method, mask=True,
                                         sigma=sigma*sf, contrastThreshold=contrastThreshold,edgeThreshold=edgeThreshold,
                                         cfac=cfac, bfac=bfac)
            k2, d2 = image.get_keypoints(band=bands[b], method=method, mask=True,
                                         sigma=sigma*sf, contrastThreshold=contrastThreshold,edgeThreshold=edgeThreshold,
                                         cfac=cfac, bfac=bfac)
        elif 'orb' in method:
            nfeatures = kwds.pop('nfeatures', 5000)
            k1, d1 = ortho.get_keypoints(band=b, method=method, mask=True, nfeatures=nfeatures, cfac=cfac, bfac=bfac)
            k2, d2 = image.get_keypoints(band=bands[b], method=method, mask=True, nfeatures=nfeatures, cfac=cfac, bfac=bfac)
        else:
            assert False, "Error - unknown matching method %s" % method
        # match points
        dist = kwds.pop('dist',0.7)
        k_o, k_h = HyImage.match_keypoints(k1, k2,
                                         d1, d2, method=method, dist=dist)

        if (not k_o is None) and vb: print("..%s=%d.." % ('rgb'[b], len(k_o)), end='')

        # transform opencv matches to sensible coordinates...
        if not k_o is None:
            k_o = np.array([k_o[:, 0, 1], k_o[:, 0, 0]]).T
            k_h = np.array([k_h[:, 0, 1], k_h[:, 0, 0]]).T

        if k_ortho is None:
            k_ortho = k_o
            k_hyper = k_h
        elif not k_o is None:
            k_ortho = np.vstack([k_ortho, k_o])
            k_hyper = np.vstack([k_hyper, k_h])

    if vb and (not k_ortho is None): print("(%d)." % len(k_ortho))

    # from here on we need some keypoints....
    assert not k_ortho is None, "Error - no valid matches found..."

    # get 3D location of tie points
    k3d = []
    for px, py in k_ortho:
        p = ortho.data[int(px), int(py), 3:6]  # extract 3d position
        k3d.append(p)
    k3d = np.array(k3d)

    #occasionaly keypoints have undefined positions... remove these ones.
    mask = np.isfinite( np.sum(k3d,axis=1) )
    assert sum(mask) > 3, "Error - at least 4 keypoints with finite positions are needed...."
    k3d = k3d[mask,:]
    k_hyper = k_hyper[mask,:]

    # solve pnp problem
    try:
        if 'pano' in cam.proj.lower():
            k_hyper_pp = pano_to_persp(k_hyper[:, 0], k_hyper[:, 1], cam.fov, cam.step, image.data.shape)
            p_est, r_est, inl = pnp(k3d, k_hyper_pp, cam.fov, image.data.shape, **kwds)
        else:
            p_est, r_est, inl = pnp(k3d, k_hyper, cam.fov, image.data.shape, **kwds)

    except: # pnp failed - plot keypoints before failing
        print("Error - PnP solution not found. Check sift feature matching?")
        if gf:
            fig, ax = ortho.quick_plot((0, 1, 2))

            if not k_ortho is None:
                ax.scatter(k_ortho[:, 0], k_ortho[:, 1], c='red', s=5)
            ax.set_title("Tie points in estimated camera view.")
            fig.show()

            # plot tie points
            fig, ax = image.quick_plot(bands, cfac=cfac, bfac=bfac)
            if not k_hyper is None:
                ax.scatter(k_hyper[:, 0], k_hyper[:, 1], c='red', s=5)
            ax.set_title("Tie points in hyperspectral scene.")
            fig.show()

        return None, None, np.inf

    #put estimate in a camera object
    est = Camera(p_est,r_est,cam.proj,cam.fov,image.data.shape,cam.step)

    # calculate final correspondances
    if 'pano' in cam.proj.lower():
        pp, vis = proj_pano(k3d[inl, :], C=p_est, a=r_est,
                            fov=cam.fov, dims=image.data.shape, step=cam.step)
    else:
        pp, vis = proj_persp(k3d[inl, :], C=p_est, a=r_est,
                             fov=cam.fov, dims=image.data.shape)

    err = np.linalg.norm(pp[:, 0:2] - k_hyper[inl, :], axis=1)
    err = np.mean(err)

    if vb: print("Solved PnP problem using %d inliers (residual error = %.1f px)." % (len(inl),err))

    # recurse using updated estimates?
    if recurse > 1:
        if vb: print('------------resolving using updated camera transform------------')
        return align_to_cloud(image, cloud, est, bands=bands,
                               method=method,recurse=recurse-1, s=s, sf=sf, vb=vb, gf=gf, **kwds)

    # plot residuals
    if gf:

        # render point cloud as background (slow)
        #fig, ax = cloud.quick_plot(est,
        #                           s=1, fill_holes=True, blur=True,
        #                           bands='rgb')

        # plot original image as background (fast)
        fig, ax = image.quick_plot(bands)

        #plot tie points and residuals
        ax.scatter(pp[vis, 0], pp[vis, 1], c='r', s=2)
        ax.scatter(pp[vis, 0], pp[vis, 1], c='r', s=2)
        ax.scatter(k_hyper[inl][:, 0], k_hyper[inl][:, 1], c='cyan', s=2)
        ax.set_xlim(0, image.data.shape[0])  # ensure axis ranges haven't been changed by weird points.
        ax.set_ylim(image.data.shape[1], 1)
        for i, p in enumerate(pp):
            ax.plot([pp[i][0], k_hyper[inl][i, 0]], [pp[i][1], k_hyper[inl][i, 1]], c='white', lw=1.0)
        ax.set_title("Reprojected keypoint residuals.")
        fig.show()

    return est, k3d[inl, :], err


def align(image1, image2, source_bands, dest_bands=None, method='affine', matchdist=0.6, vb=False, **kwds):
    """
    Coregister an image or numpy array to a target image using SIFT keypoints and the specified image transform.

    Args:
        image1 (hylite.HyImage): the reference image.
        image2 (hylite.HyImage): the image to transform.
        source_bands (tuple): A tuple defining the bands to use in image 1.
        dest_bands (tuple): A tuple defining the bands to use in image 2. Defaults to source_bands.
        method (str): the method to use. Options are 'affine', 'piecewise_affine' or 'polynomial'.
        matchdist (float): the SIFT matching distance threshold. Default is 0.6.
        vb (bool): create graphical output figures for debugging. Default is False.
        **kwds*: keywords are passed to HyImage.get_keypoints( ... ).

    Returns:
        A transformed image
    """
    assert isinstance(image1, hylite.HyImage) and isinstance(image2,
                                                             hylite.HyImage), "Error - images myst be HyImage instances."

    if dest_bands is None:
        dest_bands = source_bands

    # extract features
    def getFeatures(I1, I2):
        pref = []
        ptarg = []
        for b1, b2 in zip(source_bands, dest_bands):
            k1, d1 = I1.get_keypoints(b1, method='sift', mask=True)
            k2, d2 = I2.get_keypoints(b2, method='sift', mask=True)
            kimg, ktg = I2.match_keypoints(k1, k2, d1, d2, method='sift', dist=matchdist)
            pref.append(kimg)
            ptarg.append(ktg)
        return np.vstack(pref), np.vstack(ptarg)

    pref, ptarg = getFeatures(image1, image2)

    if vb:
        fig, ax = image1.quick_plot(source_bands)
        ax.set_title("Initial features")
        ax.scatter(pref[:, 0, 1], pref[:, 0, 0], s=4, color='r')
        ax.scatter(ptarg[:, 0, 1], ptarg[:, 0, 0], s=4, color='g')
        xx = np.array([[pref[i, 0, 1], ptarg[i, 0, 1]] for i in range(pref.shape[0])]).T
        yy = np.array([[pref[i, 0, 0], ptarg[i, 0, 0]] for i in range(pref.shape[0])]).T
        ax.plot(xx, yy, color='white', alpha=0.2)
        fig.show()

    # compute transform
    try:
        from skimage import transform as tf
    except:
        assert False, "Error - please install scikit image (pip install scikit-image) to use this functionality."

    if 'piecewise' in method.lower():
        tform = tf.PiecewiseAffineTransform()
    elif 'affine' in method.lower():
        tform = tf.AffineTransform()
    elif 'poly' in method.lower():
        tform = tf.PolynomialTransform()

    tform.estimate(pref[:, 0, :], ptarg[:, 0, :])

    # apply transform
    mapped = image2.copy(data=False)
    mapped.data = tf.warp(image2.data, tform, output_shape=(image1.xdim(), image1.ydim()))

    # final plot?
    if vb:
        # get transformed matches
        pref, ptarg = getFeatures(image1, mapped)  # get new matches to show residuals
        fig, ax = mapped.quick_plot(dest_bands)
        ax.set_title("Residual features")
        ax.scatter(pref[:, 0, 1], pref[:, 0, 0], s=4, color='r')
        ax.scatter(ptarg[:, 0, 1], ptarg[:, 0, 0], s=4, color='g')
        xx = np.array([[pref[i, 0, 1], ptarg[i, 0, 1]] for i in range(pref.shape[0])]).T
        yy = np.array([[pref[i, 0, 0], ptarg[i, 0, 0]] for i in range(pref.shape[0])]).T
        ax.plot(xx, yy, color='white', alpha=0.2)
        fig.show()

    return mapped


#### DEPRECTIATED
def align_images(image1, image2, warp=True, **kwds):
    """
    Coregister an image or numpy array to a target image.

    Args:
        image1 (hylite.HyImage): the image to transform
        image2 (hylite.HyImage): the reference image to fit too.
        **kwds: Optional keywords include:

             - image_bands = the band(s) in the image to use for matching. If an integer or wavelength is
                             passed then a single band will be used. A tuple contingin a minimum index or
                             wavelength can also be passed, in which case bands will be averaged. Default is
                             (0,3) (i.e. the first 3-bands).
             - target_bands = the band(s) in the target image to use, in the same format as image_bands. Default
                              is (0,3) (i.e. the first 3-bands).
             - method = the method used for image warping. Default is 'affine'.
             - features = the feature detector to use. Can be 'sift' (default) or 'orb'.
             - warp = use dense flow to warp the image to better fit the target. Default is True.
             - dist = the distance threshold used for keypoint matching. Default is 0.75.
             - rthresh = the ransac inlier threshold for ransac outlier detection. Default is 10.0. Increase for more tolerance.

    Returns:
        A numpy array containing the coregistered image data.
    """
    print("Warning: align_images is depreciated. Please use piecewise_align, polynomial_align or affine_align instead.")

    assert isinstance(image1,HyImage) and isinstance(image2,HyImage), "Error - images myst be HyImage instances."

    #get image features
    bands = kwds.get("image_bands",(0,3))
    k_image, d_image = image1.get_keypoints(bands,method=kwds.get('features', 'sift'), mask=True)

    #get target features
    bands = kwds.get("target_bands",(0,3))
    k_target, d_target = image2.get_keypoints(bands,method=kwds.get('features', 'sift'), mask=True)

    #match features
    k_image,k_target = image2.match_keypoints(k_image, k_target,
                                              d_image, d_target,
                                              method=kwds.get('features', 'sift'), dist=kwds.get('dist',0.75))

    #filter dodgy points using ransac model
    src_pts = k_image
    dst_pts = k_target

    #filter dodgy points using ransac model
    assert (src_pts is not None) and (dst_pts is not None), "Error - no valid matches found."
    H, status = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, kwds.get('rthresh', 10.0))
    dst_mask = dst_pts[:, 0, :] * status
    src_mask = src_pts[:, 0, :] * status
    dst_mask = dst_mask[dst_mask.all(1)]
    src_mask = src_mask[src_mask.all(1)]

    method = kwds.get('method', 'affine').lower() # get transform method
    if 'affine' in method: # warp with affine transform
        dst_mask = np.expand_dims(dst_mask, axis=1)
        src_mask = np.expand_dims(src_mask, axis=1)
        #M = cv2.estimateRigidTransform(src_mask, dst_mask, False) # estimate affine transform
        M = cv2.estimateAffinePartial2D(src_mask, dst_mask)[0]

        # if > 512 bands, cut in half for open-cv compatability
        if image1.band_count() > 1025:
            assert False, "Too many baaaands!!!"
        if image1.band_count() < 512: # easy
            #mapped = cv2.warpPerspective(image1.data, M, (image2.ydim(), image2.xdim()))
            mapped = cv2.warpAffine(image1.data, M, (image2.data.shape[1], image2.data.shape[0])) # apply
        else: # slightly less easy
            div = int(image1.data.shape[2] / 2) # split dataset

            # map each part
            mapped1 = cv2.warpAffine(image1.data[...,:div], M, (image2.data.shape[1], image2.data.shape[0]))
            mapped2 = cv2.warpAffine(image1.data[...,div:], M, (image2.data.shape[1], image2.data.shape[0]))
            #mapped1 = cv2.warpPerspective(image1.data[..., :div], H, (image2.ydim(), image2.xdim()))
            #mapped2 = cv2.warpPerspective(image1.data[..., div:], H, (image2.ydim(), image2.xdim()))

            # rejoin
            mapped = np.concatenate((mapped1, mapped2), axis=2)

    elif 'poly' in method: # warp with polynomial
        try:
            from skimage import transform as tf
        except:
            assert False, "Error - please install scikit image to use poly mode: pip install scikit-image"
        tform3 = tf.estimate_transform('polynomial', dst_mask, src_mask)
        mapped = tf.warp(image1.data, tform3, output_shape=(image2.xdim(), image2.ydim()))
    else:
        assert False, "Unknown transform method %s." % method

    #refine using deep flow?
    if warp:
        _, dmap = deepWarp( np.nanmean( mapped, axis=2 ), np.nanmean( image2.data, axis=2) )
        mapped = cv2.remap( mapped, dmap, None, cv2.INTER_LINEAR)
    return mapped

