from hylite import HyImage
from hylite.project.basic import *
from hylite.project.camera import Camera
import cv2
import numpy as np
import hylite

def deepWarp(image,target):

    """
    Uses deep optical flow to warp one image to match another.

    *Arguments*
     - source = the image to deform. Must be numpy array.
     - target = the target image to fit the source image too. Must be numpy array.

    *Returns*:
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

def align_to_cloud(image, cloud, cam, bands=hylite.RGB,
                   method='sift', recurse=2, s=2, sf=3.0, cfac=0.0, bfac=0.0,
                   vb=True, gf=True, **kwds):
    """
    Aligns a hyperspectral image to a point cloud based on an approximate camera position and
    view direction. This inital position must be reasonable to ensure good tie-points can be
    created and used to solve for the camera position.

    *Arguments*:
     - image = a HyImage object to match.
     - cloud = a georeferenced HyCloud to match to.
     - cam = a Camera object containing camera data (fov, step, dims) and initial position/orientation estimate.
     - bands = a tuple containing the 3 hyperspectral bands to match agains the RGB bands of the point cloud. Default is
             io.HyImage.RGB.
     - method = the matching method to use. Can be 'sift' (default) or 'orb'.
     - recurse = The number of rounds of matching/alignment to perform. Default is 2.
     - s = the point size to use for rendering. Default is 2. Must be integer.
     - sf = increase resolution of rendered images to enhance point matching. Default is 3.
     - cfac = contrast adjustment to apply to hyperspectral bands before matching. Default is 0.0.
     - bfac = brightness adjustment to apply to hyperspectral bands before matching. Default is 0.0.
     - vb = True if messages should be written to the console. Default is True.
     - gf = True if a plot showing matching keypoints and residuals should be created. Useful for checking
               matching accuracy, troubleshooting and evaluating the accuracy of the reconstructed camera position.
               Default is True.
    *Keywords*:
     - keyword arguments are passed to the hylite.project.p2p( ... ).

    *Returns*:
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
        k1, d1 = ortho.get_keypoints(band=b, method=method, mask=True, sigma=2.0 * sf, cfac=cfac, bfac=bfac)
        k2, d2 = image.get_keypoints(band=bands[b], method=method, mask=True, sigma=2.0, cfac=cfac, bfac=bfac)

        # match points
        k_o, k_h = HyImage.match_keypoints(k1, k2,
                                         d1, d2, method=method, dist=0.7)

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
            p_est, r_est, inl = pnp(k3d, k_hyper_pp, cam.fov, image.data.shape, ransac=True, **kwds)
        else:
            p_est, r_est, inl = pnp(k3d, k_hyper, cam.fov, image.data.shape, ransac=True, **kwds)
    except: # pnp failed - plot keypoints before failing
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
        fig, ax = image.quick_plot(hylite.RGB)

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

def align_images(image1, image2, warp=True, **kwds):
    """
    Coregister an image or numpy array to a target image.

    *Arguments*:
     - image1= the image to transform
     - image2 = the reference image to fit too.

    *Keywords*:
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

    *Returns*:
     - a numpy array containing the coregistered image data.
    ethod = the matching method to use. Can be 'sift' (default) or 'orb'.
     - warp = use dense flow to warp the image to better fit the target. Default is True.
     - dist = the distance threshold used for keypoint matching. Default is 0.75.
     - rthresh = the ransac inlier threshold for ransac outlier detection. Default is 10.0. Increase for more tolerance.

    *Returns*:
     - a numpy array containing the coregistered image data.
    """

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
        M = cv2.estimateRigidTransform(src_mask, dst_mask, False) # estimate affine transform
        mapped = cv2.warpAffine(image1.data, M, (image2.data.shape[1], image2.data.shape[0])) # apply
    elif 'poly' in method: # warp with polynomial
        from skimage import transform as tf
        tform3 = tf.estimate_transform('polynomial', dst_mask, src_mask)
        mapped = tf.warp(image1.data, tform3, output_shape=(image2.xdim(), image2.ydim()))
    else:
        assert False, "Unknown transform method %s." % method

    #refine using deep flow?
    if warp:
        _, dmap = deepWarp( np.nanmean( mapped, axis=2 ), np.nanmean( image2.data, axis=2) )
        mapped = cv2.remap( mapped, dmap, None, cv2.INTER_LINEAR)
    return mapped


