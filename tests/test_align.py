import unittest
import os
from pathlib import Path
import hylite.io as io
from hylite.project.align import *
import shutil
from tempfile import mkdtemp
import numpy as np

class TestAlign(unittest.TestCase):
    def test_warp(self):
        # load test image
        image1 = io.load(os.path.join(os.path.join(str(Path(__file__).parent.parent), "test_data"),"image.hdr"))
        image1.data = np.dstack( [ image1.data, image1.data ] ) # make dataset > 512 bands

        # create slightly offset second image
        image2 = image1.copy()
        image2.data = image2.data[5:, 5:, : ].copy() # add an offset
        image2.push_to_header()

        # test align with method=affine
        # n.b. there's not really any way to check if the align actually worked... so just try a bunch of combinations
        np.warnings.filterwarnings('ignore') # supress warnings when comparing to nan
        align_images( image1, image2, image_bands=(30, 50), target_bands=(30, 50),
                                warp=True, features='sift', method='affine' )
        align_images(image1, image2, image_bands=(450., 600.), target_bands=(450., 600.),
                     warp=True, features='sift', method='poly')

    def test_align_to_cloud(self):

        # generate a basic geometry
        x, y = np.meshgrid(np.linspace(-10, 10), np.linspace(-10, 10))
        xyz = np.vstack([x.ravel(), y.ravel(), np.zeros_like(x.ravel())]).T
        cloud = hylite.HyCloud( xyz )

        # project onto a camera
        img_dims = (20, 20)
        cam = Camera(np.array([0, 0, 40]), np.array([0, 0, 90]), 'persp', 25., dims=img_dims)
        pp, vis = proj_persp(xyz, cam.pos, cam.ori, fov=cam.fov, dims=cam.dims)

        # choose 20 random points
        idx = [np.random.randint(0, np.sum(vis)) for i in range(20)]
        pixels = pp[vis,:][idx, :2 ]
        points = np.arange(xyz.shape[0])[vis][idx]
        cam_est, err = align_to_cloud_manual(cloud, cam, points, pixels)

        #print(err, np.linalg.norm(cam.pos - cam_est.pos), np.linalg.norm(cam.ori - cam_est.ori))
        self.assertAlmostEqual(np.linalg.norm(cam.pos - cam_est.pos), 0, 3 )
        self.assertAlmostEqual( np.sin( np.deg2rad( np.linalg.norm(cam.ori - cam_est.ori) ) ), 0, 3 )


if __name__ == '__main__':
    unittest.main()
