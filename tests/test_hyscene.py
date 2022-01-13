import unittest

from tempfile import mkdtemp
import shutil

import hylite
from hylite import HyScene, HyCloud, HyImage
from hylite.project import Camera, Pushbroom, blend_scenes
import numpy as np
class MyTestCase(unittest.TestCase):
    def build_dummy_data(self):

        # build an example cloud
        x, y = np.meshgrid(np.linspace(-10, 10), np.linspace(-10, 10))
        xyz = np.vstack([x.ravel(), y.ravel(), np.zeros_like(x.ravel())]).T
        klm = np.zeros(xyz.shape)
        klm[:,2] = 1.0
        rgb = (np.random.rand( *xyz.shape ) * 255).astype(np.uint8)
        self.cloud = HyCloud( xyz, rgb=rgb, normals=klm )

        # build an example image
        dims = (20, 20, 3)
        self.image = HyImage( np.random.rand( *dims ) )
        self.image.set_wavelengths( hylite.RGB )

        # build associated camera
        pos = np.array([0, 0, 40])
        ori = np.array([0, 0, 90])
        fov = 25.
        self.cam = Camera( pos, ori, 'persp', fov, dims)

        # and a track
        self.swath = HyImage(np.random.rand(dims[0],100,3))
        self.swath.set_wavelengths(hylite.RGB)
        cp = np.zeros( (100, 3) )
        cp[:, 0] +=  np.linspace(-10, 10, 100)
        cp[:, 1] +=  np.linspace(-10, 10, 100)
        cp[:, 2] = 80.
        co = np.zeros( (100,3) )
        self.track = Pushbroom( cp, co, fov / dims[0], fov / dims[0], (dims[0], cp.shape[0]) )
        #print(self.track.R[0].as_matrix())
        #print(self.track.R[0].as_matrix()@(self.cloud.xyz[0,:]-cp[0,:]))

    def test_construction(self):
        self.build_dummy_data()

        # make a test directory
        pth = mkdtemp()
        try:
            # init a scene
            S = HyScene(pth,"Scene1")

            # build using normal camera
            S.construct( self.image, self.cloud, self.cam, occ_tol=1, maxf=100, s=5 )

            # build using pushbroom camera
            S2 = HyScene(pth, "Scene2")
            S2.construct( self.swath, self.cloud, self.track, occ_tol=1, maxf=100, s=(5,1) )

            # test projections using normal camera
            cld = S.push_to_cloud( hylite.RGB, method='best' )
            img = S.push_to_image( 'klm', method='closest')
            self.assertAlmostEquals( np.nanmax(img.data), 1.0, 2)
            self.assertAlmostEquals(np.nanmax(cld.data), 1.0, 2 )

            # test projections using pushbroom camera
            cld = S2.push_to_cloud(hylite.RGB, method='best')
            img = S2.push_to_image('klm', method='closest')

            # test blending
            O = blend_scenes(pth + '/blendtest.hyc', dict(test=[S, S2]), method='average', hist_eq=False, trim=False,
                         vb=True, clean=True)
            self.assertEqual( O.test.point_count(), 2500 )

        except:
            shutil.rmtree(pth)  # delete temp directory
            self.assertFalse(True, "Error - could not construct HyScene instance")

        shutil.rmtree(pth)  # delete temp directory




if __name__ == '__main__':
    unittest.main()
