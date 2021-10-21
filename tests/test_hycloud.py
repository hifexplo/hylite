import unittest
import numpy as np
from tests import genCloud, genImage
from hylite.project import Camera

class TestHyCloud(unittest.TestCase):
    def test_cloud(self):

        cloud = genCloud(npoints = 1000, nbands=10)
        self.assertEqual(cloud.point_count(), 1000)
        self.assertEqual(cloud.has_rgb(), True)
        self.assertEqual(cloud.has_normals(), True)
        self.assertEqual(cloud.has_bands(), True)

        n0 = cloud.normals[0, :].copy()
        cloud.flip_normals()
        self.assertEqual( np.sum( n0 + cloud.normals[0,:] ), 0 )

        # test generate normals
        cloud.compute_normals(1.0, vb=True)

        cloud.filter_points(0,val=(0.1,0.5),trim=True)
        self.assertGreaterEqual(np.nanmin(cloud.data[:,0]), 0.1)
        self.assertLessEqual(np.nanmax(cloud.data[:,0]), 0.5)
        self.assertLess(cloud.point_count(),1000)

        # test rendering
        cam = Camera( pos = np.array([0.0,0.0,10.0]), ori=np.array([0.0,0.0,0.0]), fov=30, proj='persp', dims=(1000,1000) )
        cloud.render( cam )

        # test projection
        image = genImage(1000,1000)
        cloud.project( image, cam)


if __name__ == '__main__':
    unittest.main()
