import unittest
import numpy as np
from tests import genCloud, genImage
from hylite.project import Camera
from hylite.project.basic import proj_persp, rasterize, proj_ortho, proj_pano
import os
from pathlib import Path
import hylite
from hylite import io
class TestHyCloud(unittest.TestCase):

    def test_projection(self):
        # load point cloud
        cloud = io.load( os.path.join(os.path.join(str(Path(__file__).parent.parent), "test_data"), "hypercloud.hdr") )
        cam = cloud.header.get_camera(0)

        # project perspective
        pp, viz = proj_persp( cloud.xyz, cam.pos, cam.ori, cam.fov, cam.dims )
        self.assertTrue(viz.all())

        R,zz = rasterize( pp, viz, cloud.rgb, cam.dims, s=2 )
        self.assertTrue( np.isfinite(zz).any() )
        self.assertTrue( np.isfinite(R).any() )

        # test rendering [ or, at least run these functions... ]
        cloud.quick_plot(hylite.RGB, cam )

        rgb = cloud.rgb.copy()
        cloud.colourise( hylite.RGB, stretch=(0.0,95) )
        self.assertEqual( (rgb == cloud.rgb).all(), False ) # check that colours have changed!
        cloud.quick_plot('rgb', cam )


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
