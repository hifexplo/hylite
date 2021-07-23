import unittest
import numpy as np
from tempfile import mkdtemp
import shutil
import os
from hylite.project import proj_persp
from hylite.project import PMap
from hylite import io

class MyTestCase(unittest.TestCase):
    def test_projection(self):

        # generate a basic geometry
        x, y = np.meshgrid( np.linspace(-10,10), np.linspace(-10,10) )
        xyz = np.vstack( [x.ravel(),y.ravel(),np.zeros_like(x.ravel())] ).T

        # project onto a camera
        img_dims = (20,20)
        pp, vis = proj_persp(xyz, np.array([0, 0, 40]),
                             a=np.array([0, 0, 90]), fov=25., dims=img_dims)

        # build projection map
        pm = PMap(*img_dims, xyz.shape[0])
        pm.set_ppc(pp, vis)

        # test basic stuff
        self.assertEqual(pm.point_count(), 1936)
        self.assertEqual(pm.pixel_count(), 400)
        self.assertEqual(pm.get_pixel_indices( 505 )[0][0,1], 18 )
        self.assertAlmostEqual(pm.get_depth( (16, 18), 505 ), 40., places=5 )
        self.assertEqual(pm.get_point_index( (16,18) )[0], 505 )
        self.assertEqual(pm.get_pixel_index( 505 )[0], (16,18), 40.)

    def test_intersections(self):
        # generate a basic geometry
        x, y = np.meshgrid(np.linspace(-10, 10), np.linspace(-10, 10))
        xyz = np.vstack([x.ravel(), y.ravel(), np.zeros_like(x.ravel())]).T

        # project onto a camera 1 and build pmap
        img_dims = (40, 20)
        pp, vis1 = proj_persp(xyz, np.array([-5, 0, 40]),
                              a=np.array([0, 0, 90]), fov=25., dims=img_dims)
        pm1 = PMap(*img_dims, xyz.shape[0])
        pm1.set_ppc(pp, vis1)

        # plt.scatter( pp[vis1,0], pp[vis1,1] )
        # plt.show()

        # project onto a camera 2 and build pmap
        img_dims = (20, 20)
        pp, vis2 = proj_persp(xyz, np.array([5, 0, 40]),
                              a=np.array([0, 0, 90]), fov=25., dims=img_dims)
        pm2 = PMap(*img_dims, xyz.shape[0])
        pm2.set_ppc(pp, vis2)

        U = np.sum(vis1 + vis2)  # number of points in union
        X = np.sum(vis1 & vis2)  # number of points in intersection

        self.assertEqual(len(pm1.union(pm2)), U)
        self.assertEqual(len(pm1.intersect(pm2)), X)

        kk, jj = pm1.intersect_pixels( pm2 )
        self.assertEqual(len(kk), X)
        self.assertEqual(len(jj), X)

    def test_io(self):

        # generate a basic geometry
        x, y = np.meshgrid(np.linspace(-10, 10), np.linspace(-10, 10))
        xyz = np.vstack([x.ravel(), y.ravel(), np.zeros_like(x.ravel())]).T

        # project onto a camera
        img_dims = (20, 20)
        pp, vis = proj_persp(xyz, np.array([0, 0, 40]),
                             a=np.array([0, 0, 90]), fov=25., dims=img_dims)

        # build projection map
        pm = PMap(*img_dims, xyz.shape[0])
        pm.set_ppc(pp, vis)

        pth = mkdtemp()

        # create temp directory
        try:
            # save pmap
            path = os.path.join(pth,"pmap.prj.npz")
            io.save(path, pm)

            # load pmap
            pm2 = io.load( path )

        except:
            shutil.rmtree(pth)  # delete temp directory
            self.assertFalse(True, "Error - could not save PMap data" )

        # checks
        self.assertTrue(os.path.exists(path), "Error - file could not be written.")
        self.assertEqual(pm.point_count(), pm2.point_count())
        self.assertEqual(pm.pixel_count(), pm2.pixel_count())

        shutil.rmtree(pth)  # delete temp directory

if __name__ == '__main__':
    unittest.main()
