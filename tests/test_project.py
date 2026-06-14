import os
import unittest
from pathlib import Path
from tempfile import mkdtemp
import shutil

import hylite
import numpy as np
from hylite import io
from tests._support import TEST_DATA, require_test_env
from hylite.project.basic import proj_persp, rasterize
from hylite.project.pmap import PMap, blend_scenes, push_geomattr, get_blend_weights
from hylite.project.camera import Camera
from hylite.project.pushbroom import Pushbroom
from hylite import HyScene, HyCloud, HyImage


class TestPMap(unittest.TestCase):
    def test_projection(self):
        require_test_env(self, "default")
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

        # test points per pixel
        npoints = pm.points_per_pixel()
        self.assertEqual(np.max(npoints.data), 9)

        npixels = pm.pixels_per_point()
        self.assertEqual(np.max(npixels), 1)

    def test_intersections(self):
        require_test_env(self, "default")
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
        require_test_env(self, "default")

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


class TestAlign(unittest.TestCase):
    def test_warp(self):
        require_test_env(self, "opencv")
        from hylite.project.align import align
        image1 = io.load(os.path.join(TEST_DATA, "image.hdr"))
        image1.data = np.dstack( [ image1.data, image1.data ] ) # make dataset > 512 bands

        # create slightly offset second image
        image2 = image1.copy()
        image2.data = image2.data[5:, 5:, : ].copy() # add an offset
        image2.push_to_header()

        # n.b. there's not really any way to check if the align actually worked... so just try a bunch of combinations
        align(image1, image2, source_bands=((30, 50),), dest_bands=((30, 50),), method='affine')
        align(image1, image2, source_bands=((450., 600.),), dest_bands=((450., 600.),), method='polynomial')

        for m in ['piecewise', 'affine', 'polynomial']:
            align(image1, image2, source_bands=(0, 1, 2), method=m)

    def test_align_to_cloud(self):
        require_test_env(self, "opencv")
        from hylite.project.align import align_to_cloud_manual, proj_persp
        from hylite.project.camera import Camera
        
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


class TestHyScene(unittest.TestCase):
    def build_dummy_data(self):

        # build an example cloud
        x, y = np.meshgrid(np.linspace(-10, 30), np.linspace(-10, 10))
        xyz = np.vstack([x.ravel(), y.ravel(), np.zeros_like(x.ravel())]).T
        klm = np.zeros(xyz.shape)
        klm[:,2] = 1.0
        rgb = (np.random.rand( *xyz.shape ) * 255).astype(np.uint8)
        self.cloud = HyCloud( xyz, rgb=rgb, normals=klm )

        # build an example image
        dims = (20, 20, 3)
        self.image = HyImage( np.full( dims, 0.75 ) )
        self.image[10,:,:] = np.nan # add some nans
        self.image.set_wavelengths( hylite.RGB )

        # build associated camera
        pos = np.array([0, 0, 40])
        ori = np.array([0, 0, 90])
        fov = 25.
        self.cam = Camera( pos, ori, 'persp', fov, dims)

        # and a track
        self.swath = HyImage(np.full((dims[0],100,3), 0.75))
        self.swath.data[:,10,:] = np.nan # add some nans
        self.swath.set_wavelengths(hylite.RGB)
        fov = 25.
        cp = np.zeros( (100, 3) )
        cp[:, 0] +=  np.linspace(-10, 10, 100)
        cp[:, 1] +=  np.linspace(-10, 10, 100)
        cp[:, 2] = 30.
        co = np.zeros( (100,3) )
        self.track = Pushbroom( cp, co, fov / dims[0], fov / dims[0], (dims[0], cp.shape[0]) )

    def test_construction(self):
        require_test_env(self, "default")
        self.build_dummy_data()

        # make a test directory
        pth = mkdtemp()
        try:
            # init a scene
            S = HyScene(pth,"Scene1")

            # build using normal camera
            S.construct( self.image, self.cloud, self.cam, occ_tol=1, maxf=100, s=5 )
            assert np.sum(S.pmap.data) > 0 # must be at least some valid mappings
            assert S.pmap.point_count() > 0 # must be some valid points
            assert S.pmap.pixel_count() > 0 # must be some valid pixels

            # build using pushbroom camera
            self.assertAlmostEqual( self.track.get_z(i=50)[2], -1, 2) # camera points down
            S2 = HyScene(pth, "Scene2")
            S2.construct( self.swath, self.cloud, self.track, occ_tol=1, maxf=100, s=(5,1), step=10 )
            assert np.sum(S2.pmap.data) > 0 # must be at least some valid mappings
            assert S2.pmap.point_count() > 0 # must be some valid points
            assert S2.pmap.pixel_count() > 0 # must be some valid pixels
            

            # test projections using normal camera
            cld = S.push_to_cloud( hylite.RGB, method='best' )
            img = S.push_to_image( 'klm', method='closest')
            self.assertAlmostEqual( np.nanmax(img.data), np.nanmax(S.cloud.normals), 2) # check 3D-2D projection worked (max of normal vectors is close)
            self.assertAlmostEqual(np.nanmax(cld.data), np.nanmax(S.image.data), 2 ) # check 2D-3D back-projection worked (max of images is close)

            # test projections using pushbroom camera
            cld = S2.push_to_cloud(hylite.RGB, method='best')
            img = S2.push_to_image('klm', method='closest')
            self.assertAlmostEqual( np.nanmax(img.data), np.nanmax(S2.cloud.normals), 2) # check 3D-2D projection worked (max of normal vectors is close)
            self.assertAlmostEqual(np.nanmax(cld.data), np.nanmax(S2.image.data), 2 ) # check 2D-3D back-projection worked (max of images is close)

            # test blending (scene.image / scene.swath RGB — blend_scenes reads each scene's image)
            S.image = self.image
            S2.image = self.swath
            # N.B. blend_scenes' default ooc=True calls s.free() after projecting each scene.
            # The lazy reloader then expects the scene to exist on disk. These scenes were
            # never persisted (and HyScene(pth, "Scene1") actually treats pth as the *name*,
            # not the root, so .save() targets a non-existent path anyway). Disable ooc so
            # the loop can re-use the in-memory scenes across weighting methods.
            for method in ['gsd','obliquity','distance','equal']:
                w = get_blend_weights([S,S2],method=method,ascloud=True) # run different weighting methods
                O = blend_scenes([S,S2], w, (0,-1), ooc=False )
                self.assertEqual( O.point_count(), 2500 )
                self.assertEqual(O.band_count(), 3)
                self.assertAlmostEqual( np.nanmax(O.data), max( np.nanmax(self.image.data), np.nanmax(self.swath.data) ), 2 ) # check normalisation during blending / averaging is correct (we are not scaling the data)
                
            # and, finally, a single check with ooc=True
            w = get_blend_weights([S,S2], method='equal',ascloud=True) # run different weighting methods
            O = blend_scenes([S,S2], w, (0,-1), ooc=True, trim=True )
            self.assertLess( O.point_count(), self.cloud.point_count() ) # check some points were deleted
            self.assertGreater( O.point_count(), 0 ) # some points were projected...
            self.assertEqual(O.band_count(), 3)
            self.assertAlmostEqual( np.nanmax(O.data), max( np.nanmax(self.image.data), np.nanmax(self.swath.data) ), 2 ) # check normalisation during blending / averaging is correct
            self.assertAlmostEqual( np.nanmin(O.data), min( np.nanmin(self.image.data), np.nanmin(self.swath.data) ), 2  ) # min should also be the same (check blend weights sum to 1 properly)
        except:
            shutil.rmtree(pth)  # delete temp directory
            self.assertFalse(True, "Error - could not construct HyScene instance")

        shutil.rmtree(pth)  # delete temp directory

class TestHyCloudProjection(unittest.TestCase):
    def test_projection(self):
        require_test_env(self, "default")
        # load point cloud
        cloud = io.load( os.path.join(TEST_DATA, "hypercloud.hdr") )
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

        ids = cloud.render('ortho', 'i')
        assert np.max(ids.data.ravel().astype(int)) > 1000 # check some point IDs exist

if __name__ == '__main__':
    unittest.main()
