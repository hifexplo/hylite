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


def _sphere_ray(az, el, dist=10.0):
    """World direction for an identity-pose spherical camera (x right, y up, look -z)."""
    azr = np.deg2rad(az)
    elr = np.deg2rad(el)
    direction = np.array([
        np.cos(elr) * np.sin(azr),
        np.sin(elr),
        -np.cos(elr) * np.cos(azr),
    ])
    return direction * dist


class TestSphereCamera(unittest.TestCase):
    def _camera(self):
        step = 0.2
        dims = (200, 150)
        az_lo = -20.0
        el_hi = 15.0
        return Camera(np.zeros(3), np.zeros(3), 'sphere', dims[1] * step, dims,
                      step=step, az_lo=az_lo, el_hi=el_hi)

    def test_projection_roundtrip(self):
        require_test_env(self, "default")
        from hylite.project.basic import proj_sphere, pix_to_ray_sphere

        cam = self._camera()
        samples = [(0.0, 0.0), (0.0, 2.0), (3.0, 0.0)]
        xyz = np.vstack([_sphere_ray(az, el) for az, el in samples])
        pp, vis = proj_sphere(xyz, cam.pos, cam.ori, cam.az_lo, cam.el_hi, cam.step, cam.dims)
        self.assertTrue(vis.all())
        for i, (az, el) in enumerate(samples):
            self.assertAlmostEqual(pp[i, 0], (az - cam.az_lo) / cam.step, places=6)
            self.assertAlmostEqual(pp[i, 1], (cam.el_hi - el) / cam.step, places=6)
            self.assertAlmostEqual(pp[i, 2], 10.0, places=6)
            ray = pix_to_ray_sphere(pp[i, 0], pp[i, 1], cam.az_lo, cam.el_hi, cam.step)
            np.testing.assert_allclose(ray, xyz[i] / 10.0, atol=1e-6)

        # a normal pointing away from the camera is culled; one pointing toward it is kept
        away = xyz[:1]
        _, hidden = proj_sphere(away, cam.pos, cam.ori, cam.az_lo, cam.el_hi, cam.step, cam.dims, normals=away)
        _, shown = proj_sphere(away, cam.pos, cam.ori, cam.az_lo, cam.el_hi, cam.step, cam.dims, normals=-away)
        self.assertFalse(hidden[0])
        self.assertTrue(shown[0])

    def test_camera_io(self):
        require_test_env(self, "basic")
        from hylite.hyheader import HyHeader
        from hylite.io.cameras import saveCameraTXT, loadCameraTXT

        cam = self._camera()
        header = HyHeader()
        header.set_camera(cam)
        loaded = header.get_camera()
        self.assertEqual(loaded.proj, 'sphere')
        self.assertEqual(loaded.dims, cam.dims)
        self.assertAlmostEqual(loaded.step, cam.step, places=5)
        self.assertAlmostEqual(loaded.az_lo, cam.az_lo, places=5)
        self.assertAlmostEqual(loaded.el_hi, cam.el_hi, places=5)
        np.testing.assert_allclose(loaded.pos, cam.pos)
        np.testing.assert_allclose(loaded.ori, cam.ori)

        pth = mkdtemp()
        try:
            path = os.path.join(pth, 'cam.txt')
            saveCameraTXT(path, cam)
            text = loadCameraTXT(path)
        finally:
            shutil.rmtree(pth)
        self.assertTrue(text.is_spherical())
        self.assertAlmostEqual(text.az_lo, cam.az_lo, places=5)
        self.assertAlmostEqual(text.el_hi, cam.el_hi, places=5)
        self.assertAlmostEqual(text.step, cam.step, places=5)

        image = HyImage(np.zeros((cam.dims[0], cam.dims[1], 1)))
        image.header['panorama azimuth range deg'] = '%.3f %.3f' % (cam.az_lo, cam.az_lo + cam.dims[0] * cam.step)
        image.header['panorama elevation range deg'] = '%.3f %.3f' % (cam.el_hi - cam.dims[1] * cam.step, cam.el_hi)
        image.header['panorama deg per pixel'] = '%.5f' % cam.step
        built = Camera.from_equirect(image, cam.pos, cam.ori)
        self.assertTrue(built.is_spherical())
        self.assertEqual(built.dims, cam.dims)
        self.assertAlmostEqual(built.az_lo, cam.az_lo, places=3)
        self.assertAlmostEqual(built.el_hi, cam.el_hi, places=3)
        self.assertAlmostEqual(built.step, cam.step, places=5)
        self.assertAlmostEqual(built.fov, cam.dims[1] * cam.step, places=5)

    def test_align_to_cloud(self):
        require_test_env(self, "opencv")
        from hylite.project.align import align_to_cloud_manual
        from hylite.project.basic import proj_sphere

        x, y = np.meshgrid(np.linspace(-10, 10, 15), np.linspace(-10, 10, 15))
        xyz = np.vstack([x.ravel(), y.ravel(), np.zeros_like(x.ravel())]).T
        cloud = hylite.HyCloud(xyz)

        step = 0.1
        dims = (800, 800)
        cam = Camera(np.array([0.0, 0.0, 40.0]), np.array([5.0, -3.0, 12.0]), 'sphere',
                     dims[1] * step, dims, step=step, az_lo=-40.0, el_hi=40.0)
        pp, vis = proj_sphere(xyz, cam.pos, cam.ori, cam.az_lo, cam.el_hi, cam.step, cam.dims)
        self.assertGreater(np.sum(vis), 20)

        idx = np.linspace(0, np.sum(vis) - 1, 20).astype(int)
        pixels = pp[vis, :][idx, :2]
        points = np.arange(xyz.shape[0])[vis][idx]
        cam_est, err = align_to_cloud_manual(cloud, cam, points, pixels)

        self.assertAlmostEqual(np.linalg.norm(cam.pos - cam_est.pos), 0, 3)
        self.assertAlmostEqual(np.sin(np.deg2rad(np.linalg.norm(cam.ori - cam_est.ori))), 0, 3)
        self.assertLess(err, 0.05)
        self.assertAlmostEqual(cam_est.az_lo, cam.az_lo)
        self.assertAlmostEqual(cam_est.el_hi, cam.el_hi)
        self.assertAlmostEqual(cam_est.step, cam.step)

    def test_scene_backproject(self):
        require_test_env(self, "default")
        cam = self._camera()
        pixels = [(40, 60, 0.2), (80, 40, 0.5), (120, 90, 0.8)]
        xyz = []
        for x, y, _ in pixels:
            az = cam.az_lo + (x + 0.5) * cam.step
            el = cam.el_hi - (y + 0.5) * cam.step
            xyz.append(_sphere_ray(az, el))
        cloud = HyCloud(np.vstack(xyz))

        image = HyImage(np.zeros((cam.dims[0], cam.dims[1], 1)))
        image.set_wavelengths(np.array([10000.0]))
        for x, y, value in pixels:
            image.data[x, y, 0] = value

        pth = mkdtemp()
        try:
            scene = HyScene('sphere', pth)
            scene.construct(image, cloud, cam, s=1, occ_tol=0, maxf=0, bf=False, vb=False)
            self.assertGreater(scene.pmap.point_count(), 0)
            self.assertGreater(scene.pmap.pixel_count(), 0)
            out = scene.push_to_cloud(0, method='closest')
            for i, (_, _, value) in enumerate(pixels):
                self.assertAlmostEqual(float(out.data[i, 0]), value, places=5)
        finally:
            shutil.rmtree(pth)


if __name__ == '__main__':
    unittest.main()
