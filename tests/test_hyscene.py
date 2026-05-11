import unittest

from tempfile import mkdtemp
import shutil

import hylite
from hylite import HyScene, HyCloud, HyImage
from hylite.project import Camera, Pushbroom, blend_scenes, push_geomattr, get_blend_weights
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
        fov = 50.
        cp = np.zeros( (100, 3) )
        cp[:, 0] +=  np.linspace(-10, 10, 100)
        cp[:, 1] +=  np.linspace(-10, 10, 100)
        cp[:, 2] = 30.
        co = np.zeros( (100,3) )
        self.track = Pushbroom( cp, co, fov / dims[0], fov / dims[0], (dims[0], cp.shape[0]) )

    def test_construction(self):
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
            O = blend_scenes([S,S2], w, (0,-1), ooc=True )
            self.assertEqual( O.point_count(), 2500 )
            self.assertEqual(O.band_count(), 3)
            self.assertAlmostEqual( np.nanmax(O.data), max( np.nanmax(self.image.data), np.nanmax(self.swath.data) ), 2 ) # check normalisation during blending / averaging is correct
            
        except:
            shutil.rmtree(pth)  # delete temp directory
            self.assertFalse(True, "Error - could not construct HyScene instance")

        shutil.rmtree(pth)  # delete temp directory




if __name__ == '__main__':
    unittest.main()
