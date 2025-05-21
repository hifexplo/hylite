import unittest
import numpy as np
from tests import genImage
import hylite

class TestHyImage(unittest.TestCase):
    def test_image(self):

        # test constructor
        image = hylite.HyImage(np.zeros((25,25,5)), wav=np.arange(5)*100)
        self.assertListEqual(list(image.get_wavelengths()), list(np.arange(5)*100))

        # create test image
        image = genImage(dimx = 1464, dimy=401, nbands=10)

        # run plotting functions
        image.quick_plot( (0,1,2), vmin=2, vmax=98 )
        image.quick_plot( 0 )

        self.assertEqual(image.xdim(), 1464)
        self.assertEqual(image.ydim(), 401)
        self.assertEqual(image.band_count(), 10)
        self.assertEqual(image.aspx(),  401 / 1464)

        # todo - add test code for georeferencing code
        # get_extent, set_projection, set_projection_EPSG, get_projection_EPSG, pix_to_world, world_to_pix

        image.flip(axis='y')
        image.data[10,10,:] = np.nan
        image.fill_holes()
        self.assertEqual( np.isfinite( image.data ).all(), True )
        image.blur()

        # resize
        nx,ny = int(1464/2), int(401/2)
        image.resize(newdims=(nx, ny))
        self.assertEqual(image.xdim(), nx)
        self.assertEqual(image.ydim(), ny)
        self.assertEqual(image.band_count(), 10)

        # extract features
        k, d = image.get_keypoints( band=0 )
        src, dst = image.match_keypoints(k,k,d,d)
        self.assertGreater(len(src), 0 ) # make sure there are some matches...

        # masking
        image.mask( np.sum(image.data,axis=2) > 0.75 )
        self.assertEqual(np.isfinite(image.data).all(), False)

if __name__ == '__main__':
    unittest.main()
