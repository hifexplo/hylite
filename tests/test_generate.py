import unittest
import numpy as np

class MyTestCase(unittest.TestCase):
    def test_generate(self):
        from hylite.reference import genImage, randomSpectra
        im,A = genImage()
        self.assertTrue(im.xdim() == 512 ) # check shape
        self.assertTrue(im.ydim() == 512)
        self.assertTrue(np.isfinite(im.data).all() ) # check all finite
        self.assertFalse((im.data == 0).all()) # check all non-zero
if __name__ == '__main__':
    unittest.main()
