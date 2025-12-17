import unittest
import os
from pathlib import Path
from hylite import io
import numpy as np

# TASNIM TO IMPLEMENT THIS
class MyTestCase(unittest.TestCase):
    def test_baseline(self):
        #image = io.load(os.path.join(os.path.join(str(Path(__file__).parent.parent), "test_data"), "image.hdr"))
        #cloud = io.load(os.path.join(os.path.join(str(Path(__file__).parent.parent), "test_data"), "hypercloud.hdr"))

        #self.assertEqual(pca.band_count(), 10)
        #self.assertLess( np.nanmax( np.abs( pca.data - pca2.data ) ), 1e-4 )

        #self.assertLess( np.nanmin(hc.data), 1 ) # assert some values are less than one
        #self.assertGreaterEqual( np.nanmin(hc.data), 0 ) # no negatives!

        pass

    def test_absorbance(self):
        from hylite.transform import convertToAbsorbance
        image = io.load(os.path.join(os.path.join(str(Path(__file__).parent.parent), "test_data"), "image.hdr"))
        cloud = io.load(os.path.join(os.path.join(str(Path(__file__).parent.parent), "test_data"), "hypercloud.hdr"))
        lib = io.load(os.path.join(os.path.join(str(Path(__file__).parent.parent), "test_data"), "library.csv"))

        for D in [image, cloud, lib]:
            absorbance = convertToAbsorbance(D, method='kubelka-munk')
            self.assertIsNotNone(absorbance)
            self.assertEqual(absorbance.band_count(), D.band_count())
            self.assertEqual(absorbance.data.shape[:-1], D.data.shape[:-1])
            self.assertGreaterEqual(np.nanmin(absorbance.data), 0)
            self.assertGreater(np.sum(np.isfinite(absorbance.data)), 0)


if __name__ == '__main__':
    unittest.main()
