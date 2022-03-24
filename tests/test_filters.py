import unittest
import os
from pathlib import Path
from hylite import io

class MyTestCase(unittest.TestCase):
    def test_PCA_MNF(self):
        image = io.load(os.path.join(os.path.join(str(Path(__file__).parent.parent), "test_data"), "image.hdr"))
        cloud = io.load(os.path.join(os.path.join(str(Path(__file__).parent.parent), "test_data"), "image.hdr"))

        # run mnf on clouds and images
        from hylite.filter import MNF
        from hylite.filter import PCA
        for data in [image, cloud]:

            # test PCA
            pca, w = PCA( data, bands=10)
            self.assertEqual(pca.band_count(), 10)

            # run MNF
            mnf, w = MNF( data, bands=10)
            self.assertEqual( mnf.band_count(), 10 )

            # run MNF denoising
            denoise, w = MNF( data, bands=10, denoise=True )
            self.assertEqual(denoise.band_count(), data.band_count())

if __name__ == '__main__':
    unittest.main()
