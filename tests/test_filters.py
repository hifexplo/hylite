import unittest
import os
from pathlib import Path
from hylite import io
import numpy as np

class MyTestCase(unittest.TestCase):
    def test_PCA_MNF(self):
        image = io.load(os.path.join(os.path.join(str(Path(__file__).parent.parent), "test_data"), "image.hdr"))
        cloud = io.load(os.path.join(os.path.join(str(Path(__file__).parent.parent), "test_data"), "hypercloud.hdr"))

        # run mnf on clouds and images
        from hylite.filter import MNF, PCA, from_loadings
        img_mask = np.full( (image.xdim(), image.ydim()), False )
        cld_mask = np.full( (cloud.point_count()), False )
        img_mask[:, :int(img_mask.shape[1]/2) ] = True
        cld_mask[:int(cloud.point_count() / 2)] = True
        for data, mask  in zip( [image, cloud], [img_mask, cld_mask]):

            # test PCA
            pca, w, m = PCA( data, bands=10, mask=mask)
            pca2 = from_loadings( data, w, m )
            self.assertEqual(pca.band_count(), 10)
            self.assertLess( np.nanmax( np.abs( pca.data - pca2.data ) ), 1e-4 )

            # run MNF
            mnf, w, m = MNF( data, bands=10, mask=mask)
            mnf2 = from_loadings(data, w, m)
            self.assertEqual( mnf.band_count(), 10 )
            self.assertLess(np.nanmax(np.abs(mnf.data - mnf2.data)), 1e-4)

            # run MNF denoising
            denoise, w, m = MNF( data, bands=10, denoise=True )
            self.assertEqual(denoise.band_count(), data.band_count())

        # test hull correction
        from hylite.correct import get_hull_corrected
        for d in [image, cloud]:
            hc = get_hull_corrected(d) # run hull correction
            self.assertLessEqual( np.nanmax(hc.data),  1 ) # assert results are not > 1
            self.assertLess( np.nanmin(hc.data), 1 ) # assert some values are less than one
            self.assertGreaterEqual( np.nanmin(hc.data), 0 ) # no negatives!
if __name__ == '__main__':
    unittest.main()
