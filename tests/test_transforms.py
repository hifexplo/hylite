import unittest
import os
from pathlib import Path
from hylite import io
import numpy as np

class MyTestCase(unittest.TestCase):
    def test_PCA_MNF(self):
        import hylite

        image = io.load(os.path.join(os.path.join(str(Path(__file__).parent.parent), "test_data"), "image.hdr"))
        cloud = io.load(os.path.join(os.path.join(str(Path(__file__).parent.parent), "test_data"), "hypercloud.hdr"))
        image.data[0,0,10] = np.nan # ensure there are at least some nans
        cloud.data[0,10] = np.nan
        numpy1 = image.data
        numpy2 = cloud.data

        # run mnf on clouds and images
        from hylite.transform import MNF, PCA, NoiseWhitener
        for X  in [cloud,image,numpy1,numpy2]:
            # get shape
            if isinstance(X, np.ndarray):
                n = X.shape[-1]
            else:
                n = X.data.shape[-1]

            # test PCA
            pca = PCA(n_components=n, normalise=False, subsample=1).fit(X)
            Xt = pca.transform(X)
            Xtt = pca.inverse_transform(Xt) # back-transform

            self.assertTrue( isinstance( Xt, type(X) ) )
            if isinstance(Xt, np.ndarray):
                self.assertTrue( Xt.shape[-1] == n )
                self.assertLess( np.nanmax( np.abs( X - Xtt ) ), 1e-4 )
            else:
                self.assertTrue( Xt.data.shape[-1] == n )
                self.assertLess( np.nanmax( np.abs( X.data - Xtt.data ) ), 1e-4 )
            
            # fit noise
            noise = NoiseWhitener(noiseMethod='spectral')
            noise.fit(X)
            if isinstance(X, hylite.HyImage): # also try spatial on image data
                noise = NoiseWhitener(noiseMethod='spatial')
                noise.fit(X)
            
            # test MNF
            mnf = MNF(n_components=n, normalise=False, subsample=1, noise=noise).fit(X)
            Xt = mnf.transform(X)
            Xtt = mnf.inverse_transform(Xt) # back-transform

            self.assertTrue( isinstance( Xt, type(X) ) )
            if isinstance(Xt, np.ndarray):
                self.assertTrue( Xt.shape[-1] == n )
                self.assertLess( np.nanmax( np.abs( X - Xtt ) ), 1e-4 )
            else:
                self.assertTrue( Xt.data.shape[-1] == n )
                self.assertLess( np.nanmax( np.abs( X.data - Xtt.data ) ), 1e-4 )

if __name__ == '__main__':
    unittest.main()
