import os
import unittest
from pathlib import Path
from tempfile import mkdtemp
import shutil

import hylite
import numpy as np
from hylite import io
from tests._support import TEST_DATA, require_test_env


class TestTransform(unittest.TestCase):
    def test_PCA_MNF(self):
        require_test_env(self, "default")
        import hylite

        image = io.load(os.path.join(TEST_DATA, "image.hdr"))
        cloud = io.load(os.path.join(TEST_DATA, "hypercloud.hdr"))
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


class TestFilterAndSample(unittest.TestCase):
    def test_overlay(self):
        require_test_env(self, "basic")
        from hylite.transform import overlay
        from tests import genImage

        img1 = genImage(dimx=12, dimy=10, nbands=4)
        img2 = genImage(dimx=12, dimy=10, nbands=4)
        img1.data[:] = 1.0
        img2.data[:] = 3.0

        combined, std = overlay([img1, img2], method="mean")
        self.assertEqual(combined.data.shape, img1.data.shape)
        np.testing.assert_allclose(combined.data, 2.0)
        np.testing.assert_allclose(std.data, 1.0)

        img2.data[0, 0, 0] = 5.0
        median, _ = overlay([img1, img2], method="median")
        self.assertAlmostEqual(median.data[0, 0, 0], 3.0)

    def test_boost_saturation(self):
        require_test_env(self, "default")
        from hylite.transform import boost_saturation
        import hylite

        image = io.load(os.path.join(TEST_DATA, "image.hdr"))
        rgb = boost_saturation(image, hylite.RGB, flip=True, sat=0.8)
        self.assertEqual(rgb.band_count(), 3)
        self.assertEqual(rgb.data.shape[:2], image.data.shape[:2])
        finite = np.isfinite(rgb.data)
        self.assertTrue(finite.any())
        self.assertGreaterEqual(np.nanmin(rgb.data[finite]), 0.0)
        self.assertLessEqual(np.nanmax(rgb.data[finite]), 1.0)

    def test_resample(self):
        require_test_env(self, "basic")
        from hylite.transform import Resample, ASTER, SENTINEL

        image = io.load(os.path.join(TEST_DATA, "image.hdr"))
        custom = Resample([(500.0, 600.0), (700.0, 800.0)])
        out = custom.apply(image)
        self.assertEqual(out.band_count(), 2)
        self.assertEqual(out.data.shape[:2], image.data.shape[:2])
        np.testing.assert_allclose(out.get_wavelengths(), [550.0, 750.0])

        band = custom.get_band(image, 1)
        self.assertEqual(band.shape, image.data.shape[:2])
        i0 = image.get_band_index(500.0, thresh=np.inf)
        i1 = image.get_band_index(600.0, thresh=np.inf)
        sl = image.data[..., i0:i1 + 1]
        n = np.isfinite(sl).sum(axis=-1)
        expected = np.full(sl.shape[:-1], np.nan, dtype=np.float64)
        np.divide(np.nansum(sl, axis=-1), n, out=expected, where=n > 0)
        np.testing.assert_allclose(band, expected)
        # out-of-range interval is nan, not an empty-slice mean
        np.testing.assert_array_equal(
            Resample([(8000.0, 9000.0)]).get_band(image, 1),
            np.full(image.data.shape[:2], np.nan),
        )

        aster = ASTER.apply(image)
        self.assertEqual(aster.band_count(), len(ASTER.bands))
        sentinel = SENTINEL.apply(image)
        self.assertEqual(sentinel.band_count(), len(SENTINEL.bands))

        from tests import genCloud
        cloud = genCloud(npoints=40, nbands=10)
        cloud_out = custom.apply(cloud)
        self.assertEqual(cloud_out.band_count(), 2)
        self.assertEqual(cloud_out.data.shape[0], cloud.data.shape[0])
        np.testing.assert_allclose(cloud_out.get_wavelengths(), [550.0, 750.0])

        lib = io.load(os.path.join(TEST_DATA, "library.csv"))
        lib_out = custom.apply(lib)
        self.assertEqual(lib_out.band_count(), 2)
        self.assertEqual(lib_out.data.shape[:-1], lib.data.shape[:-1])


class TestAbsorbance(unittest.TestCase):
    def test_absorbance(self):
        require_test_env(self, "default")
        from hylite.transform import convertToAbsorbance
        image = io.load(os.path.join(TEST_DATA, "image.hdr"))
        cloud = io.load(os.path.join(TEST_DATA, "hypercloud.hdr"))
        lib = io.load(os.path.join(TEST_DATA, "library.csv"))
        for D in [image, cloud, lib]:
            absorbance = convertToAbsorbance(D, method='kubelka-munk')
            self.assertIsNotNone(absorbance)
            self.assertEqual(absorbance.band_count(), D.band_count())
            self.assertEqual(absorbance.data.shape[:-1], D.data.shape[:-1])
            self.assertGreaterEqual(np.nanmin(absorbance.data), 0)
            self.assertGreater(np.sum(np.isfinite(absorbance.data)), 0)

if __name__ == '__main__':
    unittest.main()
