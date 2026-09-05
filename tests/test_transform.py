import os
import unittest
from pathlib import Path
from tempfile import mkdtemp
import shutil

import hylite
import numpy as np
from hylite import io
from tests._support import TEST_DATA, require_test_env


def _component_noise(data):
    """Normalised roughness of each band along axis 0 (higher = noisier)."""
    data = np.asarray(data)
    spatial = tuple(range(data.ndim - 1))
    delta = np.nanmean(np.abs(np.diff(data, axis=0)), axis=spatial)
    return delta / (np.nanstd(data, axis=spatial) + 1e-12)


def _assert_mnf_noise_increases(testcase, data):
    """First MNF components must be cleaner than later ones (Green et al. 1988)."""
    cn = _component_noise(data)
    testcase.assertLess(cn[0], cn[-1], "first MNF component is noisier than the last")
    testcase.assertNotEqual(int(np.argmax(cn)), 0, "first MNF component is the noisiest")
    k = max(1, len(cn) // 5)
    testcase.assertLess(
        np.median(cn[:k]),
        np.median(cn[-k:]),
        "early MNF components are not cleaner than late ones",
    )


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
            
            methods = ['spectral']
            if isinstance(X, hylite.HyImage) or (isinstance(X, np.ndarray) and X.ndim == 3):
                methods.append('spatial')

            for method in methods:
                noise = NoiseWhitener(noiseMethod=method)
                noise.fit(X)
                if method == 'spectral':
                    off = noise.Wn_.copy()
                    np.fill_diagonal(off, 0.0)
                    self.assertLess(np.max(np.abs(off)), 1e-12)
                    self.assertTrue(np.all(np.diag(noise.Wn_) > 0))

                mnf = MNF(n_components=n, normalise=False, subsample=1, noise=noise).fit(X)
                Xt = mnf.transform(X)
                Xtt = mnf.inverse_transform(Xt)

                self.assertTrue(isinstance(Xt, type(X)))
                data_in = X if isinstance(X, np.ndarray) else X.data
                data_t = Xt if isinstance(Xt, np.ndarray) else Xt.data
                data_tt = Xtt if isinstance(Xtt, np.ndarray) else Xtt.data
                self.assertEqual(data_t.shape[-1], n)
                self.assertLess(np.nanmax(np.abs(data_in - data_tt)), 1e-4)
                _assert_mnf_noise_increases(self, data_t)
                if method == 'spectral':
                    # padding bug put ~all of PC1 into bands 0–1; smoothness residual must not
                    self.assertLess(np.sum(mnf._pca.components_[0, :2] ** 2), 0.5)

    def test_spectral_noise_sigma(self):
        """Spectral whitening recovers per-band σ from a smooth cube plus white noise."""
        require_test_env(self, "default")
        from hylite.transform import NoiseWhitener, MNF

        rng = np.random.default_rng(0)
        h, w, b = 32, 40, 60
        t = np.linspace(0, 2 * np.pi, b)
        signal = 0.4 + 0.3 * np.sin(t)
        yy, xx = np.mgrid[0:h, 0:w]
        cube = (0.5 + 0.5 * np.sin(xx / 7.0) * np.cos(yy / 9.0))[..., None] * signal
        true_sigma = 0.01 + 0.02 * np.linspace(0, 1, b) ** 2
        cube = cube + rng.normal(0.0, 1.0, cube.shape) * true_sigma

        noise = NoiseWhitener(noiseMethod='spectral', subsample=1).fit(cube)
        np.testing.assert_allclose(noise.estimate[2:-2], true_sigma[2:-2], rtol=0.15)
        off = noise.Wn_.copy()
        np.fill_diagonal(off, 0.0)
        self.assertLess(np.max(np.abs(off)), 1e-12)

        mnf = MNF(n_components=5, subsample=1, noise=noise).fit(cube)
        Xt = mnf.transform(cube)
        _assert_mnf_noise_increases(self, Xt)
        cn = _component_noise(Xt)
        self.assertLess(cn[0], 0.3)
        self.assertGreater(cn[-1], 0.8)


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
