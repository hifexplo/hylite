import os
import unittest

import numpy as np
from hylite import io
from tests._support import TEST_DATA, require_test_env, upgrade_test_env


class TestCorrect(unittest.TestCase):
    def test_correct_path_absorption(self):
        require_test_env(self, "lite")
        from hylite.correct.illumination import correct_path_absorption
        image = io.load(os.path.join(TEST_DATA, "image.hdr"))
        cloud = io.load(os.path.join(TEST_DATA, "hypercloud.hdr"))
        for D in [image, cloud]:
            Xhc = correct_path_absorption(D, atabs=2200., vb=False)
            self.assertEqual(D.data.shape, Xhc.data.shape)
            self.assertGreaterEqual(np.nanmin(Xhc.data), 0.0)
            self.assertLessEqual(np.nanmax(Xhc.data), 1.0)

    def test_panel(self):
        require_test_env(self, "lite")
        image = io.load(os.path.join(TEST_DATA, "image.hdr"))
        from hylite.correct import Panel
        from hylite.reference.spectra import R90
        rad = np.nanmean(image.data[:10, :10, :], axis=(0, 1))
        P = Panel(R90, rad, strict=True, wavelengths=image.get_wavelengths())

        if not upgrade_test_env("default"):
            return
        P.quick_plot()


class TestHullCorrection(unittest.TestCase):
    def test_hull(self):
        require_test_env(self, "lite")
        from hylite.correct import get_hull_corrected
        image = io.load(os.path.join(TEST_DATA, "image.hdr"))
        cloud = io.load(os.path.join(TEST_DATA, "image.hdr"))
        Xhc = get_hull_corrected(image.data, vb=False)
        self.assertGreaterEqual(np.nanmin(Xhc), 0.0)
        self.assertLessEqual(np.nanmax(Xhc), 1.0)
        self.assertEqual(image.data.shape, Xhc.shape)
        for D in [image, cloud]:
            Xhc = get_hull_corrected(D, vb=False)
            self.assertEqual(image.data.shape, Xhc.data.shape)
            self.assertGreaterEqual(np.nanmin(Xhc.data), 0.0)
            self.assertLessEqual(np.nanmax(Xhc.data), 1.0)


class TestEqualize(unittest.TestCase):
    def test_hist_eq(self):
        require_test_env(self, "basic")
        from hylite.correct.equalize import hist_eq

        ref = np.linspace(0, 1, 256)
        src = np.random.RandomState(0).rand(32, 32)
        matched = hist_eq(src, ref)
        self.assertEqual(matched.shape, src.shape)
        self.assertEqual(matched.dtype, src.dtype)

        data = np.linspace(0, 1, 100)
        np.testing.assert_allclose(hist_eq(data, data), data)

        # NaNs in source are preserved
        src_nan = src.copy()
        src_nan[0, 0] = np.nan
        matched_nan = hist_eq(src_nan, ref)
        self.assertTrue(np.isnan(matched_nan[0, 0]))

    def test_norm_eq(self):
        require_test_env(self, "basic")
        from hylite.correct.equalize import norm_eq

        adj = np.random.RandomState(1).rand(10, 10, 3)
        adj_s = adj[2:8, 2:8, :]
        ref_s = np.random.RandomState(2).rand(6, 6, 3) * 2 + 0.5

        out = norm_eq(adj, adj_s, ref_s, per_band=True, inplace=False)
        np.testing.assert_allclose(np.nanmean(out[2:8, 2:8, :], axis=(0, 1)),
                                   np.nanmean(ref_s, axis=(0, 1)), rtol=1e-5)
        np.testing.assert_allclose(np.nanstd(out[2:8, 2:8, :], axis=(0, 1)),
                                   np.nanstd(ref_s, axis=(0, 1)), rtol=1e-5)
        self.assertFalse(np.allclose(out, adj))

        adj_copy = adj.copy()
        norm_eq(adj_copy, adj_s, ref_s, per_band=False, inplace=True)
        self.assertFalse(np.allclose(adj_copy, adj))


if __name__ == '__main__':
    unittest.main()
