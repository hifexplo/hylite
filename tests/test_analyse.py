import os
import unittest
from io import BytesIO
from tempfile import mkdtemp
import shutil

import numpy as np
import hylite
from hylite import io
from hylite.analyse.fourier import (
    HyFourier,
    FourierArchive,
    HYFOURIER_EXTENSION,
    FOURIER_ARCHIVE_EXTENSION,
    _parseSearchQuery,
    _sampleNames,
    _formatArchiveSampleName,
    _parseArchiveSampleName,
    _displayNameMatchesQuery,
    _archiveDisplayNameMatchesQuery,
    _nameMatch,
    _merge_or_search_results,
)
from hylite.analyse.mwl import MWL
from hylite.hycloud import HyCloud
from hylite.hyheader import HyHeader
from hylite.hyimage import HyImage
from hylite.hylibrary import HyLibrary

from tests._support import TEST_DATA, require_test_env, upgrade_test_env
from hylite._deps import optional


def _test_subset(source, n_points=300):
    """Return a manageable subset of each test dataset for faster unit tests."""
    if source.is_image():
        return source.crop(0, min(80, source.xdim()), 0, source.ydim())
    if hasattr(source, 'xyz') and source.xyz is not None:
        sub = source.copy()
        n = min(n_points, sub.data.shape[0])
        sub.data = sub.data[:n]
        sub.xyz = sub.xyz[:n]
        if sub.normals is not None:
            sub.normals = sub.normals[:n]
        if sub.rgb is not None:
            sub.rgb = sub.rgb[:n]
        return sub
    return source


class TestHyFourier(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not upgrade_test_env("default"):
            raise unittest.SkipTest("requires hylite[default]")
        cls.image = _test_subset(io.load(os.path.join(TEST_DATA, "image.hdr")))
        cls.cloud = _test_subset(io.load(os.path.join(TEST_DATA, "hypercloud.hdr")))
        cls.library = io.load(os.path.join(TEST_DATA, "library.csv"))

    def setUp(self):
        require_test_env(self, "default")

    def test_construct_and_reconstruct(self):
        for source in [self.image, self.cloud, self.library]:
            with self.subTest(type=type(source).__name__):
                hyfourier = HyFourier(source, padding="reflect", max_freq=0.25, vb=False)
                self.assertIsInstance(hyfourier, HyFourier)
                n_spectra = int(np.prod(source.data.shape[:-1]))
                self.assertEqual(hyfourier.data.shape[0], n_spectra)
                self.assertEqual(hyfourier.data.shape[2], 2)
                self.assertEqual(hyfourier.data.dtype, np.int16)
                self.assertGreater(hyfourier._scale, 0)
                self.assertEqual(hyfourier.original_shape, source.data.shape)
                self.assertEqual(len(hyfourier.get_wavelengths()), source.band_count())

                recon = hyfourier.toHyData()
                self.assertEqual(recon.data.shape, source.data.shape)

                mask = np.isfinite(source.data) & np.isfinite(recon.data)
                self.assertGreater(mask.sum(), 0)
                err = np.nanmax(np.abs(source.data[mask] - recon.data[mask]))
                self.assertLess(err, 0.35)

    def test_to_hydata(self):
        cases = (
            (self.library, HyLibrary),
            (self.cloud, HyCloud),
            (self.image, HyImage),
        )
        for source, cls in cases:
            with self.subTest(type=cls.__name__):
                hyfourier = HyFourier(source, padding='reflect', max_freq=0.25, vb=False)
                self.assertEqual(hyfourier.header['fourier source type'], cls.__name__)
                out = hyfourier.toHyData()
                self.assertIsInstance(out, cls)
                self.assertEqual(out.data.shape, source.data.shape)
                np.testing.assert_allclose(out.get_wavelengths(), source.get_wavelengths())
                if cls is HyCloud:
                    np.testing.assert_array_equal(out.xyz, source.xyz)
                mask = np.isfinite(source.data) & np.isfinite(out.data)
                self.assertGreater(mask.sum(), 0)
                err = np.nanmax(np.abs(source.data[mask] - out.data[mask]))
                self.assertLess(err, 0.35)

    def test_minima_mwl(self):
        feature_range = (2100.0, 2400.0)
        for source in [self.image, self.cloud, self.library]:
            with self.subTest(type=type(source).__name__):
                hyfourier = HyFourier(source, padding="cosine", max_freq=0.25, vb=False)
                mwl = hyfourier.minima(*feature_range, n_features=1)
                self.assertIsInstance(mwl, MWL)
                self.assertGreater(np.nanmax(mwl[0, 'pos']), feature_range[0])

    def test_save_load(self):
        tmp = mkdtemp()
        try:
            hyfourier = HyFourier(self.image, padding="reflect", max_freq=0.25, vb=False)
            out_path = os.path.join(tmp, "fourier_image")
            hyfourier.save(out_path)

            loaded = HyFourier.load(out_path)
            self.assertEqual(loaded.data.shape, hyfourier.data.shape)
            self.assertEqual(loaded.data.dtype, np.int16)
            np.testing.assert_array_equal(loaded.data, hyfourier.data)
            self.assertAlmostEqual(loaded._scale, hyfourier._scale)
            np.testing.assert_allclose(loaded.get_wavelengths(), hyfourier.get_wavelengths())

            recon = loaded.toHyData()
            self.assertIsInstance(recon, HyImage)
            self.assertEqual(recon.data.shape, self.image.data.shape)
            self.assertEqual(loaded.header['fourier source type'], 'HyImage')
        finally:
            shutil.rmtree(tmp)

    def test_kde_gaussians_and_grid(self):
        feature_range = (2100.0, 2400.0)
        hyfourier = HyFourier(self.library, padding="reflect", max_freq=0.25, vb=False)
        gaussians = hyfourier.kde(*feature_range, sigma=10.0)
        self.assertEqual(len(gaussians), hyfourier.n_spectra)
        self.assertGreater(len(gaussians[0]), 0)
        g0 = gaussians[0][0]
        self.assertIn('mu', g0)
        self.assertIn('sigma', g0)
        self.assertIn('weight', g0)
        self.assertIn(g0['kind'], ('minimum', 'maximum'))
        self.assertEqual(g0['sigma'], 10.0)
        self.assertGreater(g0['weight'], 0.0)

        grid = hyfourier.kde(*feature_range, grid=True, index=0)
        self.assertEqual(grid.shape, (len(hyfourier.get_wavelengths()),))
        self.assertGreater(np.nanmax(grid), 0.0)

        custom = np.linspace(feature_range[0], feature_range[1], 50)
        grid2 = hyfourier.kde(*feature_range, grid=custom, index=0)
        self.assertEqual(grid2.shape, custom.shape)

    def test_search_feature_and_name(self):
        hyfourier = HyFourier(self.library, padding="reflect", max_freq=0.25, vb=False)
        names, scores = hyfourier.search('2200', confidence=10.0, n_result=5)
        self.assertEqual(len(names), len(scores))
        self.assertGreater(len(names), 0)
        self.assertGreater(scores[0], 0.0)
        self.assertTrue(np.all(scores[:-1] >= scores[1:]))
        self.assertTrue(np.all((scores >= 0.0) & (scores <= 1.0)))

        sample_name = self.library.get_sample_names()[0]
        names2, scores2 = hyfourier.search(sample_name, n_result=5)
        self.assertIn(sample_name, names2)
        self.assertEqual(scores2[names2.index(sample_name)], 1.0)

        names3, scores3 = hyfourier.search('!9999', confidence=10.0, n_result=3)
        self.assertEqual(len(names3), 3)
        self.assertGreater(scores3[0], 0.0)

    def test_precompute_extrema_sidecars(self):
        tmp = mkdtemp()
        try:
            hyfourier = HyFourier(self.library, padding="reflect", max_freq=0.25, vb=False)
            out_path = os.path.join(tmp, "fourier_lib")
            sidecar = hyfourier.precomputeExtrema(kde_sigma=10.0, vb=False)
            self.assertIn('min_offsets', sidecar)
            self.assertEqual(len(sidecar['min_offsets']), hyfourier.n_spectra + 1)
            self.assertNotIn('inf_wavelength', sidecar)
            self.assertNotIn('min_left_width', sidecar)
            self.assertEqual(sidecar['min_wavelength'].dtype, np.float16)
            self.assertEqual(sidecar['min_prominence'].dtype, np.float16)
            self.assertIsNotNone(hyfourier._kde_sidecar)
            self.assertFalse(os.path.isfile(out_path + HYFOURIER_EXTENSION))

            hyfourier.save(out_path)
            archive_path = out_path + HYFOURIER_EXTENSION
            self.assertTrue(os.path.isfile(archive_path))

            loaded = HyFourier.load(out_path)
            self.assertIsNotNone(loaded._extrema_sidecar)
            self.assertIsNone(loaded._kde_sidecar)
            names, scores = loaded.search('2200', confidence=10.0, n_result=5)
            self.assertIsNotNone(loaded._kde_sidecar)
            self.assertEqual(len(names), len(scores))
            self.assertGreater(scores[0], 0.0)

            with np.load(archive_path, allow_pickle=False) as blob:
                self.assertFalse(any(k.startswith('kde_') for k in blob.files))
                self.assertFalse(any('left_width' in k or 'right_width' in k for k in blob.files))
                self.assertEqual(blob['ext_min_wavelength'].dtype, np.float16)
        finally:
            shutil.rmtree(tmp)

    def test_search_exclude_by_kind(self):
        features, _ = _parseSearchQuery('!2200')
        self.assertEqual(features[0]['kind'], 'minimum')
        self.assertTrue(features[0]['exclude'])
        features, _ = _parseSearchQuery('!^2200')
        self.assertEqual(features[0]['kind'], 'maximum')
        self.assertTrue(features[0]['exclude'])

    def test_name_match_and_scoring(self):
        name = '[silica] splib07b_Quartz_GDS31_BECKa_AREF'
        self.assertEqual(_nameMatch(name, ['quartz', 'beck']), 1.0)
        self.assertEqual(_nameMatch(name, ['quartz']), 1.0)
        self.assertEqual(_nameMatch(name, ['beck']), 1.0)
        self.assertEqual(_nameMatch(name, ['quartz', 'asdf']), 0.5)
        self.assertEqual(_nameMatch(name, ['kaolinite', 'beck']), 0.5)
        self.assertEqual(_nameMatch(name, ['kaolinite', 'asdf']), 0.0)

    def test_search_name_and_tokens(self):
        hyfourier = HyFourier(self.library, padding='reflect', max_freq=0.25, vb=False)
        sample_names = [str(name) for name in self.library.get_sample_names()]
        target = '2016_EH-005'
        self.assertIn(target, sample_names)

        names, scores = hyfourier.search('EH-005 EH', n_result=10)
        self.assertIn(target, names)
        self.assertEqual(scores[names.index(target)], 1.0)
        self.assertGreaterEqual(scores[0], scores[-1])

        partial_names, partial_scores = hyfourier.search('EH-005', n_result=len(sample_names))
        self.assertEqual(partial_scores[partial_names.index(target)], 1.0)
        self.assertGreater(partial_scores[0], 0.0)

    def test_search_or_subqueries(self):
        hyfourier = HyFourier(self.library, padding='reflect', max_freq=0.25, vb=False)
        sample_names = [str(name) for name in self.library.get_sample_names()]
        left, right = sample_names[0], sample_names[1]

        names, scores = hyfourier.search(f'{left}|{right}', n_result=5)
        self.assertIn(left, names)
        self.assertIn(right, names)
        self.assertGreater(scores[names.index(left)], 0.0)
        self.assertGreater(scores[names.index(right)], 0.0)

        merged = _merge_or_search_results([
            ([left, sample_names[2]], np.array([1.0, 0.5])),
            ([right, sample_names[3]], np.array([1.0, 0.5])),
        ], n_result=4)
        self.assertEqual(merged[0], [left, right, sample_names[2], sample_names[3]])

    def test_fourier_archive_or_search(self):
        archive = FourierArchive()
        archive['a'] = HyFourier(self.library, padding='reflect', max_freq=0.25, vb=False)
        sample_names = [str(name) for name in self.library.get_sample_names()]
        left, right = sample_names[0], sample_names[1]

        names, scores = archive.search(f'{left}|{right}', n_result=5)
        self.assertEqual(len(names), len(scores))
        qualified_left = _formatArchiveSampleName('a', left)
        qualified_right = _formatArchiveSampleName('a', right)
        self.assertIn(qualified_left, names)
        self.assertIn(qualified_right, names)

    def test_sample_names(self):
        header = HyHeader()
        header['sample names'] = np.array(['Alpha', 'Beta', 'Gamma'], dtype=str)
        header['group epidote'] = np.array([1, 2], dtype=np.int32)
        names = _sampleNames(header, 3, (3, 1, 10), (3, 1))
        self.assertEqual(names[0], 'Alpha')
        self.assertEqual(names[1], '[epidote] Beta')
        self.assertEqual(names[2], '[epidote] Gamma')

        image_names = _sampleNames(HyHeader(), 6, (2, 3, 10), (2, 3))
        self.assertEqual(image_names[0], '(0,0)')
        self.assertEqual(image_names[1], '(0,1)')
        self.assertEqual(image_names[3], '(1,0)')

        point_names = _sampleNames(HyHeader(), 4, (4, 10), (4,))
        self.assertEqual(point_names, ['S0', 'S1', 'S2', 'S3'])

        hyfourier = HyFourier(self.image, padding="reflect", max_freq=0.25, vb=False)
        cropped = _sampleNames(hyfourier.header, hyfourier.n_spectra, hyfourier.original_shape, hyfourier.spatial_shape)
        self.assertEqual(cropped[0], '(0,0)')
        self.assertEqual(len(cropped), hyfourier.n_spectra)

    def test_get_spectra_by_name(self):
        hyfourier = HyFourier(self.library, padding='reflect', max_freq=0.25, vb=False)
        full = hyfourier.getSpectra()
        names = _sampleNames(
            hyfourier.header, hyfourier.n_spectra, hyfourier.original_shape, hyfourier.spatial_shape,
        )
        display_name = names[0]

        subset = hyfourier.getSpectra(display_name)
        self.assertIsInstance(subset, HyLibrary)
        self.assertEqual(subset.sample_count(), 1)
        self.assertEqual(subset.get_sample_names(), [display_name])
        np.testing.assert_allclose(subset.data[0, 0], full.data[0, 0], rtol=1e-4, atol=1e-4, equal_nan=True)

        two = hyfourier.getSpectra(names[:2])
        self.assertEqual(two.sample_count(), 2)
        self.assertEqual(list(two.get_sample_names()), names[:2])

        with self.assertRaises(ValueError):
            hyfourier.getSpectra('not-a-real-name')

        by_exact = hyfourier.getSpectraByName(display_name)
        self.assertEqual(by_exact.sample_count(), 1)
        self.assertEqual(by_exact.get_sample_names(), [display_name])

        bare_name = display_name.split(' ', 1)[-1] if display_name.startswith('[') else display_name
        by_bare = hyfourier.getSpectraByName(bare_name)
        self.assertEqual(by_bare.sample_count(), 1)
        self.assertEqual(by_bare.get_sample_names()[0], display_name)

        with self.assertRaises(ValueError):
            hyfourier.getSpectraByName('am-21')

        with self.assertRaises(ValueError):
            hyfourier.getSpectraByName('not-a-real-name')

    def test_get_spectra_by_name_matching(self):
        display = '[topaz] splib07b_Topaz_HS184.3B_ASDNGb_AREF'
        bare = 'splib07b_Topaz_HS184.3B_ASDNGb_AREF'
        qualified = '(beck) [topaz] splib07b_Topaz_HS184.3B_BECKb_AREF'

        self.assertTrue(_displayNameMatchesQuery(display, display))
        self.assertTrue(_displayNameMatchesQuery(display, bare))
        self.assertTrue(_displayNameMatchesQuery(display, '[topaz] ' + bare))
        self.assertFalse(_displayNameMatchesQuery(display, 'Topaz'))
        self.assertFalse(_displayNameMatchesQuery(display, bare[:-4]))

        self.assertTrue(_archiveDisplayNameMatchesQuery('beck', display.replace('ASDNGb', 'BECKb'), qualified))
        self.assertTrue(_archiveDisplayNameMatchesQuery('beck', display.replace('ASDNGb', 'BECKb'), display.replace('ASDNGb', 'BECKb')))
        self.assertTrue(_archiveDisplayNameMatchesQuery('beck', display.replace('ASDNGb', 'BECKb'), bare.replace('ASDNGb', 'BECKb')))
        self.assertFalse(_archiveDisplayNameMatchesQuery('asdng', display, qualified))

    def test_fourier_archive(self):
        tmp = mkdtemp()
        try:
            archive = FourierArchive()
            for source in [self.library, self.image]:
                key = type(source).__name__.lower().replace('hy', '')
                archive[key] = HyFourier(source, padding='reflect', max_freq=0.25, vb=False)
                archive[key].precomputeExtrema(vb=False)

            out_path = os.path.join(tmp, 'multi')
            archive.save(out_path)
            fda_path = out_path + FOURIER_ARCHIVE_EXTENSION
            self.assertTrue(os.path.isfile(fda_path))

            loaded = FourierArchive.load(out_path)
            self.assertEqual(set(loaded.keys()), set(archive.keys()))
            with open(fda_path, 'rb') as fh:
                fda_bytes = fh.read()
            loaded_bytes = FourierArchive.load_bytes(fda_bytes)
            loaded_buffer = FourierArchive.load_from_buffer(BytesIO(fda_bytes))
            self.assertEqual(set(loaded_bytes.keys()), set(archive.keys()))
            np.testing.assert_array_equal(loaded_bytes['library'].data, archive['library'].data)
            np.testing.assert_array_equal(loaded_buffer['image'].data, archive['image'].data)

            with self.assertRaises(TypeError):
                FourierArchive.load_bytes('not-bytes')
            for key in archive:
                np.testing.assert_array_equal(loaded[key].data, archive[key].data)
                names, scores = loaded[key].search('2200', confidence=10.0, n_result=3)
                self.assertEqual(len(names), 3)
                self.assertGreater(scores[0], 0.0)

            with self.assertRaises(TypeError):
                archive['bad'] = self.library

            names, scores = archive.search('2200', confidence=10.0, n_result=5)
            self.assertEqual(len(names), len(scores))
            self.assertGreater(len(names), 0)
            for name in names:
                self.assertRegex(name, r'^\((library|image)\) ')
                key, inner = _parseArchiveSampleName(name)
                self.assertIn(key, archive)
                self.assertEqual(name, _formatArchiveSampleName(key, inner))

            top_name = names[0]
            subset = archive.getSpectra(top_name)
            self.assertIsInstance(subset, HyLibrary)
            self.assertEqual(subset.sample_count(), 1)

            key, inner = _parseArchiveSampleName(top_name)
            direct = archive[key].getSpectra(inner)
            np.testing.assert_allclose(subset.data, direct.data, equal_nan=True)

            lib_key = 'library'
            lib_names = _sampleNames(
                archive[lib_key].header,
                archive[lib_key].n_spectra,
                archive[lib_key].original_shape,
                archive[lib_key].spatial_shape,
            )
            first_lib_name = lib_names[0]
            bare_name = first_lib_name.split(' ', 1)[-1] if first_lib_name.startswith('[') else first_lib_name

            by_qualified = archive.getSpectraByName(_formatArchiveSampleName(lib_key, first_lib_name))
            self.assertEqual(by_qualified.sample_count(), 1)
            self.assertEqual(by_qualified.get_sample_names()[0], first_lib_name)

            by_display = archive.getSpectraByName(first_lib_name)
            self.assertEqual(by_display.get_sample_names()[0], first_lib_name)

            by_bare = archive.getSpectraByName(bare_name)
            self.assertEqual(by_bare.get_sample_names()[0], first_lib_name)

            with self.assertRaises(ValueError):
                archive.getSpectraByName(first_lib_name[:8])

            with self.assertRaises(ValueError):
                archive.getSpectraByName('not-a-real-name')
        finally:
            shutil.rmtree(tmp)


class TestSAM(unittest.TestCase):
    def test_sam(self):
        require_test_env(self, "lite")
        from hylite.analyse.sam import SAM, spectral_angles
        from hylite import io

        image = io.load(os.path.join(TEST_DATA, "image.hdr"))
        cloud = io.load(os.path.join(TEST_DATA, "hypercloud.hdr"))

        for data in [image, cloud]:
            em1 = data.X(onlyFinite=True)[0]
            em2 = data.X(onlyFinite=True)[-1]

            ang = spectral_angles([em1, em2], data.X())
            self.assertTrue(np.isfinite(ang).any())

            em3 = data.X(onlyFinite=True)[-10]
            sam = SAM(data, [[em3], [em1, em2]])
            self.assertTrue(np.isfinite(sam.data).any())
            self.assertEqual(int(sam.X(onlyFinite=True)[0, 0]), 1)
            self.assertEqual(int(sam.X(onlyFinite=True)[-1, 0]), 1)
            self.assertEqual(int(sam.X(onlyFinite=True)[-10, 0]), 0)

            arr = np.vstack([em3, em2])[:, None, :]
            lib = hylite.HyLibrary(arr, lab=['EM1', 'EM2'], wav=data.get_wavelengths())
            sam = SAM(data, lib)
            self.assertTrue(np.isfinite(sam.data).any())
            self.assertEqual(int(sam.X(onlyFinite=True)[-1, 0]), 1)
            self.assertEqual(int(sam.X(onlyFinite=True)[-10, 0]), 0)


class TestUnmixing(unittest.TestCase):
    def test_unmixing(self):
        require_test_env(self, "lite")
        import hylite
        from hylite import io
        from hylite.analyse.unmixing import mix, unmix, endmembers

        image = io.load(os.path.join(TEST_DATA, "image.hdr"))
        cloud = io.load(os.path.join(TEST_DATA, "hypercloud.hdr"))

        has_default = upgrade_test_env("default")
        has_pysptools = optional("pysptools") is not None
        has_fcls = optional("cvxopt") is not None

        for data in [image, cloud]:
            em1 = data.X(onlyFinite=True)[0]
            em2 = data.X(onlyFinite=True)[-1]
            E = hylite.HyLibrary(np.vstack([em1, em2]),
                                 lab=['A', 'B'], wav=data.get_wavelengths())
            A = data.copy()
            A.data = np.random.uniform(size=(data.data.shape[:-1]) + (2,))
            A.data = A.data / np.sum(A.data, axis=-1)[..., None]

            X = mix(A, E)
            self.assertTrue(X.data.shape[-1] == E.data.shape[-1])

            if has_default:
                A2 = unmix(X, E, method='nnls')
                self.assertLess(np.mean(np.abs(A2.data - A.data)), 1e-4)

            if has_fcls:
                A2 = unmix(X, E, method='fcls')
                self.assertLess(np.mean(np.abs(A2.data - A.data)), 1e-4)

            if has_pysptools:
                for m in ['atgp', 'fippi', 'nfindr', 'ppi']:
                    em, ix = endmembers(X, 3, method=m)

                    if len(ix.shape) > 1:
                        self.assertLess(np.max(np.abs(em.data[0, 0, :] - X.data[*ix[0], :])), 1e-6)
                    else:
                        self.assertLess(np.max(np.abs(em.data[0, 0, :] - X.data[ix[0], :])), 1e-6)

                    self.assertLess(min(np.mean(np.abs(em1 - em.data[:, 0, :])),
                                        np.mean(np.abs(em2 - em.data[:, 0, :]))), 0.05)


class TestMWL(unittest.TestCase):
    def test_mwl(self):
        require_test_env(self, "lite")
        from hylite import io
        from hylite.analyse.mwl import minimum_wavelength, colourise_mwl

        image = io.load(os.path.join(TEST_DATA, "image.hdr"))
        image.data[:50, :, :] = np.nan
        cloud = io.load(os.path.join(TEST_DATA, "image.hdr"))

        image.header.set_sample_points('A', [(20, 15)])
        image.header.set_sample_points('B', [(80, 15)])
        image.header.set_sample_points('C', [(140, 15)])
        image.header['class names'] = ['A', 'B', 'C']
        from hylite.hylibrary import from_indices
        lib = from_indices(image,
                           [image.header.get_sample_points(n)[0] for n in image.header.get_class_names()],
                           names=image.header.get_class_names(),
                           s=5)

        for D in [lib, image, cloud]:
            mwl = minimum_wavelength(D, 2100., 2380.,
                                     trend='hull', method='minmax',
                                     n=1, nthreads=1, vb=False, xtol=0.1, ftol=0.1)
            self.assertGreater(np.nanmax(mwl.model.data[..., 1]), 2100.)

            mwl = minimum_wavelength(D, 2100., 2380., trend='hull', method='quad', n=2, nthreads=1, vb=True, xtol=0.1,
                                     ftol=0.1)
            self.assertGreater(np.nanmax(mwl.model.data[..., 1]), 2100.)

            mwl = minimum_wavelength(D, 2100., 2380., trend='hull', method='poly', n=2, nthreads=1, vb=True, xtol=0.1,
                                     ftol=0.1)
            self.assertGreater(np.nanmax(mwl.model.data[..., 1]), 2100.)

            mwl = minimum_wavelength(D, 2100., 2380., trend='hull', method='gauss', n=1, nthreads=1, vb=True, xtol=0.1, ftol=0.1)
            self.assertGreater(np.nanmax(mwl.model.data[..., 1]), 2100.)

            M = minimum_wavelength(D, minw=2100., maxw=2400., sym=False, method='gauss', n=3, vb=True, xtol=0.1, ftol=0.1)
            mask = np.isfinite(M[0, 0])

            M.sortByDepth()
            self.assertTrue((M[0, 'depth'][mask] >= M[1, 'depth'][mask]).all())
            self.assertTrue((M[2, 'depth'][mask] >= M[2, 'depth'][mask]).all())

            M.sortByPos()
            self.assertTrue((M[0, 'pos'][mask] <= M[1, 'pos'][mask]).all())
            self.assertTrue((M[2, 'pos'][mask] <= M[2, 'pos'][mask]).all())

            deepest = M.deepest(2100., 2400.)
            M.sortByDepth()
            self.assertTrue((np.nan_to_num(deepest.data) == np.nan_to_num(M[0].data)).all())

            closest = M.closest(2100., depth_cutoff=0)
            M.sortByPos()
            self.assertTrue((np.nan_to_num(closest.data) == np.nan_to_num(M[0].data)).all())

            closest = M.closest(2200., valid_range=(2195., 2205.))
            self.assertTrue(np.nanmin(closest.data[..., 1]) >= 2195., "Error - %s" % np.nanmin(closest.data[..., 1]))
            self.assertTrue(np.nanmin(closest.data[..., 1]) <= 2205., "Error - %s" % np.nanmin(closest.data[..., 1]))

            M = minimum_wavelength(D, minw=2100., maxw=2400., sym=False, method='gauss', n=3, vb=True, xtol=0.1,
                                   ftol=0.1, minima=False)
            M.sortByDepth()
            self.assertGreater(np.nanmax(M[0, 'depth']), 0)

            if upgrade_test_env("default"):
                colourise_mwl(M.closest(2200., valid_range=(2150., 2230.)))[1].quick_plot()

                M1 = minimum_wavelength(D, minw=2100., maxw=2400., sym=True, method='gauss', n=3, vb=True, nthreads=-1, xtol=0.1, ftol=0.1)
                M1.evaluate()
                M1.classify(5, nf=3)
                M1.residual()
                M1.quick_plot()
                M1.quick_plot(step=3)


class TestDecisionTree(unittest.TestCase):
    def test_decision_tree(self):
        require_test_env(self, "basic")
        from hylite.analyse.dtree import decision_tree

        layer0 = np.array([[True, False], [True, True]])
        layer1 = np.array([[False, True], [True, False]])
        out, names = decision_tree(
            [layer0, layer1],
            {(True, False): "A", (True, True): "B"},
        )
        self.assertEqual(names, ["Unknown", "A", "B"])
        self.assertEqual(out[0, 0], names.index("A"))
        self.assertEqual(out[1, 0], names.index("B"))
        self.assertEqual(out[0, 1], 0)  # (False, *) unmatched -> Unknown
        self.assertEqual(out[1, 1], names.index("A"))  # (True, False)

        # None skips a layer when matching a label key
        layer2 = np.array([[True, False], [False, True]])
        out2, names2 = decision_tree(
            [layer0, layer1, layer2],
            {(True, None, True): "C"},
        )
        self.assertEqual(out2[1, 1], names2.index("C"))


class TestIndices(unittest.TestCase):
    def test_band_ratio(self):
        require_test_env(self, "lite")
        from hylite import io
        from hylite.analyse.indices import band_ratio, NDVI, SKY, SHADE

        image = io.load(os.path.join(TEST_DATA, "image.hdr"))
        cloud = io.load(os.path.join(TEST_DATA, "hypercloud.hdr"))

        for D in [image, cloud]:
            ratio = band_ratio(D, 800.0, 670.0)
            self.assertEqual(ratio.band_count(), 1)
            self.assertEqual(ratio.data.shape[:-1], D.data.shape[:-1])
            self.assertIn("/", ratio.get_band_names()[0])

            ndvi = NDVI(D)
            self.assertEqual(ndvi.get_band_names()[0], "NDVI")
            finite = np.isfinite(ndvi.data)
            if finite.any():
                self.assertGreaterEqual(np.nanmax(ndvi.data[finite]), -1.0)
                self.assertLessEqual(np.nanmax(ndvi.data[finite]), 1.0)

            sky = SKY(D)
            self.assertEqual(sky.band_count(), 1)
            shade = SHADE(D)
            self.assertEqual(shade.band_count(), 1)

        # combined numerator from discrete bands
        combo = band_ratio(image, [100, 200], 300)
        self.assertEqual(combo.band_count(), 1)
        self.assertTrue(np.isfinite(combo.data).any())


if __name__ == '__main__':
    unittest.main()
