import os
import shutil
import unittest
from tempfile import mkdtemp

import numpy as np

from hylite import io
from tests import genCloud, genImage
from tests._support import TEST_DATA, require_test_env


def _scale_data(data, factor=1.0):
    out = data.copy(data=True)
    out.data = out.data * factor
    return out


def _copy_and_scale(in_path, out_path, factor=2.0):
    data = io.load(in_path)
    out = _scale_data(data, factor=factor)
    io.save(out_path, out)


class TestMultiprocessing(unittest.TestCase):
    def test_split_merge(self):
        require_test_env(self, "basic")
        from hylite.multiprocessing import _split, _merge

        cloud = genCloud(npoints=101, nbands=4)
        chunks = _split(cloud, 4)
        self.assertEqual(len(chunks), 4)
        self.assertEqual(sum(c.point_count() for c in chunks), cloud.point_count())

        merged = _merge(chunks, shape=cloud.data.shape[:-1])
        np.testing.assert_allclose(merged.xyz, cloud.xyz)
        np.testing.assert_allclose(merged.data, cloud.data)

        image = genImage(dimx=20, dimy=15, nbands=3)
        ichunks = _split(image, 3)
        imerged = _merge(ichunks, shape=image.data.shape[:-1])
        np.testing.assert_allclose(imerged.data, image.data)

    def test_parallel_chunks(self):
        require_test_env(self, "lite")
        from hylite.multiprocessing import parallel_chunks

        cloud = genCloud(npoints=120, nbands=5)
        result = parallel_chunks(_scale_data, cloud, 2.0, nthreads=2)
        np.testing.assert_allclose(result.data, cloud.data * 2.0)
        np.testing.assert_allclose(result.xyz, cloud.xyz)

        image = genImage(dimx=24, dimy=18, nbands=4)
        img_out = parallel_chunks(_scale_data, image, 0.5, nthreads=2)
        np.testing.assert_allclose(img_out.data, image.data * 0.5)

    def test_parallel_datasets(self):
        require_test_env(self, "lite")
        from hylite.multiprocessing import parallel_datasets

        pth = mkdtemp()
        try:
            original = io.load(os.path.join(TEST_DATA, "image.hdr"))
            in_paths = []
            out_paths = []
            for i in range(2):
                in_path = os.path.join(pth, "in_%d.hdr" % i)
                out_path = os.path.join(pth, "out_%d.hdr" % i)
                io.save(in_path, original)
                in_paths.append(in_path)
                out_paths.append(out_path)

            parallel_datasets(_copy_and_scale, in_paths, out_paths, nthreads=2, factor=3.0)

            for out_path in out_paths:
                scaled = io.load(out_path)
                self.assertEqual(scaled.data.shape, original.data.shape)
                np.testing.assert_allclose(scaled.data, original.data * 3.0)
        finally:
            shutil.rmtree(pth)


if __name__ == '__main__':
    unittest.main()
