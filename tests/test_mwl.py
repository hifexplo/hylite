import unittest
import os
from hylite import io
from pathlib import Path
from hylite.analyse.mwl import *
import shutil
from tempfile import mkdtemp
import numpy as np

from hylite.io import saveMultiMWL, loadMultiMWL


class MyTestCase(unittest.TestCase):
    def test_hull(self):
        from hylite.correct import get_hull_corrected
        image = io.load(os.path.join(str(Path(__file__).parent.parent), "test_data/image.hdr"))
        cloud = io.load(os.path.join(str(Path(__file__).parent.parent), "test_data/image.hdr"))

        # test hull correction on numpy array
        Xhc = get_hull_corrected( image.data,vb=False )
        self.assertTrue( np.nanmax(Xhc) <= 1.0 )
        self.assertTrue( np.nanmin(Xhc) >- 0.0)
        self.assertEqual( image.data.shape, Xhc.shape )

        # test hull correction on HyData instances
        for D in [image, cloud]:
            Xhc = get_hull_corrected(D,vb=False)
            self.assertEqual(image.data.shape, Xhc.data.shape)
            self.assertTrue(np.nanmax(Xhc.data) <= 1.0)
            self.assertTrue(np.nanmin(Xhc.data) >= 0.0)

    def test_mwl(self):
        image = io.load(os.path.join(str(Path(__file__).parent.parent), "test_data/image.hdr"))
        cloud = io.load(os.path.join(str(Path(__file__).parent.parent), "test_data/image.hdr"))

        for D in [image,cloud]:
            # test normal mwl
            mwl = minimum_wavelength(D, 2100., 2380., trend='hull', method='minmax', n=1, nthreads=1, vb=False)
            self.assertGreater(np.nanmax(mwl.data[..., 1]), 2100.)  # check some valid features were identified

            mwl = minimum_wavelength(D, 2100., 2380., trend='hull', method='gauss', n=1, nthreads=1, vb=True)
            self.assertGreater(np.nanmax(mwl.data[..., 1]), 2100.)  # check some valid features were identified

            # test multi-mwl
            M = minimum_wavelength( D, minw=2100., maxw=2400., sym=False,method='gauss',n=3, vb=True )
            mask = np.isfinite(M[0, 0])

            # check depth sorting
            M.sortByDepth()
            self.assertTrue( (M[0, 'depth'][mask] >= M[1, 'depth'][mask]).all())
            self.assertTrue( (M[2, 'depth'][mask] >= M[2, 'depth'][mask]).all())

            # check pos sorting
            M.sortByPos()
            self.assertTrue( (M[0, 'pos'][mask] <= M[1, 'pos'][mask]).all())
            self.assertTrue( (M[2, 'pos'][mask] <= M[2, 'pos'][mask]).all())

            # check deepest feature extraction
            deepest = M.deepest(2100., 2400.)
            M.sortByDepth()
            self.assertTrue( (np.nan_to_num(deepest.data) == np.nan_to_num(M[0].data)).all() )

            # check closest feature extraction
            closest = M.closest(2100.)
            M.sortByPos()
            self.assertTrue( (np.nan_to_num(closest.data) == np.nan_to_num(M[0].data)).all() )

            closest = M.closest(2200., valid_range=(2195., 2205.))
            self.assertTrue( np.nanmin(closest.data[..., 1]) >= 2195., "Error - %s" % np.nanmin(closest.data[..., 1]))
            self.assertTrue( np.nanmin(closest.data[..., 1]) <= 2205., "Error - %s" % np.nanmin(closest.data[..., 1]))

            # test colourise function [ just run it to check for crashes ]
            rgb, leg = colourise_mwl( M.closest(2200., valid_range=(2150., 2230.) ) )

            # test multithreading (again; just see if it runs for now )
            M1 = minimum_wavelength( D, minw=2100., maxw=2400., sym=False, method='gauss',n=3, vb=True, nthreads=-1 )

    def test_TPT(self):
        from hylite.filter import TPT
        image = io.load(os.path.join(str(Path(__file__).parent.parent), "test_data/image.hdr"))
        tpt,p,d = TPT(image, sigma=10., window=7, thresh=0, vb=False)

if __name__ == '__main__':
    unittest.main()
