import unittest
import os
from hylite import io
from pathlib import Path
from hylite.analyse.mwl import *
import shutil
from tempfile import mkdtemp
import numpy as np


class MyTestCase(unittest.TestCase):
    def test_hull(self):
        from hylite.correct import get_hull_corrected
        image = io.load(os.path.join(os.path.join(str(Path(__file__).parent.parent), "test_data"),"image.hdr"))
        cloud = io.load(os.path.join(os.path.join(str(Path(__file__).parent.parent), "test_data"),"image.hdr"))

        # test hull correction on numpy array
        Xhc = get_hull_corrected( image.data,vb=False )
        self.assertGreaterEqual( np.nanmin(Xhc), 0.0 )
        self.assertLessEqual( np.nanmax(Xhc), 1.0 )
        self.assertEqual( image.data.shape, Xhc.shape )

        # test hull correction on HyData instances
        for D in [image, cloud]:
            Xhc = get_hull_corrected(D,vb=False)
            self.assertEqual(image.data.shape, Xhc.data.shape)
            self.assertGreaterEqual(np.nanmin(Xhc.data), 0.0)
            self.assertLessEqual(np.nanmax(Xhc.data), 1.0)

    def test_mwl(self):
        image = io.load(os.path.join(os.path.join(str(Path(__file__).parent.parent), "test_data"),"image.hdr"))
        image.data[:50,:,:] = np.nan # add some nans to make more realistic
        cloud = io.load(os.path.join(os.path.join(str(Path(__file__).parent.parent), "test_data"),"image.hdr"))

        # also create a HyLibrary instance
        image.header.set_sample_points('A', [(20, 15)]) # label some seed pixels in each sample
        image.header.set_sample_points('B', [(80, 15)])
        image.header.set_sample_points('C', [(140, 15)])
        image.header['class names'] = ['A', 'B', 'C']
        from hylite.hylibrary import from_indices
        lib = from_indices(image,
                           [image.header.get_sample_points(n)[0] for n in image.header.get_class_names()],
                           names=image.header.get_class_names(),
                           s=5)


        for D in [lib,image,cloud]:
            # test normal mwl
            mwl = minimum_wavelength(D, 2100., 2380.,
                                     trend='hull', method='minmax',
                                     n=1, nthreads=1, vb=False, xtol=0.1, ftol=0.1)
            self.assertGreater(np.nanmax(mwl.model.data[..., 1]), 2100.)  # check some valid features were identified

            mwl = minimum_wavelength(D, 2100., 2380., trend='hull', method='gauss', n=1, nthreads=1, vb=True, xtol=0.1, ftol=0.1)
            self.assertGreater(np.nanmax(mwl.model.data[..., 1]), 2100.)  # check some valid features were identified

            # test multi-mwl
            M = minimum_wavelength( D, minw=2100., maxw=2400., sym=False,method='gauss',n=3, vb=True, xtol=0.1, ftol=0.1)
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
            M1 = minimum_wavelength( D, minw=2100., maxw=2400., sym=True, method='gauss',n=3, vb=True, nthreads=-1, xtol=0.1, ftol=0.1)

            # run some other fun functions
            ev = M1.evaluate()
            cls = M1.classify(5,nf=3)
            rs = M1.residual()

            # run plotting code
            fig,ax = M1.quick_plot()
            fig, ax = M1.quick_plot(step=3)

    def testIO(self):
        image = io.load(os.path.join(os.path.join(str(Path(__file__).parent.parent), "test_data"),"image.hdr"))
        M = minimum_wavelength(image, minw=2100., maxw=2400., sym=False, method='gauss', n=2, vb=True)

        pth = mkdtemp()

        # do save and load
        eq0,eq1 = False,False
        df = np.inf
        try:
            io.save(pth+'/test', M)
            eq0 = os.path.exists(os.path.join(pth,'test.mwl')) # save worked?
            M2 = io.load(os.path.join(pth,'test.hdr'))

            # check identical
            eq1 = (M2.x == M.x).all()
            df = np.nanmax( M.model.data - M2.model.data ) # max difference

        except:
            shutil.rmtree(pth)  # delete temp directory
            self.assertFalse(True, "Error..." )
        shutil.rmtree(pth)  # delete temp directory

        self.assertTrue(eq0) # fail here if something went wrong
        self.assertTrue(eq1)
        self.assertTrue(df < 1e-2) # difference should be very small
    def test_TPT(self):
        from hylite.filter import TPT
        image = io.load(os.path.join(os.path.join(str(Path(__file__).parent.parent), "test_data"),"image.hdr"))
        tpt,p,d = TPT(image, sigma=10., window=7, thresh=0, vb=False)

if __name__ == '__main__':
    unittest.main()
