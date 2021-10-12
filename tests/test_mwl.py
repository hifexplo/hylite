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
    def test_mwl(self):
        image = io.load(os.path.join(str(Path(__file__).parent.parent), "test_data/image.hdr"))
        image.data = image.data[0:20,:,:] # crop to speed up

        # test normal mwl
        mwl = minimum_wavelength(image, 2100., 2380., trend='hull', method='minmax', n=1, threads=1, vb=False)
        self.assertGreater(np.nanmax(mwl.data[..., 0]), 2100.)  # check some valid features were identified

        mwl = minimum_wavelength(image, 2100., 2380., trend='hull', method='gauss', n=1, threads=1, vb=False)
        self.assertGreater(np.nanmax(mwl.data[..., 0]), 2100.)  # check some valid features were identified

        # test multi mwl
        np.warnings.filterwarnings('ignore')  # supress warnings when comparing to nan
        mwl = minimum_wavelength( image, 2100., 2380., trend='hull', method='gauss', n=3, threads=1, vb=False )
        self.assertGreater(np.nanmax(mwl[0].data[..., 0]), 2100.) # check some valid features were identified

        # test sort function
        mwl = sortMultiMWL(mwl, 'pos')

        # test rgb mapping
        rgb = colourise_mwl( mwl[0] )
        self.assertTrue( np.isfinite( rgb[0].data ).any() )

        # test save/load
        pth = mkdtemp()
        try:

            # using function directly
            saveMultiMWL(os.path.join(pth, "mwl.hdr"), mwl )
            mwl2 = loadMultiMWL( os.path.join(pth, "mwl.hdr") )

            # using hylite.io
            io.save( os.path.join(pth, "mwl2.hdr"), mwl )
            mwl3 = io.load( os.path.join(pth, "mwl2.hdr") )
            shutil.rmtree(pth) # delete temp directory
        except:
            shutil.rmtree(pth)  # delete temp directory
            self.assertFalse(True, "Error - could not save mwl data?")
        for n in range(3): # check loaded files more or less match saved ones
            for lmwl in [mwl2, mwl3]:
                self.assertAlmostEquals(np.nanmax(mwl[n].data[..., 0]), np.nanmax(lmwl[n].data[..., 0]),
                                        2, msg="Error - load incorrect.")
                self.assertGreater(np.nanmax(lmwl[n].data[..., 0]), 2100.)

        # test multithreading
        mwl3 = minimum_wavelength(image, 2100., 2380., trend='hull', method='gauss', n=3, threads=2, vb=False)
        mwl3 = sortMultiMWL(mwl3, 'pos')
        for n in range(3): # check results more-or-less match single-threaded ones
            self.assertAlmostEquals(np.nanmax(mwl[n].data[..., 0]), np.nanmax(mwl3[n].data[..., 0]),
                                    2, msg="Error - multi mwl is incorrect.")
            self.assertGreater(np.nanmax(mwl3[n].data[..., 0]), 2100.)
    def test_TPT(self):

        from hylite.filter import TPT
        image = io.load(os.path.join(str(Path(__file__).parent.parent), "test_data/image.hdr"))
        tpt,p,d = TPT(image, sigma=10., window=7, thresh=0, vb=False)

        # test TPT MWL
        from hylite.analyse import minimum_wavelength
        mwl = minimum_wavelength(image, 2100., 2380., trend='hull', method='tpt', n=1, threads=2, vb=False)
        self.assertGreater( np.nanmax(mwl.data[...,0]), 2100. )
if __name__ == '__main__':
    unittest.main()
