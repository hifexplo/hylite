import unittest
import os
from hylite import io
from pathlib import Path
from hylite.analyse.mwl import *
import shutil
from tempfile import mkdtemp
import numpy as np
class MyTestCase(unittest.TestCase):
    def test_mwl(self):
        image = io.load(os.path.join(str(Path(__file__).parent.parent), "test_data/image.hdr"))
        image.data = image.data[0:20,:,:] # crop to speed up

        # test normal mwl
        np.warnings.filterwarnings('ignore')  # supress warnings when comparing to nan
        mwl = minimum_wavelength( image, 2100., 2380., trend='hull', method='gauss', n=3, threads=1, vb=False )

        # test sort function
        mwl = sortMultiMWL(mwl, 'pos')

        # test rgb mapping
        rgb = colourise_mwl( mwl[0] )

        # test save/load
        pth = mkdtemp()
        try:
            saveMultiMWL(os.path.join(pth, "mwl.hdr"), mwl )
            mwl2 = loadMultiMWL( os.path.join(pth, "mwl.hdr") )
            shutil.rmtree(pth) # delete temp directory
        except:
            shutil.rmtree(pth)  # delete temp directory
            self.assertFalse(True, "Error - could not save mwl data?")
        for n in range(3): # check loaded files more or less match saved ones
            self.assertAlmostEquals(np.nanmax(mwl[n].data[..., 0]), np.nanmax(mwl2[n].data[..., 0]),
                                    2, msg="Error - load incorrect.")
        # test multithreading
        mwl3 = minimum_wavelength(image, 2100., 2380., trend='hull', method='gauss', n=3, threads=2, vb=False)
        mwl3 = sortMultiMWL(mwl3, 'pos')
        for n in range(3): # check results more-or-less match single-threaded ones
            self.assertAlmostEquals(np.nanmax(mwl[n].data[..., 0]), np.nanmax(mwl3[n].data[..., 0]),
                                    2, msg="Error - multi mwl is incorrect.")
if __name__ == '__main__':
    unittest.main()
