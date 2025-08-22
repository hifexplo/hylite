import unittest
import hylite
from hylite import io
from pathlib import Path
import numpy as np
import os

class MyTestCase(unittest.TestCase):
    def test_sam(self):
        from hylite.analyse.sam import SAM, spectral_angles

        image = io.load(os.path.join(os.path.join(str(Path(__file__).parent.parent), "test_data"),"image.hdr"))
        cloud = io.load(os.path.join(os.path.join(str(Path(__file__).parent.parent), "test_data"),"hypercloud.hdr"))

        for data in [image, cloud]:
            em1 = data.X(onlyFinite=True)[0]
            em2 = data.X(onlyFinite=True)[-1]

            # check spectral angles function runs
            ang = spectral_angles( [em1,em2], data.X() )
            self.assertTrue(np.isfinite(ang).any())

            # run SAM
            em3 = data.X(onlyFinite=True)[-10]
            sam = SAM(data, [[em3], [em1,em2] ])
            self.assertTrue(np.isfinite(sam.data).any())
            self.assertEqual( int( sam.X( onlyFinite=True )[0,0]), 1 )
            self.assertEqual(int(sam.X(onlyFinite=True)[-1, 0]), 1 )
            self.assertEqual(int(sam.X(onlyFinite=True)[-10, 0]), 0 )

            # run SAM with a library
            arr = np.vstack( [em3,em2] )[:,None,:]
            lib = hylite.HyLibrary( arr, lab=['EM1','EM2'], wav=data.get_wavelengths() )
            sam = SAM(data, lib)
            self.assertTrue(np.isfinite(sam.data).any())
            self.assertEqual(int(sam.X(onlyFinite=True)[-1, 0]), 1 )
            self.assertEqual(int(sam.X(onlyFinite=True)[-10, 0]), 0 )