import unittest, os
from hylite import io
from pathlib import Path
import numpy as np

class MyTestCase(unittest.TestCase):
    def test_correct_path_absorption(self):
        from hylite.correct.illumination import correct_path_absorption
        image = io.load(os.path.join(os.path.join(str(Path(__file__).parent.parent), "test_data"),"image.hdr"))
        cloud = io.load(os.path.join(os.path.join(str(Path(__file__).parent.parent), "test_data"), "hypercloud.hdr") )
        # test hull correction on HyData instances
        for D in [image, cloud]:
            Xhc = correct_path_absorption(D, atabs=2200.,vb=False)
            self.assertEqual(D.data.shape, Xhc.data.shape)
            self.assertGreaterEqual(np.nanmin(Xhc.data), 0.0)
            self.assertLessEqual(np.nanmax(Xhc.data), 1.0)

    def test_panel(self):
        image = io.load(os.path.join(os.path.join(str(Path(__file__).parent.parent), "test_data"),"image.hdr"))
        from hylite.correct import Panel
        from hylite.reference.spectra import R90
        rad = np.nanmean( image.data[:10,:10,:], axis=(0,1))
        P = Panel( R90, rad, strict=True, wavelengths=image.get_wavelengths() )
        fig,ax = P.quick_plot()
if __name__ == '__main__':
    unittest.main()
