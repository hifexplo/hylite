import unittest
import numpy as np
from tests import genImage
from hylite.sensors import Fenix, FX10, Rikola_HSC2, Rikola_RSC1, Rikola
from hylite.reference.spectra import R90

class TestSensors(unittest.TestCase):
    def test_fenix(self):
        # create fake scene and dark and white reference
        scene = genImage(Fenix.ypixels(),500,450)
        dark = genImage(Fenix.ypixels(),100,450)
        white = genImage(Fenix.ypixels(),100,450)

        #give reasonable pixel magnitudes
        scene.data[:,:,0:175] *= 4095 # VNIR sensor
        scene.data[:, :, 175:] *= 65535 # SWIR sensor
        dark.data[:, :, 0:175] *= 0.1*4095  # VNIR sensor
        dark.data[:, :, 175:] *= 0.1*65535  # SWIR sensor
        white.data = (white.data + 2.0) / 3
        white.data[:, :, 0:175] *= 0.75*4095  # VNIR sensor
        white.data[:, :, 175:] *= 0.75*65535  # SWIR sensor

        Fenix.set_dark_ref( dark )
        Fenix.set_white_ref_spectra( R90 )
        Fenix.set_white_ref( white )
        Fenix.correct_image( scene, verbose=False )

        self.assertEqual( np.isfinite(scene.data).all(), True )

    def test_fx(self):
        scene = genImage(FX10.ypixels(),500,224)
        dark = genImage(FX10.ypixels(),100,224)
        white = genImage(FX10.ypixels(),100,224)

        scene.data *= 65535  # n.b. I have no idea what FX pixel numbers should go to
        dark.data *= 0.1 * 65535
        white.data = (white.data + 2.0) / 3
        white.data *= 0.75 * 65535

        FX10.set_dark_ref( dark )
        FX10.set_white_ref( white )
        FX10.correct_image(scene, verbose=False )

if __name__ == '__main__':
    unittest.main()
