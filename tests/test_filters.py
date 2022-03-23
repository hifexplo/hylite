import unittest
import os
from pathlib import Path
from hylite import io

class MyTestCase(unittest.TestCase):
    def test_MNF(self):
        image = io.load(os.path.join(os.path.join(str(Path(__file__).parent.parent), "test_data"), "image.hdr"))
        cloud = io.load(os.path.join(os.path.join(str(Path(__file__).parent.parent), "test_data"), "image.hdr"))

        # run mnf on clouds and images
        from hylite.filter import MNF
        for data in [image, cloud]:
            mnf, w = MNF( data, bands=10)
            self.assertEqual( data.band_count(), 10 )


        self.assertEqual(True, False)


if __name__ == '__main__':
    unittest.main()
