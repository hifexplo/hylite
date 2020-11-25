import unittest
import os
from hylite import io
from pathlib import Path
from tempfile import mkdtemp
import shutil

class TestHyImage(unittest.TestCase):
    def test_load(self):
        self.img = io.load(os.path.join(str(Path(__file__).parent.parent), "test_data/image.hdr"))
        self.lib = io.load(os.path.join(str(Path(__file__).parent.parent), "test_data/library.csv"))
        self.cld = io.load(os.path.join(str(Path(__file__).parent.parent), "test_data/hypercloud.hdr"))

    def test_save(self):
        self.test_load() # load datasets
        for data in [self.img, self.lib, self.cld]:
            pth = mkdtemp()
            try:
                io.save(os.path.join(pth, "data.hdr"), data )
                shutil.rmtree(pth) # delete temp directory
            except:
                shutil.rmtree(pth)  # delete temp directory
                self.assertFalse(True, "Error - could not save data of type %s" % str(type(data)))

if __name__ == '__main__':
    unittest.main()