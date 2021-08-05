import unittest
import os
import hylite
from hylite import io
from pathlib import Path
from tempfile import mkdtemp
import shutil
import numpy as np

class TestHyImage(unittest.TestCase):
    def test_load(self):
        self.img = io.load(os.path.join(str(Path(__file__).parent.parent), "test_data/image.hdr"))
        self.lib = io.load(os.path.join(str(Path(__file__).parent.parent), "test_data/library.csv"))
        self.cld = io.load(os.path.join(str(Path(__file__).parent.parent), "test_data/hypercloud.hdr"))

        # test load with SPy
        img = io.loadWithSPy(os.path.join(str(Path(__file__).parent.parent), "test_data/image.hdr"))
        self.assertTrue( np.nanmax( np.abs(self.img.data - img.data ) ) < 0.01 ) # more or less equal datsets

    def test_save(self):
        self.test_load() # load datasets
        pth = mkdtemp()

        # test lib and cloud
        try:
            for data in [self.lib, self.cld]:
                io.save(os.path.join(pth, "data.hdr"), data)
            # test image(s)
            for data in [self.img]:
                    # save with default (GDAL?)
                    print(os.path.join(pth, "data.hdr"))
                    io.save(os.path.join(pth, "data.hdr"), data )
                    self.assertEqual( os.path.exists(os.path.join(pth, "data.hdr")), True)
                    data2 = io.load(os.path.join(pth, "data.hdr")) # reload it

                    # save with SPy
                    io.saveWithSPy(os.path.join(pth, "data2.hdr"), data )
                    self.assertEqual(os.path.exists(os.path.join(pth, "data2.hdr")), True)
                    data3 = io.load(os.path.join(pth, "data2.hdr")) # reload it

                    # assert equal
                    self.assertTrue(np.nanmax(np.abs(data.data - data2.data)) < 0.01)  # more or less equal datsets
                    self.assertTrue(np.nanmax(np.abs(data.data - data3.data)) < 0.01)  # more or less equal datsets
        except:
            shutil.rmtree(pth)  # delete temp directory
            self.assertFalse(True, "Error - could not save data of type %s" % str(type(data)))

        shutil.rmtree(pth)  # delete temp directory

    def test_hycollection(self):

        # load some data
        self.test_load()  # load datasets
        pth = mkdtemp()
        try:
            # build a HyCollection
            C = hylite.HyCollection( name = "testC", root = pth )
            C.img = self.img
            C.cld = self.cld
            C.lib = self.lib
            C.val = 100.
            C.arr = np.linspace(0,100)
            C.bool = True

            # save it
            io.save( os.path.join(pth, "testC.hdr"), C )

            # load it
            C2 = io.load( os.path.join(pth, "testC.hdr") )

            # test it
            self.assertEqual( C2.val, C.val )
            self.assertTrue( (C2.arr == C.arr).all() )
            self.assertTrue( C2.bool )
            self.assertEqual(C2.val, C.val)
            self.assertEqual(C2.val, C.val)
            self.assertEqual(C2.img.xdim(), C.img.xdim())
            self.assertEqual(C2.cld.point_count(), C.cld.point_count())
            self.assertEqual(C2.lib.sample_count(), C.lib.sample_count())
        except:
            shutil.rmtree(pth)  # delete temp directory
            self.assertFalse(True, "Error - could not create, load or save HyCollection." )
        shutil.rmtree(pth)  # delete temp directory

if __name__ == '__main__':
    unittest.main()