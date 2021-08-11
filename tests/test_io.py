import unittest
import os
import hylite
from hylite import io
from pathlib import Path
from tempfile import mkdtemp
import shutil
import numpy as np

from hylite.project import Camera, Pushbroom

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
            # test image(s) with GDAL and SPy
            for data in [self.img]:
                    # save with default (GDAL?)
                    #print(os.path.join(pth, "data.hdr"))
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

            # test saving camera objects
            cam = Camera( np.ones(3), np.ones(3), 'pano', 32.2, (100,100), step=0.1 ) # build test camera
            track = Pushbroom( np.ones((1000,3)), np.ones((1000,3)), 0.05, 30.04, (100,1000)) # test pushbroom camera

            # test saving camera
            data = cam
            io.save(os.path.join(pth, "camera"), data )
            self.assertTrue(os.path.exists(os.path.join(pth, "camera.cam")))

            # test saving track
            data = track
            io.save(os.path.join(pth, "track"), data )
            self.assertTrue(os.path.exists(os.path.join(pth, "track.brm")))

            # test loading camera
            cam2 = io.load( os.path.join(pth, "camera.cam"))
            self.assertTrue( (np.abs(cam2.pos - cam.pos) < 0.01).all() )
            self.assertTrue((np.abs(cam2.ori - cam.ori) < 0.01).all())
            self.assertTrue(cam2.dims[0] == cam.dims[0])
            self.assertEqual(cam2.proj, cam.proj)

            # test loading pushbroom
            track2 = io.load(os.path.join(pth, "track.brm"))
            self.assertTrue((np.abs(track2.cp - track.cp) < 0.001).all())
            self.assertTrue((np.abs(track2.co - track.co) < 0.001).all())
            self.assertTrue(track2.dims[0] == track.dims[0])
            self.assertTrue( np.abs(track2.pl - track.pl) < 0.001 )

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
            C.x = None # this should be ignored
            C.bool = True

            # save it
            io.save( os.path.join(pth, "testC.hdr"), C )
            #print(os.listdir( os.path.join(pth, "testC.hyc/") ) )
            self.assertTrue(os.path.exists(os.path.join(pth, "testC.hyc/arr.npy")))  # check numpy array has been saved
            self.assertTrue(os.path.exists(os.path.join(pth, "testC.hyc/img.hdr")))  # check image has been saved
            #C.print()

            # load it
            C2 = io.load( os.path.join(pth, "testC.hdr") )
            #C2.print()

            # test it
            self.assertEqual( C2.val, C.val )
            self.assertTrue( (C2.arr == C.arr).all() )
            self.assertTrue( C2.bool )
            self.assertEqual(C2.val, C.val)
            self.assertEqual(C2.val, C.val)
            self.assertEqual(C2.img.xdim(), C.img.xdim())
            self.assertEqual(C2.cld.point_count(), C.cld.point_count())
            self.assertEqual(C2.lib.sample_count(), C.lib.sample_count())

            # test cleaning
            C2.bool = None
            C2.val = None
            C2.img = None
            C2.arr = None
            C2.clean()

            self.assertFalse( 'bool' in C2.header )
            self.assertFalse('val' in C2.header)
            self.assertFalse(os.path.exists(os.path.join(pth, "testC.hyc/val.npy")))  # check numpy array has been deleted
            self.assertFalse( os.path.exists( os.path.join(pth, "testC.hyc/img.hdr") )) # check image has been deleted

            # add a relative path!
            C2.addExternal( 'relobject', os.path.join(pth, "testC.hyc/lib.csv") )
            self.assertTrue( isinstance(C2.relobject, hylite.HyLibrary) )

            # test saving collection in a different location
            io.save(os.path.join(pth, "testD.hdr"), C2 )
            self.assertTrue(os.path.exists(os.path.join(pth, "testD.hyc/cld.hdr")))  # check cloud has been copied across

            # load this collection
            C3 = io.load(os.path.join(pth, "testD.hyc"))
            C3.inner = io.load( os.path.join(pth, "testC.hdr") ) # add nested collection
            C3.inner.arr2 = np.full( 40, 3.0 ) # add new thing to nested collection
            io.save(os.path.join(pth, "testE.hyc"), C3 )
            self.assertTrue(os.path.exists(os.path.join(pth, "testE.hyc/cld.hdr"))) # check cloud has been copied across
            self.assertTrue(os.path.exists(os.path.join(pth,
                                                        "testE.hyc/inner.hyc/cld.hdr"))) # check cloud has been copied across
            self.assertTrue(os.path.exists(os.path.join(pth,
                                                        "testE.hyc/inner.hyc/arr2.npy"))) # check cloud has been copied across
            self.assertTrue(isinstance(C3.relobject, hylite.HyLibrary)) # check relative path link can be loaded

            # test quicksave function
            C3.save()
        except:
            shutil.rmtree(pth)  # delete temp directory
            self.assertFalse(True, "Error - could not create, load or save HyCollection." )
        shutil.rmtree(pth)  # delete temp directory

if __name__ == '__main__':
    unittest.main()