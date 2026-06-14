import unittest
import os
import hylite
from hylite import io
from pathlib import Path
from tempfile import mkdtemp
import shutil
import numpy as np
from hylite.project.camera import Camera
from hylite.project.pushbroom import Pushbroom
from tests._support import TEST_DATA, require_test_env, upgrade_test_env


class TestIO(unittest.TestCase):
    def test_load(self):
        require_test_env(self, "basic")
        self.img = io.load(os.path.join(TEST_DATA, "image.hdr"))
        self.lib = io.load(os.path.join(TEST_DATA, "library.csv"))

        if not upgrade_test_env("lite"): return
        self.cld = io.load(os.path.join(TEST_DATA, "hypercloud.hdr"))

        if not upgrade_test_env("default"): return
        img = io.loadWithSPy(os.path.join(TEST_DATA, "image.hdr"))
        self.assertTrue(np.nanmax(np.abs(self.img.data - img.data)) < 0.01)

        if not upgrade_test_env("all"): return
        img = io.loadWithGDAL(os.path.join(TEST_DATA, "image.hdr"))
        self.assertTrue(np.nanmax(np.abs(self.img.data - img.data)) < 0.01)

    def test_loadtxt(self):
        require_test_env(self, "basic")
        lib = io.load(os.path.join(TEST_DATA, "library.csv"))
        pth = mkdtemp()
        try:
            io.saveLibraryTXT(os.path.join(pth,"libtxt.txt"), lib )
            io.saveLibraryCSV(os.path.join(pth, "libcsv.csv"), lib)
            
            lib2 = io.loadLibraryTXT(os.path.join(pth,"libtxt.txt"))
            lib3 = io.loadLibraryCSV(os.path.join(pth, "libcsv.csv"))
            for l in [lib2, lib3]:
                self.assertLess( np.max( np.abs( l.data - lib.data ) ), 1e-5 )
                self.assertLess( np.max(np.abs(l.get_wavelengths() - lib.get_wavelengths())), 1e-5 )

            # test loading from directory
            for i,mineral in enumerate(['quartz', 'biotite','phlogopite']): # build directory
                io.saveLibraryTXT(os.path.join(pth,"library/%s/_%d.txt"%(mineral,i)), lib )
            lib = io.loadLibraryDIR(os.path.join(pth,"library"))
            self.assertIn('phlogopite', lib.get_sample_names())
            self.assertEqual(lib.data.shape[0],3)
            self.assertEqual(lib.data.shape[1],57)
        except:
            shutil.rmtree(pth)  # delete temp directory
            self.assertFalse(True, "Error - could not load or save spectral library to text format.")

    def test_save(self):
        pth = mkdtemp()
        try:
            # TEST BASIC IO
            require_test_env(self, "basic")
            io.usegdal = False
            img = io.load(os.path.join(TEST_DATA, "image.hdr"))
            lib = io.load(os.path.join(TEST_DATA, "library.csv"))

            io.save(os.path.join(pth, "img_basic.hdr"), img)
            img2 = io.load(os.path.join(pth, "img_basic.hdr"))
            self.assertAlmostEqual(np.nanmax(np.abs(img.data - img2.data)), 0, 6)

            from hylite.io.libraries import saveLibraryTXT, loadLibraryTXT
            lib_path = os.path.join(pth, "lib_basic.txt")
            saveLibraryTXT(lib_path, lib)
            lib_txt = loadLibraryTXT(lib_path)
            self.assertLess(np.max(np.abs(lib.data - lib_txt.data)), 1e-5)
            self.assertLess(np.max(np.abs(lib.get_wavelengths() - lib_txt.get_wavelengths())), 1e-5)

            cam = Camera(np.ones(3), np.ones(3), 'pano', 32.2, (100, 100), step=0.1)
            track = Pushbroom(np.ones((1000, 3)), np.ones((1000, 3)), 0.05, 30.04, (100, 1000))

            io.save(os.path.join(pth, "camera"), cam)
            self.assertTrue(os.path.exists(os.path.join(pth, "camera.cam")))
            cam2 = io.load(os.path.join(pth, "camera.cam"))
            self.assertTrue((np.abs(cam2.pos - cam.pos) < 0.01).all())
            self.assertTrue((np.abs(cam2.ori - cam.ori) < 0.01).all())
            self.assertEqual(cam2.dims[0], cam.dims[0])
            self.assertEqual(cam2.proj, cam.proj)

            io.save(os.path.join(pth, "track"), track)
            self.assertTrue(os.path.exists(os.path.join(pth, "track.brm")))
            track2 = io.load(os.path.join(pth, "track.brm"))
            self.assertTrue((np.abs(track2.cp - track.cp) < 0.001).all())
            self.assertTrue((np.abs(track2.co - track.co) < 0.001).all())
            self.assertEqual(track2.dims[0], track.dims[0])
            self.assertTrue(np.abs(track2.pl - track.pl) < 0.001)

            # TEST HYPERCLOUD (PLYFILE) BASED IO
            if not upgrade_test_env("lite"): return
            cld = io.load(os.path.join(TEST_DATA, "hypercloud.hdr"))
            for data, name in [(lib, "lib_lite"), (cld, "cld_lite")]:
                path = os.path.join(pth, "%s.hdr" % name)
                io.save(path, data)
                data2 = io.load(path)
                self.assertAlmostEqual(np.nanmax(np.abs(data.data - data2.data)), 0, 6)

            # TEST SPY-BASED IO
            if not upgrade_test_env("default"): return
            io.usegdal = False
            io.saveWithSPy(os.path.join(pth, "img_spy.hdr"), img)
            self.assertTrue(os.path.exists(os.path.join(pth, "img_spy.hdr")))
            img_spy = io.load(os.path.join(pth, "img_spy.hdr"))
            self.assertAlmostEqual(np.nanmax(np.abs(img.data - img_spy.data)), 0, 6)
            rgb = img.export_bands(hylite.RGB)
            rgb.percent_clip(1, 99, per_band=True, clip=True)
            rgb.data = (rgb.data * 255).astype(np.uint8)
            rgb.header['magic_key'] = '42'
            io.save(os.path.join(pth, "rgb.hdr"), rgb)
            self.assertTrue(os.path.exists(os.path.join(pth, "rgb.png")))
            rgb2 = io.load(os.path.join(pth, "rgb.hdr"))
            self.assertEqual(rgb2.header['magic_key'], '42')

            from hylite.analyse import saveLegend
            saveLegend('Red stuff', 'Green stuff', 'Blue stuff', os.path.join(pth, 'legend.png'))
            self.assertTrue(os.path.exists(os.path.join(pth, 'legend.png')))

            # TEST GDAL-BASED IO
            if not upgrade_test_env("all"): return
            io.usegdal = True
            io.save(os.path.join(pth, "img_gdal.hdr"), img)
            self.assertTrue(os.path.exists(os.path.join(pth, "img_gdal.hdr")))
            img_gdal = io.load(os.path.join(pth, "img_gdal.hdr"))
            self.assertAlmostEqual(np.nanmax(np.abs(img.data - img_gdal.data)), 0, 6)

            io.saveWithGDAL(os.path.join(pth, "img_gdal2.hdr"), img)
            self.assertTrue(os.path.exists(os.path.join(pth, "img_gdal2.hdr")))
            img_gdal2 = io.load(os.path.join(pth, "img_gdal2.hdr"))
            self.assertAlmostEqual(np.nanmax(np.abs(img.data - img_gdal2.data)), 0, 6)
        except Exception as exc:
            self.fail("Error - could not save IO data: %s" % exc)
        finally:
            from hylite._deps import optional
            io.usegdal = optional("osgeo") is not None
            if os.path.exists(pth):
                shutil.rmtree(pth)

    def test_subset(self):
        require_test_env(self, "default")
        from hylite.io.images import loadSubset

        # load whole image for reference
        path = os.path.join(TEST_DATA, "image.hdr")
        image = io.load(path)

        # load subset and check that dimensions and values match
        subset = loadSubset(path, bands=hylite.SWIR )
        self.assertEqual(subset.xdim(), image.xdim())
        self.assertEqual(subset.ydim(), image.ydim())
        self.assertAlmostEqual(np.nanmax( np.abs(image.export_bands(hylite.SWIR).data - subset.data ) ), 0 )

        # load a pixel and check that the dimensions and values match

    def test_loadSED(self):
        require_test_env(self, "lite")
        pth = os.path.join(TEST_DATA, "sedLib")
        assert os.path.exists(pth)

        from hylite.io.libraries import loadLibrarySED
        lib = loadLibrarySED(pth)
        assert '1456045_00115' in lib.get_sample_names() # check sample names
        assert lib.data.shape[0] == 2 # two samples in this library
        assert lib.data.shape[1] == 1 # one measurement per sample
        assert lib.data.shape[2] == 1024 # 1024 bands
        assert np.abs((np.mean(lib.data)-10.258820312500001)) < 1e-5 # check spectral data


class TestHyCollection(unittest.TestCase):
    def getTestData(self):
        image = io.load(os.path.join(TEST_DATA, "image.hdr"))
        image.data[:50, :, :] = np.nan  # add some nans to make more realistic
        cloud = io.load(os.path.join(TEST_DATA, "hypercloud.hdr"))

        # also create a HyLibrary instance
        image.header.set_sample_points('A', [(20, 15)])  # label some seed pixels in each sample
        image.header.set_sample_points('B', [(80, 15)])
        image.header.set_sample_points('C', [(140, 15)])
        image.header['class names'] = ['A', 'B', 'C']
        from hylite.hylibrary import from_indices
        lib = from_indices(image,
                           [image.header.get_sample_points(n)[0] for n in image.header.get_class_names()],
                           names=image.header.get_class_names(),
                           s=5)
        return image, cloud, lib

    def getTestImage(self):
        image = io.load(os.path.join(TEST_DATA, "image.hdr"))
        image.data[:50, :, :] = np.nan
        return image

    def test_save_header_only(self):
        require_test_env(self, "lite")
        from hylite import HyCollection
        pth = mkdtemp()  # create output directory
        try:
            C = HyCollection("test", pth ) # create a HyCollection
            C.attr = "foo"
            self.assertEqual(C.file_type, 'Hylite Collection') # check "file type" key is loaded as file_type
            self.assertEqual(C.file_type, C.header['file type'])  # check "file type" key is loaded as file_type
            C.save()

            C2 = io.load(os.path.join(pth,'test.hdr'))
            self.assertEqual(C.attr, "foo")
            self.assertEqual(C2.file_type, 'Hylite Collection')  # check "file type" key is loaded as file_type
            self.assertEqual(C2.file_type, C2.header['file type'])  # check "file type" key is loaded as file_type
        except:
            shutil.rmtree(pth)  # delete temp directory
            self.assertFalse(True, "Error - failed save header only test")
        shutil.rmtree(pth)  # delete temp directory



    def test_basic(self):
        require_test_env(self, "lite")
        from hylite import HyCollection
        pth = mkdtemp() # create output directory
        array = np.random.rand(50)  # create some random data
        image, cloud, lib = self.getTestData()
        try:
            ### Create and save a HyCollection
            C = HyCollection("test", pth ) # create a HyCollection
            C.array = array # put in numpy array
            C.image = image
            C.cloud = cloud
            C.lib = lib

            X = C.cloud # call getter
            # check that the directory does not exist yet!
            print( C.getDirectory(makedirs=False) )
            self.assertFalse( os.path.exists( C.getDirectory(makedirs=False) ) )
            # test get and set
            self.assertEqual( C.get('image'), image )
            self.assertEqual( C['image'], image) # also test dict-like behaviour

            C.set('image2', image)
            self.assertEqual(C.image2, image )
            C['image3'] = image # also test dict-like behaviour
            self.assertEqual(C.image3, image )

            # test loaded function
            self.assertTrue( C.loaded('image2'), "Error in loaded(...) function.")
            # test save
            C.save_attr('image2') # save image attribute
            C.set('image3', image, save=True) # save on set
            C.save() # save all attributes

            ### Reload it
            C2 = io.load( os.path.join(pth,'test.hdr'))
            self.assertFalse( C2.loaded('image2'), "Error in loaded(...) function.")

            # check getAttributes() function
            self.assertTrue( len(C2.getAttributes(True)) == 0, "Error - getAttributes(ram=True) returned attributes on disk." )
            self.assertEqual( len(C2.getAttributes(False)), len(C.getAttributes(False) )), "Error - getAttributes(ram=True) did not return attributes on disk."
            # print(C.getAttributes(False))
            # print(C2.getAttributes(False))

            # check equality
            thresh = 1e-5
            self.assertTrue( np.max( np.abs( C2.array - array ) ) < thresh )
            self.assertTrue(np.nanmax(np.abs(C2.image.data - image.data)) < thresh)
            self.assertTrue(np.nanmax(np.abs(C2.cloud.data - cloud.data)) < thresh)
            self.assertTrue(np.nanmax(np.abs(C2.lib.data - lib.data)) < thresh)

            # check add JSON file works too
            C.mydict = {"Key" : "Value"}
            C.save()
            C.free()
            self.assertTrue("Key" in C.mydict) # will throw an error if save failed or if load failed
            
        except:
            shutil.rmtree(pth)  # delete temp directory
            self.assertFalse(True, "Error - failed basic HyCollection tests")
        shutil.rmtree(pth)  # delete temp directory

    def test_image_png(self):
        require_test_env(self, "default")  # needs Pillow for uint8 PNG export
        from hylite import HyCollection
        pth = mkdtemp()  # create output directory
        image = self.getTestImage()
        try:
            C = HyCollection("test", pth)  # create a HyCollection
            C.image = image
            C.rgb = image.export_bands(hylite.RGB)
            C.rgb.percent_clip()
            C.rgb.header['magic_key'] = '42'
            C.rgb.data = (C.rgb.data * 255).astype(np.uint8) # convert to uint8 - this should be saved as .png
            C.save()
            C.free()

            self.assertTrue(os.path.exists(os.path.join(C.getDirectory(), "rgb.hdr"))) # check header file written
            self.assertTrue( os.path.exists( os.path.join( C.getDirectory(), "rgb.png" ) ) ) # this should be png
            self.assertTrue(os.path.exists(os.path.join(C.getDirectory(), "image.dat"))) # this should be dat
            self.assertEqual( C.rgb.data.dtype, np.uint8 ) # load from disk and check type
            self.assertListEqual(list(C.rgb.get_wavelengths()), [503.4, 551.19, 681.63] ) # check header info is preserved
            self.assertEqual(C.rgb.header['magic_key'], '42')

        except:
            shutil.rmtree(pth)  # delete temp directory
            self.assertFalse(True, "Error - failed nested HyCollection tests")
        shutil.rmtree(pth)  # delete temp directory

    def test_nested(self):
        require_test_env(self, "lite")
        from hylite import HyCollection
        pth = mkdtemp() # create output directory
        array = np.random.rand(50)  # create some random data
        image, cloud, lib = self.getTestData()
        try:
            ### Create and save a HyCollection
            C = HyCollection("test", pth ) # create a HyCollection
            C.funky_data_base = 'foobar'
            C.addSub( "SC1" ) # create a subcollection
            C.SC1.array = array
            C.SC1.image = image
            C.SC1.funky_data_A = array

            SC2 = C.addSub( "SC2" ) # add another subcollection
            SC2.cloud = cloud
            SC2.lib = lib
            SC2.funky_data_B = lib

            C.addSub("SC3") # add a subcollection with no data directory (only header)
            C.SC3.message = "hi!"
            C.SC3.funky_data_C = 'This is the answer!'
            C.SC3.another_image = image.export_bands(hylite.RGB)
            C.save()

            ### Reload it
            C2 = io.load( os.path.join(pth,'test.hdr'))

            # check equality
            thresh = 1e-5
            self.assertTrue( np.max( np.abs( C2.SC1.array - array ) ) < thresh )
            self.assertTrue(np.nanmax(np.abs(C2.SC1.image.data - image.data)) < thresh)
            self.assertTrue(np.nanmax(np.abs(C2.SC2.cloud.data - cloud.data)) < thresh)
            self.assertTrue(np.nanmax(np.abs(C2.SC2.lib.data - lib.data)) < thresh)
            self.assertTrue('hi' in C2.SC3.message )

            # test recursive searching
            C2.free() # used for testing queries on datasets that are not loaded into RAM
            C2.print()
            for _C in [C, C2]:
                query = _C.query(name_pattern='funky_data', recurse = True, ram_only=False)
                self.assertListEqual(sorted(query), sorted(['funky_data_base', 'funky_data_B', 'funky_data_A', 'funky_data_C']) )
                query = _C.query(name_pattern='funky_data', recurse=False, ram_only=False)
                self.assertListEqual(query, ['funky_data_base']) # only matches a single argument
                query = _C.query(ext_pattern=['npy', 'ndarray'], recurse=True, ram_only=False) # matches string attributes
                self.assertListEqual(query, sorted(['array', 'funky_data_A']))
                query = _C.query(ext_pattern=['dat', 'HyImage'], recurse=True,
                                ram_only=False)  # matches string attributes
                self.assertListEqual(query, ['another_image', 'image'])
                query = _C.query(ext_pattern=['hyc', 'HyCollection'], recurse=False, ram_only=False)  # test no recurse
                query2 = _C.query(ext_pattern=['hyc', 'HyCollection'], recurse=True, ram_only=False,
                                recurse_matches=False )  # test no recurse matches
                self.assertListEqual(query, query2)
        except:
            shutil.rmtree(pth)  # delete temp directory
            self.assertFalse(True, "Error - failed nested HyCollection tests")
        shutil.rmtree(pth)  # delete temp directory

    def test_io_roundtrip(self):
        require_test_env(self, "lite")
        pth = mkdtemp()
        try:
            image, cloud, lib = self.getTestData()
            C = hylite.HyCollection(name="testC", root=pth)
            C.img = image
            C.cld = cloud
            C.lib = lib
            C.val = 100.
            C.arr = np.linspace(0, 100)
            C.x = None
            C.bool = True

            io.save(os.path.join(pth, "testC.hdr"), C)
            self.assertTrue(os.path.exists(os.path.join(pth, "testC.hyc", "arr.npy")))
            self.assertTrue(os.path.exists(os.path.join(pth, "testC.hyc", "img.hdr")))

            C2 = io.load(os.path.join(pth, "testC.hdr"))
            self.assertEqual(C2.val, C.val)
            self.assertTrue((C2.arr == C.arr).all())
            self.assertTrue(C2.bool)
            self.assertEqual(C2.img.xdim(), image.xdim())
            self.assertEqual(C2.cld.point_count(), cloud.point_count())
            self.assertEqual(C2.lib.sample_count(), lib.sample_count())

            C2.bool = None
            C2.val = None
            C2.img = None
            C2.arr = None
            C2.clean()
            self.assertFalse('bool' in C2.header)
            self.assertFalse('val' in C2.header)

            C2.addExternal('relobject', os.path.join(pth, "testC.hyc", "lib.lib"))
            self.assertTrue(isinstance(C2.relobject, hylite.HyLibrary))

            io.save(os.path.join(pth, "testD.hdr"), C2)
            C3 = io.load(os.path.join(pth, "testD.hyc"))
            C3.inner = io.load(os.path.join(pth, "testC.hdr"))
            C3.inner.arr2 = np.full(40, 3.0)
            io.save(os.path.join(pth, "testE.hyc"), C3)
            C3.save()
        except Exception:
            shutil.rmtree(pth)
            self.fail("Error - could not create, load or save HyCollection via io")
        shutil.rmtree(pth)


class TestMWLIO(unittest.TestCase):
    def test_mwl_io(self):
        require_test_env(self, "lite")
        from hylite.analyse.mwl import minimum_wavelength
        image = io.load(os.path.join(TEST_DATA, "image.hdr"))
        M = minimum_wavelength(image, minw=2100., maxw=2400., sym=False, method='gauss', n=2, vb=True)
        pth = mkdtemp()
        eq0, eq1, df = False, False, np.inf
        try:
            io.save(pth + '/test', M)
            eq0 = os.path.exists(os.path.join(pth, 'test.mwl'))
            M2 = io.load(os.path.join(pth, 'test.hdr'))
            eq1 = (M2.x == M.x).all()
            df = np.nanmax(M.model.data - M2.model.data)
        except Exception:
            shutil.rmtree(pth)
            self.fail("Error saving or loading MWL")
        shutil.rmtree(pth)
        self.assertTrue(eq0)
        self.assertTrue(eq1)
        self.assertTrue(df < 1e-2)


if __name__ == '__main__':
    unittest.main()