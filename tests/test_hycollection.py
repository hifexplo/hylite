import unittest
from tempfile import mkdtemp
import shutil
import numpy as np
import hylite
from hylite import io
import os
from pathlib import Path

class MyTestCase(unittest.TestCase):
    def getTestData(self):
        image = io.load(os.path.join(os.path.join(str(Path(__file__).parent.parent), "test_data"), "image.hdr"))
        image.data[:50, :, :] = np.nan  # add some nans to make more realistic
        cloud = io.load(os.path.join(os.path.join(str(Path(__file__).parent.parent), "test_data"), "hypercloud.hdr"))

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

    def test_save_header_only(self):
        from hylite import HyCollection
        pth = mkdtemp()  # create output directory
        try:
            C = HyCollection("test", pth ) # create a HyCollection
            C.attr = "foo"
            self.assertEquals(C.file_type, 'Hylite Collection') # check "file type" key is loaded as file_type
            self.assertEquals(C.file_type, C.header['file type'])  # check "file type" key is loaded as file_type
            C.save()
            C2 = io.load(os.path.join(pth,'test.hdr'))
            self.assertEquals(C.attr, "foo")
            self.assertEquals(C2.file_type, 'Hylite Collection')  # check "file type" key is loaded as file_type
            self.assertEquals(C2.file_type, C2.header['file type'])  # check "file type" key is loaded as file_type
        except:
            shutil.rmtree(pth)  # delete temp directory
            self.assertFalse(True, "Error - failed save header only test")
        shutil.rmtree(pth)  # delete temp directory



    def test_basic(self):
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
            C.set('image2', image)
            self.assertEqual(C.image2, image )

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
        except:
            shutil.rmtree(pth)  # delete temp directory
            self.assertFalse(True, "Error - failed basic HyCollection tests")
        shutil.rmtree(pth)  # delete temp directory

    def test_image_png(self):
        from hylite import HyCollection
        pth = mkdtemp()  # create output directory
        array = np.random.rand(50)  # create some random data
        image, cloud, lib = self.getTestData()
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

if __name__ == '__main__':
    unittest.main()
