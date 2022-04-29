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
        cloud = io.load(os.path.join(os.path.join(str(Path(__file__).parent.parent), "test_data"), "image.hdr"))

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

            # test get and set
            self.assertEqual( C.get('image'), image )
            C.set('image2', image)
            self.assertEqual(C.image2, image )

            # test save
            C.save()

            ### Reload it
            C2 = io.load( os.path.join(pth,'test.hdr'))

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

    def test_nested(self):
        from hylite import HyCollection
        pth = mkdtemp() # create output directory
        array = np.random.rand(50)  # create some random data
        image, cloud, lib = self.getTestData()
        try:
            ### Create and save a HyCollection
            C = HyCollection("test", pth ) # create a HyCollection
            C.addSub( "SC1" ) # create a subcollection
            C.SC1.array = array
            C.SC1.image = image

            SC2 = C.addSub( "SC2" ) # add another subcollection
            SC2.cloud = cloud
            SC2.lib = lib

            C.addSub("SC3") # add a subcollection with no data directory (only header)
            C.SC3.message = "hi!"

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
        except:
            shutil.rmtree(pth)  # delete temp directory
            self.assertFalse(True, "Error - failed nested HyCollection tests")
        shutil.rmtree(pth)  # delete temp directory

if __name__ == '__main__':
    unittest.main()
