import unittest
import os
from pathlib import Path
import hylite.io as io
from hylite.project.align import *
import shutil
from tempfile import mkdtemp
import numpy as np

class TestAlign(unittest.TestCase):
    def test_library(self):
        image = io.load(os.path.join(os.path.join(str(Path(__file__).parent.parent), "test_data"),"image.hdr"))

        # label some seed pixels in each sample
        image.header.set_sample_points('A', [(20, 15)])
        image.header.set_sample_points('B', [(80, 15)])
        image.header.set_sample_points('C', [(140, 15)])
        image.header['class names'] = ['A', 'B', 'C']

        # test building library with sample positions only
        from hylite.hylibrary import from_indices
        lib = from_indices(image,
                             [image.header.get_sample_points(n)[0] for n in image.header.get_class_names()],
                             names=image.header.get_class_names(),
                             s=5)
        self.assertEqual( lib.data.shape[0], 3 )

        # expand labels using grab-cut
        from hylite.filter import label_blocks
        cls = label_blocks(image, s=5,  # number of pixels to label outside of seed point
                           epad=1,  # ignore these pixels near edges (can be dodgy sometimes)
                           erode=1,  # apply erode filter to avoid pixels near sample edges
                           boost=10,  # boost contrast before labelling
                           bands=hylite.SWIR)


        # test building library and plotting functions
        from hylite.hylibrary import from_classification
        for sample in [50, (50,),'all', (5,50,95)]:
            lib = from_classification( image, cls, ignore=[0], subsample=sample )
            lib.quick_plot(color=['r','g','b'], clip=(0,50,100))
            lib.quick_plot(color=['r', 'g', 'b'], clip=50)

        # test copy functions (well... run them)
        lib2 = lib.copy(data=False)
        lib2 = lib.copy(data=True)

        # test merging / splitting
        lib.set_sample_names(['A','B','C'])
        lib2 = lib[['A','A','B']] # check merging of names
        self.assertEqual( lib2.data.shape[0], 2 )
        self.assertEqual(lib2.data.shape[1], 6)
        lib2 = lib[[1, 'A', 'B']] # check indices and names merge seamlessly
        self.assertEqual(lib2.data.shape[0], 2)
        self.assertEqual(lib2.data.shape[1], 6)

        # add groups
        lib.add_group("Group1", ['A','B'])
        lib.add_group("Group2", ['C'])

        # get them
        groups = lib.get_groups()
        lib2 = lib.get_group('Group1')
        lib3 = lib.get_group('Group2')
        self.assertTrue( 'Group1' in groups )
        self.assertTrue('Group2' in groups )
        self.assertTrue( lib.has_groups() )
        self.assertEqual( lib2.data.shape[0], 2)
        self.assertEqual(lib3.data.shape[0], 1)

        # test collapse
        self.assertEqual(lib.data.shape[0],3) # three samples
        lib3 = lib.collapse()
        self.assertEqual(lib3.data.shape[0],2) # two samples after collapsing groups

        # test squash (at least, run it)
        lib4 = lib.squash()
        self.assertEqual( lib4.data.shape[1], 1 )
        self.assertEqual( lib4.data.shape[0], lib.data.shape[0] )

        # test fancy plotting
        lib.quick_plot( collapse=True, hc=True )

    def test_construction(self):
        image = io.load(os.path.join(os.path.join(str(Path(__file__).parent.parent), "test_data"), "image.hdr"))




if __name__ == '__main__':
    unittest.main()
