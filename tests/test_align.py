import unittest
import os
from pathlib import Path
import hylite.io as io
from hylite.project.align import *
import shutil
from tempfile import mkdtemp
import numpy as np

class MyTestCase(unittest.TestCase):
    def test_mwl(self):
        # load test image
        image1 = io.load(os.path.join(str(Path(__file__).parent.parent), "test_data/image.hdr"))

        # create slightly offset second image
        image2 = image1.copy()
        image2.data = image2.data[5:, 5:, : ].copy() # add an offset
        image2.push_to_header()

        # test align with method=affine
        # n.b. there's not really any way to check if the align actually worked... so just try a bunch of combinations
        np.warnings.filterwarnings('ignore') # supress warnings when comparing to nan
        align_images( image1, image2, image_bands=(30, 50), target_bands=(30, 50),
                                warp=True, features='sift', method='affine' )
        align_images(image1, image2, image_bands=(450., 600.), target_bands=(450., 600.),
                     warp=True, features='sift', method='poly')

if __name__ == '__main__':
    unittest.main()
