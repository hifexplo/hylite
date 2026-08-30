import unittest

import numpy as np
from tests._support import require_test_env


class TestReferenceGenerate(unittest.TestCase):
    def test_generate(self):
        require_test_env(self, "default")
        from hylite.reference import genImage, randomSpectra
        im, A = genImage()
        self.assertTrue(callable(randomSpectra))
        self.assertTrue(im.ydim() == 512)
        self.assertTrue(np.isfinite(im.data).all())
        self.assertFalse((im.data == 0).all())

if __name__ == '__main__':
    unittest.main()
