import unittest
import hylite
from hylite import HyFeature
from hylite.reference.features import Minerals
import numpy as np

class MyTestCase(unittest.TestCase):
    def test_construct(self):
        assert Minerals.Mica_K is not None
        assert Minerals.Chlorite_Fe is not None

    def test_multigauss(self):
        x = np.linspace(2100., 2400., 500 )
        y = HyFeature.gaussian(x, 2200., 200., 0.5 )
        self.assertAlmostEquals( np.max(y), 1.0, 2 )
        self.assertAlmostEquals(np.min(y), 0.5, 2)

        y = HyFeature.multi_gauss(x, [2200.,2340.], [200., 200.], [0.5,0.5])
        print(np.min(y), np.max(y))
        self.assertAlmostEquals(np.max(y), 1.0, 2)
        self.assertAlmostEquals(np.min(y), 0.5, 2)

    def test_plotting(self):
        x = np.linspace(2100., 2400., 500)
        Minerals.Mica_K[0].quick_plot()

        Minerals.Mica_K[0].quick_plot(method='')

    def test_fitting(self):
        pass

if __name__ == '__main__':
    unittest.main()
