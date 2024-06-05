import unittest
import hylite
from hylite import io
from pathlib import Path
import numpy as np
import os

class MyTestCase(unittest.TestCase):
    def test_unmixing(self):
        from hylite.analyse.unmixing import mix, unmix, endmembers

        image = io.load(os.path.join(os.path.join(str(Path(__file__).parent.parent), "test_data"),"image.hdr"))
        cloud = io.load(os.path.join(os.path.join(str(Path(__file__).parent.parent), "test_data"),"hypercloud.hdr"))
        for data in [image, cloud]:
            # build random dataset for unmixing
            em1 = data.X(onlyFinite=True)[0]
            em2 = data.X(onlyFinite=True)[-1]
            E = hylite.HyLibrary( np.vstack([em1, em2]), 
                                 lab = ['A','B'], wav=data.get_wavelengths() )
            A = data.copy()
            A.data = np.random.uniform(size=(data.data.shape[:-1]) + (2,))
            A.data = A.data / np.sum(A.data, axis=-1)[...,None] # close to sum to 1

            # run forward model
            X = mix( A, E)
            self.assertTrue(X.data.shape[-1] == E.data.shape[-1]) # check last dimension is correct

            # run backward model
            for m in ['nnls', 'fcls']:
                A2 = unmix( X, E, method=m)
                self.assertLess( np.mean(np.abs(A2.data - A.data)), 1e-4 )

            # also check for (pure) endmembers
            for m in ['atgp', 'fippi', 'nfindr', 'ppi']:
                em, ix = endmembers(X, 2, method=m )

                # check that endmembers match their indices
                if len(ix.shape) > 1:
                    self.assertLess( np.max(np.abs(em.data[0,0,:] - X.data[*ix[0], :])), 1e-6 )
                else:
                    self.assertLess( np.max(np.abs(em.data[0,0,:] - X.data[ix[0], :])), 1e-6 )

                # check that the endmembers are at least similar to the real ones
                self.assertLess( min( np.mean( np.abs(em1 - em.data[:,0,:]) ),
                                  np.mean( np.abs(em2 - em.data[:,0,:]) ) ), 0.05 ) # average 5% difference
                
                # check that we can use these endmembers directly for unmixing
                # A3 = unmix( X, em, method='nnls') # commented out due to slowness....