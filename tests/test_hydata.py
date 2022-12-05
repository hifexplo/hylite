import unittest
from hylite.project import Camera
from hylite.reference.spectra import R90
from hylite.correct.panel import Panel

import numpy as np
from tests import genHeader, genCloud, genImage

class TestHyData(unittest.TestCase):
    def test_header(self):

        #load header from file
        header = genHeader()

        # check basics
        self.assertEqual(header.has_band_names(), False)
        self.assertEqual(header.has_wavelengths(), True)
        self.assertEqual(header.has_fwhm(), False)
        self.assertEqual(header.band_count(), 450)
        self.assertEqual(len(header.get_wavelengths()), 450)

        # check copy and set functions
        header2 = header.copy()
        header2.set_wavelengths( np.zeros_like(header.get_wavelengths()))
        header2.set_band_names(["Band %d" for i in range(header.band_count())])
        self.assertEqual( (header.get_wavelengths() == header2.get_wavelengths()[0]).any(), False )

        # check drop bands
        header3 = header.copy()
        header3.set_band_names(["Band %d" for i in range(header.band_count())])
        mask = np.full( header2.band_count(), True )
        mask[0:4] = False
        header3.drop_bands(mask)
        self.assertEqual(header3.band_count(), 4)
        self.assertEqual(len(header3.get_wavelengths()), 4)
        self.assertEqual(len(header3.get_band_names()), 4)

        # check set Camera
        # define camera properties and initial location/orientation estimate
        cam = Camera(pos=np.asarray([665875.0, 4162695, 272]),  # np.array([666290.454, 4162697.93, 268.521235])
                     ori=np.array([43, 80, 130]),  # np.array([50.0,-83.0,-137.0])
                     proj='pano', fov=32.3, step=0.084,
                     dims=(1464, 401))
        header.set_camera(cam)
        cam2 = header.get_camera()

        self.assertEqual((cam2.pos == cam.pos).all(), True)
        self.assertEqual((cam2.ori == cam.ori).all(), True)
        self.assertEqual(cam2.proj, cam.proj)
        self.assertEqual(cam2.dims, cam.dims)
        self.assertEqual(cam2.fov, cam.fov)
        self.assertEqual(cam2.step, cam.step)

        # check set panel
        panel = Panel( R90, np.zeros( header.band_count() ), wavelengths=header.get_wavelengths() )
        header.add_panel(panel)
        self.assertEqual( len(header.get_panel_names()), 1)
        panel2 = header.get_panel('R90')
        self.assertEqual( np.sum( panel2.get_mean_radiance() ), 0 )
        self.assertEqual(panel2.get_mean_radiance().shape[0], header.band_count())
        self.assertEqual( panel2.material.get_name().lower(), R90.get_name().lower())

    def test_data(self):
        # check functions for images and cloud data
        lines = [401, 1]
        samples = [1464, 1000]
        for i,data in enumerate( [genImage(dimx = 1464, dimy=401, nbands=10), genCloud(npoints = 1000, nbands=10)] ):
            # check basics
            self.assertEqual(data.has_wavelengths(), True)
            self.assertEqual(data.has_band_names(), True)
            self.assertEqual(data.has_fwhm(), True)
            self.assertEqual(data.band_count(), 10)
            self.assertEqual(data.samples(), samples[i])
            self.assertEqual(data.lines(), lines[i])
            self.assertEqual(data.is_int(), False)
            self.assertEqual(data.is_float(), True)

            # check band names
            data.set_band_names([a for a in 'abcdefghijklmnop'[:10]])
            self.assertEqual( data.get_band_index('e'), 4 )

            # check export (which also checks copy etc.)
            data2 = data.export_bands( (0,5) )
            self.assertEqual(len(data2.get_wavelengths()), 6)
            self.assertEqual(len(data2.get_fwhm()), 6)
            self.assertEqual(data2.data.shape[-1], 6 )

            # nans
            data2.mask_bands(3,-1) # mask bands from 3rd to last
            self.assertEqual(data2.data.shape[-1], 6) # bands should still exist
            self.assertEqual( np.isfinite(data2.data[...,3:]).any(), False ) # all of last bands should be nan
            data2.delete_nan_bands()
            self.assertEqual(data2.data.shape[-1], 3)  # bands should have been deleted
            self.assertEqual(len(data2.get_wavelengths()), 3) # as should associated header data
            self.assertEqual(len(data2.get_fwhm()), 3) # as should associated header data

            # test set as nan
            self.assertEqual(np.isfinite(data2.data[..., 2]).all(), True)
            data2.data[..., :] = 0
            data2.set_as_nan(0)
            self.assertEqual( np.isfinite( data2.data[...,2] ).all(), False )

            # get band
            self.assertEqual( np.isfinite( data2.get_band(2)).any(), False)
            self.assertEqual( data2.get_band_grey(0).dtype, np.uint8 )
            self.assertEqual( data2.get_band_index(500.0), 0 )

            # check compression
            tv = data.data.ravel()[0]
            data.compress()
            self.assertEqual(data.data.dtype, np.uint16)
            data.decompress()
            self.assertEqual(data.data.dtype, np.float32)
            self.assertAlmostEqual(data.data.ravel()[0], tv, 3)

            # check smoothing works with nan bands
            data.mask_bands(1, 3)
            data.mask_bands(8, -1)
            data.smooth_median(window=3)
            data.smooth_savgol(window=3, chunk=True)

            # percent clip
            data.percent_clip(5,95,per_band=False,clip=True)
            data.percent_clip(5, 95, per_band=True, clip=True)

            # normalise
            data.normalise()

            # resampling
            sub = data.resample( data.get_wavelengths()[2::4], agg = True, thresh=30. )
            self.assertEqual( sub.band_count(), int( data.band_count() / 4 ) )
            sub2 = data.resample(data.get_wavelengths()[2::5], agg=False, thresh=30.)
            self.assertEqual(sub2.band_count(), int(data.band_count() / 5))
            rg = [ (data.get_wavelengths()[i], data.get_wavelengths()[i+2]) for i in range(0,data.band_count()-2, 2) ]
            sub3 = data.resample(rg, agg=True, thresh=0.1)
            self.assertEqual(sub3.band_count(), len(rg))
            sub4 = data.resample( np.linspace(0.,1000.), partial=True)

            # test fill holes
            data.data = np.nan_to_num(data.data, posinf=0, neginf=0) # remove any stray nans
            self.assertTrue(np.isfinite(data.data).all())
            data.data[...,5] = np.nan # add some nans
            data.fill_gaps() # remove them again!
            self.assertTrue(np.isfinite(data.data).all() ) # check they were removed

            # test reshaping to feature vectors
            data.data[..., 5] = np.nan  # add some nans
            self.assertSequenceEqual( data.X().shape, np.reshape(data.data, (-1, data.band_count())).shape ) # check shape is unchanged
            self.assertNotEquals(data.X(True).shape[0], np.reshape(data.data, (-1, data.band_count())).shape[0] ) # check nans are removed
            self.assertTrue( np.isfinite( data.X(True).all()) )

            data.set_raveled( np.zeros_like( data.X(True)), onlyFinite=True ) # set with nan-mask
            self.assertFalse( np.isfinite(data.data ).all() ) # check nans persist

            data.set_raveled(np.zeros_like(data.X()), onlyFinite=False)  # set without nan-mask
            self.assertTrue(np.isfinite(data.data).all())  # check nans are gone

            # N.B. data.data is now all zero!

if __name__ == '__main__':
    unittest.main()
