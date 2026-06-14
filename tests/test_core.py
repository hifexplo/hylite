import unittest
import os
from pathlib import Path

import hylite
import numpy as np
from hylite import io
from hylite.project.camera import Camera
from tests import genHeader, genCloud, genImage
from tests._support import TEST_DATA, require_test_env


class TestHyData(unittest.TestCase):
    def test_header(self):
        require_test_env(self, "basic")
        from hylite.project.camera import Camera
        from hylite.correct.panel import Panel
        from hylite.reference.spectra import R90
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
        require_test_env(self, "default")
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

            # check nasty characters are dropped to avoid issues in header files....
            data.set_band_names(["%s,2\n{}"%b for b in 'abcdefghijklmnop'[:10]])
            for c in ',\n{}':
                self.assertTrue(c not in ''.join(data.get_band_names()))
            #print('|'.join(data.get_band_names()))

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

            # check quantize
            for m in ['kmeans']: # , 'minibatch', 'birch']:
                index,lib = data.getQuantized(n=255, cmeth=m, vthresh=10, subsample=50, mask=None )
                self.assertEqual( np.max(index.data), 255 )
                self.assertEqual( lib.data.shape[0], 256 )

            # check reconstruction from quanta
            rc = hylite.HyData.fromQuanta( index, lib )
            self.assertTrue( (np.array(rc.data.shape) == np.array(data.data.shape)).all() )
            self.assertListEqual( list(rc.get_wavelengths()), list(data.get_wavelengths()) )
            self.assertListEqual( list(rc.get_band_names()), list(data.get_band_names()) )

            # check smoothing works with nan bands
            data.mask_bands(1, 3)
            data.mask_bands(8, -1)
            data.smooth_median(window=3)
            data.smooth_savgol(window=3, chunk=True)

            # percent clip
            data.percent_clip(5,95,per_band=False,clip=True)
            data.percent_clip(5, 95, per_band=True, clip=True)

            # expression evaluation
            out = data.eval('b2:b5 + b8')
            w0 = data.get_wavelengths()[1]
            w1 = data.get_wavelengths()[3]
            self.assertEqual(out.data.shape[-1], 1 ) # should have 1 band
            out = data.eval('b2:b5 + b8 | %d/(%d+$1)**$2'%(w0, w1))
            self.assertEqual(out.data.shape[-1], 2 ) # should have 2 bands

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
            self.assertNotEqual(data.X(True).shape[0], np.reshape(data.data, (-1, data.band_count())).shape[0] ) # check nans are removed
            self.assertTrue( np.isfinite( data.X(True).all()) )

            data.set_raveled( np.zeros_like( data.X(True)), onlyFinite=True ) # set with nan-mask
            self.assertFalse( np.isfinite(data.data ).all() ) # check nans persist

            data.set_raveled(np.zeros_like(data.X()), onlyFinite=False)  # set without nan-mask
            self.assertTrue(np.isfinite(data.data).all())  # check nans are gone

            # N.B. data.data is now all zero!

    def test_getitem_setitem(self):
        require_test_env(self, "default")
        image = genImage(dimx=10, dimy=5, nbands=10)
        cloud = genCloud(npoints=50, nbands=10)

        # get individual bands / pixels using float wavelengths
        self.assertTrue(np.allclose(image[0, ..., 550.0], image.data[0, ..., image.get_band_index(550.0)]))
        self.assertTrue(np.allclose(image[..., 550.0], image.get_band(550.0)))

        band_slice = image[..., 500.0:600.0] # get slice based on float wavelength range
        i0 = image.get_band_index(500.0)
        i1 = image.get_band_index(600.0) + 1
        self.assertEqual(band_slice.shape[-1], i1 - i0)

        # slice based on band names
        name_slice = image[..., 'Band 1':'Band 3']
        j0 = image.get_band_index('Band 1')
        j1 = image.get_band_index('Band 3') + 1
        self.assertEqual(name_slice.shape[-1], j1 - j0)
        self.assertTrue(np.allclose(name_slice, image.data[..., j0:j1]))
        self.assertTrue(np.allclose(image['Band 2':'Band 4'], image.data[..., image.get_band_index('Band 2'):image.get_band_index('Band 4') + 1]))

        # same checks for clouds (different shaped data array)
        self.assertTrue(np.allclose(cloud[0, 550.0], cloud.data[0, cloud.get_band_index(550.0)]))
        self.assertTrue(np.allclose(cloud[..., 550.0], cloud.get_band(550.0)))

        # check header key retrieval / setting
        for data in (image, cloud):
            data['sensor'] = 'test_sensor'
            self.assertEqual(data['sensor'], 'test_sensor')
            data.header['sensor'] = 'hdr_sensor'
            self.assertEqual(data['sensor'], 'hdr_sensor')

            # also check value setting in data array
            data[..., 0] = 42.0
            self.assertTrue((data.data[..., 0] == 42.0).all())

        # check a more complex slice
        image[2:4, 1:3, 700.0] = 99.0
        b = image.get_band_index(700.0)
        self.assertTrue((image.data[2:4, 1:3, b] == 99.0).all())
        self.assertTrue(np.allclose(image[2, 1, 'Band 3'], image.get_band('Band 3')[2, 1]))

        cloud[5, 800.0] = 11.0
        self.assertEqual(cloud.data[5, cloud.get_band_index(800.0)], 11.0)

class TestHyImage(unittest.TestCase):
    def test_image(self):
        require_test_env(self, "opencv")
        image = hylite.HyImage(np.zeros((25,25,5)), wav=np.arange(5)*100)
        self.assertListEqual(list(image.get_wavelengths()), list(np.arange(5)*100))

        # create test image
        image = genImage(dimx = 1464, dimy=401, nbands=10)

        # check basics
        self.assertEqual(image.xdim(), 1464)
        self.assertEqual(image.ydim(), 401)
        self.assertEqual(image.band_count(), 10)
        self.assertEqual(image.aspx(),  401 / 1464)

        # run plotting functions
        image.quick_plot( (0,1,2), vmin=2, vmax=98 )
        image.quick_plot( 0 )

        # ------------------------------------------------------------------
        # Set a known affine transform for georeferencing tests
        # GDAL format: [x0, px_w, rot_x, y0, rot_y, px_h]
        # ------------------------------------------------------------------

        # do we have GDAL?
        gdal = True
        try:
            from osgeo import gdal
            gdal = True
        except:
            gdal = False

        image.affine = np.array( [1000.0, 2.0, 0.0, 2000.0, 0.0, -2.0] )
        if gdal:
            image.set_projection_EPSG('EPSG:32633') # set an EPSG code as this is needed for pix to world tests

        # ------------------------------------------------------------------
        # Resize and test affine update
        # ------------------------------------------------------------------
        old_affine = image.affine.copy()
        old_x, old_y = image.xdim(), image.ydim()

        nx, ny = int(old_x / 2), int(old_y / 2)
        image.resize(newdims=(nx, ny))

        self.assertEqual(image.xdim(), nx)
        self.assertEqual(image.ydim(), ny)
        self.assertEqual(image.band_count(), 10)

        # pixel scaling factors
        sx = old_x / nx
        sy = old_y / ny

        # origin should be unchanged
        self.assertAlmostEqual(image.affine[0], old_affine[0])
        self.assertAlmostEqual(image.affine[3], old_affine[3])

        # pixel size should scale
        self.assertAlmostEqual(image.affine[1], old_affine[1] * sx)
        self.assertAlmostEqual(image.affine[5], old_affine[5] * sy)

        # rotation terms (if any) should scale consistently
        self.assertAlmostEqual(image.affine[2], old_affine[2] * sy)
        self.assertAlmostEqual(image.affine[4], old_affine[4] * sx)

        # world-space invariant check (strongest test)
        if gdal:
            x0, y0 = image.pix_to_world(0, 0)
            x1, y1 = image.pix_to_world(nx, ny)

            ox0, oy0 = hylite.HyImage(
                np.zeros((old_x, old_y, 1)),
                affine=old_affine,
                projection=image.projection
            ).pix_to_world(0, 0)

            ox1, oy1 = hylite.HyImage(
                np.zeros((old_x, old_y, 1)),
                affine=old_affine,
                projection=image.projection
            ).pix_to_world(old_x, old_y)

            self.assertAlmostEqual(x0, ox0)
            self.assertAlmostEqual(y0, oy0)
            self.assertAlmostEqual(x1, ox1)
            self.assertAlmostEqual(y1, oy1)

            # Check crop invariance
            xmin, xmax = 100, 300
            ymin, ymax = 50, 200

            cropped = image.crop(xmin, xmax, ymin, ymax)

            # shape check
            self.assertEqual(cropped.xdim(), xmax - xmin)
            self.assertEqual(cropped.ydim(), ymax - ymin)

            # cropped origin must match original pixel location
            cx0, cy0 = cropped.pix_to_world(0, 0)
            oxc, oyc = image.pix_to_world(xmin, ymin)

            self.assertAlmostEqual(cx0, oxc)
            self.assertAlmostEqual(cy0, oyc)

            # cropped far corner must align too
            cx1, cy1 = cropped.pix_to_world(
                cropped.xdim(),
                cropped.ydim()
            )
            ox1, oy1 = image.pix_to_world(xmax, ymax)

            self.assertAlmostEqual(cx1, ox1)
            self.assertAlmostEqual(cy1, oy1)

        # test some image manipulations
        image.flip(axis='y')
        image.data[10,10,:] = np.nan
        image.fill_holes()
        self.assertEqual( np.isfinite( image.data ).all(), True )
        image.blur()

        # extract features
        k, d = image.get_keypoints( band=0 )
        src, dst = image.match_keypoints(k,k,d,d)
        self.assertGreater(len(src), 0 ) # make sure there are some matches...

        # masking
        image.mask( np.sum(image.data,axis=2) > 0.75 )
        self.assertEqual(np.isfinite(image.data).all(), False)
    
    def test_tile_and_mosaic_affine_correctness(self):
        require_test_env(self, "gdal")
        import numpy as np
        import hylite

        from osgeo import gdal

        # Create a rotated / skewed test image with structured signal
        nx, ny, nb = 128, 128, 5

        x = np.linspace(0, 2 * np.pi, nx)
        y = np.linspace(0, 2 * np.pi, ny)
        xx, yy = np.meshgrid(x, y, indexing="ij")

        data = np.zeros((nx, ny, nb), dtype=np.float32)

        for b in range(nb):
            # long-wavelength spatial signal + band-specific phase
            data[..., b] = (
                np.sin(xx * 0.5 + b * 0.3) +
                np.cos(yy * 0.5 - b * 0.2)
            )

        data[::32, ::32, :] = 1.0 # add some spikes
        # normalize to [0,1] for numerical stability
        data -= data.min()
        data /= data.max()

        affine = [
            1000.0,   # x origin
            2.0,      # pixel width
            0.5,      # x skew
            2000.0,   # y origin
            -0.3,     # y skew
            -2.0      # pixel height (north-up)
        ]

        img = hylite.HyImage(
            data,
            affine=affine,
            wav=np.arange(nb)
        )
        img.set_projection_EPSG('EPSG:32633')  # UTM zone 33N

        # Tile image
        tile_size = (32, 32)
        tiles = img.tile(tile_size)

        self.assertGreater(len(tiles), 0)

        # 1. Verify tile affine correctness
        for t in tiles:
            # Pick tile origin pixel in original image
            # Compute expected world coordinate
            tx0, ty0 = t.pix_to_world(0, 0)
            px0, py0 = img.pix_to_world(t.header['xleft'], t.header['ytop'])
            self.assertAlmostEqual( tx0, px0, places=5 )
            self.assertAlmostEqual( ty0, py0, places=5 )

            # Recover pixel offset by inverse mapping
            px, py = img.world_to_pix(tx0, ty0)
            self.assertAlmostEqual(px, t.header['xleft'], places=5)
            self.assertAlmostEqual(py, t.header['ytop'], places=5)

            # World-space invariant
            ox, oy = img.pix_to_world(int(round(px)), int(round(py)))
            self.assertAlmostEqual(tx0, ox, places=6)
            self.assertAlmostEqual(ty0, oy, places=6)

        # 2. Mosaic tiles back into a NEW grid
        for b in ['first', 'mean', 'max', 'min', 'median']:
            mosaic = hylite.HyImage.mosaic(tiles, blend=b, out_shape=[128,128])

            self.assertAlmostEqual(mosaic.affine[0], img.affine[0])
            self.assertAlmostEqual(mosaic.affine[1], img.affine[1])
            self.assertAlmostEqual(mosaic.affine[2], img.affine[2])
            self.assertAlmostEqual(mosaic.affine[3], img.affine[3])
            self.assertAlmostEqual(mosaic.affine[4], img.affine[4])
            self.assertAlmostEqual(mosaic.affine[5], img.affine[5])

            diff = np.abs(mosaic.data - data)
            mask = np.isfinite(diff)
            self.assertLess(np.nanmean(diff[mask]), 1e-3)

class TestHyLibrary(unittest.TestCase):
    def test_library(self):
        require_test_env(self, "opencv") # openCV is used to extract library from seed pixels
        image = io.load(os.path.join(TEST_DATA, "image.hdr"))

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

        # build simple classification from seed regions (sample points are x, y)
        fg = np.zeros((image.xdim(), image.ydim()), dtype=int)
        s = 5
        for i, name in enumerate(image.header.get_class_names()):
            for x, y in image.header.get_sample_points(name):
                fg[max(x - s, 0):min(x + s, fg.shape[0]),
                   max(y - s, 0):min(y + s, fg.shape[1])] = i + 1
        cls = hylite.HyImage(fg[:, :, None])
        cls.header["file type"] = "ENVI Classification"
        cls.header['classes'] = len(image.header.get_class_names())
        cls.header['class names'] = ['background'] + image.header.get_class_names()

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

        # header and array indexing via HyData base
        lib['custom field'] = 'value'
        self.assertEqual(lib['custom field'], 'value')
        self.assertTrue(np.allclose(lib[0, :, 550.0], lib.data[0, :, lib.get_band_index(550.0)]))

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
        require_test_env(self, "lite")
        image = io.load(os.path.join(TEST_DATA, "image.hdr"))


class TestHyCloud(unittest.TestCase):
    def test_cloud(self):
        require_test_env(self, "default")
        cloud = genCloud(npoints=1000, nbands=10)
        self.assertEqual(cloud.point_count(), 1000)
        self.assertEqual(cloud.has_rgb(), True)
        self.assertEqual(cloud.has_normals(), True)
        self.assertEqual(cloud.has_bands(), True)

        n0 = cloud.normals[0, :].copy()
        cloud.flip_normals()
        self.assertEqual(np.sum(n0 + cloud.normals[0, :]), 0)

        cloud.compute_normals(1.0, vb=True)

        cloud.filter_points(0, val=(0.1, 0.5), trim=True)
        self.assertGreaterEqual(np.nanmin(cloud.data[:, 0]), 0.1)
        self.assertLessEqual(np.nanmax(cloud.data[:, 0]), 0.5)
        self.assertLess(cloud.point_count(), 1000)

        cam = Camera(pos=np.array([0.0, 0.0, 10.0]), ori=np.array([0.0, 0.0, 0.0]), fov=30, proj='persp', dims=(1000, 1000))
        cloud.render(cam)

        ortho = cloud.render(cam='ortho', bands='rgb', res=0.05)
        self.assertEqual(ortho.data.ndim, 3)
        self.assertEqual(ortho.band_count(), 3)
        self.assertGreater(ortho.xdim(), 0)
        self.assertGreater(ortho.ydim(), 0)
        self.assertAlmostEqual(ortho.affine[1], 0.05)
        self.assertAlmostEqual(ortho.affine[5], -0.05)

        from hylite.project.basic import proj_ortho
        xyz = np.array([[0.0, 0.0, 1.0], [1.0, 1.0, 2.0]])
        pp, vis = proj_ortho(xyz, np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0]),
                             s=1.0, y_down=True, cull=False)
        np.testing.assert_allclose(pp, [[0.0, 1.0, 1.0], [1.0, 0.0, 2.0]])
        self.assertTrue(vis.all())

        image = genImage(1000, 1000)
        cloud.project(image, cam)


class TestHyFeature(unittest.TestCase):
    def test_multigauss(self):
        require_test_env(self, "basic")
        from hylite import HyFeature
        x = np.linspace(2100., 2400., 500)
        y = HyFeature.gaussian(x, 2200., 200., 0.5)
        self.assertAlmostEqual(np.max(y), 1.0, 2)
        self.assertAlmostEqual(np.min(y), 0.5, 2)
        y = HyFeature.multi_gauss(x, [2200., 2340.], [200., 200.], [0.5, 0.5])
        self.assertAlmostEqual(np.max(y), 1.0, 2)
        self.assertAlmostEqual(np.min(y), 0.5, 2)

    def test_fitting(self):
        require_test_env(self, "basic")

if __name__ == '__main__':
    unittest.main()
