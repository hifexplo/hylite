import unittest
import numpy as np
from tests import genImage
import hylite

class TestHyImage(unittest.TestCase):
    def test_image(self):

        # test constructor
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
        import numpy as np
        import hylite

        try:
            from osgeo import gdal
        except ImportError:
            self.skipTest("GDAL not available")

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

if __name__ == '__main__':
    unittest.main()
