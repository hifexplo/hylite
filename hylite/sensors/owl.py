import numpy as np
from pathlib import Path
import hylite.io as io
from hylite.sensors import Sensor
from scipy.ndimage import median_filter


class OWL(Sensor):
    """
    Implementation of sensor corrections for the OWL sensor.
    """

    @classmethod
    def name(cls):
        """
        Returns this sensors name
        """
        return "OWL"

    @classmethod
    def fov(cls):
        """
        Return the (vertical) sensor field of view .
        """
        return 32.3

    @classmethod
    def ypixels(cls):
        """
        Return the number of pixels in the y-dimension.
        """
        return 640  # n.b. sensor has 384 pixels but this is resized to 401 on lens correction

    @classmethod
    def xpixels(cls):
        """
        Return the number of pixels in the x-dimension (==1 for line scanners).
        """
        return 1  # the OWL is a line-scanner

    @classmethod
    def pitch(cls):
        """
        Return the pitch of the each pixel in the y-dimension (though most pixels are square).
        """
        return 0.05

    @classmethod
    def correct_image(cls, image, verbose=True, **kwds):

        """
        Apply sensor corrections to an image.

        Args:
            image (hylite.HyImage): a hyImage instance of an image captured using this sensor.
            verbose (bool): true if updates/progress should be printed to the console. Default is False.
            **kwds: Optional keywords include:

                 - rad = true if image should be converted to radiance by applying dark and white references. Default is True.
                 - bpr = replace bad pixels (only for raw data). Default is True.
                 - flip = true if image should be flipped (if camera mounted backwards in core
                          scanner). Default is False.
        """

        # get kwds
        rad = kwds.get("rad", True)
        bpr = kwds.get("bpr", True)

        lim = image.get_band_index(12368.)
        image.data = image.data[..., :lim]
        image.set_wavelengths(image.get_wavelengths()[:lim])

        if rad:
            if verbose: print("Converting to radiance... ", end="", flush="True")

            # convert from int to float
            image.data = image.data.astype(np.float32)

            # flag infs
            image.data[image.data == 65535.] = np.nan

            # apply dark reference
            if cls.dark is None:
                print("Warning: dark calibration not found; no dark correction was applied! Something smells dodgy...")
            else:
                dref = np.nanmean(cls.dark.data, axis=1)  # calculate dark reference
                image.data[:, :, :] -= dref[:, None, :lim]  # apply dark calibration

            # apply white reference (if specified)
            if not cls.white is None:
                # calculate white reference radiance
                white = np.nanmean(cls.white.data.astype(np.float32),
                                   axis=1) - dref  # average each line and subtract dark reference
                # also estimate noise per-band (useful for eg., MNFs)
                noise = np.nanstd(white, axis=0)

                refl = np.ones(white.shape[1])  # assume pure white

                # apply white reference
                cfac = refl[None, :] / white
                image.data[:, :, :] *= cfac[:, None, :lim]
                noise *= np.nanmean(cfac, axis=0)
                image.header['band_noise'] = noise
            if verbose: print("DONE.")

        ##############################################################
        # replace bad pixels with an average of the surrounding ones
        ##############################################################
        if bpr:
            if verbose: print("Filtering bad pixels... ", end="", flush="True")
            invalids = np.argwhere(np.isnan(image.data) | np.isinf(image.data))  # search for bad pixels
            for px, py, band in invalids:
                n = 0
                sum = 0
                for xx in range(px - 1, px + 2):
                    for yy in range(py - 1, py + 2):
                        if xx == px and yy == py: continue  # skip invalid pixel
                        if xx < 0 or yy < 0 or xx >= image.data.shape[0] or yy >= image.data.shape[
                            1]: continue  # skip out of bounds pixels
                        if image.data[xx][yy][band] == np.nan or image.data[xx][yy][
                            band] == np.inf: continue  # maybe neighbour is nan also
                        n += 1
                        sum += image.data[xx][yy][band]
                if n > 0: sum /= n  # do averaging
                image.data[px][py][band] = sum
            if verbose: print("DONE.")

        # Denoise LWIR along sensor plane
        image.data = median_filter(image.data, size=(3, 1, 3), mode="mirror")

        # rotate image so that scanning direction is horizontal rather than vertical)
        image.data = np.rot90(image.data)  # np.transpose(remap, (1, 0, 2))
        image.data = np.flip(image.data, axis=1)
        image.set_band_names(None)  # delete band names as they get super annoying
    @classmethod
    def correct_folder(cls, path, **kwds):

        """
        Many sensors use simple/common data structures to store data/headers/dark reference etc. Hence it is often easiest
        to pass an output folder to the sensor for correction.

        Args:
            path (str): a path to the folder containing the sensor specific data.
            **kwds: keywords are passed directly to correct_image, except for:

                - verbose = True if print outputs should be made to update progress. Default is True.

        Returns:
            A hyImage to which all sensor-specific corrections have been applied. Note that this will generally not include
           topographic or atmospheric corrections.

        """

        verbose = kwds.get("verbose", True)
        kwds["verbose"] = verbose

        imgs = [str(p) for p in Path(path).rglob("capture/*.hdr")]  # all image files [including data]
        dark = [str(p) for p in Path(path).rglob("capture/DARKREF*.hdr")]  # dark reference file
        white = [str(p) for p in Path(path).rglob("capture/WHITEREF*.hdr")]  # an white reference data (core scanner)
        refl = [str(p) for p in Path(path).rglob("capture/REFL*.hdr")]  # any processed reflectance data (SiSu Rock)
        for d in dark:
            del imgs[imgs.index(d)]
        for w in white:
            del imgs[imgs.index(w)]
        for r in refl:
            del imgs[imgs.index(r)]

        if len(imgs) > 1 or len(
            dark) > 1: assert False, "Error - multiple scenes found in folder. Double check file path..."
        if len(imgs) == 0 or len(
            dark) == 0: assert False, "Error - no image or dark calibration found in folder. Double check file path... %s" % path

        if verbose: print('\nLoading image %s' % imgs[0])

        # load image
        image = io.load(imgs[0])
        OWL.set_dark_ref(dark[0])

        if len(white) > 0:  # white reference exists
            OWL.set_white_ref(white[0])

        # correct
        OWL.correct_image(image, **kwds)

        # return corrected image
        return image
