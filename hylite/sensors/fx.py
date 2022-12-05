import os
import numpy as np
from pathlib import Path
import hylite.io as io
from hylite.sensors import Sensor

"""
Implementation of shared processing for all FX class sensors. This class should not be instantiated though - use FX10 and FX17 instead. 
"""
# noinspection PyAbstractClass
class FX(Sensor):

    @classmethod
    def correct_image(cls, image, verbose=False, **kwds):

        """
        Apply sensor corrections to an image. For the FX series sensors this just applies a dark correction.

        Args:
            image (hylite.HyImage): a hyImage instance of an image captured using this sensor.
            verbose (bool): true if updates/progress should be printed to the console. Default is False.
        """

        if cls.dark is None:
            print("Warning: dark calibration not found; could not perform radiance correction.")

        else:
            #convert to float
            image.data =image.data.astype(np.float32)

            #calculate pixel corrections from dark reference
            if verbose: print("Applying dark calibration... ", end="", flush="True")
            dref = np.mean(cls.dark.data, axis=1)

            #apply dark correction
            image.data -= dref[:,None,:]

            if verbose: print("DONE.")
            
        #transpose and flip (fx images are always rotated 90 deg for some reason)
        image.data = np.transpose(image.data,(1,0,2))
        image.data = np.flip(image.data,axis=1)

    @classmethod
    def correct_folder(cls, path):

        """
        Many sensors use simple/common data structures to store data/headers/dark reference etc. Hence it is often easiest
        to pass an output folder to the sensor for correction.

        Args
            path (str): a path to the folder containing the sensor specific data.

        Returns:
            A hyImage to which all sensor-specific corrections have been applied. Note that this will generally not include
            topographic or atmospheric corrections.

        """

        assert os.path.exists(path), "Error: invalid directory '$s'" % path

        assert False, "Error - folder correction for FX data is not yet implemented."

        #todo

        pass

class FX10(FX):

    """
    Specific details/methods for FX10 sensor
    """
    @classmethod
    def name(cls):
        """
        Returns this sensors name
        """
        return "FX10"

    @classmethod
    def fov(cls):
        """
        Return the (vertical) sensor field of view .
        """
        return 54.0

    @classmethod
    def ypixels(cls):
        """
        Return the number of pixels in the y-dimension.
        """
        return 1024

    @classmethod
    def xpixels(cls):
        """
        Return the number of pixels in the x-dimension (==1 for line scanners).
        """
        return 1 # line scanner

    @classmethod
    def pitch(cls):
        """
        Return the pitch of the each pixel in the y-dimension (though most pixels are square).
        """
        return 9.97 / 1000

class FX17(FX):


    """
    Specific details/methods for FX10 sensor
    """

    @classmethod
    def name(cls):
        """
        Returns this sensors name
        """
        return "FX17"

    @classmethod
    def fov(cls):
        """
        Return the (vertical) sensor field of view .
        """
        return 75.0

    @classmethod
    def ypixels(cls):
        """
        Return the number of pixels in the y-dimension.
        """
        return 640

    @classmethod
    def xpixels(cls):
        """
        Return the number of pixels in the x-dimension (==1 for line scanners).
        """
        return 1 # line scanner

    @classmethod
    def pitch(cls):
        """
        Return the pitch of the each pixel in the y-dimension (though most pixels are square).
        """
        return 18.7 / 1000

class FX50(FX):
    """
    Implementation of sensor corrections for the FX50 sensor.
    """

    @classmethod
    def name(cls):
        """
        Returns this sensors name
        """
        return "FX50"

    @classmethod
    def fov(cls):
        """
        Return the (vertical) sensor field of view .
        """
        return 45.

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
        return 1  # the FX50 is a line-scanner

    @classmethod
    def pitch(cls):
        """
        Return the pitch of the each pixel in the y-dimension (though most pixels are square).
        """
        return 0.07

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
                image.data[:, :, :] -= dref[:, None, :]  # apply dark calibration

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
                image.data[:, :, :] *= cfac[:, None, :]
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
            **kwds: Optional keywords include:

                - verbose = True if print outputs should be made to update progress. Default is True.
                - other keywords are passed directly to correct_image.

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
        FX50.set_dark_ref(dark[0])

        if len(white) > 0:  # white reference exists
            FX50.set_white_ref(white[0])

        # correct
        FX50.correct_image(image, **kwds)

        # return corrected image
        return image
