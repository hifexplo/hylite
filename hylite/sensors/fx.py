from .sensor import Sensor
import numpy as np
import os

"""
Implementation of shared processing for all FX class sensors. This class should not be instantiated though - use FX10 and FX17 instead. 
"""
# noinspection PyAbstractClass
class FX(Sensor):

    @classmethod
    def correct_image(cls, image, verbose=False, **kwds):

        """
        Apply sensor corrections to an image. For the FX series sensors this just applies a dark correction.

        *Arguments*:
         - image = a hyImage instance of an image captured using this sensor.
         - verbose = true if updates/progress should be printed to the console. Default is False.
        *Keywords*:
         - keywords as defined by inherited classes (sensor specific)
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

        *Arguments*:
         - path = a path to the folder containing the sensor specific data.

        *Keywords*:
         - keywords as defined by inherited classes (sensor specific)

        *Returns*:
         - a hyImage to which all sensor-specific corrections have been applied. Note that this will generally not include
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