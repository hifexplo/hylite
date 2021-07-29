from abc import ABCMeta, abstractmethod
import os
import numpy as np
import hylite.io as io
from hylite.project import Camera

class Sensor(object):
    """
    Base class for hyperspectral camera specific implementations to inherit. This defines the basic functions that
    every sensor class is expected to implement.
    """

    __metaclass__ = ABCMeta

    #set all default references to none
    dark = None # dark reference image
    white = None # white reference image
    white_spectra = None # reflectance of white (or grey) target

    @abstractmethod
    def name(cls):
        """
        Returns this sensors name
        """

        pass #implement this in inherited classes

    @classmethod
    def focal_length(cls):
        """
        Return the sensor focal length (in pixels).
        """
        return cls.ypixels() / (2 * np.tan(np.deg2rad(cls.fov() / 2)))

    @abstractmethod
    def fov(cls):
        """
        Return the (vertical) sensor field of view .
        """
        pass # implement this in inherited classes

    @abstractmethod
    def ypixels(cls):
        """
        Return the number of pixels in the y-dimension.
        """
        pass # implement this in inherited classes

    @abstractmethod
    def xpixels(cls):
        """
        Return the number of pixels in the x-dimension (==1 for line scanners).
        """
        pass # implement this in inherited classes

    @abstractmethod
    def pitch(cls):
        """
        Return the pitch of the each pixel in the x-dimension (though most pixels are square). Note that for
        line scanners this will be an angular pitch.
        """
        pass # implement this in inherited classes

    @classmethod
    def correct_image(cls, image, verbose=False, **kwds):
        """
        Apply sensor corrections to an image.

        *Arguments*:
         - image = a hyImage instance of an image captured using this sensor.
         - verbose = true if updates/progress should be printed to the console. Default is False.
        *Keywords*:
         - keywords as defined by inherited classes (sensor specific)
        """

        pass  # implement this in inherited classes

    @abstractmethod
    def correct_folder(cls, path ):
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

        pass  # implement this in inherited classes

    @classmethod
    def set_dark_ref(cls, image):
        """
        Sets the dark reference to be used for sensor corrections.
        """

        if isinstance(image, str):
            assert os.path.exists(image), "Error: %s is not a valid file path or hyperspectral image."
            image = io.load(image)
        assert isinstance(image, io.HyImage) or image is None, "Error: dark reference must be an image or None."
        Sensor.dark = image

    @classmethod
    def set_white_ref_spectra(cls, spectra):
        """
        Set the known reflectance spectra of the white reference.

        *Arguments*:
         - spectra = hylite.reference.spectra.Target instance defining the known reflectance spectra of the white reference.
                     If None, then the target is assumed to be pure white (reflectance = 1).
        """
        Sensor.white_spectra = spectra # store reference spectra

    @classmethod
    def set_white_ref(cls, image):
        """
        Sets the white reference to be used for sensor corrections.

        *Arguments*:
         - image = the white reference image.
        """

        if isinstance(image, str):
            assert os.path.exists(image), "Error: %s is not a valid file path or hyperspectral image."
            image = io.load(image)
        assert isinstance(image, io.HyImage), "Error: white reference must be an image."
        Sensor.white = image # store white reference

    @classmethod
    def get_camera(cls):
        """
        Return a hylite.project.Camera object with fov, pitch and projection corresponding to this sensor.
        """
        if cls.xpixels() == 1: # is this a line-scanner?
            return Camera(np.zeros(3), np.zeros(3), 'pano', cls.fov(), (cls.xpixels(), cls.ypixels()), cls.pitch())
        else:
            return Camera(np.zeros(3), np.zeros(3), 'pano', cls.fov(), (cls.xpixels(), cls.ypixels()) )
