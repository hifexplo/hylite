import numpy as np
from pathlib import Path
import hylite.io as io
from hylite.sensors import Sensor
from scipy.ndimage import median_filter
from hylite.filter import combine
import glob
import os

def radiance_to_brightness_temp(wavenumber_cm, radiance):
    """
    Convert radiance spectra to brightness temperature.
    
    Parameters
    ----------
    wavenumber_cm : array_like
        Wavenumber(s) in cm^-1
    radiance : array_like
        Radiance in W / (m^2 sr cm^-1)
    
    Returns
    -------
    Tb : ndarray
        Brightness temperature in Kelvin
    """
    # Physical constants
    h = 6.62607015e-34     # Planck's constant [J s]
    c = 2.99792458e8       # speed of light [m/s]
    k_B = 1.380649e-23     # Boltzmann constant [J/K]
    
    # Convert wavenumber to SI units [1/m]
    wn_m = np.array(wavenumber_cm) * 100.0  
    
    # Planck radiance inversion
    # Radiance given per unit wavenumber in cm^-1, so formula uses wn in cm^-1
    term = (2 * h * c**2 * wn_m**3) / radiance
    Tb = (h * c * wn_m) / (k_B * np.log1p(term))
    
    return Tb

class TelopsNano(Sensor):
    """
    Implementation of sensor corrections for the Telops Hypercam Nano sensor.
    """

    @classmethod
    def name(cls):
        """
        Returns this sensors name
        """
        return "Nano"

    @classmethod
    def fov(cls):
        """
        Return the (vertical) sensor field of view .
        """
        return 32.3 # TODO - replace this with the correct value once known.

    @classmethod
    def ypixels(cls):
        """
        Return the number of pixels in the y-dimension.
        """
        return 160  # n.b. sensor has 384 pixels but this is resized to 401 on lens correction

    @classmethod
    def xpixels(cls):
        """
        Return the number of pixels in the x-dimension (==1 for line scanners).
        """
        return 320  # the OWL is a line-scanner

    @classmethod
    def pitch(cls):
        """
        Return the pitch of the each pixel in the y-dimension (though most pixels are square).
        """
        return 0.05

    @classmethod
    def correct_image( cls, image, verbose=True, **kwds):
        """
        Apply sensor corrections to an image.

        Args:
            image (hylite.HyImage): a hyImage instance of an image captured using this sensor.
            verbose (bool): true if updates/progress should be printed to the console. Default is False.
            **kwds: Optional keywords include:

                    - bright = true if image should be converted to brightness temperature by inverting the Plank function. Default is True.
                    - denoise = denoising factor for total variation denoising (see skimage docs). Set to 0 (default) to disable. 
                    - flipY = true if image should be flipped on the Y-axis. Default is True.
                    - flipX = true if image should be flipped on the X-axis. Default is True.
        """

        # get kwds
        bright = kwds.get("bright", True)
        denoise = kwds.get('denoise', 0)
        flipY = kwds.get('flipY', True)
        flipX = kwds.get('flipX', True)

        if flipX:
            image.flip('x')
        if flipY:
            image.flip('y')
        

        # convert to brightness temperature
        if bright:
            image.data = radiance_to_brightness_temp( image.get_wavelengths(), image.data ) # N.B. image.wavelengths are in wavenumbers!
        
        # despeckle and denoise
        if denoise > 0:
            from skimage.restoration import denoise_tv_chambolle
            image.data = denoise_tv_chambolle(image.data, weight=denoise, channel_axis=-1) # denoise

        image.set_band_names(None)  # delete band names as they get super annoying
        return image
        
    
    @classmethod
    def correct_folder(cls, path, **kwds):
        """
        Load and average a directory full of TelopsNano images.

        Args:
            path (str): a path to the folder containing the data. All images in this folder are loaded, corrected and then 
            the median spectra for each pixel is returned.
            **kwds: keywords are passed directly to correct_image, except for:
                - verbose = True if print outputs should be made to update progress. Default is True.
        Returns:
            A HyImage containing the corrected and averaged data.
        """
        images = glob.glob(os.path.join( path, '/*.hdr') ) # get all images in directory
        images = [TelopsNano.correct_image(io.load(p)) for p in images] # load and correct them
        median, std = combine( images, method='median', warp=False) # take a median spectra for each pixel
        return median

