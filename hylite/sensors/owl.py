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
                 - denoise = denoising factor for total variation denoising (see skimage docs). Default is 1.
                 - dead = percentage of pixels expected to be dead. Default is 2.5.
                 - blink = percentage of pixels expected to be blinking. Default is 10.
                 - flip = true if image should be flipped (if camera mounted backwards in core
                          scanner). Default is False.
        """

        # get kwds
        rad = kwds.get("rad", True)
        lim = image.get_band_index(kwds.get("lim", 12368.))
        denoise = kwds.get('denoise', 1)
        dead = kwds.get('denoise', 2.5)
        blink = kwds.get('blink', 10)
        flip = kwds.get("flip", False)

        # drop pixels above specified limit wavelength (these are shite)
        image.data = image.data[..., :lim]
        image.set_wavelengths(image.get_wavelengths()[:lim])
        image.rot90() # rotate so cross-track is y-axis

        if rad:
            if verbose: print("Converting to radiance... ", end="", flush="True")

            # convert from int to float
            image.data = image.data.astype(np.float32)

            # store exposure histogram for QAQC
            counts, bins = np.histogram(image.data.ravel(), bins=50, range=(0, 2**14) )
            image.header['Bins'] = bins
            image.header['Raw Levels'] = counts
            image.header['ninvalid'] = counts[-1]
            
            # flag over-exposed pixels
            image.data[image.data >= (2**14 - 1)] = np.nan

            # load white reference
            whiteref=None
            if cls.white is not None:
                whiteref = cls.white.copy()
                whiteref.rot90()
                whiteref.data = whiteref.data[:,:,:lim]
                whiteref.data = whiteref.data.astype(np.float32)

            # apply dark reference
            if cls.dark is None:
                print("Warning: dark calibration not found; no dark correction was applied! Something smells dodgy...")
            else:
                darkref = cls.dark.copy()
                darkref.rot90()
                darkref.data = darkref.data[:,:,:lim]
                darkref.data = darkref.data.astype(np.float32)

                # detect blinking pixels
                darkmean = np.mean( darkref.data, axis=0 ) # calculate dark reference
                darksigma = np.std( darkref.data, axis=0)
                blinkmask = darksigma > np.percentile(darksigma,100-blink)
                
                # do dark subtraction
                for img in [image, whiteref]:
                    if img is not None:
                        img.data = img.data.astype(np.float32)
                
                        # detect dead pixels
                        imgsigma = np.std( img.data, axis=0 )
                        deadpixels = imgsigma < np.percentile(imgsigma, dead)

                        # store darkref histogram for QAQC
                        counts, bins = np.histogram(darkmean.ravel(), bins=50, range=(0, 2**14) )
                        img.header['Dark Levels'] = counts

                        # despeckle and denoise
                        from skimage.filters import median
                        from skimage.morphology import white_tophat, black_tophat, ball
                        from skimage.restoration import denoise_tv_chambolle
                        mask = np.logical_or(blinkmask)
                        smooth = median( img.data, footprint=np.ones((1,5,5)))
                        img.data[ :, mask ] = smooth[:, mask ] # replace dead / stuck pixels
                        footprint = ball(3)[::3,::3,:] # ellipse footprint
                        img.data = img.data + black_tophat(img.data, footprint) # dark despeckle
                        img.data = img.data - white_tophat(img.data, footprint) # bright despeckle
                        #img.data = denoise_tv_chambolle(img.data, weight=denoise, channel_axis=-1) # denoise

                    # store corrected counts for QAQC
                    counts, bins = np.histogram(img.data.ravel(), bins=50, range=(0, 2**14) )
                    img.header['Corrected Levels'] = counts

            # apply white reference (if specified)
            if whiteref is None:
                # calculate white reference radiance
                refl = np.ones(whiteref.data.shape[-1])  # assume pure white

                # apply white reference
                cfac = refl[None, :] / np.nanmean( whiteref.data, axis=0 )
                image.data[:, :, :] *= cfac[None, :, :]

            if verbose: print("DONE.")

        # also estimate noise per-band (useful for eg., MNFs)
        if False:
            if cls.white is not None:
                white = median_filter(white, size=(5,3,5), mode='mirror') # also apply to white panel for noise estimation
                noise = np.nanstd(white, axis=(0, 1))
                image.header['band noise'] = noise

        if flip:
            image.data = np.flip(image.data, axis=1) # rotate image so that scanning direction is horizontal rather than vertical)
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
