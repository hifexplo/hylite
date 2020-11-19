import numpy as np
from pathlib import Path
import cv2
import numpy.ma as ma
import hylite.io as io
from .sensor import Sensor

class Fenix(Sensor):
    """
    Implementation of sensor corrections for the Fenix sensor.
    """

    cal2 = None
    cal4 = None
    cal8 = None

    @classmethod
    def name(cls):
        """
        Returns this sensors name
        """
        return "FENIX"

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
        return 384 # n.b. sensor has 384 pixels but this is resized to 401 on lens correction

    @classmethod
    def xpixels(cls):
        """
        Return the number of pixels in the x-dimension (==1 for line scanners).
        """
        return 1 # the Fenix is a line-scanner

    @classmethod
    def pitch(cls):
        """
        Return the pitch of the each pixel in the y-dimension (though most pixels are square).
        """
        return 0.084

    @classmethod
    def correct_image(cls, image, verbose=True, **kwds):

        """
        Apply sensor corrections to an image.

        *Arguments*:
         - image = a hyImage instance of an image captured using this sensor.
         - verbose = true if updates/progress should be printed to the console. Default is False.
        *Keywords*:
         - rad = true if image should be converted to radiance by applying dark and white references. Default is True.
         - bpr = replace bad pixels (only for raw data). Default is True.
         - shift = shift bands to account for time-delay between their acquisitions.
                   Only use for near-field sensing (e.g. drill-core scans). Default is False.
         - flip = true if image should be flipped before applying lens correction (if camera mounted backwards in core
                  scanner). Default is False.
         - lens = apply GLTX lens correction to remove lens distortion. Default is True.
        """

        #get kwds
        rad = kwds.get("rad", True)
        bpr = kwds.get("bpr", True)
        shift = kwds.get("shift", False)
        lens = kwds.get("lens", True)

        if rad:
            if verbose: print("Converting to radiance... ", end="", flush="True")

            #convert from int to float
            image.data = image.data.astype(np.float32)

            # apply dark reference
            if cls.dark is None:
                print("Warning: dark calibration not found; no dark correction was applied! Something smells dodgy...")
            else:
                dref = np.nanmean(cls.dark.data, axis=1) #calculate dark reference
                image.data[:, :, :] -= dref[:, None, :]  # apply dark calibration

            # apply laboratory calibration
            assert not image.header is None, "Error: image must be linked to a header file (.hdr) for FENIX correction."
            binning = int(image.header.get('binning', [2])[0])
            if binning == 2:
                if Fenix.cal2 is None:
                    Fenix.cal2 = io.loadWithGDAL( str(Path(__file__).parent / "calibration_data/fenix/Radiometric_2x2_1x1.hdr") )
                cal = Fenix.cal2.data[:, 0, :]
                calw = Fenix.cal2.get_wavelengths()
            elif binning == 4:
                if Fenix.cal4 is None:
                    Fenix.cal4 = io.loadWithGDAL( str(Path(__file__).parent / "calibration_data/fenix/Radiometric_4x2_1x1.hdr") )
                cal = Fenix.cal4.data[:, 0, :]
                calw = Fenix.cal4.get_wavelengths()
            elif binning == 8:
                if Fenix.cal8 is None:
                    Fenix.cal8 = io.loadWithGDAL( str(Path(__file__).parent / "calibration_data/fenix/Radiometric_8x2_1x1.hdr") )
                cal = Fenix.cal8.data[:, 0, :]
                calw = Fenix.cal8.get_wavelengths()
            else:
                assert False, "Error: calibration data for binning=%d does not exist" % binning

            # convert from saturation to radiance based on laboratory calibration file
            cal = cal[:, [np.argmin(calw < w) for w in
                          image.get_wavelengths()]]  # match image wavelengths to calibration wavelengths
            image.data[:, :, :] *= cal[:, None, :]  # apply to image

            # apply white reference (if specified)
            if not cls.white is None:

                # calculate white reference radiance
                white = np.nanmean( cls.white.data.astype(np.float32), axis=1 ) - dref # average each line and subtract dark reference
                white *= cal # apply laboratory calibration to white reference

                # extract white (or grey) reference reflectance
                if cls.white_spectra is None:
                    refl = np.zeros(white.shape[1]) + 1.0  # assume pure white
                else:
                    # get known target spectra
                    refl = cls.white_spectra.get_reflectance()

                    # match bands with this image
                    idx = [ np.argmin( cls.white_spectra.get_wavelengths() < w ) for w in image.get_wavelengths() ]
                    refl = refl[idx]

                # apply white reference
                cfac = refl[None,:] / white
                image.data[:,:,:] *= cfac[ :, None, : ]

            if verbose: print("DONE.")

        ##############################################################
        #replace bad pixels with an average of the surrounding ones
        ##############################################################
        if bpr:
            if verbose: print("Filtering bad pixels... ", end="", flush="True")
            invalids = np.argwhere(np.isnan(image.data) | np.isinf(image.data)) #search for bad pixels
            for px, py, band in invalids:
                n = 0
                sum = 0
                for xx in range(px-1,px+2):
                    for yy in range(py-1,py+2):
                        if xx == px and yy == py: continue #skip invalid pixel
                        if xx < 0 or yy < 0 or xx >= image.data.shape[0] or yy >= image.data.shape[1]: continue #skip out of bounds pixels
                        if image.data[xx][yy][band] == np.nan or image.data[xx][yy][band] == np.inf: continue #maybe neighbour is nan also
                        n += 1
                        sum += image.data[xx][yy][band]
                if n > 0: sum /= n #do averaging
                image.data[px][py][band] = sum
            if verbose: print("DONE.")

        ######################################################################################
        #sensor alignment - identify tie points and apply rigid transform to second sensor
        ######################################################################################
        if shift:
            if verbose: print("Correcting sensor shift (SIFT)... ", end="", flush="True")

            #extract sift features from a band in each sensor
            m = 'sift' #matching method
            xp1, des1 = image.get_keypoints(100,method=m)
            xp2, des2 = image.get_keypoints(200,method=m)

            #match features
            src_pts, dst_pts = io.HyImage.match_keypoints(xp1,xp2,des1,des2,method=m)

            #use RANSAC and a homography transform model to discard bad points
            H, status = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 3.0)
            dst_mask = dst_pts[:, 0, :] * status
            src_mask = src_pts[:, 0, :] * status
            dst_mask = dst_mask[dst_mask.all(1)]
            src_mask = src_mask[src_mask.all(1)]
            dst_mask = np.expand_dims(dst_mask, axis=1)
            src_mask = np.expand_dims(src_mask, axis=1)

            #estimate translation matrix
            M = cv2.estimateRigidTransform(dst_mask, src_mask, False)

            #transform bands from second sensor (>175)
            for i in range(175, image.band_count()):
                image.data[:, :, i] = cv2.warpAffine(image.data[:, :, i], M, (image.data.shape[1], image.data.shape[0]))

            if verbose: print("DONE.")

        #############################
        #apply lens correction
        #############################
        if lens:
            if verbose: print("Applying lens correction... ", end="", flush="True")

            # load pixel offsets
            m = np.array([[0.0, 1.14, 2.28, 3.41, 4.55, 5.68, 6.81, 7.94, 9.07, 10.19, 11.32, 12.44, 13.56, 14.68, 15.8, 16.92, 18.03, 19.15, 20.26, 21.37, 22.48, 23.59, 24.69, 25.8, 26.9, 28.0, 29.11, 30.21, 31.3, 32.4, 33.5, 34.59, 35.68, 36.78, 37.87, 38.95, 40.04, 41.13, 42.21, 43.3, 44.38, 45.46, 46.54, 47.62, 48.7, 49.78, 50.85, 51.93, 53.0, 54.07, 55.14, 56.21, 57.28, 58.35, 59.42, 60.48, 61.54, 62.61, 63.67, 64.73, 65.79, 66.85, 67.91, 68.97, 70.02, 71.08, 72.13, 73.18, 74.24, 75.29, 76.34, 77.39, 78.44, 79.48, 80.53, 81.58, 82.62, 83.67, 84.71, 85.75, 86.79, 87.83, 88.87, 89.91, 90.95, 91.99, 93.02, 94.06, 95.1, 96.13, 97.16, 98.2, 99.23, 100.26, 101.29, 102.32, 103.35, 104.38, 105.41, 106.43, 107.46, 108.48, 109.51, 110.53, 111.56, 112.58, 113.6, 114.63, 115.65, 116.67, 117.69, 118.71, 119.73, 120.75, 121.76, 122.78, 123.8, 124.81, 125.83, 126.84, 127.86, 128.87, 129.89, 130.9, 131.91, 132.93, 133.94, 134.95, 135.96, 136.97, 137.98, 138.99, 140.0, 141.01, 142.02, 143.03, 144.03, 145.04, 146.05, 147.05, 148.06, 149.07, 150.07, 151.08, 152.08, 153.09, 154.09, 155.09, 156.1, 157.1, 158.1, 159.11, 160.11, 161.11, 162.11, 163.12, 164.12, 165.12, 166.12, 167.12, 168.12, 169.12, 170.12, 171.12, 172.12, 173.12, 174.12, 175.12, 176.12, 177.12, 178.12, 179.12, 180.12, 181.12, 182.12, 183.12, 184.12, 185.11, 186.11, 187.11, 188.11, 189.11, 190.11, 191.1, 192.1, 193.1, 194.1, 195.1, 196.1, 197.09, 198.09, 199.09, 200.09, 201.09, 202.09, 203.08, 204.08, 205.08, 206.08, 207.08, 208.08, 209.08, 210.08, 211.08, 212.08, 213.07, 214.08, 215.07, 216.07, 217.08, 218.07, 219.08, 220.08, 221.08, 222.08, 223.08, 224.08, 225.08, 226.08, 227.09, 228.09, 229.09, 230.09, 231.1, 232.1, 233.11, 234.11, 235.11, 236.12, 237.12, 238.13, 239.13, 240.14, 241.15, 242.15, 243.16, 244.17, 245.17, 246.18, 247.19, 248.2, 249.21, 250.22, 251.23, 252.24, 253.25, 254.26, 255.27, 256.29, 257.3, 258.31, 259.32, 260.34, 261.35, 262.37, 263.38, 264.4, 265.42, 266.43, 267.45, 268.47, 269.49, 270.51, 271.53, 272.55, 273.57, 274.59, 275.61, 276.63, 277.66, 278.68, 279.71, 280.73, 281.76, 282.78, 283.81, 284.84, 285.87, 286.9, 287.93, 288.96, 289.99, 291.02, 292.05, 293.09, 294.12, 295.15, 296.19, 297.23, 298.26, 299.3, 300.34, 301.38, 302.42, 303.46, 304.5, 305.55, 306.59, 307.63, 308.68, 309.73, 310.77, 311.82, 312.87, 313.92, 314.97, 316.02, 317.07, 318.13, 319.18, 320.24, 321.29, 322.35, 323.41, 324.47, 325.53, 326.59, 327.65, 328.71, 329.78, 330.84, 331.91, 332.98, 334.04, 335.11, 336.18, 337.25, 338.33, 339.4, 340.47, 341.55, 342.63, 343.7, 344.78, 345.86, 346.95, 348.03, 349.11, 350.2, 351.28, 352.37, 353.46, 354.55, 355.64, 356.73, 357.83, 358.92, 360.02, 361.11, 362.21, 363.31, 364.41, 365.52, 366.62, 367.73, 368.83, 369.94, 371.05, 372.16, 373.27, 374.39, 375.5, 376.62, 377.74, 378.86, 379.98, 381.1, 382.22, 383.35, 384.47, 385.6, 386.73, 387.86, 389.0, 390.13, 391.27, 392.4, 393.54, 394.68, 395.82, 396.97, 398.11, 399.26, 400.41],
                          [11.82, 11.64, 11.46, 11.28, 11.1, 10.93, 10.76, 10.58, 10.41, 10.24, 10.07, 9.91, 9.74, 9.58,9.41, 9.25, 9.09, 8.93, 8.77, 8.61, 8.45, 8.3, 8.15, 7.99, 7.84, 7.69, 7.54, 7.39, 7.24, 7.1,6.95, 6.81, 6.67, 6.53, 6.39, 6.25, 6.11, 5.97, 5.84, 5.7, 5.57, 5.44, 5.3, 5.17, 5.05, 4.92,4.79, 4.66, 4.54, 4.42, 4.29, 4.17, 4.05, 3.93, 3.81, 3.7, 3.58, 3.46, 3.35, 3.24, 3.12,3.01, 2.9, 2.79, 2.69, 2.58, 2.47, 2.37, 2.26, 2.16, 2.06, 1.96, 1.86, 1.76, 1.66, 1.56,1.47, 1.37, 1.28, 1.18, 1.09, 1.0, 0.91, 0.82, 0.73, 0.64, 0.56, 0.47, 0.39, 0.3, 0.22, 0.14,0.06, -0.02, -0.1, -0.18, -0.26, -0.33, -0.41, -0.48, -0.56, -0.63, -0.7, -0.77, -0.84,-0.91, -0.98, -1.05, -1.11, -1.18, -1.24, -1.31, -1.37, -1.43, -1.49, -1.55, -1.61, -1.67,-1.73, -1.79, -1.84, -1.9, -1.95, -2.0, -2.06, -2.11, -2.16, -2.21, -2.26, -2.31, -2.35,-2.4, -2.45, -2.49, -2.53, -2.58, -2.62, -2.66, -2.7, -2.74, -2.78, -2.82, -2.86, -2.89,-2.93, -2.96, -3.0, -3.03, -3.06, -3.09, -3.12, -3.15, -3.18, -3.21, -3.24, -3.27, -3.29,-3.32, -3.34, -3.36, -3.39, -3.41, -3.43, -3.45, -3.47, -3.49, -3.5, -3.52, -3.54, -3.55,-3.57, -3.58, -3.59, -3.6, -3.62, -3.63, -3.64, -3.64, -3.65, -3.66, -3.67, -3.67, -3.68,-3.68, -3.68, -3.68, -3.69, -3.69, -3.69, -3.69, -3.68, -3.68, -3.68, -3.67, -3.67, -3.66,-3.66, -3.65, -3.64, -3.63, -3.62, -3.61, -3.6, -3.59, -3.57, -3.56, -3.55, -3.53, -3.52,-3.5, -3.48, -3.46, -3.44, -3.42, -3.4, -3.38, -3.36, -3.33, -3.31, -3.28, -3.26, -3.23,-3.2, -3.17, -3.14, -3.11, -3.08, -3.05, -3.02, -2.99, -2.95, -2.91, -2.88, -2.84, -2.8,-2.77, -2.73, -2.69, -2.65, -2.6, -2.56, -2.52, -2.47, -2.43, -2.38, -2.34, -2.29, -2.24,-2.19, -2.14, -2.09, -2.03, -1.98, -1.93, -1.87, -1.82, -1.76, -1.7, -1.65, -1.59, -1.53,-1.47, -1.4, -1.34, -1.28, -1.21, -1.15, -1.08, -1.02, -0.95, -0.88, -0.81, -0.74, -0.67,-0.59, -0.52, -0.45, -0.37, -0.29, -0.22, -0.14, -0.06, 0.02, 0.1, 0.18, 0.27, 0.35, 0.43,0.52, 0.61, 0.69, 0.78, 0.87, 0.96, 1.05, 1.15, 1.24, 1.33, 1.43, 1.53, 1.62, 1.72, 1.82,1.92, 2.02, 2.12, 2.23, 2.33, 2.44, 2.54, 2.65, 2.76, 2.87, 2.98, 3.09, 3.2, 3.32, 3.43,3.55, 3.66, 3.78, 3.9, 4.02, 4.14, 4.26, 4.38, 4.51, 4.63, 4.76, 4.89, 5.02, 5.15, 5.28,5.41, 5.54, 5.68, 5.81, 5.95, 6.08, 6.22, 6.36, 6.5, 6.65, 6.79, 6.93, 7.08, 7.23, 7.37,7.52, 7.67, 7.82, 7.98, 8.13, 8.29, 8.44, 8.6, 8.76, 8.92, 9.08, 9.24, 9.41, 9.57, 9.74, 9.9,10.07, 10.24, 10.41, 10.59, 10.76, 10.93, 11.11, 11.29, 11.47, 11.65, 11.83, 12.01, 12.19,12.38, 12.57, 12.75, 12.94, 13.13, 13.33, 13.52]
                          ]).T

            # flip x-axis correction if camera was mounted backwards
            if kwds.get("flip", False):
                m[:,1] *= -1

            # convert to displacement vectors
            dmap = np.zeros((image.data.shape[0], image.data.shape[1], 2))
            dmap[:, :, 0] += -m[:, None, 1]  # displacements in x
            dmap[:, :, 0] -= np.min(-m[:, 1])  # avoid negative displacements
            dmap[:, :, 1] += (m[:, 0] - np.arange(image.data.shape[0]))[:, None]  # displacements in y

            # calculate width/height of corrected image
            width = int(image.data.shape[1] + np.max(m[:, 1]) - np.min(m[:, 1]))
            height = int(np.ceil(np.max(m[:, 0])))

            # resize displacement map to output dimensions
            dmap = cv2.resize(dmap, (width, height), cv2.INTER_LINEAR)

            # use displacement vectors to calculate mapping from output coordinates to original coordinates
            xx, yy = np.meshgrid(range(dmap.shape[1]), range(dmap.shape[0]))
            idx = np.dstack([xx, yy]).astype(np.float32)
            idx[:, :, 0] -= dmap[:, :, 0]
            idx[:, :, 1] -= dmap[:, :, 1]

            # apply remapping
            if image.data.shape[-1] < 512:  # open-cv cannot handle more than 512 bands at a time
                remap = cv2.remap(image.data, idx, None, cv2.INTER_LINEAR)
            else:  # we need to split into different stacks with < 512 bands and then recombine
                remap = []
                mn = 0
                mx = 500
                while mn < image.data.shape[-1]:
                    if mx > image.data.shape[-1]:
                        mx = image.data.shape[-1]

                    # apply mapping to slice of bands
                    remap.append(cv2.remap(image.data[:, :, mn:mx], idx, None, cv2.INTER_LINEAR))

                    # get next slice
                    mn = mx
                    mx += 500

                # stack
                remap = np.dstack(remap)

            image.data = remap
            if verbose: print("DONE.")

        # rotate image so that scanning direction is horizontal rather than vertical)
        image.data = np.rot90(image.data)  # np.transpose(remap, (1, 0, 2))
        image.data = np.flip(image.data, axis=1)

    @classmethod
    def correct_folder(cls, path, **kwds):

        """
        Many sensors use simple/common data structures to store data/headers/dark reference etc. Hence it is often easiest
        to pass an output folder to the sensor for correction.

        *Arguments*:
         - path = a path to the folder containing the sensor specific data.

        *Keywords*:
         - verbose = True if print outputs should be made to update progress. Default is True.
         - calib = Calibration spectra for any white references found.
         - other keywords are passed directly to correct_image.

        *Returns*:
         - a hyImage to which all sensor-specific corrections have been applied. Note that this will generally not include
           topographic or atmospheric corrections.

        """

        verbose = kwds.get("verbose", True)
        kwds["verbose"] = verbose

        imgs = [str(p) for p in Path(path).rglob("capture/*.hdr")] # all image files [including data]
        dark = [str(p) for p in Path(path).rglob("capture/DARKREF*.hdr")] # dark reference file
        white = [str(p) for p in Path(path).rglob("capture/WHITEREF*.hdr")] # an white reference data (core scanner)
        refl = [str(p) for p in Path(path).rglob("capture/REFL*.hdr")] # any processed reflectance data (SiSu Rock)
        for d in dark:
            del imgs[imgs.index(d)]
        for w in white:
            del imgs[imgs.index(w)]
        for r in refl:
            del imgs[imgs.index(r)]

        if len(imgs) > 1 or len(dark) > 1: assert False, "Error - multiple scenes found in folder. Double check file path..."
        if len(imgs) == 0 or len(dark) == 0: assert False, "Error - no image or dark calibration found in folder. Double check file path... %s" % path

        if verbose: print('\nLoading image %s' % imgs[0])

        #load image
        image = io.loadWithGDAL(imgs[0])
        Fenix.set_dark_ref(dark[0])
        if len(white) > 0: # white reference exists
            Fenix.set_white_ref(white[0])

        #correct
        Fenix.correct_image(image,**kwds)

        #return corrected image
        return image