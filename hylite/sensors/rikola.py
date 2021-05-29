import os
import hylite.io as io
from .sensor import Sensor
import numpy as np
from pathlib import Path
import cv2
from multiprocessing import Pool
from tqdm import tqdm
import piexif
import matplotlib.pyplot as plt

class Rikola(Sensor):
    """
    Implementation of sensor corrections for the Rikola sensor.
    """

    # constants used for boundary tracking when correcting for sensor shift (spatial)
    LEFT_BOUND = 99999
    RIGHT_BOUND = 99998
    TOP_BOUND = 99997
    BOTTOM_BOUND = 99996

    @classmethod
    def correct_image(cls, image, verbose=True, **kwds):
        """
        Apply sensor corrections to an image.

        *Arguments*:
         - image = a hyImage instance of an image captured using this sensor.
         - verbose = true if updates/progress should be printed to the console. Default is True.

        *Keywords*:
         - crop = True if the image should be cropped so that only pixels covered by all bands after alignment are retained. Default is True.
         - align = True if the image bands should be coaligned using sift (due to delayed aquisition times). Default is True.
         - lens = True if the RIKOLA lens correction should be apply to correct for lens distortion. Default is True.
         - match_band = the band to co-register data to. This should be a good quality band (low noise etc.). Default is 30.
         - contrast_thresh = the contrast threshold used for SIFT matching. Default is 0.01.
         - sigma = the sigma to use for SIFT matching. Default is 1.5.
         - edge_thresh = the edge threshold to use for sift matching. Default is 10.
         - eq = equalise before sift? 'True' or 'False'. Default is 'False'.
         - match_dist = maximum matching distance for SIFT. Default is 0.7.
         - min_match = the minimum number of matches to apply band coregistration. Default is 5.
         - warp = the warping method used to align bands. Can be 'affine', 'homography' or 'flow'.
           Default is 'flow' aligns each band using 'affine' and then refines the result using dense optical flow.
           This is slow but corrects for errors associated with perspective and non-flat topography....

        """

        #get keywords
        contrast_thresh = kwds.get("contrast_thresh", 0.01)
        sigma = kwds.get("sigma", 1.5)
        edge_thresh = kwds.get("edge_thresh", 10)
        eq = kwds.get("eq", False)
        match_dist = kwds.get("match_dist", 0.7)
        min_match = kwds.get("min_match", 5)
        warp = kwds.get("warp", 'flow')
        match_band = kwds.get("match_band", 30)
        crop = kwds.get('crop', True)
        align = kwds.get('align', True)
        lens = kwds.get('lens', True)

        # n.b. rikola camera already applies dark correction. Hence, we only need apply a lens correction and align the bands

        ###############################
        # LENS CORRECTION
        ###############################

        #Lens correction for NEW rikola camera
        if image.samples() == 1024: #new rikola

            #delete dodgy bands at start and end of spectral range
            image.band_names = image.header.get_wavelengths() #np.fromstring( image.header['wavelength'], sep="," ) #load band wavelengths
            lower = np.argmin( image.band_names < 512 ) #first valid band
            upper = np.argmin( image.band_names < 914 ) #last valid band
            image.data = image.data[:,:,lower:upper] #delete dodgy bands
            image.band_names = image.band_names[lower:upper]

            if lens:
                #lens correction for sensor 1
                RIKMat = np.array(
                    [[1559.169765826131 - 2.1075939987083148, -0.46558291934094703, 512. - 12.049303428088125],
                     [0, 1559.169765826131, 512. + 10.043851873473248], [0, 0, 1]])
                RIKdist = np.array([[-0.22431431139011579, -0.25788278936781356, -0.00070123282637077225,
                                      0.0023607284552227088, 1.3341497419995652]])
                m, roi = cv2.getOptimalNewCameraMatrix(RIKMat, RIKdist, (image.lines(), image.samples()), 1,
                                                       (image.lines(), image.samples()))


                #lens correction for sensor 2
                RIKMat2 = np.array(
                    [[1562.7573572611377 - 3.2133196257811703, 1.6878264077638325, 512. - 14.06301214860064],
                     [0, 1562.7573572611377, 512. + 11.549906117342049], [0, 0, 1]])
                RIKdist2 = np.array([[-0.23799506683807267, -0.079495552793610855, 0.00085334990816270302,
                                      0.0017596141854012402, 0.78590417508343857]])
                m2, roi2 = cv2.getOptimalNewCameraMatrix(RIKMat2, RIKdist2, (image.lines(), image.samples()), 1,
                                                       (image.lines(), image.samples()))

                #find sensor gap where we change corrections (648 nm band)
                sensgap = np.argmin(image.get_wavelengths() < 648)

                # transpose into img[y][x][b]
                image.data = np.transpose(image.data, (1, 0, 2))

                # apply lens correction
                if verbose: print("Applying lens correction... ", end="", flush="True")
                image.data[:,:,0:sensgap] = cv2.undistort(image.data[:,:,0:sensgap], RIKMat, RIKdist, None, m)
                image.data[:,:,sensgap:] = cv2.undistort(image.data[:, :, sensgap::], RIKMat2, RIKdist2, None, m2)

                # crop
                x, y, w, h = roi2
                image.data = image.data[x:(x + w - 10), y:(y + h - 10)]

                if verbose: print("DONE.", flush="True")

                # transpose back to img[x][y][b]
                image.data = np.transpose(image.data, (1, 0, 2))


        #lens correction for OLD rikola cameara
        elif lens:
            # define camera correction parameters
            # camera matrix = ([[fx,skew,cx],[0,fy,cy],[0,0,1]])
            # distortion coefficients = ([[k1,k2,kp1,p2,k3]])
            # n.b. depending on the aquisition mode the rikola reads either the whole sensor (1010 lines) or half the sensor (658 lines). Hence we chose the camera matrix accordingly
            if image.lines() == 1010: #old rikola, full frame
                RIKMat = np.array([[1580, -0.37740987561726941, 532.14269389794072],
                                   [0, 1586.5023476977308, 552.87899983359080],
                                   [0, 0, 1]])

                RIKdist = np.array([[-0.34016504377397944, 0.15595251253428483, 0.00032919179911211665,
                                     0.00016579344155373088, 0.051315602289989909]])

            elif image.lines() == 648: #old rikola, half frame
                RIKMat = np.array([[1580.9821817891338, -0.053468464819987738, 537.09531859948970],
                                   [0, 1580.4094746112266, 369.76442407125506],
                                   [0, 0, 1]])
                RIKdist = np.array([[-0.31408677145500508, -0.26653256083139154, 0.00028155583639827883,
                                     0.00025705469073531660, 2.4100464839836362]])
            else:
                assert False, "Error - invalid number of lines (%d) for RIKOLA image. Should be either 1010 or 648" % (
                    image.lines())

            # create calibration map
            m, roi = cv2.getOptimalNewCameraMatrix(RIKMat, RIKdist, (image.lines(), image.samples()), 1,
                                                   (image.lines(), image.samples()))

            # transpose into img[y][x][b]
            image.data = np.transpose(image.data, (1, 0, 2))

            # loop through bands and apply correction
            if verbose: print("Applying lens correction... ", end="", flush="True")

            # apply lens correction
            image.data = cv2.undistort(image.data, RIKMat, RIKdist, None, m)

            # crop
            x, y, w, h = roi
            image.data = image.data[x:(x + w - 10), y:(y + h - 10)]

            if verbose: print("DONE.", flush="True")

            # transpose back to img[x][y][b]
            image.data = np.transpose(image.data, (1, 0, 2))

        ###############################
        # COREGISTER BANDS
        ###############################

        # flag boundary pixels in each band so we can crop it after warping
        image.edge_mask = np.zeros(image.data[:, :, 0:4].shape, dtype=np.float32)
        image.edge_mask[0:3, :, 0] = 1.0
        image.edge_mask[-4:-1, :, 1] = 1.0
        image.edge_mask[:, 0:3, 2] = 1.0
        image.edge_mask[:, -4:-1, 3] = 1.0
        edge_accum = np.zeros_like(image.edge_mask)
        if align:
            # identify SIFT features for each band
            if verbose: print("Identifying SIFT features... ", end="", flush="True")
            features = [image.get_keypoints(b) for b in range(image.band_count())]

            if verbose: print("DONE.", flush="True")

            if verbose: print("Stacking bands... ", end="", flush="True")

            # loop through bands and find matches
            matches = []
            for b in range(0, image.band_count()):
                if b == match_band:
                    matches.append(None)
                    continue  # skip reference band as it doesn't need to be moved...
                if verbose: print("Matching bands... %d " % int((100 * b) / image.band_count()) + "% \r", end="", flush="True")

                # can we match this band with the reference band?
                src_pts, dst_pts = io.HyImage.match_keypoints(features[b][0], features[match_band][0],
                                                         features[b][1], features[match_band][1],
                                                         method='SIFT', dist=match_dist)
                matches.append((src_pts, dst_pts))

            if verbose: print("Matching bands... DONE.                          ", flush="True")

            # apply transformation to each band
            for b in range(0, image.band_count()):
                if b == match_band: continue  # skip reference band as it doesn't need to be moved...
                if verbose: print("Stacking bands... %d " % int((100 * b) / image.band_count()) + "% \r", end="",flush="True")

                # get match points
                src_pts, dst_pts = matches[b]
                if src_pts is None:  # couldn't match with reference band, try matching with one of the already aligned bands
                    print("Warning - could not directly band %d. Attempting indirect alignment... " % b, end="",
                          flush="True")
                    i = 1
                    while src_pts is None:
                        if b - i < 0:  # run out of bands
                            print("Failed to align band" % b)
                            break

                        # calculate features in aligned band
                        feat2 = image.get_keypoints(b-i, eq=eq, contrastThreshold=contrast_thresh,
                                            edgeThreshold=edge_thresh, sigma=sigma)

                        # try matching
                        src_pts, dst_pts = io.HyImage.match_keypoints(features[b][0], feat2[0],
                                                                features[b][1], feat2[1],
                                                                dist=match_dist, method='sift')

                    print("Aligned to band %d" % (b - i))

                # do alignment
                if not src_pts is None:
                    if ('affine' in warp.lower()) or ('flow' in warp.lower()):
                        H, status = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                        dst_mask = dst_pts[:, 0, :] * status
                        src_mask = src_pts[:, 0, :] * status
                        dst_mask = dst_mask[dst_mask.all(1)]
                        src_mask = src_mask[src_mask.all(1)]
                        dst_mask = np.expand_dims(dst_mask, axis=1)
                        src_mask = np.expand_dims(src_mask, axis=1)
                        #M = cv2.estimateRigidTransform(src_mask, dst_mask, False)
                        M = cv2.estimateAffinePartial2D(src_mask, dst_mask)[0]

                        # apply to image
                        image.data[:, :, b] = cv2.warpAffine(image.data[:, :, b], M,
                                                             (image.data.shape[1], image.data.shape[0]))

                        # apply to edge mask
                        for e in range(0, 4):
                            edge_accum[:, :, e] += cv2.warpAffine(image.edge_mask[:, :, e], M,
                                                                  (image.edge_mask.shape[1], image.edge_mask.shape[0]))

                    elif ('homography' in warp.lower()):
                        H, status = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)

                        # apply to image
                        image.data[:, :, b] = cv2.warpPerspective(image.data[:, :, b], H,
                                                                  (image.data.shape[1], image.data.shape[0]))

                        # apply to edge mask
                        for e in range(0, 4):
                            edge_accum[:, :, e] += cv2.warpPerspective(image.edge_mask[:, :, e], H,
                                                                       (image.edge_mask.shape[1],
                                                                        image.edge_mask.shape[0]))
                    else:
                        assert False, "Error - unknown transformation type. Should be 'flow' or 'homography' or 'affine'."

            # update edge mask
            image.edge_mask = edge_accum

            # crop image to area of complete overlap
            xmin = np.min(np.where(np.sum(image.edge_mask[:, :, 0], axis=1) == 0.0))
            xmax = np.max(np.where(np.sum(image.edge_mask[:, :, 1], axis=1) == 0.0))
            ymin = np.min(np.where(np.sum(image.edge_mask[:, :, 2], axis=0) == 0.0))
            ymax = np.max(np.where(np.sum(image.edge_mask[:, :, 3], axis=0) == 0.0))
            if crop:
                image.data = image.data[xmin:xmax, ymin:ymax, :]
            else:
                image.data[:xmin, :, :] = 0.
                image.data[xmax:, :, :] = 0.
                image.data[:, :ymin, :] = 0.
                image.data[:, ymax:, :] = 0.

            # apply flow transform to aligned bands
            if 'flow' in warp.lower():

                # reset edge mask
                image.edge_mask = np.zeros(image.data[:, :, 0:4].shape, dtype=np.float32)
                image.edge_mask[0:3, :, 0] = 1.0
                image.edge_mask[-4:-1, :, 1] = 1.0
                image.edge_mask[:, 0:3, 2] = 1.0
                image.edge_mask[:, -4:-1, 3] = 1.0
                edge_accum = np.zeros_like(image.edge_mask)
                alg = cv2.optflow.createOptFlow_DeepFlow()
                X, Y = np.meshgrid(range(image.data.shape[1]), range(image.data.shape[0]))
                map = np.dstack([X, Y]).astype(np.float32)
                bnd1 = np.uint8(255 * (image.data[:, :, 0] - np.nanmin(image.data[:, :, 0])) /
                         (np.nanmax(image.data[:, :, 0]) - np.nanmin(image.data[:, :, 0])))
                for b in range(1, image.band_count()):
                    if verbose: print("Warping bands... %d " % int((100 * b) / image.band_count()) + "% \r", end="",
                                      flush="True")
                    bnd2 = np.uint8(255 * (image.data[:, :, b] - np.nanmin(image.data[:, :, b])) /
                         (np.nanmax(image.data[:, :, b]) - np.nanmin(image.data[:, :, b])))
                    bnd1[bnd2 == 0.] = 0.
                    bnd2[bnd1 == 0.] = 0.
                    flow = alg.calc(bnd1, bnd2, None)
                    map[:, :, 0] += flow[:, :, 0]
                    map[:, :, 1] += flow[:, :, 1]
                    image.data[:, :, b] = cv2.remap(image.data[:, :, b], map, None, cv2.INTER_LINEAR)
                    image.data[:, :, b][bnd2 == 0.] = 0.
                    for e in range(0, 4):
                        edge_accum[:, :, e] += cv2.remap(image.edge_mask[:, :, e], map, None, cv2.INTER_LINEAR)
                    bnd1 = bnd2

                image.edge_mask = edge_accum

                # crop again

                xmin = np.min(np.where(np.sum(image.edge_mask[:, :, 0], axis=1) == 0.0))
                xmax = np.max(np.where(np.sum(image.edge_mask[:, :, 1], axis=1) == 0.0))
                ymin = np.min(np.where(np.sum(image.edge_mask[:, :, 2], axis=0) == 0.0))
                ymax = np.max(np.where(np.sum(image.edge_mask[:, :, 3], axis=0) == 0.0))
                if crop:
                    image.data = image.data[xmin:xmax, ymin:ymax, :]
                else:
                    image.data[:xmin, :, :] = 0.
                    image.data[xmax:, :, :] = 0.
                    image.data[:, :ymin, :] = 0.
                    image.data[:, ymax:, :] = 0.
            if verbose: print("Warping bands...  DONE.                         ", flush="True")

    @classmethod
    def correct_folder(cls, path, **kwds):
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
         - multi = True if multiple threads will be spawned to processes images in parallel. Default is True.
         - nthreads = the number of worker threads to spawn if multithreaded is true. Default is the number of CPU cores.

        """

        assert os.path.isdir(path), "Error - %s is not a valid directory path" % path
        assert os.path.exists(path), "Error - could not find %s" % path

        #get keywords
        multi = kwds.get('multi', True)
        nthread = kwds.get('nthreads', os.cpu_count())
        overwrite = kwds.get('overwrite', False)

        if nthread is None:
            print("Warning - could not identify CPU count. Multithreading disabled.")
            multi = False

        #find images to process
        images = list(Path(path).rglob("*Calib*[0-9][0-9][0-9][0-9][0-9].hdr")) #old rikola files
        images += list(Path(path).rglob("*Sequence*[0-9][0-9][0-9][0-9][0-9].hdr")) #new rikola files

        if not overwrite:  # ignore already corrected files
            images = [x for x in images if list(Path(path).rglob(x.name[:-4] + "_CORRECTED.hdr")) == []]

        corrected = []
        print("Processing %d images with the following settings: %s" % (len(images),kwds))
        if multi: #run in multithreaded
            p = Pool(processes = nthread) #setup pool object with n threads
            corrected = list(tqdm( p.imap_unordered(Rikola._cmp, [(i, kwds) for i in images]), total=len(images),leave=False)) #distribute tasks
        else: #not multithreaded
            for pth in tqdm(images,leave=False):
                # noinspection PyCallByClass
                corrected.append(Rikola._cmp( (pth, kwds) ) )#

        return corrected

    # define worker function for correct_folder( ... ) multithreading
    @classmethod
    def _cmp(cls, args):
        pth = str(args[0])
        kwds = args[1]
        image = io.loadWithGDAL(pth)  # load image
        Rikola.correct_image(image, False, **kwds)  # correct image
        io.saveWithGDAL(os.path.splitext(pth)[0] + "_CORRECTED.hdr", image)  # save corrected image
        return os.path.splitext(pth)[0] + "_CORRECTED.hdr"  # return corrected path

    @classmethod
    def GPS_JPG(cls, MAIN):
        """
        Creates geotagged RGB JPGs for a folder of calibrated Rikola images and stores them in a subfolder "RGB/"

        *Arguments*:
         - MAIN = path to the folder containing the image data (ending with "CORRECTED.dat")
                  as well as the acquisition-specific "TASKFILE.TXT"

        """

        try:
            import osgeo.gdal as gdal # todo - remove GDAL dependency for this function (replace with Pillow?)
        except:
            assert False, "Error - the GPS_JPG function requires GDAL to be installed."

        # create new folder for file storage
        if not os.path.exists(MAIN + 'RGB/'):
            os.makedirs(MAIN + 'RGB/')

        # read taskfile.txt containing GPS information
        with open(MAIN + 'TASKFILE.TXT') as f:
            task = f.readlines()
        task = np.asarray(task[60:])
        taskfile = np.asarray([task[i].split(",") for i in range(len(task))])

        # collect filelist for processing
        filelist = [root + '/' + file for root, _, files in os.walk(MAIN) for file in files if
                    "_CORRECTED.dat" in file and "xml" not in file]

        # processing for each file in filelist
        for file in filelist:

            # read 3 bands (RGB) as PIL image
            raster = gdal.Open(file)
            band1 = raster.GetRasterBand(23).ReadAsArray()
            band2 = raster.GetRasterBand(7).ReadAsArray()
            band3 = raster.GetRasterBand(1).ReadAsArray()
            arr1 = band1.astype(np.float)
            maxim = np.nanmax(band1)
            band1_255 = np.uint8(arr1 / maxim * 255)
            arr1 = band2.astype(np.float)
            maxim = np.nanmax(arr1)
            band2_255 = np.uint8(arr1 / maxim * 255)
            arr1 = band3.astype(np.float)
            maxim = np.nanmax(arr1)
            band3_255 = np.uint8(arr1 / maxim * 255)

            # find and extract file-relevant info from taskfile
            rawfilename = os.path.split(file)[1].replace('_CORRECTED.dat', '.DAT')[6:]
            filenumber = np.where(taskfile[:, 0] == rawfilename)
            test = taskfile[filenumber, 16].tolist()
            altitude = float('[]'.join(test[0]))
            longi = taskfile[filenumber, 11].tolist()
            long = float('[]'.join(longi[0])) / 100
            lati = taskfile[filenumber, 9].tolist()
            lat = float('[]'.join(lati[0])) / 100
            WEi = taskfile[filenumber, 12].tolist()
            WE = ('[]'.join(WEi[0]))
            NSi = taskfile[filenumber, 10].tolist()
            NS = ('[]'.join(NSi[0]))
            time = taskfile[1, 6]
            date = str("%02d" % (int((time.split(' ')[1]).split('.')[2]),) + ':' + "%02d" % (
            int((time.split(' ')[1]).split('.')[1]),) + ':' + "%02d" % (int((time.split(' ')[1]).split('.')[0]),))
            out = MAIN + 'RGB/' + os.path.split(file)[-1][:-4] + '.jpg'

            # save jpg with exif using piexif
            exif_dict = {'0th': {},
                         '1st': {},
                         'Exif': {},
                         'GPS': {
                             1: NS,  # latituderef
                             2: ((int(lat), 1), (int(100 * (lat - int(lat))), 1),
                                 (int((100 * (lat - int(lat)) - int(100 * (lat - int(lat)))) * 6000), 100)),  # Latitude
                             3: WE,  # Longituteref
                             4: ((int(long), 1), (int(100 * (long - int(long))), 1),
                                 (int((100 * (long - int(long)) - int(100 * (long - int(long)))) * 6000), 100)),
                             # longitude
                             6: (abs(int(altitude * 100)), 100),  # Altitude
                             7: (
                             (int((time.split(' ')[2]).split(':')[0]), 1), (int((time.split(' ')[2]).split(':')[1]), 1),
                             (int((np.float((time.split(' ')[2]).split(':')[2])) * 100), 10)),
                             29: date},  # timestamp
                         'Interop': {},
                         'thumbnail': None}
            exif_bytes = piexif.dump(exif_dict)
            plt.imsave(out,np.dstack((band1_255, band2_255, band3_255)))
            piexif.insert(exif_bytes, out)

            """
            # save image and write EXIF information with exiftools instead
            lat=(int(lat))+(int(100*(lat-int(lat))))/60.+(100*(lat-int(lat))-int(100*(lat-int(lat))))*6000/360000.0
            long=(int(long))+(int(100*(long-int(long))))/60.+(100*(long-int(long))-int(100*(long-int(long))))*6000/360000.0
            im.save(out)
            pic = out.encode("utf-8")
            with exiftool.ExifTool() as et:
                et.execute(("-GPSLatitudeRef=" + NS).encode("utf-8"), b"-overwrite_original", pic)
                et.execute(("-GPSLongitudeRef=" + WE).encode("utf-8"),b"-overwrite_original", pic)
                et.execute(("-GPSLatitude=" + str(lat)).encode("utf-8"), b"-overwrite_original", pic)
                et.execute(("-GPSLongitude=" + str(long)).encode("utf-8"), b"-overwrite_original", pic)
                et.execute(("-GPSAltitude=" + str(altitude)).encode("utf-8"), b"-overwrite_original", pic)
                et.execute(("-GPSTimeStamp=" + str(time)).encode("utf-8"), b"-overwrite_original", pic)
                et.execute(("-GPSDateStamp=" + str(date)).encode("utf-8"), b"-overwrite_original", pic)
            """

class Rikola_RSC1( Rikola ): # old Rikola

    """
    Sensor specific details for the Rikola camera
    """
    @classmethod
    def name(cls):
        """
        Returns this sensors name
        """
        return "Rikola RSC-1"

    @classmethod
    def fov(cls):
        """
        Return the (vertical) sensor field of view .
        """
        return 36.5

    @classmethod
    def ypixels(cls):
        """
        Return the number of pixels in the y-dimension.
        """
        return 1010

    @classmethod
    def xpixels(cls):
        """
        Return the number of pixels in the x-dimension (==1 for line scanners).
        """
        return 1010

    @classmethod
    def pitch(cls):
        """
        Return the pitch (mm) of the each pixel in the y-dimension (though most pixels are square).
        """
        return 0.0055

class Rikola_HSC2( Rikola ): # new Rikola

    """
    Sensor specific details for the Rikola camera
    """

    @classmethod
    def name(cls):
        """
        Returns this sensors name
        """
        return "Rikola HSC-2"

    @classmethod
    def fov(cls):
        """
        Return the (vertical) sensor field of view .
        """
        return 36.8

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
        return 1024

    @classmethod
    def pitch(cls):
        """
        Return the pitch of the each pixel in the y-dimension (though most pixels are square).
        """
        return 0.0055