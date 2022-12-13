"""
Manage metadata and other header file information.
"""

import copy
import numpy as np

class HyHeader( dict ):
    """
    A generic class for encapsulating header files (.hdr) data. We use these for point clouds and spectral
    libraries as well as images.
    """
    def __init__(self):
        super().__init__()

        #default values
        self['file type'] = 'ENVI Standard'

    def has_band_names(self):
        """
        Return true if band names are defined
        """
        return 'band names' in self

    def has_wavelengths(self):
        """
        Return true if wavelengths are defined.
        """
        return ('wavelength' in self)

    def has_fwhm(self):
        """
        Return true if FWHM are defined.
        """
        return 'fwhm' in self

    def has_bbl(self):
        """
        Returns true if a bad band list has been defined.
        """
        return 'bbl' in self

    def band_count(self):
        """
        Return the number of bands specified in this header file.
        """
        if self.has_wavelengths():
            return len(self.get_wavelengths())
        elif self.has_band_names():
            return len(self.get_band_names())
        elif self.has_fwhm():
            return len(self.get_fwhm())
        elif 'bands' in self:
            return int(self['bands'])
        return 0 # image has no bands

    def get_band_names(self):
        """
        Get list of band names from header file.
        """
        assert 'band names' in self, "Error - header file has no band names."
        return self.get_list('band names', str)

    def get_wavelengths(self):
        """
        Get list of band wavelengths from header file.
        """
        if 'wavelength' in self:
            return self['wavelength'].copy()
        else:
            assert False, "Error - header file has no wavelength information."


    def get_bbl(self):
        """
        Get the list of bad bands from header file. This will return an
        array scored True if a band is good and False if it is bad.
        """
        assert 'bbl' in self, "Error - header file has no bad band information."
        return np.array(self['bbl']) == 1 # internally we store as 0 = bad band and 1 = good band.

    def get_fwhm(self):
        """
        Get list of band widths from header file.
        """
        assert 'fwhm' in self, "Error - header file has no FWHM information."
        return self['fwhm'].copy()

    def set_band_names(self, band_names):
        """
        Set list of band names from header file.

        Args:
            band_names (list): a list of band names.
        """
        if band_names is None:
            if 'band names' in self:
                del self['band names']
        else:
            assert isinstance(band_names, list) or isinstance(band_names, np.ndarray), "Error - band_names must be a list or numpy array."
            self['band names'] = list(band_names)

    def set_wavelengths(self, wavelengths):
        """
        Set list of band wavelengths from header file.

        Args:
            wavelengths (list, ndarray): a list or numpy array of wavelengths.
        """
        if wavelengths is None:
            if 'wavelength' in self:
                del self['wavelength']
        else:
            assert isinstance(wavelengths, list) or isinstance(wavelengths, np.ndarray) or isinstance(wavelengths, tuple), "Error - wavelengths must be list, tuple or numpy array."
            self['wavelength'] = np.array(wavelengths).astype(np.float)

    def set_fwhm(self, fwhm):
        """
        Set list of band wavelengths from header file.

        Args:
            fwhm (list, ndarray): a list or numpy array of band widths.
        """
        if fwhm is None:
            if 'fwhm' in self:
                del self['fwhm']
        else:
            assert isinstance(fwhm, list) or isinstance(fwhm, np.ndarray), "Error - fwhm must be list or numpy array."
            self['fwhm'] = np.array(fwhm).astype(np.float)

    def set_bbl(self, bbl):
        """
        Set list of bad bands.

        Args:
            bbl (list, ndarray): a list or numpy array scored True for good bads and False for bad ones, or None to remove bbl.
        """
        if bbl is None:
            if 'bbl' in self:
                del self['bbl']
        else:
            assert isinstance(bbl, list) or isinstance(bbl, np.ndarray), "Error - bbl must be list or numpy array."
            self['bbl'] = np.array(bbl).astype(np.uint8) # internally we store as 0 (bad band) or 1 (good band).

    def drop_bands(self, mask):
        """
        Remove all header data (e.g. band names, wavelengths) associated with the specified bands. Used when removing or
        deleting bands from an image.

        Args:
            mask (list, ndarray): either a list of band indices to remove or a boolean numpy array with True for bands that should remove.
        """
        # how many bands are there?
        nbands = self.band_count()

        # create boolean mask
        mask = np.array(mask)
        if mask.dtype == np.bool:
            keep = np.logical_not(mask)  # flip so we have array with True for bands to keep
        elif issubclass(mask.dtype, np.integer):
            keep = np.full(nbands, True)
            keep[mask] = False  # drop bands
        else:
            "Error - unrecognised data type %s." % mask.dtype

        # check we have a value per band
        #assert len(keep) == nbands, "Error - data has %d values but mask is length %d" % (nbands, len(keep))
        # N.B. this assert has been removed because it is triggered when a header contains no wavelength or band names
        # (and so the header file cannot know how many bands to expect in the image)

        skip = ['default bands', 'temperature', 'camera',
                'class lookup', 'class names', 'coordinate system string', 'description',
                'geo points', 'map info', 'projection info', 'spectra names']
        for key, value in self.items():
            if key in skip:  # these keys are lists but should never be trimmed (as they don't relate to bands)
                continue
            if isinstance(value, str):
                if "," in value:
                    value = np.array(value.split(","))  # split on commas
            if isinstance(value, list):
                if len(value) == len(keep):
                    self[key] = list(np.array(value)[keep])  # drop bands
            if isinstance(value, np.ndarray):
                if len(value) == len(keep):
                    self[key] = value[keep]  # drop bands

        #update band count
        self['bands'] = str(np.sum(keep))

    def drop_all_bands(self):
        """
        Remove all attributes in the header file that are specified per band. Useful if a dataset is copied and the bands
        overwritten.
        """

        # how many bands are there?
        nbands = self.band_count()
        if nbands == 0:
            return # no bands

        skip = ['default bands', 'temperature', 'camera',
                'class lookup', 'class names', 'coordinate system string', 'description',
                'geo points', 'map info', 'projection info', 'spectra names']
        to_del = []
        for key, value in self.items():
            if key in skip:  # these keys are lists but should never be trimmed (as they don't relate to bands)
                continue
            if isinstance(value, str):
                if "," in value:
                    value = np.array(value.split(","))  # split on commas
            if isinstance(value, list):
                if len(value) == nbands:
                    to_del.append(key)
            if isinstance(value, np.ndarray):
                if len(value) == nbands:
                    to_del.append(key)

        # do delete
        for k in to_del:
            del self[k]

        #update band count
        self['bands'] = str(0)

    def copy(self):
        """
        Make a deep copy of this header class.

        Returns:
            a new header instance.
        """

        return copy.deepcopy(self)

    def print(self):
        """
        Print metadata to console
        """

        for key, value in self.items():
            s = str(value).strip()
            print("%s = { %s } " % (key, s))

    ############################################################
    ## Functions related to other data we store in header files
    ############################################################
    def get_list(self, name, dtype=np.float):
        """
        Get generic list data from the header as a numpy array.

        Args:
            name (str): the header key used to access the list.
            dtype (str): the numpy data type that should be returned. Default is float.

        Returns:
            a numpy array containing the header data, or None if the key does not exist.
        """

        if not name in self:
            return None
        out = self[name]
        if isinstance(out, str):
            out = np.array( [v.strip() for v in out.split(',')] ) # split string into array
        try:
            return out.astype(dtype) # convert to the correct type and return
        except:
            return out # return as e.g. string - is better than failing.

    def set_camera(self, camera, n=0):
        """
        Store information on the position and orientation of this image to the image header.

        Args:
            camera (hylite.project.Camera): the camera instance to store.
            n (int): the camera id (integer), if multiple cameras are stored (e.g. for hyperclouds). Default is 0.
        """

        # pose = cx cy cz rx ry rz projection
        self["camera %d pose" % n] = "%f %f %f %f %f %f" % (camera.pos[0], camera.pos[1], camera.pos[2],
                                                     camera.ori[0], camera.ori[1], camera.ori[2])

        # internals = x pixels, y pixels, vertical field of view, angular step (for panoramic cameras)
        if camera.is_panoramic():
            self["camera %d internals" % n] = "%s %d %d %f %f" % (
            camera.proj, camera.dims[0], camera.dims[1], camera.fov, camera.step)
        else:
            self["camera %d internals" % n] = "%s %d %d %f" % (camera.proj, camera.dims[0], camera.dims[1], camera.fov)

    def get_camera(self, n = 0):
        """
        Extract a camera object containing the internal and external camera properties (or None if they don't exist).

        Args:
            n (int): the camera id (integer), if multiple cameras are stored (e.g. for hyperclouds). Default is 0.
        """
        from hylite.project.camera import Camera #n.b. do import here to avoid circular dependencies

        # check for single camera (for backward compatability)
        if 'camera pose' in self and 'camera internals' in self:
            self['camera %d pose'%n] = self['camera pose']
            self['camera %d internals'%n] = self['camera internals']
            del self['camera pose']
            del self['camera internals']

        # no camera info? bail!
        if not ('camera %d pose'%n in self and 'camera %d internals'%n in self):
            return None

        pose = self['camera %d pose' % n].split(' ')
        internal = self["camera %d internals" % n].split(' ')
        pos = np.array([float(pose[0]), float(pose[1]), float(pose[2])])
        ori = np.array([float(pose[3]), float(pose[4]), float(pose[5])])
        proj = internal[0]
        dims = (int(internal[1]), int(internal[2]))
        fov = float(internal[3])
        if 'pano' in proj:
            step = float(internal[4])
            return Camera(pos=pos, ori=ori, proj=proj, fov=fov, dims=dims, step=step)
        elif 'persp' in proj:
            return Camera(pos=pos, ori=ori, proj=proj, fov=fov, dims=dims)
        else:
            assert False, "Error - '%s' is an unknown camera projection." % proj

    def has_camera(self):
        """
        Return true if this header file has defined camera pose information.
        """
        return not (self.get_camera() is None)

    def get_panel_names(self):
        """
        Get the name of all panels defined in this header file.
        """
        names = []
        for k in self.keys():
            if "target" in k: # this key relates to a target
                name = k.split(' ')[1]
                if not name in names:
                    names.append(name)
        return names

    def add_panel(self, panel, name=None):
        """
        Add the specified panel to  the header file

        Args:
            panel (hylite.correct.Panel): the panel reflectance and measured radiance spectra.
            name (str): the name of this panel. If None (default) then the name of the panel material will be used (e.g., 'R90').
        """
        assert (panel.get_wavelengths() == self.get_wavelengths()).all(), "Error - target wavelengths and header wavelengths do not match."
        if name is None:
            name = panel.material.get_name()
        name = name.lower()  # use lower case
        self['target %s reflectance' % name] = panel.get_reflectance()
        self['target %s radiance' % name] = panel.get_mean_radiance()
        if panel.normal is not None: # store normal vector
            self['target %s normal' % name] = panel.normal
        if panel.skyview is not None:
            self['target %s skyview' % name] = panel.skyview
        if panel.alpha is not None:
            self['target %s alpha' % name] = panel.alpha

    def get_panel(self, name):
        """
        Get the specified panel from this header

        Args:
            name (str): the name of the target to return.

        Returns:
             a Panel instance containing the reflectance and radiance data, or None if it does not exist.
        """

        from hylite.reference.spectra import Target # n.b. do import here to avoid circular dependencies
        from hylite.correct.panel import Panel

        name = name.lower() # use lower case
        if not ('target %s reflectance' % name in self and 'target %s radiance' % name in self):
            return None # no target found

        reflectance = self['target %s reflectance' % name]
        radiance = self['target %s radiance' % name]

        if isinstance(reflectance,str):
            reflectance = np.fromstring(reflectance, sep=",")
            radiance = np.fromstring(radiance, sep=",")

        reflectance = reflectance.astype(np.float32) # check type
        radiance = radiance.astype(np.float32)

        # create dummy Target instance
        material = Target(self.get_wavelengths(), reflectance, name=name)

        # create Panel instance and return
        P = Panel( material, radiance, wavelengths=self.get_wavelengths() )

        # get normal vector (if defined)
        if ('target %s normal' % name) in self:
            normal = self['target %s normal' % name]
            if isinstance(normal, str):  # parse string
                P.set_normal( np.fromstring(self['target %s normal' % name], sep=",") )
            elif isinstance(normal, list):
                P.set_normal( np.array(normal) )
            else:
                assert isinstance(normal, np.ndarray), "Error - %s is an invalid normal." % normal

        # get illu factors if defined
        if ('target %s alpha' % name) in self:
            P.alpha = float(self['target %s alpha' % name])
        if ('target %s skyview' % name) in self:
            P.skyview = float(self['target %s skyview' % name])
        return P

    def remove_panel(self, name = None):
        """
        Removes the specified calibration panel from this header.

        Args:
            name (str): the name of the panel to remove. If None (default) then all panels will be removed.
        """
        if name is not None:
            name = name.lower()  # use lower case

        # find keys to remove
        to_del = []
        for k in self.keys():
            if "target" in k: # this key relates to a target
                if not name is None and not name in k:
                    continue # skip this one
                to_del.append(k)

        # remove them
        for k in to_del:
            self.pop(k, None)

    def get_data_ignore_value(self ):
        if 'data ignore value' in self:
            return float(self['data ignore value'])
        else:
            return np.nan

    def set_data_ignore_value(self, val):
        assert isinstance(val, float) or isinstance(val, int), "Error - %s is an invalid data type for data ignore values" % type(val)
        self['data ignore value'] = str(val)

    def get_class_names(self):
        if not 'class names' in self:
            return None
        if isinstance(self['class names'], list):
            return self['class names']
        elif isinstance(self['class names'],str):
            return [s.strip() for s in self['class names'].split(',')]

    def get_sample_points(self, name):
        """
        Get points that are associated with the specified sample name.
        """
        key = 'sample %s' % name
        if not key in self:
            key = 'sample %s' % name.lower()
            if not key in self:
                return []

        if isinstance(self[key], list):
            return self[key]
        elif isinstance(self[key], str):
            if ')' in self[key]: # image coordinates - parse tuple
                # parse string
                pnts = []
                splt = self[key].split('),')
                for tpl in splt:
                    tpl = tpl.replace('(', '')
                    tpl = tpl.replace(')', '')
                    tpl = tpl.split(',')
                    pnts.append((int(tpl[0]), int(tpl[1])))
            else:
                splt = self[key].split(',') # point cloud - simply treat as IDs
                pnts = (int(i) for i in splt)
            return pnts

    def set_sample_points(self, name, points):
        """
        Set points that are associated with the specified sample name.

        Args:
            name (str): the name of the sample points to set
            points (list): a list of points: [(x1,y1), (x2,y2) ... ] or [id1,id2,id3, ...]
        """
        key = 'sample %s' % name
        self[key] = points
