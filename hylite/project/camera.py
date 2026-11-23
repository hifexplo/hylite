"""
A simple data structure for storing camera pose information.
"""

import numpy as np

from hylite._deps import require

class Camera( object ):
    """
    A utility class for storing camera properties (position, dimensions, fov etc.).
    """

    def __init__(self, pos, ori, proj, fov, dims, step = None, az_lo=None, el_hi=None):
        """
        Args:
            pos (ndarray): the camera position vector.
            ori (ndarray): the camera orientation vector (pitch, roll, yaw) in degrees.
            proj (str): the project type: 'persp', 'pano', or 'sphere'.
            fov (float): the camera field of view (degrees). For a sphere this is the vertical
                angular height of the image (ydim * step) and is not used by the projection.
            dims (tuple): the pixel dimensions of the frame.
            step (float): the angular step of pixels. Required for panoramic and spherical cameras.
            az_lo (float): azimuth (degrees) of the left edge of a spherical image. Positive is right.
            el_hi (float): elevation (degrees) of the top edge of a spherical image. Positive is up.
        """

        assert isinstance(pos, np.ndarray) and len(pos)==3, \
                                            "Error - camera_position must be a numpy array with length 3. "
        self.pos = pos.copy()

        assert isinstance(ori, np.ndarray) and len(ori) == 3, \
                                            "Error - camera_ori must be a numpy array with length 3. "
        self.ori = ori.copy()

        assert 'persp' in proj.lower() or 'pano' in proj.lower() or 'sphere' in proj.lower(), \
                           "Error - camera type should be persp, pano or sphere."
        self.proj = proj

        assert len(dims) >= 2, "Error, dims must be a tuple or list of length 2."
        self.dims = dims

        assert np.isfinite(fov), "Error, fov must be numeric."
        self.fov = fov
        self.step = step
        self.az_lo = None if az_lo is None else float(az_lo)
        self.el_hi = None if el_hi is None else float(el_hi)
        if self.is_spherical():
            assert self.step is not None and self.step > 0, "Error - sphere cameras require a positive pixel step."
            assert self.az_lo is not None and self.el_hi is not None, \
                "Error - sphere cameras require az_lo and el_hi."

    @classmethod
    def from_equirect(cls, image, pos, ori):
        """
        Build a spherical camera from an equirectangular mosaic.

        The header stores the crop as ``panorama azimuth range deg`` (left and right
        edges), ``panorama elevation range deg`` (bottom and top edges), and
        ``panorama deg per pixel``.

        Args:
            image: a HyImage whose header contains those three keys.
            pos: camera position (x, y, z).
            ori: camera orientation (pitch, roll, yaw) in degrees.

        Returns:
            A Camera with ``proj='sphere'``. ``pos`` and ``ori`` are the station pose;
            the header only fixes the angular crop.
        """
        hdr = image.header
        az_key = 'panorama azimuth range deg'
        el_key = 'panorama elevation range deg'
        step_key = 'panorama deg per pixel'
        assert az_key in hdr and el_key in hdr and step_key in hdr, \
            "Error - image header has no equirectangular panorama bounds."
        az_lo = float(str(hdr[az_key]).split()[0])
        el_hi = float(str(hdr[el_key]).split()[1])
        step = float(hdr[step_key])
        dims = (int(image.xdim()), int(image.ydim()))
        fov = dims[1] * step
        return cls(np.array(pos, dtype=float), np.array(ori, dtype=float), 'sphere', fov, dims,
                   step=step, az_lo=az_lo, el_hi=el_hi)

    def clone(self):
        """
        Create a deep copy of a camera instance.
        """
        return Camera( self.pos.copy(), self.ori.copy(), self.proj, self.fov, self.dims, self.step,
                       az_lo=self.az_lo, el_hi=self.el_hi )

    def set_transform(self, camera_pos, camera_ori):
        """
        Sets the position and rotation of this camera.

        Args:
            camera_pos (ndarray): the new position.
            camera_ori (ndarray): the new orientation.
        """
        assert isinstance(camera_pos, np.ndarray) and len(camera_pos) == 3, \
                                        "Error - camera_position must be a numpy array with length 3. "
        self.pos = camera_pos.copy()

        assert isinstance(camera_ori, np.ndarray) and len(camera_ori) == 3, \
                                        "Error - camera_ori must be a numpy array with length 3. "
        self.ori = camera_ori.copy()

    def xdim(self):
        """
        Return camera x-dimension (pixels).
        """
        return self.dims[0]

    def ydim(self):
        """
        Return camera y-dimension (pixels).
        """
        return self.dims[1]

    def is_perspective(self):
        """
        Returns true if this camera uses a perspective project.
        """
        return 'persp' in self.proj.lower()

    def is_panoramic(self):
        """
        Returns true if this camera uses a panoramic project.
        """
        return 'pano' in self.proj.lower()

    def is_spherical(self):
        """
        Returns true if this camera uses an equirectangular spherical projection.
        """
        return 'sphere' in self.proj.lower()

    def get_rotation_matrix(self):
        """
        Return the rotation matrix of this camera (based on its Euler angles).
        """
        spatial = require("scipy").spatial
        return spatial.transform.Rotation.from_euler('XYZ', -self.ori, degrees=True).as_matrix()
