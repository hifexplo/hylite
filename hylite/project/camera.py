import numpy as np
from scipy import spatial

class Camera( object ):
    """
    A utility class for storing camera properties (position, dimensions, fov etc.).
    """

    def __init__(self, pos, ori, proj, fov, dims, step = None):
        """
        Creates a new camera object.

        *Arguments*:
         - pos = the camera position vector.
         - ori = the camera orientation vector.
         - proj = the project type: 'persp' or 'pano'.
         - fov = the camera field of view (degrees).
         - dims = the pixel dimensions of the frame.
         - step = the angular step of pixels in the x-direction for panoramic projections. Default is None.
        """

        assert isinstance(pos, np.ndarray) and len(pos)==3, \
                                            "Error - camera_position must be a numpy array with length 3. "
        self.pos = pos.copy()

        assert isinstance(ori, np.ndarray) and len(ori) == 3, \
                                            "Error - camera_ori must be a numpy array with length 3. "
        self.ori = ori.copy()

        assert 'persp' in proj.lower() or 'pano' in proj.lower(), \
                           "Error - camera type should be pano or persp."
        self.proj = proj

        assert len(dims) >= 2, "Error, dims must be a tuple or list of length 2."
        self.dims = dims

        assert np.isfinite(fov), "Error, fov must be numeric."
        self.fov = fov
        self.step = step

    def clone(self):
        """
        Create a deep copy of a camera instance.
        """
        return Camera( self.pos.copy(), self.ori.copy(), self.proj, self.fov, self.dims, self.step )

    def set_transform(self, camera_pos, camera_ori):
        """
        Sets the position and rotation of this camera.

        *Arguments*:
         - camera_pos = the new position.
         - camera_ori = the new orientation.
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

    def get_rotation_matrix(self):
        """
        Return the rotation matrix of this camera (based on its Euler angles).
        """
        return spatial.transform.Rotation.from_euler('XYZ', -self.ori, degrees=True).as_matrix()
