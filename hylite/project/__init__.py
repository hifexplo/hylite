"""
Project points between 2D image coordinates and 3D world coordinates. Also includes related problems such as
camera localisation.
"""

from .basic import *
from .camera import Camera

_PMAP_NAMES = {
    "PMap", "push_to_cloud", "push_to_image", "blend_scenes",
    "push_geomattr", "get_blend_weights",
}
_PUSHBROOM_NAMES = {
    "Pushbroom", "project_pushbroom", "optimize_boresight",
}


def __getattr__(name):
    if name in _PMAP_NAMES:
        from . import pmap
        return getattr(pmap, name)
    if name in _PUSHBROOM_NAMES:
        from . import pushbroom
        return getattr(pushbroom, name)
    raise AttributeError("module %r has no attribute %r" % (__name__, name))
