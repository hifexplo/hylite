"""
Save and load viewing positions associated with data acquisition or rendering 3D point cloud data.
"""

import numpy as np

from hylite.io import makeDirs
from hylite.project import Camera, Pushbroom


def saveCameraTXT( path, camera ):
    """
    Saves a Camera view to a simple text format. The camera can either be an instance of Camera or
    Pushbroom.

    Args:
        path: the path to save the file.
        camera: a camera object containing the data to save.
    """

    # make directories if need be
    makeDirs( path )

    with open(path,'w') as f:

        if isinstance( camera, Camera ):
            f.write("Camera # camera type\n")
            f.write('%.3f,%.3f,%.3f #camera position\n' % (camera.pos[0],camera.pos[1],camera.pos[2]))
            f.write('%.3f,%.3f,%.3f #camera ori\n' % (camera.ori[0], camera.ori[1], camera.ori[2]))
            f.write('%s #projection type\n' % camera.proj)
            f.write('%.3f #vertical field of view (deg)\n' % camera.fov)
            f.write('%d,%d #dims\n' % (camera.dims[0],camera.dims[1]))
            if 'pano' in camera.proj.lower():
                f.write('%.3f # angular pitch (x)\n' % camera.step)
        elif isinstance( camera, Pushbroom ):
            f.write("Pushbroom # camera type\n")
            f.write('%.3f,%.3f # xfov, yfov\n' % (camera.xfov, camera.lfov))
            f.write('%d,%d # xdim, ydim\n' % (camera.dims[0], camera.dims[1]))
            for i in range(3): # write position arrays
                f.write("%s\n" % ','.join(["%.6f" % v for v in camera.cp[:,i]]))
            for i in range(3):  # write orientation arrays
                f.write("%s\n" % ','.join(["%.6f" % v for v in camera.co[:, i]]))
        else:
            assert False, '%s is not a Camera or Pushbroom object' % type(camera)


def loadCameraTXT( path ):
    """
    Loads a Camera view from a simple text format.

    Returns:
        a Camera object containing the camera properties.
    """

    with open(path, 'r') as f:
        lines = f.readlines()

        #trim comments
        lines = [l.split('#')[0].strip() for l in lines]
        if 'camera' in lines[0].lower():
            pos = np.fromstring( lines[1], sep=',')
            ori = np.fromstring( lines[2], sep=',')
            proj = lines[3]
            fov = float(lines[4])
            dims = tuple( np.fromstring( lines[5], sep=',').astype(np.int))
            step  = None
            if len(lines) > 6:
                step = float( lines[6] )
            return Camera(pos, ori, proj, fov, dims, step)
        elif 'pushbroom' in lines[0].lower():

            xfov, lfov = np.fromstring( lines[1], sep=',')
            dims = tuple( np.fromstring( lines[2], sep=',').astype(np.int))
            cp = np.array([ np.fromstring(lines[3+i], sep=',') for i in range(3)]).T
            co = np.array([ np.fromstring(lines[6+i], sep=',') for i in range(3)]).T

            return Pushbroom( cp, co,  xfov, lfov, dims )