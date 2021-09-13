import numpy as np
import hylite

def estimate_incidence(normals, sunvec):
    """
    Utility function to estimate the cosine of incidence angles based on normals and calculated sun position.

    *Arguments*:
     - normals = either: (1) a HyImage with band 0 = nx, band 1 = ny and band 2 = nz, (2) HyCloud instance containing
                 normals, or (3) mx3 numpy array of normal vectors.
     - sunvec = a numpy array containing the sun illumination vector (as calculated by estimate_sun_vec(...)).

    *Returns*:
     - list of incidence angles matching the shape of input data (but with a single band only).
    """

    # extract normal vectors
    if isinstance(normals, hylite.HyCloud):
        N = normals.normals[:, :3]
        outshape = normals.point_count()
    elif isinstance(normals, hylite.HyImage):
        N = normals.get_raveled()[:, :3]
        outshape = (normals.xdim(), normals.ydim())
    else:
        N = normals.reshape((-1, 3))
        outshape = normals.shape[:-1]

    # normalize normals (just to be safe)
    N = N / np.linalg.norm(N, axis=1)[:, None]

    # calculate cosine of angles used in correction
    cosInc = np.dot(-N, sunvec)  # cos incidence angle

    # return in same shape as original data
    return cosInc.reshape(outshape)

def calcOrenNayar(normals, view, source, roughness=0.3):
    """
    Calculate OrenNayar reflectance across a scene.

    *Arguments*:
     - normals = a (...,3) array containing normal vectors for each point or pixel.
     - view = a (...,3) array containing viewing directions for each point or pixel.
     - source = a (3,) array containing the illumination direction.
     - roughness = a float describing the OrenNayar roughness parameter. Default is 0.3.

    *Returns*:
     - a (...,1) numpy array containing the Oren-Nayar reflectance factors.
    """

    # check array shapes are compatible
    assert normals.shape == view.shape, "Error - normals array and view array must have the same shape."
    assert normals.shape[-1] == 3, "Error - the last axis of the normal and view arrays must have 3 values (x,y,z)."
    assert source.shape[0] == 3 and len(source.shape) == 1, "Error - source must be a 3-D vector."

    outshape = normals.shape[:-1] + (1,)  # output shape
    N = normals.reshape((-1, 3))  # flatten
    V = view.reshape((-1, 3))
    r2 = roughness ** 2
    I = source  # shorthand for later

    # calculate roughness terms
    A = 1.0 - (0.5 * r2) / (r2 + 0.33)
    B = (0.45 * r2) / (r2 + 0.09)

    LdotN = estimate_incidence(N, I)  # I . N [ = cos (incidence angle) ]
    VdotN = np.sum(V * N, axis=-1)  # V . N [ = cos( viewing angle ) ]

    # remove backfaces
    irradiance = LdotN.copy()
    irradiance[irradiance < 0] = 0

    # convert cosines to radians
    angleViewNormal = np.arccos(VdotN);
    angleLightNormal = np.arccos(LdotN);

    a = (V - N * VdotN[:, None]) / np.linalg.norm((V - N * VdotN[:, None]), axis=-1)[:, None]
    b = (I - N * LdotN[:, None]) / np.linalg.norm((I - N * LdotN[:, None]), axis=-1)[:, None]

    angleDiff = np.sum(a * b, axis=-1)
    angleDiff[angleDiff < 0] = 0

    alpha = np.nanmax(np.dstack([angleViewNormal, angleLightNormal]), axis=-1)[0, :]
    beta = np.nanmin(np.dstack([angleViewNormal, angleLightNormal]), axis=-1)[0, :]

    # return
    return (irradiance * (A + B * angleDiff * np.sin(alpha) * np.tan(beta))).reshape(outshape)

def calcLambert(normals, source):
    """
    Calculate lambertian reflectance across a scene.

    *Arguments*:
     - normals = a (...,3) array containing normal vectors for each point or pixel.
     - source = a (3,) array containing the illumination direction.

    *Returns*:
     - a (...,1) numpy array containing the Lambertian reflectance factors.
    """

    # check array shapes are compatible
    assert normals.shape[-1] == 3, "Error - the last axis of the normal and view arrays must have 3 values (x,y,z)."
    assert source.shape[0] == 3 and len(source.shape) == 1, "Error - source must be a 3-D vector."

    outshape = normals.shape[:-1] + (1,)  # output shape

    # I . N [ = cos (incidence angle) ]
    return np.clip(estimate_incidence(normals.reshape((-1, 3)), source), 0, 1).reshape(outshape)

