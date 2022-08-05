"""
Functions for calculating spectral angles and doing spectral angle mapping.
"""

import numpy as np

def spectral_angles(reference, spectra):
    """
    Calculate the angle between each spectra and each reference.

    Args:
        reference: a numpy array of reference spectra.
        spectra: a numpy array of spectra to compare to the references.

    Returns:
        a 2D numpy array such that [spectra i][reference j] gives the angle between the i'th spectra and j'th reference spectra.
    """

    # normalise spectra
    reference = np.array(reference) / np.linalg.norm(np.array(reference), axis=1)[:, None]
    spectra = np.array(spectra) / np.linalg.norm(spectra, axis=1)[:, None]

    # calculate angle
    out = np.zeros((len(reference), len(spectra)))
    for i in range(len(reference)):
        out[i, :] = np.arccos(np.dot(reference[i], spectra.T))

    return out


def SAM(data, ref_spec):
    """
    Apply a spectral angle classification based on reference spectra.

    Args:
        data: the HyData instance (e.g. image or cloud) to apply the classification to.
        ref_spec: a list containing lists of spectra for each class. i.e.:
                    ref_spect = [ [class1_spec1, class1_spec2],[class2_spec1, class2_spec2], ... ]
    Returns:
        a HyData instance with the same type as data containing two bands: the class index, and the spectral angle to this (closest) class.
    """
    R = []
    L = []
    for i, S in enumerate(ref_spec):
        for s in S:
            L.append(i)  # label of this spectra
            R.append(s)  # append spectra

    # calculate angles
    ang = spectral_angles(np.array(R), data.X())

    # extract classifications
    sam = np.take(L, np.argmin(ang, axis=0)).astype(np.float32)  # find best matching class
    ang = np.rad2deg(np.min(ang, axis=0)).astype(np.float32)  # calculate spectral angle

    # mask
    sam[np.isnan(data.X()).all(axis=-1)] = np.nan
    ang[np.isnan(data.X()).all(axis=-1)] = np.nan

    # reshape and return
    out = data.copy(data=False)
    out.set_raveled(np.array([sam, ang]).T, shape=data.data.shape[:-1] + (2,))
    out.set_wavelengths(None)
    out.set_band_names(["Class", "Angle"])
    return out