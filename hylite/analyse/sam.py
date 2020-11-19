"""
Functions for calculating spectral angles and doing spectral angle mapping.
"""

import numpy as np

def spectral_angles(reference, spectra):
    """
    Calculate the angle between each spectra and each reference.

    *Arguments*:
     - reference = a numpy array of reference spectra.
     - spectra = a numpy array of spectra to compare to the references.

    *Returns*: a 2D numpy array such that [spectra i][reference j] gives the angle between
               the i'th spectra and j'th reference spectra.
    """

    # normalise spectra
    reference = np.array(reference) / np.linalg.norm(np.array(reference), axis=1)[:, None]
    spectra = np.array(spectra) / np.linalg.norm(spectra, axis=1)[:, None]

    # calculate angle
    out = np.zeros((len(reference), len(spectra)))
    for i in range(len(reference)):
        out[i, :] = np.arccos(np.dot(reference[i], spectra.T))

    return out