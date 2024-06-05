"""
Reference spectra such as calibration targets or spectral libraries.
"""

import numpy as np
import os
import glob
from pathlib import Path

class Target(object):
    """
    Class defining the reflectance properties of a calibration target.
    """
    def __init__(self, wavelength, reflectance, name = "Custom"):
        """
        Load a custom target from a .txt file with a band,reflectance pair on each line. Lines not of this format will be ignored.

        Args:
            wavelength (ndarray): the calibration wavelengths (in nanometers).
            reflectance (ndarray): the target reflectances.
            name (str): the name of this target.
        """

        self.band = np.array(wavelength)
        self.refl = np.array(reflectance)
        self.name = name

        assert self.band.shape[0] > 0, "Error: Calibration data must cover multiple wavelengths"
        assert self.band.shape == self.refl.shape, "Error: wavelength and reflectance data must have the same length."

    def get_name(self):
        """
        Return the name of this reference target.
        """
        return self.name

    def get_wavelengths(self):
        """
        Get the wavelengths corresponding to the reflectance measurements for this target.
        """
        return self.band

    def get_reflectance(self):
        """
        Get the reflectance measurements for this target.
        """
        return self.refl

    def writeTXT(self, path):
        """
        Save this calibration target to txt file
        """
        if os.path.isfile( path):
            path = os.path.dirname(path) #make sure path is a directory
        if not os.path.exists(path): #and it exists
            os.makedirs(path)

        #create file
        path = str( Path(path) / "%s.txt" % self.get_name())
        with open(path,'w') as f:
            w = self.get_wavelengths()
            r = self.get_reflectance()
            for i in range(len(w)):
                f.write("%f,%f\n" % (w[i],r[i]))

def loadTarget( path ):
    """
    Load a target from a .txt file with a band,reflectance pair on each line. Lines not of this format will be ignored.

    Args:
        path (str): path to the file to load.
    """

    band = []
    refl = []
    name = os.path.splitext(os.path.basename(path))[0]  # get name as filename
    with open(path, 'r') as f:
        lines = f.readlines()
        for l in lines:
            if ',' in l:
                split = l.split(",")  # comma delimited
            elif "\t" in l:
                split = l.split('\t')  # tab delimited
            else:
                split = l.split(' ')  # space delimited
            if len(split) == 2:
                band.append(float(split[0]))
                refl.append(float(split[1]))

    return Target( band, refl, name )

def loadDirectory( path ):
    """
    Load all spectra (stored as .txt files) in a directory.

    Args:
        path (str): the directory to search.

    Returns:
        A dictionary of reference (with target names as keys)
    """

    files = glob.glob( str( Path(path) / "*.txt" ) )
    targets = {}
    for f in files:
        try:
            t = loadTarget(f)
            targets[t.get_name()] = t
        except:
            print("Warning: could not parse target %s" % os.path.dirname(f))

    return targets


# load reference data from associated directories
spectralon = loadDirectory( str( Path(__file__).parent / "spectralon" ) )
PVC = loadDirectory( str( Path(__file__).parent / "pvc" ) )
custom = loadDirectory( str( Path(__file__).parent / "pvc" ) )

# use these to create local variables (for easy access/import)
locals().update(spectralon)
locals().update(PVC)
locals().update(custom)