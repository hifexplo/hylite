"""
Simple class for encapsulating light source properties.
"""
from hylite.correct.illumination import sph2cart

class SourceModel(object):
    """
    Class for encapsulating light source propertes (spectra, position, etc.)
    """
    def __init__(self, direction=None, spectra=1.0, ):
        """
        Create a new light source.

        *Arguments*:
         - direction = the (downward pointing) illumination vector. Can be none if illumination is omnidirectional.
         - spectra = a n-d numpy array defining the radiance of the light source in each band. E.g. downwelling illumination
                     calculated using calibration panels. Default is 1.0 (pure white).
        """

        self.spectra = spectra
        self.illuVec = direction
        if self.illuVec is not None:
            if len(self.illuVec) == 3:
                if self.illuVec[2] > 0: # z-component should always point downwards
                    self.illuVec *= -1
            elif len(self.illuVec) == 2:
                self.illuVec = sph2cart( self.illuVec[0] + 180, self.illuVec[1] ) # n.b. +180 flips direction from vector point at sun to vector
            else:
                assert False, "Error: direction must be a (3,) cartesian vector or a (2,) vector with azimuth, elevation (in degrees)."

    def r(self):
        """
        Return the radiance spectra for this source.sdfdfdfdfd
        """
        return self.spectra

    def d(self):
        """
        Return the direction vector for this source, or None if the source is omnidirectional.
        """
        return self.illuVec