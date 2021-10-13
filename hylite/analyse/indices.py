import numpy as np


def band_ratio(data, num, den):
    """
    Calculate a band ratio to map broad absorbion features (e.g. iron).

    *Arguments*:
     - data = HyData instance to calculate a band ratio for.
     - num = the numerator of the band ratio. Integers are treated as indices, floats as wavelengths.

                If a tuple is passed then values between band1[0] and band1[1] will be averaged before computing the
               band ratio. Lists of tuples/floats or indices will be summed (for calculating ratios of the form (a+b)/c).

     - den = the denominator of the band ratio. Values are treated like in band1.
    *Returns*:
     - a new HyData instance containing the band ratio.

    """
    if not isinstance(num, list):
        num = [num]
    if not isinstance(den, list):
        den = [den]

    # calculate numerator and denominator
    numdata = np.zeros(data.data.shape[:-1])
    dendata = np.zeros(data.data.shape[:-1])
    names = []
    for D, exp in zip([numdata, dendata], [num, den]):
        name = None
        for n in exp:
            if isinstance(n, tuple):
                # calculate addition or subtraction
                sign = 1
                if n[0] < 0 or n[1] < 0:
                    sign = -1
                    n = (abs(n[0]), abs(n[1]))

                assert n[0] < n[1], "Error - invalid slice %.1f:%.1f" % n
                idx = slice(data.get_band_index(n[0]),
                            data.get_band_index(n[1]), 1)
                if name is None:
                    name = "%.1f:%.1f" % n
                else:
                    if sign > 0:
                        name += " + %.1f:%.1f" % n
                    else:
                        name += " - %.1f:%.1f" % n
            else:
                sign = 1
                if n < 0:
                    sign = -1
                    n = abs(n)

                idx = data.get_band_index(n)
                if name is None:
                    name = "%.1f" % n
                else:
                    if sign > 0:
                        name += " + %.1f" % n
                    else:
                        name += " - %.1f" % n

            # accumulate
            slc = data.data[..., idx]
            if isinstance(idx, slice):
                slc = np.nanmean(slc, axis=-1)
            D += sign * slc

        # store name
        names.append(name)

    # calculate output
    out = data.copy(data=False)
    out.header.drop_all_bands()  # drop band specific attributes
    out.data = (numdata / dendata)[..., None]

    # set band name
    out.set_band_names(['(%s) / (%s)' % (names[0], names[1])])
    out.push_to_header()

    # return HyData instance
    return out

def NDVI(data):
    """
    Calculate NDVI.

    *Arguments*:
     - data = the HyData instance to calculate a NDVI for.
    *Returns*:
     - a new HyData instance containing the band ratio.
    """

    idxNIR = data.get_band_index(800.0)
    idxRed = data.get_band_index(670.0)

    # create output HyData instance
    out = data.copy(data=False)
    out.header.drop_all_bands()  # drop band specific attributes

    # calculate NDVI
    with np.errstate(all='ignore'):  # ignore div0 errors
        out.data = ((data.data[..., idxNIR] - data.data[..., idxRed]) / (
                data.data[..., idxNIR] + data.data[..., idxRed]))[..., None]

    # set band name
    out.set_band_names(['NDVI'])
    out.push_to_header()

    return out

def SKY(data):
    """
    Calculate ratio between blue (479.89 nm and featureless swir (1688.64 nm) to identify sky pixels and (sometimes)
    very distance pixels due to blue scattering.

    *Arguments*:
     - data = the HyData instance to calculate a NDVI for.
    *Returns*:
     - a new HyData instance containing the SKY band ratio.
    """

    return band_ratio(data, 479.89, 1688.64)

def SHADE(data):
    """
    Calculate ratio between blue (479.89 nm and near infrared (1688.64 nm) to identify shadow pixels (at least in theory).

    Generally this should work well, except for objects that are very blue. Luckily blue things are not common in Geology.

    *Arguments*:
     - data = the HyData instance to calculate SHADE for.
    *Returns*:
     - a new HyData instance containing the SHADE band ratio.
    """

    return band_ratio(data, 480.0, 800.0)

