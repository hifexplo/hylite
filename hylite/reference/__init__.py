"""
Package containing spectral reference information for common minerals and calibration targets. Reference spectra are
located in the reference package while information on specific *absorption* features is stored in the *features* package.
"""

# lazy: generate.py needs gfit/skimage and must not run on `import hylite`
_GENERATE_NAMES = frozenset({"randomSpectra", "genImage"})


def __getattr__(name):
    if name in _GENERATE_NAMES:
        from . import generate
        return getattr(generate, name)
    raise AttributeError("module %r has no attribute %r" % (__name__, name))


def __dir__():
    return sorted(list(globals().keys()) + list(_GENERATE_NAMES))
