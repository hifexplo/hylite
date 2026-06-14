"""
Optional dependency imports for hylite.

- require_on_load(...): call at submodule import for deps needed by any use of that submodule.
- require(...): call inside functions that need a package only for that code path.
- optional(...): soft probe (e.g. GDAL availability) without raising.
- simulateEnv(tier): restrict visible packages for testing (basic|lite|default|opencv|gdal|all).
"""

import importlib
from functools import lru_cache

# Packages included when simulating each install target.
_TIER_INCLUDES = {
    "basic": frozenset(["basic"]),
    "lite": frozenset(["basic", "lite"]),
    "default": frozenset(["basic", "lite", "default"]),
    "opencv": frozenset(["basic", "lite", "default", "opencv"]),
    "gdal": frozenset(["basic", "lite", "default", "gdal"]),
    "all": frozenset(["basic", "lite", "default", "opencv", "gdal"]),
}

# import_name -> (pip package name, minimum tier)
_REGISTRY = {
    "gfit.util": ("gfit", "basic"),
    "gfit": ("gfit", "basic"),
    "tqdm": ("tqdm", "basic"),
    "utm": ("utm", "lite"),
    "pytz": ("pytz", "lite"),
    "laspy": ("laspy", "lite"),
    "plyfile": ("plyfile", "lite"),
    "astral": ("astral", "lite"),
    "astral.sun": ("astral", "lite"),
    "piexif": ("piexif", "lite"),
    "numba": ("numba", "lite"),
    "natsort": ("natsort", "lite"),
    "matplotlib": ("matplotlib", "default"),
    "matplotlib.pyplot": ("matplotlib", "default"),
    "scipy": ("scipy", "default"),
    "scipy.cluster.hierarchy": ("scipy", "default"),
    "scipy.ndimage": ("scipy", "default"),
    "scipy.optimize": ("scipy", "default"),
    "scipy.sparse": ("scipy", "default"),
    "scipy.spatial": ("scipy", "default"),
    "scipy.stats": ("scipy", "default"),
    "PIL": ("Pillow", "default"),
    "skimage": ("scikit-image", "default"),
    "spectral": ("spectral", "default"),
    "roipoly": ("roipoly", "default"),
    "sklearn": ("scikit-learn", "default"),
    "pysptools": ("pysptools", None),  # external optional; not bundled in hylite extras
    "cv2": ("opencv-contrib-python", "opencv"),
    "osgeo": ("GDAL", "gdal"),
    "osgeo.gdal": ("GDAL", "gdal"),
}

_simulated_tier = "all"


def _registry_entry(name):
    if name in _REGISTRY:
        return _REGISTRY[name]
    parts = name.split(".")
    for i in range(len(parts), 0, -1):
        parent = ".".join(parts[:i])
        if parent in _REGISTRY:
            return _REGISTRY[parent]
    return (name, "default")


def _package_tier(name):
    """Minimum install tier for an import name."""
    return _registry_entry(name)[1]


def _tier_available(name):
    tier = _package_tier(name)
    if tier is None:
        return True
    return tier in _TIER_INCLUDES[_simulated_tier]


def _install_hint(name):
    pkg, tier = _registry_entry(name)
    if tier is None:
        return "pip install %s" % pkg
    if tier == "default":
        return "pip install hylite  (includes %s)" % pkg
    return "pip install hylite[%s]  (includes %s)" % (tier, pkg)


def simulated_tier():
    """Return the currently simulated install target (default `all`)."""
    return _simulated_tier


def resetSimulateEnv():
    """Restore full dependency visibility (`all` tier)."""
    global _simulated_tier
    _simulated_tier = "all"
    require.cache_clear()


def simulateEnv(tier):
    """
    Simulate running with a given `hylite` install target.

    Targets: `basic`, `lite`, `default`, `opencv`, `gdal`, `all`.

    While a test runs, :func:`require` and :func:`optional` treat packages outside
    the simulated target as unavailable.

    Returns `True` if every package for *tier* is physically installed (the test
    can run). Returns `False` if any are missing (the test should skip).
    """
    global _simulated_tier
    tier = str(tier).lower().strip()
    if tier == "full":
        tier = "all"
    if tier not in _TIER_INCLUDES:
        raise ValueError(
            "tier must be one of: basic, lite, default, opencv, gdal, all (got %r)" % tier
        )
    _simulated_tier = tier
    require.cache_clear()
    allowed = _TIER_INCLUDES[tier]
    for name, (_, package_tier) in _REGISTRY.items():
        if package_tier is None or package_tier not in allowed:
            continue
        try:
            importlib.import_module(name)
        except ImportError:
            return False
    return True


@lru_cache(maxsize=None)
def require(name):
    """
    Import an optional dependency or raise ImportError with install instructions.

    Args:
        name (str): module name passed to importlib.import_module(...).

    Returns:
        the imported module.
    """
    hint = _install_hint(name)
    if not _tier_available(name):
        raise ImportError(
            "'%s' is not available in simulated hylite[%s]. Install with: %s"
            % (name, _simulated_tier, hint)
        )
    try:
        return importlib.import_module(name)
    except ImportError as exc:
        raise ImportError(
            "'%s' is required. Install with: %s" % (name, hint)
        ) from exc


def require_on_load(module_name, *names):
    """
    Validate dependencies when a hylite submodule is imported.

    Use for packages required by any typical use of the submodule (fail fast).
    """
    for name in names:
        try:
            require(name)
        except ImportError as exc:
            raise ImportError(
                "Cannot import hylite.%s: %s" % (module_name, exc)
            ) from exc


def optional(name, default=None):
    """
    Return an imported module, or default if it is not installed.

    Useful for capability probes (e.g. GDAL availability in io).
    """
    if not _tier_available(name):
        return default
    try:
        return importlib.import_module(name)
    except ImportError:
        return default
