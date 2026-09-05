import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

# Install tiers (see hylite/_deps.py for runtime tier checks).
# pip install hylite          -> DEFAULT (install_requires)
# pip install hylite[opencv]  -> DEFAULT + OpenCV
# pip install hylite[gdal]    -> DEFAULT + GDAL
# pip install hylite[all]     -> DEFAULT + OpenCV + GDAL
_BASIC = [  # these are really core dependencies
    "numpy",
    "tqdm",
    "gfit",
]

_LITE = _BASIC + [ # these lightweight deps needed for some / many tasks 
    "utm",
    "pytz",
    "laspy",
    "plyfile",
    "astral",
    "piexif",
    "numba",
    "natsort",
]
_DEFAULT = _LITE + [ # default install includes common scientific python stack (matplotlib, scipy, etc)
    "matplotlib>=3",
    "Pillow",
    "scikit-image",
    # used widely in io / filters / transforms but not part of the named stacks above:
    "spectral",
    "roipoly",
    "scikit-learn",
    "scipy>=1.4",
]
_OPENCV = ["opencv-contrib-python>=4.5"]
_GDAL = ["GDAL>=3"]

setuptools.setup(
    name="hylite",
    version="1.42",
    author="Helmholtz Institute Freiberg",
    author_email="s.thiele@hzdr.de",
    description="Open-source toolbox for hyperspectral geology.",
    long_description="A python package for loading, correcting, projecting and analysing hyperspectral datasets, with particular emphasis on geological applications.",
    long_description_content_type="text/markdown",
    url="https://github.com/samthiele/hylite",
    packages=setuptools.find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering"
    ],
    keywords='hyperspectral data analysis hypercloud geology mineral mapping',
    python_requires='>=3.6',
    install_requires=_DEFAULT,
    extras_require={
        "opencv": _OPENCV,
        "gdal": _GDAL,
        "all": _OPENCV + _GDAL,
    },
    project_urls={
        'Source': 'https://github.com/samthiele/hylite',
        'Documentation': 'https://hifexplo.github.io/hylite/hylite.html',
    },
    package_data={"": ["*.txt", "*.hdr", "*.cal", "*.dat"]}
)
