import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

##TODO - do this properly
setuptools.setup(
    name="hylite",
    version="1.2",
    author="Helmholtz Institute Freiberg",
    author_email="s.thiele@hzdr.de",
    description="Open-source toolbox for hyperspectral geology.",
    long_description="A python package for loading, correcting, projecting and analysing hyperspectral datasets, with particular emphasis on geological applications.",
    long_description_content_type="text/markdown",
    url="https://github.com/samthiele/hylite",
    #package_dir={'': 'hylite'},
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
    install_requires=["scipy>=1.4", "matplotlib>=3", "numpy",
                      "imageio","opencv-python", "opencv-contrib-python",
                      "tqdm", "roipoly", "spectral", "utm",
                      "laspy","plyfile","astral","piexif","gfit"],

    # List additional URLs that are relevant to your project as a dict.
    #
    # This field corresponds to the "Project-URL" metadata fields:
    # https://packaging.python.org/specifications/core-metadata/#project-url-multiple-use
    #
    # Examples listed include a pattern for specifying where the package tracks
    # issues, where the source is hosted, where to say thanks to the package
    # maintainers, and where to support the project financially. The key is
    # what's used to render the link text on PyPI.
    project_urls={  # Optional
        'Source': 'https://github.com/samthiele/hylite',
    },

    package_data={"": ["*.txt","*.hdr","*.cal","*.dat"] }
    #todo - include data files here with data_files = [ .... ]
)
