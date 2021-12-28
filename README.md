hylite
----------

*hylite* is an open-source python package for preprocessing imagery from a variety of hyperspectral sensors
and fusing the results with high-resolution point-cloud data to generate seamless and radiometrically corrected
hyperclouds.  A variety of analysis techniques are also implemented, including multi-feature gaussian minimum wavelength mapping,
dimensionality reduction and spectral angle mapping. Reference spectra from spectral libraries, ground or laboratory measurements
can also be integrated and used to perform supervised classifications using machine learning techniques.

A key design feature of *hylite* is polymorphism between different spectral data types, such that spectral libraries,
images and point clouds can be easily analysed and integrated. Pre-processing workflows for each of these
data types have also been implemented.

*hylite* also includes a variety of tools for visualising different hyperspectral datasets and associated derivatives. For
example, minimum wavelength maps can be easily calculated and visualised for spectral libraries, laboratory scans and
outdoor scenes.

------

![workflow image](docs/workflow.png)


*Preprocessing and correction workflows implemented in hylite for different data types.*

-----------

![hypercloud image](docs/mwl.png)


*Example visualisations of minimum wavelength maps calculated for (a) imagery of rock samples acquired using a core-scanner
and (b) a hypercloud of an open-pit mine.*

----------

Examples
--------------

Try the *hylite* hyperspectral toolbox using some example notebooks on Binder:

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/samthiele/hylite_demo/HEAD)


Release notes
--------------

#### Version 1.2

New features:
* projection of push-broom data using `hylite.project.Pushbroom`
* `HyCollection` class for easily loading / saving large numbers of data files 
* Completely rewritten `HyLibrary` class for easily merging, resampling and splitting spectral libraries.
* Added `align_to_cloud_manual` function for locating cameras with manually chosen tiepoints.

Improvements:
* Completely re-written minimum wavelength mapping code for improved performance (thanks Numba!)
Installation (using PIP).
* Simplified structure for topographic and atmospheric corrections for cleaner code and increased flexibility.
* Many improvements to plotting functions.
* Greatly simplified input output code by wrapping specific funtions in generic `io.load` and `io.save`.
* Removed GDAL as a required dependency (SPy will be used instead if GDAl can't be found). Note that SPy can have 
  unpredictable behaviour for non-reflectance files (outside of 0 - 1 range).
* Increased performance of `get_hull_corrected` and `rasterize` functions using Numba.
* Significantly expanded penetration of test functions (though more work is needed here still).

Installation
--------------

1. Create and activate a new python environment (anacona users only)

```
conda create -n hylite
conda activate hylite
````


------------

2 Install *hylite* with pip.

`pip install hylite`


Installation (from GitHub)
--------------

1. Create and activate a new python environment (anacona users only)

```
conda create -n hylite
conda activate hylite
````

2. Download and unzip hylite from GitHub (or clone it using `git clone https://github.com/samthiele/hylite.git`)

3. Navigate into the hylite directory using terminal and install using setuptools:

`python setup.py install`


Optional dependencies:
------------

A variety of other python packages might be needed depending on how you use _hylite_. These include:
 - _GDAL_: needed if working with georeferenced images (e.g. geotiffs, some envi files).
 - _jupyter_: recommended as coding interface when using hylite for exploratory data analysis.


Testing installation
----------------------

Check *hylite* is installed by opening a python console and running:

```python
import hylite
```

A better test of the installation can be performed by downloading the test data included in this repository, launching python or a jupyter notebook
and running the following code:

```python

import hylite
from hylite import io

lib = io.load( 'test_data/library.csv' )
lib.quick_plot()

image = io.load( 'test_data/image.hdr' )
image.quick_plot(hylite.RGB)

cloud = io.load( 'test_data/hypercloud.ply' )
cloud.quick_plot(cloud.header.get_camera(0), hylite.RGB)
```

Other test functionality is included in the _tests_ directory.

Next steps
-------------

1. Navigate to the template_notebooks directory and launch a jupyter notebook server.
2. Find a notebook that does what you need (or extend one to do what you want).
3. Happy processing! :D

Citing *hylite*
---------------

If you use *hylite* for your work then please cite:


```
Thiele, S. T., Lorenz, S., et al., (2021). Multi-scale, multi-sensor data
integration for automated 3-D geological mapping. Ore Geology Reviews. DOI: j.oregeorev.2021.104252
```
https://doi.org/10.1016/j.oregeorev.2021.104252


Contributing to  hylite
-------------------------

Cool additions are welcomed!
Please feel free to submit pull requests through GitHub or get in touch with us directly if
you have any questions. Bug reports are also welcomed (though please do try to be specific).

Documentation
---------------

For more information on how to use *hylite*, please refer to https://hylite.readthedocs.io/en/latest/index.html.

---------------
