hylite
----------

*hylite* is an open-source python package for preprocessing imagery from a variety of hyperspectral sensors
and fusing the results with high-resolution point-cloud data to generate seamless and radiometrically corrected
hyperclouds.  A variety of analysis techniques are also implemented, including minimum wavelength mapping,
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


Installation
-------------

1. Create and activate a new python environment (anacona users only)

```
conda create -n hylite
conda activate hylite
````

2. Install GDAL

`conda install gdal`


3. Check gdal works by opening a python console and trying to import it
```python
from osgeo import gdal
```
On some environments GDAL can be difficult to install.

<details>
  <summary>Install with PIP</summary>

------------

4.1 Install *hylite* with pip.

`pip install hylite`

------------

</details>

<details>
  <summary>Install from Github</summary>

------------

4.1 Download the *hylite* directory from Github.

4.2 Unzip this directory and navigate to it in a terminal.

4.3 Install with setuptools by typing the following into terminal:

`python setup.py install`

4.4 Fix any issues  ¯\_(ツ)_/¯

------------

</details>

5. Check *hylite* is installed by opening a python console and running:

```python
import hylite
```

Testing installation
----------------------

A simple test of the installation can be performed by downloading the test data included in this repository, launching python or a jupyter notebook
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

Next steps
-------------

1. Navigate to the *template_notebooks* directory and launch a jupyter notebook server.
2. Find a notebook that does what you need (or extend one to do what you want).
3. Happy processing! :D

Citing *hylite*
---------------

If you use *hylite* for your work then please cite:

***Thiele, S. T., Lorenz, S., et al., (2021). Multi-scale, multi-sensor data
integration for automated 3-D geological mapping using hylite. Ore Geology Reviews.***

(currently under review; stay tuned!)

Contributing to  hylite
-------------------------

Cool additions are welcomed!
Please feel free to submit pull requests through GitHub or get in touch with us directly if
you have any questions. Bug reports are also welcomed (though please do try to be specific).

Documentation
---------------

For more information on how to use *hylite*, please refer to https://hylite.readthedocs.io/en/latest/index.html.

---------------
