Install
========================================================

Create and activate a new python environment (anaconda users only).

.. code:: shell

  conda create -n hylite
  conda activate hylite

*hylite* can then be most easily installed via pip:

.. code:: shell
    pip install hylite

----------------------

If you want the newest version of *hylite* then you can install it directly from GitHub.

First, download and unzip hylite from GitHub (or clone it using `git clone https://github.com/samthiele/hylite.git`)

Then navigate into the hylite directory using terminal and install using setuptools:

.. code:: shell
    python setup.py install

And.... Fix any issues  ¯\_(ツ)_/¯

------

Finally, check hylite is installed by opening a python console and running:

.. code:: python

    import hylite

Testing installation
----------------------

A simple test of the installation can be performed by downloading the test data included in this repository, launching python or a jupyter notebook
and running the following code:

.. code:: python

    import hylite
    from hylite import io

    lib = io.load( 'test_data/library.csv' )
    lib.quick_plot()

    image = io.load( 'test_data/image.hdr' )
    image.quick_plot(hylite.RGB)

    cloud = io.load( 'test_data/hypercloud.ply' )
    cloud.quick_plot(cloud.header.get_camera(0), hylite.RGB)

Other test functionality is included in the _tests_ directory.

Optional dependencies:
------------

A variety of other python packages might be needed depending on how you use *hylite*. These include:
 - _GDAL_: needed if working with georeferenced images (e.g. geotiffs, some envi files).
 - _jupyter_: recommended as coding interface when using hylite for exploratory data analysis.

