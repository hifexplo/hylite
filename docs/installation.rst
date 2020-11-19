Install
========================================================

1. Create and activate a new python environment (for anaconda users).

2. Install GDAL

.. code:: shell

    conda install gdal


3. Check gdal works by opening a python console and trying to import it

.. code:: python

    import gdal

On some environments GDAL can be difficult to install.

-------

4. Install *hylite* (PIP):
----------------------

4a. Install hylite from PIP

.. code:: shell
    pip install hylite

4. Install *hylite* (Github):
----------------------

4a. Download the hylite directory from Github.
4b. Unzip this directory and navigate to it in a terminal.
4c. Install with setuptools by typing the following into terminal:

.. code:: shell

    python setup.py install

4d. Fix any issues  ¯\_(ツ)_/¯

------

5. Check hylite is installed by opening a python console and running:

.. code:: python

    import hylite

Testing installation
----------------------

A simple test of the installation can be performed by downloading the test data included in this repository, launching python or a jupyter notebook
and running the following code:

.. code:: python

    import hylite
    from hylite import io

    lib = io.loadLibraryCSV( 'test_data/library.csv' )
    lib.quick_plot()

    image = io.loadWithGDAL( 'test_data/image.hdr' )
    image.quick_plot(hylite.RGB)

    cloud = io.loadCloudPLY( 'template_notebooks/demo_data/hypercloud.ply' )
    cloud.quick_plot(cloud.header.get_camera(0), hylite.RGB)
