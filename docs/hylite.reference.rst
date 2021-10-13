hylite.reference
========================
Generic package for encapsulating reference hyperspectral data such as calibration panels (*hylite.reference.spectra*)
and common features (*hylite.reference.features*). In the future functionality will be added here to link with common
spectral libraries (e.g., USGS).

.. automodule:: hylite.reference
   :members:
   :undoc-members:
   :show-inheritance:

---------------------------------

.. toctree::
   :maxdepth: 2

   hylite.reference

---------------------------------

Reference Spectra
-----------------------
Reference spectra used for radiance - reflectance conversions (e.g. ELC). To include your own reference spectra,
add .txt files containing wavelength, reflectance pairs (separated by commas) to *hylite.reference.custom*.

.. toctree::
   :maxdepth: 4

   hylite.reference.custom
   hylite.reference.pvc
   hylite.reference.spectralon

.. automodule:: hylite.reference.spectra
   :members:
   :undoc-members:
   :show-inheritance:

hylite.reference.spectra.spectralon package
===========================================

Reference spectra for Spectralon panels as measured by Helmholtz Institute Freiberg.

Module contents
---------------

.. automodule:: hylite.reference.spectra.spectralon
   :members:
   :undoc-members:
   :show-inheritance:


hylite.reference.spectra.custom package
=======================================

Add custom reference spectra here as .txt files containing a wavelength, reflectance pair on each line, for example:

1000.0, 0.9
1001.0, 0.91
1002.0, 0.89

These files are loaded automatically when you import hylite.reference and can be accessed as follows:

`from hylite.reference.custom import **filename**`

Module contents
---------------

.. automodule:: hylite.reference.spectra.custom
   :members:
   :undoc-members:
   :show-inheritance:



---------------------------------


Reference Features
-----------------------

This module contains a collection of HyFeature instances that approximate commonly
encountered absorbtion features. These are organised into two themes: *features*,
which contains generic absorbtions, and *minerals* which characterises the spectral
signature of commonly observed minerals.

These data are largely used for labelling/annotating figures and are not necessarily accurate.

.. automodule:: hylite.reference.features
   :members:
   :undoc-members:
   :show-inheritance:



