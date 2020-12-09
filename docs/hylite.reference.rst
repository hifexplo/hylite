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


---------------------------------



Reference Features
-----------------------

.. automodule:: hylite.reference.features
   :members:
   :undoc-members:
   :show-inheritance:

