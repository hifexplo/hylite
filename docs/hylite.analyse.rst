hylite.analyse
======================

.. automodule:: hylite.analyse
   :members:
   :undoc-members:
   :show-inheritance:

---------------------------------

.. toctree::
   :maxdepth: 2

   hylite.analyse

---------------------------------


Band ratios
---------------------------------

This package contains a generic and flexible set of functions for calculating common
band ratios and spectral indices.

.. automodule:: hylite.analyse.indices
   :members:
   :undoc-members:
   :show-inheritance:


---------------------------------



Minimum wavelength mapping
-----------------------------

This package contains a variety of functions for calculating, processing and visualising
minimum wavelength maps. These can be simple minimimum wavelength maps fitting a single
feature to a small spectral range, or complex multi-feature maps that combine several gaussian
or lorentzian features to reproduce observed spectra.


.. automodule:: hylite.analyse.mwl
   :members:
   :undoc-members:
   :show-inheritance:


---------------------------------



Classification
---------------

*hylite* does not specifically implement classification methods as these are excellently covered
by the scikit-learn package, but we do provide some useful functions for converting *hylite* data types
into labelled feature vectors to be used with scikit-learn.

**Decision tree**

.. toctree::
   :maxdepth: 4

   hylite.analyse.supervised
   hylite.analyse.unsupervised

.. automodule:: hylite.analyse.dtree
   :members:
   :undoc-members:
   :show-inheritance:

**Spectral angle mapping**

.. automodule:: hylite.analyse.sam
   :members:
   :undoc-members:
   :show-inheritance:

