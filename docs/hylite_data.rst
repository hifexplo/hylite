Data structures
========================

Hylite uses the following five classes to store and manipulate hyperspectral data.
These all inherit from HyData and so share much common functionality.

The hyperspectral data itself is stored in a numpy array *HyData.data*, indexed as [x,y,band] for images and
[ id, band ] for point clouds or spectral libraries.



---------------------------------


.. toctree::
   :maxdepth: 2

   hylite_data


---------------------------------



HyData
--------------------

.. automodule:: hylite.hydata
   :members:
   :undoc-members:
   :show-inheritance:


---------------------------------


HyHeader
--------------------

.. automodule:: hylite.hyheader
   :members:
   :undoc-members:
   :show-inheritance:


---------------------------------

HyImage
---------------------

.. automodule:: hylite.hyimage
   :members:
   :undoc-members:
   :show-inheritance:


---------------------------------



HyCloud
---------------------

.. automodule:: hylite.hycloud
   :members:
   :undoc-members:
   :show-inheritance:


---------------------------------



HyLibrary
-----------------------

.. automodule:: hylite.hylibrary
   :members:
   :undoc-members:
   :show-inheritance:


---------------------------------


HyCollection
---------------------

.. automodule:: hylite.hycollection
   :members:
   :undoc-members:
   :show-inheritance:


HyScene
---------------------

.. automodule:: hylite.hyscene
   :members:
   :undoc-members:
   :show-inheritance:


HyFeature
---------------------------------

.. automodule:: hylite.hyfeature
   :members:
