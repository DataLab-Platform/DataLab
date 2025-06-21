"""
Algorithms (:mod:`sigima_.algorithms`)
----------------------------------

This package contains the algorithms used by the DataLab project. Those algorithms
operate directly on NumPy arrays and are designed to be used in the DataLab pipeline,
but can be used independently as well.

.. seealso::

    The :mod:`sigima_.algorithms` package is the main entry point for the DataLab
    algorithms when manipulating NumPy arrays. See the :mod:`sigima_.computation`
    package for algorithms that operate directly on DataLab objects (i.e.
    :class:`sigima_.obj.SignalObj` and :class:`sigima_.obj.ImageObj`).

The algorithms are organized in subpackages according to their purpose. The following
subpackages are available:

- :mod:`sigima_.algorithms.signal`: Signal processing algorithms
- :mod:`sigima_.algorithms.image`: Image processing algorithms
- :mod:`sigima_.algorithms.datatypes`: Data type conversion algorithms
- :mod:`sigima_.algorithms.coordinates`: Coordinate conversion algorithms

Signal Processing Algorithms
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: sigima_.algorithms.signal
   :members:

Image Processing Algorithms
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: sigima_.algorithms.image
   :members:

Data Type Conversion Algorithms
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: sigima_.algorithms.datatypes
   :members:

Coordinate Conversion Algorithms
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: sigima_.algorithms.coordinates
   :members:

"""
