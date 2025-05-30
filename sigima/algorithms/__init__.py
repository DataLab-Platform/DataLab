"""
Algorithms (:mod:`sigima.algorithms`)
----------------------------------

This package contains the algorithms used by the DataLab project. Those algorithms
operate directly on NumPy arrays and are designed to be used in the DataLab pipeline,
but can be used independently as well.

.. seealso::

    The :mod:`sigima.algorithms` package is the main entry point for the DataLab
    algorithms when manipulating NumPy arrays. See the :mod:`sigima.computation`
    package for algorithms that operate directly on DataLab objects (i.e.
    :class:`cdl.obj.SignalObj` and :class:`cdl.obj.ImageObj`).

The algorithms are organized in subpackages according to their purpose. The following
subpackages are available:

- :mod:`sigima.algorithms.signal`: Signal processing algorithms
- :mod:`sigima.algorithms.image`: Image processing algorithms
- :mod:`sigima.algorithms.datatypes`: Data type conversion algorithms
- :mod:`sigima.algorithms.coordinates`: Coordinate conversion algorithms

Signal Processing Algorithms
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: sigima.algorithms.signal
   :members:

Image Processing Algorithms
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: sigima.algorithms.image
   :members:

Data Type Conversion Algorithms
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: sigima.algorithms.datatypes
   :members:

Coordinate Conversion Algorithms
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: sigima.algorithms.coordinates
   :members:

"""
