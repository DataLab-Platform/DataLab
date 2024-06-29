"""
Algorithms (:mod:`cdl.algorithms`)
----------------------------------

This package contains the algorithms used by the DataLab project. Those algorithms
operate directly on NumPy arrays and are designed to be used in the DataLab pipeline,
but can be used independently as well.

.. seealso::

    The :mod:`cdl.algorithms` package is the main entry point for the DataLab
    algorithms when manipulating NumPy arrays. See the :mod:`cdl.computation`
    package for algorithms that operate directly on DataLab objects (i.e.
    :class:`cdl.obj.SignalObj` and :class:`cdl.obj.ImageObj`).

The algorithms are organized in subpackages according to their purpose. The following
subpackages are available:

- :mod:`cdl.algorithms.signal`: Signal processing algorithms
- :mod:`cdl.algorithms.image`: Image processing algorithms
- :mod:`cdl.algorithms.datatypes`: Data type conversion algorithms
- :mod:`cdl.algorithms.coordinates`: Coordinate conversion algorithms

Signal Processing Algorithms
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: cdl.algorithms.signal
   :members:

Image Processing Algorithms
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: cdl.algorithms.image
   :members:

Data Type Conversion Algorithms
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: cdl.algorithms.datatypes
   :members:

Coordinate Conversion Algorithms
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: cdl.algorithms.coordinates
   :members:

Curve Fitting Algorithms
^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: cdl.algorithms.signal
   :members:
"""
