"""
Computation (:mod:`cdl.computation`)
-----------------------------------------

This package contains the computation functions used by the DataLab project.
Those functions operate directly on DataLab objects (i.e. :class:`cdl.obj.SignalObj`
and :class:`cdl.obj.ImageObj`) and are designed to be used in the DataLab pipeline,
but can be used independently as well.

.. seealso::

    The :mod:`cdl.computation` package is the main entry point for the DataLab
    computation functions when manipulating DataLab objects.
    See the :mod:`cdl.algorithms` package for algorithms that operate directly on
    NumPy arrays.

Each computation module defines a set of computation objects, that is, functions
that implement processing features and classes that implement the corresponding
parameters (in the form of :py:class:`guidata.dataset.datatypes.Dataset` subclasses).
The computation functions takes a DataLab object (e.g. :class:`cdl.obj.SignalObj`)
and a parameter object (e.g. :py:class:`cdl.param.MovingAverageParam`) as input
and return a DataLab object as output (the result of the computation). The parameter
object is used to configure the computation function (e.g. the size of the moving
average window).

In DataLab overall architecture, the purpose of this package is to provide the
computation functions that are used by the :mod:`cdl.core.gui.processor` module,
based on the algorithms defined in the :mod:`cdl.algorithms` module and on the
data model defined in the :mod:`cdl.obj` (or :mod:`cdl.core.model`) module.

The computation modules are organized in subpackages according to their purpose.
The following subpackages are available:

- :mod:`cdl.computation.base`: Common processing features
- :mod:`cdl.computation.signal`: Signal processing features
- :mod:`cdl.computation.image`: Image processing features

Common processing features
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: cdl.computation.base
   :members:

Signal processing features
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: cdl.computation.signal
   :members:

Image processing features
^^^^^^^^^^^^^^^^^^^^^^^^^

Base image processing features
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: cdl.computation.image
   :members:

Exposure correction features
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: cdl.computation.image.exposure
    :members:

Restoration features
~~~~~~~~~~~~~~~~~~~~

.. automodule:: cdl.computation.image.restoration
    :members:

Morphological features
~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: cdl.computation.image.morphology
    :members:

Edge detection features
~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: cdl.computation.image.edges

Detection features
~~~~~~~~~~~~~~~~~~

.. automodule:: cdl.computation.image.detection
    :members:
"""
