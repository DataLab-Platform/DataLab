.. _ref-to-model:

Internal data model
===================

In its internal data model, DataLab stores data using two main classes:

* `cdlapp.core.model.signal.SignalObj`, which represents a signal object, and
* `cdlapp.core.model.image.ImageObj`, which represents an image object.

These classes are defined in the `cdlapp.core.model` package but are exposed
publicly in the `cdlapp.obj` package.

Also, DataLab uses many different datasets (based on guidata's `DataSet` class)
to store the parameters of the computations. These datasets are defined in
different modules but are exposed publicly in the `cdlapp.param` package.

Public API
^^^^^^^^^^

The public API is the following:

.. automodule:: cdlapp.obj
    :members:
    :show-inheritance:

.. automodule:: cdlapp.param
    :members:
    :show-inheritance:

.. automodule:: cdlapp.core.model.signal
    :members:
    :show-inheritance:

.. automodule:: cdlapp.core.model.image
    :members:
    :show-inheritance:
