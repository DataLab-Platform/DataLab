.. _ref-to-model:

Internal data model
===================

.. meta::
    :description: Internal model of DataLab, the open-source scientific data analysis and visualisation platform
    :keywords: DataLab, internal model, data model, signal, image, dataset, parameter, computation, scientific data analysis, visualisation, platform

In its internal data model, DataLab stores data using two main classes:

* :class:`sigima.obj.SignalObj`, which represents a signal object, and
* :class:`sigima.obj.ImageObj`, which represents an image object.

These classes are defined in the :mod:`sigima.obj` package.

Also, DataLab uses many different datasets (based on guidata's ``DataSet`` class)
to store the parameters of the computations. These datasets are defined in
different modules but are exposed publicly in the :mod:`sigima.params` package.

.. seealso::

    The :ref:`api` section for more information on the public API.
