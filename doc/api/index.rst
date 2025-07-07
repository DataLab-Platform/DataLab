.. _api:

API
===

The public Application Programming Interface (API) of DataLab offers a set of
functions to access the DataLab features. Those functions are available in various
submodules of the `datalab` package. The following table lists the available submodules
and their purpose:

.. list-table::
    :header-rows: 1
    :align: left

    * - Submodule
      - Purpose

    * - :mod:`sigima.tools`
      - Algorithms for data analysis, which operates on NumPy arrays

    * - :mod:`sigima.params`
      - Convenience module to access the DataLab sets of parameters (instances of :class:`guidata.dataset.DataSet` objects)

    * - :mod:`sigima.objects`
      - Convenience module to access the DataLab objects (:class:`sigima.objects.SignalObj` or :class:`sigima.objects.ImageObj`) and related functions

    * - :mod:`sigima.proc`
      - Computation functions, which operate on DataLab objects (:class:`sigima.objects.SignalObj` or :class:`sigima.objects.ImageObj`)

    * - :mod:`datalab.proxy`
      - Proxy objects to access the DataLab interface from a Python script or a remote application

.. toctree::
   :maxdepth: 2
   :caption: Public features:

   tools
   params
   objects
   proc
   proxy

.. toctree::
   :maxdepth: 1
   :caption: Internal features:

   gui/index
   gui/main
   gui/panel
   gui/actionhandler
   gui/objectview
   gui/plothandler
   gui/roieditor
   gui/processor
   gui/docks
   gui/h5io
