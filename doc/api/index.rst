.. _api:

API
===

The public Application Programming Interface (API) of DataLab offers a set of functions to access the DataLab features. Those functions are available in various submodules of the `sigima` and `datalab` packages. The `sigima` package is the computation engine of DataLab, while the `datalab` package provides the interface to access the application features. The following table lists the available submodules and their purpose:

.. list-table::
    :header-rows: 1
    :align: left

    * - Submodule
      - Purpose

    * - :mod:`sigima.tools`
      - Algorithms for data analysis (operating on NumPy arrays) which purpose is to fill in the gaps of common scientific libraries (NumPy, SciPy, scikit-image, etc.), offering consistent tools for computation functions (see :mod:`sigima.proc`)

    * - :mod:`sigima.params`
      - Sets of parameters for configuring computation functions (these parameters are instances of :class:`guidata.dataset.DataSet` objects)

    * - :mod:`sigima.objects`
      - Object model for signals and images (:class:`sigima.objects.SignalObj` and :class:`sigima.objects.ImageObj`) and related functions

    * - :mod:`sigima.proc`
      - Computation functions, which operate on signal and image objects (:class:`sigima.objects.SignalObj` or :class:`sigima.objects.ImageObj`)

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
