Roadmap
=======

Future milestones
-----------------

Features or enhancements
^^^^^^^^^^^^^^^^^^^^^^^^

* Add support for data acquisition:

  - It would be nice to be able to acquire data from various sources
    (e.g. a camera, a digitizer, a spectrometer, etc.) directly from DataLab
  - This would allow to use DataLab as a data acquisition software, and to
    process the acquired data immediately after
  - Although there is currently no design for this feature, it could be
    implemented by creating a new plugin family, and by defining a common
    API for data acquisition plugins
  - One of the possible technical solutions could be to rely on `PyMoDAQ <https://pymodaq.cnrs.fr/>`_,
    a Python package for data acquisition, which is already compatible with
    various hardware devices - *how about a collaboration with the PyMoDAQ
    developers?*

* Create a DataLab math library:

  - This library would be a Python package, and would contain all the
    mathematical functions and algorithms used in DataLab:
    - A low-level algorithms API operating on NumPy arrays
    - The base non-GUI data model of DataLab (e.g. signals, images)
    - A high-level computing API operating on DataLab objects (e.g. signals, images)
  - It would be used by DataLab itself, but could also be used by third-party software
    (e.g. Jupyter notebooks, Spyder, Visual Studio Code, etc.)
  - Finally, this library would be a good way to share DataLab's mathematical features
    with the scientific community: a collection of algorithms and functions
    that are well-tested, well-documented, and easy to use
  - *Note*: it is already possible to use DataLab's processing features from outside
    DataLab by importing the `cdl` Python package, but this package also contains
    the GUI code, which is not always needed (e.g. when using DataLab from a Jupyter
    notebook). The idea here is to create a new package that would contain only the
    mathematical features of DataLab, without the GUI code.

.. note:: The DataLab math library could be an opportunity to reconsider the design
    of the DataLab processing functions. Currently, the processing functions working
    on signal and image objects rely on `guidata.dataset.DataSet` objects for input
    parameters. This is very convenient for the developer because it allows to create
    a GUI for the processing functions automatically, but it is not very flexible for
    the user because it forces to instantiate a `DataSet` object with the right
    parameters before calling the processing function (this can be cumbersome especially
    when dealing with simple processing functions requiring only a few parameters).
    Thus, it could be interesting to consider a more flexible and simple design, where
    the processing parameters would be passed as keyword arguments to the processing
    functions. The `DataSet` objects could be handled internally by the processing
    functions (e.g. by calling the `DataSet.create` method with the keyword arguments
    passed by the user). This would allow to keep the automatic GUI generation feature
    for the processing functions, but would also allow to call the processing functions
    directly with keyword arguments, without having to create a `DataSet` object first.

* Create a Jupyter plugin for interactive data analysis with DataLab:

  - Using DataLab from a Jupyter notebook is already possible, thanks to the
    remote control features (see :ref:`ref-to-remote-control`), but it would
    be nice to have a dedicated plugin
  - This plugin would allow to use DataLab as a Jupyter kernel, and to
    display DataLab numerical results in Jupyter notebooks or the other way
    around (e.g. display Jupyter results in DataLab)
  - This plugin would also allow to use DataLab's processing features from
    Jupyter notebooks
  - A typical use case could also consist in using DataLab for manipulating
    signals or images efficiently, and using Jupyter for custom data analysis
    based on specific / home-made algorithms
  - This plugin could be implemented by using the Jupyter kernel interface
    (see above)

* Create a Spyder plugin for interactive data analysis connected with DataLab:

  - This is exactly the same use case as for the Jupyter plugin, but for
    Spyder
  - This plugin could also be implemented by using the Jupyter kernel interface
    (see above)

* Add support for time series (see
  `Issue #27 <https://github.com/DataLab-Platform/DataLab/issues/27>`_)

* Add a Jupyter kernel interface to DataLab:

  - This would allow to use DataLab from other software, such as Jupyter
    notebooks, Spyder or Visual Studio Code
  - This would also allow to share data between DataLab and other software
    (e.g. display DataLab numerical results in Jupyter notebooks or the other
    way around, display Jupyter results in DataLab, etc.)
  - After a first and quick look, it seems that the Jupyter kernel interface
    is not straightforward to implement, so that it may not be worth the effort
    (the communication between DataLab and Jupyter is currently already possible
    thanks to the remote control features)

Maintenance
^^^^^^^^^^^

* 2024: switch to gRPC for remote control (instead of XML-RPC), if there is a
  need for a more efficient communication protocol (see
  `Issue #18 <https://github.com/DataLab-Platform/DataLab/issues/18>`_)

* 2025: drop PyQt5 support (end-of-life: mid-2025), and switch to PyQt6 ;
  this should be straightforward, thanks to the `qtpy` compatibility layer
  and to the fact that `PlotPyStack` is already compatible with PyQt6)

Other tasks
^^^^^^^^^^^

* Create a DataLab plugin template (see
  `Issue #26 <https://github.com/DataLab-Platform/DataLab/issues/26>`_)

Past milestones
---------------

DataLab 0.11
^^^^^^^^^^^^

* Add a drag-and-drop feature to the signal and image panels, to allow reordering
  signals and images (see
  `Issue #17 <https://github.com/DataLab-Platform/DataLab/issues/17>`_)

* Add "Move up" and "Move down" buttons to the signal and image panels, to allow
  reordering signals and images (see
  `Issue #22 <https://github.com/DataLab-Platform/DataLab/issues/22>`_)

* Add 1D convolution, interpolation, resampling and detrending features

DataLab 0.10
^^^^^^^^^^^^

* Develop a very simple DataLab plugin to demonstrate the plugin system

* Serialize curve and image styles in HDF5 files

* Add an "Auto-refresh" global option, to be able to disable the automatic
  refresh of the main window when doing multiple processing steps, thus
  improving performance

* Improve curve readability (e.g. avoid dashed lines, use contrasted colors,
  and use anti-aliasing)

DataLab 0.9
^^^^^^^^^^^

* Python 3.11 is the new reference

* Run computations in a separate process:

  - Execute a "computing server" in background, in another process
  - For each computation, send serialized data and computing function
    to the server and wait for the result
  - It is then possible to stop any computation at any time by killing the
    server process and restarting it (eventually after incrementing the
    communication port number)

* Optimize image displaying performance

* Add preferences dialog box

* Add new image processing features: denoising, ...

* Image processing results: added support for polygon shapes (e.g. for
  contour detection)

* New plugin system: API for third-party extensions

   - Objective #1: a plugin must be manageable using a single Python script,
     which includes an extension of `ImageProcessor`, `ActionHandler`
     and new file format support
   - Objective #2: plugins must be simply stored in a folder wich defaults
     to the user directory (same folder as ".DataLab.ini" configuration
     file)

* Add a macro-command system:

  - New embedded Python editor
  - Scripts using the same API as high-level applicative test scenarios
  - Support for macro recording

* Add an xmlrpc server to allow DataLab remote control:

  - Controlling DataLab main features (open a signal or an image,
    open a HDF5 file, etc.) and processing features
    (run a computation, etc.)
  - Take control of DataLab from a third-party software
  - Run interactive calculations from an IDE
    (e.g. Spyder or Visual Studio Code)
