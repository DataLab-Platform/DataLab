Roadmap
=======

Future milestones
-----------------

Features
^^^^^^^^

* Add a Jupyter kernel interface to DataLab:

  - This would allow to use DataLab from other software, such as Jupyter
    notebooks, Spyder or Visual Studio Code
  - This would also allow to share data between DataLab and other software
    (e.g. display DataLab numerical results in Jupyter notebooks or the other
    way around, display Jupyter results in DataLab, etc.)

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

* Make tutorial videos: plugin system, remote control features, etc. (see
  `Issue #25 <https://github.com/DataLab-Platform/DataLab/issues/25>`_)

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
