Roadmap
=======

Future milestones
-----------------

Features
^^^^^^^^

* Add support for timeseries

* Add support for multichannel timeseries

* Develop a Jupyter plugin for interactive data analysis connected with DataLab

* Develop a Spyder plugin for interactive data analysis connected with DataLab

Other tasks
^^^^^^^^^^^

* Develop a very simple DataLab plugin to demonstrate the plugin system

* Develop a DataLab plugin template

* Make a video tutorial about the plugin system and remote control features

Past milestones
---------------

DataLab 1.0
^^^^^^^^^^^

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

CodraFT 2.2
^^^^^^^^^^^

* Add default image visualization settings in .INI configuration file

CodraFT 2.1
^^^^^^^^^^^

* "Open in a new window" feature: add support for multiple separate windows,
  thus allowing to visualize for example two images side by side

* New demo mode

* New command line option features (open/browse HDF5 files at startup)

* ROI features:

  - Add an option to extract multiples ROI on either
    one signal/image (current behavior) or one signal/image per ROI
  - Images: create ROI using array masks
  - Images: add support for circular ROI

CodraFT 2.0
^^^^^^^^^^^

* New data processing and visualization features (see below)

* Fully automated high-level processing features for internal testing purpose,
  as well as embedding DataLab in a third-party software

* Extensive test suite (unit tests and application tests)
  with 90% feature coverage

CodraFT 1.7
^^^^^^^^^^^

* Major redesign

* Python 3.8 is the new reference

* Dropped Python 2 support

CodraFT 1.6
^^^^^^^^^^^

* Last release supporting Python 2
