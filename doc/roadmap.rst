Roadmap
=======

Future milestones
-----------------

CodraFT 2.1
^^^^^^^^^^^

* Run computations in a separate process:

  - Execute a "computing server" in background, in another process
  - Define a communication protocol between this process and
    CodraFT GUI process based on TCP sockets
  - For each computation, send pickled data and computing function
    to the server and wait for the result
  - It will then possible to stop any computation at any time by killing the
    server process and restarting it (eventually after incrementing the
    communication port number)

* "Open in a new window" feature: add support for multiple separate windows,
  thus allowing to visualize for example two images side by side

* ROI features:

  - Add an option to extract multiples ROI on either
    one signal/image (current behavior) or one signal/image per ROI
  - Images: create ROI using array masks
  - Images: add support for circular ROI

* Optimize image displaying performance

Past milestones
---------------

CodraFT 2.0
^^^^^^^^^^^

* New data processing and visualization features (see below)

* Fully automated high-level processing features for internal testing purpose,
  as well as embedding CodraFT in a third-party software

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
