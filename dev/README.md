Setting up DataLab development environment
==========================================

Python distribution
-------------------

DataLab requires the following :

* Python (e.g. WinPython)

* Additional Python packages

Installing all required packages :

    pip install --upgrade -r dev\requirements.txt

ℹ️ See [Installation](https://cdlapp.readthedocs.io/en/latest/intro/installation.html)
for more details on reference Python and Qt versions.

Test data
---------

DataLab test data are located in different folders, depending on their nature or origin.

Required data for unit tests are located in "cdl\data\tests" (public data).

A second folder %CDL_DATA% (optional) may be defined for additional tests which are
still under development (or for confidential data).

Specific environment variables
------------------------------

Enable the "debug" mode (no stdin/stdout redirection towards internal console) :

    @REM Mode DEBUG
    set DEBUG=1

Adding support for mathematical equations in documentation:

    @REM LaTeX executable must be in Windows PATH, for mathematical equations rendering
    @REM Example with MiKTeX :
    set PATH=C:\\Apps\\miktex-portable\\texmfs\\install\\miktex\\bin\\x64;%PATH%

Visual Studio Code configuration used in `launch.json` and `tasks.json`
(examples) :

    @REM Development environment
    set CDL_PYTHONEXE=C:\C2OIQ-DevCDL\python-3.8.10.amd64\python.exe
    @REM Folder containing additional working test data
    set CDL_DATA=C:\Dev\Projets\CDL_data

Other requirements
------------------

The following applications are required to build the executable and the installer:

* NSIS (Nullsoft Scriptable Install System)
* InkScape (for SVG to PNG conversion)
* ImageMagick (for PNG to BMP conversion, and for icon creation)
