Setting up CobraDataLab development environment
==========================================

Python distribution
-------------------

CobraDataLab requires the following :

* Python 3.8.10 (e.g. WinPython)

* Additional Python packages

Installing all required packages :

    pip install --upgrade -r dev\requirements.txt

Test data
---------

CobraDataLab test data are located in different folders, depending on their nature or origin.

Required data for unit tests are located in "cdl\data\tests" (public data).

A second folder %DATA_CDL% (optional) may be defined for additional tests which are
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

    @REM Development PYTHONPATH (needed if new required features have been added in
    @REM development version of dependencies like guidata and guiqwt, for example)
    set PYTHONPATH_CDL=C:\dev\libre\guidata;C:\dev\libre\guiqwt
    @REM Development environment
    set PYTHON_CDL_DEV=C:\C2OIQ-DevCDL\python-3.8.10.amd64\python.exe
    @REM Folder containing additional working test data
    set DATA_CDL=C:\Dev\Projets\CDL_data
    @REM Release environment
    set PYTHON_CDL_RLS=C:\C2OIQ\python-3.8.10.amd64\python.exe

Other requirements
------------------

The following applications are required to build the executable and the installer:

* NSIS (Nullsoft Scriptable Install System)
* InkScape (for SVG to PNG conversion)
* ImageMagick (for PNG to BMP conversion, and for icon creation)
