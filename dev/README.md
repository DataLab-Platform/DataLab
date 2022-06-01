Setting up CodraFT development environment
==========================================

Python distribution
-------------------

CodraFT requires the following :

* Python 3.8.10 (e.g. WinPython)

* Additional Python packages

Installing all required packages :

    pip install --upgrade -r dev\requirements.txt

Test data
---------

CodraFT test data are located in different folders, depending on their nature or origin.

Required data for unit tests are located in "codraft\data\tests" (public data).

A second folder %DATA_CODRAFT% (optional) may be defined for additional tests which are
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
    set PYTHONPATH_CODRAFT=C:\dev\libre\guidata;C:\dev\libre\guiqwt
    @REM Development environment
    set PYTHON_CODRAFT_DEV=C:\C2OIQ-DevCodraFT\python-3.8.10.amd64\python.exe
    @REM Folder containing additional working test data
    set DATA_CODRAFT=C:\Dev\Projets\CodraFT_data
    @REM Release environment
    set PYTHON_CODRAFT_RLS=C:\C2OIQ\python-3.8.10.amd64\python.exe
