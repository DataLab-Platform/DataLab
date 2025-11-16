@echo off
REM ======================================================
REM Generic Python Installer build script
REM ------------------------------------------
REM Licensed under the terms of the BSD 3-Clause
REM (see datalab/LICENSE for details)
REM ======================================================
call %~dp0utils GetScriptPath SCRIPTPATH
call %FUNC% GetLibName LIBNAME
call %FUNC% SetPythonPath
call %FUNC% UsePython
call %FUNC% GetVersion VERSION
call %FUNC% GetModName MODNAME
call %FUNC% GetDLProjectPath DATALAB_MODULE_PATH

echo Modname=%MODNAME%
echo LibName=%LIBNAME%
echo LibVersion=%VERSION%
echo DLModulePath=%DATALAB_MODULE_PATH%
echo ===========================================================================
echo Python=%PYTHON%
echo WINPYDIRBASE=%WINPYDIRBASE%
echo PYTHONPATH=%PYTHONPATH%
echo ===========================================================================

call %FUNC% EndOfScript