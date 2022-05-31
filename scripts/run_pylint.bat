@echo off
REM This script was derived from PythonQwt project
REM ======================================================
REM Test launcher script
REM ======================================================
REM Licensed under the terms of the MIT License
REM Copyright (c) 2020 Pierre Raybaut
REM (see PythonQwt LICENSE file for more details)
REM ======================================================
setlocal
call %~dp0utils GetScriptPath SCRIPTPATH
call %FUNC% GetLibName LIBNAME
call %FUNC% GetModName MODNAME
call %FUNC% SetPythonPath
call %FUNC% UseWinPython
set PYLINT_ARG=%*
if "%PYLINT_ARG%"=="" set PYLINT_ARG=%MODNAME% --disable=fixme
python -m pylint --rcfile=%RCFILE% %PYLINT_ARG%
call %FUNC% EndOfScript