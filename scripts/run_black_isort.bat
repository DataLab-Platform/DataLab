@echo off
REM This script was derived from PythonQwt project
REM ======================================================
REM Run black and isort code analysis tool
REM ======================================================
REM Licensed under the terms of the MIT License
REM Copyright (c) 2020 Pierre Raybaut
REM (see PythonQwt LICENSE file for more details)
REM ======================================================
setlocal
call %~dp0utils GetScriptPath SCRIPTPATH
call %FUNC% GetLibName LIBNAME
call %FUNC% SetPythonPath
set PYTHON=%CDL_PYTHONEXE%
call %FUNC% UsePython
python -m black .
python -m isort --profile black .
call %FUNC% EndOfScript