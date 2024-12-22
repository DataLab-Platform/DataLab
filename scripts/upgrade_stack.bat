@echo off
REM This script was derived from PythonQwt project
REM ======================================================
REM Update only PlotPyStack required packages
REM ======================================================
REM Licensed under the terms of the MIT License
REM Copyright (c) 2020 Pierre Raybaut
REM (see PythonQwt LICENSE file for more details)
REM ======================================================
setlocal
call %~dp0utils GetScriptPath SCRIPTPATH
call %FUNC% UsePython
cd %SCRIPTPATH%\..
python.exe -m pip install --upgrade pip
pip install --upgrade PythonQwt guidata PlotPy
call %FUNC% EndOfScript