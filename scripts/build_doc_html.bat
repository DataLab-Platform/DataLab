@echo off
REM This script was copied from PythonQwt project
REM ======================================================
REM Documentation build script
REM ======================================================
REM Licensed under the terms of the MIT License
REM Copyright (c) 2020 Pierre Raybaut
REM (see PythonQwt LICENSE file for more details)
REM ======================================================
call %~dp0utils GetScriptPath SCRIPTPATH
call %FUNC% GetLibName LIBNAME
call %FUNC% GetModName MODNAME
call %FUNC% SetPythonPath
call %FUNC% UsePython
call %FUNC% GetVersion CDL_VERSION
cd %SCRIPTPATH%\..
%PYTHON% doc\update_requirements.py

@REM Set light mode for Qt applications and clean previous documentation ===============
set QT_COLOR_MODE=light
if exist %MODNAME%\data\doc ( rmdir /s /q %MODNAME%\data\doc )
mkdir %MODNAME%\data\doc

@REM Build documentation ===============================================================
set LANG=fr
if exist build\doc ( rmdir /s /q build\doc )
sphinx-build -b html -D language=fr doc build\doc
start build\doc\index.html

call %FUNC% EndOfScript