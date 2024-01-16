@echo off
REM This script was copied from PythonQwt project
REM ======================================================
REM Build GitHub Pages documentation
REM ======================================================
REM Licensed under the terms of the MIT License
REM Copyright (c) 2020 Pierre Raybaut
REM (see PythonQwt LICENSE file for more details)
REM ======================================================

setlocal enabledelayedexpansion

@REM Get the target path for GitHub Pages from `CDL_GHPAGES` environment variable:
@REM if this variable is not defined, interrupt the script and show an error message
if not defined CDL_GHPAGES (
    echo ERROR: CDL_GHPAGES environment variable is not defined.
    echo Please define it to the path of the local clone of the GitHub Pages repository.
    echo For instance:
    echo     set CDL_GHPAGES=C:\Dev\DataLab_GitHub_Pages
    echo.
    echo Then, run this script again.
    exit /b 1
)

call %~dp0utils GetScriptPath SCRIPTPATH
call %FUNC% GetLibName LIBNAME
call %FUNC% GetModName MODNAME
call %FUNC% SetPythonPath
call %FUNC% UsePython
call %FUNC% GetVersion CDL_VERSION
cd %SCRIPTPATH%\..
%PYTHON% doc\update_requirements.py

set QT_COLOR_MODE=light

@REM Build documentation ===============================================================
for %%L in (fr en) do (
    set LANG=%%L
    set TARGET=%CDL_GHPAGES%\%%L
    %PYTHON% doc/update_screenshots.py
    if exist !TARGET! ( rmdir /s /q !TARGET! )
    sphinx-build -b html -D language=%%L doc !TARGET!
)

call %FUNC% EndOfScript