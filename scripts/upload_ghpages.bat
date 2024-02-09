@echo off
REM This script was copied from PythonQwt project
REM ======================================================
REM Upload GitHub Pages documentation
REM ======================================================
REM Licensed under the terms of the MIT License
REM Copyright (c) 2020 Pierre Raybaut
REM (see PythonQwt LICENSE file for more details)
REM ======================================================

@REM Get the target path for GitHub Pages from `CDL_GHPAGES` environment variable:
@REM if this variable is not defined, interrupt the script and show an error message
if not defined CDL_GHPAGES (
    echo ERROR: CDL_GHPAGES environment variable is not defined.
    echo Please define it to the path of the local clone of the GitHub Pages repository.
    echo For instance:
    echo     set CDL_GHPAGES=C:\Dev\DataLab-Platform.github.io
    echo.
    echo Then, run this script again.
    exit /b 1
)

call %~dp0utils GetScriptPath SCRIPTPATH

pushd %CDL_GHPAGES%
git checkout main
git add .
git commit -m "Update documentation"
git push origin main
popd

call %FUNC% EndOfScript