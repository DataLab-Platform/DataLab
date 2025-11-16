@echo off
REM This script was copied from PythonQwt project
REM ======================================================
REM Package build script
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
call %FUNC% GetVersion DATALAB_VERSION

set REPODIR=%SCRIPTPATH%\..

@REM Clone repository in a temporary directory
set CLONEDIR=%REPODIR%\..\%LIBNAME%-tempdir
if exist %CLONEDIR% ( rmdir /s /q %CLONEDIR% )
git clone -l -s . %CLONEDIR%

pushd %CLONEDIR%
@REM Build source package
%PYTHON% -m build --sdist
@REM Build wheel package with PDF documentation embedded
mkdir %MODNAME%\data\doc
copy %REPODIR%\%MODNAME%\data\doc\*.pdf %MODNAME%\data\doc
%PYTHON% -m build --wheel
popd

if not exist %REPODIR%\dist ( mkdir %REPODIR%\dist )
copy %CLONEDIR%\dist\*.whl %REPODIR%\dist
copy %CLONEDIR%\dist\*.tar.gz %REPODIR%\dist

rmdir /s /q %CLONEDIR%
call %FUNC% EndOfScript