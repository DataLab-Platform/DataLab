@echo off
REM ======================================================
REM WiX Installer build script
REM ------------------------------------------
REM Licensed under the terms of the BSD 3-Clause
REM (see datalab/LICENSE for details)
REM ======================================================
call %~dp0utils GetScriptPath SCRIPTPATH
call %FUNC% GetLibName LIBNAME
call %FUNC% SetPythonPath
call %FUNC% UsePython
call %FUNC% GetVersion VERSION

set ROOTPATH=%SCRIPTPATH%\..
set RSCPATH=%ROOTPATH%\resources
set WIXPATH=%ROOTPATH%\wix

REM Note: WiX UI bitmaps (wix\dialog.bmp, wix\banner.bmp) are committed under
REM git and regenerated on demand via `scripts\build_resources.bat` when the
REM source SVG files change. The release pipeline therefore does not require
REM Inkscape or ImageMagick.

echo Generating .wxs file for MSI installer...
%PYTHON% "%WIXPATH%\makewxs.py" DataLab %VERSION%

echo Building MSI installer...
wix build "%WIXPATH%\DataLab-%VERSION%.wxs" -ext WixToolset.UI.wixext

call %FUNC% EndOfScript