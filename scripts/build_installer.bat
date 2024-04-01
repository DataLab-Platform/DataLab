@echo off
REM ======================================================
REM WiX Installer build script
REM ------------------------------------------
REM Licensed under the terms of the BSD 3-Clause
REM (see cdl/LICENSE for details)
REM ======================================================
call %~dp0utils GetScriptPath SCRIPTPATH
set ROOTPATH=%SCRIPTPATH%\..

set RSCPATH=%ROOTPATH%\resources
set WIXPATH=%ROOTPATH%\wix

@REM Generating images for WiX installer
set INKSCAPE_PATH="C:\Program Files\Inkscape\bin\inkscape.exe"
%INKSCAPE_PATH% "%RSCPATH%\DataLab-WixUIDialog.svg" -o "temp.png" -w 493 -h 312
magick convert "temp.png" bmp3:"%WIXPATH%\dialog.bmp"
%INKSCAPE_PATH% "%RSCPATH%\DataLab-WixUIBanner.svg" -o "temp.png" -w 493 -h 58
magick convert "temp.png" bmp3:"%WIXPATH%\banner.bmp"
del "temp.png"

@REM Generating .wxs file for WiX installer
call %FUNC% GetLibName LIBNAME
call %FUNC% SetPythonPath
call %FUNC% UsePython
call %FUNC% GetVersion VERSION
%PYTHON% "%WIXPATH%\makewxs.py" %LIBNAME% %VERSION%

@REM Building WiX Installer
wix build "%WIXPATH%\%LIBNAME%-%VERSION%.wxs" -ext WixToolset.UI.wixext

call %FUNC% EndOfScript