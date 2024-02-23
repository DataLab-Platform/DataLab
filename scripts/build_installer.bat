@echo off
REM ======================================================
REM Generic Python Installer build script
REM ------------------------------------------
REM Licensed under the terms of the BSD 3-Clause
REM (see cdl/LICENSE for details)
REM ======================================================
call %~dp0utils GetScriptPath SCRIPTPATH
set ROOTPATH=%SCRIPTPATH%\..

set RSCPATH=%ROOTPATH%\resources
set NSISPATH=%ROOTPATH%\nsis

@REM Generating images for NSIS installer
set INKSCAPE_PATH="C:\Program Files\Inkscape\bin\inkscape.exe"
%INKSCAPE_PATH% "%RSCPATH%\win.svg" -o "%NSISPATH%\temp.png" -w 164 -h 314
magick convert "%NSISPATH%\temp.png" bmp3:"%NSISPATH%\win.bmp"
%INKSCAPE_PATH% "%RSCPATH%\banner.svg" -o "%NSISPATH%\temp.png" -w 150 -h 57
magick convert "%NSISPATH%\temp.png" bmp3:"%NSISPATH%\banner.bmp"
del "%NSISPATH%\temp.png"

@REM Generating icons for NSIS installer
for %%s in (16 24 32 48 128 256) do (
  %INKSCAPE_PATH% "%RSCPATH%\install.svg" -o "%NSISPATH%\install-%%s.png" -w %%s -h %%s
  %INKSCAPE_PATH% "%RSCPATH%\uninstall.svg" -o "%NSISPATH%\uninstall-%%s.png" -w %%s -h %%s
)
magick convert "%NSISPATH%\install-*.png" "%NSISPATH%\install.ico"
magick convert "%NSISPATH%\uninstall-*.png" "%NSISPATH%\uninstall.ico"
del "%NSISPATH%\install-*.png"
del "%NSISPATH%\uninstall-*.png"

call %FUNC% GetLibName LIBNAME
call %FUNC% SetPythonPath
call %FUNC% UsePython
call %FUNC% GetVersion VERSION
call %FUNC% GetVersionWithoutAlphaBeta VERSION_WITHOUT_ALPHABETA

@REM Building NSIS installer
set NSIS_DIST_PATH=%ROOTPATH%\dist\%LIBNAME%
set NSIS_PRODUCT_NAME=%LIBNAME%
set NSIS_PRODUCT_ID=%LIBNAME%
set NSIS_INSTALLDIR=C:\%LIBNAME%
set NSIS_PRODUCT_VERSION=%VERSION%
set NSIS_INSTALLER_VERSION=%VERSION_WITHOUT_ALPHABETA%.0
"C:\Program Files (x86)\NSIS\makensis.exe" nsis\installer.nsi

call %FUNC% EndOfScript