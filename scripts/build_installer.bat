@echo off
REM ======================================================
REM Generic Python Installer build script
REM ------------------------------------------
REM Copyright (c) 2021-2023 Codra
REM ======================================================
call %~dp0utils GetScriptPath SCRIPTPATH
set ROOTPATH=%SCRIPTPATH%\..

set RSCPATH=%ROOTPATH%\resources

@REM Generating images for NSIS installer
set BMPPATH=%ROOTPATH%\nsis\images
set INKSCAPE_PATH="C:\Program Files\Inkscape\bin\inkscape.exe"
%INKSCAPE_PATH% "%RSCPATH%\win.svg" -o "%BMPPATH%\temp.png" -w 164 -h 314
magick convert "%BMPPATH%\temp.png" bmp3:"%BMPPATH%\win.bmp"
%INKSCAPE_PATH% "%RSCPATH%\banner.svg" -o "%BMPPATH%\temp.png" -w 150 -h 57
magick convert "%BMPPATH%\temp.png" bmp3:"%BMPPATH%\banner.bmp"
del "%BMPPATH%\temp.png"

@REM Generating icons for NSIS installer
set ICOPATH=%ROOTPATH%\nsis\icons
for %%s in (16 24 32 48 128 256) do (
  %INKSCAPE_PATH% "%RSCPATH%\install.svg" -o "%ICOPATH%\install-%%s.png" -w %%s -h %%s
  %INKSCAPE_PATH% "%RSCPATH%\uninstall.svg" -o "%ICOPATH%\uninstall-%%s.png" -w %%s -h %%s
)
magick convert "%ICOPATH%\install-*.png" "%ICOPATH%\install.ico"
magick convert "%ICOPATH%\uninstall-*.png" "%ICOPATH%\uninstall.ico"
del "%ICOPATH%\install-*.png"
del "%ICOPATH%\uninstall-*.png"

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