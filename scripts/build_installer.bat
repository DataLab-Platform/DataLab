@echo off
REM ======================================================
REM Generic Python Installer build script
REM ------------------------------------------
REM Copyright (c) 2021 CODRA
REM ======================================================
call %~dp0utils GetScriptPath SCRIPTPATH
call %FUNC% GetLibName LIBNAME
call %FUNC% SetPythonPath
call %FUNC% UseWinPython
call %FUNC% GetVersion VERSION
set ROOTPATH=%SCRIPTPATH%\..\
cd %ROOTPATH%

set NSIS_DIST_PATH=%ROOTPATH%\dist\%LIBNAME%
set NSIS_PRODUCT_NAME=%LIBNAME%
set NSIS_PRODUCT_ID=%LIBNAME%
set NSIS_INSTALLDIR=C:\%LIBNAME%
set NSIS_PRODUCT_VERSION=%VERSION%
set NSIS_INSTALLER_VERSION=%VERSION%.0
"C:\Program Files (x86)\NSIS\makensis.exe" nsis\installer.nsi

call %FUNC% EndOfScript