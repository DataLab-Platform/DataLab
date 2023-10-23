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
set PATH=C:\Program Files\HTML Help Workshop;C:\Program Files (x86)\HTML Help Workshop;%PATH%
@REM Update screenshots
set QT_COLOR_MODE=light
set LANG=fr
%PYTHON% doc/update_screenshots.py
@REM Build documentation
if exist %MODNAME%\data\doc ( rmdir /s /q %MODNAME%\data\doc )
sphinx-build -D language=fr -b singlehtml doc %MODNAME%\data\doc
ren %MODNAME%\data\doc\index.html index_fr.html
@REM Update screenshots
set LANG=en
%PYTHON% doc/update_screenshots.py
sphinx-build -D language=en -b singlehtml doc %MODNAME%\data\doc
@REM explorer %MODNAME%\data\doc
call %FUNC% EndOfScript