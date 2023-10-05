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
if exist build\doc ( rmdir /s /q build\doc )
sphinx-build -D language=fr -D htmlhelp_basename=%LIBNAME%_fr -b htmlhelp doc build\doc
hhc build\doc\%LIBNAME%_fr.hhp
@REM Update screenshots
set LANG=en
%PYTHON% doc/update_screenshots.py
sphinx-build -D language=en -b htmlhelp doc build\doc
hhc build\doc\%LIBNAME%.hhp
move /y build\doc\*.chm %MODNAME%\data
sphinx-build -b html doc build\doc
call %FUNC% EndOfScript