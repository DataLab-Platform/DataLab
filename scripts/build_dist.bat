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
call %FUNC% UseWinPython
if exist MANIFEST ( del /q MANIFEST )
python create_dephash.py
python setup.py sdist bdist_wheel
move /y %MODNAME%\data\*.chm .
python setup.py build sdist
move /y .\*.chm %MODNAME%\data
rmdir /s /q %LIBNAME%.egg-info
call %FUNC% EndOfScript