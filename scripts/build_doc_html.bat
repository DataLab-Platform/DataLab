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
call %FUNC% SetPythonPath
call %FUNC% UsePython
cd %SCRIPTPATH%\..

@REM Build documentation ===============================================================
set LANG=fr
if exist build\doc ( rmdir /s /q build\doc )
sphinx-build -b html -D language=%LANG% doc build\doc
start build\doc\index.html

call %FUNC% EndOfScript