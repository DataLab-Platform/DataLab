@echo off
REM This script was derived from PythonQwt project
REM ======================================================
REM Run coverage code analysis tool
REM ======================================================
REM Licensed under the terms of the MIT License
REM Copyright (c) 2020 Pierre Raybaut
REM (see PythonQwt LICENSE file for more details)
REM ======================================================
setlocal
call %~dp0utils GetScriptPath SCRIPTPATH
call %FUNC% GetLibName LIBNAME
call %FUNC% GetModName MODNAME
call %FUNC% SetPythonPath
call %FUNC% UsePython
if exist sitecustomize.py ( del /q sitecustomize.py )
echo import coverage> sitecustomize.py
echo coverage.process_startup()>> sitecustomize.py
set COVERAGE_PROCESS_START=%SCRIPTPATH%\..\.coveragerc
coverage run -m cdl.tests.all_tests %* --timeout 600
@REM coverage report -m
coverage combine
coverage html
start .\htmlcov\index.html
if exist sitecustomize.py ( del /q sitecustomize.py )
call %FUNC% EndOfScript