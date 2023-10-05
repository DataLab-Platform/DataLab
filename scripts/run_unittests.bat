@echo off
REM This script was copied from PythonQwt project
REM ======================================================
REM Unattended test script
REM ======================================================
REM Licensed under the terms of the MIT License
REM Copyright (c) 2020 Pierre Raybaut
REM (see PythonQwt LICENSE file for more details)
REM ======================================================
setlocal
call %~dp0utils GetScriptPath SCRIPTPATH
call %FUNC% GetModName MODNAME
call %FUNC% SetPythonPath
call %FUNC% UsePython
python -m %MODNAME%.tests.all_tests
call %FUNC% EndOfScript