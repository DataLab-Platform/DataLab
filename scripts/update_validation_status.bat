@echo off
REM This script was copied from PythonQwt project
REM ======================================================
REM Update validation status
REM ======================================================
REM Licensed under the terms of the MIT License
REM Copyright (c) 2020 Pierre Raybaut
REM (see PythonQwt LICENSE file for more details)
REM ======================================================
call %~dp0utils GetScriptPath SCRIPTPATH
call %FUNC% SetPythonPath
call %FUNC% UsePython
cd %SCRIPTPATH%\..
%PYTHON% doc\update_validation_status.py