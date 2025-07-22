@echo off
REM This script was copied from PythonQwt project
REM ======================================================
REM Update statically generated documentation resources
REM ======================================================
REM Licensed under the terms of the MIT License
REM Copyright (c) 2020 Pierre Raybaut
REM (see PythonQwt LICENSE file for more details)
REM ======================================================
call %~dp0utils GetScriptPath SCRIPTPATH
call %FUNC% SetPythonPath
call %FUNC% UsePython
cd %SCRIPTPATH%\..
%PYTHON% -m guidata.utils.genreqs all
%PYTHON% doc\update_validation_status.py
%PYTHON% doc\update_processor_methods.py
