@echo off
REM This script was copied from PythonQwt project
REM ======================================================
REM Preview GitHub Pages documentation
REM ======================================================
REM Licensed under the terms of the MIT License
REM Copyright (c) 2020 Pierre Raybaut
REM (see PythonQwt LICENSE file for more details)
REM ======================================================

call %~dp0utils GetScriptPath SCRIPTPATH
call %FUNC% SetPythonPath
call %FUNC% UsePython

start http://localhost:8000

cd %CDL_GHPAGES%
%PYTHON% -m http.server

call %FUNC% EndOfScript