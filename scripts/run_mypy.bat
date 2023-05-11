@echo off
REM This script was derived from PythonQwt project
REM ======================================================
REM Run mypy code analysis tool
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
call %FUNC% UseWinPython
mypy --strict --ignore-missing-imports --allow-subclassing-any %MODNAME%
call %FUNC% EndOfScript