@echo off
REM This script was derived from PythonQwt project
REM ======================================================
REM Run gettext translation tool
REM ======================================================
REM Licensed under the terms of the MIT License
REM Copyright (c) 2020 Pierre Raybaut
REM (see PythonQwt LICENSE file for more details)
REM ======================================================
setlocal
call %~dp0utils GetScriptPath SCRIPTPATH
call %FUNC% SetPythonPath
call %FUNC% UsePython
call %FUNC% GetModName MODNAME
python -c "from guidata.utils.gettext_helpers import do_%1; do_%1('%MODNAME%')"
call %FUNC% EndOfScript