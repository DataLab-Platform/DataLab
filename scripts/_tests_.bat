@echo off
REM ======================================================
REM Generic Python Installer build script
REM ------------------------------------------
REM Copyright (c) 2021 Codra
REM ======================================================
call %~dp0utils GetScriptPath SCRIPTPATH
call %FUNC% GetLibName LIBNAME
call %FUNC% SetPythonPath
call %FUNC% UsePython
call %FUNC% GetVersion VERSION
call %FUNC% GetModName MODNAME

echo Modname=%MODNAME%
echo LibName=%LIBNAME%
echo LibVersion=%VERSION%

call %FUNC% EndOfScript