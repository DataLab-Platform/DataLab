@echo off
REM This script adapted from PythonQwt project
REM ======================================================
REM Executable build script
REM ======================================================
REM Licensed under the terms of the MIT License
REM Copyright (c) 2020 Pierre Raybaut
REM (see PythonQwt LICENSE file for more details)
REM ======================================================
call %~dp0utils GetScriptPath SCRIPTPATH
call %FUNC% GetLibName LIBNAME
call %FUNC% SetPythonPath
call %FUNC% UseWinPython
call %FUNC% GetVersion VERSION
pyinstaller %LIBNAME%.spec --noconfirm
"C:\Program Files\7-Zip\7z.exe" a -mx1 "dist\%LIBNAME%-v%VERSION%_exe.zip" dist\%LIBNAME%
call %FUNC% EndOfScript