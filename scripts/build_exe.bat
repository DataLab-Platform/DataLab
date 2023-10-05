@echo off
REM This script adapted from PythonQwt project
REM ======================================================
REM Executable build script
REM ======================================================
REM Licensed under the terms of the MIT License
REM Copyright (c) 2020-2023 Pierre Raybaut
REM (see PythonQwt LICENSE file for more details)
REM ======================================================
call %~dp0utils GetScriptPath SCRIPTPATH
call %FUNC% GetLibName LIBNAME
set ROOTPATH=%SCRIPTPATH%\..\

@REM Generating icon
set INKSCAPE_PATH="C:\Program Files\Inkscape\bin\inkscape.exe"
set RESPATH=%ROOTPATH%resources
for %%s in (16 24 32 48 128 256) do (
  %INKSCAPE_PATH% "%RESPATH%\%LIBNAME%.svg" -o "%RESPATH%\tmp-%%s.png" -w %%s -h %%s
)
magick convert "%RESPATH%\tmp-*.png" "%RESPATH%\%LIBNAME%.ico"
del "%RESPATH%\tmp-*.png"

@REM Building executable
call %FUNC% SetPythonPath
call %FUNC% UsePython
call %FUNC% GetVersion VERSION
pyinstaller %LIBNAME%.spec --noconfirm
"C:\Program Files\7-Zip\7z.exe" a -mx1 "dist\%LIBNAME%-v%VERSION%_exe.zip" dist\%LIBNAME%
call %FUNC% EndOfScript