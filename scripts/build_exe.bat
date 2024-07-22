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
call %FUNC% SetPythonPath
call %FUNC% UsePython
call %FUNC% GetVersion VERSION
set REPODIR=%SCRIPTPATH%\..

@REM Keep this around for Qt hooks debugging (the output must not be empty):
@REM %PYTHON%  -c "from PyInstaller.utils.hooks import qt; print(qt.pyqt5_library_info.location)"

@REM Clone repository in a temporary directory
set CLONEDIR=%REPODIR%\..\%LIBNAME%-tempdir
if exist %CLONEDIR% ( rmdir /s /q %CLONEDIR% )
git clone -l -s . %CLONEDIR%
pushd %CLONEDIR%

@REM Generating icon
set INKSCAPE_PATH="C:\Program Files\Inkscape\bin\inkscape.exe"
set RESPATH=%CLONEDIR%\resources
for %%s in (16 24 32 48 128 256) do (
  %INKSCAPE_PATH% "%RESPATH%\%LIBNAME%.svg" -o "%RESPATH%\tmp-%%s.png" -w %%s -h %%s
)
magick "%RESPATH%\tmp-*.png" "%RESPATH%\%LIBNAME%.ico"
del "%RESPATH%\tmp-*.png"

@REM Building executable
pyinstaller %LIBNAME%.spec --noconfirm --clean
cd dist
set ZIPNAME=%LIBNAME%-v%VERSION%_exe.zip
"C:\Program Files\7-Zip\7z.exe" a -mx1 "%ZIPNAME%" %LIBNAME%
popd

if not exist %REPODIR%\dist ( mkdir %REPODIR%\dist )
@REM Move zipped executable to dist directory
move %CLONEDIR%\dist\%ZIPNAME% %REPODIR%\dist
@REM Move generated folder to dist directory
move %CLONEDIR%\dist\%LIBNAME% %REPODIR%\dist

rmdir /s /q %CLONEDIR%
call %FUNC% EndOfScript