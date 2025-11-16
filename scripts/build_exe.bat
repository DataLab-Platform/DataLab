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

@REM Backup PYTHONPATH
set OLD_PYTHONPATH=%PYTHONPATH%
@REM Make a new virtual environment for building the executable
set VENV_DIR=%REPODIR%\..\%LIBNAME%-temp-venv
if exist %VENV_DIR% ( rmdir /s /q %VENV_DIR% )
%PYTHON% -m venv %VENV_DIR%
set PYTHON=%VENV_DIR%\Scripts\python.exe
set PYTHONPATH=%CLONEDIR%
@REM Install required packages in the virtual environment
%PYTHON% -m pip install --upgrade pip setuptools wheel
@REM Install DataLab and its dependencies
%PYTHON% -m pip install .[exe]

@REM Generating icon
set INKSCAPE_PATH="C:\Program Files\Inkscape\bin\inkscape.exe"
set RESPATH=%CLONEDIR%\resources
for %%s in (16 24 32 48 128 256) do (
  %INKSCAPE_PATH% "%RESPATH%\DataLab.svg" -o "%RESPATH%\tmp-%%s.png" -w %%s -h %%s
)
magick "%RESPATH%\tmp-*.png" "%RESPATH%\DataLab.ico"
del "%RESPATH%\tmp-*.png"

@REM Generate build manifest
%PYTHON% scripts\generate_manifest.py

@REM Building executable
%PYTHON% -m PyInstaller DataLab.spec --noconfirm --clean

@REM Windows 7 SP1 compatibility fix
copy "%RESPATH%\api-ms-win-core-path-l1-1-0.dll" "dist\DataLab\_internal" /Y

@REM Zipping executable
cd dist
set ZIPNAME=DataLab-v%VERSION%_exe.zip
"C:\Program Files\7-Zip\7z.exe" a -mx1 "%ZIPNAME%" DataLab
popd

if not exist %REPODIR%\dist ( mkdir %REPODIR%\dist )
@REM Move zipped executable to dist directory
move %CLONEDIR%\dist\%ZIPNAME% %REPODIR%\dist
@REM Move generated folder to dist directory
move %CLONEDIR%\dist\DataLab %REPODIR%\dist

@REM Cleanup temporary directories
rmdir /s /q %CLONEDIR%
rmdir /s /q %VENV_DIR%
@REM Restore PYTHONPATH
set PYTHONPATH=%OLD_PYTHONPATH%

call %FUNC% EndOfScript