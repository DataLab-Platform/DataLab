@echo off
REM This script was copied from PythonQwt project
REM ======================================================
REM Documentation build script
REM ======================================================
REM Licensed under the terms of the MIT License
REM Copyright (c) 2020 Pierre Raybaut
REM (see PythonQwt LICENSE file for more details)
REM ======================================================
call %~dp0utils GetScriptPath SCRIPTPATH
call %FUNC% GetLibName LIBNAME
call %FUNC% GetModName MODNAME
call %FUNC% SetPythonPath
call %FUNC% UsePython
call %FUNC% GetVersion CDL_VERSION
cd %SCRIPTPATH%\..
%PYTHON% doc\update_requirements.py
set PATH=C:\Program Files\HTML Help Workshop;C:\Program Files (x86)\HTML Help Workshop;%PATH%

@REM Build documentation in french =====================================================
@REM Update screenshots
set QT_COLOR_MODE=light
set LANG=fr
%PYTHON% doc/update_screenshots.py
@REM Build documentation
if exist %MODNAME%\data\doc ( rmdir /s /q %MODNAME%\data\doc )
sphinx-build -D language=fr -b singlehtml doc %MODNAME%\data\doc
@REM Rename index.html to index_fr.html and update links
ren %MODNAME%\data\doc\index.html index_fr.html
pushd %MODNAME%\data\doc
%PYTHON% -c "with open('index_fr.html', 'r+', encoding='utf-8') as f: content = f.read(); f.seek(0); f.write(content.replace('index.html', 'index_fr.html')); f.truncate()"
popd

@REM Build documentation in english ====================================================
@REM Update screenshots
set LANG=en
%PYTHON% doc/update_screenshots.py
sphinx-build -D language=en -b singlehtml doc %MODNAME%\data\doc
@REM explorer %MODNAME%\data\doc
call %FUNC% EndOfScript