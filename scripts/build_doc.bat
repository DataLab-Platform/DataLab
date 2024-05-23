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
%PYTHON% doc\update_validation_status.py

@REM Set light mode for Qt applications and clean previous documentation ===============
set QT_COLOR_MODE=light
if exist %MODNAME%\data\doc ( rmdir /s /q %MODNAME%\data\doc )
mkdir %MODNAME%\data\doc

@REM Build documentation ===============================================================
for %%L in (fr en) do (
    @REM -------------------------------------------------------------------------------
    @REM Create dummy PDF file, otherwise the PDF menu entry in "?" menu
    @REM won't be visible in the automatic screenshot
    echo Dummy PDF file > %MODNAME%\data\doc\%LIBNAME%_%%L.pdf
    @REM -------------------------------------------------------------------------------
    set LANG=%%L
    %PYTHON% doc/update_screenshots.py
    if exist build\doc ( rmdir /s /q build\doc )
    sphinx-build -b latex -D language=%%L doc build\doc
    cd build\doc
    echo Building PDF documentation for %%L...
    pdflatex -interaction=nonstopmode -quiet %LIBNAME%.tex
    @REM Build again to fix table of contents (workaround)
    pdflatex -interaction=nonstopmode -quiet %LIBNAME%.tex
    echo Done.
    cd ..\..
    move /Y build\doc\%LIBNAME%.pdf %MODNAME%\data\doc\%LIBNAME%_%%L.pdf
)

@REM explorer %MODNAME%\data\doc

call %FUNC% EndOfScript