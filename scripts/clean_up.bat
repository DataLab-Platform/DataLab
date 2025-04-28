@echo off
REM This script was copied from PythonQwt project
REM ======================================================
REM Clean up repository
REM ======================================================
REM Licensed under the terms of the MIT License
REM Copyright (c) 2020 Pierre Raybaut
REM (see PythonQwt LICENSE file for more details)
REM ======================================================
call %~dp0utils GetScriptPath SCRIPTPATH
call %FUNC% GetLibName LIBNAME
call %FUNC% GetModName MODNAME
cd %SCRIPTPATH%\..\

@REM Removing files/directories related to Python/doc build process
if exist %LIBNAME%.egg-info ( rmdir /s /q %LIBNAME%.egg-info )
if exist %MODNAME%.egg-info ( rmdir /s /q %MODNAME%.egg-info )
if exist MANIFEST ( del /q MANIFEST )
if exist build ( rmdir /s /q build )
if exist dist ( rmdir /s /q dist )
if exist doc\_build ( rmdir /s /q doc\_build )

@REM Removing cache files/directories related to Python execution
del /s /q *.pyc 1>nul 2>&1
del /s /q *.pyo 1>nul 2>&1
FOR /d /r %%d IN ("__pycache__") DO @IF EXIST "%%d" rd /s /q "%%d"

@REM Removing version control backup files
del /s /q *~ 1>nul 2>&1
del /s /q *.bak 1>nul 2>&1
del /s /q *.orig 1>nul 2>&1

@REM Removing localization template files
if exist doc\locale\pot ( rmdir /s /q doc\locale\pot )
del /s /q %MODNAME%\locale\%MODNAME%.pot 1>nul 2>&1

@REM Removing files related to documentation generation
del /q doc/contributing/changelog.md 1>nul 2>&1

@REM Removing generated PDF documentation files
del /q %MODNAME%\data\doc\%LIBNAME%*.pdf 1>nul 2>&1

@REM Log files
del /s /q *.log 1>nul 2>&1

@REM Removing files/directories related to Coverage and pytest
if exist .coverage ( del /q .coverage )
if exist coverage.xml ( del /q coverage.xml )
if exist htmlcov ( rmdir /s /q htmlcov )
del /s /q .coverage.* 1>nul 2>&1
if exist sitecustomize.py ( del /q sitecustomize.py )
if exist .pytest_cache ( rmdir /s /q .pytest_cache )

@REM Removing files/directories related to WiX installer
if exist wix\bin ( rmdir /s /q wix\bin )
if exist wix\obj ( rmdir /s /q wix\obj )
if exist wix\*.bmp ( del /q wix\*.bmp )
if exist wix\*.wixpdb ( del /q wix\*.wixpdb )
del /q wix\%LIBNAME%*.wxs 1>nul 2>&1
