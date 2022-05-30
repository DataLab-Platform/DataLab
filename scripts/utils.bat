@echo off
set FUNC=%0
call:%*
goto Exit

REM ======================================================
REM Utilities for deployment, test and build scripts
REM ======================================================
REM Licensed under the terms of the MIT License
REM Copyright (c) 2020 Pierre Raybaut
REM (see LICENSE file for more details)
REM ======================================================

:GetScriptPath
set _tmp_=%~dp0
if %_tmp_:~-1%==\ set %1=%_tmp_:~0,-1%
EXIT /B 0

:GetLibName
pushd %~dp0..
for %%I in (.) do set %1=%%~nxI
popd
goto:eof

:GetModName
pushd %~dp0..
for /D %%I in (*) DO (
    if exist %%I\__init__.py (
        set %1=%%I
        goto :found_module
    )
)
:found_module
popd
goto:eof

:GetVersion
call:GetModName MODNAME
call:SetPythonPath
echo import %MODNAME%;print(%MODNAME%.__version__) | python > _tmp_.txt
set /p %1=<_tmp_.txt
del _tmp_.txt
goto:eof

:SetPythonPath
set PYTHONPATH=%~dp0..;%PYTHONPATH%
goto:eof

:UseWinPython
if defined WINPYVER (goto:eof)
if not defined PYTHON (goto :nopython)
for %%a in ("%PYTHON%") do set "p_dir=%%~dpa"
for %%a in (%p_dir:~0,-1%) do set "WINPYDIRBASE=%%~dpa"
:nopython
if defined WINPYDIRBASE (
    call %WINPYDIRBASE%\scripts\env.bat
    call :ShowTitle "Using WinPython from %WINPYDIRBASE%"
) else (
    echo Warning: WINPYDIRBASE environment variable is not defined, switching to system Python
    echo ********
    echo (if nothing happens, that's probably because Python is not installed either:
    echo please set the WINPYDIRBASE variable to select WinPython directory, or install Python)
    )
goto:eof

:ShowTitle
@echo:
@echo ========= %~1 =========
@echo:
goto:eof

:EndOfScript
@echo:
@echo **********************************************************************************
@echo:
if not defined UNATTENDED (
    @echo End of script
    pause
    )
goto:eof

:Exit
exit /b