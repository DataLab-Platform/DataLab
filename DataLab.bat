@echo off
for %%a in ("%CDL_PYTHONEXE%") do set "p_dir=%%~dpa"
for %%a in (%p_dir:~0,-1%) do set "WINPYDIRBASE=%%~dpa"
call %WINPYDIRBASE%scripts\env_for_icons.bat %*
cd/D %~dp0
set ORIGINAL_PYTHONPATH=%PYTHONPATH%
for /F "tokens=*" %%A in (.env) do (set %%A)
set PYTHONPATH=%PYTHONPATH%;%ORIGINAL_PYTHONPATH%
start "" "%WINPYDIR%\pythonw.exe" cdl\start.pyw %*