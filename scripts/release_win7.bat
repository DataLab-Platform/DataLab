@echo off

call %~dp0utils GetScriptPath SCRIPTPATH
call %FUNC% GetLibName LIBNAME
call %FUNC% GetModName MODNAME
call %FUNC% SetPythonPath
call %FUNC% UsePython
call %FUNC% GetVersion VERSION

echo ===========================================================================
echo Making %LIBNAME% v%VERSION% release with %WINPYDIRBASE% for Windows 7
echo ===========================================================================

set destdir=releases\%LIBNAME%-v%VERSION%-release
if not exist %destdir% ( mkdir %destdir% )
ren %LIBNAME%-%VERSION%.exe %LIBNAME%-%VERSION%-Win7.exe
move %LIBNAME%-%VERSION%-Win7.exe %destdir%
explorer %destdir%

call %FUNC% EndOfScript