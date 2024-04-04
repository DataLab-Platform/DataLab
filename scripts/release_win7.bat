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
pushd "wix"
ren %LIBNAME%-%VERSION%.msi %LIBNAME%-%VERSION%-Win7.msi
popd
move wix\%LIBNAME%-%VERSION%-Win7.msi %destdir%
explorer %destdir%

call %FUNC% EndOfScript