@echo off

set INKSCAPE_PATH="C:\Program Files\Inkscape\bin\inkscape.exe"

@REM Deploying images and icons
copy DataLab.svg ..\datalab\data\logo
%INKSCAPE_PATH% "DataLab-Title.svg" -o "..\doc\_static\DataLab-Title.png" -w 190
%INKSCAPE_PATH% "DataLab-Frontpage.svg" -o "..\doc\_static\DataLab-Frontpage.png" -w 1300
%INKSCAPE_PATH% "DataLab-Splash.svg" -o "..\datalab\data\logo\DataLab-Splash.png" -w 350
%INKSCAPE_PATH% "DataLab-Watermark.svg" -o "..\datalab\data\logo\DataLab-Watermark.png" -w 225
%INKSCAPE_PATH% "DataLab-Banner.svg" -o "..\doc\images\DataLab-Banner.png" -w 364
%INKSCAPE_PATH% "DataLab-Banner.svg" -o "..\datalab\data\logo\DataLab-Banner-150.png" -w 150
%INKSCAPE_PATH% "DataLab-Screenshot-Theme.svg" -o "..\doc\images\DataLab-Screenshot-Theme.png" -w 982
%INKSCAPE_PATH% "DataLab-Overview.svg" -o "..\doc\images\DataLab-Overview.png" -w 1250
%INKSCAPE_PATH% "DataLab-Windows-Installer.svg" -o "..\doc\images\shots\windows_installer.png" -w 900

@REM Generating icon
call :generate_icon "DataLab"
call :generate_icon "DataLab-Reset"

@REM Generating images for WiX installer
%INKSCAPE_PATH% "WixUIBanner.svg" -o "temp.png" -w 493 -h 58
magick "temp.png" bmp3:"banner.bmp"
%INKSCAPE_PATH% "WixUIDialog.svg" -o "temp.png" -w 493 -h 312
magick "temp.png" bmp3:"dialog.bmp"
del "temp.png"
move /y *.bmp ..\wix

goto:eof

:generate_icon
set ICON_NAME=%1
for %%s in (16 24 32 48 128 256) do (
  %INKSCAPE_PATH% "%ICON_NAME%.svg" -o "tmp-%%s.png" -w %%s -h %%s
)
magick "tmp-*.png" "%ICON_NAME%.ico"
del "tmp-*.png"
goto:eof
