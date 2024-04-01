@echo off

set INKSCAPE_PATH="C:\Program Files\Inkscape\bin\inkscape.exe"

@REM Deploying images and icons
copy DataLab.svg ..\cdl\data\logo
%INKSCAPE_PATH% "DataLab-Title.svg" -o "DataLab-Title.png" -w 190
copy DataLab-Title.png ..\doc\_static
%INKSCAPE_PATH% "DataLab-Frontpage.svg" -o "DataLab-Frontpage.png" -w 1300
copy DataLab-Frontpage.png ..\doc\_static
%INKSCAPE_PATH% "DataLab-Splash.svg" -o "DataLab-Splash.png" -w 300
copy DataLab-Splash.png ..\cdl\data\logo
%INKSCAPE_PATH% "DataLab-Watermark.svg" -o "..\cdl\data\logo\DataLab-Watermark.png" -w 300
%INKSCAPE_PATH% "DataLab-Banner.svg" -o "DataLab-Banner.png" -w 364
copy DataLab-Banner.png ..\doc\images
%INKSCAPE_PATH% "DataLab-Banner.svg" -o "DataLab-Banner-150.png" -w 150
move DataLab-Banner-150.png ..\cdl\data\logo
%INKSCAPE_PATH% "DataLab-Screenshot-Theme.svg" -o "DataLab-Screenshot-Theme.png" -w 982
copy DataLab-Screenshot-Theme.png ..\doc\images
%INKSCAPE_PATH% "DataLab-Overview.svg" -o "DataLab-Overview.png" -w 1250
copy DataLab-Overview.png ..\doc\images
%INKSCAPE_PATH% "DataLab-Windows-Installer.svg" -o "windows_installer.png" -w 900
move windows_installer.png ..\doc\images\shots

@REM Generating icon
for %%s in (16 24 32 48 128 256) do (
  %INKSCAPE_PATH% "DataLab.svg" -o "tmp-%%s.png" -w %%s -h %%s
)
magick convert "tmp-*.png" "DataLab.ico"
del "tmp-*.png"

@REM Generating images for WiX installer
%INKSCAPE_PATH% "DataLab-WixUIBanner.svg" -o "temp.png" -w 493 -h 58
magick convert "temp.png" bmp3:"banner.bmp"
%INKSCAPE_PATH% "DataLab-WixUIDialog.svg" -o "temp.png" -w 493 -h 312
magick convert "temp.png" bmp3:"dialog.bmp"
del "temp.png"
move /y *.bmp ..\wix

@REM Generating icons for NSIS installer
for %%s in (16 24 32 48 128 256) do (
  %INKSCAPE_PATH% "install.svg" -o "install-%%s.png" -w %%s -h %%s
  %INKSCAPE_PATH% "uninstall.svg" -o "uninstall-%%s.png" -w %%s -h %%s
)
magick convert "install-*.png" "install.ico"
magick convert "uninstall-*.png" "uninstall.ico"
del "install-*.png"
del "uninstall-*.png"
move /y *install.ico ..\nsis
