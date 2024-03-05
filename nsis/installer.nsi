/*

Licensed under the terms of the BSD 3-Clause
(see cdl/LICENSE for details)

*/

!define DISTDIR "$%NSIS_DIST_PATH%"
!define PRODNAME "$%NSIS_PRODUCT_NAME%"
!define INSTALLDIR "$%NSIS_INSTALLDIR%"
!define ID "$%NSIS_PRODUCT_ID%"
!define VERSIONPROD "$%NSIS_PRODUCT_VERSION%"
!define VERSIONINST "$%NSIS_INSTALLER_VERSION%"
!define COPYRIGHT "$%NSIS_COPYRIGHT_INFO%"
!define HELPLINK "$%NSIS_HELP_LINK%"
!define URLUPDATEINFO "$%NSIS_URLUPDATEINFO%"
!define URLINFOABOUT "$%NSIS_URLINFOABOUT%"

Unicode true
SetCompressor /SOLID zlib

; Includes
;------------------------------------------------------------------------------
!include "MUI2.nsh"
!include "Sections.nsh"
!include "FileFunc.nsh"
!include "Locate.nsh"
!include "include\UninstallExisting.nsh"

; General
;------------------------------------------------------------------------------
Name "${ID}"
OutFile "..\${ID}-${VERSIONPROD}.exe"
InstallDir "${INSTALLDIR}"
BrandingText "${PRODNAME}"
XPStyle on
RequestExecutionLevel user

; Interface Configuration
;------------------------------------------------------------------------------
!define MUI_HEADERIMAGE
!define MUI_HEADERIMAGE_BITMAP "banner.bmp"
!define MUI_HEADERIMAGE_UNBITMAP "banner.bmp"
!define MUI_ABORTWARNING
!define MUI_ICON "install.ico"
!define MUI_UNICON "uninstall.ico"

; Pages
;------------------------------------------------------------------------------
!define MUI_WELCOMEFINISHPAGE_BITMAP "win.bmp"
!define MUI_WELCOMEPAGE_TEXT "$(welcome_str)"
!define MUI_UNWELCOMEFINISHPAGE_BITMAP "win.bmp"
!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!define MUI_FINISHPAGE_NOAUTOCLOSE
!define MUI_FINISHPAGE_REBOOTLATER_DEFAULT
!define MUI_FINISHPAGE_LINK "$(explore_str)"
!define MUI_FINISHPAGE_LINK_LOCATION "$INSTDIR"
!define MUI_FINISHPAGE_RUN
!define MUI_FINISHPAGE_RUN_TEXT "$(run_str)"
!define MUI_FINISHPAGE_RUN_FUNCTION "LaunchLink"
!insertmacro MUI_PAGE_FINISH
!insertmacro MUI_UNPAGE_WELCOME
!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES
!insertmacro MUI_UNPAGE_FINISH
!insertmacro MUI_LANGUAGE "English"
!insertmacro MUI_LANGUAGE "French"
LangString welcome_str ${LANG_ENGLISH} "This will install ${PRODNAME} on your computer.$\r$\n${PRODNAME} deploys itself as a simple folder unzipping to a custom target directory. Installing this software has thus limited effect on the operating system and will not compromise any existing Python installation.$\r$\n$\r$\nPlease click on Next to continue."
LangString welcome_str ${LANG_FRENCH} "Vous êtes sur le point d'installer ${PRODNAME} sur votre ordinateur.$\r$\n${PRODNAME} s'installe par une simple copie de fichiers dans un répertoire de destination paramétrable. L'installation n'est pas invasive et peut cohabiter (sans effet de bord) avec n'importe quelle autre distribution Python déjà installée sur cet ordinateur.$\r$\n$\r$\nCliquez sur Suivant pour continuer."
LangString explore_str ${LANG_ENGLISH} "Browse installation directory"
LangString explore_str ${LANG_FRENCH} "Explorer le dossier d'installation"
LangString uninstall_str ${LANG_ENGLISH} "Uninstall ${PRODNAME}"
LangString uninstall_str ${LANG_FRENCH} "Désinstaller ${PRODNAME}"
LangString run_str ${LANG_ENGLISH} "Run ${PRODNAME}"
LangString run_str ${LANG_FRENCH} "Démarrer ${PRODNAME}"
LangString busy_str ${LANG_ENGLISH} "Installer is already running"
LangString busy_str ${LANG_FRENCH} "L'installeur est déjà en cours d'ex�cution"
LangString uninst_str ${LANG_ENGLISH} "Permanantly remove ${PRODNAME}?"
LangString uninst_str ${LANG_FRENCH} "Souhaitez-vous réellement supprimer ${PRODNAME} ?"
LangString uninstprev_str ${LANG_ENGLISH} "Remove ${PRODNAME} already installed version?"
LangString uninstprev_str ${LANG_FRENCH} "Souhaitez-vous supprimer la version déjà installée de ${PRODNAME} ?"
LangString uninstprev_failed_str ${LANG_ENGLISH} "Failed to uninstall, continue anyway?"
LangString uninstprev_failed_str ${LANG_FRENCH} "La désinstallation a échouée, souhaitez-vous néanmoins continuer ?"

; Installer Sections
;------------------------------------------------------------------------------
!define PUBLISHER "DataLab Platform Developers"
!define UINSTREG "SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\${ID}"
Section "" SecWinPython
    SectionIn RO
    SetOutPath "$INSTDIR"
    File /r "${DISTDIR}\*.*"
    WriteUninstaller $INSTDIR\uninstaller.exe
    createDirectory "$SMPROGRAMS\${PRODNAME}"
    createShortCut "$SMPROGRAMS\${PRODNAME}\${PRODNAME}.lnk" "$INSTDIR\${PRODNAME}.exe"
    createShortCut "$SMPROGRAMS\${PRODNAME}\$(explore_str).lnk" "$INSTDIR\"
    createShortCut "$SMPROGRAMS\${PRODNAME}\$(uninstall_str).lnk" "$INSTDIR\uninstaller.exe"
    WriteRegStr HKCU "${UINSTREG}" "DisplayName" "${PRODNAME}"
    WriteRegStr HKCU "${UINSTREG}" "DisplayIcon" "$INSTDIR\${PRODNAME}.exe"
    WriteRegStr HKCU "${UINSTREG}" "DisplayVersion" "${VERSIONPROD}"
    WriteRegStr HKCU "${UINSTREG}" "Publisher" "${PUBLISHER}"
    WriteRegStr HKCU "${UINSTREG}" "HelpLink" "${HELPLINK}"
    WriteRegStr HKCU "${UINSTREG}" "URLUpdateInfo" "${URLUPDATEINFO}"
    WriteRegStr HKCU "${UINSTREG}" "URLInfoAbout" "${URLINFOABOUT}"
    ${GetSize} "$INSTDIR" "/S=0K" $0 $1 $2
    IntFmt $0 "0x%08X" $0
    WriteRegDWORD HKCU "${UINSTREG}" "EstimatedSize" "$0"
    WriteRegStr HKCU "${UINSTREG}" "UninstallString" "$\"$INSTDIR\uninstaller.exe$\""
    WriteRegStr HKCU "${UINSTREG}" "QuietUninstallString" "$\"$INSTDIR\uninstaller.exe$\" /S"
SectionEnd

Section "Uninstall"
    DeleteRegKey HKCU "${UINSTREG}"
    RMDir /r "$SMPROGRAMS\${PRODNAME}"
    Delete $INSTDIR\uninstaller.exe
    ; **********************************************************************************
    ; The following is adapted to a PyInstaller 6.0 "onedir mode" distribution
    Delete "$INSTDIR\${PRODNAME}.exe"
    RMDir /r $INSTDIR\_internal
    RMDir $INSTDIR\plugins
    RMDir $INSTDIR
    ; **********************************************************************************
SectionEnd


; Functions
;------------------------------------------------------------------------------
Function .onInit
    ; Check if an instance of this installer is already running
    System::Call 'kernel32::CreateMutexA(i 0, i 0, t "${ID}") i .r1 ?e'
    Pop $R0
    StrCmp $R0 0 +3
        MessageBox MB_OK|MB_ICONEXCLAMATION "$(busy_str)"
        Abort
    ReadRegStr $0 HKCU "${UINSTREG}" "QuietUninstallString"
    ${If} $0 != ""
    ${AndIf} ${Cmd} `MessageBox MB_YESNO|MB_ICONQUESTION "$(uninstprev_str)" /SD IDYES IDYES`
        !insertmacro UninstallExisting $0 $0
        ${If} $0 <> 0
            MessageBox MB_YESNO|MB_ICONSTOP "$(uninstprev_failed_str)" /SD IDYES IDYES +2
                Abort
        ${EndIf}
    ${EndIf}
FunctionEnd

Function LaunchLink
  ExecShell "" "$INSTDIR\${PRODNAME}.exe"
FunctionEnd

; Descriptions
;------------------------------------------------------------------------------
VIAddVersionKey "ProductName" "${PRODNAME} ${VERSIONPROD}"
VIAddVersionKey "CompanyName" "${PUBLISHER}"
VIAddVersionKey "LegalCopyright" "${COPYRIGHT}"
VIAddVersionKey "FileDescription" "${PRODNAME}"
VIAddVersionKey "FileVersion" "${VERSIONPROD}"
VIProductVersion "${VERSIONINST}"