<#
.SYNOPSIS
    Create a disposable venv, install DataLab + dependencies into it, then run
    pytest from the *installed* version (site-packages). Output is shown live
    AND saved to <venv>.log with a summary header. The venv can be deleted
    automatically at the end of the run (the log is always kept).

.DESCRIPTION
    Self-contained script, runnable from any location: venvs and logs are created
    in a "runs" sub-folder of the script directory ($PSScriptRoot\runs), using
    absolute paths. That sub-folder is git-ignored.

    Machine-specific settings (the "Python version -> interpreter" map and an
    optional requirements.txt override) live in a git-ignored PowerShell data
    file next to the script, "Run-DataLabTests.config.psd1". A committed template
    "Run-DataLabTests.config.template.psd1" documents the format; copy it and edit
    the paths. The script fails early with guidance if the config file is missing.

    Three choices (offered via interactive menus when not passed as parameters):
      1. Package source:
           * pypi    : DataLab and all its dependencies from PyPI.
           * develop : Sigima and DataLab from the latest commit of their
                       "develop" branch (git+URL@develop); the rest from PyPI.
           * main    : same but from the "main" branch of Sigima and DataLab.
      2. Python version: 3.9 to 3.14, or "all" to iterate over every version in
         turn (one venv + one log per version).
      3. Whether to delete the venvs at the end of the run (default: yes). The
         .log files are always kept.

.PARAMETER Source
    "pypi", "develop", "main", or a DataLab tag from a pre-defined combination
    (v1.0.1, v1.0.2, v1.0.3, v1.0.4, v1.1.0, v1.1.1, v1.2.0). If absent: asked
    interactively (default pypi). In develop/main mode, Sigima and DataLab are
    installed from the matching git branch (unless -Branch is provided). For a
    tag combo, DataLab is installed at the chosen tag and Sigima at the
    associated tag (table $TagCombos), e.g. DataLab v1.2.0 + Sigima v1.1.2 --
    useful to bisect a regression on the DataLab side. NB: very old DataLab tags
    may expose a different package name (import may fail).

.PARAMETER PythonVersion
    3.9, 3.10, 3.11, 3.12, 3.13, 3.14 or "all". If absent: asked interactively
    (default 3.11). Only versions present in the config file can be run.

.PARAMETER DeleteVenv
    $true (default) deletes the venv(s) at the end of the run; $false keeps them.
    If not passed as a parameter: asked interactively (default yes).

.PARAMETER ShowWindows
    Show the Qt windows during the tests (disables QT_QPA_PLATFORM=offscreen).
    Useful to reproduce graphics-resource (GDI) issues on Windows.

.PARAMETER GdiProbe
    Enable the "on-the-fly" GDI probe (Windows only). ENABLED BY DEFAULT ($true);
    pass -GdiProbe:$false to disable it. Injects the standalone pytest plugin
    scripts\pytest_gdi_probe.py (via -p, without installing it into the venv) to
    track the number of GDI objects during the tests. Produces two CSVs next to
    the log (<venv>.gdi-pertest.csv and <venv>.gdi-timeline.csv) and reports the
    GDI peak (max of the in-process timeline) in the header (line "GDI objects
    peak").
    As the plugin is not installed, the SAME local version instruments every
    matrix cell (pypi/main/develop) identically. The GDI signal is much stronger
    in -ShowWindows (native) mode than in offscreen.

    How the two probes complement each other depending on the mode:
      * Offscreen mode (default): the QPA backend is purely raster (no native
        windows), so a leak does NOT show up as a rising GDI count but as rising
        RAM. Watch "Memory peak" instead.
      * -ShowWindows (native) mode: real windows allocate GDI handles; a leak
        then shows up as a growing GDI count (GDI probe + per-test delta). This
        is the mode to favour for tracking GDI exhaustion (the ~10000-handle
        crash is intrinsically a native phenomenon).

.PARAMETER TimeoutSeconds
    Maximum time per test in seconds (anti-hang guard, especially in native
    mode), applied via the pytest-timeout plugin ('thread' method on Windows).
    Default 300; 0 to disable. A test exceeding this delay terminates the
    session (with a stack dump) instead of freezing the cell for hours (e.g. a
    modal dialog left open). Can be overridden via -PytestArgs '--timeout=N'.

.PARAMETER PytestArgs
    Arguments passed as-is to pytest (e.g. -k pattern, -x, --durations=20, a test
    sub-path, etc.).

.PARAMETER RequirementsPath
    Path to DataLab's requirements.txt. If empty (default), it is resolved to
    "<repo>\requirements.txt" relative to the script location, or to the
    'RequirementsPath' entry of the config file if set.

.PARAMETER DataLabVersion
    (pypi mode) Specific version of datalab-platform (e.g. "1.2.3"). Default: latest.

.PARAMETER Branch
    (develop mode) git branch to install. Default: develop.

.PARAMETER Matrix
    Matrix mode: automatically runs the report matrix instead of a single run --
    Source {develop, main} x PythonQwt {pypi, <chosen fix>} x window mode(s) x
    Python 3.9-3.14, sequentially. A few questions (fix variant, mode(s), venv
    deletion) then confirmation of the number of cells. Produces a console
    summary and an aggregated Markdown report (per-cell detail + Source x
    PythonQwt x mode synthesis) in runs\. In fully interactive mode (no
    parameter), a "Run type" menu also offers this mode. Tip: run on an idle
    machine for comparable durations.

.PARAMETER NoTestDataInjection
    Disable automatically completing the installed package with the test data
    missing from old wheels. By default, for any source OTHER than develop/main
    (PyPI, tags v1.0.x/v1.1.x/v1.2.0), the script copies from the local checkout
    -- in COPY-IF-ABSENT mode -- the config .h5 files and the plugin/macro
    templates that the published packages omitted. This avoids known test
    failures (IndexError on the 2nd .h5 of config_unit_test, missing
    plugin/macro templates) that only reflect a packaging defect already fixed on
    develop/main and needlessly polluted the log.

.EXAMPLE
    .\Run-DataLabTests.ps1 -Source pypi -PythonVersion 3.11

.EXAMPLE
    .\Run-DataLabTests.ps1 -Source v1.2.0 -PythonVersion 3.11 -ShowWindows

.EXAMPLE
    .\Run-DataLabTests.ps1 -Source develop -PythonVersion all -DeleteVenv:$false

.EXAMPLE
    .\Run-DataLabTests.ps1 -PythonVersion 3.9 -PytestArgs '-k','signal','--durations=20'

.EXAMPLE
    .\Run-DataLabTests.ps1 -PythonQwtBranch v0.15.0

.EXAMPLE
    .\Run-DataLabTests.ps1 -Matrix

.EXAMPLE
    .\Run-DataLabTests.ps1 -GdiProbe -ShowWindows -PythonVersion 3.11

.EXAMPLE
    .\Run-DataLabTests.ps1 -PythonVersion 3.12 -NoJsonReport
#>

[CmdletBinding()]
param(
    [ValidateSet('pypi', 'develop', 'main',
        'v1.0.1', 'v1.0.2', 'v1.0.3', 'v1.0.4', 'v1.1.0', 'v1.1.1', 'v1.2.0')]
    [string] $Source,

    [ValidateSet('3.9', '3.10', '3.11', '3.12', '3.13', '3.14', 'all')]
    [string] $PythonVersion,

    [bool] $DeleteVenv,

    [switch] $ShowWindows,

    # GDI probe (Windows) ENABLED BY DEFAULT: injects scripts\pytest_gdi_probe.py
    # via -p (without installing it into the venv) + reports the GDI peak of the
    # venv processes. Disable with -GdiProbe:$false.
    [bool] $GdiProbe = $true,

    # Matrix mode: automatically runs the report matrix
    # (Source {develop,main} x PythonQwt {pypi, <fix>} x mode x Python 3.9-3.14).
    # Produces a console summary + an aggregated Markdown report in runs\.
    [switch] $Matrix,

    # By default, the script logs the duration of EACH test
    # ('--durations=0 --durations-min=0', all setup/call/teardown phases) AND
    # generates a detailed per-test JSON report via the pytest-json-report plugin
    # (installed automatically into the venv), written next to the log:
    # <venv>.report.json. -NoJsonReport disables only the JSON report (durations
    # are still shown in the log).
    [switch] $NoJsonReport,

    # Anti-hang guard: maximum time per test in seconds, applied via the
    # pytest-timeout plugin ('thread' method on Windows). Prevents a stuck test
    # (e.g. a modal dialog left open in native mode) from freezing the cell for
    # hours. Default 300 s; 0 to disable; can be overridden via
    # -PytestArgs '--timeout=N'.
    [int] $TimeoutSeconds = 300,

    [string[]] $PytestArgs = @(),

    # Path to DataLab's requirements.txt. Empty (default) => resolved to
    # <repo>\requirements.txt (or the config 'RequirementsPath' entry).
    [string] $RequirementsPath = '',

    [string] $DataLabVersion = '',

    [string] $Branch = 'develop',

    # PythonQwt: 'pypi' (default) keeps the version resolved by pip; otherwise the
    # name of a branch or commit to install from the PythonQwt git repository
    # (e.g. 'v0.15.0', 'master', a SHA).
    [string] $PythonQwtBranch,

    # By default, for any source OTHER than develop/main (PyPI, old tags), the
    # script completes the installed package with the test data that old wheels
    # do not ship (config .h5 files, plugin and macro templates), copied from the
    # local checkout in "copy-if-absent" mode. This avoids known test failures
    # that pollute the log. -NoTestDataInjection disables this behaviour.
    [switch] $NoTestDataInjection
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

# --- Common path resolution --------------------------------------------------

$ScriptDir = $PSScriptRoot
if ([string]::IsNullOrEmpty($ScriptDir)) { $ScriptDir = (Get-Location).Path }

# DataLab repository root (the script lives in <repo>\scripts). Used as the
# canonical source to complete old wheels with the missing test data, and to
# resolve requirements.txt relative to the repository.
$RepoRoot = Split-Path -Parent $ScriptDir

# --- Machine-specific configuration ------------------------------------------

# The "Python version -> interpreter" map (and an optional requirements path)
# live in a git-ignored PowerShell data file next to this script. A committed
# template documents the format; the script fails early if the real file is
# missing.
$ConfigPath   = Join-Path $ScriptDir 'Run-DataLabTests.config.psd1'
$TemplatePath = Join-Path $ScriptDir 'Run-DataLabTests.config.template.psd1'
if (-not (Test-Path -LiteralPath $ConfigPath)) {
    throw ("Configuration file not found: $ConfigPath`n" +
        "Copy the template and edit the interpreter paths for your machine:`n" +
        "    Copy-Item '$TemplatePath' '$ConfigPath'`n" +
        "Then edit '$ConfigPath' (see the comments inside).")
}
$Config = Import-PowerShellDataFile -LiteralPath $ConfigPath
# Use ContainsKey rather than member access: under Set-StrictMode -Version
# Latest, accessing a missing hashtable key via "$Config.Foo" throws instead of
# returning $null.
if (-not $Config.ContainsKey('PythonPaths')) {
    throw "Config '$ConfigPath' must define a 'PythonPaths' hashtable (version -> python.exe)."
}
# Import-PowerShellDataFile returns an UNORDERED hashtable: re-sort the keys by
# version so that "all" / the matrix iterate 3.9 -> 3.14 deterministically.
$PythonPaths = [ordered]@{}
foreach ($key in ($Config.PythonPaths.Keys | Sort-Object { [version]$_ })) {
    $PythonPaths[$key] = $Config.PythonPaths[$key]
}
$AllVersions = @($PythonPaths.Keys)

# requirements.txt: explicit -RequirementsPath wins, else the config override,
# else <repo>\requirements.txt (resolved relative to the script location).
if ([string]::IsNullOrWhiteSpace($RequirementsPath)) {
    $RequirementsPath = if ($Config.ContainsKey('RequirementsPath')) { $Config.RequirementsPath }
    else { Join-Path $RepoRoot 'requirements.txt' }
}

# git repositories ("develop"/"main" sources and PythonQwt override).
$SigimaGitUrl  = 'https://github.com/DataLab-Platform/sigima.git'
$DataLabGitUrl = 'https://github.com/DataLab-Platform/DataLab.git'
$PythonQwtGitUrl = 'https://github.com/PlotPyStack/PythonQwt.git'

# DataLab/Sigima tag combinations (to bisect a DataLab regression). The key is
# the DataLab tag (also used as the $Source token for file naming); the value is
# the Sigima tag to install alongside. Chosen via the 1st "Source" menu or
# directly via -Source <DataLab tag>.
$TagCombos = [ordered]@{
    'v1.0.1' = 'v1.0.2'
    'v1.0.2' = 'v1.0.3'
    'v1.0.3' = 'v1.0.5'
    'v1.0.4' = 'v1.1.1'
    'v1.1.0' = 'v1.1.1'
    'v1.1.1' = 'v1.1.1'
    'v1.2.0' = 'v1.1.2'
}

# --- General utilities -------------------------------------------------------

function Format-Duration {
    param([TimeSpan] $Span)
    '{0:00}h{1:00}m{2:00}s' -f [int]$Span.TotalHours, $Span.Minutes, $Span.Seconds
}

function Get-ExitCodeInfo {
    <#
        Translate a pytest exit code into a readable description. Distinguishes
        pytest's "business" codes (0-5) from native Windows crashes (negative
        NTSTATUS codes), which indicate a crash of the Python process itself
        (invalid memory access, overflow, resource exhaustion, etc.).
    #>
    param([int] $Code)

    if ($Code -eq 0) { return 'success (all tests pass)' }

    $pytestCodes = @{
        1 = 'pytest: some tests failed'
        2 = 'pytest: execution interrupted (Ctrl-C or collection error)'
        3 = 'pytest: internal error'
        4 = 'pytest: usage error (command line)'
        5 = 'pytest: no test collected'
    }
    if ($pytestCodes.ContainsKey($Code)) { return $pytestCodes[$Code] }

    # Native crash: present the code as an unsigned hexadecimal (NTSTATUS).
    # NB: 0xFFFFFFFFL must have the L (long) suffix, otherwise PowerShell parses
    # 0xFFFFFFFF as an int32 equal to -1 and the mask fails.
    $unsigned = [uint32]([int64]$Code -band 0xFFFFFFFFL)
    $hex = ('0x{0:X8}' -f $unsigned)
    $ntstatus = @{
        '0xC0000005' = 'ACCESS_VIOLATION (invalid memory access: native bug, corruption, or GDI/handle exhaustion)'
        '0xC00000FD' = 'STACK_OVERFLOW (stack overflow: infinite recursion)'
        '0xC0000374' = 'HEAP_CORRUPTION (heap corruption)'
        '0xC0000409' = 'STACK_BUFFER_OVERRUN (/GS stack canary)'
        '0xC000001D' = 'ILLEGAL_INSTRUCTION'
        '0xC0000094' = 'INTEGER_DIVIDE_BY_ZERO'
        '0xC000013A' = 'CONTROL_C_EXIT (Ctrl-C)'
        '0x80000003' = 'BREAKPOINT (assert / native breakpoint)'
    }
    if ($ntstatus.ContainsKey($hex)) {
        return "NATIVE CRASH $hex : $($ntstatus[$hex])"
    }
    return "abnormal exit code $Code ($hex) - likely native crash"
}

function Test-AbnormalExit {
    # True if the code does not match a pytest "business" exit (0-5).
    param([int] $Code)
    return ($Code -lt 0 -or $Code -gt 5)
}

function Get-FreezeVersion {
    param([string[]] $Freeze, [string] $PackageName)
    foreach ($line in $Freeze) {
        if ([regex]::IsMatch($line, '^(?i)' + [regex]::Escape($PackageName) + '\s*[=@]')) {
            return $line.Trim()
        }
    }
    return "$PackageName (absent)"
}

function Select-FromMenu {
    <#
        Display a numbered menu and return the chosen option. Empty input =
        default value. Loops until the input is valid.
    #>
    param(
        [Parameter(Mandatory)] [string]   $Title,
        [Parameter(Mandatory)] [string[]] $Options,
        [Parameter(Mandatory)] [string]   $Default
    )
    Write-Host ''
    Write-Host $Title -ForegroundColor Cyan
    for ($i = 0; $i -lt $Options.Count; $i++) {
        $mark = if ($Options[$i] -eq $Default) { ' (default)' } else { '' }
        Write-Host ('  [{0}] {1}{2}' -f ($i + 1), $Options[$i], $mark)
    }
    while ($true) {
        $resp = Read-Host "Choice (1-$($Options.Count), Enter = default)"
        if ([string]::IsNullOrWhiteSpace($resp)) { return $Default }
        $n = 0
        if ([int]::TryParse($resp, [ref] $n) -and $n -ge 1 -and $n -le $Options.Count) {
            return $Options[$n - 1]
        }
        Write-Host 'Invalid input.' -ForegroundColor Yellow
    }
}

# --- Interactive selection (when parameters are not provided) -----------------

# Preliminary menu: single run or full matrix (only in fully interactive mode,
# i.e. when none of -Matrix, -Source, -PythonVersion are passed).
if (-not $PSBoundParameters.ContainsKey('Matrix') -and
    -not $PSBoundParameters.ContainsKey('Source') -and
    -not $PSBoundParameters.ContainsKey('PythonVersion')) {
    $runType = Select-FromMenu -Title 'Run type:' `
        -Options @('single', 'full matrix (report)') -Default 'single'
    $Matrix = ($runType -like 'full matrix*')
}

if (-not $Matrix -and -not $PSBoundParameters.ContainsKey('Source')) {
    $comboOpts = @($TagCombos.Keys | ForEach-Object { "$_ + Sigima $($TagCombos[$_])" })
    $srcChoice = Select-FromMenu -Title 'DataLab/Sigima package source:' `
        -Options (@('pypi', 'develop', 'main') + $comboOpts) -Default 'pypi'
    # For a tag combo, the 1st token of the label is the DataLab tag (= $Source).
    $Source = ($srcChoice -split ' ')[0]
}
# Effective git branch for DataLab/Sigima in non-pypi mode: derived from the
# source ('develop' or 'main') unless -Branch is provided explicitly.
if (-not $PSBoundParameters.ContainsKey('Branch') -and $Source -in @('develop', 'main')) {
    $Branch = $Source
}
if (-not $Matrix -and -not $PSBoundParameters.ContainsKey('PythonVersion')) {
    $PythonVersion = Select-FromMenu -Title 'Python version:' `
        -Options @('3.9', '3.10', '3.11', '3.12', '3.13', '3.14', 'all') -Default '3.11'
}
if (-not $Matrix -and -not $PSBoundParameters.ContainsKey('DeleteVenv')) {
    $delChoice = Select-FromMenu -Title 'Delete the venv(s) at the end of the run?' `
        -Options @('yes', 'no') -Default 'yes'
    $DeleteVenv = ($delChoice -eq 'yes')
}
if (-not $Matrix -and -not $PSBoundParameters.ContainsKey('PythonQwtBranch')) {
    $qwtChoice = Select-FromMenu -Title 'PythonQwt to test:' `
        -Options @('pypi', 'master', 'v0.15.0', '(enter a branch/commit...)') `
        -Default 'pypi'
    if ($qwtChoice -eq '(enter a branch/commit...)') {
        $qwtChoice = (Read-Host 'PythonQwt branch or commit').Trim()
        if ([string]::IsNullOrWhiteSpace($qwtChoice)) { $qwtChoice = 'pypi' }
    }
    $PythonQwtBranch = $qwtChoice
}
if (-not $Matrix -and -not $PSBoundParameters.ContainsKey('ShowWindows')) {
    $offOpt = 'offscreen (default) -- almost no GDI signal; leak visible via memory peak'
    $natOpt = 'native --show-windows -- real windows, usable GDI signal (per-test delta)'
    $winChoice = Select-FromMenu -Title 'Window mode (affects the GDI probe):' `
        -Options @($offOpt, $natOpt) -Default $offOpt
    $ShowWindows = ($winChoice -eq $natOpt)
}
# Normalise: empty or 'pypi' => no override (PyPI version resolved by pip).
$UsePythonQwtBranch = -not (
    [string]::IsNullOrWhiteSpace($PythonQwtBranch) -or $PythonQwtBranch -eq 'pypi'
)

$pyLabel = if ($PythonVersion -eq 'all') { "all ($($AllVersions -join ', '))" } else { $PythonVersion }
$qwtLabel = if ($UsePythonQwtBranch) { "git @ $PythonQwtBranch" } else { 'pypi' }
if (-not $Matrix) {
    Write-Host ''
    Write-Host ('=> Source = {0} | Python = {1} | PythonQwt = {2} | Delete venv = {3} | GDI probe = {4}' -f `
            $Source, $pyLabel, $qwtLabel, $(if ($DeleteVenv) { 'yes' } else { 'no' }), $(if ($GdiProbe) { 'yes' } else { 'no' })) -ForegroundColor Green
    if ($GdiProbe -and -not $ShowWindows) {
        Write-Host ('   (GDI probe in offscreen mode: almost no GDI signal expected ' +
            '-- add -ShowWindows for a usable signal; in offscreen the leak ' +
            'remains visible via the memory peak.)') -ForegroundColor DarkYellow
    }
}

# --- Runs directory + reproducible environment -------------------------------

if (-not (Test-Path -LiteralPath $RequirementsPath)) {
    throw "Requirements file not found: $RequirementsPath"
}

# Venvs (heavy) and logs are grouped in a "runs" sub-folder. That folder is
# git-ignored (runs\.gitignore) so it does not pollute the repository.
$RunsDir = Join-Path $ScriptDir 'runs'
if (-not (Test-Path -LiteralPath $RunsDir)) {
    New-Item -ItemType Directory -Path $RunsDir | Out-Null
}
$GitIgnore = Join-Path $RunsDir '.gitignore'
if (-not (Test-Path -LiteralPath $GitIgnore)) {
    Set-Content -LiteralPath $GitIgnore -Value '*' -Encoding ASCII
}

# Reproducible, non-interactive environment (common to all runs).
$env:QT_API = 'pyqt5'
$env:PYTHONUTF8 = '1'
$env:GUIDATA_UNATTENDED = '1'
# Recent PyQt5 wheels (5.15.11, from Python 3.12 on) no longer ship the
# "Qt5/lib/fonts" folder, and Qt warns "QFontDatabase: Cannot find font
# directory ...". We fix it cleanly (without hiding the warning) by pointing Qt
# at the Windows system fonts, which are a valid font directory.
# NB: guidata >= 3.14.5 already sets QT_QPA_FONTDIR in offscreen mode (see
# guidata._configure_fontdir). We keep it here as a safety net for "pypi" mode,
# which might install a guidata older than the fix.
$WinFontDir = Join-Path $env:WINDIR 'Fonts'
if (Test-Path -LiteralPath $WinFontDir) {
    $env:QT_QPA_FONTDIR = $WinFontDir
}
# Defensive cleanup: if a previous session (or another tool) left the debug
# allocator or dev-mode enabled, they would crash ensurepip when creating the
# venv. We neutralise them explicitly.
$env:PYTHONDEVMODE = $null
$env:PYTHONMALLOC = $null
Remove-Item Env:PYTHONDEVMODE -ErrorAction SilentlyContinue
Remove-Item Env:PYTHONMALLOC -ErrorAction SilentlyContinue
# Native-crash diagnostics: faulthandler prints a Python traceback on
# SIGSEGV / invalid memory access; unbuffered output so no line is lost at the
# moment of a hard crash.
# NB: we do NOT enable PYTHONDEVMODE (which forces the debug memory allocator
# PYTHONMALLOC=debug). The latter crashes ensurepip/pip when creating the venv
# on some WinPython builds ("Debug memory block ... FORBIDDENBYTE").
$env:PYTHONFAULTHANDLER = '1'
$env:PYTHONUNBUFFERED = '1'

# --- Core: a full run for one Python version ---------------------------------

function Invoke-DataLabTestRun {
    <#
        Run a full cycle (venv + install + pytest + log) for a given Python
        version. Returns the pytest exit code.
    #>
    param(
        [Parameter(Mandatory)] [string] $PyVersion,
        [Parameter(Mandatory)] [string] $PyExe
    )

    $stamp    = Get-Date -Format 'yyyyMMdd-HHmmss'
    # Tokens encoded in the name of ALL produced files (venv, .log, .report.json,
    # .gdi-*.csv) to find your way around the matrix:
    #  * PythonQwt: 'pypi' or the sanitised branch/tag/commit ('/' and separators
    #    -> '_', invalid characters removed);
    #  * Window mode: 'offscreen' (default) or 'native' (-ShowWindows).
    # Without these tokens, two cells differing only by PythonQwt or the mode
    # would produce identical names (collision) at the same timestamp.
    $qwtToken = if ($UsePythonQwtBranch) {
        (($PythonQwtBranch -replace '[\\/:\s]+', '_') -replace '[^\w.\-]', '')
    }
    else { 'pypi' }
    # Cap the token length (deep venv paths close to MAX_PATH).
    if ($qwtToken.Length -gt 28) { $qwtToken = $qwtToken.Substring(0, 28) }
    $modeToken = if ($ShowWindows) { 'native' } else { 'offscreen' }
    $venvName = "datalab-$Source-qwt-$qwtToken-$modeToken-py$PyVersion-$stamp"
    $venvDir  = Join-Path $RunsDir $venvName
    $venvPy   = Join-Path $venvDir 'Scripts\python.exe'
    $finalLog = Join-Path $RunsDir "$venvName.log"
    $partLog  = Join-Path $RunsDir "$venvName.log.part"
    if (Test-Path -LiteralPath $partLog) { Remove-Item -LiteralPath $partLog -Force }

    # Local helpers capturing $partLog
    function Write-Log {
        param([string] $Message = '')
        Write-Host $Message
        Add-Content -LiteralPath $partLog -Value $Message -Encoding UTF8
    }
    function Invoke-Step {
        param(
            [Parameter(Mandatory)] [string]   $Title,
            [Parameter(Mandatory)] [string]   $Exe,
            [Parameter(Mandatory)] [string[]] $Arguments,
            [switch] $AllowFailure
        )
        $banner = ('=' * 78)
        Write-Log ''
        Write-Log $banner
        Write-Log "  $Title"
        Write-Log "  > $Exe $($Arguments -join ' ')"
        Write-Log $banner
        & $Exe @Arguments 2>&1 | ForEach-Object {
            $line = "$_"
            Write-Host $line
            Add-Content -LiteralPath $partLog -Value $line -Encoding UTF8
        }
        $code = $LASTEXITCODE
        if ($code -ne 0 -and -not $AllowFailure) {
            throw "Step '$Title' failed (code $code)."
        }
        return $code
    }

    $totalSw = [System.Diagnostics.Stopwatch]::StartNew()
    Write-Log "Partial log     : $partLog"
    Write-Log "Venv            : $venvDir"
    Write-Log "Python          : $PyExe"
    Write-Log "Source          : $Source"

    $pytestExit = 1
    $peakMB = $null
    $gdiPeak = $null
    $jsonReport = $null
    $testSw = [System.Diagnostics.Stopwatch]::new()
    try {
        # 1. Create the venv
        Invoke-Step -Title '1. Creating the venv' -Exe $PyExe `
            -Arguments @('-m', 'venv', $venvDir) | Out-Null
        if (-not (Test-Path -LiteralPath $venvPy)) {
            throw "The venv was not created correctly: $venvPy"
        }

        # 2. Upgrade pip
        Invoke-Step -Title '2. Upgrading pip' -Exe $venvPy `
            -Arguments @('-m', 'pip', 'install', '--upgrade', 'pip') | Out-Null

        # 3. Install DataLab's dependencies
        Invoke-Step -Title '3. Installing dependencies (requirements.txt)' -Exe $venvPy `
            -Arguments @('-m', 'pip', 'install', '-r', $RequirementsPath) | Out-Null

        # 4. Install DataLab (+ Sigima in develop mode)
        if ($Source -eq 'pypi') {
            $pkg = if ([string]::IsNullOrWhiteSpace($DataLabVersion)) { 'datalab-platform' }
            else { "datalab-platform==$DataLabVersion" }
            Invoke-Step -Title '4. Installing DataLab (PyPI)' -Exe $venvPy `
                -Arguments @('-m', 'pip', 'install', $pkg) | Out-Null
        }
        else {
            # Tag combo (DataLab tag != Sigima tag) OR develop/main branch.
            if ($TagCombos.Contains($Source)) {
                $dlRef = $Source
                $sigRef = $TagCombos[$Source]
            }
            else {
                $dlRef = $Branch
                $sigRef = $Branch
            }
            Invoke-Step -Title "4a. Installing DataLab (git @ $dlRef)" -Exe $venvPy `
                -Arguments @('-m', 'pip', 'install', "git+$DataLabGitUrl@$dlRef") | Out-Null
            Invoke-Step -Title "4b. Installing Sigima (git @ $sigRef)" -Exe $venvPy `
                -Arguments @('-m', 'pip', 'install', '--force-reinstall', '--no-deps', "git+$SigimaGitUrl@$sigRef") | Out-Null
        }

        # 4c. Optional PythonQwt override from a git branch/commit.
        # Force-reinstall without deps to replace the PyPI version pulled in by
        # PlotPy, while keeping the rest of the dependency tree.
        if ($UsePythonQwtBranch) {
            Invoke-Step -Title "4c. Installing PythonQwt (git @ $PythonQwtBranch)" -Exe $venvPy `
                -Arguments @('-m', 'pip', 'install', '--force-reinstall', '--no-deps', "git+$PythonQwtGitUrl@$PythonQwtBranch") | Out-Null
        }

        # 4d. pytest-json-report plugin for the per-test duration report.
        if (-not $NoJsonReport) {
            Invoke-Step -Title '4d. Installing pytest-json-report' -Exe $venvPy `
                -Arguments @('-m', 'pip', 'install', 'pytest-json-report') | Out-Null
        }

        # 4e. pytest-timeout plugin: per-test anti-hang guard (native).
        if ($TimeoutSeconds -gt 0) {
            Invoke-Step -Title '4e. Installing pytest-timeout' -Exe $venvPy `
                -Arguments @('-m', 'pip', 'install', 'pytest-timeout') | Out-Null
        }

        # 5. Locate the installed package (site-packages)
        $datalabDir = (& $venvPy -c "import datalab, os; print(os.path.dirname(datalab.__file__))").Trim()
        if (-not (Test-Path -LiteralPath $datalabDir)) {
            throw "datalab package directory not found: $datalabDir"
        }
        # Normalise the real on-disk path CASE (e.g. "lib" -> "Lib"). Without
        # this, pytest --import-mode=importlib loads conftest.py twice under two
        # different cases (CLI path vs resolved path) and fails with
        # "AssertionError: ... is already loaded with path ...".
        $datalabDir = (Get-Item -LiteralPath $datalabDir).FullName
        Write-Log ''
        Write-Log "datalab package installed in: $datalabDir"

        # 5b. Inject the test data missing from old wheels.
        # Published wheels (PyPI) and old tags do not ship some data files needed
        # by the tests (several .h5 for the config, the plugin and macro
        # templates). This packaging defect was fixed on develop/main, but
        # polluted the logs of older versions with known, uninteresting test
        # failures. We therefore complete the installed package from the local
        # checkout (canonical source), in COPY-IF-ABSENT mode (we never overwrite
        # a shipped file, which preserves any version-specific data). Skipped for
        # develop/main (already fixed) and disabled via -NoTestDataInjection.
        if (-not $NoTestDataInjection -and $Source -notin @('develop', 'main')) {
            Write-Log ''
            Write-Log ('=' * 78)
            Write-Log '  5b. Injecting test data missing from old wheels'
            Write-Log ('=' * 78)
            # Each entry: source folder (local checkout), file pattern, target
            # sub-folder relative to the installed package, and exclusions.
            $injectMappings = @(
                @{ Src = (Join-Path $RepoRoot 'datalab\data\tests'); Pattern = '*.h5';
                   Rel = 'data\tests'; Exclude = @() }
                @{ Src = (Join-Path $RepoRoot 'datalab\tests\features\plugins\templates');
                   Pattern = '*.py.template'; Rel = 'tests\features\plugins\templates'; Exclude = @() }
                @{ Src = (Join-Path $RepoRoot 'datalab\gui\macros_templates'); Pattern = '*.py';
                   Rel = 'gui\macros_templates'; Exclude = @('__init__.py') }
            )
            $copiedTotal = 0
            foreach ($map in $injectMappings) {
                if (-not (Test-Path -LiteralPath $map.Src)) {
                    Write-Log "  (source missing, skipped): $($map.Src)"
                    continue
                }
                $dstDir = Join-Path $datalabDir $map.Rel
                $srcFiles = @(Get-ChildItem -LiteralPath $map.Src -Filter $map.Pattern -File `
                        -ErrorAction SilentlyContinue |
                        Where-Object { $map.Exclude -notcontains $_.Name })
                foreach ($srcFile in $srcFiles) {
                    $dstFile = Join-Path $dstDir $srcFile.Name
                    if (Test-Path -LiteralPath $dstFile) { continue }
                    if (-not (Test-Path -LiteralPath $dstDir)) {
                        New-Item -ItemType Directory -Path $dstDir -Force | Out-Null
                    }
                    Copy-Item -LiteralPath $srcFile.FullName -Destination $dstFile -Force
                    Write-Log "  + $($map.Rel)\$($srcFile.Name)"
                    $copiedTotal++
                }
            }
            if ($copiedTotal -eq 0) {
                Write-Log '  No file to inject (package already complete).'
            }
            else {
                Write-Log "  $copiedTotal test-data file(s) injected."
            }
        }
        else {
            Write-Log ''
            Write-Log "Test-data injection skipped (source '$Source')."
        }


        # Inventory of installed versions (for the header + log)
        Write-Log ''
        Write-Log ('=' * 78)
        Write-Log '  Package inventory (pip freeze)'
        Write-Log ('=' * 78)
        $freeze = & $venvPy -m pip freeze 2>&1
        $freeze | ForEach-Object { Write-Log "$_" }

        # 6. Run pytest against the installed package.
        # pytest displays paths relative to the INVOCATION directory (cwd), not
        # the rootdir. To shorten the display to "datalab\tests\...", we run
        # pytest FROM the site-packages folder (Push-Location below) and pass the
        # package as a RELATIVE path.
        $sitePackages = Split-Path -Parent $datalabDir
        $datalabLeaf  = Split-Path -Leaf $datalabDir   # "datalab"
        $pytestArgsFull = @('-m', 'pytest', $datalabLeaf, '--import-mode=importlib', '-p', 'no:cacheprovider')
        # By default: duration of EACH test, no threshold (all setup/call/teardown
        # phases). The user can override via -PytestArgs '--durations=N'.
        if (-not ($PytestArgs -match '^--durations')) {
            $pytestArgsFull += @('--durations=0', '--durations-min=0')
        }
        # Detailed per-test JSON report (by default; -NoJsonReport disables it).
        # Requires the pytest-json-report plugin installed at step 4d.
        if (-not $NoJsonReport) {
            $jsonReport = Join-Path $RunsDir "$venvName.report.json"
            $pytestArgsFull += @('--json-report', "--json-report-file=$jsonReport")
        }
        # Anti-hang guard: maximum time per test (pytest-timeout, 'thread' method
        # on Windows). Prevents a modal dialog left open in native mode from
        # freezing the cell for hours. Can be overridden via -PytestArgs.
        if ($TimeoutSeconds -gt 0 -and -not ($PytestArgs -match '^--timeout')) {
            $pytestArgsFull += @("--timeout=$TimeoutSeconds", '--timeout-method=thread')
        }
        # "On-the-fly" GDI probe: injects the standalone plugin
        # scripts\pytest_gdi_probe.py via -p (without installing it into the
        # venv). The script folder is added to PYTHONPATH for the duration of the
        # pytest call so that `-p pytest_gdi_probe` is importable. The SAME local
        # version thus instruments every source (pypi/main/develop) identically,
        # without touching the installed DataLab code.
        if ($GdiProbe) {
            $env:PYTHONPATH = if ([string]::IsNullOrEmpty($env:PYTHONPATH)) { $ScriptDir }
            else { "$ScriptDir;$env:PYTHONPATH" }
            $gdiPrefix = Join-Path $RunsDir $venvName
            $pytestArgsFull += @('-p', 'pytest_gdi_probe', '--gdi-probe', "--gdi-csv-prefix=$gdiPrefix")
        }
        # Higher verbosity to identify the LAST test run before a hard crash (the
        # test name appears before its execution with -v).
        if (-not ($PytestArgs -match '^-(v|q|--verbose|--quiet)$')) { $pytestArgsFull += '-v' }
        if (-not ($PytestArgs -match '^-r')) { $pytestArgsFull += '-rA' }
        if (-not ($PytestArgs -match '^--tb')) { $pytestArgsFull += '--tb=long' }
        # Capture local variables in error tracebacks.
        if ($PytestArgs -notcontains '-l' -and $PytestArgs -notcontains '--showlocals') {
            $pytestArgsFull += '-l'
        }
        if ($ShowWindows) { $pytestArgsFull += '--show-windows' }
        $pytestArgsFull += $PytestArgs

        # Memory sampler: a background job reads, every second, the cumulative
        # working set of the venv's python.exe processes, to spot an abnormal
        # rise (leak) just before a crash. The peak is logged. The job stops as
        # soon as a sentinel file appears.
        $memStop = Join-Path $RunsDir "$venvName.memstop"
        if (Test-Path -LiteralPath $memStop) { Remove-Item -LiteralPath $memStop -Force }
        $memJob = Start-Job -ScriptBlock {
            param($VenvRoot, $StopFile)
            $peak = [int64]0
            # We use Win32_Process (CIM) rather than Get-Process: the
            # Process.Path (MainModule.FileName) property often returns $null
            # ("Access denied") and the main pytest process was then never
            # counted, giving an absurd peak (~12 MB). Win32_Process's
            # ExecutablePath / WorkingSetSize are reliable.
            while (-not (Test-Path -LiteralPath $StopFile)) {
                try {
                    $procs = Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
                        Where-Object {
                            $_.ExecutablePath -and $_.ExecutablePath.StartsWith(
                                $VenvRoot, [System.StringComparison]::OrdinalIgnoreCase)
                        }
                    if ($procs) {
                        $sum = ($procs | Measure-Object -Property WorkingSetSize -Sum).Sum
                        if ($sum -gt $peak) { $peak = [int64]$sum }
                    }
                }
                catch { }
                Start-Sleep -Milliseconds 1000
            }
            return $peak
        } -ArgumentList $venvDir, $memStop

        # GDI note: no external PowerShell sampler. The GetGuiResources/OpenProcess
        # sampler had proved unreliable (read 0 even in native mode, while the
        # in-process probe measured the real peak). The GDI peak is now read AFTER
        # pytest from the timeline produced by the in-process pytest_gdi_probe
        # (<venv>.gdi-timeline.csv), a reliable source.

        $testSw = [System.Diagnostics.Stopwatch]::StartNew()
        # Run pytest from site-packages for short displayed paths
        # ("datalab\tests\..."). DataLab's pyproject.toml config is still
        # discovered normally by walking up the tree.
        Push-Location -LiteralPath $sitePackages
        try {
            $pytestExit = Invoke-Step -Title '6. Running pytest' -Exe $venvPy `
                -Arguments $pytestArgsFull -AllowFailure
        }
        finally {
            Pop-Location
        }
        $testSw.Stop()
        $totalSw.Stop()

        # Stop the memory sampler and read the peak.
        New-Item -ItemType File -Path $memStop -Force | Out-Null
        try {
            $peakBytes = Receive-Job -Job $memJob -Wait -ErrorAction SilentlyContinue
            if ($peakBytes) { $peakMB = [math]::Round([int64]$peakBytes / 1MB, 1) }
        }
        catch { }
        finally {
            Remove-Job -Job $memJob -Force -ErrorAction SilentlyContinue
            Remove-Item -LiteralPath $memStop -Force -ErrorAction SilentlyContinue
        }

        # Read the GDI peak from the in-process timeline (max of gdi_count),
        # produced by the pytest_gdi_probe plugin. Reliable, unlike an external
        # sampler. Absent if the run crashed before sessionfinish.
        if ($GdiProbe -and $gdiPrefix) {
            $gdiTimeline = "$gdiPrefix.gdi-timeline.csv"
            if (Test-Path -LiteralPath $gdiTimeline) {
                try {
                    $gdiMax = (Import-Csv -LiteralPath $gdiTimeline |
                        Measure-Object -Property gdi_count -Maximum).Maximum
                    if ($null -ne $gdiMax) { $gdiPeak = [int]$gdiMax }
                }
                catch { }
            }
        }

        # Explicit logging of the exit code (crucial in case of a crash).
        $exitInfo = Get-ExitCodeInfo -Code $pytestExit
        Write-Log ''
        Write-Log ('-' * 78)
        Write-Log ("  pytest exit code: $pytestExit  ->  $exitInfo")
        if ($null -ne $peakMB) {
            Write-Log ("  Memory peak (pytest+children) : $peakMB MB")
        }
        if ($null -ne $gdiPeak) {
            Write-Log ("  GDI objects peak (venv processes) : $gdiPeak")
        }
        if (Test-AbnormalExit -Code $pytestExit) {
            Write-Log '  /!\ ABNORMAL end: the Python process probably crashed (native crash).'
            Write-Log '      If a faulthandler traceback appears above, it points to the crash site.'
            Write-Log '      The venv will be KEPT for investigation (see below).'
        }
        Write-Log ('-' * 78)


        # --- Header construction ---------------------------------------------
        $cpu = Get-CimInstance -ClassName Win32_Processor | Select-Object -First 1
        $cs  = Get-CimInstance -ClassName Win32_ComputerSystem
        $os  = Get-CimInstance -ClassName Win32_OperatingSystem
        $ramTotalGB = [math]::Round($cs.TotalPhysicalMemory / 1GB, 1)
        $ramFreeGB  = [math]::Round(($os.FreePhysicalMemory * 1KB) / 1GB, 1)
        $pyVersionFull = (& $venvPy -c "import platform; print(platform.python_version())").Trim()
        $qtInfo = (& $venvPy -c "import qtpy; print(qtpy.API_NAME, qtpy.PYQT_VERSION or qtpy.PYSIDE_VERSION, '/ Qt', qtpy.QT_VERSION)" 2>&1)
        $exitInfo = Get-ExitCodeInfo -Code $pytestExit
        $testStatus = if ($pytestExit -eq 0) { 'SUCCESS (all tests pass)' }
        elseif (Test-AbnormalExit -Code $pytestExit) { "NATIVE CRASH (code=$pytestExit)" }
        else { "TEST FAILURE (pytest code=$pytestExit)" }

        $header = New-Object System.Collections.Generic.List[string]
        $header.Add('#' * 78)
        $header.Add('# DataLab TEST REPORT')
        $header.Add('#' * 78)
        $header.Add('')
        $header.Add('[Run characteristics]')
        $header.Add(("  {0,-20}: {1}" -f 'Date (start)', (Get-Date).ToString('yyyy-MM-dd HH:mm:ss')))
        $header.Add(("  {0,-20}: {1}" -f 'Source', $Source))
        if ($TagCombos.Contains($Source)) {
            $header.Add(("  {0,-20}: {1} / {2}" -f 'DataLab / Sigima', $Source, $TagCombos[$Source]))
        }
        elseif ($Source -ne 'pypi') {
            $header.Add(("  {0,-20}: {1}" -f 'git branch', $Branch))
        }
        $header.Add(("  {0,-20}: {1}" -f 'PythonQwt', $(if ($UsePythonQwtBranch) { "git @ $PythonQwtBranch" } else { 'pypi' })))
        $header.Add(("  {0,-20}: {1}" -f 'Python requested', $PyVersion))
        $header.Add(("  {0,-20}: {1}" -f 'Python (actual)', $pyVersionFull))
        $header.Add(("  {0,-20}: {1}" -f 'Interpreter', $PyExe))
        $header.Add(("  {0,-20}: {1}" -f 'Qt binding', $qtInfo))
        $header.Add(("  {0,-20}: {1}" -f 'Window mode', $(if ($ShowWindows) { 'show-windows (native)' } else { 'offscreen' })))
        $header.Add(("  {0,-20}: {1}" -f 'Venv', $venvDir))
        $header.Add(("  {0,-20}: {1}" -f 'Venv deleted', $(if ($DeleteVenv) { 'yes (at end of run)' } else { 'no' })))
        $header.Add(("  {0,-20}: {1}" -f 'Requirements', $RequirementsPath))
        $header.Add(("  {0,-20}: {1}" -f 'pytest arguments', ($pytestArgsFull -join ' ')))
        if (-not $NoJsonReport) {
            $header.Add(("  {0,-20}: {1}" -f 'JSON duration report', $jsonReport))
        }
        if ($GdiProbe) {
            $header.Add(("  {0,-20}: {1}.gdi-pertest.csv / .gdi-timeline.csv" -f 'GDI probe (CSV)', (Join-Path $RunsDir $venvName)))
        }
        $header.Add('')
        $header.Add('[Key package versions]')
        foreach ($p in @('datalab-platform', 'Sigima', 'PlotPy', 'guidata', 'PythonQwt', 'PyQt5', 'NumPy', 'SciPy', 'scikit-image')) {
            $header.Add(("  {0}" -f (Get-FreezeVersion -Freeze $freeze -PackageName $p)))
        }
        $header.Add('')
        $header.Add('[Test result]')
        $header.Add(("  {0,-20}: {1}" -f 'Status', $testStatus))
        $header.Add(("  {0,-20}: {1} ({2})" -f 'Exit code', $pytestExit, $exitInfo))
        if ($null -ne $peakMB) {
            $header.Add(("  {0,-20}: {1} MB" -f 'Process memory peak', $peakMB))
        }
        if ($null -ne $gdiPeak) {
            $header.Add(("  {0,-20}: {1}" -f 'GDI objects peak', $gdiPeak))
        }
        $header.Add(("  {0,-20}: {1}" -f 'Tests duration', (Format-Duration $testSw.Elapsed)))
        $header.Add(("  {0,-20}: {1}" -f 'Total duration (run)', (Format-Duration $totalSw.Elapsed)))
        $header.Add('')
        $header.Add('[Machine]')
        $header.Add(("  {0,-20}: {1}" -f 'Name', $env:COMPUTERNAME))
        $header.Add(("  {0,-20}: {1} (version {2}, build {3})" -f 'System', $os.Caption, $os.Version, $os.BuildNumber))
        $header.Add(("  {0,-20}: {1}" -f 'Architecture', $env:PROCESSOR_ARCHITECTURE))
        $header.Add(("  {0,-20}: {1}" -f 'CPU', $cpu.Name.Trim()))
        $header.Add(("  {0,-20}: {1} / {2}" -f 'Cores / threads', $cpu.NumberOfCores, $cpu.NumberOfLogicalProcessors))
        $header.Add(("  {0,-20}: {1} GB" -f 'Total RAM', $ramTotalGB))
        $header.Add(("  {0,-20}: {1} GB" -f 'Free RAM (end)', $ramFreeGB))
        $header.Add('')
        $header.Add('#' * 78)
        $header.Add('# DETAILED OUTPUT')
        $header.Add('#' * 78)
        $header.Add('')

        # --- Assemble the final log ------------------------------------------
        Set-Content -LiteralPath $finalLog -Value $header -Encoding UTF8
        Get-Content -LiteralPath $partLog | Add-Content -LiteralPath $finalLog -Encoding UTF8
    }
    finally {
        if (Test-Path -LiteralPath $partLog) { Remove-Item -LiteralPath $partLog -Force }
        # On a native crash (abnormal end), we KEEP the venv so it can be replayed
        # manually and investigated, even if deletion was requested. Otherwise,
        # delete it if requested (the .log file is always kept).
        $abnormal = Test-AbnormalExit -Code $pytestExit
        if ($DeleteVenv -and -not $abnormal -and (Test-Path -LiteralPath $venvDir)) {
            Write-Host "  Deleting venv: $venvDir" -ForegroundColor DarkGray
            Remove-Item -LiteralPath $venvDir -Recurse -Force -ErrorAction SilentlyContinue
        }
        elseif ($DeleteVenv -and $abnormal) {
            Write-Host "  /!\ Abnormal end (crash): venv KEPT for investigation -> $venvDir" -ForegroundColor Yellow
        }
    }

    # Console summary for this run
    $venvKept = (-not $DeleteVenv) -or (Test-AbnormalExit -Code $pytestExit)
    Write-Host ''
    Write-Host ('=' * 78)
    Write-Host ("  [py $PyVersion] Log : $finalLog")
    if ((-not $NoJsonReport) -and $jsonReport -and (Test-Path -LiteralPath $jsonReport)) {
        Write-Host ("  [py $PyVersion] JSON report : $jsonReport")
    }
    if ($venvKept -and (Test-Path -LiteralPath $venvDir)) {
        Write-Host "  [py $PyVersion] Venv kept : $venvDir"
    }
    Write-Host ('=' * 78)

    return [pscustomobject]@{
        Exit        = $pytestExit
        Status      = $testStatus
        Duration    = (Format-Duration $testSw.Elapsed)
        DurationSec = [int]$testSw.Elapsed.TotalSeconds
        PeakMB      = $peakMB
        PeakGdi     = $gdiPeak
        LogPath     = $finalLog
    }
}

# --- Full matrix (Source x PythonQwt x mode x Python) ------------------------

function Invoke-DataLabMatrix {
    <#
        Automates the report matrix. Reuses Invoke-DataLabTestRun by setting the
        script-scope variables ($script:*) before each cell, then produces a
        console summary + an aggregated Markdown report in runs\.
    #>
    # --- Axes ---
    $matrixSources = @('develop', 'main')

    $fix = Select-FromMenu -Title 'Matrix: PythonQwt variant to compare against PyPI:' `
        -Options @('v0.15.0', 'master', '(enter...)') `
        -Default 'v0.15.0'
    if ($fix -eq '(enter...)') { $fix = (Read-Host 'PythonQwt branch/tag/commit').Trim() }
    $matrixQwt = @('pypi')
    if (-not [string]::IsNullOrWhiteSpace($fix) -and $fix -ne 'pypi') { $matrixQwt += $fix }

    $modeChoice = Select-FromMenu -Title 'Matrix: window mode(s):' `
        -Options @('offscreen', 'native', 'both') -Default 'offscreen'
    $matrixModes = switch ($modeChoice) {
        'native' { @($true) }
        'both'   { @($false, $true) }
        default  { @($false) }
    }

    $delChoice = Select-FromMenu -Title 'Matrix: delete the venvs at the end of each run?' `
        -Options @('yes', 'no') -Default 'yes'
    $script:DeleteVenv = ($delChoice -eq 'yes')

    $matrixPy = $AllVersions

    # --- Plan + confirmation ---
    $combos = New-Object System.Collections.Generic.List[object]
    foreach ($s in $matrixSources) {
        foreach ($q in $matrixQwt) {
            foreach ($m in $matrixModes) {
                foreach ($v in $matrixPy) {
                    $combos.Add([pscustomobject]@{ Source = $s; Qwt = $q; Native = $m; Py = $v })
                }
            }
        }
    }
    $modeLabels = ($matrixModes | ForEach-Object { if ($_) { 'native' } else { 'offscreen' } }) -join ', '
    Write-Host ''
    Write-Host ('=' * 78) -ForegroundColor Cyan
    Write-Host ("  MATRIX: $($combos.Count) sequential cells") -ForegroundColor Cyan
    Write-Host ("  Sources    : $($matrixSources -join ', ')")
    Write-Host ("  PythonQwt  : $($matrixQwt -join ', ')")
    Write-Host ("  Modes      : $modeLabels")
    Write-Host ("  Python     : $($matrixPy -join ', ')")
    Write-Host ('  Tip        : run on an idle machine (overnight) for comparable durations.')
    Write-Host ('=' * 78) -ForegroundColor Cyan
    $confirm = Read-Host "Run $($combos.Count) runs? (y/N)"
    if ($confirm -notmatch '^(y|yes)$') { Write-Host 'Cancelled.' -ForegroundColor Yellow; return }

    # --- Pre-validate the interpreters ---
    foreach ($v in ($matrixPy | Select-Object -Unique)) {
        if (-not (Test-Path -LiteralPath $PythonPaths[$v])) {
            throw "Python interpreter not found for $v : $($PythonPaths[$v])"
        }
    }

    # --- Sequential execution ---
    $rows = New-Object System.Collections.Generic.List[object]
    $i = 0
    foreach ($c in $combos) {
        $i++
        $script:Source = $c.Source
        $script:Branch = $c.Source
        $script:ShowWindows = [switch]$c.Native
        if ($c.Qwt -eq 'pypi') {
            $script:PythonQwtBranch = 'pypi'
            $script:UsePythonQwtBranch = $false
        }
        else {
            $script:PythonQwtBranch = $c.Qwt
            $script:UsePythonQwtBranch = $true
        }
        $modeLabel = if ($c.Native) { 'native' } else { 'offscreen' }
        Write-Host ''
        Write-Host ('#' * 78) -ForegroundColor Magenta
        Write-Host ("#  [$i/$($combos.Count)] $($c.Source) | qwt=$($c.Qwt) | $modeLabel | py$($c.Py)") -ForegroundColor Magenta
        Write-Host ('#' * 78) -ForegroundColor Magenta

        $r = Invoke-DataLabTestRun -PyVersion $c.Py -PyExe $PythonPaths[$c.Py]
        $rows.Add([pscustomobject]@{
                Source = $c.Source; Qwt = $c.Qwt; Mode = $modeLabel; Py = $c.Py
                Exit = $r.Exit; Status = $r.Status; Duration = $r.Duration
                DurationSec = $r.DurationSec; PeakMB = $r.PeakMB; PeakGdi = $r.PeakGdi
            })
    }

    # --- Console summary ---
    Write-Host ''
    Write-Host ('=' * 78)
    Write-Host '  MATRIX SUMMARY'
    Write-Host ('=' * 78)
    foreach ($row in $rows) {
        $st = if ($row.Exit -eq 0) { 'SUCCESS' }
        elseif ($row.Exit -lt 0 -or $row.Exit -gt 5) { "CRASH($($row.Exit))" }
        else { "FAIL($($row.Exit))" }
        Write-Host ("  {0,-7} qwt={1,-30} {2,-9} py{3,-5} {4,-9} {5}" -f `
                $row.Source, $row.Qwt, $row.Mode, $row.Py, $st, $row.Duration)
    }

    # --- Aggregated Markdown export ---
    $stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
    $mdPath = Join-Path $RunsDir "matrix-$stamp.md"
    $md = New-Object System.Collections.Generic.List[string]
    $md.Add("# DataLab matrix -- $stamp")
    $md.Add('')
    $md.Add('## Per-cell detail')
    $md.Add('')
    $md.Add('| Source | PythonQwt | Mode | Python | Status | Duration | Mem peak | GDI peak |')
    $md.Add('| --- | --- | --- | --- | --- | --- | --- | --- |')
    foreach ($row in $rows) {
        $st = if ($row.Exit -eq 0) { 'OK' }
        elseif ($row.Exit -lt 0 -or $row.Exit -gt 5) { "CRASH $($row.Exit)" }
        else { "FAIL $($row.Exit)" }
        $mem = if ($null -ne $row.PeakMB) { "$($row.PeakMB) MB" } else { '-' }
        $gdi = if ($null -ne $row.PeakGdi) { "$($row.PeakGdi)" } else { '-' }
        $md.Add("| $($row.Source) | $($row.Qwt) | $($row.Mode) | $($row.Py) | $st | $($row.Duration) | $mem | $gdi |")
    }
    $md.Add('')
    $md.Add('## Synthesis (Source x PythonQwt x Mode)')
    $md.Add('')
    $md.Add('| Source | PythonQwt | Mode | Passed | Duration range |')
    $md.Add('| --- | --- | --- | --- | --- |')
    foreach ($g in ($rows | Group-Object Source, Qwt, Mode)) {
        $ok = @($g.Group | Where-Object { $_.Exit -eq 0 }).Count
        $tot = @($g.Group).Count
        $secs = @($g.Group | ForEach-Object { $_.DurationSec } | Where-Object { $null -ne $_ })
        $range = if ($secs.Count) {
            $mn = [int]($secs | Measure-Object -Minimum).Minimum
            $mx = [int]($secs | Measure-Object -Maximum).Maximum
            ('{0:00}m{1:00}s-{2:00}m{3:00}s' -f [int][math]::Floor($mn / 60), ($mn % 60), [int][math]::Floor($mx / 60), ($mx % 60))
        }
        else { '-' }
        $f = $g.Group[0]
        $md.Add("| $($f.Source) | $($f.Qwt) | $($f.Mode) | $ok/$tot | $range |")
    }
    Set-Content -LiteralPath $mdPath -Value $md -Encoding UTF8
    Write-Host ''
    Write-Host ("  Matrix Markdown report : $mdPath") -ForegroundColor Green
    Write-Host ("  Logs                   : $RunsDir")
    Write-Host ('=' * 78)
}

# --- Loop over the requested version(s) --------------------------------------

if ($Matrix) {
    Invoke-DataLabMatrix
    return
}

$versionsToRun = if ($PythonVersion -eq 'all') { $AllVersions } else { @($PythonVersion) }

# Pre-validate the interpreters to fail early.
foreach ($v in $versionsToRun) {
    if (-not (Test-Path -LiteralPath $PythonPaths[$v])) {
        throw "Python interpreter not found for $v : $($PythonPaths[$v])"
    }
}

$results = [ordered]@{}
$globalExit = 0
foreach ($v in $versionsToRun) {
    Write-Host ''
    Write-Host ('#' * 78) -ForegroundColor Magenta
    Write-Host ("#  RUN Python $v  (source: $Source)") -ForegroundColor Magenta
    Write-Host ('#' * 78) -ForegroundColor Magenta

    $res = Invoke-DataLabTestRun -PyVersion $v -PyExe $PythonPaths[$v]
    $results[$v] = $res
    if ($res.Exit -ne 0) { $globalExit = 1 }
}

# --- Global summary ----------------------------------------------------------

Write-Host ''
Write-Host ('=' * 78)
Write-Host '  SUMMARY'
Write-Host ('=' * 78)
foreach ($v in $results.Keys) {
    $code = $results[$v].Exit
    $status = if ($code -eq 0) { 'SUCCESS' }
    elseif ($code -lt 0 -or $code -gt 5) { "NATIVE CRASH (code $code)" }
    else { "TEST FAILURE (code $code)" }
    Write-Host ("  Python {0,-5} : {1}" -f $v, $status)
}
Write-Host ("  Venvs deleted   : {0} (kept on native crash)" -f $(if ($DeleteVenv) { 'yes' } else { 'no' }))
Write-Host ("  Logs            : {0}" -f $RunsDir)
Write-Host ('=' * 78)

exit $globalExit
