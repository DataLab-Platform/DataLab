# Machine-specific configuration for scripts/Run-DataLabTests.ps1
#
# This is a TEMPLATE. Copy it to "Run-DataLabTests.config.psd1" (same folder)
# and edit the values to match your machine. The real config file is git-ignored
# so your local paths never end up in the repository:
#
#     Copy-Item .\Run-DataLabTests.config.template.psd1 .\Run-DataLabTests.config.psd1
#
# The script fails early with guidance if "Run-DataLabTests.config.psd1" is
# missing.

@{
    # Map of "Python version" -> absolute path to the interpreter (python.exe).
    #
    # Only the versions you actually have installed need to be listed; the script
    # validates the path of each version it is asked to run (and the "all" /
    # matrix modes iterate over the versions listed here, in ascending order).
    #
    # The example paths below point to WinPython distributions; adapt them.
    PythonPaths = @{
        '3.9'  = 'C:\path\to\python.exe'
        '3.10' = 'C:\path\to\python.exe'
        '3.11' = 'C:\path\to\python.exe'
        '3.12' = 'C:\path\to\python.exe'
        '3.13' = 'C:\path\to\python.exe'
        '3.14' = 'C:\path\to\python.exe'
    }

    # Optional: override the DataLab requirements.txt path. Leave this entry
    # commented out (or absent) to use "<repo>\requirements.txt", resolved
    # relative to the script location.
    # RequirementsPath = 'C:\custom\path\requirements.txt'
}
