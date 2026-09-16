$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
python src\lightbox_calibration.py @args
