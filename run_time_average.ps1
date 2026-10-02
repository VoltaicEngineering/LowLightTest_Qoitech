$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
python src\time_average.py @args
