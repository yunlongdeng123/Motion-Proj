param([ValidateRange(1,65535)][int]$Port = 8871)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
Write-Host "Gaussian Evidence Lab: http://127.0.0.1:$Port/"
Write-Host 'Adjustable CPU geometry sampling. No training or GPU is required.'
python server.py --port $Port --remote-cpu
