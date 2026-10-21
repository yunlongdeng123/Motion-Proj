$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
Write-Host 'Gaussian Evidence Lab: frozen 30K GPU diagnostics only. No training.'
Write-Host 'Requires wm-vgpu-1011 SSH and an idle GPU. Close with Ctrl+C to release our worker.'
Write-Host 'Open http://127.0.0.1:8770/'
python server.py --port 8770 --remote-cpu --remote-gpu
