$ErrorActionPreference = 'Stop'
$diagFolder = Split-Path -Parent $MyInvocation.MyCommand.Path
$diagPython = (Get-Command python -ErrorAction Stop).Source
try {
    $diagHealth = Invoke-RestMethod -Uri 'http://127.0.0.1:8767/api/health' -TimeoutSec 2
    if ($diagHealth.app -eq 'Gaussian Evidence Lab') {
        Start-Process 'http://127.0.0.1:8767/'
        return
    }
    throw 'Port 8767 is used by another service. Choose another port with server.py.'
} catch [System.Net.WebException] { }
Start-Process -FilePath $diagPython -ArgumentList @("`"$(Join-Path $diagFolder 'server.py')`"", '--remote-cpu', '--port', '8767') -WorkingDirectory $diagFolder -WindowStyle Hidden
Start-Sleep -Milliseconds 900
Start-Process 'http://127.0.0.1:8767/'
