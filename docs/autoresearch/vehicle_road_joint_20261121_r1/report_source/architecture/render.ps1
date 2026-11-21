$ErrorActionPreference = 'Stop'
$sourceDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$workspaceDir = (Resolve-Path (Join-Path $sourceDir '..\..\..')).Path
$outputPath = Join-Path $workspaceDir 'outputs\vehicle_road_joint_20261121\architecture.png'
$chromeExe = 'C:\Program Files\Google\Chrome\Application\chrome.exe'
$profileDir = Join-Path $workspaceDir 'work\vehicle_road_joint_20261121\tmp\architecture-chrome'
python (Join-Path $sourceDir 'generate.py')
if ($LASTEXITCODE -ne 0) { throw 'SVG generation failed.' }
$htmlUri = ([System.Uri](Join-Path $sourceDir 'render.html')).AbsoluteUri
$chromeArguments = @('--headless=new', '--disable-gpu', '--hide-scrollbars', '--force-device-scale-factor=1', '--no-first-run', '--no-default-browser-check', "--user-data-dir=$profileDir", "--screenshot=$outputPath", '--window-size=2560,1600', '--virtual-time-budget=1000', $htmlUri)
$chromeProcess = Start-Process -FilePath $chromeExe -ArgumentList $chromeArguments -WindowStyle Hidden -PassThru -Wait
if ($chromeProcess.ExitCode -ne 0) { throw "Chrome rendering failed: $($chromeProcess.ExitCode)" }
Get-Item -LiteralPath $outputPath | Select-Object FullName, Length
