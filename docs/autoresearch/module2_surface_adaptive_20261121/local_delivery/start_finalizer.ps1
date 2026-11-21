$ErrorActionPreference='Stop'
$taskRoot=Split-Path -Parent $MyInvocation.MyCommand.Path
$taskProject=Split-Path -Parent (Split-Path -Parent $taskRoot)
$taskLogs=Join-Path $taskProject 'outputs/road_material_three_rounds_20261121/surface_adaptive'
New-Item -ItemType Directory -Force -Path $taskLogs | Out-Null
$taskLock=Join-Path $taskLogs 'finalizer.lock'
if(Test-Path -LiteralPath $taskLock){
    $taskLockPid=[int](Get-Content -LiteralPath $taskLock)
    $taskExisting=Get-Process -Id $taskLockPid -ErrorAction SilentlyContinue
    if($taskExisting){Write-Output ('Already running: '+$taskLockPid);exit 0}
    Remove-Item -LiteralPath $taskLock
}
$taskPython=(Get-Command python).Source
$taskScript=Join-Path $taskRoot 'finalizer.py'
$taskArguments=@('-u',('"'+$taskScript+'"'))
$taskStamp=Get-Date -Format 'yyyyMMdd_HHmmss'
$taskWorker=Start-Process -FilePath $taskPython -ArgumentList $taskArguments -WindowStyle Hidden -RedirectStandardOutput (Join-Path $taskLogs ($taskStamp+'.stdout.log')) -RedirectStandardError (Join-Path $taskLogs ($taskStamp+'.stderr.log')) -PassThru
Write-Output ('Started surface finalizer: '+$taskWorker.Id)
