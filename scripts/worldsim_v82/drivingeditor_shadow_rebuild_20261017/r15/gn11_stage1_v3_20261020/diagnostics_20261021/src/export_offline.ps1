param(
    [string]$Destination = $PSScriptRoot,
    [string]$HostAlias = 'wm-vgpu-1011',
    [string]$RemoteSource = '/root/autodl-tmp/motion_proj_v82_gn11_v3/scripts/worldsim_v82/drivingeditor_shadow_rebuild_20261017/r15/gn11_stage1_v3_20261020/diagnostics_20261021/src',
    [string[]]$Arms = @()
)
$ErrorActionPreference = 'Stop'
if ($HostAlias -notmatch '^[A-Za-z0-9_.-]+$' -or $RemoteSource -notmatch '^/[A-Za-z0-9_./-]+$') { throw 'Invalid SSH alias or source path' }
foreach ($Arm in $Arms) { if ($Arm -notmatch '^[A-Za-z0-9_-]+$') { throw 'Invalid experiment name' } }
if (-not (Test-Path -LiteralPath (Join-Path $Destination 'index.html'))) { throw 'Destination must be an existing diagnostic package with index.html' }
$ExportId = [DateTime]::UtcNow.ToString('yyyyMMdd_HHmmss')
$RemoteOutput = "/root/autodl-tmp/gsdiag_offline_$ExportId"
$ArmArguments = if ($Arms.Count) { '--arms ' + ($Arms -join ' ') } else { '' }
Write-Host 'Exporting offline geometry on AutoDL CPU. Original checkpoints and native evidence stay unchanged.'
ssh $HostAlias "cd '$RemoteSource' && CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /root/autodl-tmp/envs/drivingeditor/bin/python export_offline.py --output '$RemoteOutput' $ArmArguments"
if ($LASTEXITCODE -ne 0) { throw 'AutoDL export failed; local package has not been updated' }
scp -C -r "${HostAlias}:$RemoteOutput/." $Destination
if ($LASTEXITCODE -ne 0) { throw 'Download incomplete; rerun to finish downloading before opening the package' }
$IndexPath = Join-Path $Destination 'index.html'
$IndexText = [IO.File]::ReadAllText($IndexPath)
if (-not $IndexText.Contains('data/geometry_index.js')) {
    $IndexText = $IndexText.Replace('<script src="sampling.js">', '<script src="data/geometry_index.js"></script><script src="offline_geometry.js"></script><script src="sampling.js">')
    [IO.File]::WriteAllText($IndexPath, $IndexText, [Text.UTF8Encoding]::new($false))
}
Write-Host "Updated $IndexPath. Open it directly; no local server is required."
