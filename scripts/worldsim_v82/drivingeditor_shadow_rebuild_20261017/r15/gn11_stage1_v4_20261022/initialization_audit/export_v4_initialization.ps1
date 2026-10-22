param(
    [string]$Destination=$PSScriptRoot,
    [string]$HostAlias='wm-vgpu-1011',
    [string]$RemoteSource='/root/autodl-tmp/motion_proj_v82_gn11_v4/scripts/worldsim_v82/drivingeditor_shadow_rebuild_20261017/r15/gn11_stage1_v4_20261022/initialization_audit'
)
$ErrorActionPreference='Stop'
if ($HostAlias -notmatch '^[A-Za-z0-9_.-]+$' -or $RemoteSource -notmatch '^/[A-Za-z0-9_./-]+$') { throw 'Invalid remote path or alias' }
if (-not (Test-Path -LiteralPath (Join-Path $Destination 'index.html'))) { throw 'Destination must be the existing diagnostic package' }
$TaskExportId=[DateTime]::UtcNow.ToString('yyyyMMdd_HHmmss')
$TaskRemoteOutput="/root/autodl-tmp/gsdiag_v4_initialization_$TaskExportId"
ssh $HostAlias "cd '$RemoteSource' && CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /root/autodl-tmp/envs/drivingeditor/bin/python export_initialization_audit.py --output '$TaskRemoteOutput'"
if ($LASTEXITCODE -ne 0) { throw 'Read-only initialization audit failed; local package was not updated' }
scp -C -r "${HostAlias}:$TaskRemoteOutput/." $Destination
if ($LASTEXITCODE -ne 0) { throw 'Download incomplete; rerun before opening the page' }
$TaskIndex=Join-Path $Destination 'index.html'
$TaskHtml=[IO.File]::ReadAllText($TaskIndex)
$TaskHtml=$TaskHtml.Replace('<script src="v4_embed.js?v=embed1"></script>','').Replace('href="#v4-initialization"','href="v4_initialization.html"')
if (-not $TaskHtml.Contains('data/v4_experiments.js')) { $TaskHtml=$TaskHtml.Replace('<script src="data/scene.js"></script>','<script src="data/scene.js"></script><script src="data/v4_experiments.js?v=slots1"></script>') }
$TaskHtml=[regex]::Replace($TaskHtml,'src="app\.js(?:\?[^"]*)?"','src="app.js?v=v4slots1"')
$TaskHtml=[regex]::Replace($TaskHtml,'src="gpu\.js(?:\?[^"]*)?"','src="gpu.js?v=v4slots1"')
[IO.File]::WriteAllText($TaskIndex,$TaskHtml,[Text.UTF8Encoding]::new($false))
Write-Host 'Updated V4 initialization audit in the original package. Open v4_initialization.html directly.'
