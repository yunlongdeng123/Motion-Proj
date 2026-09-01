# WorldSim V7 P0 storage cleanup manifest

- task: `WS-V7-P0-STORAGE-CLEANUP-01`
- date: `2026-09-01`
- host: AutoDL `wm-3090-0811`
- pre-cleanup filesystem: `/root/autodl-tmp` = `600G total / 462G used / 139G available / 77%`
- audited reclaim: `23,027,117,355 bytes` (`21.446 GiB`)
- status: `executed_and_verified`.
- post-cleanup filesystem: `/root/autodl-tmp` = `600G total / 440G used / 161G available / 74%`.

## Deletion set

| Absolute path | Bytes | Why it is outside V7 | Recovery |
|---|---:|---|---|
| `/root/autodl-tmp/models/hunyuan3d-omni-70e803bfb4e1` | 13,514,733,097 | V6.1-only Hunyuan3D capability/ME-2 weights; no V7 code, config, paper, or data-prep reference | Re-download `tencent/Hunyuan3D-Omni` at revision `70e803bfb4e127d534049d8ab8c8cb511780d485`; the V6.1 config and official source checkout are retained |
| `/root/autodl-tmp/models/hunyuan3d-2.1-0b94677654c5` | 8,022,040,597 | V6.1-only image arm weights; no V7 reference | Re-download `tencent/Hunyuan3D-2.1` at revision `0b94677654c57bb9a6b6845cd7b704ccf551d327`; the V6.1 config and official source checkout are retained |
| `/root/autodl-tmp/models/worldsim_v32/harmonizer` | 1,448,893,309 | V3.2/V3.3 appearance harmonizer weight bundle; V7 physical compiler does not consume appearance enhancement as occupancy | Re-download `nvidia/Harmonizer`; repository configs and `/root/autodl-tmp/third_party/worldsim_v32/harmonizer` are retained |
| `/root/autodl-tmp/av2_test` | 41,450,352 | Interrupted speed-test fragment for a train log; not part of the frozen 30-log AV2 val cohort | Re-run unsigned S3 copy from `s3://argoverse/datasets/av2/sensor/train/00a6ffc1-6ce9-3bc3-a060-6006e9893a1a/` |

All four targets resolved exactly to the listed absolute paths, were ordinary directories owned by `root:root`, and were
checked before deletion. The first three are immutable, publicly recoverable model weights; their configs, research records,
run evidence, source checkouts, and recovery revisions remain in place.

## Explicitly preserved

- `/root/autodl-tmp/data/worldsim_v4` and all nuScenes raw/processed inputs required by V7;
- all `/root/autodl-tmp/runs/worldsim_v4`, V5, V6, V6.7 canonical evidence;
- all repository configs, docs, failure ledgers, and historical scripts;
- all V6.7/V7 model assets and the remotely installed `/root/autodl-tmp/bin/s5cmd`;
- local `D:\datasets\av2` frozen-cohort download.

Pre-V5 runs were not deleted: `docs/ARTIFACT_RETENTION.md` marks the V4 canonical runs/checkpoints as dependencies of later
config chains. This cleanup therefore removes only re-downloadable legacy weight bundles and the disposable AV2 speed-test
fragment, not unique research evidence.

## Execution result

All four exact targets were deleted sequentially and verified absent. Available space rose from `139G` to `161G`; the
filesystem utilization fell from `77%` to `74%`. No run directory, dataset directory, repository history, or V7 dependency
was touched.
