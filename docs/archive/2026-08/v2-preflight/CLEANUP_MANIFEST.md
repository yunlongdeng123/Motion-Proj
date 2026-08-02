# V2 启动前存储清理清单

- 日期：2026-08-02
- 状态：`done`
- 授权：用户要求删除占空间的中间产物，只保留 V2 计划需要的资产
- 项目：`/root/autodl-tmp/motion_proj`
- 删除前数据盘：250G total / 185G used / 66G available
- 预计删除：`63,284,939,678` bytes（约 58.93 GiB，目录硬链接/文件系统统计可能使实际释放略有差异）
- docs 恢复点：`/root/autodl-tmp/motion_proj_backups/docs-before-v2-preflight-2026-08-02/`

## 1. 删除前安全检查

- GPU 0 MiB；无训练、推理、下载或评测进程；
- 无活跃研究 tmux/controller；
- Cursor/Jupyter/TensorBoard 为用户服务，不处理；
- 所有目标均已解析为 `/root/autodl-tmp/` 下的字面绝对路径；
- Git 当前只有未跟踪的 V2 计划；删除前先提交本清单与文档归档。

## 2. 精确删除目标

| Bytes | 绝对路径 | 原因 / 恢复方式 |
|---:|---|---|
| 8,570,158,353 | `/root/autodl-tmp/pip_cache` | 旧 pip HTTP cache；V2 使用 `/root/autodl-tmp/cache/pip` + 镜像重建 |
| 5,346,380,054 | `/root/autodl-tmp/conda_pkgs` | 旧 Conda package cache；V2 使用项目级 TUNA 与 `CONDA_PKGS_DIRS` 重建 |
| 8,481,779,460 | `/root/autodl-tmp/envs/dggt` | V1 失败环境；M1 明确要求新建 `dggt-v2` |
| 72,472,722 | `/root/autodl-tmp/envs/dggt.interrupted-20260729T094923` | 中断环境副本；失败证据已在正式 run |
| 6,698,152,563 | `/root/autodl-tmp/envs/resim` | 已关闭 ReSim V6 环境；V2 不使用 |
| 5,867,820,963 | `/root/autodl-tmp/envs/adgs-dpt` | AD-GS 已完成的可重建预处理环境；V2 不重复训练 |
| 1,341,395,378 | `/root/autodl-tmp/checkpoints/depth_anything_v2` | DPT 可按固定 source/weight 重下；processed depth 已保留 |
| 5,394,489,250 | `/root/autodl-tmp/checkpoints/dggt/model_latest_nuscenes.pt.partial` | 不完整下载；完整 preload 已另存并校验 |
| 1,096,130,347 | `/root/autodl-tmp/data/dynamic_recon/processed_failed` | 两次失败预处理大副本；日志/manifest/terminal 保留 |
| 1,298,140,749 | `/root/autodl-tmp/runs/dynamic_recon/DR-M3-ADGS-0230-01/20260727T195611__scene0230__s0-r3/model_100` | profiling 中间 checkpoint/render |
| 1,394,877,620 | `/root/autodl-tmp/runs/dynamic_recon/DR-M3-ADGS-0230-01/20260727T195611__scene0230__s0-r3/model_1000` | profiling 中间 checkpoint/render |
| 1,243,016,754 | `/root/autodl-tmp/runs/dynamic_recon/DR-M4-ADGS-6SCENE-01/20260728T131642__scene0242__s0-r3-wm3090/model_100` | profiling 中间 checkpoint/render |
| 1,301,521,503 | `/root/autodl-tmp/runs/dynamic_recon/DR-M4-ADGS-6SCENE-01/20260728T131642__scene0242__s0-r3-wm3090/model_1000` | profiling 中间 checkpoint/render |
| 1,326,233,234 | `/root/autodl-tmp/runs/dynamic_recon/DR-M4-ADGS-6SCENE-01/20260728T155551__scene0255__s0-wm3090/model_100` | profiling 中间 checkpoint/render |
| 1,426,841,101 | `/root/autodl-tmp/runs/dynamic_recon/DR-M4-ADGS-6SCENE-01/20260728T155551__scene0255__s0-wm3090/model_1000` | profiling 中间 checkpoint/render |
| 681,327,242 | `/root/autodl-tmp/runs/dynamic_recon/DR-M4-ADGS-6SCENE-01/20260728T155551__scene0255__s0-wm3090/work` | 成功后遗留临时 mask work；processed 正本保留 |
| 1,302,269,747 | `/root/autodl-tmp/runs/dynamic_recon/DR-M4-ADGS-6SCENE-01/20260728T193546__scene0295__s0-wm3090/model_100` | profiling 中间 checkpoint/render |
| 1,400,518,321 | `/root/autodl-tmp/runs/dynamic_recon/DR-M4-ADGS-6SCENE-01/20260728T193546__scene0295__s0-wm3090/model_1000` | profiling 中间 checkpoint/render |
| 679,654,068 | `/root/autodl-tmp/runs/dynamic_recon/DR-M4-ADGS-6SCENE-01/20260728T193546__scene0295__s0-wm3090/work` | 成功后遗留临时 mask work；processed 正本保留 |
| 1,281,647,590 | `/root/autodl-tmp/runs/dynamic_recon/DR-M4-ADGS-6SCENE-01/20260729T012643__scene0518__s0-wm3090/model_100` | profiling 中间 checkpoint/render |
| 1,370,508,560 | `/root/autodl-tmp/runs/dynamic_recon/DR-M4-ADGS-6SCENE-01/20260729T012643__scene0518__s0-wm3090/model_1000` | profiling 中间 checkpoint/render |
| 649,213,384 | `/root/autodl-tmp/runs/dynamic_recon/DR-M4-ADGS-6SCENE-01/20260729T012643__scene0518__s0-wm3090/work` | 成功后遗留临时 mask work；processed 正本保留 |
| 1,192,465,527 | `/root/autodl-tmp/runs/dynamic_recon/DR-M4-ADGS-6SCENE-01/20260729T054205__scene0749__s0-wm3090/model_100` | profiling 中间 checkpoint/render |
| 1,247,560,460 | `/root/autodl-tmp/runs/dynamic_recon/DR-M4-ADGS-6SCENE-01/20260729T054205__scene0749__s0-wm3090/model_1000` | profiling 中间 checkpoint/render |
| 532,533,555 | `/root/autodl-tmp/runs/dynamic_recon/DR-M4-ADGS-6SCENE-01/20260729T054205__scene0749__s0-wm3090/work` | 成功后遗留临时 mask work；processed 正本保留 |
| 641,549,519 | `/root/autodl-tmp/runs/dynamic_recon/DR-M3-ADGS-0230-01/20260727T182247__scene0230__s0-r2/work` | blocked run 大型 work；轻量失败证据保留 |
| 939,888,677 | `/root/autodl-tmp/runs/dynamic_recon/DR-M4-ADGS-6SCENE-01/20260727T235743__scene0242__s0/model_100` | blocked run profiling checkpoint；轻量失败证据保留 |
| 506,392,977 | `/root/autodl-tmp/runs/dynamic_recon/DR-M4-ADGS-6SCENE-01/20260727T235743__scene0242__s0/work` | blocked run 大型 work；轻量失败证据保留 |

## 3. 明确保留

- 六个正式 `model_60000/`，包括 checkpoint、official test/train renders、GT、视频和结果；
- 所有 run 的 `manifest/resolved/source_snapshot/stages/logs/metrics/summary/terminal`；
- AD-GS 39G processed 数据，因为 M1 adapter 仍直接依赖 `image/sky/semantic`；
- raw subset、manifests、历史 OccGS/DriveStudio 最小资产；
- DGGT full preload：`5,411,266,466` bytes，SHA-256
  `fd15644b3a878849470cbf5f0f9eae39167cfec1b853092898ae754c4f3acde9`；
- `motionproj/drivestudio/adgs/adgs-sam` 环境；
- Grounding DINO HF cache、Grounded-SAM-2 与 CoTracker3 权重；
- 原始数据、第三方源码、Git 文档/归档和人工材料。

## 4. 完成后验证

- 所有清单目标均不存在；按删除前统计移除 `63,284,939,678` bytes；
- 数据盘从 `185G used / 66G available` 变为 `127G used / 124G available`（51%）；
- `/root/autodl-tmp/runs/dynamic_recon` 从 30G 降至 11G；
- 六个 `model_60000` 均存在，逐 run `terminal.status=done`，point cloud 非空；
- 每个最终模型仍有 42 张 official test render 和 138 张 train render，`results.json` 可解析；
- aggregate `summary.json` 为 `status=done / coverage=1.0 / all_gates_passed=true`；
- DGGT full preload 仍为 `5,411,266,466` bytes，SHA-256 复核通过；
- raw subset 2.9G、AD-GS processed 39G、manifests 1.7M 均驻留；
- `motionproj/drivestudio/adgs/adgs-sam` 四个核心环境均驻留；
- 直接执行 `pytest` 暴露 console-script 不含仓库根目录的 import 路径问题；改用规范命令
  `python -m pytest -q tests/test_dr_pseudo_tracks.py tests/test_v71_actor_registry.py` 后 `7 passed`；
  V2 计划和迁移指南已据此校准；
- 活动文档和两个新归档入口的相对 Markdown 链接检查为 `missing_links=0`。

删除的目录/文件不进入回收站；恢复方式仅为按固定版本重新下载/重建。V1 的轻量证据与 V2 输入未删除。
