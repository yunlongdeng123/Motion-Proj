# 历史产物保留与清理账本

> **批次**：`STORAGE-RETENTION-20260719`
> **登记日期**：2026-07-19
> **状态**：已完成并验证
> **清理前代码基线**：`73bd1a4`，`main...origin/main`，工作树 clean
> **研究边界**：本批次只维护非 Git 产物，不授权 ReSim 下载、C1-BOOT、推理、训练、双卡或人工 verdict 代填

## 1. 保留原则

正式 run ID、历史事实和负结论不可覆盖，但不要求永久保留所有 checkpoint、candidate 视频、adapter 或中间
tensor。已关闭路线的大型二进制载荷可在以下条件同时满足时清理：

1. 删除范围在数据操作前以绝对路径冻结，并且每个目标都解析到 `/root/autodl-tmp` 下；
2. manifest、resolved config、metrics、events、summary、result、人工 verdict 与终止标记继续驻留；
3. 正式人工标注、待评审材料和 RF-11 的 common-support abstention 证据受保护；
4. 可下载权重固定上游仓库、revision、文件清单和已有 metadata object ID；
5. 清理前后核对保护证据 SHA-256，且清理后可用空间不少于 80 GB；
6. 若空间仍不足，不得自行扩大到 nuScenes、环境、评审材料或 evaluator 资产。

本批次不为待删大文件制作数据备份。Git 文档、轻量 run evidence 与本账本就是保留记录；需要重算时必须使用
新 run ID，不能重建或覆盖原 run 目录。

## 2. 清理前基线

2026-07-19 登记时：

| 检查 | 结果 |
|---|---|
| Git | `73bd1a4`；`main...origin/main`；无 tracked/untracked 变更 |
| tmux | 无 server |
| 训练/推理进程 | 无 `accelerate`、`torchrun` 或训练脚本；仅 JupyterLab/TensorBoard 常驻 |
| GPU 设备 | 当前 shell 未暴露 `/dev/nvidia*`，`/usr/bin/nvidia-smi` 是不可执行的空占位文件，因此不存在可占用本实例 GPU 的设备进程 |
| 数据盘 | `/dev/md0`，总计 `137,438,953,472` 字节，已用 `92,389,117,952`，可用 `45,049,835,520`，`df -h` 显示 128G / 87G / 42G |
| 本地 nuScenes | `/root/autodl-tmp/data/nuscenes`，`du -sh` 约 35G；受保护 |
| 冻结删除目标 | 75 个精确路径，逻辑大小合计 `46,525,314,508` 字节（46.53 GB / 43.33 GiB） |

`ROUTE_PIVOT_FINAL_REPORT.md`、V5 计划、`BACKBONE_MIGRATION_AUDIT.md` 和 C0 run 中的 42 GB
继续保留为 2026-07-18 当时事实快照，不用清理后数值回写。

## 3. SVD-XT 可恢复信息

- Hugging Face 仓库：`stabilityai/stable-video-diffusion-img2vid-xt`
- 固定 revision：`9e43909513c6714f1bc78bcb44d96e733cd242aa`
- 重建入口：[`scripts/download_weights.md`](../scripts/download_weights.md)
- 清理前本地路径：`/root/autodl-tmp/weights/svd-xt`
- 清理前完整目录：`32,608,949,417` 字节（32.61 GB）
- 清理后驻留状态：`non-resident`；历史 asset check 明确报告无本地权重，不影响单元测试
- 内容边界：两个 monolithic safetensors，加上 Diffusers image encoder、UNet、VAE 的 `full` 与 `fp16`
  权重。旧文档的“约 10 GB”只接近单个 monolithic 文件，不是完整快照占用。

下表的 object ID 来自现有 `.cache/huggingface/download/**/*.metadata` 第二行；它是下载时记录的上游对象
标识，不是本批次重新计算的文件 SHA-256。

| 相对文件 | 字节 | Hugging Face metadata object ID |
|---|---:|---|
| `.gitattributes` | 1,571 | `78ff5b07c33f29189b6901023fc9f4d5724f31f7` |
| `LICENSE.md` | 11,852 | `1d9ce2ee1067327543544de197291726e4fc57a4` |
| `README.md` | 8,156 | `16b65b7ca9bd5a548fe89ff5463a1d6733ead1f7` |
| `comparison.png` | 146,654 | `6a355280c143517c8472151ef8715061c6f436b8` |
| `feature_extractor/preprocessor_config.json` | 518 | `0d9d33b883843d1b370da781f3943051067e1b2c` |
| `image_encoder/config.json` | 685 | `1f5518ccff586593a40f4eaf0e75c066dca54bec` |
| `image_encoder/model.fp16.safetensors` | 1,264,217,240 | `ae616c24393dd1854372b0639e5541666f7521cbe219669255e865cb7f89466a` |
| `image_encoder/model.safetensors` | 2,528,371,296 | `ed1e5af7b4042ca30ec29999a4a5cfcac90b7fb610fd05ace834f2dcbb763eab` |
| `model_index.json` | 496 | `814cc99f8674db1df84d0fff0d4e5535e745d328` |
| `output_tile.gif` | 18,630,497 | `2340a9809e36fa9634633c7cc5fd256737c620ba47151726c85173512dc5c8ff` |
| `scheduler/scheduler_config.json` | 533 | `05ea60ddb0d95607f9306e020e4ea355664d0275` |
| `svd_xt.safetensors` | 9,559,625,980 | `b2652c23d64a1da5f14d55011b9b6dce55f2e72e395719f1cd1f8a079b00a451` |
| `svd_xt_image_decoder.safetensors` | 9,503,252,964 | `99aa889bf6d1ca28e026755b83ba37e3072ad79b45dd4c94fae14bee7482263b` |
| `unet/config.json` | 984 | `2a30c09f6764459c04d7dc10bf5b4bbf1e5ebc73` |
| `unet/diffusion_pytorch_model.fp16.safetensors` | 3,049,435,868 | `9fbc02e90f37d422f5e3a4aeaee95f6629dc8c45ca211b951626e930daf2bddf` |
| `unet/diffusion_pytorch_model.safetensors` | 6,098,682,464 | `7783d82729af04f26ded4641a5952617fe331fc46add332fb9e47674fecc6ad7` |
| `vae/config.json` | 607 | `7c27c35b4e6ab0e705d46306f60d36839b680c03` |
| `vae/diffusion_pytorch_model.fp16.safetensors` | 195,531,910 | `af602cd0eb4ad6086ec94fbf1438dfb1be5ec9ac03fd0215640854e90d6463a3` |
| `vae/diffusion_pytorch_model.safetensors` | 391,017,740 | `5d92aa595a53d9da9faf594f09910ee869d5d567c8bb0362d5095673c69997d6` |

## 4. 冻结删除范围

| 类别 | 目标数 | 清理前逻辑字节 | RF / 理由 | 保留事实 |
|---|---:|---:|---|---|
| SVD-XT 完整本地快照 | 1 | 32,608,949,417 | SVD 路线已关闭；C1 使用 ReSim；固定 revision 可重建 | 本节完整下载账本、历史 manifest/result 与文档 |
| 全部既有非空历史 `runs/**/ckpts/` | 39 | 11,763,536,899 | `RF-01`、`RF-06`；已完成/失败训练的 optimizer/adapter 载荷 | 父 run 的 manifest、resolved、metrics、events、summary、日志与终止标记 |
| B0 128-video candidate pool | 1 | 1,416,132,207 | `RF-15`；machine gate 已拒绝 | 全部生成索引、评分、排名、machine gate、result、summary 与 sensitivity evidence |
| F0 rejected variants + noise bank | 2 | 67,060,112 | `RF-06`；0 个 variant 同时通过 correction/locality | manifest、resolved、metrics、Pareto、summary、日志与 `COMPLETE` |
| A1 rejected `feature_records/` | 1 | 54,932,107 | `RF-14`；21/21 配置均未产生合法 actor candidate | scene split、queries、index、control/primary metrics、result 与 summary |
| PA1/PA2 中间目录 | 31 | 614,703,766 | `RF-09`–`RF-12`；candidate/constructor/latent 路线已关闭 | JSON/JSONL 结论、人工 review、UPO v1/v2 common-support 证据与终止标记 |
| **合计** | **75** | **46,525,314,508** | 只删除冻结路径 | 不动态扩大范围 |

### 4.1 SVD、B0、F0 与 A1

```text
/root/autodl-tmp/weights/svd-xt
/root/autodl-tmp/runs/route-pivot-b0-natural-rollout-s20260718-v1/candidates
/root/autodl-tmp/runs/autoresearch-f0-endpoint-s20260713-6845411/variants
/root/autodl-tmp/runs/autoresearch-f0-endpoint-s20260713-6845411/noise_bank.pt
/root/autodl-tmp/runs/route-pivot-a1-feature-scan-s20260718-v2/feature_records
```

### 4.2 既有 39 个非空 checkpoint 目录

```text
/root/autodl-tmp/runs/motionproj_v1/ckpts
/root/autodl-tmp/runs/p2-determinism-a4-dirty/ckpts
/root/autodl-tmp/runs/p2-determinism-b4-dirty/ckpts
/root/autodl-tmp/runs/p2-exactresume-interrupted4-dirty3/ckpts
/root/autodl-tmp/runs/p2-resume-continuous12-6c6261f/ckpts
/root/autodl-tmp/runs/p2-resume-continuous12-f1b568c/ckpts
/root/autodl-tmp/runs/p2-resume-continuous12-fa375b2/ckpts
/root/autodl-tmp/runs/p2-resume-continuous4-f1b568c/ckpts
/root/autodl-tmp/runs/p2-resume-interrupted12-6c6261f/ckpts
/root/autodl-tmp/runs/p2-resume-interrupted12-f1b568c/ckpts
/root/autodl-tmp/runs/p2-resume-interrupted12-fa375b2/ckpts
/root/autodl-tmp/runs/p2-rngresume-continuous4-dirty2/ckpts
/root/autodl-tmp/runs/p2-rngresume-interrupted4-dirty2/ckpts
/root/autodl-tmp/runs/p2-train-flow2-ba53f20/ckpts
/root/autodl-tmp/runs/p2-train-flow2-dirty-stopgrad/ckpts
/root/autodl-tmp/runs/p2-train-full2-9d9b28e/ckpts
/root/autodl-tmp/runs/p2-train-replay2-ced5e35/ckpts
/root/autodl-tmp/runs/p2-train-synthetic2-ba53f20/ckpts
/root/autodl-tmp/runs/p2-tune-mini/trials/tune-smoke2/ckpts
/root/autodl-tmp/runs/p2-tune-mini/trials/tune-t0-s100-180a8f6dfc/ckpts
/root/autodl-tmp/runs/p2-tune-mini/trials/tune-t1-s100-a11a64832b/ckpts
/root/autodl-tmp/runs/p2-tune-mini/trials/tune-t10-s100-6f451fa77e-s300-6f451fa77e/ckpts
/root/autodl-tmp/runs/p2-tune-mini/trials/tune-t10-s100-6f451fa77e/ckpts
/root/autodl-tmp/runs/p2-tune-mini/trials/tune-t11-s100-1441277e7f/ckpts
/root/autodl-tmp/runs/p2-tune-mini/trials/tune-t12-s100-c70da52680/ckpts
/root/autodl-tmp/runs/p2-tune-mini/trials/tune-t13-s100-361d60028a/ckpts
/root/autodl-tmp/runs/p2-tune-mini/trials/tune-t14-s100-50a2afc0d3-s300-50a2afc0d3/ckpts
/root/autodl-tmp/runs/p2-tune-mini/trials/tune-t14-s100-50a2afc0d3/ckpts
/root/autodl-tmp/runs/p2-tune-mini/trials/tune-t15-s100-bc37ae7eee/ckpts
/root/autodl-tmp/runs/p2-tune-mini/trials/tune-t2-s100-a7f98ea51c/ckpts
/root/autodl-tmp/runs/p2-tune-mini/trials/tune-t3-s100-618c6eb1de/ckpts
/root/autodl-tmp/runs/p2-tune-mini/trials/tune-t4-s100-ca89099ae4/ckpts
/root/autodl-tmp/runs/p2-tune-mini/trials/tune-t5-s100-9fae4dcf5b/ckpts
/root/autodl-tmp/runs/p2-tune-mini/trials/tune-t6-s100-82eb1e9cbd/ckpts
/root/autodl-tmp/runs/p2-tune-mini/trials/tune-t7-s100-def9a17d55-s300-def9a17d55/ckpts
/root/autodl-tmp/runs/p2-tune-mini/trials/tune-t7-s100-def9a17d55/ckpts
/root/autodl-tmp/runs/p2-tune-mini/trials/tune-t8-s100-ff84413cf5/ckpts
/root/autodl-tmp/runs/p2-tune-mini/trials/tune-t9-s100-05e5eed95f-s300-05e5eed95f/ckpts
/root/autodl-tmp/runs/p2-tune-mini/trials/tune-t9-s100-05e5eed95f/ckpts
```

### 4.3 PA1/PA2 的 31 个中间目录

```text
/root/autodl-tmp/runs/autoresearch-pa1-branch-s20260715-v2/candidates
/root/autodl-tmp/runs/autoresearch-pa1-branch-s20260715-v2/diagnostic_independent
/root/autodl-tmp/runs/autoresearch-pa1-branch-s20260715-v3/candidates
/root/autodl-tmp/runs/autoresearch-pa1-branch-s20260715-v3/diagnostic_independent
/root/autodl-tmp/runs/autoresearch-pa1-branch-s20260715-v4/candidates
/root/autodl-tmp/runs/autoresearch-pa1-branch-s20260715-v4/diagnostic_independent
/root/autodl-tmp/runs/autoresearch-pa1-branch-s20260715-v5/candidates
/root/autodl-tmp/runs/autoresearch-pa1-branch-s20260715-v5/diagnostic_independent
/root/autodl-tmp/runs/autoresearch-pa2-cand-fallback-s20260716-v1/candidates
/root/autodl-tmp/runs/autoresearch-pa2-cand-fallback-s20260716-v1/diagnostic_independent
/root/autodl-tmp/runs/autoresearch-pa2-pair-extension-s20260715-v1/candidates
/root/autodl-tmp/runs/autoresearch-pa2-pair-extension-s20260715-v1/constructor_baselines
/root/autodl-tmp/runs/autoresearch-pa2-pair-extension-s20260715-v1/diagnostic_independent
/root/autodl-tmp/runs/autoresearch-pa2-pair-s20260715-v1/candidates
/root/autodl-tmp/runs/autoresearch-pa2-pair-s20260715-v1/constructor_baselines
/root/autodl-tmp/runs/autoresearch-pa2-pair-s20260715-v1/diagnostic_independent
/root/autodl-tmp/runs/autoresearch-pa2-pair-smoke-s20260715-v1/candidates
/root/autodl-tmp/runs/autoresearch-pa2-pair-smoke-s20260715-v1/constructor_baselines
/root/autodl-tmp/runs/autoresearch-pa2-pair-smoke-s20260715-v1/diagnostic_independent
/root/autodl-tmp/runs/autoresearch-pa2-pair-smoke-s20260715-v2/candidates
/root/autodl-tmp/runs/autoresearch-pa2-pair-smoke-s20260715-v2/constructor_baselines
/root/autodl-tmp/runs/autoresearch-pa2-pair-smoke-s20260715-v2/diagnostic_independent
/root/autodl-tmp/runs/autoresearch-pa2-pair-smoke-s20260715-v3/candidates
/root/autodl-tmp/runs/autoresearch-pa2-pair-smoke-s20260715-v3/constructor_baselines
/root/autodl-tmp/runs/autoresearch-pa2-pair-smoke-s20260715-v3/diagnostic_independent
/root/autodl-tmp/runs/autoresearch-pa2-pair-smoke-s20260715-v4/candidates
/root/autodl-tmp/runs/autoresearch-pa2-pair-smoke-s20260715-v4/constructor_baselines
/root/autodl-tmp/runs/autoresearch-pa2-pair-smoke-s20260715-v4/diagnostic_independent
/root/autodl-tmp/runs/autoresearch-pa2-pair-smoke-s20260715-v5/candidates
/root/autodl-tmp/runs/autoresearch-pa2-pair-smoke-s20260715-v5/constructor_baselines
/root/autodl-tmp/runs/autoresearch-pa2-pair-smoke-s20260715-v5/diagnostic_independent
```

删除阶段只使用以上 75 个字面绝对路径；不会再次用 `find`、`runs/**`、`*-v*` 或其他动态模式决定目标。

## 5. 明确保留范围

- `/root/autodl-tmp/runs/autoresearch-pa2-pair-expanded-s20260715-v1/reviews.jsonl` 的正式 48 条人工
  verdict，以及 `review/`、review cases、prompt、private mapping 和 summary；
- `/root/autodl-tmp/runs/route-pivot-r1-temporal-s20260718-v1/review/` 的 32 个待标注 pair；
- `/root/autodl-tmp/runs/route-pivot-a0-real-motion-s20260718-v3/{panels,review}/` 的 12 个待标注
  panel；
- 所有 run 的 manifest、resolved YAML、metrics、events、summary、result、review verdict 与终止标记；
- UPO v1/v2 的 `query_sets.jsonl`、`paired_tracks.jsonl`、`common_support.jsonl`、
  `bootstrap_intervals.jsonl`、graphs、calibration/holdout/stress summary 和 `COMPLETE`；
- nuScenes、本地 `motionproj` 环境、CoTracker3、RAFT、Depth/evaluator 资产与 C0 迁移证据；
- B0 的 scored/ranking/machine-gate/result/sensitivity，F0 的 metrics/Pareto/summary，A1 的 query/index/probe
  metrics/result，以及 PA1/PA2 的轻量 JSON/JSONL 结论。

## 6. 清理前保护哈希

目录聚合哈希的算法是：按绝对路径排序文件，对每个文件执行 `sha256sum`，再对完整 checksum stream
执行一次 `sha256sum`。因此文件内容、文件集合或路径变化都会改变结果。

| 保护集合 | 文件数 | 清理前 SHA-256 |
|---|---:|---|
| PA2 正式 48 条 `reviews.jsonl` | 1 | `a313d1e23282807e232c2a7e27b1f646b809b62add38f9a3060bc5ab94cab012` |
| R1 32-pair `review/` package | 68 | `96dab39380b1b815c7be44d8f2216bbeef44fad136c6c8dad81ffd4cd517c929` |
| A0 v3 12-panel `panels/ + review/` package | 15 | `6cd8a25d913289c6c06cad78463002b51d82591f6a0e45113039cafd369b605e` |
| UPO v1/v2 全部文件 | 32 | `2a7c405cd4dd143b5bc4a83784830b6120ce6cdf68dbfdfe6025ea295b9359a9` |
| 534 个核心 manifest/resolved/metrics/events/summary/result/review/终止文件 | 534 | `5f7a616293870012cc227d58d3ae34728c28310a3f1d75d10813cf5527a60d77` |

快照文档与 C0 证据的清理前 SHA-256：

| 路径 | SHA-256 |
|---|---|
| `docs/ROUTE_PIVOT_FINAL_REPORT.md` | `f72113d8f75fd4950f0393ccf2ed3d2ba83e8f9a795f340284d5f31805476132` |
| `docs/MOTION_ROUTE_PIVOT_AUTORESEARCH_PLAN_V5.md` | `14e92920d617400afbcc23e22fba35610a0591892b57c47d3cedc637fddcd810` |
| `docs/BACKBONE_MIGRATION_AUDIT.md` | `743347df7dba017ab613a5b3b1720bfdd284eb293cca855ab9943fabf5c401cd` |
| C0 `summary.json` | `20a7c4845a48cef824a393346b81ee70b740cdf779261c4ee102fd6ba2d782ef` |

## 7. 执行与收口结果

数据删除在清单提交 `2d520565118bf0f0413f78461b8cb837a683f1ea` 之后执行。删除命令只接收第 4 节的
75 个字面路径；执行前再次确认 Git clean、无 tmux/训练进程、无可用 NVIDIA device，并逐项确认
`realpath -e` 与冻结路径相同。

| 项目 | 结果 |
|---|---|
| 清单提交 | `2d52056 docs(storage): 固化历史产物保留与清理清单` |
| 实际删除目标 / 逻辑字节 | 75 / `46,525,314,508`；与冻结清单完全一致 |
| 文件系统实际回收 | `46,541,172,736` 字节；按清理前后 `df -B1` 的 used/avail 差值计算 |
| 清理后 `df` | 总计 `137,438,953,472`，已用 `45,847,945,216`，可用 `91,591,008,256` 字节；`df -h` 为 128G / 43G / 86G，使用率 34% |
| 清理后 `du` | `/root/autodl-tmp` `45,847,945,216`；data `36,638,400,512`；envs `8,352,190,464`；runs `572,469,248`；weights `101,892,096`；cache `0` 字节 |
| 75 个目标不存在 | 75/75；非空 `ckpts/` 计数为 0 |
| 受保护路径存在 | 48 条正式 review、R1 32 pair、A0 12 panel、UPO v1/v2、nuScenes、环境、CoTracker3、RAFT/Depth 与 C0 evidence 全部存在 |
| 保护 SHA-256 不变 | 第 6 节五组 evidence aggregate 与四个历史快照逐项相同 |
| JSON/JSONL/YAML 解析 | 513 个 JSON、317 个 JSONL（96,765 条非空记录）和 167 个 YAML 全部可解析 |
| 历史 asset check | 0 failures / 2 warnings；CUDA 当前不可用，SVD-XT 明确报告 no local HF cache candidate，即 `non-resident` |
| `git diff --check` | 第二笔提交前 staged/unstaged 检查通过 |
| 完整 `pytest -q` | 容器 cgroup 上限为 2 GiB，单进程在真实 nuScenes 项处被 OOM 137；逐测试文件隔离运行同一套件，JUnit 汇总为 73/73 files、247 tests、0 failures、0 errors、0 skipped |
| C1/ReSim 动作 | 不下载、不推理、不训练 |

清理后可用空间为 85.3 GiB，超过 80 GB 收口门槛，也满足当前文档中的 70–80 GB C1-BOOT 存储前置条件。
这只关闭了存储 blocker；`RESEARCH_STATUS.md` 仍不授权 C1-BOOT。
