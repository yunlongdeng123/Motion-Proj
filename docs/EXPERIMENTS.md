# Experiments

- 更新时间：2026-07-26
- 活跃路线：动态驾驶场景重建与反事实编辑
- 权威方案：[`DYNAMIC_DRIVING_RECONSTRUCTION_PLAN_V1.md`](DYNAMIC_DRIVING_RECONSTRUCTION_PLAN_V1.md)
- 历史快照：[`archive/2026-07/cutin-mining-closed/EXPERIMENTS_CUTIN_FINAL_SNAPSHOT.md`](archive/2026-07/cutin-mining-closed/EXPERIMENTS_CUTIN_FINAL_SNAPSHOT.md)

本文件只登记当前路线。V1–V7.1、OccGS 和 cut-in 的完整实验历史已归档；失败事实继续由
[`RESEARCH_FAILURES.md`](RESEARCH_FAILURES.md) 约束，不因精简活动台账而删除。

## 1. 状态词

只使用：

```text
pending | running | blocked | done | rejected
```

`done` 表示预注册门禁满足；`blocked` 表示需要外部资源/权限或 upstream 修复；`rejected` 表示研究门禁失败，不能靠
重命名、挑场景或放宽阈值继续。

## 2. 活跃注册表

| Run ID | 状态 | 目标 | 输入 | Primary gate | 证据路径 |
|---|---|---|---|---|---|
| `DR-M0-ARCHIVE-01` | done | 封存 cut-in、清理可再生产物 | 历史 docs/data/runs | 保留项完整、删除项精确、无 OOM | `docs/archive/2026-07/cutin-mining-closed/` |
| `DR-M1-PLAN-01` | done | 官方调研与下一阶段完整方案 | 官方论文/代码、本地只读资产 | baseline/data/env/experiments/stops/review 全部闭合 | `docs/DYNAMIC_DRIVING_RECONSTRUCTION_PLAN_V1.md` |
| `DR-M2-ENV-ASSET-01` | pending | AD-GS 环境与六场景资产 smoke | 官方六 scenes、独立 envs | 编译/单步/文件计数/typed provenance 全过 | `/root/autodl-tmp/runs/dynamic_recon/` |
| `DR-M3-ADGS-0230-01` | pending | scene-0230 exact pipeline | frames 10..69、三相机、900×1600 | 60k 正常结束、官方 render/metrics 完整 | 同上 |
| `DR-M4-ADGS-6SCENE-01` | pending | AD-GS 六场景数值复现 | 官方六 scenes | PSNR≥30.56、SSIM≥0.915、LPIPS≤0.184 | 同上 |
| `DR-M5-DGGT-NUSC-01` | pending | DGGT 推理级对照 | 同六 scenes 固定窗口 | upstream smoke；完整输入预算与速度/质量报告 | 同上 |
| `DR-M6-STRESS-01` | pending | 重建/编辑/去遮挡/噪声压力测试 | 六 scenes、冻结对象与编辑幅度 | ≥3 scenes 重复同一失败 | 同上 |
| `DR-M7-HYPOTHESIS-01` | pending | 唯一创新假设预注册 | M6 failure matrix | novelty、truth tier、primary、baseline 冻结 | `docs/` |
| `DR-M8-METHOD-01` | pending | 方法/消融/统计 | 六 scenes、3 seeds | primary 改善且 guardrails 不退化 | `/root/autodl-tmp/runs/dynamic_recon/` |
| `DR-M9-HUMAN-01` | pending | 盲审与最终人工包 | 冻结全量 clips/metrics | 用户/指定评审完成 verdict | `docs/human-review/` |

## 3. `DR-M0-ARCHIVE-01`

### 结果

- 状态：`done`
- 文档快照：
  `/root/autodl-tmp/motion_proj_backups/docs-before-direction-pivot-2026-07-26/`
- 回收：约 5 GiB；
- 删除：3 个 N1 可再生 10 Hz cache、3 个 B0 中间 checkpoint、2 个 H1C 失败渲染副本、
  5 个干净 worktree、41 个冗余 `.codexbak.*`；V7.1 审计目录内 20 个已索引备份继续保留；
- 保留：raw nuScenes、mini comparator、trainval annotations、final checkpoints、正式指标/日志、审核证据、
  `RESEARCH_FAILURES.md`；
- `memory.events`: `oom=0`, `oom_kill=0`。

### 证据

- [`archive/2026-07/cutin-mining-closed/CLEANUP_MANIFEST.md`](archive/2026-07/cutin-mining-closed/CLEANUP_MANIFEST.md)
- [`archive/2026-07/cutin-mining-closed/README.md`](archive/2026-07/cutin-mining-closed/README.md)

## 4. `DR-M1-PLAN-01`

### 结果

- 状态：`done`
- 主基线：AD-GS；
- 前馈对照：DGGT inference-only；
- 编辑参考：DrivingEditor；
- conditional geometry comparator：VAD-GS；
- primary data：AD-GS 官方 nuScenes 六 scenes；
- 当前资产结论：左右前相机和 sweeps 不完整，必须选择性提取；
- 当前资源结论：2 GiB 接近上限，本轮不能安装或运行。

### 复现锚点

```text
scenes = 0230,0242,0255,0295,0518,0749
frames = 10..69 inclusive
cameras = CAM_FRONT_LEFT,CAM_FRONT,CAM_FRONT_RIGHT
resolution = 900x1600
iterations = 60000
paper_mean = PSNR 31.06 / SSIM 0.925 / LPIPS 0.164
```

### 下一步

等待用户开放至少 32 GB RAM 和 1×24 GB GPU。资源未开放前 `DR-M2-ENV-ASSET-01` 不启动。

## 5. `DR-M2-ENV-ASSET-01` 预注册

### 输入

- official AD-GS commit；
- official environment.yaml；
- scene-0230 首先；
- 本机只读 nuScenes tar shards。

### 固定顺序

1. resource preflight；
2. pin source/license；
3. create AD-GS env；
4. compile rasterizers；
5. one-step forward/backward；
6. DPT/SAM single-image smoke；
7. exact asset member manifest；
8. scene-0230 selective extraction；
9. upstream preprocess structural audit。

### 单卡停止

- memory ≥90% limit；
- OOM/RC137；
- GPU OOM；
- disk free <20 GiB；
- 需要少相机/低分辨率才能继续。

任一触发后状态写 `blocked`，不重跑。

## 6. `DR-M3/M4` 预注册

### M3

scene-0230 先 100 iterations、再 1,000 iterations 做资源画像；只有投影满足资源合同才跑官方 60k。
M3 不以单场景数字对齐论文，只验证 exact pipeline 完整。

### M4

六场景全部运行，三项必须同时通过：

```text
mean PSNR >= 30.56
mean SSIM >= 0.915
mean LPIPS(VGG) <= 0.184
```

报告所有 per-scene、mean、worst-case、coverage。失败只允许一次有明确根因的重跑；仍失败则 `blocked`。

## 7. `DR-M5` 预注册

DGGT 不做训练。固定三个 4-frame windows/scene：

```text
10..13
34..37
66..69
```

先官方 native protocol，再做 common-observation diagnostic。必须报告 DGGT 与 AD-GS 不同的输入帧数、pose 使用、
逐场景优化和 resize，禁止写成 matched leaderboard。

## 8. `DR-M6` 预注册

### 对象

每 scene 最多两个：

- 可见支持最高的 `high-support`；
- 仍满足最低门槛但支持最低的 `boundary-support`。

不足两个如实记 coverage。

### 编辑

```text
lateral +0.5/+1.0/+1.5 m
time shift -0.5/+0.5 s
speed 0.75x/1.25x
stop 1.0 s then restart
delete
```

不赋予 cut-in/merge 语义。

### 真值

- Tier A：held-out real observation；
- Tier B：geometric support；
- Tier C：unsupported，只评 uncertainty/ABSTAIN/人审。

### 噪声

固定 `0230/0242/0255`，one-factor-at-a-time，3 seeds；完整级别见权威计划第 11.5 节。

## 9. `DR-M7/M8` 预注册

只有 M6 在 ≥3 scenes 重复失败才选方法。当前预期优先考察：

```text
编辑诱发 visibility recomputation
+ evidence-typed/confidence-aware Gaussians
+ non-target perception preservation
```

但 VAD-GS、GA-GS、DrivingEditor、DenoiseGS、Perception-aware 3DGS 的 claim 边界必须先审计。若无稳定失败，
M7=`rejected`，不硬造模块。

方法实验至少 3 seeds，matched scene/frame/camera/actor/edit/seed/budget，报告 CI、worst-case 与 coverage。

## 10. 历史路线入口

- cut-in 最终封存：[`archive/2026-07/cutin-mining-closed/README.md`](archive/2026-07/cutin-mining-closed/README.md)
- cut-in 结束时状态：[`archive/2026-07/cutin-mining-closed/RESEARCH_STATUS_CUTIN_FINAL_SNAPSHOT.md`](archive/2026-07/cutin-mining-closed/RESEARCH_STATUS_CUTIN_FINAL_SNAPSHOT.md)
- cut-in 结束时实验台账：[`archive/2026-07/cutin-mining-closed/EXPERIMENTS_CUTIN_FINAL_SNAPSHOT.md`](archive/2026-07/cutin-mining-closed/EXPERIMENTS_CUTIN_FINAL_SNAPSHOT.md)
- OccGS V7/V7.1：[`archive/2026-07/v7-feasibility/`](archive/2026-07/v7-feasibility/)
- 所有失败：[`RESEARCH_FAILURES.md`](RESEARCH_FAILURES.md)
