# WorldSim V5 M1：30k 正式基线与 Structured Unary 诊断

日期：2026-08-14

任务：`WS-V5-M1-STRUCTURED-OWNERSHIP-01`

用途：技术报告正文/附录的轻量事实入口；重型 checkpoint 与 NPZ 保留在不可变 run 目录。

## 结论

8 个 frozen development scenes 的 StreetGS base 均完成 30k，并由独立 r035 对 checkpoint bytes/SHA、run 终态与资源分母逐项复核。scene0471 的 frozen SAM evidence r036 与 unary diagnostic r037 均完成，base checkpoint 在推理前后字节完全一致。

r037 支持 reliability-aware unary 的机制方向：B1/B3 相对 B0 在 Gaussian proxy 与 2D SAM-proxy 上均改善 IoU/calibration，2D Boundary F1 分别提高 `+0.107437/+0.105079`。但 2D FN semantic mass 同时增加 `+0.091532/+0.095477`，所以结论不是“方法通过”，而是：

> unary 方向有支持，但存在明确 FN tradeoff；不选择 B1/B3，不自动解锁 graph，不读取 validation/test。

## 8-scene 30k base

| Scene | Formal run | Duration (s) | Checkpoint bytes | Gaussian count（BG/Rigid） | Peak GPU MiB | Checkpoint SHA-256 |
|---|---|---:|---:|---:|---:|---|
| 0471 | r027 | 2025.140 | 288,937,974 | 809,902 / 49,711 | 23,396 | `496356ca2d8f31b4e8593b294eebba1f068a0c7fddbd86ab1717c377a1793cfa` |
| 1087 | r028 | 1814.695 | 305,219,702 | 930,979 / 244 | 21,716 | `84c34b837e8083a84a77de0662c6c702ce978a2e8e11a4038d487e1f6c9754be` |
| 0379 | r029 | 2136.735 | 365,700,214 | 1,186,659 / 632 | 23,464 | `d77fa13f5ebe1d469222c3df78eb316d836178dca608d00c17f0af3a56d3042d` |
| 0998 | r030 | 2733.878 | 477,402,742 | 1,578,331 / 79,068 | 24,054 | `4ad88a7e3a8a1df83bbb01675a7945cf4b496b299e4d5e3f356ca061143c0516` |
| 0359 | r031 | 2300.078 | 415,637,174 | 1,387,860 / 9,896 | 23,686 | `ca1ec48e2acc3883405f4d1a25b7682e2de18abd1a93342b751d8cfc7c9b6048` |
| 0875 | r032 | 2057.192 | 310,077,046 | 889,059 / 59,915 | 20,792 | `d72286163ff246c493747bbd29fba0723419d0138209b18917108043edde7cc7` |
| 0535 | r033 | 2158.847 | 365,295,478 | 1,162,092 / 21,981 | 23,898 | `a74669b2f22c6c4445fcf972d716b77b54be555ee2eedb88ac4ebfd118472838` |
| 0436 | r034 | 3166.334 | 391,824,182 | 1,140,699 / 151,229 | 23,976 | `ee03a8165bc1b335dab867c3b4066e0870fd550110b7f916e9fbdde9e7f5eb4a` |

r035 audit：`8/8 × 30,000`，total duration=`18,392.900159636512 s`，total checkpoint bytes=`2,920,094,512`，maximum peak GPU/cgroup=`24,054 MiB / 27,706,277,888 bytes`，formal source commit=`267faba485148a6e60e79864848c8376016b6369`。所有 checkpoint payload 均重哈希一致；没有读取 validation/test quality，也没有 method inference。

## Frozen SAM evidence（r036）

- canonical run：`/root/autodl-tmp/runs/worldsim_v5/WS-V5-M1-STRUCTURED-OWNERSHIP-01/20260814T172400Z__m1-scene0471-sam-sparse-s0-r036`
- source commit：`6ce988b4d4e02ab920b2a6f19878d827145473cc`
- 预注册视图：15 evidence + 15 development evaluation；available/accepted=`18/18`
- prompts：`17 actors / 62 boxes / 61 accepted boxes`
- network/held-out/method inference：`false/false/false`
- summary/mask-manifest SHA：`d66f04a05a5a0ee8fb94b423e20296ec019bc8fe1e56ebc6c57fb1c80495d487 / f7a9f5e9f022c8f8685be89e7cdd7d808f13081106c56b07e5c87260ac72a213`

## Structured unary（r037）

- canonical run：`/root/autodl-tmp/runs/worldsim_v5/WS-V5-M1-STRUCTURED-OWNERSHIP-01/20260814T173032Z__m1-scene0471-unary-diagnostic-s0-r037`
- source commit：`3fec3ab41a4d0b245b2cb55cd1354f2339532146`
- denominator：`859,613 Gaussians`；`15` evidence views；evaluation=`8 accepted + 7 abstained`
- frozen threshold：`0.5`；parameter search=`false`
- duration/peak GPU：`555.1997426208109 s / 13,987 MiB`
- checkpoint before/after SHA：均为 `496356ca2d8f31b4e8593b294eebba1f068a0c7fddbd86ab1717c377a1793cfa`

### Gaussian membership proxy

| Arm | IoU | Brier | ECE | NLL | FP mass | FN mass |
|---|---:|---:|---:|---:|---:|---:|
| B0 | 0.867323 | 0.009692 | 0.033437 | 0.041575 | 0.035385 | 0.032277 |
| B1 | 0.998413 | 0.003552 | 0.018667 | 0.020815 | 0.019166 | 0.010828 |
| B3 | 0.999698 | 0.003180 | 0.019391 | 0.021271 | 0.019419 | 0.018961 |

### 2D frozen SAM-proxy evaluation

| Arm | IoU | Boundary F1 | Brier | ECE | NLL | FP mass | FN mass |
|---|---:|---:|---:|---:|---:|---:|---:|
| B0 | 0.268201 | 0.189959 | 0.210897 | 0.237214 | 0.959039 | 0.246497 | 0.237811 |
| B1 | 0.384866 | 0.297396 | 0.109158 | 0.126231 | 0.744577 | 0.103568 | 0.329343 |
| B3 | 0.384363 | 0.295038 | 0.107430 | 0.124782 | 0.653935 | 0.108044 | 0.333288 |

| Delta vs B0 | ΔIoU | ΔBoundary F1 | ΔBrier | ΔECE | ΔNLL | ΔFP mass | ΔFN mass |
|---|---:|---:|---:|---:|---:|---:|---:|
| B1 | +0.116665 | +0.107437 | -0.101739 | -0.110983 | -0.214462 | -0.142929 | +0.091532 |
| B3 | +0.116161 | +0.105079 | -0.103467 | -0.112431 | -0.305104 | -0.138453 | +0.095477 |

## 解释边界与下一门

1. 这是单个 development scene 的机制诊断，evaluation target 仍来自冻结 SAM，而不是人工独立 GT；不能代表 validation。
2. B1 的 Boundary F1/IoU/FN 略优，B3 的 Brier/ECE/NLL 略优；差异不足以选择 arm。
3. 两臂 FN tradeoff 都超过计划 validation gate 的 `+0.01`，必须在 graph 诊断中原样保留，不能通过事后改阈值消除。
4. 下一步只允许冻结小型 graph mechanism protocol；Transformer、hierarchical B2、semantic split、validation/test/KITTI quality 继续禁止。

## 完整性哈希

| Artifact | SHA-256 |
|---|---|
| r035 summary | `4a540d24cd8bfa18c9d63cdcbabe08dcded7a2de88de116695e431187cb6738b` |
| r035 manifest | `bba0892345225f5d4527402943d17b7806207714c7a9355641eda3b2cda72119` |
| r036 summary | `d66f04a05a5a0ee8fb94b423e20296ec019bc8fe1e56ebc6c57fb1c80495d487` |
| r036 mask manifest | `f7a9f5e9f022c8f8685be89e7cdd7d808f13081106c56b07e5c87260ac72a213` |
| r037 summary | `dd8b2a9e5f09f130f948c9de2b6b8eaa5bea9ab714278bed7fa56a633dd7a22d` |
| r037 status | `fb68e06d174a5e6a6859a50c07392b6895102feecf3944e712dc9861de1736ce` |
| r037 fingerprint | `70d2f878e9042b632d4f3355cc350d02ae661127c738130019506aa77c4480e4` |
| r037 manifest | `80ff775de6a4fc4c748cf3ec9570c2ab0ce10e817fbe5cdcd1bef69eeee871c4` |
| r037 diagnostics | `88e256b9f07149cdfbf94da26e7d59b83c2071cb4485e41cf80717f0eac0d755` |
| r037 resolved config | `a09ac4f9359df36e1c9ff90fdba83da5de5cac519eb52979d6444211df50e291` |
