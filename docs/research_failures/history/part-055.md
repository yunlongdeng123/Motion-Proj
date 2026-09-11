# 历史原始记录 055

> 冻结历史，不是当前执行规则。当前规则见 [scaling law](../../../auto-research_scaling_law.md) 与 [AGENTS](../../../AGENTS.md)。
> 原文件：`docs/RESEARCH_FAILURES.md`；迁移前提交 `abbd431c`。原有相对链接按原目录解释，证据路径原样保留。
> 按索引读取相关小节；旧哈希、过度门控和资源指令均不重新生效。

## 1. V1–V6 版本总览与 V1/V2 汇总

| 版本 | 终态/核心推翻 | 主要工程坑 | 详细证据入口 |
|---|---|---|---|
| V1 | AD-GS 六场景复现成立，但 persistent identity 不存在；唯一候选与已有工作直接重合，M7 rejected，M8/M9 未授权 | DGGT `pointops2` PEP 517 隔离缺 torch；训练完成、评估依赖和容器生命周期必须分开 | 下方 `V1-F01`–`V1-F06`、`PIVOT-F14B/F15/F16`、V1 frozen archive |
| V2 | M0–M4 闭环；M5 三场景压力测试只完成部分资产，不能写成路线完成；局部保持不等于删除后背景真实 | 空 shell PATH、CUDA/headers/包版本、devkit schema、SM 8.6、render 累积内存、子进程回收、空 actor slice | 下方 `V2-F01`–`V2-F09`、V2 继承门禁、`EXPERIMENTS.md` V2 注册表 |
| V3/V3.1 | A1 保持 off；A2 是 boundary/全局/成本 tradeoff；A3 局部精修不晋级；P1 pruning rejected；P2/P3 只支持存储/资产拆分 claim | 相机标签、随机 CUDA 初始化、分辨率三层合同、runtime state key、资源 ceiling、前馈前置条件 | `V3-F01`–`V3-F25` |
| V3.2/V3.3 | S4 temporal 未完成、S5 语义生产链回退；RoadPatch/asset/release 只在冻结场景和协议成立，不构成跨场景 dominance | frozen base identity、empty-target、模型/视图可用性、类型/枚举严格比较、确定性 archive | `V32-*`、`V33-*` 详细章 |
| V4 | M1 scene-disjoint validation rejected；M2 selective routing 成立但 geometry MAE 退化 `+3.3908 m`；M3 仅在冻结 18-scene exact-once test confirmed | cohort 非确定性、split leak、SSH 断管、解释器分层、CUDA arch、immutable run/staging、完整 denominator | live canonical `V4-F01`–`V4-F49` |
| V5 | M1/M2/M3 全部 rejected；structured graph 不稳定、无 absolute geometry-safe candidate、constraint projection 信号不足 | KITTI calib/OXTS 语义、缺 LiDAR 帧、provenance enum、launcher 原子目录、heading metric 和 long-run stdout | `V5-F01`–`V5-F59` |
| V5.1 | M1-only 已收尾、无 promoted candidate；U2/B3 保留为 V5.2 comparator。LUDVIG uplift/raw graph、progressive、simple voxel node、Gaussian Grouping 与 exact faithful Trace3D operator 均按各自冻结门 rejected；Stage H 未运行，保持 pending 并由 V5.2 observation-source scope 取代 | uplift 无 actor margin；progressive/node elevation 的 IoU/FN 跨场失稳；identity coverage/persistence 不足；Trace3D alpha 跨 fresh process 非确定；另有零长 KNN、跨 shell、解释器/helper/CUDA 初始化、PDF/CLI、partial staging、solver/license/stdout、bytecode/cache、SAM 显存、batch sensitivity 与 CUBLAS 恢复边界 | `V51-F01`–`V51-F66` |
| V5.2 | 18-case 人工复核冻结 `9 BASE_FAILURE + 8 M123_ELIGIBLE + 1 unresolved`；M1/M3 症状匹配增强但 causal bridge 未通过，M2 降级为 safety/abstention | 原 census 的 actor/boundary 指标可被 global collapse 污染；eligible case 必须保持 `5 Discovery design + 3 one-shot Confirmation` | `V52-F01`–`V52-F02` |
| V6 | V5.2 TrackBayes 主线已由 world-compiler direction reset 取代；G0–G3 与 R1 capability gate 已通过，尚无方法质量结论 | pytest import/runtime profile 不能混用；历史外部资产可能只剩 manifest；冻结 plan 的 exact allowlist 必须显式纳入 terminal closeout；formal capability run 禁止 dirty source | `V6-F01`–`V6-F05`；G0–G3/R1 governance artifacts；V6 plan |
| V6.1 | 最小实验负结论收口：oracle `10/28, 0 false-safe`，GaussianWorld/IR-WM 均恢复 `10/28` 表面支持但各自 `10/10 false-safe`；ME-4 未解锁 | predicted argmax Occupancy 不能升级为安全 authority；第三 backend、threshold/grid/history/verifier sweep 均冻结 | `V61-F01`–`V61-F13`；`V61_MINIMUM_EXPERIMENT_CLOSEOUT.md` |
| V6.2 | CPSC-Lite family负结论收口：P6与唯一P6R均为`4/28,4/4 false-safe`，P7/P8未解锁 | evidence dropout把source-valid UNKNOWN从82.7%降到63.9%但未改变四个unsafe accepts；query-wise projection不能提供hidden surface authority；第二recovery、O_eval调参、backbone/backend/sweep冻结；未来复开需native logits/features、独立calibration与hidden-surface risk supervision；不新增哈希/校验和/指纹 | `V62-F01`–`V62-F07`；`P6R_EVIDENCE_DROPOUT_CLOSEOUT.md` |
| V6.3 | P2D native pointwise rejected；P3/P4 passed；P5/P5D objective collapse；P5R恢复训练candidate；P6 B3在两scene均输Native B2，surface family closed negative，P7锁定 | 训练内feasible不得冒充stage candidate；B3 tail与area同时失败后禁止继续B4/B5/M0、换seed/模型/门或读取legacy/H/T；未来复开必须是fresh uncertainty representation与conditional-coverage新版本 | `V63-F01`–`V63-F24`；`ARXIV_EVIDENCE_INDEX.md`；`P6_SURFACE_FAMILY_CLOSEOUT.md`；各P2D/P3/P4/P5/P5D/P5R/P6 prereg |
| V6.4 | full-native MLP与conditional M0通过独立exact-once；M1在untouched fixed-opportunity denominator相对支持；P11 collision critic经一次独立threshold recovery仍rejected，版本终态report-ready | U2绝对弱、U3高FPR、PCA calibration失败；selected与fixed denominator必须分开；共享盘I/O以restricted-shard、per-scene staging、ready-first GPU queue恢复；critic unsafe prior与ranking跨cohort漂移 | `V64-F01`–`V64-F28`；`V64_RESEARCH_FAMILY_CLOSEOUT.md`；`ARXIV_EVIDENCE_INDEX.md`；各P6R/P4C/P10/P11 closeout |
| V6.5 | visited-state reliability ranking与单调校准可迁移；one-shot direct action selection benefit失败并终止 | q0只可作给定轨迹访问状态的reliability diagnostic；禁止第二confirmation、阈值/lattice/critic救援 | `V65-F01`–`V65-F19`；`V65_ARXIV_TECHNICAL_REPORT.md`；`ARXIV_EVIDENCE_INDEX.md` |
| V6.6 | terminal/report-ready：两级certificate/HARP bake/P8R synthetic response支持；P7 surface repair终局负结果；P9未执行 | `V66-F02 closed_negative_after_single_recovery`；`V66-F03 resolved_by_single_implementation_recovery`；下一failure id=`V66-F04` | `WORLDSIM_V6_6_HARP_COMPILER_PLAN.md`；`docs/autoresearch/worldsim_v66/V66_ARXIV_TECHNICAL_REPORT.md` |

P4C conditional compiler freeze没有新增failure：它只把已读calibration中“50%的3个failure全部在rain”迁移为单一固定
coverage map，并在任何新quality read前冻结新8-scene confirmation。若formal replay不满足预注册coverage/risk gate，直接登记
algorithm failure并关闭该candidate，不改mapping或扫描第二版本。
新confirmation入口前free disk仅29 GiB；只回收精确的13 GiB pip下载缓存后为41 GiB，未删除formal run、模型、环境或processed
资产。这是预防性可恢复空间管理，不新增failure ID。
P4C确认执行继续直接复用V64-F16的scene-ready迁移并将新temporary raw改为独立精确路径；在target read前已冻结所有run ID、
单preprocess/双GPU并发和controller cleanup ownership。入口未出现新failure。

### 1.1 V1 汇总条目

- `V1-F01`（`algorithm/evaluation`, `active`）：AD-GS 六场景 exact reproduction 只证明 frozen baseline 可复现，
  不证明对象级编辑、未知背景恢复或新方法成立。证据为 V1 M4 aggregate；禁止把复现分数重命名为贡献。
- `V1-F02`（`engineering`, `resolved in V2`）：DGGT V1 在 input staging 前被 `pointops2` 的 PEP 517 隔离构建
  缺少 torch 阻塞，没有质量/速度数字。V2 用固定 compiler/runtime/headers 和 upstream 非隔离安装解除工程前置；
  V1 terminal 仍保持 blocked，详见 `PIVOT-F14B`。
- `V1-F03`（`algorithm/data`, `active`）：六场景 pseudo ID 最长支持仅 `1/6/1/1/2/1` 帧，checkpoint 只有
  二值 `obj`，`0/12` object slots 可评；`persistent_object_identity_unavailable` 不能靠事后几何关联回填。
- `V1-F04`（`algorithm/governance`, `active`）：候选“恢复持久身份并绑定 actor 后做轨迹编辑”与 InstDrive、Director、
  OmniRe、HorizonForge、G²Editor 直接重合；适配 AD-GS 是工程，不足以通过 novelty gate。M7 保持 rejected。
- `V1-F05`（`protocol/evaluation`, `active`）：M7 拒绝后 M8/M9 未授权，0 seeds、0 proposed metrics、human verdict
  为 `null`。禁止事后补 endpoint、把 0 coverage 写成提升或由 Codex 代填人评。
- `V1-F06`（`data/governance`, `active`）：早期 cut-in 路线没有官方召回率分母，strict-v2 在 675 scenes 仅
  `1 PASS / 1 scene`；这说明当前可验证事件池过稀，不说明 nuScenes 没有 cut-in。cut-in 只能作可选演示，不能再
  承担主数据入口或论文成立条件。

V1 canonical 状态、实验和专项报告保存在 `docs/archive/2026-07/dynamic-reconstruction-v1/`；其中的
`RESEARCH_FAILURES.md` 是冻结快照，不再单独维护。

### 1.2 V2 汇总条目

- `V2-F01`（`engineering`, `resolved`）：非登录 shell 中裸 `python` 不在 PATH；runner 必须显式绑定解释器，
  不运行 `conda init`，也不能把 PATH 错误写成网络/依赖失败。
- `V2-F02`（`engineering`, `resolved`）：DGGT r1–r7 依次暴露 pip backtracking、CUDA compiler/runtime mismatch、
  cusparse headers、transformers/diffusers/torch schema、`flow_vis` 和 retry schema；每次修复必须新 run，native 已完成
  的阶段不被后续 common-eval blocked 覆盖。
- `V2-F03`（`data/protocol`, `resolved`）：磁盘 `sample.json` 没有 devkit runtime `anns` 反向索引，Decimal、
  invalid projection schema 和 nearest-sweep 也会破坏 exact mapping；最终必须以 exact `sample_token` 为主键。
- `V2-F04`（`resource/protocol`, `active`）：30k checkpoint 完成不等于累积 full render 完成；后者在 577/588 时
  越过 90% cgroup 合同并安全停止。训练与 post-render 必须分开裁决，不得删 checkpoint 或写成 OOM/方法失败。
- `V2-F05`（`engineering`, `resolved`）：CUDA 扩展 import 成功不代表包含 RTX 3090 SM 8.6 kernel；必须从冻结源码
  按目标 arch 重编并运行真实 forward/backward，不能只做 import smoke。
- `V2-F06`（`data/evaluation`, `active`）：训练可把非目标 actor Gaussian slice 裁为空；registry 必须显式 unavailable，
  选定 actor 必须非空，禁止静默删 denominator 或把空 slice 当成功删除。
- `V2-F07`（`engineering/resource`, `resolved`）：外层 timeout 不会回收 `start_new_session=True` 的子进程；必须按
  精确 PGID 清理并保留 interrupted terminal，长任务使用 detached controller 和独立日志。
- `V2-F08`（`algorithm/protocol`, `active`）：M5 只完成 0230/0242 checkpoint 与 0255 诊断；三场景×两 actor×四编辑、
  pseudo-hole/perception/final matrix 未完成。空 tensor `torch.cat` 是工程阻塞，不是 3DGS 方法失败，也不允许把部分资产
  写成 M5 done。
- `V2-F09`（`evaluation/algorithm`, `active`）：lateral/delete non-target PSNR 93/95 dB 主要是硬局部保持构造，
  不能证明 source footprint 后背景、边界或时序真实；后续必须把 outside preservation 与 hole/depth/boundary/temporal
  指标分开。

### 1.3 V4 历史编号冲突校正

2026-08-17 统一账本时发现，旧追加段落重复使用了 `V4-F30`–`V4-F33`。为保证后续引用唯一，本文件将 live canonical
编号校正如下；archive 快照保持原字节和旧编号，不回写：

- B0/D0 段的 `V4-F17`–`V4-F33` 保持不变；
- 历史 M1 development/validation `V4-F30`–`V4-F34` → live `V4-F34`–`V4-F38`；
- 历史 M1 rejection/M2 validation `V4-F35`–`V4-F39` → live `V4-F39`–`V4-F43`；
- 历史 M3 `V4-F40`–`V4-F45` → live `V4-F44`–`V4-F49`。

新文档、代码和 run manifest 只引用 live canonical ID；核对旧 commit/归档时同时记录“historical ID → live ID”。

<a id="detail-v64"></a>

