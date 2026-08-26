# Motion-Proj 统一失败、风险与防重复账本

> **最后更新**：2026-08-26
> **唯一活跃失败事实源**：本文件 `docs/RESEARCH_FAILURES.md`
> **覆盖范围**：V1–V6.4、V7/V7.1、N1/cut-in 与跨路线工程/资源/协议教训
> **事实边界**：失败事实以 canonical run、`docs/EXPERIMENTS.md`、`docs/RESEARCH_STATUS.md` 和冻结证据为准

本文件是仓库中唯一持续维护的 failure ledger。`docs/archive/**/RESEARCH_FAILURES*.md` 只是对应 commit 的不可变
历史快照；`WS_*_FAILURE_FORENSICS.md` 是专项诊断报告，不是第二本账。V6.3报告索引
`docs/autoresearch/worldsim_v63/ARXIV_EVIDENCE_INDEX.md`只导航本账与canonical evidence，也不是第二本失败账。新路线、新版本和新实验不得再创建并行的
`*_FAILURES.md` 事实源。

### V6.4 当前边界（2026-08-26）

- V6.4 从`research/worldsim-v6.3-surface-tail@c192955`直接建立；`V63-F24`仍关闭 Surface family，新的合法路线只能研究
  native aleatoric/epistemic uncertainty、scene/stratum conditional risk 与独立 case-level calibration。
- 首个核心假设已冻结为原生 U0 对比 geometry-conditioned feature-density U2；旧 4+2 scene 只作机制诊断，禁止作为
  fresh V6.4 claim，也不允许由该结果读取 calibration/confirmation/test。
- retrospective U2 已在两个旧 evaluation scene 都优于 U0，但 FPR@95TPR 仍高，只授权建立 fresh cohort，不授权
  authority/calibration claim。当前 `V64-F01--F03`均为 resolved engineering/operations，不得写成算法负结论。
- compact fresh cohort 已从 V6.1–V6.3 未读 quality 的 scene 中按 metadata-only 冻结。r2 证明候选还必须存在于冻结的
  train temporal metadata；恢复队列在任何 fresh quality read 前改冻并登记`V64-F05`。不得把更早版本曾出现过但未进入
  V6.1–V6.3 UQ路线的scene错判为legacy，也不得用本轮后续质量回改cohort或复用r2部分产物。
- r3 已完整生成6-scene/72-target native sidecar，单卡资源通过；这只是capability，不得提前写成fresh UQ成立。
- fresh UQ已在target quality读取前冻结为同一PCA-16/GMM-4和两条晋级门；同数据换seed、PCA/GMM或改scene均禁止。
- fresh U2虽过相对门，但两scene内AUROC都约0.498且FPR95约0.96，登记`V64-F10 active`。后续只允许已冻结的
  fit-only PCA-16 logistic risk head执行一次；不得把监督标签用于evaluation拟合、扫描超参或扩展split。
- U3已通过两fresh scene绝对AUROC门，但高FPR95保持`V64-F11 active`。独立calibration/confirmation已按metadata-only冻结为
  `16+8 scenes`；当前与旧evaluation score均不得回流修改cohort、risk rule或head。
- P6整批准备在共享盘扫描`>1 h`后仍为`9/10 shards`且GPU空闲，登记`V64-F12 active`；恢复采用scene-ready有界
  producer-consumer，不重复已完成shard、不做无关GPU filler，confirmation仍保持锁定。
- P6 prep r1又暴露固定`1176/196`不适用于`nbr_samples=41`的scene-1045；登记`V64-F13
  recovery_frozen_pre_quality`，恢复只使用metadata派生帧数并复用已完成raw/scene。
- P6 r2已完成24场景并删除临时raw，`V64-F12/V64-F13`分别由producer-consumer与variable-length恢复；短SSH命令继承
  stdin导致本地编排不退出，登记`V64-F14 resolved_operations`并按OpenSSH官方`-n`修复。
- 独立192-case校准在最低5% coverage仍为`41/192` failure、risk/UCB=`0.2135/0.2929`，登记`V64-F15 active`；
  confirmation target保持未读，禁止放宽risk合同或删stratum。

### V6.3 报告使用边界（2026-08-26）

- active scientific negative evidence=`V63-F02, V63-F24`：原生特征没有解除逐点路线的`4/4 false-safe`，而B3
  Surface-Mean随后在P6两scene均输Native B2并按Stop2关闭surface family。
- recovered scientific failure=`V63-F19`：P5 positive-authority collapse已由P5D确认并由P5R primal-dual恢复训练侧可行性；
  该恢复不能覆盖后续P6 stage rejection。
- `V63-F01/F03--F18/F20--F23`均为resolved或resolved_preexecution工程、协议、数据表示、数值、metadata或operations
  记录；论文附录可用作复现教训，但不得当作算法negative count。
- B4/B5/M0和P7--P11是`not executed/locked`，不是失败attempt；本次文档审计没有新增failure ID，也没有重分类旧失败。

## 0. 使用合同与渐进式导航

### 0.1 渐进式读取

1. 每次研究任务先读本节和“V1–V5 版本总览”，确认当前问题属于算法、数据、评测、协议、工程、资源还是治理风险。
2. 根据 task、模块和关键词用 `rg` 定位 failure ID；只展开命中的完整条目及其相邻门禁，不默认一次加载全部长账本。
3. 新计划至少引用一个直接相关 failure ID；涉及跨路线复用时，再读取对应版本详细章和 canonical evidence。
4. 只有在做全局路线审计、迁移或报告附录时才通读全文件；归档快照仅在核对当时字节或历史编号时读取。

### 0.2 渐进式写入

1. 新坑先在当前版本详细章追加一个唯一 ID，再更新本文件顶部版本总览；不要复制整章或另建版本 failure 文档。
2. 旧结论被新证据解除时不得删除原条目；在原条目追加 `resolved/superseded`、新证据、仍然成立的边界和新 ID。
3. 写入只包含可复用的失败事实与防重复门禁；逐步日志、完整 stdout 和大表留在 run，实验结果表留在
   `EXPERIMENTS.md`。
4. 每个正式实验在启动前登记 `failure_ledger_refs`，收口时登记 `failure_ledger_delta`。任何 `blocked/rejected`、
   前提被推翻、工程恢复、门禁失败或旧风险解除，都必须在同一逻辑提交中更新本文件；若确无新增，只在实验台账写
   `failure_ledger_delta=none`，避免向失败账本灌入成功流水账。

### 0.3 单条记录最小 schema

| 字段 | 必填内容 |
|---|---|
| ID / 分类 | 唯一 `<路线>-FNN`；分类为 `algorithm/data/evaluation/protocol/engineering/resource/governance` 之一或组合 |
| 状态 | `active/resolved/superseded`；实验 task 状态仍只用 `pending/running/blocked/done/rejected` |
| 观察事实 | 分母、错误、指标、资源或 terminal；不把推断写成事实 |
| 根因与推翻项 | 已确认根因，以及它推翻了哪个假设、实现合同或旧结论 |
| 防重复与复开 | 禁止的事后调参/删分母/覆盖 run，以及合法复开需要的新证据 |
| 证据 | task/run ID、commit、summary/manifest/文档路径；工程失败与算法 reject 分开 |

### 0.4 目录

- [V1–V6 版本总览与 V1/V2 汇总](#1-v1v6-版本总览与-v1v2-汇总)
- [V6.4 详细账本](#detail-v64)
- [V6.3 详细账本](#detail-v63)
- [V6.2 详细账本](#detail-v62)
- [V6.1 详细账本](#detail-v61)
- [V6 详细账本](#detail-v6)
- [V5.1 详细账本](#detail-v51)
- [V5 详细账本](#detail-v5)
- [V4 详细账本](#detail-v4)
- [V3.3/V3.2/V3.1 详细账本](#detail-v3)
- [V2 继承门禁](#detail-v2)
- [V7/V7.1、N1/cut-in 与历史路线](#detail-legacy)
- [跨路线原则与新实验检查表](#detail-cross-route)

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
| V6.4 | U2过相对门但场景内弱；fit-only U3独立校准失败；full-native MLP以40%通过独立校准与exact-once确认（1/96 failure）；conditional确认输入恢复中 | `python -m pytest`、LocalTUN、conda、disk path、train temporal metadata、summary名、PowerShell`$()`、surface成本、region内稀疏预测类、pooled/within-scene偏移、高FPR95、整批I/O屏障、固定场景帧数、SSH stdin生命周期、batch-narrowed tar catalog、empty actor frame key、temporal split membership | `V64-F01`–`V64-F18`；`P4N_FRESH_UQ_CLOSEOUT.md`；`P5_SUPERVISED_RISK_CLOSEOUT.md`；`P4C_TEMPORAL_MEMBERSHIP_RECOVERY_FREEZE.md` |

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

## V6.4 原生不确定性编译器详细账本（2026-08-26）

- `V64-F01`（`engineering/runtime`, `resolved_pre_quality_read`）：P0 定向投影测试首次使用控制台入口
  `pytest -q tests/worldsim_v62/test_projection.py`，测试收集阶段即因仓库根目录未进入 `sys.path`触发
  `ModuleNotFoundError: motion_proj`。失败发生在 formal run、GPU context、数据与 V6.4 quality read 之前；没有科学结果。
  依据 pytest 官方 import-path 合同，仅把入口改为
  `python -m pytest -q tests/worldsim_v62/test_projection.py`，结果 `1 passed in 1.59s`。防重复：仓库内测试统一用
  `python -m pytest`；不为入口差异改包结构、污染环境或扩展回归矩阵。证据=
  `WS-V64-P0-SCOPE-GIT-01`、`docs/autoresearch/worldsim_v64/P0_SCOPE.md`、
  `https://docs.pytest.org/en/stable/explanation/pythonpath.html`。

- `V64-F02`（`engineering/operations`, `resolved_pre_formal_run`）：UQ prereg commit 首次两次普通
  `git push`在 30 秒窗口内无输出，远端 ref 保持 `04b343f`，本地 `f1764de`未丢失，formal run 尚未启动。GitHub
  官方状态页显示 Git Operations operational；本地 `localtun sessions`确认当前 AutoDL session 的 remote proxy 后，
  仅为该 push 显式设置 HTTP/HTTPS/ALL proxy，普通 push 成功推进远端到 `f1764de`。防重复：远端网络慢时先检查
  GitHub 状态和当前 LocalTUN session；端口是会话级，禁止复用旧记忆值，也不 force push。证据=`f1764de`、
  `https://www.githubstatus.com/`。

- `V64-F03`（`engineering/runtime`, `resolved_post_run_read`）：canonical UQ run 已成功结束后，第一次只读 summary
  命令在非登录 shell 直接调用裸 `python`，因环境未激活在读取文件前返回 `command not found`。按仓库环境合同 source
  `conda.sh`并激活 `motionproj`后，同一只读程序完成逐 scene 指标读取；run、summary 与模型均未修改。防重复：任何
  非登录 Python 命令显式激活环境；该错误不登记为算法或 formal run failure。证据=
  `run://worldsim_v64/WS-V64-P3-NATIVE-UQ-01/20260826T080200Z__uq-retrospective-s0-r1`、
  `https://docs.conda.io/projects/conda/en/25.1.x/dev-guide/deep-dives/activation.html`。

- `V64-F04`（`engineering/runtime`, `resolved_pre_data_read`）：fresh sidecar 首次 formal 入口
  `20260826T081300Z__fresh-native-s0-r1`在 wrapper 调用继承 runner 后，因 task parent 尚不存在而对
  `shutil.disk_usage`触发 `FileNotFoundError`。run leaf 未创建，GPU、processed scene、IR-WM 与 quality 均未触达，
  canonical run=`null`。恢复只在 wrapper 中先 `mkdir` task parent，再由未改的 runner 创建 exclusive run leaf；
  cohort、seed、资源门和 denominator 不变。防重复：disk probe 必须绑定已存在的挂载内路径，不把前置目录缺失写成磁盘
  不足或算法失败。证据=`WS-V64-P2-FRESH-NATIVE-SIDECAR-01`、
  `https://docs.python.org/3/library/shutil.html#shutil.disk_usage`。

- `V64-F05`（`data/interface`, `resolved_pre_quality_read`）：fresh sidecar r2=
  `20260826T081500Z__fresh-native-s0-r2`启动冻结 IR-WM worker 后，初始 cohort 中 val-split 的`scene-0100`与
  `scene-0632`在`nuscenes_temporal_infos_train.pkl`查询处触发`KeyError`。`scene-0230`已完成12个native units，
  worker wall=`35.8975 s`、peak GPU=`4.1305 GiB`，blocked run leaf总计`528 MiB`；其余scene未形成完整denominator，
  canonical=`null`，target evidence与任何fresh quality均未读。根因是selector只核对processed/raw可用性，却漏掉冻结
  extractor的train temporal metadata membership。检索IR-WM、BEVFormer与DriveStudio官方数据准备合同后，恢复仅在
  pre-quality 阶段改冻为六个均在train temporal metadata、且未进入V6.1–V6.3 quality ledger的scene；evaluation两scene
  从本机raw数据用官方DriveStudio流程物化。防重复：保留r2，不把12个部分unit混入r3，不生成val temporal metadata救旧
  cohort，不改变seed/targets/model/UQ/门；r3必须使用全新exclusive leaf完成72-unit denominator。证据=
  `docs/autoresearch/worldsim_v64/P2_FRESH_COHORT_FREEZE.md`、
  `https://github.com/ziyc/drivestudio/blob/main/docs/NuScenes.md`、
  `https://github.com/fundamentalvision/bevformer`、`https://github.com/APRIL-ZJU/IR-WM/blob/ir-wm/README.md`。

- `V64-F06`（`engineering/operations`, `resolved_post_run_read`）：r3 已成功输出正式summary后，首次只读收口程序
  假定文件名为`P2_NATIVE_SUMMARY.json`，对不存在路径调用`Path.read_text()`触发`FileNotFoundError`；run、GPU、
  artifacts与quality均未修改。实际继承的V6.3 extractor写出`P2_SUMMARY.json`。按Python pathlib官方合同先枚举run根目录，
  再读取实际文件；canonical r3与全部指标不变。防重复：继承runner的consumer不得根据task或版本猜文件名，先用明确目录
  枚举或读取runner源码中的输出合同；不为只读路径错误重跑formal。证据=
  `run://worldsim_v64/WS-V64-P2-FRESH-NATIVE-SIDECAR-01/20260826T082600Z__fresh-native-s0-r3`、
  `docs/autoresearch/worldsim_v64/P2_FRESH_SIDECAR_CLOSEOUT.md`、
  `https://docs.python.org/3/library/pathlib.html`。

- `V64-F07`（`engineering/operations`, `resolved_pre_run`）：fresh evidence首次launcher在Windows PowerShell传给SSH的
  双引号字符串中使用`$(git status --porcelain)`；PowerShell按官方解析规则在本地提前执行subexpression，因本地工作目录
  不是repo返回fatal。远端evidence run目录仍不存在、数据/quality/GPU均未触达。恢复仅删除嵌入式subexpression，在单独
  只读命令已确认远端branch clean与目标路径不存在后，使用同一config和固定r1启动。防重复：PowerShell到SSH的双引号
  参数不嵌入`$()`或`$var`远端shell表达式；状态检查拆成独立命令，不把本地解析错误写成formal failure。证据=
  `run://worldsim_v64/WS-V64-P2E-FRESH-EVIDENCE-01/20260826T084000Z__fresh-evidence-s0-r1`、
  `https://learn.microsoft.com/en-us/powershell/module/microsoft.powershell.core/about/about_parsing`。

- `V64-F08`（`resource/protocol`, `resolved_by_native_voxel_recovery`）：继承的V6.3 surface compiler历史formal为
  `72 units / 47,568.47 s wall / 3,334.28 s max unit`。fresh surface r1运行约4分钟时仍为`0/72 units`、4 KiB，两个worker
  CPU各约100%且无资源异常；照旧执行预计浪费约13小时，并生成当前UQ不消费的signed-distance、patch、normal、actor与
  proposal registry。检索OCCUQ ICRA 2025的原生`200×200×16` voxel-level feature GMM，以及CuPy/cuCIM exact EDT后，
  选择前者：在UQ score读取前按精确PGID停止并保留partial，预注册唯一native-boundary-voxel r1；不安装CuPy、不继续旧
  full-stack。防重复：不得把partial写成算法失败，不得删除/覆盖r1；native r1后禁止回旧surface、换EDT/denominator或
  sweep救结果。证据=`docs/autoresearch/worldsim_v64/P4N_NATIVE_VOXEL_UQ_RECOVERY_FREEZE.md`、
  `https://github.com/ika-rwth-aachen/OCCUQ`、
  `https://docs.cupy.dev/en/latest/reference/generated/cupyx.scipy.ndimage.distance_transform_edt.html`。

- `V64-F09`（`algorithm/interface`, `resolved_pre_evaluation`）：native-voxel r1在四个fit scene采样后，冻结occupied-boundary
  denominator的预测FREE geometry组仅`43`点，小于4-component GMM固定最低`80`点，`NativeFeatureDensityUQ.fit`抛错；
  r1仅8 KiB resolved/status，无model、evaluation score或gate verdict。根因是把retrospective全surface上的双预测geometry
  条件化机械搬到几乎全occupied的region。OCCUQ官方`gmm_utils.py`按真实voxel类别收集feature，并在推理时对类密度做
  `logsumexp`边缘化。恢复因此固定为region整体一个boundary-global GMM-4；不是降低样本门或扫描组件数。防重复：不得复制
  43点、降80门、把evaluation并入fit或回到双组；v2只执行r2一次，其他输入/gate不变。证据=
  `docs/autoresearch/worldsim_v64/P4N_NATIVE_VOXEL_UQ_RECOVERY_FREEZE.md`、
  `https://raw.githubusercontent.com/ika-rwth-aachen/OCCUQ/main/tools/gmm_utils.py`。

- `V64-F10`（`algorithm/evaluation`, `active`）：native-voxel r2按冻结协议完成并通过相对门：pooled U2 AUROC=
  `0.518545`、较最佳U0增`0.083047`，scene support=`2/2`。但两个scene内U2 AUROC仅`0.498387/0.498295`，
  FPR@95TPR=`0.965465/0.960623`；scene-0359 AP低于prevalence，scene-0998的50% coverage risk高于prevalence。
  pooled改善可能部分来自scene-level prevalence/score shift，不能包装成可靠场景内ranking、authority或calibration。
  顶会迁移依据：OCCUQ将dense UQ supervision与feature GMM分工；ReliOcc采用plug-and-play hybrid voxel uncertainty；EvOcc
  用evidence supervision显式建模unobserved/contradicting evidence。恢复只允许先冻结一个用四fit scenes hidden-FREE标签训练的
  轻量risk head，再在相同两scene分母执行一次；禁止扫描GMM/PCA/seed/denominator/gate或读取更多split救结果。证据=
  `docs/autoresearch/worldsim_v64/P4N_FRESH_UQ_CLOSEOUT.md`、`https://github.com/ika-rwth-aachen/OCCUQ`、
  `https://doi.org/10.24963/ijcai.2025/220`、
  `https://openaccess.thecvf.com/content/CVPR2025/papers/Kalble_EvOcc_Accurate_Semantic_Occupancy_for_Automated_Driving_Using_Evidence_Theory_CVPR_2025_paper.pdf`。

- `V64-F11`（`algorithm/evaluation`, `active`）：P5固定监督risk head按预注册通过pooled和两scene AUROC门，pooled U3
  AUROC/AUPRC=`0.658118/0.148720`，scene AUROC=`0.640682/0.636266`。但FPR@95TPR仍为pooled`0.867738`、
  scene`0.859069/0.907021`；logistic输出也未在独立calibration set上校准。故本run只支持ranking，不支持低误报authority、
  calibrated probability、conditional coverage或safety。检索ICLR 2024 Conformal Risk Control后，合法恢复必须先冻结新的
  scene-disjoint calibration/confirmation cohort，以scene/unit为交换单元选择单调selective set，并在untouched confirmation
  一次验证；不得用已读scene-0359/0998选threshold、扫描risk/coverage或把voxel当独立样本制造虚假样本量。证据=
  `docs/autoresearch/worldsim_v64/P5_SUPERVISED_RISK_CLOSEOUT.md`、
  `https://proceedings.iclr.cc/paper_files/paper/2024/hash/f3549ef9b5ff520a7e41ff3cc306ab2b-Abstract-Conference.html`、
  `https://github.com/aangelopoulos/conformal-risk`。

- `V64-F12`（`resource/operations`, `resolved_by_pipeline`）：P6正式准备入口扫描共享盘官方tar超过一小时后，已完成
  `9/10`个shard、临时raw约`14 GiB`、盘余量约`45 GiB`，但旧入口必须等全部raw和24个processed scene完成才启动
  IR-WM，观测GPU=`0% / 1 MiB`。24-scene native按既有6-scene输出外推约`13.3 GiB`，再叠加processed和临时raw，
  证明此前`~21.6 GiB`整批持久化估算缺少足够余量。该事实是I/O/调度阻塞，不是模型或数据质量结论。检索NVIDIA
  DALI异步pipelined execution、bounded prefetch queue及WebDataset shard streaming后，恢复保留已完成shard工作，并在
  DriveStudio scene达到冻结的`1176 images + 196 lidar`后立即按scene送入IR-WM，最多两个GPU worker；先处理16-scene
  calibration，模型冻结后再处理锁定confirmation。禁止重复扫描已完成shard、启动无关GPU filler、降低quality边界、把
  confirmation提前读入校准或用多卡掩盖共享盘瓶颈。证据=
  `run://worldsim_v64/WS-V64-P6-CALIBRATION-SIDECAR-01/20260826T100000Z__calibration-prep-s0-r1`、
  `https://docs.nvidia.com/deeplearning/dali/archives/dali_190/user-guide/docs/advanced_topics_performance_tuning.html`、
  `https://github.com/webdataset/webdataset`。

- `V64-F13`（`data/interface`, `resolved_pre_quality_read`）：prep r1在全部tar扫描完成、首个scene-1045官方
  DriveStudio转换成功后，以`images=1206, lidar=201`对旧六场景硬编码的`1176/196`做比较并抛错。r1未读
  Occupancy/UQ/hidden-FREE或calibration/confirmation质量；临时raw和完整首场景均保留。根因是nuScenes scene记录有
  `nbr_samples`，而`interpolate_N=4`的官方DriveStudio时间表长度为`(nbr_samples-1)*5+1`；scene-1045的
  `nbr_samples=41`故应为201帧、六相机1206图，不是文件缺失或重复。恢复只从冻结metadata派生每scene期望数，并让新r2
  显式复用现有临时raw和完整scene；不删除额外合法帧、不重扫tar、不改变12 target、cohort、seed或backend。首场景已独立
  完成`12/12` native targets，证明201帧接口可供IR-WM消费，但不构成质量结论。证据=
  `run://worldsim_v64/WS-V64-P6-CALIBRATION-SIDECAR-01/20260826T100000Z__calibration-prep-s0-r1`、
  `run://worldsim_v64/WS-V64-P6-CALIBRATION-SIDECAR-01/20260826T111500Z__calibration-native-scene-1045-s0-r1`、
  `https://github.com/ziyc/drivestudio`、`https://www.nuscenes.org/nuscenes?frame=0&sceneId=scene-0011&view=regular`。

- `V64-F14`（`engineering/operations`, `resolved_pre_quality_read`）：Windows PowerShell中的两个长驻feed lane反复调用
  短`ssh` readiness/publish命令；远端命令已经结束，但client继承PTY stdin后未退出，导致下一scene不推进。一次lane恢复时
  scene-0810的原远端worker仍在运行，新wrapper只在发现run leaf已存在时抛`FileExistsError`，没有覆盖或重复GPU计算；
  原worker随后正常完成`12/12`。OpenSSH官方手册明确后台/编排调用用`-n`禁止读取stdin；所有短检查、publish和wrapper调用
  加`-n`后lane连续推进，双worker达到100% GPU。防重复：不得因client挂起杀未知远端进程或新建重复run；先查remote PID/
  summary，再恢复缺失scene。证据=`https://man.openbsd.org/ssh`及P6逐scene run leaves。

- `V64-F15`（`algorithm/evaluation`, `resolved_by_new_version`）：冻结U3在16个独立calibration scene的192个case上没有任何正coverage
  通过case risk合同。最低5% coverage已有`41/192` failure，empirical risk=`0.213542`、simultaneous UCB=
  `0.292860`；night/vulnerable-transit分别`16/48`与`13/48`，所以不是Bonferroni或Clopper-Pearson过严。10%到50%
  coverage的failure继续增至`54,62,74,80,93`。根因边界是PCA16线性risk ranking不能跨新night/rain/construction/
  vulnerable场景提供case-level hidden-FREE控制；P5两scene AUROC通过不再足以解锁calibration/authority。confirmation target仍
  未读。禁止降低epsilon=0.05、提高conflict threshold=0.05、删stratum、读confirmation选策略或添加<5%事后coverage。
  合法复开必须是新模型版本：16个已消费scene只作development training，当前8个untouched scene作独立calibration，并先
  metadata-only冻结新confirmation。迁移依据=`https://proceedings.mlr.press/v97/geifman19a`、
  `https://proceedings.neurips.cc/paper/2019/hash/0c4b1eeb45c90b52bfb9d07943d855ab-Abstract.html`、
  `https://openaccess.thecvf.com/content_iccv_2017/html/Lin_Focal_Loss_for_ICCV_2017_paper.html`；closeout=
  `docs/autoresearch/worldsim_v64/P6_CASE_CALIBRATION_CLOSEOUT.md`。
  迁移在读取原confirmation前已冻结：16个已读scene仅作development，原8个quality-unread scene转独立calibration；新
  confirmation按剩余metadata-only pool/seed1固定为`1023,1105,0903,0451,0981,0537,0789,0157`。模型固定为完整
  273D的`128/64` focal-loss MLP且不做超参扫描。此冻结没有读取新quality、没有产生新failure ID；详见
  `docs/autoresearch/worldsim_v64/P6R_SELECTIVE_MLP_FREEZE.md`。
  第一阶段正式训练已完成：`786054` points，loss=`0.0337864->0.0251443`，development AUROC=`0.8811503`
  （仅描述），GPU fit=`10.1545 s`。模型现已冻结且原8-scene calibration仍未读；这既不关闭V64-F15，也不产生新
  failure。下一判定只来自预注册的96-case独立校准。
  独立证据现已在模型冻结后一次完成`8 scenes/96 units`，source-role overlap与query均为0；尚未读取模型分数或选择
  coverage，故V64-F15状态不变且没有新增failure ID。
  P6R独立评分随后以0.05--0.40 coverage全部得到`0/96` failure和simultaneous UCB=`0.048647`，选择最大通过40%；
  50%为`3/96`、UCB=`0.103218`而正确拒绝。故失败以“新模型版本解决”关闭；原PCA16线性U3负结论不改写，且新
  confirmation仍未读。证据=`run://worldsim_v64/WS-V64-P6R-CALIBRATION-01/20260826T141500Z__case-calibration-s0-r1`。
  exact-once confirmation只在冻结40%上读分一次，得到`1/96` failure；四strata分别`0/24,1/24,0/24,0/24`，
  总体和分层gate均通过。V64-F15因此以独立校准加新确认的完整新版本证据收口，但不产生现实安全声明。

- `V64-F16`（`resource/operations`, `resolved_by_scene_ready_streaming_and_catalog_finalize`）：exact-once新8 scene在现有raw cache均无payload；需要约24.8k sensor member。
  前一P6整批扫描虽已学习43033个member->shard映射，但`scan_shards`写回时只保留当前batch，故unseen batch仍会触发10个
  `.tgz`全扫，预计重现>1h GPU空转屏障。WIDS明确把index用于稀疏random access，ratarmount为compressed tar持久化SQLite
  index；结合当前代码不新增依赖，迁移为superset member->shard catalog，并以scene raw-ready为边界并发DriveStudio preprocess
  和最多两个IR-WM consumer。本批不可避免的一次scan完成后目录可复用；禁止重新裁catalog、等待整批processed才启GPU或用
  多卡掩盖I/O。证据=`https://github.com/webdataset/wids`、`https://github.com/mxmlnkn/ratarmount`和
  `docs/autoresearch/worldsim_v64/P6R_CONFIRMATION_EXECUTION_FREEZE.md`。
  下游exact-once合同已在target read前固定为40% policy、overall最多4/96且每stratum最多1/24 loss；不以本I/O失败
  改变科学gate。
  scene-ready priority scheduling已完成全部`8 scenes/96 blind targets`：先完成单shard/相关shard组，DriveStudio与IR-WM按
  scene流水，最大worker显存`4.1314 GiB`；没有等待整批processed或启用多卡。故GPU-idle/全批屏障部分已恢复；superset
  catalog的剩余EOF写回和可重建临时raw删除仍由原prep controller收口，完成后再把V64-F16标为resolved。
  exact-once评分后剩余scanner恢复并正常EOF：prep=`20260826T143000Z__confirmation-prep-s0-r1`完成8 scene，superset
  catalog=`57338 entries / 6880063 bytes`，temporary raw由controller删除。至此资源failure正式关闭；总wall `5872.4206 s`
  保留为首次稀疏scan成本证据，不把它误写成GPU wall。

- `V64-F17`（`data/interface`, `resolved_pre_score`）：exact-once evidence r1完成33/96 units后，scene-1105 frame62
  在`load_frame_boxes`用直接dict索引触发`KeyError`。processed审计显示该scene缺0--9、56--64的frame_instances键；这些
  frame在instances_info中逐一为0 annotation，`missing_with_annotations=[]`，不是sensor缺失或hidden target异常。nuScenes
  官方devkit对non-keyframe box使用相邻sample annotation插值，没有annotation时返回空/当前集合；故common loader把缺键解释为
  empty actor list。r2以hardlink复用33个完整NPZ、只算剩63；NPZ未存储的三个summary字段显式null，不伪造。禁止重算33、
  改scene/policy/gate或用target score挑恢复。证据=`https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/nuscenes.py`
  和`docs/autoresearch/worldsim_v64/P6R_CONFIRMATION_EVIDENCE_RECOVERY_FREEZE.md`。
  r2 canonical=`20260826T152500Z__confirmation-evidence-s0-r2`按上述合同完成`96/96`，其中33 hardlink复用、63新算，
  query/role overlap均0，wall=`74.6360 s`；模型分数仍未读，故以pre-score状态关闭。
  随后的exact-once评分成功消费该证据一次且未触发第二次恢复，确认本interface failure没有改变冻结策略或coverage。

- `V64-F18`（`data/interface`, `resolved_pre_quality`）：P4C v1 metadata-only selection只检查nuScenes scene table、
  sample count与used-scene ledger，漏掉IR-WM train temporal pickle membership。7个scene完成blind native；`scene-0276`
  DriveStudio完成后在worker读取`payload["infos"][scene]`处`KeyError`，native output/target/model score均未读。官方BEVFormer/
  IR-WM使用生成的temporal train/val infos，不能用未分割scene table替代membership。恢复保留7个valid leaf并只替换无效scene：
  从commit `4813438`重建seed2 fallback；`scene-0572`是`flipped`误命中substring `ped`，首个token-valid且temporal-member
  的vulnerable候选为`scene-0813(631)`。不得重选其余7 scene、改policy/model/gate或读取quality挑替换。
  因v1 controller持有旧catalog snapshot，replacement写独立JSON并在两者结束后union，避免两个`os.replace` writer互相丢更新；
  证据=`https://github.com/APRIL-ZJU/IR-WM/blob/ir-wm/README.md`、`https://github.com/fundamentalvision/BEVFormer`和
  `docs/autoresearch/worldsim_v64/P4C_TEMPORAL_MEMBERSHIP_RECOVERY_FREEZE.md`。
  冻结replacement随后完成raw准备、DriveStudio与blind IR-WM native；corrected aggregate复用7个valid leaf并加入
  `scene-0813`，得到`8 scenes/96 targets/4423846027 bytes`，maximum worker peak=`4.1314 GiB`。全过程未读取confirmation
  target/quality/model score，未改变C0/M0、模型、gate或96-case denominator，故本interface failure在quality read前关闭。
  canonical=`run://worldsim_v64/WS-V64-P4C-CONDITIONAL-CONFIRMATION-SIDECAR-01/20260826T170000Z__native-aggregate-s0-r1`。
  corrected evidence随后一次完成`96/96 units`、query/source-role overlap=`0/0`且未触发同类membership或actor-frame错误；
  model score仍未读，没有新增failure ID。evidence=`run://worldsim_v64/WS-V64-P4C-CONDITIONAL-CONFIRMATION-EVIDENCE-01/20260826T171500Z__confirmation-evidence-s0-r1`。
  exact-once scorer随后只读冻结C0/M0一次，两臂均为`0/96` failure且M0 coverage uplift=`0.0750164`，三项gate全部通过；
  没有触发第二次恢复、mapping选择或新failure。本条保持`resolved_pre_quality`，并由fresh positive result确认恢复没有改变合同。
  下游P10M state-bake在该结果后冻结为target-free materialization：只读METHOD/native/model并让package-only consumer读取结果；
  未发现新blocker或新增failure ID，不回开本条。
  P10M formal随后一次完成96个package，M0比C0新增`74499`个emitted voxels且两项gate通过；state bake target read=false、
  runtime model/evidence access=false，没有新增failure ID。该结果不把voxel materialization外推为GS/sensor/collision authority。

- `V64-F19`（`integration/resource`, `resolved_by_sparse_gaussian_adapter`）：P10M fresh cohort的8个nuScenes scene均无同场景StreetGS/
  SceneIR checkpoint；旧V6 GS runtime只绑定其他scene，并以manifest SHA256/双bake bit-exact为入口，直接复用既不满足same-scene
  语义，也违反V6.4禁止新增hash/checksum/fingerprint的约束。不得把跨scene checkpoint硬接到fresh package、恢复旧hash治理，或把
  voxel package直接称为photorealistic GS。检索GaussianFormer的sparse semantic Gaussians、GaussianWorld的
  `{position,scale,rotation,semantic,feature}`表示与GaussianOcc的voxel-grid uniform scale/identity rotation后，冻结P10G最小迁移：
  每个M0 emitted voxel一个Gaussian，fixed `scale=0.256m/opacity=0.95/identity rotation`，GPU probabilistic BEV splat；只读P10M
  package，不读target/model/StreetGS。证据=`https://github.com/huang-yh/GaussianFormer`、
  `https://openaccess.thecvf.com/content/CVPR2025/papers/Zuo_GaussianWorld_Gaussian_World_Model_for_Streaming_3D_Occupancy_Prediction_CVPR_2025_paper.pdf`、
  `https://openaccess.thecvf.com/content/ICCV2025/papers/Gan_GaussianOcc_Fully_Self-supervised_and_Efficient_3D_Occupancy_Estimation_with_Gaussian_ICCV_2025_paper.pdf`。
  formal P10G一次完成`96/96` package，生成`534581`个M0 semantic Gaussians并在GPU BEV splat获得相对C0
  `+41016` support cells；target/model/StreetGS access均false，两项gate通过。因此“无法在no-hash/same-scene约束下进入任何
  Gaussian consumer”的integration blocker关闭；photorealistic StreetGS/sensor binding仍不在该恢复声明内。
  后续P10R直接冻结为logged future-lidar route corridor semantic consumer；只读P10G package与pose，不产生新failure ID，也不
  用route overlay冒充photorealistic或collision recovery。
  P10R formal在36/96 cases得到`+375` route support cells并通过冻结门，但C0/M0 binary intercept均为96/96、additional intercept
  cases=0；这是明确的metric saturation边界而非新implementation failure。不得把support gain包装成更多collision case被拦截。

- `V64-F20`（`evaluation/metric`, `resolved_by_route_local_cell_severity`）：P10R binary route intercept对C0/M0均为`96/96`，使case-level
  hit metric完全饱和；虽然M0在36 cases新增375 support cells，但不能回答这些新增state是否把hidden FREE写成OCCUPIED。不得事后缩短
  horizon、缩小corridor或提高density threshold制造未饱和case。Waymo Occupancy Flow以固定current-ego grid做cell-level occupancy
  metric，Implicit Occupancy Flow允许planner在连续时空点query，soft collision optimization使用连续势能而非binary hit；迁移为
  同一冻结2s/1.5m corridor上的route-local target hidden-FREE rate。policy/model/route均不改，只允许一次target audit；pooled M0
  conflict门保持原0.05，case failure只描述。证据=`https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/protos/occupancy_flow_metrics.proto`、
  `https://openaccess.thecvf.com/content/CVPR2023/html/Agro_Implicit_Occupancy_Flow_Fields_for_Perception_and_Prediction_in_Self-Driving_CVPR_2023_paper.html`、
  `https://openaccess.thecvf.com/content/CVPR2023W/E2EAD/papers/Kedia_Integrated_Perception_and_Planning_for_Autonomous_Vehicle_Navigation_An_Optimization-Based_CVPRW_2023_paper.pdf`。
  P10C一次读取冻结target后得到M0 route emitted=`10013`、conflict=`43`、pooled rate=`0.004294`，并相对C0新增563 state；
  cell-level severity因此成功打破binary metric saturation，本条关闭。但5/96局部case仍超0.05，转入V64-F21，不由pooled pass覆盖。

- `V64-F21`（`evaluation/tail-risk`, `closed_negative_tail_authority`）：P10C pooled M0 route hidden-FREE rate虽仅
  `0.004294`，仍有5/96 case局部超过0.05，最高`scene-0895/f152=0.106383`；其余包括`0876/f047=0.076923`、
  `0876/f182=0.069444`、`0454/f122=0.063492`、`0895/f137=0.057692`。这阻止把pooled severity提升为route/collision
  authority。不得在已读target上调policy、改route或挑scene。CVaR对最坏尾部而非均值进行风险汇总，且PAC-Bayesian CVaR工作
  明确区分empirical tail与generalization bound；故只冻结alpha0.10/worst10 empirical audit，M0门仍0.05，不做优化或population
  声明。证据=`https://proceedings.neurips.cc/paper/2015/hash/64223ccf70bbb65a3a4aceac37e21016-Abstract.html`、
  `https://proceedings.neurips.cc/paper/2020/hash/d02e9bdc27a894e882fa0c9055c99722-Abstract.html`。
  P10T rows-only formal得到C0/M0 worst10 empirical CVaR=`0.0504298/0.0517085`，M0比C0高`0.0012787`且超过0.05；
  verdict=`rejected_empirical_route_tail`。本次未重读target、未改policy或tail fraction。故current M0 route/collision tail authority
  正式关闭并锁定P11；不得用P4C pooled/fresh pass、P10G support或P10R exposure覆盖该负结论。合法恢复必须是新版本、独立
  calibration与新confirmation，不能回调本次frozen M0。
  新版本恢复已冻结为P10R2/M1：把已消费P6R confirmation降格为development/calibration cohort，保持每case总selected count与
  M0相同，仅把route corridor名义覆盖限制到独立C0=`0.40`，并按原冻结risk score把释放预算重分配到non-route；模型、M0
  stratum coverage、2s/1.5m route与worst10 tail均不变。M1只可在该consumed cohort形成candidate，不能关闭本条；若两项冻结门
  通过，仍需metadata-only冻结全新temporal-member confirmation并exact-once确认。不得扫route cap、改尾部比例、重训模型，或用
  M1 calibration结果改写current M0 negative closeout。
  P10R2 formal一次完成：总coverage delta=`0`，route selected `5912->3826`、hidden-FREE conflict `23->9`，route worst10
  empirical CVaR `0.0220499->0.0114783`，M1最大case rate=`0.0454545`，两项candidate gate通过。该结果来自已消费cohort，
  因此只支持进入fresh confirmation；V64-F21仍保持`closed_negative_tail_authority`且P11仍锁定。下一步不得复用该cohort作确认，
  也不得因margin较大再扩route cap；只允许按metadata冻结未读质量的新temporal-member cohort并exact-once检验固定M1。
  fresh confirmation现已在任何target/model-score read前按seed3冻结为`1020,1016,0596,0590,0006,0472,0070,0371`，
  8/8均为IR-WM train temporal member且>=40 samples。选择只用description/name/count/index与当前124-scene排除集合；固定
  M1、2s/1.5m route、worst10和两项gate均未变。该prereg不关闭V64-F21；只有新96-case exact-once结果才能决定M1是否获得
  bounded fresh empirical route-tail authority，且无论结果如何都不改写历史M0负结论。

- `V64-F22`（`resource/operations`, `resolved_by_io_reassignment`）：P4C科学链已完成并推送后，两套cleanup controller仍为
  不同required-member集合重复顺序扫描同一10个official tar，持续占用NVMe且不产生新科学证据；这会把P10R2 fresh
  confirmation的scene-ready GPU feed推迟到可选catalog union之后。再次确认P4C native aggregate、evidence与exact-once summary
  均完整后，终止两个scanner tree；不把未完成union写成成功，而是明确放弃optional catalog enrichment。仅删除预注册为official
  tar可恢复的`worldsim_v64_p4c_raw_batch`与`worldsim_v64_p4c_replacement_raw_batch`（约6.8GiB），保留全部processed、native、
  evidence、model、run artifacts及已有`57338-entry/6880063-byte`catalog。I/O随后只服务P10R2一套新扫描与scene-ready feeder；
  不新增hash/checksum/fingerprint，也不改变任何科学policy/gate/result。

- `V64-F23`（`resource/scheduling`, `recovery_frozen_pre_target`）：P10R2 10-shard scan完成并成功流水化`0590/0596/0070`
  三个native leaf后，`scene-1020(778)`已由第二producer写成canonical processed，但对应feeder线程仍排在另一长耗时
  preprocess mutex后，形成head-of-line blocking并让GPU空闲。不得增加无关GPU filler或重算有效leaf。NVIDIA DALI明确以
  asynchronous pipelined execution和分离CPU/GPU prefetch queues隐藏阶段时延；迁移为同一feeder prefix的可恢复调度：启动先复用
  `passed=true,target_count=12` leaf，canonical processed直接绕过preprocess lock进入GPU semaphore。只丢弃当前可从raw重建的
  staging partial；cohort/model/policy/targets/gates/canonical IDs完全不变，target与model score仍未读。证据=
  `https://docs.nvidia.com/deeplearning/dali/user-guide/docs/pipeline.html`。
  ready-first恢复最终复用6个complete leaf，并只对`0006/0371`启动两个GPU worker；8 leaf全部12/12通过，本条关闭为
  `resolved_by_ready_first_resume`。

- `V64-F24`（`resource/operations`, `resolved_by_producer_single_owner`）：全shard scan结束后，prep主循环与feeder各自成为
  DriveStudio producer，先对不同scene并行有利，但随后同时开始`scene-0371(288)`，若均完成会竞争同一canonical目录。
  在任何duplicate canonical write前终止较晚的prep producer/tree，保留feeder较早staging、已落盘catalog和全部complete outputs。
  feeder第一次恢复还遇到`scene-0006`仅有run目录无summary的中断partial；确认无进程占用且仅1个未完成文件后精确删除并从
  complete canonical processed重建。最终8/8 processed、8/8 native、96 targets全部通过。prep以新r2和
  `--reuse-temporary-raw`只读8个complete canonical scene，`0.8171s`写summary并删除raw，不重扫tar/重做preprocess。
  防重复：scene-ready阶段只有feeder拥有producer写权；prep在stream结束后只作reuse finalize。科学cohort/policy/target lock未变。
  后续fresh evidence一次完成96/96 units、0 reuse与0 source-role overlap，未复发temporal membership、producer或partial问题；
  V64-F23/F24保持关闭，不新增failure ID。target现已读，故此后只允许预注册exact-once scorer，不得再改M1或cohort。

- `V64-F25`（`evaluation/generalization`, `resolved_exact_empirical_cohort_relative_confirmation`）：P10R2 prereg的绝对M1 route CVaR门在fresh
  96 cases通过（`0.0403133<=0.05`），总coverage严格保持，故formal verdict按合同为supported。但calibration中的相对改善没有
  确认：fresh M0 CVaR=`0.0391815`，M1-M0=`+0.0011318`；M1 pointwise failures从1增至2、maximum从`0.06818`升至
  `0.08333`。同时M1 route selected/conflicts从`8117/54`降到`4971/20`，说明绝对冲突质量下降但case-rate尾部受更小分母与
  稀疏离散事件支配。不得用absolute gate pass声称相对改善，也不得在已读confirmation上调route cap、tail fraction或挑case。
  合法下一步必须先检索denominator-stable sparse risk / occupancy-flow severity方法，再冻结rows-only诊断或新版本；P11 comparative
  authority保持锁定。current M0 P10T负结论与M1 absolute fresh pass分别保留，互不覆盖。
  检索Waymo Occupancy Flow fixed ego-grid cell metrics、Occupancy Flow Fields与Implicit Occupancy Flow后，冻结P10R3为
  `conflict count / route-eligible voxel count`的fixed-opportunity rows-only诊断，在consumed calibration与fresh confirmation分别
  使用同一worst10。该post-hoc诊断无confirmatory gate，不可关闭本条；只用于判断selected-only可变分母是否解释方向反转，且
  不得借结果回调M1或解锁P11。证据=`https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/protos/occupancy_flow_metrics.proto`、
  `https://waymo.com/research/occupancy-flow-fields-for-motion-forecasting-in-autonomous-driving/`、
  `https://openaccess.thecvf.com/content/CVPR2023/papers/Agro_Implicit_Occupancy_Flow_Fields_for_Perception_and_Prediction_in_Self-Driving_CVPR_2023_paper.pdf`。
  P10R3 canonical rows-only结果在consumed calibration与fresh confirmation的固定分母worst10均为M1更低：
  `0.0132351->0.00455240`与`0.0216470->0.0149832`；pooled density也分别下降`0.00143870/0.00265563`。
  这使“selected-only可变分母导致方向反转”成为一致的描述性诊断，但P10R3是在读过fresh confirmation后冻结，不能作为独立
  confirmation；本条继续active，禁止据此回写P10R2 formal verdict或解锁P11。下一合法动作是先检索paired sparse-event
  confirmatory设计，再决定是否在从未读quality的test cohort冻结一次固定分母exact-once，不能复用已读cohort做显著性包装。
  检索NeurIPS 2021 `rliable`与ICLR 2024 Conformal Risk Control后，P10R4冻结一次untouched 96-case test：保留个体成对
  方向作描述，不做bootstrap/significance；CRC因目标是单调expected loss而不迁移为tail gate。三项confirmatory gate只包含
  coverage保持、fixed-denominator worst10不劣、pooled fixed density不劣。test未读前不设最小effect、不改M1；若失败则本条
  terminal rejected并保持P11锁定，若通过也只关闭exact empirical cohort层面的relative问题。输入I/O改为单遍metadata与
  raw-only producer/单feeder，避免GPU因重复8次`sample_data.json`扫描或duplicate preprocess owner空等。
  P10R4唯一untouched exact-once最终三门全过：coverage delta=`0`，fixed-opportunity worst10 M0/M1=
  `0.020725740/0.010821074`（delta=`-0.009904666`），pooled fixed density=`0.004944667/0.002001413`
  （delta=`-0.002943254`）；paired M1 lower/equal/higher=`18/78/0`。因此本条只在独立96-case exact empirical cohort的相对
  fixed-opportunity层面关闭，并解除由本条造成的P11 bounded-design锁。不得把它写成P10R2 selected-denominator formal重判，
  也不覆盖V64-F21对current M0的负结论；population、physical collision、planning、closed-loop与safety仍无authority。

- `V64-F26`（`io/execution`, `resolved_by_restricted_shards_and_dual_queue`）：P10R4首个raw-only入口发现`14437` required members均不在持久catalog，
  10个`.tgz`并发扫描约4分钟仅到`4--10%`，workers主要处于page wait且GPU尚无完整scene。CPython tarfile对gzip selected
  members仍需顺序流；ratarmount/rapidgzip可建seek-point index，但为一次性cohort新建10份index仍先消耗全量扫描。
  现有`71555`条semantic member→shard catalog显示七scene的capture prefix唯一落在05/06/08/10，`scene-0668`由相邻
  temporal range与已经原子落盘的exact-prefix files冻结到07。恢复只扫描`05,06,07,08,10`，保留all-shard尝试已完成文件，
  原workers停后只删除其`.partial.<pid>`；继续使用已运行的唯一feeder，不启动第二preprocess producer。若任一member找不到，
  restricted scan必须失败并回到未猜测的全量扫描，不得换scene或读quality。科学合同与test unread状态不变。
  restricted r1在进入scan前因既有resume目录仍执行`mkdir(exist_ok=false)`退出；该run不含新增archive读、preprocess、GPU或
  target read。恢复只把显式`--resume-raw-scan`的mkdir改为`exist_ok=true`，默认新run防覆盖不变；以新r2继续，不新增failure ID。
  r2使scene-0598 native以`45.4004s/4.1314GiB`完成，但单preprocess mutex下一scene转换超过2分钟，GPU再次出现供给缺口。
  按既有DALI分离CPU/GPU queue依据，停止feeder parent但让唯一in-flight scene-0462预处理完成，保留0598 native；同prefix
  feeder恢复为两个独立per-scene staging与`2 preprocess / 2 native` slots。不得对同scene启动第二owner；完整canonical/native
  必须reuse。科学合同与test unread不变，本恢复仍归V64-F26。
  canonical r2最终在`1807.8114s`扫描05/06/07/08/10并找齐`14437/14437`；per-shard命中
  `5401/1824/1818/1783/3611`也揭示capture prefix会跨archive boundary，但冻结五分片union完整。catalog增至
  `85992 entries`。raw完成时双队列已完成0598/0462 native且GPU峰值均`4.1314GiB`，故本条关闭；若后续native/evidence
  出现科学或独立工程故障应另记，不得重开全量tar scan。

- `V64-F27`（`io/execution`, `resolved_by_exact_stage_path_and_reuse`）：双preprocess独立target已完成scene-1084/1081，但DriveStudio实际将
  `..._processed_824`重写到`..._processed_10Hz_824/trainval/824`；feeder按常规append查找`..._824_10Hz`，因此在
  canonical install/native前抛出。824/821 stage分别有完整`1206/201`与`1176/196` images/lidar且无native partial。
  parent已停，唯一in-flight 424/522不终止、不重复；修复只镜像DriveStudio既有字符串重写。进程退出后四scene原子安装，
  night两scene用冻结underlying native command与原计划run dirs直接供GPU，patched feeder随后同prefix复用。若任一stage计数
  不完整则只重建该scene；不得删除完整stage、换scene或读test quality。
  恢复最终复用4个complete native leaf，并对其余4 scene完成同prefix native；最后两scene从stage ready到native启动仅等待
  `0.0646/0.0625s`。aggregate为`8 scenes / 96 targets / 4423846058 bytes / passed`，峰值worker显存`4.1314GiB`；
  test target/quality/model score仍未读。finalizer只登记8个complete canonical scene并删除可重建raw，故本条关闭；后续evidence或
  exact-once若失败必须按其实际阶段登记，不能重跑native或改cohort。

<a id="detail-v63"></a>

## V6.3 SurfNCC 防重复结论（2026-08-24）

- `V63-F01`（`engineering`, `resolved`）：P0 integration branch 的首次定向验证错误指向不存在的
  `tests/test_worldsim_v62_projection.py`，第二次改到真实文件后又因 pytest 进程未包含 repo-local `PYTHONPATH` 而在
  import collection 阶段触发 `ModuleNotFoundError: motion_proj`。两次均未读取数据/quality、未运行 GPU、未改变源码，
  不是 V6.2 projection 回归或 V6.3 算法失败。唯一恢复是在同一 conda 环境用
  `PYTHONPATH=. pytest -q tests/worldsim_v62/test_projection.py`，结果 `1 passed`。防重复：远端 repo 的定向 pytest
  必须使用真实 `tests/<version>/...` 路径并显式提供 repo-local import path；不为这一入口错误增加 smoke/regression
  矩阵。证据=`WS-V63-P0-SCOPE-GIT-01` 与 P0 shell terminal。

- `V63-F02`（`algorithm/evaluation`, `active`）：P2D canonical=
  `20260824T145924Z__native-pointwise-s0-r1` 用冻结 P5 best 与真实 per-cell IR-WM logits/BEV 执行 unchanged legacy28
  gate，Native B2仍为`4/28 ACCEPT,4/4 false-safe`，接受集合仍是scene-0242四个missing-route-support cases；R10=
  `2/3`、Actor gain=`0`、static/disocclusion gain=`2`、mask-area=`0.094024`、accepted FREE conflict mean/worst=
  `0.045783/0.092105`。source-valid UNKNOWN=`0.639211`，safe-OCC retention=`1.0`，hard violations=`0/939206`。
  与prototype P6/P6R相同的接受集合和false-safe说明V62-F05的feature bridge是加重因素而非主因；已推翻“只要恢复
  native feature，逐voxel CPSC即可获得hidden-surface authority”。防重复：不得对P2D重训、调threshold/seed/grid、
  用legacy O_eval选epoch或把mean conflict过门包装成安全。按预注册迁移到P3：Point Transformer V3官方实现支持
  efficient serialized point neighborhoods，visibility-aware reconstruction明确把FREE visibility作为surface约束，CVaR
  直接优化局部尾部；V6.3只迁移已在P1冻结的deterministic surface topology + patch CVaR，不改变alpha/cohort/gates。
  证据=`docs/autoresearch/worldsim_v63/P2D_NATIVE_POINTWISE_PREREG.md`、
  `https://openaccess.thecvf.com/content/CVPR2024/papers/Wu_Point_Transformer_V3_Simpler_Faster_Stronger_CVPR_2024_paper.pdf`、
  `https://pmc.ncbi.nlm.nih.gov/articles/PMC4897344/`。

- `V63-F03`（`engineering`, `resolved`）：P3 probe r1=`20260824T150842Z__surface-probe-s20260824-r1`
  在`_native_occupied_target_grid`对长度`300/300/40`的三个target-grid axis arrays调用`numpy.stack`时触发
  `ValueError: all input arrays must have the same shape`。该返回值未被调用方消费，失败发生在surface extraction、
  target supervision与任何quality gate之前，run仅4 KB，不能写成surface方法失败。NumPy官方`stack`合同要求每个输入
  shape相同；恢复为返回三个独立axis arrays，并在同轮pre-run audit中把route-support类型更新限定到对应local surface、
  法向量统计限定为finite unit vectors、target grid超出native z范围的点显式标为invalid而非用`100% valid`作错误门禁。
  这些都是接口/统计修复，不改变proposal volume、6-connected topology、patch参数、cohort或科研门槛。r1不可覆盖；
  revision 2复用冻结配置。防重复：不同长度的坐标轴不得stack；native coverage必须作为显式映射事实交给后续模型处理，
  不得偷偷clip或删除proposal。证据=`docs/autoresearch/worldsim_v63/P3_SURFACE_CORPUS_PREREG.md`、
  `https://numpy.org/doc/2.0/reference/generated/numpy.stack.html`。

- `V63-F04`（`engineering`, `resolved`）：P3 probe r2=`20260824T151429Z__surface-probe-s20260824-r2`
  在runner入口触发`FileExistsError`：外层launcher先`mkdir`了叶run directory，runner为保护不可变run又显式拒绝已存在
  路径。0 unit、0 surface、0 quality read；不能解释为F03恢复失败或科研结果。Python官方`Path.mkdir`说明默认
  `exist_ok=False`时目标存在即抛`FileExistsError`。恢复只让launcher确保task父目录存在、把叶目录留给runner原子创建；
  不修改源码、配置或科研合同。r2目录和console保留，revision 3使用新路径。防重复：带immutable-run自建语义的runner
  不得由外层预建叶目录。证据=`docs/autoresearch/worldsim_v63/P3_SURFACE_CORPUS_PREREG.md`、
  `https://docs.python.org/3/library/pathlib.html#pathlib.Path.mkdir`。

- `V63-F05`（`engineering/data-representation`, `resolved`）：P3 probe r3=
  `20260824T151618Z__surface-probe-s20260824-r3`首次完整产出`191 surfaces/498 patches/152226 points`，但101个
  微小static components的至少一点法向量无效（85个singleton，其余component size 3–11），令minimum normal-valid=
  `0`、probe未过。根因不是surface缺失：对称孤立voxel的六个外露面法向量相消，centroid fallback在离散medial-axis点
  也为零。Gradient-SDF说明SDF梯度给出normal但medial axis因最近surface不唯一而奇异；Open3D法向量接口要求需要时
  显式按camera location定向。恢复只在face-sum和centroid fallback都为零时，用target sensor viewpoint给出确定性单位
  方向，最后仅为sensor恰与点重合保留固定轴退路；不删除tiny proposal、不改变volume/topology/patch/cohort/gate。
  r3及其完整诊断保持不可变，r4使用新run。防重复：不得把tiny components静默过滤来换取normal-valid=1，也不得使用
  随机法向量。证据=`docs/autoresearch/worldsim_v63/P3_SURFACE_CORPUS_PREREG.md`、
  `https://openaccess.thecvf.com/content/CVPR2022/papers/Sommer_Gradient-SDF_A_Semi-Implicit_Surface_Representation_for_3D_Reconstruction_CVPR_2022_paper.pdf`、
  `https://www.open3d.org/docs/release/python_api/open3d.geometry.PointCloud.html`。

- `V63-F06`（`engineering/protocol`, `resolved`）：P3 r4=`20260824T152300Z__surface-probe-s20260824-r4`虽以
  `minimum normal-valid=1.0`和8/8 negative contracts得到runner `passed=true`，但formal前对照P1冻结point encoder
  schema发现payload缺signed FREE/OCC distance、patch-local coordinate、method/target behind-hit与第四个temporal support，
  且`ray_hit_order`字段实际保存raw metric distance。该问题不会改变r4的geometry capability，却使其不足以喂给冻结
  SurfNCC，故不得把r4写成完整P3 pass。恢复使用SciPy exact Euclidean distance transform按0.2m sampling生成仅依赖
  method-visible evidence的signed distances；patch coordinate减冻结patch centroid；hit order在每个surface ray bundle内按
  distance+lexicographic tie-break归一化，并另存raw distance；同时显式补behind-hit、temporal UNKNOWN与actor observed-hit。
  r5=`20260824T152843Z__surface-probe-s20260824-r5`验证上述aggregate字段后，P4 loader审计继续发现aggregate counts无法
  执行冻结的整段temporal-window dropout；同一恢复因此再补每个method sweep的state/contradiction `[point,sweep]`
  矩阵，并用配置中的单一required-field清单防止再次静默漏项。VideoMAE与MaST-Pre支持结构化时间mask应保留时间维，
  但V6.3不迁移其高mask ratio/预训练目标。无target信息进入proposal/feature decision、无新超参或quality选择。
  防重复：capacity前必须逐字段对齐P1 schema；不能用字段名掩盖语义错位、用aggregate冒充per-sweep，或事后删掉冻结输入
  以让loader先跑。证据=
  `docs/autoresearch/worldsim_v63/P3_SURFACE_CORPUS_PREREG.md`、
  `https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.distance_transform_edt.html`、
  `https://openaccess.thecvf.com/content/ICCV2021/papers/Zhao_Point_Transformer_ICCV_2021_paper.pdf`、
  `https://proceedings.neurips.cc/paper_files/paper/2022/file/416f9cb3276121c42eebb86352a4354a-Paper-Conference.pdf`、
  `https://openaccess.thecvf.com/content/ICCV2023/papers/Shen_Masked_Spatio-Temporal_Structure_Prediction_for_Self-supervised_Learning_on_Point_Cloud_ICCV_2023_paper.pdf`。

- `V63-F07`（`engineering`, `resolved`）：r6 pre-run per-sweep窄检查首次凭scene名猜测processed path为
  `trainval/000`，在读取`instances_info.json`时触发`FileNotFoundError`；没有创建run或读取quality。冻结cohort是该映射的
  唯一事实源，实际`scene-0071 processed_index=68`；改用`trainval/068`后检查通过，state与contradiction均为
  `[3,300,300,40]`且逐voxel FREE+OCC+UNKNOWN count恒等于3。防重复：raw processed目录只按cohort metadata中的
  `processed_index`解析，不从scene display name猜目录；这一入口错误不扩展smoke/regression。证据=P3 r6 pre-run shell与
  `configs/worldsim_v62/p2_development_cohort_v1.yaml`。

- `V63-F08`（`engineering`, `resolved`）：P4尚未解锁执行时的temporary synthetic AMP interface r1在
  `binary_cross_entropy(sigmoid(hidden_free/authority))`触发PyTorch RuntimeError；128个随机点、无真实surface/quality、
  未创建正式run。PyTorch官方AMP文档明确说明BCELoss backward梯度可能无法用FP16表示，autocast因此主动拒绝，并要求
  使用`binary_cross_entropy_with_logits`。恢复同时输出hidden-FREE/authority logits供loss使用，推理概率仍为sigmoid；
  r2合成forward/backward finite，proposal-token gradient存在。防重复：FP16训练的二元head必须保留logits并在autocast
  下用BCE-with-logits，不通过禁用AMP或转FP32绕开冻结precision。证据=`docs/autoresearch/worldsim_v63/P4_CAPACITY_PREREG.md`、
  `https://docs.pytorch.org/docs/stable/amp.html#prefer-binary-cross-entropy-with-logits-over-binary-cross-entropy`。

- `V63-F09`（`engineering`, `resolved`）：未来P5的packed-proposal synthetic r1把NumPy API名迁到PyTorch，调用不存在的
  `torch.flatnonzero`而在token selection前失败；128随机点、无真实surface/quality、无正式P5 run。PyTorch官方提供
  `torch.nonzero(input, as_tuple=False)`返回二维索引；对一维mask用`.squeeze(1)`得到所需索引。替换后r2完成2个proposal、
  4个patch的FP16 forward/backward，proposal CVaR shape=`[2]`且Transformer/proposal-token gradient非零。防重复：
  NumPy的`flatnonzero`不得假设存在于torch namespace；CUDA mask索引统一使用官方`torch.nonzero`/`torch.where`接口。
  证据=temporary packed-interface terminal、`https://docs.pytorch.org/docs/stable/generated/torch.nonzero.html`。

- `V63-F10`（`engineering/protocol`, `resolved_preexecution`）：P4/P5正式执行前对首40个P3完成单元做method-only结构读取，
  发现每个单元恰有一个surface超过冻结的8192-point microbatch（`40/40`，最大`173488` points）；这些大surface的完整
  patch set平均`297.45`、最大`417` tokens。H-P4-001只取最大proposal的首chunk，既未覆盖全部点，也只让每个chunk
  独立生成proposal token；这只能证明局部point memory，不能证明P1冻结的complete-proposal interaction。没有创建P4 run、
  启动GPU或读取target quality。H-P4-001因此在执行前withdrawn；H-P4-002保持同一两个unit、模型宽度、两步AdamW、
  accumulation4、CVaR/gate/resource不变，改为先按8192点编码完整patch，再把当前proposal的全部patch token汇合后运行
  两层attention与唯一proposal token。Set Transformer与Perceiver支持用小型set/latent bottleneck承接大输入；本迁移
  不增加learned token、删点、改分辨率或改denominator。未切块路径的12项输出模块化等价审计max abs diff=`0.0`。
  防重复：capacity不得用首chunk冒充完整proposal；point microbatch只能切point graph，proposal identity与patch context
  必须在上层重组。证据=`docs/autoresearch/worldsim_v63/P4_CAPACITY_PREREG.md`、
  `https://proceedings.mlr.press/v97/lee19d.html`、`https://proceedings.mlr.press/v139/jaegle21a.html`。

- `V63-F11`（`engineering/protocol`, `resolved_preexecution`）：同一次P5执行前审计发现packed chunk曾各自计算actor/safe/
  unsafe标签、各自抽structural dropout，并在移除hard/temporal/actor-observed evidence后仍保留原`authority_bits`输入与
  原authority标签；selection还把各chunk hidden-FREE CVaR的最大值当完整proposal CVaR。这会让同一proposal跨chunk
  标签/selector漂移、从辅助authority通道看见已mask支持，并改变tail统计。没有真实P5 run、checkpoint或quality read。
  恢复把actor/safe/unsafe/full point count绑定完整proposal；每epoch/proposal只抽一个semantic selector并由所有chunk
  消费；遮蔽后从剩余method/temporal支持重算authority输入和监督，保留合法Actor current/swept与closure支持；selection
  汇合全部hidden-FREE点后精确计算alpha0.90 proposal CVaR，并统一为hard projection优先、仅learned low-authority OCC
  转UNKNOWN的最终decision后再统计coverage/UNKNOWN/accuracy。训练端明确保留内存受限的packed stochastic CVaR surrogate，
  不冒充exact full-batch optimizer。MAE支持“encoder不可见mask输入、原semantic target仍监督”；minibatch risk文献提示
  tail functional在小batch上可能有偏。防重复：任何evidence-derived辅助通道必须与mask同步；chunk-local统计不得冒充
  proposal统计。证据=`docs/autoresearch/worldsim_v63/P5_TRAIN_PREREG.md`、
  `https://openaccess.thecvf.com/content/CVPR2022/html/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022_paper.html`、
  `https://arxiv.org/abs/2301.11724`、
  `https://openaccess.thecvf.com/content/CVPR2025/papers/Kalble_EvOcc_Accurate_Semantic_Occupancy_for_Automated_Driving_Using_Evidence_Theory_CVPR_2025_paper.pdf`。

- `V63-F12`（`engineering/protocol`, `resolved_preexecution`）：完整proposal context恢复后继续逐loss审计，发现ranking仍只在
  当前packed chunk恰好共现的safe/unsafe proposals间配对；跨chunk的大proposal会多次占用同一safe对照，而不共现的
  nearest-size pair永远没有loss。这违反P1冻结的complete-proposal、同actor/static stratum、nearest full-point-count
  一对一匹配。没有真实P5 run、checkpoint或quality read。Cross-Batch Memory证明小batch pair mining可由历史embedding
  扩展，但其stale queue与额外memory state在本项目无必要：每个unit的完整patch token set本来就小。恢复先从完整unit
  metadata一次性生成一对一pair，再用当前权重的detached完整patch-token cache运行可微proposal attention/risk head，
  每unit只施加一次全proposal ranking；point losses仍按8192 chunks有界。margin=`0.10`、weight=`0.25`、labels、cohort、
  optimizer与denominator均不变，也没有新增queue/momentum/hyperparameter。防重复：proposal-level pair loss不得由chunk
  共现关系定义；batching只能限制point graph，不能改变匹配集合。聚焦语义审计把safe/unsafe proposals置于两个不同chunk，
  仍得到冻结unit pair=`[(0,1)]`。证据=
  `docs/autoresearch/worldsim_v63/P5_TRAIN_PREREG.md`、
  `https://openaccess.thecvf.com/content_CVPR_2020/html/Wang_Cross-Batch_Memory_for_Embedding_Learning_CVPR_2020_paper.html`。

- `V63-F13`（`engineering/protocol`, `resolved_preexecution`）：batch invariance审计发现6-neighbor edge曾按surface identity
  构建，但一个大surface按patch边界进入多个point chunks时，跨chunk patch边会被静默删除；同一对patch偶尔共处一个chunk
  时该边又存在，因此输出依赖packing而非冻结几何。没有真实P4/P5 run、GPU结果或quality read。GraphSAINT明确指出induced
  subgraph minibatch丢失外部边会产生sampling bias；Point-BERT则提供local patch先编码为token、再由Transformer组合的成熟
  分层结构。V6.3不引入随机subgraph sampling、归一化估计或halo超参，而是把两层deterministic 6-neighbor aggregation的
  local neighborhood明确绑定到已冻结的完整patch；patch最大2048且从不切分，所以边集合与8192 packing无关，跨patch交互
  由完整proposal patch attention承担。proposal surface/6-connectivity、patch membership、模型层数、point features、cohort
  与denominator不变。防重复：任何point microbatch必须保持local encoder的计算单元完整；不能让edge存在性取决于邻patch
  是否碰巧同batch。两完整patch的聚焦语义审计得到full/split有向边数=`4/4`。证据=
  `docs/autoresearch/worldsim_v63/P4_CAPACITY_PREREG.md`、
  `https://openreview.net/pdf?id=BJe8pkHFwS`、
  `https://openaccess.thecvf.com/content/CVPR2022/html/Yu_Point-BERT_Pre-Training_3D_Point_Cloud_Transformers_With_Masked_Point_Modeling_CVPR_2022_paper.html`。

- `V63-F14`（`engineering/protocol`, `resolved_preexecution`）：训练端已把matched ranking限定为完整scene/frame unit，
  但selection汇总曾把24个selection units的proposal rows一次交给全局nearest-size matcher，因此safe/unsafe可跨scene/frame
  配对；完整proposal risk虽正确，checkpoint objective仍会受跨案例规模巧合影响。没有真实P5 run、checkpoint或quality
  read。CVPR 2016 lifted structured embedding与CVPR 2022 graph sampling都说明pair mining的候选关系/采样边界是目标的一部分，
  不能把扩大候选池当作中性的batch实现。恢复仅按`(scene,target_frame)`分组执行原有actor/static、nearest full-point-count、
  one-to-one matcher，再对有pair的unit loss等权平均；margin=`0.10`、weight=`0.25`、proposal risk、cohort、threshold与gate
  均不变。聚焦synthetic把safe/unsafe分别置于两个unit得到`0 pair`。防重复：train/selection的proposal matching边界必须
  同为完整unit；不得跨case挖pair或引入memory queue。证据=`docs/autoresearch/worldsim_v63/P5_TRAIN_PREREG.md`、
  `https://openaccess.thecvf.com/content_cvpr_2016/html/Song_Deep_Metric_Learning_CVPR_2016_paper.html`、
  `https://openaccess.thecvf.com/content/CVPR2022/html/Liao_Graph_Sampling_Based_Deep_Metric_Learning_for_Generalizable_Person_Re-Identification_CVPR_2022_paper.html`。

- `V63-F15`（`engineering/evaluation`, `resolved_preexecution`）：P4已有`cvar_gradient_nonzero` gate曾在total loss backward后
  检查`hidden_free_head.weight.grad`，但该head同时受BCE-with-logits监督；即使proposal CVaR图断开，BCE也足以让flag
  为true，造成capacity假阳性。没有真实P4 run、quality read或scientific denominator。CVaR优化的一手工作明确把risk
  objective的梯度作为优化对象，PyTorch官方`autograd.grad`提供指定outputs到inputs的直接VJP。恢复在原forward图上对
  `proposal_cvar.mean()`分别向state/hidden-free/authority heads求梯度并只接受finite nonzero direct path；聚焦synthetic
  三条head路径均通过。原gate名称、阈值、units、steps、模型与资源合同不变，也未新增回归矩阵。防重复：多项loss共享
  parameter时，不得以总梯度证明某个特定loss已连通。证据=`docs/autoresearch/worldsim_v63/P4_CAPACITY_PREREG.md`、
  `https://docs.pytorch.org/docs/stable/generated/torch.autograd.grad.html`、
  `https://proceedings.mlr.press/v235/kim24x.html`。

- `V63-F16`（`data/evaluation`, `resolved`）：P3 formal终态前语义审计发现surface registry与summary字段
  `hidden_free_count/hidden_free_point_count`实际只计算`target_state==FREE`，没有同时要求
  `method_state==UNKNOWN && !method_contradiction`，所以该描述统计不能按hidden-FREE引用。point NPZ中的method/target/
  contradiction、proposal/patch/native features均正确；P4不消费此计数，P5 training/selection从point arrays使用正确布尔
  条件，因此不是标签污染或语料重建失败。NeurIPS dataset documentation实践强调保留旧版本并显式记录metadata限制。
  恢复边界：不改写formal artifact；terminal后从72个原始NPZ一次重算得到target FREE/OCC/UNKNOWN=
  `1545584/335050/9702367`、correct hidden-FREE=`688837`，已在三本账与P3 prereg登记勘误；未来materializer
  以additive v2把`target_free/occupied/unknown`与正确`hidden_free`分字段输出。
  canonical r6 probe一次重算得到target FREE/OCC/UNKNOWN=`19609/3891/128726`、correct hidden-FREE=`8311`，确认旧值
  `19609`只是target-FREE。
  禁止因描述字段误名重跑13.213小时正确point corpus，也禁止继续引用旧summary的hidden-FREE数字。证据=
  `docs/autoresearch/worldsim_v63/P3_SURFACE_CORPUS_PREREG.md`、
  `https://arxiv.org/abs/1803.09010`、
  `https://papers.neurips.cc/paper_files/paper/2024/file/605bbd006beee7e0589a51d6a50dcae1-Supplemental-Datasets_and_Benchmarks_Track.pdf`。

- `V63-F17`（`engineering/numerics`, `resolved`）：H-P4-002 r1 canonical=
  `20260825T045854Z__capacity-h002-s0-r1`在11.181s完成全部2 train/2 selection complete proposals，peak仅
  `0.196070 GiB`，finite loss、direct CVaR三head gradient、proposal-token gradient、hard violations=`0`、checkpoint
  reload与selection finite均成立；但unscale后的total gradients含nonfinite，且same-model/reloaded FP16 forward max abs
  difference均为`9.059906e-6`，所以冻结finite与exact-zero determinism gate诚实未过。没有quality conclusion、calibration、
  confirmation或test read，不是资源/算法失败。PyTorch官方AMP文档说明默认initial scale可能使FP16 gradient overflow；
  reproducibility文档说明CUDA SDPA不同backend/backward确定性不同，math backend配合deterministic algorithms可确定执行。
  唯一有界r2恢复保留FP16但固定GradScaler initial scale=`1024`，禁用flash/memory-efficient SDPA、只启用math SDPA并开启
  deterministic algorithms。模型、units、dropout、loss、optimizer LR/WD、2 steps、accum4、gate与22GiB ceiling均不变。
  r3=`20260825T051200Z__capacity-h002-s0-r3`在同一合同下以finite gradient与exact-zero repeat/reload正式通过。
  防重复：不得放宽exact-zero阈值、忽略nonfinite flag、增加steps或把r1写成quality negative。
  证据=`docs/autoresearch/worldsim_v63/P4_CAPACITY_PREREG.md`、
  `https://docs.pytorch.org/docs/stable/amp.html`、
  `https://docs.pytorch.org/docs/stable/notes/randomness.html`。

- `V63-F18`（`engineering/runtime`, `resolved`）：H-P4-002 r2 canonical=
  `20260825T050400Z__capacity-h002-s0-r2`在第一个CUDA math attention forward处被PyTorch deterministic-algorithm runtime
  拒绝：CUDA>=10.2的cuBLAS矩阵运算只有在进程启动前设置`CUBLAS_WORKSPACE_CONFIG=:4096:8`或`:16:8`后才允许确定执行。
  r2在任何optimizer step、capacity summary、quality/calibration/confirmation/test read之前终止，run leaf为空；因此它没有检验
  F17的AMP-scale或exact-zero恢复，不是第二个科研/数值尝试。NVIDIA cuBLAS官方结果可重复性说明`:4096:8`会固定workspace
  配置且增加约24 MiB，PyTorch deterministic文档对CUDA matmul给出同一前置条件。恢复把`:4096:8`同时绑定到launcher和
  runner的pre-torch-import环境，并由P4/P5配置显式记录；新增开销远低于22 GiB ceiling。r3继续F17的同一次有界恢复，
  model/data/FP16/AMP scale/SDPA backend/dropout/loss/optimizer/steps/accum/gates均不变。防重复：不得关闭determinism、放宽
  exact-zero门或把入口异常写成capacity/quality失败。r3实际peak=`0.256589 GiB`、wall=`11.863s`并正式passed，说明环境
  恢复闭合而无需新增资源或协议变化。证据=`docs/autoresearch/worldsim_v63/P4_CAPACITY_PREREG.md`、
  `https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility`、
  `https://docs.pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html`。

- `V63-F19`（`algorithm/evaluation`, `resolved_by_constrained_recovery_p6_unlocked`）：P5 canonical=
  `run://worldsim_v63/WS-V63-P5-SURFNCC-TRAIN-01/20260825T051530Z__surfncc-train-s0-r1`完成全部48个train与24个
  scene-disjoint selection units，`7 epochs/1792 steps`、finite training、peak=`0.403084 GiB`且累计hard violations=`0`；
  runner据此正确报告capacity/training `passed=true`。但冻结lexicographic objective选择的epoch 3仅是best
  training-objective checkpoint：safe-OCC retention=`0`、emitted-OCC coverage=`0.0371977<0.10`、source-valid UNKNOWN=
  `0.861807>0.60`。该checkpoint把有正向OCC支持的安全曲面与危险/缺证据曲面一起拒绝，不能作为SurfNCC candidate，
  不能用低false-safe或低tail掩盖零仿真效用。这是positive-authority collapse症状；现有证据尚不足以区分representation/
  supervision重叠、raw/post-projection decision composition或weighted-objective optimization collapse，故不得提前把根因写成
  ordinary underfit或任一优化结论。

  P5D H002 canonical=`run://worldsim_v63/WS-V63-P5D-AUTHORITY-COLLAPSE-DIAGNOSTIC-01/20260825T084844Z__authority-diagnostic-s0-r2`
  已把根因收敛：safe-OCC raw/projected/post-authority decision均为实际`[FREE,OCCUPIED,UNKNOWN]=[153,0,62301]`且
  authority veto=`0`，排除hard projection与decision composition；raw `P(OCC)`虽以AUC=`0.722684`保留弱排序，绝对
  mean仅`0.006459`，`q_AUTH` AUC也仅`0.578070`。weighted tail/retention gradient mean比=`5.531x`，direct-tail与
  state-head比分别=`1.715x/1.732x`，tail-retention cosine mean=`-0.411568`；retention loss mean=`0.968547`。
  因此primary root确认为weighted-objective optimization collapse，evidence-authority supervision弱对齐为次级机制，
  而不是solver或authority veto失败。

  P5R canonical=
  `run://worldsim_v63/WS-V63-P5R-CONSTRAINED-SURFNCC-TRAIN-01/20260825T091631Z__constrained-train-s0-r1`
  以同一SurfNCC representation、数据、hard projection、seed0与P5 epoch3 model-only warm-start运行proxy primal-dual；
  retention/emitted-OCC/non-UNKNOWN改为约束，旧weighted retention term置0。formal完成`10 epochs/2560 steps`、finite、
  hard violations=`0`，未读P6/calibration/H/T。best feasible epoch6的retention=`0.721226`、coverage=`0.114148`、
  non-UNKNOWN=`0.686101`，四项exact gate全过，tail+rank=`0.520541`，因此`candidate_promotable=true`并解锁P6。
  epoch 7–9连续三轮没有更优feasible candidate后按patience停止；尤其epoch 8/9虽tail更低但coverage/UNKNOWN失门，未覆盖
  epoch6。由此F19的positive-authority collapse已由约束优化闭合，而不是靠降低gate或回改solver闭合。

  防重复：不得增加epoch、换seed、加大模型、改变CVaR alpha、降低retention/coverage/UNKNOWN gate、提高
  `lambda_ret`，也不得回改已连续零违反的FREE/OCC projection、ray hard constraint、lifecycle或V6.2 solver。P5R不再追加
  recovery/sweep；合法下一步仅为冻结best candidate进入原P6 fresh matched AB。P6必须保留Native B2、surface encoder、
  CVaR与authority消融及原晋级门；P5R的candidate pass不能冒充P6/校准/confirmation/deployment结论。
  证据=`docs/autoresearch/worldsim_v63/P5_TRAIN_PREREG.md`、
  `docs/autoresearch/worldsim_v63/P5D_AUTHORITY_COLLAPSE_DIAGNOSTIC_PREREG.md`、
  `configs/worldsim_v63/p5d_authority_collapse_diagnostic_v1.yaml`、
  `scripts/run_worldsim_v63_p5d_authority_diagnostic.py`、
  `configs/worldsim_v63/p5r_constrained_surfncc_train_v1.yaml`、
  `scripts/run_worldsim_v63_p5r_constrained_train.py`、
  `https://proceedings.mlr.press/v97/geifman19a.html`、`https://proceedings.mlr.press/v98/cotter19a.html`、
  `https://proceedings.mlr.press/v97/cotter19b.html`。

- `V63-F20`（`engineering/runtime`, `resolved`）：H-P5D-001第一次formal入口在创建run leaf、读取P5
  checkpoint/train arrays或建立CUDA context前，将`shutil.disk_usage`直接调用于尚不存在的
  `/root/autodl-tmp/runs/worldsim_v63/WS-V63-P5D-AUTHORITY-COLLAPSE-DIAGNOSTIC-01`，触发`FileNotFoundError`。
  新task namespace按设计尚未由runner创建，所以canonical run=`null`，没有任何分布、梯度或科学结论。Python官方
  `shutil.disk_usage(path)`要求path指向已有filesystem位置；`Path.mkdir(parents=True)`才负责创建缺失父目录。
  H-P5D-002只在disk check前向上寻找最近已存在父目录并执行相同20 GiB资源检查，之后仍由formal runner创建唯一leaf；
  checkpoint、48-unit分布、4-unit gradient probe、模型/FP16、threshold/gate、零optimizer与P6/H/T locks全部不变。
  防重复：新task resource check不得假设namespace已存在，也不得为了通过检查预创建并冒充failed/canonical run；本恢复不
  增加smoke或质量读取。证据=`scripts/run_worldsim_v63_p5d_authority_diagnostic.py`、
  `https://docs.python.org/3/library/shutil.html#shutil.disk_usage`、
  `https://docs.python.org/3/library/pathlib.html#pathlib.Path.mkdir`。

- `V63-F21`（`evaluation/metadata`, `resolved`）：P5D H002 canonical的`DECISION_STAGE_COUNTS.json`把
  `class_order`描述文字写成`UNKNOWN/FREE/OCCUPIED`，但生成counts的`torch.bincount`直接以argmax class index为bin，冻结
  project constants实际为`FREE_INDEX=0/OCCUPIED_INDEX=1/UNKNOWN_INDEX=2`。因此三个counts数组本身、raw/projected/
  post-authority等值关系、authority veto=`0`、全部distribution/gradient/summary均正确；错误只在数组标签文字。正确
  safe-OCC counts为`FREE/OCCUPIED/UNKNOWN=153/0/62301`，不是旧文字顺序的解释。canonical artifact保持不可变，runner
  future label已改成`FREE/OCCUPIED/UNKNOWN`；不为13分钟正确诊断重跑，也不改写artifact。防重复：任何class-count数组
  必须从同模块index constants生成或明确按constants记录order，不凭tri-state自然语言习惯手写顺序。证据=
  `motion_proj/worldsim_v62/projection.py`、`scripts/run_worldsim_v63_p5d_authority_diagnostic.py`、
  `https://docs.pytorch.org/docs/stable/generated/torch.argmax.html`、
  `https://docs.pytorch.org/docs/stable/generated/torch.bincount.html`。

- `V63-F22`（`engineering/operations`, `resolved`）：P5R terminal文档收口的首次SSH备份命令在本地PowerShell双引号中
  使用远端`$b`，变量在发送前被本地展开，导致远端备份在复制前失败并在项目外创建`/docs`重复副本树；随后一次包含
  `$(realpath /docs)`的保护命令也先被本地PowerShell解释并在任何删除动作前失败。两次均未修改
  `/root/autodl-tmp/motion_proj`、canonical run、checkpoint或Git工作树。只读`find`确认`/docs`全部是可由原仓库恢复的
  重复副本后，以显式绝对目标删除该树；随后不用变量或命令替换，以显式
  `/tmp/worldsim_v63_pre_p5rclose_20260825T1440Z`成功备份七个文档。P6 prereg同步后的inline `python -c` YAML检查又因
  同一PowerShell→SSH引号层在读文件前`SyntaxError`；改为将只读Python源码经stdin传给远端解释器后验证通过，项目仍未变。
  防重复：从PowerShell发送SSH文件操作时，不在双引号命令中使用远端`$var`、`$(...)`或嵌套inline Python字符串；备份、
  清理目标使用已解析的显式绝对路径并拆成独立步骤，结构验证统一经stdin发送。

- `V63-F23`（`engineering/runtime`, `resolved_pre_quality_read`）：P6 B0/B1/B2首次formal入口直接执行
  `python scripts/run_worldsim_v63_p6_development_ab.py`，解释器按官方合同只把输入脚本所在`scripts/`目录置于module search
  path首位，因而在首个project import触发`ModuleNotFoundError: motion_proj`。失败发生在run leaf创建、P3/native数据、B2
  checkpoint与CUDA context之前，canonical run=`null`，没有P6 quality或科学结果。恢复不改源码/配置/denominator/gate，
  只从repo root改用`python -m scripts.run_worldsim_v63_p6_development_ab`，使当前目录进入module path；同解释器`--help`
  入口验证通过。防重复：repo-local runner若import project package或兄弟`scripts` module，formal launcher统一用`python -m`
  或已安装console entry point，不把direct-file import失败登记为算法reject，也不为此扩展smoke矩阵。证据=
  `https://docs.python.org/3/library/sys_path_init.html`、
  `https://packaging.python.org/en/latest/guides/creating-and-packaging-command-line-tools/`。

- `V63-F24`（`algorithm/evaluation`, `active route-closed`）：P6 B3 Surface-Mean虽在训练内冻结epoch1 feasible checkpoint
  （hard0、retention=`0.636863`、OCC coverage=`0.285326`、UNKNOWN=`0.550411`），但统一逐scene stage evaluator在两scene
  都不优于冻结Native B2。scene-0450 common surface hidden-FREE CVaR=`0.596685 vs 0.497850`，相对改善=
  `-19.852%`，accepted area ratio=`0.406270`且source-valid UNKNOWN=`0.651678>0.60`；scene-1089 tail=
  `0.655861 vs 0.465122`，改善=`-41.008%`，area ratio=`0.499323`。两scene hard0、retention、case、actor/static过门，
  说明失败不是hard solver回归或all-UNKNOWN，而是surface architecture在保留一定OCC后仍同时放大hidden-FREE tail并显著
  收缩相对Native B2的写入面积。supporting scenes=`0/2`，H-P6-001 rejected。

  主计划Stop2因此关闭surface architecture family：B4 Surface-Max、B5 Surface-CVaR和M0 authority均不执行，H-P6-002/
  H-P6-003关闭未读，P7没有frozen P6 M0输入而保持locked；legacy/calibration/confirmation/test均未读。不得用pooled
  retention、训练内candidate、较高accuracy或hard0掩盖逐scene tail/area失败，也不得换seed、加大模型、改CVaR alpha、
  降低area/UNKNOWN/2%门或先读legacy/H/T复开。未来合法复开必须在新版本预注册feature-level aleatoric/epistemic
  uncertainty与scene/stratum-conditional coverage约束，并使用fresh development denominator；相关候选仅为EvOcc
  （CVPR 2025）、ReliOcc（IJCAI 2025）、OCCUQ（ICRA 2025开源）及UAI 2024 conditional robust optimization，不构成
  V6.3 recovery授权。证据=`docs/autoresearch/worldsim_v63/P6_SURFACE_FAMILY_CLOSEOUT.md`、
  `run://worldsim_v63/WS-V63-P6-DEVELOPMENT-AB-01/20260826T014500Z__b3-eval-s0-r1`、
  `https://openaccess.thecvf.com/content/CVPR2025/html/Kalble_EvOcc_Accurate_Semantic_Occupancy_for_Automated_Driving_Using_Evidence_Theory_CVPR_2025_paper.html`、
  `https://www.ijcai.org/proceedings/2025/220`、`https://github.com/ika-rwth-aachen/OCCUQ`、
  `https://proceedings.mlr.press/v244/chenreddy24a.html`。

<a id="detail-v62"></a>

## V6.2 CPSC 防重复结论（2026-08-24）

- `V62-F01`（`algorithm/evaluation`, `active`）：V6.1 oracle Occupancy 在 legacy28 得到 `10/28 ACCEPT` 且
  `0 false-safe`，说明物理状态补全存在真实上界；GaussianWorld 和 IR-WM 的 learned argmax Occupancy 都得到同一
  `10/28` 表面支持，但各自接受项全部为 `10/10 false-safe`。已确认的共同根因是 dense learned prior 能覆盖 proposal，
  却没有把真实 observed FREE 当作不可违反的前向约束；这推翻“感知型 argmax Occupancy 可直接成为 world compiler
  物理权威”，不推翻 learned prior 作为软信息源。防重复：不得以第三 backend、confidence threshold、entropy、grid、
  history window、checkpoint、verifier 放宽或 observed-FREE 事后全 veto 复开 V6.1。合法复开仅限 V6.2 CPSC：方法输入内
  hard FREE/OCC、contradiction→UNKNOWN、可推翻 prior、anti-trivial coverage 和独立 false-safe 评测；若 B1 hard clip
  已达 `>=5/28, 0 false-safe`，应诚实转为 projection-only compiler。证据=`V61-F11,V61-F13`、
  `docs/autoresearch/worldsim_v61/V61_MINIMUM_EXPERIMENT_CLOSEOUT.md`、
  `docs/WORLDSIM_V6_2_CONSTRAINT_AWARE_PHYSICAL_STATE_COMPLETION_PLAN.md`。
- `V62-F02`（`data/protocol`, `resolved`）：P2 r1 query probe 的 `method/target_state` 沿用 V6.1 evidence 编码
  `UNKNOWN/FREE/OCCUPIED=0/1/2`，而 P3 model distribution 固定 `FREE/OCCUPIED/UNKNOWN=0/1/2`；字段名未显式区分，
  若直接训练会静默互换 UNKNOWN/FREE/OCC 标签。r1 只做 CPU 资源/池探针，未训练、未产出科学结果或 formal dataset。
  r2 在任何 formal materialization 前把字段拆成 `*_evidence_state` 与 remapped `*_class_index`，并确认两者范围0..2；
  canonical probe=`20260824T082318Z__query-probe-s20260824-r2`。防重复：loader 只能把 `target_class_index` 送入
  three-state loss，把 `*_evidence_state` 作为 hard-evidence feature；禁止依赖裸整数碰巧相同或在 loader 中无名 remap。

- `V62-F03`（`data/algorithm`, `resolved`）：P2 formal r1 在 `scene-1012/f152` 因 instantaneous
  `actor_envelope` pool=`0` 停止；该帧并非没有 actor，而是4个当前 actor 全在冻结 ROI 外，其中一个 actor 在可见
  method sweep `f146` 仍穿过 ROI。只按 target-frame box 构造 actor pool 推翻了“每个冻结 target 当前 ROI 都含 actor”
  的隐含假设，也会诱使实现删除固定的15k actor query。参考 QueryOcc 的相邻时刻独立4D查询以及动态稀疏 query 的
  时序传播，恢复方案把 actor query support 固定为 `current target envelope ∪ visible method-sweep envelopes`；时序包络
  只定义 query support，不升级为 hard OCC evidence、不读取 dropout/target evidence，也不挪用 actor quota。定点复现
  `20260824T083403Z__actor-sweep-repro-s20260824-r5`：current=`0`、visible swept=`450` voxels、actor-type query=
  `15000/15000`、total=`100000`、exit=`0`。防重复：不得因某个 target 当前 ROI 无 actor 而删 unit、删 actor query、
  改 ROI 或把 target evidence 当 method input；若 visible method sweep 也无 actor support，必须登记新的 cohort-level
  事实并重新审视 actor-query定义，不能静默转采 EASY FREE。formal r1=
  `20260824T082601Z__query-dataset-s20260824-r1`，未完成 manifest、未用于训练或质量结论。恢复后的 formal r2=
  `20260824T083654Z__query-dataset-s20260824-r2` 已完成72/72 units、7.2M queries，combined actor pool 0空、
  source-role overlap=0；该成功不新增 failure ID。

- `V62-F04`（`engineering`, `resolved`）：P4 probe r1 在 official IR-WM plugin import 阶段、GPU forward 与任何
  sidecar 写入前失败；隔离 Python 可执行文件虽来自 `worldsim-v61-irwm`，controller 却继承外层 shell PATH，导致
  PyTorch `cpp_extension.load()` 的 `verify_ninja_availability()` 找不到 env 内已安装的 `bin/ninja`。这不是缺依赖、
  CUDA 不兼容、IR-WM 方法失败或数据失败。PyTorch 官方实现明确通过 PATH 调用 `ninja --version`，V6.1 已成功的
  IR-WM controller 也显式 prepend env `bin` 并冻结 `TORCH_CUDA_ARCH_LIST=8.6`。恢复仅复用同一环境合同：prepend
  env bin，设置 `PYTHONNOUSERSITE=1`、OMP/MKL threads、CUDA device与SM 8.6；不安装包、不改模型/输入/query或门槛。
  failed probe=`20260824T085711Z__prior-sidecar-probe-s1-r1`，无科学输出；防重复：用隔离 Python 启动 native/CUDA
  worker时不得假设其 bin 自动进入 PATH，也不得把 loader import failure记成方法 rejection。恢复后的同输入 r2=
  `20260824T085956Z__prior-sidecar-probe-s1-r2` 已产生100k query-aligned sidecar，peak=`4.05GiB`、target evidence
  read=`false`；因此 F04 保持 resolved，不新增 recovery。

- `V62-F05`（`data/protocol`, `resolved for artifact-bounded P6`）：P6 接口审计推翻“V6.1 已冻结P5可直接消费的
  prior sidecar”。canonical ME3R 的四个IR-WM输出仅有argmax `class_label[200,200,16]`、occupied mask和网格/pose，
  没有17 logits或256D BEV；禁止重跑IR-WM，故逐cell uncertainty/latent不可精确恢复。阶段表还有第二处冲突：B2需要
  Tier-C校准阈值、B4需要未训练的no-evidence-dropout checkpoint、full M0需要P8 grouped conformal，三者在P6均不存在。
  防重复：不得从硬label伪造逐cell置信度、用legacy O_eval拟合adapter/threshold、重跑backbone、补训多臂或把B5冒充
  conformal M0。参考CVPR 2022 ProtoSeg的非参数训练特征均值，恢复只用P5 train split按17 class求query-weighted
  logits/BEV prototype并查表，P5保持frozen；24-unit只读失真审计agreement=`0.896898`、bridge hidden-FREE=
  `0.399349`、safe-OCC=`0.872897`、hard violation=`0`。合法P6只比较B0/B1/B3/B5，明确B2/B4 unavailable、M0 defer
  P8；bridge claim始终是lossy artifact transfer，不是native sidecar parity。证据=`P6_LEGACY_INTERFACE.md`、
  `configs/worldsim_v62/p6_legacy28_v1.yaml`、`motion_proj/worldsim_v62/legacy_bridge.py`。

- `V62-F06`（`algorithm/evaluation`, `active; recovery exhausted`）：P6 canonical=`20260824T095529Z__legacy28-s0-r1` 在同一28-case上
  B5仅`4/28 ACCEPT`且`4/4 false-safe`，mask-area=`0.09402`、R10=`2/3`、Actor新增=`0`；source-valid UNKNOWN=
  `0.82735` 超过0.50。B3与B5 case decision完全相同；B5 hard projection仍是`0/939206`违规，oracle accepted surface
  safe-OCC retention=`1.0`。B1虽把accepted FREE conflict从B0 mean/worst=`0.26748/0.57057`降到
  `0.05058/0.11722`，但仍`10/10 false-safe`，所以简单hard clip既不安全也未触发Stop 1。根因边界：argmax-only
  prototype input造成严重missing-feature shift，evidential head高UNKNOWN仍保留4个hidden-unsafe route surface；局部
  projection只能保证观测cell，不能恢复丢失特征或证明隐藏表面。禁止用本次O_eval调threshold/prototype、改grid/window、
  重跑IR-WM、删case、放松UNKNOWN/FREE gate或改选第二backend。唯一合法复开=`P6R evidence dropout`：依据CVPR 2022
  Modality-Agnostic Learning，只用P2/P4 train模拟`p=0.5` prototype feature loss，并由frozen full-view P5 teacher做
  `0.25 KL`一致性；相同P6 gate一次性复评。若P6R失败，CPSC-Lite关闭，不再换projection/set-valued recovery。
  P6R formal r2=`20260824T101705Z__feature-dropout-train-s0-r2` 已按冻结复合目标选best epoch2；objective改善但
  prototype hidden-FREE false-OCC为`0.41441`，尚未解除本条。只有未改门槛的legacy28 recovery可以裁决F06，训练
  selection不能替代false-safe结果。
  唯一P6R legacy recovery=`20260824T102709Z__feature-dropout-legacy28-s0-r1` 仍为`4/28 ACCEPT,4/4 false-safe`，
  接受集合完全相同；UNKNOWN虽从`0.827351`降到`0.638518`，仍超过0.50，mask-area=`0.094024`、R10=`2/3`、Actor
  gain=`0`、worst FREE conflict=`0.087379`。因此missing-feature exposure缓解abstention但没有建立hidden-surface
  authority，本条从“允许唯一recovery”更新为“recovery exhausted / family closed”。后续不得选择projection architecture
  或set-valued head作为第二recovery，也不得绕行P7/P8 calibration。未来新版本复开至少要求native per-voxel logits/features、
  独立calibration cohort与直接hidden-surface false-safe risk supervision，并在任何legacy评分前重新scope-freeze。

- `V62-F07`（`engineering`, `resolved`）：P6R首次formal入口
  `20260824T101047Z__feature-dropout-train-s0-r1` 从source=`d8f69d0`创建run后，在pure-prototype baseline selection
  的首个`compute_cpsc_losses`调用触发`KeyError: prior_tristate`；optimizer steps=`0`、checkpoint=`0`、legacy O_eval
  read=`0`，因此不是训练不稳定或机制rejection。根因是recovery runner自定义mapping batch漏传原P5 prior-preserve
  loss需要的字段。PyTorch官方`torch.utils.data`说明mapping sample/batch由collation保留键，调用方必须完整提供约定字段；
  恢复前已核对loss的全部batch访问，确认没有第二个遗漏键。修复不能把full-view先验静态塞入所有路径：selection传
  `bridge_prior[:,18:21]`，训练传逐query混合后的`corrupt_prior[:,18:21]`，使loss与student实际证据视图一致。
  failed r1保持不可变；revision 2只改batch合同与run revision，不改模型、数据、p/KL、loss权重、seed、资源或legacy
  gate，不增probe/smoke/回归矩阵。证据：PyTorch DataLoader官方文档
  `https://docs.pytorch.org/docs/stable/data.html`、P6R terminal/config/runner。

<a id="detail-v61"></a>

## V6.1 Occupancy-verified world compiler 防重复结论（2026-08-22）

- `V61-F01`（`engineering/protocol`, `resolved`）：`WS-V61-H-P0-001` 首次正式启动在创建 run
  directory、读取 R9/R10/raw evidence、GPU、训练或生成器之前，对尚不存在的
  `/root/autodl-tmp/runs/worldsim_v61` 调用 `shutil.disk_usage`，触发 `FileNotFoundError`。没有 canonical run，
  也没有方法结果；不得把它记成 Occupancy/SceneIR-O rejection。`WS-V61-H-P0-002` 只在资源审计前以
  `mkdir(parents=True, exist_ok=True)` 创建精确 namespace，并增加缺失目录单测；R9/R10 hashes、28-case、scene mapping、
  truth tiers、threshold/stop rules、资源门与 confirmation lock 全部不变。H002 从干净提交 `6247fd8` 运行并使全部
  P0 gate PASS；canonical=`run://worldsim_v61/WS-V61-P0-SCOPE-FREEZE-01/20260822T100812Z__scope-freeze-s20260822-r1`，
  gate SHA=`fb2a416a...ae40`。仍然成立的边界：任何新路线 runner 都必须先创建自己的精确 namespace，禁止把父目录的
  可用空间当成子 namespace 已存在。

<a id="detail-v6"></a>

## V6 可验证世界编译器新增防重复结论（2026-08-21）

- `V6-F01`（`governance/research-direction`, `active`）：V5.2.1 的 badcase census、人工归因、Base Validity、
  immutable run、exact-once、UNKNOWN/abstention 与 failure-ledger 纪律继续有效，但它们没有建立 TrackBayes/M3 的
  causal bridge，也未解决偏离 logged trajectory 后的大场景扩展、生成内容可信固化和闭环复用。Stage H/BKI 从未执行，
  不得写成算法 reject；V5.2 M123 autoresearch 主线状态为 `superseded_by_v6_direction_reset`，M1/M2/M3 只作为
  SceneIR provenance、factorized validity 和 dynamics verifier 的子系统证据。V6 的合法复开边界是跨 frontend 的
  SceneIR/support/provenance/verify/bake/deterministic-runtime 问题，禁止把 V6 再退化为 StreetGS repair、TrackBayes-only、
  KNN/Graph/BKI 或 cut-in mining 主线。证据=`WS-V6-G0-REPO-CONVERGENCE-01`、
  `docs/WORLDSIM_V6_VERIFIABLE_WORLD_COMPILER_AUTORESEARCH_PLAN.md`、
  `docs/autoresearch/worldsim_v6/governance/REPO_PREFLIGHT.json`；方法质量 run=`0`。
- `V6-F02`（`engineering/protocol`, `resolved`）：G2 首次裸 `pytest -q` 在 collection 阶段出现 `12` 个
  `motion_proj/scripts` import error，显式 `PYTHONPATH=$PWD` 后才执行测试；随后 motionproj interpreter 下的 4 个
  V5.1 frozen-runtime tests 因预期 `/root/autodl-tmp/envs/drivestudio/bin/python / torch 2.1.2+cu118` 而失败，使用合同
  指定解释器后对应四文件 `15 passed`。根因是 integration runner 把 repo import root 与历史 runtime profile 当成单一
  环境默认值，不是 merge、算法或冻结结果漂移。以后全量 gate 必须显式注入 repo root，并按 config runtime 分组；不得
  删除 exact-runtime tests 或放宽版本字段。证据=`WS-V6-G2-BRANCH-CONVERGENCE-01`、最终 motionproj
  `1443 passed / 1 skipped` + DriveStudio `15 passed`。
- `V6-F03`（`engineering/asset-integrity`, `resolved with retained asset boundary`）：G2 frozen-asset regression 发现
  Instant NuRec official checkout、P2 selected checkpoint 和 P3 的 `158` 个 chunk payload 缺失，但 manifest、source
  checkpoint 与冻结 hash 仍在；这会造成 5 个 asset-dependent tests 失败，不能倒写历史方法 reject。Instant NuRec 按 exact
  public commit/tree 恢复；P2 从 immutable source 确定性重建并命中 `432,111,754 bytes / 7be87e8b...7448`；P3 重建
  payload 对旧 manifest `158/158` bytes/hash exact 后只补缺失文件，未覆盖旧 manifest。P2 recovery r1 因传相对 protocol
  path 在 snapshot 前 blocked，r2 改为绝对路径后成功；旧 r1 保留。以后清理 canonical selected asset 必须同步保留可执行
  materializer 与 exact source，恢复只能新 run→逐 hash 比较→补缺失字节，禁止生成近似资产或改旧 manifest。证据=
  `20260821T073335Z__g2-p2-asset-recovery-s0-r1`、`20260821T073353Z__g2-p2-asset-recovery-s0-r2`、
  `20260821T073459Z__g2-p3-asset-recovery-s0-r1`、R0/P3 `23 passed`。
- `V6-F04`（`protocol/governance`, `resolved`）：V5.1 `_validate_normative_plan_binding` 只允许 P0 base hash 与
  `b359541` Stage-B append hash，但当前 canonical 文档已在 `a9dede0` 冻结 terminal closeout，SHA-256=
  `a0e764f3...fe1d`；因此 4 个 protocol tests 在到达各自语义断言前统一被旧 allowlist 拦截。修复不改历史 config、plan
  或 gate，只把该 exact terminal commit/hash 加入 fail-closed allowlist，并把回归期望指向 terminal hash；任何未知第四种
  状态仍拒绝。禁止用“任意后继 commit”或跳过 hash 来救测试。证据=`WS-V6-G2-BRANCH-CONVERGENCE-01`、
  `tests/test_worldsim_v51_protocol.py=9 passed`。
- `V6-F05`（`engineering/provenance`, `resolved with noncanonical run retained`）：R1 首个实例
  `20260821T081500Z__r1-capability-s0-r1` 在 capability runner/config/tests 仍未提交时执行，summary 如实记录
  `source_dirty=true`；手工指定的目录时间标签还晚于真实完成时间。其能力事实虽通过，但不能作为 canonical closeout，旧目录
  保留且不覆盖。修复是在 runner 创建 run 前读取 `git status --porcelain`，dirty 即 fail-closed；先提交
  `d981df7fdde5458eb3878193c4a76f6dcf926ad4`，再由 runner 自动生成真实 UTC 标签的新实例
  `20260821T080610Z__r1-capability-s0-r1`，其 `source_dirty=false`、gate PASS。以后不得用“内容看起来正确”绕过
  source cleanliness 或手工修正旧 terminal；工程实例与 canonical evidence 必须分开登记。

<a id="detail-v52"></a>

## V5.2 人工归因与 M123 causal bridge 防重复结论（2026-08-20）

- `V52-F01`（`evaluation/attribution`, `active`）：V5.2.1 的 `GLOBAL_RGB / ACTOR_RGB / BOUNDARY` failure label 是合法
  census 结果，但不能自动解释为 M1/M2/M3 的模块失败。用户指定评审者对代表性 18-case package 完成 `18/18` 逐图复核后，
  冻结 `9 BASE_FAILURE + 8 M123_ELIGIBLE + 1 ATTRIBUTION_UNRESOLVED`：AD-GS 的多条 actor/boundary case 实际由白屏、
  单色、全局 smear 主导；即使 ownership 完美也无法恢复这些画面。不得用 BASE_FAILURE case 评价 TrackBayes、M3 delta 或
  M2 router，也不得删除这些 case 来改善基座 aggregate。所有后续 M123 run 必须先执行 P0 Base Validity Gate，并在完整报告中
  单独保留 base sentinels。证据=`WS-V521-P11-HUMAN-ATTRIBUTION-01`、canonical run=
  `/root/autodl-tmp/runs/worldsim_v521/20260820T130000Z__p11-human-review-attribution-s0-r001`、cases SHA=
  `d89f4a4b...381f`。
- `V52-F02`（`evaluation/causal-identification`, `active`）：8 个 StreetGS eligible case 的视觉症状与 M1 observation scarcity、
  M3 actor trajectory/visibility 高度相容，但 panel 不能证明失败 pixel 恰好来自 low-observation/uncertain Gaussian，也不能证明
  ghost 会被 actor-pose warp 解释。因此状态只能是 `DIRECTION_SUPPORTED_CAUSAL_BRIDGE_PENDING` /
  `SYMPTOM_OVERLAP_STRONG_EXACT_TEMPORAL_BRIDGE_PENDING`，不得从人工诊断直接晋级 TrackBayes 或修改 M3。合法复开必须保持
  Discovery design `5`（#05/#10/#11/#16/#17）与 one-shot Confirmation `3`（#06/#12/#18）分离，先冻结并执行 exact
  pixel→Gaussian/U2-B3 observability bridge 与 `unwarped/flow-warped/pose-warped` temporal bridge；Confirmation 不得选 arm、
  threshold 或 metric。M2 只消费已通过 candidate 的 uncertainty/validity 并决定 execute/abstain；geometry undefined 时不得
  写 geometry-safe。证据=`docs/run_manifests/worldsim-v5.2.1-human-review-attribution-v1/` 与
  `configs/worldsim_v52/m123_autoresearch_v1.yaml`。

<a id="detail-v51"></a>

## V5.1 M1-only 新增防重复结论（2026-08-17）

- `V51-F01`（`engineering`, `resolved`）：首轮执行
  `pytest -q tests/test_worldsim_v51_protocol.py tests/test_audit_worldsim_v51_start.py` 在 collection 阶段因测试文件直接
  import `motion_proj.worldsim_v51`、但未显式把仓库根加入 `sys.path` 而报 `ModuleNotFoundError: motion_proj`；同轮
  `python scripts/audit_worldsim_v51_start.py --help` 已通过，所以该 terminal 推翻的是“pytest 启动环境总会自动注入
  repo root”的工程假设，不是 P0 协议或算法失败。修复在测试 import 前按绝对 `ROOT` 注入路径并以原命令回归；失败时
  没有运行方法、读取 validation/test/KITTI quality 或产生质量数字。后续直接脚本与测试入口都必须有独立 import smoke，
  禁止把 collection error 计入方法分母。证据：`WS-V51-P0-M1-SCOPE-FREEZE-01`、
  `tests/test_worldsim_v51_protocol.py`、`tests/test_audit_worldsim_v51_start.py`。
- `V51-F02`（`engineering/evaluation`, `resolved`）：A0 runner 的 metric 单测把输入 `float32(0.1)` 产生的
  `0.10000000149011612` 与 Python 十进制 `0.1` 做严格相等，导致 `1 failed / 6 passed`；这推翻的是“人工十进制常数
  可以作为 bit-exact 浮点 oracle”的测试假设，不是 frozen metric 定义或 A0 canonical replay 失败。修复只把人工常数
  断言改为 `pytest.approx`；正式 A0 仍把同一实现重算值与 canonical JSON float 做 `delta == 0.0`，posterior/statistics
  仍逐 bit 比较。禁止为了让 exact gate 通过而对 canonical metric 使用容差、舍入或字符串截断。失败时未启动正式 run、
  GPU renderer、方法推理或 validation/test/KITTI quality read。证据：`WS-V51-M1-A-UNARY-OBSERVABILITY-01`、
  `tests/test_replay_worldsim_v51_v5_unary.py`。
- `V51-F03`（`engineering/evaluation`, `resolved`）：A1 规定 `visibility >= 0.01` 为 inclusive gate，但冻结 NPZ 中
  visibility 是 `float32`；若直接与 Python double `0.01` 比较，存储的 `float32(0.01)` 会因表示略小而被误判为
  false，首轮测试得到 `[False,False,False]` 而非 `[True,False,False]`。这会真实改变 observation denominator，不能靠
  放宽单测解决。修复是在比较前把配置阈值量化为 observation dtype，同时在诊断中分别记录 configured/applied value；
  门仍是 inclusive，未读取 evaluation quality 或搜索阈值。禁止用 epsilon、容差或事后改 threshold 隐式改变分母。
  证据：`WS-V51-M1-A-UNARY-OBSERVABILITY-01`、`motion_proj/worldsim_v51/evidence/visibility.py`、
  `tests/test_worldsim_v51_visibility.py`。
- `V51-F04`（`algorithm/protocol`, `resolved before A2 quality read`）：A2 首轮只读 evidence-statistics 检查发现，
  若直接在“全部 Gaussian”上取 effective-count 下分位数、entropy/disagreement 上分位数并用 OR 组成 UNKNOWN，
  scene-0379/1087 的 count 与 disagreement 分位数会同时退化为 `0`；inclusive `disagreement >= 0` 会把全部 Gaussian
  判为 UNKNOWN，Gaussian coverage 直接变成 `0`。根因是未观测但由冻结 base-model prior 明确赋类的 Gaussian 占
  `67.39%/97.20%`，它们不能和真正有 semantic observation 的校准总体混在一起。A2 在任何 evaluation artifact 或
  quality metric 读取前，把阈值总体冻结为“三个 H scene 中 effective-count>0 的 A1 Gaussian pooled population”，并用
  `high entropy AND (low count OR high disagreement)` 保留 entropy 作为必要条件；三个阈值固定为该总体的
  `Q25(count)=0.19274792820215225`、`Q75(entropy)=0.005402358970383594`、
  `Q75(disagreement)=8.494543610182426e-12`。禁止把全量分位数退化误写成 A2 方法负结果，也禁止在看到 A2 evaluation
  quality 后改总体、分位点、布尔规则或图像 abstain threshold。证据：
  `configs/worldsim_v51/m1_unary_unknown_v1.yaml` 与 A1 posterior SHA binding；S/validation/test/KITTI 仍未读取。
- `V51-F05`（`algorithm/protocol`, `rejected by A3 r005`）：计划 A3-1 提议以
  `n_eff=(sum r)^2/(sum r^2+epsilon)` 限制 A3-0 的 fractional concentration `sum r`，并解释为
  correlation-aware。但 A1 reliability 逐 observation 严格在 `[0,1]`，因此忽略仅用于数值稳定的 epsilon 时恒有
  `sum r^2 <= sum r`，进而 `n_eff >= sum r`：作为上限必然是 no-op，直接替换则会提高而非降低 posterior
  concentration。更根本的是该式只见单 observation 权重及其平方和，没有任何 view-pair correlation observable；
  调换 view 顺序或相关结构而保持权重集合时结果不变，不能支持“10 个高度相关 view 不等于 10 个独立证据”的命题。
  r005 用 v2 evidence-only audit 对 `944,443` 个 positive-count Gaussian 复现：无 epsilon 时
  `n_eff<sum(r)` 为 `0`，absolute cap change>`1e-9` 为 `0`；若直接 replacement，`940,762/944,443=99.6102%`
  Gaussian concentration 被放大。结论=`a3_kish_cap_rejected_structural_noop_not_correlation_aware`；A3 不启动 GPU
  quality arm，按原计划只解锁独立 A4。禁止为挽救 A3 事后加入相关系数/时间核/feature similarity；那将是新机制，
  不是原 A3-1 的修复。r005 未读 evaluation artifact/quality、validation/test/KITTI，failure delta=`V51-F05/F06`。
- `V51-F06`（`engineering/evaluation`, `resolved without quality read`）：A3 audit v1/r004 用相对 cap change 判断
  epsilon 是否产生“有意义修正”，但 0471/0379 存在 reliability=`1.401298464324817e-45` 的 float32 最小次正规数；
  `epsilon=1e-12` 使这些近零 mass 的相对变化达到 `1.0`，尽管三个场景最大绝对 cap reduction 仅约
  `2.5e-13`。r004 因此合法保留为 `done/inconclusive`，不是 A3 得到有效 concentration reduction，也不是方法质量
  失败；它没有读取 evaluation artifact/quality、启动 GPU renderer 或改变 posterior。v2 新配置绑定 r004 与 v1 hash，
  在新结果前把 meaningful gate 改为 absolute cap change>`1e-9`，同时继续报告相对量作诊断。禁止用近零分母的巨大
  相对数宣称机制有效，也禁止覆盖 r004 terminal；只能用新 r005 重放同一 45 份 evidence observation。
- `V51-F07`（`algorithm/novelty`, `rejected by A4 r006`）：CIF 原论文将 occupancy probability 与 conditional
  instance distribution 分开，并明确针对 appearance opacity 与 occupancy 混淆；其完整方法还包含 learned deformable
  Gaussian instance field、identity calibration 与 semantic resampling。V5.1 计划明确不引入后三类机制，而当前 renderer
  已把 appearance `base_opacity` 与 conditional ownership sidecar 相乘，A1 已分离 visibility eligibility，A2 已分离
  UNKNOWN。因而 A4 若把 `base_opacity` 当 occupancy，会在 renderer 中二次乘 alpha 且违反参考机制；若用
  visibility/effective-count，会再次把不可见误作不存在；若对已实例化 Gaussian 设 occupancy=1，则与现有 renderer
  bit-exact no-op。r006 绑定三个 A2 posterior 与 renderer/visibility/abstention 源码，确认 occupancy field=`0/3`、
  constant-one 对现有 renderer=`3/3 bit exact`，而复用 appearance opacity=`3/3 non-exact` 且会二次缩放。结论=
  `a4_cif_decoupling_rejected_no_independent_occupancy_observable`；未读 evaluation artifact/quality、未启动 GPU/training。
  禁止把已有 A1/A2 分解重新命名为 CIF 增益，或在结果后偷偷解锁完整 CIF 训练、校准/重采样。参考：
  [CVPR 2026 official paper](https://openaccess.thecvf.com/content/CVPR2026/html/Wu_Consistent_Instance_Field_for_Dynamic_Scene_Understanding_CVPR_2026_paper.html)。
- `V51-F08`（`engineering/resource`, `resolved without duplicate run`）：首次尝试并行启动 S unary materialization 时，
  PowerShell→SSH→远端 Bash 的多层转义把 `$!/$?` 保留成字面量，导致外层 `wait/test` 解析失败。只读 PID、GPU、run 目录
  与 status 审计确认 scene-0998 只有一个正式进程，scene-0359 根本没有启动；因此保留 r049 单实例自然完成，再以独立前台
  命令串行执行 r050。两者均为 `done`、checkpoint 前后 SHA exact，且无重复 scene/candidate 分母。3090 实测后段显存保留
  分别达到约 `22 GiB/20 GiB`，也推翻了“两个 unary materialization 可安全共卡并发”的资源假设。以后 Windows 发起的远端
  后台编排不得内联依赖 `$!/$?`；优先每个长 run 使用独立前台 SSH session，确需后台时使用远端脚本/控制器并单独审计
  PID、日志与 terminal。wrapper 失败不得写成方法失败，也不得因外层 exit code 重跑已封口的 immutable run。证据：
  r049/r050、source commit `6950597`、`configs/worldsim_v51/stage_a_screening_v1.yaml`。
- `V51-F09`（`algorithm/evaluation`, `rejected by Stage A r007`）：A1 visibility 在 H 上通过的 scene-balanced
  `ΔBoundary-F1=+0.001155713` 没有在预注册 S=`0998/0359` 上复现。S 两场 delta 分别为
  `-0.0000904944/+0.0000574359`，只有 `1/2` 非负、`0/2` 达到 clearly-positive `+0.001`，均值为
  `-0.0000165293`；尽管 mean IoU/Brier/ECE 略改善且 FN 增量仍在门内，冻结 gate 是合取，A1 必须 rejected。
  这推翻“hard visibility eligibility 的 H 小效应可跨开发场景稳定复制”，不是 Bayesian U2/B3 基线失败。禁止根据 IoU
  或 calibration 的微小正向分量保留 A1、删除 0998、放宽 clearly-positive 门或在同一 S 上重选 visibility threshold。
  合法复开需要新机制、新任务和未读场景；V5.1 当前冻结 U2/B3，不再继续复杂化 Bayesian family。证据：r007、
  `configs/worldsim_v51/stage_a_closeout_v1.yaml`。
- `V51-F10`（`algorithm/evaluation`, `rejected by Stage A r007`）：A2 UNKNOWN 在 S 上仍能集中错误：scene-balanced
  accepted/abstained error=`0.0148416/0.134393`，两场均有非空 denominator；但 coverage 在 0998/0359 为
  `0.250105/0.864765`，scene-balanced mean=`0.557435<0.60`，未通过冻结 selective gate。0998 的 UNKNOWN Gaussian
  ratio=`34.5512%`，0359 仅=`0.7891%`，说明 H 分位数规则的场景依赖很强；同时 A2 conditional posterior 与 A1 相同，
  继承 `V51-F09` 的 conditional gate 失败。因此 A2 rejected，不能用较高 unknown recall 或 error separation 掩盖可用覆盖率
  不足，也不能在看到 S 后调整 Q25/Q75、布尔规则或 image threshold。证据：r007；failure delta=`V51-F09/F10`。
- `V51-F11`（`protocol/governance`, `resolved by explicit user authorization on 2026-08-17`）：normative plan 对 Stage A 全失败后的解锁规则内部冲突。§10.8
  明写“所有 Stage A arm 都失败”时保留 U1/U2 并进入 Stage B；附录“八、Stage A 后如何解锁”却只在 Stage A
  candidate 通过 S 时允许 `WS-V51-M1-B-LUDVIG-UPLIFT-01`。r007 的真实状态正是 A1–A4 全 rejected、fallback=
  `U2/B3`，因此执行者不能静默挑选有利条款，也不能把“进入 Stage B”的研究顺序当成独立授权。Stage B 保持
  `pending/locked`；合法复开必须由用户明确选择“授权 U2/B3 fallback 进入 Stage B”或“按 candidate-pass 条款关闭
  M1”，再用 freeze-only commit 统一 normative/short plan/config。用户于 2026-08-17 明确选择“授权 U2/B3 fallback
  继续 M1”，并要求单 arm/scene/工程/paper failure 留档后自动进入下一冻结路线；因此本治理阻塞解除，但原条款冲突和
  r007 结论不删除。解法采用 executable authorization overlay 绑定原 normative/P0/Stage A/proposal SHA，不原地改写
  冻结字节；M2/M3 与 validation/test/KITTI 锁保持。该问题不是算法负结果，解除时仍未读取 C/validation/test/KITTI
  quality。证据：`docs/WORLDSIM_V5_1_M1_TOPCONF_PLAN.md`、`configs/worldsim_v51/stage_a_closeout_v1.yaml`、
  `configs/worldsim_v51/stage_b_authorization_v1.yaml`、`docs/WS_V51_STAGE_B_PREFLIGHT.md`。
- `V51-F12`（`engineering/resource/governance`, `resolved by r003 asset + r004 resource smoke`）：Stage B 的 faithful 第一版要求官方 DINOv2
  ViT-g/14 registers，但 2026-08-17 只读审计只找到 Depth-Anything-V2 内部 DINOv2 模块，torch/HuggingFace cache
  均无对应官方 checkpoint；“存在 DINOv2 源文件”不能写成 LUDVIG 资产已冻结。官方 LUDVIG README 记录的测试平台是
  A6000 48GB，当前 RTX 3090 只有 24576 MiB，且 Stage A 单个 unary materialization 已实测约 20–22 GiB，因此
  DINO extraction 与 DriveStudio renderer 禁止同进程或同卡并发常驻。授权后必须先冻结 upstream commit、官方模型来源、
  checkpoint SHA/license、preprocess 与 PCA population/seed，再采用“离线 DINO sidecar→释放进程→renderer uplift”分段
  执行。官方 checkpoint HEAD bytes=`4,546,140,349`，multipart ETag 不是 SHA；官方 extractor 又会把全 camera raw
  feature 预分配在 GPU。V5.1 只允许先下载后全 SHA，再用 CPU memmap/40-D patch-grid streaming 与 dense-parity test
  保持语义；缺资产或 OOM 只记工程/resource terminal，不得写成 feature uplift 失败，也不得临时换小模型、降分辨率后
  仍称 faithful port。
  资产缺失子项由 r003 解除：official bytes=`4,546,140,349`、SHA-256=`746ecb8c...a283`，本地重算
  8 MiB×542-part ETag exact。r004 随后在 clean source=`935d2b2` 上以官方 commit=`7764ea0f...25fc8`、原始分辨率
  预处理、ViT-g/14 registers 与 strict state dict 完成 one-image forward：params=`1,136,486,912`、keys=`568`、
  missing/unexpected=`0/0`，4 个输出 shape 全 exact；GPU sampled/Torch reserved peak=`6,702/6,376 MiB`、cgroup peak=
  `15,701,860,352 bytes`，显著低于预注册门，资源风险因此 resolved。该解除不代表 feature uplift 有效，也不解除
  DINO→释放进程→renderer 的顺序合同；后续仍禁止同卡并发、临时换小模型/分辨率，且必须先过 operator parity。
  证据：`configs/worldsim_v51/stage_b_preflight_v1.yaml`、`docs/WS_V51_STAGE_B_PREFLIGHT.md`、
  `configs/worldsim_v51/stage_b_dinov2_resource_freeze_v1.yaml`、r004 canonical run。
- `V51-F13`（`engineering/protocol`, `resolved without method execution`）：P0 scope config 已把 normative plan
  SHA-256 冻结为 `3d7f7481...`，但 Stage A closeout commit `3d33262` 曾直接向该长计划加入 5 行执行进展，使当前
  HEAD SHA 漂到 `b119cd56...`。Stage B preflight 运行 `pytest -q tests/test_worldsim_v51_protocol.py` 时因此得到
  `2 failed / 1 passed`，即使本轮新增注记撤回后仍复现，证明这是 inherited drift。这推翻“冻结后的 normative plan
  仍可作普通活文档窄改”的工程假设，不是 P0、Stage B 算法或数据失败。修复用新提交移除这 5 行，把当前状态继续保留
  在 short plan/status/experiments，恢复原 plan SHA exact；不改写历史，也不只改 expected SHA 掩盖漂移。若用户授权后
  确需统一解锁规则，必须建立显式 supersession/migration 并同步 P0 binding。失败时没有下载 checkpoint、启动
  method/GPU run 或读取 C/validation/test/KITTI quality。证据：`3d33262`、
  `configs/worldsim_v51/p0_m1_scope_v1.yaml`、`tests/test_worldsim_v51_protocol.py`、
  `docs/WS_V51_STAGE_B_PREFLIGHT.md`。
- `V51-F14`（`engineering/protocol`, `resolved by r010`）：LUDVIG DINO extractor 的 PCA 路径不是天然确定性合同。
  `PCA(n_components=40)` 没有 `random_state`，大矩阵会走 randomized solver；当 patch 数超过 500,000 时还用未设 seed
  的 `np.random.choice` subsample。更隐蔽的是 GPU path 用 PyTorch `std`（默认 correction=1），CPU path 用 NumPy
  `std`（correction=0），所以为省显存切到 CPU 会改变标准化与全部 feature。V5.1 proposal 冻结 H evidence=
  `45 views×7,296 patches=328,320`，明确不触发 subsample；固定 std correction=1、randomized PCA
  random_state=`20260814`、40-D、whiten=false，并把 scaler/PCA state 持久化后只 transform S/C。不得把 solver/seed/std
  差异当作 backbone 增益或在 S/C refit；这是 reproducibility hardening，不是参数搜索。本轮未下载模型、提取 feature 或
  读取质量。证据：LUDVIG `predictors/dino.py`、`configs/worldsim_v51/stage_b_freeze_proposal_v1.yaml`、
  `docs/WS_V51_STAGE_B_FREEZE_PROPOSAL.md`。r010 已对 45 H views exact 执行该 hardened contract：首图 raw feature
  repeat bit-exact，PCA state deterministic NPZ repeat byte-exact，45 个 sidecar 的 file/content SHA 全 exact，raw memmap
  成功后删除，PCA state SHA=`fe9eea72...3231c8`；因此本条 resolved。该解决只证明 feature/PCA 可复现，不证明
  LUDVIG uplift 或方法质量有效，S/C 仍只准 transform、不得 refit。
- `V51-F15`（`evaluation/governance`, `resolved by r015 without promoting proxy to method input`）：Stage B 的 same-actor/actor-background metric 可从
  frozen `RigidNodes.points_ids[:,0]` 与 Background row 构造，但这是 base-model membership proxy，不是真实 ownership GT。
  若把该 proxy 输入 DINO/PCA/uplift/权重会形成标签泄漏；若只凭 proxy margin 解锁 Graph，则会把模型自身表示循环证明为
  语义正确。proposal 将其限制为 evaluation-only stratum，强制写
  `model_membership_proxy_not_ground_truth`，并同时报告不消费 membership 的 same-Gaussian repeatability 与 heldout DINO
  reprojection。无 eligible actor 的 scene 必须保留 abstain；不得降低 32-Gaussian eligibility、删 1087/0379 或只报大
  Rigid 场景。Stage B 未获授权，本轮没有产生 metric。证据：V5 formal30k r027–r034 metadata、
  `configs/worldsim_v51/stage_b_freeze_proposal_v1.yaml`、`docs/WS_V51_STAGE_B_FREEZE_PROPOSAL.md`。r015 已严格按
  evaluation-only 声明执行：proxy 未进入 method/PCA/uplift，1087 因无 eligible actor 保留 abstain，同时报告不消费
  membership 的 repeatability 与 heldout reprojection；治理风险因此 resolved，但方法因 margin 失败另记 `V51-F31`。
- `V51-F16`（`engineering/resource`, `resolved by parallel r003`）：DINOv2 asset r002 在已 source network turbo、official URL、
  fixed target 与 curl resume 合同下运行约 `106 s`，连续 prefix 仅增长到 `26,566,656 / 4,546,140,349 bytes`；
  按稳定窗口外推需数小时。执行者精确核对并 `TERM` 唯一 curl PID，runner 以 `exit=-15` 写入 blocked terminal，
  final asset 不存在，`.partial` 及其 SHA=`934ef5aa...e2265` 保留。该事实推翻“代理单连接足以在合理实验窗口完成 4.5 GB
  official asset”的工程假设，不是 checkpoint 损坏、DINO/LUDVIG 方法或 GPU 失败。合法恢复必须新 run ID，冻结 prefix
  bytes/SHA，以互不重叠的 fixed HTTP ranges 并行下载；每段验证 range bytes/SHA，assembly 后同时通过 full SHA-256 与
  S3 multipart ETag=`3d1b...-542`（8 MiB×542 parts）才可原子发布。禁止覆盖 r002、删除 prefix 后假装首次下载、
  使用镜像/不同权重或只凭 total bytes/remote ETag 宣称完成。证据：
  `20260817T141600Z__m1-stage-b-dinov2-asset-s20260814-r002`、
  `configs/worldsim_v51/stage_b_dinov2_download_parallel_v1.yaml`。r003 以 14 ranges 在 `1504.935 s` 完成；逐段
  bytes/SHA、assembled full SHA=`746ecb8c...a283`、multipart ETag=`3d1b...-542` 与 terminal/manifest 二次复核全 exact，
  final 原子发布后精确删除 `15 files / 4,546,140,349 bytes` staging。本条工程恢复因此 resolved；当时仍 active 的
  ViT-g 24GB resource smoke 风险 `V51-F12` 后由 r004 独立解除。
- `V51-F17`（`engineering/protocol`, `resolved before formal r005`）：synthetic operator parity 首轮 unit regression=
  `2 failed / 8 passed`，两处 failure 都来自同一个 below-Gaussian-view-mass 夹具：它把 24 个 intersection 全设为
  minimum contribution=`1e-4`，所以聚合 mass 实际为 `0.0024≥0.001`，被测算子正确保留该 group，而测试错误地要求
  drop。这推翻的是“逐 intersection 在 floor 上就能构造低于 group floor”的夹具假设，不是 B0/B1 公式、dense oracle、
  lazy bilinear 或 LUDVIG 方法失败。修复只把该 group 改成 5 个 `1e-4`、其余为 0，使 mass=`0.0005`；不改阈值、
  operator、seed 或通过标准。失败发生在 formal r005 创建前，未加载 DINO/renderer、未读真实 feature 或质量；禁止用
  降低 group floor 掩盖夹具错误。证据：`tests/test_worldsim_v51_feature_uplift.py`、
  `scripts/audit_worldsim_v51_stage_b_operator_parity.py` 的 pre-formal regression。修正后 19/19 regression PASS；formal
  r005 又以 11/11 checks PASS，并真实观测 `8 Gaussian-view → 7 kept + 1 dropped`，确认本条 resolved。
- `V51-F18`（`engineering`, `resolved before r005 result-freeze commit`）：新增 result-freeze test 先通过 canonical run
  文件存在/SHA、summary status/checks/checkpoint immutable，随后因把局部变量 `validate_freeze` 简化为 `freeze` 时漏改
  两条 parity 断言，得到 `NameError` 与 `1 failed / 19 passed`。这推翻“机械重命名后所有引用自然一致”的测试维护假设，
  不推翻 r005 artifact、operator parity 或任一质量结论。修复只替换两处旧变量名，并重跑同一 20-test regression；
  禁止重跑/覆盖 r005 或修改 freeze 数字来绕过测试。证据：`tests/test_worldsim_v51_feature_uplift.py`、
  `configs/worldsim_v51/stage_b_operator_parity_freeze_v1.yaml`。
- `V51-F19`（`engineering/protocol`, `resolved by v2/r007 reaching renderer`）：one-H-view formal r006 在
  `_build_runtime()` 导入 DriveStudio `models.gaussians.basics` 时因 `ModuleNotFoundError: pytorch3d` blocked；v1 config 错把
  入口冻结为 motionproj Python，而历史 DriveStudio 运行合同使用独立 `/root/autodl-tmp/envs/drivestudio/bin/python`。
  terminal 发生在 dataset/trainer 构造和 renderer 启动前，0 intersection、0 denominator、0 quality；这推翻“主项目环境可
  直接承载 DriveStudio CUDA 依赖”的工程假设，不是 renderer/LUDVIG/资源失败。合法恢复必须保留 r006，以 v2 + 新 r007
  只替换 interpreter，并在 formal 内 exact 核对 executable、torch=`2.1.2+cu118`、CUDA=`11.8`、`pytorch3d/gsplat`
  imports；不得安装包污染 motionproj env 或改 view/floor/resource gate。证据：r006 status SHA=`06b74ec9...b4be3`、
  `configs/worldsim_v51/stage_b_one_view_contribution_v1.yaml` 与 v2 recovery config。v2/r007 已 exact 使用该环境并完成
  dataset/trainer/checkpoint/renderer 启动，本条因此 resolved；r007 的后续尺寸阻塞另记 `V51-F20`。
- `V51-F20`（`engineering/protocol`, `resolved by v3/r008 reaching post-render resource gate`）：r007 到达真实单视图 renderer 后，v2 把 sensor
  JPEG `1600×900` 错冻为 renderer width/height，触发 fail-closed。冻结 r027 source config 已明确三路
  `downscale_when_loading=[2,2,2]`，现有 V5 SAM/actor configs 与历史 `V3-F18` 也记录 model-native=`800×450`；这是重复
  违反三层尺寸合同，不是 renderer 或 contribution 质量失败。r007 在 intersection inventory 前停止，且旧错误文本未写出
  observed/expected 数值。合法恢复必须保留 r007，以 v3/r008 显式同时冻结 sensor/downscale/model-native 三层尺寸并增强
  错误文本；不得改 checkpoint、view、support floor 或把 800×450 写成降分辨率调参。loader 会基础设施性物化
  image/mask/LiDAR，但 runner 不消费其值；二者须分开记录。证据：r007 status SHA=`da279515...8d3c9`、r027
  `config.yaml` SHA=`eb22faea...9c6d`、`configs/worldsim_v51/stage_b_one_view_contribution_v3.yaml`。v3/r008 已按
  `800×450` 完成 renderer 并进入 post-render 资源门，因此本条 resolved；后续资源阻塞另记 `V51-F21`。
- `V51-F21`（`engineering/resource/protocol`, `resolved by v4/r009`）：r008 在真实单视图 renderer 和
  contribution 汇总完成后，NVIDIA peak=`14,234 MiB` 超过预注册 ceiling=`12,288 MiB`，故 status=`blocked`；
  cgroup peak 仅 `9,598,074,880 bytes`，89 个采样无错误，进程正常退出且 GPU 已释放，不能误写成 OOM、renderer
  或算法质量失败。原 runner 又在资源门通过后才写 contribution/resource artifact，使 blocked run 只保留
  status/events/resolved/resource-samples；这会降低失败诊断可审计性。禁止覆盖 r008，合法恢复只能新建 v4/r009：
  保持 scene/view/checkpoint/renderer/two-floor/quality locks 不变，仅把 NVIDIA/Torch ceiling 提升为 `16,384 MiB`
  （仍低于 24 GiB），并在资源判定前先持久化只读 denominator/resource 诊断。r008 status/resource-samples SHA=
  `8b8ebe17...b2118bf / fc0f9788...a90932`；不得用这次工程资源事实选择或调节算法质量。v4/r009 已在
  `14,234 MiB NVIDIA / 13,882 MiB Torch reserved / 9,593,946,112 bytes cgroup` 下通过，诊断 artifacts 也在 gate 前
  持久化，因此本条 resolved；冻结结果见 `stage_b_one_view_contribution_freeze_v1.yaml`。
- `V51-F22`（`engineering/audit`, `resolved immediately`）：r009 完成后的第一次独立逐文件 verifier 把 manifest
  payload key 硬编码为 `files`，但该 runner 的冻结 schema 使用 `inventory`，只读命令因此 `KeyError: 'files'`；
  formal r009 及任何 artifact 均未改变。修正 verifier 按冻结 schema 读取 `inventory` 后，8/8 manifest entries 的
  SHA/bytes exact，run=`10 files / 28,156 bytes`。后续 verifier 必须先读 schema/key，不能跨 runner 猜测 manifest 字段。
- `V51-F23`（`engineering/resource/protocol`, `resolved by v2/r012`）：H 45-view sparse uplift r011 已完整处理
  3 scenes×15 views、写出 6 个 Gaussian feature sidecar 并证明 3 个 base checkpoint before/after exact，但 post-compute
  NVIDIA peak=`20,554 MiB` 与 Torch reserved=`20,202 MiB` 超过 v1 预注册的共同 ceiling=`18,432 MiB`，故 formal status
  必须保持 `blocked`。cgroup peak 仅 `13,328,011,264 bytes`，799 个 resource samples 无 monitor error，runner 正常释放 GPU；
  这推翻的是“单-view 14,234 MiB 足以外推三场景 streaming full-run 低于 18 GiB”的资源假设，不推翻 sparse transpose、
  B0/B1、DINO/PCA 或任何 method-quality 结论。合法恢复必须保留 r011 及其 gate 前诊断，以 v2 + 新 r012 从冻结原输入
  完整重跑；唯一改变为 NVIDIA/Torch ceiling=`22,528 MiB`（仍低于 24 GiB），不得复用 blocked sidecar、改 scene/view/floor、
  调整 operator 或读取 membership/quality。r011 失败证据已独立验证 6/6 NPZ file/content identity、manifest chain 与
  checkpoint immutability exact；status/resources/report/manifest/resource-samples SHA=
  `a450cdaf...0eee6/0312e190...8a37/98140571...99c9/88956448...dae6/76422821...ae9c`。配置证据为
  `configs/worldsim_v51/stage_b_h_uplift_v1.yaml` 与 v2 recovery config。v2/r012 从原冻结输入完整重跑，在相同 observed
  NVIDIA/Torch reserved=`20,554/20,202 MiB` 下通过 22 GiB 门；6/6 sidecar、19/19 manifest、3 checkpoint 与全部 locks
  独立审计 exact，因此本条 resolved。该解决不证明 B0/B1 方法质量有效，`V51-F15` 仍须由预注册 evaluation-only 门处理。
- `V51-F24`（`engineering/protocol`, `resolved before formal r013`）：H heldout feature 预注册回归中，新 runner/sidecar
  测试已在 frozen motionproj interpreter 得到 `8 passed`，但随后一次聚合命令又在同一 interpreter 调用依赖 DriveStudio
  runtime 的 `test_worldsim_v51_h_uplift.py`，该测试按合同报告 runtime mismatch，汇总为 `1 failed / 9 passed`。这只是测试
  调用环境错误，未创建 formal run、未启动 GPU、未读 membership/uplift quality，不能写成 H uplift 或 heldout transform 失败。
  合法修复是按 `V51-F19` 的环境分层分别执行：DINO/sidecar/heldout tests 使用 motionproj Python，renderer/uplift tests
  使用 drivestudio Python；禁止为让聚合命令通过而改任一冻结 runtime config 或向主环境安装 DriveStudio CUDA 依赖。
  r013 config 将本条纳入 `failure_ledger_refs`，并须在 clean prereg commit 前保留两个解释器各自的 PASS 证据。
- `V51-F25`（`engineering`, `resolved before formal r013`）：从 Windows PowerShell 发出的第二次聚合 SSH 命令在
  双引号内包含 `$(find ...)`；本地 shell 在 SSH 前提前解释该命令替换，并把远端 `find -name` 片段误当成 PowerShell
  命令，最终本地报 `-name is not recognized`、远端 bash 报 unmatched quote。测试没有启动、GPU 未使用、仓库未修改、
  quality 未读。防重复门禁：跨 PowerShell→SSH 的命令不得嵌套未转义命令替换；本次改为独立执行测试、`git diff --check`
  与只读 `find`，不使用 shell substitution 控制流。不得把 launcher quoting 失败计入算法或测试 verdict。
- `V51-F26`（`engineering`, `resolved during r013 post-run audit`）：首次只读 r013 inspection 又把多语句 Python
  `-c` 嵌入 PowerShell 的双引号 SSH 字符串，本地 parser 在远端执行前把 Python 括号误解释为 PowerShell，报
  `An expression was expected after '('`；没有命令抵达远端、没有 artifact/repo/GPU/quality 状态变化。该问题与
  `V51-F25` 同属跨 shell 引号边界，但触发面是嵌套 Python source。合法修复为用 `apply_patch` 创建独立只读 auditor、
  `scp` 到精确 `/tmp` 路径后以固定参数执行；后续禁止在 PowerShell→SSH 双层命令中内嵌多语句 Python source。
- `V51-F27`（`engineering`, `resolved during r013 post-run audit`）：同步 auditor 与三份台账时，命令工作目录是
  local staging 根而三份 source path 误写成 `docs/...`，因此 auditor 已成功传到精确 `/tmp/audit_r013.py`，随后三个
  docs scp 在本地以 `stat local ... No such file` 失败；远端 repo 和 run 均未被部分修改。修复只把三份 source path
  写成 `motion_proj/docs/...` 并逐项同步；防重复要求多 source scp 前先按当前 workdir 解析 source，且不能把部分成功
  的前序传输误当成整条命令成功。
- `V51-F28`（`engineering/resource/protocol`, `resolved by v2/r015`）：formal H evaluation r014 已完整处理
  3 scenes、90/90 evidence/evaluation views并先持久化 3 scene reports 与完整只读 report，但 terminal resource gate
  观测 NVIDIA peak=`22,570 MiB > 22,528 MiB`、Torch reserved=`23,354 MiB > 22,528 MiB`，因此 status 必须保持
  `blocked`。cgroup peak=`14,305,161,216 bytes`，1,208 samples、0 monitor errors、duration=`897.647 s`，GPU 已释放；
  这推翻的是“r012 的 22 GiB uplift ceiling 足以覆盖双向 evaluation sparse projection”的资源外推，不是 H gate verdict。
  禁止读取 blocked r014 的 scene/aggregate quality、覆盖 run 或据其数值改 metric/pair/proxy/gate。合法恢复只允许新 v2/r015
  把 NVIDIA/Torch ceiling 同时提高到 `24,000 MiB`（仍低于 24,576 MiB 卡容量），其余 base config 逐字继承，并从
  r012/r010/r013 原冻结输入完整重跑。r014 status/resources/report/progress/resource-samples SHA=
  `6409545b...f6d1/ffc98a00...674e/510f82ec...227c/61475cb1...06ed/8fae05eb...7cf`；10 files，无 partial。
  v2/r015 从原 freeze 完整重跑，在同一 NVIDIA/Torch=`22,570/23,354 MiB` 下通过 `24,000 MiB` 门，故本条 resolved；
  r015 的 H verdict 由独立质量门决定，不能倒写 r014 为成功。
- `V51-F29`（`engineering/audit`, `resolved immediately`）：r014 blocked metadata 首次只读 hash 命令把实际
  `events.jsonl` 误写成 `events.json`，`sha256sum` 因该单项不存在返回非零，使后续 `&& find` 没有执行；前面其余 hash
  已正常输出，run/repo/quality 均未改变。修正为冻结 schema 的 `events.jsonl` 后 SHA=`31fff013...a7ce`，并完成 10-file
  bytes inventory。后续 verifier 必须从 runner/schema 读取精确 artifact 名，不凭相邻 runner 猜扩展名。
- `V51-F30`（`engineering/audit`, `resolved during r015 closeout`）：独立 auditor 首先要求 blocked r014 与 recovery r015
  report 在删除 `seconds` 后逐 Python float exact，得到 assertion failure。递归定位显示差异均为并行 CPU sparse/BLAS reduction
  的末位浮点扰动；离散字段、denominator、checkpoint、gate verdict 全 exact。修正审计合同为离散字段 exact、float
  absolute tolerance=`1e-12`，共 241 个差异，最大仅 `4.9760e-13`，复核 PASS。禁止把非 bit-exact reduction 误写成方法
  不可复现，也不得用宽松相对容差掩盖 gate 翻转；任何离散/gate 差异或 float 超过 `1e-12` 仍须 fail。
- `V51-F31`（`algorithm/evaluation`, `active rejected-route prevention`）：canonical r015 的 H gate 为 rejected。3 scenes
  中仅 0471/0379 evaluable，B1 actor-background margins=`-0.121280/-0.098618`，正场景=`0/2`，scene-balanced=
  `-0.109949<0`；1087 按冻结 32-Gaussian rule 无 eligible active actor，必须 abstain。Rigid coverage mean=`0.842910`
  与 heldout reprojection `B1-B0=+0.022777` 均过门，说明 normalized transpose 能改善 2D feature reconstruction，
  但不能证明 actor 内 feature 比最近 Background 更紧致；0471/0379 的 B0 margin 也已为负，且 B1 没有救回该前提。
  这推翻 LUDVIG uplift 可直接支撑 driving Gaussian semantic graph 的核心假设，因此按预注册同时 reject raw LUDVIG graph，
  不得降低 actor minimum、删除 1087、只报 reprojection 或先加 Bayesian/SAM/motion edge 救 graph。下一合法路线是独立
  faithful progressive propagation；S/C/validation/test/KITTI 仍未读。
- `V51-F32`（`engineering/launcher`, `resolved before D0 preregistration`）：首次 D0 只读 inventory 把含
  `U2|B3|Bayesian|...` 的正则直接嵌入 PowerShell→SSH 命令，外层 shell 抢先解释 `|`，导致远端 `sed` address
  截断并把各 regex 分支误当命令；没有文件、run、GPU 或 quality 状态变化。后续统一把多行只读脚本 UTF-8 base64
  编码后交给远端 `bash`，避免跨 shell parser 改写。不得把 launcher quoting failure 计入 D0 方法 verdict。
- `V51-F33`（`engineering/runtime`, `resolved before D0 preregistration`）：第二次 artifact inventory 在远端使用裸
  `python`，但主机 PATH 按合同没有该命令；三个 YAML path 已被 `rg` 只读打印，内嵌解析均未运行，也未改状态。
  修正为显式 `/root/autodl-tmp/envs/motionproj/bin/python` 后完成 NPZ identity/count/quantile 审计。后续所有 V5.1
  runner/auditor 必须使用冻结解释器绝对路径，禁止把 shell PATH 差异写成数据或算法失败。2026-08-20 收尾清理预审
  首次又假设 `/usr/bin/python3` 存在并在任何 inventory 写入前失败；随后显式使用 `/root/miniconda3/bin/python` 运行同一
  审计，候选集合未变化。该复发没有研究资产状态变化，进一步要求一次性维护的 cleanup 工具也必须绑定已探测的绝对解释器。
- `V51-F34`（`engineering/runtime`, `resolved before D0 preregistration`）：D0 扩大回归时又用 motionproj Python
  调用了依赖 DriveStudio runtime identity 的 H evaluation config test，得到该项 runtime drift；D0 新测试本身已
  `4/4 PASS`，没有 formal run、GPU 或 quality read。这是 `V51-F24` 的重复触发，说明仅在文档记环境分层不足。
  r020 freeze 的扩大回归再次用 motionproj 聚合 26 个 V5.1 test files，结果 `95 passed / 3 runtime-drift failed`；失败项
  正是三个已冻结为 DriveStudio Python 的旧 H tests，本次 E0a 定向 8 项均 PASS。按 suite 拆分后再用 frozen DriveStudio
  interpreter 补跑 3 项通过。后续回归命令必须在生成 file list 时就按 runtime 分组并分别记录结果；不得改 runtime
  freeze 来迁就聚合命令，也不得把这一环境错误写成算法 regression。
- `V51-F35`（`engineering/protocol`, `resolved before D0 preregistration`）：同一扩大回归发现 P0 scope 与 Stage-B
  authorization 仍保留最初 top-plan SHA=`3d7f7481...`，而 commit `b359541` 为预注册 H heldout contract 对计划做了
  17 行 append-only 更新，当前 SHA=`b4888476...`；旧 validator 只允许单 hash，导致 4 个既有 protocol tests 在真实
  route assertion 前 fail。禁止改写两份历史 freeze 的 recorded SHA。修复在 validator 中显式固定
  `base hash → authorized append hash` 两状态链，第三种状态仍 fail-closed，并增加 current/historical 双 hash 回归。
- `V51-F36`（`engineering/test`, `resolved before formal D0 operator run`）：首个 progressive expansion unit fixture
  预期 `p=0.5` 节点最终 UNKNOWN，却把该节点直接连到 `p=0.01` Background seed。两者 L2-normalized binary
  distribution cosine 约 `0.714`，高于冻结最低 threshold=`0.5`，故算子把它合法扩张为 Background；失败的是 fixture
  的“无支持”假设，不是算法。修复只删除这条边，使该节点真正孤立；其余阈值/公式/实现不变，5/5 operator tests PASS。
  禁止为满足错误预期而提高阈值、加入 confidence gate 或改变 UNKNOWN 语义。
- `V51-F37`（`algorithm/evaluation`, `active; D0 rejected by r018`）：faithful SAI3D-style raw-Gaussian progressive
  propagation 在 frozen H matched 12 views 上只通过 BF1 两项门：positive scene=`2/3`、scene-balanced BF1=
  `+0.0002196`；IoU=`-0.0714543<0` 与 FN semantic mass=`+0.1694766>+0.02` 同时 FAIL。0471 的
  BF1/IoU 有改善，但 FN 仍 `+0.080830`；1087/0379 的 IoU=`-0.159417/-0.220397`、FN=
  `+0.218146/+0.209454`，说明减少 FP/提高部分 calibration 并不能补偿 actor 漏检和跨场不稳定。这推翻“在 raw
  Gaussian 上按冻结 KNN 与多视图 SAM affinity 做 progressive growing 可稳定优于 U2/B3”的假设，不推翻所有 graph
  或 super-primitive 路线。D1 永久跳过；禁止按 r018 调 thresholds/hops/seeds/affinity 或重读 H。合法后续只有按冻结
  顺序进入 Stage E，先以 no-quality E0 检查 node elevation 是否提高 observation density，再按其门禁决定 E1/E2。
  证据：r018，source=`2cd98b3`，summary/manifest=`b08c7276...62d6/792660e3...010c`，independent metric/gate
  replay=`18c12f4d...0d2`，freeze=`configs/worldsim_v51/stage_d_progressive_h_evaluation_freeze_v1.yaml`。
- `V51-F38`（`engineering/shell`, `resolved during r019 monitoring`）：旁路进度查询再次在 PowerShell→SSH 边界使用
  `$run`，本地 shell 先展开变量并破坏远端引号，得到 `unexpected EOF`；正式 r019 进程独立运行、未受影响，也没有
  repo/run 写入。后续监控只能使用绝对字面路径或仓库内 CLI，禁止跨 shell 传未编码变量。这是 `V51-F32` 的复发，
  说明“已知坑”仍需由可执行入口而非记忆约束。
- `V51-F39`（`engineering/data-contract`, `resolved by v2/r020`）：formal r019 在 0471/1087 完成后，
  因 0379 frozen KNN 含 `34/7,123,746` zero-length edges 而 blocked；v1 把“用于 voxel scale 的 edge length”错误
  写成全量严格正值。三场 nonfinite edge 均为 0，0379 仍有 `7,123,712` positive edges，因此这是 quantile 输入合同
  过强，不是算法质量、OOM 或 corrupted geometry。r019 terminal/13 files 保留，partial assignments 禁止晋级/复用。
  合法 recovery v2 只排除零长 edge 的 scale statistic，保留全部 Gaussian，其他 quantiles/gate/views/locks byte-semantic
  继承，并以新 r020 完整重跑；r020 三场/九档 assignment、metrics 与 gate 独立复算 exact，report=`8df03b2a...5d34`。
  防重复边界仍成立：不得把 zero edge 直接删除出后续 topology，也不得借 recovery 改 voxel level。
- `V51-F40`（`engineering/shell`, `resolved with enforced command boundary during r020 freeze`）：量化 F39 时首次 base64 远端 Python 命令仍错误嵌入双引号，
  PowerShell 把 `base64.b64decode(...)` 当作本地命令，远端再次未执行。修复不是继续堆转义，而是新增可测试的只读
  `scripts/audit_worldsim_v51_e0a_edges.py`；CLI test PASS，三场 edge identity/zero/nonfinite/positive quantile 审计完成，
  report=`30493d5d...bc5`。r020 freeze 时又误用一次跨 PowerShell/SSH inline `python -c`，只产生远端 SyntaxError、未写入
  run；随即改为本地解析 YAML、远端只运行仓库 CLI/pytest。后续需要多语句远端分析时必须先落地仓库内 auditor，
  禁止临时内嵌脚本；单语句也不得跨两层 shell 手写嵌套引号。r022 审计前的旁路摘要查询再次因 heredoc 嵌套引号
  得到 `unexpected EOF`，随后发现远端没有 `jq`；两者均未写 run。改为 scp 冻结 JSON 后在本地只读解析，并由仓库
  auditor 完成正式审计。进入 Stage F 后又有一次含 `$f` 的远端 loop 被 PowerShell 提前展开，以及一次嵌套
  `python -c` 验证命令 SyntaxError；均在 formal r023 前、无 run/asset/repo 状态变化。该复发进一步说明远端临时解析
  不是证据入口；正式 source audit 必须由仓库 runner 完成。r025 freeze 上传后的 YAML smoke 又因 PowerShell 中手写
  `python -c` 反斜杠转义得到 `unterminated string literal`，紧接着尝试 stdin 单层命令仍被本地 quote stripping 破坏；
  两次测试链都在该点停止且未修改 run/repo 文件。最终不再修补 shell quoting，改为仓库 pytest 直接加载 freeze YAML，
  并配合 `git diff --check` 复核。该 recurrence 不影响此前已 PASS 的 r025 独立 auditor。2026-08-20 cleanup inventory 又有
  两次 inline `awk`/shell-loop 因 PowerShell→SSH quoting 失败；两次均为只读、没有创建/修改/删除研究资产。最终把审计和
  fail-closed deletion 放入固定 Python 文件，以 exact JSON plan 执行。禁止再为临时汇总跨两层 shell 拼循环、变量或 awk。
- `V51-F41`（`engineering/environment`, `resolved during r020 audit`）：本地 `autodl-stage/motion_proj` 只是按约束用于
  `apply_patch` 的 partial staging tree，不包含完整 `motion_proj.worldsim_v5` package；误在该目录收集 E0a 联合测试时触发
  `ModuleNotFoundError`。这不是 canonical repo、r020 或 auditor 失败。修复为只在本地做语法检查/编辑，将新增文件同步到
  远端完整 clean checkout 后运行同一测试，结果 `8 passed`；随后 r020 独立审计通过。后续不得把 partial staging 当作
  可运行 checkout，也不得为迎合该环境复制缺失 package 或修改 import path。r022 审计阶段在同一 staging tree 误跑
  `git diff --check`，因它不含 `.git` 只打印 usage；命令没有修改文件，正式 CLI test 与审计仍在远端完整 checkout PASS。
- `V51-F42`（`algorithm/evaluation`, `active; E0B rejected by r022`）：simple voxel super-primitive control 的
  `fine_q50 + member-unary mean + visibility-weighted SAM mean + max visibility + frozen D0 propagation` 在 frozen H matched
  12 views 上未能优于 U2/B3，也未能稳定优于 raw D0。相对 U2/B3 虽有 BF1 positive scenes=`2/3`，scene-balanced
  BF1=`-0.0002566`、IoU=`-0.0925468`、FN=`+0.1899473` 全部 FAIL；相对 D0 的 BF1 nonnegative scenes 仅
  `1/3`，mean BF1=`-0.0004762`、IoU=`-0.0210926`、FN=`+0.0204707`，四项机制门全 FAIL。0379 相对 D0
  `ΔIoU=-0.064618 / ΔFN=+0.067752`，说明确定性 voxel 合并与 member evidence 平均会扩散弱/错误证据，结构密度提升
  不能推出语义质量提升；1087 的近 no-op 也未形成可泛化收益。该结果只推翻这套 simple node-elevation 实例，不推翻
  faithful Gaussian Grouping 或所有 graph/node 方法。E1 PanoGS 与 E2 AG²aussian 按预注册停止，禁止依据 r022 调
  voxel level、node aggregation、seed/threshold/hop、删除 0379 或重读 H；下一合法路线是 Gaussian Grouping faithful
  source audit 与 no-quality preflight。证据：r022 summary/manifest=`4964a2f0...3d4/3c5a2fbe...7aa`，independent dual-gate
  replay=`5ced73db...104f`，freeze=`configs/worldsim_v51/stage_e_e0b_h_evaluation_freeze_v1.yaml`。
- `V51-F43`（`engineering/tooling`, `resolved before F0 preregistration`）：下载并哈希 Gaussian Grouping official PDF 后，
  远端没有 `pdfinfo/pdftotext`；桌面依赖清单给出的 Poppler override/fallback 也不可执行。改用 bundled Python 的
  `pdfplumber` 读 18 页，但首次输出受 Windows GBK 限制，遇到作者脚注符号触发 `UnicodeEncodeError`；只设置任务级
  `PYTHONIOENCODING=utf-8` 后完成全文提取，并用已存在的 `pypdfium2` 渲染方法第 6–8 页检查公式与图示。没有安装
  系统包、没有改 PDF、也未触及方法数据。后续 PDF source audit 优先复用 bundled Python 并显式 UTF-8，不假设远端或
  dependency catalog 中声明的 Poppler binary 实际存在；工具缺失不得写成论文或算法证据。
- `V51-F44`（`engineering/runner`, `resolved by r024`）：F0 source preflight runner 复用 Stage-B `_git`
  helper 时写成 `_git("rev-parse", "HEAD")`，但 helper 签名是 `_git(project, *args)`，实际调用变成
  `git -C rev-parse HEAD` 并在 source commit 读取处失败。r023 此时只创建 run 目录和 `resolved_config.yaml=7,796`
  bytes，尚未写 status、启动 resource monitor、读取 source/data/schema、运行 CUDA smoke 或读取任何 quality；它是不可晋级
  的 incomplete shell。合法 recovery 只新增 `repository_source_identity(PROJECT)` 显式绑定与参数顺序回归，用新 clean
  commit/r024 从头运行；r024 已越过 source identity 并执行到 adapter smoke 前，证明本项修复有效。不得删除 r023、
  手补 terminal 或借机改变 F0 source/method/data contract。
- `V51-F45`（`engineering/runtime`, `resolved by r025`）：r024 完成 official source identity、代码语义与三场
  train-only metadata/observation schema 的内存检查后，在 16D adapter smoke 前直接调用
  `torch.cuda.reset_peak_memory_stats(torch.device("cuda:0"))`；当前进程尚未初始化 CUDA context，PyTorch 返回
  `Invalid device argument 0: did you call init?`。r024 status=`blocked`，只有 resolved/status/events/resource samples
  四文件=`9,449 bytes`，没有 source report、CUDA render、SAM/DEVA/identity training 或 quality read。合法 recovery 只按
  `set_device → one-scalar allocation/context init → reset peak → smoke` 顺序执行并加入 call-order regression，新 r025 从头
  重跑；r025 与独立 replay 均得到 `[1,32,32,16]` render、`189` positive-alpha pixels、`48/48` identity gradients、
  base gradients absent，GPU peak=`310 MiB`，证明初始化顺序修复有效。禁止放宽 resource ceiling 或把已在 blocked run
  内存检查过的数据结果晋级；canonical 只认 r025。
- `V51-F46`（`data-contract/algorithm`, `active prerequisite after r025`）：Gaussian Grouping official identity mechanism
  需要 SAM everything masks 经 DEVA semionline 关联后的跨视图一致 short IDs；r025 独立核对三场各 15 个 train-only
  observation 后，45/45 只有 binary actor-union probability 等同一套 23 fields，没有任何
  `instance/identity/object_id/mask_id/class_id` label。`instances_info/frame_instances` 中的 stable actor token 只描述场景级
  track metadata，不提供每像素 mask；三个 formal checkpoint 也只能提供已训练 Gaussian state，不能反推出监督标签。
  同时 official DEVA propagation 与 SAM ViT-H weights 均 absent，现有 SAM2.1 Hiera Large 虽有 checkpoint，但不符合上游
  SAM-v1 everything-mode source contract。该缺口不否定 source core 或 frozen-base 16D adapter（两者 r025 PASS），但
  `current_training_input_ready=false /identity_training_authorized=false`。合法下一步只能先预注册并执行 train-only F0a
  asset acquisition + SAM/DEVA identity-mask materialization，冻结 URL/SHA、输出 schema、确定性、资源与 partial recovery；
  禁止用 metadata、binary U2/B3、SAM2 或 evaluation target 代替，禁止在 materialization 冻结前启动 F0 training。
  证据：r025 summary=`da4890d...988`、audit=`14d2b78b...8b64`、
  `configs/worldsim_v51/stage_f_f0_source_preflight_freeze_v1.yaml`。
- `V51-F47`（`engineering/orchestration/tooling`, `resolved during r026`）：首次启动 r026 时把远端 Linux command 与
  `/root/autodl-tmp/motion_proj` workdir 直接交给桌面本地 `bash` tool，Windows 在建立 SSH 前返回
  `CreateProcess ... 目录名称无效`；远端 run/path/assets 均尚不存在，因此这不是 r026 blocked terminal，修复为从本地
  PowerShell 显式 `ssh wm-3090-0811` 后按原 prereg run ID 启动。r026 完成后只读汇总文件字节时又假设远端存在 `bc`，
  在已打印 auditor hash 与 status/manifest size 后报 `bc: command not found`；它没有修改 run，完整字节数改由已审计的
  manifest inventory 加 status/manifest exact size 得到 `51,021`。这两次都推翻“tool shell/workdir 与常用 CLI 可跨本地/
  远端默认存在”的工程假设，不影响 r026 的 `done` 或 asset hashes。后续远端命令必须从 Windows 使用 SSH alias，run
  证据计算由仓库 runner/auditor 完成，禁止临时依赖未冻结的 `bc/jq/python -c`。证据：r026 audit=`5a360f42...817c`，
  freeze=`configs/worldsim_v51/stage_f_f0a_asset_source_acquisition_freeze_v1.yaml`。
- `V51-F48`（`engineering/data-contract`, `resolved before formal r027`）：r027 prereg config 首稿手录
  scene-0471/frame0/camera0 SHA 时，把 canonical `093d38e8d8d8f12...819e` 漏写一个 `d8` 成
  `093d38e8d8f12...819e`。formal-config pytest 在任何 r027 run 目录、wheel/env mutation、GPU/model/image decode 前
  fail-closed；远端 `sha256sum` 与 r026 已独立审计的 selected manifest 一致，证明是配置转录错误而非图像漂移。修复只从
  r026 canonical record 恢复完整 SHA，并以原测试重跑；不得改图、重选 view、放宽 hash 或把该失败写成 SAM/DEVA 结果。
  后续长 identity 必须由 manifest 机器传递并保留 config-validation test，禁止凭聊天摘要/截断 hash 手工补全。
- `V51-F49`（`engineering/license/runtime`, `resolved by r028/r029`）：r027 已 atomic 构建 isolated venv 并安装
  exact `supervision=0.14.0/PuLP=2.7.0/gurobipy=10.0.3`，但第一个 Gurobi tiny MILP 在创建 model 时报告
  `License expired 2024-10-28`，因此 status=`blocked`。失败发生在 one-view upstream CLI 前：没有加载 DEVA/SAM 权重、
  没有 GPU model forward、没有 decode input/mask、没有 quality；4 files=`12,861 bytes`，不得把它写成 identity mechanism
  或 association 失败。根因是把上游 `gurobipy>=10.0.3` 的最低版本误冻结成 exact 10.0.3，而该 wheel 的内置 restricted
  runtime 已过期；服务器没有另一个 `gurobi.lic` 可续用。合法 recovery v2 只将 Gurobi 提升到当前 index 可得的
  `12.0.3`（仍满足上游版本下界），使用全新 wheelhouse/venv/r028 重跑，其他 source/assets/input/CLI/resource/locks
  不变；仍要求 Gurobi tiny optimum，禁止直接跳过 gate 或静默采用 PuLP 后声称 faithful。r028/r029 的 Gurobi 12.0.3
  tiny model 均得到 status=`2`、solution=`1.0`，因此 license 前置已解除；后续 stdout 与 GPU failures 分别独立记账。
- `V51-F50`（`engineering/source-provenance`, `resolved before r028`）：r027 environment path verification 用子 Python
  import frozen DEVA source，默认 bytecode policy 在 Gaussian Grouping checkout 内生成 4 个未跟踪 `__pycache__` 目录；
  tracked diff 为 0，但后续 v2 config 的 clean-source gate 正确 fail-closed。该污染不是 upstream 修改、算法失败或 r028 run；
  在确认 exact paths 后只删除这 4 个由本任务生成的 cache，并让 runner 的全部子进程继承
  `PYTHONDONTWRITEBYTECODE=1`。第一次清理 wrapper 又因 PowerShell 提前处理 `$p` 而出现 quote EOF，未删除或修改任何
  文件；随后改用 4 个显式绝对 target 完成清理，两个 external repo 恢复 clean。禁止把 `git status` 放宽为忽略 untracked，
  也不得把 source tree 内 cache 纳入冻结；后续 import smoke 必须同时核对 commit/tree/clean。
- `V51-F51`（`engineering/runner`, `resolved by r029`）：v2/r028 成功获取并安装
  `gurobipy=12.0.3`，Gurobi 不再抛 license-expired/CalledProcessError；但 restricted-license banner 与 runner 自己打印的
  JSON 同时写 stdout，v2 对整段 `json.loads` 而得到 `JSONDecodeError line 1 column 1`。r028 因此在 solver output parse
  blocked，4 files=`10,642 bytes`；one-view CLI、DEVA/SAM load、input/mask decode 与 quality 仍全部未发生，不能把它写成
  solver、association 或算法失败。v3/r029 唯一修复是解析最后一个非空 stdout 行为 JSON，并把前置 banner 原样存档；
  solver status=`OPTIMAL` 与 solution=`1` 的门不放宽，环境/权重/view/CLI/resource/locks 全继承 v2。禁止粗暴丢弃全部
  stdout、用正则猜数字或因 parser failure 绕过 Gurobi gate。r029 已越过两种 solver gate 并启动 official CLI，证明
  terminal-line parser 恢复有效；r029 后续 OOM 另记 `V51-F52`，不得倒写本项未解决。
- `V51-F52`（`engineering/resource`, `active; allocator-only recovery disproved by r030`）：r029 首次完整越过 environment、Gurobi/PuLP 和
  official model-load gate，在唯一 scene-0471/frame0/camera0 输入上执行 SAM ViT-H everything mask；上游默认
  points-per-side/batch=`64/64`，在 `BatchMaskData.cat` 尝试再分配 `6.74 GiB` 时 CUDA OOM。错误现场为 GPU total/free=
  `23.56/6.72 GiB`、process=`16.83 GiB`、PyTorch allocated/reserved-unallocated=`10.74/5.77 GiB`，独立 resource samples
  peak=`17,246 MiB`。r029 status=`blocked`，6 files=`22,458 bytes`，没有 mask、`pred.json`、identity quality 或 cross-view
  association denominator；这推翻“24GB 可直接运行 official default one-view”的资源假设，不推翻 Gaussian Grouping 算法。
  第一合法 recovery/r030 只设置 allocator `max_split_size_mb:128`，source/view/size/IoU/grid/batch/ceilings 均不变；若仍
  OOM，必须保留 r030 后另行预注册 `SAM_NUM_POINTS_PER_BATCH` 单变量 batching adaptation，并补 parity/repeatability，禁止
  同轮缩图、改 grid/阈值或把工程失败写成 algorithm reject。r030 已证明该 allocator-only recovery 不足，后续事实续记
  `V51-F55`。证据：r029 stderr=`205266c9...403`、resource=`a0d825a8...01a`、status=`ff75badc...f66`。
- `V51-F53`（`engineering/source-provenance`, `resolved by r030 asset publish`）：r029 模型构造首次触发冻结 DEVA source
  `deva/model/resnet.py` 中的 `model_zoo.load_url`，隐式从 PyTorch URL 下载 ResNet50/18 到用户级 `/root/.cache`。两份
  资产分别为 `102,502,400 bytes /19c8e357...097` 与 `46,827,520 bytes /5c106cde...13f8`；此前 F0a asset freeze 未枚举
  这项 transitive dependency，故“官方 DEVA+SAM 两权重已经覆盖全部模型资产”的来源合同被推翻。合法 recovery 固定 source
  literal URL、bytes/full SHA，把 exact cache 原子复制到专用
  `/root/autodl-tmp/models/gaussian_grouping_v51_stage_f/torch_home/hub/checkpoints`，并让 subprocess 固定 `TORCH_HOME`；禁止
  继续依赖用户 cache、重新下载未验 hash、修改 frozen upstream 或把 ResNet 权重混称 DEVA checkpoint。canonical 目标独立
  审计前不得删除原 cache；审计后只允许精确清理由本次生成且 hash 匹配的两个源文件。r030 前置已把两份资产原子发布到
  dedicated target；独立复核 bytes/full SHA、无 `.partial`，且 official CLI stderr 没有 download 行，证明 `TORCH_HOME`
  生效。本项来源缺口已解除。r032 canonical audit PASS 后，两份原 cache 源副本已按 exact path 精确删除；dedicated
  `TORCH_HOME` copies full SHA 保持，后续不再依赖用户 cache 或网络。
- `V51-F54`（`engineering/orchestration`, `resolved before prereg commit`）：第一次提交 v4 时把包含括号与多段正文的
  `git commit -m` 放进 Windows PowerShell→SSH→bash 双层命令；本地层提前剥掉远端参数引号，bash 在 Conventional Commit
  标题的 `(` 前直接 syntax error。失败发生在 Git 创建 commit、push、r030 run 或 canonical asset publish 之前；staged diff
  保持不变，不是 config、test 或算法失败。这是 `V51-F47` 跨 shell 合同的复发：恢复改为在本地用 patch 生成独立 commit
  message 文件、scp 到远端临时路径，再以 `git commit -F` 单参数读取；禁止继续手工嵌套长 `-m`、省略正文或覆盖 staged
  内容。提交后必须精确删除临时 message，并重查 branch/status/commit。
- `V51-F55`（`engineering/resource`, `active; batch-only recovery disproved by r031`）：r030 在 source=`33c013d` 上保持同一 input、官方
  point grid/batch=`64/64` 与全部方法参数，仅增加 `max_split_size_mb:128`。reserved-unallocated 已由 r029 的 `5.77 GiB`
  降到 `578.72 MiB`，说明碎片明显减少，但同一 `BatchMaskData.cat` 仍尝试分配 `9.49 GiB`，free=`9.16 GiB` 而 OOM；
  sampled GPU peak=`24,098 MiB`，超过 prereg `24,000 MiB`，cgroup=`18,035,429,376 bytes`，101 samples/0 errors。
  6 files=`25,873 bytes`，没有 mask、metadata 或 quality。该结果推翻 allocator-only 足以让 official default batch 在 3090
  上运行的假设，也提示主要约束已不是 allocator fragmentation；仍不构成算法 reject。下一合法 recovery/r031 只把上游
  文档明确称为 parallel point prompts 的 `SAM_NUM_POINTS_PER_BATCH=64→32`，保留 points-per-side=`64`，并要求成功后另做
  batch parity/repeatability；禁止同轮改 grid、size、IoU、模型或资源 gate。证据：r030 stderr=`15d9bd12...bef`、resource=
  `39060722...6ad`、status=`d38fd753...0f7`。r031 已证明 batch32 仍不足，后续累计规模事实续记 `V51-F57`。
- `V51-F56`（`engineering/orchestration`, `resolved before r031 prereg`）：为核对 batch 参数来源所发的只读 `rg` 命令在
  Windows PowerShell→SSH→bash 双层字符串内包含 alternation `|`，引号被提前剥离后 bash 把后半段当命令，返回
  `points_per_batch: command not found`。没有 repo/run/asset 状态变化；恢复为单关键词、无 pipe 的 `rg`，定位到 DEVA
  `docs/DEMO.md`、`ext_eval_args.py` 和 `automatic_sam.py`：参数默认 64，定义为每批并行 point prompts，并直接传给 SAM
  `points_per_batch`。今后临时只读 SSH 查询也必须避开嵌套 alternation/pipe，复杂查询落到本地或正式 auditor。
- `V51-F57`（`engineering/resource`, `resolved for 24GB by r032 grid32; default-grid boundary remains`）：r031 精确执行 v5 的唯一变化
  `SAM_NUM_POINTS_PER_BATCH=64→32`，stdout 确认 side/batch=`64/32`，但仍在同一 `MaskData.cat` 累积点 OOM：request/free=
  `9.32/9.31 GiB`、allocated/reserved-unallocated=`13.34 GiB/599.11 MiB`，GPU peak=`24,066 MiB`、cgroup peak=
  `18,052,734,976 bytes`、119 samples/0 errors。6 files=`28,677 bytes`，mask/metadata/quality 均 absent。相对 r030 的
  `9.49 GiB` request 仅下降约 `0.17 GiB`，推翻“缩并行 batch 可解决最终累计 masks 峰值”的假设；继续 batch16 是重复
  调参，未授权。源码表明每批 full-resolution masks 在 NMS 前累积，规模主要受 points-per-side² 控制；DEVA 官方文档又明确
  建议降低 `SAM_NUM_POINTS_PER_SIDE` 来减少 automatic queries。下一合法 recovery/r032 只设 side=`32`（1024 prompts），
  batch=`32` 与其他参数/门禁不动；它必须标作 documented resource adaptation，成功后需同-grid batch parity、3-view
  association/repeatability 和后续质量门，禁止把 resource PASS 冒充 default-grid parity。证据：r031 stderr=
  `b822aab6...692`、resource=`0a06475d...af4`、status=`99d081ee...e23`。r032 以 side/batch=`32/32` 在 GPU peak
  `23,954 MiB` 内完成 output schema，解除当前 24GB execution prerequisite；但 default grid64 仍不可运行，grid32 quality/
  association 尚未证明，不能删除 r029–r031 或声称 exact-default parity。
- `V51-F58`（`engineering/orchestration`, `resolved after r032 audit`）：按 `V51-F53` 清理门禁删除两份用户 cache 前，首次
  wrapper 用嵌套 `$(sha256sum ... | cut -d " " -f1)`；PowerShell/SSH/bash 再次破坏 delimiter 引号，`cut` 在第一个
  `&&` 前退出，两份文件均未删除。已有 r032 independent audit 保存 source/canonical full SHA，恢复改为无 pipe/无命令替换
  的两个显式 `rm -f`，随后验证源路径 absent 且 canonical SHA 分别保持 `5c106cde...13f8/19c8e357...0097`。禁止在双层
  shell 中拼 checksum parser；以后先由 auditor 落证据，再用 exact path 单动作清理并独立验证。
- `V51-F59`（`protocol/data-contract`, `resolved by r033 association subgate; one-view boundary remains`）：r032 的 mask 是合法 `900×1600 uint8`，但 histogram=
  `{0:1,440,000}`。这不是 SAM grid32 quality reject：唯一输入少于 semionline `num_voting_frames=3`，upstream flush 没有形成
  cross-view consensus，因此 all-background 正是预先声明的 one-view 边界。它同时推翻“one-view resource PASS 可证明
  identity masks ready”的隐含推断。下一步必须在冻结 grid32 上做 same-grid batch parity，并用至少 3 个按时序排序的
  train-only views 检查 non-empty masks、stable short IDs、repeatability 与资源；在此之前禁止 full materialization、identity
  training 或把 annotation_count=1 当成实例覆盖。证据：r032 mask=`0bf854a1...59d`、audit=`cebe07fd...cd5`、freeze=
  `configs/worldsim_v51/stage_f_f0a_environment_one_view_smoke_freeze_v1.yaml`。r033 在同一 grid32/batch32 上用冻结的
  frame=`0/40/80` 完成 3-frame voting：三张 mask 全 non-empty，19 个 positive short IDs 至少跨 2 帧，且 batch32 repeat
  三 mask/metadata bit-exact，因此“one-view 不能证明 identity input”的前置边界已解除；该证据不读质量，也不解除 r033
  的 batch/resource 失败（`V51-F60/F61`）。
- `V51-F60`（`algorithm/implementation-contract`, `resolved for current method selection by r034; sensitivity boundary remains`）：r033 预注册把同 grid32 的
  `SAM_NUM_POINTS_PER_BATCH=32→16` 视为 execution-memory parity 臂，但 batch16 与 batch32 的三张短 ID mask 和 `pred.json`
  全不 exact。逐帧不同 label pixels=`208,647/288,527/244,696`，exact fraction=`0.855106/0.799634/0.830072`，binary
  foreground IoU=`0.961177/0.995201/0.969622`；batch32 IDs 含 `36/62/95`，batch16 含 `13/63/96`。与此同时
  batch32 primary↔repeat 的三张 mask 和 metadata 均 bit-exact，association/non-empty 也通过，故差异不能归因于无约束随机
  重跑。已确认的推翻项是“batch 只改变显存、不改变输出”；更深机制可能涉及 AMP batch-shape 数值路径与候选/NMS 边界，
  但 r033 没有证明具体源码根因。禁止放宽 exact 门、用高 foreground IoU 冒充 identity parity，或从 batch16/32 中按结果
  挑一臂。合法恢复只允许在新 run 恢复 upstream default batch64、保持 grid32/输入/阈值不变并独立检查 repeatability、
  association 与资源；若 batch64 不可运行，则该 faithful-input 路径必须保持 blocked 而不是继续调 batch。证据：r033 source=
  `191d3e4...12f`、parity=`7a6db15f...7ae`、audit=`a5a7d5c8...fa7d`。r034 恢复 upstream default batch64，保持
  grid32/三帧/其他参数，primary↔repeat 三 mask/metadata bit-exact、association PASS；当前方法选择因此固定 batch64，不再
  依赖被证伪的 batch32 execution-only 解释。本条的 batch-sensitive 事实仍是 active anti-regression boundary：后续任何
  batch 改动都是方法变化，必须新协议且不能用 r034 质量作等价担保。
- `V51-F61`（`engineering/resource/protocol`, `resolved by r034 physical-headroom contract; r033 remains failed`）：r033 三臂串行而非并发，independent audit 从 241 条
  resource samples 重算 NVIDIA peak=`24,116 MiB > 24,000 MiB`，超过预注册门 116 MiB；cgroup peak=
  `17,956,044,800 bytes`、event wall=`78.917s`、monitor errors=`0`。runner 先因 parity fail-closed，故没有执行后置 resource
  adjudication 或发布 `resources.json/summary/manifest`；这不能让已记录的 GPU 越门消失。它推翻“r032 one-view peak
  23,954 MiB 可直接外推到三视图三臂合同”的资源假设，不是 mask quality reject。禁止倒写 r033 为资源 PASS、事后把旧门
  提到 24,116，或把无 OOM 等同有安全余量。新 batch64 smoke 若修改 ceiling，必须在启动前绑定卡总显存、明确保留 headroom
  与复开理由；若 OOM/越新门则停止该 recovery，不得继续缩 batch 回到已证伪的 execution-only 解释。证据：r033 resource=
  `db4e6d17...8e9c`、status=`e027888e...7234`、audit=`22,939 bytes /a5a7d5c8...fa7d`。r034 没有修改 r033 门，
  而是新预注册 card total=`24,576 MiB`、minimum headroom=`256 MiB`、peak ceiling=`24,320 MiB`；upstream batch64 两臂
  实测 peak=`24,092 MiB`、headroom=`484 MiB`、cgroup=`17,957,322,752 bytes`、142 samples/0 errors，全门通过并由
  audit=`e0988f50...5258` 重放。因此当前三视图 upstream-batch64 resource prerequisite resolved；45-view materialization
  仍需独立总时长/磁盘/输出分母门，不能从两臂 smoke 外推。
- `V51-F62`（`engineering/CUDA-runtime/resource-boundary`, `resolved_recovery; root_cause_unproven`）：r035 依冻结顺序串行做三场 45-view train-only
  materialization；0471 已完成 `15 masks +pred.json`，但 1087 official grid32/upstream-batch64/AMP subprocess 处理前两张
  后，在第三张触发 three-frame vote，于 `consensus_associated.py:58 spatial_alignment` 的 `value @ affinity` 返回
  `CUDA CUBLAS_STATUS_INTERNAL_ERROR / cublasGemmStridedBatchedExFix`，因此 1087 没有 canonical mask/pred/report，0379
  未启动。该错误不是显式 PyTorch OOM；resource samples 重放 peak=`24,124/24,576 MiB`、headroom=`452 MiB`、cgroup=
  `17,961,271,296 bytes`、174 samples/0 errors，仍通过 r035 预注册数值门，所以现阶段既不能武断归因 OOM，也不能用
  headroom 数值排除 allocator/CUBLAS workspace/driver 异常。它推翻了“r034 同 batch64 三视图 PASS 可直接外推任意场景
  的 45-view execution stability”，但没有推翻 Gaussian Grouping identity 算法或证明 mask quality 失败。禁止把 0471 的
  `15/45` partial 写成 full materialization、原地续跑/覆盖 r035、跳过 1087、改场景顺序、缩 batch 回到已证明会改变输出
  的配置，或读取 partial quality 再选 recovery。合法下一步只能新预注册 exact 1087 `000_0/000_1/000_2` 三视图，保持
  grid32/batch64/AMP/size480/thresholds 并启用 `CUDA_LAUNCH_BLOCKING=1` 定位是否可重放；diagnostic 输出不得进入质量或
  training。证据：r035 source=`e4d64d3...1424`、status/events/resource-samples=`c3f917bd...f61/7d2221b5...0b7/
  d46d632d...4e2`、stderr=`f626efc6...8a5`、audit=`25,311 bytes /6d217a7e...13e1 /PASS`。r036 在 exact
  1087 首三视图、相同 method 且 `CUDA_LAUNCH_BLOCKING=1` 下串行 fresh-process replay 两次：第一次在相同 GEMM
  位置复现、第二次成功并生成 3 个 schema-valid mask/pred，故 deterministic input/shape 必现假设被推翻；与此同时一成一败
  证明 runtime 还不具备 materialization 所需的 repeatable execution。r036 resource 仍 PASS，成功输出未读质量且不能补写
  r035。下一步先做预注册 runtime health/control-vs-target reproducibility gate；禁止用第二次偶然成功直接重启 45-view。
  r036 evidence=`summary 32e59c85...3ea /audit 5,077 bytes, ec7cfa36...34f6 /freeze
  configs/worldsim_v51/stage_f_f0e_scene1087_cuda_fault_localization_freeze_v1.yaml`。r037 用同卡、同 frozen method、同
  `CUDA_LAUNCH_BLOCKING=1` 做 A–B–A–B：两次 0471 known-good control 均成功、mask/pred 彼此 exact 且与 r034 SHA
  相同；夹在其间的两次 1087 target 均在相同 CUBLAS site 失败。故“整张 GPU/所有三视图已普遍失效”不成立，失败收窄为
  target-path process instability；但 r036 曾有一次 target success，仍不能倒写 deterministic data failure。ECC/page/row 对
  RTX3090 为 N/A，dmesg 又无权限，二者都不能冒充健康证明。合法下一步是 source-neutral trace，在不改 upstream 文件/
  tensor 内容/方法参数的前提下记录 control/target matmul tensor metadata 与 allocator 状态；任何 trace 输出仍不得参与质量或
  training。r037 evidence=`summary 5fd4a4e8...df8 /audit 8,245 bytes, 2fb76f32...d50d /freeze
  configs/worldsim_v51/stage_f_f0f_cuda_runtime_health_reproducibility_freeze_v1.yaml`。r038 source-neutral trace 的 control/
  target 都成功且输出分别 exact 对齐 r034/r036 success；control 两个 matmul 是 `26/36 objects`，target 是 `3/52`，两侧
  affinity 都为 `[1,1620,1620]`，故“target 首个 matmul 更大所以必败”被直接推翻。更关键的 allocator observation 是：
  control pre-matmul driver-free 仅约 `35/55 MiB`、allocator retry=`0`；target success process 已发生一次 allocator retry，
  cache 被释放后 pre-matmul free 约 `18.15/17.23 GiB`。这使 allocator-cache/CUBLAS-workspace state 成为有证据的 active
  hypothesis，但不是根因证明：trace timing 可能扰动执行，且 control 在低 free 下仍成功。合法 recovery 只允许预注册在
  frozen line58 matmul 前执行 `torch.cuda.empty_cache()`，不改 tensor/operator/grid/batch/AMP，并要求 control/target 双 repeat
  对既有 success hashes bit-exact；禁止把 cache observation 写成 OOM 或跳过 parity 直接 full materialization。r038 evidence=
  `summary e9db6152...8f46 /audit 16,025 bytes, a8cbdb5b...4047 /freeze
  configs/worldsim_v51/stage_f_f0g_target_tensor_allocator_instrumentation_freeze_v1.yaml`。r039 在每个 frozen matmul 前调用
  `torch.cuda.empty_cache()`，A–B–A–B 四个 fresh process 全部成功；8 次 intervention 均有 before/after allocator 证据，
  control/target 双 repeat 与 r034、r036/r038 success hashes 全部 exact，且资源门 PASS。因此 empty-cache 是当前合法的
  execution recovery candidate，并推翻“释放 cache 必然改变 identity 输出”的担忧；但三视图 parity 不能外推 1087
  15-view/全 45-view，`V51-F62` 仍 active。下一步只允许单场 1087 15-view recovery，禁止直接把 r039 写成 full
  materialization ready。r039 evidence=`summary d720af4e...9505 /audit 8,625 bytes, fda57ee4...88ab /freeze
  configs/worldsim_v51/stage_f_f0h_pre_matmul_empty_cache_parity_freeze_v1.yaml`。r040 把同一 intervention 扩到 exact
  scene-1087 15-view：单 fresh process 完成 `15 uint8 900×1600 masks +pred.json`，6/6 observed matmul 均有
  empty-cache before/after 证据，resource gate 与独立审计全部 PASS；未读取 mask 内容质量。因此“该 recovery 只对三视图
  probe 有效、扩到 1087 15-view 必然失败”已被推翻，但它仍不能外推 fresh 三场 45-view，`V51-F62` 保持 active。
  下一步只允许预注册新目录、按 `0471→1087→0379` 串行的 45-view recovery；禁止续写 r035、复用 r035 partial、先读
  r040 quality，或直接放行 identity training。r040 evidence=`summary 312a0277...a65 /audit 9,254 bytes,
  1393c664...67c /freeze configs/worldsim_v51/stage_f_f0i_scene1087_15_view_empty_cache_recovery_freeze_v1.yaml`。
  r041 再从 exact r026 manifest 新建三组 scene-local 输入，按 `0471→1087→0379` 三个 fresh process 串行执行；三场
  全部成功，45/45 schema masks、3/3 pred、18/18 pre-matmul empty-cache evidence、output record chain 与资源门均由
  独立审计重放。因此 r035 暴露的 full-materialization execution failure 已在 frozen empty-cache intervention 下解除，
  `V51-F62` 改为 resolved recovery；但 trace timing/allocator-cache/CUBLAS workspace 中哪一个是唯一根因仍未证明，不能
  写成 OOM root cause。该结论也完全不包含 mask quality、actor identity alignment 或 training readiness；下一步必须新预
  注册质量/对齐门，禁止把“45 个文件存在”写成算法有效。r041 evidence=`summary f3ee3ad1...c183 /materialization
  32b5d8d3...1b7f /audit 18,462 bytes, acd5a91b...31d2 /freeze configs/worldsim_v51/stage_f_f0j_fresh_45_view_
  empty_cache_materialization_freeze_v1.yaml`。
- `V51-F63`（`algorithm/instance-quality-alignment`, `rejected`）：faithful Gaussian Grouping 的 45-view materialization 在
  F0l 首次读取 frozen train-only weak support 后，只通过 scene-1087，scene-0471/0379 均失败。0471 的 foreground
  coverage/one-to-one identity recall/persistent-track fraction=`0.122784/0.080747/0`，0379=`0.238278/0.202933/0`，分别远低于
  读像素前冻结的 `0.70/0.35/0.50`；两场 assignment efficiency=`0.937/1.0`，说明问题不是主要由 short-ID collision
  造成，而是 actor support 大量未覆盖且同一 3D track 的 assigned ID 没有跨两个 eligible views 持续。1087 的
  `0.859091/0.505009/0.5` 全门通过不能覆盖 all-three-scene contract。该负结论推翻“只要 full materialization 稳定，
  faithful DEVA short IDs 就足以作为当前 driving identity training 输入”；它不是 CUDA/资源 blocked，也不允许训练后再救。
  限制：DriveStudio dynamic union 只作 foreground weak support，3D projected boxes 只作 track attribution，不是真值 instance
  segmentation；因此结论严格限于当前三场、frame→camera view order 与 adapter，不能外推为 Gaussian Grouping 普遍无效。
  F1/F2/identity training 关闭，下一步按冻结路线进入 Trace3D source/method/immutable-base adapter preflight。证据：r043
  summary=`f13c8094...da8a`、report=`b1e4bb40...95ed`、audit=`4,210 bytes /f478fbd9...4320`、freeze=
  `configs/worldsim_v51/stage_f_f0l_train_only_quality_identity_alignment_freeze_v1.yaml`。
- `V51-F64`（`engineering/tool-availability`, `resolved`）：Trace3D G0 r044 已成功原子发布 official PDF 与 exact
  repo commit/tree，随后仅在读取 PDF 页数时因系统没有 `pdfinfo` executable 抛出 `FileNotFoundError`。异常发生在任何
  repository semantic report、source execution、submodule initialization、model download 或 image/mask/quality read 前；run
  只留下 running status 与 start event，必须保留且不能事后补成 blocked/done。该失败推翻“AutoDL 默认具备 Poppler CLI”
  的工程假设，不涉及 Trace3D 方法可行性。合法恢复只在新 run exact 复核现有 paper=`2,390,825 bytes /d50eda07...47e4`
  与 repo=`7465ad94...c442/tree 22d30d19...a05d/clean`，用标准库 PDF `/Type /Page` marker 计数替代 external tool；不得
  删除或重下已发布资产，也不得借恢复 init submodules/执行源码/读质量。closeout=
  `configs/worldsim_v51/stage_g_g0_trace3d_source_method_preflight_r044_closeout_v1.yaml`。r045 以原资产 exact reuse、标准库
  page-marker count=`11` 完成恢复；独立 audit `053cf574...d3b` PASS，故该工程失败关闭。r044 历史终态不倒写；本次修复不构成
  Trace3D 方法成功或训练授权。
- `V51-F65`（`engineering/algorithm-determinism`, `resolved_method_rejection`）：Trace3D exact unpatched CUDA extension 在 r046 的
  preregistered synthetic class-response gates 上 PASS，但相同 config/input/extension 的独立 fresh-process audit 将 foreground
  alpha weight 从 `0.0267562941` 重放为 `0.0056084292`（absolute diff=`0.0211478649`）；hard class vector 均为 `[0,1]`。
  official `id_trace.cu` 在多个 pixel threads 面向同一 per-Gaussian/class global weight 时使用普通 `+=` 而不是原子累加，
  这是与漂移相容的 source-level hazard，但一次跨进程差异尚不足以把根因写死。不得因 hard argmax 一致就进入真实 U2/B3
  adapter，也不得事后给 r046 增加失败门或直接 patch upstream。合法下一步是在全新预注册 run 中以多个 fresh processes 同时
  冻结 hard/alpha exact determinism；FAIL 时 faithful Trace3D operator rejected 并按路线转 BKI/graph-free，PASS 才允许预注册
  real tensor/camera adapter。r046 capability PASS 与本风险并存，且都不构成质量证据。
  r047 按上述规则执行 8 个 fresh processes、每进程 hard 两次与 alpha 两次；16 个 hard vectors 均为 `[0,1]`，但
  alpha exact vectors 为 `0.0056084292/0.0267562941` 两种，故唯一数 `2 > 1` 并被独立 audit `98c72ba7...d31`
  重算确认。该结果足以拒绝当前 exact unpatched faithful operator，但仍不把普通 `+=` 写成已证明的唯一根因，也不外推
  所有 Trace3D 实现。按预注册 failover 不 patch、不进入 real adapter，路线转 `WS-V51-M1-H-GRAPHFREE-01`。
- `V51-F66`（`algorithm/governance`, `superseded`）：Stage H 的 faithful BKI/graph-free fallback 在 V5.1 内没有启动，
  task status 保持 `pending`、execution=`false`，并由 V5.2 scope 取代；因此它既不是 done，也不是 empirical rejected。
  收口依据是累计而非新增质量读数：r018 的 progressive propagation 只有 ΔBoundary-F1=`+0.0002196`，同时
  ΔIoU=`-0.0714543`、ΔFN=`+0.1694766`；r022 的 simple voxel node 虽提高 observation density，相对 U2/B3 的
  BF1/IoU/FN 仍为 `-0.0002566/-0.0925468/+0.1899473`；r043 的 0471/0379 identity recall 仅
  `0.080747/0.202933` 且 persistence=`0/0`；r047 的 faithful Trace3D alpha 又不能跨 fresh process exact 重放。
  这些事实共同推翻“在不改变 evidence source 的情况下继续替换传播器仍有较高边际收益”的 V5.1 资源分配假设；它们
  支持当前瓶颈为 effective observation structural missing，但不证明 BKI 或所有空间 kernel 普遍无效。禁止在 V5.1
  继续做 BKI source preflight、kernel/threshold 调参，或把 `superseded` 改写成 BKI reject。合法复开只能在 V5.2 先引入
  独立新观测源，并在首次方法质量读取前冻结 coverage、identity persistence、fresh-process reproducibility 与跨场分母；
  证据=`configs/worldsim_v51/m1_closeout_v1.yaml`、`docs/archive/2026-08/worldsim-v51-m1-closeout/README.md`，
  authoring base=`fc07b99`，failure delta=`V51-F66`。

<a id="detail-v5"></a>

## V5 M3 constraint projection 新增防重复结论（2026-08-14）

- `V5-F52`：V4 M3 canonical r238/r335 的 baseline 是 `FRAME_INDEPENDENT`，不是 V5 要求的 `T2_V4_FROZEN_SE3_BSPLINE`。V4 的 `30.41%/34.39%` warp 改善不能直接成为 V5 T2 comparator 证据；V5 必须在同一 fresh clip 上重跑 T2–T5。
- `V5-F53`：M2 rejected 后，M3 REMOVE 不能隐式复用 geometry-safe repair。REMOVE 只保留 exact bypass、semantic reintroduction 与 rollback checks，不进入 trajectory physics denominator；M3 正结果永远不得倒写 M2 成功。
- `V5-F54`：r002 是 config identity blocked，不是 clip inventory 质量失败。缺少 `protocol_audit.conclusion` 在 annotation streaming 前触发 KeyError；无图片/LiDAR blob/quality/GPU 读数。terminal 保留，只以新 r003 修复。
- `V5-F55`：低速位置抖动不能定义 velocity heading，倒车也不是 heading inconsistency。r004 的 T2 38 项违例全部来自旧 heading metric；禁止用 speed>`0.1m/s` 且 forward-only 的结果宣称 projection 有大幅物理改善。当前修正规则是 speed>`1m/s` 且 forward/reverse mismatch 取小者。
- `V5-F56`：POCS 更新量固定点不等于物理可行。r004 T5 的逐帧 heading correction 产生 `20` yaw-rate + `14` heading violations，却曾报告 converged；现在只有剩余 total violations=`0` 才允许 convergence。禁止用旧 converged flag 支持方法。
- `V5-F57`：r005 exact replay 后 T2=`15/16 safe`，仅 `1/16` request 有 `2` 项违例；T5 降到 `1` 项，但预注册最小 evaluable 是 `8`。当前结论必须是 insufficient signal，不能因相对 reduction=`50%` 而解锁 renderer、选择 arm 或进入 validation。
- `V5-F58`：不得通过降低 heading speed floor、取消 reverse、降低物理 caps、删除 T2-safe requests 或改 minimum-evaluable gate 来“修复”r005。复开只能注册新的 desired-motion hypothesis 与新 run，明确保留 result-aware development 身份；collision/render/validation 仍需独立门。
- `V5-F59`：r006 已把 V5 constraint-projection M3 正式收口为 `rejected`。禁止事后扩大 lane shift/acceleration stress template 来人为制造 T2 violations；V4 M3 时序正结果继续成立，但只能写作历史 baseline，不能倒写 V5 constraint projection 成功。未来复开必须是独立新路线与新冻结协议。

## V5 M2 cross-view scaffold 与拒绝收口新增防重复结论（2026-08-14）

- `V5-F46`：repair asset provenance 必须来自已有不可变枚举。r012 使用新字符串 `cross_view_background_depth_scaffold`，在首个 asset、GPU 与方法质量读取前被拒绝；该 terminal 是工程 blocked，不是 G4 质量结果。修复只改为既有 `native_scene_donor` 并新建 r013，不覆盖 r012。
- `V5-F47`：G4 的 Gaussianization 后改善 `17/22` 不能替代 raw 相对门。r013 raw 仅 `12/22`，低于冻结的 `14/22`，且 raw/post absolute-safe 均为 `0/22`；G4 必须保持 rejected。其 direct projection mean/median 仅约 `4.73%/0.78%`，不得靠 Gaussianization 或 fallback 隐藏覆盖不足。
- `V5-F48`：G5 的相对支持不能覆盖绝对几何安全失败。r014 raw/post 改善=`15/22`、`19/22`，mean delta=`-3.270320/-4.023966m`，但 raw/post absolute-safe 仅 `1/22`、`0/22`；正式写法只能是“model proxy 上相对改善，未形成 geometry-safe candidate”，不得写成 M2 成功或跨视角恢复真实背景。
- `V5-F49`：多相机投影覆盖不是独立真值。r014 any/direct/extrapolation/fallback mean=`60.40%/15.57%/47.99%/36.44%`，LiDAR projected mean≈`0.8%`；大量信息来自 bounded extrapolation 与 G0 fallback。禁止把 source 数量、投影覆盖率或 proxy MAE 当作 same-view hidden-background GT confidence。
- `V5-F50`：G5 绝对门失败触发结果前 hard stop。禁止事后搜索 absolute threshold、camera/time source grid、fusion、disagreement、extrapolation radius、Gaussian stride/opacity，亦不得自动解锁神经 surface；任何复开必须是新科研假设、新 task、新冻结协议与独立 evidence source。
- `V5-F51`：r015 已把 `WS-V5-M2-GEOMETRY-FIRST-REPAIR-01` 正式收口为 `rejected`，method/router/validation 均未解锁。后续 M3 是独立任务，任何 M3 正结果不得倒写 M2 成功；`WS-V5-M2-GEOMETRY-FEASIBLE-ROUTER-01` 保持 locked，不能在没有 geometry-safe candidate 时单独运行。

## V5 M2 geometry-first 新增防重复结论（2026-08-14）

- `V5-F34`：同一 view 的 actor-union mask 不是合法的一次修复 request。r002/r003 把多个 actor 合成一个 hole，最大 union 达 `152,410` pixels，导致大洞支配均值；r004 已恢复 `one_actor_one_view_one_hole`，得到 `23=22 accepted+1 rejected` 且 union pixel-exact replay。后续方法选择不得复用 union-mask 分母。
- `V5-F35`：base renderer `background_depth` 是维护一致性 proxy，不是 same-view hidden-background GT。r005 的 reference confidence mean/median 只有 `0.0585/0.0582`，范围 `0–0.1557`；任何 raw/post MAE 必须与 `model_proxy_not_ground_truth` 同表，不得写成真实道路深度恢复。
- `V5-F36`：G0 在逐 actor r005 上 raw absolute fail=`22/22`，raw MAE mean/median=`8.5872/8.7151m`；同时 Gaussianization primary=`16/22`。不得因为 G0 是最简单 arm 或相对复杂 surface 较稳，就把它写成 geometry-safe candidate。
- `V5-F37`：G1 piecewise plane 只有 `5/22` 请求改善 `>=0.5m`，candidate−G0 mean/median=`+3.658565/+2.300282m`。它在当前 model proxy 上正式 rejected；禁止根据局部 side 拟合直觉继续调 piece 分区后复活同一 arm。
- `V5-F38`：G2 MLS 的收益由少数洞驱动。它在一个 `118,022`-pixel actor 上改善 `11.194128m`，但仅 `8/22` 请求达到改善门，mean/median delta=`+3.005506/+1.620793m`。不得引用最大改善或自适应带宽完成度掩盖广泛稳定性失败。
- `V5-F39`：G3 quadratic 的 median delta=`-1.489037m` 不能覆盖 mean=`+0.103693m` 与 improvement count=`11/22<14/22`。r009 已按冻结 gate rejected；不得事后将判据改成 median-only，也不得沿用 request-unit 错误的 r003 作为相反证据。
- `V5-F40`：blocked terminal 必须与方法失败分离。r001 是 unavailable-view denominator 合同错误；r007 是 artifact serializer 局部变量遮蔽导致 `KeyError: 0`。两者没有可用质量 summary，不能进入 arm 均值，也不能覆盖目录或改 terminal；修复只允许新 run ID。
- `V5-F41`：G1/G2/G3 全部 rejected 且 G0 raw `22/22` fail，当前不存在 safe candidate。按照 V5 causal order，feasibility-first router、validation 和神经 surface 都不得解锁；下一步只能先诊断/修复 Gaussianization representation 与 alpha compositing，之后重新经过独立 development gate。
- `V5-F42`：formal run 的目标目录必须由 runner 原子创建，不能为了 stdout 重定向提前 `mkdir`。r010 在任何模型加载/GPU/质量读取前被 overwrite guard 拒绝；其 blocked terminal、events 和 run.log 保留。修复只改变 launcher，把日志放到 run 外部并用新 r011；不得删除 r010 或把它计入方法分母。
- `V5-F43`：提高 Gaussian asset opacity 不能修复当前 representation gap。r011 的 OPAQUE−BASE 在 `0/22` 请求改善 `>=0.1m`，mean/median post-MAE delta=`+0.059686/+0.065773m`；在 dense 条件下继续提高 opacity 也退化 `+0.035533m`。禁止继续提高 opacity、改 alpha 阈值或把 background mixing 写成已支持机制。
- `V5-F44`：stride `2→1` 的 DENSE arm 在 `20/22` 请求改善，mean/median delta=`-0.424179/-0.480927m`，但这是 frozen scene0471 model proxy 上的机制取证，不是 geometry-safe method selection。G0 raw 仍 `22/22` absolute fail，validation/KITTI/独立 GT 均未读取；不得直接把 stride-1 送入 validation、改 router，或把 post-render 改善写成真实道路恢复。
- `V5-F45`：factorial arm 通过数必须解释为因素对比，不能按臂名投票。r011 中 DENSE 与 DENSE_OPAQUE 都过门，正式 summary 因此保守写作 `multiple_gaussianization_factors_have_broad_mechanism_support`；但 OPAQUE 在 sparse/dense 两个条件都退化，描述性 density/opacity main effect=`-0.436256/+0.047609m`。后续只允许冻结 density representation repair 并重新过独立 gate，不能把组合臂通过误写成 opacity 或 interaction 获支持。

## V5 M1 structured unary 新增防重复结论（2026-08-14）

- `V5-F20`：renderer 的逐 pixel intersection 不是独立多视角证据；同一 Gaussian 在同一 view 覆盖更多像素时，若逐行更新 Beta，会把屏幕面积伪装成 view count。V5 必须先按 `Gaussian × view` 聚合 contribution，再以 `1-exp(-mass)` 形成饱和 visibility，B0/B1 每 Gaussian 每 view 最多一票；不得以原始 intersection 行数扩大 evidence denominator。
- `V5-F21`：只改 arm 名称不能形成 Bayesian ablation。若 B0/B1/B3 共享同一 soft signal 与同一 reliability，它们在代数上会退化为同一方法。结果读取前已冻结为 `B0=hard unweighted`、`B1=hard reliability-weighted`、`B3=soft SAM probability × reliability fractional count`；B2 继续延后，禁止在读到 scene0471 指标后改定义。
- `V5-F22`：ownership posterior 不是新的 Gaussian opacity。若直接把 posterior 当 opacity，会给原 base 中近透明的 Gaussian 注入虚假 semantic mass。所有 2D ownership evaluation 必须渲染 `immutable base opacity × ownership probability`，并在运行前后复核 base checkpoint SHA；不得用 posterior-only rasterization 形成虚假 IoU/Boundary F1 改善。
- `V5-F23`：scene0471 的 annotation prompt 集与 checkpoint RigidNodes 表示没有天然一一对应。冻结规则选出 `17` 个轨迹大于 1 m 的非自行车 vehicle（`15 car + 2 construction`），而 formal checkpoint 只有 `15` 个 RigidNodes instances；差异可能来自无足够 LiDAR 的标注 actor。2D SAM union 与 3D base-model proxy 必须分别报告，不得把表示缺口静默算成 unary FN、删除 prompt 或补造 Gaussian。
- `V5-F24`：直接调用 `process_camera/collect_gaussians` 会绕过 `SceneGraphTrainer.forward()` 中的 timeline 设置；若不显式按 `normed_time` 更新 `trainer.cur_frame` 与各 Gaussian model 的 `set_cur_frame`，动态 actor 会被错误地固定在旧帧。V5 sidecar runner 必须复现该状态迁移并做 nearest-timestamp 回归测试；仅图像 frame ID 正确不足以证明动态 Gaussian pose 正确。
- `V5-F25`：scene0471 r037 的 B1/B3 虽同时改善 IoU、Boundary F1、Brier、ECE、NLL 与 FP semantic mass，但 2D FN semantic mass 相对 B0 分别增加 `+0.0915315/+0.0954773`，明显超过计划 validation gate 的 `+0.01` 容忍量。不得只摘录正指标把 r037 写成 M1 成功，也不得以 aggregate calibration 改善掩盖漏检代价；后续 graph 诊断必须逐臂保留 FN、per-view denominator 与 abstain。
- `V5-F26`：r037 是单个 development scene、固定 `0.5` 阈值、`8 accepted + 7 abstained` evaluation views 上的 SAM-proxy 机制诊断；它能推翻“reliability-aware unary 完全无方向信号”，但不能证明 topology graph 必要、不能选择 B1/B3、不能代表 8-scene validation。graph 只能在单独预注册协议后启动，禁止根据 r037 结果补调 unary 参数、读取 validation 或直接扩展 Transformer/semantic split。
- `V5-F27`：r038 的 G3 在 scene0471 2D SAM-proxy 上只带来 `+0.008585/+0.006245` Boundary F1（B1/B3），虽然方向一致且 FN 增量小于 `0.002`，但 Gaussian membership proxy 的 IoU 与 Boundary F1 同时退化。不得只摘录 2D 正指标把 graph 写成已通过，也不得只看 proxy 负指标否定全部图机制；两套口径都必须保留，并在 result-blind development replication 后再决定 formal arm。
- `V5-F28`：r038 的 `cross_proxy_affinity_ratio` 使用 Background/RigidNodes membership 仅做事后 leakage 审计；graph candidate/affinity 明确不消费该字段。G1→G3 从 `0.0083646` 降到 `0.0040198` 证明物理 affinity 更少跨越 base proxy，不等于真实语义边界 GT 或 graph 必要性。禁止把 proxy 反馈进建图、据此调 k/扩散率，或直接解锁 semantic split/validation。
- `V5-F29`：长时 sidecar 不能依赖 SSH stdout 生命周期。scene1087 unary r041 已完成主要计算，却在关闭的输出管道上写日志触发 `BrokenPipeError` 并合法标记 `blocked`；不得把它写成方法失败、覆盖目录或复用编号。正式长任务必须在启动时脱离 SSH 并把 stdout/stderr 重定向到独立日志；同一冻结配置以新编号 r042 完成，r041 继续作为基础设施失败证据。
- `V5-F30`：scene0471 的 `8 accepted + 7 abstain` 不能硬编码成 graph 的全场景分母。r044 在读取 scene1087 的绑定 unary 后被该常量 fail-closed；修复 `d55a067` 改为 summary/diagnostics 双重验证 accepted、abstain、总分母与 B1/B3 `(frame,camera)` 键，并要求 accepted>0。r044 是通用化合同失败，不是数据或 graph 质量失败；修复后只能使用新 run r045。
- `V5-F31`：三场景 frozen SAM 的可用视图为 scene0471/1087/0379=`18/2/6`（各 30），对应 unary 可评估分母=`8+7 / 1+14 / 3+12`。scene1087 的负方向只来自 1 个可评估视图，不能扩大为总体失败；但 result-blind replication 必须保留该稀疏场景和全部 abstain，禁止删场景、补 prompt、补 mask 或只报告可评估视图较多的场景。
- `V5-F32`：G3 三场景复制门正式失败。六个 `scene × unary` 单元只有 `3/6` 个 Boundary F1 为正（门槛 `>=4/6`）；虽然 mean ΔBoundary-F1=`+0.0016107723`、mean ΔFN-mass=`+0.0025676789` 单项通过，但 scene1087 的 G1 cross-proxy affinity 已为 `0`，G3=`1.2800523e-29`，逐场严格下降也失败。不得用正均值覆盖稳定性门、选择 G3、读取 validation 或直接堆 Transformer。semantic split 仍是条件任务，必须先用独立 boundary-residual forensic 证明 boundary ambiguity 是主要残差；当前不自动解锁。
- `V5-F33`：boundary error enrichment 高不等于 boundary 是主要残差。r001 六个单元的 enrichment=`3.83×–280.98×`，但 boundary-primary=`0/6`，mean boundary classification/semantic-error share 只有 `0.402095/0.248353`。尤其 scene0379 虽约 68% threshold error 位于极小边界带，boundary semantic-error mass 仍只有 26%–36%；不得只引用 enrichment 解锁 split。M1B 条件未成立，semantic split/Transformer/validation 继续禁止，M1 structured ownership 收口为 rejected。

## V5 KITTI archive / adapter 新增防重复结论（2026-08-14）

- `V5-F01`：官方 KITTI Tracking calibration 不是统一的 `key: values` 语法。实际 `P0`–`P3` 行带冒号，`R_rect`、`Tr_velo_cam`、`Tr_imu_velo` 行不带冒号；V4 `_read_numeric_table()` 会静默忽略后三类行，真实 adapter 将缺失 rectification/extrinsic。V5 必须同时解析 colon/whitespace 两种格式，并对矩阵 shape、finite、handedness 和投影做 2-sequence smoke；不得把 zip layout ready 写成 calibration gate 已通过。
- `V5-F02`：官方 tracking OXTS 每行是 `30` 个导航/IMU 字段，不是 12-value `3×4` world pose。V4 `_load_pose_matrices()` 会截取前 12 个值并错误解释为位姿，行数与 sensor frame 相等也不能证明 object/world/camera chain 正确。V5 必须按官方语义从 latitude/longitude/altitude/roll/pitch/yaw 构造 pose，并结合 `Tr_imu_velo` 验证坐标链；禁止直接复用 V4 OXTS path 当 pose matrix。
- `V5-F03`：central directory 可读、成员集合对齐和全 archive SHA-256 只证明压缩包可冻结、可进入 staging，不等于真实 adapter 已完成。合法晋级仍需要独立 `.partial` 解压、post-extract member/frame audit、2-sequence 坐标/pose/track-ID smoke 和新 manifest；不得从 archive metadata 直接写 `WS-V5-D1-KITTI-ADAPTER-01=done`。
- `V5-F04`：KITTI Tracking 官方 testing split 没有 `label_02`。testing sequences 可用于无标签 adapter/engineering smoke，但不能进入需要 track/box GT 的 cross-domain 质量主表。V5 10-sequence formal 必须从 21 个 labeled training sequences 中在结果前冻结；不得用 testing split 扩大带 GT denominator。
- `V5-F05`：实际 archive 的 `training/0001` 不是三传感器全帧严格对齐：`image_02/image_03/velodyne=447/447/443`，LiDAR 缺 `000177`–`000180`。这不是 ZIP 损坏或全 KITTI 缺失，但会使“每帧都有 stereo+LiDAR”的 adapter 合同失败。V5 必须在结果前冻结 common-frame/abstain 与 coverage denominator，逐序列记录被排除帧；不得静默 `set` 取交集、补造 LiDAR、删掉 0001 或写成 447/447 完整 multimodal coverage。
- `V5-F06`：V4 M2 geometry risk 使用 `clip(hole_geometry_mae_m / 0.5, 0, 1)`；r219–r221 的 `214` 个 candidates 中 `192` 个饱和为 1，且所有 MAE `>=0.5 m` 的 `192/192` candidates 都相同。`57/130` 个有候选 request 存在“未归一化 rendered MAE 不同、geometry risk 全相同”的碰撞。V5 不得只改 geometry 权重或阈值后宣称解决；任何 mapping 必须保持 tail rank，并报告 saturation ratio、unique risk、rank correlation 与 bad-tail distinguishability。
- `V5-F07`：M2 `+3.3908096237 m` 是保留 abstain 的 policy-level scene-balanced delta，不等于 83 个 accepted repair candidate 相对 TELEA 退化。accepted-only 同请求 router/TELEA=`1.62295/2.01453 m`，而 47 个 risk-abstain 的 atomic no-op/TELEA request mean=`16.58283/2.60817 m`。V4 caveat 不得删除，但后续必须同时报告 accepted geometry、abstain geometry、coverage 和 full-denominator valid yield；不得把两种口径互相替代。
- `V5-F08`：V4 M1 canonical state 已把 observation 压成正/负 count 与乘积 weight，未持久化 per-view observation、投影 boundary distance、Gaussian center/covariance 或 neighborhood/topology disagreement。仅凭 r200 state 不能证明 SAM 错、graph 必然有效或 topology 是唯一根因。M1-D0 必须先生成带 provenance 的 per-Gaussian/per-view diagnostic；缺字段时保持 `running/blocked_evidence_missing`，不能从 aggregate Boundary F1 直接跳到完整 graph 实现。
- `V5-F09`：`WS-V5-M1-D0-BAYES-FORENSICS-01=done` 只表示历史分母已机器重算、缺失字段采集合约已冻结。canonical conclusion=`blocked_evidence_missing_contract_frozen`；不得把 task done 改写成 evidence complete、graph 已验证、M1 rejection 被推翻或可直接训练 full structured ownership。
- `V5-F10`：M2 的 retrospective geometry oracle 只按现有 rendered `hole_geometry_mae_m` 相对排序；其 reference 是 immutable base `Background_depth`，不是 same-view hidden-background GT。即使 `62/83` accepted 与该 oracle 一致，也不能证明 candidate 物理正确。必须先补 `reference_source/confidence` 与 raw→pre-Gaussian→post-render 三段误差，随后才允许在 fresh development 拟合 non-saturating mapping。
- `V5-F11`：P0 freeze-only commit=`dfe7526c7a83ca12d7fa9f6c5a11a29ea7b27b19` 只冻结 scope、historical bindings、missing-evidence schema 与审计器。它不包含 fresh scene selection、模型实现或质量结果；任何后续工作必须通过 P0 formal audit，并继续保持 fresh/test/KITTI quality 未读与 parameter search=false。
- `V5-F12`：fresh 8/8/20 的冻结只使用官方 split、scene context、actor annotation/LiDAR-count metadata proxy 与 sensor-keyframe completeness；没有展开图像/LiDAR blob，也没有读取 reconstruction/edit/M1/M2/M3 quality。20 个 test scene 的身份出现在 freeze manifest 不等于 test quality 已读；`V5_TEST_FREEZE.json` 与 exact-once ledger 形成前，禁止加载其内容或指标。
- `V5-F13`：fresh development cohort 冻结后，8 个 scene 的 processed 为 `0/8`；三前向相机+LiDAR keyframe 的 `0/1280` 只是早期粗审计，不能作为 DriveStudio 10Hz preprocess 的完整分母。核对上游后，真实合同是六相机+`LIDAR_TOP` 的完整 keyframe/sweep 时间链，metadata-only 精确分母为 `14,220` files、当前 `0` present。这是 selective extraction/preprocess 工程前置，不是 M1 质量失败；不得退回 V4 scenes、替换 frozen cohort、提前读 validation，或为省事解压全部约 294 GB blobs。必须一次扫描 metadata、按 member→archive 选择性抽取，并保留逐 scene/sensor/file denominator 与内容哈希。
- `V5-F14`：V4 semantic mask NPZ 实际保留 SAM2 `logits/raw_binary/binary`，因此 V5 不得把最终 binary 当作唯一 confidence，也不得为补 confidence 重新运行或更换 SAM。V5 使用 frozen logit 的 sigmoid 作为 observation probability；quality gate rejected mask 仍保留 raw logit 供诊断，但必须把 positive/negative/reliability 全部置零并显式记录 availability，禁止把拒绝样本误当作背景负证据。
- `V5-F15`：Gaussian 最小 covariance 主轴只能作为 renderer-native surface-normal proxy，不是 LiDAR/mesh ground-truth normal；其符号还具有本征向量二义性。V5 必须用 reference camera 定向、验证 covariance 正定与 available normal 单位范数，并把 `normal_is_ground_truth=false` 固化进 config。若 graph 改善，不能据此宣称已恢复真实表面法线。
- `V5-F16`：DriveStudio 原生 nuScenes preprocess 完成不等于 StreetGS 训练输入已闭合；它生成 images/calibration/LiDAR/object/dynamic masks，但不生成 StreetGS loader 必需的三训练相机 `sky_masks`。V5 首个 profile r003 在任何训练迭代前因 `sky_masks/000_0.png` 缺失合法 `blocked`，summary SHA=`a2802430984ab369143be609088df514e3ed0943563b23ee0a5b3bee02e214f7`。这不是 reconstruction 质量失败，不得覆盖 r003、伪造空 mask 或把 preprocess 8/8 改写成失败；必须先用已冻结本地 SegFormer revision、offline/atomic 协议派生每 scene `frames×3` masks，绑定独立 manifest/SHA，再以新 run ID 重跑 profile。

- `V5-F19`：Python 包内单测通过不等于脚本可从仓库根直接启动。KITTI audit r002 attempt 在读取任何 payload 前因 `ModuleNotFoundError: motion_proj` 失败；import 发生在 runner main 前，因此没有生成 run 目录。不得伪造 r002 terminal、复用该 ID 或把它写成数据/坐标质量失败。修复必须在 package import 前显式加入 project root，并增加 `script.py --help` 直接入口回归测试；提交 `43fe090...` 后以 r003 新 ID 完成真实 smoke。
- `V5-F18`：单场或部分场景的 100-step 成功不能解锁全量 formal，也不能解释成 reconstruction 质量成功。必须保留 8-scene denominator，每场验证 step-100 checkpoint、finite means、summary/status/fingerprint/run-manifest、clean source 与 checkpoint bytes/SHA；formal runner 必须再次读取已提交的 cohort binding。r019–r026 的 `8/8 done` 只证明训练链路与资源门可用，尚未读取 development quality，也不允许跳过 30k base、改用 profile checkpoint 做 structured ownership 结论。
- `V5-F17`：sky mask 文件存在不等于训练输入已合法绑定。V5 必须同时验证 8 个独立 run 的 summary、run manifest、sky-mask manifest、冻结 SegFormer revision、`frames×3` denominator 与全部 PNG bytes/SHA，并把这些 identity 通过新 overlay 绑定到不可变 reconstruction base 配置；不得回写被 r003 引用的 base 配置、只数文件名后开训，或把 segmentation inference 误写成 method inference。r011–r018 已按该协议闭合 `4704/4704`，只解锁 `profile100`，不直接解锁 30k formal 或质量结论。

<a id="detail-v4"></a>

## V4 M3 / 18-scene exact-once 防重复结论（2026-08-13）

- `V4-F44`：r258 因 18 场 sky masks 尚未齐全而 fail-closed，r277 因上游假定 instance timeline 稠密而在 scene-0919 暴露稀疏时间轴合同错误；两者都是资产/兼容性失败，不是模型质量失败。只允许以提交 `d5a4794e` 的稀疏 timeline 兼容修复及 r278 100-step smoke 解锁正式训练，不得覆盖失败 run 或提前读 test quality。
- `V4-F45`：M3 validation r238 的完整 denominator 是 `3 evaluable + 3 abstain = 6`。30.4106% warp L1 改善与 2.6470% temporal LPIPS 改善只来自可评场景；不得删除 abstain、写成 6/6 质量成功，或外推到长时序/非三前向相机。
- `V4-F46`：REMOVE 使用 exact bypass，零时序增益是冻结组合合同的结果；M3 通过依赖预注册的 across-operation temporal gate，不代表每个 operation 都严格改善。不得事后取消 bypass、改 operation 权重或只报告 LATERAL/INSERT。
- `V4-F47`：M2 晋级不消除 geometry 风险。hole geometry MAE 的 signed improvement 为 `-3.3908096237 m`（即误差退化 `+3.3908096237 m`）；18-scene 时序结论无论为 `confirmed`，都不得改写成 repair geometry dominance。
- `V4-F48`：18-scene test 使用 committed freeze 与 exact-once ledger；每场 attempt marker 在任何 test content/quality read 前以 exclusive create 写入，已消费 attempt 禁止重跑。canonical ledger=`/root/autodl-tmp/runs/worldsim_v4/WS-V4-M3-TEMPORAL-DELTA-01/20260813T222011Z__m3-test-exact-once-ledger-s0`，attempt/completion=`18/18`；聚合器只读 run evidence，未重读 test source content。
- `V4-F49`：test 的 abstain 必须留在 18-scene denominator。canonical `20260813T225624Z__m3-test-aggregate18-s0-r335` 为 `12 evaluable + 6 abstain`、conclusion=`confirmed`；不得把 evaluable-only gate 写成全 18 场成功，也不得因 `not_confirmed` 复用同一 test 调参或因 `confirmed` 扩大声明边界。

## V4 M1 rejection / M2 validation 新增防重复结论（2026-08-13）

- `V4-F39`：M1 的 development 正结果不能覆盖 scene-disjoint validation 负结果。validation 只有
  `3/6` scenes 可评，方向支持=`0/6`，Boundary F1/Brier/ECE 均反向；base/checkpoint exact 且没有 validation
  重搜。M1 必须保持 `rejected`，不得继续加 feature、transformer、改 threshold，或把 M2 成功倒写成 M1 成功。
- `V4-F40`：M2 validation 的完整 denominator 是 `6 scenes / 154 requests`。scene-1089/0862/1012 的
  `ABSTAIN_NO_ACTOR` 和 scene-0317 的 24 个 `ABSTAIN_NO_ROLE_MATCHED_ERASE_PACKAGE` 都必须保留；不得只用
  130 个具备 role asset 的请求或 3 个可评场景改写 coverage。canonical coverage 固定为 `83/154=0.5389610390`。
- `V4-F41`：validation 不允许重新选择 baseline、risk weights 或 threshold。matched baseline 必须沿用 development
  冻结的 `TELEA`，router 必须沿用 `uncertainty_forward/threshold=1.0`。即使 validation 上其他 arm 的 composite
  error 更低，也不得事后改 comparator 或路由 operating point。
- `V4-F42`：M2 通过的是预注册的合取门，不是所有 repair 轴支配。相对 TELEA，router 的 global PSNR/SSIM/LPIPS、
  hole PSNR、static LiDAR 和 selective-risk separation 通过，但 hole geometry MAE 从 `2.1435024986 m` 退化到
  `5.5343121223 m`。不得把 `hole_any_endpoint` 通过写成 geometry 改善、真值背景恢复或全面优于 Telea。
- `V4-F43`：selective-risk 成立只表示 frozen uncertainty 排序在当前 validation 请求上有误差分离：abstained
  counterfactual error 比 accepted 高 `0.1241311528`。它不证明 71 个 abstain 已被成功修复，也不允许把 abstain
  从 usable-yield 分母删除。M3 与 18-test 必须继续同时报告 coverage、abstain、blocked 和 worst-case。

## V4 M1 / validation 新增防重复结论（2026-08-12）

- `V4-F34`：同一 scene 的历史 V3.3 train mask 不能自动视为符合 V4 冻结的
  `sample_index mod 5` partition。scene-0230 的 development target 审计发现真实 train/evaluation
  frame overlap，因此必须 `ABSTAIN_LEGACY_SPLIT_LEAK`；不得放宽 split、删除该 scene denominator，
  或把旧 heldout 结果改名为 development。
- `V4-F35`：M1 六场景质量均值只允许在可评 scenes 上计算，但 coverage denominator 必须保持全部六场。
  r124 的 `2 evaluable + 4 abstain` 是协议事实，不得把 2/2 改写成 6/6 成功或静默删除 abstain。
- `V4-F36`：validation 只能复用 development 冻结的 evidence arm、calibrator、mask threshold 与 temporal
  retention；禁止在六个 validation scenes 上再次执行 arm search、calibration fit 或 threshold search。
- `V4-F37`：长时间 archive scan 不能依附会超时断开的 SSH stdout。r128 的 10 个 worker 在扫描约
  58 分钟后因外层 SSH 断管触发 `BrokenPipeError`；这不是数据缺失，也不得覆盖该 run。重试必须使用
  stdin=`/dev/null`、stdout/stderr 文件重定向、parent PID=1 的 detached 进程，并复用已提取的非空文件。
- `V4-F38`：Python 环境必须按 stage 显式区分。validation raw extraction 需要
  `/root/autodl-tmp/envs/motionproj/bin/python`（含 `ijson`）；StreetGS/V3.3 GPU runtime 使用
  `/root/autodl-tmp/envs/drivestudio/bin/python`。r127/r129 分别保留缺依赖与错误解释器路径证据，
  不通过临时安装或删除失败记录掩盖环境错误。


> **历史合并注记（2026-08-12）**：以下 V4 D0/B0 与更早内容在当日从旧账本合入；当前权威元数据、目录和写入合同
> 以上方 2026-08-17 统一入口为准。完整 `RF-01`–`RF-18` 原文见
> [`archive/2026-07/v7-feasibility/RESEARCH_FAILURES_RF01_RF18.md`](archive/2026-07/v7-feasibility/RESEARCH_FAILURES_RF01_RF18.md)。

本文件保留仍约束后续路线的历史结论，并把 H1-11D 的失败严格分为“观察到的事实、合理推断、尚未知、
复开条件”。归档不会使旧失败失效；任何新计划复用旧机制时仍须满足原 RF 的重开条件。

## V4 D0 防重复结论（2026-08-11）

- `V4-F08`：nuScenes metadata 的实际可用入口是 `/root/autodl-pub/nuScenes` 中的官方 archive，不是历史代码里的
  `/root/autodl-tmp/data/nuscenes` 或空的 DriveStudio data 目录。D0 只展开 `v1.0-trainval_meta.tgz`，不得为 cohort
  选择展开 549 GB sensor blobs、下载副本或把空目录写成数据集失败。
- `V4-F09`：nuScenes log filename 不能用未锚定的三组 `NN-NN-NN` 正则取小时。首版把
  `n015-2018-11-14-19-09-14+0800` 的月份 `11` 当成小时，测试将夜间误分为白天。修复必须解析完整
  `YYYY-MM-DD-HH-MM-SS±ZZZZ` 尾部；不得以放宽测试或手工改 scene 标签绕过。
- `V4-F10`：三个既有 processed scene 是 infrastructure anchors，不是结果前随机样本，也不能用其 V3/V3.3 质量选择
  test。D0 只把 0230/0242/0255 固定在 development，并以 metadata-only diversity 补足其余 scenes；后续不得把
  anchor smoke 写成 baseline 或方法质量结论。
- `V4-F11`：只冻结 scene name 列表不足以复验 cohort。D0 config 同时冻结 30 scene 的 actor/edit/clip/frame/sensor
  完整记录，formal builder 逐字段比对并锁 cohort SHA；任何 metadata 更新或构建逻辑变化都必须新建 run，不能静默
  沿用 `eda9f684...44578`。
- `V4-F12`：metadata donor support 是场景级 proxy，不是重建后 donor 的图像/几何质量。D0 可以用它做结果前分层，
  但 B0/M2 不得把 strong/medium/weak 标签直接当作 repair 成功、真实性或 quality ground truth。
- `V4-F13`：确定性 greedy selection 不能对无序 `set` 直接做普通浮点求和。r1/r2 恰好重建一致，r3 在另一
  `PYTHONHASHSEED` 下因同分 score 的末位舍入选择了不同 scene，freeze gate 以 `af36f51a...3447 !=
  eda9f684...44578` 拦截。r4 必须对 tag 排序并用 `math.fsum`，测试须跨多个 hash seed；不得固定环境变量掩盖算法
  非确定性，也不得把 r3 候选倒写成新 cohort。

## V4 D1 防重复结论（2026-08-11）

- `V4-F14`：requested dataset path 与 symlink-resolved physical path 必须同时记录。D1 r1 只写了
  `/autodl-pub/data/KITTI`，容易被误读为审计了不同根；r2 明确 `/root/autodl-pub/KITTI ->
  /autodl-pub/data/KITTI` 且两者均缺失。不得用 `Path.resolve()` 后的单一路径抹掉用户合同。
- `V4-F15`：synthetic 12-gate pass 只证明 adapter schema/坐标/投影/track-ID 检查可执行，不等于真实 KITTI adapter
  smoke。真实目录缺失时任务必须保持 `blocked`，不能将 unit fixture 写成两 sequence evidence 或 10-sequence
  cross-domain 结果。
- `V4-F16`：KITTI 原生合同只有 `image_02/image_03` 两路彩色相机。不得为了匹配 nuScenes 三相机伪造第三视角；
  tracking 缺失时才审计 raw，且 pose/tracklet/calibration 任一门失败都必须 `blocked_dataset_adapter`。

## V4 B0 防重复结论（2026-08-11）

- `V4-F17`：历史 summary/metrics 记录 checkpoint 曾经存在，不等于当前 checkpoint 可执行。B0 首次磁盘审计中，
  scene-0230/0242/0255 的历史 StreetGS 文件均已不在路径；只能保留 bytes/hash provenance，不能把历史
  `exists=true` 或旧质量数值登记为当前 executable。正式 matrix 必须在每次 run 重新检查真实文件。
- `V4-F18`：AD-GS historical aggregate 的 6-scene metrics 只覆盖旧 cohort，且其历史 source/env/checkpoint 路径已缺失。
  与 V4 development 重叠的 0230/0242/0255 也只能记 `historical_metrics_only`；2026-08-12 新恢复的 exact official
  source/env 只解除执行前置，不使历史 metric executable。不得把旧三场景数值拼接新三场景、用 aggregate mean 代替
  scene rows，或把环境 smoke 写成 V4 same-split checkpoint。
- `V4-F19`：baseline inventory 的 `blocked` terminal 是当前资产前置条件未齐，不是 B0 task 的永久 blocked 或方法失败。
  B0 继续 `running`，通过新 run 补齐资产；旧 inventory 不覆盖。只有 V3.3/StreetGS/AD-GS 各 6/6 且统一 evaluator
  完整后才可收口，不能因 M1 实现更有趣而跳过 matched baseline。
- `V4-F20`：DriveStudio preprocess 会把输出根再追加 `_10Hz`，并按零填充 scene index 写目录。r4 的上游命令成功
  不等于 runner 目录合同成功；必须验证真实输出 root、scene dir、`1,176 RGB / 196 LiDAR` 后才能登记 done，不能移动
  或重命名一个未审计路径来掩盖合同错误。
- `V4-F21`：远端网络不可达不能用镜像、floating revision 或未校验模型绕过。sky-model r10 因 `Errno 101` 保留
  blocked；本机只从官方固定 revision URL 获取三文件，传输前后均按 bytes/SHA256 exact 校验，远端恢复经临时目录
  原子发布且 generation 保持 offline。任何 staging 漂移都必须 fail-closed。
- `V4-F22`：preprocess 预建的空 `sky_masks/` 与已有推理产物不是同一状态。r12 因旧 runner 只看目录存在而 blocked；
  修复只允许非 symlink 的空目录，发布前再次 `rmdir`，已有 mask、非目录或 partial 一律拒绝，不能覆盖正式产物。
- `V4-F23`：StreetGS 100-step profile 只证明训练链、checkpoint schema 和单卡资源门可执行。r16 不能计入 6-scene
  30k formal coverage，也不能读取/登记质量或据此宣称 baseline 已完成；每场 formal 必须新建不可变 run。
- `V4-F24`：六场景 StreetGS 的 Gaussian 数、wall 与 peak GPU 差异很大；scene-0255/0048 sampled peak 达
  `24,092/24,000 MiB`，scene-0994 final RigidNodes 仅 `1,029`。不能用一个 scene 的 profile 外推所有资源，不能因
  actor 稀疏补点或删 scene，也不能把无 OOM 的近上限运行倒写成资源失败。主表保留每场分母与工程行。
- `V4-F25`：StreetGS 原生 `test_image_stride=10` 不是冻结的 `sample_index mod 5` 三分区。r17/r20/r22/r24/r26/r28
  即使完成 30k、checkpoint finite 且未主动读 test quality，余数 4 的 heldout 输入仍可能进入训练，因此只能保留为
  protocol-mismatch provenance；r29 的 `StreetGS=6` 被 corrected inventory r33 明确推翻。不得用“训练成功”替代
  matched-contract 合规，也不得覆盖旧 run 来修正历史。
- `V4-F26`：不读取 test quality 还不够，训练进程也必须在 I/O 层隔离 development/heldout。AD-GS adapter 正式训练
  只物化 `train` 的 354 张图；兼容补丁增加 `--disable_test_evaluation`，避免上游在 final iteration 自动将 test
  iterations 加入评测。审计/统一 evaluator 可以显式物化三分区，但训练 runner 不得复用该全量目录。
- `V4-F27`：source checkout、权重和 Python 包必须分别固定 commit/bytes/SHA，环境恢复成功不等于 baseline scene
  executable。r34 从冻结本地环境离线复制，编译 `simple_knn` 与 `diff_gaussian_rasterization` 并通过真实 CUDA
  forward/backward smoke；在 strict preprocess + checkpoint 完成前，AD-GS coverage 仍为 `0/6`。
- `V4-F28`：传输/命令包装失败与模型失败必须分开。CoTracker/plyfile 的首次远端校验受 shell quoting 影响，DPT 下载
  曾出现两个进程指向同一 partial，发布后的附带 `stat` 也曾因 quoting 失败；这些尝试均未被登记为 canonical。
  只有停止冲突进程、临时路径原子发布并对最终 bytes/SHA 做独立复验后才可使用，且不得把 wrapper failure 写成
  权重、CUDA 或算法失败。
- `V4-F29`：preprocess 失败后的目标目录不是可静默复用的 canonical。r35 因启动包装器预建 run 目录而被不可变门拒绝；
  r36 因环境构建留下的未跟踪目录被 source-audit 拒绝；r37 完成 adapter/depth/segment 后因可选诊断依赖
  `flow_vis` 缺失而在 flow 启动时 blocked。三者均未训练或读取 dev/heldout；r37 partial 移入
  `work/codex-backups/2026-08-12-adgs-r37-partial-scene0230`，不覆盖、不伪装 resume。正式 flow 通过显式
  no-visualization 合同移除诊断视频依赖，CUDA extension 后续从 run-local source copy 构建，避免再次污染 official checkout。
- `V4-F30`：修掉 preprocess 可视化依赖不等于训练 import graph 已解除同名依赖。r39 尚未进入 iteration，
  `loss_utils -> flow_utils` 就因全局 `flow_vis` import blocked；正确修复是只在 TensorBoard flow 图真正调用时 lazy import，
  并把 `utils/flow_utils.py` 纳入 exact compatibility patch。不得安装非必要诊断包来掩盖正式无评测训练合同。
- `V4-F31`：Python import 成功不证明 CUDA extension 包含当前 GPU kernel。r41 能加载 PyTorch3D 0.7.5，
  但 inherited `_C.so` 在 RTX3090 KNN 首次执行时报 `no kernel image`；必须从 clean frozen source 在 run-local
  目录以 `TORCH_CUDA_ARCH_LIST=8.6` 重编，并在环境 smoke 中真实调用 `knn_points`。r42 完成该合同后 r43 才进入
  100/100 iteration；r39/r41 继续保留 blocked，不倒写为成功。
- `V4-F32`：进度条达到 60k 或路径上存在 checkpoint 不等于 scene executable。r44 只有在 formal step、run 内
  `point_cloud/deform/env` 三文件 bytes/SHA、fingerprint/manifest、source HEAD、六修改文件与兼容补丁全部精确后，
  才由 r45 从 `AD-GS 0/6` 更新为 `1/6`；StreetGS 同样从“存在即计数”收紧为 runtime+bytes+SHA 精确。
  `AD-GS-2026-07-27.patch` 是 zero-context patch，reverse-check 必须显式传 `--unidiff-zero`，否则会产生审计假阴性。
- `V4-F33`：单个方法达到 6/6 不等于 B0 完成。StreetGS r32/r46/r48/r50/r52/r54 已按 strict mod5、
  checkpoint bytes/SHA 与 clean r55 inventory 收口为 6/6，但 V3.3/AD-GS 仍各为 1/6，统一 evaluator 也尚未生成
  完整 scene rows；不得据此启动 M1、读取 test quality，或把 inventory 的 `matched_baseline_assets_incomplete`
  倒写成 StreetGS 失败。后续只补缺失方法/场景并保留旧 inventory。

## V4 P0 防重复结论（2026-08-11）

- `V4-F01`：计划草案记录的 HEAD 不是执行时事实。草案写 `main@144ed19`，P0 实查为 `main@2108430`，且 V3.3
  收口 `e6663e1` 已进入其历史。V4 必须从真实 `main` 建分支，不得回退旧 HEAD 或把草案 provenance 写成 canonical。
- `V4-F02`：计划写“`/root/autodl-pub/KITTI` 已在公共盘”不等于当前机器已挂载。P0 实查目录不存在，状态固定为
  `blocked_local_dataset_missing`。不得创建空目录、下载 KITTI、借其他 layout 冒充，或把外部缺盘写成 adapter/算法失败。
- `V4-F03`：H0 在计划表中写了 `conditional`，但同一计划规定任务状态只允许
  `pending/running/blocked/done/rejected`。V4 注册表将 H0 规范化为 `pending`，条件授权单独记录；不得引入第六种状态。
- `V4-F04`：一手论文、项目页或官方源码存在不等于 baseline 已在本机 single RTX 3090 + same split 执行。SplatAD、
  IDSplat、HorizonForge、RecEdit-Drive 等必须分开记录 paper/source/executable 状态；没有 matched run 不填数值。
- `V4-F05`：KITTI 缺失不阻塞 D0 nuScenes cohort，但会阻塞 single-card closure 的 KITTI adapter smoke。不得因此提前
  读取 nuScenes test、跳过跨数据集条件，或用多卡/新下载掩盖外部前提。
- `V4-F06`：V4 的公式必须同时对应 config、代码、ablation 和可计算指标。P0 只冻结 schema，不代表 M1/M2/M3
  已实现或有效；后续失败必须按预注册早停，不得继续堆 evidence feature、diffusion 或数学包装。
- `V4-F07`：formal run 通过不代表可跳过提交前 whitespace gate。P0 r1 使用的方法合同正确，但未跟踪计划的参考文献
  含 Markdown 行尾空格，`git diff --check` 拒绝提交；规范引用格式后 plan/config SHA 改变。r1 保留 noncanonical
  done，r2 对最终字节重新审计并成为 canonical；不得倒写或覆盖 r1。

<a id="detail-v3"></a>

## V3.3 R0 防重复结论（2026-08-11）

- `V33-F41`：R0 不能把 JSON 中“语义相近”的类型或枚举视为相同。diagnostic 前三次分别把 S2 的空列表写成
  数值 0、把 S3 `heldout` 写成 `heldout_confirmation`、把 S4 `real_renderer_evaluation` 写成
  `evaluation`，均 fail-closed。以后 verifier 必须比较原始类型/枚举；不得用字符串归一化掩盖 schema 漂移。
- `V33-F42`：正式 instance-field validator 通过不代表 NPZ 必须有未约定的 `schema_version`。r4 在报告层
  冗余读取该字段而 failed；修复只移除报告假设，仍执行完整 validator。以后不能把“自己希望存在的字段”变成
  canonical 资产失败，也不能因此跳过正式 schema 校验。
- `V33-F43`：RoadPatch 成为 V3.3 主方法不等于已在 matched 协议下胜过 V3.2 Telea。两者 base、空间语义与
  评测协议不同；R0 答案固定为 `not_directly_ranked`。不得用 B1 相对 B0 的 heldout gate 写成 Telea head-to-head，
  也不得因缺直接排名否定 RoadPatch 的 3D-native/provenance/heldout 成立结论。
- `V33-F44`：内容寻址 release 不允许为“完整”而复制 579 MB base checkpoint。R0 package 含 O1 sidecar、
  RoadPatch/A4/S4 delta、production renders 与 external reference，forbidden model suffix count=`0`；离线 verifier
  同时锁 file set/bytes/SHA。任何新增 `.pth/.pt/.ckpt/.safetensors` 或未登记文件都必须拒绝。
- `V33-F45`：deterministic archive 不能包含当前 run timestamp、绝对输出路径或可变 ZIP metadata。R0 release
  ledger 只引用固定 canonical 输入，ZIP entry 排序/1980 timestamp/permission/compression 固定；diagnostic、
  formal 和同 run replay SHA 均为 `cffaad16...44a7`。以后新增 release 字段必须先证明跨 run byte-exact。
- `V33-F46`：R0 的 `v33_supported` 只覆盖 scene-0230 主链、冻结确认视图和单 RTX 3090。它不证明
  scene-0242/0255 完整 V3.3 transfer、生成 actor GT、相邻视频时序、闭环安全或传感器真实性。F0 LiDAR-EVS
  仍是 conditional 新任务，不能倒写进 R0 的 4/4 success criteria。

## V3.3 S5 防重复结论（2026-08-11）

- `V33-F35`：unconstrained Harmonizer 不能因“只做视觉润色”进入删除生产链。canonical r4 的 edit target
  delete candidate 让冻结 SAM2 semantic mass/fraction 增加 `+0.126399/+0.133885`；production raw fallback
  两项均为 `0`。以后不得关闭 detector、改用候选图作 delete，或只展示另外四个未触发视图。
- `V33-F36`：跨视图平均改善不能掩盖单视图确认失败。五视图 contact 平均为改善，但 heldout f060/c1 的
  contact L1 delta=`+0.422686`，超过冻结逐视图上限 `+0.25`；因此 G1 必须 rejected、production=G0。
  不得改成 aggregate gate、放宽上限、删除该视图或用 edit target 的强改善抵消它。
- `V33-F37`：heldout confirmation 不是第二个开发集。S5 先只用 f091/c1、f005/c0、f065/c1 选择 G1，随后才
  读取 f020/c0、f060/c1；看到 F36 后不得调 contact/shadow 区域、权重、cap 或阈值并继续在同一 heldout 上
  宣称泛化。合法复开需要新假设、新 task 和未读 confirmation 数据；R0 只登记当前负结果。
- `V33-F38`：冻结推理环境不应为共享模块的无关顶层依赖而污染。r1 的 Harmonizer 已完成，SAM2 启动时因
  `semantic_gate.py` 顶层 SciPy import failed；SAM2 只消费 semantic mass/decision，并不构建 gate。修复是将
  SciPy 限在 builder 内 lazy import，不是往冻结 SAM2 环境临时安装包；r1 terminal 保持 failed。
- `V33-F39`：R3D2 official code、Apache license 与 clean commit 不等于存在作者 pretrained pipeline。
  canonical S5 只登记 `blocked_pretrained_model_unavailable`，`model_loaded=false/training=false`。不得拿
  SD-Turbo/TAESD base、Harmonizer 或自训权重冒充 R3D2-fast。
- `V33-F40`：S5 的五个冻结视图不是相邻视频帧，不能从 deterministic image SHA、跨 run exact 或五视图
  quality 推出 temporal consistency。canonical 明确 `not_evaluated_non_temporal_frozen_five_view_protocol`；
  时序 claim 需要独立连续视频协议、新指标与新 run。

## V3.3 S4 防重复结论（2026-08-11）

- `V33-F30`：S1 `hard_instance_id` 是候选身份集合，不等于所有 Background 候选都应被硬 ERASE。r2 将
  high actor 的 `36,736 Background + 4,525 Rigid` 全部设为零，虽然 target coverage=`0.999741`，但目标外
  L1=`0.821965>0.5`，方法按冻结门 rejected。不得通过放宽 L1 门、扩大 target mask 或只报告 coverage 复活该臂。
  合法修复是使用 S1 已训练 instance opacity 的 MAP 正类 `p>=0.5` 选择 Background，同时保留全部物理 Rigid core；
  r8 以 `1,614+4,525` 行将目标外 L1 降到 `0.225349`，门与视角未变。
- `V33-F31`：S2 canonical `roadpatch_delta.npz` 的 104 行同时服务 high/boundary 两个 actor，不是任一单 edit
  都应加载的整体。S4 high package 必须按冻结 `target_role=high_support` 只取 25 行，并保留 parent delta SHA；
  把 boundary 的 79 行混入 high edit 会产生无关场景变化，不能用“同属 Background repair”掩盖。
- `V33-F32`：恢复同一批 Python/Parameter 对象是必要条件，但不足以证明可回滚。renderer 可能持有缓存或顺序状态；
  每个 stack 卸载后必须重新执行 source render 并比较 tensor schema+bytes SHA。canonical r8 为 5 视角×4 stack=
  `20/20` exact，另有 full deterministic replay 和 replay rollback；以后不能只比较 checkpoint SHA 或 object id。
- `V33-F33`：immutable base + delta package 不能把 579 MB checkpoint 复制到 base 目录后仍称“小型 delta”。
  canonical package 的 base 只含 checkpoint/registry reference descriptor，完整 checkpoint copy=`0`，最大 payload
  `3,942,422 bytes`；任何 materialized deployment checkpoint 必须作为独立 deployment factor，不得改写 authoring state。
- `V33-F34`：S4 r3/r4 的方法和指标有效，但最终核心增加 fail-closed policy validator 后其 source snapshot 不再是
  提交态；r5/r6 与最终方法一致，但 builder 后续补齐异常 terminal。最终只认完整重跑 r7/r8 为 canonical；不得以
  “只是校验/失败处理代码”绕过源码 byte-exact 合同。

## V3.3 S3 防重复结论（2026-08-11）

- `V33-F22`：DriveStudio `instances_info.obj_to_world` 在冻结数据中可直接是 list，也可能是历史 JSON string。
  selector r0 只接受 string，在首个候选 fail-closed，未发布 selection。修复必须对两种 schema 都验证 4×4 shape
  与 finite，不得无校验 `json.loads` 或续写 r0；r1/r2 已以新 run 收口。
- `V33-F23`：view selector 的确定性不能用复制 manifest 冒充。high r1→r2、boundary r7→r8 都重新执行全部
  D2 original/delete 候选渲染，formal 传入 diagnostic 的 expected selection/input SHA 并得到 byte-exact；候选池
  必须继续排除 heldout 与 reserved development，不能让评估帧参与 view selection。
- `V33-F24`：更多输入视图不自动等于更好 actor。high development 中 A4 被选择是因为冻结 metric order 下
  IoU/boundary 改善且 LPIPS/PSNR/背景漂移/横移碎裂 retention 全过；heldout 只允许 A0 与 frozen A4 一次确认。
  不得从 heldout 重选 A1/A2/A4，或把生成背面写成 GT accuracy。
- `V33-F25`：V3.2 manual A0 只绑定 high-support token `af663...c5c29`，不是 boundary actor 基线。boundary
  评估必须使用同 identity 的 immutable D2 native actor，不能复用 high A0、换 actor 或仅凭 class 相同跳过三元核对。
- `V33-F26`：Asset Harvester 对 boundary A4 成功生成 PLY/NPZ 不等于 production override 可接受。r12 中 A4
  相对 native 的 IoU/boundary F1 从 `0.666562/0.555343` 降到 `0.624832/0.492141`，LPIPS/PSNR 也失败；
  决策为 `ABSTAIN_GENERATED_OVERRIDE`，未读 boundary heldout。不得通过放宽门、读取 heldout 或只展示 orbit
  render 复活该资产。
- `V33-F27`：CLI 编排提供错误 inference manifest SHA 时，importer 必须在物化前 fail-closed。boundary r10
  因传错 SHA 保留失败证据；正确 SHA 只用于新 r11。不得删除 r10 后续写、绕过 SHA，或把编排错误写成模型失败。
- `V33-F28`：canonical eval 的 source snapshot 必须等于最终提交源码。high r5/r6 的指标与 r13/r14 完全相同，
  但 evaluator 后来增加 native baseline 支持，旧快照不再等于提交态；因此只把 r13/r14 作为 canonical。不得用
  “逻辑没变”绕过 byte-exact 合同。
- `V33-F29`：scene-0242/0255 有旧 V3 checkpoint 不等于存在本任务冻结的 V3.3 S1/S2 mask/actor 输入链。
  在 boundary transfer 已拒绝的情况下更不能混用旧资产补齐跨场景表。合法复开需要新 task、每场景 exact identity/
  mask/checkpoint 协议、冻结 high actor policy 和新 run；旧 S3 terminal 不续写。

## V3.3 S2 防重复结论（2026-08-11）

- `V33-F15`：正式 run 目录不能在 runner 注册前由 `nohup ... > run.log` 预创建。r0 因 shell 先创建
  `run.log` 而触发 non-empty run-directory fail-closed；不得删除日志后续写。正式托管应把 launcher 日志写到
  run 目录外，或由 runner 创建目录后再写；r10/r11 使用新 run 收口。
- `V33-F16`：V3.1 P3 package 的空间网格是 `(x,y)` 且绑定 V3.2 P2 FP16 mixed checkpoint，不是当前 D2 FP32
  原生 Background 的道路索引。DriveStudio 首个 CAM_FRONT 是 OpenCV `x-right/y-down/z-forward`，道路 BEV
  必须使用 `(x,z)`；r1 因错误要求 P3 manifest 绑定 D2 exact SHA 而失败。不得为复用旧索引而放宽 checkpoint hash。
- `V33-F17`：相机内参在当前 DriveStudio 输入中是 9 个值（`fx/fy/cx/cy + 5 distortion`），不是只含 4 个值；
  r2 在正式物化前 fail closed。所有 adapter 必须显式接受已冻结 schema、校验前四项和 distortion 长度，不能静默切片
  后假称已适配其他相机模型。
- `V33-F18`：对整格直接取 `max_scale/max_plane_residual` 会被一个天空、立面或跨层 Gaussian 污染。r3 得到
  `53,541` patches 但 `0` valid；这不证明场景没有道路 donor。修复应先逐行排除 actor/generated/低 support/
  scale outlier，再确定性选择 `<=0.75 m` 的 densest vertical slab，最后做 plane/normal 门。r10 由此得到
  `822` valid patches；不得回到 whole-cell 放宽阈值。
- `V33-F19`：cross-view sidecar 的 `visible_view_count` 与 front-camera frustum observation 是两个合同。
  r4/r5 已有 valid 4 m patch，但把 `minimum_multi_camera_count=2` 当作当前六相机逐相机观测计数，导致两个真实
  target 的 top-5 手工门失败；当前实现明确要求 sidecar `visible_view_count>=5` 且至少一个 front-camera frustum
  observation，不虚构不可得的逐相机 visibility。
- `V33-F20`：donor 几何合格不等于新增 Gaussian 数量可以无限。r8 的 2,150-row dense delta 在 development
  selection 中可见，但 heldout PSNR/SSIM 退化 `-0.8553 dB/-0.00619`，保持 rejected。修复不是从 heldout 选
  top-K，而是在候选资格阶段冻结 `maximum_rows_per_target=512`，让搜索选择最小可见 delta；r11 最终为
  `25+79=104` rows，并通过全部 heldout 门。不得复活 r8 或事后改写其 terminal。
- `V33-F21`：官方 Inpaint360GS source clean、Apache-2.0 不等于当前 StreetGS/3090 上已复现。官方声明
  RTX 4090/CUDA 11.8，并需要主环境、独立 LaMa 环境、CropFormer/Big-LaMa/SAM/DeAOT/GroundingDINO 权重；
  官方代码没有 DriveStudio/StreetGS checkpoint adapter。r12 因这些前置条件 fail-closed 为
  `blocked_single_3090`、`official_execution_attempted=false`。该状态不是 B2 质量负结论，也不得用 Telea、base
  SAM 或自写 RoadPatch 输出冒充官方 Inpaint360GS。

## V3.3 S1 防重复结论（2026-08-11）

- `V33-F07`：磁盘可用空间在 P0 后被外部流程扩大，同时 V3.2 SAM2 checkout/weight/runtime 被删除；不能把
  canonical train masks 仍存在误判为 heldout 推理环境仍可用。普通 clone 又被大型 demo checkout 拖住，已终止本任务
  PID 并保留 `sam2.incomplete-20260811T0138`。恢复只能 sparse checkout exact commit、下载 exact weight，另建
  隔离环境并冻结 package list；不得修改 DriveStudio 环境或复用不完整 checkout。
- `V33-F08`：heldout-target r2 在旧 `/root/autodl-tmp/envs/worldsim-v32-sam/bin/python` 不存在时 exit=`127`；
  prompts 虽生成但 run terminal=`failed`，不得续写。新环境必须复原 V3.2 记录的 Python/torch/torchvision 版本并
  新建 run；r4 已按此收口。
- `V33-F09`：SAM2 singleton predictor 在当前 exact runtime 返回 `[object,1,H,W]`，而旧兼容路径也可能给
  `[object,H,W]`。r3 无条件 `unsqueeze` 产生 5D interpolation 错误；修复必须显式接受 rank 3/4、拒绝其他 rank，
  并新建 r4。r3 未发布正式 mask，不得当质量证据。
- `V33-F10`：更宽的 O3 ambiguous reassignment 并不自动改善边界。100-step development smoke 中 O3 的
  boundary F1/IoU=`0.123499/0.160504`，低于 O1 的 `0.149382/0.181752`，且 FP 更高；O3 已排除。除非提出
  新的几何邻域/身份证据并使用新 task/run，否则不得因“候选更多”重开。
- `V33-F11`：O1 在 heldout 显著改善 boundary/IoU/NBD/FP，但 FN mass 从 `0.061278` 增到 `0.109356`，
  identity presence 仍为 `0.972973`。因此只能声明对象边界与 false-positive 抑制突破，不能声明全面支配或完整
  召回；S2 delete mask 必须继续报告 FN/残留语义，不能用 O1 的高 precision 掩盖漏删。
- `V33-F12`：`np.savez_compressed` 默认把当前时间写入 ZIP entry header；即使 r6/r7 的 O0 全部数组 exact，文件
  SHA 也会漂移。r6 因此保留为 done noncanonical。r7 writer 固定 entry 排序、1980 ZIP timestamp、权限与压缩
  参数，并用同一 field 二次写入 byte-exact 测试锁定容器确定性。该合同不等于宣称 CUDA 训练位级确定性：
  r6→r7 O1 最大 logit/opacity 漂移为 `0.001357 / 8.918e-05`，但 heldout aggregate exact。
- `V33-F13`：正式 run 的 source snapshot 必须与最终提交源码 byte exact，纯 EOF 空白也不能例外。r7 方法与门禁
  均通过，但提交前 `git diff --check` 清理了 4 个新文件的多余 EOF 空行，导致其中 2 个冻结快照不再与待提交源码
  exact；r7 因此降为 done noncanonical，不能仅凭“空白不影响算法”继续引用为 canonical。
- `V33-F14`：长 GPU 任务不能把前台 SSH 生命周期当作任务托管。r8 已完成模型与 finalizer，但 124 秒调用超时关闭
  stdout，外层 `tee` 收到 SIGPIPE，terminal 按预注册 trap 写成 `failed / exit 141`；不得把已有 summary 反推成 done。
  r9 改用 `nohup` 后台托管并以只读 SSH 轮询，最终正常 `done`、GPU 释放、9 个 source snapshots 全 exact。

## V3.3 P0 防重复结论（2026-08-11）

- `V33-F01`：官方 SAM3.1 source 可 checkout 不等于 checkpoint 可执行。当前代码固定为 `96914d2`，但
  `hf auth whoami` 为未登录且 cache 无 SAM3.1；不得绕过 gated access、猜权重 revision/hash，或让该门阻塞
  dual-opacity 主假设。S1 必须 exact fallback 到 V3.2 SAM2.1 canonical masks；未来解锁需新 task/protocol/run。
- `V33-F02`：论文写 code available、GitHub 仓库存在或项目页可访问，不等于存在 runnable implementation。
  GS-RoadPatching `468f812` 只有 HTML/CSS/JS/图片、无算法源码和根 LICENSE；OP2GS、3D-GIMP、FocusGS、
  LiDAR-EVS 也没有可固定官方 runnable source。后续只能称 `*-inspired` 或 `audit_only`，不得写 reproduction。
- `V33-F03`：R3D2 `3fc6e31` 已公开 Apache-2.0 训练/export/eval 代码，但仓库只声明下载 `sd-turbo/taesd`
  base，没有作者训练并导出的 R3D2 pipeline。单卡从零训练不是“补齐 inference”，且被计划禁止；S5 保持
  `weights_blocked`，不能拿 base diffusion 输出冒充 R3D2。
- `V33-F04`：GOR-IS source release 不消除许可与运行合同。根许可证只允许 non-commercial research/evaluation，
  torch/CUDA 未 pin，且要求 nvdiffrast、CUDA rasterizer 和 OptiX gtracer；没有 pretrained manifest。它只能作
  optional audit，不能抢占 RoadPatch 主线或被写成单卡已验证 baseline。
- `V33-F05`：Inpaint360GS 的官方 source/Apache-2.0 只支持进入 adapter/preflight。上游验证环境是 RTX 4090 /
  CUDA 11.8，并依赖外部 CropFormer/LaMa 权重；在 StreetGS split、相机、分辨率和输入 schema 冻结前，不得
  安装/训练。若 24 GiB 下必须静默降正式分辨率、改 heldout 或改相机数，必须 `blocked_single_3090`。
- `V33-F06`：P0 重新 hash 的 D2/S2/S3/mixed/chunk 五资产 exact 与 V3.2 `36 passed` 只证明 immutable baseline
  仍成立，不证明 V3.3 方法有效。P0 全程无训练/模型推理；S1 必须另建 protocol、run 和指标证据。

## V3.2 终局处置与复开门禁（2026-08-11）

V3.2 已以 `WS-V32-R0-INTEGRATION-01=done`、整体 `none_plan_complete` 收口。归档位于
[`archive/2026-08/worldsim-v3.2/`](archive/2026-08/worldsim-v3.2/README.md)。下面的 `V3-F34`–`V3-F46`
继续约束任何后续路线，但不构成继续执行 V3.2 的任务清单。

| 分支 | 终局处置 | 禁止的延续方式 | 合法复开条件 |
|---|---|---|---|
| S1 semantic lift | `done`，canonical r6 | 复用 identity-invalid r5；绕过 ID/token/rigid 三元核对 | 新数据或新语义假设；新 task/protocol/run；继续 fail-closed identity |
| S2 background inpaint | `done`，canonical r3 | 把 Telea unseen RGB 当作 geometry/GT；复活退化的 r2 | 独立深度或多视图证据；预注册未观测 3D 门；新 run |
| S3 actor harvest | `done`，canonical r3 | 把生成背面写成 GT；倒写 CUDA preflight 失败的 r2 | 新 actor/方法假设；固定 source/weight/license；新 task/run |
| S4 harmonizer | task `done`，non-temporal `excluded diagnostic`；temporal `blocked` | 仅凭锐化或全图指标把删除区重生车辆纳入生产链；绕过 gated 权重 | 合法取得 gated base；显式 semantic preservation + temporal gate；新 task/protocol/run |
| S5 multiview upper bound | `blocked`，未授权 | 猜测许可证、移植无根许可证代码或把未执行写成质量结论 | 明确可执行许可证与权重；独立资源审计；用户重新授权的新 task |
| R0 integration | `done`，canonical r4 | 从 exact package 外推 streaming、跨场景、闭环安全或 GT correctness | 为对应 claim 增加独立数据、协议、测量与 run；不得续写 r4 |

统一复开规则：外部门禁解除只改变“是否可提出新任务”，不会把 S4 temporal 或 S5 自动变成当前任务。任何复开都
必须引用相关失败 ID，使用新 task ID、新冻结 protocol 和不可复用 run ID；旧 `blocked/rejected/done` terminal
保持不可改写。当前 `next_action=none_plan_complete`。

## V3.2 防重复结论（2026-08-10）

- `V3-F34`：actor role 必须同时绑定 dataset instance ID、`instances_info.id`/instance token 与 checkpoint
  rigid model index。只分别验证 class、token registry 和 core count 会允许“2D mask 属于 actor A、D2 core 属于 actor B”
  的静默错配。所有 prompt、semantic lift、asset generation 在运行前必须 fail-closed 核对三元 identity；旧 r5
  因 ID `5`/token `af663…` 错配而失效，不得通过后续 adapter 或人工选图补救。

- `V3-F35`：AutoDL 根 `.condarc` 可把 `nvidia` channel 重写到缺包镜像；第三方官方 setup 的 CUDA channel
  不能只凭 channel 名复现。Asset Harvester 必须使用 `--override-channels` 和明确的官方 NVIDIA/defaults URL，
  同时记录 setup 日志；TUNA 对应 404 不能被误判成官方包不存在。
- `V3-F36`：复制第三方 setup 脚本到 `/tmp` 后，基于脚本位置计算的 `REPO_DIR` 会静默变成 `/tmp`。
  transport-only patch 必须冻结真实 checkout 绝对路径，并让恢复 wrapper 显式接收 formal `RUN_DIR`；不得把
  环境完成结果写入已拒绝的旧 run。
- `V3-F37`：`gsplat` 的浅克隆或 transport-only 复制不会自动带上 GLM submodule。Asset Harvester
  环境不能只以 `pip install` 成功为准；必须固定 `gsplat` commit、初始化 GLM 到 exact commit，
  并在当前 GPU 上 import CUDA extension。
- `V3-F38`：第三方 setup 子进程里的 conda activation 不会传回父 wrapper。恢复脚本后续记录、
  校验或 formal 推理必须使用明确的环境 Python 绝对路径，不得依赖裸 `python`
  或父 shell 的隐式 PATH。
- `V3-F39`：PyTorch 2.10 下 `torch.cuda.manual_seed_all` 不会立即建立 CUDA context；在此前调用
  `reset_peak_memory_stats(0)` 可以在模型加载前报 `Invalid device argument`。资源监控 runner 必须先
  `set_device` + `cuda.init`，再清零峰值计数器。S3 r2 因此在 GPU peak=`0 MiB`、无部分输出时
  `rejected`；修复后必须新建 formal run，不得改写 r2。
- `V3-F40`：边界目标的时间邻近帧不等于存在静态世界几何重叠。S2 r1 的固定支持集合跨过 frame `31` 后，
  boundary mask 的有效跨视图覆盖低于 `32` 像素；放宽深度门也不能修复零几何重叠。后续只能在 train-only
  视图上冻结 exhaustive camera/frame geometry audit，再新建 run；不得读取 held-out 来选支持帧，也不得单纯
  放宽深度容差掩盖视锥不重叠。
- `V3-F41`：2D unseen completion 可生成不等于其深度足以作为全时段静态 Background。S2 r2 把全部未观测
  Telea 区域写入 3D checkpoint 后，四路 held-out 平均 PSNR/SSIM 退化 `0.495842 dB / 0.007160`，形成后续帧
  灰色遮挡；候选必须拒绝。S2 r3 保留完整 2D unseen artifact/provenance，但高支持 checkpoint 只持久化
  cross-view observed geometry，并重新通过未放宽的 held-out 门。后续不得把 inpainted RGB 自动升级为
  geometry-grounded world state；若要持久化 unseen 3D，必须增加独立深度/多视图证据和新预注册门。
- `V3-F42`：Harmonizer 导出 JIT 不是脱离官方 NGC runtime 即可原样执行的普通 TorchScript。当前权重包含
  `tex_ts::rmsnorm_fwd_inf_ts`，PyTorch 2.10 还会把两个 einops shape scalar 随 `map_location` 移到 CUDA，
  造成 shape tensor CPU/CUDA 冲突。当前适配只允许使用独立公式验证为 BF16 exact 的 RMSNorm 回退，并将整数
  1/2 shape scalar 放回 CPU；必须记录 runtime deviation 和测试，不能写成 untouched official runtime。
- `V3-F43`：生成式 final-render enhancer 可以恢复外观，同时破坏明确的 counterfactual 语义。S4 r2/r3 在
  G1 remove+inpaint 区域重新生成 actor-like 黑色车辆外观；r3 的 mask 内 L1=`14.217278`、changed fraction
  `0.541750`，失败冻结 `12.0 / 0.40` 门。全图 PSNR、outside drift 或“看起来更锐利”都不能覆盖 actor deletion
  失败；non-temporal Harmonizer 仅保留 optional diagnostic，不得默认进入 remove 输出链。未来复开必须增加
  显式 semantic conditioning/preservation 和连续帧 temporal gate，并取得 gated Cosmos base 的合法授权。
- `V3-F44`：DriveStudio 只读 forward 使用 `torch.inference_mode()` 不等于 trainer 已切换到 eval。训练态
  renderer 会对 `means2d` 调用 `retain_grad()`，与 inference tensor 冲突；R0 r2 因此在首个 forward 明确
  `rejected`。所有只读 replay 必须在每次 state load 后显式 `trainer.set_eval()`；失败 run 不得补写结果，
  只能修复代码后新建唯一 run。
- `V3-F45`：质量门必须冻结指标、区域、单位和范数；`MAE<=1 uint8` 不能被实现成逐像素 L∞/max-error
  `<=1`。R0 r3 的 source→mixed PSNR=`67.24–68.43 dB`、MAE=`0.0093–0.0123`，但两视角存在极少量
  max error=`2`，因错误合同仍保持 `rejected`。修复必须更新 protocol/runner hash 后新建 run，不能在旧结果上
  改 gate 或把 max error 隐藏。
- `V3-F46`：R0 `done` 只证明当前 scene 的 V3.2 资产可追踪集成与固定三视角 storage/package 等价。
  `GENERATED_BACKGROUND` 和 `GENERATED_ACTOR` 仍不是 GT；S4 仍被排除；S5 仍阻塞。432 MB mixed checkpoint、
  444 MB chunk payload 与 8.36 GiB 峰值也不证明 streaming、load、render、跨场景泛化或闭环安全收益；未来相关
  claim 必须有独立 protocol、数据与测量，不能由 R0 exact reassembly 外推。

- `V3-F18`：A3 R1 的工程链可逐位重放，但 heldout 评测越过冻结 GPU ceiling，且资源无效 diagnostic 为
  geometry 改善与 RGB safeguard 退化并存。后续不得提高旧 ceiling、替换旧 renderer 或继续调同一四步配方。
- `V3-F19`：传感器原始尺寸、checkpoint 原生加载尺寸和评测输出尺寸是三个不同合同。A4-P0 v1 的
  `1600×900` 误记不能在后续路线重现；每次 profile 必须同时记录三层尺寸和 downscale 来源。
- `V3-F20`：checkpoint `state_dict` 键不等于加载后的 runtime attribute。任何新资产注册、恢复或 streaming
  代码必须同时审计保存端、加载端赋值和 live object，不能从序列化 schema 猜运行时 API。
- `V3-F21`：A4-P1 最小预注册剪枝臂 b05 已违反 global/non-target 质量门。不得事后增加 b01/b02、放宽
  `0.10 dB` 或只挑 actor/boundary 指标，把同一结果改写成剪枝成功。
- `V3-F22`：A4-P2 只证明选择性 FP16 参数存储在冻结质量门内可把 checkpoint 减少 `25.35%`；它没有证明
  renderer、load、FPS 或 peak VRAM 加速，且 Gaussian means 不能安全降为 FP16。
- `V3-F23`：A4-P3 只证明 159-file static/actor chunk package 可 exact 重组。package 比 source 大
  `2.79%`、load/reassembly 更慢；没有 demand loading、cache policy 和驻留集测量时，不能声称 streaming/LOD 收益。
- `V3-F24`：F0 只完成 Instant NuRec 官方能力与本机前置审计；本机没有执行 inference，standalone CLI 只导出
  static PLY。不得把它写成前馈基线质量失败，也不得把 static PLY 当完整 dynamic WorldSim checkpoint。
- `V3-F25`：R0 的 63 inputs、23 decisions、12 deliverables 与 P3 package exact 只证明 V3.1 证据链闭环。
  它不证明 D2 dominance、R1/P1 有效、P2/P3 加速、完整 world model、跨场景泛化或闭环安全。

## V3 启动时必须先读的结论（2026-08-05）

- `V3-F01`：M4 的 non-target PSNR 93/95 dB 是硬局部编辑的构造/保持性证据，不是编辑后视觉质量。
- `V3-F02`：DriveStudio 已有 Affine、CamPose 与 LiDAR 初始化；A1 必须做 off/native/enhanced 消融，
  不得把上游能力改名为新增模块。
- `V3-F03`：V2 M5 未完成。0230/0242 checkpoint、Tier A/B/C 和 0255 诊断可复用，但不得把部分资产
  写成三场景压力测试通过。
- `V3-F04`：scene-0255 是小输入 CUDA `torch.cat` 工程阻塞且无 OOM 证据，不能写成 3DGS 方法失败。
- `V3-F05`：三个 scene 只支撑模型消融和工程判断；不得外推 trainval、夜间、长时或复杂交互。
- `V3-F06`：Instant NuRec 等工作已经改变前馈基线边界；DGGT 只作历史范式对照，不做跨分辨率、跨输入、
  跨训练预算的 leaderboard。
- `V3-F07`：persistent identity、actor binding、scene graph 和基础 trajectory edit 已由上游与 V2 覆盖，
  不能作为 V3 模型贡献。
- `V3-F08`：rolling shutter 需要真实 readout direction/time；没有 metadata 时必须 `not_supported`，不得
  从帧时间或相机顺序推测行曝光时间。
- `V3-F09`：actor-aware densification 必须分 D0–D3 小步消融；不得一次加入 boundary、LiDAR、visibility、
  residual 后只报告一个合并结果。
- `V3-F10`：编辑后 local refinement 的 unknown background 仍是 unknown；只允许 Tier-A、多视图或 LiDAR
  支持监督，Tier B/C 不得当伪真值回传。
- `V3-F11`：全图 PSNR/SSIM 不能代替 actor/边界质量；counterfactual mask 也不是真值分割，必须同时报告
  visible-image/pixel coverage，避免目标未渲染时通过缩小分母得到虚高指标。
- `V3-F12`：nuScenes processed camera ID 必须以数据加载器事实源映射；显示标签写错会把非相邻相机当成
  预注册相机对，已有 formal 必须 rejected 后重跑，不能只改图标题。
- `V3-F13`：seed=0 不保证 CUDA visibility filter 后的随机背景初始化逐点/逐计数复现。记录的 LiDAR/actor
  tensor exact 可作门禁，重建初始化 depth 只能作 witness，不能冒充源训练初始化 exact residual。
- `V3-F14`：局部 role、全图画质或 learned correction 稳定性改善不能替代预注册阶段主端点。C2/C3 未通过
  E1/E2 合同就不能为了保留增强模块而成为 C*。
- `V3-F15`：确认场景的原始端点方向可以与开发场景相反。不得把完整 Pareto 合同的 `done_off` 改写成
  “C0 在所有场景、所有指标都最好”，也不得只挑 0255 E1/E2 error 改写 C*。
- `V3-F16`：A2-D2 的边界改善、global/non-target 退化与更高训练成本构成严格 Pareto tradeoff。不得用新增
  事后标量权重把它改写成 D2 dominance；后续采用 D2 必须同时登记 D1 fallback 和完整退化轴。
- `V3-F17`：Gaussian ancestry、counterfactual footprint 和未提交 V2 M5 产物都不能自动升级为 A3 真值。
  ancestry 只证明来源，paired mask 只是模型诊断；A3 必须使用已提交输入和 typed support，S-C 保持 unsupported。

### V3-F01：局部保持不等于编辑质量

M4 的 lateral/delete non-target PSNR=`93.394483/95.598042`，主要来自编辑器只改变目标 actor 并保留其他
Gaussian。它证明实现没有意外改动非目标区域，但不能证明 source footprint 后方背景正确、actor 边界自然或
连续帧无闪烁。V3 必须把 outside preservation 与 Tier-A hole、depth ordering、boundary 和 temporal 指标分开。

### V3-F02：原生校准和初始化不能重复发明

DriveStudio `e59bda4` 的 `AffineTransform` 已输出 RGB affine，`CameraOptModule` 已学习 3D 平移和 6D 旋转
残差，数据集也已从 LiDAR 初始化背景/实例。A1 的合法动作是关闭/原生/增强的受控消融，以及 support provenance
审计；不能把启用原生 config 写成新成像、位姿或 LiDAR 模块。

### V3-F03/F04：V2 M5 部分证据与 scene-0255 工程阻塞必须分开

V2 M5 没有生成预注册的 24 条有效序列和 final matrix。scene-0230/0242 checkpoint 是有效训练资产；
scene-0255 训练则阻塞于 `datasets/driving_dataset.py` 实例点列表的 CUDA `torch.cat`。r27 观察到 166 个
CUDA float32 tensors、152 个 `(0, 3)` 空 tensor、177 scalars，且 `oom/oom_kill=0`。V3 A0 可以基于此做
最小 compatibility fix，但必须使用新 task/run，不能改写 M5 terminal，也不能由诊断完成推断训练完成。

V3 A0 已用 `436cfc1` 实现配对过滤：点与颜色按同一个 empty-row 条件过滤，全空时返回 prototype view。
canonical smoke `20260805T161656Z__scene0255-catfix-s0-r2` 在原生错误复现后完成真实 dataset init、1-step
优化与 checkpoint，说明该工程阻塞已在 smoke 范围解除。随后新 30k run
`20260805T162355Z__scene0255-native30k-s0-r1` 完成 checkpoint、registry 与 held-out 评估；0230/0242 通过
严格等价合同复用。该兼容问题现已闭环，但只证明工程修复和 A0 基线成立，不证明任何 A1/A2 方法提升。

### V3-F11：全图质量与模型差分 mask 都有明确边界

A0 中 scene-0242 全图 PSNR=`29.107`，高于 0230/0255，但其 high actor 区域 PSNR=`19.788`，反而是三场景
最低。scene-0255 boundary actor 区域 SSIM=`0.526`，也没有被全图 SSIM=`0.743` 反映。后续 A1/A2 不得只用
全图指标判断动态对象提升。

A0 actor mask 来自同一 checkpoint 的 original 与 actor-delete 配对渲染差分，是模型 counterfactual
diagnostic，不是 nuScenes 真值 segmentation。如果模型没有画出 actor，mask 会缩小；因此每个结果必须同时报告
candidate/visible image、effect pixel coverage 和 `ABSTAIN`。tight-crop LPIPS 用固定 8px padding 与 256px 输出，
不能和全图 DriveStudio LPIPS 混为同一指标。

A0 finalizer r1 因复用 checkpoint run 使用 `source_training_resources`、原生 run 使用 `train_resources` 而
`blocked`。这是汇总 schema 兼容失败，不是模型失败；`00ba4e8` 增加显式 provenance 归一化，r2 为唯一完成矩阵。

### V3-F05/F06/F07：结论规模与研究边界

三个固定 scene 足以比较相同数据、预算和实现下的 A0–A4，但不构成数据规模、天气、城市或交互分布覆盖。
Instant NuRec、OmniRe、IDSplat、SplatAD、ADGaussian、Real2Sim、RoVES 等工作分别覆盖前馈分层重建、
实例场景图、传感器和物理方向；V3 的价值来自完整复现、窄模型改动、负结果和工程 Pareto，而不是重新命名
已公开能力。只有 A2/A3 在至少 2/3 场景方向一致且资源稳定，才讨论扩展场景。

### V3-F08/F09/F10：禁止不可归因或无真值捷径

rolling shutter 没有 row timing 就不能实现；actor-aware densification 必须从 actor/background threshold 与
quota 开始，再分别增加 boundary/residual 和 LiDAR/visibility；local refinement 必须冻结 affected set 外参数，
并区分 expected/first-hit/measured depth。不得用 hard-composition outside=0、原图 actor 像素或未知区域的
自洽渲染作为方法成功证据。

### V3-F12：相机标签错误会污染跨相机端点

A1-E0 初版沿用了错误的显示顺序 `0=FRONT_LEFT / 1=FRONT / 2=FRONT_RIGHT`，但 DriveStudio nuScenes
事实源明确为 `0=FRONT / 1=FRONT_LEFT / 2=FRONT_RIGHT`。结果是名义上的相邻相机对可能实际落到
左右两侧非相邻画面，零支持也会被错误解释为模型现象。首次 formal
`20260806T140703Z__scene0230-c0-a1-e0-formal-full-s0-r1` 因此已标记
`rejected / INVALID_CAMERA_ID_LABEL_MAPPING`，原 terminal/manifest/summary 以 `*.original_done.json` 保留；
`d85ef27` 修复后 C0/C1 使用新唯一 run 回填。

防重复门禁：相机 ID/name 映射必须来自训练数据加载器或预处理权威列表，写入 resolved config 并纳入 hash；
QA 必须验证投影落在实际重叠的建筑/路面。若映射错误，所有受影响正式结果必须 rejected，不得通过重命名
已有 JSON、图片或曲线继续使用。

### V3-F13：随机 CUDA 可见性筛选不等于 exact 初始化 replay

A1 最小 LiDAR provenance 的 strict smoke
`20260806T142900Z__scene0230-a1-lidar-provenance-smoke1-s0-r1` 观察到：800,000 个背景 LiDAR 点、全部 24 个
actor point/color tensor、75,002 个 RigidNodes 初始点均 exact match，但随机 near/far 球面候选经过 CUDA
visibility filter 后，背景初始 Gaussian 数从源运行 946,484 变为 replay 的 946,597；后续 replay 又得到
946,309 和 946,291。这不是 LiDAR 输入变化，也没有训练或 checkpoint 修改。

防重复门禁：冻结的 `a1_lidar_provenance_v1.yaml` 要求记录 LiDAR/actor tensor exact match，并记录随机球面
候选、visibility mask SHA 和计数；背景 exact replay 固定为 `report_not_gate`，不允许事后设置“接近即可”的计数
容差。正式初始 depth residual 必须标为
`seed0_reconstructed_initialization_witness_not_exact_source_initialization`。要获得源训练初始化的 exact depth，未来
必须在训练创建时直接持久化 post-filter 初始化 tensors；A2 的逐 Gaussian ancestry 仍需独立 instrumentation。

### V3-F14：局部改善不能替代阶段主端点

scene-0230 中，C2 的 boundary-support E2 mean/P90 从 C0 的 `0.003547/0.006353` 改善为
`0.003346/0.005447`，但 high-support E2 P90 退化到 `0.011734`，actor/boundary LPIPS 也整体退化；因此不能把
单个 role 的改善提升为整个 E2 端点改善。C3 的全图 PSNR/LPIPS、boundary actor 质量和 learned pose correction
稳定性均最好，但 E1 median/P90 与两个 E2 role 仍未严格优于 C0。

A1-S0-v1 在结果已可见后、确认场景前把 V3.1 7.5 操作化为无容差严格 Pareto，并如实披露该时点；没有新增
事后数值阈值。正式结论必须是 `C*=C0-off / done_off`。不得更换 role、放宽端点、只引用 C3 全图画质或把
learned correction 幅值写成 pose GT，以强行保留增强模块。

### V3-F15：完整合同通过不等于每项指标方向一致

scene-0242 的 C0 在 global、E1 和 high E2 上优于 C1；scene-0255 则相反，C1 的 E1 median/P90 和两个 E2
role error 都更低。但 0255 C1 的 high E2 coverage 从 `23.529%` 降至 `21.569%`，boundary/high actor LPIPS
均退化，因而仍未通过冻结的“主端点改善、另一端点不退化、appearance LPIPS 可接受”完整合同。

A1 finalizer 的合法表述是：C1 在两个确认场景均不 eligible，C*=C0 保持 `done_off`，同时原始端点方向具有
scene dependence。禁止写成“C0 普遍校准更优”，也禁止忽略 coverage/appearance 只引用 0255 error 重选 C1。

### V3-F16：边界优先分支选择不等于全面方法提升

A2 formal 中，D2 相对 D1 的 boundary-support boundary-band PSNR/SSIM/LPIPS 从
`25.770024/.821572/.048382` 改善到 `26.171399/.828868/.044568`；但 global 从
`27.770024/.850915/.177704` 退化到 `27.703188/.850333/.178344`，non-target PSNR/SSIM 也下降，训练
wall time 从 `2099.33 s` 增至 `2720.82 s`。fixed 与 matched strict-quality、quality-cost 裁决都为
`tradeoff_non_dominated`，且 matched D2 只是 fixed 30k 的 exact alias，不是独立复现。

A3 采用 D2 是因为 A2 的预注册靶点包含 actor boundary，并且 D2 在该边界带三项指标同时改善；这是完整结果
可见后的工程资产路由，不是新增数值门槛、统计显著性或 D2 对 D1 的支配结论。任何后续报告都必须同时保留
D1 quota-only fallback，披露 D2 的 global/部分 actor/non-target/cost 退化，并禁止只摘录边界带结果宣称 A2
“全面提升”。单场景 scene-0230 也不能支持跨场景泛化结论。

### V3-F17：来源账本与 paired mask 不等于局部精修监督真值

D2 final checkpoint 的 Background ancestry 完整对齐 `1,205,164` 个 Gaussian，其中 `240,528` 个
`init_source=LIDAR` direct roots，其余含 random、split 与 clone；`nearest_lidar_distance` 对部分 lineage 有限，
但它是出生/父子来源记录，不是当前 target ray 的 T0 measured depth。A3 只能把 calibrated LiDAR projection 的
`depth_lidar_measured` 当 T0，把 first-hit 当 T1 ordering，把 expected depth 保持 diagnostic。

同样，source/edited footprint 来自同一 checkpoint 的 paired RGB difference，只能定位干预区域，不能充当真值
segmentation 或删除后的背景 RGB。S-A RGB 监督必须来自排除 target view 的 alternate camera/time 真实观测并有
calibrated reprojection；S-B 只使用 measured LiDAR 或至少两视图 geometry，禁止 RGB loss；S-C 不更新、不 seed、
不进入 loss，只报告 coverage/uncertainty/ABSTAIN。

当前工作树中的 V2 M5 protocol、`stress_metrics.py` 和 stress runner 均未提交且属于被冻结的用户工作，A3 不得
通过 import 或复制其结果建立隐式依赖。只能复用已提交并按 SHA 冻结的 M4 edit、paired mask、typed-depth 与
registry 接口；否则无法形成 clean source commit，也会把 V2 未闭环事实倒写成 V3 证据。

### V3-F18：A3 工程可重放不等于局部精修可晋级

A3 R1 已证明四个 S-B/T0 unit 的 opacity/scale 更新可以逐位重放，但可变集合只有 `51` 个 Background rows、
四个 unit 合计只有 `8` 个 T0 geometry pixels，S-A/RGB 为 `0/ABSTAIN`。heldout r2/r4/r5 的单 view 峰值稳定在
`14,241–14,245 MiB`，超过结果前冻结的 `12,288 MiB` ceiling；r5 的资源无效 diagnostic 同时出现 depth-order
改善和 non-target/original-global RGB MSE 严格退化，exact Pareto 为 `tradeoff_non_dominated`。

因此后续若复开局部精修，研究变量必须先变为“可观测支持如何获得、分层和拒绝”，而不是继续调 R1 的 step、
LR、alpha、mask dilation 或旧 renderer。合法复开需要新任务、新协议、与 heldout 隔离的支持审计，以及冻结前
证明 S-A 或更充分 T0/多视图证据确实存在；否则保持 `A3*=R0/D2 exact alias`。

### V3-F19：部署尺寸必须分为 sensor、model-native 与 evaluation 三层

A4-P0 v1 把 nuScenes sensor `1600×900` 写成 checkpoint 原生分辨率，但 source config 已固定
`downscale_when_loading=[2,2,2]`，真实模型加载和 render 均为 `800×450`。v1 保持 `blocked`，v2 只纠正输入
合同并在新 run 完整重跑；不能使用 v1 的性能数字关闭 P0。

后续所有 runtime、质量和资源协议必须显式记录 sensor resolution、source-config downscale、model-native
resolution 和最终 evaluation/output resolution。任何一层变化都是新的实验因子，不能用“native”一词静默折叠。

### V3-F20：序列化 schema 不能替代 live runtime API 审计

A4-P5 r1 已生成合法 registry，却把 checkpoint key `points_ids` 当作加载后的 `RigidNodes.points_ids`；实际
`load_state_dict` 会把它写入 `self.point_ids`。r1 因此保留 `blocked`，修复后的 r2 才通过 14/14 audits。

后续做 lazy loading、分块恢复、资产注册或 checkpoint 迁移时，必须分别验证 state key、load hook、live attribute
和调用方，并用真实 fresh-process reload 测试锁定。不能因为 checkpoint 中存在字段，就推断运行时对象暴露同名接口。

### V3-F21：预注册最小剪枝臂失败后不能事后缩小 fraction

A4-P1 的 b05/b10/b20 均通过结构、reload、count 和资源审计，但最小 b05 已使 global occupied PSNR、global PSNR
和 non-target PSNR 分别退化 `0.117684/0.110926/0.125462 dB`，超过冻结 `0.10 dB` 门；更大 fraction 失败更多
端点。局部 actor/boundary 指标保持或改善不能覆盖全局与非目标区失败。

后续不得在同一 ranking、视图和结果上新增 b01/b02、改变阈值或只报局部轴。若重新研究压缩，必须有不同的、
结果前可解释的结构假设与新预注册；单纯把 fraction 调小不是新的研究问题。

### V3-F22：FP16 存储压缩不等于端到端加速

A4-P2 把 Background/RigidNodes 的 10 个 scale/quat/feature/opacity tensors 转为 FP16，checkpoint 从
`578,819,674` 降到 `432,111,754` bytes，31/31 quality safeguards 通过。但 candidate 的 load、P50 和 FPS 没有
形成一致加速，renderer 输入仍显式转回 FP32；source audit 还表明 Background means 若做 FP16 roundtrip，最大
空间误差接近 `1 m`。

因此当前合法 claim 仅为 `mixed_precision_parameter_storage_fp32_render`。后续若研究低精度执行，必须单独冻结
renderer dtype、数值误差、质量、peak resident memory 和 latency 合同；不得从文件变小推断 Tensor Core、VRAM
或实时收益，也不得把 means、trajectory 或 provenance 一并降精度。

### V3-F23：exact chunk package 不等于 streaming/LOD 系统

A4-P3 的 133 static + 24 actor + skeleton + manifest 共 159 files 可 exact 重组，57 RGB SHA、31 endpoints、
85 tensor paths 和 source/registry immutability 全通过。但 package 比 source checkpoint 大 `2.792171%`，全量读取的
load/reassembly 与 render 均未加速，filesystem cache 也未控制。

后续只有在实现真实 demand loading、明确 working set/cache/eviction、记录首帧与稳态 latency、peak resident bytes、
I/O bytes 和 exact fallback 后，才可研究 streaming/LOD。继续把同一 159-file package 全量读入内存，只能叫资产
分离，不能叫部署加速。

### V3-F24：F0 前置失败不是前馈方法质量失败

Instant NuRec canonical audit 只通过 4/11 prerequisites；Python 3.11、uv、30 GiB VRAM、100 GB free disk、exact
weights、licensed NCore input 与 terms record 未同时满足，所以 `inference_command_constructed=false`。官方 standalone
CLI 的实际输出又只含 static PLY，不含 dynamic/sky/ISP/actor registry/trajectory/depth。

后续不得把“本机未运行”写成 upstream quality reject，也不得用 static PLY 与 StreetGS 完整 checkpoint 做假等价
比较。只有硬件、许可、数据和 converter 接口全部独立满足后，才能用新任务做窄范围前馈 pilot。

### V3-F25：证据链 exact 不等于研究主张自动成立

R0 canonical 的 63/63 inputs、23/23 decisions、12/12 deliverables、26/26 manifest files 与 P3 159-file package
全部 exact，证明 V3.1 可以从冻结事实恢复同一结论与生产链。它没有增加场景、seed、真值、闭环控制或新的方法臂。

后续研究必须从一个明确、可证伪的新问题出发，并说明新增证据解除哪一条失败约束；不能把 R0 的可复现性重新命名为
完整 world model、跨场景泛化、物理真实性或安全性。若主张涉及 A2/A3 方法，至少需要独立场景确认；若主张涉及
部署收益，必须直接测量对应 runtime/working-set 端点。

<a id="detail-v2"></a>

## V2 启动时必须先读的结论（2026-08-02）

- `PIVOT-F03`：AD-GS exact reproduction 已完成；V2 只读最终 checkpoint/render/metrics，不重复训练。
- `PIVOT-F04`：可见性建模不等于未观测背景真值；M5 必须保留 Tier A/B/C。
- `PIVOT-F05/F14`：资源、外部实例与方法失败分开；OOM/重启不能写成模型质量结论。
- `PIVOT-F06/F07/F08`：换机、非登录 shell 与浮动权重必须重新审计；镜像不能改变固定版本。
- `PIVOT-F10/F11/F12/F13`：PNG/JPEG、COLMAP 并发、cgroup 90% 和合法空占位均已有失败证据。
- `PIVOT-F14B`：V1 pointops2 的直接根因是 PEP 517 隔离构建缺少 torch；V2 先按 upstream
  `python setup.py install`，不重复原 `pip install .`。
- `PIVOT-F15`：AD-GS camera-local pseudo ID 与二值 `obj` 不能支持对象级编辑；V2 以 nuScenes
  `instance_token` 只做评测真值，不注入 AD-GS 训练。
- `PIVOT-F16`：持久身份、actor binding 与基础轨迹编辑本身已不新；V2 必须先产生跨三场景真实失败，
  再做新的 novelty gate。

存储清理只使历史环境和中间 checkpoint non-resident，不撤销上述失败，也不允许重新运行已关闭路线。

<a id="detail-legacy"></a>

## N1 kinematics-first 第三次 reject 与第四版约束（2026-07-25）

### N1-F12：地图分支收敛不是车辆横向机动

**观察**

- 第三次人审文件：
  `/root/autodl-tmp/runs/event_first/N1-EVENT-KINEMATIC-01/v71_n1-event-kinematic-01__kinematic-v1__s0__20260725T092427030639Z__8c2247b6/audit/review_working.jsonl`；
- review SHA256：
  `005cd74b874833808435fd2f47387d1d8e446cdea2d3a5cae6146e34bf331e96`；
- 12/12 已审，`TRUE_POSITIVE=0`、`FALSE_POSITIVE=12`、`UNCERTAIN=0`，precision=`0`；
- subject maneuver 为 `INVALID` 12/12；failure code 为
  `SUBJECT_NO_LATERAL_MANEUVER=12`、`ROUTE_CONTINUATION=11`、`NORMAL_TURN=1`、
  `MAP_MATCH_JITTER=1`；
- 第三版 12/12 机器候选都是 `converging_branch_merge`。规则只验证 source/target
  地图分支在几何上汇合，却没有验证车辆中心/车身相对接收车道发生 outside→inside 横移；
- target corridor 人审 12/12 为 `VALID` 并不能挽救 subject maneuver。地图画对了，不等于事件成立。

**根因**

第三版仍把“actor 沿一条会汇入 target 的道路行驶”当成“actor 主动切入 target 车流”。车辆可以保持正常
转向/道路中心线跟随，而道路本身向另一分支收敛；仅比较 source/target approach heading 或地图 token
变化仍会把路形变化误写成车辆运动学。

**防重复**

- 必须直接从原始 2 Hz annotation 计算 subject 相对接收 corridor 的连续横向状态；
- 至少观察目标车道中心外→中心内，并在进入后保持名义 1 s；10 Hz 插值不参与物理门；
- 进入前还必须与接收 corridor 近似同向，避免把大角度路口/主路续接的几何距离收敛当作 cut-in；
- 不再把 `merge` 地图类别、multiple incoming、route token change 或道路弯曲本身当正例。

### N1-F13：接收车必须来自独立目标车流，不能复用 subject 后车

**观察**

- 第三次 review 中 rear 为 `INVALID` 2/12、front 为 `INVALID` 1/12；
- 第三版 corridor 构造会贪心选择与 subject source 最顺的 incoming，再在该 corridor 上找 rear；
- 因而所谓 rear 往往就是 subject 原队列中的后车，而不是被切入目标车流的接收车；
- K3-004 选错 front branch；K3-007、K3-010 选错 rear branch。其余多项虽被人审写成 corridor
  `VALID`，也只说明地图链连续，不证明 receiver 角色语义成立。

**第四版硬约束**

1. parallel lane change 的 target chain 显式排除 source token；
2. merge 只枚举 `target` 的 direct incoming 中不同于 subject source 的分支；
3. RECEIVER 必须在进入前后保持同一 identity、同向、最近后车次序与 `[0.5,40] m` bumper gap；
4. subject/receiver 之间不得遗漏更近同 corridor 车辆；
5. negative control 也必须存在持续 receiver，不能用孤车普通直行冒充交互密度等价 control。

### N1-F14：第三次裁决的研究失败与工程失败必须分开

**研究裁决**

- clean adjudication commit：`1fbbbc1`；
- 成功 run：
  `/root/autodl-tmp/runs/event_first/N1-EVENT-KINEMATIC-AUDIT-01/v71_n1-event-kinematic-audit-01__human-audit-reject-v1__s0__20260725T155754010881Z__4c51f0d9`；
- 唯一终态 `REJECTED`，`n2_authorized=false`。

**保留的工程失败**

第一次 formal adjudication 使用了错误的 audit-manifest 指纹键，在写入研究产物前失败：
`.../v71_n1-event-kinematic-audit-01__human-audit-reject-v1__s0__20260725T155523736677Z__4c51f0d9/`。
该目录保留 `FAILED/failure.json`，原因是 `engineering_manifest_key_mismatch`。修复将
`artifact_set_sha256` 更正为实际 schema 的 `immutable_artifact_set_sha256`，并把所有输入校验提前到
run 目录创建之前。不得删除失败尝试或把它统计成 research reject。

### N1-F15：四图全常驻与重型 map API 会触发 2 GiB cgroup 峰值

**观察**

- 第四版首个 development smoke 在算法开始前以 `RC=137` 被杀；
- 容器 `memory.max=2147483648`，当时常驻服务已占约 `1.85 GiB`；
- 官方 `nuscenes.map_expansion.map_api` 的导入会连带 OpenCV、Matplotlib、Shapely 和渲染 API；
  单是 import probe RSS 就从约 `58 MiB` 增至约 `212 MiB`；
- 同时常驻四张 `NuScenesMap`、完整 sample/instance JSON 行和 128-scene dense batch 会进一步放大峰值。

**工程修复**

- 新增只读取 `lane`、`lane_connector`、`arcline_path_3`、`connectivity` 的轻量 map reader；
- arcline 离散化与官方 devkit reference 在单测中逐点一致；
- map index 改为一次只缓存一个 location，calibration/evaluation 按 location 排序；
- `sample.json`、`instance.json` 改为 ijson 流式最小字段投影，scene builder 复用同一 metadata source；
- scene batch 冻结为 32；不得通过杀死用户编辑器进程、修改容器上限或跳过地图证据来“解决”。

### N1-F16：负对照配置契约缺项导致首个正式 K4 工程失败

**失败事实**

- 失败 run：
  `/root/autodl-tmp/runs/event_first/N1-EVENT-CUTIN-01/v71_n1-event-cutin-01__receiver-cutin-v1__s0__20260725T170948229629Z__46186120`；
- 失败代码提交：`f5c9bbe4c819abce42e1cca0b8800e16a77af680`；
- 正式日志：
  `/root/autodl-tmp/runs/event_first/n1k4_formal_f5c9bbe.log`，SHA256
  `b9ac6d3cce2e731f16aad7bc6a068eaf09c54439ad654bb0a3a9d0c58f63a487`；
- calibration 已运行，进入首个 formal evaluation batch 后，在构造 same-actor 30-frame
  lane-keeping negative control 时抛出 `KeyError: min_median_speed_mps`；
- `lane_keeping_features` 实际接收 `kinematics_control`，但该字段只写在 `cutin` 下；同一调用下一步还会需要
  `max_acceleration_mps2`，原 YAML 也遗漏；
- 失败发生在 `event_pool.json`、`summary.json` 与任何研究裁决写入前。因此它是工程失败，不是机器 gate
  reject，更不是第四次人工评测结论；`n2_authorized=false`。

**保留与修复**

- 旧目录不删除、不改造成成功 run；写入结构化 `failure.json` 与 `FAILED`，原 `RUNNING` 被保留为
  `RUNNING.invalidated`；
- 修复提交 `8581d4dcd1bf9a4f92b426c601e1149c804afc5a` 同时补入
  `kinematics_control.min_median_speed_mps=0.5` 和 `max_acceleration_mps2=12.0`；
- 新增启动前 `_validate_config_contract`，在加载 nuScenes metadata 前检查 delayed runtime dependency、
  `receiver_cutin` 审核 schema 与 `never_start_n2_from_this_run=true`；
- 新增 post-run-directory 异常处理：后续未捕获异常自动写 `FAILED/failure.json`、清除活动
  `RUNNING`，并强制 `n2_authorized=false`；
- 27 项相关测试通过后才从新 run ID 完整重跑。

**防重复**

1. development pilot 必须覆盖至少一个 positive actor 的 negative-control 搜索；“positive=1、
   negative=0”不能被误读为该分支已执行；
2. 所有按候选稀疏触发的配置依赖必须在启动时校验，不能等全量运行数分钟后才由 `KeyError` 暴露；
3. 任何残留 `RUNNING` 的异常目录必须先结构化归档，再开始新 run；禁止覆盖、续跑或统计为 research
   reject；
4. 修复配置/异常落盘不授权改变冻结 K4 阈值、评估 scene 或候选排序。

### N1-F17：重复扫描 583 MB 标注文件产生 cgroup 页缓存压力与外部 SIGKILL

**失败事实**

- 失败 run：
  `/root/autodl-tmp/runs/event_first/N1-EVENT-CUTIN-01/v71_n1-event-cutin-01__receiver-cutin-v1__s0__20260725T171746938858Z__5b1634e3`；
- 失败代码提交：`8581d4dcd1bf9a4f92b426c601e1149c804afc5a`；
- 正式日志：
  `/root/autodl-tmp/runs/event_first/n1k4_formal_8581d4d.log`，SHA256
  `7a89e5f5ab88c53a6d9531dedc56a9db302cef8dab144ade7da75a91f3c09191`；
- calibration 与前 96/685 个 evaluation scenes 已执行，随后 shell 报 `Killed`；没有
  `event_pool.json`、`summary.json` 或研究裁决；
- 本层 cgroup `memory.max=2147483648`，事件计数仍为 `oom=0`、`oom_kill=0`，因此不能把信号来源伪写成
  kernel OOM；终态登记为 `external_sigkill_under_cgroup_memory_pressure`；
- `sample_annotation.json` 大小为 `583417244` bytes。失败后进程已消失时，
  `memory.current=1704148992`、file cache=`639545344` bytes；
- 对该只读标注文件执行 `POSIX_FADV_DONTNEED` 后，在没有停止编辑器/Jupyter/TensorBoard 等用户服务的前提下，
  `memory.current` 立即降为 `1169817600`、file cache 降为 `102739968` bytes。

**根因边界**

证据支持以下工程推断：每个 32-scene batch 都顺序扫描 583 MB 标注表，读页长期计入 2 GiB cgroup；
进程 RSS、既有服务和文件页缓存共同逼近硬上限，外部管理层随后发送 SIGKILL。由于无内核日志权限且
`oom_kill=0`，不能声称已证明具体 killer；但页缓存释放的前后差值直接证明了主要可控压力源。

**修复与复验**

- 修复提交 `f13eb0f1e39b608de1c5e698cd678c2dfd8365a4`；
- 所有大型顺序输入在读取前标记 `POSIX_FADV_SEQUENTIAL`，读取后标记
  `POSIX_FADV_DONTNEED`；
- per-scene JSON 改为 `json.load(file_handle)`，不再由 `read_text` 同时常驻字符串和解析对象；
- 每批显式删除 dense scene payload、执行 `gc.collect` 与 glibc `malloc_trim`；
- 每批日志新增 process RSS 与 cgroup current；正式启动若缺 POSIX page-cache control 则 fail closed；
- 30 项相关测试通过。新正式 run 在 96 scenes 的同一死亡点记录
  RSS `602673152`、cgroup current `1707110400` bytes，且继续运行，证明修复覆盖了原路径；
- 成功 run 最终完成 685/685 scenes；最后一批 RSS `510734336`、cgroup current
  `1612763136` bytes，`oom=0`、`oom_kill=0`。它以独立 run ID
  `...T173015103731Z__5b1634e3` 和唯一 `AWAITING_HUMAN_REVIEW` 结束。

**防重复**

1. Python RSS 不是 2 GiB 容器的完整内存分母；必须同时记录 anon、file cache、cgroup current 与
   `memory.events`；
2. 流式解析只限制 Python 对象，不自动释放内核页缓存；反复全表扫描必须有 cache-pressure 策略；
3. 不得以杀死用户服务、跳过正式场景、降低地图分辨率或减少校准标签来换取“成功”；
4. SIGKILL 无法触发 Python exception handler，因此监控器必须把残留 `RUNNING` 另行结构化封存；
5. 该修复只改变 I/O/内存生命周期，不改变 K4 候选、阈值、排序、scene split 或人工门槛。

### 第四版 calibration 冻结结果与禁止矩阵

第四版只用第二、三次全部 49 条人工标签调试阈值，所有 26 个已审 scene 从 formal evaluation 排除。
截至冻结前 development replay：

- 第三次 FP 拒绝 `12/12`；
- 第二次 FP 拒绝 `35/35`；
- 第二次 TP 保留 `1/2`；
- 被保留真例同时满足目标车道中心外→中心内、进入后稳定、进入前近似同向和 RECEIVER 前后身份连续；
- 另一个旧 TP 因没有独立 RECEIVER 的 pre identity support 被拒绝，不用旧 overall 标签覆盖新事件定义。

| 快捷做法 | 为什么无效 | 第四版合法替代 |
|---|---|---|
| 降低分支/多 incoming 门槛 | 仍把地图属性当车辆行为 | 原始 2 Hz center/box outside→inside |
| 复用 subject source-stream rear | 重演 K3 rear 污染 | 独立 direct incoming / target lane RECEIVER |
| 只要求进入后有 rear | 无法证明被切入车流在事件前已存在 | 同一 RECEIVER pre/post identity |
| 因 0.999 s 拒绝名义三帧 1 s | nuScenes 时间戳有毫秒抖动 | 冻结 20 ms timestamp tolerance，仍需 3 个 2 Hz 帧 |
| 在正式 train 结果上再调阈值/scene | evaluation 泄漏 | 阈值只由 49 条旧审标签冻结 |
| 缩短 30-frame negative 或允许 overlap | 改写 matched-control 问题 | physical event window + 0.5 s guard，control 仍 30 frames |
| 自动启动 N2 | 三次 reject 后边界更严格 | 第四次用户裁决 + 新授权前 `n2_authorized=false` |

## N1 kinematics-first 第三版（2026-07-25，已人工 REJECTED）

### N1-F09：候选真实性与 matched-control 支持是两个独立门槛

**观察**

- clean commit `aa162ef4dea808ad28ca7e56f1273f106e9c0e49` 上的 official train 694-scene
  formal run 完成 8,631 transitions → 1,879 topology-pass → 244 physical-motion-pass →
  12 interaction candidates；
- 12 candidates 覆盖 9 scenes，达到 candidate `≥12` 与 scene `≥6`；
- same-actor lane-keeping negative 只有 2，same-actor pair 只有 2，均低于冻结阈值 4；
- 因此 parent `machine_gate_passed=false`；parent 的唯一 terminal 保持
  `AWAITING_HUMAN_REVIEW`，后续独立 adjudication 已按 12/12 FP 写成 `REJECTED`；
- `AWAITING_HUMAN_REVIEW` 只是当时的审计就绪状态，从未表示 machine pass 或 N1/N2 授权。

**Pair 失败的冻结诊断**

对 12 个 positive actor 重放原 30-frame negative 搜索，不改 event pool：

| 主阻塞 | actor 数 | 观察 |
|---|---:|---|
| paired | 2 | 仅 `scene-0870` 两个 actor |
| 无 30-frame stable run | 1 | actor 轨迹支持太短 |
| 所有窗口与 positive overlap | 5 | 4–27 个候选窗口全部重叠 |
| non-overlap lane-keeping 存在，但 interaction 全失败 | 4 | 共 25 个 lane-keeping PASS windows，全部缺 center front/rear |

其中 6/10 未配对 actor 没有可用的非重叠长控制窗口；另外 4/10 没有等价的双侧 interaction control。
这不是把 gap 或速度阈值稍微放宽就能解决的问题。

**禁止快捷修补**

- 不把 30-frame 缩短到刚好得到 4 pairs；
- 不允许 negative 与 positive event overlap；
- 不用不同 actor 冒充 same-actor control，也不只挑有 pair 的两个 actor 报告；
- 不把普通 lane-keeping 但缺 front/rear 的窗口当成与正例等价的 interaction negative；
- 不因人工可能判真而把 `machine_research_support` 改成 true。

**可能突破**

第三次人工已表明 12/12 merge 候选均不真实，因此先修 subject/receiver 语义，再谈 control 扩展。
若第四版人审真实性通过但 same-actor control 仍不足，可新预注册“更长日志中的同 actor control”或
“matched-other-actor control”；后者必须显式匹配 scene、类别、速度、道路与交互密度。二者都不能回写
第三版 run。

### N1-F10：第三版最终候选只覆盖 converging-branch merge

**观察**

- 244 个 physical-motion-pass 包含 181 merge、63 parallel lane change；
- interaction 层有 215 个在中心关键帧缺 front/rear、17 个 temporal identity/bumper-gap 失败；
- 最终 12/12 candidates 全是 `converging_branch_merge`，parallel lane-change 为 0。

**能下的结论**

第三次人审只能估计这 12 个 converging-branch merge 的真实性。即使全部为真，也不能声称第三版已经覆盖
一般 lane change/cut-in；同时也不能断言 63 个 physical lane-change 都是假事件，因为它们是在更严格的
双侧 interaction 层失败。

**复开条件**

先把 `subject maneuver authenticity` 与 `front+rear gap-insertion interaction` 拆成两个预注册层。
可对 63 个 physical lane-change 建独立 diagnostic audit，但不得事后补进当前 12 条、降低当前 machine gate
或把 subject-only event 当 interaction positive。

### N1-F11：完整审核包不等于每项都有前视相机可见证据

**观察**

- 正式包包含 12/12 panels、evidence、topdown、checklist、prompt 和逐文件 SHA256；
- 本机 full train 数据完整覆盖 CAM_FRONT，但其他五个相机目录只有 mini 规模，不能为这些 formal scenes
  提供稳定六相机视图；
- 首尾面板 QA 正常；部分 subject/front/rear 不在 CAM_FRONT 视野，但 2 Hz annotation topdown 仍存在；
- 40 个 immutable audit files 复算 0 hash mismatch；空白 review validator 按预期 fail closed。

**审核边界**

不在 CAM_FRONT 中的角色不能被猜测为 VALID。评审先使用 topdown、vector centerline 和跨时刻 identity；
若相机与 annotation 冲突或证据仍不足，必须判 `UNCERTAIN` 并记录
`INSUFFICIENT_VISUAL_EVIDENCE`。补六相机或 raw LiDAR 需要独立资产/用途授权，不能偷偷进入本轮或 N2。

### N1 第三版历史禁止重试矩阵

| 快捷做法 | 为什么无效 | 合法后续 |
|---|---|---|
| 把 parent `AWAITING_HUMAN_REVIEW` 写成 pass | negative/pair 失败且人工 12/12 FP | 独立 adjudication 已 `REJECTED` |
| 用人工结果覆盖 pair gate | authenticity 与 comparison support 是不同问题 | 两类 gate 均保留 |
| 缩短/重叠 negative window | 事后改变 matched-control 定义 | 新任务、新 split、新预注册 |
| 用其他 actor 补足 same-actor pair | 混入 actor/scene confound | 预注册 matched-other-actor 设计 |
| 把 63 个 physical lane changes 加入 positive | 它们没有通过冻结 interaction | 单独 subject-only diagnostic |
| 用单侧 front 或 rear 算 interaction | 改变“插入双侧 gap”的研究对象 | 另立事件 subtype |
| 没有相机框也猜 TRUE/FALSE | 把证据缺失转成标签 | `UNCERTAIN` |
| 审核后自动启动 N2 | 本 run 明确 `n2_authorized=false` | 新授权 + 新 gate |

## N1 full-domain 第二次 reject（2026-07-25）

### N1-F05：把 target 多 incoming 的地图类别误当成 subject 行为

**观察**

- 父机器 run `N1-EVENT-FULL-01` 在 val 146 上报 37 个 positive，其中 topology 为 35 merge + 2 lane change；
- 完成人审文件 SHA256 为
  `ae71b31e02faf1d783c36748e629e85acf32a132f35ce2f98102a5f62201dd05`；
- 用户确认的逐项结果为 `TRUE_POSITIVE=2`、`FALSE_POSITIVE=35`、`UNCERTAIN=0`，机器候选精度
  `2/37=0.054054`；
- 多数 reviewer notes 明确指出：subject 沿与 target 共线的主路 lane/connector 正常直行，真正汇入
  target 的是另一条 incoming branch；旧规则却只因 `target_incoming_count>=2` 就把 subject 标为 merge；
- 独立 audit adjudication：
  `/root/autodl-tmp/runs/event_first/N1-EVENT-FULL-AUDIT-01/v71_n1-event-full-audit-01__human-audit-reject-v1__s0__20260725T083929632491Z__6507cbac/`，
  唯一终态 `REJECTED`。

**能下的结论**

graph-corridor 修复了邻车跨 token fragmentation，却没有证明 subject 本身执行了 lateral maneuver。
“target 有多个 incoming”是地图节点属性，不是 actor-specific merge 证据；第二次 N1 不能进入 N2。

**不能下的结论**

不能据此断言 full nuScenes 没有真实 lane change/merge，也不能把 2 个标注 TP 当成已验证的完整事件池。
旧 audit panel 没有把 subject/front/rear 的 3D identity 投影到图像，且 reviewer 字段包含多个来源；
用户已整体确认 reject，但单条 TP 仍只能作为第三版 calibration 标签，不得直接进入 formal evaluation。

### N1-F06：10 Hz 插值 cadence 被错误提升为物理证据

**观察**

nuScenes `sample` 是 2 Hz 标注关键帧。第二版把 2 Hz box 线性/SLERP 插值到 10 Hz 后，用连续 lane-token run
寻找 transition；该做法对齐了 DriveStudio cadence，却没有产生新的物理观测。第二次人审 notes 多次指出
短 token 切换、轨迹插值或 map assignment 假象。

**防重复**

- 第三版速度、加速度、yaw-rate、lane preference、front/rear persistence 只能用原始 2 Hz keyframe；
- 10 Hz 只用于 frame 对齐、可视化和复现旧 transition 候选，不得计算导数或宣称 0.1 s 观测；
- 至少 3 个 pre 和 3 个 post keyframes；不足时为 `UNKNOWN`，不得靠插值补齐。

### N1-F07：单时刻中心距不是持续物理交互

**观察**

第二版在单一 relation frame 上用中心线 `s` 与中心距 `[2,60] m` 选择 front/rear，没有扣除 box extent，
也不要求同一 front/rear identity 跨时刻持续。37 个 machine positive 中 36 个至少依赖一个跨-token 邻车，
因此 branch 选择错误会直接翻转结果。

**第三版解除条件**

1. target corridor 每个 graph edge 同时满足方向连续和 endpoint 连续，并只选单一最连续分支；
2. 使用 oriented box 在 lane tangent 上的投影半长，报告 bumper gap 与 center gap；
3. 至少 2/3 个连续 2 Hz keyframe 保持同一 front/rear identity、方向和次序；
4. 同时报告 longitudinal speed、closing speed、headway/TTC；它们是诊断，不得替代人审。

### N1-F08：第二版审核合同与 provenance 不足

**观察**

- 父机器 run 诚实记录 `code_dirty=true`；它可定位但不是 clean-commit formal baseline；
- 旧人审清单给了逐项 verdict 定义，却未预注册聚合阈值；
- 因此第二次 reject adjudication 没有查看结果后补造阈值，只登记用户明确决定；
- 旧 CAM_FRONT 清单没有身份 box overlay，容易把画面中“真正并道的另一辆车”认成 subject。

**复开条件**

第三版必须在 clean commit 上运行；正式 audit pack 同时提供盲序、subject/front/rear 颜色框、2 Hz 俯视轨迹、
逐项 component verdict、failure codes、完整提示词、immutable file hashes、预注册统计阈值和独立
adjudication 命令。Agent 不得填写人工 verdict。

### N1 full-domain 禁止重试矩阵

| 快捷做法 | 为什么无效 | 第三版允许替代 |
|---|---|---|
| 继续调 `graph_hops` 或 gap | 35/37 误报的主体事件本身不成立 | actor-specific 2 Hz kinematics 先行 |
| target 有多个 incoming 就叫 merge | 把地图节点类别当行为 | 比较 source 与主路 incoming 的 approach geometry |
| 用已审 val 37 条挑最终阈值并在同一 split 报结果 | calibration/evaluation 泄漏 | val 只 calibration；official train formal evaluation |
| 从 10 Hz 插值计算速度/横移 | 人造高频证据 | 原始 sample timestamp + 2 Hz boxes |
| 单帧 front/rear 中心距 | branch/identity 易跳变，忽略车长 | branch-safe corridor + temporal identity + bumper gap |
| 把 2 个旧 TP 直接当第三版正例 | panel identity 仍有未决风险 | 只作 calibration；第三版候选重新盲审 |
| 机器候选一出现就启动 N2 | 人工真实性与样本支持尚未通过 | `AWAITING_HUMAN_REVIEW`，`n2_authorized=false` |

## N1 mini event-pool reject（2026-07-24）

### N1-F01：interaction-support failure

**观察**

- N0 map-expansion、scene→map 与 pose contract 已通过，不再是资产缺失；
- 45 个 source-only eligible actors 产生 71 个 stable token transitions；
- topology taxonomy：39 route continuations、19 merges、3 lane changes、10 unresolved；
- 19 merges + 3 lane changes 共 22 个 topology-pass candidates；
- 22/22 的 exact-target-token front/rear relation 为 FAIL；
- 18 个没有 target-token 邻车，4 个只有 front、没有 rear；0 个同时满足 2–60 m front/rear；
- positive=0、negative pairing=0、same-actor pair=0、positive scenes=0，唯一终态 `REJECTED`。

**能下的结论**

冻结 mini split 不支持可比较 interaction event pool，N2–N5 不触发。地图缺失不是旧 H1 的唯一根因；
补地图后 mini interaction support 仍为零。

**不能下的结论**

不能写成“人类绝对看不到任何交互”或“full nuScenes 也没有事件”。exact target token 可能把同一
longitudinal corridor 上的 actor 分到相邻 lane/connector token；该表示风险尚未独立校准。

**复开条件**

mini run 不复开。新的路线必须：

1. 使用不同 run/task ID；
2. 以 22 topology-pass mini cases 仅作 calibration/audit，不作 formal evaluation；
3. 在 graph corridor 上定义 route-aligned curvilinear front/rear，而非后验放宽欧氏半径；
4. calibration 与 evaluation scenes 分离；
5. 优先在 full nuScenes trainval annotations/metadata 上冻结并评估。

### N1-F02：exact-token corridor fragmentation

**观察**

71 transitions 中 39 个只是 directed route continuation，说明官方 lane graph 将连续道路划分为多个
lane/lane_connector token。当前 interaction 只接受 relation frame 上与 subject 完全相同的 target token。

**推断**

该规则高精度但可能低 recall，尤其在 lane→connector→lane 或短 lane segment 附近。它是 0 interaction
PASS 的一个可能贡献因素，但不是已证实的唯一原因；mini 本身也可能确实缺少前后车。

**禁止快捷修补**

- 不把“相邻 token”全部并入；
- 不把只有 front 或只有 rear 改成 positive；
- 不把 82–89 m front 后验纳入 60 m；
- 不在同一 22 cases 上调 graph hops、gap 或 heading 直到出现 positive。

允许的修复是先定义有向 corridor、route-aligned `s` 和 branch disambiguation，再由独立 calibration
审计冻结；formal evaluation 必须 scene-disjoint。

### N1-F03：mini scale 与静止对象密度

**观察**

- 003/005/004 eligible actors 为 7/22/16；
- 因首尾位移不足 5 m 被拒的 actor 为 107/17/5；
- eligible pose map-match coverage 为 88.89% / 95.60% / 93.36%；
- 官方 full nuScenes 有 1,000 个约 20 秒 scenes，850 个为 train/val，而当前 formal pool 只有 3 scenes。

**结论**

mini 三场景对多 scene interaction event pool 的统计支持不足。下一步应扩数据底座，不应换 actor 或删场景。
优先同域 `v1.0-trainval` annotations/metadata，只有其 event gate 仍失败才评估 nuPlan/Waymo。

### N1-F04：negative=0 的语义

N1 只为已经有 positive 的 actor 构造 same-actor comparable negative。因此 `negative=0` 是
`positive actor set=∅` 的结构结果，不证明没有稳定非事件窗口。后续报告必须同时给出 positive actor 分母，
不得把 negative=0 解释为数据中全是事件或完全无普通驾驶。

### N1 禁止重试矩阵

| 快捷做法 | 为什么无效 | 允许替代 |
|---|---|---|
| 删除 rear requirement | 改变冻结 interaction claim | corridor calibration + scene-disjoint evaluation |
| 扩大 60 m 到覆盖 82–89 m | 看结果后调阈值 | 在新 calibration pool 依据任务时间窗冻结 |
| exact token 改成任意相邻 token | 可能跨 branch/对向车道误配 | directed corridor + route-aligned `s` |
| 从 22 cases 挑“看起来像”的 positive | 人工/后验标签泄漏 | 完整盲审协议；calibration 不进入 eval |
| 在 005 单 scene 继续 | 删除失败 scene、失去多 scene gate | full trainval scene-disjoint split |
| 直接启动 N2/N3/render | 没有 comparable event | 新 N1 先通过 |

## 0. H1 reject 执行摘要

### 0.1 为什么 reject

| ID | 层级 | 观察到的事实 | 能下的结论 | 不能下的结论 |
|---|---|---|---|---|
| `H1-F01` | 事件存在性 | 30 proposals：0 positive、25 negative、5 source-positive/non-event、0 same-actor pair | 冻结 proposal bank 不支持 H3 或配对因果比较 | “occupancy 一定无效”或“换几个 actor 就会成功” |
| `H1-F02` | certificate 精度 | D1 TP=15、FP=5、precision=0.75 < 0.80 | H1-CERT 按预注册 reject | 仅因 recall=0.8824 就称 certificate 通过 |
| `H1-F03` | certificate 覆盖 | D1 UNKNOWN=10/30、PASS=0、PASS coverage=0 | 当前证据无法给出足够确定的正判定 | 把 UNKNOWN 排除或并入 PASS 后重算 |
| `H1-F04` | repair 吞吐 | D2 reject=30/30、export=0、usable yield=0 | H1-PROJ 按预注册 reject；外部 rate 不可定义 | “导出集 0/0 违规，所以修复完美” |
| `H1-F05` | 数据效用 | 无 positive pair，H1 已拒绝 | H3 不触发 | 以 RGB 差分、accept rate 或 proxy 代替下游任务 |
| `H1-F06` | 高成本阶段 | H1 前置 gate 失败 | H2/render audit/blind pack 不实例化是正确停止 | “没跑 H2，所以 H1 结论不完整” |
| `H1-F07` | 统计实现 | 首版 aggregate 把 rejection 计成零违例 | 聚合 bug 已修复且不影响方法输出 | 用首版 aggregate 支持方法 claim |
| `H1-F08` | 资产/证据 | 本机 map 只有 raster PNG；base UNKNOWN 约 96–98% | lane/road support 与独立覆盖存在硬缺口 | 从 raster 或 learned occupancy 静默补成真值 |

### 0.2 冻结证据

- 正式 run：
  `/root/autodl-tmp/runs/occgs_resim/v71/V7-H1-11D/v71_v7-h1-11d__pilot-3-matched__s0__20260723T155755269940Z__cf8d5ebc/`；
- proposal-bank SHA256：
  `f8986915f8d2be0cddddfa6be86f4d2d1ece456c12bf9a962cafec78fd058cd7`；
- config SHA256：
  `cf8d5ebc1429e076fc5142aa6a759a18f54b7f3f937c8423d51505a094bc9fe3`；
- C/D1 realized trajectory 30/30 identity；
- C external hard violation 17/30：003=5/10、005=7/10、004=5/10；
- D1：15 TP、5 FP、2 FN（含 abstention）、20 FAIL、10 UNKNOWN、0 PASS；
- D2：0 accept/export、0 usable yield；
- 唯一 terminal marker：`REJECTED`。

### 0.3 逐失败点根因、卡点与复开要求

#### `H1-F01`：proposal-support failure

**观察**

- source-only eligibility 固定为 3 scenes × 2 actors；
- 每 actor 固定 P1–P5，共 30 个 proposal；
- scenario-effect 没有产生任何 `0→1` positive；
- 5 个 source-positive case 在 proposal 后成为 non-event，25 个为 negative；
- 没有 same-actor positive/negative pair。
- 后续只读 continuity 审计发现，冻结 actor 003:38、003:35、005:23 在完整连续 track 内的 world
  displacement 仅 0.88 / 0.29 / 0.76 m；原 source-only 排序偏好长、清晰 track，但没有事件相关性。

**推断**

固定横向位移只满足几何“移动过”，没有以 lane topology、corridor crossing、target-lane front/rear gap、
duration 或 interaction 定义事件。该设计与“cut-in/merge 正例”的目标错位。这是由结果和 schema 支持的
最强解释，但尚未通过 vector map 重标注，所以不能断言每个 case 的唯一失败原因。

**卡点**

- 本机缺 nuScenes map-expansion vector JSON；
- mini 三场景的真实事件上限未知；
- 没有先冻结 natural-event pool，就不能知道是 proposal family 失败还是场景本身无事件。

**复开条件**

不是改 P1–P5。必须新建 event-first 路线：先冻结 map/track 事件定义和 actor pool，证明存在预定数量的
positive/negative 与 same-actor pair，然后才允许生成候选。若 mini 事件池不足，应 reject mini pool 或
请求新数据授权。

#### `H1-F02`：certificate precision failure

**观察**

5 个 FP 全来自 scene 004 actor 8；certificate 报告 5 个 static-overlap voxels，而独立 raw LiDAR
检查为 0 points。D1 precision 为 0.75，低于冻结门槛 0.80。

**推断**

结果与 coarse voxel quantization、box-to-voxel 接触或证据层不一致相符；尚不能证明是哪一个机制，也不能
从“0 raw points”推出空间一定安全，因为 LiDAR 可能受遮挡和采样稀疏影响。

**卡点**

- `0.4m` 离散 grid 将连续几何压成二值接触；
- 单一 voxel overlap 缺少距离、置信度和观测支持；
- static/dynamic 分层仍可能受历史 sweep 和运动补偿影响；
- raw point absence 不是 free-space ground truth。

**复开条件**

在 scene-disjoint calibration pool 上比较 coarse voxel 与 motion-compensated raw sweeps 的连续
point-to-OBB/swept-volume distance；逐类报告量化、动态残影、遮挡、地图边界和标注误差。门槛必须在
冻结评估前预注册，不能用 actor 004:8 调到通过。

#### `H1-F03`：coverage/abstention failure

**观察**

三场景 base unknown 约为 97.10% / 96.04% / 97.57%；D1 10/30 UNKNOWN，PASS coverage 为 0。
两个 FN 位于 005 的 P3/P5，D1 known fraction 为 0，而 raw LiDAR 只有 3/2 points。

**推断**

当前 single/coarse observation 无法支持大部分 free/occupied 判定。两个 FN 说明“极少 raw points”
也不能自动解决判定；具体是遮挡、采样、时序或标注问题仍未知。

**卡点**

- raw LiDAR 稀疏；
- 缺 vector drivable/lane polygons；
- 多 sweep 若不做动态/ego motion compensation 会制造 ghost；
- learned completion 会提高表面 coverage，却失去独立真值身份。

**复开条件**

增加独立 evidence，而不是调低 known-fraction：官方 vector map、ego/dynamic compensated sweeps、
显式 truth tier 与 uncertainty。继续报告 PASS/FAIL coverage 和 abstention；任何 learned occupancy
只能是附加证据层，不能作为外部 evaluator。

#### `H1-F04`：repair all-reject failure

**观察**

D2 没有接受或导出任何 proposal；usable yield=0，外部 violation rate 无分母。

**推断**

当前 projection/repair 约束组合没有可用工作区，或者 proposal 全都离可行域过远。因为 0 export，无法
区分“repair 算法差”与“输入候选全不可修复”各自贡献。

**卡点**

- 没有成功样本用于 paired outcome；
- 先验 proposal 不由 lane-reachable set 生成；
- 二值 certificate 既可能过严又可能不准；
- H2/H3 都依赖 D2 产出，故被同时锁死。

**复开条件**

先通过 N1 证明事件存在，再以 lane graph/target state 生成 reachable proposal；冻结 minimum usable yield、
comparable export 数和外部 evaluator。若仍为 all-reject，直接 reject proposal/repair family。

#### `H1-F05`：metric aggregation bug

**观察**

首版 summary 把 rejection 计为零违例，使 0 export 看起来像 0% external violation。唯一允许的
`metric_aggregation_bug` 修复保留了旧 aggregate；修复后无 export 时 fail closed。修复提交为
`b82c540`，不改变 proposal、trajectory、certificate 或 D2 输出。

**防重复**

- 所有 rate 必须同时报告 numerator、denominator、rejected、unknown；
- denominator=0 时写 `undefined`，不能写 0；
- terminal decision 必须读取 comparable export 和 usable yield；
- 原始 aggregate 不覆盖，修复生成新版本并记录 migration。

#### `H1-F06`：地图资产与证据缺口

**观察**

`/root/autodl-tmp/data/nuscenes/maps/` 只有 4 个 PNG，没有 vector JSON；本机没有 Waymo/nuPlan 数据。
`/root/autodl-tmp` 约有 65G 可用空间。

**卡点**

官方 lane graph/drivable polygon 暂不可查询；不能可靠地定义 target lane、connectivity、off-road 或
corridor crossing。DriveStudio adapter 代码的存在不等于数据和许可就绪。

**复开条件**

先生成最小资产清单并取得下载授权；保存来源、许可、大小、SHA256 和 scene→map 映射。不得从 raster PNG
反推正式 lane graph，也不得静默下载全量 Waymo/nuPlan。

### 0.4 禁止重试矩阵

| 快捷做法 | 为什么无效 | 允许的替代 |
|---|---|---|
| 降 known-fraction / coverage | 把无证据改名为有证据 | 增加独立 map/raw evidence |
| UNKNOWN 并入 PASS/FAIL | 改变预注册语义和分母 | 继续三态并单列 coverage |
| 删除 S1、005 或 004 actor 8 | 后验删难例 | scene-disjoint 新 pool |
| 换 actor、方向、P1–P5 幅度 | 用结果挑 proposal | 先冻结 event definition 与 actor pool |
| 0 export 报 0 violation | denominator=0 | 报 undefined + yield=0 |
| multi-sweep 直接堆叠 | 动态物体会 ghost | ego/dynamic motion compensation |
| learned occupancy 当 GT | 方法与 evaluator 循环 | raw/map 独立 evaluator + calibration |
| GS floaters/画质当安全证据 | renderer 不是物理传感器 | GS 只在 N4 导出 |
| 先做 H2/H3/scale | 没有 comparable positive | N1–N3 先过门 |
| 重命名 OccGS 复开 | 没有解除原失败 | 新路线必须满足复开条件 |

### 0.5 可复用资产

失败不否定以下工程资产：

- coordinate contract、`WorldState`、typed label/depth；
- run contract、artifact index、terminal marker 和 fail-closed aggregate；
- object-centric GS reconstruction/renderer；
- D1/D2 接口和 `PASS/FAIL/UNKNOWN` schema；
- 冻结 H1 bank 作为负对照与回归 fixture。

复用这些资产不能继承 H1 claim；新路线必须有新 preregistration、独立 event pool 和 evaluator。

## 1. 仍直接约束 V7 的历史结论

| ID | 状态 | 对 V7 的约束 |
|---|---|---|
| `RF-05` | rejected | 合法轨迹/点或局部像素变化不等于 RGB、遮挡、source removal、depth、identity 与标签都合法 |
| `RF-06` | rejected | 局部 loss 或 mask 不保证参数/输出只在局部改变；必须测 outside、boundary、frame-0 与 held-out |
| `RF-08` | limitation | 可复现的机器 evaluator 不等于绝对物理真值，更不能替代人工 verdict |
| `RF-09` | rejected | same-scene、shared identity 或结构合法不等于人类能辨别方法收益 |
| `RF-16` | limitation | layout/trajectory controllability 不等于 action-disentangled actor physics 或数据效用 |
| `RF-18` | rejected | ReSim `exp0_no_carla` 的 E-vs-F action response 不足；V7 不得借归档重开 C1P/C1S |

其他 RF 仍完整有效，但当前 OccGS 计划不直接复用对应的 SVD projection/preference 配方。

## 2. V7 风险索引

| ID | 状态 | 风险 | 禁止的快捷修补 |
|---|---|---|---|
| `V7-RISK-01` | rejected_v71 | occupancy 已接入 11D，但 certificate precision 与 repair yield 均未过预注册 gate | 因为 occupancy 文件存在或 D2 无 export 就宣称 H1 通过 |
| `V7-RISK-02` | limitation | C0 24/24 是按效应 top-k 的机器筛选，不是用户人工评测 | 写成 human pass，或只报 top-k 隐藏 46/62 全分布 |
| `V7-RISK-03` | open_risk | L0 mask 来自 RGB 差分，outside=0 由 hard composition 构造保证 | 用 0 leakage 宣称 occupancy-guided completion 有质量收益 |
| `V7-RISK-04` | open_risk | U0 以极端 V4 为 naive 对照且没有下游任务 | 把 accept rate / RGB signal 写成优于 naive GS 或 mAP 收益 |
| `V7-RISK-05` | legacy_limitation | V7 既有 run 缺正式 manifest、resolved config 与终态标记；V7.1 新 run 已由 EV-10 fail closed | 事后猜 seed/fingerprint 或伪造 immutable provenance |
| `V7-RISK-06` | open_risk | 只覆盖 mini 三场景，S1 held-out 质量偏弱 | 先扩规模、只筛容易场景或把三场景外推为论文结论 |
| `V7-RISK-07` | interface_mitigated_v71 | 11C 已闭合 WorldState→renderer→typed-label 工程链；occupancy repair 的方法增益仍未验证 | 把 label-sync 工程通过写成 occupancy certificate/projection 通过 |
| `V7-RISK-08` | legacy_risk_mitigated_v71 | O0 坐标注释、metadata 与实际变换含义不一致；11A 已冻结显式 frame 合同 | 沿用含义不明的 `pose/T`，或在 round-trip 前计算 H1 指标 |
| `V7-RISK-09` | confirmed_mitigated_v71 | 旧 rotated-corner AABB 使 PILOT-3 动态体素量膨胀 1.72–2.83 倍；扁平语义不能诚实移除 actor | 把旧 O0 AABB 当正式安全几何，或移除 actor 后把体积恢复为 free |
| `V7-RISK-10` | confirmed_failure_v71 | 高 UNKNOWN 在 11D 导致 10/30 D1 abstain、D2 30/30 拒绝与 0 usable yield | 把 UNKNOWN 并入 PASS/FAIL，或降低观测门槛追求 yield |
| `V7-RISK-15` | architecture_mitigated_v71 | certificate detection 与 trajectory projection 若混组会混淆检测和修复收益 | D1 修改 C trajectory，或把 D1/D2 合成单一 validity 数字 |
| `V7-RISK-16` | confirmed_failure_v71 | 冻结 30-proposal bank 得到 0 个 0→1 positive 和 0 个 same-actor pair | 用位移幅度或 RGB 差分代替 scenario-effect gate，或事后换 actor |
| `V7-RISK-17` | confirmed_mitigated_v71 | 单一 `depth` 名称会混淆 expected、first-hit 与 LiDAR measured truth tier；11C 已强制分名和 sidecar | 把 expected depth 登记为 measured GT，或省略 validity/truth-tier |

## 3. 风险详情与解除条件

### V7-RISK-01：occupancy 尚未进入方法

**观察**

- `occupancy/build_scene_occupancy.py` 独立写出 per-frame grid；
- `resim/s0_trajectory_editor.py` 只检查横向运动学、yaw、actor/ego 距离和粗横向范围；
- `resim/c0_counterfactual_render.py` 改写 RigidNodes pose，但没有查询 occupancy；
- `resim/l0_local_completion.py` 用 V0/edited RGB 差分构 mask。

**边界**

O0 是有用的世界状态基础设施，但当前不能支持“occupancy 提高合法性”或“occupancy-guided completion”主张。

**解除条件**

按 `V7-H1-11` 建立统一 actor/state mapping，让 occupancy 进入 edit certificate、visibility 与标签重生，并对
matched kinematic-only/naive baselines 做非循环消融。只添加一次 occupancy lookup 或 post-hoc filter 不足以解除。

### V7-RISK-02：机器 top-k 不等于人工合法率

**观察**

- C0 全部可见 case 为 46/62 machine legal；
- 24/24 是按 mean edit effect 排序后的 top-24；
- 当前 `reviews/` 目录是机器面板与机器 JSON，没有用户填写的 verdict。

**边界**

可表述为“机器筛选 top-24 均满足当前规则”，不得表述为“24/24 人工合法”或用其估计全候选分布。

**解除条件**

先冻结 blind sample、逐项 rubric、失败优先级、JSONL schema 与聚合阈值，再由用户或指定评审者完成 verdict。
agent 不代填，也不以机器字段映射成人工答案。

### V7-RISK-03：L0 primary metric 目前是构造不变量

**观察**

hard composition 直接复制 mask 外的 edited GS，因此 outside-mask L1 必然为 0；当前 12 帧结果只验证实现遵守
公式。mask 由 RGB 差分阈值和膨胀获得，不包含 ray visibility、unknown/free 或 source footprint geometry。

**边界**

L0 只证明 local composition 工程可行。没有证据表明 Telea 改善视觉、时序、depth 或 identity。

**解除条件**

使用 geometry-derived disocclusion mask，并在有真值的 pseudo-hole 上比较 no completion、Telea 与局部生成；
primary 必须包含 inside quality、boundary、temporal、depth/instance，而不是继续调阈值追 outside=0。

### V7-RISK-04：U0 proxy 不识别数据效用

**观察**

`naive_V4` 是约 39–50 m 的强制横移负例；它被拒绝只能证明 validator 能识别一个极端错误。当前没有训练
detector、occupancy model 或 event classifier，JSON 明确记录 `u0_full_map_pass=false`。

**边界**

不能声称 OccGS 优于 matched naive GS、real-only 或提供下游增益。

**解除条件**

对相同 proposal、相同样本量和相同训练预算比较 R / R+naive / R+OccGS / R+OccGS+completion，并使用
scene-disjoint split、至少 3 seeds 和任务指标。三场景只可用于 pipeline smoke。

### V7-RISK-05：既有 run provenance 不完整

**观察**

`runs/occgs_resim/` 现有 B0/C0/L0/U0 目录未发现 `manifest.json`、`resolved.yaml` 或终态标记。B0 仍有
`config.yaml`、metrics、checkpoint；其他阶段有 JSON 报告，但不足以满足正式 run contract。

**边界**

现有数值可作为 retrospective evidence，不能声称是完整、不可变、可从 manifest 一键复现的正式 run。

**解除条件**

`V7-EV-10` 为既有证据生成显式缺失项索引；所有新 run 通过 fail-closed wrapper 产生完整协议。禁止事后补造
未知字段或覆盖旧目录。

**2026-07-23 缓解结果**

- `V7_EVIDENCE_INDEX.json` 已逐文件索引 B0/O0/S0/C0/L0/U0 的 1,610 个文件，并保留正式字段的
  `missing/unknown_not_inferred`；
- V7.1 run contract 对 run ID 复用、三层 hash、artifact bytes、summary、冲突终态标记和 optional
  `not_triggered` 分支 fail closed；
- 正式 smoke 在 commit `3590558` 上以唯一 `COMPLETE` 结束，25 项相关测试通过。

该缓解只约束 V7.1 新 run；V7 旧 run 的 provenance 缺口不可逆，仍保持 retrospective/legacy limitation。

### V7-RISK-06：场景覆盖与质量

**观察**

本机只有 mini 10 scenes 具备前向完整 sweep；feasibility 只使用 3 scenes。S1 test PSNR/SSIM 为 20.18/0.472，
明显弱于 S0/S2。

**边界**

当前结果不能外推到 trainval、长时、多相机、夜间或复杂交互；也不能只删掉 S1 后报告更好均值。

**解除条件**

H1 先在冻结三场景与 worst-case 上通过，再审计可获得的 scene-disjoint 数据。扩展必须保留困难场景分层、
真实/插值 provenance 与相同门禁。

### V7-RISK-07：标签链未闭环

**观察**

C0 已改写 RigidNodes pose 并输出 RGB/depth/rigid 分量，但尚未形成统一的 semantic、instance、2D/3D box、
occupancy 与 visibility regeneration 流水线。

**边界**

“label synchronization”当前只可称 proxy/interface 可行，不是完整传感器与标签一致性。

**解除条件**

同一 world-state record 驱动 renderer 与所有标签 writer，逐帧验证 pose、depth、mask、box 和 occupancy 共位；
对缺失/不可见标签 fail closed。

**2026-07-23 缓解结果**

- 11C 在 PILOT-3 的 V0/V1、三场景、三前向相机上生成 18 个样本和 432 个 typed sidecar；
- 独立审计验证 18/18 样本、6/6 WorldState hash、temporal identity、三相机覆盖、instance-depth z-order 与
  state-specific safety/observation/render-support 引用；
- expected、first-hit、LiDAR measured depth 分名，有限 semantic scope 和 visibility provenance 均写入 sidecar；
- S1 保留，正式 run 以唯一 `COMPLETE` 结束。

该结果只解除 renderer/label 工程接口风险；11D 之前仍不能声称 occupancy certificate 或 repair 有方法收益。

### V7-RISK-08：O0 坐标框架歧义已确认

**观察**

- `occupancy/build_scene_occupancy.py` 文件头将 grid 描述为首帧 ego-centric；
- `meta.json` 将同一产物描述为 per-frame ego-centric；
- 实际实现每帧读取 `lidar_pose/{t}.txt`，以其逆矩阵把 world box 变换到 grid，同时直接使用 sensor-local
  LiDAR 点。因此产物实际是 per-frame LiDAR-sensor grid，而不是首帧固定 grid，也不能在未审计 LiDAR-to-ego
  外参前简称 ego frame；
- DriveStudio 则以起始 `CAM_FRONT` 的 `camera_to_world` 逆矩阵定义 model frame。

**边界**

现有 O0 数值仍可作为 coarse retrospective evidence，但在显式记录 `T_grid_world`、`T_model_world`、
`T_world_camera` 并通过 world→model/grid→world round trip 前，不得用于 H1 合法性指标。

**解除条件**

`V7-H1-11A` 统一使用 `T_dst_src` 命名，修正新 schema/adapter 的 frame 声明，以 synthetic fixtures 和
PILOT-3 原始标定验证 translation、yaw、box corners、camera projection 及 checkpoint pose round trip。
旧 O0 文件不原地改写；正式 H1 evidence 产生新版本与新 fingerprint。

**2026-07-23 缓解结果**

- 11A 将 annotation/model/grid/camera/LiDAR frame 分别冻结为 world、start-CAM_FRONT、per-frame-LiDAR、
  `T_world_camera` 与 `T_world_lidar`；
- 三场景 1,679 个 actor poses 的 translation、rotation、box 和三前向相机投影 round-trip gate 通过；
- registry 跨独立进程重建 hash 完全一致，正式 run 以唯一 `COMPLETE` 结束。

旧 O0 metadata 不原地改写，故该风险仍是 retrospective artifact 的 legacy limitation；V7.1 后续模块必须引用
11A coordinate contract 和新 fingerprint。

### V7-RISK-09/10：AABB 膨胀与高 UNKNOWN 已确认

**观察**

- 在完全相同的 PILOT-3 raw annotation、grid 和 240 帧上，旧 rotated-corner AABB 相对 oriented-box
  center-inclusion 的动态体素量比分别为 003 `1.721×`、005 `2.249×`、004 `2.833×`；
- 分离 dynamic instance layers 后，base unknown 比例仍为 `97.10% / 96.04% / 97.57%`；
- source actor removal 后原体积恢复 UNKNOWN，不会恢复 FREE；edited layer 可独立 remove/insert，三场景未出现
  layer overlap；
- 缺少 nuScenes map-expansion polygons 时 road-support 与 off-road control 保持 UNKNOWN。

**边界**

11B 已消除 AABB 作为正式动态几何和扁平 layer 删除污染，但没有降低 observation sparsity。30 条可测真实
controls 的 retention 为 100%，collision/teleport 可检测负例为 2/2；然而加入 road-support 后 32 条完整
certificate 全为 UNKNOWN。这是诚实 abstention，不是 H1-CERT pass。

**后续约束**

D1 必须报告 precision、recall、abstention 和 PASS coverage；UNKNOWN 不进入 TP/FP/FN。只有独立观测或 map
证据能把 UNKNOWN 变为可判定状态，禁止通过调大 unknown threshold、把 box 当 background surface 或用 Gaussian
floaters 补 safety evidence。

### V7-RISK-15/16：certificate/projector 与 scenario effect 必须继续拆分

11B 已冻结 `scenario-effect-v1` 的纯 3D 0→1/0→0 gate、same-actor pair schema 和
`certificate-calibration-v1` 三态接口。11D 必须让 D1 逐字节复用 C trajectory，D2 才允许修改轨迹；位移 proposal
若未形成冻结的 corridor crossing、duration、gap 与 TTC/headway 条件，只能标为 non-event，不能靠命名成为
cut-in/merge positive。

### V7-RISK-17：typed depth 语义混淆已缓解

11C 把 depth 冻结为三个不同产品：diagnostic expected depth、T1 Gaussian first-hit depth、T0 LiDAR measured
depth；每个产品有独立 validity、definition、truth tier 与 artifact sidecar。独立审计确认三类各 18 个，且没有
expected-as-measured 混写。后续 export/evaluator 必须继续按产品名和 truth tier 消费，不能重新折叠成无类型
`depth`。

### V7-H1-11D：H1-CERT / H1-PROJ 预注册拒绝

**冻结事实**

- source-only eligibility 覆盖 3 scenes × 2 actors，P1–P5 共 30 proposals；S1 未删除；
- C/D1 realized trajectory hash 30/30 完全相同；
- D1：precision `0.75`、recall `0.8824`、abstention `0.3333`、PASS coverage `0`；
- C external hard violation `17/30`；
- D2：0/30 export、0 usable yield，external rate 不可定义；
- scenario-effect：0 positive、25 negative、5 source-positive/non-event，0 same-actor pair。

**裁决**

H1-CERT 因 precision 低于 `0.80` 拒绝；H1-PROJ 因拒绝全部 proposal、无 comparable export、usable yield
低于 `70%` 拒绝。按路线转向规则停止 OccGS 方法 claim，只保留 object-centric GS、WorldState、typed label、
certificate/evaluator 与 run-contract 基础设施。

**唯一修复与防重复**

首版聚合把 rejection 计成零违规，已作为 `metric_aggregation_bug` 唯一修复，旧 aggregate 保留。修复未改变
方法输出；第二版对无 export 的 rate fail closed。不得继续：

- 调低 known-evidence/coverage 门槛把 UNKNOWN 改成 PASS；
- 删除 005/S1 或 004 actor 8；
- 根据现有结果重选 actor、方向、proposal 或 event threshold；
- 用固定-pool `0/30 violation` 隐藏 D2 的 `30/30 reject`；
- 因 recall 达标而隐藏 precision fail，或把 UNKNOWN 排除后重算；
- 在当前配方上继续 H2/H3/scale。

## N1 receiver-centric cut-in final：第四轮后新增防重复项（2026-07-26）

### N1-F18：receiver branch merge 的 13/13 历史假阳性不能靠阈值微调挽回

第四版旧 parent 的 18 个 machine candidates 中有 13 个 receiver-branch merge；第四轮人工裁决表明这类
历史 branch-merge 候选均为 `FALSE_POSITIVE`。因此 branch topology、`target_incoming_count`、shared successor
或 token change 永远不能单独证明 cut-in。final v2 把该类别固定为
`ABSTAIN/UNSUPPORTED_BRANCH_MERGE_MODE`；不得为了候选数量重新放宽它。

### N1-F19：support-count 不能替代完整 receiver identity 时序

K4-012 暴露了“support 数量足够”仍可能跨 raw 帧切换接收车身份的问题。legacy fixture 的 `1→38` 与 v2
raw map 重放中观测到的 `9→1→9` 是不同窗口/枚举证据，二者都不能被静默等同为连续 receiver。final v2
要求 required raw frame 全窗唯一 non-null identity、last-post anchor 和每帧 rank/gap/path-clear；任一身份
切换必须 FAIL 或 ABSTAIN，不能被总 support count 抵消。

### N1-F20：弯道 map jitter 的 post heading 不能借由宽松窗口穿透

K4-015 证明 source/target 局部不平行或 post heading 过大时，几何横向收敛可以伪装成切入。final v2 使用
local parallel overlap、raw post-heading、累计 yaw 和 raw-only provenance；它不是针对 scene/token 的黑名单。
禁止为了保留 K4-015 或增加 PASS 数而放宽这些 hard gate。

### N1-F21：CAM_FRONT 的五帧截图不能承担角色/时序的完整证明

单相机可见性和五帧页面截断无法可靠展示 SUBJECT、RECEIVER、source/target corridor 的完整 raw 窗口。
审核 V2 因而以逐 raw-frame topdown、2 Hz signals、actor-ID switch 标注和固定 camera-unavailable 警告为主；
相机只作可选证据。看不清必须 `UNCERTAIN`，不得肉眼猜身份或通过下载未授权传感器补洞。

### N1-F22：final v2 不是旧阈值的第五次微调

本轮只吸收第四轮已完成的校准信息，变更的是事件语义和证据链：parallel-only subject body entry、独立
receiver 全时序、raw 2 Hz hard evidence、三态 first-failure、streaming worker 与 blind/debug 分离。K4 只做
固定 regression；Resource Contract V1 在任何 final scene 前失败，用户复开后的 V2 已按 N1-F25 修正为
675 scenes 并完整运行，但同样没有用于调参。后续若研究 branch merge 或新资产，必须是新的任务 ID、
预注册与 scene-disjoint 评估。

### N1-F23：共享 cgroup 的 start 合同是研究终止门，而非可绕过的工程告警

final formal 在 clean commit `7104f5c` 的 preflight 已将 runner 自身 RSS 降至 `20,705,280` bytes，仍记录
`cgroup_memory_current_bytes=1,523,929,088`，超过冻结上限 `1,350,000,000`。它在任何 evaluation scene 前
安全失败；独立裁决冻结证据并以 `REJECTED/stop_nuscenes_cutin_mining` 结束。development override 的 32/96
smoke、清页缓存或 K4 回归均不能替代正式 start 合同。禁止杀死 Cursor/Jupyter/TensorBoard 等用户服务、修改
正式阈值、截断正式 split（当时 expected 常量误写为 669，见 N1-F25）或把这次结果说成“nuScenes 没有
cut-in”；`n2_authorized=false` 保持不变。

**2026-07-26 用户复开授权**

上述 `REJECTED` 仍是 Resource Contract V1 下不可改写的历史裁决，但不再代表任务永久停滞。用户随后显式
扩大容器内存并授权继续本次 final：现场复核 `memory.max=128,849,018,880` bytes（120 GiB），原
`2,147,483,648` bytes（2 GiB）资源前提已改变。因此必须保留失败 parent 与独立拒绝裁决，同时使用新的
Resource Contract V2、全新 config fingerprint 和不可复用 run ID 恢复 scene-disjoint formal
（经 N1-F25 确认为 675 scenes）；不得覆盖或续写
V1 失败目录，也不得把 V2 成功倒写成 V1 当时没有失败。

### N1-F24：内存不足时停止并等待资源授权，不继续死磕

**新执行规则**

1. 任何正式或开发任务若触发启动/运行 stop 阈值、`RC=137`、SIGKILL，或观察到持续逼近 cgroup
   `memory.max`，立即停止启动新 batch，并尽最大可能写入结构化 `FAILED/failure.json`、最后完成 scene、
   process RSS、cgroup current、anon/file cache 与 `memory.events`；
2. 不通过反复重跑、杀死 Cursor/Jupyter/TensorBoard 等用户服务、缩短正式 split、降低证据质量、跳过
   audit、修改研究阈值或清理不属于本任务的缓存来争抢资源；
3. 把失败点、最低所需资源和恢复命令回报用户，然后等待用户开放资源；没有新的明确授权时不得自行恢复；
4. 用户开放资源后，先记录新的 `memory.max/current` 和授权时间，版本化 resource contract，使用新 run ID
   从冻结研究配置重新运行。资源合同变化只允许调整资源阈值，不允许调整 cut-in taxonomy、hard gate、
   calibration/evaluation split、抽样或人工聚合门槛；
5. 资源暂停与研究拒绝分开登记。未读取 prospective evaluation scene 的资源失败不能被写成方法精度失败，
   后续在新授权下完成的结果也不能删除或覆盖先前工程失败证据。

### N1-F25：final 的 669-scene 预期是 split 算术错误，不是 evaluation 集合定义

Resource Contract V2 的首次 clean formal run：
`/root/autodl-tmp/runs/event_first/N1-EVENT-CUTIN-FINAL-01/v71_n1-event-cutin-final-01__receiver-cutin-final-v1__s0__20260726T142634031503Z__5c8c65d7`
在 K4 regression 通过后、任何 evaluation scene 或 candidate 读取前 fail closed，错误为
`final evaluation scene 数不匹配: 675 != 669`。

独立复算表明：nuScenes official `train` 为 700 scenes；冻结的 42 个 calibration scenes 中，25 个属于
`train`、17 个属于 `val`，且没有 split 外 scene。因此 scene-disjoint evaluation 的确定数量是
`700 - 25 = 675`。`_resolve_evaluation_scenes` 已正确执行
`set(train) - set(all_calibration_scenes)`；错误只在 YAML 的 `expected_scene_count` 常量把不属于 train 的
calibration scene 也错误计入了减法。

合法修复仅为把 Resource Contract V2 配置中的 assertion 从 669 改为 675，并生成新 config fingerprint、
新 clean commit 和新 run ID。不得借此增删 calibration scene、显式挑选 evaluation scene、查看 candidate 后
改 split，或修改 taxonomy、strict gate、K4、抽样与人工门槛。失败 run 必须保留为工程契约失败，不能统计成
research reject。

### N1-F26：strict v2 在 675 scenes 上只有 1 个 PASS，不能靠人审单例或放宽规则扩池

Resource Contract V2 的 clean formal run：
`/root/autodl-tmp/runs/event_first/N1-EVENT-CUTIN-FINAL-01/v71_n1-event-cutin-final-01__receiver-cutin-final-v1__s0__20260726T142941598714Z__883fae9a`
在 commit `beee1de`、seed 0、config fingerprint
`883fae9a6514c0bff5bba8bcaf81a22c79e6d719586221596a7d4b5364c337da` 上完成 675/675 scenes。

结果为 `ABSTAIN=1,556`、`FAIL=200`、`PASS=1`，唯一 PASS 只覆盖 1 scene；冻结 machine-readiness 要求
至少 3 candidates / 3 scenes，故 parent 以
`REJECTED / stop_nuscenes_cutin_mining_too_sparse` 结束。K4、raw-only 和资源合同检查均通过，峰值 batch
process RSS 为 `337,154,048` bytes、cgroup current 为 `4,556,898,304` bytes；这次拒绝不再是资源失败。

独立稀疏终局人工包保留唯一 PASS 和 3 条 diagnostic，但人工结果不能改变数量门失败：即使唯一 PASS 被判为
TP，仍只有 1 TP / 1 scene，低于 sparse 的 3/3。禁止为了形成池而把 ABSTAIN 提升为 PASS、恢复
`receiver_branch_merge`、放宽 raw/parallel/receiver 时序门、事后改 scene 或把单例人工真实性外推为总体
precision。准确结论是“当前冻结 strict v2 的 prospective pool 过稀”，不是“nuScenes 没有 cut-in”。

<a id="detail-cross-route"></a>

## 4. 跨路线必须保留的原则

1. 先证明监督/比较对象存在，再训练或扩量。
2. occupancy、编辑、渲染和标签必须共享同一显式状态，不允许旁路文档绑定。
3. matched baseline 使用相同 proposal、scene、actor、幅度、seed 与预算。
4. top-k 只用于诊断，不替代全分布、coverage 与 worst-case。
5. machine pass 只解锁下一门禁，不自动成为 human verdict、论文 claim 或 scale 授权。
6. hard composition 的局部性与 completion 的质量是两个独立门禁。
7. 下游效用必须由任务指标证明，不能由约束 accept rate、RGB 差分或 PSNR 代替。
8. 工程失败与 research reject 分开登记；既有 provenance 缺失必须诚实标记。
9. 失败范围不能过度外推，但也不能通过改名、放宽阈值或只挑成功场景重复旧问题。

## 5. 新实验防重复检查表

- [ ] 是否明确引用了相关 `RF-*` 与 `V7-RISK-*`？
- [ ] occupancy 是否真正进入决策/状态链，而非只在磁盘上存在？
- [ ] baseline 是否 matched，而非故意构造的极端负例？
- [ ] primary endpoint 是否避免“方法规则自己定义方法成功”的循环论证？
- [ ] 是否同时报告全分布、coverage、per-scene 与 worst case？
- [ ] completion 是否测 inside quality/temporal/depth，而非只测 outside exact？
- [ ] human verdict 是否只由用户/指定评审者填写？
- [ ] run 是否有唯一 ID、resolved config、fingerprint、metrics、summary 与终态标记？
- [ ] 哪个单卡门禁失败时停止，什么条件才允许 scale？

## 6. 2026-07-26 路线转向新增失败与防重复项

### PIVOT-F01：nuScenes cut-in 没有可验证的召回率分母

**观察**

nuScenes 官方公开的是场景、样本、对象实例、类别、属性、3D 框、传感器与地图等结构；`scene.description`
是自由文本，不是事件级 cut-in 真值。官方没有发布 cut-in 场景占比、逐事件标签或可直接用于召回率计算的全集分母。
四轮挖掘和最终 675-scene prospective run 最多只能测量“冻结规则产出的候选质量”，不能测量“数据集中所有
cut-in 被召回了多少”。

**最终证据**

- `N1-F26`：strict v2 在 675 个 scene 上只有 `1 PASS / 1 scene`；
- 最终人工稀疏包即使把唯一 PASS 判为真阳性，也仍低于预注册的 `3 candidates / 3 scenes`；
- 该结果不能外推为“nuScenes 没有 cut-in”，也不能用事后放宽规则伪造召回。

**裁决**

`cut-in mining` 状态固定为 `rejected / frozen`。以后 cut-in 只允许作为已经具备重建与编辑能力后的可选演示，
不再承担数据集入口、方法定义、训练前置条件或论文成立条件。

**解除条件**

只有新的、独立的数据源提供事件级真值及明确分母，或新的任务本身不需要宣称事件召回率，才允许创建全新任务 ID
重新讨论；不得恢复当前 strict-v2 阈值调参。

### PIVOT-F02：贡献漂移——工程系统吞噬了重建与编辑研究

**观察**

过去路线的主要投入逐步变成事件挖掘、地图匹配、接收车身份、规则校准、候选审核与资源合同。它们改善了审计性，
却没有自然回答动态对象几何、连续运动表示、遮挡/去遮挡、反事实轨迹编辑或下游感知一致性。

**边界**

这不否定已形成的 WorldState、typed label、run contract、审计与人工审核基础设施；它只否定把“更好的 cut-in
挖掘器”作为 3DGS/4D 重建论文的核心贡献。

**后续约束**

新路线必须先复现公开强基线，再通过重建/编辑压力测试选择创新点。数据工程模块只能服务于冻结实验，不得重新成为
论文主任务。每个里程碑都必须说明它直接回答的重建或编辑问题。

### PIVOT-F03：未完成 exact reproduction 前禁止集成式“改进”

**观察**

`RF-05/06/08/09/16/18` 与 `V7-RISK-03/04/06/07/10/16/17` 共同表明：输入、状态、比较对象、覆盖率和真值定义
未冻结时，模块堆叠会把工程可运行误当成方法收益。AD-GS 的公开 nuScenes 协议提供了固定 scene、帧区间、预处理、
训练和评测入口，适合作为新的事实锚点。

**禁令**

在 AD-GS exact reproduction 门禁通过前，不得：

- 合并 Motion-Proj/StreetGS/OccGS 模块；
- 加 occupancy、物理约束、扩散补全、感知损失或轨迹编辑；
- 更换为自选事件场景、调低分辨率后对齐论文指标或只展示成功帧；
- 把兼容性补丁、预处理修复或运行成功表述为方法改进。

任何 unavoidable compatibility patch 必须独立提交、最小化、附 upstream diff 与消融；原始基线结果必须保留。

### PIVOT-F04：不能把“可见性建模”泛化成未观测背景已经解决

**观察**

AD-GS 的双向时间可见性用于动态对象生命周期和已观察运动建模；VAD-GS（CVPR 2026）的 visibility-aware
densification 已覆盖稀疏观测下的几何补密；DrivingEditor 支持对象删除/添加；Real2Sim 进一步展示对象级编辑与
物理交互。这意味着“增加一个 visibility 模块”或“支持平移对象”本身已经不足以构成新意。

**仍未闭合的问题**

反事实轨迹编辑会同时制造原位置去遮挡、新位置遮挡、跨相机深度排序和证据外外观。当前项目只有在下列内容形成
联合、可验证方案时才可提出方法 claim：

- 编辑诱发的显式可见性重计算；
- 未观测区域的真实性/置信度与拒绝机制；
- 跨视角、跨时间一致的背景恢复；
- 目标区预期变化与非目标区感知保持。

**防重复**

创新选择前必须把 AD-GS、VAD-GS、DrivingEditor、DGGT/ReconDrive 和当时最新工作重新做一次代码可用性与
claim 边界审计；不得把已有 visibility-aware densification 或基本对象变换重新命名为贡献。

### PIVOT-F05：资源不足时研究停机规则跨路线继续生效

`N1-F24` 是项目级规则，不属于 cut-in 专属逻辑。本轮 cgroup 为 `memory.max=2,147,483,648` bytes，轻量元数据
审计后 `memory.current` 一度达到 `2,129,526,784` bytes，因此立即停止 Python 扫描、conda 求解、下载、
预处理和训练，只继续轻量文本/文件操作。后续任何新路线任务遇到相同条件时，必须保存失败/现场证据并等待用户开放
资源；不得反复重跑、杀用户服务或偷偷缩减正式协议。

### PIVOT-F06：旧机器 smoke 证据不能替代新实例复验

迁移到 RTX 4080 SUPER 新容器后，已有环境目录和旧 RTX 4090 日志仍然存在，但它们不能证明当前 driver、CUDA、
扩展 ABI、显存与 cgroup 合同可用。M2 因此在新机器上重新执行 AD-GS forward/backward、DPT、SAM2、
Grounding DINO HF 和 CoTracker3 smoke，并为每项保存独立退出码。

后续换机或容器重建时，即使复用同一 env/checkpoint，也必须生成新的 instance 级环境证据；旧日志只能作为历史，
不能复制为当前 PASS。

### PIVOT-F07：非 login shell 的 PATH 不能作为 CUDA provenance

M2 首次当前机器采集因非 login shell 找不到 `nvcc` 提前失败，而 `/usr/local/cuda/bin/nvcc` 实际存在。环境报告
已改为显式设置 `CUDA_HOME` 并调用绝对路径，同时传播 smoke 的真实退出码。

后续自动任务必须显式记录并使用 toolkit 路径；“命令不在 PATH”与“机器没有 CUDA toolkit”必须分开裁决。

### PIVOT-F08：在线浮动模型不能进入 exact reproduction

upstream 的 CoTracker `torch.hub` 在线 `main` 与未固定 revision 的 Hugging Face 模型都会随时间变化。M2 将
CoTracker repo、离线 checkpoint、Grounding DINO HF revision 与 snapshot fingerprint 全部固定并哈希，
运行时使用 offline mode。

后续 baseline 不得在正式 run 中联网追随 `main`、latest 或未固定 snapshot；若必须升级，使用新 config
fingerprint 和新 run instance。

### PIVOT-F09：tar 页缓存与 nuScenes auxiliary 都属于资源/资产合同

并行流式扫描约 294 GB tar 时，文件页缓存计入本容器 cgroup，首个扫描实例峰值达到 `57,001,484,288` bytes。
本任务只对自己已读过的 tar 文件区间调用 `POSIX_FADV_DONTNEED`，没有全局 `drop_caches`、杀用户服务或清理
其他任务缓存。

第一次结构审计还发现 1,440 个 RGB/LiDAR payload 齐全并不足以初始化 nuScenes devkit；`map.json` 引用的
4 个静态 map masks 同样是必需资产。失败实例
`20260727T165549__e49a4e-4080s-r2` 保留为 `blocked`，补齐并哈希登记 maps 后由新实例
`20260727T180733__e49a4e-4080s-r3` 通过。以后 selective extraction 必须同时审计运行库隐式依赖的 auxiliary
文件，不能只按训练脚本直接打开的传感器路径计数。

### PIVOT-F10：AD-GS 的 PNG 输出与 SAM2 的 JPEG-only 枚举不兼容

M3 首个 scene-0230 实例完成 `prepare_raw` 和 180 张 depth 后，在 sky mask 初始化时报
`no images found`。AD-GS `scripts/nuscene/nuscene.py` 固定写 `000000.png`，其 `semantic.py` 自己也会枚举
PNG；但 Grounded-SAM-2 `load_video_frames_from_jpg_images` 只按 `.jpg/.jpeg` 扩展名建立 video frame 列表。
这不是空数据、模型失败或资源 OOM。

失败实例 `20260727T181617__scene0230__s0` 保留为 `blocked`。最小兼容性修复只在 instance work dir 中为每个
PNG 建立相同字节内容的 `.jpg` 硬链接，跨文件系统时复制原始字节；PIL 已验证按内容可无损读取，不做 JPEG 转码，
不改 Grounding/SAM 模型、box/text 阈值、mask、帧序或评测。修复后的 AD-GS patch SHA-256 为
`114c3976af2c80d1da5581b401b3a099f22a7483347fc401113c8439bc991eb9`，必须由新 M3 instance 复验。

### PIVOT-F11：COLMAP 默认全核并发会越过本机 cgroup 内存门禁

M3 第二个实例 `20260727T182247__scene0230__s0-r2` 已完成 180 张 depth/object/sky/semantic 与
138/138 flow，在 COLMAP feature extraction 阶段发现 upstream 未指定线程数，COLMAP 自动使用容器可见的
128 个 CPU threads。cgroup memory 峰值达到 `62,265,835,520` bytes，并连续两个采样超过
`memory.max=66,571,993,088` 的 90% 停止线；runner 只终止本 run 的进程组，`oom=0 / oom_kill=0`，
失败实例与部分 COLMAP 目录均保留。

这不是图像、SIFT、匹配或几何协议失败。最小资源兼容修复显式传入
`SiftExtraction.num_threads=16` 与 `SiftMatching.num_threads=16`；不改分辨率、相机、帧、SIFT 参数、
exhaustive matching 或评测。r3 的 COLMAP 已完成 138/138 图像注册与 70,933 points，阶段峰值降至
`35,117,174,784` bytes。当前完整 compatibility patch SHA-256 为
`49b4c06ecec6c30f1e80b5abf4d46970920f9d71952acbda273774d9b5b34f48`。

以后在 CPU 核数远大于内存预算的容器内运行 COLMAP，必须显式登记并发数并纳入资源合同；不得把减少图像、
降低分辨率或删相机伪装成等价的资源修复。

### PIVOT-F12：跨场景连续执行时，official render 可先于 OOM 触发 cgroup 90% 合同

M3 scene-0230 的 60k render 峰值已达到 `59,530,678,272` bytes，距离注册的 90% 停止线仅
384,115,507 bytes。M4 scene-0242 严格串行完成全部 preprocess 后，100-step train 峰值为
`59,359,428,608` bytes；随后的 official render 在第 2/138 帧连续两个采样达到 90%，峰值
`59,996,393,472` bytes，比停止线高约 81.6 MB。runner 只向本 stage 进程组发送 `SIGTERM`，
stage `rc=-15`、runner `rc=1`，`oom=0 / oom_kill=0`，没有影响其他服务。

该结果说明当前 `memory.max=66,571,993,088` bytes 对六场景连续 exact reproduction 没有足够安全余量；
“尚未 OOM”不能用来绕过预注册停止线。blocked 实例
`/root/autodl-tmp/runs/dynamic_recon/DR-M4-ADGS-6SCENE-01/20260727T235743__scene0242__s0/`
必须保留，不在同一资源合同下立即重跑，也不得降分辨率、删相机、调模型或全局 `drop_caches`。

恢复时应先提高 cgroup 内存额度，建议至少 80 GiB、推荐 96 GiB，再创建新 instance；允许复用逐文件冻结的
processed scene，但必须记录来源、哈希和新资源合同。若无法增加资源，则 M4 保持 `blocked`，不得将只有
scene-0230 的结果写成六场景论文复现。

### PIVOT-F13：processed scene 复用校验必须区分关键产物与合法空占位

RTX 3090 换机后的首个 scene-0242 复用实例
`/root/autodl-tmp/runs/dynamic_recon/DR-M4-ADGS-6SCENE-01/20260728T131533__scene0242__s0-r2-wm3090/`
在训练前 fail closed：新增递归哈希校验把 COLMAP 的合法 0-byte `created/sparse/model/points3D.txt`
占位文件误判为损坏。旧实例以 `blocked` 保留，没有修改 processed scene 或启动训练。

合法修复允许 COLMAP 非关键占位文件为 0 bytes，同时继续强制 `database.db`、`cameras.txt`、
`images.txt`、`colmap.ply` 和所有训练直接消费的 image/depth/mask/flow/meta/point cloud 非空；
复用后必须重新运行独立 processed audit。修复后的新实例 output fingerprint 为
`32bf9ccaa108273b69286625a0c7aaacb04fd9d76f243daff976206d0b7ef4f6`，138/138 registered images
审计通过。不得删除空占位文件、伪造非空内容或因此重跑昂贵预处理。

### PIVOT-F14：容器实例重建必须与 OOM/方法失败分开

M5 首个正式 DGGT 实例
`/root/autodl-tmp/runs/dynamic_recon/DR-M5-DGGT-NUSC-01/20260729T094923__native-nusc-s0-wm3090/`
在 `env_torch` 下载期间停止。日志没有 stage 终态、OOM 或 `oom_kill` 增量；当前容器 PID 1 的启动时间为
2026-07-29 13:13:43 +08:00，晚于日志停止时间，故裁决为外部容器实例重建，而不是 DGGT 精度、显存或方法失败。

旧 run 与旧 controller 已原子标为 `blocked`，部分环境移动到
`/root/autodl-tmp/envs/dggt.interrupted-20260729T094923/`，没有覆盖或删除。恢复使用新 run
`/root/autodl-tmp/runs/dynamic_recon/DR-M5-DGGT-NUSC-01/20260729T133346__native-nusc-s0-wm3090/`；以后同类中断必须先核对 PID 1 启动时间、stage marker、launcher 与 OOM 证据，再决定是否创建
新实例，禁止把 stale `running` 当成任务仍存活。

### PIVOT-F14B：pointops2 的 PEP 517 build isolation 没有继承已安装 torch

恢复实例完成 Python 3.10、torch 2.4.1 和全部 requirements；resolver 最终选择
`rerun-sdk 0.23.1 / opencv-python 4.11.0.86 / numpy 1.26.4`。随后 upstream pointops2 执行普通
`pip install .` 时，PEP 517 临时 build env 在读取 setup requirements 阶段报
`ModuleNotFoundError: No module named 'torch'`。正式 stage `rc=1`，峰值 cgroup memory
`16,839,843,840` bytes、GPU 0 MiB，`oom=0 / oom_kill=0`。

正式 blocked 证据：
`/root/autodl-tmp/runs/dynamic_recon/DR-M5-DGGT-NUSC-01/20260729T133346__native-nusc-s0-wm3090/`。
checkpoint、native inference 和 common-observation metrics 均未启动；没有在同一实例事后加入
`--no-build-isolation` 覆盖失败。该结果属于明确 upstream packaging blocked，不是 DGGT 质量、显存或方法裁决，
并按权威计划第 15.1 节满足继续 M6 的替代前置证据。

### PIVOT-F15：AD-GS 冻结 pseudo ID 与 checkpoint 都不能支持单对象编辑

M6 直接审计训练前冻结的 `semantic/mask_*.npy`。按 camera-local ID 统计，六个官方场景最长支持帧为
`1 / 6 / 1 / 1 / 2 / 1`，全部低于预注册 `≥20/60`；processed scene 也没有冻结的 vehicle track artifact。
与此同时，六个 60k checkpoint 的 `point_cloud.ply` 均只有二值 `obj∈{0,1}`，没有持久 instance ID。

正式证据：
`/root/autodl-tmp/runs/dynamic_recon/DR-M6-STRESS-01/20260729T145645__identity-audit-s0-wm3090/`。稳定失败
`persistent_object_identity_unavailable` 在 6/6 scenes 重复。对象编辑、pseudo-hole 和噪声行全部保留为
`ABSTAIN`，0/12 object slots 没有从 coverage 分母删除。禁止在看到 checkpoint/场景结果后用几何 Hungarian
轨迹回填 M6 baseline；这类重关联只能作为新方法候选，并必须先过 novelty。

### PIVOT-F16：instance-aware 与 driving edit 已被 2025–2026 工作直接覆盖

M7 只沿决策表考察 A“可编辑运动表示与轨迹不确定性”。重新核对官方来源后：InstDrive 已用 SAM pseudo masks
学习动态驾驶场景 2D/3D instance identity；Director 已做 4D Gaussian identity consistency；OmniRe 已用
actor scene graph/canonical vehicle nodes 做仿真；HorizonForge 与 G²Editor 已覆盖车辆轨迹操作、删除和遮挡区
恢复。

正式 evidence：
`/root/autodl-tmp/runs/dynamic_recon/DR-M7-HYPOTHESIS-01/20260729T145748__novelty-audit-s0-wm3090/`。候选的持久身份、actor-centric Gaussian binding、时序一致性和轨迹/对象编辑
核心机制均为 direct overlap；confidence/ABSTAIN 是评测与安全护栏，不构成独立技术 delta，剩余差异只是
AD-GS 适配工程。因此 M7=`rejected`，不注册事后 primary endpoint；M8/M9 均
`rejected / not authorized`，不得通过改名、挑场景或把 0 coverage 写成改进继续。

### PIVOT-F17：DGGT 扩展构建必须同时固定 compiler、headers 和 Python 依赖上界

V2 M1 表明，“已安装 torch cu121”并不足以证明 CUDA extension 可构建。宿主只有 CUDA 11.8
toolkit，会在 pointops2 编译时与 torch 2.4.1+cu121 硬失配；只补 `nvcc` 又会缺
cusparse 等 headers。正确合同是在前缀环境固定 NVIDIA CUDA 12.1 compiler/runtime/headers，
传播 `CUDA_HOME/CPATH/LD_LIBRARY_PATH`，再按 upstream `python setup.py install`。

同一里程碑还暴露了浮动 Python 树的独立风险：transformers 5.x 使用 torch 2.4.1 未提供的
DTensor API，diffusers 0.39 触发 torch schema 不兼容。最终固定
`transformers 4.48.3 / tokenizers 0.21.0 / diffusers 0.32.2 / numpy 1.26.4 /
opencv-python 4.11.0.86 / rerun-sdk 0.23.1 / flow-vis 0.1`。

对应 blocked runs 为
`20260802T120027Z__native-nusc-s0`、`120943Z__...-r2`、`122213Z__...-r3`、
`122904Z__...-r4`、`124347Z__...-r5`。这些失败是构建/依赖证据，不能推断 DGGT
方法质量。

### PIVOT-F18：原生阶段完成不应被后续评估依赖失败覆盖

M1 r6 已完成 18/18 1-view 和 18/18 3-view，但 common evaluator 导入 AD-GS 冻结
`loss_utils` 时因 `flow_vis` 未安装而 blocked。原生输出本身未损坏，但主 terminal 已转为
blocked，禁止为了“好看的 done”改写。

恢复方式是新建 r8，对 r6 `native_summary.json/metrics.json`、每个 stage 和输出哈希做
fail-closed 引用后只执行 common diagnostic。r7 中重试封装自身的 `KeyError` 也以新的
blocked run 保留，再由 r8 完成。后续所有 multi-stage run 必须把“可复用的完成阶段”与
“整个 instance 的 terminal 终态”分开；重试不得修改旧 terminal。

### PIVOT-F19：nuScenes devkit 反向索引与磁盘 metadata 不是同一 schema

M2 r1 直接读取官方磁盘 `sample.json` 时发现其中没有 `anns`；该字段是 nuScenes devkit
初始化后才注入的反向索引，不是原始 JSON 合同。正式适配器改为流式扫描
`sample_annotation.sample_token`；由于这个外键非唯一，不得用单值 dict 覆盖同一 sample
的多个 annotation。`ijson` 还必须以 `use_float=True` 读取，否则 Decimal 会污染严格 JSON
运行合同。

同一里程碑还表明，“时间最近”不足以建立 raw annotation 到 camera sweep 的真值映射。r4
中 scene-0242 boundary actor 命中更近的 sweep，但 sweep 所属 `sample_token` 与 raw 2 Hz annotation
不同，因此在 QA 前即 blocked。正确规则是先限定 exact sample token，再在候选内最小化
timestamp delta；正式 r5 达到 `4356/4356` exact mappings。后续不得仅按文件名或时间
猜测 raw/processed/render 映射。

### PIVOT-F20：CUDA 扩展 import 成功不等于包含当前 GPU 架构

M3 的 DriveStudio 环境能正常 import `gsplat` 和 `nvdiffrast`，但旧二进制没有 RTX 3090 的
SM 8.6 kernel：前者在 SH rasterization 报 `no kernel image`，后者在 EnvLight 路径报 CUDA 209。
只做 import smoke 无法发现此类错误。恢复时分别固定官方源码 commit，以
`TORCH_CUDA_ARCH_LIST=8.6+PTX` 重建，并执行真实 CUDA forward/backward；旧 `.so` 先备份，
没有修改算子语义或模型配置。

对应 blocked runs 为 r4、r6、r7；正式 binary SHA-256 为 gsplat
`6d7c8e5a...dd6131`、nvdiffrast `0d18f767...96499`。以后 CUDA 扩展 readiness 必须包含
目标 GPU 上的实际 kernel forward/backward，不能只看包版本和 import。

### PIVOT-F21：训练完成 checkpoint 与累积式 post-render 必须分开裁决

M3 r8 的 30k 原生训练已保存 `step=30000` checkpoint，但上游随后将 588 个 full-render 结果累积在
内存中；在 `577/588` 时 cgroup memory 连续两次超过 90%，资源守卫发送 SIGTERM。`oom=0 /
oom_kill=0`，checkpoint 字节数与 step 完整。r8 仍保持 `blocked`，不得改写为 done；r12 通过新的
不可变 run 对 checkpoint step/bytes/hash 和原失败 terminal 做窄范围复核，再执行流式 27-image
edit smoke 完成 M3。

同一恢复链还发现，正式训练会把某个非目标 rigid model 的全部 Gaussian 裁剪掉。token、dataset column
和 model index 仍是一一映射，但 checkpoint slice 为空。registry v2 因此将其明确标成
`unavailable_empty_checkpoint_slice`，同时对正式选中 actor 继续要求非空。禁止为了全 registry 看起来
完整而伪造 slice，也禁止因一个非目标空 slice 丢弃 23 个真实非空映射。

### PIVOT-F22：外层 timeout 不会自动回收独立 session 的 GPU 子进程

M4 controller 用 `subprocess.Popen(..., start_new_session=True)` 隔离正式渲染，使 SSH/tmux 断开不应
误杀长任务；相应地，用外层 `timeout` 调试 controller 时，SIGINT 只终止父进程，子进程会以 PPID 1
继续占用 GPU。`debug_controller_s0_r5` 复现了该行为；残留子进程通过已核实的精确 PGID 发送 SIGTERM
回收，GPU 从约 `8.1 GiB` 回到 `0 MiB`，没有终止用户服务。

以后不得用外层 timeout 探测会派生独立 session 的 controller。正式运行应直接由 nohup/tmux 托管，
同时监控 controller PID、child PID、terminal 和 resource.jsonl；确需中止时必须核实 process tree 后
显式回收 child process group。`r5/r6` 的 running terminal 保留为中断证据，不改写成 done。

### PIVOT-F23：SE(3) 一致性容差必须覆盖 float32 往返误差

M4 单帧 r1 的 actor transform 先由 checkpoint float32 tensor 变换，再写入 JSON 并读回，最大平移误差
略高于 `1e-6 m`；其余 15 项检查均通过。把该值当几何失败会制造假阴性。协议在查看正式全量结果前
固定为 `1e-4 m`，r2/r3 冒烟通过，正式 196 帧实测最大误差为
`3.814697265625e-06 m`，rotation/size/canonical drift 均为零。容差变更只反映数值精度，不降低
1 m 编辑幅度，也不得据此为真正的轨迹偏差放宽门禁。

### PIVOT-F24：冻结 heldout 资源门失败不能靠事后更换 renderer 或提高阈值挽回

A3 R1 在结果前冻结 `12,288 MiB` PyTorch allocated GPU ceiling。三条完成全部 R0/R1 指标计算的只读路径
`r2/r4/r5` 分别达到 `14,241.777 / 14,244.924 / 14,241.399 MiB`；wall、cgroup、run bytes 与 OOM delta
均通过。资源审计、CPU checkpoint staging、Rigid quota device 兼容和逐 view `trainer.info` 释放都没有改变这一
单 view 峰值。继续把 renderer 改为 `packed=true`、分块/降分辨率，或把 ceiling 提高到观测值以上，会在看到结果后
改变 source-render 路径或预算，不再是预注册评测。

r5 的资源无效 diagnostic 也不能救回方法：S-B depth-order violation 从 `0.915792` 降至 `0.908173`，但
non-target 与 original-global RGB MSE 都按 exact comparator 严格变差，故为 `tradeoff_non_dominated`。正确分层是：
r5 run 保留 `blocked`，R1 方法臂登记 `rejected_resource_gate_and_diagnostic_tradeoff`，A3 任务以负结果 `done`；
生产路由回退到 R0/D2 immutable exact alias。以后若研究 packed/分块渲染，只能在 A4 作为新的部署因子另行冻结，
不得倒写 A3 heldout 结论或解锁 R2–R4。

### PIVOT-F25：部署 profile 必须区分传感器原始尺寸与 checkpoint 原生加载尺寸

A4-P0 v1 在新测量前把 scene-0230 的 nuScenes 传感器尺寸 `1600×900` 冻结为“原生分辨率”。formal r1 实际
完成 2 次 warm-up 与 9 次 measured render 后，11 行全部为 `800×450`，因此只在 finalize 的
`native_resolution_exact` audit 失败；资源、输入 hash、无训练/无 checkpoint、同步矩阵与无 torch resume audit
均通过。source config 事后审计确认三路相机已冻结 `data.pixel_source.downscale_when_loading=[2,2,2]`，故当前
checkpoint 的模型原生加载/渲染尺寸本来就是 `800×450`。

不得修改 v1 或把 r1 改写为 done，也不得用 r1 性能数字关闭 P0。正确处理是保留 r1=`blocked`，冻结其 protocol、
manifest、runtime stage/rows、resource audit 与 terminal hash；创建 v2 只纠正分辨率语义，再从新目录完整重跑。
后续协议必须同时记录 sensor resolution、source-config downscale 与 model-native render resolution；“native”一词
不能在这三层之间无来源转换。该纠错不授权降低分辨率、切换 renderer、改变资源 ceiling 或开启 P1/P2/P3/P5。

### PIVOT-F26：checkpoint state key 不能冒充加载后的模型运行时属性

A4-P5 formal r1 已通过 9 项输入审计并生成 `14,729-byte` reference-only deployment registry，但 fresh DriveStudio
worker 在 checkpoint 成功加载后读取 `RigidNodes.points_ids` 时报 `AttributeError`。源码事实是：checkpoint
`state_dict` 以 `points_ids` 为序列化键，`load_state_dict` 将它弹出并写入运行时属性 `self.point_ids`。两者语义
相关但接口层不同；直接把 checkpoint 键拼成对象属性会使恢复链在资产已经物化后失败。

r1=`20260809T155209Z__a4-p5-registry-resume-s0-r1` 保留 `blocked`，terminal SHA=`61d30a11...773e`；其已生成
registry SHA=`e48bccdf...9039d` 不覆盖。修复 `0e899b2` 只通过 fail-closed helper 读取 runtime `point_ids`，并用
两条测试分别锁定有效属性和拒绝 `points_ids` 别名；没有修改 P5 protocol、输入、资源 ceiling 或审计口径。
新目录 r2 以相同 registry SHA 完成 14/14 audits，证明问题属于 runner/runtime contract，不是资产或方法失败。

以后凡从 checkpoint 结构推断 live module API，必须同时核对 `state_dict` 保存端、加载端赋值和加载后的真实对象，
并用回归测试锁定层级；旧失败 run 维持 `blocked`，不得因修复后的新 run 成功而倒写。

### PIVOT-F27：最小预注册剪枝臂失败后不能事后补更小 fraction 或放宽质量门

A4-P1 在结果前固定 source/b05/b10/b20 四臂，并要求 global、actor、boundary 与 non-target 的全部 safeguard 同时
通过。canonical r1=`20260809T165058Z__a4-p1-contribution-prune-s0-r1` 完成 36-view contribution、三臂物化、
四臂 57-view 质量和 9-view runtime，21/21 audits 全 true，资源门通过；因此它不是工程或资源 `blocked`。

最小候选 b05 已将 checkpoint 减少 `23,881,368 bytes`，全部 row/invariant/reload/count 审计 exact，但 global
occupied PSNR、global PSNR 与 non-target PSNR 分别退化 `0.117684/0.110926/0.125462 dB`，超过冻结的
`0.10 dB` 上限；b10/b20 分别失败 12/15 个端点。局部 actor/boundary 指标的保持或改善不能覆盖全局与非目标区
失败。运行时 P50/FPS 也非随 fraction 单调，且 filesystem cache 未控制，只能报告，不能充当事后选择理由。

正确裁决是 P1 experiment=`done`、method=`rejected_quality_or_integrity_gate`、生产资产 exact fallback 到 source。
不能在看到 b05 失败后新增 b01/b02、改排名视图、放宽 `0.10 dB` 或只保留通过的局部端点；这些都属于新的预注册
实验，而不是当前 P1 的修复。该负结果只约束 scene-0230/seed-0/冻结视图矩阵，不外推为所有贡献度剪枝均失败。

### PIVOT-F28：顶层 `named_parameters()` 不保证覆盖普通映射中的子模型参数

A4-P2 formal r1=`20260809T174337Z__a4-p2-mixed-precision-s0-r1` 已成功完成 10-field checkpoint conversion、
source/candidate 57-view quality、两臂 runtime、aggregate 与 no-torch resume；aggregate 也按冻结门选择 mixed arm。
但 finalizer 的 `checkpoint_reduction_and_runtime_matrix_exact` 唯一失败：账本只调用
`trainer.named_parameters()`，而 DriveStudio 把 Gaussian 子模型保存在普通 `trainer.models` 映射中，并未注册成
顶层 `ModuleDict`。因此账本只看见 LPIPS 等 `9,883,392 bytes`，把 source/candidate 错记为相同总量且没有 FP16
bucket，尽管候选 checkpoint 已实际从 `578,819,674` 降到 `432,111,754 bytes`。

这属于 evidence collection defect，不是 conversion、质量、资源或方法失败。r1 保留 `blocked`，terminal SHA=
`5ef3dab60ff934af19ff547c0f7e7cd0fe74b83000476888a630341ee39474c0`；不得手改 r1 runtime stage 或把它倒写为
done。修复 `dcf2822` 显式遍历 `trainer.models` 中每个 module 的 parameters，并按 Parameter identity 去重；回归 fixture
故意使用未注册的普通映射，锁定 `models.Background._scales` 等字段必须进入账本。协议、字段、阈值、renderer 与
selection 均未改变。

新目录 canonical r2=`20260809T174850Z__a4-p2-mixed-precision-s0-r2` 完成 19/19 audits，正确记录 source/candidate
persistent bytes=`394,641,424 / 247,936,208` 与 candidate FP16 bucket=`146,705,216 bytes`，并选择 mixed arm。
以后审计复合模型时，必须同时核对容器是否已注册为 `nn.Module`、顶层 traversal 覆盖范围与逐字段预期集合；
证据账本缺失不得用 checkpoint 文件变小或 runtime 成功来推断补齐。

### PIVOT-F29：无卡实例必须以 cgroup 内存为资源合同，不能读取宿主机 `free`

V3.2 S0 在 AutoDL 无卡开机模式中观察到 `free -b` 暴露宿主机约 810 GB 内存，但
`/sys/fs/cgroup/memory.max=2,147,483,648`，实际只有 2 GiB。CoIn 的完整 partial-clone checkout 在大量 blob
物化时把 `memory.current` 推近上限且长时间无进展；继续并发创建环境或校验大权重会把可回收 page cache 与真实
匿名内存混在一起，增加无意义的 OOM 风险。

正确处理是停止该 checkout，把残留移到仓库外备份，并改用 `--no-checkout --filter=blob:none` 固定 commit/tree；
所有下载使用流式落盘，依赖安装和 hash 校验串行执行。GPU、VRAM 与 driver 则必须在有卡重启后重新审计，不能把
无卡模式的 `nvidia-smi: permission denied` 写成硬件不存在。后续任何资源 preflight 都必须同时记录
`memory.max`、`memory.current`、`memory.events` 和数据盘余量；宿主机 `free` 只作诊断，不作授权。

### PIVOT-F30：原子发布目录不能把 `.partial` 绝对路径写进 manifest

S1 prompt preparer 首次在 `s1_prompt_v1.partial` 内生成绝对 `video_dir`，发布时将目录改名为
`s1_prompt_v1`，manifest 内部却仍指向已经不存在的 `.partial` 路径。SAM2 因此把它判定为既非 MP4 也非 JPEG
目录，r1 在真实 GPU 启动后立即失败。修复是 manifest 只保存相对 `sam_inputs/...`，消费者相对 manifest
父目录解析；原子 rename 后重新验证每个目录和 JPEG 链接。任何会整体 rename 的 staged asset 都不得在内部保存
staging 绝对路径。

### PIVOT-F31：SAM2 `reverse=True` 默认从最早 prompt 开始，可能合法地产生零帧

首次双向传播实现只设置 `reverse=True`，但官方 predictor 默认 `start_frame_idx=min(condition frames)`；train-only
block 的首个 prompt 常在 local frame 0，反向 processing order 因此为空。调用成功和进度条 `0it` 不能证明反向
覆盖。正确做法是显式用 block 内最晚 prompt 作为 reverse start，并按每个 object 自己的 prompt frame 过滤输出；
r5 中实际产生 13 个 prompt 之前的 mask，才构成双向证据。

### PIVOT-F32：mask QC 必须在同一像素坐标系比较

r4 将 SAM logits 从源图 `1600×900` resize 到模型原生 `800×450`，却直接与源图坐标的逐帧 3D box 比较，
造成 `235/263` 假拒绝。修复后 box 按 exact x/y 比例映射到 800×450，r5 为 `212 accepted / 51
fail-closed`，其中 43 个是近空 mask。以后任何 IoU、centroid、boundary 或 area ratio 门禁都必须同时记录 source
size、target size 与变换；不同尺度间的数值不得直接进入裁决。

### PIVOT-F33：大规模 Gaussian 重复索引累加不得使用逐元素 `np.add.at`

S1 r2 在每个视图的数百万 ray/Gaussian intersections 上多次使用 `np.add.at`，CPU 单核成为瓶颈；同时该版本
仍缺计划要求的 negative views、depth-consistency rate 和 boundary score，因此保留 250 个 mask 后以 exit 143
终止，不得作为完成证据。r5 改用 `np.bincount(minlength=total)` 和向量化 view count，263-view lift wall
`770.733s`，并保存完整 posterior schema。研究 runner 必须输出阶段进度；“CPU 持续运行”不能替代复杂度审计。

### V6-F06：数据 adapter 的运行环境必须覆盖写出阶段依赖

R3 首次正式目录
`20260821T085802Z__support-deviation-s20260821-r1` 在完成 scene-0242 图像、标定与点云聚合后，
于 `store_ply` 写出阶段因主环境缺少 `plyfile` 失败。该 run 保留 `failed` terminal，不改写为完成；
失败发生在任何 checkpoint 推理、质量读取或确认集读取之前，因此不是方法负结果，也不是 GPU/数据资源不足。
修复只把冻结的 adapter 命令路由到已经具备 AD-GS 依赖的 `/root/autodl-tmp/envs/adgs/bin/python`，
不改变数据分区、场景、checkpoint、support 假设、指标或门槛。以后环境 readiness 必须覆盖 adapter 的最终序列化依赖，
不能以脚本启动和主体循环成功代替端到端环境兼容性。

### V6-F07：只读 renderer 的数据 loader 仍可能强制读取训练期辅助字段

R3 第二次正式目录
`20260821T090109Z__support-deviation-s20260821-r1` 已完成 scene-0242 adapter 和 StreetGS 全部冻结渲染，
随后 AD-GS `Scene` 构造因 adapter 没有 `depth/000000.npy` 失败。V4 adapter 按设计只生成图像、语义、天空、
位姿与点云，而 AD-GS nuScenes loader 即使在只读 checkpoint 渲染时仍无条件加载每张训练期 depth；源码检索确认
`gaussian_renderer` 不读取 `viewpoint_camera.depth`，R3 指标也只使用 DriveStudio 导出的真实稀疏 LiDAR。
因此该 run 保留 `failed` terminal，不改写为方法负结果。

修复是在每个不可变新 run 的 adapter 内生成全零 float32 depth 占位文件，并写出独立 audit，明确标注
`loader_field_only=true`、renderer/指标不消费以及真实几何证据来源；不修改 AD-GS checkout、checkpoint、
开发/确认分区、指标或假设。以后复用训练代码做只读渲染时，必须区分 loader 的强制 schema 字段与实际计算依赖，
占位值只能用于经源码证明不被实验结果消费的字段。

### V6-F08：通用场景点云不能替代 checkpoint 对应的 object-aware loader 资产

R3 第三次正式目录
`20260821T090552Z__support-deviation-s20260821-r1` 已再次完成 scene-0242 adapter 与 StreetGS 渲染，
AD-GS 随后在 `readnuScenesInfo` 对 `obj_id[..., 0]` 索引时失败。通用 V4 adapter 生成的 `points3d.ply`
只有 xyz/rgb/time，没有 AD-GS 训练 adapter 的 `obj` property；即使 checkpoint 加载随后会覆盖 Gaussian 初始化，
`Scene` loader 仍先强制构造 object-aware point cloud。该 run 保留 `failed`，不是方法或资源负结果。

修复不伪造 object id，而是把同一场景、同一冻结 checkpoint 训练时使用的
`adgs_processed_v4/train/<scene>/points3d.ply` 复制进新 development adapter；绑定前验证 PLY header 的
`property float obj`，记录通用点云 hash、冻结训练点云路径/hash 和复制后 hash。开发图像、位姿与分区继续来自
新 adapter，checkpoint 与指标不变。以后给冻结模型换 evaluation camera 集时，应复用训练时与模型结构耦合的
初始化/registry 资产，只替换经协议允许的观测与相机字段。

### V6-F09：lazy camera 偏移前必须整体迁移设备，不能只新建 CUDA 外参

R3 第四次正式目录已成功完成 scene-0242 的 object-aware `Scene` 构造与 checkpoint restore，
在首个 novel camera 的 `full_proj_transform` 计算处失败：冻结 AD-GS 配置启用 `lazy_load_to_gpu`，
原 camera 的 `projection_matrix` 留在 CPU，而 R3 worker 直接把新 `world_view_transform` 建在 CUDA，导致 BMM
设备不一致。该 run 保留 `failed`，资源峰值远低于门槛，不属于方法或资源负结果。

修复在任何偏移或编辑前调用上游 `Camera.cuda()`，一次性迁移 image/depth/semantic/sky 与全部变换矩阵，
再深拷贝和修改外参；这与上游 `Camera.to()` 合同一致，不改变相机数值、checkpoint、renderer、指标或门槛。
以后 lazy evaluation path 必须把 camera 作为一个设备一致的整体处理，不能只迁移新创建的 tensor。

### V6-F10：structured array 拼接后必须使用字段索引，不能依赖 recarray 属性

R3 第五次正式目录
`20260821T091503Z__support-deviation-s20260821-r1` 已完成 2 场景 × 2 frontend 的全部 adapter、
checkpoint restore、横向/前向/actor-edit 渲染和 worker audit，共 80 个 render；汇总阶段把 `np.rec.fromarrays`
结果经 `np.concatenate` 拼接后得到 structured `ndarray`，代码仍以 `values.y/values.x` 访问字段，触发
`AttributeError`。该 run 的 terminal 保留 `failed`，不得倒写为 done；渲染与 checkpoint 证据本身完整。

修复统一改用 `values["y"]` 等 structured-array 字段索引。为避免无意义重跑冻结 renderer，建立独立
analysis-only recovery run：先逐个重算全部 render SHA、核对每个 worker 的 render count、checkpoint 前后 hash、
无训练/无确认集 audit 与 adapter 分区，再从只读失败目录计算指标；新目录记录原 run/commit/terminal hash、
分析 commit 与聚合 content hash，原目录不修改。以后 post-render 工程失败可复用已验证的不可变证据，
但必须新建 terminal 和完整 provenance，不能在原失败目录续写。

### V6-F11：Gaussian `source_indices` 是 chunk-local 身份，不能跨模型直接判全局唯一

R5 v0 正式目录 `20260821T094101Z__provenance-s20260821-r1` 已生成 provenance package，并达到
chunk=`24/24`、actor=`23/23`、primitive=`1,267,870/1,267,870` 覆盖；但 raw `source_indices` 的全局 unique
count 只有 `1,095,606`，因此 100% identity gate fail-closed。原因是 StreetGS Background 与各 Rigid model
分别维护局部 source-index 空间，actor 数值可与 Background 重叠；这不表示 primitive 丢失。

v0 run 与 config 保持 failed/frozen。v1 不放宽全局唯一门，而把 primitive identity 改为
`(chunk_id, source_index)` 复合键：先要求每个 chunk 内 source index 唯一，再要求 chunk id 唯一，二者合取形成
全局唯一身份。provenance 字段、source-type 分离、覆盖率和无 confirmation/训练边界均不变。以后任何跨 chunk
primitive registry 都必须显式携带命名空间，不能把上游局部索引误当全局主键。

### V6-F12：R7 source manifest 必须绑定实测完整 SHA，不能使用手工摘录值

R7 首次正式入口在创建 run 目录与读取 render payload 前 fail-closed：预注册配置中的 R3 recovery manifest
SHA 使用了手工摘录值 `e866dae3b84c...`，而冻结文件实测 SHA 为
`e866dae35a4ff17fb75791ff395f45504f8f779d57e75121354b8be388595acc`。两者不一致，因此程序按 source
identity contract 立即拒绝；没有 pseudo-hole、proposal、质量指标、训练或 confirmation 读取，也没有可写成
`rejected` 的方法结果。

修复只替换为 `sha256sum` 实测的完整 source manifest SHA；R7 hypothesis、cohort、hole 定义、verifier 阈值、
decoy、gate 与资源合同均不改变。以后冻结跨 run source identity 时，必须从机器可读 artifact 或现场哈希复制完整值，
不得从状态文档里的短写或人工记忆还原。

### V6-F13：R7 verifier 必须显式归一 frontend 的 singleton-channel 维度

R7 首个有 run 目录的正式实例
`20260821T100107Z__oracle-missing-world-s20260821-r1` 在第一个 pseudo-hole mask 构造时失败：StreetGS/AD-GS
冻结渲染对 depth/dynamic opacity 保留了 `H×W` 与 `H×W×1` 两种合法 singleton-channel 表示，初版代码无条件
取 `[...,0]`，把二维 opacity 错切成长度 `H` 的向量，随后与 `H×W` mask 广播失败。该目录 terminal 保持
`failed`；尚未形成完整 denominator、gate 或方法结论。

修复新增唯一的 plane normalization：只接受 `H×W`、`H×W×1` 或 `1×H×W`，统一返回二维数组，其他形状
继续 fail-closed；所有 depth/semantic verifier 和 usable-region 路径共同使用它。hypothesis、pseudo-hole、decoy、
阈值和 source render 均不改变。

### V6-F14：actor/disocclusion pseudo-hole 不能假设每个 frontend 都导出非空 dynamic opacity

R7 第二个有 run 目录的正式实例
`20260821T100228Z__oracle-missing-world-s20260821-r1` 已完成 scene-0242 与部分 scene-0048 cases，随后在
scene-0048/StreetGS 的 disocclusion mask 上 fail-closed：该冻结 renderer 的 `dynamic_opacity` 在此帧为空，初版
mask 得到 0 pixels，低于预注册 `256` denominator。该目录保持 `failed`，不汇报不完整 gate。

修复不降低 minimum pixels，也不读取 confirmation，而是使用本实验本来就冻结的 `base` 与
`actor_remove_all` 配对渲染：以 RGB 变化或同 frontend 可比 depth 变化，加上可用的 dynamic opacity，形成 actor
evidence；disocclusion 对其确定性膨胀，actor-removal hole 使用原 evidence。这样 StreetGS/AD-GS 都以“实际 actor
edit effect”而不是可选 buffer 的存在作为 denominator，其他 hole、verifier、decoy 与 gate 不变。

### V6-F15：actor pseudo-hole denominator 必须先满足非空 actor-effect 证据

R7 第三个失败目录 `20260821T100348Z__oracle-missing-world-s20260821-r1` 在使用 RGB/depth actor-edit evidence 后，
scene-0048/StreetGS/frame-52 的 disocclusion mask 仍严格为 0。R3 冻结 `ACTOR_EDIT_EFFECTS.jsonl` 证实该 frontend
在 scene-0048 的 frame 52/57 上，`actor_remove_all`、translate、time-shift 的 global effect 与 nonzero fraction
全部为 0；因此这四个 actor-removal/disocclusion Cartesian cases 没有可构造的真实 pseudo-hole denominator。

这不是 oracle verifier 的负结果。预注册 `WS-V6-H-R7-001` 因 32-case minimum experiment 无法实例化而标记
`invalidated_pre_gate`，不倒写 gate。替代假设 `WS-V6-H-R7-002` 在 proposal 评分前冻结 eligibility：route/side
仍保留全部 16 cases；actor/disocclusion 要求原冻结 evidence 至少 256 pixels，不合格的四项显式记录 structural
ABSTAIN；剩余 28 oracle + 28 decoy 才进入完全相同的 verifier/bake 门禁。

### V6-F16：冻结 LaMa 配置必须用 OmegaConf 解析字段引用

R8 首轮正式目录 `20260821T103805Z__frozen-generator-s20260821-r1` 中，Big-LaMa 在构造
FFC generator 时失败。官方 `config.yaml` 的 `downsample_conv_kwargs` 与
`resnet_conv_kwargs` 使用 `${generator...}` 字段引用；直接用 PyYAML 读取会把引用保留为
字符串，最终在通道比例转整数时触发 `ValueError`。该候选被如实记为 `failed`，首轮 gate
保持 `rejected`，不改写为方法结论。

修复只把 generator 子树改为 `OmegaConf.load` 后 `resolve=True`，仍使用同一官方配置、源码、
checkpoint、输入、seed、阈值与选择规则；不解析或消费任何训练/确认数据。以后复用带插值的
冻结配置时，必须在进入模型构造前解析并审计最终标量，不能把 YAML 语法读取成功当成配置已实例化。

### V6-F17：扩散 inpaint 输出尺寸必须显式绑定冻结输入尺寸

同一 R8 首轮中，SD-v1.5 pipeline 虽完成模型加载和推理，但未显式传入 `height/width`，输出采用
默认 `512×512`，与冻结输入及 mask 的 `512×288` 不一致，overlay 布尔索引因此触发
`IndexError`。该候选同样保留 `failed`，不是 GPU、权重或方法负结论。

修复是在冻结 pipeline 调用中显式设置 `height=image.shape[0]`、`width=image.shape[1]`；不 resize
实验输入、不改变 mask、prompt、steps、guidance、seed、候选或门槛。后续所有生成器 adapter
必须把空间尺寸当作调用合同，并在 compositing 前 fail-closed 核对 image/mask/proposal 三者形状。

### V6-F18：推理 adapter 不得反序列化 checkpoint 的训练器对象

R8 第二轮正式目录 `20260821T104107Z__frozen-generator-s20260821-r1` 中，SD-v1.5 已完整通过
4 cases × 2 repeats 的 capability/resource gate，但 Big-LaMa 在 `torch.load` 时尝试恢复 checkpoint
中未参与推理的 Lightning callback，因轻量推理环境没有完整训练框架而失败。冻结规则要求两个 ungated
候选都被实际执行，所以该轮仍如实 `rejected`；SD 的通过不能事后放松这条规则。

修复改用 PyTorch `weights_only=True`，只读取 tensor/state_dict，再按冻结 generator 结构严格加载；
不安装或执行训练器，不改变 checkpoint 字节、候选、案例、输入、seed、阈值、资源合同或选择规则。
以后第三方训练 checkpoint 的推理入口必须默认最小反序列化面，训练回调、优化器和日志对象不得成为
部署环境的隐式依赖。

### V6-F19：weights-only 仍需显式白名单化归档中的非张量全局类型

R8 第三轮正式目录 `20260821T104241Z__frozen-generator-s20260821-r1` 中，Big-LaMa 的
`weights_only=True` 正确拒绝了归档内未白名单化的
`pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint`；SD-v1.5 再次完整通过，但
双候选执行门仍未满足，所以该轮保持 `rejected`。这不是显存或模型能力失败。

修复为该精确全局名注册无方法、无训练行为的占位类型，并只把该类型加入 PyTorch safe globals；
checkpoint 继续以 `weights_only=True` 读取，后续仍只消费 `state_dict`。不引入 Lightning 训练栈，
不执行 callback，也不改变任何正式实验变量。若归档再暴露未预注册类型则继续 fail-closed，禁止切回
不受限反序列化来绕过门控。

### V6-F20：safe-global 修复必须基于静态完整清单，不能逐异常猜测

R8 第四轮正式目录 `20260821T104432Z__frozen-generator-s20260821-r1` 在白名单化
`ModelCheckpoint` 后继续 fail-closed，下一项未注册类型为 `omegaconf.dictconfig.DictConfig`；
SD-v1.5 仍完整通过而双候选执行门仍失败。逐次靠异常暴露类型会无意义地重复正式运行。

修复先用 `zipfile + pickletools` 静态读取官方 checkpoint 的 `data.pkl` GLOBAL 指令，不执行
反序列化；完整非内建清单只有 OmegaConf 的 `ContainerMetadata/Metadata/DictConfig/ListConfig/AnyNode`、
`typing.Any` 和已知 `ModelCheckpoint`，其余为 PyTorch tensor/标准容器重建函数。adapter 一次性白名单化
这些精确类型后仍使用 `weights_only=True`，实验变量与选择规则不变。以后同类归档应先做静态类型清单，
再建立最小安全白名单，避免把正式 run 当依赖探针。

### V6-F21：Python 2 pickle 内建名迁移后也必须进入 safe-global 清单

R8 第五轮正式目录 `20260821T104627Z__frozen-generator-s20260821-r1` 中，OmegaConf 类型完成
白名单后，weights-only loader 继续拒绝由旧归档 `__builtin__.dict` 迁移得到的 `builtins.dict`；
SD-v1.5 再次通过，双候选执行门仍保持 `rejected`。静态 GLOBAL 清单此前列出了旧模块名，但未把
Python 3 运行时映射后的内建类型显式加入 safe globals。

修复补入清单中出现的标准容器及迁移类型：`dict/list/int/OrderedDict/defaultdict`；仍不对白名单外
类型开放，不切换为 unrestricted pickle。模型、checkpoint、输入、seed、阈值与选择规则不变。
以后审计跨 Python 版本的 checkpoint 时，GLOBAL 清单必须同时记录归档名与当前运行时解析后的类型名。

### V6-F22：跨阶段 source manifest 必须复制机器实测完整 SHA

R9 首次正式入口在创建 run 目录前 fail-closed：冻结配置把 R7 manifest 的
`73dbb2ba11bc12bc...` 手工误抄为 `73dbb2ba11bc4e22...`。现场 `sha256sum` 与 R7 closeout 均确认
源文件仍为 `73dbb2ba11bc12bc4e22ca13765af10d14ac5d183e1d529add5e6d619f2a4d0c`；因此没有 proposal、
verifier、hidden target、训练、confirmation 或方法结果产生。

修复只替换为机器实测的完整 source manifest SHA，不改变 R9 hypothesis、模型、cohort、arm、阈值、
gate 或资源合同。这是与 V6-F12 同类的 provenance 抄录错误；后续跨 run 配置应由 manifest artifact
自动生成，禁止再次从缩写或人工记忆恢复完整哈希。

### V6-F23：冻结 semantic checkpoint 必须 strict 重建 auxiliary head

R9 首个有 run 目录的正式实例 `20260821T110446Z__independent-arms-s20260821-r1` 已先生成全部
28 个 Big-LaMa proposal，随后 semantic worker 在 strict load 时拒绝 checkpoint 中的
`aux_classifier.*`：初版 adapter 以 `aux_loss=False` 构造 DeepLabV3，遗漏了训练 checkpoint 保留的官方
auxiliary head。run 保持 `failed`，未产生 arm verdict、融合或 bake，也不是模型质量/资源负结论。

修复只以 `aux_loss=True` 重建同一 19-class DeepLabV3-ResNet50，并继续 strict load 全部参数；正式推理
仍只消费主输出 `out`，aux head 不参与 arm score。权重字节、cohort、proposal、threshold、gate 与资源合同
均不改变。以后冻结视觉 checkpoint 的结构审计必须覆盖所有 state-dict head，不能以“推理不消费”为由
在 strict identity 前删除参数。

### V6-F24：第三方 checkpoint 的主 head 与 auxiliary head 类数可能不一致

R9 第二个正式 run `20260821T110616Z__independent-arms-s20260821-r1` 在启用 aux 结构后继续由
strict load 拒绝：归档主 classifier 为 Cityscapes 19 类，但 `aux_classifier.4` 仍是 torchvision 默认
21 类（权重 `21×256×1×1`），而统一 `num_classes=19` 构造出的 aux head 为 19 类。run 保持
`failed`；28 proposals 已生成，但无 verifier verdict、融合或 bake。

修复精确重建归档结构：主 head 保持 19 类，aux 最后一层单独恢复为 21 类，然后 strict load 全 state dict；
aux 输出仍不被正式分数消费。模型权重、P3 动态类定义、cohort、threshold、gate 和资源合同均不改变。
以后第三方 segmentation checkpoint 必须逐 head 审计 shape，不能假设所有 classifier 共用同一 label count。

### V6-F25：capability 最优的 generator 不等于 verifier-arm 质量最优

R9 canonical rejected run `20260821T110743Z__independent-arms-s20260821-r1` 在 28 个 matched
development pseudo-holes 上证明 Big-LaMa 虽是 R8 的资源最优候选，但没有 verifier arm 可进入 R10：P1/P2
均 `0/28` ACCEPT；P3 在 12 个 actor-evidence cases 中接受 6 个，却有 1 个 false-safe，率 `1/6=0.1667`
高于冻结 `0.10`；P4 正确 `28/28` ABSTAIN。P0 photo/geometry false-safe 均为 `1.0`，P3 为
`0.5833`。outside-mask exact、无融合/无 bake/无 confirmation 均通过，峰值仅 `428 MiB`，不是资源失败。

H-R9-001 正确裁决为 `rejected`，不得放宽 photo/depth/semantic 阈值或把全拒绝写成有效 verifier。
新 H-R9-002 只切换到 R8 已完成 capability gate 的第二候选 SD-v1.5；沿用完全相同的 28-case cohort、
hidden observations、P1–P4、truth 定义、threshold、gate、模型和资源上限。Big-LaMa 与 SD 结果必须分属
独立不可变 run，禁止在看到 SD 结果后混合选择 per-case generator。

### V6-F26：单图生成候选无法把不可观测 missing-world 内容变成可验证事实

H-R9-002 canonical rejected run `20260821T111228Z__independent-arms-s20260821-r1` 用冻结
SD-v1.5 替换 Big-LaMa，并保持所有 verifier 与 gate 不变。P1 仍 `0/28` ACCEPT；P2 仅 `2/28=0.0714`
且低于冻结 `0.10` coverage；P3 与 Big-LaMa 同为 `6/12` ACCEPT、`1/6=0.1667` false-safe；P4
`28/28` ABSTAIN。outside-mask exact，峰值 `2696 MiB`，无融合、bake、训练或 confirmation。因此
H-R9-002 同样是方法质量 `rejected`，不是工程/资源 blocked。

两种单图 inpainting 都失败后，不得调松阈值、按案例混选生成器或转向 gated 23.8GB FLUX 权重来规避
负结果。新 H-R9-003 将唯一变量改为冻结 cross-frontend reconstructed proposal：同 scene/frame/edit variant
使用另一 frontend 的对齐 RGB 填入 mask；P1–P4、truth、threshold、gate 和 denominator 原样保留。该 proposal
仍标记 reconstructed，两个 frontend 来自同一传感器支持，不能解释为新增观测或独立 ground truth。

### V6-F27：正式入口必须显式建立仓库模块搜索路径

R12 第一次启动命令在创建 run 目录、加载模型或执行 GPU 推理前失败：直接运行
`python scripts/worldsim_v6/run_logsim.py` 时，Python 只把脚本目录加入模块搜索路径，因而无法导入仓库根目录下的
`motion_proj` 包并抛出 `ModuleNotFoundError`。该失败没有产生样本、指标或方法结论，也不是资源失败。

修复只在入口脚本中根据 `__file__` 把仓库根目录加入 `sys.path`，不改 R12 hypothesis、cohort、输入哈希、模型、
阈值、gate 或资源合同。后续以完全相同命令重跑；任何模型或指标失败仍独立登记，不能用本次入口错误掩盖。

### V6-F28：启用 CUDA 确定性算法前必须冻结 cuBLAS workspace 配置

R12 首个有 run 目录的正式实例 `20260821T114117Z__logsim-s20260821-r1` 已完成两项静态 chunk 的 CPU
重放构造并成功严格加载冻结 DeepLab checkpoint，但第一次 GPU forward 被 PyTorch fail-closed：代码启用了
`torch.use_deterministic_algorithms(True)`，CUDA 10.2+ 的 cuBLAS 路径还要求进程启动前设置
`CUBLAS_WORKSPACE_CONFIG=:4096:8`。该 run 保持 `blocked`，没有感知输出、完整 gate 或方法结论；也没有发生 OOM。

修复只在启动感知子进程前加入这一确定性环境变量，继续保留 deterministic algorithms、同一 checkpoint、4 个输入、
同一 cohort、阈值与资源上限。不得关闭确定性检查来换取通过；修复后新建独立 run 重试。

### V6-F29：世界空间 z-buffer 必须把空视锥投影作为有效零覆盖结果

R13 首个正式 run `20260821T120059Z__worldspace-route-s20260821-r1` 已完成两个 verified chunk 的世界坐标
提升，但部分大幅路线偏离没有任何点落入目标视锥。初版 z-buffer 仍为零长度索引构造了长度 1 的首元素布尔 mask，
触发 `IndexError`。该 run 保持 `blocked`，没有完整 baseline matrix、gate 或方法结论，也不是资源失败。

修复只在 z-buffer 中对零个可见点直接返回空的 x/y/z/source-index；下游按预注册协议记录
`projected_pixel_count=0`、指标不可用且 route-support fail。不得删掉这些偏离、降低分母或将空投影改写成 ABSTAIN。
其余输入、深度、标定、四方法、阈值、gate 和资源合同均不变，并以独立 run 重试。

### V6-F30：WorldSim evaluator 必须统一合法的 singleton-channel depth plane

R13 第二个正式 run `20260821T120208Z__worldspace-route-s20260821-r1` 在加载偏离路线的 StreetGS depth 时
fail-closed。该 renderer 保存合法的 `H×W×1` float depth，而 evaluator 的 PIL resize 入口只接收 `H×W`，
因此抛出 `TypeError`。run 保持 `blocked`，尚无完整 48-row baseline matrix 或 gate；没有 GPU/内存问题。

修复在 resize 前只接受 `H×W` 或 `H×W×1`，后者显式去掉最后 singleton channel；其他形状继续拒绝。
这与 V6-F13 的 plane normalization 原则一致，但本次记录覆盖独立的 R13 evaluator。不得改 depth 数值、插值模式、
样本、四方法、阈值或 gate，修复后以新 run 重试。

### V6-F31：verifier 相对深度不得直接解释为 WorldSim 米制相机 z

H-R13-001 canonical rejected run `20260821T120310Z__worldspace-route-s20260821-r1` 成功把两个 R11 chunk
封装为 58,273 个所谓世界点，但估计总面积仅 `0.01993 m²`，且 12/12 个非零路线偏离均为零投影覆盖，
usable lateral route 为 `0.0 m`。V6 的 matched false-safe 仍为 `0/3`，相对 naive 的降低为 `0.8214`，
所以拒绝原因不是安全 gate，而是 R9 depth 只为仿射对齐后的 verifier 几何比较服务，不能直接当作米制 z 做相机平移。

H-R13-001 按预注册门槛正式 `rejected`，不得通过放宽 256-pixel、0.12 photo 或 0.30 geometry 门槛恢复。
H-R13-002 只替换深度来源：使用同帧冻结 logged LiDAR metric depth，并限定最近填充距离不超过 8 个
512×288 像素；世界提升、四方法、12 个偏离、评估阈值、false-safe gate 和资源合同全部保持不变。

### V6-F32：无可见性约束的关键帧点云 union 会放大遮挡错误而非扩展路线

H-R13-003 canonical rejected run `20260821T121112Z__worldspace-fusion-s20260821-r1` 将两个 metric world chunk
按冻结 5cm voxel 做无目标视角过滤的 union。57,997 个输入点因近表面重复被折叠为 5,868 点，没有形成预期 densification；
共同 lateral route 从 `3.0m` 退化到 `2.0m`，5m mean geometry MRE 从 `9.0572` 升至 `10.6579`，
相对变化为 `-17.67%`，两帧 5m 继续失败。因此假设正式 `rejected`，不是工程或资源 blocked。

不得靠扫描更小 voxel 或放宽 photo/geometry 门槛复活该 union。H-R13-004 转向不同机制：保持 H-R13-002 的
逐帧 world points 和 RGB，不做 union/densification；只用同一冻结 support 中三相机 logged LiDAR 投影到目标视角，
在 4 像素邻域和 0.30 相对深度差内保留可见点。StreetGS truth proxy 只用于最终评估，不进入过滤。

### V6-F33：跨阶段安全摘要必须读取冻结 schema 的完整方法键

H-R13-004 首个正式 run `20260821T121825Z__worldspace-visibility-s20260821-r1` 已完成两次目标视角
LiDAR 投影和全部 12 个偏移的指标计算，但在汇总继承 H-R13-002 的 V6 false-safe 时 fail-closed。冻结摘要使用
`baseline_safety.v6_generate_verify_bake.false_safe_rate`，初版 runner 却读取了不存在的缩写键
`baseline_safety.v6.joint_false_safe_rate`，因此抛出 `KeyError`，run 保持 `blocked`，没有形成 gate 或方法结论。

修复只按冻结摘要的实际 schema 读取完整方法键和 `false_safe_rate` 字段；不改 world points、目标视角 LiDAR、4 像素/
0.30 visibility 合同、12 个偏移、质量阈值、false-safe 数值或资源合同。以后跨阶段消费结构化摘要时，必须把完整方法标识和
字段名纳入配置/manifest 合同，禁止用人工缩写推断 schema。

### V6-F34：删除式可见性筛选不能补齐远路线新暴露表面，且 5m 深度代理失去米制有效性

H-R13-004 canonical rejected run `20260821T121956Z__worldspace-visibility-s20260821-r1` 使用冻结三相机
logged LiDAR 在每个目标视角执行 4 像素/0.30 相对深度筛选。它保持共同 lateral route `3.0m`、精确复跑、源不可变和
V6 false-safe `0.0`，但两帧 5m 几何 MRE 仍为 `9.0323/9.4071`，photo MAE 为 `0.1404/0.1495`，
两帧均失败；5m 仍保留约 `78%` 的旧点，说明仅删除矛盾点并未提供新暴露表面。

独立诊断还显示，5m 目标视角投影后的实测 LiDAR 中位 z 约 `13.17/13.34m`，而 StreetGS 目标 depth proxy 中位数仅
`1.30/1.22`，该 proxy 在此外推距离不能作为米制几何真值。不得扫描 visibility 阈值或放宽 photo/geometry gate 来追逐
这个失效代理。路线偏移结论固定为 H-R13-002 已验证的 lateral `3m`、forward `2m`；后续直接进入计划尚未覆盖的 actor
add/remove、trajectory modification 与 traffic-density typed edit 实验。

### V6-F35：跨 run 回放内容等价比较必须排除非语义 repeat 序号

H-R13-005 首个正式 run `20260821T122830Z__dynamic-edits-s20260821-r1` 的三个 V6 typed edit 均通过全部
编辑、依赖闭包、时序和精确复跑检查，但总 gate 因 `base_matches_frozen_r12_replay=false` 保持 `rejected`。逐字段定位确认
唯一差异是当前重新加载调用使用 `repeat_index=0`，而冻结文件 `DYNAMIC_REPLAY_REPEAT1.json` 记录 `repeat_index=1`；
`replay_content_sha256` 及 actor、trajectory、semantic、collision、sensor、event 全部内容一致。

修复只在跨 run 内容等价比较的两侧移除非语义 `repeat_index`，仍严格比较冻结内容 hash 和所有功能字段；不改三个 edit、
四方法臂、actor/时间戳分母、碰撞计算、false-safe、资源合同或任何阈值。repeat 序号继续保留在各自运行记录内，但不得被当作
compiled-world 内容漂移。

### V6-F36：浮点 renderer RGB 归一化必须容忍轻微大于 1 的辐射 overshoot

H-R13-006 首个正式 run `20260821T123716Z__actor-sensor-perception-s20260821-r1` 完成 16 次 DeepLab
推理且精确复跑，但 AD-GS 两个 case 的 target RGB effect 被错误压到约 `0.00035`，继而使感知变化近零。StreetGS 两例保持
约 `0.09` target effect。根因是 AD-GS 的归一化浮点 RGB 存在轻微 `>1` overshoot，初版主 runner 把它误判为 0--255
后再除 255；worker 同样没有把它放大到 uint8 动态范围。该 run 的 AD-GS 指标无效，不能作为方法 rejection。

修复把浮点 renderer 合同明确为最大值 `<=2.0` 时仍按归一化辐射值处理：主 runner 直接 clip 到 `[0,1]`，perception
worker 乘 255 后 round/clip 为 uint8；只有明显大于 2 的数组才按 0--255 输入。冻结 render 字节、模型、case、mask、
阈值、repeat、资源和 gate 全部不变。以后不同 frontend 的 RGB 必须由显式 range contract 归一化，禁止用严格 `max<=1`
启发式区分编码。

### V6-F37：局部 RGB 编辑不保证全帧感知输出局部，宽感受野必须进入 verifier 因子设计

H-R13-006 canonical rejected run `20260821T123835Z__actor-sensor-perception-s20260821-r1` 在修复 RGB
range 后确认 StreetGS/AD-GS 两帧共 4 个 actor-remove case 均具有强 sensor locality：target RGB MAE
`0.0872--0.0954`、outside RGB MAE `0.000175--0.000399`、locality enrichment `224--498x`，16 次
DeepLab 推理精确复跑且峰值仅 `688MiB`。但全帧 DeepLab 在 target 外仍改变 `6.34%--10.31%` 标签，4 个 case
只有 AD-GS frame57 达到 2x perception locality，故假设按预注册 gate 正式 `rejected`。

不得把 outside 2% 或 enrichment 2x 阈值调松，也不得因 RGB 局部就宣称 perception failure 已解决。H-R13-007 改用
factorized ROI 机制：固定 256px tile/128px candidate stride，只根据 logged dynamic opacity 选择最高 actor fraction target
和与其不重叠的最低 actor fraction static tile，对两者独立执行冻结模型；full-frame rejection 永久保留，不被 ROI 结果覆盖。

### V6-F38：remove-all 编辑不能为 factorized perception 提供无 actor 的静态 ROI 对照

H-R13-007 canonical rejected run `20260821T124443Z__factorized-perception-s20260821-r1` 按冻结
256px tile/128px stride，在每个 frontend/frame 选择最高 actor fraction target 与不重叠的最低 actor fraction static。
四个 target 的 actor fraction 为 `0.753--0.936`，但四个所谓 static 仍为 `0.099--0.373`，全部超过预注册
`0.01` 上限；static RGB MAE 也为 `0.0095--0.0267`，证明 remove-all 操作本身横跨全图，而不是 tile 选择偶然失败。

不得扫描 tile 大小/stride 或放宽 static denominator。H-R13-008 改变真正的因果变量：利用 StreetGS 冻结 checkpoint 的
per-Gaussian `point_ids` 与 `instances_fv`，只删除在两帧均可见且 Gaussian 数最多的单个 model actor；actor 选择不读取
RGB/semantic outcome。冻结 AD-GS 不保留可审计 per-actor ID，必须 ABSTAIN，不得伪造跨 frontend 单 actor 对齐。

### V6-F39：actor Gaussian 数量不等于下游感知敏感度，单启发式选择必须扩展为完整分母

H-R13-008 canonical rejected run `20260821T125151Z__single-actor-perception-s20260821-r1` 从两帧均可见的
12 个 StreetGS model actor 中，按最大 Gaussian 数且最小 index 的预注册规则选中 index `2`（`13,490` Gaussians）。
logged rerender 对冻结 R3 RGB 的 MAE 为精确 `0`，单 actor 删除的 effect pixel 为 `10,271/9,047`，target RGB
MAE 为 `0.0210/0.0286`，outside RGB MAE 仅约 `1e-6`；但 target DeepLab label change 只有
`0.00068/0.0`，两帧都未达到冻结 2% 感知效应门。

该负结论拒绝“Gaussian 最多 actor 最能触发 perception”的启发式，不得改选第二大 actor 当作同一假设 recovery，也不得放宽
2% threshold。H-R13-009 一次性评估冻结 metadata 定义的全部 12 个 eligible actor，以固定完整分母报告 ACCEPT/ABSTAIN
覆盖率；只有两帧都通过完全相同门槛的 actor 才可被 V6 接受，其余必须显式 abstain。

## 7. 历史新路线启动前附加检查

- [ ] 是否明确说明该步骤直接服务于重建、编辑或可信评测，而不是重新做事件挖掘？
- [ ] AD-GS exact reproduction 是否已经通过冻结门禁？
- [ ] 是否把 upstream 原始结果与 compatibility patch 结果分开？
- [ ] 是否对 VAD-GS 等已公开的 visibility/completion 工作做 novelty 边界核对？
- [ ] 反事实无真值指标是否有真实 held-out/pseudo-hole 证据，而不是自洽规则？
- [ ] 是否同时评估目标区变化、非目标区保持、几何/时序一致性和下游感知？
- [ ] 遇到内存/GPU不足时是否按 `N1-F24/PIVOT-F05` 停机并等待授权？

## 8. WorldSim 后续正式消融前检查

- [ ] 是否使用 V3 task ID、新 run 和冻结 config/source hash，而不是续写 V2 terminal？
- [ ] 是否保持 scene-0230/0242/0255、split、seed、相机、步数和 actor cohort 不变？
- [ ] 是否把原生 Affine/CamPose/LiDAR init 与新增实现分开？
- [ ] rolling-shutter 路径是否有真实 row timing；没有时是否显式 `not_supported`？
- [ ] actor-aware 变化是否只增加一个可归因因子，并保留 module-off 原生等价测试？
- [ ] 是否同时报告 actor/boundary 质量、GS 数、训练时间、VRAM 和 non-target 保持？
- [ ] local refinement 是否冻结 affected set 外参数，并只使用 Tier-A/多视图/LiDAR 可观测证据？
- [ ] expected/first-hit/measured depth 是否继续保持 typed separation？
- [ ] 工程 `blocked`、方法负结果 `rejected` 和任务完成 `done` 是否没有混写？
- [ ] 结论是否明确限制在三场景消融，不写成大规模泛化或闭环安全结论？
- [ ] 新路线是否只选择一个 primary hypothesis，并说明它具体解除 `V3-F18`–`V3-F25` 中哪一项？
- [ ] 是否在任何训练、推理或新结果读取前冻结 matched baseline、主端点、资源门、停止条件和确认场景？
- [ ] 是否避免把更小剪枝 fraction、提高旧资源 ceiling、全量读取 chunk 或继续调 R1 配方伪装成新研究？

### V6-F40：可见 actor cohort 分母不得替代 SceneIR 全量 actor 分母

H-R13-010 canonical run `20260821T130810Z__sceneir-sensor-binding-s20260821-r1` 正确完成同一 scene-0242 checkpoint 的 SceneIR 编译、model index `0` 到 `actor_0000` / `streetgs_actor_0000` / `12,390` primitives 的绑定、actor remove、两次 fresh replay，以及未受影响 actor state、trajectory、semantic label 和 collision pair 的精确保持；继承的 H-R13-009 V6 false-safe 仍为 `0`。但是 preregistration 把 H-R13-009 中“两帧均可见”的 `12` 个 actor cohort 错当成 checkpoint 转换后的 SceneIR 全量 actor 数，实际冻结 converter 输出为 `27` 个 actor；因此预注册的 `15→14`、`2940→2744`、`20580→17836` 与实际 `27→26`、`5292→5096`、`68796→63700` 不符，typed dependency-closure gate 按约定拒绝。

该 run 保持 `rejected`，不能用其余检查通过来覆盖错误分母。恢复假设 H-R13-011 只把全量分母来源改为编辑前冻结 checkpoint 的 deterministic SceneIR converter 输出，并预注册上述实际总量；不修改 checkpoint、actor mapping、edit、replay、quality threshold、继承 verdict、资源合同或 unsupported claim。以后必须把 visibility/evaluation cohort 与 compiled-world total denominator 分别命名和冻结。

### V6-F41：下游 regression consumer 必须冻结上游 gate 的完整 decision 值

H-PT1-001 首个 formal run `20260821T132127Z__regression-utility-s20260821-r1` 在读取 scene-0230 development contract 时 fail-closed。上游 H-R13-005 gate 的实际 decision 是 `accept_typed_dynamic_edit_dependency_closure`，初版 consumer 却硬编码了缩写 `accept_typed_dynamic_edits`，因此在读取 heldout replay、构造四类 mutation 或计算三臂 quality metric 前即抛出 `PT1RegressionError`；该 run 只有 `TERMINAL.json`，不得产生方法结论。

修复把 scene-0230 与 scene-0242 两个上游 gate 的完整 accepted decision 值写入冻结 config，consumer 只按 config 精确比较。不得改变四类 stale-factor mutation、三方法臂、false-safe/detection gate、source hash、资源合同或 unsupported claim；修复后以新 run 重试 H-PT1-001。跨阶段消费者以后不得从 task 名或人类缩写猜测 structured gate value。

### V6-F42：policy 输入不得直接包含 verifier 的 decision statistic

H-PT2-001 canonical run `20260821T133158Z__risk-policy-s20260821-r1` 的数值 gate 全部为真，V6 arm 在 scene-0255 heldout 上达到 balanced accuracy `1.0`、false-safe `0`、safe-route completion `1.0`，并把 naive stale-label arm 的 false-safe 从 `1.0` 降为 `0`。但是 Real-only arm 同样达到 balanced accuracy `1.0` 和 false-safe `0`，尽管其 98 条训练行的 positive fraction 为 `0`。原因是 policy 直接接收 signed AABB clearance，而 hazard label 正是 `clearance<=0`；这等于把 verifier 判定边界编码进输入，Real-only 只需把阈值放在最小 logged clearance 以下就会偶然分开固定 synthetic offsets。

因此该 run 只保留“naive stale labels 有害”的诊断，不晋级为 incremental post-training utility；初版 gate 缺少对 Real-only 的增益约束也是方法治理缺口。H-PT2-002 保持三场景、frame 分母、clone offsets、label、heldout 与任务指标不变，移除 signed clearance/AABB extent/hazard verdict，只向 policy 提供原始绝对 ego-relative forward/lateral position；固定 axis-aligned rectangle ERM 候选网格，并新增相对 Real-only 与 naive 两者 false-safe 至少降低 `0.50` 的 gate。以后任何 learned verifier/policy 实验必须审计 feature 是否直接重编码 label rule。

### V6-F43：声明值经坐标变换后必须做远小于物理阈值间隔的数值 canonicalization

H-PT2-002 canonical run `20260821T133626Z__risk-policy-s20260821-r1` 在移除 signed-clearance feature leakage 后，使 Real-only 与 naive arm 都成为 constant CONTINUE，heldout false-safe 均为 `1.0`；V6 raw-position rectangle policy 把 false-safe 降至 `0.1122449`，safe-route completion 保持 `1.0`，但 balanced accuracy `0.9438776` 与 false-safe 仍未达到冻结的 `0.95` / `0.05` 门，故方法正式 `rejected`。

逐行诊断确认 11 个漏检全部是配置声明的 `3.0m` clone：`inv(T_ego) @ (T_ego @ T_offset)` 后 raw forward feature 变成 `3.00000000000011–3.00000000000045`，严格 `<=3.0` 比较失败；没有其他 hazard 漏检，也没有 false brake。H-PT2-003 只在 policy raw feature 写入前按 9 位小数 canonicalize（`1e-9m`），label 仍按未取整 signed clearance 计算；该尺度高于浮点漂移、但比最小候选阈值间隔小至少九个数量级。不得改样本、offset、threshold grid、label 或质量门。以后由声明变换生成的控制量必须把表示 canonicalization 与方法容差分开冻结。

### V6-F44：单一 zero-lateral intervention 训练不能支持二维风险泛化

H-PT3-001 canonical run `20260821T134426Z__intervention-robustness-s20260821-r1` 冻结 H-PT2-003 的全部 policy 参数，在未参与 PT2 训练或选择的 scene-0048 上评估 forward `1.5/4.5/7.5m` × nonzero lateral `0.75/1.5/2.5m` 的 441 个 edit rows，加 49 个 clean rows。分母含 232 hazards / 258 safe；冻结 V6 policy 的 lateral threshold 为 `0.0m`，因此对 232 hazards 检出 `0`，false-safe `1.0`，与 Real-only/naive constant policies 完全相同。source immutability、repeat exact、safe-route completion 均通过，所以是方法性 `rejected`。

不得把 PT2 的跨场景同 intervention pass 写成二维 policy generalization，也不得只把冻结 lateral threshold 放宽。H-PT3-002 改变训练 evidence：在 scene-0230/0242 使用 forward `0/2/4/6/8m` × lateral `0/1/2/3m` 的完整 factorized typed-edit grid，V6 重算标签、naive 保留 stale labels；scene-0048 使用与训练离散的 half-offset grid，所有质量门不变。以后 policy coverage 必须按 intervention factor denominator 报告，不能只报 scene denominator。

### V6-F45：最近 actor 的位置不足以表达 box overlap，必须保留 factorized extent

H-PT3-002 canonical run `20260821T134843Z__factorized-policy-training-s20260821-r1` 在 scene-0230/0242 完成 1,960 条二维 typed clone train rows，并在 scene-0048 的 980 条离散 half-offset edits 上评估。V6 相对两基线把 false-safe 从 `1.0` 降至 `0.4505208`、safe-route completion 保持 `1.0`，但 balanced accuracy 仅 `0.7747396`，未过冻结门，正式 `rejected`。

诊断显示 position-only policy 为每条 episode 只保留 signed-clearance 最小 actor 的 `|x|/|y|`，却丢掉 projected box half-extent 与 yaw；远处 safe clone 会由更近但窄或旋转的真实 actor 取代特征，同一位置因此对应不同 overlap label。训练 lateral `2.0m` 恰在默认 box 边界，未 canonicalize 的 factor label 还出现 66/32 等浮点混合。H-PT3-003 保持数据、grid、三臂、heldout 和质量门，新增 raw projected half-extents，按 1e-9m canonicalize forward/lateral factor labels，分别训练两个 logistic overlap heads 后 AND；不得输入 signed clearance 或最终 hazard verdict。以后几何 policy 的 factor representation 必须保留决定接触边界的尺寸/朝向信息。

### V6-F46：单一 synthetic box size/yaw 无法识别可迁移的 extent 系数

H-PT3-003 canonical run `20260821T135421Z__factorized-policy-training-s20260821-r1` 使用 raw `|x|/|y|` 与 projected half-extents，分别训练 forward/lateral logistic overlap heads。V6 train overall balanced accuracy 为 `1.0`，forward/lateral factor train accuracy 为 `1.0/0.9986`，但在 disjoint scene-0048 grid 上 balanced accuracy `0.7963`、false-safe `0.22135`、safe-route completion `0.81395`，未过冻结门，正式 `rejected`；两基线 false-safe 均为 `1.0`。

这不是优化未收敛，而是 synthetic train 全部采用默认 `4.5x2.0m`、relative yaw `0°`，projected extent 几乎常数，position 与 extent 的相反系数无法由 synthetic positives/negatives 识别，只能依赖稀疏真实 box 分布，换场景即漂移。H-PT3-004 保留 factor-head 架构、loss、task gates 与场景，扩展训练 denominator 为三种 size × 四种 yaw × 原 position grid；heldout 使用完全离散的三种 size × 三种 yaw。不得输入 signed gap 或固定解析碰撞公式来伪装 learning gain。以后 intervention coverage 必须同时报告 position、size 与 orientation factors。

### V6-F47：factor head 的训练目标必须与冻结的 balanced 指标对齐

H-PT3-004 canonical run `20260821T135942Z__factorized-policy-training-s20260821-r1` 在 23,520 条 multi-size/multi-yaw train interventions 与 8,820 条完全离散 heldout interventions 上运行。V6 把 false-safe 从 Real-only 的 `0.97133`、naive 的 `1.0` 降至 `0.14821`，但 balanced accuracy `0.85613`、safe-route completion `0.86047`，仍未通过冻结的 `0.90/0.10/0.90` 门，因此正式 `rejected`。

训练的 lateral factor 正例比例为 `0.88174`，原始 unweighted BCE 按出现频率主导梯度，与 task 从一开始冻结的 balanced accuracy/false-safe/completion 三目标不一致；这会同时留下 `14.82%` false-safe 与 `13.95%` false brake。H-PT3-005 保持场景、position/size/yaw 分母、raw feature、标签、三臂、步数和所有 gate 不变，只让每个 factor head 的正负类别在 BCE 梯度中各占一半。不得通过调决策阈值、改 heldout 或放宽门来恢复。

### V6-F48：类别平衡 BCE 不是可分 factor boundary 的充分机制

H-PT3-005 canonical run `20260821T140408Z__factorized-policy-training-s20260821-r1` 只把两个 logistic factor heads 改为正负类别总权重各半，其余数据、特征、三臂和 gate 均不变。V6 heldout balanced accuracy 降为 `0.82294`，false-safe 升为 `0.24746`，completion 为 `0.89334`；naive false-safe 也由 `1.0` 变为 `0.67863`，导致相对 naive 的 reduction 只有 `0.43116`。所有冻结质量门仍未通过，正式 `rejected`。

两个 V6 head 的 train balanced accuracy 仍只有 `0.97503/0.96014`，说明仅重加权有限步 smooth BCE 并没有形成稳定分离 margin；它改变了错误权衡，却没有消除训练边界错误。H-PT3-006 保持相同 denominator、raw feature 和原 gate，改用 deterministic class-balanced linear max-margin heads，并保留两 head AND。不得再通过类别权重或阈值扫描恢复。

### V6-F49：可分训练集上的最大间隔仍受离散 intervention 支持分辨率约束

H-PT3-006 canonical run `20260821T140847Z__factorized-policy-training-s20260821-r1` 使用相同 raw features 与 multi-size/multi-yaw 分母，将两个 factor heads 换为 class-balanced linear max-margin。V6 的 forward、lateral 与 joint train balanced accuracy 均为 `1.0`；heldout false-safe 降到 `0.05226`，相对 Real-only/naive 的 reduction 为 `0.66264/0.62637`，三项均过门。但 safe-route completion 只有 `0.84790`，balanced accuracy `0.89782`，仍未通过冻结门，正式 `rejected`。

原 train position 只有 forward `0/2/4/6/8` × lateral `0/1/2/3`，即使训练完全可分，最大间隔也可落在相邻采样位置之间；heldout 恰使用离散半步位置，暴露 `15.21%` false brake。H-PT3-007 只加密 train position denominator，新增位置全部与 heldout position set 离散，保持 size/yaw、heldout、features、SVM 和 gate 不变。不得调 SVM threshold 或删除 near-boundary heldout rows。

### V6-F50：set aggregation 下不能只用 false-safe 单轴增益比较会过度刹车的基线

H-PT6-001 canonical run `20260821T142626Z__compositional-risk-s20260821-r1` 在 scene0450 的 784 个双 clone episodes 上，用同一 frozen per-actor policy 对 logged actors 与两 clone 做 Boolean OR。V6 balanced accuracy/false-safe/completion 为 `1.0/0/1.0`；Real-only 为 `0.5/1.0/1.0`；naive 为 `0.52211/0.08844/0.13265`。V6 的全部绝对质量门通过，但相对 naive 的 false-safe reduction 最大只能是 `0.08844`，无法达到从 single-clone task 继承的 `0.50`，所以 H-PT6-001 仍正式 `rejected`。

naive 并未获得可用策略，而是 set OR 后以 `86.73%` false-brake 换取较低 false-safe。H-PT6-002 不追认旧 run，保持 frozen policy、scene、episode、label 和绝对门不变，预注册 paired Pareto gate：V6 对每个 baseline 的 false-safe 不更差、completion 不更差，并且 balanced accuracy 至少高 `0.20`。以后 actor-set policy 不得只用某一风险轴的下降评价会全刹车的 baseline。

### V6-F51：multi-actor development 的完美结果未在 one-shot confirmation 保持 false-safe

H-PT7-001 canonical run `20260821T143359Z__compositional-risk-confirmation-s20260821-r1` 在 attempt 先于质量读取、policy 与 H-PT6-002 gate 均冻结、scene0862 与八个 clone case tuples 全新的合同下，评估 784 个双 clone episodes。V6 balanced accuracy `0.92333`、safe-route completion `1.0`，且仍 Pareto 支配两基线；但 600 个 hazard 中漏掉 92 个，false-safe `0.15333`，超过冻结的 `0.10` 绝对门，因此 one-shot confirmation 正式 `rejected` 并消费。

不得用 scene0862 的漏检分布为该 multi-actor candidate 调 policy、case、threshold 或 gate，也不得用其余 Pareto checks 通过覆盖 false-safe failure。该 family 关闭。后续 H-PT8-001 来自此前一直显式保留的 `ABSTAIN_NO_LONGITUDINAL_CONTROLLER`，使用预注册 kinematic scenario grid 开始独立的 closed-loop utility family，不读取 PT7 badcase 生成参数。

### V6-F52：closed-loop collision 分母必须区分 policy 可避免与动力学不可避免，并审计 Real-only 监督

H-PT8-001 canonical run `20260821T143902Z__closed-loop-utility-s20260821-r1` 在 360 个静止单 actor、五秒、jerk-limited 纵向 scenarios 上，将三个 frozen policy arms 作为相同三秒 preview controller 的碰撞信号。V6 collision rate `0.25`、safe completion `0.9375`、comfort `1.0`、balanced accuracy `0.84375`，未过冻结门，正式 `rejected`；Real-only 与 V6 完全相同，naive balanced accuracy `0.69554`。

70 个 V6 collisions 按 6/8/10m/s 为 `3/22/45`，按 15/20/25/30/35m 初距为 `41/22/7/0/0`，并随 actor size 增大，说明原 denominator 把 t=0 已无法在同一 decel/jerk contract 下停车的场景也算成 policy false-safe。同时，Real-only equality 提醒后续必须审计它继承的 factor-label supervision；不得通过调 preview horizon 掩盖 baseline equality。H-PT8-002 只加入相同 dynamics 的 t=0 full-brake oracle，把 hazards 分为 avoidable/unavoidable，保留全部计数并只在 avoidable stratum 评价 policy collision；其余均不变。

### V6-F53：真实 box factor supervision 已解释静止 AABB preview，不能人为削弱 Real-only 制造增益

H-PT8-002 canonical run `20260821T144333Z__closed-loop-utility-s20260821-r1` 用同一 jerk/decel contract 的 t=0 full-brake oracle 将 280 个 uncontrolled hazards 分为 210 个 avoidable 与 70 个 unavoidable。V6 在 avoidable stratum collision `0`、safe completion `0.9375`、balanced accuracy `0.96875`，说明 H-PT8-001 的绝对 collision failure 来自可行性分母；但 Real-only 的三项指标完全相同，增量门仍失败，H-PT8-002 正式 `rejected`。

Real-only factor heads 虽无最终 clean collision positives，但合法读取了 logged box 的 forward/lateral overlap factor labels，已经足以学习静止 AABB 边界。不得删除这些真实监督、重命名 baseline 或只比较更弱 naive 来制造 V6 closed-loop gain；该静止单 actor family 关闭。H-R14-001 转回 core compiler 的既有 H-C 缺口，测试邻帧 temporal evidence-conditioned proposal，不由 PT8 结果选择参数。

### V6-F54：独立 verifier arms 合格不等于 factor conjunction 有足够 usable coverage

H-R15-001 canonical run `20260821T145504Z__factorized-verification-s20260821-r1` 只消费 H-R14-001 独立通过的 P1 photo 与 P2 geometry decisions，按双 ACCEPT 才 ACCEPT、双 REJECT 才 REJECT、disagreement 全 ABSTAIN。28 cases 中只有 1 个 joint ACCEPT、13 个 ABSTAIN、14 个 REJECT；joint false-safe 为 `0`、相对 P0 reduction `0.89286`，但 accept coverage `0.03571` 未达到冻结 `0.05`，因此正式 `rejected`。

不得用 OR、放宽 verifier threshold、把 disagreement 计作 ACCEPT 或把门改成 1 case 来恢复。RGB-only ECC proposal 对 photo 优化，却没有约束 geometry，导致两个独立有效 arm 的接受集合错位。H-R16-001 保留同一 temporal source、28-case denominator、P0-P4 thresholds 与后续 conjunction，仅把 outside-mask alignment image 改为等权 RGB gray + robust normalized render inverse-depth；render depth 只用于 proposal alignment，不充当 P2 truth。

### V6-F55：无类型约束的双模型语义共识会形成相关性共同错误

H-R20-001 canonical run `20260821T152456Z__semantic-consensus-s20260821-r1` 在冻结的 12 个 R16 semantic-evidence cases 上，以 DeepLabV3-ResNet50 与 SegFormer-B0 的 hole 内 dynamic-mask IoU `>=0.70` 作为唯一决策证据。该机制成功拒绝了 R16 原先唯一的边缘 false-safe `scene-0048__ad_gs__f057__actor_removal_hole`，但两个模型在 `scene-0048__ad_gs__f052__disocclusion` 上以 IoU `0.88267` 共同预测了错误动态内容，使 2 个 ACCEPT 中 1 个为 false-safe，false-safe rate `0.50`、相对 P0 reduction 仅 `0.08333`，正式 `rejected`。

不得把架构不同等同于错误独立，也不得扫描 consensus threshold；相关模型会在编辑语义不同的 hole 上共同犯错。H-R21-001 保留相同模型、12-case denominator、truth 与质量门，新增由 compiler edit type 决定的 typed semantic contract：`actor_removal_hole` 沿用冻结的 `0.50` 双模型 dynamic IoU 门；`disocclusion` 只有两个模型在 hole 内都预测零 dynamic pixel 才可 ACCEPT。决策仍不读取 target dynamic truth，P4 保持 ABSTAIN。

### V6-F56：actor edit-mask 的轴对齐二阶矩不足以恢复可接受的纹理对应

H-R22-001 canonical run `20260821T153531Z__independent-arms-s20260821-r1` 对六个 `actor_removal_hole` 只使用源 actor-edit mask 与目标已知 hole mask 的中心和轴向二阶矩，估计 axis-aligned affine 后搬运邻帧 actor RGB。actor 子集 P2 geometry 得到 `1/6` ACCEPT、false-safe `0` 并通过独立 gate，但 P1 photo 为 `0/6` ACCEPT，导致要求 P1/P2 同时独立合格的正式 gate 拒绝。四个 photo truth-safe cases 的 masked RGB MAE 仍为 `0.06299–0.07655`，高于冻结 `0.05`；整体 P1/P2 通过只来自未改变的非 actor cases，不能覆盖 actor 子集失败。

不得放宽 P1 阈值或用整体分母掩盖 actor failure。轴对齐中心/尺度对齐无法表示邻帧 actor 的旋转、剪切和透视轮廓变化。H-R23-001 保留相同源/目标 masks、六个 actor 分母、非 actor RGB-D ECC、verifier 与全部 gates，只把 actor 配准改为：以二阶矩 affine 为初始化，在两个二值 edit-mask 的 signed-distance field 上执行一次冻结 homography ECC；仍不读取 target RGB/depth/dynamic。

### V6-F57：edit-mask 轮廓 homography 仍不能建立跨时刻 actor 纹理对应

H-R23-001 canonical run `20260821T154110Z__independent-arms-s20260821-r1` 在相同六个 actor cases 上，以 R22 moment affine 初始化 signed-distance-field homography ECC。actor P1 仍为 `0/6` ACCEPT，P2 仍只有 `1/6` ACCEPT；SDF warp 让若干 photo MAE 从 R22 的 `0.063–0.076` 恶化到 `0.069–0.116`，仅一个 case 改善至 `0.05748`，仍未过冻结 `0.05`。因此整体 P1/P2 合格仍不能覆盖 actor subset，正式 `rejected`。

不得继续扫描 SDF ROI、iteration 或 photo threshold。二值 actor-edit support 约束的是轮廓，不携带姿态、可见面与光照的跨时刻纹理对应；更灵活的 homography 会扭曲错误纹理。既有 H-R9-003 已独立证明 same-time cross-frontend proposal 在 actor 子集上 P1 `4/6`、P2 `1/6` 且 false-safe 均为 `0`。H-R24-001 因此不再拟合 mask，而按 typed asset route 复用已验证来源：static/disocclusion 保持 R16 temporal RGB-D，actor_removal 使用同帧 cross-frontend；所有 verifier 与 actor gates 不变。

### V6-F58：小 actor 分母上的固定 false-safe rate delta 可以数学上不可达

H-R24-001 canonical run `20260821T154603Z__independent-arms-s20260821-r1` 按 hole type 复用已接受的 R16 temporal RGB-D static source 与 H-R9-003 same-time cross-frontend actor source。actor 子集 P1 为 `4/6` ACCEPT、false-safe `0`，P2 为 `1/6` ACCEPT、false-safe `0`；两臂的绝对 coverage/risk 都通过。但 actor P1 的 P0 只有 `1/6` false-safe，最大可能 rate reduction 为 `0.16667`，无法达到从 28-case 全局臂继承的固定 `0.25`，因此要求 P1/P2 均合格的 H-R24-001 按原合同正式 `rejected`。

不得追认 R24 gate，也不得删除唯一 P0 错误或降低风险要求。H-R25-001 保持 R24 proposals、decisions、truth 与 P1/P2 阈值完全冻结，只预注册适合六例小分母的 actor Pareto gate：每臂 coverage `>=0.10`、false-safe rate `<=0.10`，并且 false-safe count 相对 P0 至少严格减少 `1`。该离散计数门在两个臂上均可实现且仍要求实际消除错误；旧 R24 run 继续保持 rejected。

### V6-F59：下游 factorized consumer 必须从拥有字段的冻结 artifact 读取 case metadata

H-R27-001 首个 formal run `20260821T160452Z__three-factor-s20260821-r1` 在组装六个 actor factorized decision 时因 `KeyError: mask_pixel_count` 失败，仅产生 failed `TERMINAL.json`，未构造 gate 或方法结论。R24 的 `verifier_worker/PER_CASE_ARMS.jsonl` 拥有 factor decisions、truth 与 proposal hash，但 `mask_pixel_count` 的 schema owner 是同一 run 的 `CASES.jsonl`；初版 consumer 错把该 metadata 当作 arm row 字段。

修复把冻结 R24 `CASES.jsonl` 及其 SHA256 加入 R27 source contract，并仅从 case-id index 读取 `mask_pixel_count`。不得改变 P1/P2/P3 decisions、truth、三因子合取规则、coverage/risk/count gate、R24 rejected 状态或确认集锁。该失败属于 artifact binding plumbing，不否定 H-R27-001，修复后以新 run 重试。

### V6-F60：factorized validity 的各因子 truth 应做逻辑积，不能要求逐例标签相等

H-R27-001 canonical retry `20260821T160717Z__three-factor-s20260821-r1` 在六个 actor cases 上得到预期决策：`1` ACCEPT、`1` REJECT、`4` ABSTAIN，全部 factor decision disagreement 都 ABSTAIN，ACCEPT 的 joint truth-safe 为真，false-safe 为 `0`，coverage 为 `1/6`。但预注册合同额外要求 photo、geometry、semantic 三种 truth label 逐例相等；实际有两个 case 的 factor truths 不同，因此 run 按合同正式 `rejected`。

不得删除这两个不一致 case 或追认 R27。photo truth 衡量 RGB 恢复、geometry truth 衡量 depth、semantic truth 衡量动态语义，它们本就可以独立真假。H-R28-001 保持全部 proposal、factor decisions、fusion 与风险门冻结，将 joint truth 明确定义为三种 factor truth 的逻辑 AND，并要求精确保留两个 cross-factor truth disagreements；R27 的唯一实质失败项必须仍是错误的 truth identity 假设。以后 factorized evaluator 必须区分 decision disagreement 与 truth-factor diversity。

### V6-F61：未展开的聚合布尔失败不能用于猜测精确 factor-diversity 分母

H-R28-001 canonical run `20260821T161052Z__three-factor-s20260821-r1` 正确使用 factor-truth product，并保持 `1` ACCEPT、`1` REJECT、`4` ABSTAIN、false-safe `0`；coverage、strict error removal、所有 disagreement ABSTAIN、source immutability 与 R27 rejection retention 全部通过。唯一失败是预注册把 cross-factor truth disagreement count 猜成 `2`，完整逐因子展开后实际为 `3`。

R27 只给出了 truth-identity 聚合布尔失败，不能推出失败 case 的精确数量。不得追认 R28 或把实际值 `3` 再硬编码成安全门。H-R29-001 保持全部六例、factor decisions、factor truths、product、fusion 与质量门冻结，要求 factor truth diversity 非零且逐例透明报告；精确 diversity count 作为描述性输出，不参与资格。以后只能从已冻结的逐例 artifact 预注册精确计数，不能从 aggregate failure 反推。

### V6-F62：aggregate actor-layer validity 不能下放给单 actor identity component

H-R32-001 canonical run `20260821T163038Z__identity-factor-s20260821-r1` 用此前接受的 H-R13-009 model-index-0 removal support 与 H-R13-011 `actor_0000` binding，在 R30/R31 layer 内得到 `4,792` 个 identity pixels，覆盖 resized actor effect support 的 `91.02%`。P1 photo 在该 support 上仍 ACCEPT（MAE `0.043739`），且 photo/geometry/semantic 三种 truth evaluation 均 safe；但 P2 geometry mean relative error 为 `0.212180`，超过冻结 `0.20`，P3 DeepLab/SegFormer dynamic IoU 仅 `0.098330`，低于冻结 `0.50`，因此 identity-specific conjunction ABSTAIN，正式 `rejected`。

不得用 aggregate R29 ACCEPT、truth-safe、接近 geometry 门或 target semantic IoU `0.9946` 覆盖独立 decision failure，也不得放宽阈值。R7/R30 的 actor layer 由 all-actor edit evidence 构成，整体可用不推出任一 identity component 可用。H-R33-001 不再修复 generated layer，而提取 H-R13-011 已接受的 observed-support SceneIR `actor_0000` Gaussian chunk 与 logged trajectory，作为明确标注的 baseline/runtime asset；generated identity route 保持 rejected。

### V6-F63：预注册记录的声明时间不得晚于正式 run，即使 source commit 已先冻结

H-R37-001 首个 formal run `20260821T170543Z__trajectory-edit-s20260821-r1` 的方法、阈值与源代码已在 run 前 commit/push，数值上也使两个 1m actor translations 都通过 compiled/native sensor equivalence；但 `HYPOTHESES.jsonl` 内手填的 `recorded_at_utc=2026-08-21T17:15:00Z` 晚于 run directory 的 `17:05:43Z`。该自相矛盾时间戳破坏了预注册审计的机器可验证顺序，因此该 run 不得作为 canonical acceptance，数值只可用于 failure diagnosis。

不得回写旧记录、追认首 run 或仅凭 Git commit 顺序忽略结构化时间字段。H-R37-002 保持相同代码路径、两个 interventions、thresholds、source denominator 与资源合同，在服务器 `date -u` 实时时钟下追加新的预注册记录并重新 commit/push 后复跑；只有 retry 可成为 R37 canonical authority。

### V6-F64：factor consumer 记录 intervention metadata 时必须从冻结 owner 绑定完整字段

H-R39-001 首个 formal attempt `20260821T172447Z__static-contact-s20260821-r1` 在 static KD-tree 和任何 decision 生成前因 `KeyError: translation_delta_m` 失败，仅产生 failed `TERMINAL.json`。R39 config 为两个 intervention 写了预期 contact decision，但 consumer 在输出 decision row 时还读取 delta；delta 的事实 owner 是冻结 R38 payload/decision，初版 config 没有显式重复绑定。

修复只在 R39 config 中补入与 R38 完全相同的 `[1,0,0]` 与 `[0,1,0]`，不得改动 static query、0.80 coverage、0.90 retention、directional control、资源合同或 source denominator。该失败属于 metadata plumbing，不读取或改变实验结果；H-R39-001 在新 commit/push 后按同一假设重试。

### V6-F65：background Gaussian 密度与 actor AABB 极值不能充当 ground-contact evidence

H-R39-001 canonical retry `20260821T172656Z__static-contact-s20260821-r1` 在 824,583 个 observed background Gaussians 上按冻结的 1.5m horizontal / 0.35m vertical / 3-point contract 查询 actor AABB bottom。logged support coverage 只有 `0.17857`，远低于 `0.80`；x+1m coverage `0.16837` 因绝对覆盖不足而 REJECT，y+1m coverage 反而为 `0.19388`，方向控制失败。run 正式 `rejected`。

不得放宽 coverage、vertical tolerance 或把 retention 单独当 ACCEPT。background splats 混合道路、立面与其他表面，Gaussian AABB minimum 又受少量低端 primitives 支配，两者组合不是 ground-contact 语义。H-R40-001 转向冻结的同前端三相机 logged LiDAR：排除 dynamic pixels 后提升到 world frame，以 actor world-Gaussian y 轴 5% 分位作为 robust support anchor、局部 LiDAR y 轴 10% 分位作为 ground proxy，仅评估 R37 实际执行的 frame57；x/y directional controls 与物理/semantic-road abstention保留。

### V6-F66：独立 runner 必须显式绑定仓库根目录后再导入项目包

H-R41-001 首次正式启动在创建 run directory 或读取任何冻结 artifact 前因 `ModuleNotFoundError: No module named 'motion_proj'` 退出。R41 runner 缺少其他 WorldSim V6 runner 已使用的仓库根目录 `sys.path` 引导；实验主体、预注册 factor decisions 与 fusion contract 均未执行，因而这不是方法拒绝。

修复只把 `scripts/worldsim_v6/run_r41_actor_edit_factor_fusion.py` 的仓库根目录插入 `sys.path`，不得修改 R37/R38/R40 hashes、两个 intervention、四因子 decisions、reject-dominates fusion、资源合同或 claim boundary。修复后必须新 commit/push，并以同一 H-R41-001 重跑；首次启动不得被追认为 canonical run。

### V6-F67：手工绑定 source digest 必须逐字符比对实际 SHA256

H-R41-001 第二次正式启动已进入 source verification，但在创建 run directory 或读取 factor decision rows 前拒绝 R37 `MANIFEST.json`。诊断显示配置值第 38 个字符误抄为 `f`：`...e8f42f5946...`，实际冻结 SHA256 为 `...e8f42c5946...`；两者长度均为 64，因此肉眼概览未能发现单字符漂移。源 artifact 本身未改变。

不得跳过或放宽 `_verify`。修复只把 R37 manifest digest 的错误字符改为实际 `c`，其余 R37/R38/R40 hashes、interventions、factor decisions、fusion contract、资源与 claim boundary 全部冻结不变。新 commit/push 后仍按 H-R41-001 重跑，第二次启动不具 canonical authority。

### V6-F68：负值向量 CLI 参数必须用 `--option=value` 绑定，避免 argparse 将其解释为新选项

H-R43-001 首个 formal run `20260821T175434Z__selected-sensor-s20260821-r1` 已完成全部 source/proposal 绑定并创建 run，但 native worker 在加载 checkpoint 前以 rc=2 退出。日志明确为 `argument --translation-delta-m: expected one argument`：选中 proposal 以负数开头的字符串 `-1.0,0.0,-0.5` 被 argparse 解释成新的 option；此前 R37 的正值向量没有暴露这个入口问题。run 只有 failed `TERMINAL.json` 与 worker log，没有 sensor 或 gate。

修复仅把调用形式从两个 argv token `--translation-delta-m`, `<negative-vector>` 改为单 token `--translation-delta-m=<negative-vector>`。不得修改 R42 proposal、translation、renderer worker、任何 sensor threshold、GPU 预算或 claim boundary。该错误不否定 H-R43-001；必须在新 commit/push 后按原假设重试，首 run 永不追认为 canonical。

### V6-F69：verified translation 不应通过破坏性 float32 world-means 重写来拥有 trajectory edit

H-R44-001 canonical run `20260821T180210Z__verified-bake-s20260821-r1` 成功生成自包含 68MB package，所有非 translation actor fields byte-exact、shifted means content-addressed、manifest 完整、双次 bake byte-exact，且 typed validity/abstention 全部保留。但把 `[-1,0,-0.5]` 直接加到原始 float32 world means 后，反算 translation 的最大误差为 `1.9073486328125e-6m`，超过预注册 `1e-6m`，因此 run 正式 `rejected`。

不得把阈值放宽到 2e-6，也不得用舍入后的数组冒充精确 trajectory ownership。H-R45-001 改变表示机制：R35 的全部 actor arrays（包括 base world means）原样 byte-exact 保存，proposal translation 由独立 content-addressed float64 `T_delta_world` trajectory 拥有；runtime 明确按齐次变换组合 base world means。这样 edit 是显式、持久、可验证的，又不迫使高精度 transform 被吸收到 float32 geometry。R44 rejected package 仅保留为失败证据，不得供 runtime 使用。

### V6-F70：trajectory event identity 不能要求每个 timestamp 的 state content hash 唯一

H-R46-001 canonical run `20260821T181019Z__detached-logsim-s20260821-r1` 从完整复制的 detached R45 package 独立加载，196 行/每行 12,390 primitives、组合误差 `0`、导数不变误差 `1.42e-14`、两次 replay aggregate SHA256 完全相同，且 source package 在 copy 后未被 loader 使用。但预注册错误要求 196 个 state content hashes 全部不同；实际只有 `142` 个唯一状态。诊断显示唯一重复组覆盖 `14.1s` 到 `19.5s` 共 `55` 个 timestamp，表示同一个 stationary geometry state 被多个合法 trajectory events 引用。run 因此正式 `rejected`。

不得给 state bytes 掺入 timestamp 以伪造不同 state，也不得删除静止尾段。H-R47-001 明确分离两种身份：`materialized_state_sha256` 继续只哈希几何内容、允许并精确报告 142 个唯一状态；`trajectory_event_sha256` 哈希 sequence index、timestamp、visibility、proposal id 与 state hash，必须对196个事件全部唯一。重复 state group 与55次静止尾段必须原样保留，detached replay、组合精度和 abstention 合同不变。

### V6-F71：actor geometry trajectory 必须同时拥有 native lifecycle，不能把 repeated terminal pose 当作 active actor

H-R49-001 canonical run `20260821T182444Z__multiframe-sensor-s20260821-r1` 在 frames `[0,57,140,141,195]` 上把 R47 detached package 与 R35+同一 delta 两条 runtime 路径逐数组比较，5/5 sensor NPZ 完全相同，runtime modes、event/state identity、repeat、state restoration、package/checkpoint immutability 与资源门均通过。但两条 compiled 路径共同遗漏 native `RigidNodes.instances_fv`：frames 141/195 的 native actor 已 inactive、opacity 为零，compiled package 仍使用固定 observed opacity，导致最大 opacity field error `0.99643`、RGB MAE `0.00501`、depth MAE `0.44633m`。因此 run 正式 `rejected`；cross-path equality 只能证明两个 consumer 同错。

不得删除141/195、放宽 sensor 阈值、把 actor effect 为0解释成无关，或继续把 stationary geometry state 等同于 active lifecycle。H-R50-001 从冻结 StreetGS native `instances_fv[:, actor_0000]` 提取完整196帧生命周期，预注册验证 frames0-140 active、141-195 inactive 的单次边界，并把 content-addressed bool lifecycle 作为独立字段 bake 进 transform-owned package。base geometry、proposal transform 与 R49 rejection 必须原样保留；后续 sensor runtime 必须用 lifecycle 乘 actor opacity。
### V6-F72：下游 perception adapter 必须显式绑定 sensor NPZ 的拥有字段

H-R53-001 首次正式启动 `20260821T185332Z__lifecycle-perception-s20260821-r1` 在产生任何感知输出前失败。冻结 R49/R51 sensor NPZ 使用 `native_rgb` 与 `compiled_rgb` 字段，而复用的旧 R13 worker 硬编码读取 `rgb`，因此抛出 `KeyError: rgb is not a file in the archive`；run 仅有输入 index 与错误日志，没有 label、gate 或方法结论。

修复新增 R53 专用隔离 worker，唯一变化是显式读取 `compiled_rgb`；冻结 R52/R49/R51/model hashes、frames57/141/195、双重复、active exact control、inactive label-change gate、资源和 claim boundary 均不变。不得把 `native_rgb` 偷换为输入或先读取标签结果调阈值。新 commit/push 后按原 H-R53-001 重试，首次启动不具 canonical authority。
### V6-F73：冻结 CUDA 感知 worker 必须在进程启动前绑定确定性 CuBLAS workspace

H-R53-001 第二次正式启动 `20260821T185611Z__lifecycle-perception-s20260821-r1` 已正确读取 `compiled_rgb` 并加载冻结 DeepLabV3，但在首个 forward、任何 label 输出前被 `torch.use_deterministic_algorithms(True)` 拒绝：CUDA>=10.2 的 CuBLAS 需要进程启动前设置 `CUBLAS_WORKSPACE_CONFIG=:4096:8`。run 仍只有输入 index 与错误日志，没有方法结果。

修复仅由 R53 主进程向隔离 worker 环境注入 `CUBLAS_WORKSPACE_CONFIG=:4096:8`，保持 deterministic algorithms 开启；不得关闭确定性模式。冻结 sources、模型、12 次推理分母、active/inactive gates、资源与 claim boundary 均不变。新 commit/push 后仍按原 H-R53-001 重试，第二次启动不具 canonical authority。
### V6-F74：跨 actor 复用 renderer-conformant translation 不保证全轨迹 interaction 可接受

H-R59-001 canonical run `20260821T192557Z__actor2-interaction-s20260821-r1` 将 R58 已通过 native renderer 的 `[-1,0,-0.5]m` translation 应用于 actor2 的完整196帧轨迹。self-kinematics 精确保持，最大 velocity/acceleration invariance error 仅 `8.88e-15/1.78e-13`；但相对 logged baseline 新增7个 AABB overlap events：actor0 在5.3--5.6s共4个，actor5 在6.2--6.4s共3个。因此 interaction factor 正式 `REJECT`，renderer conformance 不能提升为 edit validity。

不得删除发生 overlap 的帧、放宽 AABB gate、用删除4个旧 overlap 抵消新增7个，或因为 R58 sensor 通过就覆盖 R59。H-R60-001 保留 R58/R59 与完整27 actor x196帧 denominator，冻结 x/z 各 `[-2,-1.5,-1,-0.5,0,0.5,1,1.5,2]m` 的80个非零 translation 网格，逐候选要求 self-kinematics ACCEPT 且新增 overlap 为0；按与被拒候选的距离、再按字典序选择最近可接受方案。contact、road、physics 与 safety 继续 ABSTAIN。

### V6-F75：StreetGS 数据 support 提取必须由拥有完整前端依赖的冻结环境执行

H-R62-001 首次正式启动 `20260821T194222Z__actor2-lidar-contact-s20260821-r1` 在读取任何 frame98 support 或产生 contact decision 前失败，`TERMINAL.json` SHA256 为 `592ba630ce84d996ff478b7314a12bfe1d5e0aedb0762bc7270e99ceaa7565d7`。主实验从通用 `motionproj` 环境直接导入冻结 StreetGS `DrivingDataset`，其模型依赖链要求 `pytorch3d`，该环境未安装，因此抛出 `ModuleNotFoundError: No module named 'pytorch3d'`。这属于环境 ownership plumbing，不构成 contact 方法结果。

修复新增隔离的 LiDAR support worker，并用配置中冻结的 DriveStudio Python 环境执行两次 frame98 三相机提取；主实验只读取两个 worker artifact、核验逐数组 repeat-exact 后运行原 contact evaluator。不得改变 R60 proposal、R61/R56/R40 authority、frame98、13,490 primitive denominator、动态像素排除、R40 quantile/radius/0.35m 阈值、预期 ACCEPT 方向、资源上限或 claim boundary。新 commit/push 后按原 H-R62-001 重试，首次启动不具 canonical authority。

### V6-F76：隔离前端 worker 必须同时绑定项目包根目录

H-R62-001 第二次正式启动 `20260821T194557Z__actor2-lidar-contact-s20260821-r1` 已切换到拥有 `pytorch3d` 的冻结 DriveStudio Python，但仍在读取 frame98 前以 worker rc=1 失败，`TERMINAL.json` SHA256 为 `1dcea9a007b18df26c4ff420fd15e8e759a1b814a5ad09184a3a4d22001420b0`。独立复现显示冻结 `DrivingDataset` 还导入项目内 `motion_proj.worldsim_v3.drivestudio_compat`，而新 worker 只加入 checkpoint backup 与 upstream 路径，遗漏当前仓库根目录，触发 `ModuleNotFoundError: No module named 'motion_proj'`。

修复只给 worker 增加显式 `--repo-root` 并在导入冻结 dataset 前插入 `sys.path`，同时令主进程在检查 rc 前落盘 worker stderr。不得改变 Python 环境、数据配置、proposal、frame、接触协议、阈值、预期方向或任何方法分母。新 commit/push 后仍按原 H-R62-001 重试；前两次启动都不具 canonical authority。

### V6-F77：单帧投影 LiDAR 无局部点时不得把 contact 缺证据解释为 edit invalid

H-R62-001 canonical run `20260821T194813Z__actor2-lidar-contact-s20260821-r1` 在 frame98 从三相机重复提取出完全一致的 `7,183` 个静态 logged-LiDAR world points，actor2 的13,490 primitives、生命周期、R60 proposal 与全部 authority 均精确绑定。但冻结2m局部查询在 logged 与 `[-1,0,0]m` 编辑中心附近都得到0个候选，最近水平距离分别为 `4.1758m` 与 `3.9666m`，因此两者 contact error 都不可计算并正式 `REJECT`。这证明的是单帧稀疏 support 不足，不是编辑破坏地面接触。

不得放宽2m半径、降低32点分母、增大0.35m误差阈值、删除 logged baseline 控制，或把两个 REJECT 宣称为编辑无效。H-R63-001 固定使用 target frame98 前后各10帧的对称21帧窗口 `[88,108]`，逐帧排除 dynamic pixels、提升到同一 world frame，并沿用已接受 R13 的0.05m deterministic voxel union；随后完全复用 R40 的 quantile/radius/denominator/error 阈值评价同一个 logged/selected pair。semantic road、physics、planning 与 safety 继续 ABSTAIN。

### V6-F78：相机投影 LiDAR 子集的时间融合仍可能无法覆盖远距 actor 接触邻域

H-R63-001 canonical run `20260821T195625Z__temporal-lidar-contact-s20260821-r1` 精确保留 R62 frame98 support，并从冻结21帧三相机数据得到 `149,723` 个 raw projected static observations、0.05m 去重后 `25,798` 个 world points；worker 双提取、坐标、窗口和 source gates 全部通过。但 logged 与 selected actor2 的2m邻域仍各为0点，两个 contact 均正式 `REJECT`。actor2 是 `vehicle.car` 且88--108帧间移动约 `1.97m`，因此失败说明相机投影稀疏子集在远距/遮挡区域不能承担 contact map，而不是简单增加同类帧数即可修复。

不得扩大投影时间窗、放宽 contact gates 或用环带高度挑选靠近 actor anchor 的平面。H-R64-001 改用同一 processed scene 的21帧360度 raw LiDAR，按每帧全部标注3D box加0.10m固定边界排除动态点，再做0.05m world voxel union；45个 raw/pose/box 输入以预先计算的聚合 SHA256 冻结。contact evaluator、logged baseline、selected proposal与全部阈值保持不变，semantic road、physics、planning与safety继续ABSTAIN。

### V6-F79：中心圆查询与 Gaussian 低分位 anchor 不适用于大尺寸 actor 的 box-filtered contact

H-R64-001 canonical run `20260821T200653Z__raw-lidar-contact-s20260821-r1` 从21帧360度 LiDAR 获得 `729,568` 点，按全部标注 box 排除 `95,432` 个动态点并形成 `73,010` 个静态5cm voxels；source、变换、动态过滤与资源门均通过。但 actor2 logged 中心查询虽有37点，Gaussian y-5% anchor 与 ground proxy 相差 `2.415m`；selected 只有18点且误差 `0.934m`，两者正式 `REJECT`。审计发现 model actor2 唯一对应 processed instance7 `vehicle.truck`，其 local +z 映射到 world -y，所以 Gaussian y-5% 是上表面而非接触底面；同时12.153m长的 truck 在 box-filter 后中心区域本就形成观测空洞。

不得继续把 R40 针对 actor0 偶然通过的中心/低分位定义当作跨 actor ground owner，也不得降低32点门。H-R65-001 显式绑定最近且有大间隔的 instance7 box，使用标注 local -z 底面拥有 contact anchor，并在 oriented footprint 外固定1m边界环查询 raw static voxels；环内用 median world-y 抑制不同高度表面，要求每个 intervention 至少64点、误差仍不超过0.35m。box只拥有几何位置，不提供 ground 高度；logged 与 selected 均须独立通过。

### V6-F80：actor package schema 分母必须把独立 lifecycle 计入 base arrays

H-R67-001 canonical run `20260821T202013Z__actor2-transform-bake-s20260821-r1` 成功产生 transform-owned package：float64 composition error 为0、双 bake byte-exact、196 transforms、13,490 primitives、四因子与 abstention 均正确，且所有源数组 hash 实际逐项相等。唯一失败是预注册把“7个 Gaussian/trajectory arrays”误写成 package 的完整 base array count=7；R56 还拥有独立 `actor_frame_validity`，实际键为8个，因此 `all_seven_base_arrays_byte_exact` 按合同正式失败。

不得删除 lifecycle、追认 H-R67-001、忽略数量门或改变任何 package bytes。H-R67-002 保持相同 R66/R56、proposal、transform、composition、repeat、资源与 claim boundary，只把 schema 分母明确为8，并同时要求8个 hash 全等且 `actor_frame_validity` 必须存在；新 commit/push 后重跑，首 run 保持 rejected authority。

### V6-F81：本地 shell 不得展开远端 Git 预检的命令替换

H-R68-001 第一次启动尝试在创建 run directory、读取任何冻结 artifact 或启动 GPU worker 前退出。PowerShell 在 SSH 到达服务器前展开了双引号字符串中的 `$(git status --porcelain)` 与 `$(git rev-parse ...)`，本地当前目录不是 Git 仓库，继而使传给远端 bash 的引号不闭合。服务器只收到语法错误；没有 R68 run、sensor、gate 或方法结果产生。

修复只把 SSH 的远端命令改为 PowerShell 单引号字面量，并把可执行 runner 的物理文件模式同步为已提交的 `100755`；不改 R68 代码、配置、冻结 source、frame98、actor2、transform/lifecycle ownership、sensor exactness、阈值、资源或 claim boundary。H-R68-001 保持 active，在 clean 且已 push 的同一 source commit 后重新启动。该失败是 launcher plumbing，不构成假设 rejection。

### V6-F82：multi-actor sensor 证据帧必须让每个被编辑 actor 对相机具有独立可见支持

H-R71-001 canonical run `20260821T205641Z__two-actor-sensor-s20260821-r1` 在 frame98 正确加载 R70，向 native checkpoint 同时应用 actor0/actor2 两个 transform，并在 compiled path 替换两组 owned fields。两组 field error、共享 RGB/depth/opacity、repeat、state restore、package/checkpoint immutability、资源和全部 abstention 均通过；但 actor0 虽 lifecycle active 且12,390个 opacity primitives 非零，其 camera effect pixels 为0。联合 sensor SHA256 因而与 R61/R68 actor2-only sensor 完全相同，只有 actor2 的19,785 effect pixels，按预注册的双 actor 可见门正式 rejected。

不得删除 `both_actors_have_visible_effect`、把 active primitives 当作 camera evidence，或追认 frame98 为双 actor runtime 成功。冻结 R51 证据表明 actor0 在采样帧中只有 frame57 具有17,568 effect pixels；冻结 R57 表明 actor2 在相邻 frame49 和后续 frame98 分别具有17,290/19,700 effect pixels。H-R71-002 因此在任何新渲染前固定 frame57，并改用冻结 R36 frame57 logged sensor 作 counterfactual baseline；所有 runtime、field/sensor、每 actor>=32 pixels、joint>=256 pixels、资源和 abstention 门保持不变。若 actor2 在 frame57 仍不可见，则保留第二次 rejection 并转向独立的可见交集搜索，不得继续猜帧。

### V6-F83：手工复制 source SHA256 时不得遗漏重复的相邻字节组

H-R82-001 第一次正式启动在创建 run directory、复制 package 或开始任何 bake 前，被冻结输入校验拒绝。R82 配置把 R70 `MANIFEST.json` 的实际 SHA256 `1583baf70c760ab700992ef9573ceb6fe59f992527445ac5eb5eb99f7795e6fe` 误抄为少一个 `5e` 字节组的 `1583baf70c760ab700992ef9573ceb6fe59f992527445ac5eb99f7795e6fe`；实际 artifact 与 R71 已使用的冻结 authority 均未变化。本次没有 run、gate、package 或方法结果。

不得跳过 `_verify`、重新生成 R70 artifact、放宽 package denominator 或追认本次启动。H-R82-002 仅修正该 source digest，并保持 R70/R80/R81、三个 actor、45 payload、34,257 primitives、588 trajectory rows、双 bake exact、资源和 claim boundary 全部不变；必须在新 commit/push 后重试。

### V6-F84：retry 配置的 hypothesis_id 必须与追加的预注册记录一致

H-R82-002 run `20260821T215606Z__three-actor-package-s20260821-r1` 数值上通过全部 package gates：3个 actor、45个 payload、34,257 primitives、588 trajectory rows、195,658,443 bytes，三棵 actor package tree byte-exact且双 bake 完全一致。但 digest repair commit 只修正 source SHA256，遗漏把 YAML `hypothesis_id` 从已关闭的 `WS-V6-H-R82-001` 更新为 active 的 `WS-V6-H-R82-002`，因此 SUMMARY 错绑旧 hypothesis。该 run 不得作为 canonical acceptance。

不得回写 run、把数值通过覆盖 provenance mismatch 或重新编号旧记录。H-R82-003 只把 YAML hypothesis binding 更新为 `WS-V6-H-R82-003`，保持已验证的 R70/R80/R81 hashes、bake bytes、所有 denominators、资源与 claim boundary 不变；新 commit/push 后重跑。

### V6-F85：稀疏固定时点的 actor lifecycle 有效性不能替代前视相机可见性

H-R95-001 canonical run `20260821T234324Z__scene0048-actor-visibility-s20260821-r1` 在第二个独立 scene0048 matched-formal30k checkpoint 上完整枚举9个 RigidNodes actor、196帧 lifecycle 与15,717个 primitives；source、partition、checkpoint immutability、GPU 和全部 denominator gate 均通过。但在预注册的 frames `[0,49,98,147,195]` 前视相机中，所有9个候选的 actor-only effect pixels 均为0，最大值仍为0，低于非平凡可见门64，因此 H-R95-001 正式 `rejected`。这证明固定五时点 lifecycle-active 不能推出 camera-visible support，并不否定 scene0048 checkpoint 或 actor 表示。

不得删除64像素门、把非零 primitives/opacity 当作屏幕可见、改用事后选中的单帧，或追认 R95 为成功。H-R96-001 保持相同 checkpoint、9个候选、前视相机、opacity阈值0.01和选择规则，改为在单个冻结进程内穷举全部196帧的所有 lifecycle-active actor/frame 对；仅在完整分母上按最大 effect pixels、actor index、frame index确定性选择。若穷举仍为0，则保留第二次 rejection 并转向三相机覆盖实验，而不是继续猜前视帧。

### V6-F86：全时域 sensor conformance 不得要求生命周期外的 actor frame_valid 恒为真

H-R98-001 canonical run `20260822T001113Z__scene0048-selector-transfer-s20260821-r1` 完成196帧 logged/edited RGBD、784次冻结 DeepLab 推理与全部资源/immutability 分母。零校准 threshold45 在 scene0048 得到 TP=30、TN=166、FP=0、FN=0，precision/recall/F1=1、skip=84.69%，优于 fixed256 的 F1=0.9831 与原生36帧 lifecycle 的 F1=0.9091。但预注册 gate 把 `package_actor_frame_valid` 作为196帧均须为真的 sensor-conformance 合取项；actor8 的冻结 lifecycle 正确地仅在 frames160..195 为真、frames0..159 为假，因此唯一方法检查 `all196_compiled_native_sensor_conformant` 与总 `passed` 为假，run 按合同正式 `rejected`。逐项诊断确认196/196帧数值 conformance、repeat 与 native state restoration 全部通过，最大 RGB MAE `1.36e-8`、depth MAE `6.19e-7m`。

不得追认或回写 R98、不得把160个 inactive frame 改成 active、不得删除全196帧 sensor 数值门，也不得重跑昂贵推理来掩盖治理错误。H-R99-001 只把合同修正为“每帧 `package_actor_frame_valid` 必须与冻结 native lifecycle 精确相等”，从 R98 的内容寻址 sensor/perception artifacts 独立重算196帧 conformance、784输出重复性与 selector 指标；R98 保持 rejected，R99 作为新的 governance-repair authority。

### V6-F87：跨场景配置不得凭摘要转抄 checkpoint authority

H-R102-001 首次正式启动在创建 run directory、读取 sensor 或启动 GPU worker 前被冻结输入校验拒绝。配置误用了不存在的 scene0255 matched-baseline 路径 `20260812T132516Z__streetgs-scene0255-matched-formal30k-s0-r50`，并把 R101 摘要中截断/误记的 digest `dba249822f22317d926cc2953d0a433f6a95e6963d35e42750b8f7074dad6acd` 当作 authority；R101 实际已通过的冻结 checkpoint 是 `20260811T214009Z__streetgs-scene0255-matched-formal30k-s0-r48`，SHA256 为 `dba24982a3f25e162b5e293165258a588cf9bd7a49e54e05d0d052de703cb2d2`。本次没有 run、gate、sensor、perception 或方法结果。

不得跳过 `_verify`、制造不存在的 checkpoint、把 launcher 失败解释成 selector rejection，或继续从人工摘要抄 authority。H-R102-002 只从已接受 R101 配置复制确切 checkpoint 路径与 SHA256，并更新 hypothesis binding；R101/R90 artifact、actor34 edit、196/784 分母、threshold45、逐帧 lifecycle 合同、资源与 claim boundary 全部不变。必须在新 commit/push 后重试。

### V6-F88：零 AABB interaction 不意味着 RGB factorial interaction 必须非零

H-R111-001 canonical run `20260822T022113Z__scene0255-two-actor-factorial-s20260821-r1` 精确绑定 R102 actor34-only、R110 actor24-only 与 R109 joint 的 00/10/01/11 冻结 sensor/perception arrays。三个 source 的 196/784 分母、repeat、logged cell 与 hashes 全部一致；actor34/actor24 条件边际分别覆盖 19/161 帧，joint 与 single-target union 的帧级 F1=1，像素 F1=0.999086，single-selector OR 对 joint target 的 F1=1。但预注册错误要求至少 1 个 RGB pixel 的 `rgb11-rgb10-rgb01+rgb00` 绝对残差超过 1/255；实际 196 帧的最大残差、平均残差与超阈像素数全部严格为 0，因此唯一 `sensor_factorial_interaction_detected` gate 失败，run 正式 `rejected`。

不得删除该 gate、追认 R111 或把精确 superposition 描述成非线性 renderer evidence。H-R112-001 以 R111 rejection 为冻结诊断 authority，改测与观测一致的新机制：sensor 层必须逐值 exact affine superposition；下游冻结 DeepLab 允许有界的非线性 residual，但 joint/single-union 像素 F1 必须不低于 0.995、对称差比例不高于 0.005，两个 actor 条件边际与帧级/selector-OR exactness 仍须保留。semantic correctness、local causality、contact、dynamics、physics、planning 与 safety 继续 ABSTAIN。

### V6-F89：正式 runner 的模块归属必须与导入路径一致

H-R116-001 首次正式启动在创建 run directory、读取冻结 artifact 或启动 GPU worker 前，以 `ModuleNotFoundError: No module named 'motion_proj.worldsim_v6.r116_scene0255_fourth_actor_edit_compiler'` 退出。入口脚本正确从项目包 `motion_proj.worldsim_v6` 导入主体，但实现文件被错误提交到 `scripts/worldsim_v6`；因此本次没有 run、gate、sensor、proposal、GPU 结果或方法结论，H-R116-001 按实现合同记为 infrastructure rejection。

不得通过临时修改 `PYTHONPATH`、从未承诺的工作树文件导入、忽略失败启动或把它追认为 actor1 方法结果。H-R117-001 仅修复模块 ownership：主体放入 `motion_proj/worldsim_v6`，runner 仍位于 `scripts/worldsim_v6` 并从项目包导入；actor1、frame195、4,489 effect pixels、838 Gaussians、196-frame lifecycle、80 translations、所有 source hashes、阈值、资源上限和 claim boundary 保持不变。必须在新 commit/push 后从干净工作树重新正式运行。

### V6-F90：RGB-difference 邻域不能假定覆盖冻结感知的全局标签响应

H-R122-001 canonical run `20260822T034305Z__spatial-impact-locality-s20260821-r1` 精确绑定 R118/R121 两个四 actor 饱和方向的392帧 sensor 与 frozen-DeepLab arrays，所有 source hashes、逐文件 hashes、196/196 正帧和资源门均通过。但预注册把实际 `450x800` 图像误写为 `600x1200`，因此 shape gate 正式失败；更关键的是方法门也独立失败：RGB-diff mask 膨胀64px 后聚合标签召回仅 `0.604784`，最差帧仅 `0.027397`，不存在预注册网格内逐帧100%覆盖半径。R122 按合同正式 `rejected`，不得因 shape 书写错误而追认其空间局部性假设。

不得只修正 shape 后删除 exact per-frame coverage、把60.48%聚合召回解释为稀疏验证成功、依赖平均 ROI 掩盖最差帧，或宣称 crop inference 等价。非 canonical 恢复诊断显示392/392帧所需半径均大于128px、391/392帧大于256px，中位所需半径约412px、最大约684px；固定256px平均已覆盖73.56%画面却仍只有92.90%标签召回。H-R123-001 必须用正确450x800分母和扩展半径网格正式复算这些非局部性边界，接受的结论应是拒绝 RGB-diff 膨胀稀疏机制，而不是放宽成近似覆盖。semantic correctness、crop equivalence、speedup、physics、planning 与 safety 继续 ABSTAIN。

### V6-F91：跨实验 source digest 必须直接复制磁盘 SHA256，不能依赖人工转抄

H-R124-001 首次正式启动于 `2026-08-22T03:57:05Z` 前在创建 run directory、聚合任何向量或产生方法结果之前被 source verifier 拒绝。R109 `SELECTOR_TRANSFER.json` 的配置 digest 被人工转抄为 `0b972e5d0ff102c1eda06a2b077f769fe836f4b3d856242f4df58dbecafc6eafd91`，而冻结磁盘文件的实际 SHA256 是 `0b972e5d0ff102c1eda06a2b077f769fe836f4b3d856242f4df75406faeafd91`。本次没有 canonical run、gate、聚合向量、指标或科学结论，按 infrastructure/source-authority rejection 记录。

不得跳过 `_verify`、修改 R109 artifact、追认 H-R124-001，或调整 threshold45、11条件、2156帧、类别支持、分离间隔及资源门来掩盖该错误。H-R125-001 只把 R109 selector digest 改为磁盘实值并更新 task/hypothesis identity；其余 policy/source authorities、condition corpus、门限、预期方向、资源合同和 claim boundary 全部保持不变，必须在新 commit/push 后从干净工作树正式运行。

### V6-F92：语料内精确阈值不能未经新条件检验就提升为前瞻不变量

H-R128-001 canonical run `20260822T042521Z__scene0230-orthogonal-holdout-s20260821-r1` 在预注册后新生成 scene0230 actor12 `[0,0,+0.5]m` 的196帧 sensor 与784个冻结 DeepLab 输出；所有 source、proposal、0新增 overlap、38,541 primitives、196帧 lifecycle、compiled/native sensor、repeat、GPU、wall 与 abstention gate 均通过。但 threshold45 在78个正帧中漏掉 frame77：RGB changed pixels 为26而 frozen-label changed pixels 为5，得到 TP77、FN1、TN118、FP0、recall0.987179、F1=0.993548。run 按 zero-error 合同正式 `rejected`。

不得追认 R128、删除 frame77、把5个标签像素降为无关、放宽 F1/recall，或在同一 holdout 上改 threshold 后宣称前瞻成功。诊断显示全部118个负帧的最大 RGB feature 仍为0、78个正帧的最小值为26，使包含 R128 的开发并集精确阈值区间缩为 `[1,26]`。H-R129-001 必须把 R128 明确降格为 threshold-revision development evidence，与 R126 的2156行和 R127 的196行合并，按预注册 max-min margin 规则选择 threshold13并只声明开发集精确性；随后必须在另一个新条件上做独立前瞻检验。

### V6-F93：AD-GS 全时域实验必须区分 train、development 与锁定的 heldout camera 分区

H-R134-001 首次正式启动 `20260822T053238Z__adgs-cross-frontend-threshold13-s20260821-r1` 在任何 sensor 或 perception 输出产生前，于请求 frame0 时退出。冻结 R3 adapter 的196时间轴只物化了118个 train、39个 development 时间步并刻意排除39个 heldout 时间步；worker 仅从 `getTestCameras()` 建表，因此只看见 development，frame0 不存在。失败目录仅4 KiB、sensor 文件数为0，不构成 threshold13 或 cross-frontend 方法结论。

不得把 heldout 图像补入 adapter、把缺帧静默删除后仍声称196分母、追认 H-R134-001，或把本次启动失败解释为 AD-GS transfer rejection。H-R134-002 只能合并 `getTrainCameras()` 与 `getTestCameras()`，从冻结 `partition.json` 预先导出 camera0 的精确118+39=157帧，并保持39个 heldout 未读；AD-GS checkpoint/edit、threshold13、正负支持、0 FP/FN、skip、资源门与所有 abstention 不变，新 commit/push 后重新正式运行。

### V6-F94：StreetGS 上冻结的单一 RGB 像素阈值不能直接宣称跨 frontend 不变

H-R134-002 canonical run `20260822T053744Z__adgs-cross-frontend-threshold13-s20260821-r2` 在 heldout 未读的前提下完成 AD-GS scene0048 的118 train+39 development 帧、157组 logged/edited sensor 与628个重复精确 DeepLab 输出；checkpoint、adapter、aggregate actor state restoration、分区、GPU、wall 与所有 abstention gate 均通过。冻结 StreetGS threshold13 在131个正帧、26个负帧上得到 TP130、FN1、TN26、FP0、recall0.992366、F1=0.996169，唯一漏检为 train frame13：RGB changed pixels=1、label changed pixels=1；run 按0-error合同正式 `rejected`。

不得追认 R134、删除 frame13、把1个标签像素降为无关、放宽 recall/F1，或把 AD-GS 数据用于回改 StreetGS threshold13 后仍称全局策略。诊断显示26个 AD-GS 负帧的最大 feature 为0、131个正帧的最小 feature 为1，开发区间是脆弱的单点 `[1,1]`。H-R135-001 只能显式声明 frontend-conditioned router：StreetGS 保持13，AD-GS 用 R134 开发集按预注册规则拟合为1；R134 保持 rejected，且 AD-GS 的39个 heldout 时间步必须在 policy freeze 后一次性验证。
### V6-F95: AD-GS threshold-1 exact classification does not survive the sole heldout confirmation

H-R136-001 canonical run `20260822T055538Z__adgs-heldout-confirmation-s20260821-r1` consumed the one allowed confirmation attempt before reading heldout quality. All 39 camera-0 heldout frames, 78 AD-GS renders, and 156 frozen DeepLab outputs completed within contract. Source immutability, adapter partitioning, repeat exactness, actor-state restoration, positive/negative support, skip, GPU, wall, output budget, and all abstention gates passed. The frozen R135 AD-GS threshold 1 nevertheless produced TP=31, TN=7, FP=1, FN=0: frame 14 changed 11 RGB pixels but changed 0 label pixels. Precision was 0.96875, recall 1.0, F1 0.984127, and the run was correctly rejected.

Do not retune a scalar threshold on these heldout rows, rerun the consumed candidate, delete frame 14, or reinterpret conservative over-execution as exact classification. The next hypothesis changes method family and objective: an exact-input identity guard may reuse cached perception only for byte-identical RGB inputs and must execute otherwise. R137 evaluates that one-sided operational contract on R134 development data plus the already frozen R133 StreetGS execution authority; R136 remains rejected.
### V6-F96: A negative comma-separated translation must not be passed as a detached argparse value

H-R138-001 canonical failed run `20260822T061548Z__adgs-antithetic-exact-input-s20260821-r1` created the exact-once attempt and completed train+heldout adapter materialization. The sensor subprocess then exited in argument parsing before frame 0: the detached token `-0.5,0.0,0.0` was treated as an option, so `--translation-world` reported that its argument was missing. No sensor array or perception output was produced, and the exact-input method was not measured. The attempt is nevertheless consumed under the preregistered any-outcome rule.

Do not rerun the same antithetic condition, claim a method rejection, or edit its run into a success. R139 uses a distinct world-z +0.5m condition and binds the vector as `--translation-world=<csv>`, while preserving the exact-once, 39-frame heldout, full/reference selective execution, identity-only reuse, reconstruction, support, resource, and abstention gates.
### V6-F97: Python booleans in formal JSON closeout code must use `False`, not JSON `false`

H-R140-001 failed run `20260822T063253Z__end-to-end-utility-s20260821-r1` verified all immutable inputs and wrote the end-to-end certificate, gate, and summary, but exited before RESOURCE_AUDIT, MANIFEST, and TERMINAL. The resource dictionary used the undefined Python name `false` for `gpu_used`, raising `NameError`. No GPU, training, confirmation read, or source mutation occurred. Although the partial gate was written, it is not a canonical acceptance because terminal closeout is incomplete.

Do not hand-create missing success artifacts or promote the partial gate. H-R140-002 changes only `false` to `False` and updates the YAML hypothesis binding; all sources, formulas, three conditions, zero-error authorities, end-to-end thresholds, resources, and claim boundaries remain unchanged for a new clean-commit run.
### V6-F98: A literal recovery must search the whole Python module, not only the first failing line

H-R140-002 failed run `20260822T063601Z__end-to-end-utility-s20260821-r1` reproduced the same certificate and gate as H001, then failed on the immediately following `training_started: false` field. The first recovery changed only `gpu_used`, leaving two lowercase JSON booleans in Python source. Again, no GPU, training, confirmation read, or source mutation occurred, and the partial gate is not canonical.

Do not promote either partial run or continue one-line-at-a-time repair. H-R140-003 is preregistered after an exhaustive `true|false|null` token search. It changes the exactly two remaining resource-audit values (`training_started`, `confirmation_content_read`) to Python `False` and updates the hypothesis binding; all scientific inputs, formulas, thresholds, denominators, budgets, and claim boundaries remain fixed.

### V61-F02：下游 runner 必须读取上游 gate 的真实嵌套 authority

H-ME1-001 正式入口完成所有冻结文件 hash 校验后，把 `ME0_GATE.json` 的通过位误读为顶层 `passed`，而
`worldsim_v61.me0_gate.v1` 的真实 authority 是 `checks.passed`。因此触发 `KeyError`；异常发生在 run directory
创建、O_method/O_eval tensor 读取、GPU ray compiler、proposal 编译和任何方法计算之前。canonical run=`null`，
不存在 oracle upper-bound 科学结果，不能把本次记成 method rejected。

不得跳过 ME-0 authority、修改 canonical ME-0 artifact、放宽 ME-1 gate，或把 launcher failure 追认为科学 attempt。
H-ME1-002 只把读取路径修正为 `document["checks"]["passed"]` 并增加嵌套 schema 回归；28-case、五臂、source
hashes、0.2m voxel/0.1m ray step、50% coverage、20% depth consistency、false-safe/stop rule 与资源预算全部不变。

### V61-F03：合法 actor ID 0 不得与 raster 的空身份 sentinel 共用

ME-2 actor control 准备审计发现，ME-0 的 scene-0048 sparse identity layer 合法包含 actor ID `0`，但 ME-1
相机 raster 用零初始化 `actor_grid`，导致 actor0 与“该 voxel 无 actor”无法区分。ME-1 primary O2 的10个 ACCEPT
全部来自 scene-0242，scene-0048 两个 actor case 均已由冻结 P1 REJECT，因此 O2=`10/28`、false-safe=`0`、
mask yield 与 primary gate 不受影响；但 canonical ME-1 的 O3 scene-0048 identity/swept 诊断不能提升为完整 actor 结论。

不得把 actor0 改号、删除 scene-0048、追认 O3 actor safety，或为此重跑 ME-1 主臂。后续实现把 empty sentinel 改为
`-1` 并增加 actor0 回归；ME-2/ME-4 必须消费修复后的 identity raster，已落盘 ME-1 run 保持不可变。

### V61-F04：冻结 source digest 必须先满足 SHA-256 的 64 字符结构合同

P4 第一次正式入口在创建 run directory、载入模型或占用 GPU 前，被 VAE source gate 拦截。实际固定 revision
`70e803bfb4e127d534049d8ab8c8cb511780d485` 的 VAE 文件为 `1311145138` bytes，实际 SHA-256 与服务器
`X-Linked-ETag` 均为 `379995ca170d8a899019125f389ba8692b2e35625ff64ddc3fdaa8c9302ac340`；预注册配置在
末尾误多录一个 `2`，形成 65 字符值。模型字节没有漂移，canonical run=`null`，不存在 capability 科学结果。

不得跳过 source gate、改写模型文件或重复下载。修复只删除多录字符，并新增所有 model digest 必须是 64 位小写
十六进制的回归测试；官方 commit、model/DINO revision、demo、seed、50 steps、512 octree、资源门与 stop rule 全部不变。

### V61-F05：离线 Hugging Face repo-id 解析必须有显式 revision ref

P4 第二次正式入口通过全部 source gate 并创建 failed run
`20260822T111747Z__voxel-smoke-s1234-r1`；Omni DiT 与 VAE 均以 0 missing/0 unexpected 成功载入，随后
`Dinov2Model.from_pretrained("facebook/dinov2-large")` 在离线模式失败。固定 snapshot 与三个文件已完整存在，但按
exact commit 下载不会自动创建默认 `refs/main`；官方 encoder 只传 repo-id、未传 revision，因而无法把默认 main
解析到已缓存 snapshot。该 run 没有生成 mesh/points 或 capability gate，不是模型能力 rejection。

不得开启正式 run 网络、重复下载 DINO、改官方 encoder 或更换 backbone。修复只按 Hugging Face 标准 cache schema
创建 `refs/main`，内容精确绑定已冻结 commit `47b73eefe95e8d44ec3623f8890bd894b6ea2d6c`；runner 在模型载入前
验证 ref、snapshot、config 与 model SHA，之后仍保持 `HF_HUB_OFFLINE=1` 和 `TRANSFORMERS_OFFLINE=1`。

### V61-F06：Hugging Face cache ref 是无换行 commit token，不是普通文本行

P4 第三次正式入口创建 failed run `20260822T112159Z__voxel-smoke-s1234-r1`，再次在相同 DINO 离线解析点失败。
运行时常量确认 `huggingface_hub.HF_HUB_CACHE` 与 `transformers.TRANSFORMERS_CACHE` 都精确指向预期 cache root，
排除了环境变量和根路径猜测。直接审计已安装 `huggingface_hub.file_download.try_to_load_from_cache` 发现，它对
`refs/main` 使用原样 `f.read()`，不执行 `strip()`；由普通文本 staging 上传的 ref 是 41 bytes，尾部 `0a` 使
revision token 与 40 字符 snapshot 目录不相等。该 run 仍未生成 mesh/points 或 capability 结果。

不得继续猜 cache roots 或重复完整 P4。修复把精确目标 ref 机械规范化为 40 bytes，并让 runner 也要求 byte-exact
40 字符内容；先单独执行一次 repo-id 的离线 DINO load smoke，只有它通过后才允许下一次正式 P4。模型与参数不变。
修复后孤立 smoke 由 repo-id 离线载入 `Dinov2Model` 的 `304368640` 个参数，故解析卡点已关闭。

### V61-F07：共享 shape 环境必须显式包含 image-only backend 的官方导入依赖

H-ME2-001 failed run `20260822T120008Z__hy3d-actor-s1234-r1` 已通过全部冻结 source gate 并构造4个 actor
inputs，随后 A0 worker 导入官方 Hunyuan3D-2.1 package 时失败。该 package 的 `__init__.py` 无条件导入
`postprocessors.py`，后者依赖官方 `requirements.txt` 精确固定的 `pymeshlab==2022.2.post3`；既有 Omni
shape-inference 环境没有这个 image-only backend 依赖。失败发生在模型载入、GPU inference、asset 生成和 method
decision 之前；canonical run=`null`，不是 A0 或 A3 能力结论。

不得跳过 A0、patch 官方 `__init__.py`、安装无关 texture/UI 全依赖、改变四臂或把本次追认为方法 rejected。
H-ME2-002 只安装官方固定的 `pymeshlab==2022.2.post3`，在配置/runner 增加 exact version gate，并执行一次
离线 base pipeline import smoke；该 smoke 已成功导入 `Hunyuan3DDiTFlowMatchingPipeline`。模型、权重、
4 units/6 cases、seed、batch、steps、octree、compiler、truth separation、thresholds、资源和 stop rule 全部不变。

### V61-F08：Omni diffusion 支持 batch>1，不代表默认 marching-cubes extractor 也支持

H-ME2-002 failed run `20260822T120519Z__hy3d-actor-s1234-r1` 完成4个有效 A0 mesh，并成功载入 Omni、完成首个
A1 的2-sample diffusion 与 VAE implicit query。runner 随后发现输出 mesh 数为1而输入数为2并 fail-closed。
官方源码显示 `extract_geometry_vanilla` 虽把 logits reshape 为 `(batch_size,X,Y,Z)`，但 marching cubes 固定读取
`grid_logits[0]`，wrapper 也固定返回一元素 list；因此第二份 latent 没有 mesh，不是随机空输出或 OOM。
本次没有 A1/A2/A3 asset、method decision 或科学结论；canonical run=`null`。

不得静默丢弃第二样本、把6-case缩为3-case、patch官方源码、把全部 diffusion 降为batch1后冒充并行，或改变生成
参数。H-ME2-003 保持昂贵 diffusion batch2与逐样本generator，令官方 pipeline 返回两份 latent，再逐份调用同一
官方 VAE 的 batch1 decode/export。H002 已完成的4个 A0 只在 plan、input hashes、report 与每个 asset hash 全部
精确后复用，不重复GPU计算。模型、controls、seed、steps、octree、guidance、compiler、truth、threshold与stop rule不变。

#### V6-F97/V6-F98 recovery 收口

H-R140-003 从干净且已推送的源提交 `a13759ba8db03e1f740ad93e246ca24f0ff2d7fa` 完成 canonical run `20260822T063937Z__end-to-end-utility-s20260821-r1`。Scientific certificate SHA256 为 `913833af47e4171e27707f71418b6625ed358b538d1c8a5a18bca5ac7f585363`，与两次 partial computation 完全一致；完整 gate、summary、manifest、resource audit 与 terminal 的 SHA256 依次为 `ac3c79c0e93f2932a076da8323b89a210ff2cbaac27ffa13079ce89ae9d07b51`、`50900ff99736055a10c32f4362176b7fc87862ae84667591077d6c17024e635b`、`1cc753b3c0a9489ced2a58b23035466ee26cba963d2abc56c84ebd4d057e5a62`、`06c110236591529d5fef5f4178bfed696b6c0ad0cfbce94c497896dc92230265` 与 `be263ba010cdb936fbd01dbfa0fe294b8022101aae348c95030d2a42d45fdb77`。

该 recovery 不删除或重分类 V6-F97/V6-F98：两个失败目录继续保持不可变，只有 H-R140-003 是 canonical。完整 account 报告 StreetGS、AD-GS development 与 AD-GS exact-once confirmation 的端到端 reduction 分别为 13.5337%、11.1434% 与 1.66365%（macro 8.78024%、worst 1.66365%），reconstruction error 为 0。Selector 研究族在此次 recovery 后冻结；R141 未执行，本收口不授权继续 threshold、actor 或方向实验。

### V61-F09：通用生成表面不能冒充场景观测一致的 Occupancy

H-ME2-003 canonical run `20260822T121848Z__hy3d-actor-s1234-r1` 完成固定四臂各 6 例。A0 image、A1 bbox、
A2 raw-LiDAR point 与 A3 O_method voxel 全部为 `0/6 ACCEPT`。主臂 A3 没有 false-safe，但每例都在 method 和
独立 O_eval 中占据已观测 FREE cell；method conflict=`6..246`，eval conflict=`8..273`。所有四臂均有同一失败，
而 A3 的 native coverage、hole coverage、silhouette 与 extent 多数已达到冻结下限，因此不是提示词、seed、纹理或
单一轮廓阈值问题。

该失败是科学机制 rejection：Hunyuan 输出的通用闭合 actor surface 没有把场景 FREE-space 作为硬约束，不能作为
Occupancy-authoritative proposal。按预注册规则永久停止本版本 Hunyuan actor 路线；不得靠 prompt/seed/steps/
octree sweep、放宽 FREE=0、事后 clipping 或 per-case 选择恢复。ME-3 学习式 occupancy 是计划中的独立机制，
仍按 GaussianWorld→OccWorld→Drive-OccWorld→IR-WM→OccSora 优先级审计，不把本失败无依据外推到该路线。

### V61-F10：tmux 正式入口必须由 wrapper 自举仓库根目录

H-ME3-GW-001 的第一次正式入口从 source=`16c0efd8d570eaa15c5c4757ddfb434af8b61ede` 启动，但 tmux
非登录环境没有仓库级 `PYTHONPATH`；Python 把 `scripts/` 而不是 repository root 放在 `sys.path[0]`，wrapper
因此在导入 `motion_proj.worldsim_v61.me3_predicted_experiment` 时立即触发 `ModuleNotFoundError`。失败发生在 run
directory 创建、source/artifact 读取、模型载入、GPU context、predicted occupancy 或 method decision 之前；
canonical run=`null`，没有科学结果，不能记为 GaussianWorld rejection。

H-ME3-GW-002 只在 wrapper 导入项目包前把 `Path(__file__).resolve().parents[1]` 加入 `sys.path`，并先用
`--help` 做无 run/GPU 的入口 smoke。GaussianWorld commit/weights、两个并行 scene workers、seed、2Hz frame schedule、
class mapping、UNKNOWN policy、28-case denominator、O_eval separation、thresholds、资源预算与 stop rule 全部不变。


### V61-F11：单卡 inference capability 通过不等于 predicted Occupancy 可以成为安全 authority

H-ME3-GW-002 canonical run `20260822T134559Z__predicted-occ-s1-r1` 完成两个 scene worker、24 次官方 streaming
inference、4 个 target occupancy、28 个 method decisions 和隐藏 O_eval 评分。预测臂与 oracle O2 得到相同的
`10/28 ACCEPT` 和 mask-area yield=`0.3983001361`，但这10例全部 false-safe；route-support 例的 hidden
observed-FREE conflict ratio=`0.766..0.958`，actor/disocclusion=`0.159..0.328`。run 正确以
`predicted_zero_false_safe=false` 拒绝。P6 的 weight/output/资源 capability 结论仍有效，但不能提升为安全性结论。

官方 GaussianWorld head、网格与类别源码，以及 DriveStudio nuScenes transform 源码和跨 metadata 数值对照都没有发现
x/y/z、class17 empty、camera order 或 lidar2img 错误。小幅前相机矩阵差异来自 nuScenes 异步相机 timestamp，后相机
在机器精度内一致。因此不得通过轴交换、投影修补或输入排列试错重开 GaussianWorld。

不得用 O_eval 选 confidence threshold、降低 UNKNOWN/verifier 门、做 grid/schedule/checkpoint sweep，或把 predicted
FREE 冒充 observed truth。已有 artifact 已证明把 observed O_method FREE 作为保守 veto 会让10个接受项全部 abstain；
无需创建零产出的重复 run。ReliOcc、α-OCC 与 OCCUQ 的可靠 uncertainty 需要训练/calibration，朴素 softmax/entropy
也不支持无校准安全声明。后续只允许先做一次 IR-WM truth-free current-state capability smoke；通过后才消耗唯一一次
ME-3 recovery，失败则停止 learned occupancy 并保留负结论。
### V61-F12：checkpoint 的零 missing gate 必须区分未使用的官方删除参数与有效 forward state

H-P7-IRWM-001 canonical run `20260822T143153Z__irwm-current-smoke-s1-r1` 已完成官方 IR-WM current-state GPU
forward，并写出 finite/nonempty occupancy。17项 gate 中15项通过；失败项只有环境版本字符串和模型零 missing。
Detectron2 使用官方 `0.6+cu111` wheel，而预注册只写 `0.6`；这不是不同 release。checkpoint 的唯一 missing keys
为 `pts_bbox_head.transformer.reference_points.weight/bias`。冻结官方 `WorldBEVFormerHead.init_weights()` 明确删除
整个 `transformer.reference_points`，其 detector decoder 在本次 `get_bev_features` current-state 路径不执行。

不得改写 H001 的 rejected terminal、直接手工把 gate 改成 PASS、给 missing 参数调值，或重复完整 GPU forward。
H-P7-IRWM-002 使用独立 P7R task，精确绑定 H001 gate/report/output/manifest/terminal 和官方删除源码，只允许
Detectron2 build suffix 与上述两项 source-proven unused missing keys；其余 H001 capability、truth-free、resource
合同全部原样要求通过。任何额外 missing/unexpected key 或 artifact 漂移都停止 learned occupancy。
### V61-F13：不同 predicted Occupancy capability 不能替代独立 observed-FREE safety authority

H-ME3-IRWM-001 canonical run `20260822T145543Z__irwm-predicted-occ-s1-r1` 完成两个并行 scene workers、
4 个 target occupancy、28 个 method decisions 和隐藏 O_eval 评分。IR-WM primary 与 oracle O2 都得到相同的
`10/28 ACCEPT`、accepted mask yield=`0.3983001361`，但全部10例 false-safe。route-support 的 hidden
FREE conflict=`0.344..0.571`，actor/disocclusion=`0.106..0.173`，均超过固定0.05；因此唯一顶层失败 gate 是
`predicted_zero_false_safe`。正式 run 无训练、calibration、threshold selection、confirmation read 或 truth 泄漏，
资源也在预算内，故这是科学机制 rejection，不是工程 blocked。

GaussianWorld 与 IR-WM 使用不同官方时序机制、类别合同和网格，却都复现 oracle 的10例接受集合且得到10/10 false-safe。
本证据拒绝在当前 development 协议中把 learned argmax occupancy 直接作为安全 authority；不否定两模型的 perception
capability，也不产生现实安全声明。不得再换 backend、选 confidence threshold、改 checkpoint/grid/history window、
放宽 verifier、用 O_eval 选阈值，或执行确定性零 yield 的 observed-FREE veto 冒充恢复。唯一 ME-3 recovery 已消费，
ME-4 不授权；V6.1 minimum experiment 以负结论收口。
