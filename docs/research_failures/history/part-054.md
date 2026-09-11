# 历史原始记录 054

> 冻结历史，不是当前执行规则。当前规则见 [scaling law](../../../auto-research_scaling_law.md) 与 [AGENTS](../../../AGENTS.md)。
> 原文件：`docs/RESEARCH_FAILURES.md`；迁移前提交 `abbd431c`。原有相对链接按原目录解释，证据路径原样保留。
> 按索引读取相关小节；旧哈希、过度门控和资源指令均不重新生效。

### V66-F02 — fixed-budget conflict triage不等于natural physical repair

- 分类：`algorithm/evaluation`；状态：`closed_negative_after_single_recovery`。
- 观察：P7 L0在固定290 action budget下处理210/307 conflict states，exposure reduction=`68.40%`且Actor/hazard proxy
  保留；但`physical_geometry_mutated=false`，所以handled只是候选，不是实际修复。Q0还出现scene yield=`5/6`，说明简单
  actor-blind ranking可能把一个scene local geometry全部送入action并造成easier-world风险。
- 推翻项：推翻“ranking/triage通过即可称P7 physical distribution成功”；不推翻P3L/P3C ranking或P6 package capability。
- 防重复/复开：禁止把action count写成repaired artifact count，禁止调50% budget/score threshold，禁止删Actor或用target
  label决定primitive retention。
- 外部检索迁移：NeuRAD sensor-aware dynamic Actor、Neural Scene Graphs track-based static/dynamic decomposition、
  Cam4DOcc instance 4D occupancy，迁移为“canonical Actor collision shell保持 + motion-compensated sensor-supported local
  surface repair”。详见`P7R_SENSOR_SUPPORTED_REPAIR_MIGRATION_FREEZE.md`。
- 合法恢复：沿用P7 L0 action set；acted boundary只保留same-Actor motion-compensated hit，target仅评估；同时过conflict
  reduction、clean/overall geometry yield、Actor/shell/track/hazard preservation gates。
- 证据：`WS-V66-P7-HAZARD-PRESERVING-DISTRIBUTION-01` /
  `20260828T092919Z__fixed-budget-distribution-s0-r1`；首次physical recovery reject=
  `WS-V66-P7R-SENSOR-SUPPORTED-ACTOR-REPAIR-01/20260828T093710Z__sensor-surface-repair-s0-r1`。
- recovery：exact hit保留过稀；P7R2依据PoinTr/SnowflakeNet/RFNet冻结Actor-local 0.512m support expansion，
  详见`P7R2_RADIUS_SUPPORT_RECOVERY_FREEZE.md`。唯一恢复仍未同时满足冲突下降与干净几何保留，family终止。

P7R point-level repair当时按冻结规则实现；target只参与post-repair metric，未进入L0 action或same-Actor hit retention
rule。该阶段`V66-F02`进入唯一恢复，最终终态见下。

P7R formal被拒绝：conflict reduction=`0.847660`，但overall/clean boundary retention=`0.383588/0.395715 < 0.40`；
7/9 gates通过不改变verdict。根因是0.2m exact evidence-voxel hit过稀；随后只执行预先冻结的P7R2 fixed
one-native-voxel `0.512m` same-Actor support neighborhood，原九门不变。

P7R2 formal同样被拒绝：overall/clean boundary retention=`0.617684/0.619549`通过，但conflict reduction=
`0.417872 < 0.50`；8/9 gates通过不改变verdict。固定支持半径增加后同时保留了过多conflict points，暴露sensor proximity
无法区分合法表面支持与Actor-local free-space conflict。按预冻结规则不试中间radius、不降gate、不改budget、不训练
completion model；`V66-F02`以`closed_negative_after_single_recovery`终止。P7 triage正结果保留，但physical repair、
RL-ready distribution与P9继续锁定。完整证据：`P7R2_RADIUS_SUPPORT_RESULT.md`。

### V66-F03 — P8低速stop-state重复应用jerk update

- Pre-run operational note：P8首次shell invocation因未设置repo `PYTHONPATH`在module import前退出；没有进入runner、
  创建run directory或读取scientific metric。只修正launcher环境后执行同一代码/config，因此不分配failure ID，也不构成
  第二次scientific read。
- 分类：`implementation/numerical`；状态：`resolved_by_single_implementation_recovery`。
- 观察：P8六场景X1 collision steps均为0，但scene-0001/0219 command jerk分别为`9.637574/7.400627m/s^3`，
  超过固定`6m/s^3`；仅4/6 scenes全门通过，formal verdict拒绝。
- 根因：零速边界先通过正常rate limiter更新acceleration，随后stop分支再次增加一个jerk step；同一离散步对command
  应用了两次rate update。不是IDM参数或Actor selection证据。
- 防重复：禁止换Actor/scene、调headway/IDM/AV/horizon、放宽jerk gate或只删低速场景。
- 外部检索迁移：Autoware longitudinal controller的DRIVE/STOPPING/STOPPED state与vehicle command longitudinal
  jerk limiter，迁移为明确stopped desired acceleration=0且每步只过一次原rate limiter。
- 唯一恢复：P8R只移除第二次increment，其余输入、参数、轨迹、指标与gates exact不变；失败关闭P8 family。
- 证据：`WS-V66-P8-REACTIVE-ACTOR-01/20260828T095440Z__reactive-actor-s0-r1`；冻结：
  `docs/autoresearch/worldsim_v66/P8R_STOP_STATE_JERK_RECOVERY_FREEZE.md`。

P8R保持全部实验参数与gates exact，只把stopped desired acceleration置0并让command每步通过原rate limiter一次。
六场景全部支持，pooled X0/X1 collision steps=`306/0`，minimum X1 gap=`1.948192m`，maximum command jerk=
`6.000000m/s^3`；`V66-F03`恢复关闭。该恢复只修复numerical update，不把synthetic response扩展到natural interaction，
也不改变P7 terminal negative或解锁P9/RL。证据：`WS-V66-P8R-STOP-STATE-JERK-RECOVERY-01/
20260828T095839Z__stop-state-jerk-recovery-s0-r1`。

下一可用编号：`V66-F04`。

### V6.5 终态边界（2026-08-28）

- `V65-F19` 为 terminal algorithm negative：P10X 5/6 gates通过，但 direct selected-action cost reduction
  `16.38% < 25%`。支持给定 `tau` 的 visited-state reliability ranking/calibration，不支持 direct action
  authority、planner、policy、closed-loop、RL或safety。
- V6.6 只可把 q0/visited-state score当 diagnostic；不得以新阈值、第二 confirmation cohort或新 critic
  复开 V6.5 family。

### V6.4 当前边界（2026-08-27）

- V6.4已终态`v64_research_complete_report_ready`。正证据边界为P6R/P4C独立选择性校准和P10R4 untouched
  fixed-opportunity exact empirical route-local risk；不支持population、physical collision、planning、closed-loop、RL或
  safety claim。P11/P11R collision critic以`V64-F28 closed_negative_after_single_recovery`终止，P11D确认unsafe prior与
  ranking同时跨cohort漂移，不授权第二次threshold recovery、大型NWM/RL或复开V6.4。
- 版本收口与arXiv索引没有新增实验或failure ID。`V64-F25`只在P10R4独立固定机会分母层面解除；P10T/current-M0
  `V64-F21`负结论及P10R2 selected-denominator relative non-improvement保持。详细终态见
  `docs/autoresearch/worldsim_v64/V64_RESEARCH_FAMILY_CLOSEOUT.md`与`ARXIV_EVIDENCE_INDEX.md`。
- shutdown前最终报告审计只读确认7个关键canonical run的目录与`summary.json/status.json`可用，并按既有`V64-F03`
  合同在非登录shell激活`motionproj`后解析；没有重跑、重算、改结果或新增failure。arXiv写作交接见
  `docs/autoresearch/worldsim_v64/V64_ARXIV_REPORT_HANDOFF.md`，本次failure ledger delta=`none`。

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

- [V6.6 详细账本](#detail-v66)
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

