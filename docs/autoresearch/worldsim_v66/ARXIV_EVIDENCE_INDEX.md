# WorldSim V6.6 ArXiv Evidence Index

- 工作标题：**HARP-Compiler：保留危险Actor的幻觉感知世界编译**
- 文档冻结：`2026-08-28`
- 分支：`research/worldsim-v6.6-harp-compiler`
- 终态：`v66_research_complete_arxiv_report_ready`
- 科学结论：two-level ranking/package与synthetic reactive capability支持；natural physical surface repair终局拒绝；RL未执行
- 本次report handoff新增scientific run：none

本文件是V6.6论文写作的证据导航。逐实验数值以`docs/EXPERIMENTS.md`为准，失败与防重复规则以
`docs/RESEARCH_FAILURES.md`为准，技术叙事见`V66_ARXIV_TECHNICAL_REPORT.md`。

## 1. Canonical run index

| 阶段 | canonical run | verdict / 核心数值 |
|---|---|---|
| P1-D | `run://worldsim_v66/WS-V66-P1-VALIDITY-HAZARD-SEPARATION-ATLAS-DEV-01/20260828T084915Z__factorial-atlas-dev-s0-r1` | 409 base Actor-units；4/4 development gates |
| P2-D | `run://worldsim_v66/WS-V66-P2-FACTOR-CERTIFICATE-DEV-01/20260828T085346Z__factor-certificate-dev-s0-r1` | 8/8 deterministic gates |
| P4-D | `run://worldsim_v66/WS-V66-P4-ARTIFACT-REPAIR-DEV-01/20260828T085755Z__repair-first-dev-s0-r1` | paired R2 6/6 development gates |
| P2N | `run://worldsim_v66/WS-V66-P2N-NATURAL-ACTOR-CONFLICT-DIAGNOSTIC-01/20260828T090228Z__natural-actor-conflict-s0-r1` | 891 rows；deterministic recall=0 |
| P3L | `run://worldsim_v66/WS-V66-P3L-ACTOR-LOCAL-GEOMETRY-HEAD-01/20260828T091036Z__local-geometry-head-s0-r1` | AUROC/AUPRC=`0.652365/0.692384` |
| P3C | `run://worldsim_v66/WS-V66-P3C-INDEPENDENT-LOCAL-GEOMETRY-CONFIRM-01/20260828T091611Z__independent-local-geometry-confirm-s0-r1` | AUROC/AUPRC=`0.761644/0.767165`；6/6 scenes |
| P6 | `run://worldsim_v66/WS-V66-P6-HARP-BAKE-01/20260828T092421Z__harp-bake-s0-r1` | 127 Actors / 581 states / 1,623,503 primitives |
| P7 | `run://worldsim_v66/WS-V66-P7-HAZARD-PRESERVING-DISTRIBUTION-01/20260828T092919Z__fixed-budget-distribution-s0-r1` | L0 exposure reduction=`0.684039`；triage only |
| P7R | `run://worldsim_v66/WS-V66-P7R-SENSOR-SUPPORTED-ACTOR-REPAIR-01/20260828T093710Z__sensor-surface-repair-s0-r1` | conflict reduction=`0.847660`；clean retention FAIL |
| P7R2 | `run://worldsim_v66/WS-V66-P7R2-RADIUS-SUPPORTED-ACTOR-REPAIR-01/20260828T094232Z__radius-surface-repair-s0-r1` | clean retention=`0.619549`；conflict reduction=`0.417872` FAIL |
| P8 | `run://worldsim_v66/WS-V66-P8-REACTIVE-ACTOR-01/20260828T095440Z__reactive-actor-s0-r1` | 4/6；stop-state jerk implementation reject |
| P8R | `run://worldsim_v66/WS-V66-P8R-STOP-STATE-JERK-RECOVERY-01/20260828T095839Z__stop-state-jerk-recovery-s0-r1` | 6/6；collision steps `306→0`；jerk=`6` |

## 2. 结果表

| 研究问题 | 分母/设置 | 结果 | 终态解释 |
|---|---:|---:|---|
| injected factor是否可分 | 409 base Actor-units / 8,180 rows | deterministic certificate 1.0/1.0 | development mechanism only |
| natural local conflict是否被coarse certificate覆盖 | 891 Actor-units，498 conflict | recall=0，AUROC=0.5 | `V66-F01`触发two-level设计 |
| learned local ranking是否迁移 | 581 independent Actor-units | AUROC/AUPRC=`0.761644/0.767165` | local REPAIR/ABSTAIN ranking only |
| package是否保留Actor | 127 unique Actors | states/metadata retention=`1/1`，removed=0 | runtime package capability |
| fixed budget是否降低暴露 | 307 conflicts / 290 actions | L0 handled 210，reduction=`0.684039` | candidate triage, not repair |
| exact sensor repair | 23,580 boundary points | conflict reduction `0.847660`; clean retention `0.395715` | reject |
| one-voxel repair | same denominator | conflict reduction `0.417872`; clean retention `0.619549` | terminal reject |
| fixed reactive response | 6 synthetic lead-brake scenes | X0/X1 collisions `306/0`; min gap `1.948192m` | narrow capability support |

## 3. Failure map

| ID | 状态 | 论文处理 |
|---|---|---|
| `V66-F01` | `resolved_by_two_level_certificate` | 证明existence与local geometry validity必须分层 |
| `V66-F02` | `closed_negative_after_single_recovery` | 主要scientific negative；P7/P9边界 |
| `V66-F03` | `resolved_by_single_implementation_recovery` | numerical appendix；不改变P8 parameters/gates |

下一可用failure ID为`V66-F04`。

## 4. Artifact inventory

| artifact | location |
|---|---|
| canonical summaries/status/configs | `/root/autodl-tmp/runs/worldsim_v66/<task>/<run>/` |
| HARP eight-file package | P6 canonical run `package/`，16,321,358 bytes |
| P7 repaired boundary artifacts | P7R/P7R2 canonical run directories |
| P8/P8R scene metrics and trajectories | corresponding canonical runs；P8R trajectory JSONL 5,202,077 bytes |
| chronological ledger | `docs/EXPERIMENTS.md` |
| failure ledger | `docs/RESEARCH_FAILURES.md` |
| terminal state | `docs/autoresearch/worldsim_v66/AUTORESEARCH_STATE.current.json` |
| per-stage result/freeze documents | `docs/autoresearch/worldsim_v66/` |

## 5. Resource and protocol boundary

- GPU：单张RTX 3090 24GB；multi-GPU未需要。
- GPU stages的observed peak allocation约`0.02359GiB`；主要工作是短GPU forward与CPU/I/O编译。
- `/root/autodl-tmp`在closeout约剩`95GB`。
- 未新增hash/checksum/fingerprint；未执行smoke/regression matrix；closeout不重算scientific metric。
- 最小handoff audit解析V6.6 run tree内24个`summary.json/status.json`，全部JSON-readable。
- P8有一次pre-run launcher import failure：未创建run或读取metric；canonical P8仍是唯一scientific read。
- P9/P10/P11未执行，因为P7 physical repair未通过。

## 6. Source-of-truth order

1. canonical run artifacts；
2. `docs/EXPERIMENTS.md`；
3. `docs/RESEARCH_FAILURES.md`；
4. `docs/RESEARCH_STATUS.md`与terminal state；
5. 本index、`V66_RESEARCH_CLOSEOUT.md`与`V66_ARXIV_TECHNICAL_REPORT.md`。

任何rounded disagreement必须回到canonical summary，不得重跑、调gate或生成第二个repair recovery。
