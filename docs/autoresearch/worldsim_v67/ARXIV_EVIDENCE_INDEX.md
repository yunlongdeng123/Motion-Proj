# WorldSim V6.7 ArXiv Evidence Index

- 工作标题：**From End-to-End Reliability Shortcuts to Actor-Uncertainty Boundary Queries**
- 分支：`research/worldsim-v6.7-anisotropic-surface`
- 状态：`P113 outcome pending final fill`
- 当前主结论：P108 scene-level independent factorization支持；P113独立uncertainty-over-clearance归因待回填

本文件是V6.7技术报告的证据导航。逐实验数值以`docs/EXPERIMENTS.md`为准，失败与防重复规则以
`docs/RESEARCH_FAILURES.md`为准，实时状态以`docs/RESEARCH_STATUS.md`为准，论文叙事见
`V67_ARXIV_TECHNICAL_REPORT.md`。

## 1. Canonical run index

| 阶段 | canonical run | 核心证据 |
|---|---|---|
| P81 | `run://worldsim_v67/WS-V67-P81-FRESH-TEST-ACTOR-EVENT-01/20260829T211500Z__fresh-test-actor-event-s0-r1` | all-row query/Actor/P73=`26/57/45`；10/10 scenes |
| P95 | `run://worldsim_v67/WS-V67-P95-TRAJECTORY-OCCUPANCY-FLIP-01/20260830T002500Z__trajectory-occupancy-flip-s0-r1` | consumed development `7/28/13` |
| P96 | `run://worldsim_v67/WS-V67-P96-OCCUPANCY-FLIP-CONFIRMATION-01/20260830T004000Z__occupancy-flip-confirmation-s0-r1` | independent `8/5/12`；relative reject |
| P102 | `run://worldsim_v67/WS-V67-P102-HIERARCHICAL-TEMPORAL-INTERACTION-01/20260830T022000Z__hierarchical-temporal-interaction-s0-r1` | development best `4/27/13` |
| P103 | `run://worldsim_v67/WS-V67-P103-HIERARCHICAL-CONFIRMATION-01/20260830T024000Z__hierarchical-confirmation-s0-r1` | same independent read `9/7/12`；relative reject |
| P107 | `run://worldsim_v67/WS-V67-P107-ACTOR-UNCERTAINTY-TUBE-01/20260830T061000Z__actor-uncertainty-tube-s0-r2` | P81/P96=`2/36/13`与`2/9/12` |
| P108 | `run://worldsim_v67/WS-V67-P108-UNCERTAINTY-TUBE-CONFIRMATION-01/20260830T063500Z__uncertainty-tube-confirmation-s0-r1` | independent P107/Actor/P75=`5/35/20` |
| P109 | `run://worldsim_v67/WS-V67-P109-DIRECTIONAL-ACTOR-UNCERTAINTY-01/20260830T062500Z__directional-actor-uncertainty-s0-r1` | directional development P81/P96均0 events |
| P110 | `run://worldsim_v67/WS-V67-P110-DIRECTIONAL-CONFIRMATION-01/20260830T064000Z__directional-confirmation-s0-r1` | same-read directional=`1`，AUROC=.96027 |
| P111 | `run://worldsim_v67/WS-V67-P111-CLEARANCE-CONFIRMATION-BASELINE-01/20260830T064500Z__clearance-confirmation-baseline-s0-r1` | same-read clearance=`1`，AUROC=.91644 |
| P112 | `run://worldsim_v67/WS-V67-P112-NONLINEAR-GAUSSIAN-CROSSING-01/20260830T065000Z__nonlinear-gaussian-crossing-s0-r1` | P96 nonlinear 3/.85852 vs linear 0/.90434；reject |
| P114 | `run://worldsim_v67/WS-V67-P114-MONOTONE-TAIL-RISK-01/20260830T071000Z__monotone-tail-risk-s0-r1` | P81/P96 AUROC gain=`-.01626/-.00137`；reject |
| P115 | `run://worldsim_v67/WS-V67-P115-SPECTRAL-ACTOR-UNCERTAINTY-01/20260830T071500Z__spectral-actor-uncertainty-s0-r1` | P81/P96 AUROC gain=`+.00945/-.05722`；reject |
| P116 | `run://worldsim_v67/WS-V67-P116-DIRECTIONAL-QUANTILE-FIELD-01/20260830T072000Z__directional-quantile-field-s0-r1` | P81/P96 AUROC gain=`-.00380/-.01503`；reject |
| P117 | `run://worldsim_v67/WS-V67-P117-FULL-COVARIANCE-ACTOR-UNCERTAINTY-01/20260830T072500Z__full-covariance-actor-uncertainty-s0-r1` | P81/P96 AUROC gain=`+.00490/+.00932`；development support |
| P118 | `run://worldsim_v67/WS-V67-P118-CORRELATION-ABLATION-01/20260830T073000Z__correlation-ablation-s0-r1` | conditional-vs-zero rho gain=`+.00030/-.00012`；reject mechanism |
| P113 | `run://worldsim_v67/WS-V67-P113-DIRECTIONAL-VS-CLEARANCE-CONFIRMATION-01/20260830T070500Z__directional-vs-clearance-s0-r1` | `PENDING_FINAL_FILL` |

上述locator已按run tree精确对齐；任何metric disagreement仍回到对应canonical summary，不重算quality。

## 2. 论文主表证据

| 问题 | 数据角色 | 结果 | 可写结论 |
|---|---|---|---|
| 全row unreliable-event排序 | P81 independent | query/Actor/P73=`26/57/45` | 窄triage signal |
| visited Actor max-error是否task-conditioned | P84--P94 same-read | 全family未稳定超过Actor-only | negative object result |
| end-to-end occupancy flip是否迁移 | P95 dev → P96 independent | `7/28/13` → `8/5/12` | development shortcut/relative reversal |
| hierarchical query是否恢复 | P102 dev → P103 secondary | `4/27/13` → `9/7/12` | relative reversal retained |
| scalar Actor uncertainty factorization | P107 consumed ×2 → P108 independent | P108=`5/35/20` | scene-level independent support |
| directional uncertainty | P109 consumed ×2 → P110 secondary | P110 directional=`1`, AUROC=.96027 | strong secondary ranking |
| geometry-only解释 | P111 same read | clearance=`1`, AUROC=.91644 | event tie, ranking gap |
| nonlinear sampled crossing | P112 consumed ×2 | cross-cohort worse than linear | reject Monte Carlo recovery |
| downstream monotone tail pool | P114 consumed ×2 | P81/P96都低于P109 max | reject aggregation recovery |
| joint low-frequency Actor sequence | P115 consumed ×2 | P81小增益、P96强退化 | reject spectral recovery |
| directional distribution-free q90 | P116 consumed ×2 | P81/P96都低于P109 Gaussian | reject quantile recovery |
| full bivariate residual covariance | P117 consumed ×2 | P81/P96均0 events，mean AUROC gain=`+.00711` | development mechanism support |
| conditional rho direct contribution | P118 same-checkpoint ablation | mean gain=`+.000094`且P96反向 | reject direct-rho attribution |
| learned uncertainty超过clearance | P113 independent | `PENDING_FINAL_FILL` | `PENDING_FINAL_FILL` |

## 3. Failure map

| ID | 状态 | 论文处理 |
|---|---|---|
| `V67-F67` | closed negative | visited max-error对象不支持task-conditioned claim |
| `V67-F68--F73` | mixed scientific/engineering chain | subtype、auxiliary与data scaling不恢复独立迁移 |
| `V67-F74` | closed negative | P95 independent relative reversal |
| `V67-F75` | closed negative | P102/P103 hierarchical relative reversal |
| `V67-F76` | resolved pre-run | launcher工作目录丢失；0 scientific exposure |
| `V67-F77` | resolved pre-optimizer | NPZ原子交付race；0 optimizer step |
| `V67-F78` | closed negative | nonlinear Gaussian sampling跨cohort退化 |
| `V67-F79` | closed negative | P114 top-k/union tail pool稀释P109 max |
| `V67-F80` | closed negative | P115 low-frequency Actor sequence在P96过度平滑 |
| `V67-F81` | closed negative | P116 directional q90低于P109 standardized margin |
| `V67-F82` | active pre-target recovery | P113 scene-0003 exact archive locator incomplete |
| `V67-F83` | closed negative | P118 conditional-rho direct mechanism not supported |
| `V67-F84` | reserved for P113 | only if frozen one-shot scientific decision fails |

## 4. Artifact inventory

| artifact | location |
|---|---|
| canonical run summaries/configs/status | `/root/autodl-tmp/runs/worldsim_v67/<task>/<run>/` |
| P107 scalar checkpoint | P107 canonical run |
| P109 directional checkpoint | P109 canonical run |
| P117 full-covariance checkpoint | P117 canonical run |
| P118 same-checkpoint rho ablation | P118 canonical run |
| P108 independent rows/summary | P108 prep与primary canonical runs |
| P111 clearance comparator | P111 canonical run |
| P113 independent rows/summary | P113 prep与primary canonical runs |
| chronological ledger | `docs/EXPERIMENTS.md` |
| failure ledger | `docs/RESEARCH_FAILURES.md` |
| live status | `docs/RESEARCH_STATUS.md` |
| technical narrative | `docs/autoresearch/worldsim_v67/V67_ARXIV_TECHNICAL_REPORT.md` |

## 5. Claim boundary

- independent表示scene-level；不是session-level，因为历史cohort可能含相邻session scene。
- occupancy-decision flip不是collision label；risk score不是calibrated probability。
- fixed-coverage排序不授权planner、policy、closed-loop、control或deployment。
- P110/P111共享P108 target read，只是prospective secondary/comparator，不是第二个独立cohort。
- P113只裁决learned directional uncertainty相对clearance-only的ranking/event增量，不重裁P108相对Actor-only/P75的主结论。
- P117在P113冻结后完成且只读consumed cohorts；它不能替换P113的P109 checkpoint或充当独立确认。
- 未新增hash/checksum/fingerprint；没有smoke/regression matrix或参数扫描。

## 6. Source-of-truth order

1. canonical `summary.json` / `status.json` / retained row artifacts；
2. `docs/EXPERIMENTS.md`；
3. `docs/RESEARCH_FAILURES.md`；
4. `docs/RESEARCH_STATUS.md`与`AUTORESEARCH_STATE.current.json`；
5. 本index与`V67_ARXIV_TECHNICAL_REPORT.md`。
