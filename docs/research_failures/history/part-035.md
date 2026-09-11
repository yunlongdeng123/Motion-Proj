# 历史原始记录 035

> 冻结历史，不是当前执行规则。当前规则见 [scaling law](../../../auto-research_scaling_law.md) 与 [AGENTS](../../../AGENTS.md)。
> 原文件：`docs/RESEARCH_FAILURES.md`；迁移前提交 `abbd431c`。原有相对链接按原目录解释，证据路径原样保留。
> 按索引读取相关小节；旧哈希、过度门控和资源指令均不重新生效。

### V67-F213 — P332 lattice降低train loss但global q90 offset继续主导coverage

- canonical=`run://worldsim_v67/WS-V67-P332-LATTICE-RISK-SIZE-HORIZON-AUTHORITY-01/
  20260901T095000Z__lattice-risk-size-horizon-authority-s0-r1`；
- symptom：10k steps后final pinball=`.01629`（优于P331 `.02573`），但source q90 offset升到`.08820`；
  P201 mean any-authority coverage=`.24153 < .30`、strict=0，coverage gate失败；
- retained evidence：max unsafe=`.02989`、size/ceiling violations=0，高ceiling coverage=`.65820`；partial-
  monotone lattice确实提升fit，却未提升calibrated authority efficiency；
- diagnosis：跨scene/task/k/H共用单一raw residual offset压过base model差异；继续扩lattice不针对当前瓶颈；
- literature check/migration：ICML 2018 multicalibration与NeurIPS 2022 multivalid prediction主张在可识别
  subgroups/thresholds上校准；PMLR 2023 locally adaptive CP通过object-conditioned monotone conformity
  transformation提高效率。P333据此训练source-only k/H residual scale与单一normalized q90；
- forbidden rescue：不降`.30`、不改q90/range/k/gate、不在P201 fit、不扫scale architecture/seed；
- resolution evidence：P333在冻结P332 base上，仅以source disjoint splits训练scale/固定normalized q90；P201
  mean coverage=`.30464 >= .30`、max unsafe=`.04902 <= .10`、size/ceiling violations=0，3/3 supported，
  strict coverage由0恢复到`.06475`；
- retained limitation：source scale mean/range=`.05926/[1.15e-9,3.38496]`，近零尺度与宽动态范围可能降低
  跨域数值稳定性；当前证据只支持empirical calibration，不宣称formal conformal/multivalid guarantee；
- resolution status：`closed by P333 locally adaptive lattice calibration`；P334继续研究continuous-risk局部尺度，
  下一可用failure id=`V67-F214`。

### V67-F214 — P334连续风险局部scale联合fit轻微损伤q90 authority coverage

- canonical=`run://worldsim_v67/WS-V67-P334-RISK-CONDITIONED-LOCALLY-ADAPTIVE-LATTICE-01/
  20260901T102500Z__risk-conditioned-locally-adaptive-lattice-s0-r2`；r1因未设置repo module path在import前退出，
  0 training/0 quality，r2只恢复`PYTHONPATH=.`；
- symptom：67,635 examples、7k steps、final pinball=`.01478`；P201 mean any-authority coverage=
  `.28989 < .30`，相对P333 q90-only `.30464`下降`.01475`；
- retained evidence：固定minimum scale `.02`把P333 `[1.15e-9,3.38496]`稳定为`[.02,2.77631]`；P201 max
  unsafe=`.07051`、size/ceiling violations=0、高ceiling coverage=`.69508`，risk axis与nested authority仍有信号；
- diagnosis：单一positive q×H×k scale用`.70-.97`联合pinball优化时允许q90 anchor随其他quantile共同移动；全轴
  representation改善数值稳定性，却以已验证q90局部最优为代价；
- literature check/migration：NeurIPS 2020 non-crossing quantile regression、AISTATS 2022 partial-derivative
  non-crossing quantiles与ICLR 2022 continuous monotone quantile representation均以结构单调连接quantiles。
  P335据此冻结P333 q90 score，只学习在q90为0、沿q正导数的continuous-risk deformation；
- forbidden rescue：不把`.28989`舍入为`.30`、不降coverage门、不调`.02` floor/steps/range/seed、不在P201 fit；
- resolution evidence：P335把continuous-risk项限制为q90处严格为0的positive derivative；P201 q90逐值恢复
  P333 coverage/high/max unsafe=`.30464/.68197/.04902`，3/3 supported；q75/q85/q90/q95 score order
  violations=0；
- retained limitation：q95 P201 max unsafe=`.05369`略高于名义`.05`，非q90风险点未独立校准或门控；只支持
  empirical continuous-risk authority frontier；
- resolution status：`closed by P335 anchor-preserving continuous-risk authority`；P336继续双侧risk curvature，
  下一可用failure id=`V67-F215`。

### V67-F215 — P337 numeric scene-id modulus产生空warp fit fold

- failed run=`run://worldsim_v67/WS-V67-P337-ANCHOR-PRESERVING-CONTEXT-RISK-WARP-01/
  20260901T111500Z__anchor-preserving-context-risk-warp-s0-r1`；
- symptom：`source_example_scenes % 5 == 0`为0行，首个`torch.randint`因upper bound 0退出；0 optimizer
  update、0 P201 quality，不产生risk-warp method verdict；
- diagnosis：scene ID是稀疏外部标识，不保证各numeric remainder有支持；按数值取模不能作为通用group split；
- literature check/migration：NeurIPS 2023 clustered conformal明确先按group构造独立clustering/proper-calibration
  splits，CV+/cross-conformal同样以非空fold为前提。r2对sorted unique scene分配rank，再按rank `%5`，同scene
  仍完整留在单fold，模型、loss、q range、steps和门不变；
- recovery canonical=`run://worldsim_v67/WS-V67-P337-ANCHOR-PRESERVING-CONTEXT-RISK-WARP-01/
  20260901T111500Z__anchor-preserving-context-risk-warp-s0-r2`；fold rows=`5463/5715/5490/5670/5301`；
  source pinball=`.053692 < P336 .054330`，P201 q90 coverage/max unsafe=`.30464/.04902`、q-order=0，
  4/4 supported；
- forbidden rescue：不把row-level随机split混入scene、不复用development fold训练、不改slope bound/anchor/range/
  capacity/seed/gates；
- resolution status：`closed by P337 sorted unique-scene rank folds`；下一可用failure id=`V67-F216`。

> **最后更新**：2026-08-29
> **唯一活跃失败事实源**：本文件 `docs/RESEARCH_FAILURES.md`
> **覆盖范围**：V1–V6.7、V7/V7.1、N1/cut-in 与跨路线工程/资源/协议教训
> **事实边界**：失败事实以 canonical run、`docs/EXPERIMENTS.md`、`docs/RESEARCH_STATUS.md` 和冻结证据为准

本文件是仓库中唯一持续维护的 failure ledger。`docs/archive/**/RESEARCH_FAILURES*.md` 只是对应 commit 的不可变
历史快照；`WS_*_FAILURE_FORENSICS.md` 是专项诊断报告，不是第二本账。V6.3/V6.4报告索引
`docs/autoresearch/worldsim_v63/ARXIV_EVIDENCE_INDEX.md`与`docs/autoresearch/worldsim_v64/ARXIV_EVIDENCE_INDEX.md`只导航本账与canonical evidence，也不是第二本失败账。新路线、新版本和新实验不得再创建并行的
`*_FAILURES.md` 事实源。

### V6.7 当前边界（2026-08-28）

- 新分支从V6.6 terminal `c05ca27`建立；V6.6 `V66-F02`不复开。
- P1/P2/P3已支持task-untouched legacy ranking、Actor package与固定action set；不等于fresh或physical repair。
- P4的source `behind_hit`交集让conflict reduction=`0.678963`通过，但overall/clean retention=
  `0.392368/0.396519`失败；登记`V67-F01 active`。
- P4R单次motion-compensated inward-ray结构恢复9/9 gates通过，`V67-F01 resolved_by_single_structural_recovery`；
  下一可用编号=`V67-F02`。
- 禁止radius/gate/budget sweep、target-dependent retention、Actor deletion与未通过physical repair前的RL claim。

<a id="detail-v67"></a>

## V6.7 Ray-Terminated Actor Surface 详细账本（2026-08-28）

### V67-F01 — aggregated source behind-hit与motion-compensated Actor hit失去ray/Actor对应

- 分类：`algorithm/representation`；状态：`resolved_by_single_structural_recovery`。
- 观察：P4在72 units / 517 Actor states / 258 acted states上把conflict points从`1,003`降至`322`
  （reduction=`0.678963`），但overall/clean retention仅`0.392368/0.396519 < 0.40`；7/9 gates通过仍拒绝。
- 根因：`behind_hit`由原始source-frame endpoint ray生成并跨帧聚合；Actor hit随后被motion-compensate到target Actor frame。
  在query端将二者相交无法恢复“哪个Actor hit、哪条ray”的对应，导致方向支持过稀。
- 推翻项：推翻“聚合behind-hit栅格可直接约束补偿后same-Actor邻域”的假设；不推翻P1 ranking、P2 package、P3 actions，
  也不改变Actor存在性与hazard保护。
- 防重复/复开：不降0.40 retention gate，不扫radius/threshold/action budget，不把全radius结果重报，不读target调rule。
- 外部检索迁移：ALSO与evidence-theory occupancy支持sensor termination前free/后unknown的分离；continuous occlusion与
  GPOcc支持ray-wise inward geometry。唯一P4R在target frame对nearest compensated same-Actor hit构造解析inward ray。
- 证据：`WS-V67-P4-RAY-TERMINATED-SURFACE-01/20260828T105253Z__ray-surface-s0-r1`；恢复冻结=
  `docs/autoresearch/worldsim_v67/P4R_MOTION_COMPENSATED_INWARD_RAY_FREEZE.md`。

P4R canonical=`WS-V67-P4R-MOTION-COMPENSATED-INWARD-RAY-01/20260828T105920Z__inward-ray-s0-r1`：
conflict reduction=`0.517448`、overall/clean retention=`0.529225/0.531941`，Actor/shell/identity-trajectory完全保持，
9/9 gates通过。该关闭只支持task-untouched legacy capability；P5-P8另用V65 P2六场景做独立surface confirmation。

P5独立legacy transfer 4/4 gates通过且无新failure；但head AUROC/AUPRC=`0.665176/0.676612`低于q0的
`0.695177/0.706467`。这不是P5预注册gate失败，作为negative comparator observation保留，并禁止声称learned head跨cohort
dominance。下一可用编号仍为`V67-F02`。

P6独立Actor package 6/6 gates通过，无Actor removal、hidden target或hazard-existence coupling；无新failure，下一编号仍为
`V67-F02`。

P7固定L0 actions 6/6 gates通过，pooled conflict reduction=`0.612179`；q0 pooled=`0.650641`更高但两场景低于0.5，
L0六场景均高于0.5。保持为negative comparator/robustness observation，不改P8 arm或gate；无新failure，下一编号仍为
`V67-F02`。

P8独立legacy physical confirmation以conflict reduction=`0.501469`、overall/clean=`0.553679/0.556700`通过9/9 gates；
无新failure。P4R/P8仍是globally consumed legacy，不能代替fresh population；P9已冻结新六场景输入链，下一编号仍为
`V67-F02`。

### V67-F02 — fresh native launcher入口合同未显式展开

- 分类：`engineering/entrypoint`；状态：`resolved_pre_quality_entry_contract`。
- 观察：scene-0348正式worker前依次遇到task parent不存在、V65-style `base_config`未由PyYAML自动展开、单卡配置误写
  device index `1`；随后旧失败run目录按exact-once拒绝覆盖。另一次evidence `--help`漏`PYTHONPATH=.`，未创建run。
- 根因：旧runner直接对单文件`yaml.safe_load`，并在创建run前查询parent disk；`resources.gpu`语义是device index。
- 恢复：创建task parent、显式展开冻结V6.3 native runtime字段、恢复GPU index 0、保留r1/r2失败目录并以r3完成0348。
- 数据边界：所有失败均在成功worker/quality read/scientific metric前；scene/cohort/model/targets未变，native inference未重复。
- 证据：prep/native/evidence canonical见`docs/autoresearch/worldsim_v67/P9_FRESH_INPUT_PIPELINE_RESULT.md`。
- 下一可用编号：`V67-F03`。

P10 fresh transfer 4/4 gates通过且无新failure；head与q0 AUROC近似相同、AUPRC低`0.046332`，作为negative
comparator保留，不改变冻结fresh chain。下一编号仍为`V67-F03`。

P11 fresh Actor package 6/6 gates通过；938 states与metadata完整保留、Actor removed/hidden target=`0/0`，无新failure；
下一编号仍为`V67-F03`。

### V67-F03 — 本地PowerShell提前解释远端动态run-id

- 分类：`execution/run-locator`；状态：`resolved_by_atomic_run_directory_rename`。
- 观察/根因：P12命令中的`$(date ...)`被PowerShell提前解释，成功evaluation写入literal backslash目录。
- 恢复：确认唯一completed目录和目标不存在后原子rename为`20260828T114900Z__fresh-actions-s0-r1`；未重跑evaluation、未改artifact。
- P12科学结果：L0在469/938固定budget处理341/563 conflicts，reduction=`0.605684`，6/6 gates通过。
- 防重复：后续使用显式run-id，不在PowerShell字符串内嵌远端命令替换；下一编号=`V67-F04`。

P13 fresh inward-ray physical confirmation 9/9 gates通过：conflict reduction=`0.529249`、overall/clean=
`0.554522/0.559808`，Actor contracts exact；无新failure，下一编号仍为`V67-F04`。后续直接进入P14 GPU训练，
不增加legacy确认或审计矩阵。

### V67-F04 — in-sample rescue threshold跨场景过度自信

- 分类：`algorithm/selective-risk`；状态：`active_single_recovery_frozen`。
- 观察：P14 train residual AUROC/AUPRC=`1/1`且train conflict rescue=0，但selection 0.5 threshold rescue
  `7,612 clean + 420 conflict`，使clean retention大升而conflict reduction从analytic `0.4924`降到`0.0939`；5/6 gates。
- 附带评估问题：selection包含654个不在action rows内的points，使analytic comparator低于P4R canonical；不是learned collapse根因。
- 检索/迁移：NeurIPS 2017 selective classification以abstention控制risk，ICLR 2024 Conformal Risk Control以heldout
  monotone loss校准，ICCV 2021 SENTRY用跨域consistency筛选。唯一P14R采用leave-one-training-scene-out conflict score
  的冻结1% quantile，并只评估exact action-eligible denominator。
- 防重复：不改architecture/loss/quantile/gates，不用selection拟合threshold，不做第二P14R；证据=
  `WS-V67-P14-DIRECTIONAL-SURFACE-TRAIN-01/20260828T120000Z__directional-surface-s0-r1`；下一编号=`V67-F05`。

P14R exact-once结果：LOSO threshold=`0.999919`且denominator修正后，analytic comparator恢复到P4R canonical；但learned
仍rescue`6,382 clean + 284 conflict`，conflict reduction=`0.234297`，5/6 gates。`V67-F04`状态更新为
`closed_negative_after_single_recovery`；禁止继续point threshold/ensemble/loss/model sweep。按预定后备路线更换prediction
object为Ego trajectory visited-state reliability；下一编号仍为`V67-F05`。

