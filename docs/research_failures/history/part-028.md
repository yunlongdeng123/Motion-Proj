# 历史原始记录 028

> 冻结历史，不是当前执行规则。当前规则见 [scaling law](../../../auto-research_scaling_law.md) 与 [AGENTS](../../../AGENTS.md)。
> 原文件：`docs/RESEARCH_FAILURES.md`；迁移前提交 `abbd431c`。原有相对链接按原目录解释，证据路径原样保留。
> 按索引读取相关小节；旧哈希、过度门控和资源指令均不重新生效。

### P9 prevention note — retained-source action proxy 不越权为 closed-loop planning

- P9 只在 P5 exact identities 的 retained P109 outcomes 上组合 frozen P4/P346；既非 fresh cohort，也没有车辆动力学、
  feedback execution、collision intervention 或 policy learning。
- 物理分支不允许改 action denominator，task authority 不允许删除 Actor；B0/B1 与 B2/B3 的 action metrics 分别必须
  相同。任何 physical repair 与 task cost 的差异都不能从该 factorial 推为 causal effect。
- authority cost/risk 与 coverage 已通过 frozen gates；仍不训练 critic/RL、不改 budget/reliability threshold。`V7-F16`
  为文档 horizon 标签错误；下一可用编号为 `V7-F17`。

### Paper layout note — P8-A expanded Table 1 visually valid

- official-template compile=7 pages；Table 1 增加 fresh P4/P6-C rows 后 LaTeX 报 `6.03pt` tabular overfull warning。
- 144-dpi page 4 visual audit 确认左右边界未裁切；pages 6/7 的新 exact-once 段、limitations、conclusion 与 references
  continuation 也无 overlap/orphan float。
- 不为消除非裁切 warning 改科学数值、缩小不可读字号或删除 negative-result boundary；无新增 failure id，下一可用仍为
  `V7-F16`。

### V7-F15 — sparsity-consistent candidate 在 fresh nuScenes 未保持 global repair ranking

- run：`run://worldsim_v7/WS-V7-P8A-FRESH-NUSCENES-EXACT-ONCE-01/20260902T200000Z__fresh-nuscenes-final-s0-r1`；
  frozen 20/20 scenes、123 Actors、quality read=1、scene replacement=0、model/threshold update=0。
- symptom：P6-C/P4 repair AUROC=`.747253/.782280`，candidate degradation=`.035027` 超过 preregistered `.02`；其余
  coverage/false-repair/selective-Chamfer gates 通过，AND verdict 仍必须 rejected。
- retained support：candidate coverage=`26.83%`、false-repair=`2.44%`、selective Chamfer=`.19689m < .23441m query`；
  这些是 frozen operating-point utility，不能覆盖 global-ranking regression。
- literature/open-source response：CVPR 2023 DGLSS 的 sparsity-invariant consistency 是 source-only LiDAR DG 机制，不保证
  本任务 selector ranking；ICML 2025 AURC 将 selective system 作为全 risk--coverage ranking 评估。已保留 scores 的
  descriptive AURC=`.12643` (P6-C) vs `.10563` (P4, lower better)，25% coverage risk=`.06452/0`，确认 failure 不只是
  threshold mismatch。sources：`https://openaccess.thecvf.com/content/CVPR2023/html/Kim_Single_Domain_Generalization_for_LiDAR_Semantic_Segmentation_CVPR_2023_paper.html`、
  `https://proceedings.mlr.press/v267/zhou25y.html`。
- resolution：P6-C 不晋升为 primary selector；P4 保持 paper-facing frozen selector。不换 scene、不调 gate/threshold、不
  训练 recovery head。已冻结 fresh AV2 exact-once 仍执行，但只作 external evidence/negative ablation，不反向选择方法。
- claim impact：可主张 sparsity intervention score-shift 降低和当前 operating-point utility，不能主张 selector 全局非劣或
  已被 fresh in-domain final test 支持。

下一可用编号：`V7-F16`。

### P8-A exact-once prevention note — fresh test 不用于方法选择

- 20 scenes 在读取任何 Actor repair/hazard/Chamfer 结果前按 metadata-only rule 冻结，并排除 P4 全部角色；正式 read 后
  禁止换 scene、删难例、重训 P6-C、重新 calibration 或移动 threshold。
- P8-A 只做 frozen P6-C/P4 同 row 比较；AV2 fresh final 仍由另一个已冻结 exact-once runner 消费，不因 P8-A 结果改变。
- scientific gate 已因 `V7-F15` 失败并保留 negative result；资源/入口失败与 scientific rejection 分开登记。下一可用编号
  为 `V7-F16`。

### Paper layout note — P7-B 后第 7 页仅 references continuation

- official-template compile=7 pages；第 7 页没有 orphan figure/table 或正文断裂，仅 bibliography continuation。
- 当前未为压回 6 页删科学边界；CVPR 主文预算仍有余量。无新增 failure id。

### V7-F14 — FP32 roundoff 造成解析 geometry-cost bound 假超差

- r1：`run://worldsim_v7/WS-V7-P7B-GEOMETRY-COST-SENSITIVITY-01/20260902T190000Z__geometry-cost-sensitivity-s0-r1`。
- all-source maximum `shift-bound=3.8147e-6`，frozen tolerance=`1e-6`；P5 exact-match strata 在所有 6 个 shifts 为
  0 violation，且 tightness 约 1，符合 subtract/divide roundoff 特征。
- 对策：解析计算改 FP64，保持 source rows、deltas、clearance floor、groups、tolerance 全部不变后 r2；禁止通过放宽
  tolerance 把 r1 改写为通过。
- outcome：r2=`run://worldsim_v7/WS-V7-P7B-GEOMETRY-COST-SENSITIVITY-01/20260902T191500Z__geometry-cost-sensitivity-s0-r2`
  为 0 violation，maximum overage=`1.42e-14`；failure 已关闭但保留 r1 provenance。

### P7-B prevention note — 确定性 Lipschitz bound 不等于扰动概率或安全证书

- frozen stress 只对 retained profile 的 uniform signed-clearance shift 验证代数上界与紧致性。
- 禁止把 bound satisfaction 写成 sensor error distribution、physical repair causal effect、closed-loop collision bound 或
  real-road safety guarantee；bound 很松时必须如实报告 tightness 与 clearance-floor crossing。
- r1 数值实现失败已登记 `V7-F14`；下一可用编号为 `V7-F15`。

### Paper claim-boundary note — P5-B 零事件不写成安全保证

- P5-B table 保留 `0/1,781` false-safe/flip 观测，同时正文明确它只有 5 个 selected Actors、来自 retained source
  outcomes、未执行 P346、非 causal effect/confidence bound。
- method 固定 `g_i=1 \not\Rightarrow a_{i,t}(c)=1`，避免把 physical selection 误写成 planning/control certificate。
- 无新增 failure id；该限制已同步 abstract、method、experiments、limitations、conclusion。

### P5-B safety-boundary note — physical repair authority 不是 trajectory authority

- canonical：`run://worldsim_v7/WS-V7-P5B-FROZEN-PHYSICAL-RELIABILITY-INTERFACE-01/20260902T183000Z__frozen-physical-reliability-interface-s0-r1`。
- 无新增 failure id：5 个 selected Actors 的 `1,781` retained rows 为 `0` false-safe、`0` decision flip，且 23 个
  geometric-harm Actors 全部 abstained。
- 必须保留的边界：selected q90 state error=`1.577m`，是 abstained `.693m` 的 `2.28x`；3.0s 达 `4.246m`。
  因此禁止把 P4 selection 当成 C3/规划控制 authority。该零事件小样本也不构成 formal guarantee。

### V7-F13 — P4 train 与 V6.7 source 的 exact Actor overlap 不足以联合轻训

- canonical：`run://worldsim_v7/WS-V7-P5-PHYSICAL-RELIABILITY-ALIGNMENT-AUDIT-01/20260902T180000Z__physical-reliability-alignment-s0-r1`。
- exact join 后 P4 train 仅 `2 scenes/5 Actors/1,320 rows`，低于 frozen `3 scenes/20 Actors`；不能构造非平凡
  train-side scene holdout。
- 拒绝手段：不放宽 identity join，不把 calibration/test Actors 回流训练，不按类别/空间近邻猜 identity，不做 scene
  shortcut joint fit。
- 对策：P4 与 P346 权重不变，只在 118 个 exact-match Actors 上做 descriptive multi-horizon interface audit。本失败是
  source identity coverage 边界，不是算力/工程失败。

### P5 alignment prevention note — 不用稀疏 scene overlap 强行联合轻训

- P4 与 V6.7 identity namespace 不同，必须经 official scene ordering 与 DriveStudio instance id 映射，不能按类别/空间
  最近邻猜 Actor 对应。
- direct joint fit 只在 P4 train 至少 3 scenes/20 Actors 对齐时允许；否则 cal/test 角色不能回流训练，避免 scene identity
  shortcut 与 role leakage。
- audit 只读 retained rows/metadata；未通过不触发 training rescue。正式结果已登记为 `V7-F13`。

### V7-F12 — ratio-only opportunity invariance 丢失 nuScenes repairability ordering

- run：`run://worldsim_v7/WS-V7-P6-OPPORTUNITY-INVARIANT-SELECTOR-01/20260902T170000Z__opportunity-invariant-s70601-r1`；
  fit 只读 P4 retained nuScenes rows，fresh AV2 compiled/scored=`0/0`。
- symptom：固定 `.5x/2x` opportunity transform feature shift=`0`，但 nuScenes-test repair AUROC=`.60728`，低于 P4
  `.64908` 与 frozen floor `.62908`；cal/test coverage=`14.29/13.16%`，test conditional failure=`36.67%`。
- root cause：将 surfel/support 全部除以 observation count 是 non-invertible compression；它移除 raw frame-count shortcut
  的同时也移除了与 repairability label 有关的 evidence amount。结构 exact invariance 不能替代 label sufficiency。
- literature/open-source response：CVPR 2023 DGLSS 用 source-only random LiDAR subsampling 与 sparsity-invariant feature
  consistency保留语义关系；CVPR 2024 LiDAR detector generalization study 也支持 source-domain low-resolution augmentation。
  下一 hypothesis 保留 raw interpretable features，只在 nuScenes 训练施加 fixed opportunity subsampling + score consistency。
- resolution/impact：ratio-only candidate=`rejected_before_external_read`；runner 阻止 external phase。P3/P4/P7 结论不变，
  fresh 20-log cohort 未消费。status=`closed_ratio_normalization_family`。

下一可用编号：`V7-F13`。

### P6-C freeze prevention note — source sparsity consistency 不读取 fresh target

- P6-C 由 `V7-F12` 的 source-domain failure 和 CVPR 2023/2024 source-augmentation evidence 提出，不读取 fresh AV2。
- `.5x/.75x`、consistency weight=`1`、seed=`70602` 与两项 fit gates 已固定；不因 fit 或未来 external 结果扫描。
- 只有 nuScenes non-inferiority 与相对 P4 intervention shift 两门通过，runner 才允许 external phase；否则 fresh quality
  保持 unread。实现只提供 formal fit，未用部分下载日志做 adapter smoke。本项不是新 failure；下一可用编号仍为
  `V7-F13`。
- formal fit 随后以 AUROC `.63239>=.62908`、shift ratio `.10668<=.70` 通过 2/2；这只解锁 fresh external read，
  不把 source intervention robustness 升级为 external guarantee，也不改变下一 failure 编号。
- external runner 仅增加 20/20 `.complete` operational precondition；partial cohort 不编译/打分，防止 IO 顺序变成隐式
  target development。它不增加质量 gate。本项仍非 failure，下一可用编号仍为 `V7-F13`。

### P6 recovery prevention note — consumed AV2 不参与机会归一化选择

- P7 已精确暴露 `observation_frame_count` shortcut，因此 P6 只能删除/归一化该机会变量，不能借 30 个 consumed AV2 logs
  选择更多 feature、seed、threshold 或 gate。
- fresh 20 logs 由 150-log metadata 排序、排除 v1 multiples-of-five indices、对 complement 固定 every-sixth 得到；冻结时
  method output/quality read=false。下载完成后只允许一次 external read。
- runner 强制拆为 nuScenes-only `fit` 与 fresh-only `external`；fit artifact/status=`model_frozen_waiting_fresh_av2`，
  防止 IO 未完成时误读部分 fresh cohort 或用逐日志结果修改模型。
- P6 即使通过也仍是 empirical transfer；factorized cross-task derivative zero 与 sensor-domain invariance 是两个不同命题。
  本项不是新 failure；随后 ratio-only fit 暴露 `V7-F12`，下一可用编号为 `V7-F13`。

### V7-F11 — factorized validity head 依赖跨域不稳定的 sensor-opportunity shortcut

- run：`run://worldsim_v7/WS-V7-P7-INTERPRETABLE-SAFETY-ENVELOPE-01/20260902T163000Z__safety-envelope-s0-r1`；
  P7 不训练、不重校准，只解释冻结 P4 r2 model/rows/threshold。
- symptom：factorized score calibration→AV2 mean=`.7614→.9783`、median=`.9453→.999992`，Wasserstein=`.2170`、
  KS=`.7041`；相同 threshold 的 coverage 从 `8.93%` 跳到 `75.55%`。64-step IG 中 observation frame count 的
  normalized attribution 从 nuScenes test `18.25%` 增到 AV2 `49.53%`。
- root cause：head 虽不读 hazard inputs，却直接读取 raw observation-frame count；不同数据集的序列长度、传感频率与
  Actor 可见窗口改变该量，模型把 sensor opportunity 当作 repairability。结构因子化消除了 cross-task leakage，但没有
  自动获得 sensor-domain invariance。
- literature response：Integrated Gradients 用于明确模型敏感度而非因果；LiDAR domain-generalization/adaptation 工作反复
  指出 density/sensor pattern shift。V7 恢复只允许 nuScenes development 上的 dimensionless support/density normalization
  或显式 sparsity-invariance，不在 AV2 consumed rows 上调 threshold。
- impact：P4 7/7 gates 和 AV2 false-repair/Chamfer 数值仍是有效 empirical zero-shot read；删除的是“域不变 selector”
  或 external formal risk guarantee 的解释。30 个 AV2 logs 已消费；恢复需 metadata-frozen 新 AV2 cohort 或 Waymo。
  status=`open_sensor_opportunity_normalization_and_fresh_external_confirmation`。

下一可用编号：`V7-F12`。

### P7 freeze prevention note — attribution 不冒充因果，曲线不用于重选 AV2 threshold

- P7 只在固定 coverage grid 上描述已完成 P4 score ordering；star 始终是 nuScenes calibration 已冻结 threshold，不能从
  AV2 curve 选择更好 operating point。
- Integrated Gradients baseline 固定为 nuScenes-train standardized mean；其 completeness 只验证 path attribution 数值，
  feature magnitude 仍不是 sensor causality、物理充分性或跨域稳定性证明。
- factorized zero leakage 来自输入图不连通，不依赖 IG/相关性；shared leakage 仍按 P4 paired swaps 报告。
- 本项不是新 failure；随后正式读取暴露 `V7-F11`，下一可用编号为 `V7-F12`。

### P4 outcome note — 支持 empirical selective transfer，不恢复跨域 formal guarantee

- r2 7/7 gates，但 nuScenes calibration 与 AV2 coverage=`8.93%/75.55%`；该 shift 不是可忽略的 calibration noise，
  而是外域 exchangeability 不成立的直接提醒。
- AV2 factorized selective false repair=`8.99%`，低于 always repair=`16.56%`；但 mean Chamfer `.1821m` 仍比
  always repair `.1770m` 高 `5.03mm`。不得写成全面支配或逐 Actor universal certificate。
- hazard coverage=`92.17%`、hazard selected failure=`2.36%`、cross-input shift=`0/0` 支持“没有靠风险输入拒绝危险 Actor”的
  结构结论；Actor/hazard retention 仍只是 immutable interface，不是上游 detection/tracking correctness。
- 本项不是新 failure；下一可用编号仍为 `V7-F11`。

