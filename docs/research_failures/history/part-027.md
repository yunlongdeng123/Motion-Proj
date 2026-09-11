# 历史原始记录 027

> 冻结历史，不是当前执行规则。当前规则见 [scaling law](../../../auto-research_scaling_law.md) 与 [AGENTS](../../../AGENTS.md)。
> 原文件：`docs/RESEARCH_FAILURES.md`；迁移前提交 `abbd431c`。原有相对链接按原目录解释，证据路径原样保留。
> 按索引读取相关小节；旧哈希、过度门控和资源指令均不重新生效。

### P3-D outcome note — completion tail不触发 source-ray carving

- run：`run://worldsim_v7/WS-V7-P3D-AV2-VISIBLE-FAILURE-ATTRIBUTION-01/20260902T224500Z__visible-failure-attribution-s0-r1`；
  all 634 Actors、all target rays、all provenance aligned。
- COMPLETE产生 `24,445/25,698` new early，但同时产生 `365,634` new hits（`14.96×` gain/cost）；总体 early相比
  query仍净减少 `9,439`。
- observable surface contradictions中 COMPLETE只有 `56/415=13.49%`，KEEP为 `359/415=86.51%`；因此 completion
  并不同时主导两种物理失败，预冻结 carving解锁条件不成立。
- resolution：关闭 source-ray carving与新 operator；不以 target-only结果删除 completion，不调 ray tolerance。保留
  `V7-F18` actor-tail，P4 select/abstain继续承担逐 Actor authority，fresh P3-C只作原合同确认。
- 本诊断没有新 failure id；下一可用仍为 `V7-F19`。

### P3-D prevention note — 不把 nearest-ray assignment尾部误判成 completion物理冲突

- P3-C 的 target→surface nearest ray与 surface→target nearest ray不是同一个匹配；前者新增 early count不能直接推出
  compiler插入了新自由空间冲突。
- P3-D 覆盖全部634 Actors与全部 target rays，只把既有输出点按 compiler原始 KEEP/PROJECT/COMPLETE provenance计数；
  不挑 top failures、不改 tolerance、不新增 gate。
- 只有 COMPLETE 同时主导 new early与surface contradiction时，才允许 source-ray space carving；KEEP/PROJECT主导则
  说明问题主要是稀疏覆盖/nearest assignment边界，禁止用表面删除“修复”指标。
- 当前为 post-result diagnostic freeze，无新 scientific verdict；下一可用 failure id保持 `V7-F19`。

### V7-F18 — pooled visibility physics改善没有形成逐 Actor 通用证书

- run：`run://worldsim_v7/WS-V7-P3C-AV2-VISIBILITY-CERTIFICATE-DEV-01/20260902T223000Z__visibility-audit-s0-r1`；
  consumed 30-log descriptive cohort，visibility definitions在指标读取前提交并推送。
- retained aggregate：query→compiled target-hit recall `.49691→.69831`、early termination `.03395→.02868`、visible
  precision `.99606→.99765`、F-score `.66304→.82156`，surface contradicted points `856→415`。
- symptom：只有 `406/634` Actors 不新增 target-early 或 surface-contradiction count；absolute-zero仅 `10/634`。
  `214` Actors early count上升，`34` Actors surface contradiction上升；其中 `20` 同时上升。
- failure concentration：hazardous Actors nonnew=`135/230`，added-early positive mass=`5,062`，高于 non-hazard的
  `2,586`；query-relative Chamfer-worse 105 Actors里只有24个 visibility F-score non-inferior。
- literature response：NeuRAD要求显式 ray/sensor model；ICCV 2023 LiDAR domain-gap study强调 paired real/sim sensor
  evaluation；CVPR 2025 EvOcc 通过 first-return ray casting区分 free/occupied。它们共同支持保留 ray partition，而不支持
  用 pooled Chamfer/precision覆盖 Actor tail。
- resolution：不调 `.20m` tolerance、不删 Actor、不改 fresh cohort/operator。下一步仅做全 Actor nearest-output provenance
  attribution；若失败主要来自 target-ray nearest matching，则修 evaluator；若来自 COMPLETE 的 source-ray free-space
  violation，才允许预冻结 source-ray space carving。fresh exact-once仍按原合同运行。
- claim impact：可主张 aggregate zero-shot observed-ray 物理改善；不可主张每个 Actor 的完整表面/碰撞/安全 certificate。

下一可用编号：`V7-F19`。

### P3-C prevention note — visibility certificate 不把遮挡/未观测空间冒充自由空间

- Chamfer 对不可见背面和实际可证伪的 early return 混为一个距离，不能单独承担三维物理 certificate；P3-C 不删除
  `V7-F09` 的109个 Actor-level worsening，而是单列其 observed-ray contradiction状态。
- target rays固定分为 early/hit/late/unmatched，surface固定分为 contradicted/supported/occluded/UNKNOWN；只有 observed
  free-space contradiction可否定表面，occluded/off-ray 永远保持 UNKNOWN。
- `.20m` lateral/depth tolerance沿用 P3，不根据30-log descriptive结果或fresh 20-log结果修改；fresh cohort不参与模型、
  threshold、operator、Actor或scene选择。
- 本项不把“无可见冲突”包装成完整表面正确、碰撞规避、规划、闭环或道路安全保证。正式 descriptive 结果随后登记
  `V7-F18`；下一可用为 `V7-F19`。

### Official-template provenance note — pinned author kit仍为官方HEAD，无新failure

- official GitHub `cvpr-org/author-kit` main HEAD仍是 pinned `2917585`，latest release仍为 CVPR2026；
- 不从搜索结果、Overleaf或第三方仓库引入所谓 CVPR2027 template；当前 draft只设置 `confYear=2027`；
- style files未改、无需重编或新增 layout gate；等官方 2027 kit发布后再迁移。无failure id，下一可用仍为 `V7-F18`。

### Paper source-convergence note — legacy TBD contract移除，无新failure

- `main_results.tex` 是早期未引用表，18个 TBD macros只由该表引用；main/supplement PDF均未渲染 TBD；
- 删除范围仅为该 dead table、18个 dead macros 与 `\todoresult` wrapper，Git可恢复；真实 P1--P9 numbers/tables未动；
- main/supplement重编译保持8/6页和原文件大小，warning集合未增加；
- source-only cleanup，不创建 failure id；下一可用仍为 `V7-F18`。

### V7 branch-base audit note — corrected V6.7 ancestry成立，无新failure

- `research/worldsim-v6.7-anisotropic-surface` terminal=`d97c3f2`，同时是当前 V7 的 exact merge-base；ancestor=true、
  V7 relative ahead/behind=`50/0`；
- 内部旧续跑描述中的 V6.4→V6.5 路线不是当前用户目标，也未被执行；当前唯一 active branch仍是 corrected V7；
- read-only Git audit，不执行 rebase/merge/checkout，不影响下载、watcher、canonical runs或论文；无failure id，下一可用
  仍为 `V7-F18`。

### Contribution-map drift note — 早期ownership已修正，无新failure

- 旧 `CONTRIBUTION_MAP.md` 仍指向 Sec. 3.1--3.4、早期文件名，并笼统写“P8 owns final exact-once numbers”；这与
  该审计时点P8-A fresh nuScenes rejection、P6-C等待fresh AV2和P9/P7-C completion不一致；
- resolution：重写为4项当前贡献、11个 canonical stages、failure boundary与 final claim checklist；明确 P4 primary、
  P8-A source exact-once only、P6-C external-only ownership；后续external已完成并以`V7-F19`补齐；
- exposure：论文正文与 canonical summaries未受影响；属于 documentation drift，不创建scientific failure id。当前下一
  可用已由后续账本推进到`V7-F20`。

### Project-page asset-index note — 完整失败案例保留且无新failure

- 索引覆盖 P3-B canonical 30/30 panels 与 30/30 MP4，并保留 13 hazardous Actors、低可见性 q05/q06 cases 与
  q07 Chamfer-worsening case；不因项目页展示目的重选素材；
- main/compact-supplement/full roles分别固定为8/10/30，camera仍按 query-point visibility选择，RGB不参与选择；
- 46MiB binary bundle不重复写入Git，索引只保存 canonical run与相对路径；不新增 hash/checksum/fingerprint；
- documentation-only，不创建新failure id；下一可用仍为 `V7-F18`。

### Paper literature note — cross-sensor motivation不越权为P6-C证书

- CVPR 2023 DGLSS 的 source subsampling consistency与 ICCV 2023 3DLabelProp 的 sequence/geometry common
  representation被用于说明已知 LiDAR sensor shift 对策；它们不直接验证本项目的 Actor repair selector；
- 主稿明确保留 exact-once source/external evaluation，不由相关工作推断 fresh AV2 verdict，也不改变 P6-C candidate；
- compile仍为8页，pages 2/7/8无裁切或重叠，只有既有 Table 1 `6.03pt` overfull；无新failure id，下一可用仍为
  `V7-F18`。

### P6-C external auto-launch prevention note — 无新failure

- watcher 只在 `ALL_COMPLETE + 20/20 .complete` 后进入 frozen external runner，未完成阶段不读 Actor target quality；
- 单 `flock` 保证不会与人工或重复 watcher 并发启动同一 exact-once evaluator；status必须仍为
  `model_frozen_waiting_fresh_av2`，否则拒绝执行；
- 下载器若在 readiness 前退出，watcher报告资源/下载异常并停止，不删除、不补换 log、不修改 cohort/model/threshold/gate；
- 当前下载 `11/20` 且无 error/retry，本项仅 orchestration，不创建新 scientific failure id；下一可用仍为 `V7-F18`。

### Supplement layout note — evidence gallery收敛且无新failure

- 初版双栏 float 使 failure heading/table 与 gallery intro 分离并产生近空白页；这是未提交的排版中间态，未改变任何
  scientific artifact、case identity、数值或 claim。
- resolution：在 failure ledger 起切换为单栏，表与十个 frozen gallery panels 使用正常浮动顺序；official-template
  最终为 `6 pages/7,219,027 bytes`，pages 1--6 逐页检查无 blank page、float reorder、clipping 或 overlap。
- gallery 固定每个 qualitative log 的 actor-rank-0，共10例；未按视觉效果或 metric 重选，且保留 q07 Chamfer
  worsening case。P7-C stable false repairs 与标准化 stress-box 限制也完整保留。
- 本阶段是 documentation/layout work，不创建新 failure id；下一可用编号仍为 `V7-F18`。

### Paper layout note — P7-C boundary integrated without new failure

- official-template=`8 pages/1,164,284 bytes`；pages 5--8 visual audit 无 clipping/overlap，page 8 仅 bibliography continuation。
- 正文如实保留 AV2 robust-select 中 `10.76%` false repair，并明确 standardized box不是 calibrated sensor noise；没有把
  network stability包装为 correctness/safety certificate。
- 只有既有 Table 1 `6.03pt` overfull warning，无新增 warning/failure id；下一可用编号仍为 `V7-F18`。

### V7-F17 — 跨域 feature-box 稳健性没有排除稳定的错误修复

- run：`run://worldsim_v7/WS-V7-P7C-VALIDITY-INTERVAL-CERTIFICATE-01/20260902T220000Z__validity-interval-s0-r1`；
  frozen P4 network/threshold、FP64 interval propagation、target-independent certificate state。
- symptom：`.10` train-std all-feature box 下，consumed AV2 的 479 个 nominal-selected Actors 有 437 个仍 robust-select，
  但其中 47 个 target false repair（`10.76%`）；nuScenes test 仅 `3/9` nominal decisions robust-select 且 0 false repair。
- interpretation：AV2 score saturation 不只是 threshold 附近脆弱性；错误决定可在较宽 feature box 内稳定。网络的局部
  threshold invariance不能推出表面修复正确、domain calibration、collision avoidance 或 road safety。
- mechanism：`.10` sensor-opportunity box 的 mean logit width 在 test/AV2=`1.5496/1.2850`，均高于 physical-surface
  `1.1329/.9836`；`log_query_points` 是 test `190/228`、AV2 `603/634` 的 top-1 interval-width feature，与 P7 的
  observation-opportunity attribution相互印证。
- resolution：保留 P4 的 empirical risk--coverage/geometry结果，但论文新增“stable decision != correct repair”边界；不把
  IBP state作为新 safety gate，不据此删 Actor、调 threshold/radius/group、适配 AV2 或改 P6-C fresh protocol。
- claim impact：可主张逐 Actor deterministic network-decision interval与机会特征解释；不可主张 physical/sensor/safety certificate。

下一可用编号：`V7-F18`。

### P7-C prevention note — feature-box certificate 不越权为现实安全保证

- P7-C 的 `.05/.10/.20` 是 nuScenes-train standardized stress radii，不来自 AV2/nuScenes sensor-noise calibration；禁止
  将 interval state 解释为真实扰动概率、coverage confidence 或跨域 exchangeability。
- `robust_select` 只表示 frozen P4 network score 在整个 box 内保持高于 frozen threshold；错误且稳定的网络决策仍可能是
  target false repair，因此 retained target failure 必须单独报告。
- feature groups、radii、threshold 与 explanation radius 在结果读取前冻结；fresh AV2 recovery rows不读，不以 consumed
  AV2 target反向调 box。正式结果随后登记 `V7-F17`；下一可用编号为 `V7-F18`。

### Paper layout note — P9 后第 8 页仅 references continuation

- official-template compile=`8 pages/1,162,514 bytes`；pages 6/7/8 visual audit 无 clipping、overlap 或 orphan float，
  第 8 页只有 bibliography continuation。
- P9 Table 5、retained-source limitation 与 `2.5s` heldout label 均可见；未把 authority conditioning 写成 causal
  planning/closed-loop/collision/safety improvement。
- 编译无新增 warning；既有 Table 1 `6.03pt` overfull 仍视觉有效。不通过改数字、删限制或缩小不可读字体压缩。
- documentation-only paper integration 不创建新 failure id；下一可用编号仍为 `V7-F17`。

### V7-F16 — P9 冻结说明误标 P346 heldout horizon

- affected run：`run://worldsim_v7/WS-V7-P9-COMPOSED-AUTHORITY-FIXED-LATTICE-01/20260902T213000Z__composed-authority-s0-r1`。
- symptom：docs freeze 写 `3.0s`；但 frozen config 未硬编码 horizon，runner 明确读取 P346 lattice artifact 的
  `heldout_horizon_index`，artifact index=`2` 对应 `2.5s`，canonical summary 也在结果生成前记录 `2.5s`。
- exposure/root cause：retained-source data 已全部消费过；错误来自人工把四 horizon 最后一项误认为 heldout，而 P346 的
  source training indices 与 heldout index 已由 upstream artifact 冻结。没有根据 P9 metrics 选择 horizon。
- resolution：以 executable artifact contract 为科学 source of truth，把 docs 的 `3.0s` 更正为 `2.5s`；不改 config/
  runner/model/data/budget/threshold，不重跑、不生成 r2。
- claim impact：r1 仍为 canonical `2.5s` retained-source demo；failure 只影响旧文字标签，不影响数值或 verdict。

下一可用编号：`V7-F17`。

