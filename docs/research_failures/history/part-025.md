# 历史原始记录 025

> 冻结历史，不是当前执行规则。当前规则见 [scaling law](../../../auto-research_scaling_law.md) 与 [AGENTS](../../../AGENTS.md)。
> 原文件：`docs/RESEARCH_FAILURES.md`；迁移前提交 `abbd431c`。原有相对链接按原目录解释，证据路径原样保留。
> 按索引读取相关小节；旧哈希、过度门控和资源指令均不重新生效。

## V7-F29 — true first-return exposes larger risk but all deletion policies pay Chamfer

- canonical=`run://worldsim_v7/WS-V7-P20-TRUE-FIRST-RETURN-AUDIT-01/
  20260903T124500Z__true-first-return-audit-r1`；fresh AV2未读。
- metric correction：always-COMPLETE total/hazard new-early由legacy `.9662/1.4362%`变为true-ray
  `5.9561/8.7421%`，说明target-nearest proxy低估首返回风险约6倍。
- P17/P17R/P19 hazard events分别`14219→12649/13386/14088`，第一门全true；Chamfer分别
  `.1994111/.1957160/.1946787m > .1945869m`，第二门全false。
- interpretation：deletion对true first-return early有集合单调性，但对hit与bidirectional Chamfer没有；这不是再调
  threshold/coverage能消除的评估噪声。
- response：关闭deletion-only policy sweep；P21形式化单调安全定理与经验交换率，不训练、不选policy、不读target。

## P21 prevention note — theorem exact, empirical frontier bounded（2026-09-02）

- 只从P20 summary派生；不重编译、不以ratio重新选policy、不增加epsilon/smoothing。
- set-inclusion只保证early非增，不保证hit或Chamfer；不得扩写成collision-free或road-safety guarantee。
- P19高效率是consumed source描述，不是fresh transfer evidence。
- artifact同时保存定理premise/result/non-guarantees与frontier；ratio零分母明确为null，不伪造有限效率。

## V7-F28 — one-slot hazard veto reduces early returns but still pays surface utility

- canonical=`run://worldsim_v7/WS-V7-P19-SPARSE-HAZARD-VETO-SOURCE-01/
  20260903T114000Z__sparse-hazard-veto-s71701-r1`；fresh AV2未读。
- action：35/79 hazard Actors各veto一个candidate，总计`35/3325`，coverage=`98.947%`；clear Actors完全不变。
- benefit：hazard new-early `2336→2302`（`1.43622→1.41532%`），total `2982→2948`。
- cost：Chamfer `.1945868→.1946787m`，new hits `39255→39119`；严格Pareto第二项false。
- prevention：关闭score-ranked candidate veto top-k/threshold/stratum/ranking变体，不以34 events收益放宽Chamfer门。
- audit response：legacy attribution先取target的Euclidean-nearest surface再检查ray，并非literal first return。P20只做一次
  frozen P17/P17R/P19 true-first-return审计，不改policy；下一failure=`V7-F29`。

## P20 prevention note — correct the ray operator, not the policy（2026-09-02）

- 每条held-out ray在lateral tolerance内取minimum positive depth；query/compiled同算子、同冻结depth tolerance。
- P17/P17R checkpoint、P19 action、Chamfer、scene均固定；不得在corrected result后选择tolerance/operator或重训。
- consumed source只作诊断，fresh AV2保持未读；P20本身不冒充独立confirmation。
- implementation只做chunked GPU `min(where(valid, depth, inf))`；不得回退target-nearest、改为soft rendering或增加
  morphology后处理。

## V7-F27 — rare Pareto-positive support collapses Actor routing to the baseline

- canonical=`run://worldsim_v7/WS-V7-P18-TWO-EXPERT-ROUTER-FIT-01/
  20260903T104500Z__two-expert-router-s71801-r1`；fresh AV2未读。
- support：train+calibration仅`3/85` P17R-dominant，consumed test仅`2/228`；router在test选择P17R=`0`、
  always-COMPLETE=`228`，accuracy=`99.12%`是majority解而不是有效authority。
- result：hazard new-early与Chamfer均精确等于baseline，第一门false、第二门true；loss `.69615→.04729`不能弥补
  runtime features对稀有dominance缺乏可迁移分离。
- oracle boundary：post-verdict两Actor oracle仅减少1个hazard early、mean Chamfer改善约`6.1e-6m`，当前expert family
  的可用增益本就极窄。关闭Actor router/class-weight/width/threshold sweep。
- response：P19把动作单位降到hazard Actor内固定一个UNKNOWN slot；不学习router、不改`.5`、不扫capacity；下一failure=
  `V7-F28`。

## P19 prevention note — fixed one-slot candidate veto（2026-09-02）

- 仅hazard Actor可veto；若存在冻结P17R score `<.5`，只把minimum-score一个candidate标UNKNOWN，其余完整保留。
- clear Actor恒always-COMPLETE；KEEP/PROJECT不可删除；不得以oracle test label直接选Actor。
- source双Pareto失败即关闭score-ranked veto，不扫top-k/threshold/strata/ranking；fresh AV2仍须source通过才可读。
- implementation直接以`np.argmin`固定首个tie，输出one-slot UNKNOWN/其余OCCUPIED；不得借实现错误扩大veto容量。

## V7-F26 — normalized hybrid supervision narrows but does not close the Pareto gap

- canonical=`run://worldsim_v7/WS-V7-P17R-HYBRID-RAY-CHAMFER-FIT-01/
  20260903T094500Z__hybrid-ray-chamfer-s71701-r1`；fresh AV2未读。
- recovery：coverage `71.43→88.96%`；P17 Chamfer regression约76.6%被收回；hazard early仍降`1.4362→1.4153%`。
- remaining failure：Chamfer `.1945868→.1957160m`，new hits `39,255→37,518`，严格Pareto第二项false。
- resolution：关闭ray loss/Chamfer mixture/threshold family，不以近似通过放宽边界。下一方向是固定experts的two-stage
  Actor routing，label仅用逐Actordominance，不引入utility weight；下一failure=`V7-F27`。

## P18 prevention note — router选择完整expert，不重开surface threshold（2026-09-02）

- experts固定为always-COMPLETE和P17R；router不得混合点集、改`.5` threshold或删除KEEP/PROJECT。
- P17R label只在Actor Chamfer/new-early双不退且至少一项严格改善时成立，否则明确回退baseline。
- nuScenes test已被base expert消费，只是router development；不得包装为新source confirmation。
- source aggregate双Pareto通过才允许fresh AV2；不扫router feature/width/seed/class weight；下一failure=`V7-F27`。
- implementation保持完整expert原子性：baseline action直接恢复always-COMPLETE质量/attribution，P17R action原样保留；不得
  在路由后重新混合两者点集。checkpoint引用而不复制/微调P17R。

## V7-F25 — ray-only first-return objective trades away 3D surface utility

- canonical=`run://worldsim_v7/WS-V7-P17-RAY-SET-COMPLETION-FIT-01/
  20260903T084500Z__ray-set-fit-s71701-r1`；0 fresh AV2 read。
- success channel：hazard new-early `1.4362→1.3760%`，clear `.4425→.3925%`，P16 non-compositionality被joint ray loss修复。
- failure channel：coverage=`71.43%`，Chamfer `.1945868→.1994111m`，new hits `39,255→34,242`；ray-only没有完整
  surface responsibility。
- prevention：不扫`.5` threshold/ray budget/epoch/seed；不得只报告early下降隐藏Chamfer/hit损失。
- one recovery：同模型同ray loss加入expected bidirectional Chamfer，两个loss各除以always-COMPLETE source reference，
  固定1:1 dimensionless sum；下一failure=`V7-F26`。

## P17 prevention note — joint ray loss, not another point-label rescue（2026-09-02）

- P17复用feature仅为控制变量；prediction object由独立三态label改为同ray candidate set的first-return depth。
- 固定core KEEP/PROJECT作为ray fallback，completion只竞争其前方位置；不得让模型删除KEEP/PROJECT来伪造改善。
- hard `.5` selection、straight-through gradient、1024 rays与Smooth-L1均在source execution前冻结，不扫温度/阈值/loss。
- source双Pareto失败即关闭，不以classification accuracy或soft rendered loss授权fresh AV2 read；下一failure=`V7-F25`。

## V7-F24 — independent completion-candidate classification is non-compositional

- canonical=`run://worldsim_v7/WS-V7-P16-EVIDENTIAL-COMPLETION-FIT-01/
  20260903T073500Z__completion-fit-s71601-r2`；fresh AV2未读。
- imbalance：source fit三态=`6/1739/25`；test=`6/3295/24`。P16 test FREE recall=`0`、UNKNOWN F1=`.0619`，
  occupied-prior accuracy `.9693`不可写成三态学习成功。
- physical symptom：completion coverage仍`.9765`，但hazard new-early从`1.4362%`升至`1.4811%`，Chamfer从
  `.1945868m`升至`.1950206m`，new hits少899；源域即双退化。
- mechanism：删除一个candidate会暴露同ray上另一更早的KEEP/PROJECT/COMPLETE点；单点label在first-return composition
  下不封闭，candidate classification不是正确prediction object。
- resolution：关闭P16 external，不扫class weight/features/threshold/seed；下一候选必须jointly optimize ray/set
  transmittance或first-return depth。第三批AV2可继续下载但保持unread。
- next failure id=`V7-F25`。

## P16 fit r1 engineering failure — diagnostics package未解包（2026-09-02）

- run=`20260903T071500Z__completion-fit-s71601-r1`；首个nuScenes source scene完成后在写diagnostics时NameError。
- root cause：`compile_nuscenes_scene`仍以`row, _`解包，但新增路径访问`package["diagnostics"]`。
- resolution：单行恢复`row, package`，r1保留failed；0 model training/checkpoint、0 source result、0 fresh AV2 read。
- classification：纯实现故障，不登记`V7-F24`，也不授权改变P16冻结features/model/loss/seed/decision。

## P16 prevention note — completion责任必须在candidate层学习（2026-09-02）

- P15证明Actor-level P4/P6-C未压低hazard COMPLETE channel；P16不得再包装另一个Actor score或低coverage conjunction。
- labels固定为held-out ray FREE/OCCUPIED/UNKNOWN；只有argmax OCCUPIED可发COMPLETE，FREE/UNKNOWN不被强制二值化。
- 模型固定11维features、`64-64-3`、seed71601与single cross-entropy objective；source test结果不得触发第二candidate。
- 既有50个AV2 logs全部consumed，不得用于feature/loss/seed/epoch/action选择；第三批10 logs在任何model/quality read前冻结。
- 不把softmax/Dirichlet/evidence mass写成formal uncertainty、calibration或safety guarantee；评价仍是首占用深度、new early与
  composite geometry。
- 禁止fresh read后扫threshold/tolerance/features或删除失败log/Actor；下一可用failure id=`V7-F24`。

## P15 paper audit — action attribution不得越界为action ablation（2026-09-02）

- main/supp已同时陈述COMPLETE主导new early/new hit、KEEP主导surface contradiction，以及P4/P6-C未压低hazard
  completion channel；不得只保留收益数字或aggregate hazard/clear ratio。
- `nearest compiled output`只提供输出来源归属，不提供移除KEEP/PROJECT/COMPLETE后的反事实；论文已使用
  `localizes/attributes`，禁止使用`causes/prevents`等action-ablation措辞。
- PROJECT零计数的KEEP-first dedup解释已同时进入main、supp与limitations，禁止据此声明PROJECT零风险。
- final main=`10 pages/1,771,543 bytes`（8 content + 2 references），supplement=`8 pages/7,228,331 bytes`；
  main 7/supp 2视觉QA通过，仅既有Table 1 `6.03pt` non-clipping overfull。

## P15 result boundary — hazard early-return负担由COMPLETE主导，selector未过滤（2026-09-02）

- canonical audit=`run://worldsim_v7/WS-V7-P15-FRESH-HAZARD-ACTION-AUDIT-01/
  20260903T044500Z__fresh-hazard-action-audit-s0-r1`；523 Actors/20 consumed-fresh logs exact join。
- mechanism split：COMPLETE占all new early/new hits=`94.70/99.94%`；KEEP占surface contradictions=`95.68%`。不得
  用任一metric替代另一metric，也不得把nearest provenance写成action ablation。
- hazard burden：always hazard/clear new-early=`1.912/.980%`；hazard是`1.951x`。COMPLETE hit/early=
  `13.67/17.62`，hazard每个new-early换得的new-hit更少。
- selector boundary：P4/P6-C selected hazard new-early rate分别是always hazard的`1.0055x/1.0010x`，未滤除该机制；
  这与P14 hazard visible-risk未降一致，但不推翻P4 Chamfer target。
- PROJECT prevention：0 output来自KEEP-first voxel dedup provenance collapse，不是PROJECT零risk/zero effect。
- classification：现有论文已分离hazard-state preservation、visibility与causal safety，故为descriptive hard-evidence
  boundary，不登记`V7-F24`；禁止调completion/tolerance/selector在本cohort recovery。

## P15 prevention note — nearest-output provenance不是action intervention（2026-09-02）

- P15以target ray的nearest compiled output给new early/hit/contradiction归因，只回答“当前输出最接近哪个action来源”，不回答
  “删除该action会怎样”；禁止使用causal、ablation或safety intervention措辞。
- PROJECT使用observed LiDAR hit，且KEEP在concatenate/dedup中优先；同voxel PROJECT会显示为KEEP。0 PROJECT count不是
  PROJECT零风险或零贡献证明，也不授权修改dedup precedence重跑。
- fresh 20-log cohort已被P3-C/P6-C消费；P15是mechanism follow-up，不是新的independent confirmation，不扫tolerance。
- hazard/clear差异必须以target-ray denominator与action share共同报告，不能只用Actor count或absolute ray count。
- 只有结果实质推翻既有claim才登记`V7-F24`；否则保留descriptive hard-evidence boundary。

## P14 result boundary — hazard preservation不是hazard-stratum visible-risk降低（2026-09-02）

- canonical：`run://worldsim_v7/WS-V7-P14-HAZARD-STRATIFIED-DEFER-01/
  20260903T031500Z__hazard-stratified-s0-r1`；523 Actors中142 hazard，0 dataset/model/fit/threshold change。
- P4 hazard selected risk=`47.37%`，高于always=`46.48%`；clear为`31.00%`，低于always=`33.86%`。总体少48个
  introduced failures中hazard仅少3、clear少45，说明aggregate改善不能外推hazard risk改善。
- P6-C hazard risk=`46.81%`且66个hazard failures不减；其34个总体reduction全部来自clear。
- apparent-safe边界：P4∧visibility观察到0 hazard failure，但只修`5/142=3.52%` hazard Actors且总gain仅`.00122m`。
- prevention：后续同时写hazard/clear coverage与risk；不得用Actor/hazard字段保留、hazard AUROC或总体risk reduction替代
  hazard-stratum visibility证据，也不得以近零hazard coverage包装safety。
- classification：既有V7明确把hazard-preserving限定为状态不变并拒绝road-safety claim，故本结果是descriptive boundary，
  不登记`V7-F24`；下一可用failure id仍为`V7-F24`。
- paper audit：exact decomposition与hazard boundary已进入main/supp；main 8 content pages、2 reference pages，相关页视觉QA
  无裁切/重叠。负边界未因聚合数字、近零hazard coverage或排版被隐藏。

