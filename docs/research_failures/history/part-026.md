# 历史原始记录 026

> 冻结历史，不是当前执行规则。当前规则见 [scaling law](../../../auto-research_scaling_law.md) 与 [AGENTS](../../../AGENTS.md)。
> 原文件：`docs/RESEARCH_FAILURES.md`；迁移前提交 `abbd431c`。原有相对链接按原目录解释，证据路径原样保留。
> 按索引读取相关小节；旧哈希、过度门控和资源指令均不重新生效。

## P14 prevention note — aggregate composite改善不得掩盖hazard-stratum代价（2026-09-02）

- P13 population risk/gain仍是hazard/clear mixture；Actor/hazard state保留100%不等于hazard stratum的visible risk降低。
- P14只用exact identity把coverage、conditional outcome和stratum share展开，不训练新head、不扫policy或utility weight。
- ICML 2022 subgroup selective-regression只提供“abstention可能放大分组退化”的方法警告；`hazardous`不是人口属性，
  本审计不作fairness、collision、planning或现实安全声明。
- CRC需要的风险控制/交换性不由本consumed AV2审计提供；禁止把identity residual为零写成distributional guarantee。
- 只有与既有claim发生实质矛盾才登记`V7-F24`，否则保留descriptive boundary，不为凑编号造failure。

## P13 prevention note — abstention不是Actor删除，conditional risk不是population risk（2026-09-02）

- P10--P12的selected risk条件在repair set上；实际policy对abstained Actors回退原query surface，不删除Actor或hazard。
- P13固定同时报告conditional failure与全523 Actors的introduced-failure mass，并以composite Chamfer gain暴露“低risk只因
  低coverage”或“修复集合实际使query变差”的情形；不新增后验utility weight/gate。
- query-only的introduced-visible failure按定义为0，只表示没有由compiler新增，不得写成query没有pre-existing violation。
- Waymo official GCS probe=`403`是license/session access condition；不登记scientific failure、不用非官方镜像替代。
- 若结果暴露新的claim错误，使用下一`V7-F23`；否则保持descriptive boundary，不为凑编号造failure。

## V7-F23 — provenance低conditional risk在defer-to-query系统中仍被query-only支配

- canonical：`run://worldsim_v7/WS-V7-P13-DEFER-TO-QUERY-COMPOSITE-01/
  20260903T023000Z__defer-to-query-s0-r1`；523 Actors/20 logs，0 dataset read/training/update。
- symptom：P11 P4∧provenance selected risk=`20.16%`看似显著优于P4，但完整policy只修124 Actors，仍给全体引入
  `25/523=4.78%` visible failures，且composite mean Chamfer gain=`-.000877m`。
- dominance：query-only保留同一523 Actors、introduced failure=`0`、gain=`0`，在risk/gain二维上严格支配provenance；
  P12 visibility-only也被P4∧visibility以更低risk和更高gain支配。
- retained frontier：P4=`28.11%/.08311m`、P6-C=`30.78%/.08895m`、P4∧visibility=`.76%/.00122m`均非支配；
  这只是consumed AV2 trade-off，不升级P6-C或恢复P12 family。
- resolution：关闭provenance policy作为系统级repair action；保留其point-witness解释。后续必须同时报告conditional risk
  与fallback-composite utility，禁止以低coverage selected risk单独包装safety。下一可用failure id=`V7-F24`。
- paper audit：main=`9 pages/1,762,749 bytes`、supplement=`8 pages/7,226,250 bytes`；V7-F20--F23 ledger与P13主文
  段均可读，无裁切/重叠，negative result未被低coverage或排版隐藏。

## P10/P11 analysis-semantics defect — corrected without data reread（2026-09-02）

- defect：字段名`nonnew_visible_violation`易歧义，但生成器定义True为safe conjunction：compiled early<=query early且
  compiled contradicted<=query contradicted。P10/P11 r1把True累计为failure。
- impact：r1 visible rates、safe AUROC/AURC、Wilson与相关gate方向无效；Chamfer、coverage、hazard、identity不受影响。
- resolution：不覆盖/删除r1，不手改summary；`d8bf0df`最小predicate fix后生成P10/P11 r2 canonical并同步全部数字。
- prevention：后续JSON字段必须同时对照生成predicate与summary prose；不新增hash/checksum/额外gate。
- paper audit：main=`9 pages/1,762,273 bytes`、supplement=`8 pages/7,225,029 bytes`；受影响页无裁切/重叠，
  negative boundaries未因排版被删除；唯一warning仍为既有Table 1 `6.03pt` overfull。

## V7-F22 — visibility-targeted source head迁移风险ranking但无法形成联合authority

- canonical：`run://worldsim_v7/WS-V7-P12-NUSCENES-VISIBILITY-AUTHORITY-01/
  20260903T004500Z__visibility-authority-s71201-r1`；source 29/56/228 Actors，AV2 523 Actors、status=`done`。
- retained positive：source-test safe-visible AUROC=`.753`、selected visible failure=`19.44%`；0 target fit下AV2 AUROC=
  `.625`，visibility-only risk/upper=`11.63/22.02%`，P4 dual=`10.26/20.98%`。
- failure：visibility-only/dual coverage=`8.22/7.46%`；Chamfer-worsening=`37.21/35.90%`；dual hazard coverage=
  `3.52%`。七门中Chamfer、10% coverage、50% hazard coverage失败。
- mechanism：单一visibility label能跨传感器迁移ranking，却把低risk建立在几何退化与危险Actor拒绝上；这是
  target-specific selective ranking，不是joint physical/safety authority。
- resolution：verdict=`rejected_visibility_targeted_source_head`；关闭seed/architecture/feature/source-coverage/
  AV2-threshold sweep，不复用consumed cohort作positive confirmation。下一可用failure id=`V7-F23`。

## WorldSim V7 P12 prevention note — dedicated visibility label仍不得使用AV2 development调模型（2026-09-02）

- P12修复的是P10 label mismatch，不是用AV2重命名P4：architecture/seed/epochs/features/source split与25% calibration
  coverage在重新生成source labels前固定。
- 已消费AV2只做一次development read；任何positive都必须另冻untouched cohort，任何negative都不能改coverage/seed/
  feature或再训candidate。
- 七门包含hazard coverage，避免像P11一样靠删除危险/稀疏Actors获得表面risk改善。失败使用`V7-F22`并关闭family。

## V7-F21 — observed provenance改善visibility但摧毁Chamfer与hazard authority

- canonical：`run://worldsim_v7/WS-V7-P11-PROVENANCE-AUTHORITY-AUDIT-01/
  20260903T011500Z__provenance-authority-s0-r2`；523 Actors/20 logs、status=`done`；r1 visible数字superseded。
- symptom：P4∧provenance coverage=`23.71%`，visible failure=`20.16%`（95% upper=`26.70%`），优于P4的
  `36.39%`；但Chamfer-worsening=`43.55%`高于P4 `14.60%`，hazard coverage仅`3.52%`。
- mechanism：completion-count unsafe-visible AUROC=`.589`，观测provenance确实提供visibility信号；问题不是方向反转，
  而是no-COMPLETE选择与repair geometry/hazard retention严重不兼容。
- retained theorem boundary：KEEP/PROJECT output各有observed termination witness仍为真，但这是point-provenance命题，
  不能推出cross-ray/future-view completeness，也不能用低hazard coverage冒充safety。
- resolution：关闭completion deletion、no-COMPLETE conjunction、completion-count threshold sweep；保持P3-D conclusion与
  P4/P3-C分权。不能用visibility point-risk改善覆盖Chamfer/hazard失败，也不能靠过滤危险Actor包装safety。

## WorldSim V7 P11 prevention note — observed-ray provenance不得偷换future-view completeness（2026-09-02）

- `COMPLETE==0`给出每个输出点的KEEP/PROJECT measurement witness，但只覆盖already observed terminations；它不先验
  排除另一个held-out ray下的early return或cross-ray contradiction。
- 五项gate在aggregate前固定，不能按completion count、coverage或hazard stratum扫阈值；P6-C不能覆盖结果。
- 若dual仍无法降低visible risk或coverage/hazard authority坍缩，登记`V7-F21`并停止provenance-gate recovery，而不是
  把低coverage写成safety proof。

## V7-F20 — nuScenes-trained repair authority对fresh AV2 visibility无置信分离

- canonical：`run://worldsim_v7/WS-V7-P10-FROZEN-PHYSICAL-AUTHORITY-AUDIT-01/
  20260903T010000Z__physical-authority-s0-r2`；523 Actors/20 logs exact frozen join、status=`done`；r1 superseded。
- symptom：P4 selected visible failure=`147/404=36.39%`，略低于always-repair `195/523=37.28%`；但one-sided 95%
  Wilson upper=`40.40%`不低于always point risk，safe-visible AUROC=`.533`，abstention只捕获`24.62%` failures。
- retained positive：selected Chamfer-worsening=`14.60%`低于always `19.50%`，Chamfer-nonworse AUROC=`.655`；说明
  P4学到了冻结target Chamfer repairability，但这个label不是bidirectional visibility consistency。
- boundary：structural hazard separation、cross-domain repair AUROC或local score stability均不能升级为physical
  certificate。P6-C在同一join也只有selected risk=`36.67%`、safe AUROC=`.537`，不能作为confidence recovery。
- resolution：保留P4 primary与P3-C visible certificate为两个独立authority objects；不在该cohort refit/recalibrate/
  scan threshold或删case。正确负边界是weak ranking/no confidence separation，不是selected point risk反向。

## WorldSim V7 P10 prevention note — validity score不得先验称为物理certificate（2026-09-02）

- P4训练目标是nuScenes repairability，并非fresh AV2的bidirectional visible-ray certificate；零结构leakage或高
  repair AUROC不能推出selected Actors没有可见三维违例。
- P10在读取aggregate前冻结exact join、两种failure、三项gate与Wilson upper；P6-C positive AV2 context不能覆盖
  P4 negative outcome，P3-C Actor tail也不能按score/threshold事后删改。
- 执行前failure delta=`none`；若strong containment任一gate失败，登记下一`V7-F20`而不是启动target recovery。

## WorldSim V7 Figure 1 teaser provenance audit（2026-09-02）

- four-quadrant teaser只使用冻结P3-B的`q01-a0` non-hazard与`q00-a0` hazard main cases；选择规则是metadata-only
  first-match，不按图像外观、深度密度、Chamfer、visibility或artifact severity择优。
- 同一valid/artifact pair保持Actor、trajectory、extent、camera和hazard不变；synthetic artifact是诊断overlay，
  不能写成photorealistic reconstruction、unseen-surface completeness或单Actor物理certificate。
- page 3 visual QA通过，原standalone camera figure已移除以避免同一证据重复展示；没有scientific failure delta。
  `V7-F18/V7-F19`仍开放，下一可用failure id保持`V7-F20`。

## WorldSim V7 C3 paper-evidence audit（2026-09-02）

- documentation gap：Method 4.3--4.4保留C3 density/joint-H对象，但此前主文只报告P5/P9接口审计，缺少冻结
  P182/P183/P199/P201 proper-score与calibration evidence；这不是新的scientific failure。
- resolution：只恢复V6.7 canonical结果到macro-driven主表，不重训、不重读target、不改变P4/P346或任何gate。
- retained boundary：P346 reused-P201 q90 unsafe=`9.09%`，但source held-out-H=`26.71%`；禁止formal multicalibration、
  cross-horizon stability、V7 repair-effect或AV2-transfer claim。
- layout：main pages 1--8包含全部content，page 9只含references，符合当前official CVPR references exclusion；不缩小
  bibliography字号或破坏模板。Pages 6--9 visual QA通过，既有Table 1 `6.03pt` warning无裁切。
- failure delta=`none`；V6.7既有失败账本不重编号，V7下一可用仍为`V7-F20`。

## WorldSim V7 latest paper-integration audit（2026-09-02）

- fresh P3-C/P6-C结果与`V7-F18/V7-F19`已进入main、supplement、主表、结果宏和contribution map；没有因外域正结果
  隐去fresh-nuScenes rejection，P4保持primary。
- final compile main=`8 pages/1,169,034 bytes`、supplement=`7 pages/7,222,572 bytes`；main pages 4--8与supplement
  pages 2--4逐页QA无裁切/重叠/断表，唯一既有`6.03pt` overfull视觉无害。
- 此审计未发现新failure；`V7-F18/V7-F19`继续开放为论文边界，下一可用failure id=`V7-F20`。

## WorldSim V7 latest failures（2026-09-02）

### P3-C fresh confirmation note — V7-F18独立复现，不新增failure

- run：`run://worldsim_v7/WS-V7-P3C-AV2-VISIBILITY-CERTIFICATE-FRESH-01/
  20260902T231500Z__fresh-visibility-s0-r1`；20 previously unused AV2 logs、523 Actors、exact-once、status=`done`。
- aggregate fresh：query→compiled hit=`.497233→.689919`、early=`.034478→.029314`、visible precision=
  `.995141→.996813`、F-score=`.663127→.815447`、surface contradictions=`912→509`；与consumed方向完全一致。
- retained tail：nonnew-visible=`62.72%`、exact-zero=`2.29%`、Chamfer-worsened=`19.50%`；102个worsened Actors中
  F-score noninferior仅`16.67%`。
- resolution：fresh read提高`V7-F18`外部可信度但不消除其Actor-level failure；继续拒绝universal certificate、
  post-hoc completion deletion、tolerance/operator scan。无新failure，下一可用=`V7-F20`。

### V7-F19 — sparsity-consistent selector在两个fresh域发生ranking reversal

- canonical external run：`run://worldsim_v7/WS-V7-P6C-SPARSITY-CONSISTENT-SELECTOR-01/
  20260902T173000Z__sparsity-consistent-s70602-r1`；20 fresh AV2 logs、523 Actors、exact-once、status=`done`。
- fresh AV2 symptom：P6-C/P4 repair AUROC=`.676168/.654837`，P6-C改善`2.13pp`；coverage=`.839388/.772467`、
  selective Chamfer=`.178291/.184133m`，外域预注册 gates通过。
- retained cost：P6-C/P4 false-repair=`.124283/.112811`，P6-C高`1.15pp`；原gate只要求低于always-repair，不能
  把“gate pass”重写成所有风险维度支配。
- independent contradiction：P8-A fresh nuScenes P6-C/P4 AUROC=`.747253/.782280`（`-3.50pp`），AURC=
  `.126429/.105633`（higher worse），已由`V7-F15`拒绝。
- boundary：source-only sparsity consistency改善机会扰动敏感度并能在fresh AV2提升ranking，但没有形成跨fresh域
  单调generalization；不得称domain invariant、universal或formal risk guarantee。
- resolution：不按AV2 positive result推翻P8-A，不refit/recalibrate/换threshold/cohort；P4保持paper primary，P6-C作为
  mixed-domain negative/positive ablation完整报告。下一可用failure id=`V7-F20`。

### P3-C fresh orchestration prevention note — P6-C后串行启动，不抢GPU或重复读取

- watcher不检查 P6-C scientific verdict，只要求 canonical summary/status正常收尾；正/负结果均进入同一个预冻结 P3-C
  fresh visibility read，避免结果条件分支。
- 20/20 marker、ALL_COMPLETE与P6-C evaluator退出后才exec；dedicated flock和existing-run-path stop防止双实例。
- watcher不下载、不训练、不改 config/tolerance/operator/cohort；P3-C runner仍是唯一指标读取者。
- commit=`b775807` 后 watcher PID=`33968` 正常取得lock并在16/20状态等待；未启动 evaluator或第二下载器。
- 当前为launch orchestration；P6-C结果已登记`V7-F19`，下一可用为 `V7-F20`。

### Paper layout note — P3-C/P3-D integration visual QA通过，无新failure

- main保持8页，新增visibility段后 pages 4--8无 clipping/overlap/orphan float；唯一 warning仍为既有 Table 1
  `6.03pt` overfull且视觉未裁切。
- supplement首次编译为8页，P7-C双栏float独占一页；只把同一图移到one-column failure ledger后，不改图、表、数字或
  prose，最终7页且无 warning。
- V7-F18 row、visibility table、completion gain--tail与最后references页均可读；没有通过删 negative boundary压页。
- paper-only integration不创建failure id；下一可用仍为 `V7-F19`。

