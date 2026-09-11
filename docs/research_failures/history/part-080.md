# 历史原始记录 080

> 冻结历史，不是当前执行规则。当前规则见 [scaling law](../../../auto-research_scaling_law.md) 与 [AGENTS](../../../AGENTS.md)。
> 原文件：`docs/RESEARCH_FAILURES.md`；迁移前提交 `abbd431c`。原有相对链接按原目录解释，证据路径原样保留。
> 按索引读取相关小节；旧哈希、过度门控和资源指令均不重新生效。

### P10X stop-rule note — combined confirmation不设恢复cohort

- candidate已由P2V/P3C/P10V冻结；P10X不产生新模型、参数、threshold或action generator；
- confirmation只有6个与已支持分量直接对应的核心gates，scene/case/subgroup其余数值仅报告；
- 若任一核心gate失败，关闭combined candidate，不换scene、不创建第二confirmation cohort、不以critic补救；
- 下一可用failure id保持`V65-F18`。

下一可用编号仍为：`V65-F18`。

### P10V outcome note — direct action ranking成功，未触发critic recovery

- run：`run://worldsim_v65/WS-V65-P10V-ACTION-VISITED-STATE-TRANSFER-01/20260828T003000Z__action-transfer-s0-r1`；
- outcome：Spearman `0.7402`、unsafe AUROC `0.8588`、pairwise concordance `0.7325`、selected cost -33.26%、
  scene `6/0/0`，6/6 gates；
- prevention result：V64-F28的learned collision critic没有被重开；无head、threshold、lattice或label recovery；
- exclusions：51/864 actions由事前16-point footprint rule排除，72 stop rows由事前合同排除，非失败或事后门控；
- claim impact：支持fixed-action visited-state ranking，但不恢复collision/planning/safety authority。

下一可用编号仍为：`V65-F18`。

### P10V input-pipeline outcome note — 无新failure

- targeted shards 2/6/9找到冻结cohort全部10,709 members，未触发same-cohort full-scan fallback；
- archive extractor使用已有partial-file atomic rename，scene-ready feeder不需哈希/校验和/指纹或内容验证；
- 6/6 preprocess、native及72-unit evidence全部完成，role overlap=0，过程未读action quality；
- P10V formal read前的下一可用failure id保持`V65-F18`。

下一可用编号仍为：`V65-F18`。

### P10V prevention note — 不重复V64-F28 collision-critic路径

- V6.4的10-feature linear critic在独立cohort发生unsafe ranking退化，已终止为`V64-F28`；
- P10V只复用质量读取前已冻结的13-action generator，不复用critic、threshold、collision label或policy gate；
- 新对象是trajectory-visited hidden-FREE rate排序，不将world-state reliability冒充collision/safety authority；
- 若直接Qmean无action ranking，立即关闭action family，不训练critic补救。

下一可用编号仍为：`V65-F18`。

### V65-F18 — P10X scene-ready feeder首次入口缺少仓库级Python import path

- attempted entry：从仓库根目录直接执行`python scripts/launch_worldsim_v65_scene_ready_preprocess.py --help`；
- symptom：入口在导入`from scripts.prepare_dr_v2_drivestudio_scene import ...`时抛
  `ModuleNotFoundError: No module named 'scripts'`；
- exposure audit：发生在参数解析、run创建、scene/member/processed/native/evidence/quality读取之前；正在运行的三个archive
  scan children未停止，formal confirmation read仍为false；
- root cause：Python直接执行文件时将脚本目录而非仓库根目录置于`sys.path[0]`；仓库内绝对`from scripts...`
  import需要显式仓库搜索路径；
- literature/open-source response：Python官方`sys.path`与命令行文档说明`python script.py`首先加入脚本所在目录，
  `PYTHONPATH`用于扩展module search path；
- resolution：不改代码和科学合同，只对该进程设置`PYTHONPATH=.`；scene-ready preprocess与native watcher均已正常运行；
- claim impact：纯入口环境失败，0科学暴露，不触发第二cohort或任何gate/threshold修改。

下一可用编号：`V65-F19`。

### P10X input-pipeline outcome note — V65-F18后无新failure

- targeted shards 3/7/8找到冻结cohort全部10,718 members，未触发same-cohort full-scan fallback；
- `PYTHONPATH=.`恢复后6/6 preprocess、native及72-unit evidence全部完成，role overlap=0；
- partial evidence的48 units与后两场preprocess/native重叠，canonical只补24 units；
- aggregate仅建立unit symlink，未复制数组、未重复native inference、未增加哈希/校验和/指纹；
- 过程未读combined quality，下一可用failure id保持`V65-F19`。

下一可用编号仍为：`V65-F19`。

### V65-F19 — P10X one-shot combined confirmation未达到direct action-selection benefit

- run：`run://worldsim_v65/WS-V65-P10X-COMBINED-CONFIRMATION-01/20260828T013000Z__combined-confirmation-s0-r1`；
- symptom：route ranking、frozen-map MSE、action ranking、unsafe AUROC和pairwise共5门通过，但lowest-Qmean 25%
  action selection将actual cost从`0.120215`降至`0.100520`，只降低`16.38%`，未达冻结`25%`；
- support retained：route Spearman=`0.6098`、frozen calibration MSE -81.39%、action Spearman=`0.7729`、unsafe
  AUROC=`0.9727`、pairwise=`0.6557`；这些支持reliability evaluation，不足以越权为direct action authority；
- heterogeneity：action selection在5 scenes改善、1 scene (`0817`)退化；`0718`虽无eligible nominal route，但其action
  selection改善，故失败不能归因于单一footprint exclusion，也不允许删scene；
- literature/open-source response：AISTATS 2025的excess-risk分解指出recalibration可消除miscalibration regret，仍可能
  留下需要更强post-training的grouping loss；UAI 2023也报告recalibration通常不直接改善downstream regret。本结果与该
  边界一致，因此不事后改metric或把5/6门包装为authority成功；
- resolution：按冻结stop rule关闭combined/direct-action candidate；不换scene、不降gate、不做第二confirmation、不训练
  critic或重开V64-F28；保留given-`tau` visited-state ranking/calibration这一窄结论；
- claim impact：V6.5不获得planner/policy/closed-loop/collision/safety authority；P10X终态为negative combined result。

下一可用编号：`V65-F20`。

### P10X terminal note — 不以新cohort修复V65-F19

- 这次是冻结candidate的唯一formal confirmation read，cache reused=false，未发生入口/资源错误；
- one-shot verdict必须由6门AND rule决定，selected-cost failure不可由其余5门覆盖；
- 任何local/multicalibration或decision-focused head都将是未来版本的新方法、新protocol和新分支，不能作为V6.5恢复；
- V6.5用于arXiv的准确贡献边界是task-conditioned visited-state reliability evaluator，而非action authority compiler。

下一可用编号仍为：`V65-F20`。

### ArXiv report handoff audit note — 无新failure

- documentation-only阶段只检查P2V/P3C/P10V/P10X canonical summary/status存在且JSON可读，并确认P10V action rows=813；
- 首次审计在非登录shell调用裸`python`时命中既有PATH约束；使用项目环境绝对路径后同一read-only检查通过；
- 未创建run、未重新计算metric、未读取新quality、未改变任何verdict，也未增加hash/checksum/fingerprint；
- 技术报告、证据索引、主计划终态与`FAILURE_ANALYSIS.md`更新不构成新科学trial。

下一可用编号仍为：`V65-F20`。
