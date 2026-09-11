# 历史原始记录 079

> 冻结历史，不是当前执行规则。当前规则见 [scaling law](../../../auto-research_scaling_law.md) 与 [AGENTS](../../../AGENTS.md)。
> 原文件：`docs/RESEARCH_FAILURES.md`；迁移前提交 `abbd431c`。原有相对链接按原目录解释，证据路径原样保留。
> 按索引读取相关小节；旧哈希、过度门控和资源指令均不重新生效。

### V65-F07 — continuous learned admission 用coverage换来了新的case/route风险

- run：`run://worldsim_v65/WS-V65-P4T-LEARNED-ADMISSION-TRAIN-ONLY-01/20260827T110000Z__learned-admission-s0-r1`；
- symptom：coverage绝对提升`6.64pp`且scene support=`7/8`，但case failures `0→1`、pooled fixed-route density
  `+8.82%`、worst-tail `+7.61%`；
- root cause：连续context预测在一个night case给出`0.521873` coverage，高于其oracle-safe `0.510822`，把
  hidden-FREE conflict从`0.047814`推到`0.050764`；弱coverage回归精度不能替代显式held-out risk control；
- literature response：CRC需要独立calibration选择风险参数；SOFT top-k是P5 allocator且只在P4有效后解锁；
  GroupDRO不能修复pooled selector自身新增failure。三者都不能在观察本结果后作为无代价后处理；
- resolution：`WS-V65-H-P4T-001` rejected，关闭learned admission，不调coverage上限/loss/seed/capacity，不准备
  fresh admission cohort，不解锁P5/CRC；
- claim impact：保留V6.4 M1；没有V6.5 differentiable-admission、calibration、planning或safety claim。

下一可用编号：`V65-F08`。

### V65-F08 — nuScenes map expansion v1.2 与当前devkit schema不兼容

- scope：R3 pre-run capability audit；没有run directory、quality read、model fit或指标；
- symptom：公共盘v1.2 expansion JSON可解压，但当前`NuScenesMap`明确拒绝`version < 1.3`；
- root cause：v1.2是prediction/arcline版本，当前devkit合同要求v1.3；官方v1.3增加lidar basemap支持并移除一条坏lane；
- literature/open-source response：nuScenes官方安装文档要求把map expansion解压到`maps/expansion`并使用匹配devkit；
- resolution：不降级devkit、不绕过版本检查、不使用PNG伪造11层语义。改用公共盘官方v1.3，独立目录能力调用得到
  `8×200×200`非空语义mask；
- claim impact：无科学结论；R3 hypothesis在恢复完成后才预注册。

下一可用编号：`V65-F09`。

### V65-F09 — 官方地图上下文被模型使用，但没有改善 voxel 风险或路线决策

- run：`run://worldsim_v65/WS-V65-P1R3-MAP-CONTEXT-TRAIN-ONLY-01/20260827T114500Z__map-context-s0-r1`；
- symptom：R3 相对 q0 的 AUROC/AUPRC 为 `-0.000496/-0.002280`，fixed-route density 完全相同
  `0.00299581→0.00299581`，scene lower/equal/higher=`1/14/1`；
- mechanism audit：真实地图相对 within-unit shuffled map 的 AUROC 为 `+0.000625`，non-route risk `-0.756%`，
  因而地图通路并非完全失效，但增量太弱且没有形成路线决策变化；
- root cause：冻结 q0 已编码大部分可由局部 road semantics 解释的物理边界；给逐 voxel 判别器追加静态地图，仍未把
  监督对象对齐到 Ego 真正执行轨迹后访问的未来 world/Actor outcome；
- literature response：CoRL 2022 Task-Relevant Failure Detection 把预测误差传播到 planning cost；PRECOG 对受控
  Ego goal 条件化其他 Actor future。二者支持改变预测对象，不支持继续扩大 map residual；
- resolution：`WS-V65-H-P1R3-001` rejected；关闭 per-voxel map/context residual，不做 seed/capacity/feature/radius
  rescue。下一步建立 trajectory-level visited-state reliability，直接预测 `(scene, unit, τ)` 的未来访问走廊 outcome；
- claim impact：无 V6.5 map-conditioned method、selection、planning 或 safety claim。

下一可用编号：`V65-F10`。

### V65-F10 — trajectory-level neural head 改善MSE却破坏决策排序

- run：`run://worldsim_v65/WS-V65-P1R4-TRAJECTORY-VISITED-STATE-01/20260827T121500Z__visited-state-s0-r1`；
- symptom：V1 MSE相对Qagg降低`87.35%`，但Spearman `0.751487→0.635127`，unsafe AUROC
  `0.978261→0.909420`，lowest-risk 40%实际cost `0.038137→0.057718`（恶化`51.35%`），scene
  lower/equal/higher=`2/7/6`；
- root cause：小样本连续回归把大量低幅cost压缩得更准，但这种校准目标没有保持安全选择所需的尾部排序；
- preserved positive：改变预测对象本身成功。直接Qagg的3/3 viability gates全过：Spearman=`0.751487`、unsafe
  AUROC=`0.978261`、selected实际cost相对全体降低`62.98%`；
- resolution：拒绝learned V1 head，不改loss/seed/capacity；保留确定性、可解释的trajectory-level Qagg，后续只做
  Actor companion与fresh prediction-object transfer，不重新训练voxel residual；
- claim impact：支持legacy train-only trajectory-level mechanism，不构成formal V6.5 selection/planning/safety claim。

下一可用编号：`V65-F11`。

### V65-F11 — per-actor 强相关不能经max聚合成trajectory-level Actor reliability

- run：`run://worldsim_v65/WS-V65-P1R5-ACTOR-FALSE-SAFE-01/20260827T123100Z__actor-false-safe-s0-r1`；
- symptom：A0 trajectory max-cost forecast Spearman=`0.626087`，低于冻结`0.70`；A1更低为`0.488696`。
  `relu(A1-A0)` monitor 对`relu(target-A0)` gap 的Spearman=`-0.054402`、AUROC=`0.522222`，lowest-monitor
  40% gap反而恶化`73.40%`；
- support audit：eval 24 trajectories/302 Actor tokens，9个positive gaps、6个zero monitors，不是零标签问题；
- root cause：per-actor P2C Spearman `0.872`被trajectory-level maximum的极值误差放大；A1时间通路本身较弱，
  其正向disagreement不代表A0 false-safe；
- resolution：`WS-V65-H-P1R5-001` rejected；不调gap threshold、不训练monitor、不改max/smooth-max聚合，Actor
  companion关闭。保留R4 world-state Qagg positive，资源转向其fresh transfer；
- claim impact：无trajectory-level Actor reliability、planning或safety claim。

下一可用编号：`V65-F12`。

### V65-F12 — smooth-tail改善any-error分离，却恶化visited-error rate与实际选择代价

- run：`run://worldsim_v65/WS-V65-P1R6-SMOOTH-TAIL-VISITED-STATE-01/20260827T124500Z__smooth-tail-s0-r1`；
- symptom：固定temperature=0.10的Qsoft-tail令unsafe AUROC `0.978261→1.000000`，但Spearman
  `0.751487→0.708230`，selected-40% actual cost `0.038137→0.048535`（恶化`27.27%`），scene
  lower/equal/higher=`4/6/5`；
- root cause：upper-tail强调了单个高q0 state，适合检测trajectory内“是否存在任何风险”，却与当前监督的visited
  hidden-FREE连续比例及matched-coverage期望代价不对齐；
- literature response：MIDAM的smoothed-max/attention需要bag-level AUC训练；RAP把风险预测耦合进robust planning；
  TAT依赖多条采样轨迹和历史。三者都不是当前单轨迹固定q0的无训练温度修复；
- resolution：`WS-V65-H-P1R6-001` rejected；关闭smooth-tail，不扫temperature/target/coverage。保留R4 Qmean并
  按冻结合同完成P2V fresh transfer；未来只有在真实candidate trajectory set存在时才可新开set-level aggregation；
- claim impact：无smooth-tail、CVaR、planner或safety claim；R4 prediction-object positive不受影响。

下一可用编号：`V65-F13`。

### V65-F13 — scene-ready native launcher未先创建task parent

- attempted run：`run://worldsim_v65/WS-V65-P2V-FRESH-NATIVE-SIDECAR-01/20260827T133000Z__fresh-visited-native-scene-0001-s0-r1`；
- symptom：launcher正确捕获`scene-0001`最终preprocess marker，但native runner在创建run directory前调用
  `shutil.disk_usage(run_dir.parent)`，因task parent尚不存在抛出`FileNotFoundError`；
- exposure audit：失败发生在model/native artifact/quality读取前，未创建failed run directory，不是科学负结果；
- root cause：旧批处理流程总由外层脚本预建task目录，新scene-ready launcher遗漏了该入口前置条件；
- literature/open-source response：Python官方`Path.mkdir(parents=True, exist_ok=True)`明确用于递归创建缺失父目录；
- resolution：launcher在创建executor前预建精确task parent；不改scientific config、scene、seed、run prefix或gates，
  从已经完成的scene marker继续；
- claim impact：无科学read，不改变`WS-V65-H-P2V-001`状态。

下一可用编号：`V65-F14`。

### V65-F14 — scene-ready launcher绕过base+overlay解析器

- attempted runs：`...fresh-visited-native-scene-0001-s0-r1`与`...scene-0219-s0-r1`；
- symptom：task/run dirs创建后，V6.3 generic runner访问`config["inputs"]`抛出`KeyError`；
- exposure audit：两个目录只含空`plans/reports/logs`，未构造worker plan、未加载IR-WM、未读native或quality；
- root cause：`p2v_native_sidecars_v1.yaml`与此前成功P2配置一样是`base_config + overlay`，但新launcher直接调用
  generic runner，绕过了负责合并的`run_worldsim_v64_fresh_sidecars.py`；
- literature/open-source response：Hydra Defaults List和OmegaConf structured config均把完整schema composition置于
  runtime access之前；项目迁移采用已有wrapper作为同一composition boundary，而非复制backend字段；
- resolution：launcher切换到已验证wrapper；空失败dirs改名保留，原r1 canonical path继续，scientific合同不变；
- claim impact：无科学read，不改变P2V hypothesis或fresh exposure计数。

下一可用编号：`V65-F15`。

### V65-F15 — formal P2V evaluator错误假设q0 logits为二维

- run：`run://worldsim_v65/WS-V65-P2V-VISITED-STATE-TRANSFER-01/20260827T141500Z__fresh-visited-transfer-s0-r1`；
- symptom：第1个unit的frozen q0 forward返回1-D logits，`.squeeze(1)`抛出`IndexError: Dimension out of range`；
- exposure audit：1个formal input/target unit加载到内存；没有Qagg、target value、aggregate metric、gate或verdict输出，
  compact cache未落盘；因此不是零input exposure，但没有可用于方法选择的科学反馈；
- root cause：保存的q0 wrapper已在forward中移除singleton output dim，evaluator重复按`[B,1]`做维度特定squeeze；
- literature/open-source response：PyTorch Linear保留除最后feature维外的batch形状，`squeeze(dim)`只适用于存在的指定维；
  对已冻结scalar scorer的batch contract使用`.reshape(-1)`兼容`[B]`与`[B,1]`；
- resolution：只改tensor view，不改数值、model、target、sampling、candidate或gates；r1标记failed，r2继续同一冻结read；
- claim impact：r1不产生科学结论；fresh input已部分暴露，必须在论文/ledger披露该engineering recovery。

下一可用编号：`V65-F16`。

### V65-F16 — P3C canonical evidence命令遗漏required processed-root

- attempted entry：`WS-V65-P3C-CALIBRATION-EVIDENCE-01/20260827T150000Z__calibration-evidence-s0-r1`命令；
- symptom：Python `argparse` 报`--processed-root` required并在解析阶段退出；
- exposure audit：未创建run directory，未读processed input、evidence或quality，未产生metric/gate/verdict；
- root cause：旧query-dataset runner将processed root作为required CLI option，手动启动只传入config与run-dir；
- literature/open-source response：Python官方`argparse`文档说明`required=True`选项缺失时`parse_args()`必须先报错，
  因此该入口不构成科学读取；
- resolution：仅补充冻结的standard processed root，同一canonical run-id成功产生72 units；不改scene、reuse、
  seed、target或gate；
- claim impact：无科学暴露、无方法选择信息；P3C仍保持formal calibration read=false。

下一可用编号：`V65-F17`。

### V65-F17 — P3C formal config指向不存在的冻结q0 artifact

- run：`run://worldsim_v65/WS-V65-P3C-MONOTONE-CALIBRATION-TRANSFER-01/20260827T154500Z__calibration-transfer-s0-r1`；
- symptom：`joblib.load` 在不存在的`WS-V64-P1-BASELINE-TRANSFER-01/.../models/full_native_selective_mlp.joblib`
  抛`FileNotFoundError`；
- exposure audit：run dir仅含resolved config和running status；0 unit/native/target read，0 q0 output，0 metric/gate，
  compact cache未产生；
- root cause：P3C config手工填入的run-relative locator没有沿用已成功P2V的artifact二元组；baseline-transfer
  run不拥有该model；
- literature/open-source response：`joblib.load` 以给定filename/path还原artifact；MLflow等实验跟踪系统同样将
  artifact path定义为run root的相对路径，run与relative path必须同步；
- resolution：更正为P2V实际使用的`WS-V64-P6R-SELECTIVE-MLP-01/.../RISK_MODEL/
  full_native_selective_mlp.joblib`；仅locator改变，同一冻结model与所有科学合同不变；
- claim impact：r1无科学结论；formal calibration read仍为false，r2允许继续。

下一可用编号：`V65-F18`。

### P3C outcome note — V65-F17后窄修复r2成功

- recovery run：`run://worldsim_v65/WS-V65-P3C-MONOTONE-CALIBRATION-TRANSFER-01/20260827T155000Z__calibration-transfer-s0-r2`；
- same-contract audit：仅q0 run-relative artifact locator改变；冻结slope/bias、scene、input、sampling、target、seed和
  gates与r1完全一致；
- outcome：60 eligible units，MSE -92.80%，5-bin calibration error -88.31%，5/5 evaluable scenes改善，
  ranking/AUROC/selected set精确不变；6/6 gates；
- interpretation：V65-F17确认为纯artifact entry failure，不会遮蔽也不夸大r2的independent calibration result。

下一可用编号仍为：`V65-F18`。

