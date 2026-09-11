# 历史原始记录 039

> 冻结历史，不是当前执行规则。当前规则见 [scaling law](../../../auto-research_scaling_law.md) 与 [AGENTS](../../../AGENTS.md)。
> 原文件：`docs/RESEARCH_FAILURES.md`；迁移前提交 `abbd431c`。原有相对链接按原目录解释，证据路径原样保留。
> 按索引读取相关小节；旧哈希、过度门控和资源指令均不重新生效。

### V67-F51 — P65 uncertainty-native quantile Actor reliability失败

- 分类：`prediction/ordered-quantile-actor-reliability`；状态：`closed_negative_after_single_trial`。
- 方法：同一query representation输出q10/q50/q90，用pinball loss并结构保证顺序；q50承担P60 reliability read，
  q10-q90提供empirical central interval。
- Protocol：train scene `%5!=0`、H `.8/1.5s`；remainder0 scene复用但confirmation H改为未见2.5s。
- 判定：P60三门+80% interval coverage `[.75,.85]`；不扫quantile/coverage/loss/horizon，不声称conformal coverage。
  结果：q50/Actor-only Spearman=`.717354/.800656`、MAE=`.151378/.153607`、AUROC
  `=.962478/.932027`；coverage=`.672228`，2/4 gates。
- 结论：CQR文献明确neural quantile可undercover且finite coverage需独立calibration；当前没有冻结calibration split，
  不做conformal或quantile sweep。下一编号=`V67-F52`。

### V67-F52 — P66 plain-Huber H2.5 isolation支持

- 分类：`ablation/long-horizon-vs-quantile-loss`；状态：`supported_isolation`。
- 方法：P60 plain-Huber exact；P65 scene split/H2.5 exact；唯一变化是去掉quantile head/pinball。
- 判定：P60三门exact。若通过，P65主要归因于uncertainty head/loss；若失败，plain方法边界为H2附近。
- 结果：query/Actor-only Spearman=`.759251/.773758`、MAE=`.149455/.209832`（降低`28.77%`）、
  AUROC=`.945655/.939205`；3/3 gates。
- 结论：H2.5 plain method成立；P65负结果定位于quantile head/loss与未校准interval。下一编号=`V67-F53`。

### V67-F53 — P67 direct binary Actor reliability失败

- 分类：`prediction/direct-binary-actor-reliability`；状态：`closed_negative_after_single_trial`。
- 对象：冻结定义`raw actor state error>1m AND predicted tau separation<=6m`，直接输出unreliable probability。
- 同read：binary query、continuous query、binary Actor-only；正类权重=train negative/positive exact ratio。
- 结果：AUROC=`.939174/.940707/.911397`；binary相对continuous=`-.001533`、相对Actor-only=`+.027777`；2/3 gates。
- 结论：不扫threshold/loss/class weight；direct binary关闭，continuous expected-error score保留。下一编号=`V67-F54`。

### V67-F54 — P68/P69 fixed-coverage selective Actor reliability支持

- 分类：`consumer/selective-regression-triage`；状态：`supported_two_splits`。
- 调研迁移：SelectiveNet（ICML 2019）与selective regression（ICML 2022）将abstention表述为risk-coverage tradeoff，
  并警告group风险可能随coverage下降而恶化；因此固定per-scene 50%，不做global threshold sweep。
- P68：cost -78.93%、unreliable prevalence -88.58%、32/32 scenes nonincreasing，continuous cost低于binary。
- P69：cost -68.80%、unreliable prevalence -87.83%、23/23 scenes nonincreasing，continuous cost低于Actor-only continuous。
- 结论：只支持reliability triage/abstention；不删除Actor、不改geometry、不称authority/planning/safety。下一编号=`V67-F55`。

### V67-F55 — P70 fresh-population绝对尺度迁移失败，selective ordering保留

- canonical：`run://worldsim_v67/WS-V67-P70-FRESH-ACTOR-RELIABILITY-01/20260829T153500Z__fresh-actor-reliability-s0-r2`；
  worldsim-v5六个fresh scene、H2.5共5,471 rows，source-overlap 276/756明确排除。
- 症状：query/Actor-only Spearman=`.808522/.777160`，但MAE=`.186774/.186914`，改善仅`.0753%`而非冻结10%；
  4门中3门通过，全面fresh transfer拒绝。AUROC=`.952704/.967216`也不支持全面优于Actor-only。
- 保留信号：50% selective cost -89.78%、unreliable prevalence -99.40%、6/6 scenes nonincreasing，query selection
  cost仍比Actor-only低`.001823`；因此不是ordering/triage失败，而是absolute scale未跨root获得相对优势。
- 启动故障：r1在数据读取前因非交互SSH shell缺项目`PYTHONPATH`退出；r2只补环境变量，未改科学合同。
- 文献迁移：ICLR 2025 regression TTA指出naive全特征alignment可能恶化回归；ICCV 2021 calibration under covariate
  shift支持以独立calibration domains学习迁移。P71只用重叠scene 276/756训练冻结backbone residual adapter，六个fresh
  scenes继续隔离；不扫adapter/epoch/lr。下一编号=`V67-F56`。

### V67-F56 — P71 target-only hidden-feature residual calibration负迁移

- canonical：`run://worldsim_v67/WS-V67-P71-RESIDUAL-ACTOR-CALIBRATION-01/20260829T160000Z__residual-actor-calibration-s0-r1`；
  276/756共1,875 H2.5 rows训练等容量query/Actor-only linear residual adapters。
- 症状：fresh query MAE `.186774→.266719`（+42.80%）、Spearman `.808522→.554614`；adapted Actor-only
  MAE `.177064`，query相对恶化50.63%。只有selective cost gate通过，1/4拒绝。
- 原因定位：calibration-domain supervised loss下降不代表跨scene transfer；hidden embedding上的instance-dependent residual拥有
  足够自由度重排分数，276/756不能代表其余六scene。
- 保留边界：adapted score的50% triage仍cost -70.99%、unreliable prevalence -84.89%、6/6 scenes不增，但显著弱于frozen P70。
- 文献迁移：CVPR 2021区分accuracy/ranking-preserving calibrator，UAI 2025强调monotonic calibration保序；P72只训练
  positive-slope affine，base score完全冻结。不扫P71 lr/epoch/width。下一编号=`V67-F57`。

### V67-F57 — P72 monotone target calibration未恢复query相对MAE优势

- canonical：`run://worldsim_v67/WS-V67-P72-MONOTONE-ACTOR-CALIBRATION-01/20260829T163000Z__monotone-actor-calibration-s0-r1`；
  query/Actor-only各只训练positive-slope affine，base score冻结。
- 结果：query scale/bias=`.906696/.004084`，MAE `.186774→.185193`（仅+.846%）；Actor-only scale/bias
  `=1.649869/.023415`，MAE `.159585`，所以query相对差16.05%。1/3 gates拒绝。
- 保留：query Spearman `.808522`、AUROC `.952704`和50% selection完全保持，cost -89.78%、unreliable prevalence -99.40%。
- 结论：关闭target calibration family，不扫非线性map/权重/epoch；P70跨root贡献限于ordering/triage。
- 文献迁移：multi-horizon direct forecasting与long/short-term trajectory modeling支持扩大训练horizon support；P73在source
  H `.8/1.5/2.5`联合训练并评估H3，而非继续处理同一H2.5尺度。下一编号=`V67-F58`。

### V67-F58 — P73 multi-horizon训练改善frozen误差但未建立query pointwise优势

- canonical：`run://worldsim_v67/WS-V67-P73-MULTI-HORIZON-ACTOR-RELIABILITY-01/20260829T170000Z__multi-horizon-actor-s0-r1`；
  299,103 base + 141,295 H2.5 rows，GPU warmup/joint训练与两段materialization重叠。
- 结果：H3 query MAE `.228784`，比frozen P66 `.289723`改善21.03%；但Actor-only `.229099`，query只低`.137%`
  而非10%。Spearman query/Actor=`.778665/.746849`，AUROC=`.974031/.966363`，2/3 gates拒绝。
- 保留：50% query selection cost `.034406`，优于Actor `.038821`和frozen `.041978`，相对all降低91.09%；
  unreliable prevalence降低97.76%。
- 结论：不降pointwise gate、不扫horizon mix/epochs；multi-horizon pointwise claim拒绝，但selective ordering进一步增强。
- 文献迁移：SelectiveNet/Selective Regression/Regression Deferral支持直接训练accept/reject函数。P74用scene-horizon内
  最低cost半集作为固定coverage admission监督，在H3.5比较query、Actor-only与P73 continuous。下一编号=`V67-F59`。

### V67-F59 — P74 direct low-risk-half admission弱于continuous expected-cost selector

- canonical：`run://worldsim_v67/WS-V67-P74-FIXED-COVERAGE-ACTOR-ADMISSION-01/20260829T173000Z__fixed-coverage-admission-s0-r1`；
  440,398 source rows训练，H3.5六scene 5,049 rows。
- 结果：query/Actor admission AUROC=`.835296/.817442`；50% selected cost query/Actor=`.067374/.067681`，
  query只低`.45%`，未达5%；P73 continuous=`.047491`，direct admission高41.87%。1/3 gates拒绝。
- 保留：query相对all `.524624`仍降低cost 87.16%，unreliable prevalence `.074668→.007927`，6/6 scenes不增；
  但这些不能覆盖对更强continuous baseline的失败。
- 结论：关闭binary admission family，不扫label quantile/BCE/coverage；保留multi-horizon continuous expected-cost selector。
- 数据对策：按DriveStudio官方modular process keys，只抽取V5 validation role的LIDAR并生成`lidar/calib/objects` 10Hz数据；
  P75在新8-scene cohort H3.5做一次fixed-coverage read，准备IO与GPU训练重叠。下一编号=`V67-F60`。

### V67-F60 — P75 validation LIDAR generic shard routing造成无效全包扫描

- 分类：`engineering/data-routing`；状态：`resolved_by_exact_scene_shards_and_scene_ready_preprocess`。
- attempted run：`WS-V67-P75-FRESH-ACTOR-COHORT-PREP-01/20260829T180000Z__validation-actor-prep-s0-r1`；
  generic extractor在member index尚未建立时把全部3,128 candidates交给10个约30GB gzip shards，2 workers长时间只扫描
  01/02；0/8 processed scenes ready，未读H3.5 target/metric。
- 根因：该通用逻辑适合未知member位置，但本cohort的scene顺序与官方trainval shard分段已经由metadata确定；继续全包扫描只浪费IO，
  不增加科学信息。P75 GPU四horizon训练已完成且主进程保持，不重训。
- 外部迁移：DriveStudio官方NuScenes流程支持按`process_keys`只生成`lidar/calib/objects`；恢复进一步按scene metadata冻结
  `0129/0170→02,0364/0384→04,0640→06,0977/0997→09,1053→10`，五包并发且每包只接所属members。
- 边界：r1保留为工程失败；r2不换scene/horizon/model/gate，不加入图像、mask、quality、hash/checksum/fingerprint。
  下一编号=`V67-F61`。

P76 dense percentile-rank与P77 group-balanced ListNet均只读source rows，并在P75 fresh validation rows产生前冻结；它们用于
填充P75输入IO期间的GPU空档，不改变P75首次fresh read。当前无新增失败，下一编号仍为`V67-F61`。

P78 fixed-coverage boundary-pair目标也在同一blind时段预注册；只有P77结束且P75数据仍未ready才执行。它不复开P74 BCE，
也不更改P75 cohort/gates。当前无新增失败，下一编号仍为`V67-F61`。

P79将horizon作为domain做固定risk-variance regularization，只在P78结束且fresh rows仍未ready时运行；明确不包装为完整
Fishr。当前无新增失败，下一编号仍为`V67-F61`。

P80预注册为一次linear horizon feature modulation，不做高阶interaction/scale sweep；仍只填充fresh IO等待期。
当前无新增失败，下一编号仍为`V67-F61`。

P75 model r1的2,400s readiness上限将早于已观测tar/preprocess wall；r2在fresh read前仅复用r1 source-H3 cache并提前
持久化模型，scientific合同不变。r1若按预期timeout则登记`V67-F61`，不得把它计作算法失败。

恢复结果：五包在`1409.8--1683.5s`完成，3,128 members全找到；scene-ready手工feeder先完成8个最小processed scenes，
父协调器恢复后8/8 `reused_ready`，prep wall=`1982.66s`。P75 r1在旧timeout前完成，故没有timeout failure；预备r2在
joint epoch1001终止且0 fresh read。F60关闭，下一编号仍为`V67-F61`。

### V67-F61 — P75 fresh H3.5未建立相对Actor-only的fixed-coverage mean-cost优势

- canonical：`run://worldsim_v67/WS-V67-P75-FRESH-VALIDATION-MULTI-HORIZON-01/20260829T180000Z__fresh-validation-actor-s0-r1`；
  8 scenes、8,000 rows、280 unreliable events，首次fresh read。
- 结果：query pointwise MAE比Actor低13.12%，AUROC高`.003494`；但50% selected mean cost `.038723`高于Actor
  `.037013`（退化4.62%），且只比P73 `.039619`低2.26%而非5%。1/3 selector gates，严格拒绝。
- 保留：相对all cost降低84.04%，8/8 scenes nonincreasing；query selected unreliable prevalence `.00175`低于
  Actor/P73 `.0025`。后者只生成新的独立可靠性hypothesis，不覆盖mean-cost失败。
- 防重复：不降5%门、不在该cohort调coverage/score fusion/threshold；全面selector claim关闭。下一编号=`V67-F62`。

### V67-F62 — P76--P80 blind source-ranking恢复均未超过P75

- 所有模型在P75 fresh rows/metrics出现前冻结，随后同cohort development read；不是五次独立confirmation。
- selected cost：P75/P76/P77/P78/P79/P80=`.038723/.043334/.053605/.052137/.049604/.045743`；五个恢复均更差。
- P76/P80相对各自Actor-only仍改善8.31%/13.83%，说明τ特征并非完全无效；但dense percentile rank、ListNet、
  boundary pairs、horizon risk variance与linear horizon-FiLM均未建立相对当前best selector的增益。
- 防重复：不扫rank temperature、pairing、V-REx weight、FiLM order、epochs或coverage；该source-ranking恢复族关闭。
  下一编号=`V67-F63`。

P81--P83已在新test-role sensor/target read前冻结：P81把primary endpoint改为P75留下的fixed-coverage unreliable-event
prevalence，P82/P83分别训练全source pairwise AUC与horizon-balanced pairwise AUC；当前tar IO与GPU training并行，尚无
新增scientific/engineering failure。P75 mean-cost负结论保持不变，下一编号仍为`V67-F63`。

P84在同一target-read前处理一个潜在对象错配：全row event metric可能被`separation>6m`的未访问状态主导，因此新增
visited-region-only denominator与去τ重复的Actor failure因子化模型。它不修改P81 protocol/verdict，当前仍无新增失败；
下一编号仍为`V67-F63`。

P85进一步把visited Actor rows按`scene/anchor/τ`聚合为“任一访问状态不可靠”的trajectory-level对象；显式
`anchor_frame`只是group字段，不是hash/checksum/fingerprint。协议在target read前冻结，当前无新增失败，下一编号仍为
`V67-F63`。

P86进一步冻结direct trajectory set-summary model；source anchor rows物化与P84 GPU训练并行。它不读取P81 target来选
aggregation/loss/radius，当前无新增失败，下一编号仍为`V67-F63`。

### V67-F63 — P86 r1 trajectory aggregation逐group全表扫描

- 分类：`engineering/cpu-group-aggregation-complexity`；状态：`resolved_before_training_or_test_read`。
- 症状：576,032 source rows使用`for group -> flatnonzero(inverse==group)`，复杂度`O(N×G)`；r1只写resolved，GPU训练、
  model artifact与test target read均为0。
- 调研/修复：按NumPy官方`unique(return_inverse)`与stable `argsort`语义，一次排序后遍历连续group slices；baseline
  group-max也用相同线性遍历。未改source、features、aggregation定义、model、loss、epochs、gates或fresh cohort。
- r2已进入GPU训练；r1不计scientific trial。下一编号=`V67-F64`。

