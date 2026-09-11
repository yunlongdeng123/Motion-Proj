# 历史原始记录 041

> 冻结历史，不是当前执行规则。当前规则见 [scaling law](../../../auto-research_scaling_law.md) 与 [AGENTS](../../../AGENTS.md)。
> 原文件：`docs/RESEARCH_FAILURES.md`；迁移前提交 `abbd431c`。原有相对链接按原目录解释，证据路径原样保留。
> 按索引读取相关小节；旧哈希、过度门控和资源指令均不重新生效。

### V67-F72 — P106 r1漏做occupancy-flip target adapter而误训Actor endpoint error

- 分类：`engineering/target-adapter`；状态：`resolved_by_exact_adapter_restore`；
- symptom：r1 development出现1,636/1,791 events、query/Actor/P75=`828/829/753`，与冻结occupancy-flip cohort的
  95 events明显不一致；检查表明P104 raw rows保留原`raw_actor_state_error_m`，P106 prep未执行P95的field rebinding；
- exposure：r1完成了错误target的训练/评估，但没有读取P96/P103，也未改变任何冻结checkpoint；该run整体作废，不作为
  scientific data-scale trial或negative result；
- literature response：CVPR 2023 IMPLICITO对query-point occupancy ground truth直接使用BCE；P106必须监督明确的
  predicted-vs-observed occupancy decision flip，而非Actor位移误差；
- resolution：r2 prep显式保留`actor_position_error_m`后，将`raw_actor_state_error_m`和`target_cost`设置为
  `occupancy_decision_flip.float`，新prep/model run-id重跑；source volume/model/loss/epochs/development/gates不变；
- claim impact：纯adapter failure，不计P106科学试验，不影响P102/P103/P96。

下一可用编号：`V67-F73`。

### V67-F73 — all-source scale-up产生negative transfer并劣于P75

- 分类：`scientific/source-domain-transfer`；状态：`closed_negative_no_target_weighting`；
- canonical：`run://worldsim_v67/WS-V67-P106-ALL-SOURCE-HIERARCHICAL-01/20260830T042000Z__all-source-hierarchical-s0-r2`；
- symptom：正确95-event development上fixed50 query/Actor/P75=`16/25/13`；query虽相对Actor减少36%、absolute减少
  66.30%，但比P75多3，prevalence ratio=`1.2308`，2/4 gates；AUROC=`.76717/.69915`；
- interpretation：原4/5 source训练的P102为`4/27/13`；一次性加入23 scenes/114,575 rows后，source trajectories
  79,478→97,441却外推退化，属于可观测negative transfer，不是容量或训练不足证据；
- literature response：CVPR 2019 Characterizing and Avoiding Negative Transfer指出弱相关source会伤害target，并以
  source filtering/weighting缓解；但这里唯一development已消费、P96未读，事后按development筛scene或学weights会污染结论；
- resolution：关闭all-source scaling，不挑remainder、不扫source subset/epoch/normalization、不引入target-dependent
  domain weights；保留P102原4/5 source模型为best与P103唯一frozen checkpoint；
- claim impact：不支持“更多source data单调改善reliability”，不影响P102 development或P96/P103 confirmation。

下一可用编号：`V67-F74`。

### V67-F74 — P95 occupancy-flip primary无法独立超过Actor-only

- 分类：`scientific/independent-generalization`；状态：`closed_negative_one_shot_primary`；
- canonical：`run://worldsim_v67/WS-V67-P96-OCCUPANCY-FLIP-CONFIRMATION-01/20260830T004000Z__occupancy-flip-confirmation-s0-r1`；
- symptom：fresh 1,720 trajectories/36 flips上fixed50 query/Actor/P75=`8/5/12`；query absolute reduction=55.50%且
  优于P75，但比Actor多60%，query/Actor AUROC=`.65542/.71181`，relative gate失败，3/4 gates；
- subtype：query=`7 false-safe+1 false-alarm`，Actor false-safe=0，P75 false-safe=10；不能把相对P75改善写成
  task-conditioned或safety gain，因为Actor-only更优；
- root cause：P95 development的τ features增益未跨cohort稳定；Actor dynamics本身携带主要可迁移risk，end-to-end query
  classifier把clearance/context相关性学成cohort-specific shortcut；
- resolution：按one-shot rule关闭P95 primary，不换scene、不降query-over-Actor gate、不用P103覆盖primary、不在同read
  选择P104--P106；
- claim impact：不支持independent trajectory-conditioned occupancy-decision reliability、collision或safety claim。

下一可用编号：`V67-F75`。

### V67-F75 — P102 hierarchical temporal secondary仍无法独立超过Actor-only

- 分类：`scientific/representation-generalization`；状态：`closed_negative_prospective_secondary`；
- canonical：`run://worldsim_v67/WS-V67-P103-HIERARCHICAL-CONFIRMATION-01/20260830T024000Z__hierarchical-confirmation-s0-r1`；
- symptom：同一fresh cohort fixed50 query/Actor/P75=`9/7/12`，absolute reduction=49.94%、query-vs-Actor=-28.57%，
  AUROC=`.74385/.67973`，3/4 gates；query false-safe/false-alarm=`7/2`；
- interpretation：hierarchical temporal tokens把development从P95的7改善到4，但independent仍落后Actor-only；这排除
  “只需更强query temporal encoder”作为P95失败的充分解释；
- literature response：CoRL MultiPath把intent/control uncertainty分层并支持closed-form space-time collision queries；
  下一对象应先估Actor uncertainty distribution，再解析投影到τ clearance，而非继续端到端query分类器；
- resolution：关闭P102/P103 representation family，不扫seed/width/pooling/auxiliary/data scale；若继续P107，P81/P96
  只作consumed development，新confirmation必须另冻target-unread cohort；
- claim impact：无independent hierarchical task-conditioned claim，不影响P81 all-row triage窄结论。

下一可用编号：`V67-F76`。

### P107 launch note — 不再训练end-to-end query classifier

- P95/P102在development强、P96/P103独立relative失败，Actor-only在fresh cohort更稳；因此不把卡点解释为继续扩模型的理由；
- literature response：MultiPath将Actor intent/control uncertainty与candidate trajectory的space-time collision query分开，
  P107迁移为Actor q90 time-local error tube加固定clearance解析投影；网络完全不读candidate τ；
- prevention：P81/P96只作已消费development，不换gate包装独立成功；q90、`.05m` floor、max聚合与fixed50一次冻结，
  禁止quantile/floor/aggregation/seed/width sweep；若失败登记`V67-F76`并换研究对象；
- execution：source materialization一交付即启动GPU训练，同时继续两个development cohort的CPU/IO；未增加hash、
  checksum、fingerprint或smoke/regression matrix，单3090资源足够。

下一可用编号仍为：`V67-F76`。

### V67-F76 — P107首次model launcher在后台分组后丢失仓库工作目录

- 分类：`engineering/process-launch`；状态：`resolved_before_run_creation`；
- symptom：同一SSH命令中prep后使用`&`，shell把带`cd`的前段置入后台子shell，model随后从`/root`解析相对
  `scripts/run_worldsim_v67_p107_actor_uncertainty_tube.py`并立即报文件不存在；
- exposure：0 model run directory、0 source/development load、0 optimizer step、0新target read；prep进程正常继续；
- resolution：不改代码/数据/quantile/floor/aggregation/steps/coverage，只以绝对script、config与PYTHONPATH启动原定
  canonical r1；model等待source artifact后自动接管GPU；
- prevention：后续独立后台作业均显式使用绝对入口，避免依赖同一复合shell中的`cd`作用域；不增加校验或测试矩阵；
- claim impact：纯launcher failure，不计科学trial，不改变P107 verdict边界。

下一可用编号：`V67-F77`。

### V67-F77 — P107 r1在source压缩写完前读取final NPZ

- 分类：`engineering/artifact-delivery-race`；状态：`resolved_before_first_optimizer_step`；
- symptom：producer创建`SOURCE_ACTOR_UNCERTAINTY_ROWS.npz`后仍在`np.savez_compressed`写zip，等待方只判断文件存在，
  r1遂在`np.load`报`BadZipFile`；
- exposure：run目录/resolved config已创建，但0 optimizer step、0 development evaluation、0新target read；producer随后
  正常完成575,596 source rows以及P81/P96两个development artifacts；
- resolution：r2只复用完整artifact立即训练；producer今后写`.partial.npz`，完成后由同目录`Path.replace`原子交付final；
  不增加hash、checksum、fingerprint或内容门控；
- claim impact：纯producer-consumer race，不计科学trial；q90/model/steps/clearance/aggregation/coverage与cohorts不变。

下一可用编号：`V67-F78`。

### P107 outcome note — V67-F77后因子化development在两个consumed cohorts一致成立

- canonical：`run://worldsim_v67/WS-V67-P107-ACTOR-UNCERTAINTY-TUBE-01/20260830T061000Z__actor-uncertainty-tube-s0-r2`；
- P81 fixed50解析τ-risk/Actor/P75=`2/36/13`，P96=`2/9/12`；两者query-over-Actor分别减少94.44%/77.78%，
  query AUROC `.92901/.87305`；
- interpretation：Actor-only q90 tube保留可迁移dynamics uncertainty，τ只通过物理boundary clearance解析进入排序，避免
  P95/P102的cohort-specific query shortcut；这是跨两个已消费cohort的机制证据，不是独立确认；
- next：冻结P107 checkpoint、normalization、q90、`.05m` floor、time/Actor max、H3.5和fixed50，另取target-unread
  cohort作一次primary confirmation；不在新target read后改score/model/gate。

下一可用编号仍为：`V67-F78`。

### P108 freeze note — 新confirmation只在scene level独立

- target-unread cohort固定为`0092/0329/0555/0012/0035/0268/0795/0917/0925/1060`，四location覆盖且cohort内
  10个distinct sessions；选择只用official split/order、location/session metadata与既有processed-path absence，不读target；
- frozen primary是P107 r2 checkpoint；q90、normalization、`.05m` clearance floor、time/Actor max、H3.5、fixed50
  全不变，只比较是否严格少于Actor-only且不多于P75，不复制P96的冗长gate matrix；
- limitation：部分session在历史cohort有相邻scene，因此只声称scene-level independent；不会写session-level、collision、
  planning或safety generalization；
- route miss只可在target materialization前定位exact shard并修locator，不换scene；target一旦读取，任何失败直接登记
  `V67-F78`并关闭primary recovery。

下一可用编号仍为：`V67-F78`。

### P109 launch note — 用方向投影检验P107 isotropic tube近似

- motivation：P107 scalar q90除以absolute clearance忽略Actor误差方向；MultiPath的分层uncertainty与closed-form query启发
  将Actor signed residual distribution沿candidate boundary normal解析投影；
- fixed method：diagonal Gaussian mean/scale、Gaussian NLL、9 time samples、linearized signed-clearance margin、time/Actor max；
  网络不读τ，且不扫full covariance、scale floor、loss、projection、aggregation、seed或coverage；
- evidence boundary：只先读已消费P81/P96；通过后才允许在P108 target rows出现前冻结prospective secondary。P108 frozen
  P107 primary、cohort和decision均不改变；
- execution：P109 source producer先原子交付source，GPU随后训练，同时producer继续development IO并与P108 archive IO重叠。

下一可用编号仍为：`V67-F78`。

### P109 outcome note — directional projection在两个consumed cohorts均为0 selected events

- canonical：`run://worldsim_v67/WS-V67-P109-DIRECTIONAL-ACTOR-UNCERTAINTY-01/20260830T062500Z__directional-actor-uncertainty-s0-r1`；
- outcome：P81 query/Actor/P75=`0/44/13`、P96=`0/5/12`，query AUROC `.96764/.90434`；6,000-step final
  Gaussian NLL=`-3.64128`，无工程/资源失败；
- literature response：最新开源semi-analytic collision研究同样将stochastic boundary crossing作为独立于spatial overlap的
  高效估计路线；这里保留更窄的linearized occupancy-flip ranking解释，不宣称Monte Carlo等价或calibrated probability；
- resolution：P109 development执行条件满足，P110 checkpoint/config在P108 target rows出现前冻结并只作同read prospective
  secondary；P108 P107-scalar primary的cohort、score和decision不改变。

下一可用编号仍为：`V67-F78`。

### P107/P109 mechanism note — clearance-only不能解释跨cohort增益

- frozen baseline：每个Actor/time取`1/max(abs(predicted separation - interaction radius), .05m)`，再time/Actor max与
  per-scene fixed50；不训练、不调floor/aggregation/coverage；
- result：consumed P81选择1/95 events、AUROC `.91404`，consumed P96选择13/36、AUROC `.79879`；对照P107=`2/2`、
  P109=`0/0`；
- interpretation：P81上的强结果有显著boundary-distance成分，但clearance-only跨到P96明显退化；Actor uncertainty与方向
  投影提供了不能由纯几何解释的稳定性证据；
- protocol impact：P108/P110在target read前仅增加同一descriptive comparator，不改变P108 frozen primary或P110 decision，
  不创建gate matrix、threshold sweep或新claim。

下一可用编号仍为：`V67-F78`。

### V67-F78 — nonlinear Gaussian occupancy sampling跨cohort劣于linearized boundary projection

- 分类：`scientific/uncertainty-query-approximation`；状态：`closed_negative_no_sampling_sweep`；
- canonical：`run://worldsim_v67/WS-V67-P112-NONLINEAR-GAUSSIAN-CROSSING-01/20260830T065000Z__nonlinear-gaussian-crossing-s0-r1`；
- symptom：固定256-sample nonlinear recomputation在P81保持0 selected events且AUROC `.97228`，但P96变为3 events/
  AUROC `.85852`，劣于P109 linearized projection的0/`.90434`；
- interpretation：unimodal diagonal Gaussian在完整2D distance非线性下的finite-sample tail会放大scale/mean误差；boundary-normal
  projection更直接对齐occupancy decision boundary，跨cohort反而稳定；
- literature response：2025 open-source semi-analytic work分别研究spatial overlap与stochastic boundary crossing；本结果支持在
  当前数据/模型保留boundary-crossing approximation，而不是假设更“精确”的spatial sampling必然更好；
- resolution：关闭sample-count/full-covariance/distribution/seed sweep，保留P109 frozen linear score；P108/P110不变；
- claim impact：不支持nonlinear sampled collision probability或calibration claim，不影响P109 development机制结果。

下一可用编号：`V67-F79`。

