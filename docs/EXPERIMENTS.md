# 实验索引

本页按实验定位报告和原始证据；当前执行状态只见 [RESEARCH_STATUS](RESEARCH_STATUS.md)。完整历史过程保留在 [V7.4 实验历史](archive/2026-09/v74-0920/EXPERIMENT_HISTORY.md)及[V7.3 完整台账](archive/2026-09/pre-v74/V73_EXPERIMENTS.md)。

| task / run 或阶段 | 记录内容 | 入口 |
|---|---|---|
| VADGS-P0-000 / V7.6进行中 | 官方nuScenes 000，VAD-GS从零训练；4k/8k权重、几何门禁与61张留出相机5早期评估；30k及scene-0230/0255待续 | [V7.6工程记录与architecture](v76/P0_ENGINEERING.md)、[4k/8k数据](autoresearch/worldsim_v76/p0-8k-evidence.json) |
| WS-V75-OMNI-REVIEW-02 / 20260923-proposal-r9-all-approved | 24个10秒case获人工批准，6个ego、18个对象；获批清单与原始参考保留。r9批准快照不是新的OmniDreams生成成绩 | [V7.5 r9收尾与architecture](autoresearch/worldsim_v75/downstream_bench/r9-closeout/README.md)、[机器可读索引](autoresearch/worldsim_v75/downstream_bench/r9-closeout/summary.json) |
| WS-V75-FIVE-BASELINES-01 / 20260923-r1，HUGSIM r9 | 7场景ground/scene各30k与导出；24/24 case各100帧事实/反事实和自动指标；10Hz适配与两个场景质量风险单列；训练checkpoint后已退役 | [收尾数量、证据与恢复边界](autoresearch/worldsim_v75/downstream_bench/r9-closeout/README.md) |
| WS-V75-FIVE-BASELINES-01 / 20260923-r1，DriveEditor r9 | 18对象case原生10帧事实/反事实；11个完成10秒迭代反事实；6个ego接口不支持、7个对象迭代未完成，不合并分母 | [收尾](autoresearch/worldsim_v75/downstream_bench/r9-closeout/README.md)、[V75-F02](research_failures/entries/V75-F02.md) |
| V7.5 r9 证据保留 / 20260925 | 保留119个MP4、逐case指标、配置和两份人工评分工作簿，119/119哈希核验；旧模型、七场景checkpoint及中间输入退役，可用空间约141→378GiB | [保留与重建限制](autoresearch/worldsim_v75/downstream_bench/r9-closeout/README.md) |
| WS-V75-OMNI-REVIEW-02 / 20260923-proposal-r1至r7-insertion-review | 历史待审提案：r7替换INSERT-02/04/05，原始参考有视频，当时生成与评分留空。后续以r9获批清单为准，不将旧待审状态当现状 | [r7历史提案](autoresearch/worldsim_v75/downstream_bench/review-20260923/insertion-review-r7.md)、[r6历史提案](autoresearch/worldsim_v75/downstream_bench/review-20260923/human-feedback-r6.md) |
| WS-V75-DOWNSTREAM-FULL-01 / 20260923-r1 | 我们固定24case：OmniDreams 24对/48段3696帧，原始参考24段、自动读出与固定五帧AI初评；ReSim单卡chunk17+offload真实ego队列；三个pilot高斯重建输入准备。静帧诊断非闭环评测，人工null、六维无总分 | [记录与architecture](autoresearch/worldsim_v75/downstream_bench/full-20260923/README.md)、[轻量证据](autoresearch/worldsim_v75/downstream_bench/full-20260923/summary.json) |
| WS-V75-DOWNSTREAM-GPU-SMOKE-01 / 20260923-r1 | 单 RTX3090：OmniDreams ego 减速2×61帧，GaussianDWM QA兼容版1条合成回答，HUGSIM移除2×9帧×6相机；ReSim VAE OOM停止，StreetGS checkpoint缺失；工程 smoke，正式bench 0 case | [报告与architecture](autoresearch/worldsim_v75/downstream_bench/gpu-smoke-20260923/README.md)、[日志/结果](autoresearch/worldsim_v75/downstream_bench/gpu-smoke-20260923/summary.json) |
| WS-V75-DOWNSTREAM-CFBENCH-PREP-02 / 20260923-r1 | GPU-free 全量准备：24/24 CPU geometry pass、24张人工审阅图；ReSim/DriveEditor/GaussianDWM/HUGSIM 输入与公开资产物化；GaussianDWM loader 合同问题、DriveEditor reference mask 风险及 HUGSIM release 路径问题；10项本地测试与55项上游测试通过；0模型前向 | [设计、发现与边界](v75/DOWNSTREAM_COUNTERFACTUAL_BENCH.md)、[manifest/qualification/assets/env/preflight/plan](autoresearch/worldsim_v75/downstream_bench/) |
| WS-V75-DOWNSTREAM-CFBENCH-PREP-01 / 20260922-r1 | 下游导向 paired-edit pilot：四类各6个共24个候选；六模型能力/资源 registry、CPU preflight、fail-closed planner、六维无总分 evaluator；0模型调用 | [设计与边界](v75/DOWNSTREAM_COUNTERFACTUAL_BENCH.md)、[manifest/preflight/plan](autoresearch/worldsim_v75/downstream_bench/) |
| WS-STORAGE-OLD-RUNS-RETIRE-01 / 20260922-r1 | 历史研究脉络与产物退役；保留轻量证据、当前 V7.5 依赖核验；0 模型调用 | [清单与恢复边界](autoresearch/old_runs_retirement_20260922/README.md) |
| WS-V75-CF-DEFINITION-01 / 20260921-r1 | WorldSim反事实质量定义、参考层级与有限实验设计；0模型调用，非新科学结果 | [定义](v75/PROBLEM.md)、[评价协议](v75/COUNTERFACTUAL_EVALUATION.md)、[审阅图](autoresearch/worldsim_v75/counterfactual_definition/) |
| WS-V75-ACTOR-REMOVAL-QUALIFY-01；GENERATION-01 / 20260921-r2；r1 | 两个曝光开发源中冻结首个A遮挡B对象对；三状态×未编辑/移除共6段702帧；reference通过，类别先验残留候选 | [对象移除报告](v75/ACTOR_REMOVAL_COUNTERFACTUAL.md)、[轻量证据](autoresearch/worldsim_v75/actor_removal_counterfactual/summary.json) |
| WS-V75-ACTOR-REMOVAL-CONFIRM-SOURCES-01 / 20260921-r1 | 固定8个未曝光日志、每个一个+6.5s窗口；第5个首次通过后停止；24次CPU检测、0重建/生成 | [对象移除报告](v75/ACTOR_REMOVAL_COUNTERFACTUAL.md)、[来源结果](autoresearch/worldsim_v75/actor_removal_counterfactual/confirmation-source-result.json) |
| WS-V75-ACTOR-REMOVAL-CONFIRM-QUALIFY-01 / 20260921-r1,r2,r3 | r1/r2为raster前环境失败、0生成；r3复用同一单次DVGT，三状态raster通过，科学输入不变 | [工程与资格边界](v75/ACTOR_REMOVAL_COUNTERFACTUAL.md)、[资格摘要](autoresearch/worldsim_v75/actor_removal_counterfactual/confirmation-qualification-summary.json) |
| WS-V75-ACTOR-REMOVAL-CONFIRM-GENERATION-01 / 20260921-r1 | 独立源三状态×两编辑6段702帧；reference/DVGT/class均A=10/10、B=10/10，reference gate失败，禁止重建排名 | [对象移除报告](v75/ACTOR_REMOVAL_COUNTERFACTUAL.md)、[评价](autoresearch/worldsim_v75/actor_removal_counterfactual/confirmation-evaluation.json) |
| WS-V75-ACTOR-REMOVAL-ONSET-01 / 20260921-r1 | A从f=0状态缺失的单段117帧机制诊断；A仍10/10，支持initial-image anchor；无闭环 | [对象移除报告](v75/ACTOR_REMOVAL_COUNTERFACTUAL.md)、[机制评价](autoresearch/worldsim_v75/actor_removal_counterfactual/onset-evaluation.json) |
| WS-V75-ACTOR-REMOVAL-G1-SCREEN-01 / 20260921-r1 | 下一批8个未曝光日志的中远距/小占比输入窗口；第5个首次通过后停止，8次CPU检测、0重建/生成 | [显著性候选](v75/ACTOR_REMOVAL_COUNTERFACTUAL.md)、[来源摘要](autoresearch/worldsim_v75/actor_removal_counterfactual/g1-salience-screen.json) |
| WS-V75-ACTOR-REMOVAL-G1-QUALIFY-01；G1-GENERATION-01 / 20260921-r1 | reference raster通过；仅未编辑/移除2段234帧，A=10/10→7/10仍未通过G1，DVGT未准入 | [门控结果](v75/ACTOR_REMOVAL_COUNTERFACTUAL.md)、[评价](autoresearch/worldsim_v75/actor_removal_counterfactual/g1-reference-evaluation.json) |
| WS-V75-ACTOR-REMOVAL-G1-FAR-SCREEN-01 / 20260921-r1 | 下一批8个未曝光日志的远距小投影窗口；0个合格A→B对、0 detector/重建/生成，门槛未放宽 | [对象移除收口](v75/ACTOR_REMOVAL_COUNTERFACTUAL.md)、[空窗口](autoresearch/worldsim_v75/actor_removal_counterfactual/g1-far-screen.json) |
| WS-V75-ACTOR-SLOWDOWN-G1-QUALIFY-01；G1-GENERATION-01 / 20260921-r1 | OmniDreams中距来源未来0.5×减速；前5帧输入严格一致，reference-state未编辑/减速各1段，后段5/5服从编辑，G1通过 | [轨迹报告](v75/ACTOR_TRAJECTORY_COUNTERFACTUAL.md)、[主图](autoresearch/worldsim_v75/actor_trajectory_counterfactual/main-state-error-figure.png) |
| WS-V75-ACTOR-SLOWDOWN-STATE-QUALIFY-01 / 20260921-r1,r2；STATE-GENERATION-01 / r1 | r1在状态误差样本读出前缺iopath结束；r2同输入完成1次官方DVGT读出及三状态资格；同一个OmniDreams新增depth-shift/depth+shape未编辑与减速4段468帧，13.27m输入误差对应55.61/32.08px生成响应误差 | [完整结果](v75/ACTOR_TRAJECTORY_COUNTERFACTUAL.md)、[轻量摘要](autoresearch/worldsim_v75/actor_trajectory_counterfactual/summary.json) |
| WS-V75-ACTOR-SLOWDOWN-CONTROL-QUALIFY-01；CONTROL-G1-01；CONTROL-STATE-01 / 20260921-r1 | 复用独立小误差源与既有未编辑序列；同一个OmniDreams新增reference/depth-shift/depth+shape减速3段351帧，0.380m输入误差下生成响应未退化 | [正反对照](v75/ACTOR_TRAJECTORY_COUNTERFACTUAL.md)、[响应图](autoresearch/worldsim_v75/actor_trajectory_counterfactual/response-error-contrast.png) |
| WS-V75-ACTOR-SLOWDOWN-CONFIRM-SCREEN-01 / 20260921-r1,r2；CONFIRM-QUALIFY-01 / r1 | r1为投影/场景字段工程错误、0检测；r2同一8日志在第6个选首个目标、1次CPU检测；reference raster后段变化139/64/0/0/0，G0失败，0重建/生成且不替换来源 | [停止结论](v75/ACTOR_TRAJECTORY_COUNTERFACTUAL.md)、[来源结果](autoresearch/worldsim_v75/actor_trajectory_counterfactual/confirmation-screen.json) |
| WS-V75-VELOCITY-PRIOR-CONTROL-01 / 20260920-r1 | 两日志20流CPU回放、世界速度零均值普通控制；主项退化、未准入生成 | [收口与全部证据](autoresearch/worldsim_v75/velocity_prior_control/README.md) |
| WS-V75-QUALIFY-01 / 20260920-r1 | 官方接口契约与资源来源核对 | [问题协议](v75/PROBLEM.md)、[V75-F01](research_failures/entries/V75-F01.md) |
| WS-V75-PREFLIGHT-01 / 20260920-r1 | 环境、真实初帧与条件、输入编码和权重预检 | [运行准备](v75/PREFLIGHT.md)、[证据](autoresearch/worldsim_v75/preflight) |
| WS-V75-BASELINE-01 / 20260920-single3090-r1 | 单卡首段、完整clean及同seed完整重复；资源与视频核验 | [单卡基线](v75/BASELINE.md)、[证据](autoresearch/worldsim_v75/baseline) |
| WS-V75-LOCALIZE-01 / 20260920-r1 | 单目标±0.5m条件干预与恢复；两seed有限实验 | [定位实验](v75/LOCALIZATION.md) |
| WS-V75-COHORT-01；NATIVE-COHORT-01 / 20260920-r1 | 来源目录与访问核验、3开发+3保留冻结、三例原生clean | [原生样例基线](v75/NATIVE_COHORT.md) |
| WS-V75-AV2-BRIDGE-01 / 20260920-r1 | Argoverse显式三维条件适配、真实未来RGB与单卡clean | [输入桥接](v75/AV2_BRIDGE.md) |
| WS-V75-NATURAL-01；NATURAL-SOURCES-01 / 20260920-r1 | 旧目标参考排除；八日志冻结、四个清楚可见目标的DVGT距离读出与普通控制 | [自然状态读出](v75/NATURAL_STATE.md) |
| WS-V75-NATURAL-ROLLOUT-01 / 20260920-r1 | 一个自然残余发现候选的五组生成、真实视频检测与额外观测修复 | [生成比较](v75/NATURAL_STATE.md)、[证据](autoresearch/worldsim_v75/natural_state/) |
| WS-V75-NATURAL-ROLLOUT-01 / 20260920-seed43 | 同一自然候选、固定输入的四组随机性复核；三项对照与实际状态链条图 | [有限复核](v75/NATURAL_STATE.md)、[逐时刻结果](autoresearch/worldsim_v75/natural_state/replication/replication_result.json) |
| WS-V75-OBSERVATION-01 / 20260920-r1 | 已有视频的固定点观测、缺失支持与普通投影控制 | [观测边界](v75/NATURAL_STATE.md)、[结果](autoresearch/worldsim_v75/natural_state/observation/result.json) |
| WS-V75-CONFIRM-SOURCES-01 / 20260920-r1 | 四日志前瞻来源窗口；真实可观测性未通过，无模型推理 | [来源与排除](v75/NATURAL_STATE.md)、[记录](autoresearch/worldsim_v75/natural_state/observation/confirmation_visual_review.json) |
| WS-V75-VISIBLE-DEV-01 / 20260920-r1 | 四日志可见性前置、八候选两读出、完整排除与普通尺度控制 | [可见性窗口](v75/VISIBLE_COHORT.md) |
| WS-V75-VISIBLE-ROLLOUT-01 / 20260920-r1 | 一例四组生成、实际二维正反结果、条件刚体比较不合格 | [报告](v75/VISIBLE_COHORT.md)、[证据](autoresearch/worldsim_v75/visible_cohort/) |
| WS-V75-CLOSEDLOOP-CONTRACT-01 / 20260920-r1 | 真实场景39帧动作→相机→条件契约；无世界模型生成或策略结论 | [闭环接口](v75/CLOSED_LOOP.md) |
| WS-V75-FOLLOWING-BASELINE-01 / 20260920-r1；20260920-natural4 | 任务来源与真实RGB策略；两个通过、一个相机不可观测 | [反馈报告](v75/FOLLOWING_CLOSED_LOOP.md) |
| WS-V75-FOLLOWING-CLOSEDLOOP-01 / 20260920-r1 | 一任务四组真实生成反馈，468帧/60决策，普通尺度与参考修复 | [报告](v75/FOLLOWING_CLOSED_LOOP.md)、[证据](autoresearch/worldsim_v75/following/) |
| WS-V75-BRAKING-TASKS-01 / 20260920-r1 | 固定12日志48起点，9个记录减速窗口，原持续前车入口0合格 | [制动任务](v75/BRAKING_TASKS.md)、[分母](autoresearch/worldsim_v75/braking_tasks/task_sources.json) |
| WS-V75-BRAKING-INTERFACE-AUDIT-01 / 20260920-r1 | 一次事后真实RGB入口诊断，5次感知；距离/速度状态误差通过同一IDM抵消，无生成 | [报告](v75/BRAKING_TASKS.md)、[证据](autoresearch/worldsim_v75/braking_tasks/interface-audit/) |
| WS-V75-RASTER-POLICY-01；APPROACH-BASELINE-01 / 20260920-r1；20260920-association-r2 | 官方高程＋真实历史的普通基线；实际关联反例修复与保存输入回放 | [接近任务](v75/APPROACH_CLOSED_LOOP.md)、[工程证据](autoresearch/worldsim_v75/approach/) |
| WS-V75-APPROACH-CLOSEDLOOP-01；APPROACH-REFERENCE-01 / 20260920-r1；20260920-association-r2 | 5组585帧含原GT工程失败；最终四组完整反馈、一次DVGT与42点截止前参照，几何修正未稳定恢复动作 | [报告](v75/APPROACH_CLOSED_LOOP.md)、[四组结果](autoresearch/worldsim_v75/approach/comparison.json) |
| WS-V75-APPROACH-STATE-CONTROL-01 / 20260920-r1 | 四组CPU直接状态反馈；468帧旧轨迹精确复现、468帧新控制，几何响应与生成/策略接口边界 | [报告](v75/APPROACH_STATE_CONTROL.md)、[结果](autoresearch/worldsim_v75/approach_state_control/result.json) |
| WS-V75-IOU-ASSOCIATION-CONTROL-01 / 20260920-r1 | 原策略175次回放访问精确复现；固定IoU关联平均改善但额外制动恶化，未进入新生成 | [控制报告](v75/APPROACH_STATE_CONTROL.md)、[结果](autoresearch/worldsim_v75/approach_state_control/iou_control/result.json) |
| WS-V75-BRAKING-DEV2-01；APPROACH-BASELINE-02 / 20260920-r1 | 新有限6×3窗口18→2→1，真实35次检测与固定策略基线通过 | [任务发现](v75/BRAKING_TASKS.md)、[分母](autoresearch/worldsim_v75/approach_dev2/screen/result.json) |
| WS-V75-APPROACH-CLOSEDLOOP-02；APPROACH-RECONSTRUCTION-02；APPROACH-REFERENCE-02 / 20260920-r1 | 第二制动任务四组468帧反馈、1次DVGT与55点额外参照；保留自然误差影响小的好案例 | [接近任务](v75/APPROACH_CLOSED_LOOP.md)、[结果](autoresearch/worldsim_v75/approach_dev2/comparison.json) |
| WS-V75-APPROACH-STATE-CONTROL-02 / 20260920-r1 | 第二任务468帧旧动作精确回放与468帧直接状态控制；双任务统一对比图 | [接近任务](v75/APPROACH_CLOSED_LOOP.md)、[状态控制](autoresearch/worldsim_v75/approach_dev2/state_control/result.json) |
| WS-V75-TEMPORAL-STATE-AUDIT-01 / 20260920-r1 | 官方时间接口、两任务过去观测与静止/CV强控制；两组234帧精确复现、四组468帧新CPU反馈，无模型调用 | [时间状态范围](v75/APPROACH_CLOSED_LOOP.md#时间状态审计近静止任务的范围边界)、[证据](autoresearch/worldsim_v75/temporal_state_audit/result.json) |
| WS-V75-MOVING-FOLLOWING-SCREEN-01 / 20260920-r1 | 复用固定6日志18起点的真实运动跟车资格；18→15→2→1近静止→0，无模型调用，窗口关闭 | [任务覆盖边界](v75/BRAKING_TASKS.md#同一固定来源的真实运动跟车资格)、[完整分母](autoresearch/worldsim_v75/moving_following_screen/result.json) |
| WS-V75-SHAPE-PRIOR-AUDIT-01 / 20260920-r1 | 两任务五组普通形状/距离适配；1170帧CPU控制，目标GT形状移出拟合、额外LiDAR单列，0次新模型调用 | [形状输入边界](v75/APPROACH_CLOSED_LOOP.md#普通形状先验与生成闭环相近间距不保证相同制动)、[完整审计](autoresearch/worldsim_v75/shape_feedback/audit/result.json) |
| WS-V75-SHAPE-FEEDBACK-01 / 20260920-r1；20260920-conditioning-r2 | 首轮目录预检错误保留；两任务两普通形状导出共468帧实际反馈，936帧新旧精确回放；2.757/0.301m配对进度差、先验部分缓解非完整恢复 | [结果与真实对比图](v75/APPROACH_CLOSED_LOOP.md#四段实际生成反馈)、[结果](autoresearch/worldsim_v75/shape_feedback/comparison.json)、[完整来源](autoresearch/worldsim_v75/shape_feedback/provenance.json) |
| WS-V75-SHAPE-CONFIRM-01 / 20260920-r1 | 唯一额外seed43：两GT通过后四配对分支，702帧生成/精确回放；主候选2.757→−0.324m、三个判据反向而关闭；第二任务小效应由普通先验恢复 | [复核结论](v75/APPROACH_CLOSED_LOOP.md#唯一额外seed复核主候选反向小效应可由普通先验恢复)、[事前规则](autoresearch/worldsim_v75/shape_confirmation/protocol.json)、[完整比较](autoresearch/worldsim_v75/shape_confirmation/review/comparison.json) |
| WS-V75-DVGT-CONTRACT-01 / 20260920-r1 | 两个旧调用的CPU契约检查：官方像素、坐标代数、原生方向与已有PnP；0次模型调用 | [接口范围](v75/NATURAL_STATE.md#原生点图契约与有限时间上下文控制)、[结果](autoresearch/worldsim_v75/native_contract/single_frame_audit/result.json) |
| WS-V75-DVGT-TEMPORAL-CONTRACT-01 / 20260920-r1 | 两次固定3时刻2Hz前向；14视角13改善1退化，严重单帧方向错位大幅减轻，余差未通过原生使用筛查；0次生成 | [完整对比](v75/NATURAL_STATE.md#两次冻结的普通历史观测控制)、[事前协议](autoresearch/worldsim_v75/native_contract/protocol.json)、[结果](autoresearch/worldsim_v75/native_contract/result.json) |
| WS-V74-P0-* / 20260910 | 数据、资源与输入角色准备 | [P0](archive/2026-09/v74-0920/WORLDSIM_V74_P0_HANDOFF.md) |
| WS-V74-METHOD-TOURNAMENT-01 | H1 WEX / RIF / DCS 与强控制 | [结果](archive/2026-09/v74-0920/WORLDSIM_V7_4_RESULTS.md)、[失败](archive/2026-09/v74-0920/WORLDSIM_V7_4_FAILURES.md) |
| H2 CPU / GPU P1 | 数据与学习能力实验、实现关闭 | [CPU](archive/2026-09/v74-0920/WORLDSIM_V7_4_H2_CPU_HANDOFF.md)、[GPU](archive/2026-09/v74-0920/WORLDSIM_V7_4_H2_GPU_P1_REPORT.md) |
| WS-V74-H2-P15-01 / switching-margin-r1 | 失效轨迹审计 | [P1.5](archive/2026-09/v74-0920/WORLDSIM_V7_4_H2_P15_FORENSICS_REPORT.md) |
| WS-V74-H2-P16-01 / 20260913__physical-metric-r2 | 普通几何控制 | [P1.6](archive/2026-09/v74-0920/WORLDSIM_V7_4_H2_P16_DIAGNOSTIC_REPORT.md) |
| WS-V74-H2-REDISCOVERY-01 / 20260915-cpu-r1 | 旧案例 CPU 回放与来源核对 | [重新取证](archive/2026-09/v74-0920/WORLDSIM_V7_4_H2_FIRST_RETURN_REDISCOVERY.md) |
| WS-V74-MAINFIG-01 / 20260915-first-return-r1；SECONDARY-01 / 20260915-secondary-r1 | 4主模型48前向、第二批24前向及表面参照 | [官方模型主图](archive/2026-09/v74-0920/WORLDSIM_V7_4_MAIN_FIGURES.md) |
| WS-SIM-IMPACT-01 / 20260915-r1 | HUGSIM-LTF 与碰撞表示 | [报告](archive/2026-09/v74-0920/WORLDSIM_SIMULATION_IMPACT.md) |
| WS-SIM-LIDAR-01；INTERACTION-01 / 20260915-r1 | LiDAR、策略/PDM 与六日志交互 | [LiDAR](archive/2026-09/v74-0920/WORLDSIM_SIMULATION_LIDAR_IMPACT.md)、[交互](archive/2026-09/v74-0920/WORLDSIM_SIMULATION_INTERACTION_FINDINGS.md) |
| Pi3X 局部资产审计 | 路面修复未消除残余 | [因果审计](archive/2026-09/v74-0920/WORLDSIM_SIMULATION_PI3X_CAUSAL_AUDIT.md) |
| WS-SIM-NATIVE-CLOSEDLOOP-01 / scene0004-step030000-full1 | SplatAD完整预算、反馈与感知控制 | [完整预算](archive/2026-09/v74-0920/WORLDSIM_SIMULATION_FULL_BUDGET_PERCEPTION.md) |
| WS-SIM-FF-LOCAL-ASSET-01 / r2 | Ω 与 Pi3X 网格编辑、重新投射与检测 | [局部资产](archive/2026-09/v74-0920/WORLDSIM_SIMULATION_LOCAL_ASSET_CAUSAL_AUDIT.md) |
| WS-SIM-FF-COHORT-PERCEPTION-01 / 20260915-r1 | 82新增+20复用检测；全部分母 | [六日志](archive/2026-09/v74-0920/WORLDSIM_SIMULATION_SIX_LOG_PERCEPTION.md) |
| WS-SIM-FRESH-NATIVE-01 / 20260915-r1 | 16组真实驾驶基线/路线控制 | [新来源](archive/2026-09/v74-0920/WORLDSIM_SIMULATION_FRESH_BASELINE.md) |
| WS-SIM-SPARSE-NATIVE-CLOSEOUT-01 / 20260920-r1 | 导入缺 terminaltables；0/40前向；依赖任务跳过 | [终态与日志](autoresearch/worldsim_simimpact/closeout_20260920/README.md) |
| WS-V74-DOCS-CLEANUP-01 / 20260920 | 文档职责、旧 failure 检索与归档链接修复；无新实验 | [整理记录](archive/2026-09/v74-0920/CLEANUP.md) |

每个 run 的真实完成数量、配置、seed、数据角色、成本和失败边界以该报告及对应 manifest 为准。不同类型执行次数不合并为独立样本。更早版本的实验从[历史归档](archive/README.md)进入。
