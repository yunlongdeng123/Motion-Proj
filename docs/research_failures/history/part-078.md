# 历史原始记录 078

> 冻结历史，不是当前执行规则。当前规则见 [scaling law](../../../auto-research_scaling_law.md) 与 [AGENTS](../../../AGENTS.md)。
> 原文件：`docs/RESEARCH_FAILURES.md`；迁移前提交 `abbd431c`。原有相对链接按原目录解释，证据路径原样保留。
> 按索引读取相关小节；旧哈希、过度门控和资源指令均不重新生效。

### V61-F09：通用生成表面不能冒充场景观测一致的 Occupancy

H-ME2-003 canonical run `20260822T121848Z__hy3d-actor-s1234-r1` 完成固定四臂各 6 例。A0 image、A1 bbox、
A2 raw-LiDAR point 与 A3 O_method voxel 全部为 `0/6 ACCEPT`。主臂 A3 没有 false-safe，但每例都在 method 和
独立 O_eval 中占据已观测 FREE cell；method conflict=`6..246`，eval conflict=`8..273`。所有四臂均有同一失败，
而 A3 的 native coverage、hole coverage、silhouette 与 extent 多数已达到冻结下限，因此不是提示词、seed、纹理或
单一轮廓阈值问题。

该失败是科学机制 rejection：Hunyuan 输出的通用闭合 actor surface 没有把场景 FREE-space 作为硬约束，不能作为
Occupancy-authoritative proposal。按预注册规则永久停止本版本 Hunyuan actor 路线；不得靠 prompt/seed/steps/
octree sweep、放宽 FREE=0、事后 clipping 或 per-case 选择恢复。ME-3 学习式 occupancy 是计划中的独立机制，
仍按 GaussianWorld→OccWorld→Drive-OccWorld→IR-WM→OccSora 优先级审计，不把本失败无依据外推到该路线。

### V61-F10：tmux 正式入口必须由 wrapper 自举仓库根目录

H-ME3-GW-001 的第一次正式入口从 source=`16c0efd8d570eaa15c5c4757ddfb434af8b61ede` 启动，但 tmux
非登录环境没有仓库级 `PYTHONPATH`；Python 把 `scripts/` 而不是 repository root 放在 `sys.path[0]`，wrapper
因此在导入 `motion_proj.worldsim_v61.me3_predicted_experiment` 时立即触发 `ModuleNotFoundError`。失败发生在 run
directory 创建、source/artifact 读取、模型载入、GPU context、predicted occupancy 或 method decision 之前；
canonical run=`null`，没有科学结果，不能记为 GaussianWorld rejection。

H-ME3-GW-002 只在 wrapper 导入项目包前把 `Path(__file__).resolve().parents[1]` 加入 `sys.path`，并先用
`--help` 做无 run/GPU 的入口 smoke。GaussianWorld commit/weights、两个并行 scene workers、seed、2Hz frame schedule、
class mapping、UNKNOWN policy、28-case denominator、O_eval separation、thresholds、资源预算与 stop rule 全部不变。


### V61-F11：单卡 inference capability 通过不等于 predicted Occupancy 可以成为安全 authority

H-ME3-GW-002 canonical run `20260822T134559Z__predicted-occ-s1-r1` 完成两个 scene worker、24 次官方 streaming
inference、4 个 target occupancy、28 个 method decisions 和隐藏 O_eval 评分。预测臂与 oracle O2 得到相同的
`10/28 ACCEPT` 和 mask-area yield=`0.3983001361`，但这10例全部 false-safe；route-support 例的 hidden
observed-FREE conflict ratio=`0.766..0.958`，actor/disocclusion=`0.159..0.328`。run 正确以
`predicted_zero_false_safe=false` 拒绝。P6 的 weight/output/资源 capability 结论仍有效，但不能提升为安全性结论。

官方 GaussianWorld head、网格与类别源码，以及 DriveStudio nuScenes transform 源码和跨 metadata 数值对照都没有发现
x/y/z、class17 empty、camera order 或 lidar2img 错误。小幅前相机矩阵差异来自 nuScenes 异步相机 timestamp，后相机
在机器精度内一致。因此不得通过轴交换、投影修补或输入排列试错重开 GaussianWorld。

不得用 O_eval 选 confidence threshold、降低 UNKNOWN/verifier 门、做 grid/schedule/checkpoint sweep，或把 predicted
FREE 冒充 observed truth。已有 artifact 已证明把 observed O_method FREE 作为保守 veto 会让10个接受项全部 abstain；
无需创建零产出的重复 run。ReliOcc、α-OCC 与 OCCUQ 的可靠 uncertainty 需要训练/calibration，朴素 softmax/entropy
也不支持无校准安全声明。后续只允许先做一次 IR-WM truth-free current-state capability smoke；通过后才消耗唯一一次
ME-3 recovery，失败则停止 learned occupancy 并保留负结论。
### V61-F12：checkpoint 的零 missing gate 必须区分未使用的官方删除参数与有效 forward state

H-P7-IRWM-001 canonical run `20260822T143153Z__irwm-current-smoke-s1-r1` 已完成官方 IR-WM current-state GPU
forward，并写出 finite/nonempty occupancy。17项 gate 中15项通过；失败项只有环境版本字符串和模型零 missing。
Detectron2 使用官方 `0.6+cu111` wheel，而预注册只写 `0.6`；这不是不同 release。checkpoint 的唯一 missing keys
为 `pts_bbox_head.transformer.reference_points.weight/bias`。冻结官方 `WorldBEVFormerHead.init_weights()` 明确删除
整个 `transformer.reference_points`，其 detector decoder 在本次 `get_bev_features` current-state 路径不执行。

不得改写 H001 的 rejected terminal、直接手工把 gate 改成 PASS、给 missing 参数调值，或重复完整 GPU forward。
H-P7-IRWM-002 使用独立 P7R task，精确绑定 H001 gate/report/output/manifest/terminal 和官方删除源码，只允许
Detectron2 build suffix 与上述两项 source-proven unused missing keys；其余 H001 capability、truth-free、resource
合同全部原样要求通过。任何额外 missing/unexpected key 或 artifact 漂移都停止 learned occupancy。
### V61-F13：不同 predicted Occupancy capability 不能替代独立 observed-FREE safety authority

H-ME3-IRWM-001 canonical run `20260822T145543Z__irwm-predicted-occ-s1-r1` 完成两个并行 scene workers、
4 个 target occupancy、28 个 method decisions 和隐藏 O_eval 评分。IR-WM primary 与 oracle O2 都得到相同的
`10/28 ACCEPT`、accepted mask yield=`0.3983001361`，但全部10例 false-safe。route-support 的 hidden
FREE conflict=`0.344..0.571`，actor/disocclusion=`0.106..0.173`，均超过固定0.05；因此唯一顶层失败 gate 是
`predicted_zero_false_safe`。正式 run 无训练、calibration、threshold selection、confirmation read 或 truth 泄漏，
资源也在预算内，故这是科学机制 rejection，不是工程 blocked。

GaussianWorld 与 IR-WM 使用不同官方时序机制、类别合同和网格，却都复现 oracle 的10例接受集合且得到10/10 false-safe。
本证据拒绝在当前 development 协议中把 learned argmax occupancy 直接作为安全 authority；不否定两模型的 perception
capability，也不产生现实安全声明。不得再换 backend、选 confidence threshold、改 checkpoint/grid/history window、
放宽 verifier、用 O_eval 选阈值，或执行确定性零 yield 的 observed-FREE veto 冒充恢复。唯一 ME-3 recovery 已消费，
ME-4 不授权；V6.1 minimum experiment 以负结论收口。
## WorldSim V6.5 failure ledger 启动审计（2026-08-27）

`WS-V65-P0-INHERITANCE-PROTOCOL-01` 没有新增科学、数据或资源失败。首次新分支 `git push -u` 未建立
upstream，根因是远端 GitHub 出站未使用当前 LocalTUN；设置会话端口后同一分支成功推送，没有代码/run/quality
影响，因此不分配 `V65-Fxx`。下一可用编号仍为 `V65-F01`。

P1 stage preregistration 与实现落盘前审计：没有新增 failure；尚未创建 P1 run、尚未读取指标，下一可用编号仍为
`V65-F01`。

### V65-F01 — trajectory condition 被错误要求改善 task-agnostic 物理标签

- task/run：`WS-V65-P1-CONDITION-SIGNAL-ATLAS-01` / `20260827T074500Z__signal-atlas-s0-r1`；
- symptom：T0 AUROC/AUPRC 相对 q0 为 `-0.000183/-0.001443`，fixed-route density 相对恶化 `5%`，
  scene lower/equal/higher=`1/13/2`；只有真实 trajectory 相对 shuffle 的 AUROC 响应 `+0.009591`；
- root cause：T0 把 trajectory residual 训练成 task-agnostic hidden-FREE 分类器；trajectory 明明是任务查询，
  却没有 task outcome / utility / actor-time supervision，目标语义与 WoTE/UniAD/VAD 的 planning-oriented 用法不一致；
- preserved evidence：canonical run、173MiB compact cache、预注册配置和模型均保留；
- resolution：`WS-V65-H-P1-001` rejected，禁止 T1/seed/capacity rescue。新建 `WS-V65-H-P1R-001`，明确分离
  frozen `r_phys` 与 nonnegative relevance-scaled `r_task`，只以 fixed-opportunity task risk 判定；
- claim impact：当前没有 V6.5 trajectory-conditioned method claim；P1R 仍是 legacy train-only mechanism。

下一可用编号：`V65-F02`。

P1R canonical run 完成且四个预注册 gate 全过，没有新增 failure。该结果只减少 1 个 sampled route conflict，
已按弱 train-only signal 记录，不以小 denominator 夸大结论。下一可用编号仍为 `V65-F02`。

P2 metadata-only cohort freeze、preparation/native/evidence/evaluator 入口落盘前没有新增 failure；formal P2 quality
尚未读取。下一可用编号仍为 `V65-F02`。

### V65-F02 — fresh scene metadata 选择未覆盖冻结 IR-WM temporal-info capability

- task/runs：preparation r1 与 `scene-0520/0781/0800` 三个 native scene r1；
- symptom：preparation 全 shard 扫描尚未完成；并发 native workers 在 sidecar、model score 与 quality 生成前均于
  `payload["infos"][scene]` 触发 `KeyError`；`scene-0106` 经 key audit 同样不可用；
- root cause：首版 cohort 只审计 V6.1–V6.4 config exposure 与 processed availability，没有检查冻结
  `nuscenes_temporal_infos_train.pkl` 的 700-key capability boundary；
- literature/open-source response：官方 BEVFormer 要求通过 `tools/create_data.py nuscenes ...` 生成
  `nuscenes_infos_temporal_{train,val}.pkl`。当前 research 为避免重新生成全量 infos、CAN bus 依赖与 schema 漂移，
  采用更窄的项目迁移：只从冻结 pickle 已支持、且未被 v61–v64 使用的 scene 中重选；
- resolution：formal quality read 仍为 false；首版 cohort 标记为
  `superseded_pre_read_capability_ineligible`。最终 cohort 冻结为
  `0996/0443/0002/0043/0023/0072`，保留失败目录和 2.5GiB partial raw，并让 recovery 复用已抽取文件；
- claim impact：没有 P2 模型或质量结论，`WS-V65-H-P2-001` 保持 active；不消耗唯一正式 P2 read。

下一可用编号：`V65-F03`。

V65-F02 recovery 已完成：preparation r2 成功生成 6/6 processed scenes；6 个 pipelined native scene runs
均 `passed=true`，72/72 targets 完整；evidence run 72/72 units `passed=true`。全程未读取 P2 quality，未出现
新的失败，因此下一可用编号仍为 `V65-F03`。

### V65-F03 — train-only monotone trajectory risk 在 fresh ranking boundary 完全等序

- task/run：`WS-V65-P2-TRAJECTORY-CONDITIONED-RISK-01` /
  `20260827T093900Z__trajectory-selection-s0-r1`；
- symptom：q0 与 task arm 均为 fixed-route `18/6975`，relative reduction=`0%`，worst-tail 完全相同，
  scene lower/equal/higher=`0/6/0`；task arm 多 4 个 non-route conflicts（relative `+0.0881%`）；
- root cause：P1R 的 legacy train-only 信号只有 1/20 sampled conflict，未迁移到 fresh 40% selection boundary；
  nonnegative trajectory residual 改变 probability，但没有产生可泛化的 route ranking 变化；
- preserved evidence：唯一正式 72-case run、case metrics、canonical native/evidence inputs 全部保留；
- resolution：`WS-V65-H-P2-001` rejected；关闭 trajectory-only score-ranking family，不执行 T1/seed/capacity/
  threshold rescue，不做第二次 P2 read；
- literature response：PRECOG/M2I/GameFormer/VAD/Implicit Occupancy Flow 均将条件建模放在 ego goal/action 与
  multi-agent future response 或连续时空查询上。后续仅允许以新 train-only hypothesis 审计 actor-time/action-outcome，
  且不得使用已消费 P2 cohort 做选择。

下一可用编号：`V65-F04`。

P2R actor-time train-only hypothesis 已在读取 token outcome 统计前冻结；当前无新增 failure，下一可用编号仍为
`V65-F04`。

### V65-F04 — 正常驾驶 legacy Actor tokens 的 1.5m binary collision support 为空

- run：`run://worldsim_v65/WS-V65-P2R-ACTOR-TIME-TRAIN-ONLY-01/20260827T100000Z__actor-time-s0-r1`；
- symptom：train=`0/476 positives`、eval=`0/302 positives`，A0/A1/shuffle AUPRC 均为 0、AUROC undefined；
- root cause：冻结的 1.5m Actor swept-envelope/ego-route 硬碰撞事件在这些正常驾驶 scenes 中不存在；
- resolution：run 与 96KiB cache 保留；不扩大半径、不换 scenes、不重跑 binary label。新建 P2C continuous
  proximity cost，固定 `exp(-distance/6m)` 与 absent=60m；
- literature response：joint dynamics+cost map（CVPR 2021）、Occupancy Flow、DTPP 与 DiffStack 都支持连续
  时空 cost/flow，而不是依赖稀有硬碰撞标签；
- claim impact：P2R 没有 actor-time 机制结论；P2C 仍为 legacy train-only。

下一可用编号：`V65-F05`。

### V65-F05 — 连续 cost 共用物化器仍硬依赖二值半径字段

- run：`run://worldsim_v65/WS-V65-P2C-ACTOR-TIME-COST-01/20260827T101500Z__actor-time-cost-s0-r1`；
- symptom：run directory/status 创建后，`_materialize` 在任何 evidence unit 读取、GPU geometry、训练或评分前，
  对不存在的 `evidence_contract.route_corridor_radius_m` 抛出 `KeyError`；
- root cause：P2R/P2C 共用 Actor-token materializer，但二值 label 的 task-specific config field 被写成公共必需字段；
- open-source response：Hydra 的 config composition 将共享基础字段与任务组差异组合；项目内采用同一窄边界，
  让未被 continuous target 消费的 binary radius 成为可选字段，而不是向 P2C 科学合同补入伪依赖；
- resolution：失败目录保留，科学输入/gates/seed/model 未改变；修复后使用新 run-id r2，未覆盖失败现场；
- claim impact：没有科学 read 或指标，不构成 P2C 负结果。

### V65-F06 — Actor×time 连续代价对时间敏感但不优于 snapshot

- run：`run://worldsim_v65/WS-V65-P2C-ACTOR-TIME-COST-01/20260827T102000Z__actor-time-cost-s0-r2`；
- symptom：A1 相对 A0 Spearman `-0.014889`、MSE relative reduction `-34.59%`；matched 40% pooled cost 虽降低
  `10.37%`，两个 eval scenes 却都恶化（lower/equal/higher=`0/0/2`）；
- mechanism audit：真实 A1 比 scene-wise shuffled-time Spearman 高 `0.098817`，故网络使用了时间特征；失败不是
  条件完全未进入网络，而是该增量在冻结 split 上没有形成优于强 snapshot geometry 的可泛化排序；
- literature response：UniAD、DTPP、DiffStack 的有效增量来自 planning-oriented joint objective 或候选策略
  cost evaluation，不支持继续放大当前独立 Actor-token MLP；
- resolution：`WS-V65-H-P2C-001` rejected；关闭 Actor/time family，不做 scale/seed/capacity/split rescue，P3
  不解锁。后续只允许审计固定 V6.4 risk 上的独立 admission，不能重开已关闭的 representation family；
- claim impact：无 Actor-time method/planning/safety claim；formal V6.5 selection read 仍为 false。

下一可用编号：`V65-F07`。

