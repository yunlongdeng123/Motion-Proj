# Research Status

## WorldSim V6.7 P8 independent surface supported / P9 fresh inputs ready（2026-08-28）

状态：`v67_p8_independent_surface_supported_p9_fresh_input_implementation_ready`；canonical=
`run://worldsim_v67/WS-V67-P8-INDEPENDENT-INWARD-RAY-SURFACE-01/20260828T110941Z__independent-inward-s0-r1`；verdict=
`supported_independent_legacy_motion_compensated_inward_ray_surface_confirmation`。

P8 baseline/repaired=`19,654/10,882`，overall retention=`0.553679`；conflict `1,021→509`，reduction=`0.501469`；
clean `18,633→10,373`，retention=`0.556700`。Actor/shell/ID-track-trajectory=`1/1/1`，removed/hazard shift/scene
yield=`0/0/1`，9/9 gates通过。P4R+P8构成两个surface-task-independent legacy cohorts正证据，但仍非fresh population。

P9在任何处理/quality read前从archive band 4的79个unprocessed且repo-unmentioned scenes按固定1/7..6/7位置选出
`0348/0360/0373/0388/0399/0414`（indices `265/277/290/304/315/328`）。已冻结72-unit prep/evidence/native配置；
scene-ready后立刻运行单卡native GPU，使archive/preprocess I/O与GPU重叠。当前95GiB可用，单3090足够，无需关机。

## WorldSim V6.7 P7 independent actions supported / P8 surface ready（2026-08-28）

状态：`v67_p7_independent_actions_supported_p8_surface_implementation_ready`；canonical=
`run://worldsim_v67/WS-V67-P7-INDEPENDENT-FIXED-ACTIONS-01/20260828T110742Z__independent-actions-s0-r1`；verdict=
`supported_v67_fixed_budget_action_set`。

570 rows / 312 conflicts / 285 fixed actions；L0/q0/oracle handled=`191/203/285`，reduction=
`0.612179/0.650641/0.913462`。L0 Actor retention/removed/hazard shift/emitted fraction/scene yield=`1/0/0/0.5/1`，
6/6 gates通过。q0 pooled仍更高，但L0六场景reduction均>0.5，q0在0072/0443低于0.5；只作描述，不新增post-hoc gate。

P8配置已机械绑定P6/P7 canonical并保持P4R exact inward-ray rule、`0.512m`与九门完全不变。下一步一次independent
physical formal run；无同cohort recovery或扫参。

## WorldSim V6.7 P6 independent package supported / P7 actions ready（2026-08-28）

状态：`v67_p6_independent_package_supported_p7_actions_implementation_ready`；canonical=
`run://worldsim_v67/WS-V67-P6-INDEPENDENT-ACTOR-PACKAGE-01/20260828T110547Z__independent-package-s0-r1`；verdict=
`supported_v67_actor_preserving_package`。

72 units内119 unique Actors、570 states、1,353,734 primitives；Actor state/metadata retention=`1/1`，removed/hidden
target=`0/0`，runtime model/hazard-existence coupling=false，6/6 gates通过。package=`8 files/15,671,545 bytes`；
wall/RSS=`13.9384s/0.8299GiB`。

P7配置只机械绑定P5 scores与P6 package，固定50% budget及六门不变。下一步一次matched-action formal run；不把action
audit称为physical repair。

## WorldSim V6.7 P5 independent transfer supported / P6 package ready（2026-08-28）

状态：`v67_p5_independent_transfer_supported_p6_package_implementation_ready`；canonical=
`run://worldsim_v67/WS-V67-P5-INDEPENDENT-GEOMETRY-TRANSFER-01/20260828T110336Z__independent-transfer-s0-r1`；verdict=
`supported_task_untouched_legacy_geometry_transfer`。

72 units产生570 Actor-units（conflict/clean=`312/258`）；head AUROC/AUPRC=`0.665176/0.676612`，相对deterministic=
`+0.165176/+0.129244`，6/6 scenes above chance、4/4 gates通过。GPU/RSS/wall=
`0.02359GiB/0.9341GiB/10.1148s`，无refit/threshold。

必须同时报告：该cohort q0=`0.695177/0.706467`，head相对q0=`-0.030001/-0.029854`，因此不支持“head跨cohort
优于q0”。P6配置只机械绑定P5 canonical并烘焙Actor-preserving package；下一步一次CPU+I/O formal run，不改变score、
Actor existence或hazard合同。

## WorldSim V6.7 P4R supported / independent confirmation ready（2026-08-28）

状态：`v67_p4r_supported_p5_independent_confirmation_ready`；canonical=
`run://worldsim_v67/WS-V67-P4R-MOTION-COMPENSATED-INWARD-RAY-01/20260828T105920Z__inward-ray-s0-r1`；verdict=
`supported_task_untouched_motion_compensated_inward_ray_surface_repair`；`V67-F01 resolved_by_single_structural_recovery`。

同一18,238 boundary中保留9,652，overall retention=`0.529225`；conflict `1,003→484`，reduction=`0.517448`；
clean `17,235→9,168`，retention=`0.531941`。Actor/shell/ID-track-trajectory=`1/1/1`，removed/hazard shift/scene
yield=`0/0/1`，9/9 gates通过；wall/RSS/GPU=`10.2274s/0.6089GiB/false`。

下一步P5已在quality read前冻结：转到独立V65 P2六场景`0996/0443/0002/0043/0023/0072`，先做冻结head GPU
transfer，再机械串联package→50% actions→exact P4R rule。该cohort从未用于V6.6/V6.7 surface，但在V65已消费，故只称
independent legacy surface confirmation。无refit/threshold/radius/budget/gate sweep。

## WorldSim V6.7 P4 rejected / P4R motion-compensated recovery ready（2026-08-28）

状态：`v67_p4_rejected_p4r_motion_compensated_inward_ray_implementation_ready`；P4 canonical=
`run://worldsim_v67/WS-V67-P4-RAY-TERMINATED-SURFACE-01/20260828T105253Z__ray-surface-s0-r1`；verdict=
`rejected_task_untouched_ray_terminated_surface_repair`；`V67-F01 active`。

P4把1,003 conflict points降到322，reduction=`0.678963`，Actor/shell/ID-track-trajectory retention=`1/1/1`，
removed/hazard shift/scene yield=`0/0/1`；但overall/clean retention=`0.392368/0.396519 < 0.40`，7/9 gates通过仍拒绝，
不降低门槛。wall/RSS/GPU=`10.6065s/0.6087GiB/false`。

卡点检索ALSO、evidence-theory occupancy、continuous occlusion与GPOcc后，定位为原始source `behind_hit`与
motion-compensated Actor hit之间丢失ray/Actor关联。唯一P4R已冻结并实现：nearest compensated same-Actor hit的
`0.512m`邻域只保留`dot(query-hit, normalize(hit-origin)) >= 0`的inward half-ball；P3固定258 actions、九门及target-only
评估不变。下一步只做窄编译检查、提交push后一次formal recovery；不扫rule/radius/budget/gate。

## WorldSim V6.7 P3 supported / ray repair implementation ready（2026-08-28）

状态：`v67_p3_actions_supported_p4_ray_repair_implementation_ready`；P3 canonical=
`run://worldsim_v67/WS-V67-P3-FIXED-ACTIONS-01/20260828T104902Z__fixed-actions-s0-r1`。固定258/517 actions处理
185/295 conflicts，reduction=`0.627119`；Actor retention/removed/hazard shift/scene yield=`1/0/0/1`，6/6 gates通过。

P4实现已就绪：repair核心新增默认false的`support_expansion_requires_behind_hit`，因此V6.6旧配置语义不变；V6.7
显式使用`exact OR (radius<=0.512m AND source behind_hit)`。target只参与post-repair metric。下一步一次formal run，
不扫ray rule/radius/budget/gate。

## WorldSim V6.7 P2 package supported / P3 actions ready（2026-08-28）

状态：`v67_p2_package_supported_p3_actions_ready`；canonical=
`run://worldsim_v67/WS-V67-P2-ACTOR-PACKAGE-01/20260828T104733Z__actor-package-s0-r1`；verdict=
`supported_v67_actor_preserving_package`。70 eligible units烘焙107 unique Actors、517 states、1,093,082 primitives；
state/metadata retention=`1/1`，removed/hidden-target=`0/0`，6/6 gates通过。P3 locator已机械绑定canonical P2，
50% budget与所有gates不变；下一步一次fixed-action formal run。

## WorldSim V6.7 package/action implementation ready（2026-08-28）

状态：`v67_p2_package_p3_action_implementation_ready`。已复用V6.6 bake/action核心，但V6.7 runner独立写入
`worldsim_v67` run tree。P2从P1 score只输出observable ranking，不携带target/model；P3固定50% budget并保持Actor/hazard
attributes exact。P3 config只允许在P2 canonical产生后替换run locator，不改变任何scientific参数。下一步P2/P3各一次CPU
formal run；不做测试矩阵。

## WorldSim V6.7 P1 transfer supported / package next（2026-08-28）

状态：`v67_p1_geometry_transfer_supported_package_next`；canonical=
`run://worldsim_v67/WS-V67-P1-GEOMETRY-TRANSFER-01/20260828T104342Z__geometry-transfer-s0-r1`；verdict=
`supported_task_untouched_legacy_geometry_transfer`。

72 units产生517 Actor-unit（conflict/clean=`295/222`）；冻结head AUROC/AUPRC=`0.710521/0.730703`，相对
deterministic=`+0.210521/+0.160103`，6/6 scenes above chance；无model/normalization refit或threshold。wall=
`10.84445s`、GPU=`0.02359GiB`、RSS=`0.93161GiB`。下一步在同一scores/evidence上烘焙Actor-preserving package与
固定50% action set，不再读取P1或改变ray-terminated rule。

## WorldSim V6.7 ray-terminated surface protocol active（2026-08-28）

状态：`v67_p0_protocol_frozen_p1_transfer_implementation_ready`；分支=
`research/worldsim-v6.7-anisotropic-surface`，base=`c05ca27`。V6.7不复开V6.6 radius family；新增方向性source evidence：
exact same-Actor hit保留，one-voxel邻域只有同时命中已有`behind_hit` ray-termination state才保留。

外部迁移来自ALSO sensor-location occupancy、CVPR 2024 evidence-theory occupancy与SelfOcc SDF/rendering constraints；
不引入大模型。P1实现已就绪：冻结V6.6 P3L head在V6.7 task-untouched的V65 P3C六场景一次transfer，GPU forward与
两线程unit prefetch重叠。下一步直接formal run；无sweep、无Actor existence authority。

## WorldSim V6.6 research complete / arXiv report ready（2026-08-28）

状态：`v66_research_complete_arxiv_report_ready`；active task/hypothesis=`null/null`。正证据为two-level Actor
certificate、independent legacy local geometry ranking、Actor-preserving HARP package、fixed-budget triage与6/6 synthetic
lead-brake response；主要负证据为P7 sensor-supported physical repair在single recovery后终止。

Plan审计确认`P7 FAIL → 不进入 RL`，因此P9/P10/P11未执行；这不是多卡或算力不足。V6.6不能称RL-ready simulator、
physical repair success、fresh generalization或safety。ArXiv写作入口：
`docs/autoresearch/worldsim_v66/V66_ARXIV_TECHNICAL_REPORT.md`、`ARXIV_EVIDENCE_INDEX.md`与
`V66_RESEARCH_CLOSEOUT.md`。Closeout未新增scientific run、回归矩阵或hash/checksum/fingerprint。

## WorldSim V6.6 P8R supported / P9 locked / closeout next（2026-08-28）

状态：`v66_p8r_supported_p9_locked_closeout_next`；canonical=
`run://worldsim_v66/WS-V66-P8R-STOP-STATE-JERK-RECOVERY-01/20260828T095839Z__stop-state-jerk-recovery-s0-r1`；
verdict=`supported_synthetic_lead_brake_reactive_actor_capability`；`V66-F03 resolved_by_single_implementation_recovery`。

同一六场景/Actor/参数/gates下，P8R只修复重复stop-state update：selected/supported=`6/6`，pooled X0/X1
collision steps=`306/0`，X1 min gap=`1.948192m`，max command jerk=`6.000000m/s^3`，identity/lifecycle与logged
path exact；all pooled/per-scene gates通过。wall=`0.83402s`、RSS=`0.50465GiB`、GPU=false。

正结论只限fixed synthetic lead-brake bounded response。P7 surface repair仍为terminal negative，因此P9/RL不满足
plan联合前置，保持locked。下一步不复开P7/P8、不跑额外回归，而是审计plan locks并整理V6.6 arXiv evidence closeout。
详见`docs/autoresearch/worldsim_v66/P8R_STOP_STATE_JERK_RESULT.md`。

## WorldSim V6.6 P8 rejected / P8R stop-state recovery implementation ready（2026-08-28）

状态：`v66_p8_rejected_p8r_stop_state_recovery_implementation_ready`；canonical=
`run://worldsim_v66/WS-V66-P8-REACTIVE-ACTOR-01/20260828T095440Z__reactive-actor-s0-r1`；verdict=
`rejected_synthetic_lead_brake_reactive_actor_capability`；`V66-F03 active_recovery_frozen`。

P8六场景全部选中，X0/X1 pooled collision steps=`306/0`，X1 minimum gap=`1.948192m`，identity/lifecycle与
logged path保持；但只4/6 scenes全门通过。scene-0001/0219的command jerk=`9.637574/7.400627 > 6m/s^3`。
根因是speed=0边界在正常rate limiter后又执行一次acceleration increment。

按卡点检索Autoware longitudinal DRIVE/STOPPING/STOPPED与vehicle command jerk limiter后，冻结唯一P8R：零速时
desired acceleration=0，并且只应用一次原`6m/s^3`rate limiter。Actor、场景、horizon、所有IDM/AV参数与gates均不改；
失败即关闭P8。实现已就绪，下一步一次formal recovery。详见`P8_REACTIVE_ACTOR_RESULT.md`与
`P8R_STOP_STATE_JERK_RECOVERY_FREEZE.md`。

## WorldSim V6.6 P8 deterministic reactive-Actor implementation ready（2026-08-28）

状态：`v66_p8_reactive_actor_implementation_ready`。已实现P6 `ACTORS.jsonl`六场景metadata-only固定选择、X0
constant-speed、X1 jerk/acceleration-bounded IDM-style响应和AV lead-brake干预；输出selected Actors、双臂trajectory、
per-scene metrics与summary。低速场景的horizon只由固定12m headway、初速和AV deceleration解析确定，确保X0有完整
collision observation，不读取X1 outcome；空间位置沿logged polyline及其terminal tangent extension生成。

全部固定参数已补全到freeze/config；不扫Actor、headway、controller或horizon。P8是短CPU capability audit，当前没有
合法的GPU训练任务：P7已终止，P9/RL锁定。下一步只做`py_compile`/config解析后一次formal run。

## WorldSim V6.6 P7 surface family terminal negative / P8 independent capability frozen（2026-08-28）

状态：`v66_p7_surface_family_terminal_negative_p8_capability_frozen`；P7R2 canonical=
`run://worldsim_v66/WS-V66-P7R2-RADIUS-SUPPORTED-ACTOR-REPAIR-01/20260828T094232Z__radius-surface-repair-s0-r1`；
verdict=`rejected_conflict_reduction_after_single_recovery`；`V66-F02 closed_negative_after_single_recovery`。

固定`0.512m` one-native-voxel支持把overall/clean boundary retention从P7R的`0.383588/0.395715`提高到
`0.617684/0.619549`，但conflict reduction从`0.847660`降到`0.417872 < 0.50`。其余Actor/shell/track/
trajectory retention=1、removed=0、hazard shift=0、scene yield=1；8/9 gates通过仍拒绝。按预先冻结规则不再扫
中间radius、不降gate、不改budget，也不临时训练completion model。P7只保留triage正结果；physical repair、
RL-ready distribution与P9保持锁定。

卡点检索UniSim、SMARTS与Waymax后，P8只作为独立响应式Actor capability推进：P6六场景各固定选一条真实Actor
轨迹，在同一logged polyline比较X0 constant-speed与X1固定IDM-style bounded response的lead-brake干预；参数与
`>=5/6`场景gate已冻结，无sweep。即便通过，也只支持窄synthetic lead-brake response，不恢复P7或解锁RL/P9。
详见`docs/autoresearch/worldsim_v66/P7R2_RADIUS_SUPPORT_RESULT.md`与
`docs/autoresearch/worldsim_v66/P8_REACTIVE_ACTOR_MIGRATION_FREEZE.md`。

## WorldSim V6.6 P7R2 one-voxel recovery implementation ready（2026-08-28）

状态：`v66_p7r2_one_voxel_recovery_implementation_ready`。P7R loader新增可选same-Actor nearest-hit radius，P7R2唯一
配置固定`0.512m`，等于native voxel side；P7R原exact语义仍为radius=0。其余L0 action set、target-only evaluation、
九个gates与Actor/shell/track/hazard保护全部不变。实现只使用cKDTree查最近motion-compensated hit，不生成新点、不训练
completion model。下一步一次formal recovery；失败关闭family。

## WorldSim V6.6 P7R rejected / P7R2 one-voxel recovery frozen（2026-08-28）

状态：`v66_p7r_exact_hit_rejected_p7r2_radius_recovery_frozen`；canonical=
`run://worldsim_v66/WS-V66-P7R-SENSOR-SUPPORTED-ACTOR-REPAIR-01/20260828T093710Z__sensor-surface-repair-s0-r1`；
verdict=`rejected_consumed_legacy_sensor_surface_repair`；`V66-F02 active`。

P7R把1,175 conflict points降到179，reduction=`0.847660`；Actor/shell/ID-track-trajectory retention=1、removed=0、
hazard shift=0、scene yield=1。但overall/clean boundary retention=`0.383588/0.395715 < 0.40`，7/9 gates通过仍拒绝，
不降低gate。

按卡点检索PoinTr、SnowflakeNet、RFNet和PoinTr official code后，冻结唯一P7R2：same-Actor hit support从exact
evidence voxel扩到一个native voxel side=`0.512m`邻域；同一L0 action set、同一九门，无radius/budget/threshold sweep。
若失败关闭family。冻结：`docs/autoresearch/worldsim_v66/P7R2_RADIUS_SUPPORT_RECOVERY_FREEZE.md`。

## WorldSim V6.6 P7R sensor-supported surface repair implementation ready（2026-08-28）

状态：`v66_p7r_sensor_supported_surface_repair_implementation_ready`。已实现冻结L0 action set上的point-level repair：
acted Actor-owned native boundary只保留映射到same-Actor motion-compensated hit的primitive，其他写UNKNOWN；未action
保持原geometry。输出实际`REPAIRED_ACTOR_BOUNDARY.npz`与per-state metrics。target FREE只在retention rule结束后评估；
Actor canonical shell/ID/track/trajectory/hazard attributes不改。formal run前只做`py_compile`/config/diff检查，下一步一次
CPU+I/O run，不扫budget/threshold。

## WorldSim V6.6 P7 triage supported / P7R physical-surface recovery frozen（2026-08-28）

状态：`v66_p7_triage_supported_p7r_sensor_surface_recovery_frozen`；canonical=
`run://worldsim_v66/WS-V66-P7-HAZARD-PRESERVING-DISTRIBUTION-01/20260828T092919Z__fixed-budget-distribution-s0-r1`；
verdict=`supported_consumed_legacy_fixed_budget_exposure_audit`；新增`V66-F02 active`。

固定290/581 action budget下，L0处理210/307 conflicts，exposure reduction=`0.684039`，优于Q0的`0.628664`；
emitted local fraction=`0.500861`、Actor retention=1、removed=0、hazard proxy shift=0、scene yield=1，6/6 gates通过。
Q0 scene yield仅`0.8333`，显示actor-blind q0可把某一scene local geometry全部送入action；L0未发生。

但P7没有改physical geometry，handled只能称triage，不能称repair或RL-ready。按卡点检索NeuRAD、Neural Scene Graphs、
Cam4DOcc及其official code后，P7R冻结为sensor-supported actor-local surface repair：沿用L0固定action set，将acted Actor
boundary中无same-Actor motion-compensated hit的primitive转UNKNOWN；canonical Actor collision shell/track/hazard保持。
target只作评估。迁移：`docs/autoresearch/worldsim_v66/P7R_SENSOR_SUPPORTED_REPAIR_MIGRATION_FREEZE.md`。

## WorldSim V6.6 P7 fixed-budget distribution audit implementation ready（2026-08-28）

状态：`v66_p7_fixed_budget_distribution_audit_implementation_ready`。已冻结50% local-action budget与N0/Q0/D0/L0/O0
matched arms；不扫budget/threshold。所有臂保留相同Actor/state/hazard attributes，action只是repair/abstain候选，
不改physical geometry。primary estimand是unhandled local-conflict exposure；L0 gates固定为reduction>=50%、Actor
retention=1、removed=0、hazard proxy shift=0、emitted local geometry>=50%、world scene yield=1。下一步一次CPU formal
audit；通过也不称physical repair或RL-ready。

## WorldSim V6.6 P6 HARP bake supported / P7 distribution audit next（2026-08-28）

状态：`v66_p6_harp_bake_supported_p7_distribution_audit_next`；canonical=
`run://worldsim_v66/WS-V66-P6-HARP-BAKE-01/20260828T092421Z__harp-bake-s0-r1`；verdict=
`supported_consumed_legacy_harp_bake_capability`。

72 units烘焙127 unique Actors、581 Actor states和1,623,503 Actor primitives；8文件共16,321,358 bytes。
Actor state retention/metadata completeness=`1/1`，removed/hidden-target fields=`0/0`，6/6 gates通过。
runtime model/hidden-target loading=false，hazard-existence coupling=false；wall=`15.57s`、RSS=`0.82530GiB`。

该capability没有实际改geometry。下一 active=`WS-V66-P7-HAZARD-PRESERVING-DISTRIBUTION-01`：固定50% local-action
budget，在Actor/hazard attributes完全保留的 matched arms 比较未处理local-conflict exposure；不得把ABSTAIN记为真实
repair，也不得用confirmation label调budget或阈值。

## WorldSim V6.6 P6 HARP bake implementation ready（2026-08-28）

状态：`v66_p6_harp_bake_implementation_ready`。已实现八文件Actor-preserving runtime package：static tri-state使用
RLE，Actor current-envelope primitives独立存储；ACTORS包含class、track/lifecycle与采样trajectory，factor/repair/
hazard/provenance均显式JSONL。bake从P3C score只读取observable模型输出字段，不读/输出target label。runtime manifest
固定model/hidden-target loading=false、hazard-existence coupling=false。无binary threshold，所有Actor保留，action仅
`RANK_REPAIR_OR_ABSTAIN`且geometry mutation=false。下一步一次formal CPU bake；不做重复replay或测试矩阵。

## WorldSim V6.6 P3C independent confirmation supported / P6 bake next（2026-08-28）

状态：`v66_p3c_independent_confirmation_supported_p6_bake_next`；canonical=
`run://worldsim_v66/WS-V66-P3C-INDEPENDENT-LOCAL-GEOMETRY-CONFIRM-01/20260828T091611Z__independent-local-geometry-confirm-s0-r1`；
verdict=`supported_independent_legacy_local_geometry_confirmation`。

冻结P3L在scene-disjoint V65 P2V 72 units上得到581 actor-unit（conflict/clean=`307/274`）。AUROC/AUPRC=
`0.761644/0.767165`，相对deterministic=`+0.261644/+0.238766`，相对q0=`+0.062127/+0.028200`；
6/6 evaluable scenes above chance，4/4 gates通过。wall=`10.77s`、GPU=`0.02359GiB`、RSS=`0.93375GiB`。
无model/normalization refit、threshold或第二confirmation。

`V66-F01`按两级certificate恢复关闭：deterministic层保护Actor existence，learned层只排序local geometry
REPAIR/ABSTAIN。结论仍限consumed legacy ranking，不是fresh V6.6或真实修复。下一 active=`WS-V66-P6-HARP-BAKE-01`：
将冻结offline compiler输出为不携带模型/hidden target/hazard existence gate的Actor-preserving runtime package。

## WorldSim V6.6 P3C independent confirmation frozen / implementation ready（2026-08-28）

状态：`v66_p3c_independent_confirmation_implementation_ready`。确认集固定为V65 P2V consumed六场景/72 units，
与P10V train、P10X selection均scene-disjoint；复用既有evidence/native，不重跑3.3GB sidecar。runner只加载P3L
checkpoint/normalization并threshold-free评分，禁止refit/sweep/second confirmation。gates=`AUROC +0.03`、
`AUPRC +0.05`相对constant，以及至少4个scene above chance；Actor existence authority继续关闭。窄验证只做
`py_compile`与diff检查，下一步exact-once formal read。

## WorldSim V6.6 P3L legacy selection supported / independent confirmation next（2026-08-28）

状态：`v66_p3l_legacy_selection_supported_independent_confirmation_next`；canonical=
`run://worldsim_v66/WS-V66-P3L-ACTOR-LOCAL-GEOMETRY-HEAD-01/20260828T091036Z__local-geometry-head-s0-r1`；
verdict=`supported_legacy_selection_local_geometry_head`。

P10V train含409 actor-unit（conflict/clean=`243/166`），P10X single selection含891（`498/393`）。固定head在
selection上AUROC/AUPRC=`0.652365/0.692384`，相对deterministic constant增加`+0.152365/+0.133461`，
相对q0增加`+0.108620/+0.079510`；6/6 evaluable scenes均above chance，4/4 gates通过。wall=`13.07s`、
GPU=`0.02359GiB`、RSS=`1.08285GiB`。无feature/architecture/seed/threshold sweep。

这只支持consumed legacy local-geometry ranking；模型只能为local REPAIR/ABSTAIN排序，Actor existence authority保持
关闭。`V66-F01`进入recovering但尚未关闭，下一步冻结另一独立V65 cohort，加载同一checkpoint/normalization exact-once
确认，不refit、不解析threshold。结果：`docs/autoresearch/worldsim_v66/P3L_LOCAL_GEOMETRY_HEAD_RESULT.md`。

## WorldSim V6.6 P3L local geometry head implementation ready（2026-08-28）

状态：`v66_p3l_local_geometry_head_implementation_ready`。已实现冻结8维instance-evidence summary、2x32 MLP、
seed0/full-batch weighted BCE，以及P10V train→P10X single selection。selection只看AUROC/AUPRC与per-scene
above-chance support，不解析threshold；模型输出不能删除Actor。窄验证仅`py_compile`与diff检查；下一步唯一一次训练/
selection，失败则关闭该feature family或先检索新机制，不在P10X扫参。

## WorldSim V6.6 P2N natural conflict diagnosed / P3L migration frozen（2026-08-28）

状态：`v66_p2n_deterministic_natural_ceiling_p3l_frozen`；canonical=
`run://worldsim_v66/WS-V66-P2N-NATURAL-ACTOR-CONFLICT-DIAGNOSTIC-01/20260828T090228Z__natural-actor-conflict-s0-r1`；
verdict=`diagnosed_natural_actor_local_geometry_conflict`；新增`V66-F01 active`。

72 units得到891 eligible actor-unit，其中observed-FREE local geometry conflict/clean=`498/393`，prevalence=
`0.5589`。q0 conflict AUROC/AUPRC=`0.543745/0.612874`，q0与hidden-FREE rate Spearman=`0.267650`；
deterministic certificate conflict recall=`0`、AUROC=`0.5`、AUPRC=`0.558923`，clean false conflict=0。诊断分母/
双类checks通过，wall=`10.99s`、GPU=`0.0236GiB`、RSS=`0.918GiB`。这推翻“Actor级support certificate足以
覆盖natural actor-owned local geometry conflict”，但不推翻Actor existence protection；local conflict不得用于删Actor。

按卡点协议完成外部检索并冻结最小迁移：Symphonies支持instance-query contextual geometry，GaussianFormer支持
object-centric sparse summary，Cam4DOcc强调instance temporal occupancy，CVPR evidential occupancy说明二值LiDAR
occupancy label质量与unknown/contradiction需要显式处理。项目迁移为两级certificate：existence继续用track/provenance，
local geometry只训练一个低容量instance-evidence MLP，特征固定为q0 mean/p90与boundary/hit/current/swept count及两个
density ratio。P10V作train、已读P10X只作selection；不扫architecture/seed/threshold。若selection ranking相对constant
certificate有预注册增量，再在另一独立V65 cohort exact-once确认。

Active=`WS-V66-P3L-ACTOR-LOCAL-GEOMETRY-HEAD-01 / WS-V66-H-P3L-001`。冻结：
`docs/autoresearch/worldsim_v66/P3L_INSTANCE_EVIDENCE_MIGRATION_FREEZE.md`；P2N结果：
`docs/autoresearch/worldsim_v66/P2N_NATURAL_ACTOR_CONFLICT_RESULT.md`；实现提交=`a92093b`。

## WorldSim V6.6 P2N natural-conflict diagnostic implementation ready（2026-08-28）

状态：`v66_p2n_natural_conflict_implementation_ready`。实现复用P1的CPU预取/GPU q0流水线，但在与P1场景不重叠的
P10X六场景读取target evidence，将Actor-grounded native boundary中`hidden_free_count>0`冻结为local geometry conflict。
该label只判局部owned geometry，不判Actor existence；不扫count/rate threshold，不refit q0/certificate。下一步唯一一次
diagnostic read；若deterministic证书出现natural ceiling，先检索相关顶会/优秀开源再设计迁移。

## WorldSim V6.6 P4-D repair-first supported / natural-conflict diagnostic next（2026-08-28）

状态：`v66_p4_development_repair_supported_p2n_natural_conflict_next`；canonical=
`run://worldsim_v66/WS-V66-P4-ARTIFACT-REPAIR-DEV-01/20260828T085755Z__repair-first-dev-s0-r1`；
verdict=`supported_development_repair_first_compiler`。

matched三臂各8,180 rows（compiled 24,540）。artifact rows的4,908个reason violations：R0 DROP降至0，但
all-hazard event retention=`0.50`、shift=`0.50`；R1 ABSTAIN保留hazard=`1.0`，但violation reduction=`0`；
R2 REPAIR降至0且hazard retention=`1.0`、shift=0、clean-hazard retention=1、Actor ID/track/trajectory exact=1、
nonartifact regression=0、hard evidence violation=0。R2 6/6 gates通过。CPU wall=`0.546s`、RSS=`0.574GiB`，
failure delta=`none`。

结论只到paired observable-factor repair：不支持RGB/full-scene geometry、natural artifact或fresh generalization。
下一 active=`WS-V66-P2N-NATURAL-ACTOR-CONFLICT-DIAGNOSTIC-01 / WS-V66-H-P2N-001`。使用与P1-D场景不重叠的
V65 P10X六场景，直接把Actor-grounded boundary中的target observed-FREE定义为local geometry conflict，测试q0与
deterministic factor certificate的cross-cohort ceiling；仍是已消费legacy diagnostic，不是fresh V6.6 claim。

结果：`docs/autoresearch/worldsim_v66/P4_DEVELOPMENT_REPAIR_RESULT.md`；实现提交=`e371489`。

## WorldSim V6.6 P4-D repair implementation ready（2026-08-28）

状态：`v66_p4_development_repair_implementation_ready`。matched三臂共享同一8,180行输入：DROP移除被证书判定的
artifact actor；ABSTAIN保留Actor但不发出局部geometry；REPAIR从同cluster clean reference恢复可观测factor，Actor
ID/track/trajectory/hazard属性不变。formal run只以R2 gate裁决，并完整报告R0/R1 tradeoff；无natural smoothing、learned
training、fresh read、sweep或hash/checksum/fingerprint。

## WorldSim V6.6 P2-D certificate supported / P4 repair next（2026-08-28）

状态：`v66_p2_development_certificate_supported_p4_repair_next`；canonical=
`run://worldsim_v66/WS-V66-P2-FACTOR-CERTIFICATE-DEV-01/20260828T085346Z__factor-certificate-dev-s0-r1`；
verdict=`supported_development_factor_certificate`。

独立入口从P1的observable factor重算8,180行certificate，不复用P1 decision。pooled artifact recall/AUROC/AUPRC=
`1/1/1`，五family recall均1；clean-hazard/clean-benign false artifact均0，legitimate hazard/benign retention均1，
Actor existence/ID/lifecycle retention均1，hazard-pair score delta=0，hard observed evidence violation=0。8/8 gates
全过。decision counts=`KEEP 4090 / ABSTAIN_LOCAL_GEOMETRY 818 / DROP_ARTIFACT_PRIMITIVE 818 / REPAIR 2454`。
CPU wall=`0.249s`、RSS=`0.532GiB`，failure delta=`none`。

该deterministic injected development benchmark已无learned ranking headroom，因此P3 learned family保持
`pending_locked_not_executed`，不浪费GPU做必然无法满足“相对P2 AUPRC +0.05”的训练。它不是算法failure；自然/fresh
benchmark若暴露deterministic ceiling才可解锁。active next=`WS-V66-P4-ARTIFACT-REPAIR-DEV-01 /
WS-V66-H-P4D-001`：matched比较DROP、ABSTAIN、REPAIR对artifact violation和hazard event的影响。

结果：`docs/autoresearch/worldsim_v66/P2_DEVELOPMENT_CERTIFICATE_RESULT.md`；实现提交=`38c67dc`。

## WorldSim V6.6 P2-D certificate implementation ready（2026-08-28）

状态：`v66_p2_development_certificate_implementation_ready`。独立P2入口只读取P1 rows中的可观测factor，重新计算
PASS/FAIL reason codes与KEEP/REPAIR/ABSTAIN/DROP_PRIMITIVE action；不信任P1已写certificate字段，也不把artifact/hazard
metadata送入certificate。Actor existence/ID/lifecycle在certificate阶段一律保留。下一步运行一次P2-D gate；无模型训练、
threshold sweep、fresh read、hash/checksum/fingerprint或测试矩阵。

## WorldSim V6.6 P1-D factorial atlas supported / P2 certificate next（2026-08-28）

状态：`v66_p1_development_supported_p2_certificate_next`；canonical=
`run://worldsim_v66/WS-V66-P1-VALIDITY-HAZARD-SEPARATION-ATLAS-DEV-01/20260828T084915Z__factorial-atlas-dev-s0-r1`；
verdict=`supported_development_factorial_separation_proceed_to_p2`。

72个已消费P10V units中得到409个eligible actor-unit base，五个artifact family各形成 paired cluster，共2,045 clusters /
8,180 rows，四象限各2,045。冻结q0在同base的representation-level corruption/hazard pair中按合同保持原分数，故
artifact与hazard AUROC/AUPRC均=`0.50/0.50`；这证明actor-blind baseline在该构造中无响应，不代表重渲染artifact后的
empirical q0 failure。reason-coded factor certificate的artifact AUROC/AUPRC=`1.0/1.0`，五family recall均1.0，
clean-hazard false artifact=`0`、legitimate hazardous retention=`1.0`、hazard-pair q0/certificate delta均=`0`。

4/4 development gates通过，failure ledger delta=`none`。资源=`8.03s / 0.0236GiB GPU / 0.910GiB RSS`；I/O预取与
GPU forward按计划重叠，单卡足够。该构造性满分只授权P2确定性证书与P4 repair capability，不授权learned superiority、
自然artifact、fresh selection、真实hazard edit、planning/RL/safety。active next=`WS-V66-P2-FACTOR-CERTIFICATE-DEV-01 /
WS-V66-H-P2D-001`；先把 reason codes/Actor existence protection 编译为独立certificate输出，再比较DROP/ABSTAIN/REPAIR。

结果：`docs/autoresearch/worldsim_v66/P1_DEVELOPMENT_FACTORIAL_ATLAS_RESULT.md`。执行源码提交=`0374315`；
结果文档由本次里程碑提交记录。

## WorldSim V6.6 P1-D evaluator implemented / formal run next（2026-08-28）

状态：`v66_p1_development_atlas_implementation_ready`。已实现 Actor envelope grounding、hit/current/swept support
统计、冻结 q0 GPU forward、entropy/margin聚合、五类可观测 artifact factor、四象限 paired rows 与 hazard-pair
invariance评估。每个unit由CPU预取，当前unit在GPU forward，避免全量I/O barrier。推理不读取 artifact family/label、
hazard label/score或variant ID；q0在representation-level corruption对中保持原score，用于检验Actor-blind结构边界，
不冒充重新渲染后的q0。

窄验证仅为三文件 `py_compile` 与 `git diff --check`，均通过；没有smoke/regression matrix、质量read、模型训练或
hash/checksum/fingerprint。下一步只运行 `WS-V66-P1-VALIDITY-HAZARD-SEPARATION-ATLAS-DEV-01` 一次并按结果更新
三账本；若存在q0未响应而factor可辨识的family，直接进入P2确定性证书。

## WorldSim V6.6 启动 / P1-D Actor factorial atlas 直接执行（2026-08-28）

状态：`v66_p1_development_atlas_running`；分支=`research/worldsim-v6.6-harp-compiler`。`main` 已按用户要求从
`origin/main` 快进合入 V6.5 终态 `288fa9f` 并推送，V6.6 随后从该 `main` 建立并推送。V6.5 terminal claim
保持：给定 Ego `tau` 的 visited-state reliability evaluator/calibrator 成立，direct action authority 因
`V65-F19` 终止；V6.6 不复开该 authority family，而改研究 Actor-preserving artifact validity。

当前 active task/hypothesis=`WS-V66-P1-VALIDITY-HAZARD-SEPARATION-ATLAS-DEV-01 /
WS-V66-H-P1D-001`。为直接进入 research，首轮使用 P10V 已消费的 6 scenes / 72 units 作为 Tier-L development
mechanism source，将 native boundary q0、Actor envelope/hit support 与确定性 artifact factor 组成
`validity x hazard` 配对 atlas。hazard intervention 首轮只改变任务属性，不冒充真实 cut-in 几何编辑；formal fresh
selection、自然 artifact 与 closed-loop claim 均未解锁。

P0 最小冻结已完成：Actor/SceneIR/evidence/r13/cut-in 资产审计、validity-hazard taxonomy、输入角色和禁止事项已落盘。
单 RTX 3090 足够；P1 q0 forward 与 CPU evidence materialization 采用预取重叠，不等待全量 I/O 后才用 GPU。
不新增 hash/checksum/fingerprint，不执行 P0 测试矩阵、宽 smoke 或回归测试。下一步：运行 P1-D atlas，若 deterministic
factor 对至少一种 q0 未响应 artifact 有信号，则直接实现 P2 certificate；若遇实现/算法卡点，先检索论文与优秀开源再迁移。

证据入口：`docs/autoresearch/worldsim_v66/V66_ACTOR_ASSET_AUDIT.md`、
`docs/autoresearch/worldsim_v66/V66_VALIDITY_HAZARD_TAXONOMY.yaml`、
`configs/worldsim_v66/p1_factorial_atlas_dev_v1.yaml`。基线提交=`288fa9f`；本里程碑提交由同次 Git 提交记录。

## WorldSim V6.4 arXiv handoff validated / remote shutdown authorized（2026-08-27）

状态保持`v64_research_complete_report_ready`；新增报告交接入口=
`docs/autoresearch/worldsim_v64/V64_ARXIV_REPORT_HANDOFF.md`。本次只读审计确认P6R、P4C、P10R2、P10R4、P11、P11R、
P11D共7个canonical run目录存在，核心`summary.json/status.json`均可解析，逐case/action与模型产物按stage保留；三本强制
账本、terminal state、arXiv evidence index和family closeout一致。无新科学执行、GPU run、smoke/regression matrix或failure ID，
也未加入hash/checksum/fingerprint。推送成功后按用户授权关闭AutoDL；V6.4内部无剩余task或unlocked stage。

## WorldSim V6.4 research complete / report ready（2026-08-27）

状态：`v64_research_complete_report_ready`；V6.4科学执行终止，active task/hypothesis=`null/null`；本里程碑无新实验、
无新failure ID。最终报告入口=`docs/autoresearch/worldsim_v64/ARXIV_EVIDENCE_INDEX.md`，版本收口=
`docs/autoresearch/worldsim_v64/V64_RESEARCH_FAMILY_CLOSEOUT.md`。

最强正证据为P10R4 untouched 96-case fixed-opportunity exact-once：M0/M1 mean coverage同为`0.474969689`，M1使
worst10 CVaR从`0.020725740`降至`0.010821074`，pooled density从`0.004944667`降至`0.002001413`，paired
lower/equal/higher=`18/78/0`。最强负证据为P11/P11R：verified unsafe-action recall在原始/独立校准后evaluation仅
`0.01087/0.62044`，P11D又确认unsafe prior与ranking同时漂移（AUROC `0.71165->0.56274`），故collision critic family
terminal negative。支持边界只到原生不确定性、条件选择性状态编译和固定分母的exact empirical route-local risk；不支持
population bound、physical collision、planning、closed-loop、RL或safety。单RTX3090足够，无多卡需求。

## WorldSim V6.4 P11D calibration-shift diagnostic complete（2026-08-27）

状态：`v64_p11d_shift_diagnosed_report_closeout_next`；canonical=
`run://worldsim_v64/WS-V64-P11D-COLLISION-CRITIC-SHIFT-DIAGNOSTIC-01/20260827T040000Z__collision-critic-shift-s0-r1`；
verdict=`diagnosed_p11_cross_cohort_score_and_prior_shift`。

calibration/evaluation unsafe prior=`0.07051/0.10978`，delta=`+0.03926`。verified unsafe q20/median score下移
`-0.05328/-0.13745`，但safe median仅`-0.000025`；AP从`0.24710`降到`0.13740`，AUROC从`0.71165`降到
`0.56274`。naive也出现AP/AUROC下降`-0.09030/-0.09310`。因此P11R失败不是可用单一阈值平移修复的prior shift，而是
unsafe ranking跨cohort退化；V64-F28/P11 terminal negative得到机制诊断支持但不产生新authority。rows-only CPU wall/RSS=
`0.0986s/0.1957GiB`。closeout=`docs/autoresearch/worldsim_v64/P11D_COLLISION_CRITIC_SHIFT_DIAGNOSTIC_CLOSEOUT.md`。

## WorldSim V6.4 P11D rows-only calibration-shift diagnostic frozen（2026-08-27）

状态：`v64_p11d_rows_only_shift_diagnostic_preregistered`；active task/hypothesis=
`WS-V64-P11D-COLLISION-CRITIC-SHIFT-DIAGNOSTIC-01 / WS-V64-H-P11D-001`。

只读P11R已落盘calibration/evaluation action rows与threshold，报告unsafe prior、AP、AUROC、safe/unsafe score quantile及
cross-cohort delta；无gate、native/evidence reread、GPU、refit、threshold/policy change或P11复开。该结果只服务V64-F28
failure characterization与技术报告。freeze=`docs/autoresearch/worldsim_v64/P11D_COLLISION_CRITIC_SHIFT_DIAGNOSTIC_FREEZE.md`。

## WorldSim V6.4 P11R independently calibrated critic rejected / P11 closed（2026-08-27）

状态：`v64_p11r_rejected_p11_closed`；canonical=
`run://worldsim_v64/WS-V64-P11R-CALIBRATED-COLLISION-CRITIC-01/20260827T034500Z__calibrated-collision-critic-s0-r1`；
verdict=`rejected_independently_calibrated_collision_critic`；V64-F28=`closed_negative_after_single_recovery`。

P10R2 calibration的Real-only/naive/verified threshold=`4.25e-18/0.191678/0.084891`；verified calibration recall=
`0.79545`。P4C exact evaluation中verified recall=`0.62044<0.80`，policy false-safe/progress/stuck=`2/0.87240/0.11458`；
Real-only以`96/96` fallback stop获得false-safe0但progress0/stuck1，故comparison与anti-trivial不可同时满足。四门仅progress/stuck
通过，recall与false-safe不劣失败。critic未重训，threshold先落盘，evaluation无selection。P11 family以负结论关闭；不得调分位、
换threshold/lattice/model、第二evaluation或训练大型NWM/RL。GPU wall/peak=`26.5528s/0.0658GiB`。closeout=
`docs/autoresearch/worldsim_v64/P11R_CALIBRATED_COLLISION_CRITIC_CLOSEOUT.md`。

## WorldSim V6.4 P11R independent threshold calibration frozen（2026-08-27）

状态：`v64_p11r_independent_threshold_calibration_preregistered`；active task/hypothesis=
`WS-V64-P11R-CALIBRATED-COLLISION-CRITIC-01 / WS-V64-H-P11R-001`；evaluation action labels read=`false`。

V64-F28恢复不重训三critic、不改13-action lattice：P10R2 downstream actions独立校准，每臂以unsafe score的20%分位解析得到
target recall=0.80的唯一threshold，无grid；threshold先落盘，之后P4C 96-case从未生成的action labels exact-once。gate只保留
verified recall>=0.80、policy false-safe不劣及progress>=0.50/stuck<=0.20。P10R4标签不得进入恢复。large NWM/RL继续锁定。
freeze=`docs/autoresearch/worldsim_v64/P11R_CALIBRATED_COLLISION_CRITIC_FREEZE.md`。

## WorldSim V6.4 P11 bounded collision critic primary gate pass / unsafe recall rejected（2026-08-27）

状态：`v64_p11_primary_gate_passed_unsafe_recall_rejected_recovery_next`；canonical=
`run://worldsim_v64/WS-V64-P11-BOUNDED-COLLISION-CRITIC-01/20260827T033000Z__bounded-collision-critic-s0-r1`；formal
verdict=`supported_bounded_unc_verified_collision_critic`；V64-F28=`active_unsafe_recall_collapse`。

Real-only/naive/verified selected-policy false-safe=`13/12/12`，mean progress均=`1.0`、stuck均=`0`，所以三项冻结gate PASS；
但1248-action denominator上unsafe recall仅=`0.02174/0/0.01087`，false-safe=`180/184/182`。verified与naive在policy
false-safe/reward完全相同，且Brier/ECE更差=`0.1743/0.1802` vs `0.1618/0.1441`，故UNC verification没有独立增量，
三臂critics均不能作为collision authority。GPU wall/peak=`26.6467s/0.0737GiB`。合法恢复只允许用未读downstream action
labels作独立解析阈值校准，再在另一未读action-label cohort exact-once；不得在已读P10R4调0.5门或重训。closeout=
`docs/autoresearch/worldsim_v64/P11_BOUNDED_COLLISION_CRITIC_CLOSEOUT.md`。

## WorldSim V6.4 P11 bounded collision critic frozen（2026-08-27）

状态：`v64_p11_bounded_collision_critic_preregistered`；active task/hypothesis=
`WS-V64-P11-BOUNDED-COLLISION-CRITIC-01 / WS-V64-H-P11-001`；evaluation action labels read=`false`。

参考Waymax、nuPlan、InterFuser与PlanT后，冻结单卡最小迁移：不用大型NWM/CARLA，P6R consumed cohort训练同一linear
critic；P10R4每case固定`3 lateral x 4 progress + stop=13` actions。三臂为Real-only、Real+all generated、Real+M1最低风险
half generated。主指标是selected-policy collision false-safe；仅加progress>=0.50与stuck<=0.20防全刹车。其余recall、
safe precision、Brier/ECE、comfort/reward只报告。模型在evaluation action label生成前冻结；无action/model/threshold/test sweep。
freeze=`docs/autoresearch/worldsim_v64/P11_BOUNDED_COLLISION_CRITIC_FREEZE.md`。

## WorldSim V6.4 P10R4 exact-once fixed-denominator relative confirmation supported（2026-08-27）

状态：`v64_p10r4_exact_empirical_relative_supported_p11_bounded_design_next`；canonical=
`run://worldsim_v64/WS-V64-P10R4-FIXED-DENOMINATOR-EXACT-ONCE-01/20260827T025000Z__exact-once-fixed-denominator-s4-r1`；
verdict=`supported_exact_once_fixed_denominator_relative_confirmation`。

untouched 96-case中M0/M1 mean total coverage均=`0.474969689`；fixed-denominator worst10 CVaR=
`0.020725740/0.010821074`，M1-M0=`-0.009904666`；pooled conflict density=`0.004944667/0.002001413`，
delta=`-0.002943254`。paired cases为M1 lower/equal/higher=`18/78/0`，描述性half-tie probability=`0.59375`。
三项冻结gate全部PASS；无refit、runtime selection、sweep或second test。V64-F25仅在独立exact empirical cohort层面关闭，
P10R2 formal和V64-F21 current-M0负结论不改写；population/collision/planning/closed-loop/safety仍不支持。P11只解锁
不训练大型NWM的有界设计审计。GPU wall/peak RSS=`11.6233s/0.8798GiB`。closeout=
`docs/autoresearch/worldsim_v64/P10R4_FIXED_DENOMINATOR_EXACT_ONCE_CLOSEOUT.md`。

## WorldSim V6.4 P10R4 untouched-test evidence complete / exact-once next（2026-08-27）

状态：`v64_p10r4_test_evidence_complete_exact_once_next`；canonical=
`run://worldsim_v64/WS-V64-P10R4-TEST-EVIDENCE-01/20260827T023500Z__test-evidence-s4-r1`；test target/model-score read=`true/false`。

冻结test cohort一次生成`8 scenes / 96 units / 118958863 bytes`，reuse=`0`、source-role overlap=`0`、queries=`0`，
maximum unit/wall=`16.8897/111.9646s`，passed。没有恢复、policy/model/route/tail/denominator/gate变更或第二份evidence；
failure ledger delta=`none`。下一步只运行冻结M0/M1 fixed-denominator exact-once一次。closeout=
`docs/autoresearch/worldsim_v64/P10R4_TEST_EVIDENCE_CLOSEOUT.md`。

## WorldSim V6.4 P10R4 untouched-test native complete / evidence next（2026-08-27）

状态：`v64_p10r4_test_native_complete_evidence_next`；canonical native=
`run://worldsim_v64/WS-V64-P10R4-TEST-SIDECAR-01/20260827T023000Z__native-aggregate-s4-r1`；test quality read=`false`。

冻结8 scene已全部完成canonical processed与native sidecar，aggregate=`8 scenes / 96 targets / 4423846058 bytes / passed`，
最大worker显存=`4.1314GiB`。双preprocess/双native feeder在最后两scene就绪后的native等待仅`0.0646/0.0625s`；
finalizer只复用8个complete scene，`0.8328s`登记后删除约`6.2GiB`可重建raw。V64-F27由精确镜像DriveStudio路径重写并
复用complete stage/native关闭。没有读取test target/quality/model score，也没有增加hash/checksum/fingerprint或测试矩阵。
下一步只生成一次冻结96-unit evidence，再执行一次fixed-denominator exact-once。closeout=
`docs/autoresearch/worldsim_v64/P10R4_TEST_SIDECAR_CLOSEOUT.md`。

## WorldSim V6.4 P10R4 dual-stage path recovery frozen（2026-08-27）

状态：`v64_p10r4_dual_stage_path_recovery_frozen`；V64-F27=`active_recovery`；test quality read=`false`。

双preprocess已完整生成scene-1084/1081，但DriveStudio把`..._processed_824`改写为
`..._processed_10Hz_824/trainval/824`，feeder却查找`..._824_10Hz`并在native前退出。824/821 stage counts=
`1206/201`与`1176/196` images/lidar，无native partial。修复仅镜像既有路径重写；parent停止，唯一in-flight 424/522
完成后连同824/821原子安装。night两scene用同一冻结native命令/计划run dir直接填GPU，之后同prefix feeder复用全部complete
leaf。科学合同不变。freeze=`docs/autoresearch/worldsim_v64/P10R4_DUAL_STAGE_PATH_RECOVERY_FREEZE.md`。

## WorldSim V6.4 P10R4 raw recovery complete / native streaming（2026-08-27）

状态：`v64_p10r4_raw_complete_native_streaming`；canonical raw=
`run://worldsim_v64/WS-V64-P10R4-TEST-SIDECAR-01/20260827T022000Z__test-raw-shard-recovery-s4-r2`；V64-F26=`resolved`。

restricted 05/06/07/08/10在`1807.8114s`找齐`14437/14437` members，命中=`5401/1824/1818/1783/3611`；catalog=
`85992 entries / 10318384 bytes`，temporary raw=`~6.2GiB`，free disk=`~21GiB`。test quality/target/model score均未读。
双preprocess feeder已有scene-0598/0462两份完整native，均`12/12` targets、peak GPU=`4.1314GiB`、wall=
`45.4004/45.3845s`；其余scene沿用同prefix并复用完整leaf。closeout=
`docs/autoresearch/worldsim_v64/P10R4_RAW_SHARD_RECOVERY_CLOSEOUT.md`。

## WorldSim V6.4 P10R4 I/O shard recovery frozen（2026-08-27）

状态：`v64_p10r4_dual_preprocess_feeder_recovery_frozen`；V64-F26=`active_recovery`；test quality read=`false`。

首个raw-only producer发现`14437` members均不在catalog，10个完整tgz并发约4分钟仅推进`4--10%`，GPU feeder无scene可用。
检索CPython tarfile与ratarmount/rapidgzip后，未为一次性cohort新建十份seek index；迁移现有`71555`条semantic catalog的
capture-prefix证据，将扫描范围冻结为`05,06,07,08,10`。七scene由exact prefix唯一映射，`0668`由相邻temporal scene范围与
已落盘同prefix files绑定07。停止all-shard workers后保留原子完成文件，只清理其process-scoped partial；feeder继续作为唯一
preprocess/native owner。科学cohort/model/policy/target/route/denominator/tail/gates不变。recovery=
`docs/autoresearch/worldsim_v64/P10R4_IO_SHARD_RECOVERY_FREEZE.md`。

首次restricted r1在扫描前因resume目录的`mkdir(exist_ok=false)`退出，未新增I/O或target read；定向修复只在显式
`--resume-raw-scan`下允许既有冻结目录，默认防覆盖语义不变。canonical recovery改为
`20260827T022000Z__test-raw-shard-recovery-s4-r2`。

scene-0598 native已完成（`45.4004s/4.1314GiB`），但单preprocess mutex使GPU在scene-0462超过2分钟转换时空闲。
保留0598完整native并让in-flight 0462完成后重启同prefix feeder；每scene使用独立staging，preprocess/native slots均冻结为2。
该CPU/GPU分队列恢复不改scene、模型、policy、target或任何gate。

## WorldSim V6.4 P10R4 untouched-test fixed-denominator confirmation frozen（2026-08-27）

状态：`v64_p10r4_untouched_test_preregistered`；active task/hypothesis=
`WS-V64-P10R4-TEST-SIDECAR-01 / WS-V64-H-P10R4-001`；test quality read=`false`。

seed4从未引用且temporal-valid metadata按night/rain/construction/vulnerable各冻结2 scene：
`1084,1081 / 0462,0820 / 0534,0598 / 0527,0668`，共96 cases。M0/M1、route `2s/1.5m`、M1 cap `0.40`、
fixed route-eligible denominator和worst10全部不变。唯一三门为coverage delta `<=1e-6`、M1-M0 fixed-CVaR `<=0`、
M1-M0 pooled fixed density `<=0`；无bootstrap/significance/refit/sweep/second test。元数据成员改为单遍解析；raw-only producer
与单一feeder重叠tar I/O、预处理和最多2个单scene GPU worker，避免duplicate producer。free disk=`27GiB`、预计峰值新增
`~14GiB`、单worker显存`~4.2GiB`，不需要多卡。freeze=`docs/autoresearch/worldsim_v64/P10R4_UNTOUCHED_TEST_FREEZE.md`。

## WorldSim V6.4 P10R3 fixed route-denominator diagnostic complete（2026-08-27）

状态：`v64_p10r3_fixed_denominator_diagnostic_complete`；canonical=
`run://worldsim_v64/WS-V64-P10R3-FIXED-DENOMINATOR-AUDIT-01/20260827T013000Z__fixed-denominator-audit-s0-r1`；verdict=
`diagnosed_fixed_denominator_direction_consistent`。

固定route-eligible分母后，consumed calibration的M0/M1 pooled density=`0.00236358/0.000924879`、worst10 CVaR=
`0.0132351/0.00455240`（delta=`-0.00868274`）；fresh confirmation对应=`0.00421776/0.00156213`与
`0.0216470/0.0149832`（delta=`-0.00666382`）。方向在两cohort一致，支持selected-only可变分母解释，但该指标在读过
confirmation后冻结，故仅为post-hoc诊断；V64-F25保持active、P11 comparative authority继续锁定。target/model/evidence
reread=`false`，policy/sweep=`none`，CPU wall/peak RSS=`0.00264s/0.1680GiB`。closeout=
`docs/autoresearch/worldsim_v64/P10R3_FIXED_ROUTE_DENOMINATOR_AUDIT_CLOSEOUT.md`。

## WorldSim V6.4 P10R3 fixed route-denominator diagnostic frozen（2026-08-27）

状态：`v64_p10r3_fixed_denominator_diagnostic_preregistered`；active task/hypothesis=
`WS-V64-P10R3-FIXED-DENOMINATOR-AUDIT-01 / WS-V64-H-P10R3-001`。

V64-F25显示selected-only rate在M1更小分母下尾部方向反转。参考Waymo fixed ego-grid AUC/Soft-IoU与Implicit Occupancy
Flow fixed spatial query，冻结rows-only诊断为每case `route conflict count / route-eligible count`，该分母在两臂间固定；在consumed
calibration与fresh confirmation分别算同一worst10。只报告direction，不设confirmatory gate，不重读target/model/evidence、不改M1、
不扫denominator/tail/route。结果不能关闭V64-F25或解锁P11。freeze=
`docs/autoresearch/worldsim_v64/P10R3_FIXED_ROUTE_DENOMINATOR_AUDIT_FREEZE.md`。

## WorldSim V6.4 P10R2 exact-once absolute tail supported / relative effect not confirmed（2026-08-27）

状态：`v64_p10r2_exact_once_absolute_supported_relative_tail_active`；canonical=
`run://worldsim_v64/WS-V64-P10R2-EXACT-ONCE-CONFIRMATION-01/20260826T203000Z__exact-once-confirmation-s3-r1`；verdict=
`supported_exact_once_route_aware_confirmation`；V64-F25=`active_relative_tail_generalization`。

fresh 96-case exact-once中M0/M1 total coverage均=`0.4749745`，delta=`0`；M1 route worst10 CVaR=`0.0403133<=0.05`，
两项预注册gate PASS。M1把route selected/conflicts从`8117/54`降到`4971/20`，但fresh M0 CVaR更低=`0.0391815`，
M1-M0=`+0.0011318`；pointwise failures=`1->2`，maximum=`0.06818->0.08333`。因此只支持M1自身的fresh observed
absolute empirical tail门，不支持“route-aware相对M0改善tail rate”的主张；current M0历史P10T负结论仍不改写，P11相对收益权限
继续锁定。model refit/runtime selection=`false/false`，GPU wall/peak RSS=`11.8041s/0.8843GiB`。closeout=
`docs/autoresearch/worldsim_v64/P10R2_EXACT_ONCE_CONFIRMATION_CLOSEOUT.md`。

## WorldSim V6.4 P10R2 fresh evidence complete / exact-once next（2026-08-27）

状态：`v64_p10r2_confirmation_evidence_complete_exact_once_next`；canonical=
`run://worldsim_v64/WS-V64-P10R2-CONFIRMATION-EVIDENCE-01/20260826T201500Z__confirmation-evidence-s3-r1`；
confirmation target/model-score read=`true/false`。

冻结8 scene一次生成`96/96` evidence units，scene=`8`、reuse=`0`、source-role overlap=`0`、queries=`0`，disk=
`94236671 bytes`，maximum unit/wall=`15.2590/108.8267s`，passed。没有恢复、policy/model变更或第二份evidence。
下一步只按冻结M0/M1、2s/1.5m route与worst10 CVaR执行一次exact-once；closeout=
`docs/autoresearch/worldsim_v64/P10R2_CONFIRMATION_EVIDENCE_CLOSEOUT.md`。

## WorldSim V6.4 P10R2 fresh native complete / evidence next（2026-08-27）

状态：`v64_p10r2_confirmation_native_complete_evidence_next`；canonical=
`run://worldsim_v64/WS-V64-P10R2-CONFIRMATION-SIDECAR-01/20260826T201000Z__native-aggregate-s3-r1`；
confirmation target/model-score read=`false/false`；V64-F23/F24=`resolved_by_ready_first_resume/producer_single_owner`。

8 scene均完成canonical processed与12-target native：aggregate=`96 targets / 4423846005 bytes / passed`，最大worker显存=
`4.1314GiB`。ready-first恢复复用6个complete native leaf，仅并行重建`0006/0371`；所有leaf wall=`44.45--59.72s`。
tar catalog扩充至`71555 entries / 8585986 bytes`。prep r1在检测到双producer同时写288前终止，r2仅复用8个complete scene，
`0.817s`完成登记并删除6.3GiB可重建raw；target/quality均未读。下一步只生成一次96-unit evidence，再运行固定M1 exact-once。
closeout=`docs/autoresearch/worldsim_v64/P10R2_CONFIRMATION_SIDECAR_CLOSEOUT.md`。

## WorldSim V6.4 P10R2 feeder lock-convoy recovery frozen（2026-08-27）

状态：`v64_p10r2_feeder_resume_reuse_ready_native`；V64-F23=`recovery_frozen_pre_target`。

10/10 tar scan完成且`0590/0596/0070` native各12 targets通过后，观察到已完整processed的`1020(778)`仍排在长耗时
preprocess mutex之后，GPU出现head-of-line idle。参考NVIDIA DALI异步pipelined execution与separate CPU/GPU prefetch queue，
恢复只改feeder调度：启动时复用`passed=true,target_count=12`的已有native leaf；canonical processed存在时绕过preprocess lock，
立即申请GPU slot。冻结cohort/policy/model/targets/gates/run IDs均不变，不重算3个有效leaf；终止当前feeder及可重建staging partial后
以同一prefix恢复。证据=`https://docs.nvidia.com/deeplearning/dali/user-guide/docs/pipeline.html`。

## WorldSim V6.4 P4C optional catalog cleanup stopped / I/O reassigned（2026-08-27）

状态：`v64_p4c_optional_catalog_enrichment_abandoned_temp_removed`；V64-F22=`resolved_by_io_reassignment`。

P4C corrected native/evidence/exact-once正式产物已再次确认完整；两套结果后置catalog scanner仍重复读取同一10 tar且不再产生
研究证据，阻塞P10R2新confirmation。故终止scanner/controller，不合并其未完成catalog增量；保留既有P6R catalog=
`57338 entries / 6880063 bytes`，删除已预注册为可从official tar重建的P4C raw与replacement raw两个临时目录，共释放约
6.8GiB，盘余约34GiB。科学产物、processed/native/evidence/model均未删除，P4C结论不变。I/O已转交唯一P10R2 prep并
scene-ready feed GPU；catalog union pending=`false`，语义是`abandoned optional enrichment`而非假称完成。

## WorldSim V6.4 P10R2 fresh route-aware confirmation frozen（2026-08-27）

状态：`v64_p10r2_fresh_confirmation_preregistered_prep_next`；active task/hypothesis=
`WS-V64-P10R2-CONFIRMATION-SIDECAR-01 / WS-V64-H-P10R2-002`；confirmation target/quality/model-score read=
`false/false/false`。

从700-scene IR-WM temporal train仅按metadata、seed3与当前124个已用scene排除集合冻结8 scene/96 case：night=
`1020(778),1016(774)`，rain=`0596(476),0590(470)`，construction=`0006(5),0472(383)`，vulnerable/transit=
`0070(67),0371(288)`；全部为temporal member且>=40 samples。M1保持原MLP、M0 conditional coverage、route cap=0.40、
总selected count不变、2s/1.5m route与worst10尾部；exact-once仅保留CVaR<=0.05和coverage delta<=1e-6两门。
prep/native/aggregate/evidence/exact IDs及单卡scene-ready feed已固定；无refit、参数扫描、第二confirmation或额外测试。
freeze=`docs/autoresearch/worldsim_v64/P10R2_CONFIRMATION_COHORT_FREEZE.md`与
`docs/autoresearch/worldsim_v64/P10R2_CONFIRMATION_EXECUTION_FREEZE.md`。

## WorldSim V6.4 P10R2 route-aware M1 calibration candidate supported（2026-08-26）

状态：`v64_p10r2_route_aware_candidate_supported_confirmation_freeze_next`；canonical=
`run://worldsim_v64/WS-V64-P10R2-ROUTE-AWARE-COMPILER-01/20260826T191500Z__route-aware-compiler-s0-r1`；verdict=
`supported_route_aware_candidate_on_consumed_calibration`。

在 consumed P6R development/calibration 的96 cases上，M0/M1 mean total coverage均为`0.4749505`，delta=`0`；route
selected voxels由`5912`降至`3826`，hidden-FREE conflicts由`23`降至`9`。M0/M1 route worst10 empirical CVaR=
`0.0220499/0.0114783`，M1最大case rate=`0.0454545`且0 case超过0.05，两项冻结门PASS。模型未重训、run中未选policy、
new confirmation未读；GPU wall/peak RSS=`11.3438s/0.8849GiB`。这只支持M1 calibration candidate，不关闭current M0的
V64-F21负结论，也不解锁P11。下一步只做metadata-only fresh temporal-member confirmation freeze，然后exact-once运行。
closeout=`docs/autoresearch/worldsim_v64/P10R2_ROUTE_AWARE_COMPILER_CLOSEOUT.md`。

## WorldSim V6.4 P10R2 route-aware M1 candidate frozen（2026-08-26）

状态：`v64_p10r2_route_aware_candidate_preregistered`；active task/hypothesis=
`WS-V64-P10R2-ROUTE-AWARE-COMPILER-01 / WS-V64-H-P10R2-001`；formal run=
`20260826T191500Z__route-aware-compiler-s0-r1`。

P10T 对 current M0 的 negative route-tail authority 保持不可改写；恢复作为新版本 M1。只复用已消费 P6R confirmation
96 cases 作 calibration/development：总 selected count 与 M0 完全相同，route corridor 内名义覆盖固定上限为独立 C0=`0.40`，
空出的 route budget 按原冻结风险分数重分配给 non-route voxels。模型不重训，M0 stratum coverage、2s/1.5m route、tail
alpha=`0.10` 均不扫描。冻结门只有 M1 route empirical CVaR `<=0.05` 与 mean total coverage delta `<=1e-6`；未触碰
新 confirmation，若 calibration candidate 通过，仍必须在 metadata-only 冻结的新 temporal-member cohort 上 exact-once 确认。
P11 对 current M0 继续锁定。freeze=`docs/autoresearch/worldsim_v64/P10R2_ROUTE_AWARE_COMPILER_FREEZE.md`。

## WorldSim V6.4 P10T empirical route-tail rejected / P11 locked（2026-08-26）

状态：`v64_p10t_route_tail_rejected_p11_locked`；canonical=
`run://worldsim_v64/WS-V64-P10T-ROUTE-TAIL-AUDIT-01/20260826T190000Z__route-tail-audit-s0-r1`；verdict=
`rejected_empirical_route_tail`；V64-F21=`closed_negative_tail_authority`。

冻结worst10/96 empirical CVaR：C0=`0.0504298`，M0=`0.0517085`，M0-C0=`+0.0012787`；M0超过0.05唯一gate，
且仍有5个pointwise case>0.05，故current frozen M0的route/collision tail authority正式拒绝。该负结果不推翻P4C fresh
case-risk、P10M materialization、P10G Gaussian splat或P10R pooled exposure；只禁止把它们提升为tail-safe collision/planning claim，
并锁定P11。target未重读、policy/model未改，CPU wall=`0.00076s`。closeout=
`docs/autoresearch/worldsim_v64/P10T_ROUTE_TAIL_AUDIT_CLOSEOUT.md`。合法下一研究只能是新版本route-aware conditional policy，需新
calibration/confirmation合同，不能回调本次M0。

## WorldSim V6.4 P10T empirical route-tail CVaR audit frozen（2026-08-26）

状态：`v64_p10t_route_tail_audit_preregistered`；active task/hypothesis=
`WS-V64-P10T-ROUTE-TAIL-AUDIT-01 / WS-V64-H-P10T-001`；V64-F21=`recovery_frozen_post_quality_no_policy_change`。

P10C pooled conflict通过但5/96 route cases局部>0.05，故collision/route authority仍未解锁。参考risk-sensitive CVaR与PAC-Bayesian
CVaR分析，冻结empirical top-decile mean：96 cases取worst 10，M0 CVaR门=`<=0.05`。只复用P10C rows，不重读target、不改
policy/route/model，不扫tail fraction；结果只作empirical tail diagnostic，不称population bound。freeze=
`docs/autoresearch/worldsim_v64/P10T_ROUTE_TAIL_AUDIT_FREEZE.md`。

## WorldSim V6.4 P10C pooled route-local conflict supported / local tail remains（2026-08-26）

状态：`v64_p10c_pooled_conflict_supported_local_tail_active`；canonical=
`run://worldsim_v64/WS-V64-P10C-ROUTE-CONFLICT-AUDIT-01/20260826T184500Z__route-conflict-audit-s0-r1`；verdict=
`supported_route_local_conflict_severity`；V64-F20=`resolved_by_route_local_cell_severity`。

C0/M0 route-emitted voxels=`9450/10013`，M0新增`563`；hidden-FREE conflicts=`34/43`，pooled rate=
`0.003598/0.004294`，M0低于0.05，两项gate PASS。新增state带来9个新增conflict；M0仍有5/96 case局部rate>0.05，最高=
`0.106383`，故仅支持pooled route-local severity，不支持per-case tail或collision authority。target read=`true`、collision GT=
`false`、GPU wall=`4.1697s`。closeout=`docs/autoresearch/worldsim_v64/P10C_ROUTE_CONFLICT_AUDIT_CLOSEOUT.md`。

## WorldSim V6.4 P10C route-local conflict severity audit frozen（2026-08-26）

状态：`v64_p10c_route_conflict_audit_preregistered`；active task/hypothesis=
`WS-V64-P10C-ROUTE-CONFLICT-AUDIT-01 / WS-V64-H-P10C-001`；V64-F20=`recovery_frozen_pre_target_audit`。

P10R binary intercept在两臂均96/96饱和，不能用于新增collision-case主张。参考Waymo cell-level occupancy metrics、Implicit
Occupancy Flow的planner-query语义与soft collision potential，恢复固定为同一2s/1.5m corridor上的route-local hidden-FREE conflict。
policy/route均不改；允许一次读取已锁定TARGET_EVIDENCE，只判断C0/M0 route-emitted voxels。仅要求M0新增route state>0且pooled
route hidden-FREE conflict<=0.05；case failures只描述。collision GT/planner仍不读。freeze=
`docs/autoresearch/worldsim_v64/P10C_ROUTE_CONFLICT_AUDIT_FREEZE.md`。

## WorldSim V6.4 P10R bounded Gaussian route exposure supported / binary intercept saturated（2026-08-26）

状态：`v64_p10r_bounded_route_exposure_supported_binary_intercept_saturated`；canonical=
`run://worldsim_v64/WS-V64-P10R-GAUSSIAN-ROUTE-CONSUMER-01/20260826T183000Z__gaussian-route-consumer-s0-r1`；verdict=
`supported_bounded_gaussian_route_exposure`。

96 case覆盖logged future route=`1241.4030m`；C0/M0 corridor support=`12081/12456 cells`，M0新增`375 cells`且分布于
`36/96` cases，两项gate PASS。两臂binary route intercept均`96/96`，additional intercept cases=`0`，因此case-level binary
interception已饱和，不能声称检测更多collision case。target/model/collision GT read=`false/false/false`，GPU wall/peak RSS=
`0.7472 s/0.6907 GiB`。结果只支持logged-route semantic exposure的细粒度增量；closeout=
`docs/autoresearch/worldsim_v64/P10R_GAUSSIAN_ROUTE_CONSUMER_CLOSEOUT.md`。

## WorldSim V6.4 P10R bounded Gaussian route consumer frozen（2026-08-26）

状态：`v64_p10r_gaussian_route_consumer_preregistered`；active task/hypothesis=
`WS-V64-P10R-GAUSSIAN-ROUTE-CONSUMER-01 / WS-V64-H-P10R-001`。

96 case均从processed `lidar_pose`取target后2秒/20帧logged ego trajectory，变换到target-lidar，冻结1.5m corridor并读取P10G
C0/M0 BEV density。consumer只报告route support/exposure与intercept case，不读target/model/collision GT，不把logged path overlay解释为
counterfactual collision或planning。仅要求96 case全部消费且M0 route support gain>0；参数不扫。freeze=
`docs/autoresearch/worldsim_v64/P10R_GAUSSIAN_ROUTE_CONSUMER_FREEZE.md`。

## WorldSim V6.4 P10G sparse Gaussian state adapter supported（2026-08-26）

状态：`v64_p10g_sparse_gaussian_adapter_supported`；canonical=
`run://worldsim_v64/WS-V64-P10G-GAUSSIAN-STATE-ADAPTER-01/20260826T181500Z__gaussian-state-adapter-s0-r1`；verdict=
`supported_sparse_gaussian_state_adapter`；V64-F19=`resolved_by_sparse_gaussian_adapter`。

96个P10M package全部转换并render：M0/C0 Gaussian count=`534581/460082`，conditional additional=`74499`；BEV support
cells=`594772/553756`，gain=`41016`。fixed Gaussian=`scale 0.256m / identity rotation / opacity 0.95 / OCCUPIED`；output=
`40148486 bytes`，GPU wall/peak RSS=`0.9840 s/0.8689 GiB`。consumer未访问target、model或StreetGS checkpoint，两项gate均PASS。
结果支持sparse semantic Gaussian参数化与probabilistic BEV splat，不支持photorealistic render/sensor/collision/planning/safety。
closeout=`docs/autoresearch/worldsim_v64/P10G_GAUSSIAN_STATE_ADAPTER_CLOSEOUT.md`；下一步转route/collision semantic consumer。

## WorldSim V6.4 P10G sparse Gaussian state adapter frozen（2026-08-26）

状态：`v64_p10g_sparse_gaussian_adapter_preregistered`；active task/hypothesis=
`WS-V64-P10G-GAUSSIAN-STATE-ADAPTER-01 / WS-V64-H-P10G-001`；V64-F19=`recovery_frozen_pre_run`。

fresh 8 scene没有同场景StreetGS checkpoint，旧V6 GS runtime又强绑定其他scene与hash-heavy governance，不能直接消费P10M。
检索GaussianFormer、GaussianWorld与GaussianOcc后迁移最小共同表示：每个M0-emitted voxel映射为一个semantic Gaussian，mean=
metric center、isotropic scale=`0.256m`、identity rotation、opacity=`0.95`、state=`OCCUPIED`；96 case在GPU上做probabilistic
BEV Gaussian superposition。runner只读P10M package，不读target/model/StreetGS。仅要求96 package全部render且M0 BEV support gain>0；
不做超参扫描或照片级/sensor/collision claim。freeze=`docs/autoresearch/worldsim_v64/P10G_GAUSSIAN_STATE_ADAPTER_FREEZE.md`。

## WorldSim V6.4 P10M target-free conditional state bake supported（2026-08-26）

状态：`v64_p10m_target_free_state_bake_supported`；canonical=
`run://worldsim_v64/WS-V64-P10M-CONDITIONAL-STATE-BAKE-01/20260826T180000Z__conditional-state-bake-s0-r1`；verdict=
`supported_target_free_conditional_state_bake`。

96个fresh package覆盖`1150300`个eligible native boundary voxels：C0发射`460082`，M0发射`534581`，新增`74499`
（construction/night/rain/vulnerable=`25199/35221/0/14079`），mean coverage uplift=`0.0750164`；以0.512m voxel计算的
nominal additional emitted volume=`9999.0865 m3`。package-only runtime消费全部96个state package，不加载模型或evidence；state bake
本身未读target evidence。两项gate均PASS，GPU wall/peak RSS=`9.3996 s/0.7806 GiB`，output=`27780960 bytes`。该结果只支持
target-free state materialization，下一步研究真实GS adapter，不把nominal voxel volume写成物理/安全有效体积。closeout=
`docs/autoresearch/worldsim_v64/P10M_CONDITIONAL_STATE_BAKE_CLOSEOUT.md`。

## WorldSim V6.4 P10M target-free conditional state bake frozen（2026-08-26）

状态：`v64_p10m_conditional_state_bake_preregistered`；active task/hypothesis=
`WS-V64-P10M-CONDITIONAL-STATE-BAKE-01 / WS-V64-H-P10M-001`。P4C exact-once支持后直接进入physical-state materialization，
跳过legacy28与额外confirmation/test矩阵。

冻结runner只读fresh `METHOD_EVIDENCE`、native logits/BEV与既有MLP，不读取`TARGET_EVIDENCE`；对96 case把C0/M0选择编译成
带metric-frame voxel centers、risk score及`OCCUPIED/UNKNOWN`状态的独立NPZ。随后semantic runtime consumer只从包中读取状态与
坐标，不加载模型或evidence。仅保留mean coverage uplift `>=0.05`与`96 packages全部可消费且M0新增状态>0`两个门；不做GS
render/collision/planning claim。freeze=`docs/autoresearch/worldsim_v64/P10M_CONDITIONAL_STATE_BAKE_FREEZE.md`。

## WorldSim V6.4 P4C exact-once conditional confirmation supported（2026-08-26）

状态：`v64_p4c_exact_once_supported_cleanup_running`；canonical=
`run://worldsim_v64/WS-V64-P4C-CONDITIONAL-EXACT-ONCE-CONFIRMATION-01/20260826T173000Z__exact-once-confirmation-s0-r1`；
confirmation target/model-score read=`true/true`，model refit/policy selection=`false/false`。

冻结C0 global 40%与M0 conditional map在fresh 96-case confirmation上各读取一次：C0 coverage/failure=
`0.3999444/0`，M0=`0.4749608/0`，absolute uplift=`0.0750164`。M0四strata均`0/24` failure；coverage uplift、overall
`<=4/96`、each-stratum `<=1/24`三项gate全部PASS，verdict=`supported_exact_once_conditional_confirmation`。GPU score=
`12.1745 s`，peak RSS=`0.7958 GiB`。结论只支持fresh observed case-risk下的冻结conditional coverage map，不外推现实安全或
downstream simulation。当前只剩双controller EOF、temporary raw删除与catalog semantic union；closeout=
`docs/autoresearch/worldsim_v64/P4C_EXACT_ONCE_CONFIRMATION_CLOSEOUT.md`。

## WorldSim V6.4 P4C confirmation evidence complete / exact-once next（2026-08-26）

状态：`v64_p4c_confirmation_evidence_complete_scoring_next`；canonical=
`run://worldsim_v64/WS-V64-P4C-CONDITIONAL-CONFIRMATION-EVIDENCE-01/20260826T171500Z__confirmation-evidence-s0-r1`；active task=
`WS-V64-P4C-CONDITIONAL-EXACT-ONCE-CONFIRMATION-01`；confirmation target/model-score read=`true/false`。

冻结corrected cohort一次完成`96 units / 8 scenes`，query/source-role overlap=`0/0`，disk=`90704718 bytes`，maximum unit/wall=
`6.7895/51.9970 s`，passed；没有reuse、failure delta、policy选择或model refit。下一步只以冻结C0=`global 0.40`和M0=
`rain 0.40; night/construction/vulnerable 0.50`执行一次exact-once评分。closeout=
`docs/autoresearch/worldsim_v64/P4C_CONFIRMATION_EVIDENCE_CLOSEOUT.md`。

## WorldSim V6.4 P4C corrected native confirmation complete / evidence next（2026-08-26）

状态：`v64_p4c_confirmation_native_complete_evidence_next`；canonical=
`run://worldsim_v64/WS-V64-P4C-CONDITIONAL-CONFIRMATION-SIDECAR-01/20260826T170000Z__native-aggregate-s0-r1`；active task=
`WS-V64-P4C-CONDITIONAL-CONFIRMATION-EVIDENCE-01`；new confirmation target/model-score read=`false/false`。

保留v1的7个有效native leaf，并以预冻结temporal member `scene-0813(631)`替换`scene-0276`；corrected aggregate完成
`8 scenes / 96 targets / 4423846027 bytes`，maximum worker peak GPU memory=`4.1314 GiB`。replacement raw wait=
`716.9761 s`，native wall=`45.2537 s`；sidecar阶段没有读取target、quality或模型分数，也没有改C0/M0/model/gate/denominator。
V64-F18=`resolved_pre_quality`。下一步只生成冻结96-unit evidence，再执行一次fixed C0/M0 exact-once评分。完整收口=
`docs/autoresearch/worldsim_v64/P4C_CONFIRMATION_SIDECAR_CLOSEOUT.md`。

## WorldSim V6.4 P4C 7/8 blind native complete / temporal-member replacement frozen（2026-08-26）

状态：`v64_p4c_confirmation_temporal_membership_recovery_frozen`；active task=
`WS-V64-P4C-CONDITIONAL-CONFIRMATION-SIDECAR-01 replacement scene-0813`；target/model-score read=`false/false`。
v1 scene-ready流水完成7个native scene；`scene-0276`在native output前因IR-WM train temporal pickle无该key而失败，V64-F18=
`recovery_frozen_pre_quality`。已确认其余7 scene均为temporal member并完成12 targets。

恢复不重算7个有效scene，不改C0/M0/model/gate/denominator；只把无效vulnerable scene替换为冻结seed2 fallback中首个
token-valid temporal member `scene-0813(631)`。replacement prep/native=`164500Z/165000Z`；aggregate/evidence/exact-once
仍为`170000Z/171500Z/173000Z`。replacement使用独立member-shard JSON，避免与仍持旧snapshot的v1 controller并发覆盖；
两controller完成后再semantic union并atomic replace。freeze=`docs/autoresearch/worldsim_v64/P4C_TEMPORAL_MEMBERSHIP_RECOVERY_FREEZE.md`。

## WorldSim V6.4 P4C scene-ready confirmation execution frozen（2026-08-26）

状态：`v64_p4c_confirmation_execution_frozen`；active task=`WS-V64-P4C-CONDITIONAL-CONFIRMATION-SIDECAR-01`；
new confirmation target/model-score read=`false/false`。blind prep/per-scene native/aggregate/evidence/exact score canonical IDs已固定；
temporary raw exact path=`/root/autodl-tmp/tmp/worldsim_v64_p4c_raw_batch`。一scene raw ready即preprocess并feed GPU，最多1个
preprocess与2个GPU worker；prep controller负责catalog superset EOF与临时raw删除。执行冻结=
`docs/autoresearch/worldsim_v64/P4C_CONFIRMATION_EXECUTION_FREEZE.md`。

## WorldSim V6.4 P4C conditional candidate supported / new confirmation input next（2026-08-26）

状态：`v64_p4c_conditional_candidate_supported`；canonical=
`run://worldsim_v64/WS-V64-P4C-CONDITIONAL-COMPILER-01/20260826T160000Z__conditional-compiler-s0-r1`；active task=
`WS-V64-P4C-CONDITIONAL-CONFIRMATION-SIDECAR-01`。已读calibration上C0 coverage/failure=`0.3999668/0`，冻结M0=
`0.4749773/0`，absolute uplift=`0.0750105`；四strata均`0/24`，四项candidate gate全PASS。model refit/run-time
selection=`false/false`，GPU wall=`13.3357 s`。该结果只冻结candidate，新confirmation仍未读。

旧P6R prep controller亦已完成：superset catalog=`57338 entries / 6880063 bytes`，可重建temporary raw已由controller
删除，V64-F16=`resolved_by_scene_ready_streaming_and_catalog_finalize`。下一步为新8-scene raw/processed/native blind sidecar，继续
scene-ready feed，避免等待全批。为给预计约12 GiB新资产留空间，只删除可重建`/root/autodl-tmp/pip_cache` 13 GiB，free disk
从29 GiB增至41 GiB；模型、环境、processed与formal runs均保留。P4C closeout=
`docs/autoresearch/worldsim_v64/P4C_CONDITIONAL_COMPILER_CALIBRATION_CLOSEOUT.md`。

## WorldSim V6.4 P4C conditional compiler frozen / calibration replay next（2026-08-26）

状态：`v64_p4c_conditional_compiler_frozen`；active task=`WS-V64-P4C-CONDITIONAL-COMPILER-01`。C0固定global
40%；M0固定rain 40%、night/construction/vulnerable-transit 50%，沿用冻结MLP、risk order与0.05 conflict threshold；
不做coverage sweep或refit。候选只来自已消费calibration上的既有结果：50%的3个failure全部在rain，其余strata为0。

新96-case confirmation在quality read前以metadata seed2冻结为`0992/1101,0454/1102,0876/0895,0321/0276`，
每stratum两scene。selection只读description、sample count和保守used-scene membership。下一步先formal replay已读calibration；
若M0达到coverage uplift>=0.05且保持0 failure，才物化新confirmation。freeze=
`docs/autoresearch/worldsim_v64/P4C_CONDITIONAL_COMPILER_FREEZE.md`。

## WorldSim V6.4 P6R exact-once confirmation supported（2026-08-26）

状态：`v64_p6r_exact_once_supported_cleanup_running`；canonical=
`run://worldsim_v64/WS-V64-P6R-EXACT-ONCE-CONFIRMATION-01/20260826T153500Z__exact-once-confirmation-s0-r1`；
confirmation target/model-score read=`true/true`，model refit/policy selection=`false/false`。

冻结full-native selective MLP只在独立校准选定的nominal 40%上读取一次96-case确认：mean realized coverage=
`0.3999405`，failure=`1/96`，empirical case risk=`0.0104167`。construction/night/rain/vulnerable-transit分别为
`0/24, 1/24, 0/24, 0/24`；冻结总体`<=4/96`与每stratum`<=1/24`两项gate均通过，verdict=
`supported_exact_once_confirmation`。GPU score=`12.5902 s`，peak RSS=`0.7907 GiB`。该证据仅支持冻结策略的
observed case-risk，不外推现实安全或下游compiler。当前只剩后台superset catalog EOF写回和临时raw回收；closeout=
`docs/autoresearch/worldsim_v64/P6R_EXACT_ONCE_CONFIRMATION_CLOSEOUT.md`。

## WorldSim V6.4 P6R confirmation evidence complete / exact-once scoring next（2026-08-26）

状态：`v64_p6r_confirmation_evidence_complete_scoring_next`；canonical=
`run://worldsim_v64/WS-V64-P6R-CONFIRMATION-EVIDENCE-01/20260826T152500Z__confirmation-evidence-s0-r2`；active task=
`WS-V64-P6R-EXACT-ONCE-CONFIRMATION-01`；confirmation target/model-score read=`true/false`。

r2完成`96/96 units / 8 scenes`：hardlink复用r1的33 units、只新算63；query/source-role overlap=`0/0`，logical disk=
`83483823 bytes`，maximum new-unit/wall=`11.5779/74.6360 s`，passed。empty actor-frame恢复生效，V64-F17=
`resolved_pre_score`。下一动作只以冻结MLP和nominal 40%执行一次96-case评分；不选coverage、不refit。closeout=
`docs/autoresearch/worldsim_v64/P6R_CONFIRMATION_EVIDENCE_CLOSEOUT.md`。

## WorldSim V6.4 P6R confirmation evidence r1 failed / empty-frame r2 frozen（2026-08-26）

状态：`v64_p6r_confirmation_evidence_empty_frame_recovery_frozen`；failed run=
`run://worldsim_v64/WS-V64-P6R-CONFIRMATION-EVIDENCE-01/20260826T151500Z__confirmation-evidence-s0-r1`；active task=
`WS-V64-P6R-CONFIRMATION-EVIDENCE-01 r2`；confirmation target/model-score read=`partial 33 units/false`。

r1完成`33/96` units后在scene-1105 frame62触发`frame_instances['62'] KeyError`。该scene的0--9与56--64键缺失，
但`instances_info`逐帧核对均为0 actor annotation，`missing_with_annotations=[]`；sensor/lidar/pose完整。依据nuScenes devkit
non-keyframe annotation interpolation语义，缺键应解释为empty actor set。登记`V64-F17 recovery_frozen_pre_score`；common loader
只改为`frame_instances.get(str(frame), [])`。

r2不重算已完成33 units：NPZ以hardlink复用、不复制；NPZ可恢复的semantic/sparse counts写回manifest，未存储的actor/raw/
dynamic point count显式为null；仅计算剩余63 units。40% policy、模型、case gate和scene均不变，尚未读取任何confirmation
model score。freeze=`docs/autoresearch/worldsim_v64/P6R_CONFIRMATION_EVIDENCE_RECOVERY_FREEZE.md`。

## WorldSim V6.4 P6R confirmation native complete / exact-once evidence next（2026-08-26）

状态：`v64_p6r_confirmation_native_complete_evidence_next`；canonical=
`run://worldsim_v64/WS-V64-P6R-CONFIRMATION-SIDECAR-01/20260826T150000Z__native-aggregate-s0-r1`；active task=
`WS-V64-P6R-CONFIRMATION-EVIDENCE-01`；confirmation target/quality read=`false/false`。

冻结新cohort的8 scene全部完成blind IR-WM：`96 targets / 4423846018 bytes`，maximum worker peak=`4.1314 GiB`；aggregate
只建symlink、不复制native数组。逐scene GPU wall均约`45.59--46.74 s`。模型、calibrated 40% policy和confirmation target均未
在sidecar阶段读取或修改。

V64-F16的scene-ready恢复已达成：按shard10→8/9→4/5→6→1/2优先组依次供给GPU，DriveStudio独占I/O窗口、IR-WM
计算窗口恢复未完成scan，避免了全10-shard/全8-scene屏障。persistent superset catalog的EOF写回与临时raw删除仍由prep
controller后台收口，不阻塞已完整native artifact。下一步只生成冻结96-unit confirmation evidence，然后fixed 40% exact-once
评分。完整sidecar收口=`docs/autoresearch/worldsim_v64/P6R_CONFIRMATION_SIDECAR_CLOSEOUT.md`。

## WorldSim V6.4 P6R confirmation execution frozen / indexed streaming recovery（2026-08-26）

状态：`v64_p6r_confirmation_execution_preregistered`；active task=`WS-V64-P6R-CONFIRMATION-SIDECAR-01`；policy=
`full-native MLP / nominal coverage 0.40`；confirmation quality read=`false`。

新8 scene在本机raw cache均为0命中，旧`worldsim_v64_p6_member_shards.json`只有前批43033个member，脚本又会在每批写回时
丢弃非本批映射，导致本批约24.8k unseen member再次全扫10个gzip tar。登记`V64-F16 active resource/operations`。检索
WebDataset WIDS indexed random access与ratarmount compressed-tar SQLite index后，迁移为持久化superset member->shard catalog，
不再裁掉历史映射；本批仍需一次顺序解压扫描，但scene raw一齐即由最多两个DriveStudio producer和两个IR-WM GPU consumer
流水消费，不等待8 scene整批屏障。

执行合同：只准备冻结`1023,1105,0903,0451,0981,0537,0789,0157`；每scene完成即blind提取12 targets；全部processed
后删可重建临时raw；再一次生成96 evidence并只评固定40% policy。没有model refit/coverage sweep/hash/checksum/fingerprint、
smoke或回归矩阵。单3090仍足够。配置=`configs/worldsim_v64/p6r_confirmation_sidecars_v1.yaml`；freeze=
`docs/autoresearch/worldsim_v64/P6R_CONFIRMATION_EXECUTION_FREEZE.md`。

Exact-once gate也已在target read前固定：40% coverage下overall最多`4/96` case loss、每stratum最多`1/24`，对应观测
risk<=0.05；不再做第二次置信界选择。runner/config已实现并等待blind sidecar完成。

## WorldSim V6.4 P6R independent calibration supported / confirmation unlocked（2026-08-26）

状态：`v64_p6r_calibration_supported_confirmation_unlocked`；canonical=
`run://worldsim_v64/WS-V64-P6R-CALIBRATION-01/20260826T141500Z__case-calibration-s0-r1`；verdict=
`supported_selective_policy`；selected nominal coverage=`0.40`；new confirmation read=`false`。

96个独立case中，coverage 0.05/0.10/0.20/0.30/0.40均为`0/96` failure，mean realized coverage分别为
`0.049961/0.099963/0.199969/0.299958/0.399967`；Bonferroni one-sided Clopper--Pearson UCB统一为`0.048647`，
低于0.05 target。50% coverage为`3/96`、empirical/UCB=`0.03125/0.103218`，三例均在rain，因此按最大通过规则选择40%，
未越界挑50%。四strata在40%各`0/24` failure。GPU wall=`13.0083 s`、peak RSS=`0.7943 GiB`。

V64-F15的PCA16线性U3拒绝保持不可变，但恢复状态改为`resolved_by_new_version`：完整273D MLP已通过独立有限样本校准。
这仍不是confirmation、authority或现实安全声明。下一步只对预先冻结的新8 scene生成blind native/evidence并以40%策略exact-once
确认，不 refit、不改policy。完整收口=`docs/autoresearch/worldsim_v64/P6R_CASE_CALIBRATION_CLOSEOUT.md`。

## WorldSim V6.4 P6R independent evidence complete / calibration scoring next（2026-08-26）

状态：`v64_p6r_independent_evidence_complete_calibration_scoring_next`；canonical=
`run://worldsim_v64/WS-V64-P6R-CALIBRATION-EVIDENCE-01/20260826T140000Z__calibration-evidence-s0-r1`；active task=
`WS-V64-P6R-CALIBRATION-01`；independent-calibration target/model-score read=`true/false`；new-confirmation read=`false`。

冻结模型之后一次生成原8-scene的`96/96` evidence units，queries=`0`、source-role overlap=`0`、disk=`118985634 bytes`、
maximum unit/wall=`15.4137/111.5142 s`。没有模型评分、coverage选择或额外gate；下一动作只运行已push的P6R case
calibration一次。Failure delta=`none`，V64-F15仍待独立结果；不加hash/checksum/fingerprint或测试矩阵。

## WorldSim V6.4 P6R MLP trained / independent calibration frozen（2026-08-26）

状态：`v64_p6r_selective_mlp_trained_independent_calibration_preregistered`；canonical=
`run://worldsim_v64/WS-V64-P6R-SELECTIVE-MLP-01/20260826T134500Z__selective-mlp-s0-r1`；active task=
`WS-V64-P6R-CALIBRATION-EVIDENCE-01`；independent-calibration/new-confirmation quality read=`false/false`。

唯一冻结训练使用16个已消费development scene的`786054`个native-boundary点（273D），hidden-FREE=`59867`、prevalence=
`0.0761614`。20-epoch focal loss从`0.0337864`单调降至`0.0251443`；development AUROC=`0.8811503`仅作描述，
不作gate。GPU fit=`10.1545 s`，总wall=`34.1934 s`，peak RSS=`3.6301 GiB`，model=`177 KiB`。没有超参扫描。

模型artifact现已冻结，原8个quality-unread scene获准一次独立case calibration；先生成`96`个无query evidence unit，再按原
协议一次评分。新confirmation八scene继续锁定。配置=`configs/worldsim_v64/p6r_calibration_evidence_v1.yaml`与
`configs/worldsim_v64/p6r_case_calibration_v1.yaml`；训练收口=
`docs/autoresearch/worldsim_v64/P6R_SELECTIVE_MLP_TRAINING_CLOSEOUT.md`。不加hash/checksum/fingerprint、smoke或回归矩阵。

## WorldSim V6.4 P6R split/model frozen / GPU training next（2026-08-26）

状态：`v64_p6r_selective_mlp_preregistered`；active task/hypothesis=
`WS-V64-P6R-SELECTIVE-MLP-01 / WS-V64-H-P6R-001`；development/calibration/new-confirmation quality read=
`true/false/false`。

P6已消费的16 scene转为development training；原8个untouched confirmation转为独立calibration。其target仍未读。
在读取它们前，从剩余588个>=40-sample IR-WM train-temporal scene中只按name/description/sample count/exclusion和seed1
冻结新confirmation：night=`1023,1105`，rain=`0903,0451`，construction=`0981,0537`，vulnerable-transit=
`0789,0157`；没有使用Occupancy、hidden-FREE、UQ或模型分数。

恢复模型固定为完整273D native输入的`273-128-64-1` MLP，GELU/dropout0.10，focal BCE gamma2/alpha0.75，
AdamW lr1e-3/wd1e-4，20 epochs，batch16384，seed0；每development scene最多49152 points。禁止width/loss/seed/
epoch/lr/sampling sweep，development AUROC仅描述，不作gate。模型落盘后才允许原8 scene一次独立校准；协议仍为5%--50%
coverage、conflict 0.05、case risk 0.05、confidence 0.95和Bonferroni Clopper--Pearson。完整冻结=
`docs/autoresearch/worldsim_v64/P6R_SELECTIVE_MLP_FREEZE.md`。不加hash/checksum/fingerprint或测试矩阵；单3090足够。

## WorldSim V6.4 P6 case calibration rejected / selective MLP recovery next（2026-08-26）

状态：`v64_p6_case_calibration_rejected_selective_mlp_recovery_next`；completed task=
`WS-V64-P6-CALIBRATION-01`；hypothesis=`WS-V64-H-P6C-001 rejected`；canonical=
`run://worldsim_v64/WS-V64-P6-CALIBRATION-01/20260826T131000Z__case-calibration-s0-r1`；confirmation target read=`false`。

16 scene / 192 case按冻结协议一次评分。最低5% coverage仍有`41/192` case的selected hidden-FREE conflict超过0.05，
empirical risk=`0.213542`、Bonferroni simultaneous upper bound=`0.292860`；construction/night/rain/vulnerable四strata失败=
`4/16/8/13`（各48 cases）。coverage 0.10/0.20/0.30/0.40/0.50的失败数=`54/62/74/80/93`，没有任何正coverage
满足risk upper bound<=0.05，故policy=`null`。CPU wall=`45.2726 s`、peak RSS=`0.3484 GiB`。

这拒绝P5的PCA16线性U3作为新场景case-level controller；不是simultaneous bound过严，因为5% empirical risk本身已超过
目标四倍。登记`V64-F15 active algorithm/evaluation`；禁止降低epsilon、提高conflict阈值、删night/vulnerable strata、
增加未注册<5% coverage或读取confirmation救结果。完整收口=
`docs/autoresearch/worldsim_v64/P6_CASE_CALIBRATION_CLOSEOUT.md`。

检索ICML 2019 SelectiveNet、NeurIPS 2019 Deep Gamblers和ICCV 2017 Focal Loss后，下一恢复只允许新版本：已消费的16
scene降为development training，用完整native feature训练一个固定小型selective MLP；当前8个quality-unread confirmation
scene改作独立calibration，并在其读取前从剩余metadata-only pool冻结新的confirmation cohort。模型/训练/coverage协议必须先
提交push；不扫描网络宽度、loss或seed。

## WorldSim V6.4 P6 native cohort 完成 / case calibration 已冻结（2026-08-26）

状态：`v64_p6_native_complete_case_calibration_preregistered`；completed prep=
`run://worldsim_v64/WS-V64-P6-CALIBRATION-SIDECAR-01/20260826T112500Z__calibration-prep-s0-r2`；active task=
`WS-V64-P6-CALIBRATION-01`；calibration/confirmation quality read=`false/false`。

prep r2复用r1已扫描raw和完整scene-1045，按metadata派生的`(nbr_samples-1)*5+1`合同完成`24/24`场景：
16 calibration + 8 confirmation，帧数按scene为`196`或`201`；wall=`2,286.7511 s`。成功后删除约`15 GiB`可重建
临时raw，盘余量回到约`36 GiB`；已有raw不变。逐场景blind IR-WM sidecar完成`24 scenes / 288 targets`，每worker
peak约`4.13 GiB`；只读取固定图像和时序metadata并生成native表征，没有读取hidden-FREE/UQ quality或confirmation target。

I/O恢复采用两个DriveStudio producer与最多两个IR-WM consumer，实测双GPU worker达到`100%`利用率、约`14 GiB`
device memory；单3090足够。`V64-F12`由流水线恢复，`V64-F13`由variable-length r2恢复。短SSH编排曾因继承stdin不退出，
按OpenSSH官方`-n`合同修复并登记`V64-F14 resolved_operations`；远端worker与已完成artifact未重跑。

下一里程碑只聚合不复制native数组，并在读取calibration target前冻结192-case协议：固定U3模型；候选coverage=
`0.05,0.10,0.20,0.30,0.40,0.50`；case loss=`selected hidden-FREE conflict >0.05`；target risk/confidence=
`0.05/0.95`；六候选使用Bonferroni one-sided Clopper-Pearson simultaneous upper bound，选最大通过coverage，全部失败则
直接reject且不读confirmation。完整freeze=`docs/autoresearch/worldsim_v64/P6_CASE_CALIBRATION_FREEZE.md`。不加hash/
checksum/fingerprint、smoke或回归矩阵。

## WorldSim V6.4 P6 I/O→GPU 流水线恢复进行中（2026-08-26）

状态：`v64_p6_preparation_running_incremental_gpu_feed_armed`；active task=
`WS-V64-P6-CALIBRATION-SIDECAR-01`；prep run=
`run://worldsim_v64/WS-V64-P6-CALIBRATION-SIDECAR-01/20260826T100000Z__calibration-prep-s0-r1`；
calibration/confirmation quality read=`false/false`。

正式准备入口已启动。共享盘扫描超过一小时后完成`9/10`个官方tar shard，临时raw约`14 GiB`、剩余盘约`45 GiB`，
GPU仍为`0% / 1 MiB`；这暴露了“全量tar完成→全量processed完成→GPU提取”的整批屏障与此前`~21.6 GiB`
持久化估算偏低，登记`V64-F12 active resource/operations`。未中止或重复前九个shard，也未启动无关GPU filler。

参考NVIDIA DALI的异步pipelined/prefetch queue和WebDataset按shard流式消费，迁移为有界生产者—消费者：sidecar wrapper
新增按partition/scene入口；每个DriveStudio scene达到冻结的`1176 images + 196 lidar`即送入IR-WM，单卡最多两个scene
worker。先流水化16-scene calibration；校准模型冻结后才读取8-scene confirmation。这样不等待24场景整批完成，也不增加
hash/checksum/fingerprint、quality gate、smoke或回归矩阵。单3090足够，不触发shutdown。

最后一个shard完成后，prep r1在首个scene-1045完成官方DriveStudio处理后因旧固定计数`1176 images / 196 lidar`
错误阻断；该scene按官方`interpolate_N=4`与metadata `nbr_samples=41`正确产生`1206 / 201`。登记`V64-F13
recovery_frozen_pre_quality`：期望帧数改为`(nbr_samples-1)*5+1`，r2只复用现有临时raw与已完成scene，不重扫tar。
与此同时scene-1045已直接完成IR-WM：`12 targets / 552,980,744 bytes / 46.2881 s / peak 4.1308 GiB`，native
feature完整，calibration/confirmation quality均未读。r1失败leaf与首场景probe均保留；下一动作是push恢复后直接启动prep r2。

## WorldSim V6.4 独立 calibration/confirmation cohort 已冻结（2026-08-26）

状态：`v64_calibration_confirmation_cohort_preregistered`；active task=
`WS-V64-P6-CALIBRATION-SIDECAR-01`；active hypothesis=`WS-V64-H-P6C-001`；calibration/confirmation quality read=
`false/false`。

从冻结IR-WM train temporal metadata的700 scene中排除V6.1–V6.3精确21个quality/config scene与当前V64六scene，
再要求>=40 samples，得到612个候选。只用description按night/rain/construction/vulnerable-transit四strata、seed0各取6个；
每stratum前4个组成16-scene calibration，后2个组成8-scene untouched confirmation。总24 scene×12 target=`288 units`，
未读取Occupancy/UQ/hidden-FREE/model quality。完整名单与索引=
`docs/autoresearch/worldsim_v64/P6_CALIBRATION_COHORT_FREEZE.md`。

资源策略：官方本地只读tar一次批量提取到精确临时目录，24个DriveStudio processed完成后删除该可重建raw batch；已有raw
不变。预计processed+native持久化约`21.6 GiB`，从当前`56 GiB`保留约`34 GiB`；双worker显存沿用`8.27 GiB`上界，
单3090足够。配置=`configs/worldsim_v64/p6_calibration_confirmation_sidecars_v1.yaml`；prep runner=
`scripts/prepare_worldsim_v64_calibration_batch.py`；sidecar wrapper已泛化读取配置partition。只做编译、提交push，然后直接准备
数据和288-unit sidecar；不加smoke/regression，不加hash/checksum/fingerprint。

## WorldSim V6.4 supervised hidden-FREE ranking 支持 / 独立校准下一步（2026-08-26）

状态：`v64_supervised_risk_supported_selective_calibration_cohort_next`；completed task=
`WS-V64-P5-SUPERVISED-RISK-01`；hypothesis=`WS-V64-H-P5-001 supported_ranking_only`；canonical=
`run://worldsim_v64/WS-V64-P5-SUPERVISED-RISK-01/20260826T093000Z__supervised-risk-s0-r1`。

固定logistic head在200,000 fit点（hidden-FREE=`18,242`）上训练，并对P4N完全相同的333,009点fresh denominator一次
评分。pooled U2/U3 AUROC=`0.518545/0.658118`，gain=`+0.139573`；U3 AUPRC=`0.148720`；scene-0359/0998
AUROC=`0.640682/0.636266`。两条绝对门`pooled>=0.60`与`both scenes>=0.55`全部通过。50% coverage pooled risk=
`0.049098`，低于prevalence=`0.082565`和U2=`0.076917`。CPU wall=`17.3115 s`、peak RSS=`0.8592 GiB`，无多卡需求。

该结果只支持supervised ranking transfer。pooled/scene FPR@95TPR仍为`0.867738/0.859069/0.907021`，因此登记
`V64-F11 active algorithm/evaluation`，禁止calibration、authority或安全claim。完整收口=
`docs/autoresearch/worldsim_v64/P5_SUPERVISED_RISK_CLOSEOUT.md`。

按ICLR 2024 Conformal Risk Control及官方实现，下一步先从quality-unread场景metadata-only冻结新的scene-disjoint
calibration与confirmation cohort；当前两evaluation scene不得参与阈值选择。先更新并push本里程碑，再做候选审计；不加重复
训练或测试矩阵。

## WorldSim V6.4 fit-only supervised risk 已冻结 / 待一次执行（2026-08-26）

状态：`v64_supervised_hidden_free_risk_preregistered`；active task=`WS-V64-P5-SUPERVISED-RISK-01`；active hypothesis=
`WS-V64-H-P5-001`；evaluation score read=`false`。正式run固定为
`20260826T093000Z__supervised-risk-s0-r1`。

P4N已按`V64-F10`收口为relative-only/weak-absolute。检索OCCUQ、ReliOcc、EvOcc的supervised/hybrid uncertainty路线后，
本恢复只复用P4N已拟合的StandardScaler+PCA-16，在四个fit scene的同一`200,000`点上用hidden-FREE标签拟合一个固定
logistic head：`C=1,class_weight=balanced,lbfgs,max_iter=200,seed0`；不使用scene ID。两fresh evaluation scene、
333,009点分母与U0/U2 comparator不变。

唯一两门预先固定为pooled U3 AUROC `>=0.60`且两个scene各自AUROC均`>=0.55`。禁止参数/feature/seed/denominator/
gate sweep、额外split或repeat；通过也只支持supervised ranking mechanism，不支持calibration、authority、conditional
coverage、安全、LoRA或downstream compiler。完整freeze=
`docs/autoresearch/worldsim_v64/P5_SUPERVISED_RISK_FREEZE.md`。只做源码编译后提交push并直接执行，无额外smoke/regression；
CPU-only，无多卡需求。

## WorldSim V6.4 fresh native-voxel UQ 相对门通过 / 绝对能力弱（2026-08-26）

状态：`v64_fresh_uq_relative_gate_pass_weak_absolute_supervised_risk_next`；completed task=
`WS-V64-P4N-FRESH-NATIVE-VOXEL-UQ-01`；hypothesis=`WS-V64-H-P4N-001 supported_relative_only_weak_absolute`；canonical=
`run://worldsim_v64/WS-V64-P4N-FRESH-NATIVE-VOXEL-UQ-01/20260826T091500Z__fresh-native-voxel-uq-s0-r2`。

四个fit scene固定采样`200,000`点；两个fresh evaluation scene完整评分`333,009`个unique native boundary voxels，
hidden-FREE=`27,495`、prevalence=`0.082565`。pooled最佳U0/U2 AUROC=`0.435498/0.518545`，增量=
`+0.083047`；AUPRC=`0.070965/0.085650`，scene support=`2/2`，故两条冻结相对门通过。CPU wall=
`22.3767 s`、peak RSS=`1.0705 GiB`，无多卡需求。

但scene-0359/0998的U2 AUROC分别仅`0.498387/0.498295`，FPR@95TPR=`0.965465/0.960623`；scene-0359
AUPRC低于其prevalence，scene-0998在50% coverage的risk也高于prevalence。pooled提升可能部分来自跨scene的prevalence/
score shift，不能升级为场景内可靠ranking。登记=`V64-F10 active algorithm/evaluation`；不扫描GMM/PCA/seed/分母/门槛，
不解锁authority、calibration、conditional coverage或安全claim。完整收口=
`docs/autoresearch/worldsim_v64/P4N_FRESH_UQ_CLOSEOUT.md`。

检索OCCUQ、ReliOcc与EvOcc后，下一步只预注册一个fit-only supervised hidden-FREE risk head：复用同一PCA-16表示、
同一200k fit点与同一333,009 evaluation分母，固定一次训练和一次fresh评分；不做额外smoke/regression或参数sweep。

## WorldSim V6.4 native-voxel fit 接口恢复已冻结（2026-08-26）

状态：`v64_native_voxel_global_gmm_recovery_preregistered`；active task=`WS-V64-P4N-FRESH-NATIVE-VOXEL-UQ-01`；
hypothesis=`WS-V64-H-P4N-001`；evaluation score read=`false`。

r1=`run://worldsim_v64/WS-V64-P4N-FRESH-NATIVE-VOXEL-UQ-01/20260826T090000Z__fresh-native-voxel-uq-s0-r1`
完成fit采样后，冻结occupied-boundary内预测FREE geometry组只有`43`点，小于GMM-4最低`80`点，在拟合阶段停止；
run只有8 KiB resolved/status，无model、无evaluation metrics。登记=`V64-F09 resolved_pre_evaluation`。

OCCUQ官方代码按真实voxel class收集feature，推理时跨类别密度边缘化；它不按待评region的预测类强制分别拟合。当前region
本身已是occupied boundary，恢复固定为一个boundary-global diagonal GMM-4。PCA-16、4 components、seed0、50k/fit-scene、
denominator、scene、targets、U0与两条gate均不变，不复制43点、不降样本线、不读evaluation补fit。完整修订=
`docs/autoresearch/worldsim_v64/P4N_NATIVE_VOXEL_UQ_RECOVERY_FREEZE.md`。

下一步编译、提交push后仅执行r2=`20260826T091500Z__fresh-native-voxel-uq-s0-r2`；CPU-only，无多卡需求。

## WorldSim V6.4 overbuilt surface 已停止 / native-voxel UQ 恢复冻结（2026-08-26）

状态：`v64_surface_resource_abort_native_voxel_uq_preregistered`；active task=
`WS-V64-P4N-FRESH-NATIVE-VOXEL-UQ-01`；active hypothesis=`WS-V64-H-P4N-001`；U0/U2 score read=`false`。

冻结V6.3 surface compiler历史72-unit wall=`47,568.47 s (13.21 h)`、max unit=`3,334.28 s`。fresh surface r1=
`run://worldsim_v64/WS-V64-P2S-FRESH-SURFACE-CORPUS-01/20260826T084500Z__fresh-surface-s0-r1`运行约4分钟仍为
`0/72 units`，只产生4 KiB negative tests；其signed-distance/patch/normal/actor registry均不被UQ消费。已终止精确
PGID `12735`并保留partial，登记=`V64-F08 resolved_by_native_voxel_recovery`。

按OCCUQ的原生voxel-level GMM迁移为唯一native boundary denominator：native argmax OCC∪method observed OCC的6邻域边界，
再限制method UNKNOWN、非contradiction、target ROI valid；不重复native voxel。scene/targets/seed/PCA-16/GMM-4/U0与两条
AUROC gate均不变。完整freeze=`docs/autoresearch/worldsim_v64/P4N_NATIVE_VOXEL_UQ_RECOVERY_FREEZE.md`。

CuPy/cuCIM EDT虽可加速旧路径，但仍保留本任务无用全栈，故不安装依赖、不继续surface。新r1 CPU-only，单3090与多卡均
不需要。下一步只做编译检查、提交push，然后执行`20260826T090000Z__fresh-native-voxel-uq-s0-r1`，不加probe。

## WorldSim V6.4 fresh evidence 完成 / surface 下一步（2026-08-26）

状态：`v64_fresh_evidence_complete_surface_next`；completed task=`WS-V64-P2E-FRESH-EVIDENCE-01`；active task=
`WS-V64-P2S-FRESH-SURFACE-CORPUS-01`；hypothesis=`WS-V64-H-P4-001`。canonical=
`run://worldsim_v64/WS-V64-P2E-FRESH-EVIDENCE-01/20260826T084000Z__fresh-evidence-s0-r1`。

固定6 scene / 72 target全部通过：output=`68,444,954 bytes`、wall=`118.2903 s`、max unit=`4.2750 s`、
source role overlap=`0`。unused query sampling关闭且`query_count=0`；method/dropout/target grid完整物化。fresh fit与evaluation
target evidence从本里程碑起为已读，但尚未计算U0/U2指标或gate。calibration/confirmation/test未读；CPU-only，无多卡需求。

首次launcher在本地PowerShell双引号内包含`$()`，被本地提前展开并在非repo目录报Git错误；远端run目录未创建。
按Microsoft PowerShell解析合同去掉subexpression后，同一冻结输入执行唯一r1，登记=`V64-F07 resolved_pre_run`。
完整收口=`docs/autoresearch/worldsim_v64/P2E_FRESH_EVIDENCE_CLOSEOUT.md`。下一步提交并push文档后直接运行固定surface r1，
不增加probe/smoke。

## WorldSim V6.4 fresh evidence/UQ 已冻结 / 待直接执行（2026-08-26）

状态：`v64_fresh_uq_preregistered_target_quality_unread`；active task=`WS-V64-P2E-FRESH-EVIDENCE-01`；active hypothesis=
`WS-V64-H-P4-001`。直接链路为72-unit evidence→72-unit surface→一次UQ，不展开完整compiler、不加smoke/regression。
fit=`0139,0230,0255,0994`；evaluation=`0359,0998`；evaluation target不进入fit。

U2保持retrospective的`PCA-16/GMM-4 diagonal/seed0`，U0保持三种softmax基线。唯一两门为pooled AUROC gain `>=0.02`
且scene support=`2/2`；其余指标只报告。通过也只支持fresh mechanism，禁止authority/calibration/conditional threshold/
LoRA/downstream claim；失败则关闭当前PCA/GMM表示，不在同数据sweep。完整freeze=
`docs/autoresearch/worldsim_v64/P2E_P4_FRESH_UQ_FREEZE.md`。

实现仅让既有V6.2 evidence runner读取config task ID、让V6.3 surface runner读取scene→native partition映射，并让既有V6.4
UQ runner读取同一映射与冻结gate；尚未读fresh target quality。预计三步均CPU，当前56 GiB磁盘和单3090足够，无多卡需求。
下一步编译检查、提交并push，然后直接运行evidence r1。

## WorldSim V6.4 fresh native sidecar 完成 / fresh evidence 下一步（2026-08-26）

状态：`v64_fresh_native_sidecars_complete_evidence_freeze_next`；completed task=
`WS-V64-P2-FRESH-NATIVE-SIDECAR-01`；hypothesis=`WS-V64-H-P2-001 supported_capability`；canonical=
`run://worldsim_v64/WS-V64-P2-FRESH-NATIVE-SIDECAR-01/20260826T082600Z__fresh-native-s0-r3`。

r3 在新leaf完成`6 scenes / 72 targets`，`all_native_features_complete=true`、output=`3,317,884,573 bytes`、
wall=`172.2085 s`、maximum worker peak=`4.1314 GiB`、双worker peak sum upper bound=`8.2628 GiB`。
prototype、target evidence、calibration、confirmation、exact-once test读取均为false。新增evaluation scene index
`276/756`已从本机raw数据物化，各为`196 LiDAR + 1,176 images`；无额外下载。单3090资源充足，无多卡需求。

完整收口=`docs/autoresearch/worldsim_v64/P2_FRESH_SIDECAR_CLOSEOUT.md`。r2 blocked partial原样保留且未复用。
正式run failure delta=`none`；收口reader首次错误假定文件名`P2_NATIVE_SUMMARY.json`，在读取前`FileNotFoundError`；
按实际run目录枚举改读继承extractor的`P2_SUMMARY.json`，登记=`V64-F06 resolved_post_run_read`，canonical未改变。

该里程碑只支持sidecar capability，不支持fresh UQ或authority。下一步直接复用既有evidence materializer建立fresh target
supervision，并在任何evaluation quality read前冻结保持不变的PCA-16/GMM-4/seed0 evaluator；不做冗长测试或参数sweep。

## WorldSim V6.4 fresh temporal metadata 恢复已冻结（2026-08-26）

状态：`v64_fresh_temporal_metadata_recovery_preregistered`；active task=`WS-V64-P2-FRESH-NATIVE-SIDECAR-01`；
hypothesis=`WS-V64-H-P2-001`。r2=
`run://worldsim_v64/WS-V64-P2-FRESH-NATIVE-SIDECAR-01/20260826T081500Z__fresh-native-s0-r2`为 blocked partial，
不是 canonical：初始 cohort 中的 val-split scene 不在冻结的`nuscenes_temporal_infos_train.pkl`，`scene-0100`与
`scene-0632`在 scene lookup 处 `KeyError`；只有`scene-0230`完成 12 units，run leaf=`528 MiB`、该 worker peak=
`4.1305 GiB`。target evidence、Occupancy/UQ quality、calibration、confirmation与test均未读。

按 IR-WM/BEVFormer temporal metadata 合同，pre-quality recovery 只加入 metadata capability 条件。新 fit=
`scene-0139,scene-0230,scene-0255,scene-0994`；新 evaluation=`scene-0359,scene-0998`；仍为每scene 12 targets、
总计72 units、seed0。六场景均未进入 V6.1–V6.3 quality ledger且存在于冻结 train temporal metadata；evaluation 两场景
将从本机已有 raw nuScenes 通过官方 DriveStudio preprocessing 物化。r2 保留且不复用，正式运行使用独立 r3 leaf。

本次登记=`V64-F05 resolved_pre_quality_read`；完整修订见
`docs/autoresearch/worldsim_v64/P2_FRESH_COHORT_FREEZE.md`。单卡预算仍足够，无多卡需求。下一步先提交并push恢复冻结，
再预处理 scene index `276/756`并直接执行r3，不增加smoke/regression。

## WorldSim V6.4 fresh sidecar 入口恢复已就绪（2026-08-26）

`WS-V64-P2-FRESH-NATIVE-SIDECAR-01`首次 formal 入口在 run leaf、GPU、数据读取之前因 task 父目录不存在而失败；
`shutil.disk_usage(run_dir.parent)`按 Python 合同要求现存路径，触发 `FileNotFoundError`。失败没有 canonical run、
quality 或资源消耗，登记=`V64-F04 resolved_pre_data_read`。

唯一恢复是在 wrapper 调用冻结 V6.3 extractor 前创建 task 父目录；cohort、72-unit denominator、IR-WM、seed、双 worker、
磁盘门与输出合同均不变。恢复提交后直接用新 run ID `r2`执行，不增加 smoke。

## WorldSim V6.4 compact fresh sidecar 已冻结 / 待提取（2026-08-26）

状态：`v64_fresh_native_sidecar_preregistered`；active task=`WS-V64-P2-FRESH-NATIVE-SIDECAR-01`；hypothesis=
`WS-V64-H-P2-001`。V6.1–V6.3 quality ledger 排除 21 个 scene 后，本机已有 processed 且未进入该三版 quality 的候选
为 9 个；只按 description、帧数与传感器完整性冻结 compact 6-scene cohort，不使用模型质量。

fresh fit=`scene-0100,scene-0230,scene-0632,scene-0781`；fresh evaluation=
`scene-0800,scene-0994`；每 scene 12 targets，共 72 units。完整冻结=
`docs/autoresearch/worldsim_v64/P2_FRESH_COHORT_FREEZE.md`，配置=
`configs/worldsim_v64/p2_fresh_native_sidecars_v1.yaml`。evaluation target 未读。

提取直接复用 V6.3 已通过的 IR-WM native worker，不新增 smoke。预计输出约 3.4 GiB、双 worker 显存上界约 8.3 GiB，
当前单卡 3090 与约 60 GiB 磁盘余量足够，无多卡需求。failure ledger delta=`none`。下一步提交并 push prereg 后
直接执行一次 72-unit formal。

## WorldSim V6.4 U2 retrospective 有信号 / fresh cohort 下一步（2026-08-26）

状态：`v64_retrospective_u2_signal_supported_fresh_cohort_next`；completed task=`WS-V64-P3-NATIVE-UQ-01`；
hypothesis=`WS-V64-H-P3-001 supported_retrospective`。canonical run=
`run://worldsim_v64/WS-V64-P3-NATIVE-UQ-01/20260826T080200Z__uq-retrospective-s0-r1`。

在 V6.3 旧机制集上，四个 fit scene 抽取 200,000 点拟合，两个 evaluation scene 的 3,169,645 个 eligible 点完整评分。
U2 feature-density pooled AUROC/AUPRC=`0.550470/0.076027`，最佳 U0=`0.497324/0.059739`，绝对增量=
`+0.053146/+0.016288`；FPR@95TPR 从`0.968577`降到`0.942892`。scene-0450 与 scene-1089 的 U2
AUROC=`0.580307/0.530461`，都优于各自最佳 U0。50% coverage 的 hidden-FREE risk=`0.052620`，低于总体
prevalence=`0.060847`，而最佳 U0 为`0.059330`。

这只支持 native feature-density 存在可迁移信号；高 FPR@95TPR、旧 scene 与 scene-1089 的有限 separation 都禁止
authority、校准或安全 claim。完整收口=`docs/autoresearch/worldsim_v64/P3_RETROSPECTIVE_CLOSEOUT.md`。运行使用 CPU，
wall=`49.964 s`、peak RSS=`1.044 GiB`，无多卡需求。下一步直接选取小型 metadata-only fresh cohort，保持同一
PCA-16/GMM-4/seed，不做 sweep。

本里程碑运行内 failure delta=`none`；外围操作新增并恢复 `V64-F02`（GitHub push 未走 LocalTUN proxy）和
`V64-F03`（非登录 summary reader 未激活 conda）。两者都没有改变 canonical run 或质量结果。

## WorldSim V6.4 最小 UQ 机制实验已冻结 / 待正式执行（2026-08-26）

状态：`v64_core_uq_retrospective_preregistered`；active task=`WS-V64-P3-NATIVE-UQ-01`；hypothesis=
`WS-V64-H-P3-001`。按用户最新指令不机械展开完整 V6.4 plan，直接比较原生 IR-WM softmax U0 与 feature-density U2。
冻结协议=`docs/autoresearch/worldsim_v64/P1_CORE_UQ_FREEZE.md`，配置=
`configs/worldsim_v64/p3_retrospective_uq_v1.yaml`。

本轮只把 V6.3 的 4 train scene / 2 selection scene 降级为 retrospective mechanism set：PCA/GMM 只拟合四个 train
scene，两个 evaluation scene 的 target 不进入拟合；结果无论正负都不构成 V6.4 fresh claim。U2 固定为
`17D logits + 256D BEV -> standardize -> PCA-16 -> FREE/OCC-conditioned 4-component diagonal GMM`；U0 为
max-probability、entropy 与 margin。当前不训练 aleatoric head，不引入 Surface、LoRA、scene ID 或阈值 sweep。

代码与配置已 staged，正式结果尚未读取；资源为 CPU + mmap 旧 sidecar，GPU 不占用，因此没有多卡需求。failure ledger
refs=`V63-F02,V63-F19,V63-F24,V64-F01`，delta=`none`。下一步仅做源码编译检查、提交并 push prereg，然后启动一次
正式机制运行。

## WorldSim V6.4 P0 完成 / P1 已解锁（2026-08-26）

状态：`v64_p0_complete_p1_unlocked`；分支=`research/worldsim-v6.4-native-uq`。分支从冻结的
`research/worldsim-v6.3-surface-tail@c192955`直接建立，V6.4 计划提交=`ca930a0`，没有从 main 重新起线。V6.3 的
`v63_surface_architecture_family_closed_negative_p7_locked`、`V63-F24`、B4/B5/M0 与 P7--P11 未执行/未读取边界保持不变。

P0 Git 前置已完成：`origin/main`从`bcd4143`普通快进到`c192955`，并保留
`origin/integration/worldsim-v6.3-to-main@c192955`。定向测试第一次用控制台`pytest`时在 collection 前因仓库根目录未进入
`sys.path`而失败；按 pytest 官方入口合同改用`python -m pytest -q tests/worldsim_v62/test_projection.py`后为`1 passed`。
该恢复没有创建 formal run、没有触达 GPU/数据/质量，也没有扩展测试矩阵；统一失败登记=`V64-F01 resolved_pre_quality_read`。

P0 证据=`docs/autoresearch/worldsim_v64/P0_SCOPE.md`与
`docs/autoresearch/worldsim_v64/AUTORESEARCH_STATE.current.json`。当前没有读取任何 V6.4 quality，默认资源仍为单卡
RTX 3090 24GB；磁盘观测余量约`60 GiB`，P1 必须冻结 sidecar 磁盘预算与资源停止线。下一步只执行
`WS-V64-P1-NOVELTY-PROTOCOL-01`：完成来源/许可证审计并冻结方法、fresh cohort、gates 与资源合同。

## WorldSim V6.3 arXiv evidence frozen / documentation audit complete（2026-08-26）

状态仍为`v63_surface_architecture_family_closed_negative_p7_locked`；本次只完成报告证据收口，没有新run、没有新failure ID，
也没有改变任何实验、门槛或终态。报告单一入口=
`docs/autoresearch/worldsim_v63/ARXIV_EVIDENCE_INDEX.md`，已把P0--P6的canonical run、正/负机制结论、验证层级、
失败分类、未执行臂与claim boundary映射到主计划第35节的技术报告结构。

最重要的三级语义已冻结：P5 epoch3只能称`best training-objective checkpoint`，因retention=`0`而不是candidate；P5R
epoch6只是在训练侧通过原gate并解锁P6；P6 B3 epoch1虽为feasible training candidate，最终仍在两scene得到`0/2`
支持而被stage reject。因此V6.3没有version-level best SurfNCC candidate，B4/B5/M0和P7--P11为未执行/locked，不是
rejected结果。

文档审计逐项对照P6 baseline、B3 train、B3 eval三个canonical summary：共同分母=`24 units/2 scenes`，B2/B3
pooled tail=`0.491496/0.608174`、OCC area=`2298450/1047186`，逐scene相对改善=`-19.852%/-41.008%`，与
`RESEARCH_STATUS.md`、`EXPERIMENTS.md`、`RESEARCH_FAILURES.md`、state和P6 closeout一致。legacy、calibration、
confirmation、exact-once test仍未读；当前只授权以冻结证据撰写技术报告，或为新版本重新预注册uncertainty与
conditional-coverage方案。

## WorldSim V6.3 P6 B3 rejected / surface architecture family closed / P7 locked（2026-08-26）

状态：`v63_surface_architecture_family_closed_negative_p7_locked`；completed task=`WS-V63-P6-DEVELOPMENT-AB-01`；
rejected hypothesis=`WS-V63-H-P6-001`；`H-P6-002/H-P6-003=closed_without_execution`；P7–P11、legacy、calibration、
confirmation与exact-once test均未解锁/未读取。

B3 stage canonical=
`run://worldsim_v63/WS-V63-P6-DEVELOPMENT-AB-01/20260826T014500Z__b3-eval-s0-r1`完整读取同一24-unit、两scene
denominator，wall=`242.604s`、peak=`0.130475 GiB`、hard violations=`0`、execution capability=`passed`，但stage=
`failed`、supporting scenes=`0/2`。pooled B3 common surface CVaR=`0.608174`，相对Native B2=`0.491496`恶化
`23.74%`；OCC surface area=`1047186`，仅B2的`45.56%`；proposal false-safe surrogate=`0.515384`，也比B2
`0.396840`恶化`29.87%`。

逐scene失败是决定性的：scene-0450 tail=`0.596685 vs 0.497850`（相对改善=`-19.85%`）、area ratio=
`0.406270`、source-valid UNKNOWN=`0.651678>0.60`；scene-1089 tail=`0.655861 vs 0.465122`（`-41.01%`）、
area ratio=`0.499323`。两scene hard0、retention、case coverage、actor/static coverage都过门，scene-1089 UNKNOWN也过门，
但两scene tail与area均失败，所以不存在可由pooled均值掩盖的局部支持。

按主计划Stop 2，B3不优于Native B2即关闭surface architecture family，不继续B4 Surface-Max、B5 Surface-CVaR或M0
authority，也没有frozen P6 M0可交给P7。训练内epoch1 candidate因此只证明约束能避免全UNKNOWN，不能升级为P6 candidate。
失败登记=`V63-F24 active route-closed`；完整closeout=
`docs/autoresearch/worldsim_v63/P6_SURFACE_FAMILY_CLOSEOUT.md`。

遇到终态后已检索顶会/优秀开源：EvOcc（CVPR 2025）以evidential target显式建模unobserved/contradiction，ReliOcc
（IJCAI 2025）与OCCUQ（ICRA 2025，开源）提供hybrid及feature-level aleatoric/epistemic uncertainty，UAI 2024
conditional robust optimization联合优化decision risk与conditional coverage。若未来以新版本复开，应预注册新的uncertainty
representation和scene/stratum-conditional coverage约束，并使用fresh development scenes；不得在V6.3上换seed、加模型、
降低area/UNKNOWN门、直接跑CVaR或读取legacy/H/T救结果。当前V6.3 autoresearch按预注册终态完成。

## WorldSim V6.3 P6 B3 mean training complete / stage evaluation unlocked（2026-08-26）

状态：`v63_p6_b3_training_complete_stage_eval_unlocked`；active task=`WS-V63-P6-DEVELOPMENT-AB-01`；active hypothesis=
`WS-V63-H-P6-001`；B3 common-evaluation verdict=`pending`；B4/B5/M0仍locked。

B3 training canonical=
`run://worldsim_v63/WS-V63-P6-DEVELOPMENT-AB-01/20260825T152000Z__b3-mean-s0-r1`按冻结mean aggregator完成
`5 epochs/1280 optimizer steps`，wall=`9181.220s (2.550h)`、peak=`0.400373 GiB`、finite=`true`、累计hard
violations=`0`，48 train+24 selection denominator完整，GPU已释放。warm start仅加载P5 epoch3 model，fresh AdamW、seed0；
authority loss/veto关闭，hard projection与三项primal-dual约束未改。

唯一best training candidate为epoch 1：hard=`0`、retention=`0.636863`、OCC coverage=`0.285326`、UNKNOWN=
`0.550411`，训练内四门可行；hidden-FREE common selection tail=`0.608174`、rank=`0.080258`、tail+rank=
`0.688432`、secondary accuracy=`0.621908`。epoch 0 tail更低但retention/UNKNOWN失门；epoch 2 tail+rank=
`0.310256`却retention/coverage/UNKNOWN=`0.336785/0.084310/0.791651`三门失败；epoch 3也失门；epoch 4 retention/
coverage过门但UNKNOWN=`0.698930`失败。连续三轮无更优feasible candidate后按patience=`3`终止。

因此`candidate_promotable=true`仅表示训练内checkpoint合同通过，不等于H-P6-001或B3 stage pass。下一步只用冻结epoch1
checkpoint运行统一B3 evaluator，对两scene分别检查相对B2 area、actor/static、retention/UNKNOWN/case coverage与common
surface CVaR改善>=2%；不重训、不选threshold。failure ledger delta=`none`。

## WorldSim V6.3 P6 native baselines complete / B3 Surface-Mean unlocked（2026-08-25）

状态：`v63_p6_baselines_complete_b3_training_unlocked`；active task=`WS-V63-P6-DEVELOPMENT-AB-01`；active hypothesis=
`WS-V63-H-P6-001`；下一阶段=`B3 independent Surface-Mean train/eval`；B4/B5/M0仍locked。

P6 baseline canonical=
`run://worldsim_v63/WS-V63-P6-DEVELOPMENT-AB-01/20260825T151200Z__baselines-s0-r1`完成全部24个selection units、
两独立scene与72条arm-unit records，wall=`36.8725s`、peak=`0.100660 GiB`、execution capability=`passed`。仅P6
selection quality已读；threshold fitted=`false`，legacy/calibration/confirmation/exact-once均未读，源码clean且GPU已释放。

B0 native argmax有`420297` hard violations；B1应用冻结projection后hard=`0`，但common hidden-FREE surface CVaR仍为
`0.791098`。冻结Native B2进一步降到pooled=`0.491496`，逐scene=`0.497850/0.465122`；B2 hard=`0`、safe-OCC
retention=`0.851056`、source-valid UNKNOWN=`0.266284`、accepted case coverage=`1.0`、emitted OCC area=
`2298450 points (coverage 0.626256)`、actor/static accepted proposals=`296/23562`。逐scene B3比较所需accepted area基准=
`1079847/1218603 points`，tail至少需分别低于约`0.487893/0.455820`才能达到冻结2% improvement。

该结果只冻结B2 comparator，不支持H-P6-001，也不把高B2 coverage或低于B1的tail解释为安全晋级。按预注册顺序，下一步从
同一P5 epoch3 model-only起点正式训练B3 mean arm；只有B3在两scene同时通过hard/retention/UNKNOWN/case/area/
actor/static与tail门，才解锁B4。failure ledger delta=`none`。

## WorldSim V6.3 P6 matched-AB implementation staged / baseline formal ready（2026-08-25）

状态：`v63_p6_implementation_staged_quality_unread`；active task=`WS-V63-P6-DEVELOPMENT-AB-01`；active hypothesis=
`WS-V63-H-P6-001`；P6 quality read=`false`；下一合法动作仅为正式B0/B1/B2 baseline evaluation。

P6实现已绑定冻结协议：`scripts/run_worldsim_v63_p6_development_ab.py`在同一24-unit、两scene denominator上实现B0 native
argmax、B1 hard projection、B2冻结V6.2 CPSC-Lite与统一surface hidden-FREE worst-10% CVaR；
`scripts/run_worldsim_v63_p6_ablation_train.py`从同一P5 epoch-3 model-only起点分别训练B3/B4/B5，只改变hidden-FREE
aggregator=`mean/max/cvar`并关闭authority loss/veto。最终anti-trivial gate同时检查hard0、retention、source-valid UNKNOWN、
case coverage、相对B2 accepted area和actor/static coverage；P7 threshold、calibration、legacy、confirmation与exact-once仍未读。

实现审计补齐了两个非科学漂移：B2 query batch固定为`16384`；B3–B5 checkpoint selection的proposal rank改为无authority的
`P(OCC)*q_HF`风险，避免训练已关闭authority而selection仍偷用旧`q_AUTH`组合。最终跨臂metric仍统一为冻结CVaR，不随
训练aggregator改变。一次聚焦检查验证四个runner/module可编译、三个聚合器数值/梯度和arm合同正确；没有capacity smoke、
额外seed、threshold或quality read。failure ledger delta=`none`；下一步在clean source提交后只运行B0/B1/B2 formal。

首次baseline入口使用`python scripts/run_worldsim_v63_p6_development_ab.py`，在run leaf创建、数据/checkpoint/GPU读取前因
`ModuleNotFoundError: motion_proj`终止，canonical run=`null`、P6 quality仍未读。Python官方路径合同说明直接脚本只把
脚本目录加入`sys.path`，而`python -m`把当前repo root加入模块搜索路径；因此恢复仅改launcher为
`python -m scripts.run_worldsim_v63_p6_development_ab`。`--help`入口验证通过，源码/配置/合同不变，登记`V63-F23 resolved`。

## WorldSim V6.3 P6 matched-AB protocol frozen / staged ablation training required（2026-08-25）

状态：`v63_p6_preregistered_implementation_pending`；active task=`WS-V63-P6-DEVELOPMENT-AB-01`；active hypothesis=
`WS-V63-H-P6-001`；P6 quality read=`not_started`；P7/legacy/H/T均locked。

P6接口审计发现：若B3/B4/B5只在同一M0输出上事后替换mean/max/CVaR，三者state decisions相同，且固定非负分布满足
`mean<=upper-tail CVaR<=max`，冻结的“B5相对better B3/B4改善>=2%”将数学上不可满足。P6因此预注册为独立matched loss
ablations：B3/B4/B5从同一P5 epoch3 model-only起点、fresh AdamW、seed0、同一48+24 denominator分别训练，仅改变
hidden-FREE aggregator=`mean/max/CVaR`并关闭authority loss/veto；其余model/data/dropout/FP16/hard projection/horizon和
三项primal-dual anti-trivial约束完全一致。M0保持P5R epoch6冻结candidate，不重训。

P6不在selection上选择case threshold。未校准anti-trivial acceptance固定为一个target unit最终至少发出一个OCC surface point；
accepted area为发出OCC的surface point数，Actor/static按最终发出OCC的proposal计数。所有learned arms再用同一个exact
surface hidden-FREE worst-10% CVaR比较，而不是各报训练aggregator。执行顺序冻结为B0→B1→B2→B3→B4→B5→M0；
B3或B5阶段失败即按plan stop rule关闭后续family。协议=
`docs/autoresearch/worldsim_v63/P6_DEVELOPMENT_AB_PREREG.md`，配置=
`configs/worldsim_v63/p6_development_ab_v1.yaml`。下一步只实现baseline evaluator与通用B3/B4/B5 trainer，不做capacity
smoke或P6 quality read。

协议同步后的首次inline `python -c` YAML检查因PowerShell→SSH引号转义在文件读取前`SyntaxError`，项目内容未变；改为将
只读Python经stdin传给远端解释器后，arm order与task identity验证通过。它并入既有`V63-F22`同类操作防重复，不新增failure。

## WorldSim V6.3 P5R constrained recovery passed / promotable SurfNCC candidate frozen / P6 unlocked（2026-08-25）

状态：`v63_p5r_complete_candidate_promotable_p6_unlocked`；completed task=
`WS-V63-P5R-CONSTRAINED-SURFNCC-TRAIN-01`；supported hypothesis=`WS-V63-H-P5R-001`；next task=
`WS-V63-P6-DEVELOPMENT-AB-01`；P6=`unlocked_pending_preregistration`，P7/confirmation/test仍locked。

P5R canonical=
`run://worldsim_v63/WS-V63-P5R-CONSTRAINED-SURFNCC-TRAIN-01/20260825T091631Z__constrained-train-s0-r1`
按预注册proxy primal-dual合同完成`10 epochs/2560 optimizer steps`，wall=`18400.384s (5.111h)`、peak=
`0.426807 GiB`、finite training=`true`、累计hard violations=`0`。训练/selection仍为`4/2 scene-disjoint scenes`；
calibration quality、confirmation、exact-once test与P6/H/T均未读。源码在formal期间保持clean，结束时GPU已释放，磁盘剩余约
`60 GiB`。

冻结best candidate为epoch 6，而不是普通argmin training loss：hard violations=`0`、safe-OCC retention=
`0.721226>=0.60`、emitted-OCC coverage=`0.114148>=0.10`、source-valid non-UNKNOWN=`0.686101>=0.40`
（UNKNOWN=`0.313899<=0.60`），四个exact gate全部通过；hidden-FREE tail=`0.464393`、matched-rank surrogate=
`0.056147`，candidate objective=`0.520541`。它因此是真正`candidate_promotable=true`的SurfNCC checkpoint。
secondary accuracy=`0.420739`偏低且coverage仅比门高`0.014148`，作为P6 matched-AB必须检验的局限完整保留，不能被
promotion boolean掩盖。

优化轨迹显示可行性与tail的真实张力：epoch 3首次全门可行但candidate objective=`0.857654`；epoch 5改善到
`0.620675`；epoch 6改善到`0.520541`。epoch 7因UNKNOWN=`0.635813`失门，epoch 8虽tail+rank=`0.430119`但
OCC coverage=`0.090615`且UNKNOWN=`0.617304`失门，epoch 9虽tail+rank=`0.449681`但coverage=`0.098617`且
UNKNOWN=`0.646926`失门；因此它们都不能覆盖epoch 6。连续三轮无更优feasible candidate后按冻结patience=`3`停止，
没有追加epoch/seed/model/threshold/CVaR/dual-rate sweep。final multipliers为retention/emit-OCC/non-UNKNOWN=
`2.520619/0.0000295/1.537027`。

该run支持`WS-V63-H-P5R-001`：保持同一representation、hard projection与数据合同，仅把retention、emitted-OCC和
non-UNKNOWN作为约束，即可打破P5的positive-authority collapse并恢复可用candidate；它不证明P6 matched AB、独立校准、
confirmation或deployment。P6现只解锁设计/预注册与一次正式fresh development matched AB；在冻结B0–B5/M0顺序、
Native B2基线、surface/CVaR/authority消融与原晋级门前，不运行P6质量读。

P5R文档收口的首次远端备份命令被本地PowerShell提前展开变量，误在远端根创建`/docs`重复副本树；后续一次带
`$(...)`的清理保护也在破坏动作前被本地解释器拒绝。两次均未改项目工作树、run或原文件。精确列举后只删除可由原仓库
恢复的`/docs`重复树，显式绝对路径备份已成功写入`/tmp/worldsim_v63_pre_p5rclose_20260825T1440Z`。登记
`V63-F22 resolved`；以后PowerShell→SSH文件操作禁止插值远端变量/命令替换。

## WorldSim V6.3 P5D objective-collapse diagnosis passed / P5R preregistered（2026-08-25）

状态：`v63_p5d_complete_objective_collapse_confirmed_p5r_preregistered`；active task=
`WS-V63-P5R-CONSTRAINED-SURFNCC-TRAIN-01`；active hypothesis=`WS-V63-H-P5R-001`；P6=`locked`。

P5D canonical=`run://worldsim_v63/WS-V63-P5D-AUTHORITY-COLLAPSE-DIAGNOSTIC-01/20260825T084844Z__authority-diagnostic-s0-r2`
正式`passed=true`：完整读取`48 train units`，safe-OCC/hidden-FREE/UNKNOWN点数分别为
`62454/495817/6036885`；固定target17四单元gradient probe=`79 batches/609288 points`；optimizer steps=`0`、training=
`false`、hard violations=`0`、wall=`796.702s`、peak=`0.323995 GiB`。selection/P6/calibration/confirmation/test均未读。
H-P5D-002 supported，H-P5D-001入口失败已由本run闭合，`V63-F20 resolved`。

机制结论分三层：

- `risk/authority composition failure`被排除为主因：safe-OCC的raw/post-projection/post-authority decision counts完全相同，
  正确class index order=`FREE/OCCUPIED/UNKNOWN`下均为`153/0/62301`，authority veto=`0`；hard projection没有抹掉任何
  usable OCC。
- representation保留弱排序但authority supervision区分不足：safe-OCC vs hidden-FREE的raw `P(OCC)` binned AUC=
  `0.722684`，但绝对均值仅`0.006459 vs 0.004181`；`q_AUTH` AUC仅`0.578070`，中位数=`0.0205 vs 0.0145`。
  authority target prevalence在safe-OCC/hidden-FREE/UNKNOWN仅=`10.31%/8.26%/9.24%`，说明证据authority标签与安全OCC
  语义只有弱对齐。
- `objective optimization collapse`被支持为主根因：safe-OCC retention component loss mean=`0.968547`、P50=`0.996666`，
  已近饱和；冻结权重后的tail training-term全模型gradient mean=`1.555512`，是retention `0.281250`的`5.531x`；仅看
  direct tail仍为`1.715x`，state-head为`1.732x`。77个同时非零batch的tail-retention gradient cosine mean/P50=
  `-0.411568/-0.370905`，显示系统性方向冲突。raw模型因而把safe OCC概率整体压扁，而不是最后policy拒绝。

P5D artifact的`DECISION_STAGE_COUNTS.json`唯一描述性错误是`class_order`文字写成UNKNOWN/FREE/OCCUPIED；实际
`torch.bincount`数组索引由冻结常量`FREE=0/OCCUPIED=1/UNKNOWN=2`决定，underlying counts、groups、distributions、
gradients与机制结论均正确。canonical run保持不可变，runner已修正未来label，登记`V63-F21 resolved`，不为metadata文字
重跑13分钟正确诊断。

下一唯一训练hypothesis=`WS-V63-H-P5R-001`已预注册为proxy primal-dual constrained recovery：从P5 epoch3模型权重
warm-start、fresh AdamW，保留模型/数据/dropout/FP16/CVaR/12 epochs/seed0/hard projection；把safe-OCC retention>=0.60、
emitted OCC coverage>=0.10和non-UNKNOWN coverage>=0.40作为约束，原离散rate更新dual、可微`P(OCC)*q_AUTH`更新model。
dual step固定0.01且不sweep。只有四个原始离散gate全过的checkpoint可叫candidate；best progress永不冒充candidate。

P5R implementation已staged并解锁formal execution：配置=
`configs/worldsim_v63/p5r_constrained_surfncc_train_v1.yaml`，runner=
`scripts/run_worldsim_v63_p5r_constrained_train.py`。每个原accumulation-4 optimizer step后，用同四batch的离散
post-authority rates更新三个非负dual；model loss使用可微proxy violation。旧retention weighted term精确置0，其他state/
tail/rank/consistency/authority权重保持不变。runner分别保存`BEST_PROGRESS`与仅feasible时存在的`BEST_CANDIDATE`，terminal
也分开报告training capability和candidate promotion；不增加capacity probe或测试矩阵。

## WorldSim V6.3 P5 training capability passed / candidate rejected / P5D ready（2026-08-25）

状态：`v63_p5d_h002_entrance_recovery_ready`；active task=
`WS-V63-P5D-AUTHORITY-COLLAPSE-DIAGNOSTIC-01`；active hypothesis=`WS-V63-H-P5D-002`；P6=`locked`。

P5 canonical=`run://worldsim_v63/WS-V63-P5-SURFNCC-TRAIN-01/20260825T051530Z__surfncc-train-s0-r1`完成
`7 epochs/1792 optimizer steps`，wall=`12111.626s (3.364h)`、peak=`0.403084 GiB`、finite training=`true`、累计
hard violations=`0`，AMP initial/final均=`1024`、math SDPA与deterministic cuBLAS合同保持不变；train/selection仍严格为
`4/2 scene-disjoint scenes`，calibration/confirmation/test均未读。runner的`passed=true`只证明完整denominator训练、数值、
资源、硬投影和checkpoint产物能力，不是SurfNCC晋级结论。

冻结lexicographic objective选择epoch 3为**best training-objective checkpoint**：hidden-FREE tail=`0.0145069`、matched
rank surrogate=`0.0815163`、primary=`0.0960231`、hard violations=`0`。但同一checkpoint的safe-OCC retention=`0`、
emitted-OCC coverage=`0.0371977 < 0.10`、source-valid UNKNOWN=`0.861807 > 0.60`，因此不满足P1/P5防all-UNKNOWN与晋级
合同，`p5_candidate_promotable=false`。它不得称为best SurfNCC candidate，也不得解锁P6。epoch 6虽出现
retention=`0.0002227`，其primary=`0.1285593`且仍远未过门，patience按冻结规则终止；不据此追选checkpoint。

连续7个epoch与best checkpoint的hard violations均为0，支持observed FREE/OCC、contradiction、lifecycle与hard projection
继续保持冻结；当前失败位于无直接硬证据曲面的learned risk/authority路径。`SafeOCCRetention=0`比低coverage更直接指向
positive-authority collapse症状：危险/缺证据曲面与有正向OCC支持的安全曲面都被拒绝。根因尚不能在
representation/supervision、risk-authority composition与weighted-objective optimization之间武断选择，登记
`V63-F19 active_diagnostic_ready`。

P5D已预注册为仅训练集、零更新的机制诊断：全部48个train units分别统计safe-OCC/hidden-FREE/UNKNOWN三组的
`q_AUTH`、raw/post-projection `P(OCC)`、point/patch/proposal tail分布和三阶段decision转移；固定四个train target-17
units测量tail/retention/authority直接梯度幅值与tail-retention cosine。它不重采structural dropout、不读selection/P6/
calibration/H/T、不改threshold/gate/hard solver，也不增加seed/epoch/model/CVaR sweep。若证据支持objective collapse，
下一训练hypothesis只允许另行预注册proxy/primal-dual constrained optimization；简单把`lambda_ret`调大不授权。

P5D implementation已staged：配置=`configs/worldsim_v63/p5d_authority_collapse_diagnostic_v1.yaml`，runner=
`scripts/run_worldsim_v63_p5d_authority_diagnostic.py`。分布使用固定1000-bin streaming histogram并输出一张六面板图；
决策计数显式区分raw argmax、hard projection和authority veto；gradient probe同时报告raw/frozen-weighted全模型及分head
L2 norm。summary只验证finite/完整train denominator/zero hard violation/resource，不用新的质量阈值自动判根因。

H-P5D-001第一次formal入口在run directory、checkpoint/data read与GPU context前因新task namespace尚不存在而失败：runner
把`shutil.disk_usage`直接调用在不存在的`run_dir.parent`，触发`FileNotFoundError`；没有run leaf或科学结果。Python官方合同
要求`disk_usage(path)`接收已有filesystem path。登记`V63-F20 resolved_recovery_ready`；H-P5D-002只向上找到最近已存在父目录
做同一20 GiB disk检查，仍由runner随后创建唯一leaf。diagnostic groups、checkpoint、48+4 units、FP16、阈值、梯度、资源与
全部data locks不变，不新增smoke/regression矩阵。

## WorldSim V6.3 P4 capacity passed / P5 training unlocked（2026-08-25）

状态：`v63_p5_training_ready`；active task=`WS-V63-P5-SURFNCC-TRAIN-01`；active hypothesis=`WS-V63-H-P5-001`。

P3 canonical formal=`run://worldsim_v63/WS-V63-P3-SURFACE-CORPUS-01/20260824T154059Z__surface-dl-s20260824-r1`
正式通过：`6 scenes/72 targets`、`86,360 surfaces/111,282 patches/86,360 proposals/11,583,001 points`；
surface type=`3,042 route-support / 82,499 static-disocclusion / 790 actor / 29 actor-swept`。minimum normal-valid=`1.0`、
maximum patch=`940<=2048`、maximum surface=`181,752`、8/8 negative contracts、missing fields=`[]`、source overlap=`0`。
output=`333,197,992 bytes`，wall=`47,568.466s`（`13.213h`），maximum unit wall=`3,334.282s`；source在启动和终态均clean，
prototype/calibration/confirmation/test read均false。P3 hypothesis supported，P4 H-P4-002 execution正式解锁。

P3终态前统计语义审计发现，run内`hidden_free_count`实际保存的是全部`target==FREE`，缺少
`method==UNKNOWN && !method_contradiction`条件；point payload中的method/target/contradiction字段正确，P4/P5 loader及
P5 loss/selection均从点字段重算，故语料和模型路径不受影响。正式run保持不可变；72个原始NPZ一次重算得到target
FREE/OCC/UNKNOWN=`1,545,584/335,050/9,702,367`、correct hidden-FREE=`688,837`，旧summary的`1,545,584`
不得按hidden-FREE引用。未来materializer以additive v2同时区分这些字段，登记`V63-F16 resolved`，不重跑正确语料。

P4 H-P4-001已在任何真实P4运行或quality read前撤回：对P3最先完成40 units的method-only结构统计显示，40/40均有
proposal超过8192 points，最大=`173488`，其完整patch set最大=`417`。只看largest proposal首个chunk会系统性丢失
proposal interaction，不能证明冻结合同。按Set Transformer/Perceiver的分层set encoding迁移为H-P4-002：点图仍以
8192-point chunks有界执行，但先汇总完整proposal的全部patch tokens，再运行2层patch attention与唯一proposal token；
训练chunk以可微token替换no-graph cache中的对应位置。模型、units、2 optimizer steps、accum4与22 GiB ceiling均不变。
登记`V63-F10 resolved_preexecution`；P4实现/配置/预注册已staged，P3 formal pass后执行已解锁。
P4原`cvar_gradient_nonzero`曾用会同时收到BCE梯度的hidden-free head总梯度做代理，可能假阳性；现用
`autograd.grad(proposal_cvar.mean(), state/hidden-free/authority heads)`直接检查CVaR图，聚焦synthetic三条head路径均
finite/nonzero，登记`V63-F15 resolved_preexecution`。既有gate含义被校正，未增加新gate或实验分母。

P4 H002 r1=`run://worldsim_v63/WS-V63-P4-CAPACITY-01/20260825T045854Z__capacity-h002-s0-r1`在11.181s终态
`passed=false`：完整train/selection proposals执行，peak=`0.1961 GiB`、loss finite、direct CVaR三head gradient nonzero、
proposal-token gradient nonzero、hard violations=`0`、checkpoint reload成功；但FP16总gradient出现nonfinite，且CUDA
attention相同/重载forward max abs diff均=`9.0599e-6`，未过冻结的finite/exact-0 gate。PyTorch官方说明GradScaler初始
scale可导致FP16 overflow，CUDA SDPA后端也有不同确定性；唯一有界恢复固定AMP initial scale=`1024`并禁用flash/
memory-efficient SDPA、只用deterministic math backend。模型、FP16、units、steps、loss、gate与22GiB ceiling不变；
当时登记`V63-F17 active_recovery_ready`；现已由r3闭合为resolved。r1保持不可变，不写成算法失败。

P4 H002 r2=`run://worldsim_v63/WS-V63-P4-CAPACITY-01/20260825T050400Z__capacity-h002-s0-r2`在第一次CUDA math
attention forward、任何optimizer step或summary前被deterministic runtime拒绝：cuBLAS矩阵运算要求进程启动前设置
`CUBLAS_WORKSPACE_CONFIG`。r2叶目录为空，quality/calibration/confirmation/test均未读，不能写成F17恢复或capacity失败。
按NVIDIA cuBLAS与PyTorch官方确定性合同，r3在launcher和pre-torch-import runner双层固定`:4096:8`；约24 MiB workspace
开销仍远低于22 GiB ceiling。除该运行时前置条件外，r1已冻结的AMP scale=`1024`、math SDPA、deterministic algorithms
及所有模型/数据/FP16/dropout/loss/optimizer/steps/accum/gates均不变。当时登记`V63-F18 active_recovery_ready`；现已由
r3闭合为resolved。r3仍是F17唯一有界恢复的第一次实际执行。

P4 canonical r3=`run://worldsim_v63/WS-V63-P4-CAPACITY-01/20260825T051200Z__capacity-h002-s0-r3`在
`11.863s`正式`passed=true`：train/selection各`2 complete proposals / 16 chunks`，maximum full proposal=
`117,663 points / 263 patches`；peak=`0.256589 GiB`，AMP scale initial/final均=`1024`，loss与unscaled gradient finite，
direct CVaR三head及proposal-token gradient nonzero，hard violations=`0`，checkpoint reload成功，repeat/reload max diff均
`0.0`。quality/calibration/confirmation/test read均false。H-P4-002 supported，`V63-F17/F18 resolved`，P4收口并只解锁
已预注册的P5完整denominator训练。

P5完整denominator实现已staged并由P4 pass解锁、尚未执行：每个complete proposal在每epoch只生成一个semantic dropout selector，所有chunks
继承全局actor/static、安全标签与point count；masked evidence同步移除temporal/observed-actor与证据派生authority通道，
temporal-window则由保留sweeps重算。selection把全部chunks的hidden-FREE probability连接后计算exact proposal CVaR，
完整patch context驱动proposal attention；训练仅声明memory-bounded stochastic CVaR surrogate。最终决策先保留硬投影，
只把method-UNKNOWN且低authority的learned OCC转UNKNOWN，coverage/retention/accuracy共享同一decision。逐loss审计又发现
ranking若只按chunk共现配对会漏掉完整unit的nearest-size pairs；现从unit metadata一次生成同stratum一对一匹配，并让当前
完整patch-token cache通过可微proposal attention/risk head每unit计算一次，未采用Cross-Batch Memory的stale queue。
selection端又发现曾把24个selection units合并后跨scene/frame配对；现与训练一致，严格在完整scene/frame unit内匹配并对
有pair的unit等权平均，不允许跨案例规模巧合改变checkpoint排序，登记`V63-F14 resolved_preexecution`。
另发现surface-wide edges会随相邻patch是否同chunk而漂移，现把两层6-neighbor local aggregation绑定完整冻结patch（从不切分、
max2048），跨patch交互只走完整proposal attention；登记`V63-F11/F12/F13 resolved_preexecution`。聚焦审计中，
modular-forward等价检查覆盖12 outputs且max abs difference=`0.0`；另以两个完整patch跨packing验证有向边数均为`4`，
分属不同chunk的safe/unsafe proposals仍生成冻结pair=`[(0,1)]`，跨unit safe/unsafe则为`0 pair`。没有真实P5
data/training、threshold搜索或新增
smoke/regression矩阵。

P3 canonical probe=`run://worldsim_v63/WS-V63-P3-SURFACE-CORPUS-01/20260824T153526Z__surface-probe-s20260824-r6`
已通过：`1 unit / 191 surfaces / 498 patches / 191 proposals / 152,226 points`，output=`3,055,106 bytes`，
wall=`201.356s`。minimum normal-valid=`1.0`、maximum patch=`635<=2048`、8/8 negative contracts、
`missing_point_feature_fields=[]`；逐sweep state/contradiction、exact signed distances、patch-local coordinate、normalized
ray order与全部native/evidence/actor/authority字段均存在。按F16正确重算的target FREE/OCC/UNKNOWN=
`19,609/3,891/128,726`、hidden-FREE=`8,311`；旧registry的`19,609`不得按hidden-FREE引用。
prototype/calibration/confirmation/test read均false。

P3 probe gate与随后72-unit formal均正式通过（历史`V63-F03–F07`保留且resolved）；probe外推的2.0h低估了大量
large surfaces，实际formal为13.213h，但仍低于24h资源线。下一步只执行预注册P4 H-P4-002，不再运行P3 probe/formal。

P3 r5=`run://worldsim_v63/WS-V63-P3-SURFACE-CORPUS-01/20260824T152843Z__surface-probe-s20260824-r5`
完成`191 surfaces/498 patches/152,226 points/3,029,206 bytes`，wall=`188.725s`、runner `passed=true`；新增
signed distances、patch-local xyz、behind-hit、四类temporal counts、normalized ray order与actor observed-hit均可读取。
但P4 structural-dropout loader设计审计确认：聚合temporal counts不能忠实执行冻结的整段`temporal_window` dropout，
必须保留每个method sweep的state/contradiction。该发现仍属于同一`V63-F06` frozen-schema completeness根因；r5只记
aggregate schema capability，不放行formal。

r6增加`[point,sweep]` temporal state/contradiction矩阵，并把P1必需字段清单写入P3配置，runner只做一次直接缺字段检查。
VideoMAE/Masked Spatio-Temporal Structure Prediction支持连续时空mask必须保留时间结构这一迁移，但mask比例仍使用P1冻结
的25%，不迁移其预训练目标或高mask ratio。r6通过后不再加probe，直接72-unit formal。

r6窄接口检查首次把scene名误当processed index并访问`trainval/000`，在文件打开前失败；立即读取冻结cohort得到
`scene-0071 -> processed_index 68`后，同一检查通过，per-sweep shapes=`[3,300,300,40]`。登记`V63-F07 resolved`，
未创建run、未改代码/科学合同，也不增加测试矩阵。

P3 r4=`run://worldsim_v63/WS-V63-P3-SURFACE-CORPUS-01/20260824T152300Z__surface-probe-s20260824-r4`
通过几何/资源门：`191 surfaces / 498 patches / 152,226 points`，minimum normal-valid=`1.0`、patch max=`635`、
8/8 negative contracts、runner `passed=true`，wall=`194.306s`、output=`2,429,675 bytes`。但formal放行前与P1 frozen
point schema逐字段对照发现：payload尚缺signed FREE/OCC distance、patch-local xyz、behind-hit与第四个temporal count，
且`ray_hit_order`误存metric distance。r4只证明geometry capability，不能升级为完整P3 pass。

按SciPy官方exact EDT补method-visible FREE/OCC signed distance；按Point Transformer的relative-position原则补patch-local
coordinate；显式保存behind-hit、temporal UNKNOWN、ray distance和bundle内normalized hit order，并补actor observed-hit。
这些是预冻结输入的实现补全，无新超参、无quality选择，不改变任何proposal/topology/label/gate；登记`V63-F06 resolved`，
r5为最后一个schema-complete probe，通过后直接72-unit formal。

P3 r3=`run://worldsim_v63/WS-V63-P3-SURFACE-CORPUS-01/20260824T151618Z__surface-probe-s20260824-r3`
首次完整构建出`191 surfaces / 498 patches / 152,226 points`，wall=`194.540s`、output=`2,429,273 bytes`；
native/evidence/registry/patch bounds/8项负向合同均完成，但`101`个微小static components（85个singleton，其余3–11
voxels）存在离散对称法向量抵消，minimum normal-valid=`0`，所以probe诚实未过。Gradient-SDF一手论文说明SDF梯度
在medial axis最近面不唯一处存在奇异性；Open3D官方接口也要求显式viewpoint orientation。r4仅对“外露面和+centroid
方向都为零”的退化点使用target-sensor viewpoint确定方向，不删proposal、不改topology/patch/cohort/gate，登记
`V63-F05 resolved`。

P3 r2=`run://worldsim_v63/WS-V63-P3-SURFACE-CORPUS-01/20260824T151429Z__surface-probe-s20260824-r2`
因外层launcher提前创建了本应由runner原子创建的新run directory而在入口触发`FileExistsError`；0 unit、0 surface、
0 quality read。Python官方`pathlib.Path.mkdir(exist_ok=False)`合同确认目标已存在必须失败。恢复仅移除launcher的叶目录
预创建，父目录已存在；登记`V63-F04 resolved`，冻结配置和source commit均不变，revision 3已就绪。

P3 r1=`run://worldsim_v63/WS-V63-P3-SURFACE-CORPUS-01/20260824T150842Z__surface-probe-s20260824-r1`
在surface extraction和任何quality判据之前失败：native-to-target helper错误地对长度分别为`300/300/40`的三条轴执行
`numpy.stack`，触发same-shape合同错误。按NumPy官方接口将未被消费的轴元组原样返回，并在重跑前收紧route-support
局部surface type更新、法向量有效统计与显式native-valid报告；均不改变冻结proposal/topology/science合同。r1保持不可变，
登记`V63-F03 resolved`，同配置revision 2已就绪。

P3实现/输入合同已冻结：static proposal=`native occupied + observed OCC - actor envelopes`，Actor proposal按method-visible
current/swept actor ID分开；先声明volume再取6-connected boundary，拓扑不改几何。每点保存native mapping、normal、
method/target/contradiction、逐method sweep时序support、ray bundle/order、actor identity/lifecycle与authority bits；patch按
lexicographic BFS冻结为64/512/2048。仅运行一个`scene-0071/f017` probe，通过即72-unit formal。

P2D native-to-pointwise interface与唯一 formal 已预注册：冻结 V6.2 P5 best，不训练、不调阈值，把 P2 完整 native
logits/BEV 按真实网格坐标映射到 legacy 0.2m grid，并保持原 legacy28/P6 gate、method-before-O_eval 顺序。该诊断只有
一次formal，不做capacity probe、seed或threshold sweep；结果无论正负均只裁决 prototype vs pointwise 根因。

P2D canonical=`run://worldsim_v63/WS-V63-P2D-NATIVE-POINTWISE-DIAGNOSTIC-01/20260824T145924Z__native-pointwise-s0-r1`
已正式rejected：Native B2=`4/28 ACCEPT,4/4 false-safe`，接受集合仍是四个scene-0242 missing-route-support cases；
R10=`2/3`、Actor/static gain=`0/2`、mask-area=`0.094024`、FREE conflict mean/worst=`0.045783/0.092105`、
UNKNOWN=`0.639211`、safe-OCC retention=`1.0`、hard violations=`0/939206`。恢复native feature没有改变决策集合，
因此prototype不是主因，pointwise/mean-query结构根因成立，登记`V63-F02 active`。

负结果后补查CVPR 2024 Point Transformer V3官方实现与visibility-aware surface reconstruction：可迁移点是高效确定性
point neighborhood/serialized patch以及将FREE visibility显式置于surface边界，而不是扩大网络或重新调阈值。P1冻结
的6-neighbor surface topology + patch CVaR方案保持不变；P2D不做recovery，直接进入P3 corpus。

P2 原生接口已实现并冻结：复用已验证的 official IR-WM current forward，每 target 直接保存完整
`200x200x16x17` logits、`200x200x256` BEV latent、argmax/entropy/margin/source-valid 为 memory-mappable arrays；
不再依赖 V6.2 query-deduplicated sidecar，也不存在 prototype。首个 formal denominator 固定为 Tier D 72 targets +
Tier L 4 targets；C/H/T按阶段解锁后生成。只允许一个 `scene-0071/f017` capability probe，通过即运行76-target formal。

唯一 P2 probe=`run://worldsim_v63/WS-V63-P2-NATIVE-SIDECAR-01/20260824T144921Z__native-probe-s1-r1` 已通过：
1 scene/1 target，完整原生数组=`46,081,727 bytes`，峰值 GPU=`4.0496 GiB`、wall=`25.19s`；fresh memory-map reload、
shape/finite均成立，prototype/target/calibration/confirmation/test read均为false。下一步直接formal，不追加probe。

P2 formal canonical=`run://worldsim_v63/WS-V63-P2-NATIVE-SIDECAR-01/20260824T145110Z__native-dl-s1-r1` 已通过：
8 scenes/76 targets（D=72,L=4），完整 sidecar=`3,502,211,483 bytes`，wall=`200.763s`；maximum worker peak=
`4.1314 GiB`、two-worker peak-sum upper bound=`8.2623 GiB`。76/76 native tensors完整、finite并可fresh mmap reload；
prototype/target/calibration/confirmation/test read均为false。P2 hypothesis成立，failure ledger delta=`none`；P2D已解锁。

P1 已完成一手文献/官方仓库审计。RELIOcc/OCCUQ/alpha-OCC/EvOcc 已覆盖 Occupancy reliability、uncertainty、evidence
与 conformal set 的单项；QueryOcc 覆盖连续4D query；Point/Set Transformer覆盖结构编码；CRC/NCRC/structured
segmentation覆盖独立校准；CVaR与visibility-aware reconstruction覆盖尾部损失与FREE约束。未发现把原生 Occupancy
feature、proposal surface、exact hard evidence、surface CVaR、positive OCC authority 与 case-level admission risk
统一成驾驶 world compiler 的直接重合。novelty gate 只对该组合通过，任何单组件均不主张贡献。

在任何 V6.3 quality read 前已冻结：完整 IR-WM `17D logits + 256D BEV` 原生 sidecar；6-connected proposal boundary；
patch `64/512/2048`；256D two-block/two-transformer surface encoder；CVaR alpha=`0.90`；全部 loss/训练超参；Tier
D/C/H/T/L scene-disjoint cohort；case score、`0..1 step .025` fixed-sequence exact-binomial calibration；risk target=
`0.05@95%`；anti-trivial、P6–P10 gates 与单卡资源合同。C/H/T分别为6/3/4 fresh scenes、72/36/48 target cases；H/T
保持 sealed。详见 `docs/autoresearch/worldsim_v63/P1_NOVELTY_PROTOCOL_FREEZE.md`。

V6.2 的 `v62_cpsc_lite_family_closed_negative`、`V62-F06 recovery exhausted` 与 P7/P8 未解锁结论保持不变。
按 V6.3 计划，V6.2 已经由临时 integration branch 以 fast-forward 合入 `main`，定向 projection test 在正确的
`PYTHONPATH=.` 合同下为 `1 passed`，随后从同步后的 `main` 新建并推送独立分支
`research/worldsim-v6.3-surface-tail`。首次定向测试因入口路径写错、第二次因未设置 repo-local import path 而在 collection
阶段失败，均未读取数据或产生科学结果；已登记 `V63-F01 resolved`。

V6.3 北极星冻结为：使用原生 17D Occupancy logits、256D BEV latent 与真实硬证据，对完整 proposal surface 做联合
编码与 patch/proposal 尾部风险控制，经 scene-disjoint case-level 独立校准后才允许 singleton OCC 写入 Physical State。
禁止 prototype bridge、legacy O_eval 调参、voxel-level 伪独立校准、mean query risk 替代 surface tail、用 all-UNKNOWN
冒充安全，以及新建哈希/校验和/指纹机制。默认资源为单卡 RTX 3090 24GB；只有冻结最小配置在一次合法资源恢复后仍
失败，才进入 `blocked_resource` 并向用户申请升级资源。

P0/P1/P2/P2D/P3/P4 当前完成；P5 H-P5-001 training ready。
calibration/confirmation/test保持sealed。

## WorldSim V6.2 CPSC-Lite family closed negative（2026-08-24）

状态：`v62_cpsc_lite_family_closed_negative`；active task=`none`；P7/P8=`not unlocked`。

V6.2 已从 V6.1 最小实验负结论的 `main@c8e9dee` 新建分支 `research/worldsim-v6.2-cpsc`。V6.1 终态
`v61_minimum_experiment_closed_negative` 保持不可变；V6.2 不再遍历第三个 Occupancy backend，而是研究 CPSC：把真实
FREE/OCC 作为前向硬约束、learned Occupancy 作为可推翻软先验，并对证据不足或矛盾区域输出 UNKNOWN。

P0 冻结了 legacy28 机制门槛、fresh development/calibration/confirmation/test 的数据纪律、IR-WM frozen 边界和
单卡 3090 资源上限。按用户约束，V6.2 新产物不加入哈希、校验和或指纹，也不复制 V6.1 的重审计/重门控体系；身份以
逻辑路径、语义版本、task/run ID 和 Git 提交记录为准，只保留与科学结论直接相关的精简验证。

P1 只读一手论文/官方仓库后未发现同时覆盖“硬观测 FREE/OCC + 可推翻 learned prior + selective UNKNOWN + proposal
bake/collision asset + world-simulation false-safe”的直接重合，novelty gate 通过。但单组件均有强先例：ReliOcc/OCCUQ
覆盖可靠性与 uncertainty，EvOcc 覆盖冲突/未知证据，alpha-OCC 覆盖分层保形集合，QueryOcc/DIO 覆盖 4D query 与
留出补全，HardNet/可微投影覆盖硬约束，MultiSafe 已把 conformal 用于 false-safe 控制。因此 CPSC 的可主张贡献被收窄为
`hard-evidence-constrained physical-state compilation` 的完整任务/接口/评测组合，不能把 uncertainty、三态、query、
projection、conformal 或 evidence dropout 单独写成新贡献。

P3 已实现独立于 V6.1 重审计 runner 的小型 PyTorch closed-form projection，约束优先级固定为
`contradiction > observed FREE/OCC > lifecycle > soft prior`。单个 synthetic contract test=`1 passed`；真实
scene-0048/f052 `O_method` fixture 抽样 48 query，hard FREE/OCC、contradiction/lifecycle→UNKNOWN 与 simplex 最大误差
均为 `0`，梯度 finite，未约束 query 梯度非零；第二个 fresh process 结果一致。canonical=
`run://worldsim_v62/WS-V62-P3-FEASIBILITY-PROJECTION-01/20260824T080731Z__projection-s0-r1`。

P2 已冻结 6 个 scene-disjoint development scenes：`scene-0071/0317/0450/0862/1012/1089`。该集合完整复用 V4 在
V6.2 结果出现前、仅按 metadata 冻结的 validation 六场景，不从已有质量结果里选子集；6/6 均属于 nuScenes 官方
train，覆盖 Boston/Singapore、day/dusk/night、dry/rain。每场固定 12 个 target=`17..182`、步长15，共72 units；
method candidate offsets=`[-6,-4,-2,0]`，每个 target 轮换留出一个 dropout sweep，其余三个作为 method input，独立
target offsets=`[-5,-3,-1,1]`。所有 processed scene 均有 6-camera、LiDAR、pose，最短 scene 191 帧，覆盖最大 offset。

P2 materializer 已在 scene-0071/f017 做单 unit、无质量读取的资源/类别探针。r2 canonical probe=
`run://worldsim_v62/WS-V62-P2-EVIDENCE-QUERY-DATASET-01/20260824T082318Z__query-probe-s20260824-r2`：100k
queries，六类 candidate pool 全部非空，source role overlap=`0`，disk=`2,036,102 bytes`，wall=`2.96s`。method 包含
168,487 FREE、11,936 OCC、4,923 contradictions、6,854 motion-compensated actor hits；target supervised query=
38,088/100,000。

r1 probe 在科学执行前暴露 V6.1 evidence state=`U/F/O 0/1/2` 与 P3 model class index=`F/O/U 0/1/2` 的潜在歧义，
没有被用于训练或结果；r2 已显式同时保存 `*_evidence_state` 与 `*_class_index`，登记 `V62-F02 resolved`。其后启动的
formal 仍不读取 occupancy quality、proposal outcome、O_eval、confirmation/test。

首次72-unit formal r1=`20260824T082601Z__query-dataset-s20260824-r1` 在 `scene-1012/f152` 暴露
instantaneous actor envelope 空池并终止，未形成最终 manifest、未进入训练或质量裁决。元数据定位显示该帧仍有4个 actor，
只是全部位于冻结 ROI 外；其中一个 actor 在可见 method sweep f146 穿过 ROI。按 QueryOcc 相邻时刻查询与动态稀疏
query 时序传播的思路，actor support 已改为 current target envelope 与 visible method-sweep envelopes 的并集；它只
影响 actor query 坐标，不把 box 变成 hard OCC，也不读取 dropout/target evidence或改任何配额。

定点 r5=`20260824T083403Z__actor-sweep-repro-s20260824-r5` exit=`0`：current actor envelope=`0`、visible
swept envelope=`450` voxels、actor-type queries=`15000/15000`、total queries=`100000`。`V62-F03 resolved`；据此
从恢复提交直接重跑 formal r2，没有追加更多 smoke。

formal r2 canonical=
`run://worldsim_v62/WS-V62-P2-EVIDENCE-QUERY-DATASET-01/20260824T083654Z__query-dataset-s20260824-r2` 已通过：
`6 scenes / 72 units / 7,200,000 queries`，每场12 units，method/dropout/target source roles=`216/72/288` 且
交集=`0`；六类 query 总数依冻结比例为 `1.8M/1.08M/1.8M/1.08M/1.08M/0.36M`。六类最小候选池=
`156406/6860/6533/382175/167/2446`，72/72 combined actor pools 非空；唯一 current-envelope 空 unit 已由
visible sweep support覆盖。target supervised rows=`2,639,153`，磁盘=`155,249,746 bytes`，wall=`151.47s`，
confirmation/test read=`false/false`。failure ledger delta=`none`（正式成功不灌入 failure ledger）。

P2 已收口，P4 预注册为复用 V6.1 frozen IR-WM environment/weights，在新 development scenes 上 batch1、scene worker
串行或最多2个，抽取与同一 query coordinates 对齐的 prior logits/selected features；不训练 IR-WM、不读取 target
evidence。用户约束覆盖原计划里的内容寻址/model hash 项：P4 只记录逻辑路径、语义版本、backend identity、task/run ID
与 Git 提交，不新增哈希、校验和或指纹。

P4 最薄接口已实现、尚未执行 GPU probe：官方 current occupancy head 提供 `200×200×16×17` logits，current
`ref_bev` 提供 `200×200×256` latent。sidecar 不按100k query重复拷贝 latent，而保存唯一3D prior cells、唯一2D BEV
cells和两组 query→cell 索引；source extent 外 query 显式标为 prior-invalid，留给 CPSC 输出 UNKNOWN/依赖硬证据。probe
固定 `scene-0071/f017`、history=`[7,12,17]`，成功后直接2-worker全量72 units。failure ledger delta=`none`。

P4 probe r1=`20260824T085711Z__prior-sidecar-probe-s1-r1` 在 plugin import、GPU forward和sidecar前被
`Ninja is required` 阻塞。env 内的 `bin/ninja` 已存在，根因是controller使用隔离 Python但未把同一 env bin prepend
到 PATH。按 PyTorch cpp-extension 官方查找机制和 V6.1 成功 worker合同，恢复只补齐 PATH/PYTHONNOUSERSITE/
OMP/MKL/CUDA arch 环境，不安装依赖、不改科学输入；`V62-F04 resolved`，随后重跑同输入 probe r2。

同输入 probe r2 canonical=
`run://worldsim_v62/WS-V62-P4-IRWM-PRIOR-SIDECAR-01/20260824T085956Z__prior-sidecar-probe-s1-r2` 已通过：
100,000 queries 中97,434个 source-valid，去重为27,467个3D prior cells与5,633个2D BEV cells；输出=
`4,002,647 bytes`，worker peak=`4.0496GiB`，official forward=`1.066s`，controller wall=`98.29s`（含首次native
extension启动）。missing keys仅V6.1已知的两项官方删除 `reference_points`，unexpected=`0`；target evidence、
confirmation、exact-once test均未读。P4 进入 formal 6-scene/72-target/max2-worker，不再追加 smoke。

P4 formal canonical=
`run://worldsim_v62/WS-V62-P4-IRWM-PRIOR-SIDECAR-01/20260824T090444Z__prior-sidecars-s1-r1` 已完成：
6 scenes、72/72 targets、7.2M query mappings；source-valid=`6,811,702`（94.607%）、invalid=`388,298`，每unit
valid最小=`91,305`。unique prior cells/unit=`23,129..38,500`，unique BEV cells/unit=`4,973..10,364`；sidecars=
`368,162,079 bytes`。72次official inference合计=`119.41s`，formal wall=`176.27s`，single-worker peak=
`4.1265GiB`、two-worker peak sum upper bound=`8.2523GiB`。6/6 workers unexpected keys=`0`，仅保留相同的两项
官方删除 key记录；target evidence/confirmation/test read=`false/false/false`。failure ledger delta=`none`。

P5 已预注册为只训练 prior adapter、query decoder、evidential head与projection-compatible residual；IR-WM 进程已退出且
权重保持 frozen。输入仅为P2 query/evidence与P4 sidecars，development内部划分和目标函数在启动训练前冻结；不读取
legacy28 O_eval、confirmation或exact-once test，也不新增哈希/校验和/指纹。先审计最薄 loader/model/loss与单卡batch
预算，再直接进入 bounded training，不铺设多轮 smoke/regression 矩阵。

P5 design 已冻结：train=`scene-0071/0317/0862/1012`（48 units），scene-disjoint selection=
`scene-0450/1089`（24 units）；后者只按预先冻结的 Boston rain 与 Holland Village night metadata选择。模型输入为17维
prior logits、entropy/tri-state/source-valid、256维BEV latent、method evidence、normalized coordinates与actor support；
query type、dropout evidence和target evidence明确不进模型。loss固定为query/evidential/hidden-FREE/safe-OCC/
actor-temporal/prior-preserve，hard-conflict target不反向要求模型违反method硬证据。训练配置=`FP16, batch16384,
accum2, AdamW 3e-4, max12 epochs, min4/patience3`，仅运行seed0。先做一次8 optimizer-step capacity probe，
通过后直接全量训练；failure ledger delta=`none`。

P5 capacity canonical=
`run://worldsim_v62/WS-V62-P5-CPSC-LITE-TRAIN-01/20260824T092410Z__cpsc-lite-capacity-s0-r1` 已通过：
3个train units、1个selection unit、8 optimizer steps，608,366 parameters，prior/query dims=`278/13`，FP16 peak=
`0.3724GiB`、wall=`4.91s`，finite best objective=`2.13624`，hard violation=`0`。8步 learned 与projection-only
只作非退化诊断：target accuracy=`0.4233 vs 0.3713`、safe-OCC retention=`0.9569 vs 0.9502`、UNKNOWN fraction=
`0.1773 vs 0.0767`，但hidden-FREE false-OCC=`0.2680 vs 0.2616` 尚未改善；因此不宣称质量pass/fail，只说明
loader/forward/backward/projection/resource合同成立。下一步直接formal 48/24-unit bounded training，不调loss/threshold。

P5 formal canonical=
`run://worldsim_v62/WS-V62-P5-CPSC-LITE-TRAIN-01/20260824T092636Z__cpsc-lite-train-s0-r1` 已通过：
48 train units、24 scene-disjoint selection units，608,366 parameters；9 epochs/1,512 optimizer steps 后按冻结 patience
提前停止，best epoch=`5`，best selection objective=`2.099165`。FP16 peak=`0.3724GiB`、wall=`341.66s`；BEST/FINAL
模型各约2.45MB，hard projection violations=`0/1,286,134`。

最佳 learned 相比同一 selection 的 projection-only：hidden-FREE false-OCC=`0.38457 vs 0.45371`，绝对下降
`0.06914`、相对下降`15.24%`；safe-OCC retention=`0.90106 vs 0.90068`，没有用UNKNOWN换取安全OCC丢失；target
accuracy=`0.48376 vs 0.35677`。learned UNKNOWN=`0.24758`、unconstrained UNKNOWN=`0.46960`，并非all-UNKNOWN。
target evidence仍只作监督，query type/dropout/target均未进入model features；IR-WM不驻留，legacy O_eval、confirmation、
exact-once test均未读。P5 hypothesis在冻结配置上成立，failure ledger delta=`none`；不追加seed/smoke矩阵。

P6 现按计划只做一次 frozen legacy28 matched mechanism benchmark：读取 V6.1 frozen IR-WM sidecar、ME0 O_method、ME1
O_eval和R10 comparator，不重跑IR-WM。主门槛固定为`ACCEPT>=5/28, false-safe=0, R10=3/3 retained, >=1 Actor
新增, >=1 static/disocclusion新增, accepted mask-area>=12%, accepted FREE conflict mean/worst<=0.05`；同时报告
UNKNOWN/ABSTAIN和oracle accepted surface safe-OCC retention，防止all-UNKNOWN。legacy28只裁决机制，不宣称fresh
scene generalization；失败时只允许先查一手来源后从projection architecture、evidence dropout、set-valued head三者中选
一个机制级恢复，不做threshold/grid/window/backend/model-size sweep或删case。

P6 接口审计发现计划文字与 canonical artifact 不一致：V6.1 ME3R 只保存`200×200×16 argmax class`，没有P5所需的
17 logits/256D BEV；同时B2需要尚未允许的Tier-C threshold calibration，B4没有no-evidence-dropout checkpoint，full
M0的grouped conformal按阶段计划要到P8才产生。`V62-F05 resolved_for_artifact_bounded_P6`：参考ProtoSeg的训练特征
类原型，只用P5四场景train split按17个argmax class求query-weighted logits/BEV均值，legacy查表；明确承认它不能恢复
逐cell uncertainty。不得重跑IR-WM、用O_eval拟合bridge或伪造B2/B4/M0。

只读失真审计覆盖24个selection units/2.4M queries，bridge fit未读selection target：full/bridge预测一致=
`0.896898`；bridge hidden-FREE false-OCC=`0.399349`，仍优于projection-only=`0.453707`；safe-OCC retention=
`0.872897`、target accuracy=`0.452581`、UNKNOWN=`0.221945`、hard violations=`0`。P6 formal固定执行B0 replay、B1 hard
clip、B3 evidential-no-projection与B5 pre-conformal；B5为primary，M0明确defer到P8。anti-trivial固定safe-OCC retention
`>=0.50`和source-valid UNKNOWN`<=0.50`。接口实现完成后只做一次formal，不增加bridge/model/threshold sweep。

P6 canonical=`run://worldsim_v62/WS-V62-P6-LEGACY28-ME-01/20260824T095529Z__legacy28-s0-r1`，source=
`d14827d`，正式 rejected。B0=`10/28,10 false-safe`；B1=`10/28,10 false-safe`，虽把accepted mean/worst FREE
conflict降到`0.05058/0.11722`，仍未触发projection-only Stop 1。B3与B5均=`4/28,4 false-safe`、mask-area=
`0.09402`；B5只保留R10 `2/3`、Actor新增=`0`、static新增=`2`。hard projection=`0/939,206 violations`，oracle
surface safe-OCC retention=`1.0`，但source-valid UNKNOWN=`0.82735`，说明主要失败是缺失logits/BEV的feature-shift与
hidden surface authority，不是硬约束或已知OCC丢失。wall=`47.20s`、peak=`0.5319GiB`，IR-WM未重跑。

`V62-F06 active`。按P6 stop rule与一手missing-modality文献，只授权一次evidence-dropout recovery：student从P5 best
继续训练，train query以`p=0.5`替换为train-only class prototype，frozen full-feature teacher提供`0.25×KL`一致性；其余
P5 task loss不变。固定`AdamW1e-4, FP16, batch16384, accum2, max6/min3/patience2, seed0`，pure prototype selection
一次选点；不读legacy O_eval、不加capacity smoke。checkpoint冻结后只运行一次相同P6 gate的P6R；失败则关闭
CPSC-Lite，不再选择第二种机制恢复。

P6R 已按预注册实现：同一608,366-parameter CPSC-Lite student/teacher均从P5 best初始化，teacher冻结；每个
train query独立以`p=0.5`切换到train-only class prototype logits/BEV，student损失为原P5 task loss加`0.25×teacher→
student base-probability KL`。selection固定pure-prototype view，并同时报告full view；P5 evidential anneal从best epoch继续，
不重置。配置=`configs/worldsim_v62/p6r_evidence_dropout_v1.yaml`，入口=
`scripts/run_worldsim_v62_p6r_evidence_dropout.py`。

首次formal入口=`20260824T101047Z__feature-dropout-train-s0-r1` 在baseline selection、任何optimizer step之前因batch缺少
`prior_tristate`触发`KeyError`；未形成checkpoint、未读取legacy O_eval，也没有科学质量结果。损失函数的完整batch读取
已一次性核对；按PyTorch官方mapping batch合同，恢复让`prior_tristate`与输入证据视图同步：pure-prototype selection
使用prototype三态，训练中的full/prototype逐query混合使用同一个`corrupt_prior[:,18:21]`。`V62-F07 resolved`；r1
保持不可变。同次静态接口核对还把尚未执行的legacy `_query_features`返回语句归位，避免唯一复评路径返回`None`；没有
新增失败run。恢复提交=`fb0744b`。

P6R formal r2 canonical=
`run://worldsim_v62/WS-V62-P6R-EVIDENCE-DROPOUT-RECOVERY-01/20260824T101705Z__feature-dropout-train-s0-r2`
已完成：5 epochs、840 optimizer steps，按冻结min3/patience2选择best epoch=`2`；wall=`383.489s`、FP16 peak=
`0.377805GiB`、output=`2,475,348 bytes`、hard violations=`0/1,286,134`。pure-prototype composite objective从
baseline `2.448369`降到`2.274951`，accuracy=`0.452581→0.462246`、safe-OCC retention=`0.872897→0.887356`，
但hidden-FREE false-OCC=`0.399349→0.414406`；full-view也为`0.384568→0.401991`。不据单项风险事后改选epoch 0/3/4，
best epoch 2按预注册复合目标冻结。训练未读legacy O_eval、confirmation/test，IR-WM未运行；下一步只执行一次完全相同
legacy28 arms/gates的P6R recovery。

P6R legacy canonical=
`run://worldsim_v62/WS-V62-P6R-EVIDENCE-DROPOUT-RECOVERY-01/20260824T102709Z__feature-dropout-legacy28-s0-r1`，
source=`d0e5950`，terminal=`rejected`。B0/B1仍=`10/28,10 false-safe`；recovered B3/B5仍=`4/28,4 false-safe`，
B5 mask-area=`0.094024`、accepted FREE conflict mean/worst=`0.049166/0.087379`、R10=`2/3`、new Actor/static=
`0/2`。B5接受集合与P6相同，均为四个scene-0242 missing-route-support cases。UNKNOWN从`0.827351`降至
`0.638518`，relative下降`22.82%`，但仍超过0.50且没有移除任何false-safe；safe-OCC retention=`1.0`，hard projection=
`0/939,206 violations`。resource=`48.109s / 0.531876GiB / 2,293,068 pre-closeout bytes / 64.104GiB free`。

失败后的一手来源复核显示，RELIOcc/OCCUQ需要原生head/features的重训或离线校准，α-OCC与conformal risk control需要
独立calibration，selective classification也只把risk/coverage权衡显式化；它们都不能在“不第二recovery、不用O_eval
调参、不重跑backbone”的V6.2边界内合法迁移。`V62-F06 active, recovery exhausted`；CPSC-Lite family按计划关闭，
P7/P8/confirmation/test均不解锁。完整证据见`docs/autoresearch/worldsim_v62/P6R_EVIDENCE_DROPOUT_CLOSEOUT.md`。

范围冻结见 `configs/worldsim_v62/p0_scope_freeze_v1.yaml` 与
`docs/autoresearch/worldsim_v62/SCOPE_FREEZE.md`；P1 failure ledger delta=`none`，继承边界=
`V62-F01,V61-F11,V61-F13`。

## WorldSim V6.1 minimum experiment 已负结论收口（2026-08-22）

状态：`v61_minimum_experiment_closed_negative`；当前无 active hypothesis，ME-4 未执行且不再授权。

最终 canonical：

```text
run://worldsim_v61/WS-V61-ME3R-IRWM-PREDICTED-OCC-01/20260822T145543Z__irwm-predicted-occ-s1-r1
```

source=`6de27f5704914711e38090c7416d7145f2a610be`。两个 IR-WM scene workers 在一张 RTX3090 并行，
每个只载模一次，完成 target52/57 的4个固定 current occupancy。primary 与 oracle O2 均为 `10/28 ACCEPT`，
accepted mask-area yield 也相同（`0.3983001361`，oracle fraction=`1.0`），但10个接受项全部 false-safe；唯一
顶层失败 gate 是 `predicted_zero_false_safe`。route-support hidden FREE conflict=`0.344..0.571`，actor/disocclusion=
`0.106..0.173`，全部超过冻结上限0.05。wall=`124.30s`，两个 worker peak sum upper bound=`8.25GiB`。

这消耗了 GaussianWorld 失败后由一手文献审计授权的唯一 recovery。GaussianWorld 与 IR-WM 两个不同机制都复现
oracle 的接受集合与 yield，却分别得到 `10/10 false-safe`，因此本协议下不能把 learned argmax occupancy 当作安全
authority。该结论不否定模型 perception capability，也不构成现实驾驶安全声明；它拒绝的是本轮无需训练/calibration
就把预测表面提升为 task-verifiable 几何真值的机制。

按预注册 stop rule：不进入 ME-4，不再换 backend、选 confidence threshold、改 grid/history/checkpoint、放宽 verifier，
也不运行会把10例全部变成 abstain 的 observed-FREE 事后 veto。V6.1 实验实现正式停止，后续只从冻结 artifact 合成
arXiv 技术报告。完整收口见 `docs/autoresearch/worldsim_v61/V61_MINIMUM_EXPERIMENT_CLOSEOUT.md`，失败登记为 `V61-F13`。

## WorldSim V6.1 P7R 已通过；唯一 ME-3 IR-WM recovery 已预注册（2026-08-22）

状态：`p7r_irwm_capability_passed / me3r_irwm_only_recovery_pre_registered`

P7R canonical：

```text
run://worldsim_v61/WS-V61-P7R-IRWM-CONTRACT-RECOVERY-01/20260822T144446Z__irwm-contract-recovery-s1-r1
```

source=`c42bf50809a8a6813d49c841be76f524edbb8bb7`。analysis-only recovery 对 H001 的完整 immutable
artifact、官方删除源码、CUDA wheel build string 和 exact missing-key 集合完成8项 gate，全部通过；wall=`0.023s`，
没有 GPU、model reload、训练或 confirmation。H001 rejected terminal 保持不变，`V61-F12` 只在新 task 中恢复，
P7R 结论严格限于 IR-WM current occupancy 的 3090 capability，不包含安全性声明。

唯一科学恢复 task=`WS-V61-ME3R-IRWM-PREDICTED-OCC-01`，hypothesis=`WS-V61-H-ME3-IRWM-001`。
两个 development scene 各启一个 worker 并在同一 RTX3090 并行；每个 worker 只载模一次，固定 target52 的历史窗口
`42/47/52` 和 target57 的 `47/52/57`。输出映射=`class0→FREE / 1..16→OCCUPIED / extent外→UNKNOWN`；
UNKNOWN 封 ray，predicted FREE 不作为 observed truth，native OBB 只在模型已预测 occupied cell 上绑定 identity。

科学 denominator/gate 与 GaussianWorld ME-3 原样一致：28 matched cases，primary 至少 `8/28`、false-safe=`0`、
严格超过 R10 的3例、accepted mask-area yield 至少为 oracle O2 的80%。method decisions 在读取 O_eval 前冻结。
本次失败即关闭 learned occupancy 和 V6.1 minimum experiment negative；不再换 backend，也不调 confidence、checkpoint、
grid、history window 或 verifier threshold。

## WorldSim V6.1 P7 有效 forward；H002 形式合同恢复已预注册（2026-08-22）

状态：`p7_irwm_forward_valid / h001_contract_rejected / h002_analysis_recovery_pre_registered`

H001 canonical：

```text
run://worldsim_v61/WS-V61-P7-IRWM-3090-SMOKE-01/20260822T143153Z__irwm-current-smoke-s1-r1
```

source=`c5728207ce5ac9b0649afb61c9eedbe418b8d1c9`。官方 IR-WM fully-decoupled checkpoint 已在 RTX3090
完成一次 truth-free current-state forward：raw logits=`1×3×1×40000×16×17`，最终 grid=`200×200×16`，
occupied/free=`40778/599222`，finite，inference=`1.066s`、worker wall=`15.45s`、peak=`4.050GiB`。
两历史帧+当前帧、六相机和 ego motion 完整；没有读取 occupancy GT、O_method、O_eval 或 confirmation，且没有启动
future decoder、planning、training 或 calibration。

H001 的17项 gate 只有 `environment_versions_exact` 与 `model_state_exact` 为 false，因此该 immutable run 继续保留
`rejected` terminal，并登记 `V61-F12`，但不能把有效 forward 误写成 capability 科学拒绝。窄源码审计确认：官方
Detectron2 0.6 CUDA11.1 wheel 的安装版本字符串为 `0.6+cu111`；checkpoint 唯一 missing keys 是
`pts_bbox_head.transformer.reference_points.{weight,bias}`，而冻结官方 `WorldBEVFormerHead.init_weights()` 会主动
`del self.transformer.reference_points`，current-BEV 路径也不调用检测 decoder。

H002=`WS-V61-P7R-IRWM-CONTRACT-RECOVERY-01` 只复用 H001 的 immutable output/report，不重复 GPU 推理。它要求
H001 除上述两项外其余 gate 全通过、所有 artifact hash 精确、Detectron2 完整 build string 精确、missing keys 恰好是
源码证明的两项、unexpected keys 为空。通过才允许预注册唯一一次 ME-3 IR-WM recovery；失败则停止 learned occupancy。
不改 checkpoint、config、input、class mapping、threshold 或 verifier，也不做第二次 capability forward。

## WorldSim V6.1 ME-3 GaussianWorld 已科学拒绝；IR-WM capability 已预注册（2026-08-22）

状态：`me3_gaussianworld_rejected / irwm_capability_pre_registered`

ME-3 canonical：

```text
run://worldsim_v61/WS-V61-ME3-PREDICTED-OCC-01/20260822T134559Z__predicted-occ-s1-r1
```

source=`4c048ecd2db834ae494deb998947136f9918d9bb`。两个官方 batch1 scene workers 在同一 RTX3090 并行完成
24 次 streaming inference，4 个 target occupancy 与 28 个 method decisions 全部落盘；wall=`28.36s`、
per-process peak sum upper bound=`4.47GiB`。预测臂得到 `10/28 ACCEPT`，mask-area yield=`0.3983001361`，
与 oracle O2 的接受集合和 yield 完全一致；但 10 个接受项全部在隐藏 O_eval 上 false-safe，因此唯一失败 gate 是
`predicted_zero_false_safe`。route-support 的 hidden observed-FREE conflict ratio=`0.766..0.958`，actor/disocclusion=
`0.159..0.328`。该结果登记为 `V61-F11`，停止 GaussianWorld argmax Occupancy 作为安全 authority。

源码审计排除了低级适配错误：GaussianWorld 官方 head 使用 `[x,y,z]` 网格、class1..16=occupied、class17=empty；
DriveStudio nuScenes preprocessing 原样保存 camera/lidar world transform，直接 `lidar2img` 与官方 temporal metadata 的
后相机矩阵在机器精度内一致，前相机小差异符合异步 sensor timestamp。因而不授权轴交换、投影修补、confidence/grid/
schedule sweep。把 observed O_method FREE 作为 veto 会令这10例全部 abstain，产出率为0，结果可由已有 artifact 直接推出，
不再为它创建形式化回测。

文献审计显示 ReliOcc、α-OCC 与 OCCUQ 的可靠 uncertainty 都需要训练或 calibration；朴素 max-softmax/entropy 也没有
足够 OoD 可靠性，不能在本轮事后选阈值。OccWorld 依赖过去 Occupancy 输入，会把 oracle 引回 predictor；
Drive-OccWorld 主分支没有发布任务权重。IR-WM 官方分支发布了 vision-centric fully-decoupled checkpoint，并显式从
历史相机建立 current BEV state，因此只预注册一次 truth-free current-state capability smoke。smoke 通过后才允许唯一
一次 ME-3 recovery；失败则终止 learned occupancy，不建安装/调参支线。

gate/arm-summary/summary/resource/manifest/terminal=`508b3551...d74 / 23efb5e5...18c / f6391f49...721 /
7c2c6104...6f4 / 0bb0618f...2fc / 25c01504...4bd`。完整审计见
`docs/autoresearch/worldsim_v61/ME3_GAUSSIANWORLD_FAILURE_AND_BACKEND_AUDIT.md`。

## WorldSim V6.1 P6 已通过；ME-3 GaussianWorld development 已预注册（2026-08-22）

状态：`p6_passed / me3_gaussianworld_predicted_pre_registered`

P6 canonical：

```text
run://worldsim_v61/WS-V61-P6-GAUSSIANWORLD-3090-SMOKE-01/20260822T132526Z__gaussianworld-smoke-s1-r1
```

source=`95c842a883652f679cb1bee93bf1db0e3092c5b2`。官方 streaming checkpoint 完整载入，missing/unexpected
keys=`0/0`，输出=`1×18×200×200×16`、occupied=`29608`、empty=`610392`；inference=`0.8524s`、
worker wall=`3.0384s`、peak=`2.1499GiB`。17 项 gate 全部通过，未读取 SurroundOcc label、O_method/O_eval/
confirmation，未训练或选阈值。gate/summary/resource/manifest/terminal=`dd59fd9e...133 / da079429...b21 /
b6dc3b48...9ac / 24b19cbb...0d9 / 8f886211...ab7`。

ME-3 固定两个 scene-level 官方 batch1 worker 在同一 RTX3090 并行，时序帧=`2,7,...,52,57`，只输出52/57。
类别映射固定为 `0→UNKNOWN / 1..16→OCCUPIED / 17→FREE`；UNKNOWN 封住射线并触发 abstain，predicted FREE
不作为观测 FREE。native OBB 只给模型已预测 OCCUPIED 的 cell 绑定 actor identity，绝不生成几何。method decisions
在读取 O_eval 前固化；主门槛为 `>=8/28`（ME-1 oracle 10例的80%）、false-safe=`0`、mask-area yield 保留
oracle 的 `>=80%` 且严格超过 V6 的3例。不训练、不 calibration、不 threshold sweep；若失败只允许先按具体失败因子
查文献，再预注册一次不降低阈值的保守 recovery。

H-ME3-GW-001 第一次正式入口在 run directory/GPU 前因 tmux 非登录环境缺少 repository root `PYTHONPATH` 而
失败，登记 `V61-F10`，不存在模型或方法结论。H-ME3-GW-002 只让 wrapper 从自身路径自举 repo root；所有科学合同
与预算不变，并在无 run/GPU 的 `--help` smoke 后从新干净提交重跑。

## WorldSim V6.1 ME-2 已完成并拒绝 Hunyuan 路线；ME-3 backend 审计中（2026-08-22）

状态：`me2_rejected / hy3d_route_stopped / me3_backend_audit_in_progress`

ME-2 canonical：

```text
run://worldsim_v61/WS-V61-ME2-HY3D-OCC-ACTOR-01/20260822T121848Z__hy3d-actor-s1234-r1
```

source=`98cec20ae808600309afd2066f7826b2d94ed0b9`。H-ME2-003 完成全部冻结工作：4 个唯一 actor unit、
16 个生成资产、四臂各 6 例、共 24 个 case-arm evaluation；昂贵 Omni diffusion 保持 batch2，只有官方明确
batch1 的 VAE/marching-cubes decode 串行。H002 的 4 个 A0 资产仅在 plan/input/report/asset hash 全部精确后
复用。正式 run 完全离线，无训练和 confirmation read；wall=`675.64s`、peak=`9.45GiB`。

结果为 A0/A1/A2/A3 均 `0/6 ACCEPT`，主臂 A3 false-safe=`0`，但没有任何可接受 case。全部四臂在 method 与
hidden eval 都出现观测 FREE-space conflict；A3 每例 method conflict=`6..246`、eval conflict=`8..273`。与此同时
A3 的 native actor coverage=`0.4949..0.8461`、hole coverage=`0.4738..0.8641`、silhouette IoU=`0.4044..0.8431`，
说明主要问题不是提示词或轮廓质量，而是通用闭合生成表面不能满足场景已观测 FREE 约束。这个结论登记为
`V61-F09`；按预注册 stop rule 停止 Hunyuan actor proposal，不改 prompt、seed、texture、steps、octree、
compiler 或 verifier threshold，也不做事后 clipping。

gate/arm-summary/summary/resource/manifest/terminal=`1eab2226...d86 / dc2222df...505 / 85e20dd9...e73 /
e438e93e...dde / f7fae41a...118 / 9b90d9eb...dc9`。下一任务严格按计划转入
`WS-V61-ME3-PREDICTED-OCC-01`：只审计一个有官方权重、与本机 nuScenes 六相机数据兼容、能在 24GB 单卡执行的
学习式 occupancy backend；优先 GaussianWorld，其次 OccWorld。ME-2 rejection 不被错误扩展为 learned occupancy
路线 rejection。

P6 已选择并预注册 GaussianWorld pretrained：官方 commit=`b43629e...4fc`，stream checkpoint/backbone/temporal
metadata 分别为 `298029831 / 177818375 / 530760430` bytes，SHA-256=`54770811...be3 / 1ee46d1c...ccf /
302fcb86...b54`。官方 metadata 同时包含 scene-0048/0242 各40 keyframes；本机已有两个 development scene 的
六相机 DriveStudio 图像与标定。smoke 固定 scene-0048/frame52、官方 camera order、官方 200×200×16/0.5m 输出，
只验证单卡权重载入、finite/nonempty 输出和 `<22GiB`；不读 SurroundOcc label、O_method/O_eval 或 confirmation，
不做 calibration/threshold selection。通过后直接进入一次 ME-3 development；失败时只审计一次 OccWorld source/
resource，不调 GaussianWorld 输入尺寸、camera order、权重或参数。详见
`docs/autoresearch/worldsim_v61/P6_GAUSSIANWORLD_SOURCE_AUDIT.md`。

## WorldSim V6.1 P4 与 ME-2 预注册/恢复历史（已由 H003 正式结果取代，2026-08-22）

历史状态：`p4_done / me2_h002_batch_decode_failure / h003_formal_retry_ready`

当时 active hypothesis=`WS-V61-H-ME2-003`，task=`WS-V61-ME2-HY3D-OCC-ACTOR-01`。V6 selector 研究族继续冻结，
V6.1 转向 Occupancy-authoritative、Gaussian-rendered、task-verifiable 的四维世界编译器，不再继续阈值、selector、
2D inpainting 或 per-case generator 混选。

P4 canonical：

```text
run://worldsim_v61/WS-V61-P4-HY3D-OMNI-3090-SMOKE-01/20260822T112707Z__voxel-smoke-s1234-r1
```

source=`a97b2743935e3a7143d5b75da9e7bc5bac95e317`。正式 worker 完全离线，用固定官方 voxel demo、seed1234、
50 steps、512 octree、guidance4.5 生成 `1,238,856` vertices / `2,477,728` faces 的 finite mesh 与非空
sampled points；wall=`235.16s`、peak=`7.90GiB`。gate/summary/manifest/terminal=`23451b2d...5cf /
8133a65b...ab7 / 7c4783cb...9a2f2 / 177ce781...8a3`，全部 capability/resource/license gate PASS。
`V61-F04/F05/F06` 保留为不可变失败证据，byte-exact DINO ref 修复后已关闭，不再继续 cache/安装探测。

ME-2 冻结四臂=`A0-image / A1-bbox / A2-point / A3-voxel`。A0 使用同系列官方 Hunyuan3D-2.1 image-only，
A1–A3 使用固定 Omni；4 个唯一 scene/frame/actor 输入按字节复用到 6 个冻结 actor cases，避免为重复 frontend
浪费生成算力，同时保留完整 case denominator。point/voxel 只读 raw LiDAR 与 `O_method`；method decisions 落盘并
冻结后才允许读取 `O_eval`。生成 mesh 只做轴置换与一个 uniform scale，不做 anisotropic warp、clipping 或 case 特判。

单次结构预检没有读取 `O_eval`、没有载入生成模型：4/4 controls finite，raw actor points 非空，target O_method
voxels=`10878..23088`，6 个 case 的最小 actor-hole coverage=`0.6322`。native LWH 已按官方 Omni 合同转换为
LHW；最大 actor `15.454m / 256 = 0.0604m`，低于冻结 `0.2m` occupancy cell，故固定 octree256 而不做分辨率 sweep。

主臂 A3 gate=`>=2/6`、false-safe=`0`、accepted FREE conflict=`0`、unfiltered swept collision=`0`。
scene-0242 只过滤 actor4 truck 与 actor15 trailer 的精确铰接 contact：141 连续帧相交，最大相对平移步长
`0.09814m`、最大相对 yaw 步长 `0.07619°`；不放宽全局碰撞阈值。失败即停止 Hunyuan 路线，不做 prompt、
texture、seed、steps、resolution 或 verifier threshold 调参。

H-ME2-001 已创建 failed run `20260822T120008Z__hy3d-actor-s1234-r1`：所有 source gate 和4个输入构造完成，
但 A0 worker 在载模/GPU推理前导入官方 Hunyuan3D-2.1 package 时缺少其 requirements 固定的
`pymeshlab==2022.2.post3`（`V61-F07`）。没有生成 asset、method decision 或科学结论。H-ME2-002 只在隔离
环境补齐该官方依赖并增加 exact version gate；一次离线 base pipeline import smoke 已通过。全部科学合同不变，
从新干净提交重试。

H-ME2-002 failed run `20260822T120519Z__hy3d-actor-s1234-r1` 已完成4个有效 A0 mesh；Omni 也完成首个
2-sample A1 diffusion/decode，但官方 vanilla extractor 把两份 SDF reshape 后只对 `grid_logits[0]` 做 marching
cubes，因此只返回1个 mesh。runner 对 `1 != 2` fail-closed，没有静默丢弃第二例（`V61-F08`）。H-ME2-003
保持 diffusion batch2，改为返回2份 latent 后逐份调用同一官方 VAE decode；只串行官方明确 batch1 的 mesh
extraction。H002 A0 只在旧 plan/input/report/assets 全部精确后复用，不重复4次 GPU 生成；科学参数和 gate 不变。

P0 精确绑定：

- V6.1 plan SHA-256=`8ac58801...38be`；
- R10 28-case baseline=`3 ACCEPT / 7 ABSTAIN / 18 REJECT`、false-safe=`0`、accepted mask pixels=`107807`；
- scene mapping=`scene-0048 -> processed 045`、`scene-0242 -> processed 191`；
- `O_method` 与 `O_eval` 使用不重叠的 raw LiDAR sweep 路径，confirmation 保持锁定；
- failure refs=`V6-F25/V6-F26/V6-F65/V6-F71/V6-F78/V6-F79`。

H-P0-001 在创建 run 或读取任何科学输入前因新 namespace 不存在而触发 `FileNotFoundError`；GPU/训练/生成器均未启动，
没有方法结论，登记为 `V61-F01`。H-P0-002 只创建精确 run namespace 后正式通过，`V61-F01` 已 resolved。

P0 canonical：

```text
run://worldsim_v61/WS-V61-P0-SCOPE-FREEZE-01/20260822T100812Z__scope-freeze-s20260822-r1
```

source=`6247fd89068615f791b428c3296faf945e713c75`；gate/summary/manifest=`fb2a416a...ae40 / e53a86f2...907c /
2ed96578...7593`。全部 gate PASS；R10=`3/28`、false-safe=`0`、case identity 与 scene mapping exact，
method/eval source paths disjoint。

ME-0 canonical：

```text
run://worldsim_v61/WS-V61-ME0-OCCIR-01/20260822T101817Z__occir-s20260822-r1
```

source=`5a3bc42eb68cfcda673df3c32d81479373b1bff3`；4 scene/frame units、8 truth tiers、28 case bindings 全部
通过。`O_method/O_eval` 的 raw LiDAR path 与 payload hash 全局互斥；每格 UNKNOWN/FREE/OCCUPIED 非零；
oriented actor volume、identity/lifecycle、source-removal→UNKNOWN、fresh-process content exact 与
`<=2.14e-14m` round-trip 均通过。gate/summary/manifest=`1e818074...8bb7 / 6e50644b...b14f /
386d99ab...59ec`；wall=`10.57s`，4 CPU workers，无训练/生成器/confirmation read。

ME-1 预注册固定五臂：冻结 Big-LaMa 的 `B0-2D`、冻结 R10 的 `B1-R10`、不增 coverage 的 `O1-GATE`、
主臂 `O2-OCC-GEOMETRY` 与带 native trajectory/lifecycle/swept OBB collision 的 `O3-OCC-4D`。编译只读
`O_method`，先固化 method decisions，再让 `O_eval` 只计算 hidden truth/false-safe。阈值来自既有合同：
0.2m voxel、0.1m ray step、R9 的 50% coverage 与 20% depth consistency；没有 case 特判或 threshold sweep。
一次结构审计显示 10 个 P1-ACCEPT case 的 method mask coverage=`73.65%..94.78%`，故直接进入正式 run。
若 O2 不能达到 `>=5/28`、false-safe=`0`、保留原3例并新增 actor+static/disocclusion，则停止模型接入。

H-ME1-001 在创建 run directory 或启动 GPU 前读取 ME-0 gate 时误把 authority 从 `checks.passed` 当成顶层
`passed`，触发 `KeyError`；无 run、无方法结果，登记为 `V61-F02`。H-ME1-002 只修正该 schema 路径并增加回归测试，
所有科学输入、arms、thresholds、预算与 stop rule 不变。

ME-1 canonical：

```text
run://worldsim_v61/WS-V61-ME1-ORACLE-OCC-PROPOSAL-01/20260822T104207Z__oracle-occ-s20260822-r1
```

source=`e422f0528c2c98e80d3cfbd8052ccb106734d043`。B0=`0/28`；B1/O1 均为 `3/28`；primary O2=
`10/28`、false-safe=`0`、accepted mask pixels=`450865`、yield=`39.83%`，保留原3例并新增3 actor+4
static/disocclusion。O3=`6/28`、false-safe=`0`；actor 例被真实 native OBB overlap（主要 actor4/15）拒绝，
不通过阈值豁免。后续控制准备另发现 actor ID0 与 empty sentinel 冲突（`V61-F03`）：不影响 O2 主结论，但 O3 的
scene-0048 identity 诊断降格；ME-2/ME-4 使用 `-1` sentinel 修复。wall=`3.60s`、peak=`0.51GiB`。
gate/summary/metrics=`6aca5f2f...246d / 61713df4...afb9 / dbb1d0a3...ffb6`，ME-2 已解锁。

P4 绑定 Hunyuan3D-Omni 官方 git commit=`4d47c0cc...bfa8`、HF model revision=`70e803bf...d485` 与
DINOv2-large=`47b73eef...2d6c`。官方一手实现声明约10GB VRAM且支持 bbox/point/voxel；正式 smoke 固定官方 voxel
demo、seed1234、50 steps、512 octree、无EMA/fast decode/sweep，离线运行并要求 mesh/points 有效、peak<22GiB。
模型使用受 Tencent community license 的地域与用途限制；本轮只在中国 AutoDL 主机科研执行，不分发模型/输出，
也不用于训练其他模型。P4 通过后直接跑固定6例 ME-2；失败则停止 Hunyuan 路线，不反复调安装/推理参数。

P4 首次入口在 run/GPU 前发现 VAE digest 被手工多录一个尾字符（`V61-F04`）；实际文件 SHA 与固定 revision
HTTP `X-Linked-ETag` 完全一致。只修正 65→64 字符的 provenance transcription，并新增 digest 结构回归；模型、
权重、demo、seed、steps、octree、gate 与 stop rule 均不变。推理环境已按官方版本收窄为 shape-inference closure，
`pip check`、CUDA、DINO cache 与官方 pipeline import 均通过，训练/UI/texture 后处理依赖不进入 P4。

第二次入口已创建 failed run `20260822T111747Z__voxel-smoke-s1234-r1`：DiT/VAE 精确载入，DINO repo-id
因 exact-commit cache 缺少默认 `refs/main` 而在离线解析处失败（`V61-F05`），尚未生成 mesh/points 或 capability
结论。修复只建立标准 cache ref 并把它精确绑定冻结 DINO commit；runner 在载模前验证 ref、snapshot、config 与
model SHA，正式入口继续完全离线，不修改官方源码、backbone 或任何推理参数。

第三次入口 `20260822T112159Z__voxel-smoke-s1234-r1` 暴露了更精确的根因（`V61-F06`）：运行时 cache
root 正确，但安装版本以原样 `f.read()` 解析 ref；staging 文件尾换行使 ref 为41 bytes，无法匹配40字符 snapshot。
外部 cache ref 已规范化为 byte-exact token；孤立离线 repo-id smoke 成功载入 `Dinov2Model` 的
`304368640` 个参数。只有该最小解析测试通过后才重新授权完整 P4，避免了继续重复载入12GB Omni 权重。

## WorldSim V6 收口：selector 研究族已冻结（2026-08-22）

状态：`selector_research_family_frozen_closeout_complete`

当前没有 active hypothesis。R141 未执行。按照最终研究决策，本研究族不再继续 threshold 13/45、新 actor、新编辑方向，也不引入新的 selector 机制。

### R140 recovery

R140 H001 与 H002 已完成科学计算，但由于 Python 源码使用小写 JSON boolean，在正式 closeout 阶段失败；它们继续作为 V6-F97 与 V6-F98 不可变保留。H003 只把剩余的 `false` 改为 `False`，所有科学输入、公式与 gate 均保持不变，并从干净且已推送的源提交 `a13759ba8db03e1f740ad93e246ca24f0ff2d7fa` 完成。

Canonical run：

```text
run://worldsim_v6/WS-V6-R140-CROSS-FRONTEND-END-TO-END-UTILITY-01/20260822T063937Z__end-to-end-utility-s20260821-r1
```

| 条件 | End-to-end reduction | Reconstruction errors |
| --- | ---: | ---: |
| StreetGS | 0.13533665047667254 | 0 |
| AD-GS development | 0.11143415340582441 | 0 |
| AD-GS exact-once confirmation | 0.016636471392706964 | 0 |
| Macro | 0.08780242509173464 | 0 |
| Worst | 0.016636471392706964 | 0 |

Full 与 selective 路径以相同方式计入 sensor time。这些数值是单次已观测 artifact cost，不是 replicated performance estimate。

Artifacts：

- certificate `913833af47e4171e27707f71418b6625ed358b538d1c8a5a18bca5ac7f585363`
- gate `ac3c79c0e93f2932a076da8323b89a210ff2cbaac27ffa13079ce89ae9d07b51`
- summary `50900ff99736055a10c32f4362176b7fc87862ae84667591077d6c17024e635b`
- manifest `1cc753b3c0a9489ced2a58b23035466ee26cba963d2abc56c84ebd4d057e5a62`
- resource audit `06c110236591529d5fef5f4178bfed696b6c0ad0cfbce94c497896dc92230265`
- terminal `be263ba010cdb936fbd01dbfa0fe294b8022101aae348c95030d2a42d45fdb77`

### Selector 最终证据

| 实验 | 状态 | 保留结论 |
| --- | --- | --- |
| R134 | rejected / V6-F94 | threshold 13 漏检 AD-GS frame 13（RGB 1、label 1）。 |
| R136 | rejected / V6-F95 | 冻结 threshold 1 在 heldout frame 14 出现 1 个 FP；精确分类声明失败。 |
| R137 | accepted development | 157 个 AD-GS 帧，调用减少 16.56%，0 false reuse，628 个 hash 全部精确。 |
| R138 | failed consumed / V6-F96 | 负数 CLI 参数在 sensor 输出前失败；不存在方法结论。 |
| R139 | accepted exact-once | 39 个 AD-GS 帧，调用减少 17.95%，0 false reuse，156 个 hash 全部精确。 |
| R140 | V6-F97/F98 recovery 后 accepted analysis | Macro 端到端 reduction 8.78%，worst 1.66%，0 reconstruction errors。 |

### 治理状态

- Failure ledger 的当前权威边界是 V6-F98；recovery 注记不删除或重分类失败 attempt。
- Selector 研究族在 R140 后冻结。R141 明确为未执行，不是 rejected，也不是 accepted。
- Confirmation 与 test 分区继续锁定。
- Claim boundary 只覆盖 operational equivalence 与已观测 wall-time accounting；不声明 semantic、physics、planning 或 safety correctness。
- 仓库收敛目标为唯一远端分支 `main`，指向本次 closeout。

详见 [selector 研究族收口](autoresearch/worldsim_v6/SELECTOR_RESEARCH_FAMILY_CLOSEOUT.md)、[failure ledger](RESEARCH_FAILURES.md) 与 [V6 plan](WORLDSIM_V6_VERIFIABLE_WORLD_COMPILER_AUTORESEARCH_PLAN.md)。
## WorldSim V6.5 启动：TAC-Compiler 直接研究入口（2026-08-27）

状态：`active_p1_signal_atlas`。

已从冻结的 `research/worldsim-v6.4-native-uq` 终态 `add2f3f` 创建并推送
`research/worldsim-v6.5-task-conditioned-authority`。V6.4 canonical runs 保持只读，不改写、不重跑；任何曾被
V6.4 读取质量的场景在 V6.5 一律为 Tier L，只允许机制诊断、warm-start 和基线复用，不能形成 V6.5 正式
selection/calibration/confirmation/test 结论。

按用户指令采用 direct-research profile：跳过冗长 P0、全量 smoke 和回归，只落盘最小继承/场景/selection
协议后直接执行 P1 连续 trajectory signal atlas 与 T0 低容量条件残差。当前资源为单张 RTX 3090（启动时
1 MiB、0%）和 `/root/autodl-tmp` 约 120 GiB 可用空间；P1–P6 的 faithful minimum 不需要多卡。I/O 采用
V6.4 canonical sidecar 只读复用与 compact feature cache，训练和评分优先放到 GPU。

当前 active hypothesis：`WS-V65-H-P1-001`。尚未读取 V6.5 正式 selection quality，尚无 V6.5 方法结论。
本里程碑 failure ledger delta：无；首次建分支推送未走 LocalTUN upstream，改用当前远端代理后成功，属于已解决的
基础设施路由事件，不登记科学失败。

### P1 train-only stage freeze

P1 已冻结两臂：`R0=frozen V6.4 q0` 与 `R1=q0 + 10D continuous trajectory FiLM residual`。16 个 Tier-L
development scenes 各取前 8 units 训练、后 4 units 评估；每训练 unit 最多 4096 points、每评估 unit 最多
8192 points。连续特征不含硬 1.5m corridor bit；硬 corridor 只用于 fixed-opportunity 指标。唯一 seed=0、容量
`10→32→16` trajectory encoder、冻结 64D q0 hidden、40 epochs，不做 sweep。当前下一动作是生成 compact
q0-hidden cache 并运行 GPU 主实验。

### P1 T0 终态与 P1R 迁移

P1 canonical run：

```text
run://worldsim_v65/WS-V65-P1-CONDITION-SIGNAL-ATLAS-01/20260827T074500Z__signal-atlas-s0-r1
```

523,910 train points、497,892 nested evaluation points；wall=`34.71s`、peak GPU=`0.132GiB`、compact cache=
`173MiB`。R0 q0 AUROC/AUPRC=`0.871759/0.407081`，R1 T0=`0.871576/0.405639`，增量分别为
`-0.000183/-0.001443`。matched 40% 下 pooled fixed-route density `0.00299581→0.00314560`，相对恶化
`5%`；scene lower/equal/higher=`1/13/2`。真实 trajectory 相对 unit 内 shuffle AUROC `+0.009591`，说明
条件被网络使用，但没有转化成对 q0 的有效增量。`WS-V65-H-P1-001` rejected，T1 attention 不解锁。

文献复核后冻结 `WS-V65-H-P1R-001`：WoTE 将 trajectory 用于未来结果/trajectory evaluation；UniAD/VAD
通过 planning query 与未来 occupancy/actor 交互。当前 T0 却让 trajectory 解释 task-agnostic static
hidden-FREE，存在 target semantics 错配。P1R 保留 q0=`r_phys`，新增连续 relevance 缩放、只允许非负增险的
`r_task`；以 task-aligned loss 训练，primary 只看 fixed-route opportunity，并锁定 non-route 不恶化。它不是
seed/容量救援，也不改变 P1 负结论。

P1R canonical：

```text
run://worldsim_v65/WS-V65-P1R-TASK-ALIGNED-RISK-01/20260827T075500Z__task-risk-s0-r1
```

同一 523,910/497,892 train/evaluation denominator 上，fixed-route density `0.00299581→0.00284602`
（20→19 conflicts，relative reduction=`5%`），worst-tail `0.01643968→0.01559935`，scene
lower/equal/higher=`1/15/0`，non-route emission risk relative change=`-0.00665%`；shuffle density 回到 q0 的
`0.00299581`。四个 gate 全过，verdict=`positive_train_only_task_risk_signal`。这是弱但无风险搬家的机制信号，
只解锁 fresh P2 T0；当前 active task=`WS-V65-P2-TRAJECTORY-CONDITIONED-RISK-01` 的 metadata-only cohort
freeze，尚未读取 formal selection。

P2 fresh representation-selection cohort 已在 quality read 前冻结：`scene-0520/0781/0800/0996/0443/0106`
（72 cases）。前三个只读复用现有 processed scene，后三个从公共 tar restricted extraction；它们均未出现在
V6.1–V6.4 method configs。P2 只允许 frozen q0 与 frozen P1R task risk 一次正式读取，严格使用计划中的
`>=10%` fixed-route reduction、`>=5/6` strict scene support 与 `<=5%` scene/non-route regression gate。

### P2 pre-read capability recovery

首版 cohort 的 `scene-0520/0781/0800/0106` 不在冻结 IR-WM temporal-info pickle 的 700 keys 中；三个并发
native workers 在任何 sidecar/model-score/quality 输出前以 `KeyError` 退出，preparation r1 也在 quality read 前
中止。该事件登记为 `V65-F02`，不构成 P2 负结果，也不消耗唯一正式 read。

卡点发生后已核对官方 BEVFormer 数据准备流程。全量重建 temporal infos 需要完整 nuScenes/CAN bus 管线且可能改变
冻结后端 schema，因此采用不读取质量的 capability-only 恢复：最终 fresh cohort 为
`scene-0996/0443/0002/0043/0023/0072`，全部是冻结 pickle 的直接 key，且未出现在 V6.1–V6.4 method configs。
72-case denominator、targets、arms、gate 与 seed 均不变。现有 2.5GiB partial raw 中 411 个非空成员将直接复用；
下一步是 recovery preparation r2，随后运行 2-worker native sidecar、evidence 与唯一一次 P2 formal read。

### P2 inputs complete：I/O/GPU 流水线

Recovery preparation canonical：

```text
run://worldsim_v65/WS-V65-P2-FRESH-PREPARATION-01/20260827T082500Z__fresh-prep-s0-r2
```

6/6 scenes 全部新生成，`partial_raw_reused=true`，新抽取 10,396 个成员，总 wall=`4011.44s`，quality
read=false。scene preprocess 完成即启动对应 GPU inference：CPU 继续处理后续 scene 时，GPU 保持最多两个
IR-WM workers；不再等待整批 I/O 完成后集中上卡。

六个 per-scene native runs 全部 `passed=true`，合计 72 targets、3,317,884,487 bytes；单 worker peak GPU
`4.1314GiB`，双 worker 上界 `8.2628GiB`，所有 target-evidence/calibration/confirmation/test read 均为 false。
Evidence canonical `20260827T091800Z__fresh-evidence-s0-r1` 同时在 CPU/I/O 运行，完成 6 scenes、72 units、
70,124,875 bytes，wall=`121.51s`、`passed=true`、query_count=0。下一步只把 per-scene units 无复制汇总为
canonical native root，然后执行唯一一次 P2 quality read。

### P2 trajectory-conditioned family 终态

Canonical：

```text
run://worldsim_v65/WS-V65-P2-TRAJECTORY-CONDITIONED-RISK-01/20260827T093900Z__trajectory-selection-s0-r1
```

唯一正式 fresh read 已消费。q0 与 task arm 的 fixed-route 结果同为 `18/6975`，density=`0.00258065`，
relative reduction=`0%`，worst-tail=`0.02935237`，scene lower/equal/higher=`0/6/0`。task arm non-route
conflicts `4538→4542`，relative change=`+0.0881%`，仍在 5% bound 内。coverage matched、monotone semantics 与
regression gates 通过，但 primary reduction 和 scene-support gates 失败；verdict=
`rejected_fresh_trajectory_condition`。wall=`10.01s`、peak GPU=`0.0409GiB`。

`V65-F03` 冻结该负结论：不重跑、不 sweep、不解锁 trajectory attention。文献迁移指向 ego action/goal 与
multi-agent future response、continuous spatiotemporal query；下一步只允许建立一个语义独立的
actor-time/action-outcome train-only hypothesis，并要求新 selection cohort，不能把它写成 P2 rescue。

### P2R actor-time/action-outcome 启动

Active hypothesis=`WS-V65-H-P2R-001`。复用 V6.3 legacy evidence 的严格 disjoint method/target sweeps，构建
Actor token：A0 只看 current Actor snapshot，A1 增加 method-visible swept history/time features；监督是独立
target sweeps 中 Actor swept envelope 与 ego future route 的交互。4 scenes train、2 scenes nested legacy eval，
不产生 V6.5 formal selection read。

该迁移对应 PRECOG/M2I/GameFormer 的 multi-agent conditional response、VAD 的 instance-level planning
constraint 与 Implicit Occupancy Flow 的 continuous spatiotemporal query。配置与 gate 已在任何 outcome 统计前冻结；
下一动作是边加载 72 个小 evidence units 边在 GPU 计算 trajectory geometry，物化 compact Actor-token cache 后立即
训练 A0/A1。

### P2R zero-support 终态与 P2C continuous-cost 迁移

P2R canonical `20260827T100000Z__actor-time-s0-r1` 完成 476 train / 302 eval Actor tokens，但 binary
positives 为 `0/476` 与 `0/302`；所有 gates 失败，verdict=`no_clear_train_only_actor_time_signal`。这登记为
`V65-F04` 的 task-support failure，不解释为 A1 模型负结果。

文献复核后 active hypothesis 改为 `WS-V65-H-P2C-001`：数据/split/features/model 不变，只把不可辨识的硬标签
换成预注册连续 target proximity cost `exp(-distance/6m)`；不读取 P2 cohort、不扩大二值半径、不做 sweep。
下一动作是用新 cache path 物化 target distance/cost，并在 GPU 比较 A0/A1 的 Spearman、MSE 与 selected mean cost。

### P2C continuous Actor-time cost 终态与表示族关闭

Canonical：

```text
run://worldsim_v65/WS-V65-P2C-ACTOR-TIME-COST-01/20260827T102000Z__actor-time-cost-s0-r2
```

连续 target 在 train/eval 的 mean 为 `0.147396/0.095976`，不再存在 P2R 的零支持问题。A0 snapshot 的
Spearman/MSE=`0.872281/0.006247`；A1 Actor×time=`0.857392/0.008407`，即 Spearman 增量 `-0.014889`、
MSE relative reduction=`-34.59%`。A1 相对 scene-wise shuffled time 仍有 `+0.098817` Spearman，证明时间通路
确实被使用；但 matched 40% 的两个 eval scenes 均恶化，lower/equal/higher=`0/0/2`。全局 selected mean cost
虽从 `0.023950→0.021468`（`-10.37%`），却没有 scene support，只有两个预注册 gates 通过。verdict=
`no_clear_train_only_continuous_actor_time_cost`；wall=`7.52s`、peak GPU=`0.593GiB`。

`V65-F05` 保留第一次入口的共享配置工程失败（训练/证据读取前 `KeyError`）；`V65-F06` 冻结上述科学负结论。
Actor/time family 已关闭，不改 cost scale、seed、容量或 split，不解锁 P3。结合 fresh P2，trajectory attention 与
task-conditioned end-to-end representation 同样保持锁定。当前只进行 P4 feasibility audit：检查在完全冻结 V6.4
risk representation、且不读取已消费 representation selection 的条件下，是否存在可用于 learned admission 的
独立 train/selection 分区；若不存在则直接关闭 admission，不制造重复实验。

### P4T learned admission train-only freeze

可行性审计确认V6.4两组互斥的96-case Tier-L evidence/native artifacts可直接形成train/nested-eval，不需要新scene
IO，也不触碰V6.5 admission selection。`WS-V65-H-P4T-001` 已冻结：V6.4 risk与M1 route cap=0.40不变，G0只从
连续可观测risk/context统计预测`[0.30,0.55]`内的per-case coverage，inference不读scene/stratum/hidden truth。
训练目标仅由train cases的最大安全前缀产生；单个`32→16` MLP、seed=0、无sweep。只有held-out Tier-L上相对
M1达到预注册增量且不增加failure，才准备fresh D-Selection-Admission；否则按Stop 4关闭admission。

### P4T learned admission 终态

Canonical：

```text
run://worldsim_v65/WS-V65-P4T-LEARNED-ADMISSION-TRAIN-ONLY-01/20260827T110000Z__learned-admission-s0-r1
```

G0 mean coverage=`0.541329`，相对M1 `0.474961`提升`+0.066368`，7/8 scenes有非负utility；但新增1个case
failure，pooled fixed-route density `0.00181015→0.00196987`（`+8.82%`），worst-10% fixed-route CVaR
`0.0158854→0.0170938`（恶化`7.61%`）。新增failure的预测coverage=`0.521873`，超过该case oracle-safe
`0.510822`，hidden-FREE conflict `0.047814→0.050764`。不能因只越界`0.000764`而事后收紧coverage。

四gate只过coverage/scene-support，verdict=`no_clear_train_only_learned_admission`，登记`V65-F07`。按Stop 4关闭
learned admission；不准备fresh admission cohort，不解锁P5 allocator/OT或CRC calibration。当前回到P1尚未执行的
R3 map/context family做只读可行性审计；它必须有不同监督语义与未消费的fresh selection方案，否则不创建run。

### R3 map/context capability recovery 与stage freeze

本地初始meta缺`maps/expansion`；公共盘v1.2虽完整但被当前devkit按版本合同拒绝，登记`V65-F08`，未读quality。
官方v1.3也已在公共盘，解压到独立research data root后，单pose能力调用成功生成`8×200×200`的
drivable/road/lane/crossing/walkway/carpark/road-divider/lane-divider mask，各主要层非空。

Active hypothesis=`WS-V65-H-P1R3-001`。R3复用P1相同16-scene Tier-L split与采样，冻结q0，新增14D连续
map/context：8层逐voxel语义、drivable signed distance、route curvature/length、route-on-drivable和local map
density；不读stratum/scene ID/hard route bit。单seed/单容量/单run，真实map必须同时改善AUROC与fixed-route risk、
有scene direction support、non-route不搬家并优于within-unit shuffle，才允许新的fresh selection cohort。

### R3 map/context 终态与预测对象迁移

Canonical：

```text
run://worldsim_v65/WS-V65-P1R3-MAP-CONTEXT-TRAIN-ONLY-01/20260827T114500Z__map-context-s0-r1
```

R3 AUROC/AUPRC=`0.871264/0.404801`，相对 q0 为 `-0.000496/-0.002280`；pooled fixed-route density
`0.00299581→0.00299581`，scene lower/equal/higher=`1/14/1`。真实地图比 within-unit shuffle 的 AUROC 高
`0.000625`，non-route risk 下降 `0.756%`，说明输入被使用但没有形成有效增量。5 gates 只通过2项，登记
`V65-F09`，关闭 per-voxel map/context residual。wall=`85.42s`、peak GPU=`0.1397GiB`；单卡资源充足。

当前方向按 task-relevant failure detection 改写：不再预测“这个 voxel 是否正确”，而预测“给定 Ego 轨迹
`τ`，未来 `H` 秒访问走廊的 world state 是否可靠”。先在未增加 quality exposure 的原 P1 train/nested-eval
缓存上形成 unit/trajectory-level 连续 outcome；若该对象有明确 held-out signal，才考虑 Actor-state companion 和
fresh action-level selection。该迁移不是 R3 rescue，也不重开已关闭的 voxel representation family。

### R4 trajectory-visited-state freeze

Active hypothesis=`WS-V65-H-P1R4-001`。监督单元从 voxel 改为 `(scene, unit, τ)`：冻结20帧/2秒 Ego future
trajectory，以1.5m corridor定义实际会访问的 sampled world states，target 是 corridor 内 hidden-FREE outcome 的
连续比例；至少16个visited points才纳入。输入只含冻结q0在visited footprint的分布、footprint/global observable
context和R3已冻结的14D map/context均值，不读取truth、scene ID或stratum。

先评估直接 `Qagg=mean(q0 risk | visited by τ)` 的预测对象可行性，再评估单个`25→32→16→1` GPU head的增量。
若Qagg可行而head无增量，保留trajectory-level聚合对象并关闭head；只有两者均失败才关闭world-state target。
缓存读取会在GPU计算q0 sigmoid时并行读取map-context，不再读取173MiB native hidden或原始sidecar。

### R4 trajectory-visited-state 终态：预测对象成立，learned head关闭

Canonical：

```text
run://worldsim_v65/WS-V65-P1R4-TRAJECTORY-VISITED-STATE-01/20260827T121500Z__visited-state-s0-r1
```

108 train / 58 nested-eval units符合至少16个visited samples；eval含6651个visited states、754个hidden-FREE
outcomes、46/58 unsafe units。直接Qagg Spearman=`0.751487`、unsafe AUROC=`0.978261`，lowest-risk 40%
actual cost=`0.038137`，相对全体`0.103005`下降`62.98%`，3/3 viability gates全过。这是本轮首次证明：将
state risk按将执行的`τ`聚合后，trajectory-level future visited-state reliability是强而可解释的预测对象。

V1 head虽然MSE降低`87.35%`且有shuffle response，却把Spearman降到`0.635127`、selected cost恶化`51.35%`，
scene=`2/7/6`；登记`V65-F10`并关闭 learned head。最终 verdict=`positive_train_only_visited_state_object_q0_
aggregation_only`。下一步并行准备 Actor false-safe companion 与独立fresh trajectory-level transfer；fresh准备期间
继续运行现有compact Actor GPU实验，避免GPU等待重IO。

### R5 Actor false-safe companion freeze

Active hypothesis=`WS-V65-H-P1R5-001`。不重训P2C：冻结A0 snapshot与A1 Actor-time两个已读模型，在GPU评分后
按`(scene, unit, τ)`取最大Actor-route proximity cost。先验证A0对realized target cost的trajectory-level forecast
viability；再以`relu(A1-A0)`作为纯可观测temporal disagreement monitor，预测`relu(target-A0)` false-safe gap。

Forecast gates固定为Spearman `>=0.70`、lowest-risk 40% target cost相对全体降低`>=25%`；monitor gates固定为
gap Spearman `>=0.30`、positive-gap AUROC `>=0.65`、lowest-monitor 40% gap降低`>=25%`。无learned monitor、
threshold、seed或capacity sweep；formal V6.5 selection read=false。

### P2V fresh visited-state transfer freeze

`WS-V65-H-P2V-001` 已在任何新target/q0读取前冻结。fresh cohort由满足“冻结700-key temporal metadata直接
可用、仓库未出现、尚未processed”的候选按scene-index排序取6个等距quantile：`0001/0219/0402/0594/0822/
1110`，12 frames/scene。描述覆盖construction、junction、residential、rain、pedestrian与night，但描述只在
quantile选择后记录，不参与换scene。

唯一candidate是R4保留的确定性Qagg，target仍为future 2s/1.5m trajectory footprint的realized hidden-FREE
fraction；不迁移R4 learned head。fresh gates：Spearman `>=0.60`、unsafe AUROC `>=0.85`、lowest-risk 40%
cost reduction `>=40%`、scene support `>=5/6`。输入物化将scene-ready I/O与native GPU/evidence CPU流水化，
首轮archive/preprocess期间执行已冻结R5 Actor GPU read。

Scene-ready native launcher以preprocess日志最后的`Processed dynamic masks`作为完成事件，最多同时运行2个
单scene native workers；因此首两个scene完成后GPU即启动，而不等待其余四个preprocess结束。它不读取quality。

### R5 Actor false-safe 终态

Canonical：`run://worldsim_v65/WS-V65-P1R5-ACTOR-FALSE-SAFE-01/20260827T123100Z__actor-false-safe-s0-r1`。
47/24 train/eval trajectories，eval 302 Actor tokens。A0 trajectory-level max forecast Spearman=`0.626087`，虽将
lowest-risk 40% target cost从`0.332236→0.245620`（`-26.07%`），但forecast只过1/2 gates；A1 descriptive
Spearman=`0.488696`。

False-safe gap有9/24 positives；Dplus monitor Spearman=`-0.054402`、AUROC=`0.522222`，selected gap
`0.033120→0.057430`（`+73.40%`），0/3 gates。登记`V65-F11`，关闭Actor trajectory monitor，不做阈值、
learned head或聚合 rescue。该GPU read与P2V archive/preprocess I/O实际重叠；当前active task转为P2V input pipeline。

### R6 smooth-tail visited-state aggregation freeze

P2V公开tar顺序扫描继续占用I/O时，使用既有compact cache启动独立GPU diagnostic，避免3090完全等待。Active
hypothesis=`WS-V65-H-P1R6-001`：在R4已成立的trajectory-level visited-state预测对象上，对比保留的
`Qmean`与唯一固定的可微`Qsoft-tail=sum softmax(q0/0.10)·q0`。目标、2秒/1.5m trajectory footprint、
minimum 16 points与nested eval均不变。

单candidate temperature、无learned head/threshold/seed/sweep；必须同时实现selected-40% cost相对Qmean降低
`>=10%`、unsafe AUROC不降、Spearman回退不超过`0.02`且scene lower>higher。该legacy train-only read不会
修改已冻结的P2V Qagg candidate；失败即关闭smooth-tail，不做温度救援。

### R6 smooth-tail终态：保留Qmean，关闭tail pooling

Canonical：`run://worldsim_v65/WS-V65-P1R6-SMOOTH-TAIL-VISITED-STATE-01/20260827T124500Z__smooth-tail-s0-r1`。
Qsoft-tail把unsafe AUROC从`0.978261→1.000000`，但Spearman从`0.751487→0.708230`（`-0.043256`），
selected-40% actual cost从`0.038137→0.048535`（恶化`27.27%`），scene lower/equal/higher=`4/6/5`。
4 gates只过AUROC，登记`V65-F12`。

ICML MIDAM的smoothed-max/attention是以bag-level AUC训练的新模型，RAP需要planner-coupled robust prediction，
TAT聚合多条候选轨迹及历史；均不支持对当前单轨迹q0做事后temperature rescue。关闭smooth-tail，P2V的唯一
candidate继续保持冻结Qmean。run wall=`0.562s`、peak GPU=`0.00195GiB`，与fresh archive scan重叠；单卡资源充足。

### R7 monotone visited-state calibration freeze

Active hypothesis=`WS-V65-H-P1R7-001`。R4 Qmean排序强但raw MSE高；R4 unconstrained head虽然校准更好却破坏
排序。R7只在108个legacy train units拟合`sigmoid(a·logit(Qmean)+b), a>0`，用58个nested-eval units一次评分。
单调约束保证表示目标仍是“给定τ，未来2秒访问状态的expected hidden-FREE rate”，而不是返回逐voxel判断。

模型仅2参数、seed=0、800 full-batch GPU epochs，无context/isotonic knots/bin/seed/capacity sweep。gates要求MSE降低
`>=50%`、5-bin calibration error降低`>=30%`、至少8/15 scenes MSE改善，同时Spearman/AUROC不退化且selected-
40% indices完全不变。该legacy diagnostic不消费formal calibration split，也不改变P2V Qmean候选。

### R7终态：单调校准保留排序并显著改善expected error

Canonical：`run://worldsim_v65/WS-V65-P1R7-MONOTONE-VISITED-STATE-CALIBRATION-01/20260827T125000Z__monotone-calibration-s0-r1`。
拟合map为`sigmoid(1.703977·logit(Qmean)-0.479222)`。MSE `0.0273778→0.00210441`（`-92.31%`），
5-bin calibration error `0.156639→0.0177814`（`-88.65%`），scene lower/equal/higher=`15/0/0`。

由于map严格单调，Spearman=`0.751487`、unsafe AUROC=`0.978261`、AUPRC=`0.994327`均精确不变；selected-
40%的23个units及actual cost=`0.0381365`完全相同。6/6 gates全过，verdict=`positive_train_only_monotone_
visited_state_calibration`。该结果只保留calibrator form供未来独立cohort，当前P2V仍只读冻结raw Qmean。wall=
`2.319s`、peak GPU=`0.00195GiB`、RSS=`0.954GiB`，与fresh archive scan重叠。

### P2V scene-ready pipeline engineering recovery

公共tar顺序扫描期间，已完成shard的member count与`scene-0001/0219/0594/0822`各自所需文件精确对应。为真正
流水化，准备器父进程被可恢复地`SIGSTOP`，剩余shard workers继续I/O；先并行转换`scene-0001/0219`，避免父进程
把预处理过程中已创建的目录误判为完成。两个scene约2.5分钟完成，最终dynamic-mask日志成为唯一native事件。

首次scene-ready native入口随后在任何run directory/model/quality read前触发`V65-F13`：runner先对不存在的task
parent执行`disk_usage`。按Python `Path.mkdir(parents=True, exist_ok=True)`合同，在launcher启动worker前创建task
parent；scientific config、scene、seed、run prefix均不变，失败无科学read。修复后从已完成日志继续`0001/0219`。

重启入口又在model/quality read前暴露`V65-F14`：launcher直接调用V6.3 generic runner，因而没有解析overlay中的
`base_config`，访问`inputs`时报`KeyError`。已成功P2流程的`run_worldsim_v64_fresh_sidecars.py`负责合并base+
overlay并写resolved config；launcher改为复用该wrapper，不在P2V配置中复制整套backend schema。两个失败目录只含
空`plans/reports/logs`，保留改名为`.failed-entry-v65-f14`后，用原冻结r1 run path继续。

### P2V inputs终态：72 units已就绪，quality仍未读

Canonical preparation=`.../20260827T123000Z__fresh-visited-prep-s0-r1`：10个tar workers提取10,705个新members，
wall=`4108.35s`；三批scene-ready preprocess提前完成6 scenes，父进程最终全部reuse并安全删除temporary raw，
quality read=false。

6个native scene runs全部passed，每scene 12 targets，wall=`56.84/56.68/44.39/53.70/55.04/48.16s`，peak
GPU最大`4.1314GiB`。aggregate=`.../20260827T133000Z__fresh-visited-native-aggregate-s0-r1`含72 targets、
`3,317,884,446` bytes，inference repeated=false，所有target evidence/calibration/confirmation/test read=false。

Canonical evidence=`.../20260827T133000Z__fresh-visited-evidence-s0-r1`含72 units、`76,067,478` bytes、role overlap=0，
wall=`58.72s`；其中前24 units在native/后续preprocess期间先算，final run以hardlink复用。输入阶段通过，下一步是
冻结Qmean的唯一formal P2V prediction-object read。

### P2V formal r1 tensor-shape entry failure

Formal r1=`.../20260827T141500Z__fresh-visited-transfer-s0-r1`在第一个unit加载native/target后、任何Qagg/metric/
gate输出和compact cache落盘前触发`V65-F15`。冻结q0 network本身返回`[B]` logits，evaluator错误假设`[B,1]`
并调用`.squeeze(1)`，产生dimension out of range。

按PyTorch output shape合同，修复仅为`network(batch).reshape(-1)`，同时兼容`[B]`和`[B,1]`且不改变数值、模型、
candidate、sampling、target或gate。r1如实计为formal input/target partial exposure（1 unit），但无科学metric disclosed；
compact cache不存在。用新run-id r2完成同一冻结read，不借故重选或调参。

### P2V fresh终态：trajectory visited-world-state prediction object成立

Canonical：`run://worldsim_v65/WS-V65-P2V-VISITED-STATE-TRANSFER-01/20260827T142000Z__fresh-visited-transfer-s0-r2`。
72 source units中9个按冻结minimum 16 visited samples排除，63 eligible units含8,862 visited points、1,055 hidden-
FREE outcomes与57 unsafe trajectories。

Qmean-target Spearman=`0.633963`（gate `>=0.60`），unsafe AUROC/AUPRC=`0.994152/0.999390`，lowest-risk
40% actual cost `0.102965→0.0522594`（降低`49.25%`，gate `>=40%`），scene lower/equal/higher=`5/1/0`。
4/4 gates全过，verdict=`supported_fresh_trajectory_visited_state_qagg`。

这把R4 legacy signal提升为fresh representation-selection支持：预测对象是给定Ego τ后未来2秒实际访问world states的
可靠性，而非任意voxel是否正确。claim不扩张到Actor、learned risk、admission、calibration、planning或safety。
r2 wall=`9.175s`、peak GPU=`0.0236GiB`、RSS=`1.143GiB`。下一条件分支是为R7单调calibrator选择独立unused
calibration cohort；不得在本P2V cohort上追加事后calibration read。

### P3C independent calibration-transfer freeze

Active hypothesis=`WS-V65-H-P3C-001`。101个processed dirs里没有独立未用direct-key scene；因此不复用P2V或
legacy quality。580个unprocessed direct-key candidates的persistent raw reuse均为0，必须重新archive scan。

为减少I/O且保留index跨度，利用21,345个已知member对12 scenes的archive band审计：每scene全部member只落在其
85-index band。固定band 1/5/10，并各取eligible scene的1/3、2/3 quantile，得到`0030/0055/0453/0501/1046/
1085`。先扫shards 1/5/10；缺member只触发相同scene的10-shard capability fallback，不换scene、不读quality。

Candidate是R7已冻结map `sigmoid(1.703977·logit(Qmean)-0.479222)`，不refit。gates：MSE `>=50%` reduction、
5-bin calibration error `>=30%` reduction、scene MSE support `>=5/6`、Spearman/AUROC不退化、selected-40%
indices完全一致。该read不产生conformal/admission/planning/safety claim。

P3C evaluator已实现为独立入口：复用P2V已修复的streamed materializer，只物化Qmean与trajectory-level target；
随后应用冻结两参数map并同时报告continuous error、5-bin calibration、scene MSE、unsafe ranking和selected-set
identity。无optimizer/refit代码，formal read前仅执行syntax/config解析。

### P3C inputs终态：三shard流水线完成，formal calibration尚未读

Canonical preparation=`run://worldsim_v65/WS-V65-P3C-CALIBRATION-PREPARATION-01/20260827T145000Z__calibration-prep-s0-r1`。
只扫描shards `1/5/10`就找到冻结6 scenes所需的10,689个public-tar members，wall=`1901.12s`；不需要全十shard
fallback。已完成shard上的scene立即预处理，父进程最终复用6/6 processed scenes并安全清除temporary raw，
quality read=false。

6个native scene runs全部passed，每scene 12 targets，wall=`77.05/71.65/51.94/50.90/47.34/50.17s`；peak
GPU最大=`4.1314GiB`。Aggregate=`run://worldsim_v65/WS-V65-P3C-CALIBRATION-NATIVE-SIDECAR-01/20260827T150000Z__calibration-native-aggregate-s0-r1`，
含72 targets、`3,317,884,541` bytes，inference repeated=false，且target evidence/calibration/confirmation/test read全为false。

Canonical evidence=`run://worldsim_v65/WS-V65-P3C-CALIBRATION-EVIDENCE-01/20260827T150000Z__calibration-evidence-s0-r1`：
72 units、`66,004,741` bytes、role overlap=0、wall=`33.85s`，其中前48 units由partial run hardlink复用。首次canonical
CLI在argparse阶段因遗漏必填`--processed-root`被拒绝（`V65-F16`），未创建run dir、未读input/quality；补入
配置已冻结的standard processed root后成功，科学合同未变。下一步是唯一一次frozen R7 map的independent
calibration-transfer read。

### P3C formal r1 artifact-locator entry failure

Formal r1=`run://worldsim_v65/WS-V65-P3C-MONOTONE-CALIBRATION-TRANSFER-01/20260827T154500Z__calibration-transfer-s0-r1`
在加载冻结q0 artifact时触发`V65-F17`：P3C config错指向不存在的V6.4 baseline-transfer `models/` path。失败发生在
任何unit/native/target读取、q0 forward、metric/gate和cache落盘之前，因此formal calibration quality仍未读。

已将locator更正为成功P2V实际使用的同一冻结artifact：`WS-V64-P6R-SELECTIVE-MLP-01/.../RISK_MODEL/
full_native_selective_mlp.joblib`。仅run-relative artifact locator改变；model、scene、target、sampling、calibrator、seed和
gates均不变。r1现场保留并标记failed，r2将继续同一冻结read。

### P3C终态：冻结单调calibrator独立迁移成立

Canonical=`run://worldsim_v65/WS-V65-P3C-MONOTONE-CALIBRATION-TRANSFER-01/20260827T155000Z__calibration-transfer-s0-r2`。
72 source units中12个按冻结minimum 16 visited samples排除；60 eligible units含6,675 visited points、708 hidden-FREE
outcomes和48 unsafe trajectories。`scene-1046`的12 units全部低于minimum footprint，因此scene MSE只在5个可评估
scenes上计算，五个全部改善（lower/equal/higher=`5/0/0`），达到预注册minimum support=`5`。

原始Qmean MSE=`0.0287445`，冻结map后=`0.00207044`，降低`92.80%`；5-bin absolute calibration error
`0.162039→0.0189368`，降低`88.31%`。严格单调性使Spearman=`0.715491`、unsafe AUROC/AUPRC=
`0.982639/0.995763`精确不变；selected-40%的24 units与actual cost=`0.0298324`也完全不变。

6/6 gates全过，verdict=`supported_independent_monotone_visited_state_calibration_transfer`。这将R7从train-only
form提升为independent cohort支持，但不产生conformal coverage、admission、planning或safety claim。r2 wall=`7.923s`、
peak GPU=`0.02359GiB`、RSS=`1.195GiB`；单RTX 3090足够。下一个研究对象是固定candidate trajectory set的
action-level visited-state ranking，不在P3C cohort上做事后sweep。

### P10V fixed-action visited-state transfer freeze

Active=`WS-V65-H-P10V-001`。不重开V6.4已失败的linear collision critic；直接复用其冻结action generator：
4个progress ratios×3个lateral offsets得到12条非停车轨迹。对每条轨迹询问未来2秒/1.5m footprint中实际访问的
world states是否可靠；stop因不访问未来footprint只报告、不作为reliability unit。

新独立cohort从574个unprocessed/unmentioned direct-key scenes中冻结：archive bands 2/6/9的1/3和2/3 quantiles，
即`0159/0184/0577/0599/0955/0983`。先只扫shards 2/6/9；缺member时只允许same-cohort full-scan，不换scene。
Primary gates为pooled Spearman `>=0.55`、unsafe AUROC `>=0.80`、within-case pairwise concordance `>=0.65`、
lowest-Qmean 25% actual-cost reduction `>=25%`、scene support `>=5`、evaluable cases `>=48`。不训练新critic、不扫参。

P10V evaluator已实现：每case只对最多8,192 boundary points做一次冻结q0 forward，然后在GPU上将同一批
points与12条action paths并行做footprint membership，不重复model inference。物化后才计算pooled、within-case pairwise、
scene support和lowest-quarter cost，并写出`ACTION_ROWS.jsonl`。当前shards 2/6/9正并行扫描，quality read=false。

### P10V inputs终态：scene-ready四级流水线完成

Preparation=`run://worldsim_v65/WS-V65-P10V-ACTION-PREPARATION-01/20260828T001000Z__action-prep-s0-r1`。
Shards 6/2/9分别约`936.3/1011.8/1069.6s`，总共找到10,709 members，无full-scan fallback；parent wall=
`1462.02s`。原子member-ready feeder把6 scenes分三批预处理，wall为`161.16/135.51/141.27/154.00/
190.33/197.18s`，并与shard scan、native GPU、evidence CPU重叠；父进程最终reuse 6/6并清理raw。

6个native runs均passed，每scene 12 targets，wall=`48.71/45.81/51.14/52.43/45.43/52.69s`，peak
GPU=`4.1314GiB`。Aggregate=`run://worldsim_v65/WS-V65-P10V-ACTION-NATIVE-SIDECAR-01/20260828T001500Z__action-native-aggregate-s0-r1`：
72 targets、`3,317,884,673` bytes、inference repeated=false，所有target/calibration/confirmation/test read=false。

Evidence partial在later preprocess/native期间先生成48 units；canonical=`run://worldsim_v65/WS-V65-P10V-ACTION-EVIDENCE-01/
20260828T001500Z__action-evidence-s0-r1`含72 units、48 reused、`75,306,035` bytes、wall=`32.17s`、role overlap=0。
输入里程碑没有新failure；formal action quality read=false。

### P10V终态：fixed candidate trajectories的visited-state ranking成立

Canonical=`run://worldsim_v65/WS-V65-P10V-ACTION-VISITED-STATE-TRANSFER-01/20260828T003000Z__action-transfer-s0-r1`。
72 cases的864条非stop actions中，51条按冻结minimum 16 visited points排除；813 eligible actions含55,411
visited points、6,826 hidden-FREE outcomes与659 unsafe actions。72个stop actions仅按合同报告并排除。

Pooled Qmean-target Spearman=`0.740235`，unsafe AUROC/AUPRC=`0.858779/0.945415`。在within-case target-cost gap
`>=0.02`的2,834个pairs上，方向一致率=`0.732534`。72/72 cases至少有2条evaluable actions；每case选lowest-
Qmean 25%后，actual cost `0.109772→0.0732644`（降低`33.26%`），scene lower/equal/higher=`6/0/0`。

6/6 gates全过，verdict=`supported_fresh_fixed_action_visited_state_ranking`。这直接支持“给定Ego `tau`，
未来2秒实际访问的world states的可靠度”在固定candidate set中可排序；没有训练critic、不读collision
label，不产生planner/policy/closed-loop/safety claim。wall=`8.459s`、peak GPU=`0.03917GiB`、RSS=`1.001GiB`。
当前cohort不做lattice/threshold/head扫描；下一步只允许将冻结的unit-level ranking、monotone calibration与fixed-action
ranking放到一个全新cohort做一次combined empirical confirmation。

### P10X one-shot combined confirmation freeze

Active=`WS-V65-H-P10X-001`。在任何confirmation quality读取前，冻结候选为：nominal-route raw Qmean、
R7两参数单调map（no refit）、以及P10V固定12-action lattice。剩余568个unprocessed direct-key scenes中，
固定未扫过的archive bands 3/7/8的1/3和2/3 quantiles：`0245/0287/0686/0718/0817/0868`。

为避免过度gating，只保留6个核心门：nominal route Spearman `.60`、冻结map MSE reduction `.50`、action
Spearman `.55`、unsafe AUROC `.80`、pairwise concordance `.65`、selected action cost reduction `.25`。只允许
一次combined quality read；失败即关闭候选，不建第二个confirmation cohort。

P10X combined evaluator已实现并通过远端Python语法检查。实现只调用一次P10V compact materialization：固定
action index 7作为nominal route，在同一批冻结q0/visited-state arrays上同时计算raw route ranking、R7 frozen
monotone MSE/calibration与12-action ranking/selection；不增加head、refit、critic或额外门。正式quality仍未读。

Preparation=`run://worldsim_v65/WS-V65-P10X-CONFIRMATION-PREPARATION-01/20260828T010000Z__confirmation-prep-s0-r1`
已启动shards 3/7/8三路并行扫描；父协调器被暂停而三个archive workers继续I/O，scene-ready preprocess与native
GPU watcher已经并行等待原子member交付。首次feeder直接脚本调用因仓库根目录未进入`sys.path`而在import阶段退出，
记录为`V65-F18`；改用进程级`PYTHONPATH=.`后两个watcher正常运行，未创建科学run、未读取quality。

Pipelined native aggregate收口器也已实现：逐场景run完成后仅建立`units/fresh_selection/<scene>`只读symlink，
汇总72个既定targets、bytes与worker peak，不复制大数组、不重复inference、不增加哈希或quality gate。

### P10X inputs终态：archive/preprocess/native/evidence重叠完成

Preparation=`run://worldsim_v65/WS-V65-P10X-CONFIRMATION-PREPARATION-01/20260828T010000Z__confirmation-prep-s0-r1`。
Shards 3/7/8分别找到`3,533/3,589/3,596` members，约`1010.9/982.5/1014.7s`；合计10,718 members，
无full-scan fallback。父协调器wall=`1564.11s`，复用feeder已完成的6/6 processed scenes并清理temporary raw。

六场景预处理wall=`160.52/151.46/175.19/159.90/199.32/196.81s`。6个native runs均passed，各12 targets，
wall=`74.11/80.51/49.38/47.15/57.61/57.41s`，peak GPU=`4.1314GiB`。Aggregate=
`run://worldsim_v65/WS-V65-P10X-CONFIRMATION-NATIVE-SIDECAR-01/20260828T010500Z__confirmation-native-aggregate-s0-r1`：
72 targets、`3,317,884,470` bytes、inference repeated=false。

前四场景的48-unit partial evidence与后两场预处理/native重叠。Canonical=`run://worldsim_v65/
WS-V65-P10X-CONFIRMATION-EVIDENCE-01/20260828T010500Z__confirmation-evidence-s0-r1`：72 units、48 reused、
`81,763,088` bytes、wall=`35.14s`、role overlap=0。输入全部ready，正式combined quality read仍为false；下一步只执行
冻结的单次P10X evaluator。

### P10X终态：reliability evaluation成立，direct action authority拒绝

Canonical=`run://worldsim_v65/WS-V65-P10X-COMBINED-CONFIRMATION-01/20260828T013000Z__combined-confirmation-s0-r1`。
72 cases/864 actions中125 actions按冻结16-point footprint排除；739 eligible actions含80,282 visited points、
10,818 hidden-FREE outcomes和577 unsafe actions。Nominal route有60 eligible cases；`scene-0718`的12条nominal routes
全部低于minimum footprint。

Route raw Spearman=`0.609813`、unsafe AUROC/AUPRC=`0.988868/0.997419`。冻结R7 map将MSE
`0.0318414→0.00592580`（降低`81.39%`），5-bin calibration error `0.159217→0.0203975`（降低`87.19%`）；
route scene MSE=`4/0/1`。Action pooled Spearman=`0.772946`、unsafe AUROC/AUPRC=`0.972730/0.991627`、
2,216-pair concordance=`0.655686`，前五个核心门均通过。

唯一失败门是direct selection benefit：65 evaluable cases中选lowest-Qmean 25%，actual cost仅
`0.120215→0.100520`（降低`16.38%`），低于冻结`25%`；scene lower/equal/higher=`5/0/1`，其中`scene-0817`
小幅退化。Verdict=`rejected_one_shot_combined_visited_state_confirmation`，记录为`V65-F19`。

准确终态是：支持“给定Ego `tau`，未来2秒访问world states的reliability ranking与expected-error calibration”，但
不支持把该分数直接编译为达到冻结benefit的action authority。按one-shot stop rule，不换scene、不放宽gate、不建第二
confirmation、不训练critic补救；无collision/planning/policy/closed-loop/population/safety claim。wall=`8.218s`、
peak GPU=`0.03917GiB`、RSS=`1.052GiB`，单RTX 3090足够。

V6.5终局证据链、canonical runs、arXiv可写/禁写claims与建议图表已汇总到
[`autoresearch/worldsim_v65/V65_RESEARCH_CLOSEOUT.md`](autoresearch/worldsim_v65/V65_RESEARCH_CLOSEOUT.md)。

### V6.5 arXiv report handoff终态

已补齐正式技术报告
[`V65_ARXIV_TECHNICAL_REPORT.md`](autoresearch/worldsim_v65/V65_ARXIV_TECHNICAL_REPORT.md)与
[`ARXIV_EVIDENCE_INDEX.md`](autoresearch/worldsim_v65/ARXIV_EVIDENCE_INDEX.md)，并将原计划状态从`active`更新为
`closed / arxiv_report_ready`。`FAILURE_ANALYSIS.md`现已覆盖`V65-F12`、`V65-F13--F18`工程恢复边界与`V65-F19`
终局decision-benefit失败，不再只停留在早期F01--F11。

报告交接前只做read-only证据可用性审计：P2V/P3C/P10V/P10X canonical summary/status均存在且JSON可读，P10V
`ACTION_ROWS.jsonl`保留813行；无active V6.5进程，run tree约16GiB，磁盘剩余约95GiB。审计不重算指标、不执行
smoke/regression、不添加hash/checksum/fingerprint。V6.5状态=`v65_research_complete_arxiv_report_ready`，无active
task/hypothesis；push成功后按用户要求关闭AutoDL远端。
