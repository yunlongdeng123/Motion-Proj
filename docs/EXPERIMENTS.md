# Experiments

## WorldSim V6.1 ME-2 H003 Hunyuan actor 四臂 REJECTED（2026-08-22）

- canonical=`run://worldsim_v61/WS-V61-ME2-HY3D-OCC-ACTOR-01/20260822T121848Z__hy3d-actor-s1234-r1`，
  source=`98cec20ae808600309afd2066f7826b2d94ed0b9`，hypothesis=`WS-V61-H-ME2-003`。
- 4 个唯一 unit、16 个生成资产、24 个 case-arm evaluation 全部完成；A0/A1/A2/A3 均=`0/6 ACCEPT`，
  primary A3 false-safe=`0`，但 `primary_voxel_minimum_accept_count` 未通过。
- 共同失败因子是所有四臂、所有 6 例都有 observed FREE-space conflict。A3 method conflict=`6..246`、hidden eval
  conflict=`8..273`；其 coverage、hole coverage 与 silhouette 已非共同瓶颈，因此不授权 prompt/seed/texture/
  steps/octree/threshold 调参，也不做 post-hoc clipping。
- wall=`675.64s`、Omni worker=`604.11s`、peak GPU=`9.45GiB`；H002 A0 exact reuse wall=`232.82s`，
  formal workers offline，无训练/confirmation read。
- gate/arm-summary/summary/resource/manifest/terminal=`1eab2226...d86 / dc2222df...505 / 85e20dd9...e73 /
  e438e93e...dde / f7fae41a...118 / 9b90d9eb...dc9`；failure ledger delta=`V61-F09`。
- decision=`stop_hy3d_without_prompt_seed_texture_tuning`；next=`WS-V61-ME3-PREDICTED-OCC-01`。该结论只拒绝
  Hunyuan actor proposal，不拒绝计划内独立的 learned occupancy 路线。

## WorldSim V6.1 ME-2 H002 infrastructure failure / H003 RECOVERY PRE-REGISTERED（2026-08-22）

- H002 failed run=`run://worldsim_v61/WS-V61-ME2-HY3D-OCC-ACTOR-01/20260822T120519Z__hy3d-actor-s1234-r1`。
  A0 已产生4个有效 mesh；Omni 完成首个2-sample A1 diffusion/decode 后，官方 vanilla extractor 只对
  `grid_logits[0]` 做 marching cubes 并返回1个 mesh，runner 以 `1!=2` fail-closed。failure=`V61-F08`，
  generated Omni assets/method decisions=`0/0`，无科学结论。
- H003 保持 diffusion batch2 与逐样本 fixed generators，改为官方 pipeline 返回2份 latent，再逐份调用同一 VAE
  batch1 decode/export。H002 的4个 A0 assets 只在旧 plan/input/report/asset hashes 全部精确后复用；不重复 A0。
- model、control、seed、50 steps、octree256、guidance、compiler、O_method/O_eval、thresholds、resource 和 stop
  rule 全部不变；新增2-latent→2次batch1 decode回归测试。

## WorldSim V6.1 ME-2 H001 infrastructure failure / H002 RECOVERY PRE-REGISTERED（2026-08-22）

- H001 failed run=`run://worldsim_v61/WS-V61-ME2-HY3D-OCC-ACTOR-01/20260822T120008Z__hy3d-actor-s1234-r1`。
  四个冻结输入已构造，但 A0 worker 在 model load/GPU inference 前因缺少官方 requirements 固定的
  `pymeshlab==2022.2.post3` 导入失败；generated assets=`0`、method decisions=`0`，failure=`V61-F07`，
  不构成 Hunyuan 方法 rejection。
- H002 只安装并绑定该 exact dependency；`Hunyuan3DDiTFlowMatchingPipeline` 离线 import smoke 已通过。
  四臂、models、4 units/6 cases、seeds、batch2、50 steps、octree256、compiler、gates、资源与 stop rule 全部不变。

## WorldSim V6.1 ME-2 Hunyuan actor 四臂 PRE-REGISTERED（2026-08-22）

- task=`WS-V61-ME2-HY3D-OCC-ACTOR-01`，hypothesis=`WS-V61-H-ME2-001`，状态=`formal_ready`。
- 固定 arms=`A0-image / A1-bbox / A2-point / A3-voxel`；4 个唯一 actor units 映射到同一冻结 6-case actor
  denominator，duplicate frontend 只复用 byte-exact asset。A0 绑定官方 Hunyuan3D-2.1 commit/model revision，
  A1–A3 绑定已通过 P4 的 Omni commit/model revision。
- input 统一为 Omni LHW 坐标；A2=`2048` deterministic raw-LiDAR actor points，A3=`8192` deterministic
  O_method actor voxel centers。单次 no-model/no-O_eval 预检为 4/4 finite、raw actor points 非空、O_method actor
  voxels=`10878..23088`、6-case minimum actor-hole coverage=`0.6322`。
- 固定 seed=`1234..1237`、batch=`2`、50 steps、octree=`256`；最大 extent 下 decode spacing 约 `0.0604m`
  小于 `0.2m` occupancy cell。无 prompt、texture、seed、step、resolution 或 threshold sweep。
- compiler 只允许 aspect axis permutation + one uniform scale；method decisions 在读 O_eval 前固化。
  surface support>=`0.80`、native actor coverage>=`0.20`、hole coverage>=`0.10`、silhouette IoU>=`0.05`，
  FREE conflict 与 unfiltered swept collision 都必须为0。
- scene-0242 只过滤证据绑定的 actor4 truck/actor15 trailer hitch pair；其余接触照常拒绝，不做全局 collision
  relaxation。primary A3=`>=2/6` 且 false-safe=`0`；失败立即停止 Hunyuan 路线，不进入调参循环。

## WorldSim V6.1 P4 Hunyuan3D-Omni 3090 smoke PASS（2026-08-22）

- canonical=`run://worldsim_v61/WS-V61-P4-HY3D-OMNI-3090-SMOKE-01/20260822T112707Z__voxel-smoke-s1234-r1`，
  source=`a97b2743935e3a7143d5b75da9e7bc5bac95e317`。
- 完全离线生成有效 voxel-controlled mesh=`1,238,856 vertices / 2,477,728 faces` 与非空 finite sampled
  points；wall=`235.16s`、peak=`7.90GiB`、disk start=`95.07GiB`，无训练或 confirmation read。
- gate/summary/manifest/terminal=`23451b2d...5cf / 8133a65b...ab7 / 7c4783cb...9a2f2 / 177ce781...8a3`；
  all PASS。`V61-F04/F05/F06` 继续保留，修复后不再重复 cache/weight 验证循环。

## WorldSim V6.1 P4 Hunyuan3D-Omni 3090 smoke PRE-REGISTERED（2026-08-22）
- task=`WS-V61-P4-HY3D-OMNI-3090-SMOKE-01`，hypothesis=`WS-V61-H-P4-001`，先建立隔离 Python3.10/
  torch2.5.1+cu124 环境并下载内容寻址权重，不在 `motionproj` 环境混装。
- official git=`4d47c0cc...bfa8`；HF model=`70e803bf...d485`；DINOv2-large=`47b73eef...2d6c`。
- 固定官方 voxel demo 的一个样本、seed1234、50 steps、512 octree、guidance4.5；无 EMA/fast decode/参数 sweep。
- formal run 必须 offline；mesh vertices/faces、finite、sampled points 均非零，peak<22GiB、wall<1200s、disk>=60GiB。
- license boundary=仅中国 AutoDL 科研执行，不分发模型/输出，不把输出训练进其他模型；不产生驾驶 actor 质量声明。

## WorldSim V6.1 ME-1 oracle Occupancy PASS（2026-08-22）
- canonical=`run://worldsim_v61/WS-V61-ME1-ORACLE-OCC-PROPOSAL-01/20260822T104207Z__oracle-occ-s20260822-r1`，source=`e422f05`。
- B0=`0/28`；B1=`3/7/18`；O1=`3/7/18`；O2=`10 ACCEPT/0 ABSTAIN/18 REJECT`，false-safe=`0`，
  accepted mask yield=`39.8300%`；O3=`6/28`、false-safe=`0`。
- O2 原 R10 三例全部保留，新增3 actor与4 static/disocclusion；method decisions 在 O_eval 读取前固化，全部 gate PASS。
- O3 actor rejection 保留 native swept OBB overlap 证据，不回改为 acceptance。wall=`3.60s`、peak GPU=`0.51GiB`。
- `V61-F03`：canonical O3 的 scene-0048 raster 把合法 actor0 与 empty=0 混同；O2 主臂不受影响，O3 该部分
  降格，后续 ME-2/ME-4 使用 empty=`-1` 修复且不重跑 ME-1 主实验。
- gate/summary/metrics/manifest=`6aca5f2f...246d / 61713df4...afb9 / dbb1d0a3...ffb6 / 63ae8e56...e7d5`；
  failure_ledger_delta=`none`；next=`WS-V61-P4-HY3D-OMNI-3090-SMOKE-01`。

## WorldSim V6.1 ME-1 H001 infrastructure failure / H002 RECOVERY PRE-REGISTERED（2026-08-22）
- H001 在 run/GPU 前因把 ME-0 gate authority `checks.passed` 误读为顶层 `passed` 而 `KeyError`；canonical=`null`，
  method result=`null`，failure=`V61-F02`。
- H002 只修正 frozen JSON schema 路径并新增 regression test；28-case、B0/B1/O1/O2/O3、0.2m/0.1m、
  50% coverage、20% depth、primary gate、资源与 confirmation lock 全部不变。

## WorldSim V6.1 ME-1 oracle Occupancy PRE-REGISTERED（2026-08-22）
- task=`WS-V61-ME1-ORACLE-OCC-PROPOSAL-01`，hypothesis=`WS-V61-H-ME1-001`，状态=`formal_run_pending`。
- 五臂=`B0-2D / B1-R10 / O1-GATE / O2-OCC-GEOMETRY / O3-OCC-4D`；primary=`O2-OCC-GEOMETRY`。
- `O_method` 在 GPU 上以 0.1m step 光线抽取 0.2m closed voxel surface，绑定冻结 R9 proposal RGB；method decisions
  在任何 `O_eval` tensor 读取前写入不可变 artifact。随后 `O_eval` 只报告 free conflict、observed support、unknown、
  projected coverage、method/eval depth overlap 与相对深度误差以及 false-safe。
- 沿用 R9 的 `coverage>=0.50`、`median relative depth error<=0.20`；一 voxel support tolerance；没有 scalar 补偿、
  case 特判、模型训练或 threshold sweep。O3 对 native OBB 做 lifecycle/rigid pose/canonical size 与 5 点 swept SAT。
- primary gate=`>=5/28`、false-safe=`0`、保留 R10 原 3 个 ACCEPT、新增至少 1 actor 和 1 static/disocclusion、
  accepted mask-area yield>=12%。失败则停止 Hunyuan3D 与 learned Occupancy 接入。
- failure_ledger_refs=`V61-F01,V6-F25,V6-F26,V6-F65,V6-F71,V6-F78,V6-F79`；failure_ledger_delta=`none`（预注册）。

## WorldSim V6.1 ME-0 SceneIR-O PASS（2026-08-22）
- canonical=`run://worldsim_v61/WS-V61-ME0-OCCIR-01/20260822T101817Z__occir-s20260822-r1`，source=`5a3bc42`。
- 4 scene/frame units、8 `O_method/O_eval` tiers、28 case bindings；method/eval raw LiDAR path 与 payload hash 全局互斥。
- 三态非空、oriented actor volume anti-AABB、identity/lifecycle、source-removal→UNKNOWN、fresh spawned-process exact、
  coordinate round-trip<=`2.14e-14m` 全通过；wall=`10.57s`，4 CPU workers，无 GPU/model/training/confirmation。
- gate/summary/manifest/terminal SHA=`1e818074...8bb7 / 6e50644b...b14f / 386d99ab...59ec / 199c9cf3...f9a7`。
- failure_ledger_delta=`none`；next=`WS-V61-ME1-ORACLE-OCC-PROPOSAL-01`。

## WorldSim V6.1 ME-0 SceneIR-O PRE-REGISTERED（2026-08-22）

- task=`WS-V61-ME0-OCCIR-01`，hypothesis=`WS-V61-H-ME0-001`，状态=`running`；formal run pending。
- 4 scene/frame units 并行构建 8 个 `O_method/O_eval` grid；0.2m target-LiDAR frame，显式 `T_dst_src`，
  oriented-box sparse actor identity/lifecycle 层，static FREE/OCCUPIED/UNKNOWN 与 actor layer 分离。
- method/eval 使用全局互斥的偶数/奇数 raw LiDAR payload；两批 fresh spawned-process 内容 exact；source removal 必须
  恢复 UNKNOWN。6 targeted tests 与 scene-0242/f52 preflight 已通过；无 GPU/model/training/confirmation。
- failure_ledger_refs=`V61-F01,V6-F26,V6-F65,V6-F71,V6-F78,V6-F79`；failure_ledger_delta=`none`（预注册）。

## WorldSim V6.1 P0 scope freeze PASS（2026-08-22）

- task=`WS-V61-P0-SCOPE-FREEZE-01`。H-P0-001 在 run 创建、source read、GPU/训练前因缺失
  `runs/worldsim_v61` namespace 被 `disk_usage` 拒绝；canonical run=`null`，无方法结论，failure=`V61-F01`。
- recovery hypothesis=`WS-V61-H-P0-002` 正式通过；canonical=
  `run://worldsim_v61/WS-V61-P0-SCOPE-FREEZE-01/20260822T100812Z__scope-freeze-s20260822-r1`，source=`6247fd8`。
- gate/summary/manifest SHA=`fb2a416a...ae40 / e53a86f2...907c / 2ed96578...7593`；全部 gate PASS，
  method/eval source paths disjoint；`V61-F01` resolved。
- 冻结 R10=`3/28`、false-safe=`0`、accepted mask pixels=`107807`；目标为 ME-1 oracle `>=5/28` 且
  false-safe=`0`，同时保留原 3 个 ACCEPT、新增至少 1 actor 与 1 static/disocclusion case。
- `O_method/O_eval` raw LiDAR sweep 路径必须 disjoint；scene mapping=`0048->045 / 0242->191`；confirmation locked。
- failure_ledger_refs=`V61-F01,V6-F25,V6-F26,V6-F65,V6-F71,V6-F78,V6-F79`；H002 failure_ledger_delta=`none`。
- next=`WS-V61-ME0-OCCIR-01`；oracle `<5/28` 时停止模型接入，禁止阈值/提示词/seed 调参。

## WorldSim V6 R10 factorized verification development PASS（2026-08-21）

- canonical=`run://worldsim_v6/WS-V6-R10-FACTORIZED-VERIFICATION-01/20260821T112323Z__factorized-verification-s20260821-r1`，
  source=`9262a89`；gate/summary/manifest/terminal SHA=`ecc70532...364 / 174dc8b5...868 / 65792d15...060 / 14c1a977...550`。
- P1+P2 fusion=`3 ACCEPT / 7 ABSTAIN / 18 REJECT`；false-safe=`0`，vs P0 reduction=`0.8214`，
  usable verified mask area=`0.09524`；semantic/dynamics 与 trajectory length 继续 ABSTAIN。
- R11 只允许两个 static route chunks；actor-removal case 缺 semantic/dynamics，不得烘焙 actor/trajectory。

## WorldSim V6 R9 cross-frontend independent verifier arms development PASS（2026-08-21）

- canonical=`run://worldsim_v6/WS-V6-R9-INDEPENDENT-VERIFIER-ARMS-01/20260821T111819Z__independent-arms-s20260821-r1`，
  source=`9a3b333`；gate/summary/manifest/terminal SHA=`297f1220...ab7 / a3e79468...393 / 212c12cc...fde / b3351a1a...2cd`。
- P1=`10/28` ACCEPT、coverage=`0.3571`、false-safe=`0`；P2=`3/28`、coverage=`0.1071`、false-safe=`0`；
  P3 false-safe=`0.1667` 被排除，P4=`28/28` ABSTAIN。R10 只准融合 P1+P2。
- proposal 来自同帧/同编辑的另一 frontend，仍是同 sensor support 的 reconstructed proposal，不是新增观测或 bake 授权。

## WorldSim V6 R9 SD-v1.5 independent verifier arms 正式实验 REJECTED（2026-08-21）

- canonical=`run://worldsim_v6/WS-V6-R9-INDEPENDENT-VERIFIER-ARMS-01/20260821T111228Z__independent-arms-s20260821-r1`，
  source=`95d2519`；gate/summary/manifest/terminal SHA=`07be410e...7fd / b1ad614c...c51 / 7db9a5fd...fb2 / 99912dd8...ff4`。
- P1=`0/28`；P2=`2/28`，coverage `0.0714<0.10`；P3=`6/12`、false-safe `0.1667>0.10`；
  P4=`28/28` ABSTAIN；eligible R10 arms=`[]`，peak=`2696 MiB`。
- H-R9-003 冻结 cross-frontend reconstructed proposal；所有 arm 阈值与 gate 不变，边界见 `V6-F26`。

## WorldSim V6 R9 Big-LaMa independent verifier arms 正式实验 REJECTED（2026-08-21）

- canonical=`run://worldsim_v6/WS-V6-R9-INDEPENDENT-VERIFIER-ARMS-01/20260821T110743Z__independent-arms-s20260821-r1`，
  source=`a7b8e7a`；gate/summary/manifest/terminal SHA=`07be410e...7fd / 6a74249f...626 / 88fe0a97...25f / 091bbec8...123`。
- P1/P2=`0/28` ACCEPT；P3=`6/12` ACCEPT、false-safe=`0.1667`；P4=`28/28` ABSTAIN；eligible R10 arms=`[]`。
  outside-mask exact、无融合/无 bake、peak=`428 MiB`，因此是质量门负结果而非工程或资源 blocked。
- H-R9-002 冻结切换到 SD-v1.5；cohort、verifier、truth、threshold 与 gate 全不变。失败与 pivot 见 `V6-F25`。

## WorldSim V6 R8 frozen proposal generator 正式实验 capability PASS（2026-08-21）

- canonical=`run://worldsim_v6/WS-V6-R8-FROZEN-PROPOSAL-GENERATOR-01/20260821T104759Z__frozen-generator-s20260821-r1`，
  source=`42758ec`；gate/summary/manifest/terminal SHA=`5d9d01f7...a1d / a027152b...0ca / ed868118...a17 / 3d2153cd...e62`。
- Big-LaMa 与 SD-v1.5 均 8/8 inference 成功并严格复现；Big-LaMa peak/median=`428 MiB / 0.037369 s`，
  SD-v1.5=`2718 MiB / 0.786339 s`，按预注册资源优先级唯一选择 Big-LaMa。
- 前五次 rejected run 及 adapter 失败完整保留，见 `V6-F16`–`V6-F21`；无训练、confirmation 或未验证 bake。
  next=`WS-V6-R9-INDEPENDENT-VERIFIER-ARMS-01`。

## WorldSim V6 R7 oracle missing-world v1 正式实验 development PASS（2026-08-21）

- canonical=`run://worldsim_v6/WS-V6-R7-ORACLE-MISSING-WORLD-01/20260821T100653Z__oracle-missing-world-s20260821-r1`，
  source=`68f1d149`；gate/summary/manifest/terminal SHA=`42180f5a...14e / fe760663...402 / 73dbb2ba...d0c / 388dc82c...012`。
- eligible oracle/decoy=`28/28`，acceptance=`1.0/0.0`；photo/depth median reduction=`1.0/1.0`，四类 hole
  minimum usable gain=`0.999540`，outside-mask/provenance exact；4 项 zero actor-effect 记录 structural ABSTAIN。
- v0 32-case denominator 在完整 gate 前失效，工程/source 失败见 `V6-F12`–`V6-F15`；无 generator、训练或
  confirmation。next=`WS-V6-R8-FROZEN-PROPOSAL-GENERATOR-01`。

## WorldSim V6 R6 factorized validity 正式实验 development PASS（2026-08-21）

- canonical=`run://worldsim_v6/WS-V6-R6-FACTORIZED-VALIDITY-01/20260821T094949Z__factorized-validity-s20260821-r1`，source=`4ff4c644`；
  gate/summary/manifest/terminal SHA=`134099e7...741 / c4d8b7ef...9a4 / e03977bb...a62 / 00773f3e...b65`。
- 48 rows × 4 tasks × 5 methods：factorized photo/geometry/semantic false-safe 均 `0.0`，coverage=
  `0.0833/0.2917/0.0833`；dynamics coverage=`0.0`，无独立 verifier 不产生 ACCEPT。
- V3 worst-case false-safe=`0.5`，factorized gain=`0.5`；保留为 conservative development rule，禁止解释为
  跨域 calibration 或 dynamics readiness。next=`WS-V6-R7-ORACLE-MISSING-WORLD-01`。

## WorldSim V6 R5 provenance field v1 正式实验 PASS（2026-08-21）

- canonical=`run://worldsim_v6/WS-V6-R5-PROVENANCE-01/20260821T094242Z__provenance-s20260821-r1`，source=`a3f93cef`；
  gate/summary/manifest/terminal SHA=`d4888ee2...b33 / a145e858...5ba / 6be1af73...034 / 58237201...89e`。
- 24 chunks、23 actors、1,267,870 primitives 的 `source_type/sensor_support/time_support/view_support/
  reconstruction_source/generation_source` 全覆盖；composite primitive identity unique，typed separation PASS。
- 当前 package 全为 reconstructed；不把 reconstructed 写成 observed，不虚构 generation source；sensor/view support
  为 unknown。v0 failed run=`20260821T094101Z__provenance-s20260821-r1` 保持 failed，见 `V6-F11`。
- next=`WS-V6-R6-FACTORIZED-VALIDITY-01`。

## WorldSim V6 R4 deterministic runtime v0 正式实验 PASS（2026-08-21）

- canonical=`run://worldsim_v6/WS-V6-R4-DETERMINISTIC-RUNTIME-01/20260821T093544Z__deterministic-runtime-s20260821-r1`；
  gate/summary/manifest/terminal SHA=`c7434353...816 / 009a79ff...58d / 12ecc293...add / 7cd63d7d...947`。
- 输入 SceneIR content=`8b3dd863...671`，`1,267,870` primitives、24 chunks、23 actors、168 unique blobs；
  3 次 fresh process audit SHA 均为 `29323a6e...ca2`，package manifest 前后 exact。
- `WORLD_STATE.json / LABELS.json / CHUNK_SELECTION.json / ACTOR_TRAJECTORY.json / RGB.npy` 五项均 byte-exact；
  RGB `array_equal=true`，无需 tolerance。GPU/training/confirmation 均 false。
- claim 仅限 compiler-owned diagnostic runtime determinism；next=`WS-V6-R5-PROVENANCE-01`。

## WorldSim V6 R3 support-deviation 正式实验完成 / hypothesis rejected（2026-08-21）

- canonical=`run://worldsim_v6/WS-V6-R3-SUPPORT-DEVIATION-01/20260821T092516Z__support-deviation-analysis-recovery-s20260821-r1`；
  render source=`20260821T091503Z__support-deviation-s20260821-r1@5144426e`，analysis=`00d26348`；
  ranking/summary/manifest/terminal SHA=`7ead0a6c...67d / 9dd3d767...50c / e866dae3...acc / 7f401392...a78`。
- scene-0242/0048 × StreetGS/AD-GS × frame 52/57，lateral=`0/0.5/1/2/3/5 m`，另含 forward 2 m 与
  actor remove/translate 1 m/trajectory +2 frames。80/80 render hash recovery 复核通过；checkpoint exact immutable。
- aggregate downstream error 随 lateral offset 均值从 `0.167773` 增至 `0.296806`；四个 scene/frontend group 的
  support ordering 均为正。但 support Spearman=`0.352456`、distance=`0.353105`、gain=`-0.000649`、
  residual=`0.056063`，故冻结 gate=`FAIL`，method=`reject_or_revise_analytic_support_before_any_learned_model`。
- AD-GS depth 与 StreetGS depth convention 不直接等价，AD-GS forward relative-depth proxy 约 `0.982`；因此 R3
  只作 sparse observation/cross-frontend development benchmark，不宣称 dense GT、跨 frontend metric calibration
  或安全闭环。actor edits 均产生非零 effect，但不是质量/真实性证明。
- engineering failures=`V6-F06`–`V6-F10`；每个旧 run 保持 failed。资源通过，下一步=
  `WS-V6-R4-DETERMINISTIC-RUNTIME-01`。

## WorldSim V6 R2 SceneIR v0 正式表示实验完成（2026-08-21）

- canonical=`run://worldsim_v6/WS-V6-R2-SCENEIR-V0-01/20260821T082835Z__sceneir-v0-s20260821-r1`，source=
  `aeaa1cbee58521082d9d30e09a6288b89cb06c4c`，summary/manifest/terminal SHA=
  `98938eef...afb / bf53c172...6b8 / 77b2fd3a...1e6`。
- StreetGS 真实 checkpoint SHA=`766648bf...cd1`：Background=`1,095,606`、RigidNodes=`172,264`、有效
  actor=`23`；第 97 帧所有支持字段通过 `atol=rtol=2e-6` 的表示等价。SceneIR package content SHA=
  `8b3dd8639d244a87afb864b8109a27273a59fa3f5eaa7861338915d6bde93671`。
- ReconDrive exact source=`d2bc397b...e3c`；不运行模型，仅用其 `get_recontrast_data` 标准输出 schema 的
  `4,096` primitive deterministic fixture 验证 static/actor split、camera、flow 与重组，content SHA=
  `8f50be22a21d0678842e4f98b3f4201d8ff21b26e58085a953795e85970f7aff`。两 package 独立新进程重载均通过。
- `quality_data_read=false / training=false / inference=false`；claim 仅限表示接口。next=
  `WS-V6-R3-SUPPORT-DEVIATION-01`。

## WorldSim V6 R1 capability audit 完成；R2 解锁（2026-08-21）

- canonical=`run://worldsim_v6/WS-V6-R1-FRONTEND-CAPABILITY-01/20260821T080610Z__r1-capability-s0-r1`，
  source=`d981df7fdde5458eb3878193c4a76f6dcf926ad4`，config SHA=`6d576728...dfb`，local capability
  SHA=`5b58e018...641`，matrix/summary/manifest/terminal SHA=
  `dd101206...957 / 2025b5f9...9c / 0c8b40a6...035 / 4ed40819...da7`。
- optimization：StreetGS=`executable`、AD-GS=`executable`，两者各有 `6` 场 exact checkpoint；CityGS 与
  LiHi-GS=`unavailable/not local`。feed-forward：ReconDrive=`adaptable`、TokenGS=`adaptable`、DGGT=
  `adaptable_from_frozen_outputs`；Instant NuRec=`audit_only`。
- ReconDrive exact checkout=`d2bc397b724d6cc021da22f8f57ad6af1cc53e3c`，公开 stage-2 checkpoint probe=
  `4,595,424,264 bytes / HF revision 64a40402...09d7`。当前 native 12 Hz input、local env 和 local weight
  缺失，因此 R1 只选择它作为 primary adapter target，不宣称 inference 已跑。TokenGS exact=
  `b16269c500a8894cda342bc9cf406e31169541e3`，需 driving camera/domain adapter。
- gate=`2 optimization executable AND 3 feed-forward executable/adaptable`，PASS；全程没有质量读取、训练或推理。
  首个 `20260821T081500Z__r1-capability-s0-r1` 为 dirty-source 非 canonical 实例，不参与 closeout，见 `V6-F05`。
- tests=`11 passed`；next=`WS-V6-R2-SCENEIR-V0-01`，先冻结 frontend-neutral schema、round-trip 与 negative tests。

## WorldSim V6 G3 分支与控制面初始化完成；R1 启动（2026-08-21）

- `WS-V6-G3-BRANCH-BOOTSTRAP-01=done`：从 `origin/main@e028c862da494d6fe85f6062eb231a80e9812978`
  创建并推送 `research/worldsim-v6-world-compiler`；分支初始提交与 parent main 完全相同。
- 首轮控制面已建立：运行计划、显式状态机、假设/反思追加账本、R1–R4 目录骨架，以及只在本机保存的
  capability 映射约束。初始化不读取质量数据、不训练、不推理，也不形成算法结论。
- next=`WS-V6-R1-FRONTEND-CAPABILITY-01`：冻结硬件/磁盘/环境/数据/checkpoint/third-party 能力快照，
  对候选前端给出 `runnable / repairable / audit_only / unavailable` 结论并以最小 smoke 验证。

## WorldSim V6 G0 完成 / G1–G3 治理前置注册（2026-08-21）

- `WS-V6-G0-REPO-CONVERGENCE-01=done`：完成 remote/fetch/branch/tag/submodule/LFS、ahead/behind 与 dirty status
  审计；origin main=`44d0e4a2468112b89a454992ecd9177d65184067`，authoring HEAD=
  `d95ce38731568bf5b79263d2996afdf39f9547c8`。唯一 untracked plan=`46,347 bytes / sha256
  e97572ad4ac7f38a8ba5ed48f03e54a0483581769c45799f5fe5738e764114c4`，已建立唯一 recovery ref 与仓库外恢复包；
  tracked/staged patch SHA 均为 empty-file SHA `e3b0c442...b855`。
- 首次 direct `git fetch --all --prune --tags` 因远端 HTTPS 无进展被安全中止；为当前 AutoDL 主机建立独立代理后，
  同一命令成功。分类=`engineering recovery`，没有 run/quality/candidate，`failure_ledger_delta=none`。
- `WS-V6-G1-DOCS-CLOSEOUT-01=done`：canonical commit/tree=
  `9bdabb3e249ce5a048a5a9b7b0ba8dc4774b3bb2 / c0af558fe4acfc524e84a67c9a540b7772cc1705`；在
  `research/worldsim-v5.1-m1` 登记 V5.2 terminal/V6 active、`V6-F01` 与本计划。JSON/link/failure-ID/三账本
  consistency、frozen V5.2 plan SHA regression 全 PASS，V5.2 frozen plan bytes 未改变，定向回归=`31 passed`。
- `WS-V6-G2-BRANCH-CONVERGENCE-01=done`：integration=`integration/pre-v6-20260821T072411Z`，audit=
  `166faa0ff977b403d46c030bb74c3a61dbfa788b`；main 从 `44d0e4a` 普通 fast-forward/push 到该 commit，rollback
  tag 与 integration remote 均已发布。完整 branch matrix、恢复与 gate 见下方 G2 小节。
- `WS-V6-G3-BRANCH-BOOTSTRAP-01=pending`：仅在 G2 gate 通过且 main 已 push 后创建并 push fresh V6 branch。
- G0–G3 只产生治理/仓库事实；V6 尚无 SceneIR、frontend、support、validity、verifier、bake 或方法质量结论，GPU
  method run=`0`。failure refs=`V6-F01, V52-F01, V52-F02, PIVOT-F02, PIVOT-F03, PIVOT-F04`。

### G2 integration gate 执行与恢复

- integration=`integration/pre-v6-20260821T072411Z`，rollback=`pre-v6-main-20260821T072411Z@44d0e4a`；
  branch matrix 证明 remote/local 历史项目分支均已被 main 或最新研究分支包含，唯一有效 merge 为
  `origin/research/worldsim-v5.1-m1@9ca03c7` 的 `147` 个 main-unique commits，执行 `--ff-only`，conflict=`0`。
- 首次裸 `pytest -q` 在 collection 因 repo root 未进入 console-script `sys.path` 得到 `12 errors / 0 tests run`；显式
  `PYTHONPATH=$PWD` 后完整执行为 `1445 passed / 1 skipped / 13 failed`。失败中 4 项是把冻结 DriveStudio runtime
  测试放到 motionproj interpreter，换 exact interpreter 后 `15 passed`；该组不是代码失败。
- 外部资产恢复：公开 Instant NuRec checkout 按 frozen commit/tree `1ce2288/96e36fa` 精确恢复，回归=`8 passed`。
  缺失 P2 checkpoint 由 immutable source checkpoint 重建为 `432,111,754 bytes / 7be87e8b...7448` exact；首个
  recovery r1 因相对 protocol path 在 snapshot 前 blocked，r2 使用绝对路径成功。P3 `158` payloads 重建后逐项匹配
  旧 manifest，原路径只补缺失字节、不覆盖 manifest；R0/P3 回归=`23 passed`。
- V5.1 protocol 原 allowlist 只覆盖 P0 与 Stage-B hash，未覆盖 `a9dede0` terminal closeout hash `a0e764f3...fe1d`；
  仅新增 exact terminal hash 常量和回归，保持旧配置 hash 不变，`tests/test_worldsim_v51_protocol.py=9 passed`。
- 最终 gate：motionproj profile 排除 4 个专用-runtime test files 后=`1443 passed / 1 skipped`；DriveStudio profile=
  `15 passed`；总计=`1458 passed / 1 skipped`。config=`284 YAML + 1 JSON` 全解析，CLI/import smoke、
  `git diff --check`、failure definition duplicate=`0`、V5.1/V5.2 plan hash 与 P2/P3 asset identity 全 PASS。
- G2 尚未产生方法质量结论，GPU method run=`0`；failure refs=`V6-F01, V6-F02, V6-F03, V6-F04, V51-F01,
  PIVOT-F24, PIVOT-F29, PIVOT-F30`，failure delta=`V6-F02/V6-F03/V6-F04`。

## V5.2.1 P11 Human Attribution PASS / V5.2 M123 Autoresearch 注册（2026-08-20）

- P11 task=`WS-V521-P11-HUMAN-ATTRIBUTION-01`，run=
  `/root/autodl-tmp/runs/worldsim_v521/20260820T130000Z__p11-human-review-attribution-s0-r001`，source=
  `259958c4b773762e38d40ba7617c61b7425080ad`，status=`done`，outcome=
  `human_attribution_and_backtest_denominator_frozen`。
- 输入 exact SHA：review cases=`5f43792e...04f`、P10 badcase registry=`29d1ca2b...68bb`、matched frame registry=
  `a5680323...1389`、人工 annotation config=`02c72ed9...43a7`；输出 cases/backtest contract=
  `d89f4a4b...381f /8149ecb9...872f`。
- denominator=`18`：`9 BASE_FAILURE + 8 M123_ELIGIBLE + 1 ATTRIBUTION_UNRESOLVED`；eligible=
  `5 Discovery design + 3 one-shot Confirmation`，全部为 StreetGS/nuscenes。人工 verdict 是视觉诊断假设，不构成 M1/M3
  因果证明；P10 registry、threshold、predicate、K、split 和 panel selection 均未改变。
- V5.2 autoresearch task family=`WS-V52-R0..R7` 已注册为 `pending`：R0 freeze → R1 Base Validity/causal bridge → R2 M1
  TrackBayes → R3 M3 warp bridge/replay → R4 M2 safety → R5 factorial fusion → R6 one-shot Confirmation → R7 closeout。
  用户授权最长 `12 h` unattended；无新人工 verdict 时不得由 Agent 代填。
- fresh validation/test/KITTI quality=`false`，Stage H/BKI=`false`，本条无算法训练或新 candidate；failure delta=
  `V52-F01/V52-F02`。

## V5.2.1 P9 one-shot Confirmation / P10 closeout PASS（2026-08-20）

- P9 run=`20260820T103000Z__p9-one-shot-confirmation-s0-r001`；在 quality decode 前冻结 taxonomy、P5、M123、P4
  exact SHA，protocol freeze SHA=`b70caa0f...76cf`。两套基座 × 6 scenes 的 `12/12` renderer audits 通过，
  Confirmation=`126 views/base`，base/actor/temporal rows=`252/252/36`，checkpoint hash gate=`true`。
- Confirmation 未 refit threshold、未改 K、未改 predicate；AD-GS global/actor/boundary=`28/9/5`，StreetGS=
  `25/11/5`，六个 base×class direction 均 confirmed。registry=`63` unique view cases，automatic panels=`47`，
  verdict/summary SHA=`8f5ca467...2b61 /33cbc03c...40ab`。
- P10 run=`20260820T113000Z__p10-base-badcase-closeout-s0-r001`，outcome=`v521_base_badcase_basis_frozen`，
  coverage=`complete_full`。最终 registry=`377` rows，panel registry=`124` rows；summary/decision/manifest SHA=
  `f3e61102...056f /392c9dbb...8135 /bcbccded...8ebc`。
- P6–P8 re-audit 结论保持 evidence-insufficient；最终 `ready_for_v522_algorithm_design=false`，next stage=
  `exact base/M1 overlap + actor identity/visibility/depth/correspondence evidence alignment`，算法 candidate=`0`。
- hash-first cleanup 删除 Discovery `1,728 files /606,607,441 bytes` 与 Confirmation
  `378 files /119,392,106 bytes` 的未入选 prediction、重复 GT 和 staging；保留全部 canonical metrics、panels、audits，
  recoverability=`regenerable_from_frozen_checkpoint_data_split_and_renderer`。
- fresh validation/test/KITTI=`false`，Stage H/BKI=`false`，training/algorithm modification/threshold search=`false`；
  完整 V5.2.1 定向测试=`28 passed`，failure delta=`none`。

## V5.2.1 P2–P8 Discovery census / taxonomy / re-audit PASS（2026-08-20）
- P2 run=`20260820T091500Z__p2-discovery-base-census-s0-r001`：AD-GS/StreetGS × 6 scenes 全部 exact 重渲染，
  Discovery=`576 views/base`，base/actor/temporal rows=`1,152/1,152/918`；LPIPS=`Alex`，V4 evaluator/mask source SHA=
  `5dcbb6e9...731 /62158cc3...e8c`。checkpoint、target、finite、same-input replay gates 全 PASS；Confirmation decoded=`0`。
- P3 frozen thresholds 使用 per-scene q10 后跨场等权 median；global/actor/boundary 独立 predicate 与 leaderboard，
  scalar score=false。registry=`314`，panel union=`88`，taxonomy/registry/leaderboard SHA=
  `8f494101...75ef /7ed78eb7...a23 /9a46b762...110`。
- P4=`77` automatic labeled panels；P5 full-denominator scene-balanced case-rate mean AD-GS/StreetGS=
  `0.285113/0.251939`，bootstrap CI95=`[0.076392,0.563973]/[0.124895,0.430075]`。
  actor/static MSE ratio median=`1.403697/1.921242`；boundary/actor MSE ratio median=`0.716317/0.725970`。
- legal undefined axes：comparable depth=`none`，visibility/occlusion/distance/speed/LiDAR denominator=`0/576`，
  temporal=`459 proxy windows/base but no B-TEMPORAL`，actor instance=`undefined_no_instance_region`。
- P6–P8：M1 exact overlap scenes=`0<2` → `M1_EVIDENCE_INSUFFICIENT_KEEP_PENDING`；M2 exact request mapping=`0` →
  `M2_EVIDENCE_INSUFFICIENT`；M3 新 B-TEMPORAL denominator=`0` → `M3_EVIDENCE_INSUFFICIENT_KEEP_PENDING`。
  V4 historical M3 confirmation 与 V5 constraint-projection rejection 均原样保留，不互相倒写。
- 下一步仅 P9 one-shot internal Confirmation；threshold/K/predicate refit=false，fresh validation/test/KITTI=false，
  Stage H/BKI=false，algorithm modification=false。failure delta=`none`。


## V5.2.1 P1 exact base asset / shared split census PASS（2026-08-20）
- task=`WS-V521-P1-BASE-ASSET-CENSUS-01`，status=`done`，outcome=`p1_gate_pass`；canonical run=
  `/root/autodl-tmp/runs/worldsim_v521/20260820T084604Z__p1-base-asset-census-s0-r001`，source=`127b216`。
- exact matched assets：AD-GS 六个 checkpoint bundle 与 StreetGS 六个 final checkpoint 全部命中既有冻结 hash；
  scene set=`0048/0139/0230/0242/0255/0994`，matched denominator=`234 canonical samples / 702 three-camera views`，
  coverage candidate=`complete_full`。
- quality-blind split 已冻结：Discovery=`192 samples / 576 views`，Confirmation=`42 samples / 126 views`；
  `BASE_ASSET_REGISTRY.json` SHA=`5afb4377...bcc4`，`DISCOVERY_CONFIRMATION_FREEZE.json` SHA=`3003f98e...a7345`，
  `MATCHED_FRAME_REGISTRY.jsonl` SHA=`a5680323...1389`。quality bytes decoded=`0`。
- historical-only audit：V1 AD-GS 六场 native checkpoint/render 为 `MISSING_BUT_MANIFESTED`，形成六条资产 blocker，
  不以重训恢复；StreetGS 旧 stride-10 六场为 `PROTOCOL_MISMATCH`。两者均不污染当前 exact matched denominator。
- 下一步只允许 P2 Discovery-only 串行重渲染/统一指标；Confirmation、fresh validation/test/KITTI、Stage H/BKI、
  algorithm modification、training、threshold search 均保持 false。failure delta=`none`。


## V5.2.1 P0 protocol/provenance/resource freeze PASS（2026-08-20）

- task=`WS-V521-P0-BASE-CENSUS-FREEZE-01`，status=`done`，outcome=`p0_gate_pass`；canonical run=
  `20260820T083826Z__p0-base-census-freeze-s0-r001`，source HEAD=`831fafe302fef7ff2760ff17c1e51da91b9be03c`，
  branch=`research/worldsim-v5.2.1-base-badcase-census`。计划 exact=`54,123 bytes /ce332dea...3c56`。
- 研究问题：先建立 AD-GS/StreetGS 的 exact base asset/census denominator、frame/camera/actor/window failure taxonomy 与
  deterministic `BC-*`/`BCE-*` registry，再判断 M1 ownership 是否独立于 base RGB/geometry badcase，以及 M2/M3 应保留为
  core、downstream 还是 evidence-insufficient；Top-K panel 不进入 prevalence 分母。
- split/quality lock：共享 canonical sample hash 不消费 base/camera/actor；P1 后、P2 前冻结具体 Discovery/Confirmation
  membership；P2 只读 Discovery，P9 才 one-shot 读取 internal Confirmation。fresh validation/test/KITTI quality=false，
  Stage H/BKI=false，algorithm modification/training/threshold search=false。
- provenance/resource gate 全过：clean Git worktree；RTX 3090=`24,576 MiB/used 1 MiB`，disk free=`75 GiB`，
  cgroup memory.max=`96,636,764,160 bytes`。P0 quality bytes decoded=`0`；下一步只执行 P1 metadata-only asset/split census。
- failure refs=`V1-F01/F03, V4-F34/F39/F42/F45/F47/F49, V5-F07/F31-F33/F47/F48/F51/F52/F57/F59,
  V51-F31/F37/F42/F63/F65/F66`；failure delta=`none`。

## V5.1 M1 closeout / Stage H superseded（2026-08-20）

- task=`WS-V51-M1-CLOSEOUT-01`，status=`done`，outcome=`closed_without_promoted_candidate`；这是对既有 r015/r018/
  r022/r043/r047 的只读聚合与治理收口，没有新方法 run、没有 quality reread，也没有启动 BKI。
- Stage H=`pending`、executed=`false`、disposition=`superseded_by_v5.2_scope`。`low_expected_gain` 是因为 BKI 仍将已有
  局部证据通过空间 kernel 补全未观测节点，没有改变 evidence source；它不是 BKI empirical reject。
- empirical conclusion=`effective_observations_are_structurally_missing`；V5.2 只有在新增独立观测源并先通过 coverage、
  identity persistence、fresh-process determinism 后，才可重新讨论 propagation/completion。
- freeze=`configs/worldsim_v51/m1_closeout_v1.yaml`；archive=`docs/archive/2026-08/worldsim-v51-m1-closeout/`；
  failure refs=`V51-F31/F37/F42/F63/F65`，delta=`V51-F66`。validation/test/KITTI/training=false。
- cleanup 采用 fixed inventory 串行删除 `156 targets /1,810 files /16,864,370 bytes` 的未跟踪编辑备份与可再生 cache；
  unsafe=`0`、remaining=`0`，canonical runs/checkpoints/data/third-party source 未触及。完整逐目标记录见 archive。

## V5.1 Stage G G0b r047 REJECT / audit / route closeout（2026-08-18）

- `8 fresh processes /16 alpha calls` 得到 `0.0056084292` 与 `0.0267562941` 两个 vector，违反 unique≤1；hard
  16 次 exact，其他 protocol/resource gates PASS。summary=`2,230 bytes /679b96e3...0b4`。
- audit=`3,880 bytes /98c72ba7...d31 /PASS`；failure=`V51-F65 resolved-as-rejection`。不 patch 上游、不读真实数据/
  质量；Trace3D faithful route closed。预注册 failover 原指向 `WS-V51-M1-H-GRAPHFREE-01`，但 Stage H 后续未执行并在
  V5.1 closeout 中保持 `pending`、由 V5.2 scope 取代。

## V5.1 Stage G G0b r047 cross-process determinism 预注册（2026-08-18）

- exact same synthetic input/extension，`8 fresh processes × (hard 3 + alpha 2)`；hard/alpha unique vector count 均须为 1，
  另锁 label response、finite/bounded 与 input immutable。source plain `+=` hazard 只审计，不 patch。
- PASS 才允许 real adapter preflight；FAIL 直接以 faithful Trace3D operator rejected 收口并转 BKI/graph-free；不读质量。

## V5.1 Stage G G0a r046 PASS / audit / determinism observation（2026-08-18）

- exact upstream build 与 synthetic class-response/repeat/input-immutability gates PASS；wheel/extension=
  `471786c6...287 /f81ef6d6...f53`。audit=`3,149 bytes /ecc3d061...81c /PASS`。
- alpha-weight original/audit=`0.0267562941 /0.0056084292`，出现跨进程漂移；新增 `V51-F65`，不事后改 r046 gate，
  但暂停真实 adapter。下一步为 frozen multi-process determinism forensic；失败即关闭 Trace3D 并转 graph-free。

## V5.1 Stage G G0a r046 reverse-tracing build/capability 预注册（2026-08-18）

- exact official `diff-id-rasterization`（1,490 tracked entries）在 cu118 DriveStudio 环境中 `--no-index` 隔离构建，
  source patch/environment mutation/submodule init=false；target 采用 `.partial → atomic rename`。
- synthetic-only gate=`1 Gaussian + 32×32 label-0/label-1`，检查 background/foreground 响应、repeat bitwise、alpha bound、
  input immutability。real U2/B3/checkpoint/camera/image/mask/quality=false；PASS 不能登记方法质量。

## V5.1 Stage G G0 r045 PASS / audit / freeze（2026-08-18）

- run/source=`20260818T220000Z__m1-stage-g-g0-trace3d-source-recovery-s20260814-r045 /ed39600...5651`；PDF
  page markers=`11`，repo commit/tree/clean 与 5 组 method source marker/hash 全 PASS；source execution/quality read=false。
- summary=`6,491 bytes /5e15c3df...b13`；audit=`2,311 bytes /053cf574...d3b /PASS`；failure `V51-F64`
  resolved。freeze=`stage_g_g0_trace3d_source_method_preflight_freeze_v1.yaml`；下一步只预注册 frozen-base operator probe。

## V5.1 Stage G G0 r045 exact-asset recovery 预注册（2026-08-18）

- run=`20260818T220000Z__m1-stage-g-g0-trace3d-source-recovery-s20260814-r045`；复用 r044 已原子发布的 exact
  paper/repo，禁止网络、重下、删除、source execution、submodule init 与质量读取。
- only change=`pdfinfo → Python stdlib /Type Page marker count`；同时按 exact file/marker/hash 审计 tracing CUDA、patch
  merge/repair、ambiguous split/prune 与普通 density control 的源码位置。PASS 仅进入 frozen-base operator preflight。

## V5.1 Stage G G0 r044 BLOCKED：pdfinfo 缺失（2026-08-18）

- official paper/repo atomic publish 已完成，随后 page-count 调用因 `pdfinfo` 不存在中断；run terminal 未写完，保留
  running evidence，不冒充 done。paper=`2,390,825/d50eda07...47e4`；repo=`7465ad94...c442/22d30d19...a05d/clean`。
- failure=`V51-F64`；v2 仅复用 exact assets 并改标准库 page marker count，不执行源码或读质量。

## V5.1 Stage G G0 r044 Trace3D source/method preflight 预注册（2026-08-18）

- target=`trace-3d/Trace3D@7465ad94...c442 + official ICCV 2025 PDF`；只 acquisition/hash/source semantics/
  dependency/submodule audit，source execution/model download/image-mask-quality reads=false。
- faithful method 包含 GIT reverse tracing、patch merge 与 ambiguous-Gaussian split/prune；immutable-base 首门只准 capability+
  no-quality disagreement diagnostic，明确不是 full training reproduction。

## V5.1 Stage F F0l r043 REJECT / audit / freeze（2026-08-18）

- run/source/tree=`20260818T200000Z__m1-stage-f-f0l-quality-alignment-s20260814-r043 /70293de...e4f0 /
  d1d6ddfb...349c`；scene pass=`0471 FAIL /1087 PASS /0379 FAIL`，all-scenes contract rejected。
- coverage=`0.122784/0.859091/0.238278`；one-to-one recall=`0.080747/0.505009/0.202933`；
  persistence=`0/0.5/0`。audit=`4,210 bytes /f478fbd9...4320 /PASS`；threshold search=false。
- freeze=`stage_f_f0l_train_only_quality_identity_alignment_freeze_v1.yaml`；failure=`V51-F63`。Gaussian Grouping
  F1/F2/training stopped；下一步仅 Trace3D official source/method/adapter preflight。

## V5.1 Stage F F0l r043 train-only quality/alignment 预注册（2026-08-18）

- run=`20260818T200000Z__m1-stage-f-f0l-quality-alignment-s20260814-r043`；exact read denominator=`45 candidate masks +
  45 dynamic weak-support masks`，RGB=0。scene-local maximum-weight one-to-one track↔short-ID assignment。
- per-scene gates=`tracks/views≥1/2, coverage≥0.70, assigned recall≥0.35, efficiency≥0.75, persistence≥0.50`；
  all-three PASS 才解锁 training smoke，FAIL 自动拒绝 Gaussian Grouping 并转 Trace3D；不做 threshold search。

## V5.1 Stage F F0k r042 input denominator PASS / audit / freeze（2026-08-18）

- run/source/tree=`20260818T190000Z__m1-stage-f-f0k-quality-input-freeze-s20260814-r042 /858e0d1...64f0c /
  8df9f291...996d`；45 views、90 projections、8,111,447 verified asset bytes；mask/image/quality reads=false。
- input/summary/audit SHA=`6640b5e1...6817/6d5a1ffd...b341/108bb60a...082b`；freeze=
  `stage_f_f0k_quality_alignment_input_freeze_freeze_v1.yaml`。下一步只预注册 F0l frozen-threshold execution。

## V5.1 Stage F F0k r042 quality/alignment input freeze 预注册（2026-08-18）

- run=`20260818T190000Z__m1-stage-f-f0k-quality-input-freeze-s20260814-r042`；只 hash/投影，不读 image/candidate/dynamic
  mask pixels。冻结 exact 45-view candidate/reference/camera/metadata 与 eligible projected actor denominator。
- weak-reference 只定义 actor foreground support；identity 由 scene-local 3D track 对 DEVA short ID 的一对一匹配评估。
  thresholds 在 mask read 前固定为 per-scene coverage/assignment-recall/efficiency/persistence=`0.70/0.35/0.75/0.50`。

## V5.1 Stage F F0j r041 full materialization PASS / audit / freeze（2026-08-18）

- run/source/tree=`20260818T180000Z__m1-stage-f-f0j-fresh-45-view-recovery-s20260814-r041 /27dfaa8...150a /
  9f433d16...abd2`；三 scene fresh success，`45 masks +3 pred`，18 次 pre-matmul empty-cache evidence 全通过。
- resolved/summary/materialization/manifest/status/events SHA=`3db1a122...3091/f3ee3ad1...c183/32b5d8d3...1b7f/
  551e42e5...e733/1dda9eb4...b763/2d76e728...f0b0`；output chain=`f1c1b44e...1d3`。
- audit=`18,462 bytes /acd5a91b...31d2 /PASS`；resources=`24,118 MiB peak /458 MiB headroom /
  17,981,091,840 cgroup /104.676977s /324 samples /0 errors`。freeze=`stage_f_f0j_fresh_45_view_empty_cache_
  materialization_freeze_v1.yaml`；V51-F62 execution-only resolved、root cause unproven；质量与 identity alignment 未读。

## V5.1 Stage F F0j r041 fresh 45-view recovery 预注册（2026-08-18）

- run=`20260818T180000Z__m1-stage-f-f0j-fresh-45-view-recovery-s20260814-r041`；auth=r040 freeze
  `ac390557...baca`。fresh exact 45 inputs，scene-local `0471→1087→0379` 三进程串行，禁止 r035 partial reuse。
- method=`grid32/batch64/AMP + pre-line58-matmul empty-cache`；pass=`45 schema masks +3 pred +every intervention
  evidence +output chain +resources`。本轮不读 mask nonzero/quality/identity alignment，PASS 后只预注册相应质量门。

## V5.1 Stage F F0i r040 scene-1087 recovery PASS / audit / freeze（2026-08-18）

- run/source/tree=`20260818T170000Z__m1-stage-f-f0i-scene1087-recovery-s20260814-r040 /9c8c503...9003 /
  ff47087a...965e`；exact 15-view subprocess success，输出 `15 masks +pred.json`，6 次 pre-matmul empty-cache 全有证据。
- resolved/summary/manifest/status/events SHA=`2d6f23f4...1a66/312a0277...a65/171f4fc5...2fcf/4cd8f7cc...fe4/
  bd35805a...1cb`；audit=`9,254 bytes /1393c664...67c /PASS`。
- resources=`24,118 MiB peak /458 MiB headroom /17,966,829,568 cgroup /29.030242s /89 samples /0 errors`；
  freeze=`stage_f_f0i_scene1087_15_view_empty_cache_recovery_freeze_v1.yaml`。failure delta=`V51-F62 recovery_
  candidate_pass_scene1087_15_view_full_materialization_not_yet_resolved`；下一步只预注册 fresh 45-view recovery。

## V5.1 Stage F F0i r040 scene-1087 15-view recovery 预注册（2026-08-18）

- run=`20260818T170000Z__m1-stage-f-f0i-scene1087-recovery-s20260814-r040`；exact 15 views，单 fresh process，
  grid32/batch64/AMP + line58 empty-cache。pass=`15 schema masks +pred +all intervention evidence +resources`；quality/
  full-materialization/training=false，PASS 后只允许 fresh 45-view recovery preregistration。

## V5.1 Stage F F0h r039 recovery PASS / audit / freeze（2026-08-18）

- run/source/tree=`20260818T160000Z__m1-stage-f-f0h-empty-cache-parity-s20260814-r039 /ba2f24f...a6b7 /
  64b4c992...a8e5`；outcome=`recovery_pass`。resolved/summary/manifest/status/events SHA=`d09d8787...4d7a/
  d720af4e...9505/28d8ce61...411a/0f9585ec...d8f8/97ddc77a...d6ed`。
- classes=`success×4`，empty-cache checks/reference checks=`true×4`，control/target pair exact=true；8 次 cache release
  由 trace 重放。audit=`8,625 bytes /fda57ee4...88ab /PASS`；resources=`24,118 MiB peak /458 MiB headroom /
  17,972,154,368 cgroup /96.057126s /301 samples /0 errors`。
- freeze=`stage_f_f0h_pre_matmul_empty_cache_parity_freeze_v1.yaml`；failure delta=`V51-F62 recovery_candidate_pass_
  full_materialization_not_yet_resolved`。下一步只做 1087 15-view recovery，不读质量。

## V5.1 Stage F F0h r039 empty-cache execution parity 预注册（2026-08-18）

- run=`20260818T160000Z__m1-stage-f-f0h-empty-cache-parity-s20260814-r039`；config=
  `configs/worldsim_v51/stage_f_f0h_pre_matmul_empty_cache_parity_v1.yaml`，auth=r038 freeze `ff3e8692...03ca`。
- intervention=`torch.cuda.empty_cache immediately before each line58 matmul`；source/tensor/operator/method unchanged。
  A–B–A–B 四进程分别是 control/target 双 repeat；trace 必须逐 matmul 落盘 before/after allocator evidence。
- pass 要求四臂 success、control/target 各 pair exact 且 exact 对齐既有 success hashes、资源全门 PASS。nonexact 或任一
  CUBLAS failure 都拒绝；不读 mask 内容质量，PASS 后也只做 1087 15-view recovery preregistration。

## V5.1 Stage F F0g r038 both-success trace / audit / freeze（2026-08-18）

- run/source/tree=`20260818T150000Z__m1-stage-f-f0g-tensor-trace-s20260814-r038 /da2169d...a5f3 /
  9a00a267...50a0`；resolved/summary/manifest/status/events SHA=`2124e720...43d6/e9db6152...8f46/ebcdfba1...c339/
  1cc45c03...8d54/a07ca061...6593`，outcome=`both_success`。
- control calls=`objects 26/36，free 36,765,696/57,737,216 bytes，allocator retries 0/0`；target calls=`objects 3/52，
  free 19,494,141,952/18,502,189,056 bytes，retries 1/1`；affinity 固定 `[1,1620,1620]`。target first smaller，
  排除“首个 target matmul 单纯更大”解释；cache/workspace 仅 hypothesis，不是 root-cause proof。
- trace SHA=`control 4e556703...99a2 /target cd80c179...5492`；outputs 与既有 success exact。audit=`16,025 bytes /
  a8cbdb5b...4047 /PASS`，manifest=`25 entries /1,284,692 logical /247,703 regular`；resources=`24,118 MiB peak /
  458 MiB headroom /48.697596s /151 samples /0 errors`。
- freeze=`stage_f_f0g_target_tensor_allocator_instrumentation_freeze_v1.yaml`；failure delta=`V51-F62 refined_allocator_
  cache_workspace_hypothesis_not_root_cause_proof`。下一门为 empty-cache execution parity，不读质量、不直接 materialize。

## V5.1 Stage F F0g r038 source-neutral tensor/allocator trace 预注册（2026-08-18）

- run=`20260818T150000Z__m1-stage-f-f0g-tensor-trace-s20260814-r038`；config=
  `configs/worldsim_v51/stage_f_f0g_target_tensor_allocator_instrumentation_v1.yaml`，auth=r037 freeze
  `18fd02be...55b7`。
- arms=`0471 temporal control_trace →1087 cross-camera target_trace`；官方参数与 `CUDA_LAUNCH_BLOCKING=1` 不变。
  trace hook 仅在 upstream line58 pre-matmul、line59 post-matmul 或 exception 读取 tensor metadata 与 allocator counters；
  frozen traced-file SHA=`a1b86e65...c5a6`。
- 明确禁止 source edit、operator monkeypatch、tensor-content read、quality/alignment/full-materialization/training。成功输出最多
  schema-read 6 masks；trace 只用于判断 shape/memory mechanism，不作为算法质量证据。

## V5.1 Stage F F0f r037 control-stable/target-failure / audit / freeze（2026-08-18）

- run/source/tree=`20260818T140000Z__m1-stage-f-f0f-runtime-repro-s20260814-r037 /3c692f4...cc3 /
  ad8d798b...89c`；outcome=`control_stable_target_failure`。resolved/summary/manifest/status/events SHA=
  `1d29728a...04d/5fd4a4e8...df8/6b157621...26d/d66df2b8...e1bf/4184d2dd...7aab`。
- A1/A2 control 均成功且 mask/pred exact，与 r034 identity 一致；B1/B2 target 均 CUBLAS expected failure、非显式 OOM、
  0 partial output。pair checks=`control success/exact true/true，target success/exact false/false`。
- health=`NVIDIA identity+ECC/page/row command rc0；ECC fields N/A；dmesg permission-denied rc1`，没有 reset/mutation；
  resources=`24,124 MiB peak /452 MiB headroom /17,969,901,568 cgroup /86.805117s /271 samples /0 errors`。
- audit=`8,245 bytes /2fb76f32...d50d /PASS`，manifest=`33 entries /1,217,783 logical /180,794 regular`；freeze=
  `stage_f_f0f_cuda_runtime_health_reproducibility_freeze_v1.yaml`。failure delta=`V51-F62 refined_control_stable_
  target_path_unstable`；下一步只做 source-neutral trace instrumentation，不读质量、不恢复 materialization。

## V5.1 Stage F F0f r037 runtime control-target reproducibility 预注册（2026-08-18）

- run=`20260818T140000Z__m1-stage-f-f0f-runtime-repro-s20260814-r037`；config=
  `configs/worldsim_v51/stage_f_f0f_cuda_runtime_health_reproducibility_v1.yaml`，authorization=r036 freeze
  `207b28f5...b62f`。
- A–B–A–B=`0471 temporal same-camera control →1087 cross-camera target →control repeat →target repeat`；全部 official
  grid32/upstream-batch64/AMP/size480/semionline，`CUDA_LAUNCH_BLOCKING=1`，fresh subprocess 串行。control/target 各自
  必须 success pair mask+pred bit-exact；CUBLAS expected failure signature 沿用 r036。
- 只读 health probes=`nvidia identity/temp/P-state +ECC/page/row +dmesg access`；不做 GPU reset/driver mutation。
  预注册 outcome 区分全局 runtime failure、control 稳而 target 不稳、全部稳定、success nonexact；不按 mask 内容选结果。
- input decode=`12`、output schema reads<=`12`，quality/alignment/materialization/training/downstream=false；资源上限沿用
  `peak 24,320 MiB /headroom 256 MiB /cgroup 60 GiB /900s`。

## V5.1 Stage F F0e r036 mixed / audit / freeze（2026-08-18）

- run/source/tree=`20260818T130000Z__m1-stage-f-f0e-cuda-localization-s20260814-r036 /223f943...6e0 /
  bb8166df...7ae`；outcome=`mixed`，conclusion=`cuda_launch_blocking_replays_mixed_r035_fault_is_not_deterministic_
  under_current_probe`。resolved/summary/manifest/status/events SHA=`8a09a0a5...801/32e59c85...3ea/7af30c35...704/
  db5366e8...b3d/b975c035...7af`。
- exact replay1=`expected_cublas_internal_failure`，returncode1，0 mask/pred，非显式 OOM；replay2=`success`，3 个
  schema-valid mask SHA=`6d679a37...e1c/c9768bbd...f02/1d46fa81...a6d`，pred=`10d55216...650`。两臂只差新进程
  次序，无输入/方法/diagnostic 参数差异，因此当前 probe 不支持 deterministic data/shape failure。
- resources=`peak/headroom 24,124/452 MiB /cgroup 17,964,371,968 bytes /43.106708s /134 samples /0 errors`；
  audit=`5,077 bytes /ec7cfa36...34f6 /PASS`，manifest=`16 entries /795,861 logical /72,116 regular-excluding-input`。
- failure delta=`V51-F62 refined_as_nondeterministic_under_blocking_probe`；freeze=
  `stage_f_f0e_scene1087_cuda_fault_localization_freeze_v1.yaml`。下一步是 runtime health/reproducibility gate；仍不读质量、
  不缩 batch、不恢复 full materialization。

## V5.1 Stage F F0e r036 scene-1087 CUDA fault localization 预注册（2026-08-18）

- formal target=`20260818T130000Z__m1-stage-f-f0e-cuda-localization-s20260814-r036`；config=
  `configs/worldsim_v51/stage_f_f0e_scene1087_cuda_fault_localization_v1.yaml`。authorization=r035 closeout
  `3cb38341...93da`，只针对 `V51-F62`，不改变 r035 terminal。
- exact inputs=`scene-1087/index827/frame0/camera0,1,2`；两次 `cuda_launch_blocking_replay_1/2` 串行且均为
  grid32/batch64/AMP/size480/semionline。唯一 intervention=`CUDA_LAUNCH_BLOCKING=1`，用于把异步 CUDA 错误定位到同步调用；
  source/assets/runtime/thresholds 与 r035 相同。
- 预注册四分支 outcome，不按成功输出质量选 recovery；success 只检查 `900×1600 uint8` schema、文件 hash 和两次 exact，
  expected failure 必须同时命中 `consensus_associated.py:58 /value @ affinity /CUBLAS_STATUS_INTERNAL_ERROR /
  cublasGemmStridedBatchedExFix` 且不得发布 partial mask/pred。
- input decode denominator=`6`，output schema reads<=`6`；quality/identity alignment/materialization/training/downstream
  全 false，禁止 smaller-batch retry。resources=`24,576 total /256 headroom /24,320 peak MiB /60 GiB cgroup /600s`。

## V5.1 Stage F F0d r035 BLOCKED / independent audit（2026-08-18）

- run/source/tree=`20260818T120000Z__m1-stage-f-f0d-train-materialization-s20260814-r035 /e4d64d3...1424 /
  25590428...8a2`；status=`blocked`，结论=`scene0471_materialized_scene1087_cublas_internal_blocked_
  scene0379_not_started`。resolved/status/events/resource-samples SHA=`04cc8045...b9b/c3f917bd...f61/
  7d2221b5...0b7/d46d632d...4e2`。
- 0471=`15 masks /14 non-empty /1 zero /6,527,167 nonzero pixels /16 stable IDs`，report/pred SHA=
  `80778cba...85a/55767ccb...653`；1087 在进度 `2/15` 后于 three-frame spatial-alignment half-precision batched
  GEMM 报 `CUBLAS_STATUS_INTERNAL_ERROR`，没有 mask/pred/report；0379 未启动。完整分母仅 `15/45 masks +1/3 pred`，
  不发布 summary/manifest/materialization-manifest/resources，不把 partial 当 full materialization。
- sampled resources=`GPU peak 24,124 MiB /452 MiB headroom /cgroup 17,961,271,296 bytes /174 samples /
  0 errors`，event wall=`55.993866s`；数值仍在预注册资源门内，故 `root_cause_confirmed=false`、
  `explicit_pytorch_oom=false`。stderr/stdout SHA=`f626efc6...8a5/2e18d212...c74`。
- independent audit=`20260818T123000Z__stage-f-f0d-r035-audit.json /25,311 bytes /6d217a7e...13e1 /PASS`；
  audit PASS 只表示 blocked 边界可重放。failure delta=`V51-F62 active`；下一步仅预注册 exact 1087 首三视图
  `CUDA_LAUNCH_BLOCKING=1` fault localization，method 仍 grid32/upstream-batch64/AMP，不读质量、不缩 batch、不续写 r035。

## V5.1 Stage F F0d r035 45-view train-only materialization 预注册（2026-08-18）

- task/run=`WS-V51-M1-F-IDENTITY-EMBEDDING-01 /20260818T120000Z__m1-stage-f-f0d-train-materialization-s20260814-r035`；
  config=`configs/worldsim_v51/stage_f_f0d_train_only_identity_mask_materialization_v1.yaml`，seed=`20260814`。
  r034 freeze=`3,841 bytes /058b22d4...fb9`；r026 input=`45 records /7,530,010 bytes /653941ec...7a4 /
  chain b3458c27...4d95`。
- scenes=`0471(index382)→1087(827)→0379(296)`，各 `5 frames×3 cameras=15` views，三个 official subprocess 串行；
  scene-local input/output 目录防止 15 种重复 filename 跨场覆盖，并隔离 short-ID namespace。method 固定 grid/batch=
  `32/64`，不允许 smaller-batch retry。
- pass requires exact `45 masks +3 pred.json`、每场 non-empty/stable-ID、output record chain、GPU headroom/resource gates；
  logical decode/read=`45/45`。本轮 quality/actor-alignment/training/heldout/downstream 均 false；failure delta=
  `none_at_preregistration`。PASS 后也只预注册 train-only mask quality/identity alignment。

## V5.1 Stage F F0c r034 canonical PASS / freeze（2026-08-18）

- run/source/tree=`20260818T110000Z__m1-stage-f-f0c-upstream-batch-s20260814-r034 /27b1958...b48 /
  552ba018...58e`；grid/batch=`32/64`，primary/repeat mask/pred exact，association/non-empty/stable-ID gates PASS。
  resolved/summary/manifest/status/events SHA=`bb5e6b34...99e/613f76ce...c30/a3efdb80...132/0f0dae9c...bc6/
  d6941555...deb`；manifest=`23 entries /483,584 logical /170,340 regular-excluding-input-target bytes`。
- mask SHA=`cbfc00d5...226/c11db011...f18/7ffe5683...593`，metadata=`f5491453...156`，nonzero=
  `1,183,290/1,257,333/954,829`，stable IDs=`19`。resource=`GPU peak 24,092 MiB /484 MiB headroom /
  cgroup 17,957,322,752 bytes /45.917s /142 samples /0 errors`；全部 prereg checks true。
- audit=`20260818T113000Z__stage-f-f0c-r034-audit.json /4,243 bytes /e0988f50...5258 /PASS`；freeze=
  `stage_f_f0c_upstream_batch_association_repeatability_freeze_v1.yaml`。failure delta=
  `V51-F60 resolved_for_f0c_with_batch_sensitivity_boundary / V51-F61 resolved_by_r034_headroom`；quality 和所有 downstream/
  heldout locks 仍 false，下一步只预注册 45-view train-only materialization。

## V5.1 Stage F F0c r034 upstream batch64 recovery 预注册（2026-08-18）

- task/run=`WS-V51-M1-F-IDENTITY-EMBEDDING-01 /20260818T110000Z__m1-stage-f-f0c-upstream-batch-s20260814-r034`；
  config=`configs/worldsim_v51/stage_f_f0c_upstream_batch_association_repeatability_v1.yaml`，seed=`20260814`，
  split=`train_only`。authorization=`stage_f_f0b_association_parity_closeout_v1.yaml /3,289 bytes /f51ee482...0ba9`。
- arms=`grid32/batch64 primary → grid32/batch64 repeat`；batch64 是 upstream default，禁止 smaller-batch retry。
  输入仍是 r026 manifest 的 scene-0471/index382/camera0 frame0/40/80；pass requires mask/pred exact repeat、non-empty、
  stable-ID>=2 frames 与全部资源门，不读质量。
- card total/headroom/peak=`24,576/256/24,320 MiB`，另锁 start<=512 MiB、cgroup<=60 GiB、wall<=1,200s、
  disk>=40 GiB、monitor errors=0。该新门不追认 r033；若失败则按 `V51-F60/F61/PIVOT-F05` 关闭 faithful recovery。
  failure delta=`none_at_preregistration`；所有 downstream/heldout locks 保持 false，M2/M3=`pending`。

## V5.1 Stage F F0b r033 canonical blocked / 独立审计（2026-08-18）

- run/source/tree=`20260818T100000Z__m1-stage-f-f0b-association-parity-s20260814-r033 /191d3e4...12f /
  c4f32ec...e36`；status=`blocked`，resolved/status/events/parity/resource-samples SHA=
  `9d155a88...56f9/e027888e...7234/8f35ca93...3b81/7a6db15f...7ae/db4e6d17...8e9c`。没有 done
  summary/manifest；29 logical files=`544,368 bytes`（含三份输入 symlink logical targets）。
- primary/repeat batch32 三 mask SHA 均为 `3263dc84...4c9a/bc46ecdb...b6b7/3efa06c8...116f`，metadata=
  `e841df5d...12c5`，完全 repeatable；三 mask nonzero pixels=`1,149,652/1,253,787/978,729`，19 个 stable IDs，
  association/non-empty 子门 PASS。batch16 三 mask 与 metadata SHA 全不同，逐帧 exact-label fraction=
  `0.855106/0.799634/0.830072`，foreground IoU=`0.961177/0.995201/0.969622`，batch exact-parity FAIL。
- sampled resources=`GPU 24,116 MiB /cgroup 17,956,044,800 bytes /78.917s /241 samples /0 errors`；GPU 超过
  预注册 24,000 MiB，其他 recorded checks PASS。audit=`20260818T103000Z__stage-f-f0b-r033-audit.json /
  22,939 bytes /a5a7d5c8...fa7d /PASS`；audit PASS 只表示 blocked 事实可独立重放。
- failure delta=`V51-F59 resolved_with_three_view_association / V51-F60 active / V51-F61 active`。quality/H/S/C/
  validation/test/KITTI/full-materialization/identity-training/F1/F2=false，M2/M3=`pending`；下一步仅预注册
  grid32/upstream-batch64 同三视图 association/repeatability/resource smoke。

## V5.1 Stage F F0b r033 三视图 association/batch parity 预注册（2026-08-18）

- task/run=`WS-V51-M1-F-IDENTITY-EMBEDDING-01 /20260818T100000Z__m1-stage-f-f0b-association-parity-s20260814-r033`；
  config=`configs/worldsim_v51/stage_f_f0b_three_view_association_parity_v1.yaml`，seed=`20260814`，split=`train_only`。
  r026 manifest=`18,410 bytes /653941ec...7a4 /45 records /record-chain b3458c27...4d95`；选择同相机三帧
  `000_0/040_0/080_0.jpg`，输入图片在三臂共 decode=`9` 次，输出 mask 共读取=`9` 次。
- arms 固定为 `grid32/batch32 primary → grid32/batch16 parity → grid32/batch32 repeat`；batch16 是同 grid
  的 execution-memory parity，不改变 prompt denominator。PASS 必须同时满足两组 mask/pred exact SHA、primary
  non-empty-mask>=1、positive short-ID 跨>=2帧、GPU<=24,000 MiB、cgroup<=60 GiB、wall<=1,800s、disk>=40 GiB、
  monitor errors=0。
- 本轮不是 quality run，不建立 grid64 quality parity、mask quality、full materialization、identity-training readiness
  或 U2/B3 uplift；quality/H/S/C/validation/test/KITTI/F1/F2 均锁定 false，M2/M3=`pending`。failure refs=
  `V51-F31/F37/F42/F43/F46/F49/F52/F53/F55/F57/F59`；`failure_ledger_delta=none_at_preregistration`。

## V5.1 Stage F F0a r032 canonical resource/schema PASS（2026-08-18）

- run/source/tree=`20260818T090000Z__m1-stage-f-f0a-environment-one-view-s20260814-r032 /29a160a...b52 /
  4681c265...10d`；terminal=`done`，grid/batch=`32/32`、1024 prompts，solver/env/assets/CLI/output/resource gates PASS。
  summary/manifest/status SHA=`f94d0ac5...64e/ede8f38b...e4ab/949067c1...b8c`；manifest logical=`13 entries /
  157,563 bytes`。
- mask=`1,476 bytes /0bf854a1...59d /900×1600 uint8 /all 1,440,000 pixels label0`，pred.json=`120 bytes /
  35fbd75a...5e8 /1 annotation`。这是 one-view<3 voting frames 的 schema boundary；quality/association claims=false，
  不能把 all-background 写成算法失败或成功。
- resources=`GPU peak 23,954 MiB /cgroup 18,044,903,424 bytes /22.494s /71 samples /0 errors`；audit=
  `1,687 bytes /cebe07fd...cd5 /PASS`。两份原 `/root/.cache` ResNet 源副本在 canonical audit 后精确删除，canonical
  `TORCH_HOME` 保留。freeze=`stage_f_f0a_environment_one_view_smoke_freeze_v1.yaml`；failure delta=
  `V51-F57 resolved_with_grid32_boundary / V51-F58 resolved / V51-F59 active`。

## V5.1 Stage F F0a r031 blocked / v6 grid recovery 预注册（2026-08-18）

- r031=`20260818T083000Z__m1-stage-f-f0a-environment-one-view-s20260814-r031`，source=`2e96f05`；CLI stdout 固定
  points-side/batch=`64/32`，仍在 cumulative `BatchMaskData.cat` OOM，request/free=`9.32/9.31 GiB`。GPU/cgroup peak=
  `24,066 MiB/18,052,734,976 bytes`，119 samples/0 errors；6 files=`28,677 bytes`，mask/metadata/quality=false。
- resolved/status/events/stdout/stderr/resource SHA=`101a0297...fb4/99d081ee...e23/20ca8fe5...fea/27e8e672...633/
  b822aab6...692/0a06475d...af4`。external source clean、run immutable，batch change 生效但不足，见 `V51-F57`。
- v6/r032 唯一变化为 official `--SAM_NUM_POINTS_PER_SIDE 32`，prompt denominator=`1024`；batch32、image、threshold、
  models、allocator 和 resource/locks 不变。该资源适配由 DEVA `docs/DEMO.md` 明确建议；不读 mask quality，也不宣称与
  64-grid parity。PASS 后预注册同 grid 的 batch parity 和 3-view association/repeatability。failure delta=
  `V51-F57 recovery_pending`。

## V5.1 Stage F F0a r030 blocked / v5 batch recovery 预注册（2026-08-18）

- r030=`20260818T080000Z__m1-stage-f-f0a-environment-one-view-s20260814-r030`，source=`33c013d`；allocator-only recovery
  仍在 official `64×64 / batch64` 的 `BatchMaskData.cat` OOM：request/free=`9.49/9.16 GiB`，allocated/
  reserved-unallocated=`13.51 GiB/578.72 MiB`。GPU/cgroup peak=`24,098 MiB/18,035,429,376 bytes`，101 samples/
  0 errors；6 files=`25,873 bytes`，mask/metadata/quality=false。
- resolved/status/events/stdout/stderr/resource SHA=`7550586d...fc5/d38fd753...0f7/be114348...05d/abe1dce1...a03/
  15d9bd12...bef/39060722...6ad`。专用 ResNet18/50 assets 的 bytes/SHA 均复核，stderr 不含下载，`.partial` absent；
  `V51-F53 resolved`。原 `/root/.cache` 两文件暂留到后续 canonical audit，不提前删除。
- v5/r031 只增加 official CLI `--SAM_NUM_POINTS_PER_BATCH 32`；point grid、image、threshold、models、allocator 与门禁不变。
  DEVA 文档把该参数定义为并行处理的 point prompts 数；本轮只测资源/output schema，不读质量。PASS 后仍需 batch parity
  与 3-view association+repeatability。一次只读 `rg` 的 alternation 被双层 shell 误解释，未改状态并已改为单关键词查询，
  见 `V51-F56 resolved`。failure delta=`V51-F53 resolved / V51-F55 recovery_pending / V51-F56 resolved`。

## V5.1 Stage F F0a r029 blocked / v4 recovery 预注册（2026-08-18）

- r029=`20260818T073000Z__m1-stage-f-f0a-environment-one-view-s20260814-r029`，source=`3e87323`；solver gates 已通过，
  official CLI 首图 SAM forward 在 `torch.cat` 尝试分配 `6.74 GiB` 时 OOM，sampled GPU peak=`17,246 MiB`。
  resolved/status/events/stdout/stderr/resource SHA=`14e58821...674/ff75bad...f66/f7de78f3...ac5/35110989...4d6/
  205266c9...403/a0d825a8...01a`；6 files=`22,458 bytes`。input decoded=true、DEVA/SAM loaded=true，mask/
  `pred.json`/quality=false；不得解释为 identity/association 质量结论。
- r029 首次暴露并下载 DEVA 隐式 torchvision ResNet50/18；v4/r030 冻结 URL/bytes/full SHA，并从原 cache 原子发布到
  `/root/autodl-tmp/models/gaussian_grouping_v51_stage_f/torch_home/hub/checkpoints`。subprocess 固定 `TORCH_HOME`，避免
  后续网络/用户 cache 漂移。
- v4 的唯一执行恢复是 allocator `max_split_size_mb:128`；SAM point grid/batch=`64/64` 与所有 method/data/resource/
  locks 不变。若仍 OOM，必须保留 r030 blocked 后另行预注册 batch-only adaptation，不得同轮改 size/grid/IoU。
  首次预提交 wrapper 被 PowerShell 提前剥掉远端 commit-message 引号，在 Git/正式 run 前退出；改用文件化消息，见
  `V51-F54 resolved`。failure delta=`V51-F51 resolved / V51-F52+V51-F53 recovery_pending / V51-F54 resolved`。

## V5.1 Stage F F0a r028 blocked / v3 recovery 预注册（2026-08-18）

- r028=`20260818T070000Z__m1-stage-f-f0a-environment-one-view-s20260814-r028`，source=`a1a179e`；Gurobi 12.0.3
  tiny solve 不再报过期 license，但 license banner+terminal JSON 被整段解析而 `JSONDecodeError`。resolved/status/events/
  resource SHA=`8ba1f8c0...7ff/76fe66b7...752/a7180aa9...c12/481ba9e8...c75`；4 files=`10,642 bytes`。
- one-view/DEVA/SAM/input decode/quality 均 false。v3/r029 只解析最后非空 JSON 行并保存 banner；solver optimum 门与所有
  method/data/resource locks 不变。failure delta=`V51-F51 recovery_pending`。

## V5.1 Stage F F0a r027 blocked / v2 recovery 预注册（2026-08-18）

- r027=`20260818T063000Z__m1-stage-f-f0a-environment-one-view-s20260814-r027`，source=`1c0d8bf`，blocked at
  Gurobi tiny model creation：exact 10.0.3 runtime license expired `2024-10-28`。resolved/status/events/resource SHA=
  `2b7fd119...af6/162e1717...bd/3008f50f...73a/58b87ce7...9fd`；4 files=`12,861 bytes`。
- wheels/venv 已构建，但无 environment report；one-view input 未 decode、DEVA/SAM/CLI/GPU 未执行、quality=false。
  v2/r028 唯一语义变化是 `gurobipy 10.0.3 →12.0.3`，仍满足 upstream `>=10.0.3`；fresh wheelhouse/venv 避免污染
  r027。import probe 产生的 4 个 untracked `__pycache__` 已清除，v2 禁止子进程写 bytecode（`V51-F50 resolved`）。
  其余 inputs/args/gates/locks exact inherited；failure delta=`V51-F49 pending/V51-F50 resolved`。

## V5.1 Stage F F0a isolated environment + one-view smoke 预注册（2026-08-18）

- planned r027=`20260818T063000Z__m1-stage-f-f0a-environment-one-view-s20260814-r027`；isolated venv 继承 frozen
  DriveStudio packages，仅补 exact `supervision 0.14.0 /PuLP 2.7.0 /gurobipy 10.0.3` wheels 与 frozen source paths。
  wheels 先落 partial wheelhouse、hash 后 atomic publish；base environment/upstream source 不修改。
- environment gate 同时要求 exact import versions、source import paths、Gurobi tiny MILP optimal 与 PuLP/CBC fallback tiny MILP
  optimal。缺 license/solver/package 是工程 blocked，不能切换模型或写成算法失败。
- one-view=`0471/0/0`；official CLI 参数不变，允许 decode 1 input +1 output mask，检查 900×1600 uint8/label≤199/
  pred.json exactly one annotation 与资源。它不达到 3-frame voting denominator，association claim=false、quality=false。
- ceilings=`GPU 24,000 MiB /cgroup 60 GiB /wall 1,800s /disk-after >=40 GiB`；full materialization/training/
  H/S/C/validation/test/KITTI=false，M2/M3=pending；failure refs 到 `V51-F48`，delta=
  `V51-F48 resolved_before_formal_run`（one-view SHA 手录漏 `d8`，config test 捕获且无 run/env/GPU）。

## V5.1 Stage F F0a r026 canonical asset/source acquisition（2026-08-18）

- run/source/tree=`20260818T053000Z__m1-stage-f-f0a-asset-source-s20260814-r026 /a77f458c...e7a1 /
  1df5c234...aaa3`；terminal=`done`，conclusion=`f0a_assets_and_sources_frozen_environment_setup_required`。
- acquired assets=`DEVA 276,911,801 bytes /52737482...5e48`、`SAM-v1 ViT-H 2,564,550,879 bytes /
  a7bf3b02...262e`；两者 sequential、atomic、无残余 partial。Grounded-Segment-Anything fork=
  `99fbbe78 /tree 89c82ae8...c97 /license 5e86c044...891 /clean`。
- selected input manifest=`45 records /7,530,010 bytes /b3458c27...4d95`，三场各 15 views；source manifest、逐图
  byte/SHA 与顺序由 auditor 独立重建 exact，pixels decoded=false、images staged=false。
- resource=`1 MiB GPU /11,761,352,704 cgroup bytes /166.742s /153 valid /0 errors`；summary/status/manifest=
  `b791ae61...b00/b9490e67...db1/2521c230...7fc`；run=`9 files /51,021 bytes`，audit=
  `1,396 bytes /5a360f42...817c`。
- environment probe exact、mutated=false；materialization/training/quality=false。failure delta=`V51-F47 resolved`；
  next=`isolated DEVA environment + one-view resource smoke preregistration`，S/C/validation/test/KITTI=false，
  F1/F2=false，M2/M3=pending。freeze=`configs/worldsim_v51/stage_f_f0a_asset_source_acquisition_freeze_v1.yaml`。

## V5.1 Stage F F0a asset/source acquisition 预注册（2026-08-18）

- planned r026=`20260818T053000Z__m1-stage-f-f0a-asset-source-s20260814-r026`；只获取 official DEVA/SAM-v1 assets
  与 `Grounded-Segment-Anything@99fbbe78` source，顺序下载、partial resume、exact bytes、full SHA、atomic publish。
  expected asset bytes=`276,911,801 +2,564,550,879`；第一次 acquisition 的 SHA 作为结果冻结，绝不按内容选模型。
- source provenance：Gaussian Grouping=`0ab60afe/Apache-2.0`；vendored DEVA=`frozen files/CC-BY-NC-SA-4.0`；
  Grounded-Segment-Anything fork 要求 exact commit/clean/segment_anything subtree/Apache-2.0。资产放
  `/root/autodl-tmp/models/gaussian_grouping_v51_stage_f`，不污染 upstream checkout。
- input denominator 绑定 frozen 240-record manifest=`be19da2e...943`，再确定性过滤出 45 个 train-only H records；
  原图只做 byte/SHA identity read，pixels decoded=false。future mask CLI/output schema 已固定，但本 run 的
  SAM/DEVA/materialization/training 均 false。
- DriveStudio runtime 仅做 module-presence probe、不 mutate；缺失依赖将在下一 clean prereg 中放进 isolated venv，先做
  one-view resource smoke，不能在 r026 临时 pip install。quality/S/C/validation/test/KITTI=false，F1/F2=false，
  M2/M3=pending；failure refs=`V51-F31/F37/F42–F46`，delta=`none_at_preregistration`。

## V5.1 Stage F F0 r025 canonical source/adapter preflight（2026-08-18）

- run/source/tree=`20260818T045000Z__m1-stage-f-f0-source-preflight-s20260814-r025 /8d68cad1...15c /
  bb73109b...6cd`；terminal=`done`，conclusion=`f0_source_adapter_preflight_done_input_materialization_required`。
  official paper=`10,225,908 bytes /61e82145...823`；repo=`0ab60afe`、tree=`036936e1...fd16`、10 source-file
  hashes/semantic tokens exact、Apache-2.0。
- `0471/1087/0379` 各核对 `5 frames ×3 cameras=15` train-only views；45/45 NPZ schema 相同且无 associated instance
  identity field。metadata `(total/active-union/repeated)`=`49/47/37, 7/7/6, 34/26/20`，只证明 stable tokens 与分母存在，
  不能作为 pixel labels。三个 checkpoint 仅校验 size/SHA、loaded=false；image/mask pixels 与 quality 均未读。
- 16D adapter replay exact：shape=`[1,32,32,16]`、alpha-positive=`189`、identity gradient nonzero=`48`、frozen-base
  gradient absent、loss=`0.0012892523081973195`。resource=`310 MiB GPU /8,810,483,712 cgroup bytes /3.154s /
  9 samples /0 errors`。
- summary/status/manifest/events SHA=`da4890d...988/926ccfdb...d3/0d486f16...349/86fe7f9d...3f0`；manifest=
  `6 entries /29,935 bytes`，full run=`8 files /31,328 bytes`；independent audit=`1,602 bytes /14d2b78b...8b64`。
- result=`source_audit_pass=true /adapter_capability_pass=true /current_training_input_ready=false /
  identity_training_authorized=false`。上游 DEVA/SAM-v1 assets absent，SAM2 non-substitute。failure delta=
  `V51-F45 resolved /V51-F46 active`；下一动作只预注册 F0a train-only SAM+DEVA mask materialization，F1/F2=false，
  S/C/validation/test/KITTI=false，M2/M3=pending。freeze=
  `configs/worldsim_v51/stage_f_f0_source_preflight_freeze_v1.yaml`。

## V5.1 Stage F F0 faithful source/frozen-base adapter preflight 预注册（2026-08-18）

- r023=`20260818T043000Z__m1-stage-f-f0-source-preflight-s20260814-r023` 在 status/monitor/data 前因 `_git`
  helper 漏传 project 参数而 fail，只有 `resolved_config.yaml=7,796 bytes`，不是方法 run。r024 已越过该点，但在 CUDA
  context 初始化前 reset peak counter 而 blocked；只含 4 files=`9,449 bytes`，无 report/training/quality。recovery planned
  r025=`20260818T045000Z__m1-stage-f-f0-source-preflight-s20260814-r025`；official paper=
  `ECCV 2024 /10,225,908 bytes /61e82145...823`，official code=`lkeab/gaussian-grouping@0ab60afe`、tree=
  `036936e1...fd16`、Apache-2.0。冻结 repo 9 个关键文件与 paper identity，不 import/修改 upstream code。
- faithful core=`SAM everything → DEVA semionline short-ID association → 16D view-independent identity encoding →`
  `alpha compositing → shared 1×1 classifier + normalized 2D CE + k5/sample1000 Euclidean KNN forward-KL`；code-semantics
  包括 top-k self、3D loss every 2 steps/weight 2、identity/classifier lr=`2.5e-3/5e-4`、30k iterations。
- adaptation 只来自 normative frozen-base 约束：geometry/appearance/opacity/actor poses 不训练，仅探测 16D feature
  rasterization 是否可微。三场只看 15 个 train-only image/observation 的存在性与 schema、instances metadata stable token；
  image/mask pixels、quality/H/S/C/validation/test/KITTI 均不读。
- 当前 expected gap=`binary actor-union observation has no cross-view instance identity labels + upstream DEVA/SAM-v1`
  `checkpoints absent`；SAM2 checkpoint 只作已存在 non-substitute。preflight 完成的下一动作是另行预注册 F0a train-only
  SAM+DEVA mask materialization，不直接训练 F0。F1/F2=false，M2/M3=pending；failure refs=
  `V51-F31/F37/F42/F43/F44/F45`，recovery delta=`V51-F44 resolved / V51-F45 pending`。

## V5.1 Stage E E0b r022 canonical H rejection（2026-08-18）

- run/source/tree=`20260818T020000Z__m1-stage-e-e0b-h-evaluation-s20260814-r022 / 3a84be68...ade /
  1bc14e34...12`，status=`rejected`。12 views 的 frozen U2/B3 G0、r018 D0、r021 E0B 三臂 persisted
  float16 probability 与同一 target 独立复算 exact；checkpoint、per-scene report、21-entry manifest 与 dual gate exact。
- primary E0B-vs-U2/B3 gate：positive BF1 scenes=`2/3 PASS`，但 mean BF1/IoU/FN=
  `-0.0002566/-0.0925468/+0.1899473`，后三项 FAIL。mechanism E0B-vs-D0 gate：nonnegative BF1 scenes=
  `1/3`，mean BF1/IoU/FN=`-0.0004762/-0.0210926/+0.0204707`，四项全 FAIL。
- scene `(BF1,IoU,FN)` vs U2/B3：0471=`(+0.124700,+0.166308,+0.075097)`、1087=
  `(+0.056490,-0.158934,+0.217539)`、0379=`(-0.181960,-0.285014,+0.277206)`；vs D0：0471=
  `(+0.001927,+0.000857,-0.005733)`、1087=`(-0.000410,+0.000483,-0.000607)`、0379=
  `(-0.002946,-0.064618,+0.067752)`。
- summary/status/manifest/events SHA=`4964a2f0...3d4/848fcd51...682/3c5a2fbe...7aa/7cb8704a...b8f`；run=
  `23 files /1,201,649 bytes`，audit=`6,314 bytes /5ced73db...104f`。resource=`10,724 MiB GPU /
  11,474,419,712 cgroup bytes /151.543s /124 valid /0 errors`。
- failure delta=`V51-F40/F41 recurrence resolved + V51-F42 active`；E0B rejected，E1/E2 stopped，禁止 H tuning/
  reread；S/C/validation/test/KITTI=false，M2/M3=pending。next=`Gaussian Grouping faithful source audit + no-quality
  preflight`；freeze=`configs/worldsim_v51/stage_e_e0b_h_evaluation_freeze_v1.yaml`。

## V5.1 Stage E E0b r022 matched H evaluation 预注册（2026-08-18）

- planned run=`20260818T020000Z__m1-stage-e-e0b-h-evaluation-s20260814-r022`；输入只绑定 frozen U2/B3 G0
  renders、r018 raw D0 float16 renders、r021 E0B Gaussian posterior 和原 12 个 H targets。baseline/D0/target/view 均
  不重算或改变，candidate render 持久化为 float16 后再计算七项 metric。
- primary gate vs U2/B3=`BF1 positive scenes>=2 + mean BF1>0 + mean IoU>=0 + mean FN<=+0.02`；mechanism
  gate vs D0=`BF1 nonnegative scenes>=2 + mean BF1>0 + mean IoU>=0 + mean FN<=0`。decision 为两门 AND，
  不能用 external baseline、单场收益或 calibration 抵消 IoU/FN failure。
- pass→freeze E0B 并预注册 E1 PanoGS faithful port；fail→`E0B rejected / E1,E2 stopped / next Gaussian Grouping`。
  parameter search、S/C/validation/test/KITTI=false，M2/M3=pending；failure refs=
  `V51-F31/F34/F37/F39/F40/F41`，prereg delta=`none`。

## V5.1 Stage E E0b r021 canonical no-quality operator（2026-08-18）

- run/source/tree=`20260818T014000Z__m1-stage-e-e0b-operator-s20260814-r021 / e573fe4f...2b74 /
  d47edb41...d802`，status=`done`。fine node count=`561,618/620,540/764,752`；directed quotient edges=
  `2,991,329/3,289,464/4,037,917`；exact-distance 1/2-hop neighbors=`3,862,658/9,141,772`、
  `4,226,568/10,201,080`、`5,211,750/11,766,570`。
- final actor/background/UNKNOWN nodes：0471=`11,703/549,640/275`，1087=`183/620,353/4`，
  0379=`257/764,434/61`；Gaussian posterior changed vs D0=`4,065/1/475`。这些只是 operator diagnostics，
  未渲染也未读取 quality。
- summary/status/manifest/events SHA=`bb17f82f...ae0/1682b276...89a/8b9059e5...ccd/ae9cefe1...5b6`；sidecar SHA=
  `1b2b7827...c2f/2cac0cf9...3ea/5b8d33db...f1f`。full replay audit=`2,635 bytes / 1e5a0564...650c`，
  三场 output arrays、node-constant readout、13-entry manifest 与 resource maxima exact。
- repeat prefix=`4,096 nodes /24,289 directed quotient edges / byte exact /925375c8...809`；resource=
  `1 MiB GPU /9,532,755,968 cgroup bytes /92.608 s`。failure delta=`none`；quality/E1/E2/validation/test/KITTI=false，
  M2/M3=pending；freeze=`configs/worldsim_v51/stage_e_e0b_same_propagation_freeze_v1.yaml`。

## V5.1 Stage E E0b fine-q50 same-propagation 预注册（2026-08-18）

- planned run=`20260818T014000Z__m1-stage-e-e0b-operator-s20260814-r021`；arm=`E0B`，selected level=
  `fine_q50`。三档均通过 E0a 时按 quantile ascending 取第一档，明确不使用 density gain、seed conflict 或 quality
  选择 level；这是 H-independent minimum-intervention rule。
- node evidence=`mean member unary / visibility-weighted valid member SAM probability / max member visibility`；topology=
  frozen raw directed KNN 的 quotient（drop self + directed dedup）后沿用 D0 exact 1/2-hop。除 node elevation、evidence
  aggregation 和 Gaussian broadcast 外，D0 propagation threshold/affinity/seed/UNKNOWN/fixed-point 合同 byte-semantic 不变。
- 输出保留 U2/B3 unary、frozen raw D0 posterior 与 E0B posterior，便于后续 matched H；本 run 只报告 topology、seed、
  propagation、与 D0 非质量差异和资源，不解析 evaluation artifact。future H primary gate 继承 D0 对 U2/B3 的四门，另要求
  E0B vs D0 至少 2/3 scene BF1 nonnegative、mean BF1 `>0`、mean IoU `>=0`、mean FN delta `<=0`。
- H/S/C/validation/test/KITTI quality=false，E1/E2=false，M2/M3=pending；failure refs=
  `V51-F31/F34/F37/F39/F40/F41`，prereg delta=`none`。

## V5.1 Stage E E0a r020 canonical structural result（2026-08-18）

- run/source/tree=`20260818T012000Z__m1-stage-e-e0a-density-s20260814-r020 / 4342ddb3...0c13 /
  2a264757...e4645`；status=`done`。三场的 fine/medium/coarse 三档均通过 no-quality existence gate；首份
  0471/fine sidecar NPZ byte repeat exact=`ad796cf7...e0b10`。
- raw mean observed views 与 zero-observation Gaussian：0471=`1.400230 / 299,571`，1087=`0.028232 / 904,933`，
  0379=`0.457917 / 760,253`。fine 档 rescue=`28,330/718/28,933`，node reduction=
  `34.6662%/33.3629%/35.5885%`；coarse 虽有更大 union gain，但 seed mixing 只是诊断，不是选择依据。
- summary/status/manifest/events SHA=`e96677e3...3c/9c9b95e0...25/d743c5ca...91/0506da6a...45c`；resources/
  samples=`f35ce64f...f8/dcef9d62...7c`。independent audit=`6,130 bytes / 8df03b2a...5d34`，复算 9/9
  assignments、45/45 view denominators、density metrics、gate 与 18-entry inventory exact。
- quality/propagation/parameter-search/E1/E2 均 false，M2/M3=pending；本结果只解锁 E0b preregistration。
  定向回归=`8 passed`；扩大回归按环境拆分后=`95 motionproj passed / 3 drivestudio passed`。
  `failure_ledger_delta=V51-F34/F40 recurrence resolved / V51-F39 resolved / V51-F41 resolved`；freeze=
  `configs/worldsim_v51/stage_e_e0a_superprimitive_probe_freeze_v1.yaml`。

## V5.1 Stage E r019 blocked / r020 recovery 预注册（2026-08-18）

- r019 source=`2e1786b`，status=`blocked`，error=`frozen KNN contains nonpositive/nonfinite edge length`；完成
  0471/1087 后在 0379 edge-scale derivation 阻塞。status/events/resource-samples SHA=
  `491ad90a...f3ff/60e8821e...5cc8/33f86bee...04f`；13 files / `28,114,119 bytes`，无 summary/manifest/terminal
  gate，partial sidecars 不作方法证据。
- read-only edge audit=`1,290 bytes / 30493d5d...bc5`：只有 0379 存在 `34/7,123,746` zero-length edges，
  三场 nonfinite 均为 0 且都有数百万 positive edges。该事实推翻 v1 的 all-positive 输入前提，不推翻 voxel density
  hypothesis；failure delta=`V51-F38 resolved / V51-F39 recovery pending / V51-F40 resolved`。
- v2/r020 只把 size statistic 定义为 positive-edge q50/q75/q90；零长 edge 不参与 quantile，但不删除任何 Gaussian、
  不改变 assignments/gate/views/quality locks。r020 从头完整重跑，禁止读取 r019 partial 数值来选 level；E0b/E1/E2
  仍 locked，quality/S/C/validation/test/KITTI=false，M2/M3=pending。

## V5.1 Stage E E0a simple voxel structural probe 预注册（2026-08-18）

- task/arm=`WS-V51-M1-E-NODE-ELEVATION-01 / E0A`；输入只继承 frozen B3 center/unary、V5 KNN edges 与 D0
  的 15 train-only availability/reliability/visibility rows。levels=`edge-length q50/q75/q90`，不预选 resolution；输出
  逐 Gaussian deterministic voxel node-id sidecar 与 observation-density/seed-mixing diagnostics。
- gate 是 no-quality existence gate：每场至少一档严格减少 node、提高 Gaussian-weighted member-view union、救回
  raw zero-observation Gaussian。它不渲染、不传播、不计算 BF1/IoU，不读取已用完的 r018 H 或 S/C/validation/test；
  PASS 后仍须另行预注册 E0b same-propagation matched A/B，不能把 density gain 写成算法提升。
- source context：PanoGS official paper=`CVPR 2025 / 3,396,570 bytes / 1d206aeb...f49c`，official repo=
  `zhaihongjia/PanoGS@8dfb69b`、tree=`c7cc9b4`、Apache-2.0；E0 被明确分类为 internal simple control，E1/PanoGS
  faithful execution 与 E2/AG²aussian 仍 locked。failure refs=`V51-F31/V51-F37`，prereg delta=`none`。

## V5.1 Stage D r018 canonical H rejection（2026-08-18）

- run=`20260818T003000Z__m1-stage-d-d0-h-evaluation-s20260814-r018`，source/tree=
  `2cd98b31...02cf/ccf42d68...ee4`，status=`rejected`；12/12 D0 renders、U2/B3 G0 与 frozen V5 G3 inputs、
  target、float16 precision、per-view metrics、equal-view/equal-scene aggregates 全部独立复算 exact。
- H checks=`BF1 positive scenes PASS (2/3) / mean BF1 PASS (+0.0002196) / IoU FAIL (-0.0714543) /`
  `FN safeguard FAIL (+0.1694766 > +0.02)`。scene `(BF1,IoU,FN)` delta：0471=
  `(+0.122773,+0.165451,+0.080830)`，1087=`(+0.056899,-0.159417,+0.218146)`，0379=
  `(-0.179014,-0.220397,+0.209454)`。因此 progressive rejected、D1 skipped、next=Stage E E0。
- manifest=`21 entries / 1,175,047 bytes`，run=`23 files / 1,179,119 bytes`；summary/status/manifest SHA=
  `b08c7276...62d6/7273c30b...ba95/792660e3...010c`。audit report=`3,888 bytes / 18c12f4d...0d2`；
  resources=`10,724 MiB GPU / 11.36GB cgroup /124.897s /108 valid /0 errors`。
- failure delta=`V51-F37 active`；禁止 D0/D1 recovery tuning。S/C/validation/test/KITTI=false，M2/M3=pending；
  result freeze=`configs/worldsim_v51/stage_d_progressive_h_evaluation_freeze_v1.yaml`。

## V5.1 Stage D D0 H matched evaluation / r018 预注册（2026-08-18）

- task/phase=`WS-V51-M1-D-PROGRESSIVE-01 / d0_h_matched_evaluation`；planned r018 只读 r017 frozen D0 与 V5
  frozen H graph/evaluation artifacts，不重算 baseline、D0 或 target。arms=`U2_B3_G0/U2_B3_G_V5/D0`，三臂统一从
  persisted float16 probability 计算同一 metrics；总分母固定 `12 views / 3 scenes`，cross-scene equal-scene。
- gate=`positive BF1 scenes >=2 + mean BF1>0 + mean IoU>=0 + mean FN<=+0.02`。成功动作仅为 freeze D0 后
  S exact-once；失败动作仅为 reject progressive、skip D1、advance super-primitive/anchor。禁止 parameter search、
  baseline/D0 recompute、target/view change 和 S/C/validation/test/KITTI read。
- identity/resource gates 包括 operator freeze、三场 graph config/run manifest、三场 D0 sidecar、live checkpoint
  Gaussian layout、checkpoint immutability、GPU start/peak、Torch/cgroup/wall 与零 monitor error；runtime 固定 DriveStudio
  torch `2.1.2+cu118`。failure refs=`V51-F31–V51-F36`，当前 prereg `failure_ledger_delta=none`。

## V5.1 Stage D r017 canonical D0 operator（2026-08-18）

- run=`20260818T001000Z__m1-stage-d-d0-operator-s20260814-r017`，source=`d3321bb`；三场 full raw-Gaussian
  topology/affinity/growing 均完成，sidecar SHA 0471/1087/0379=`169054ad...5d55/37a4cadc...c6d1/`
  `1e9c059c...3b5b5`。U2/B3 input arrays byte/array exact，D0 labels/posterior/assignment-level schema exact。
- seed→final counts：0471 actor `47,369→52,764`、UNKNOWN `1,085`；1087 `236→244`、UNKNOWN `4`；0379
  `607→621`、UNKNOWN `85`。affinity 的 jointly-visible topology 明显 scene-dependent，但当前禁止把结构统计解释成
  BF1/IoU 结论。
- resources=`9.38GB cgroup / 1MiB GPU /132.908s /65 valid samples /0 errors`，OOM counters 不变；manifest=
  `13 files / 8,823,721 bytes`，run=`15 / 8,826,262 bytes`。freeze=`stage_d_progressive_operator_freeze_v1.yaml`；
  quality/S/C/validation/test/KITTI=false，M2/M3=pending，failure delta=`V51-F36 resolved`。

## V5.1 Stage D D0 clean-room operator / full-H r017 预注册（2026-08-18）

- module=`motion_proj/worldsim_v51/progressive_propagation.py`；formula pipeline=`exact 1/2-hop geometry topology →`
  `visibility-weighted multi-view soft-binary cosine → member/distance-weighted region affinity → 0.9…0.5 fixed-point growing`。
  U2/B3 为 seed/baseline，未观测或 exact conflict 最终 UNKNOWN，upstream code import=false。
- tests=`5 operator +4 runner/config =9 passed`；包括 edge-order exact determinism 和 deterministic NPZ input loader。
  初始 UNKNOWN fixture 实际与 Background 相邻且 cosine≈`0.714`，在 frozen `0.5` level 被吸收是正确行为；fixture
  纠正为孤立 node，登记 `V51-F36 resolved`，没有 formal result 或 quality read。
- planned r017=`m1-stage-d-d0-operator-s20260814-r017`；输入为 r016 freeze + 三场 frozen B3/15 train-only observation
  sidecars/V5 KNN topology，只输出 labels/posterior/assignment-level、affinity/propagation diagnostics 和 resource evidence。
  不解析 V5 quality diagnostics、不 render evaluation；S/C/validation/test/KITTI=false，M2/M3=pending。

## V5.1 Stage D r016 canonical D0 preflight（2026-08-18）

- run=`20260818T000000Z__m1-stage-d-d0-preflight-s20260814-r016`，source=`99a626b`，report=
  `10,433 bytes / b84cb719...4b7b`；第二次独立执行与 canonical byte exact。
- checks=`29 identities + 3 Gaussian counts + 3 directed-edge denominators + 3 evaluation-view denominators +`
  `U2/B3 survivor + LUDVIG rejected terminal + upstream commit/tree/license inventory`，全部 PASS。
- 本 run 未解析 diagnostics/evaluation quality，只对其文件做 SHA identity；quality/S/C/validation/test/KITTI=false，
  M2/M3=pending。freeze=`stage_d_progressive_preflight_freeze_v1.yaml`，failure delta=`none`。

## V5.1 Stage D D0 progressive preflight 预注册（2026-08-18）

- task/arm=`WS-V51-M1-D-PROGRESSIVE-01 / D0`；paper-zero source=official SAI3D CVPR 2024，repo commit/tree=
  `1d9a6a2/7320924`。仓库无显式 LICENSE，执行 policy=`clean_room_reimplementation_from_paper_equations_no_upstream_code_copy`。
- faithful port=`raw Gaussian + frozen KNN geometry adjacency + visibility-weighted multi-view SAM soft binary cosine +`
  `region-to-node weighted affinity`；固定 hop=`2`、decay=`0.5`、thresholds=`0.9/0.8/0.7/0.6/0.5`、U2/B3
  high-confidence seeds、exact tie/final unsupported=`UNKNOWN`。禁止 DINO uplift、learned parameter、Bayesian/SAM/motion
  innovation、node coarsening、threshold search 和 one-shot global smoothing。
- H inputs 只绑定 V5 immutable r037/r042/r043 B3 unary 与 r038/r045/r046 topology/manifest，预计 `12` matched
  evaluation views；当前 preflight 只 hash，不解析 quality-bearing diagnostics。H gate 固定为 BF1 positive scenes `>=2/3`、
  mean BF1 `>0`、mean IoU delta `>=0`、mean FN delta `<=+0.02`；失败即 skip D1 并转 Stage E。
- D0 regression=`4 passed`。扩大回归暴露旧 plan-hash 单值 validator 与一次错误解释器聚合，分别登记并修复为
  authorized append hash chain、双环境分组；formal preflight 必须在 clean prereg commit 后运行。
  S/C/validation/test/KITTI quality=false，M2/M3=pending；failure refs=`V51-F31–F35`。

## V5.1 Stage B r015 canonical H evaluation reject（2026-08-18）

- run=`20260817T173940Z__m1-stage-b-h-evaluation-s20260814-r015`，source=`0a79a56`，status=`rejected`；
  H views=`45 evidence +45 evaluation`，3 checkpoint identities exact，final heldout/S/C/validation/test 未读。
- per scene 0471/1087/0379：Rigid coverage=`0.931484/0.975410/0.621835`；B1 eligible actors=`13/0/1`；
  B1 margin=`-0.121280/abstain/-0.098618`；heldout `B1-B0=+0.026951/+0.023009/+0.018372`。
- gate checks=`evaluable PASS / positive-margin FAIL / mean-margin FAIL / rigid-coverage PASS / heldout PASS`；final=
  `reject_ludvig_uplift_and_raw_graph`。repeatability B0/B1 也为 `0.859851/0.851373`、`0.870764/0.866528`、
  `0.859621/0.853277`，没有提供 B1 跨视图优势。
- resources=`22,570 MiB NVIDIA / 23,354 MiB Torch reserved / 14,221,561,856-byte cgroup / 896.320 s`，
  v2 24,000 MiB gate PASS。audit=`12 manifest entries / 14 files / 314,994 bytes` exact；r014/r015 float max delta
  `4.976e-13`，离散/gate exact。freeze=`stage_b_h_evaluation_freeze_v1.yaml`；failure delta=`V51-F15/F28/F30/F31`。

## V5.1 Stage B r014 resource blocked / r015 recovery 预注册（2026-08-18）

- r014 source=`9b151c8`，90/90 views 与三场 diagnostics 已写出，随后因 NVIDIA peak=`22,570>22,528 MiB`
  fail-closed；Torch reserved=`23,354>22,528 MiB`，cgroup=`14,305,161,216 bytes`，duration=`897.647 s`。
- blocked report 只做 SHA/bytes 封存，不读取 scene 或 aggregate quality；status=`6409545b...f6d1`，report=
  `510f82ec...227c`，resources=`ffc98a00...674e`，resource samples=`8fae05eb...7cf`。failure delta=`V51-F28`。
- v2/r015 唯一允许差异为 resource NVIDIA/Torch ceilings `22,528→24,000 MiB`；算法、split、pair/proxy、H gate、
  quality locks 不变，禁止复用 blocked output。首次 inventory 的 `events.json` 文件名笔误登记 `V51-F29 resolved`。

## V5.1 Stage B H evaluation-only r014 预注册（2026-08-18）

- inputs=`r012 uplift freeze + r010 evidence feature freeze + r013 evaluation feature freeze + 3 immutable H checkpoints`；
  view denominator=`3 scenes × (15 evidence + 15 evaluation)=90`，final heldout remainder=`4` 不读。
- proxy=`model_membership_proxy_not_ground_truth`，evaluation-only。reference=`frame80/camera1`；active Rigid actor
  至少 32 covered rows，pair cap=4,096，seed 由 `SHA256(20260814|scene|actor)` 固定；actor 内 equal，scene equal。
- repeatability 逐 evidence view 比较 frozen aggregate feature 与同 Gaussian single-view transpose；actor margin=
  same-actor pair cosine minus actor-to-nearest-covered-Background cosine；heldout reprojection 逐 view 在 exact common
  B0/B1 pixel denominator 上比较 frozen DINO target。
- gate=`evaluable scenes>=2; positive B1-margin scenes>=2; mean B1 margin>0; mean Rigid coverage>=0.60;`
  `mean heldout(B1-B0)>=-0.01`。resource=`22,528 MiB NVIDIA/Torch / 80 GiB cgroup / 7,200 s`。
- 当前只允许 pure/config regression 与 clean prereg commit，尚未读取 quality。S/C/validation/test/KITTI=false，
  M2/M3=pending；failure refs 包含 active `V51-F15`，delta=`pending`。

## V5.1 Stage B r013 canonical H heldout features（2026-08-18）

- run=`20260817T170028Z__m1-stage-b-h-eval-feature-s20260814-r013`，source=`b359541`，status=`done`；精确读取
  H frames=`2/42/82/122/162`，未触碰 remainder=`4`。checkpoint immutable，raw/transform first-repeat 均 bit-exact。
- outputs=`45 deterministic [40,64,114] float32 NPZ`，feature manifest=`8824a8dc...f73c`，record chain=
  `2ca3f8bc...9d50`，总 bytes=`48,452,027`；独立 auditor 对 45/45 files/arrays/identities/finite 与 55/55 terminal
  inventory entries 全部 exact，run=`57 files / 48,547,857 bytes`。
- resource=`6,702 MiB NVIDIA / 6,376 MiB Torch reserved / 17,320,468,480-byte cgroup / 70.806 s / 123 samples / 0 errors`。
  freeze=`stage_b_h_eval_feature_freeze_v1.yaml`；failure delta=`V51-F24/F25/F26/F27 resolved`。
- r013 不含 membership proxy、renderer、uplift feature 或 method quality read；S/C/validation/test/KITTI=false，
  M2/M3=pending。下一门是另行 clean preregister H evaluation-only metrics/gate，不能先看质量。

## V5.1 Stage B H heldout feature r013 预注册（2026-08-18）

- denominator=`3 H scenes × 5 evaluation frames × 3 cameras = 45 views`；frames 固定为
  `2/42/82/122/162`，只读 evaluation remainder=`2`，禁止触碰 final heldout remainder=`4`。
- extraction=`official ViT-g14-reg4 / 1596×896 / last-of-four / raw float32`；transform 只消费 r010 frozen
  `feature_mean/std + pca_mean/components`，PCA refit/subsample/search 全部禁止。首图 raw inference 与 transform
  分别做一次 bit-exact repeat。
- planned outputs=`45 deterministic [40,64,114] float32 NPZ + identity manifest + resource/report`；checkpoint
  before/after SHA 必须 exact。resource gate=`22,528 MiB NVIDIA/Torch reserved / 80 GiB cgroup / 1,800 s`。
- 本阶段 `membership_proxy/renderer/uplift_feature/method_quality=false`，不能查看 r012 B0/B1 质量；
  S/C/validation/test/KITTI=false，M2/M3=pending。failure refs 含 active `V51-F15`，delta=`pending`。
- pre-formal regression 首轮新 heldout suite=`8 passed`；一次附加聚合命令因错误地用 motionproj Python 调用
  DriveStudio uplift test 而得到 `1 failed / 9 passed`。未创建 run、未读 quality；登记 `V51-F24 resolved`，须按双环境重跑。
- 随后的 PowerShell→SSH 聚合命令又因本地提前解释 `$(find ...)` 而在任何测试前失败；无 repo/GPU/quality 状态变化，
  登记 `V51-F25 resolved`，后续拆分成无 command substitution 的独立命令。
- r013 post-run 首次 inspection 的多语句 Python `-c` 同样在 PowerShell 本地解析阶段失败；远端未执行，登记
  `V51-F26 resolved`，独立 auditor 文件替代嵌套 source。
- auditor 已传到 `/tmp`，但同一 scp 命令中的 docs source 相对 staging workdir 少一层 `motion_proj/` 而本地失败；
  run/repo 无部分修改，登记 `V51-F27 resolved` 后按精确 source path 重传。

## V5.1 Stage B r012 canonical H uplift（2026-08-18）

- run=`20260817T163100Z__m1-stage-b-h-uplift-s20260814-r012`，source=`4fc07cb`，status=`done`；输入仍为 r005
  operator freeze、r010 PCA/45 feature sidecars 与 V5 r027/r028/r029 immutable checkpoints，未复用 r011 blocked sidecar。
- outputs=`3 scene reports + 6 Gaussian feature NPZ + identity manifest`；coverage 0471/1087/0379=
  `0.8986823140/0.8479816328/0.8529442234`，B0/B1 L2=`1826.1010/2077.5490/1875.4642`，checkpoint 全 exact。
- resources=`20,554 MiB NVIDIA / 20,202 MiB Torch reserved / 14,450,888,704-byte cgroup / 621.170 s`，
  v2 22 GiB gate PASS。manifest=`19 files / 811,269,697 bytes`，run=`21 files / 811,273,469 bytes`；独立逐文件、
  逐数组审计 exact。failure delta=`V51-F23 resolved`。
- freeze=`stage_b_h_uplift_freeze_v1.yaml`；本 run 没有 proxy/quality，S/C/validation/test/KITTI 未读，M2/M3=pending。

## V5.1 Stage B r011 resource blocked / r012 recovery 预注册（2026-08-18）

- r011=`20260817T161351Z__m1-stage-b-h-uplift-s20260814-r011`，source=`40f4d64`，status=`blocked`，reason=
  `ProtocolError: H uplift NVIDIA peak 超限`。45/45 views、3 scene reports、6 NPZ sidecars 和 checkpoint before/after exact
  已计算并落盘，但 resource gate 失败，禁止提升为 canonical result。
- resource=`20,554 MiB NVIDIA / 19,314.634 MiB Torch allocated / 20,202 MiB Torch reserved /
  13,328,011,264-byte cgroup / 588.750 s / 799 valid samples / 0 monitor errors`；pre-registered ceiling=
  `18,432 MiB NVIDIA/Torch`。failure delta=`V51-F23`，属于预算低估，不是 OOM 或 method-quality negative。
- blocked evidence audit=6/6 NPZ file/content identity exact，manifest chain=`5339b880...8e12`，status/resources/report/
  manifest/resource-samples SHA=`a450cdaf...0eee6/0312e190...8a37/98140571...99c9/88956448...dae6/76422821...ae9c`；
  该审计仅保证失败证据完整，不改变 blocked 状态。
- recovery config=`stage_b_h_uplift_v2.yaml`；唯一允许差异为两项 GPU ceiling=`22,528 MiB`。r012 从原冻结输入完整重跑，
  不复用 r011 sidecar；算法、view、PCA、support floor、resource cgroup/disk/timeout 和所有质量锁不变。
- r011 没有读取 membership proxy/method quality，S/C/validation/test/KITTI 未读，M2/M3=`pending`。

## V5.1 Stage B H uplift r011 预注册（2026-08-18）

- inputs=`r005 operator freeze + r010 PCA/45 patch-grid sidecars + H r027/r028/r029 checkpoints/configs`；view order=
  `scene 0471→1087→0379, frame 0/40/80/120/160, camera 0/1/2`，image index=`frame×3+camera`。
- common support=`intersection≥1e-4, Gaussian-view mass≥1e-3, epsilon=1e-8`；B0/B1 唯一差异保持 view saturation
  vs normalized renderer mass。CSR float64 实现不生成 intersection×40 dense tensor。
- formal outputs=`3 scene reports + 6 Gaussian feature NPZ + identity manifest + resource evidence`；checkpoints 前后 exact，
  B0/B1 coverage 必须相同且 feature L2 difference>0。failure delta=`pending`。
- locks：dataset loader 可基础设施物化 image/mask/LiDAR，但 runner 仅取 timestamp/image-id/camera；不消费其值，
  不读 membership proxy/quality，S/C/validation/test/KITTI=false，M2/M3=pending。

## V5.1 Stage B r010 canonical H feature/PCA（2026-08-18）

- run=`20260817T155859Z__m1-stage-b-h-feature-pca-s20260814-r010`，source=`11c35fd`，status=`done`；
  45 H images、328,320×1,536 raw population，first image repeat bit-exact，DINO checkpoint immutable。
- PCA state：mean/std float64、PCA mean/components/singular values float32，correction=`1`、randomized `40D`、seed=
  `20260814`、whiten=false、subsample=false；state SHA=`fe9eea72...3231c8`，repeat writer byte-exact。
- feature manifest=`45 records`、chain SHA=`4c3689da...c3289`；所有 NPZ `[40,64,114] float32` 的 file SHA 和
  array content SHA 二次复核无错误。raw memmap 按成功合同删除。
- resources=`6,702 MiB NVIDIA / 6,376 MiB Torch reserved / 15,635,017,728 bytes cgroup / 104.472 s`；
  summary/status/manifest=`c6b81374.../1e8b78da.../160efe34...`。failure delta=`V51-F14 resolved`。
- freeze=`stage_b_h_feature_pca_freeze_v1.yaml`；本 run 不含 renderer/uplift/proxy/quality，S/C/validation/test/KITTI 未读。

## V5.1 Stage B H feature/PCA r010 预注册（2026-08-17）

- input denominator=`3 H scenes / 45 views / 328,320 patches / raw 1536D`；image/source/checkpoint/resource/
  one-view-contribution freezes 逐 SHA 绑定。仅 H RGB 用于 DINO feature extraction，其他角色和 quality 全锁。
- extraction=`official ViT-g14-reg4, last-of-four, fp16 autocast, raw float32 CPU memmap`；first-view full feature
  repeat 必须 bit-exact，45 个 raw feature 逐 content SHA，checkpoint before/after exact。
- PCA=`dataset-level mean/std correction1, fixed two-pass float64 accumulator, standardized float32, randomized 40D,
  seed 20260814, whiten=false, no subsample`；persisted state=`feature_mean/std + pca_mean/components/singular_values`。
- outputs=`45 deterministic NPZ patch grids [40,64,114]`、identity manifest、resource/report；raw memmap 成功后不保留。
  failure refs 含 `V51-F14/F15`，delta=`pending`；formal 前只允许 unit/config regression 和 clean prereg commit。

## V5.1 Stage B r009 canonical contribution denominator（2026-08-17）

- run=`20260817T154359Z__m1-stage-b-one-view-contribution-s20260814-r009`，source=`7f0c6c9`，status=`done`；
  scene/view=`0471/H/frame0/camera0`，model-native=`800×450`，Gaussian=`859,613`，checkpoint immutable。
- raw=`47,378,525 rows / 299,051.805624 mass`；intersection floor `1e-4` 后=`32,030,248 rows /
  298,668.303850 mass`；view-mass floor `1e-3` 后=`313,764 Gaussian`，global coverage=`0.3650061132`；
  `41,995` 个有 intersection support 的 Gaussian 因 view mass 不足被丢弃。完整 rows 不落盘。
- GPU start/peak=`1/14,234 MiB`，Torch allocated/reserved=`13,389.991/13,882 MiB`，cgroup peak=
  `9,593,946,112 bytes`，duration=`61.109 s`，resource gate PASS。summary/status/manifest/fingerprint=
  `b1e2282a.../c0bd3501.../6439a1a9.../4376c139...`；manifest recheck=`8/8 exact`。
- result freeze=`configs/worldsim_v51/stage_b_one_view_contribution_freeze_v1.yaml`；failure delta=
  `V51-F21 resolved + V51-F22 resolved`。本 run 没有 feature/PCA/quality，validation/test/KITTI 锁未触碰。

## V5.1 Stage B r008 resource blocked / r009 recovery 预注册（2026-08-17）

- r008=`20260817T153826Z__m1-stage-b-one-view-contribution-s20260814-r008`，source=`eb334fa`，status=`blocked`，
  reason=`ProtocolError: one-view contribution NVIDIA peak 超限`。正确的 model-native `800×450` renderer 已完成，
  failure 发生在 contribution 汇总后的资源门，不是 denominator 或质量判定。
- 资源事实：GPU start/min/peak=`4/4/14,234 MiB`，cgroup peak=`9,598,074,880 bytes`，valid/error samples=
  `89/0`；预注册 NVIDIA ceiling=`12,288 MiB`。status/events/resolved/resource-samples SHA=
  `8b8ebe17.../ed216a5b.../c284d64d.../fc0f9788...`，run=`18,502 bytes`，保留不复用。
- failure delta=`V51-F21`；`V51-F20 resolved`。v4/r009 recovery 仅将 NVIDIA/Torch ceiling 提升为
  `16,384 MiB` 并把诊断 artifact 写入提前到资源 gate 之前；所有算法、数据、floor、quality locks 不变。
- 本次运行没有 summary/quality/feature/PCA；validation/test/KITTI 未读，M2/M3=`pending`。

## V5.1 Stage B r007 renderer-size blocked / r008 recovery 预注册（2026-08-17）

- r007=`20260817T153300Z__m1-stage-b-one-view-contribution-s20260814-r007`，source=`e06b5ff`，status=`blocked`，
  reason=`ProtocolError: renderer width/height 漂移`。环境恢复成功并到达 renderer，但在 inventory 前 fail-closed；
  denominator/feature/quality 数值仍为 0。
- source config 与既有 V5 configs exact 证明 sensor=`1600×900` 经 `downscale_when_loading=[2,2,2]` 后 checkpoint/model
  native render=`800×450`；v2 把 sensor 尺寸错误复用为 renderer 尺寸。登记 `V51-F20`。
- r007 status/events/resolved/resource-samples SHA=`da279515.../365d212b.../d08a5f96.../8b2ca135...`；run 保留不复用。
  v3/r008 只冻结三层尺寸并强化 observed/expected error，其他合同不变。
- loader 基础设施 materialize image/mask/LiDAR=true，但 runner RGB/LiDAR/membership consumption=false；不把 loader I/O
  冒充 feature/quality read。`V51-F19 resolved`，`V51-F20 active recovery preregistered`。

## V5.1 Stage B r006 blocked / r007 environment recovery 预注册（2026-08-17）

- r006=`20260817T152900Z__m1-stage-b-one-view-contribution-s20260814-r006`，source=`5e59443`，status=`blocked`，
  reason=`ModuleNotFoundError: No module named 'pytorch3d'`。阻塞点在 import，dataset/trainer/renderer/intersection 均未启动，
  没有 denominator 或质量数值。
- r006 status/events/resolved/resource-samples SHA=`06b74ec9.../914fa591.../1b2cb043.../f6157f3f...`；run 仅 4 个
  terminal/config/resource 文件，保留且不复用。failure delta=`V51-F19`。
- v2 只把 runtime 从 `motionproj` 改为 frozen `drivestudio` interpreter，并绑定 torch/CUDA=
  `2.1.2+cu118 / 11.8`、required imports=`pytorch3d,gsplat`；预检 import 已通过。planned recovery r007 使用新 run ID。
- denominator、view、checkpoint、resource 与 no-quality locks 全部逐字继承 v1；不能借恢复改 floor/场景/分辨率或输出。

## V5.1 Stage B one-H-view renderer contribution 预注册（2026-08-17）

- planned r006=`m1-stage-b-one-view-contribution-s20260814-r006`；scene/frame/camera=`0471/0/0`，image index=`0`、
  size=`1600×900`、SHA=`093d38e8...5819e`。formal base r027 summary/checkpoint/source config SHA 逐项继承，expected
  Gaussian=`859,613`。
- config/runner/test=`configs/worldsim_v51/stage_b_one_view_contribution_v1.yaml`、
  `scripts/smoke_worldsim_v51_one_view_contribution.py`、`tests/test_worldsim_v51_contribution_inventory.py`。
  contribution=`alpha×T_before_alpha`，intersection floor=`1e-4`、Gaussian-view mass floor=`1e-3`。
- frozen outputs 仅为 row/mass、两级 support、全局 Gaussian/pixel coverage、quantiles、resource 与 checkpoint
  immutability；不持久化 intersection rows。dataset 可物化 image tensor，但 RGB/LiDAR values 不消费；membership proxy、
  DINO、PCA、feature uplift 与 quality metrics 均不消费/不计算。
- resource ceiling=`12,288 MiB NVIDIA/Torch reserved / 48 GiB cgroup / 900 s`；DINO concurrent=false。
  validation/test/KITTI=false，M2/M3=pending；failure refs 已绑定，预注册 delta=`pending`。

## V5.1 Stage B r005 canonical synthetic operator result（2026-08-17）

- run=`20260817T151900Z__m1-stage-b-operator-parity-s20260814-r005`，source=`1efa7dd`，status=`done`；
  LUDVIG commit/tree/license exact，不 vendor；checkpoint before/after SHA=`746ecb8c...a283` exact。
- parity：dense oracle B0/B1 max error=`0/0`；constant=`0`；lazy bilinear=`1.1920929e-7≤2e-6`；row/chunk
  order bit-exact；B0/B1 difference L2=`0.0829221>1e-4`。全部 11 checks PASS。
- support fixture=`240 input / 173 intersection-supported`，Gaussian-view=`8 before / 7 kept / 1 dropped`，covered=
  `4/5`；两级 frozen floor 与 zero denominator 都被真实触发，不是 vacuous pass。
- summary/status/manifest/fingerprint/resolved/metrics/events/parity-report SHA=`d15b82d1.../5683bf42.../0fc3fe51.../`
  `340c6b83.../f95b7a03.../c741d167.../0ec9e5bf.../c0a4319c...`；manifest=`6 files / 12,521 bytes`、
  run=`14,025 bytes`，独立 recheck exact。
- result freeze=`configs/worldsim_v51/stage_b_operator_parity_freeze_v1.yaml`；failure delta=`none`。DINO/PCA/renderer/
  real feature/quality/validation/test/KITTI=false，M2/M3=pending；下一门只做一个 H view contribution inventory。
- result-freeze regression 首轮=`1 failed / 19 passed`，原因是测试变量 `validate_freeze→freeze` 重命名漏改两条断言；
  run hash/assert 已先通过。修复仅改测试名，登记 `V51-F18 resolved`；r005 immutable summary 的 delta 仍为 `none`。

## V5.1 Stage B synthetic B0/B1 operator parity 预注册（2026-08-17）

- planned r005 suffix=`m1-stage-b-operator-parity-s20260814-r005`；source operator provenance 固定为 LUDVIG
  `4461fc515439bb498a75d71738a1e73cf7a452ed`、tree=`4d1287b5...fb70d`，non-commercial license；external checkout
  clean，project 不 vendor 上游源码。
- 上游 `utils/solver.py::uplifting()` + `apply_weights.cu` 的 faithful denominator 是跨所有 view/pixel 的
  `sum(alpha*T)`；numerator 是 `sum(feature*alpha*T)`。B1 原样实现 normalized transpose；B0 在同一 filtered
  Gaussian-view support 上使用 `1-exp(-mass)` 后跨 view 聚合。optional pruning=false。
- thresholds/dtype 固定为 intersection `≥1e-4`、Gaussian-view mass `≥1e-3`、epsilon=`1e-8`、float64 accumulator、
  float32 output；canonical group ordering 必须让 row/chunk permutation bit-exact。lazy sampler 必须与
  `torch.interpolate(mode=bilinear,align_corners=False)` 全 pixel dense map 在 `2e-6` 内一致。
- config/module/runner/tests=`configs/worldsim_v51/stage_b_operator_parity_v1.yaml`、
  `motion_proj/worldsim_v51/feature_uplift.py`、`scripts/audit_worldsim_v51_stage_b_operator_parity.py`、
  `tests/test_worldsim_v51_feature_uplift.py`。checkpoint 在 formal 前后完整 SHA exact；failure delta 预注册=`pending`。
- DINO/PCA/sidecar/renderer/真实 image feature/method quality/H/S/C/validation/test/KITTI 均为 false；M2/M3=`pending`。
  synthetic PASS 只解锁一个 H-view contribution denominator smoke，不直接解锁 H quality。
- pre-formal regression 首轮=`2 failed / 8 passed`，共同根因是 below-view-mass fixture 的 24×`1e-4=0.0024` 实际
  高于 `0.001` floor；不是 operator parity 失败。修复只把该组改成 5×`1e-4=0.0005`，阈值、公式和 oracle 不变；
  formal r005 尚未创建。failure delta=`V51-F17 resolved`。

## V5.1 Stage B DINOv2 ViT-g r004 canonical resource result（2026-08-17）

- canonical run=`20260817T150400Z__m1-stage-b-dinov2-resource-smoke-s20260814-r004`，source=`935d2b2`，
  status=`done`；source/checkpoint/input identity exact，official ViT-g params=`1,136,486,912`，state-dict keys=`568`，
  strict missing/unexpected=`0/0`。
- last-four normalized reshape outputs=`4×[1,1536,64,114]`，dtype=`float32`、selected finite；selected mean/std/min/max=
  `0.0111025693 / 1.3353331089 / -14.2739315 / 17.1518726`。该诊断不含 PCA、uplift、renderer 或质量 metric。
- resources PASS：GPU start/sampled peak=`1/6,702 MiB`，Torch peak allocated/reserved=`6,067.956/6,376 MiB`，
  cgroup peak=`15,701,860,352 bytes`，samples/errors=`49/0`；cleanup allocated/reserved=`8.125/48 MiB`。
- summary/status/manifest/fingerprint/resolved/metrics/events/resource-samples/diagnostics/resources SHA=
  `27ae3bd2.../97f4dccc.../fc8cf1ab.../d99e0590.../517e4f18.../ae236948.../5f9e083d.../d569c278.../`
  `59d9dbc0.../5c6f74fe...`。manifest=`8 files / 23,039 bytes`、run=`24,854 bytes`，独立 recheck exact。
- result freeze=`configs/worldsim_v51/stage_b_dinov2_resource_freeze_v1.yaml`；failure ledger delta=`V51-F12 resolved`。
  quality/screening/confirmation/validation/test/KITTI=`false`，M2/M3=`pending`。下一门仅 synthetic B0/B1 operator parity。

## V5.1 Stage B DINOv2 ViT-g one-image resource smoke 预注册（2026-08-17）

- task=`WS-V51-M1-B-LUDVIG-UPLIFT-01`；计划 canonical run suffix=
  `m1-stage-b-dinov2-resource-smoke-s20260814-r004`。本门只验证 faithful official ViT-g 在 RTX 3090 上可装载、
  严格匹配权重并生成预期张量，不产生 paper-method quality 结论。
- source identity：external checkout=`/root/autodl-tmp/third_party/dinov2-v51-stage-b`，origin=
  `https://github.com/facebookresearch/dinov2.git`，commit/tree=`7764ea0f...25fc8 / 2a27257b...12b3f43`，
  worktree clean；LICENSE/hubconf SHA=`600cc67c...b7b2 / c1f5090e...a6f64`。
- asset identity 继承 `configs/worldsim_v51/stage_b_dinov2_asset_freeze_v1.yaml` SHA=`dfa17a4a...9f9fd`；checkpoint=
  `/root/autodl-tmp/models/dinov2/dinov2_vitg14_reg4_pretrain.pth`，bytes/SHA=
  `4,546,140,349 / 746ecb8c...a283`。唯一输入 scene-0471/index-382/frame-0/camera-0 的 bytes/SHA=
  `99,906 / 093d38e8...5819e`，原图 `1600×900`。
- preprocessing/model contract：bilinear resize 到 `1596×896`、ImageNet mean/std；official
  `dinov2_vitg14_reg`，patch=`14`、register tokens=`4`、raw dim=`1536`；strict `weights_only` CPU load，
  FP32 参数 + FP16 autocast inference，last-four `norm=true/reshape=true` 输出必须逐层 exact 等于
  `[1,1536,64,114]` 且 selected output finite。
- resource contract：GPU start used `≤2,048 MiB`、sampled/torch peak `≤22,528 MiB`、cgroup peak `≤80 GiB`、
  interval=`0.5 s`、timeout=`900 s`；single DINO process，renderer 不启动。禁止 smaller-model/resolution fallback。
- config/runner=`configs/worldsim_v51/stage_b_dinov2_resource_smoke_v1.yaml` /
  `scripts/smoke_worldsim_v51_dinov2_resource.py`。PCA、feature persistence、method、quality、screening/confirmation、
  validation/test/KITTI 均为 false；M2/M3=`pending`。failure refs=`V51-F12/F14/F15/F16`，delta 在 terminal 后填写。

## V5.1 Stage B 授权迁移 / formal freeze 预注册（2026-08-17）

| Task | 状态 | 当前分母 | 本次允许 | 质量锁 |
|---|---|---|---|---|
| `WS-V51-M1-B-LUDVIG-UPLIFT-01` | running | H/S/C=`3/2/3`；images=`240`；base checkpoints=`8` | authorization migration + input SHA freeze | H/S/C quality 在 r001 均不读；validation/test/KITTI 锁定 |
| `WS-V51-M2` | pending | 0 | none | locked |
| `WS-V51-M3` | pending | 0 | none | locked |

- explicit authorization 选择 normative plan §10.8 的 `U2/B3 fallback` 分支并解除 Stage B 独立授权锁；旧 P0、Stage A
  与 `draft_freeze_only_not_authorized` proposal 均保持原字节，新的 executable overlay=
  `configs/worldsim_v51/stage_b_authorization_v1.yaml`。`V51-F11` 由 governance active 转为 resolved，不倒写 r007。
- route order 预先冻结为 LUDVIG uplift/semantic graph→progressive propagation→super-primitive/anchor→Gaussian
  Grouping→Trace3D→BKI/graph-free；所有 arm 保留 `U2/B3` matched baseline。paper faithful port 失败时写 immutable
  terminal + unified failure ledger 并进入下一路线；通过后才允许机制创新。
- formal r001 runner=`scripts/freeze_worldsim_v51_stage_b.py`，只读并哈希 240 张固定 JPEG 与 V5 r027–r034 的 8 个
  checkpoint，验证 image bytes/dimension、scene/frame/camera、checkpoint SHA、Gaussian counts 与 validation/test
  未读字段。预注册 failure refs=`V5-F20/F22/F23/F24/F26/F29/F31`、`V51-F08–F15`；本次治理 delta=`V51-F11`。
- r001 不下载 checkpoint、不启动 DINO/renderer、feature extraction、quality read 或 parameter search。通过后另以
  资产下载/完整 SHA terminal 冻结官方 DINO checkpoint；H→S→C 的 gate 与 PCA/operator 字段继续逐字继承 proposal。

### r001 canonical result

- run=`/root/autodl-tmp/runs/worldsim_v51/WS-V51-M1-B-LUDVIG-UPLIFT-01/20260817T141000Z__m1-stage-b-input-freeze-s20260814-r001`；
  source=`22149613b9fc958b2bb5351300dd53fdc0d3d221`；status=`done`；conclusion=
  `stage_b_authorized_u2_b3_fallback_and_input_identity_frozen`。
- image denominator=`240 files / 39,747,172 bytes`，record-chain SHA=`247e220f...54ab4`；8 个 scene 的尺寸、
  scene/frame/camera、逐文件 SHA 均 exact。checkpoint denominator=`8`，record-chain SHA=`4c7e5eec...15c0d`；
  V5 r027–r034 checkpoint SHA 与 Background/RigidNodes counts 全部 exact。
- summary/status/manifest/fingerprint/resolved/metrics/events/image-manifest/checkpoint-manifest SHA=
  `f6aae6f6.../8b4c9aec.../8c50882e.../88a4fa17.../3e18161b.../95dd03a9.../0428475a.../be19da2e.../`
  `8b84bf9a...`；manifest inventory=`7 files / 123,976 bytes`，run total=`129,809 bytes`，二次逐文件复核通过。
- authorization config SHA=`34fc22ad...78e0a`；machine freeze=
  `configs/worldsim_v51/stage_b_input_freeze_v1.yaml`。download/model/feature/quality/validation/test/KITTI=
  `false/false/false/false/false/false/false`，M2/M3=`pending/pending`；failure delta=`V51-F11`。

### DINOv2 asset download 预注册

- config=`configs/worldsim_v51/stage_b_dinov2_download_v1.yaml`，绑定 input freeze SHA=`16aafac7...7eb61`；runner=
  `scripts/fetch_worldsim_v51_dinov2_asset.py`。仅允许 official URL→固定 target，使用 network turbo、curl resume、
  5 retries、`.partial` staging，禁止覆盖尺寸错误的已有 final。
- expected bytes=`4,546,140,349`，下载后保留至少 `10 GiB` free；multipart ETag 不作内容哈希。只有完整 bytes 与
  SHA-256 双重验证后才 atomic publish。formal run 保存 resolved config、download log、JSONL events/metrics、asset
  record、fingerprint、manifest、summary/status。
- model load/feature/method/quality/validation/test/KITTI/GPU 均未授权于本 run；M2/M3=`pending`。failure refs=
  `V51-F11–F15`，预注册 delta=`pending`；网络或 partial 失败只作工程 terminal，保留 partial 后使用新 run ID resume，
  不得写成 DINO/LUDVIG 方法失败。

### r002 single-connection blocked / parallel recovery 预注册

- r002=`20260817T141600Z__m1-stage-b-dinov2-asset-s20260814-r002`，source=`2c92061`；在 turbo proxy 下约
  `106 s` 后 prefix=`26,566,656/4,546,140,349 bytes`，持续吞吐不足。执行者只终止 PID identity exact 的 curl；
  runner 自行写 `blocked / curl exit=-15 / partial_retained_for_resume=true`，final 不存在。
- r002 status/events/metrics/download-log/resolved SHA=`e5f3273c.../96741d85.../bf604adc.../46a2950c.../8997fd3a...`；
  prefix SHA=`934ef5aa...e2265`。failure delta=`V51-F16`，分类为工程/资源恢复，不进入方法分母。
- recovery config=`configs/worldsim_v51/stage_b_dinov2_download_parallel_v1.yaml`，runner=
  `scripts/fetch_worldsim_v51_dinov2_asset_parallel.py`。冻结 prefix 后把剩余 bytes 切成 14 个无 gap/overlap HTTP ranges；
  每段必须 range bytes + SHA exact，最终 assembly 同时校验 full SHA-256 与 S3 multipart ETag（part size=`8 MiB`、
  count=`542`、expected=`3d1b...-542`）。通过后 atomic publish 并只清理精确 prefix/segment staging。
- recovery 仍使用 official URL + network turbo；不换镜像/模型/分辨率。model/feature/method/quality/GPU 均为 false，
  validation/test/KITTI 锁定，M2/M3=`pending`。

### r003 canonical parallel asset result

- r003=`20260817T142400Z__m1-stage-b-dinov2-parallel-s20260814-r003`，source=`de6221f`，status=`done`，
  parallel download=`1504.934575 s`。14 segment 的 range bytes/SHA 全 exact；prefix=`26,566,656 bytes`。
- published asset=`/root/autodl-tmp/models/dinov2/dinov2_vitg14_reg4_pretrain.pth`，bytes=
  `4,546,140,349`，SHA-256=`746ecb8c6301c645c5c855be91687d274587d6e48fdaec4a729753160b34a283`；
  local multipart ETag=`3d1b1c4501eac45d83af24b811e3bea9-542`、part=`8,388,608 bytes × 542`，与 official exact。
- atomic publish 后清理 prefix+14 segments=`15 files / 4,546,140,349 bytes`；final free=`85,562,351,616 bytes`。
  summary/status/manifest/fingerprint/resolved/metrics/events/asset/segments SHA=`6e79804a.../16cc40ed.../98ce7751.../`
  `06e34644.../dccb1fce.../f5acb8c7.../2c2ef11b.../b4913ccf.../459e698e...`；manifest=`21 files /
  961,129 bytes`，run=`969,151 bytes`，独立 terminal/asset ETag/manifest recheck exact。
- machine freeze=`configs/worldsim_v51/stage_b_dinov2_asset_freeze_v1.yaml`；model load/feature/method/quality/
  validation/test/KITTI=`false`，M2/M3=`pending`。failure delta=`V51-F16 resolved`；下一门仅 source freeze + one-image
  ViT-g resource smoke，不直接读取 H quality。

## V5.1 Stage A S one-shot screening / closeout（2026-08-17）

| Scene | Eval views | A1 ΔBF1 | A1 ΔIoU | A1 ΔFN | A1 ΔBrier | A1 ΔECE | A2 coverage | accepted error | abstained error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0998 | 12 | -0.0000904944 | +0.000997358 | +0.000353190 | -0.000307524 | -0.000331509 | 0.250105 | 0.0289614 | 0.209930 |
| 0359 | 9 | +0.0000574359 | +0.000315630 | +0.000367882 | -0.000011430 | -0.000036688 | 0.864765 | 0.0007217 | 0.0588556 |
| scene-balanced | 2 | -0.0000165293 | +0.000656494 | +0.000360536 | -0.000159477 | -0.000184098 | 0.557435 | 0.0148416 | 0.134393 |

- canonical r007=`/root/autodl-tmp/runs/worldsim_v51/WS-V51-M1-A-UNARY-OBSERVABILITY-01/20260817T140000Z__m1-stage-a-s-screening-s20260814-r007`；
  source=`dc24f28e1de21b0fb5d1cbb41c959c3d51624a38`，duration=`254.024654 s`，peak GPU=`8393 MiB`。
- 输入链：SAM r047/r048 → immutable B3/evidence r049/r050 → single candidate r007；accepted evaluation denominator=
  `12+9=21`。A1 conditional 与 A2 conditional 共用同一 render；A2 只额外渲染 UNKNOWN，未重复 candidate/scene 分母。
- A1 gate：BF1 nonnegative=`1/2<2`、clearly positive=`0/2<1`、mean BF1<0，FAIL；FN/Brier/ECE caps 通过不改变
  合取裁决。A2 selective：两场 error concentration 均成立，但 mean coverage=`0.557435<0.60`，FAIL；A2 还继承 A1
  conditional FAIL。final survivor=`U2_B3`，Stage A 不再继续 Bayesian family。
- 0998/0359 UNKNOWN Gaussian ratio=`34.5512%/0.7891%`，只登记为场景依赖诊断，不据此事后调阈值。两 checkpoint
  前后 exact；parameter search、C/validation/test/KITTI read/tuning 均为 false。
- summary/status/manifest/fingerprint/diagnostics SHA=`094b4ae1aa9e35830952ee9bfb5cb03a2cb990cf43f79a1f63a23ba28f78e20c /`
  `03d319da5e876c78f3ca14ca83d39fc990882caacc048c0d01c1ba520a50a993 /`
  `7f60d0974fa0ab0ab472bb95b0eadc1a453e7f58dbddd7aff7553b8308f69a0e /`
  `67f1001d81b77af2c733cc9163b20cf3442c55ff773e55667e8aa51af2e8f6c6 /`
  `15834f48f3f3cb1b996a208c9c3d4bf6bab696c160d433fda8210b52153f6e67`；failure delta=`V51-F09/F10`。

## V5.1 Stage A A4 CIF-lite identifiability audit（2026-08-17）

| Check | 0471 | 1087 | 0379 | 结论 |
|---|---:|---:|---:|---|
| Independent occupancy field in A2 | 0 | 0 | 0 | unavailable |
| Constant occupancy=1 vs existing renderer | bit exact | bit exact | bit exact | no-op |
| Reuse appearance opacity as occupancy | non-exact | non-exact | non-exact | double-count alpha / forbidden |

- canonical r006=`/root/autodl-tmp/runs/worldsim_v51/WS-V51-M1-A-UNARY-OBSERVABILITY-01/20260817T122000Z__m1-a4-cif-identifiability-audit-s20260814-r006`；
  source=`cee8b66849e5c556a79e05c813d45f225efa7814`，duration=`0.323701 s`，conclusion=
  `a4_cif_decoupling_rejected_no_independent_occupancy_observable`。
- CIF official contract 区分 occupancy、conditional instance distribution、visibility 与 appearance opacity；V5.1 计划又
  禁止完整 learned deformable field、identity calibration 与 resampling。用 visibility/count 估 occupancy 会重建 A1 已修复的
  missingness conflation；constant one 则严格 no-op，故没有可归因 A4 quality arm。
- r006 未读 evaluation artifact/quality、未运行 GPU/training/search。summary/status/fingerprint/manifest/diagnostics SHA=
  `bb87357a834af3e2b2b9956ad193602a18961992cce27dc6c9486e7798087f18 /`
  `a7eeef6c14cef04d2964ba6df8969bbc3c144ab6e2abc5f570b3db50d8fe2e6c /`
  `ae6953d2108c2648d8f6a447d56a6649637ab83e8d059962098d43621e0bf9cd /`
  `072ad60ae2525b1a3c4cc6869fc4f7ca58ab765e2268902713d543ddf4718c30 /`
  `ba43869b4429b71aa513040af577c5cbea6f5fef09558c2b29605b8d14318aad`；failure delta=`V51-F07`。

## V5.1 Stage A A3 Kish effective-count pre-quality audit（2026-08-17）

| Run | 状态 | Observed Gaussians | Meaningful cap change | Replacement amplified | Quality/GPU |
|---|---|---:|---:|---:|---|
| r004 v1 | done / inconclusive | 944,443 | 相对门被 subnormal denominator 污染 | 940,762 | none / none |
| r005 v2 | done / mechanism rejected | 944,443 | 0 above absolute `1e-9` | 940,762 (99.6102%) | none / none |

- canonical r005=`/root/autodl-tmp/runs/worldsim_v51/WS-V51-M1-A-UNARY-OBSERVABILITY-01/20260817T120000Z__m1-a3-effective-count-audit-v2-s20260814-r005`；
  source=`150b0721cb0e0acf630846d815a7ab2f287ceea8`，duration=`4.219242 s`。
- r005 对 0471/1087/0379 的 A2 parent effective count 全部 float32 exact；无 epsilon 的 Kish count 在
  `944,443/944,443` Gaussian 上均不低于 A3-0 fractional concentration。作为 cap 没有有意义变化，直接 replacement
  则在 `99.6102%` Gaussian 上提高 concentration；formula 不含 correlation observable。
- r004 最大 absolute cap reduction 只有约 `2.5e-13`，但 float32 最小次正规 reliability=`1.401298e-45` 使相对量
  达 1.0；v2/r005 绑定旧 run/config SHA 后修正 audit endpoint，详见 `V51-F06`。两轮均未读 evaluation artifact/quality、
  未启动 renderer、未搜索参数；A3 机制 rejected，不产生质量臂。
- r005 summary/status/fingerprint/manifest/diagnostics SHA=`9e599c5f1d2efe4cfc6f6e92d6ce234d0d36e9c564d445348d622a07bb753987 /`
  `1cde89f295b6b60f619323d055973192b1879aca8378eba12a4a42da32dea091 /`
  `7e6c728b068f6f36d152b83434c551ca50eca334617fa38c4278e39e0bdeacee /`
  `2f3d7f017e99a03cacd2f5e6cf70aadfcf89ba4436b249373fc0a04d5ad15f1c /`
  `a147843a0f3f8167822fa1e26aceb6122d32a00d98cf34c42bb9ff00af29e32a`；failure delta=`V51-F05/F06`。

## V5.1 Stage A A2 semantic UNKNOWN / ABSTAIN（2026-08-17）

| Scene | Eval views | Gaussian UNKNOWN | Coverage | Error@coverage | Accepted abs error | Abstained abs error | UNKNOWN recall on errors |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0471 | 8 | 20.4240% | 46.4865% | 0.038468 | 0.040882 | 0.251780 | 86.8498% |
| 1087 | 1 | 0.1749% | 73.6836% | 0.001892 | 0.003579 | 0.169013 | 95.9160% |
| 0379 | 3 | 0.2478% | 95.8186% | 0.000143 | 0.000213 | 0.071956 | 87.5421% |
| scene-balanced mean | 3 scenes | — | 71.99625% | — | 0.014891 | 0.164250 | — |

- canonical r003=`/root/autodl-tmp/runs/worldsim_v51/WS-V51-M1-A-UNARY-OBSERVABILITY-01/20260817T113000Z__m1-a2-unknown-h-s20260814-r003`；
  source=`7e783f1fe04cc05cdd206b56086ef0f02a4215ee`，seed=`20260814`，duration=`192.314456 s`，
  peak GPU=`8393 MiB`。
- 阈值总体/规则/图像阈值在读取 r003 quality 前冻结为 positive-count H/A1 pooled Gaussian、
  `Q25 count / Q75 entropy / Q75 disagreement`、`high entropy AND (low count OR high disagreement)`、`0.5`；
  三个 posterior 输入逐 SHA 绑定，阈值从 `944,443` 个 Gaussian exact reproduction。
- A1 conditional rerender=`12/12 byte exact`，conditional metrics A2−A1=`0`，checkpoints=`3/3 exact`。A2 的传统
  BF1/IoU/calibration gate 因保持 A1 conditional posterior 而继续通过；selective gate 的 scene-balanced coverage=
  `0.7199625`，abstained−accepted error=`+0.14935825`，全部场景同时存在 accepted/abstained denominator。
- 0471 coverage=`0.4648653`，未达到逐场 60%；冻结合同只要求 scene-balanced mean 60%，故不倒写 r003 PASS，
  但不得宣称 uniform coverage。1087 只有 1 个 evaluation view；A2 仍是 H-only candidate，不读 S/C/validation/test/KITTI。
- summary/status/fingerprint/manifest/diagnostics SHA=`c9a821395da41b09fa124971c3f3e4e6f702987a3a587b208650294f85ae53b5 /`
  `cb67b786ab8d8918d54c1c73907eaa51dd8ee185c63ac81670b11ca03cd0a7c0 /`
  `2c5f06004f82c192c1b1b2893c643813cadcd08b00860240af6d0da20c9f01d3 /`
  `56bf02070de0519946513a0b71404814ae79ee008ca7b5109e4e1c82f68b6c52 /`
  `206faa66c393f1a9d95db75490ade57803dfbf2724daa2edce91e2699db8c9ff`；failure delta=`none`。

## V5.1 Stage A A1 visibility-masked B3（2026-08-17）

| Scene | Eval denominator | Valid obs ratio | ΔBF1 | ΔIoU | ΔFN mass | ΔBrier | ΔECE |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0471 | 8 | 0.857484 | +0.000020836 | +0.000124291 | +0.000484780 | -0.000044421 | -0.000436149 |
| 1087 | 1 | 0.967938 | 0.000000000 | 0.000000000 | +0.000000529 | -0.000000729 | +0.000010298 |
| 0379 | 3 | 0.826925 | +0.003446302 | +0.001256639 | +0.002831752 | +0.000003233 | -0.000008710 |
| scene-balanced mean | 3 scenes | — | +0.001155713 | +0.000460310 | +0.001105687 | -0.000013972 | -0.000144854 |

- canonical r002=`/root/autodl-tmp/runs/worldsim_v51/WS-V51-M1-A-UNARY-OBSERVABILITY-01/20260817T104000Z__m1-a1-visibility-h-s20260814-r002`；
  source=`38bc9b44c6c86d58173930aa019745b8a9a8e00b`，seed=`20260814`，duration=`193.228059 s`，
  peak GPU=`8382 MiB`。
- A1 唯一变量为 hard visibility eligibility；configured/applied threshold=`0.01/0.009999999776482582`。
  阈值来自 H evidence distribution，quality read=false；未引入 UNKNOWN/effective-count/CIF/feature/graph。
- B3 GPU rerender=`12/12 byte exact`，B3 aggregate metric replay delta=`0`，checkpoint pre/post=`3/3 exact`。
  A1 H gate 五项全部通过，BF1 positive scenes=`2/3`；正式结论仅为 candidate，未读 S/C/validation/test/KITTI。
- summary/status/fingerprint/manifest SHA=`74246312e257b39edaba72b6750f68a3d61f8f5933a142c1063e759cf0dc2a79 /`
  `c4dbac0d444fecc6f8154105122d2b8b811d283989a12ffcec80a31ff4691b73 /`
  `3cca740c69d4ed33c3949d3d5b07ebf6a17c294af6e6d2366401fe540f665cd9 /`
  `fbec5f19e2a29547c2538aad0be59ee225a9973bb57d41aa706141a57377e092`。
- `failure_ledger_refs`：`V5-F20–F26/F29–F32`、`V51-F01–F03`；`failure_ledger_delta=none`。

## V5.1 Stage A A0 V5 unary exact replay（2026-08-17）

| Run | 状态 | 实际重算分母 | Exact gate | 结论 |
|---|---|---:|---|---|
| r001 A0 replay | done | 3 H scenes；45 observation files；9 arm×scene | 54 array groups=`0 mismatch`；54 Gaussian metric deltas=`0.0` | A0 passed；只解锁 A1 |

- canonical=`/root/autodl-tmp/runs/worldsim_v51/WS-V51-M1-A-UNARY-OBSERVABILITY-01/20260817T102000Z__m1-a0-v5-unary-replay-s20260814-r001`，
  source=`1e2361658b85e1f12145867164238ce81ecb55ea`，config SHA=`ba8fdcaa...`。
- r037/r042/r043 每场 B0/B1/B3 的 `unary_posterior/unary_uncertainty/effective_evidence_count/`
  `multi_view_disagreement/boundary_ambiguity/depth_support` 均逐 bit exact；Gaussian Brier/ECE/IoU/NLL/FP/FN
  按原实现重算且逐 float delta=`0.0`。
- 2D evaluation 本轮未重新执行 GPU renderer；证据边界是 canonical evaluation artifact bytes、manifest 与每场 12 个
  核心 generation source SHA exact。该边界不影响 A0 Bayesian 累积逻辑复现，但不能被写成新的 2D quality run。
- summary/status/fingerprint/manifest SHA=`b9b33bbd8304bb184e9388b4c102a49236a30ded1ae7d10c071cfc9859914878 /`
  `5d695add5efb535da91da619f520f6a3f7c9b78b717e612e03422279e435e432 /`
  `6c7466b3a2f8dfb4fe905841fcfe91bce230200862fba56e843811a237d700ae /`
  `c4a5e4fdb7189cbfca0c756365f21ac6d981a6e9c30616548870c204c3103d3d`。
- `failure_ledger_refs`：`V5-F20–F26/F29–F31`、`V51-F01/F02`；`failure_ledger_delta=none`。
  无 method inference、parameter search、validation/test quality read 或 KITTI tuning。

## V5.1 M1-only P0/D0 start audit（2026-08-17）

| Task / Run | 状态 | 冻结分母 | 结果 | 下一门 |
|---|---|---|---|---|
| `WS-V51-P0-M1-SCOPE-FREEZE-01` r001 | done | M1-only scope；V5 parent `44d0e4a` | scope/授权/quality locks exact；M2/M3 pending | P0 closed |
| `WS-V51-D0-DEV-ROLE-FREEZE-01` r001 | done | H/S/C=`3/2/3`；validation/test=`8/20` | 原 V5 cohort 顺序 exact；validation/test quality unread | D0 closed |
| `WS-V51-M1-A-UNARY-OBSERVABILITY-01` A0 binding | running | r037/r042/r043=`159 files / 680,254,598 bytes` | canonical inputs、manifest inventory 与 checkpoint identity exact | posterior/metric exact replay |

- canonical=`/root/autodl-tmp/runs/worldsim_v51/WS-V51-P0-M1-SCOPE-FREEZE-01/20260817T101000Z__p0-start-audit-s0-r001`，
  source=`58953a57557b97f449c4d83db7d11132ddda5e73`；summary/status/fingerprint/manifest SHA=
  `6d495ce26c211843e69dd9034dccfc916f17311dc59edaf5e7115ed32723ef9c / 8a724b06563ff1cc4181f0760db9dc0013fc9897d7a38a3d3bdc08005fd1bd93 /`
  `b52b63d342034fa9c2fabe858ad0f1d18d5ee6d67e9c67472e82c725aa643958 / 8ab0ad66eddedece7cfe6db4871172b07ae2c80430c8ddba156df76ce2941dc5`。
- `failure_ledger_refs`：`V5-F09/F11–F14/F18/F20–F26/F29–F33`；`failure_ledger_delta`：start audit=`none`，
  实现阶段新增并已解除测试入口工程失败 `V51-F01`。
- 首轮窄测最初在 collection 阶段因 repo root 未注入而 blocked；修复后
  `pytest -q tests/test_worldsim_v51_protocol.py tests/test_audit_worldsim_v51_start.py`=`4 passed`，
  `python scripts/audit_worldsim_v51_start.py --help` 通过。无训练、方法推理、parameter search、
  validation/test quality read 或 KITTI tuning。

## V1–V5 failure ledger 治理登记（2026-08-17）

| Task ID | 状态 | 输入范围 | 结果 | 研究边界 |
|---|---|---|---|---|
| `DOC-FAILURE-LEDGER-01` | done | `AGENTS.md`、docs 导航、V1–V5 失败/风险历史与 archive 证据 | 唯一 live ledger；V1=`F01–F06`、V2=`F01–F09` 汇总；V4 live=`F01–F49` 唯一连续；commit=`4e512d9` | 没有训练、推理、quality/data read、split/seed/fingerprint 或科学结论变化 |

- `failure_ledger_refs`：V1 frozen archive、`PIVOT-F03/F14B/F15/F16`、V2 注册表、现有 V3/V4/V5 条目。
- `failure_ledger_delta`：登记 V1/V2 紧凑 canonical 条目，校正 V4 重复编号并保留 historical→live 映射；
  没有新增方法质量结论，也没有删除或解除既有失败。
- 治理验证：live definition bullets=`207`、duplicate IDs=`0`、V4 IDs=`49/49` contiguous、显式导航 anchors=`6`、
  `docs/` backup files/directories=`0/0`；`git diff --check` 通过。
- 后续实验在 plan/config/run metadata 写 `failure_ledger_refs`，结束时在本台账写 `failure_ledger_delta`；若无新增写
  `none`，若出现失败、推翻或解除则同一逻辑提交更新 `docs/RESEARCH_FAILURES.md`。

## V4 文档归档/存储维护（2026-08-17）

| Task ID | 状态 | 输入范围 | 结果 | 研究边界 |
|---|---|---|---|---|
| `WS-V4-DOC-CLEANUP-02` | done | V4 final archive、两个 scratch `tmp`、旧 `mnt` staging、`docs/` 编辑恢复副本 | V4 SHA=`78/78 OK`；`tmp`=`0/0 entries`；`mnt=absent`；`docs backup=0`；commit=`3598ef7a` | 无训练/推理/quality read；不改变 V4/V5 结论或授权 |

- 删除前规模：两个 `tmp` 合计 `9,388,434 bytes / 62 files`；`mnt=126,234,111 bytes / 154 files`；
  `docs` 恢复副本=`233 + 50 files`。V7.1 H1 reject 的 `SHA256SUMS` 已按保留的 7 个 canonical 文档重建并通过。
- 保留 `/root/autodl-tmp/motion_proj/work/codex-backups/`（`1,647 files / 979,529,557 bytes`），因为
  `RESEARCH_FAILURES.md` 仍引用其中的 AD-GS partial scene；该目录不是本次普通 scratch 清理目标。
- 完整清理清单：`docs/archive/2026-08/worldsim-v4-cleanup-2026-08-17/CLEANUP_MANIFEST.md`。

## V5 M3 protocol / clip inventory / trajectory mechanism r001–r005（2026-08-14）

| Run | 状态 | 分母 | 主要结果 | 结论 |
|---|---|---:|---|---|
| r001 protocol audit | done | 8 fresh dev bases | T2=V4 frozen B-spline；T3–T5；REMOVE 不进 physics；V4 aggregate 不复用 | implementation only unlocked |
| r002 clip inventory | blocked | 0 quality reads | config 缺 `protocol_audit.conclusion`，streaming 前 KeyError | 工程 blocked；新 r003 修复 |
| r003 clip inventory | done | 8 scenes | `8 ready + 0 abstain`，各 7 keyframes；annotation metadata-only | trajectory mechanism metrics unlocked |
| r004 mechanism v1 | done | 16 requests | T2/T5 violations=`38/34`；evaluable=`7`；improved=`5`；safe regressions=`2` | insufficient；heading measurement artifact |
| r005 measurement v2 | done | 16 requests | T2 safe/evaluable=`15/1`；T2/T5=`2/1`；reduction=`50%`；endpoint/contact=`16/16` | insufficient；nonconfirmatory replay，无 unlock |
| r006 rejection closeout | done / task rejected | `4 completed + 1 blocked` | renderer/collision/method/validation/test 全 false | `m3_rejected_constraint_projection_not_needed_on_frozen_requests` |

- r005 只修正 heading 可观测性、reverse 语义和 convergence 定义，desired templates、T2–T5、caps、分母与 gate threshold 全部 exact replay。
- r005 summary/status/diagnostics/decision SHA=`56fd2223.../7bd7aba3.../3ba7c2eb.../3e2299e4...`；完整 r001–r005 哈希见 M3 archive metadata。

## V5 M2 cross-view scaffold / formal closeout r012–r015（2026-08-14）

| Run | 状态 | 冻结变量 | 主要结果 | 结论 |
|---|---|---|---|---|
| r012 G4 launch | blocked | 同相机 `±5/±10/±15` source grid；stride-1；12px bounded extrapolation | provenance schema 在首个 asset 拒绝；无 GPU/方法读数 | 工程 terminal 保留，使用既有 `native_scene_donor` 后新 run |
| r013 G4 | done | target-reference-blind source projection；median fusion；G0 fallback | raw/post 改善=`12/22`、`17/22`；raw mean/median delta=`-1.188137/-0.655964m`；absolute-safe=`0/22,0/22` | `g4_cross_view_scaffold_relative_gate_rejected` |
| r014 G5 | done | 三相机、同帧其他相机与 `±5/±10/±15`；其余门槛不变 | raw/post 改善=`15/22`、`19/22`；raw mean/median=`-3.270320/-2.785312m`；absolute-safe=`1/22,0/22` | relative supported，absolute-safe failed；无方法选择 |
| r015 closeout | done / task rejected | 绑定 r001/r007/r010/r012 blocked 与 r004/r005/r006/r008/r009/r011/r013/r014 completed | completed/blocked=`8/4`；router/validation/neural unlock=false | `m2_rejected_no_absolute_geometry_safe_candidate`；next independent task=M3 |

- r013/r014 均为 `23=22 evaluable+1 abstain`，r005 baseline replay=`22/22 exact`；target hole interior reference 不可供 candidate 使用，validation/test/KITTI quality 未读。
- r014 G5 any/direct/extrapolated/fallback coverage mean=`60.40%/15.57%/47.99%/36.44%`；LiDAR projected mean≈`0.8%`。覆盖诊断只解释失败，不授权调参。
- r015 summary/status/fingerprint/manifest/resolved/events/decision-ledger SHA 依次为 `27a6613d...`、`b8f6e66f...`、`0ac2b374...`、`56d1858a...`、`d9117bd8...`、`fb1eb1e6...`、`63b9e36b...`；manifest=`8/8 exact`。完整哈希见 M2 archive metadata。

## V5 M2 geometry-first / Gaussianization r001–r011（2026-08-14）

| Run | 状态 | 正式分母/阶段 | 结论 |
|---|---|---|---|
| r001 staged G0 | blocked | 6 frozen views | 把 2 个 SAM-unavailable view 错当成必须有 mask；无方法结论 |
| r002 staged G0 | done | 4 union-mask evaluable + 2 abstain | raw/pre mean=`21.760946m`，post mean=`6.392995m`；仅 staged diagnostic |
| r003 union G0/G3 | done | 4 union-mask evaluable + 2 abstain | G3 raw improvement=`1/4`；后因 request unit 不等价，只作负证据 |
| r004 per-actor masks | done | `23=22 accepted+1 rejected` | union pixel-exact replay；冻结正式 request unit |
| r005 per-actor G0 | done | `22 evaluable+1 abstain` | raw fail=`22/22`；Gaussianization primary=`16/22` |
| r006 per-actor G0/G1 | done | `22+1` | improvement=`5/22`，mean/median delta=`+3.658565/+2.300282m`，G1 rejected |
| r007 per-actor G0/G2 | blocked | serializer 前已计算，未形成 summary | arm tuple 变量遮蔽导致 `KeyError: 0`；无方法结论 |
| r008 per-actor G0/G2 | done | `22+1` | improvement=`8/22`，mean/median delta=`+3.005506/+1.620793m`，G2 rejected |
| r009 per-actor G0/G3 | done | `22+1` | improvement=`11/22`，mean/median delta=`+0.103693/-1.489037m`，G3 rejected |
| r010 Gaussianization launch | blocked | formal prepare 前 | launcher 预建目录触发 overwrite guard；无 GPU、无方法读数 |
| r011 `2×2` Gaussianization factors | done | `22 evaluable+1 abstain` | BASE exact=`22/22`；DENSE/DENSE_OPAQUE 改善=`20/22、19/22`，OPAQUE=`0/22`；density 机制受支持，未选择方法臂 |

- r004/r005/r006/r008/r009 summary SHA=`e931c09fa4c1f6ca34b9b302bc5902cf9d1b183d77f6eaaeecea1083d27911a5 / b93c4be44c597762a899e46ef4c89c5a25ca1c54e09824ceca4d82c46f18eb15 / 015a507471cb18fa728d98548d1714c8898125e4f165c0f662e797bdfca30fb9 / 906bfc7b569265d6806e25b42bb855352b64b4e1cdb4562fb267aadf72efcf92 / 8a253325f5b689239712b6de3e862ddfd1b1b10adb82ac6583286c755ef0d06c`。
- surface gate 在结果前固定为：`>=18` evaluable、`>=14` 请求改善 `>=0.5m`、mean/median delta 均 `<0`。没有 arm 通过；G0 也不是 safe candidate，只是比较基准。
- r011 机制门固定为 `>=18` evaluable、`>=14` 请求改善 `>=0.1m`、mean/median candidate−BASE 均 `<0`。DENSE−BASE mean/median=`-0.424179/-0.480927m`，OPAQUE−BASE=`+0.059686/+0.065773m`；DENSE_OPAQUE 比 DENSE 又退化 `+0.035533m`。这支持 density 因子，不授权按 r011 选择 stride-1 method。
- reference=`base_background_depth_model_proxy_not_ground_truth`；r005 confidence mean/median=`0.0585/0.0582`，independent GT claim=false。validation/test/KITTI quality 与 router search/refit 均未发生。
- r011 summary/status/fingerprint/manifest/diagnostics SHA=`47a08899b5e82e8297029b58d56aabfc0fee7afd1822d6b498d9a1734de87204 / 5a4a3ebb764afc50e4b715d0968acd3601c279e315bbab809d5780c1eb23d076 / 35338750eb6656c811ec41b33c6955dccd137b1ed3b32ed20f88d5e78343bf5c / acc977a3200af4f211d69dddf5b5b8de609e66e8902db72b8dacfeda60f3cc28 / 234adf931e6b0abc0808b0858e38c40f7b76f119ed8b32881d3a16150231b1cf`；manifest inventory=`31/31 exact`。
- 详细路径和 appendix 表见 `docs/WS_V5_M2_GEOMETRY_FIRST_DEVELOPMENT.md`、`M2_R001_R009_METADATA.json` 与 `M2_R010_R011_GAUSSIANIZATION_METADATA.json`。

## V5 M1B boundary-residual forensic（2026-08-14）

- canonical=`/root/autodl-tmp/runs/worldsim_v5/WS-V5-M1B-D0-BOUNDARY-RESIDUAL-FORENSICS-01/20260814T192757Z__m1b-d0-boundary-residual-forensics-s0-r001`，source=`bb4ebb5ee57f3dfd86b7189cc08e71a77e19e00f`。
- r038/r045/r046 共 `12` 个冻结 evaluation views、`6` 个 scene×unary G0 cells；3px target boundary band 与三项 primary gate 在读取 NPZ 分布前冻结。
- boundary-primary=`0/6`，mean boundary classification-error share=`0.4020948014`，mean boundary semantic-error mass share=`0.2483529331`；结论=`boundary_ambiguity_not_primary_semantic_split_remains_locked`。
- summary/status/fingerprint/manifest/audit SHA=`ddecf415bd71fcf920b6fab5f38ee74edea93aa085b48a73947480c3c186c35d / 65ea0db4a1714b668923bd927ee7664c89bce34e133d3df97f622dbd78371e84 / 1e5605c0056274f03fdb0b3653528d1a2bb1b8801c7d1b190e8e8390e11c2e2b / a22b382d6a02d5821c38bc4fa061946abd13ef2300c05ad7f47051ab7b0b88a7 / fd128a3b19373473b97c07d7c8733c0640f526f338911e4247d8cba7bc515c35`。
- semantic split 未授权、未启动；M1 当前正式收口为 rejected，V5 转入 M2。详见 `docs/WS_V5_M1B_BOUNDARY_RESIDUAL_FORENSICS.md`。

## V5 M1 三场景 result-blind replication（2026-08-14）

| Run | 状态 | 场景/阶段 | 关键分母或失败原因 |
|---|---|---|---|
| r039 | done | scene1087 SAM | `2/30` available，1 actor / 2 accepted boxes |
| r040 | done | scene0379 SAM | `6/30` available，5 actors / 6 accepted boxes |
| r041 | blocked | scene1087 unary attempt | SSH 输出管道关闭后 `BrokenPipeError`；保留、不覆盖 |
| r042 | done | scene1087 unary | `931,223` Gaussians，`1 accepted + 14 abstain` |
| r043 | done | scene0379 unary | `1,187,291` Gaussians，`3 accepted + 12 abstain` |
| r044 | blocked | scene1087 graph attempt | scene0471 的 `8+7` 分母被硬编码；合同 fail-closed |
| r045 | done | scene1087 graph | `5,587,338` edges，G0 exact replay r042 |
| r046 | done | scene0379 graph | `7,123,746` edges，G0 exact replay r043 |

- 复制 cohort 在结果前固定为前三个 development scenes=`0471/1087/0379`；六个 G3-vs-G0 单元中 Boundary F1 正向=`3/6`，低于 `>=4/6`。
- 聚合 mean ΔBoundary-F1=`+0.0016107723`、mean ΔFN-mass=`+0.0025676789`；scene-level topology `G3<G1`=`2/3`。总裁决=`physical_graph_development_replication_rejected_3of6_boundary_support`。
- validation/test/KITTI quality、parameter search、formal arm selection、semantic split/Transformer unlock 均为 false。完整表、run 路径与哈希见 `docs/WS_V5_M1_DEVELOPMENT_REPLICATION.md`，机器可读副本见 `docs/archive/2026-08/worldsim-v5-m1/M1_R039_R046_REPLICATION_METADATA.json`。

## V5 M1 formal30k / SAM / structured unary / physical graph（2026-08-14）

| Run | 状态 | 关键分母 | 审计结论 |
|---|---|---|---|
| r035 `m1-formal30k-batch-audit` | done | `8/8 scenes × 30k` | 8 份 formal run/checkpoint 全量重哈希一致；validation/test quality 未读 |
| r036 `m1-scene0471-sam-sparse` | done | `30 views / 18 accepted` | frozen SAM evidence；17 actors、62/61 prompt/accepted boxes；无网络 |
| r037 `m1-scene0471-unary-diagnostic` | done | `859,613 Gaussians / 15 evidence / 8+7 eval` | B0/B1/B3 机制诊断；checkpoint exact；无搜索、无 graph、无 held-out |
| r038 `m1-scene0471-graph-diagnostic` | done | `859,613 Gaussians / 5,157,678 edges / 8 eval` | B1/B3×G0–G3；G0 exact replay；无搜索、无 validation/held-out |

- r035 canonical=`/root/autodl-tmp/runs/worldsim_v5/WS-V5-M1-STRUCTURED-OWNERSHIP-01/20260814T172300Z__m1-formal30k-batch-audit-s0-r035`；summary/status/fingerprint/manifest SHA=`4a540d24cd8bfa18c9d63cdcbabe08dcded7a2de88de116695e431187cb6738b / 0c941a372125e1f893bc31c15eed2cb55dbfef33fc005bf2b8122edec7626607 / ea3d7f8aadef39c7c99b2cd610091fb20e4a386ba6f8a987d16358eaaee6fb8e / bba0892345225f5d4527402943d17b7806207714c7a9355641eda3b2cda72119`。
- r036 canonical=`/root/autodl-tmp/runs/worldsim_v5/WS-V5-M1-STRUCTURED-OWNERSHIP-01/20260814T172400Z__m1-scene0471-sam-sparse-s0-r036`；summary/status/fingerprint/manifest/mask-manifest SHA=`d66f04a05a5a0ee8fb94b423e20296ec019bc8fe1e56ebc6c57fb1c80495d487 / 7fda172e7fb9e07ab520b563b2e358fb0ac70c734f7c32f2039d901a7be690c4 / 9e8181fad12ac76f0c0ffbd25cf3d39061e3893960d167f1b5497b42e51b79a1 / 3cd9728cd47622752a80d20635e254b05b31fc13126659c7c03d06b7be3fc13e / f7a9f5e9f022c8f8685be89e7cdd7d808f13081106c56b07e5c87260ac72a213`。
- r037 canonical=`/root/autodl-tmp/runs/worldsim_v5/WS-V5-M1-STRUCTURED-OWNERSHIP-01/20260814T173032Z__m1-scene0471-unary-diagnostic-s0-r037`；summary/status/fingerprint/manifest/diagnostics/resolved-config SHA=`dd8b2a9e5f09f130f948c9de2b6b8eaa5bea9ab714278bed7fa56a633dd7a22d / fb68e06d174a5e6a6859a50c07392b6895102feecf3944e712dc9861de1736ce / 70d2f878e9042b632d4f3355cc350d02ae661127c738130019506aa77c4480e4 / 80ff775de6a4fc4c748cf3ec9570c2ab0ce10e817fbe5cdcd1bef69eeee871c4 / 88e256b9f07149cdfbf94da26e7d59b83c2071cb4485e41cf80717f0eac0d755 / a09ac4f9359df36e1c9ff90fdba83da5de5cac519eb52979d6444211df50e291`。
- B1/B3 均相对 B0 提升 scene0471 的 2D IoU、Boundary F1 与三项 calibration，并降低 FP semantic mass；但 FN semantic mass 分别增加 `+0.0915315/+0.0954773`，超过计划 validation 容忍量 `+0.01`。本次只登记 unary 方向支持与 FN tradeoff，不做 arm selection，不把单 scene SAM-proxy 诊断写成 validation 通过。
- r038 canonical=`/root/autodl-tmp/runs/worldsim_v5/WS-V5-M1-STRUCTURED-OWNERSHIP-01/20260814T180451Z__m1-scene0471-graph-diagnostic-s0-r038`；summary/status/fingerprint/manifest/diagnostics/edges/posteriors/resolved-config SHA=`c64e52a9de2a43cbb89564cbf61610746fdec210eb2a4f0efb32bc6463f7faf1 / 2dc9234c7ea2bc5b45f30d22f6626a84a88044537764ccae2983a6338ba30dbf / cbd9619893048592b23fb5149b5eab447e3aefcf642cd2ae3f91cd66cdf518fc / 83ad4099f61bfb0566f38f9fd30f4197897ac825b36b2c0de1d00dddad23f027 / ab81462375a2ea7faec73051aec807808f41426e872f954a249fdee19bfb9d2b / dcf9a846d21068c40a1ebb0b0334f58f3939f6b1df8da80b3184be7ea3956406 / f6dd79138e30f55798c8f16a635b078e98999d6a9db861242b9df6960accd6d3 / 52959aaad0d29175507e815e7657f91ec1a7a4402ab3a25af793513e6ed9e749`。
- G3 在 B1/B3 两个 unary 输入上都取得小幅、同方向 2D 改善，FN 增量均小于 `0.002`；物理 affinity 与 barrier 还把 proxy cross-boundary affinity 相对 Euclidean G1 降低约 `51.94%`。但 Gaussian membership proxy 的 IoU/Boundary F1 退化，且所有结果只来自 scene0471；当前只保留 G3 mechanism preference，formal arm selection=`false`。
- 可复用表格、8-scene base 明细与失败边界见 `docs/WS_V5_M1_FORMAL_BASE_UNARY_DIAGNOSTIC.md`；机器可读轻量索引见 `docs/archive/2026-08/worldsim-v5-m1/M1_R035_R037_METADATA.json` 与 `M1_R038_GRAPH_METADATA.json`。

## V5 KITTI Tracking adapter smoke（2026-08-14）

- extraction canonical r001=`1805 files / 2,104,258,586 bytes`，summary/manifest SHA=`0ba90a7496d8f8a41dd147ef13579c9ad8d0a25aea7ee829d275ca162df1c363 / 96585bf46127f2fd5eca0a123afe068be6f3922bc73e0362d3019af4c25bc8b3`。
- 直接 audit attempt r002 在任何 payload read 前因 project import root 缺失失败，且因 module import 发生在 runner main 前未生成 run 目录；修复提交=`43fe090db160d6c9bceb6974937a4c20a2d7a760`，未复用该 attempt ID。
- adapter canonical r003=`done`：0000/0001 coverage=`1.0 / 0.9910514541387024`，0001 显式 LiDAR abstain=`[177,178,179,180]`；summary/status/fingerprint/manifest SHA=`3b27cb9fa9b06f563b690cc44b1466e622b578bc88294b450ed254e8192a970b / 404b204ed1a7ff26a1bd6f277e80d6a2e4c66690ef29f0452678cb1506b76dd2 / 2df0f6535f63509f61fe6f72c483955a98c01a9010bd0b71bdfff7364ab5be56 / 099f136af9f519820412d0d3f25fbaaacb969706dd8e4536c7864b27c7fb90ec`。
- 本 smoke 未读方法质量、未训练、未推理、未调参，也未冻结 10-sequence cross-domain pool。

## V5 M1 structured evidence / development 数据闭环（历史执行快照；由上节取代）

- base reconstruction profile batch r019–r026=`8/8 done`：每场 100 steps，checkpoint bytes 合计=`2,592,731,152`、训练耗时合计=`463.532647 s`、peak GPU=`9142 MiB`；8 个 summary 与 checkpoint 均重新哈希通过，source=`200ece4ebe59031b5546f285d2251482446ab162` clean。
- profile summary SHA（scene 0471/1087/0379/0998/0359/0875/0535/0436）=`8c419b3edd9c1e6bac13d82071d659bbe5039de742c9ff2af96105f89e2dcd2f / d2e6ffc128caed3c025930be64e061bfc2af3e4e045cfb3654e8a4673a688202 / 0a570a9774accbe795999322709681a8ed232eb889eb1576df5b6bff2fbb8f8f / f8bbcf00e55a78dff527dadabb13f73641cbc8b40afa1e80a9da1eac52447c66 / 6372b8bb14fc3b641f9838ed1e1527a727f7b8e44ead924b1a73458feb594624 / fa5897d304804e4c19b4e5ae6e8a5d7effbf8d416fb6fa7c7f1031239c7d6cbc / 21686ac5d31bd3b3e71c4c0a1ec23de2545f0cb66b1434bffa04d871e8e94b39 / ef27a83d2af8b991efdd3f73e6cab8089e9837ea7cfd8691e1b23df980fac080`。
- formal 配置=`configs/worldsim_v5/m1_development_reconstruction_formal_v1.yaml`，只解锁冻结 development 上的 30k base training；质量读取、render、validation/test 和 arm search 仍禁止。
- sky-mask canonical batch r011–r018 已完成：scene=`0471/1087/0379/0998/0359/0875/0535/0436`，mask denominator=`588/588/573/588/588/588/603/588=4704`，全 payload rehash exact；汇总 bytes/duration/weighted sky fraction=`14,058,820 / 1067.213706 s / 0.0655167343`。
- summary SHA（按上述 scene 顺序）=`97d509133f3907cfa04d486ab59a5d7801ba689f42d0d8b8decf87822551e208 / 1158ea3c4b2bace8b1cfcd5435b1806f7ffd41bdfa7759af79c6dc6ac9c9fe73 / 401ab6f5a25b55786fd2c6afffea3226901fb17436e1aaf6c3b94b16a4ce91a2 / af2210cefbbc45bb656e5374e090c80e615aec69962542e607771ef3e108e2f3 / 01a3b283034aef256620eed582951e452e7300bc78f3cfee9450085d6a16602a / 4b92a231e24fc2d302f3861fbb509358b9ac260a2fda24346363d76893c5d0ad / f8a56ca7d1797f884e7f6a1e73478fddddcc46c3faa3d0a84c301cb56a7107ab / c6d879aa538acee7c7334064773375ce5b13e160bc67aac61fcfde7d3a6356e3`。
- 新训练 overlay=`configs/worldsim_v5/m1_development_reconstruction_skybound_v1.yaml`；它只叠加已核验的 sky-mask identity，原 base 配置保持字节不变。下一批 run 必须使用新 ID、clean source，并先完成全部 8 场 `profile100`。

| Task ID | 状态 | 当前证据 / 边界 |
|---|---|---|
| `WS-V5-M1-STRUCTURED-OWNERSHIP-01` | running | schema/effective-count unary + `14,220/14,220` raw + `8/8` processed；尚无 development quality，当前进入 base reconstruction |

- 新 schema=`motion_proj/worldsim_v5/evidence_schema.py`：per-Gaussian 保存 center/covariance/normal/prior/unary/uncertainty/effective count/view disagreement/boundary ambiguity/depth/LiDAR/motion；per-view 保存 Gaussian/view/frame/camera/pixel/SAM/boundary distance/depth residual/LiDAR/view-angle/positive-negative/reliability；per-edge 保存 Gaussian COO、Mahalanobis/normal/motion distance、boundary barrier 与 affinity。
- unary=`motion_proj/worldsim_v5/bayesian_unary.py`：按 SAM confidence、visibility、boundary distance、depth residual 与 view angle 形成 observation reliability，再用 fractional effective positive/negative count 更新 Beta；同时输出 view disagreement 与 boundary ambiguity，不提前压成单一 scalar risk。
- unit evidence：deterministic NPZ 两次写出 SHA exact；availability bit、SAM logit/probability、covariance、normal、observation/edge index 与 probability range 均 fail-closed；streaming 与 batch unary finalize 一致。当前 V5 定向 suite=`33 passed`。
- raw preparation canonical r001：十个本地官方 shard 单遍扫描命中=`0/0/0/3486/3555/1820/0/1780/1786/1793`，总计=`14,220 files / 3,996,996,012 bytes`；没有铺开约 294 GB 全量 blobs。batch manifest content SHA=`65fc5363ca13f9124fe6165a84a7857339943d64b87a756127950ab19c4611b6`，formal summary SHA=`0c164e46873ecca4e2878d2d9937960d9b1df916ee25e92d075abc9d1ea0c213`。
- preprocess canonical r002：`8/8 scenes / 2,497,238,886 bytes / 1148.674389 s`；每场景均验证 images/extrinsics/intrinsics/LiDAR/LiDAR pose/三类 dynamic mask/两份 instance JSON，并生成完整逐文件 inventory 后原子发布。timeline=`191 frames(scene-0379) / 201(scene-0535) / 196(other six)`，不得静默裁成共同长度。
- r002 summary/status/fingerprint/manifest SHA=`dcdd3450328669c26eed0316e2088e1f501fad965ed10ad8d344c37fda36f9c0 / 21702f7442824a2e7fd5e66b120511fc3c28121f68d8c711c521c7e858eacfa7 / 147c6d4d4024e12ea26b1e9f0cf7ebb9723d334aa83e1ec199a86c7529eb7a2c / c7140b4001bcfb108e83b5569a5d25a1176f61c66a3dcb1c5d17c3ddfa10f391`；checkpoint 明确为 `N/A_data_preparation`。
- 首个 base profile r003 在 loader 初始化、训练 step 0 前因 `sky_masks/000_0.png` 缺失终止，status/summary SHA=`950931fda48ed436f2410424be4c01c6f391cf9333516cfcb8503ecff7f5165f / a2802430984ab369143be609088df514e3ed0943563b23ee0a5b3bee02e214f7`。分类=`derived_input_missing`，不是模型质量失败；run 保留且不复用。
- 当前先以冻结本地 SegFormer、offline/atomic 协议派生三训练相机 `4704` sky masks，并为每个 mask 保存 bytes/SHA/sky fraction；之后以新 run ID 重跑 8-scene `profile100`，再运行 30k immutable base。正式 unary diagnostic 前 graph=`disabled`，B2 hierarchical/Transformer/semantic split 均未授权；quality/训练迭代/method inference/arm-selection 仍=`0/0/0/0`。

## V5 fresh nuScenes 8/8/20 metadata-only freeze（2026-08-14）

| Task ID | 状态 | canonical evidence |
|---|---|---|
| `WS-V5-D0-NUSCENES-FRESH-COHORT-01` | done | diagnostic r001 + frozen replay r002；cohort SHA=`553373159023218b44615be27aeeb5533a6c585be276e06425235fe09b6b48b1`；r002 summary SHA=`0ea5ff1f5fd16fc278269acbd11e9998c8e3e67d74245a55bdf89a5d09896aad` |

- selection contract commit=`07eb6dbbbcbccdd4dc4661bcb68b63c5ae742fb0`；freeze commit=`8821bd9ad8c3f99b3b39829385728dc37533bb93`。diagnostic 与 formal replay 均遍历相同 850-scene raw metadata pool，并对完整 cohort artifact 做 byte-exact 比较。
- development=`scene-0471/1087/0379/0998/0359/0875/0535/0436`；validation=`scene-0170/0364/0997/0384/0129/0640/0977/1053`；test=`scene-0016/0627/0523/0344/1059/0330/0923/1071/0784/0963/0771/0039/0635/0099/0101/1066/0630/0910/0556/1068`。
- 所有 36 scenes 与 V4 30-scene cohort 严格不相交；development/validation 来自 official train，test 来自 official val；required channels=`CAM_FRONT/CAM_FRONT_LEFT/CAM_FRONT_RIGHT/LIDAR_TOP`，frame partition 与 2–4 s actor clip 合同逐场验证。
- metadata inventory SHA=`63d0a70646615a5bc074faacee9838a8c7c4729a6e091a143435588ba53829f9`，candidate JSONL SHA=`5be022825b7eb98bfc9ddbd1b22e85e1bdf9b9b9d23e8fda9b647b05bf73079f`；selection seed=`2216484596`。
- sensor blob expansion/training/model inference/fresh quality/test quality/parameter search=`false/false/false/false/false/false`。本任务不构成任何模型质量结论；20 test scenes 只冻结身份，尚未消费 quality read。

## V5 P0 M1/M2 retrospective forensic 正式审计（2026-08-14）

| Task ID | 状态 | canonical evidence |
|---|---|---|
| `WS-V5-P0-SCOPE-FREEZE-01` | done | freeze-only commit=`dfe7526c7a83ca12d7fa9f6c5a11a29ea7b27b19`；formal closeout=`20260814T091100Z__p0-scope-closeout-s0-r001`；summary SHA=`ca4248cff7085d8d5a57c842827b1a549b6d1d82fa95c2f81a2976c1192f5d38` |
| `WS-V5-M1-D0-BAYES-FORENSICS-01` | done | r001=`20260814T090500Z__m1-d0-bayes-forensics-s0-r001`；summary SHA=`55006fff260d1bdacb8781492abc3b9f9c6f8bcb5351d2644ad9311c7034d82f` |
| `WS-V5-M2-D0-GEOMETRY-FORENSICS-01` | done | r001=`20260814T090600Z__m2-d0-geometry-forensics-s0-r001`；summary SHA=`33708f5165c04fb22a79bc985da36caf1b907fef8d038ac789e31b1debc5e0c0` |

- M1 formal run 逐文件验证 r200 summary/metrics/manifest 和四份 state NPZ SHA；重算四个 state 的 O1-proxy target、posterior、uncertainty、unobserved 与 mixed-view 分母，并将缺失的 per-view/geometry/topology 字段写入 `artifacts/state_audit.json`。结论=`blocked_evidence_missing_contract_frozen`，V4 M1 仍为 `rejected`。
- M2 formal run 逐文件验证 r222 summary/manifest/router/table 和六个 scene summary SHA；复现 request/candidate=`154/214`、saturation=`192/214`、same-risk collision=`57/130`、accepted oracle exact/positive-regret=`62/21`、accepted mean/max regret=`0.3083979811/7.7560878992 m`。
- denominator decomposition：accepted `83` 的 router/TELEA request mean=`1.6585427687/2.0334827211 m`、delta=`-0.3749399524 m`；risk-abstain `47` 的 delta=`+13.9746599451 m`；role-asset blocked `24` delta=`0`；full `154` request mean delta=`+4.0629155933 m`、scene-balanced delta=`+3.3908096237 m`。
- 产物协议：两个 run 均包含 `resolved_config.yaml / fingerprint.json / events.jsonl / summary.json / manifest.json / status.json / source_snapshot / artifacts/*_audit.json`；checkpoint 明确为 historical audit 不适用，而非漏交。
- provenance：M1/M2 source commit=`dfe7526c7a83ca12d7fa9f6c5a11a29ea7b27b19`；P0 closeout source commit=`28c1d607de0a0ba72895806184348db4d3216de0`。P0 checklist 全项通过，fresh quality/test quality/training/parameter search/router refit=`false/false/false/false/false`；下一任务为结果盲 fresh cohort metadata freeze。

## V5 KITTI Tracking archive / metadata 审计（2026-08-14）

| Task ID | 状态 | canonical evidence |
|---|---|---|
| `WS-V5-D1-KITTI-ARCHIVE-AUDIT-01` | done | `docs/KITTI_TRACKING_ARCHIVE_AUDIT_V5.md`、`docs/KITTI_TRACKING_ARCHIVE_METADATA_V5.json`；manifest=`56388fc64e36c77ebac5a6ee761aa1a17297faeb876347715e8c6e9d52ec23a7` |
| `WS-V5-D1-KITTI-ADAPTER-01` | blocked | `training/0001` LiDAR 缺 `000177`–`000180`；calibration/OXTS parser 待修复；尚无真实 2-sequence smoke |

- archive auditor source HEAD=`1b64d668a90796666af7de8d53a6b8d4eaba7839`；7 个原始 ZIP 的 SHA-256 冻结在 `docs/KITTI_TRACKING_ARCHIVES_V5.sha256`，全 archive bytes 已读取，小包 `ZipFile.testzip()` 通过，大包未额外做全 payload 解码 CRC。
- 7 包 total=`67,746,799,901 bytes`（`63.094 GiB / 67.747 GB`），central members：velodyne/image_02/image_03=`19,099/19,103/19,103`，label/oxts/calib/devkit=`21/50/50/20`。
- training/testing=`21/29 sequences`，image frame denominator=`8,008/11,095`。除 `training/0001` 的 4 个 LiDAR frame gap 外，stereo/LiDAR exact alignment、label timeline、OXTS row count、calibration key 和 devkit seqmap gates 均通过。
- metadata 记录每个 training sequence 的 sensor frames、label rows、annotated frames、track/class 分布、OXTS 行宽与 calibration keys；testing split 显式保持 label=`N/A`，不虚构 GT。
- storage gate 通过：free before=`99.800 GiB`、预计 extract=`63.090 GiB`、20 GiB safety margin 后 expected free=`36.710 GiB`。本 run download/extraction/quality/training/parameter-search=`0/0/0/0/0`。
- 验证：`python -m pytest -q tests/test_audit_worldsim_v5_kitti_archives.py tests/test_worldsim_v4_kitti_track_id.py`=`10 passed`；canonical JSON 解析与内嵌 manifest SHA-256 重算必须通过后再提交。

## V5 P0 / retrospective forensics 启动（2026-08-14）

| Task ID | 状态 | 当前证据 / 边界 |
|---|---|---|
| `WS-V5-P0-SCOPE-FREEZE-01` | running | `configs/worldsim_v5/p0_scope_v1.yaml`；科学问题与禁止项已登记，freeze commit 尚未生成 |
| `WS-V5-D0-NUSCENES-FRESH-COHORT-01` | pending | `configs/worldsim_v5/nuscenes_fresh_cohort_v1.yaml`；只冻结 8/8/20、seed derivation 与 V4 30-scene exclusion，scene list 为空 |
| `WS-V5-M1-D0-BAYES-FORENSICS-01` | running | `docs/WS_V5_M1_FAILURE_FORENSICS.md`；读取 V4 r200/r201 historical diagnostic，未读 fresh quality |
| `WS-V5-M2-D0-GEOMETRY-FORENSICS-01` | running | `docs/WS_V5_M2_GEOMETRY_FORENSICS.md`；读取 V4 r212/r219–r222 historical diagnostic，未改 V4 artifacts |

- P0 start HEAD=`79dafff0c520ab3cbb8d8d73acfd87bb9225b427`；V5 plan start SHA=`abdeeec7aaa6d08efc0f30bd92c46325e3666e1f42d1f7965bed88913ac0edd0`。
- M1 evidence binding：r200 summary/metrics/manifest SHA=`57d732ba13e46cd57758d5a272d1fbb2d4e21c8a85e41a2b59a849ee975d0309 / 0ae07b58822d7be37e5220a781d951725e802404ce52881c1281bb7974e1d504 / 43af8f881667da578ebd567bca2c9dd17492fc270d8be1a5cfe1d849af05ede5`；r201 summary SHA=`470338f225a0ea22e4d5df75d71e86a516c142bd766d1a3ed08009a55fe8fec2`。
- M2 evidence binding：r212 summary SHA=`93e166d9bed748fcd96adb94ff314b73059caa05d32d7941a6b490d7246430a9`；r222 summary/manifest/router-decisions/table SHA=`6bfeb3c6a1e8f1905da936d4e83c93828c030a301ee9d4bedae081c7cc6b1a95 / 702cdb487643bbe633a164d24b9664f35bebc754186fb0845cec4b46250447dd / e59ce5f17e6c3271825875c057e0d777ad9aca95f0ef4b8e07b74d946022caf3 / c7bcadfef2b23b3889c0d130eaf634c3faeca3f4c17abd76898152517b0a86fb`。
- 本阶段只读取 V4 historical artifacts 并生成文档/config；fresh nuScenes content/quality、KITTI quality、training、parameter search=`0/0/0/0/0`。

## V4 终局文档归档（2026-08-14）

| Task ID | 状态 | canonical evidence |
|---|---|---|
| `WS-V4-DOC-ARCHIVE-01` | done | `docs/archive/2026-08/worldsim-v4-final/`；commit=`c7e4c969a95536d26d0a17a1c0d1d548f9a247dc`；`78/78` SHA-256 checks passed |

- source closeout HEAD=`403c5703a755c999d42a5ec3eb063db6cc751761`；没有重跑、续写或修改任何 V4 canonical run。
- 归档复制 M1 r200/r201、M2 r212/r222、M3 r238/r335/r336 的轻量证据，保留核心账本和 V4 计划/审计快照；checkpoint、渲染帧、训练缓存均未复制。
- 附录索引明确绑定 M1 rejection、M2 `+3.3908096237 m` hole geometry 退化、M3 `12 evaluable + 6 abstain` test 结论与 exact-once provenance。
- 验证：`sha256sum -c docs/archive/2026-08/worldsim-v4-final/SHA256SUMS`=`78/78 OK`；`git diff --cached --check`=passed。

## V4 M3 validation / 18-scene exact-once test 注册（2026-08-13）

| Task ID | 状态 | canonical evidence |
|---|---|---|
| `WS-V4-M1-EVIDENCE-FIELD-01` | rejected | r200=`3 evaluable + 3 abstain`、directional support=`0/6`；r201 rejection audit |
| `WS-V4-M2-REPAIR-ROUTER-01` | done | r222 gate passed；router=`uncertainty_forward@1.0`，accepted/abstain=`83/71`，geometry tradeoff=`+3.3908096237 m` |
| `WS-V4-M3-TEMPORAL-DELTA-01` | done | r238 validation passed；`20260813T225624Z__m3-test-aggregate18-s0-r335` test conclusion=`confirmed`，denominator=`12+6=18` |

### M3 validation freeze

- canonical=`/root/autodl-tmp/runs/worldsim_v4/WS-V4-M3-TEMPORAL-DELTA-01/20260813T075800Z__m3-validation-confirmation-s0-r238`；全部预注册门通过，test freeze authorized，test quality 尚未读取。
- frozen parameters=`{"acceleration_regularization": 0.1, "control_point_count": 4, "evidence_retention": 0.5, "warp_blend_alpha": 0.4}`；validation optimization/read=`false`，3 个 abstain 保留在 6-scene denominator。

### 18-scene test

- freeze=`/root/autodl-tmp/motion_proj/V4_TEST_FREEZE.json`；freeze-only commit=`83cb82872bf747c2b1c79fbc2a9982320f972413`，source parent=`029d819e0abb63d2edacb811be9ea2153589e92f`，execution plan 固定 r317–r334。
- ledger=`/root/autodl-tmp/runs/worldsim_v4/WS-V4-M3-TEMPORAL-DELTA-01/20260813T222011Z__m3-test-exact-once-ledger-s0`；attempt/completion=`18/18`，quality-read scene count 由冻结资产可执行性决定，所有 abstain 保留在 18-scene denominator。
- aggregate=`/root/autodl-tmp/runs/worldsim_v4/WS-V4-M3-TEMPORAL-DELTA-01/20260813T225624Z__m3-test-aggregate18-s0-r335`（`20260813T225624Z__m3-test-aggregate18-s0-r335`，r335）；evaluable/abstain=`12/6`，gate=`true`，conclusion=`confirmed`。
- 没有 test-time parameter search、threshold search 或 source-content reread；失败/非确认也按预注册合同收口，不触发同一 test 的第二次读取。

## V4 当前注册表补充（2026-08-13，覆盖下方旧进度行）

| Task ID | 状态 | 当前证据 / 下一门禁 |
|---|---|---|
| `WS-V4-B0-MATCHED-BASELINES-01` | done | 六场 strict matched baseline 与最终审计 r117 已冻结 |
| `WS-V4-M1-EVIDENCE-FIELD-01` | rejected | validation r200=`3 evaluable + 3 abstain`、方向支持 `0/6`；rejection audit r201；禁止扩 feature |
| `WS-V4-M2-REPAIR-ROUTER-01` | done | development r212 冻结 router/TELEA；validation r222 全部门通过，`m3_authorized=true` |
| `WS-V4-M3-TEMPORAL-DELTA-01` | running | M2 已解锁；当前只允许 6-development-scene 2–4 s clips，先 smoke→freeze→validation |
| `WS-V4-E0-NUSCENES-SCALE-01` | pending | M3 通过后先生成 test-freeze commit；18 test 仍未读且只能只读一次 |
| `WS-V4-D1-KITTI-ADAPTER-01` | blocked | 等用户自行复制真实 KITTI；禁止下载、禁止以 synthetic fixture 冒充完成 |

### M1 validation rejection freeze

- canonical=
  `/root/autodl-tmp/runs/worldsim_v4/WS-V4-M1-EVIDENCE-FIELD-01/20260812T204156Z__m1-validation-six-scene-confirmation-s0-r200`；
  `3/6` evaluable，方向支持=`0/6`，Boundary F1/FN mass/Brier/ECE delta=
  `-0.0664623346/+0.0083741268/+0.0024487362/+0.0024972500`；confirmation=`reject`。
- rejection audit=
  `/root/autodl-tmp/runs/worldsim_v4/WS-V4-M1-EVIDENCE-FIELD-01/20260812T210150Z__m1-validation-rejection-audit-s0-r201`；
  task=`rejected`，M1 feature expansion=false，M2 fallback scope=`evidence_routed_delta_compiler`。
- frozen development selection 未改，base RGB/checkpoint exact，validation arm/calibration/threshold search=false；
  test quality read=false。冻结提交=`b7a8fdf`。

### M2 development freeze / validation closeout

- development scene runs r206–r211 保留全部 `6` scenes；selection r212=
  `/root/autodl-tmp/runs/worldsim_v4/WS-V4-M2-REPAIR-ROUTER-01/20260812T233139Z__m2-development-router-selection-s0-r212`；
  frozen weights=`uncertainty_forward`、threshold=`1.0`、tie=`OBSERVED/DONOR/GENERATED`，best matched
  non-router=`TELEA`，development gate passed。
- validation scene runs r216–r221：scene-0071/0317/0450 formal done，scene-1089/0862/1012 retained
  `ABSTAIN_NO_ACTOR`；scene-0317 的 boundary role 另有 24 个 measured atomic abstain。checkpoint 全部 before/after exact，
  validation optimization=false，test quality read=false。
- canonical aggregate=
  `/root/autodl-tmp/runs/worldsim_v4/WS-V4-M2-REPAIR-ROUTER-01/20260813T064330Z__m2-validation-confirmation-s0-r222`；
  source=`1cc90b1865ef24f3bbef0add775d0fdf4be0d491`，`6 scenes / 154 requests / 214 candidates`，
  evaluable/scene-abstain=`3/3`、role-asset-abstain=`24`、accepted/abstain=`83/71`。
- 相对 development-frozen TELEA：global PSNR/SSIM/LPIPS delta=
  `+0.0539729695/+0.0004358785/-0.0007536737`，hole PSNR=`+3.1797798583 dB`，static LiDAR MAE
  degradation=`+0.0000499586 m`，selective separation=`+0.1241311528`；全部 gate passed。
  hole geometry MAE 同时退化 `+3.3908096237 m`，必须作为 tradeoff 报告。
- summary/manifest/status SHA=`6bfeb3c6...b1a95 / 702cdb48...47dd / 4fcc7b6e...e75b`；manifest
  `8/8` inventory exact；实现/绑定提交=`1cc90b1`；M3 解锁，18 test 仍未授权。

## V4 当前注册表补充（2026-08-12）

| Task ID | 状态 | 当前证据 / 下一门禁 |
|---|---|---|
| `WS-V4-B0-MATCHED-BASELINES-01` | done | 六场 strict matched baseline 与最终审计 r117 已冻结；commit `55232a4/9e322fa` |
| `WS-V4-M1-EVIDENCE-FIELD-01` | running | smoke r121 与 development r124 gate 均通过；r126 已冻结选择，当前只允许六场 validation confirmation |
| `WS-V4-M2-REPAIR-ROUTER-01` | pending | 等 M1 validation 闭环后执行 development matched router；不得提前读取 test quality |
| `WS-V4-M3-TEMPORAL-DELTA-01` | pending | 等 M2 development 冻结后执行 2–4 s nuScenes clips |
| `WS-V4-E0-NUSCENES-SCALE-01` | pending | 先完成 6 validation；18 test 仍封存且只能在 test freeze commit 后读取一次 |
| `WS-V4-D1-KITTI-ADAPTER-01` | blocked | 等用户自行复制真实 KITTI；禁止下载、禁止用 synthetic fixture 冒充数据集完成 |

### M1 development canonical

- run：`/root/autodl-tmp/runs/worldsim_v4/WS-V4-M1-EVIDENCE-FIELD-01/20260812T121734Z__m1-development6-s0-r124`；
  source=`451073a0ff592d2bd69240d0f33f9124c567eeac`，seed=`0`，单卡 RTX 3090。
- frozen arm=`raw__risk_100`，calibration=`raw`，threshold=`0.5`，temporal retention=`0.75`；
  scene accounting=`6 required / 2 evaluable / 4 abstain`。
- 相对 V3.3 O1：Boundary F1=`+0.1255247811`、FN semantic mass=`+0.0054849633`、
  Brier=`-0.0115803990`、ECE=`-0.0311158595`；M1 预注册 gate=`pass`。
- base RGB 与 checkpoint before/after exact；peak CUDA allocated/reserved=
  `8,802,204,672 / 8,927,576,064 bytes`；heldout/test quality 均未读。
- freeze audit r126=`done`；配置冻结提交=`b39d49b`。validation 数据/reconstruction 配置提交=`06d56ee`。

### M1 validation 数据提取

- r127：错误使用 DriveStudio Python，因缺 `ijson` 在提取前 fail-closed。
- r128：本地官方 10-shard 扫描期间 SSH 超时断管，terminal=`blocked/BrokenPipeError`；已写入约
  `6.7 GiB` raw union 的非空文件保留复用。
- r129：detached launcher 使用了不存在的 `/root/miniconda3/envs/motionproj/bin/python`，未形成正式 run；
  正确环境位于 `/root/autodl-tmp/envs/motionproj`。
- r130：detached parent PID=1，`done`；10-shard scan wall=`3,521.8796 s`，精确绑定
  `10,647` 个 sensor members，六场 required count=`1773/1747/1778/1784/1783/1782`；
  summary/inventory SHA=`6f0b0933...76854c / 41b2d5eb...ee5c`，Git=`06d56ee` clean，
  `no_download=true`、test quality 未读。`extracted_this_run=0` 表示完整复用 r128 已原子写入的非空文件。


- 更新时间：2026-08-12
- 当前路线：WorldSim V4 / EviDelta-GS
- 当前执行授权：`WS-V4-B0-MATCHED-BASELINES-01` 6-development-scene evaluator/baseline；不得读取 test quality
- 当前方案：[`WORLDSIM_V4_EVIDELTA_GS_PLAN.md`](WORLDSIM_V4_EVIDELTA_GS_PLAN.md)
- V3.2 终局归档：[`archive/2026-08/worldsim-v3.2/`](archive/2026-08/worldsim-v3.2/README.md)
- V3.1 终局归档：[`archive/2026-08/worldsim-v3.1/`](archive/2026-08/worldsim-v3.1/README.md)
- V2 历史方案：[`DYNAMIC_DRIVING_EDITING_DIAGNOSTIC_PLAN_V2.md`](DYNAMIC_DRIVING_EDITING_DIAGNOSTIC_PLAN_V2.md)
- V1 最终台账：
  [`archive/2026-07/dynamic-reconstruction-v1/EXPERIMENTS.md`](archive/2026-07/dynamic-reconstruction-v1/EXPERIMENTS.md)

本文件保留 V2 完整执行证据，并从 2026-08-05 起登记 V3。V2 M0–M4 已完成；M5 部分执行后停止扩张，
保持 `pending` 历史终态；M6–M8 不再授权。A0、A1 与 A2 已完成；A2 fixed/matched 正式裁决为
`tradeoff_non_dominated`。`WS-V3-A3-LOCAL-REFINE-01` 已以 R1 资源门失败和 diagnostic tradeoff 的负结果
`done`，`A3*=R0-off`；A4 已 `done`，P0、P5、P1、P2 与 P3 全部闭环；P1 候选被质量门拒绝并回退到 source，
P2 canonical r2 选择 mixed checkpoint，P3 canonical r1 选择 exact chunk package。F0 canonical audit 已 `done`，
本机 inference 前置条件未通过，F1=`conditional_not_unlocked`；R0 canonical 已 `done`，V3.1=`none_plan_complete`。
V3.1 终态保持冻结；V3.2 S1 canonical r6 已完成，S3 canonical r3 已生成并选定 Asset Harvester
2-view actor asset，S2 canonical r3 已选定 3DGIC-adapted generated-background checkpoint。S4 canonical r3
完成 non-temporal JIT，但删除语义保持门失败，只保留 optional diagnostic；temporal arm 受 gated Cosmos base
阻塞。R0 canonical r4 已完成语义扩展、mixed storage、S3 actor override registry、exact chunk package 与三视角
单卡验证；全部单卡 RTX 3090 可执行环节终态为 `done`，S5 继续 `blocked`。V3.2 整体终态为
`none_plan_complete`；本注册表自 2026-08-11 起冻结为历史事实，不授权续跑 S4 temporal、S5 或旧 rejected run。

## V4 注册表

| Task ID | 状态 | 目标 | 当前证据/门禁 |
|---|---|---|---|
| `WS-V4-P0-SCOPE-PAPER-FREEZE-01` | done | 冻结 claim、HEAD、数学、baseline、数据、指标与一手来源 | `main@2108430`；15 sources；KITTI missing 如实阻塞；P0 不训练 |
| `WS-V4-D0-NUSCENES-COHORT-01` | done | 结果前冻结 6 dev + 6 val + 18 test | canonical r4；850 candidates；2-scene preprocess smoke passed；无训练/test quality |
| `WS-V4-D1-KITTI-ADAPTER-01` | blocked | 本地 tracking/raw adapter | code + 12-gate synthetic contract done；canonical r2=`blocked_local_dataset_missing`；禁止下载 |
| `WS-V4-B0-MATCHED-BASELINES-01` | running | V3.3/StreetGS/AD-GS same-split replay | 当前唯一授权：6 development scenes evaluator；M1 仍未授权 |
| `WS-V4-M1-EVIDENCE-FIELD-01` | pending | Bayesian/calibrated temporal evidence | B0 未闭环前未授权 |
| `WS-V4-M2-REPAIR-ROUTER-01` | pending | Bayes-risk repair compiler | M1 未收口前未授权 |
| `WS-V4-M3-TEMPORAL-DELTA-01` | pending | `SE(3)` B-spline temporal delta | M2 未收口前未授权 |
| `WS-V4-E0-NUSCENES-SCALE-01` | pending | 6 dev + 6 val + 18 test | test freeze 提交前不得读取 test quality |
| `WS-V4-E1-KITTI-CROSSDATA-01` | pending | 2 smoke + 10 formal | D1 exact 且 nuScenes 参数冻结后解锁 |
| `WS-V4-E2-ENGINEERING-BENCH-01` | pending | 生产效率 benchmark | 方法冻结后执行 |
| `WS-V4-E3-DOWNSTREAM-GAP-01` | pending | perception real-to-sim gap | 方法冻结后执行 |
| `WS-V4-H0-HUMAN-STUDY-01` | pending | 条件式人评 | 只有自动/定量门完成后单独解锁 |
| `WS-V4-R0-RELEASE-01` | pending | exact paper release | 前序收口后解锁 |
| `WS-V4-W0-PAPER-01` | pending | 技术报告 / paper | 只写已有证据 |

### `WS-V4-P0-SCOPE-PAPER-FREEZE-01` closeout

- 实际起点为 `main@21084309480895f5541196a06191a5dffb4e30c1`；计划草案里的 `144ed19` 保留为历史点；
  `e6663e1` 和 `144ed19` 均在当前 `main` 历史中；
- 创建 `research/worldsim-v4-evidelta`，冻结 EviDelta-GS 三模块、30-scene nuScenes 6/6/18 split、
  KITTI frozen cross-domain、Tier A/B baseline、PSNR/SSIM/LPIPS-Alex、scene-level statistics 与工程指标；
- 审计 15 个一手工作；“官方论文/源码存在”与“本机 executable”分开记录，不可执行路线不填数值；
- `/root/autodl-pub/KITTI` 当前不存在，登记 `blocked_local_dataset_missing`；没有创建目录或网络下载；
- P0 时 GPU=`RTX 3090 24,576 MiB`、GPU process=`0`、cgroup OOM/kill=`0/0`、disk free 约 `193 GiB`；
- training/model inference/weight download 均为 false；P0 只解锁 D0。
- canonical=`20260811T080636Z__p0-scope-formal-s0-r2`；config/summary/manifest/status SHA=
  `248bde62...aaa8 / aba1fbcf...1283 / ec32e983...3970 / b3941601...272a`；5 份 source snapshot exact，
  run=`101,624 bytes`；r1 因提交前只规范计划参考文献 whitespace 导致 source/config SHA 改变，保留为 noncanonical
  done，r2 对最终字节重新审计。

### `WS-V4-D0-NUSCENES-COHORT-01` closeout

- 从官方 nuScenes `v1.0-trainval` 元数据枚举 850 个候选，只按结果前 metadata 分层；development/validation
  全来自官方 train，18 个 test 全来自官方 val，scene name/token 全局互斥；
- 冻结 6 development、6 validation、18 test；完整配置保存 30 scene 的 high/difficult actor、remove/lateral/insert、
  2–4 秒 continuous clip、3-front-camera + LIDAR_TOP 与逐帧 train/development/heldout 划分；
- cohort SHA=`eda9f684...44578`；formal 构建会逐字段比较 `scene_records`，不只校验名单；30/30 scene 均找到
  high/difficult actor，连续片段范围=`2.899163–3.150218 s`；
- scene-0230/0242 只复用已有 preprocess 产物做 smoke：每 scene `196` 帧、`1,176` 图像、`196` LiDAR，实例 JSON
  与抽样 artifact SHA 全通过；没有训练、模型推理或质量读取；
- canonical=`20260811T084108Z__d0-cohort-formal-s40117-r4`；config/summary/manifest/status/cohort SHA=
  `ed47c0da...667a1 / ec96970d...f276 / 3349a636...6b30 / 1dfd5db4...4461 / eda9f684...44578`；
  12 files=`1,221,825 bytes`，11 个 metadata table fingerprint exact；diagnostic r1 保留为 noncanonical；r2 因
  CRLF→LF source bytes 改变降为 noncanonical；r3 被 cohort SHA freeze gate 拦截并记录 `blocked` terminal，定位为
  set 浮点求和顺序受 Python hash seed 影响；r4 使用排序 + `math.fsum` 并恢复原 cohort SHA；
- D0 定向测试=`8 passed`，D0/P0 + V3.3/V3.2 联合回归=`106 passed`。下一任务 D1 必须在 public KITTI
  缺失时输出合法 blocked terminal，禁止下载；随后才进入 6-development-scene B0 matched baseline，M1/M2/M3 与
  test quality 仍未授权。

### `WS-V4-D1-KITTI-ADAPTER-01` blocked closeout

- 实现 tracking-first / raw-fallback 自动 layout discovery、KITTI 原生 `image_02/image_03` 双相机合同、calibration、
  track ID/3D box、LiDAR/box projection、pose/timestamp、stereo 与 frame-leak 检查；KITTI threshold search=false；
- synthetic tracking/raw fixtures 的 12 项 adapter gates 全通过，定向测试=`7 passed`；D1/D0/P0 + V3.3/V3.2
  联合回归=`113 passed`；这些测试不冒充真实 KITTI smoke 或 cross-domain quality；
- `/root/autodl-pub -> /autodl-pub/data`；requested `/root/autodl-pub/KITTI` 与 resolved
  `/autodl-pub/data/KITTI` 均不存在，没有创建目录或下载；terminal=`blocked_local_dataset_missing`；
- canonical blocked run=`20260811T085210Z__d1-kitti-layout-formal-s0-r2`；
  config/summary/manifest/status/fingerprint SHA=`bffe6eaa...93d4 / 05eeb265...9ca9 / 7c9ca256...f582 /
  598eac51...d54b / 233e464e...fdc5`；14 files=`45,214 bytes`；r1 只记 physical path，r2 补齐 requested +
  symlink-resolved provenance；
- D1 只在真实数据挂载后以新 run 复开；E1 继续 pending。当前转入 B0 nuScenes development，不因此宣称
  single-card closure，也不读取 test quality。

### `WS-V4-B0-MATCHED-BASELINES-01` evaluator + inventory milestone

- 冻结 `metrics_v1.yaml`：PSNR/SSIM/LPIPS-Alex、五类区域、无 GT/空区域 undefined、scene-level
  mean/median/std/IQR/bootstrap CI 与 paired bootstrap/sign-flip/Wilcoxon；failed/blocked/abstain 保留 denominator；
- baseline 区域协议显式冻结为 actor=`dynamic_masks/all>0`、static=`not actor and not egocar`、boundary=L1
  半径 3 px 形态学带，未编辑 baseline 的 edit_roi 为空；development-only scene evaluator 已加入文件/partition/重复键门；
- 实现统一 evaluator、scene statistics 和 engineering raw-row 派生，定向测试 `9 passed`；该里程碑未启动训练、
  模型推理或 test quality 读取；
- `baseline_matrix_v1.yaml` 精确锁 D0 六个 development scenes、same split/resolution、DriveStudio
  `e59bda4`、V3.3 `e6663e1` 与 AD-GS `9a20851`；分辨率层级固定为 sensor `1600×900`、source downscale=2、
  model/metric `800×450`；historical metric/provenance 不算 executable；
- diagnostic inventory=`20260811T090951Z__b0-inventory-diagnostic-s0-r1`，terminal=
  `blocked / matched_baseline_assets_incomplete`，coverage=`V3.3 1/6、StreetGS 0/6、AD-GS 0/6`；run 只说明
  当前磁盘资产缺口，B0 task 继续 `running`；
- 三个历史 StreetGS checkpoint 的 summary/bytes provenance 仍可审计，但文件当前不在磁盘；AD-GS historical
  六场景 metrics 存在，而 source/env/checkpoint 不存在。后续必须新 run 重训/恢复，不能把历史数值填入 matched 主表。
- raw extraction canonical r3 从官方 10 个 nuScenes tar shard 物化缺失 scene-0048/0139/0994，`5,264/5,264`
  members，fingerprint=`c51f4162...e38`；没有下载数据或读取 test quality；
- preprocess r5/r6/r7 均 done；六个 scene index `045/110/179/191/204/752` 现均为 `1,176 RGB / 196 LiDAR`。
  r4 的真实上游输出保留，但因 `_10Hz` root 合同不匹配为 blocked；
- remote sky-model r10 因网络不可达 blocked；官方 fixed-revision 本机 exact staging 由 r11 离线恢复完成，fingerprint=
  `15c200fd...90bfd`。sky r12 因预建空目录误拒绝 blocked；r13/r14/r15 分别为三个新 scene 原子生成
  `588/588` masks，fingerprint=`c35a9dbe...c3eb / 5f6a2fe7...c702 / 455dcb20...e6c7`；
- StreetGS r9 在 iteration 前因缺 sky mask blocked；r16 profile100 完成，wall=`90.8965 s`、checkpoint=
  `340,298,602 bytes / SHA 446297b8...3af`、peak GPU=`9,004 MiB`、OOM/kill=`0/0`。该 profile 只解锁
  六场景 30k formal，B0 coverage 仍为 `0/6 formal`，M1 与 test quality 继续未授权。
- StreetGS r17/r20/r22/r24/r26/r28 均完成 30k、means finite、OOM/kill=`0/0`，但 2026-08-12 复核确认其
  `test_image_stride=10` 不满足冻结的 `sample_index mod 5` 三分区合同；六个 run 与 r29=`6/1/0` 只保留为
  protocol-mismatch provenance，不计 strict matched coverage；
- strict scene-0230 r32=`20260811T154831Z__streetgs-scene0230-matched-formal30k-s0-r32` 30k done，wall=
  `3,200.0184 s`，checkpoint=`386,410,166 bytes / SHA 766648bf...af97cd1`，Background/RigidNodes=
  `1,095,606/172,264`，peak GPU=`23,892 MiB`、peak cgroup=`15,281,917,952 bytes`，OOM/kill=`0/0`，
  test quality 未读；
- corrected inventory r33=`20260811T165009Z__baseline-matched-correction-s0-r33`，terminal blocked，strict
  coverage=`StreetGS/V3.3/AD-GS 1/1/0`，inventory/fingerprint SHA=
  `73d36544...bea9e / c19fba13...e285853`；它覆盖当前结论但不覆盖旧 run；
- strict scene-0242 r46=`20260811T210253Z__streetgs-scene0242-matched-formal30k-s0-r46` 30k done，
  wall=`1,998.0482 s`，checkpoint=`302,953,462 bytes / dd41a34d...52bc0`，Background/Rigid=
  `824,583/92,170`，peak GPU/cgroup=`17,530 MiB / 23,842,824,192 bytes`，OOM/kill=`0/0`，
  无 test/full render；r47 内容寻址 inventory=`StreetGS/V3.3/AD-GS 2/1/1`，
  inventory/fingerprint=`89c72659...eafad / b91f7c76...712d6`；
- strict scene-0255 r48=`20260811T214009Z__streetgs-scene0255-matched-formal30k-s0-r48` 30k done，
  wall=`2,392.0649 s`，checkpoint=`444,340,086 bytes / dba24982...cb2d2`，Background/Rigid=
  `1,478,401/38,721`，peak GPU/cgroup=`23,932 MiB / 24,132,476,928 bytes`，OOM/kill=`0/0`，
  无 test/full render；r49 inventory=`StreetGS/V3.3/AD-GS 3/1/1`，
  inventory/fingerprint=`79b6b1d0...c86c85f / bd822e61...9641a`；
- strict scene-0048 r50、scene-0994 r52、scene-0139 r54 均 30k done，wall=
  `2,179.8120/1,806.3743/2,069.0543 s`；checkpoint bytes=`332,725,750/279,185,462/314,307,830`，
  SHA=`70d02a0b...b00d2 / 3e2b2534...3aea / 4fff4452...8dfe`；Background/Rigid=
  `1,030,993/15,717 / 819,952/932 / 962,074/7,219`，peak GPU=
  `23,694/20,970/23,056 MiB`，资源事件与 test/full render 均为 `0`；
- StreetGS 六场 registry 提交=`a4ee23a`；r55=`20260812T001330Z__baseline-streetgs-sixscene-registration-s0-r55`
  在 clean HEAD 得到 coverage=`StreetGS/V3.3/AD-GS 6/1/1`，inventory/fingerprint=
  `8bc62596...be3a1 / 4f12c1d2...32372`；
- AD-GS official exact source=`9a208512`、DPT/CoTracker weights 与 train-only adapter 已恢复；环境 r34=
  `20260811T165030Z__adgs-environment-restore-r34` 离线完成，torch=`2.1.2+cu118`，两个 CUDA 扩展真实
  forward/backward smoke passed，OOM/kill=`0/0`。该结果只解锁 strict preprocess/profile，不计 scene coverage；
- AD-GS preprocess r35/r36/r37 分别被 run-dir 不可变门、official source 未跟踪 build 目录、可选 `flow_vis`
  诊断依赖 fail-closed；均未训练或读取 dev/heldout。r37 已完成的 adapter/depth/segment partial 移入规定 backup，
  no-visualization flow 合同与 run-local extension build source 作为新提交修复，不覆盖旧 run；
- AD-GS scene-0230 canonical train-only preprocess r38=
  `20260811T171507Z__adgs-scene0230-preprocess-s0-r38` done，文件计数=`354/354/354/354/285`，flow wall=
  `3,046.9930 s`，peak GPU/cgroup=`20,112 MiB / 22,384,893,952 bytes`，OOM/kill=`0/0`，无可视化视频且
  development/heldout/test quality 均未读；fingerprint/manifest=`d44a0530...fbf2c37 / cefae230...c05873`；
- profile r39 因训练 import 全局依赖可选 `flow_vis`、r41 因 inherited PyTorch3D binary 无 sm86 KNN kernel
  分别在 iteration 前 blocked；均未读 dev/heldout。r40 exact `roma 1.5.7` runtime restore done；r42 从 clean
  `pytorch3d@2f11ddc5` run-local 重编 sm86，build=`1,151.39 s`，KNN + Gaussian CUDA smoke passed，
  新 binary=`10,313,184 bytes / eca71e2c...e3084`，fingerprint=`03ec74e8...05f7fb`；
- profile r43=`20260811T185145Z__adgs-scene0230-profile100-s0-r43` done，100 steps train stage=`40.3595 s`，
  peak GPU/cgroup=`6,012 MiB / 30,991,519,744 bytes`，三文件 checkpoint SHA=
  `bc364930...16c5 / 8205e276...e5ab / 45644cb4...8e77`，test quality 未读；profile 不计 coverage；
- formal r44=`20260811T185600Z__adgs-scene0230-formal60k-s0-r44` done，60k stage=`7,054.6221 s`，
  peak GPU/cgroup=`16,692 MiB / 33,680,572,416 bytes`，OOM/kill=`0/0`；三文件 bytes=
  `413,905,347/435,921,657/805,307,528`，SHA=`f17ed27f...a0cbb / c725f952...c84b0 /
  c3233b71...e4d34`，train-only partition 且 development/heldout/test quality 均未读；
- fail-closed registry 提交=`904e395`；r45=`20260811T205840Z__baseline-adgs-formal-registration-s0-r45`
  重新校验 runtime/source/patch、formal step、run 内三文件 bytes/SHA 与 fingerprint/manifest，得到 strict coverage=
  `StreetGS/V3.3/AD-GS 1/1/1`，inventory/fingerprint=`4bf7cf68...ad6b / 3db524d2...49e5`；
- run-local extension build source=`9f839fd`，baseline region/evaluator=`abd82d8`，runtime import/sm86 fixes=
  `c3600be/eed00cb`；联合定向测试=`50 passed`；
- B0 继续 running；StreetGS strict 六场已齐，下一步补齐 AD-GS/V3.3 其余五场 same-split，
  再运行统一 evaluator。M1 与 test quality 继续未授权。

## V3.3 注册表

| Task ID | 状态 | 目标 | 当前证据/门禁 |
|---|---|---|---|
| `WS-V33-P0-ROUTE-SOTA-AUDIT-01` | done | 冻结 V3.2、切换分支、审计 source/license/weights/hardware | canonical r2；10 sources；5 个 V3.2 大资产 SHA exact；36+4 tests |
| `WS-V33-S1-OBJECT-AWARE-GS-01` | done | SAM2.1 fallback + dual instance-opacity field | canonical formal r9；O1 selected；base exact；51 tests |
| `WS-V33-S2-ROADPATCH-INPAINT-01` | done | RoadPatch-Lite + Inpaint360GS baseline | r10 index + r11 B1 canonical；r12 B2 blocked_single_3090 |
| `WS-V33-S3-ASSET-VIEWSELECT-01` | done | Asset Harvester auto 1/2/4-view | high A4 heldout accepted；boundary override ABSTAIN；63 tests |
| `WS-V33-S4-SPATIAL-DELTA-01` | done | immutable base + erase/insert delta | canonical r7/r8；20 rollback exact；posterior-gated erase；9 tests |
| `WS-V33-S5-SEMANTIC-RENDER-01` | done | semantic-gated render；R3D2 conditional | canonical r4；G1 heldout rejected；G0 production；delete 5/5 safe |
| `WS-V33-R0-INTEGRATION-01` | done | 单卡完整集成与 exact package | canonical r7；44 inputs；10/10 gates；76-file release；v33_supported |
| `WS-V33-F0-LIDAR-EVS-AUDIT-01` | pending | 条件式未来 LiDAR 扩展 | 不阻塞 R0，当前未授权 |

### `WS-V33-P0-ROUTE-SOTA-AUDIT-01` canonical closeout

- branch=`research/worldsim-v3.3-object-maintenance`，baseline=`a055fc6727dddacd194665d5c997a1fe47c2d2f4`；
- canonical=
  `/root/autodl-tmp/runs/worldsim_v33/WS-V33-P0-ROUTE-SOTA-AUDIT-01/20260810T171744Z__p0-source-audit-s0-r2`，
  terminal=`done`；config/summary/manifest/status SHA-256=
  `29c167fe...9ae2 / 08806b5f...d68a / 2603ff0e...a12d / 91096d0e...072d`；
- r1 的审计门全部通过，但 runner 未把 auditor/module/test 快照进 run，故保留为 noncanonical `done` 证据；r2 新增
  三份 source snapshot（SHA=`96024691...c065 / 5760318f...fec3 / 0fe51b7c...a778`），不改写 r1；
- 10 个 source=`2 executable / 2 weights_blocked / 5 source_not_released / 1 audit_only`；5 个有 Git checkout 的新 source
  与继承 Asset Harvester 均 commit/tree clean exact，存在的 5 份 root license SHA exact；
- SAM3.1=`96914d2`，当前 `hf auth whoami=Not logged in` 且无 cached checkpoint，故 SAM3 arms `weights_blocked`；
  OP2GS/3D-GIMP/FocusGS/LiDAR-EVS 无官方 runnable source；GS-RoadPatching=`468f812` 只有 project-page assets；
- Inpaint360GS=`d54c893 / Apache-2.0 / source executable`，只在 S2 adapter 后做最小单卡 preflight；R3D2=
  `3fc6e31 / Apache-2.0 / exported author model absent`，不从零训练；GOR-IS=`eb36acc / noncommercial audit_only`；
- D2/S2/S3/mixed/chunk 五资产重新 hash 全 exact；V3.2 R0 8/8 gates 与 `36 passed` 回归复核通过；P0 auditor=
  `4 passed`；第一次无 `PYTHONPATH=.` 的 pytest 只在 collection 阶段失败，按仓库入口重跑无逻辑失败；
- GPU=`RTX 3090 24,576 MiB`、cgroup max=`96,636,764,160 bytes`、OOM/kill=`0/0`、disk free 约 `40 GiB`；
  P0 training/model inference/install/large-weight download/DriveStudio mutation 均为 false；
- P0 关闭时只解锁 S1；该历史门禁已由下节 S1 closeout 满足，当前授权以文件头与 STATUS 为准。

### `WS-V33-S1-OBJECT-AWARE-GS-01` canonical closeout

- diagnostic r0 完整贯通 O0/O1/O3，但 `diagnostic_steps_override=1`，只作工程证据；development smoke r1
  使用固定 10 个 development frames、零 heldout，O1 相对 O0 同时改善 boundary F1、NBD、IoU 与 FP mass，
  因而冻结 `formal_selected_arm=O1_dual_opacity`；O3 未入选；
- heldout-target r2 因外部清理后的旧 SAM Python 路径不存在而 fail closed；r3 在首个 block 暴露 logits rank
  兼容错误，未正式发布 mask；修复后 canonical r4=`20260810T180231Z__s1-heldout-targets-s0-r4`，
  31 blocks / 37 accepted / 0 rejected，summary/manifest/status SHA=
  `686c7100...20ff / d2b2ab14...3050 / b186664e...b4d1`；
- r4 使用 SAM2 source=`2b90b9f`、checkpoint=`898,083,611 bytes / 2647878d...318`；隔离 runtime=
  Python `3.10.20`、torch `2.5.1+cu124`、torchvision `0.20.1+cu124`，peak reserved=
  `2,099,249,152 bytes`；19 个 heldout frames 从文件级标记 `optimization_forbidden=true`；
- canonical formal r9=
  `/root/autodl-tmp/runs/worldsim_v33/WS-V33-S1-OBJECT-AWARE-GS-01/20260810T183154Z__s1-instance-field-formal-s0-r9`，
  terminal=`done`；config/summary/manifest/status SHA=
  `9afa48aa...3150 / 4ab311a6...8c4a / e1b858fd...584e / 9394d15e...03b9`；
- O0 heldout：boundary F1=`0.068960`、IoU=`0.063253`、NBD=`0.144958`、FP/FN mass=
  `0.900308 / 0.061278`；O1：`0.336158 / 0.330727 / 0.105280 / 0.623276 / 0.109356`；
  对应 boundary F1 `+387.47%`、IoU `+422.87%`、NBD `-27.37%`、FP mass `-30.77%`，但 FN mass
  增加，按范围限定为 precision/boundary breakthrough，不宣称所有指标全面支配；
- O1 field=`5,882,296 bytes`、SHA=`23b2403ccb47e2e2c6b5fa3d22a9a6d93815d9f9bcbc6d11b66f035831adc8d7`，
  包含 `1,309,868` 行、`65,989` assigned/trainable Gaussian；11 个近似身份冲突背景点 fail closed 未分配；
- formal optimization=`300 steps / 15.115s`，peak allocated/reserved=
  `8,001,482,240 / 8,084,520,960 bytes`；D2 checkpoint SHA 前后 exact，base means/scales/quats/SH/RGB
  opacity 均未进入 optimizer；
- r5 为输入正确但缺 runner 入口 identity 三元重核的 noncanonical formal；r6 补齐重核，但暴露
  `np.savez_compressed` ZIP timestamp 导致相同 O0 数组文件 SHA 漂移；r7 改为固定排序/timestamp/权限/压缩参数，
  同一数组重写后 O0/O1 SHA exact，但其 source snapshot 与提交前 EOF 空白清理不再 exact；r8 在同步 SSH 超时后
  以 SIGPIPE/141 failed；r9 由后台托管重跑，9 个源码快照与待提交文件 SHA exact。r7→r9 O1 有小幅 CUDA
  数值漂移，但两 arm aggregate 指标 exact；
- 定向回归=`51 passed`，py_compile、bash syntax、git diff check 通过；下一步只解锁 S2。

### `WS-V33-S2-ROADPATCH-INPAINT-01` canonical closeout

- diagnostic r0 因外层 `nohup` 在注册前创建 run.log 而拒绝 non-empty run 目录；r1 识别出旧 P3 绑定 P2 FP16
  且使用 `(x,y)` 网格；r2 识别出内参为 9 值；三者均保留 failed，未续写；
- r3 的 whole-cell geometry gate 被天空/立面 outlier 污染，`53,541` patches 中 `0` valid；r4–r6 引入
  row-level fail-closed、densest vertical slab 与明确的 sidecar/front-camera support，得到 `822` valid patches；
- canonical index r10=
  `/root/autodl-tmp/runs/worldsim_v33/WS-V33-S2-ROADPATCH-INPAINT-01/20260810T193004Z__s2-patch-index-formal-s0-r10`，
  terminal=`done`；config/index/manifest/summary/status SHA-256=
  `34022784...af79 / 51561eec...a4c / 565741c5...8845 / 4216c652...f1b / 754ab982...063`；
- r10 从 D2 FP32 `1,205,164` 个 native Background rows 中保留 `702,506` eligible rows，建立
  `15,591` 个 1/2/4 m patches，valid=`617/160/45`；index=`4,146,483 bytes`，generated donors=`0`；
- high/boundary 两个 target anchors 由 S1 object-aware delete mask ∩ accepted SAM2 mask、target-view first-hit
  depth 与 cross-view support 冻结，均选择最小覆盖的 4 m patch，且各自 top-5 通过 geometry/visibility/separation；
- r7 完成真实 GPU anchor/top-5 preflight；r8 用 2,150-row dense delta 时 heldout PSNR/SSIM delta=
  `-0.8553 dB/-0.00619`，保持 rejected；r9 在冻结 `maximum_rows_per_target=512` 后以 104 rows 通过门，
  但在 formal index manifest 前，仅作 diagnostic；
- canonical RoadPatch r11=
  `/root/autodl-tmp/runs/worldsim_v33/WS-V33-S2-ROADPATCH-INPAINT-01/20260810T193140Z__s2-roadpatch-formal-s0-r11`，
  terminal=`done`；summary/acceptance/selection/delta/status SHA-256=
  `5de28a02...17d / 9be39845...cd0 / da54ad6d...0f0 / a3105313...014 / f4780ce1...c87`；
- r11 选择 high=`p4-x-000008-z+000009 / 25 rows`、boundary=`p4-x-000009-z+000009 / 79 rows`；
  delta=`24,557 bytes`，保留 exact native row/chunk provenance、rigid `(x,z)` transform、opacity feather、bounded
  RGB affine、scale clamp；candidate 只临时挂载 Background 后渲染并恢复同一对象，base checkpoint 不变；
- heldout B0→B1 mean：PSNR `28.1571546740→28.0731240061`、SSIM
  `0.87145031937→0.87054233897`、LPIPS `0.14966575801→0.15152705088`；冻结门分别为
  `-0.1 dB/-0.005/+0.01`，全部通过；static PSNR `+0.00286484865 dB`，static LiDAR MAE
  `-0.00525204837 m`；
- r11 wall=`69.335 s`，peak CUDA allocated/reserved=
  `8,337,670,144 / 8,420,065,280 bytes`；D2 checkpoint before/after SHA exact；
- Inpaint360GS canonical preflight r12=
  `/root/autodl-tmp/runs/worldsim_v33/WS-V33-S2-ROADPATCH-INPAINT-01/20260810T193426Z__s2-inpaint360gs-preflight-s0-r12`；
  source=`d54c893285c6cb27788e05cce607e7d3cca6388a`、tree clean、Apache-2.0 license
  SHA=`41d80577...5a10`；config/preflight/summary/status SHA=
  `f6dc0291...1845 / 91b5c6a0...30b / 263f336b...8e3 / 292e90df...ec9`；
- 官方要求 RTX 4090/CUDA 11.8、main+LaMa 隔离环境以及 CropFormer/Big-LaMa/SAM/DeAOT/GroundingDINO
  权重；当前 RTX 3090 24,576 MiB 上上述环境/权重与 StreetGS adapter 均缺失，故
  `blocked_single_3090`、`official_execution_attempted=false`；不生成 B2 质量指标；
- RoadPatch 专项=`6 passed`，V3.3/V3.2 定向回归=`52 passed`，py_compile 与 r10/r11/r12 的 8 个
  canonical source snapshots byte-exact；
- S2 以 B1 RoadPatch 为 canonical，B2 作为可审计外部阻塞保留；下一步只解锁 S3。

### `WS-V33-S3-ASSET-VIEWSELECT-01` canonical closeout

- high selector diagnostic r1 与 formal r2 的 selection/input manifest byte-exact；r2=
  `/root/autodl-tmp/runs/worldsim_v33/WS-V33-S3-ASSET-VIEWSELECT-01/20260810T201345Z__s3-viewselect-high-formal-s0-r2`；
  `130` candidates / `119` eligible，heldout/development read=false；selection/input/summary/status SHA=
  `192e5035...b7be9 / 34b1e09e...ef2e7 / 65485a63...0311 / b4c0af50...376c`；
- high 自动集合：A1=`f091 c1`，A2=`f000 c0 + f091 c1`，A4=`f011 c0 + f083/f089/f094 c1`；
  selector 使用 frozen area/mask/sharpness/D2 visibility/occlusion/truncation Q_view 与 yaw/time/camera Q_set；
- high AH r3=`20260810T201830Z__s3-asset-high-formal-s0-r3`，official source=`767b243` clean、
  三权重和 VAE/C-RADIO exact、HF offline；PLY SHA=`13ff42b6...299b / 9bb7f925...8c4a / 9e9875b8...6ca0`，
  inference manifest=`e33fcc65...2d90`，wall=`160.189 s`，peak=`20,137 MiB`，OOM/high=`0`；
- importer r4=`20260810T202210Z__s3-import-high-formal-s0-r4`；A1/A2/A4 Gaussian=
  `101,988/100,783/99,241`，asset SHA=`00397310...5290 / 1a6d9300...6859 / 06d5db85...ec13`，
  deterministic reserialization/reload exact；
- high development canonical r13=`20260810T205300Z__s3-eval-high-development-formal-s0-r13`；
  A0/A1/A2/A4 IoU=`.669876/.658477/.664463/.701490`，boundary F1=`.517563/.497629/.544166/.604799`，
  LPIPS=`.090755/.092720/.095948/.098533`，PSNR=`17.718824/18.480375/17.980733/17.694031`；
  三 auto arm 的六项 retention gate 全过，A4 由冻结 metric order 选中，decision=`28d4f75c...82bf`；
- high heldout canonical r14=`20260810T205600Z__s3-eval-high-heldout-formal-s0-r14`；只比较 A0/A4，
  IoU=`.704974→.728464`、boundary F1=`.505017→.564906`、LPIPS=`.094170→.102697`、
  PSNR=`17.025697→17.009936`；四项 gate 全过，decision=`795ecbc5...8032`，无 optimization，checkpoint exact；
- boundary selector diagnostic/formal exact；formal r8=`20260810T203700Z__s3-viewselect-boundary-formal-s0-r8`，
  `135/127` candidates/eligible，A4=`f039 c0 + f146/f151/f156 c1`，yaw=`17.95°→84.53°`；selection/input=
  `ba05b224...505f / 5cb0ab6c...5856`；
- boundary AH r9/import r11 完成，inference=`2bd2d011...54bd3`，A4=`94,835 Gaussian / 3,632,764 bytes /
  9b2295e5...5dd1f`；错误 CLI SHA 的 importer r10 fail-closed 并保留，未续写；
- V3.2 无 boundary manual asset；r12 使用 immutable D2 native baseline。native vs A4 IoU=
  `.666562/.624832`、boundary F1=`.555343/.492141`、LPIPS=`.015414/.111043`、PSNR=
  `31.006882/16.396747`；A4 retention 失败，生产 override=`ABSTAIN_GENERATED_OVERRIDE`，boundary heldout 未读；
- scene-0242/0255 缺少同协议冻结的 V3.3 S1/S2 输入链，且 boundary transfer 已拒绝，条件确认未执行；
- selector r0 的 `obj_to_world` list schema 与 importer r10 的错误 CLI hash 都按新 run 修复；high r5/r6 因最终
  evaluator snapshot 漂移降为有效 noncanonical，r13/r14 重跑后与提交态 byte-exact；
- S3 专项=`11 passed`，V3.3/V3.2 定向回归=`63 passed`；py_compile/diff check/source snapshot exact；
  下一步只解锁 S4，且只允许 high A4=`06d5db85...ec13` 进入 production delta。

### `WS-V33-S4-SPATIAL-DELTA-01` canonical closeout

- initial all-hard package r1 completed；real-render r2=`rejected`，唯一失败门为 target 外 L1
  `0.8219646>0.5`；其余 exact rollback、toggle、resource、checkpoint/registry integrity 均通过；
- r2 暴露 S1 的 36,736 个 high Background hard assignments 大多是低 instance-opacity 候选；最终 protocol
  保持原视角/门不变，以概率 MAP 边界 `p(instance)>=0.5` 选择 Background，Rigid core 4,525 行仍全部 ERASE；
- canonical package r7=
  `/root/autodl-tmp/runs/worldsim_v33/WS-V33-S4-SPATIAL-DELTA-01/20260810T221300Z__s4-package-canonical-s0-r7`；
  config/package/summary/status SHA=`4b318a67...508a / 3be8ce88...ee43 / cbde9600...0c63 / 4c8332d6...375f`；
- package 只含 base checkpoint/registry reference descriptor，delta/inventory/stacks 共 `4,007,120 bytes`，
  最大 payload=`3,942,422 bytes`，完整 checkpoint copy=`0`；r3/r5/r7 package manifest byte-exact；
- ERASE=`1,614 Background + 4,525 Rigid`，base row deletion=`0`、runtime effective opacity exact zero；
  high RoadPatch=`25` rows（不混入 boundary 79 rows），A4 actor=`99,241` rows，insert provenance 逐行完整；
- canonical evaluation r8=
  `/root/autodl-tmp/runs/worldsim_v33/WS-V33-S4-SPATIAL-DELTA-01/20260810T221700Z__s4-eval-canonical-s0-r8`；
  summary/status/decision SHA=`6f143040...9085 / 87d33d95...c5e5 / 19e3aba6...db9`；
- edit target f091/c1 的 erase/background/actor/full effect pixels=`27,000/6,663/14,844/28,218`，
  erase/actor mask coverage=`0.999741/0.849298`，outside L1=`0.225349<=0.5`；
- 5 个固定视角、4 个 overlay stack 均在卸载后重新 source-render，`20/20` SHA exact；full target render
  二次 replay SHA=`451ae330...50bff` exact，replay rollback exact；duplicate insert index=`0`；
- wall=`66.181 s`、peak CUDA allocated/reserved=`8,433,577,472/8,527,020,032 bytes`，run bytes=
  `11,744,674`，OOM/kill=`0/0`，无训练/optimizer；S4 专项=`9 passed`、V3.3/V3.2 定向回归=`72 passed`，
  py_compile、bash syntax 与 source snapshot exact；
- r3/r4 为有效 noncanonical（最终 validator 后 snapshot 漂移），r5/r6 与最终方法一致但 builder fail-terminal
  尚未进入 snapshot；r7/r8 才是 canonical。下一步只解锁 S5。

### `WS-V33-S5-SEMANTIC-RENDER-01` canonical closeout

- r1=`failed`：input 与 Harmonizer 已完成，但 SAM2 冻结环境没有 SciPy，共享 semantic module 顶层导入形态学
  依赖；修复为 gate builder 内 lazy import，没有安装包或修改冻结环境；
- r2 首次完整完成；development 选择 G1，heldout f060/c1 contact delta=`+0.422686>+0.25`，final arm
  自动回退 G0；r3 仅消除只读 NumPy view warning；r4 只清理 input-prep EOF 空白，协议/阈值/选择均未改；
- canonical r4=
  `/root/autodl-tmp/runs/worldsim_v33/WS-V33-S5-SEMANTIC-RENDER-01/20260810T220500Z__s5-semantic-gate-canonical-s0-r4`；
  config/input/Harmonizer/SAM2 SHA=`b3848289...8c89 / 939a829e...635c / 1da253d8...3863 / c03fe7c9...8b19`；
  summary/status/decision SHA=`1e0bfb59...0761 / 969bb009...fb97 / 988b6647...6159d`；
- semantic gate residual cap=`12/255`、far weight=`0`；五视图 far changed pixels=`0`、actor interior delta=`0`；
  development boundary/contact delta=`-1.837229/-2.771866`；
- delete production 全部 raw 3D exact copy，SAM2 production mass/fraction delta 5/5=`0/0`；unconstrained
  candidate 在 f091/c1 的 mass/fraction=`+0.126399/+0.133885`，1/5 被标记；
- Harmonizer/SAM2 wall=`30.180697/5.701133 s`、peak NVIDIA sampled=`3,553/2,399 MiB`、peak torch
  reserved=`3,940/2,070 MiB`，run bytes=
  `34,548,858`，OOM/kill=`0/0`；r2–r4 的 30 个 RGB 产物跨三次 run SHA 全 exact，decision SHA exact；
- R3D2 commit/tree/license exact，但作者 exported pretrained model 不存在；状态
  `blocked_pretrained_model_unavailable`，无模型加载/训练；temporal 因非相邻五视图显式 not-evaluated；
- S5 专项=`8 passed`，V3.3/V3.2 定向回归=`80 passed`。S5 task=`done`，G1 method=`rejected`，
  production=`G0_raw_3d`；下一步只解锁 R0。

### `WS-V33-R0-INTEGRATION-01` canonical closeout

- diagnostic `222435/222453/222511` 分别因 S2 empty-list、S3 heldout phase、S4 real-render stage 的 exact
  schema 枚举误写而 failed；`222526` 移除报告层冗余 `schema_version` 假设，正式 instance-field validator 已通过；
  `222549` 修复 release `tools/` 目录创建；旧 terminal 均未覆盖；
- diagnostic r6=`20260810T222610Z__r0-integration-diagnostic-s0-r1` 首次 10/10 gates 通过，并冻结 archive SHA；
  formal config 将 expected SHA 固定后执行 canonical r7=
  `/root/autodl-tmp/runs/worldsim_v33/WS-V33-R0-INTEGRATION-01/20260810T222701Z__r0-integration-canonical-s0-r7`；
- config/summary/status SHA=`4b4a20b9...73a9 / c1903255...a2b2 / 0a1396f4...aa4`；44 inputs
  exact，O1/RoadPatch/A4 schema validator 通过，S4 rollback=`20/20`，S5 production exact-safe=`5/5`；
- selected chain=`D2→O1→B1→A4→posterior-gated delta→S5 G0→persistent storage reference→exact release`；
  4/4 required success criteria，overall=`v33_supported`；
- release=`76 files / 18,432,994 payload bytes`、39 JSON evidence、full checkpoint copy=`0`；archive=
  `13,760,114 bytes / cffaad16...44a7`，双构建 exact；解包 content manifest=`e386c14b...a6c53` exact；
- standalone verifier 的 directory/archive 两种模式均返回 valid；R0 wall=`2.721847 s`、GPU compute max=`0`、
  run bytes=`50,851,476`、OOM/kill=`0/0`；S1–S5 selected wall=`379.552 s`、peak=`20,137 MiB`；
- 对 RoadPatch vs Telea 结论为 `not_directly_ranked`；blocked SOTA 不作质量失败；R0 专项=`6 passed`、
  V3.3/V3.2 定向回归=`86 passed`；当前无下一执行授权，F0 LiDAR-EVS 保持 conditional。

## 1. 状态词

只使用：

```text
pending | running | blocked | done | rejected
```

`done` 表示预注册门禁满足；`blocked` 表示工程、资源或外部依赖阻塞；`rejected` 表示研究门禁失败。

## V3.2 注册表

| Task ID | 状态 | 目标 | 当前证据/门禁 |
|---|---|---|---|
| `WS-V32-S0-ROUTE-AND-SOTA-AUDIT-01` | done | source/license/weight/hardware audit | 11 个 official HEAD exact；SAM2.1 large weight bytes/SHA exact；V3.1 immutable |
| `WS-V32-S1-SEMANTIC-LIFT-01` | done | SAM2 temporal mask + Gaussian semantic posterior | r6 identity-aware finalizer 通过；398 masks、334 accepted、0 heldout leak、6 smoke |
| `WS-V32-S2-BACKGROUND-INPAINT-01` | done | background 3D inpainting | canonical r3；1,896 generated rows；两目标与四路 held-out 全部门禁通过 |
| `WS-V32-S3-ASSET-HARVEST-01` | done | complete dynamic actor asset | canonical r3；1/2-view 皆完成，选定 2-view；4 个真实视角回注渲染与资源门通过 |
| `WS-V32-S4-HARMONIZER-01` | done | visual harmonization | canonical r3；non-temporal 执行完成但语义删除门失败，optional diagnostic；temporal externally blocked |
| `WS-V32-S5-MULTIVIEW-UPPERBOUND-01` | blocked | multi-view editing upper bound | 未授权；许可证门未通过 |
| `WS-V32-R0-INTEGRATION-01` | done | V3.2 integration | canonical r4；8/8 gates，mixed checkpoint、extended semantics、actor override 与 exact chunk package selected |

### `WS-V32-S0-ROUTE-AND-SOTA-AUDIT-01` 完成证据

- branch=`research/worldsim-v3.2-semantic-repair`，baseline=`d91e80e`；
- source truth：`configs/worldsim_v32/s0_sources_v1.yaml`；报告：`docs/WS_V32_S0_SOTA_AUDIT.md`；
- SAM2 checkout=`2b90b9f5ceec907a1c18123530e92e794ad901a4`；SAM2.1 large HF revision=
  `665f8e2ad61cf5f53d65644ff27c8ee525124610`；checkpoint=`898,083,611 bytes`，SHA-256=
  `2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318`；
- 无卡模式 cgroup max=`2,147,483,648 bytes`；未运行 torch GPU、模型推理或训练；
- D2 checkpoint/A3*=R0/P2/P3/V3.1 terminal 未 mutation；S1 为唯一 next authorization。

### `WS-V32-S1-SEMANTIC-LIFT-01` 无卡准备（未启动 GPU 任务）

- 冻结配置：`configs/worldsim_v32/s1_semantic_lift_v1.yaml`；
- 无卡验收事实：`configs/worldsim_v32/s1_no_gpu_preflight_v1.yaml`；独立环境为 torch
  `2.5.1+cu124` / torchvision `0.20.1+cu124`，`pip check` 无破损依赖；
- prompt asset：`/root/autodl-tmp/assets/worldsim_v32/s1_prompt_v1/prompt_manifest.json`，SHA-256
  `f60131687e5b4e814dd41a47e69e4b6d4d83d626e20221c43462332d03d9e69d`；
- split：177 个 train frames / 19 个 heldout frames，manifest 检查为零 heldout 泄漏；
- 静态验收：Python compile、4 个 schema/projection/split 单测、runner `bash -n`、`git diff --check` 均通过；
- 未执行 SAM2 mask、Gaussian lift、original/delete/lateral smoke，故 S1 保持 `pending`。

### `WS-V32-S1-SEMANTIC-LIFT-01` r5 invalidation 与 canonical r6

- r5 的旧 finalizer 只核对字段、SHA 与 rigid core count，没有核对 `dataset_instance_id` 对应的
  `instances_info.id` 是否等于同一 role 的 `instance_token`；
- high-support 旧配置为 dataset ID `5` / token `af663…` / rigid index `5`，实际 ID `5` 的 token 是
  `bf9a…`，`af663…` 的真实 dataset ID 是 `13`；因此 r5 的高支持 SAM mask 与 D2 actor core 不是同一对象；
- 新 validator 同时核对 dataset ID ↔ token ↔ actor registry rigid index；v2 负测 exit=`1`，v3 正测通过；
- v3 config SHA-256=`377cd95999dcc02d15782fce06940952826c410d5f4df13846e5dd4c58304960`，
  prompt v3 SHA-256=`8c43b59175da1598b9720bb71d35d573647651ee4075c44ac7b0e265931f6ccf`；
- r6=`20260810T101739Z__s1-semantic-lift-s0-r6` 已 `done`；actor identity contract=`validated`；
  final summary SHA-256=`482dcd067ee91952536e863cded1e18cffa1003bbd3f1b0caa9a18380e93bb4a`；
- 398 masks=`334 accepted / 64 rejected`，heldout leaks=`0`；SAM2 wall=`81.605s`，peak
  allocated/reserved=`1,908,027,904 / 2,204,106,752 bytes`；
- semantic lift wall=`919.436s`，peak allocated/reserved=`15,723,618,816 / 24,683,479,040 bytes`；
  high-support labels=`1,230,548 / 4,525 / 36,767 / 38,028`，boundary-support labels=
  `1,276,927 / 3,728 / 21,033 / 8,180`（negative/core/semantic/ambiguous）；
- 6 个 original/delete/lateral smoke 均非空；D2 checkpoint before/after SHA-256 均为
  `1a061247a753c0d8c9aa7835a52efa2ab1ddc79141a6168adc18b9748de66e7c`；S1=`done`；
- 以下 r5 数值只保留为失效证据，不再支撑任何 selected/canonical 结论。

### `WS-V32-S2-BACKGROUND-INPAINT-01` canonical r3

- 官方 source 审计：3DGIC commit=`0fdbaed680264c02d6222c573434618eb21a44a1`、license SHA-256=
  `c2297cb5b2dd996979a6031ae7c5e112be310f87595c1dc40340be820e0d67e5`；Inpaint360GS commit=
  `d54c893285c6cb27788e05cce607e7d3cca6388a`、Apache-2.0 license SHA-256=
  `41d805773f2aa0b36c2fb69491f64c3079fe3e0671c9848680645fc9e65d5a10`。当前 StreetGS checkpoint 与两者
  原生 schema 都不相同；正式实现明确称为 3DGIC depth-guided cross-view 原理适配，不冒充上游原生运行；
- r1=`20260810T120554Z__s2-3dgic-adapted-s0-r1` 在生成任何候选 checkpoint 前因 boundary-support
  跨视图像素低于 `32` 而 `rejected`。保留的 train-only exhaustive diagnostic 显示目标 frame `31` 之后三相机
  均无几何重叠，重新冻结 frame `24..29` / CAM_FRONT，不使用 held-out；
- r2=`20260810T121342Z__s2-3dgic-adapted-s0-r2` 完成 3,940 行候选与严格重载，但把 high-support
  未观测 Telea 像素持久化进静态背景，导致四路 held-out 平均 PSNR/SSIM delta=
  `-0.495842 dB / -0.007160`，候选为 `candidate_selected=false`；
- canonical r3=`20260810T121829Z__s2-3dgic-adapted-s0-r3`，config/runner SHA-256=
  `8ad962d84009b44464ca70347fec8c935b012ad727e8e233e03648dc41defe29` /
  `8a3c9dae6c937828ff4193bcfc64bda09e0720ba64632c1bdb5848ad6b3de93c`；基础 projection/adapter 测试
  `6 passed`；
- high-support mask=`15,461` pixels，train-only cross-view observed=`7,189`、multi-support=`1,788`、
  unseen Telea=`8,272`；checkpoint 仅持久化 observed geometry。boundary-support mask=`288`，observed=`46`、
  multi-support=`44`、unseen=`242`，小目标保留完整补全。所有 unseen 只声明 provenance、view consistency 与
  artifact，不声明 GT accuracy；
- append=`1,896` rows，Background=`1,205,164 → 1,207,060`；旧 means exact，候选 strict reload、V3.1
  ancestry 对齐，权威 sidecar 对每个新行记录 `GENERATED_BACKGROUND`、confidence、observed flag、target code 与
  source pixel；
- high/boundary candidate effect=`9,928 / 176` pixels，outside L1=`0.042503 / 0.005122`；candidate 对
  completion-reference mask PSNR=`17.127871 / 21.363783 dB`，均高于 source delete 的
  `15.969350 / 19.123467 dB`；
- 四路 held-out 只在生成后读取；平均 PSNR/SSIM/LPIPS delta=
  `-0.022958 dB / -0.000528 / +0.000301`，通过冻结 `-0.1 dB / -0.005 / +0.01` 门；
- selected checkpoint/summary/provenance SHA-256=
  `3d6e13d47291f5b5949ff3adf5598b6e0cffb930c4cbff2200c6e708d82e6e0f` /
  `a07bbf7a1b160d352fd0d3d08be9e217a3d27648eeffec7841f443b5bc871407` /
  `1baf73b81205f66cfe30a6ea3385cdf960b3d8952648031fb34be26a7ef758cc`；D2 source before/after SHA exact；
  wall=`63.908s`，peak CUDA allocated/reserved=`8,051,344,384 / 8,141,144,064 bytes`，NVIDIA sampled=
  `8,125 MiB`，cgroup peak=`39,369,183,232 bytes`，OOM=`0`。S2=`done`，selected=canonical r3。

### `WS-V32-S3-ASSET-HARVEST-01` canonical r3

- r2=`20260810T103527Z__s3-asset-harvest-s0-r2` 因在 CUDA context 显式初始化前调用 PyTorch 2.10
  峰值显存计数器而 `rejected`；实际模型推理未启动、GPU peak=`0 MiB`、无部分输出；
- canonical r3=`20260810T112505Z__s3-asset-harvest-s0-r3`，官方 source commit=
  `767b2439ce47a8b2513038ae0fb2073026f89ee8`，config SHA-256=
  `72a5901348265369e14636090ca02b1c61b10b454bfa76e869188036aefc1cdb`；
- actor identity=dataset ID `13` / token `af663…` / rigid index `5`；CAM_FRONT_LEFT frame `91/51` 均为
  直接 prompt、非 heldout，SAM 被 D2 actor-effect 完整覆盖；input manifest SHA-256=
  `35555c431c44754b0d6fc2a019d7ef9ccf4d47cd7ea2cc8733189ebe2e6cf2dd`；
- Asset Harvester 1/2-view 各生成 `16` 个新视角和非空 `gaussians.ply`；inference wall=
  `113.981s`，peak CUDA allocated/reserved=`15,522,251,776 / 20,772,290,560 bytes`，NVIDIA sampled=
  `20,137 MiB`，cgroup peak=`48,426,651,648 bytes`，OOM=`0`；
- 导入后 1-view/2-view 分别为 `102,303 / 99,045` Gaussians，exact reload，3σ bounds 与冻结
  LWH=`4.51 / 1.76 / 1.65 m` 的最大误差为 `2.22e-16 m`；
- 两资产各在 frame `91/51` 完成 original/lateral/delete，四个 formal render 的 D2 checkpoint
  before/after SHA exact；
- 1-view mean IoU/F1/PSNR/LPIPS=`0.723918 / 0.499815 / 16.078961 dB / 0.104843`；
  2-view=`0.733945 / 0.459813 / 16.671399 dB / 0.094894`；综合指标与前/侧/后目视 QA，
  selected=`high_support_2view`，同时保留 boundary F1 较低的限制；
- summary/evaluation SHA-256=
  `8dc4fc930229fbb17343b0bbcf9ccda632ac54b2e5301d4ca6448bda0d99c2d1` /
  `224eda1a6480941592cef843a685b7c73a70833cbf0a81e0618385466c180a3e`；S3=`done`，生成背面不声明 GT correctness。

### `WS-V32-S4-HARMONIZER-01` canonical r3

- 官方 source commit=`dd5799e50855c5bcb1f6ef52a77b5b644b4798c0`；non-temporal JIT model bytes/SHA-256=
  `1,448,843,112 / ece8e2daa914e8c2a027a2da94e0eb2064491d5b3fd8514009fae9a442e06e90`；code 与 model license
  分别为 Apache-2.0 / NVIDIA Open Model License；
- temporal checkpoint 的官方链还要求 gated Cosmos 0.6B base。固定 revision=`dd55b6858b22ad569976bff207880b8fea839da7`，
  当前无 HF 授权，403 证据保留且未绕过；因此不做 temporal consistency claim；
- 当前 PyTorch 2.10+cu128 环境通过公式等价 RMSNorm 回退和两个 shape scalar device 修复运行导出图；
  BF16 operator 对独立参考式 exact、max abs error=`0`，适配器测试 `4 passed`；
- r1=`20260810T131510Z__s4-harmonizer-nontemporal-s0-r1` 在模型前向前因 device 初始化顺序错误
  `rejected`；r2 工程门通过但视觉 QA 发现 G1 删除区被生成式模型重新解释成车辆，失去候选资格；
- canonical r3=`20260810T131909Z__s4-harmonizer-nontemporal-s0-r3` 覆盖 5 张同 camera 输入：
  G0 original ×2、G1 semantic remove+inpaint ×1、G2 selected Asset Harvester lateral ×2；
  800×450→1024×576→800×450，direct bilinear、无 crop/pad；
- G0/G1/G2 mean outside-mask L1=`3.543039 / 3.831952 / 3.640835` uint8；G1 inside L1=
  `14.217278`、inside changed fraction >8=`0.541750`，同时失败冻结 `<=12.0 / <=0.40` 门；
  `non_temporal_candidate_selected=false`，`final_disposition=optional_diagnostic`，不得默认串入删除输出；
- 5/5 输出均有 `HARMONIZED_2D` provenance，无 3D 写回；D2、S2 selected checkpoint、S3 selected actor
  before/after SHA exact。wall=`35.048s`，常态 median inference=`0.3386s`，peak NVIDIA=`4,077 MiB`，
  peak CUDA reserved=`4,131,389,440 bytes`，OOM=`0`；
- summary/status/grid SHA-256=`4543b5fa2543f6f42aa65f0dbc17f11899de1cc7ebad4aed653200e881f1ba39` /
  `42465759974c60f0fa5407969b12ccf8aeb5952ed7c36904b378a86163b78e51` /
  `086b08b7ab57de7a27d28dda28a84109579ff8cfae15216f88533085e19f3cbf`；S4 task=`done`。

### `WS-V32-R0-INTEGRATION-01` canonical r4

- r1=`20260810T134019Z__r0-final-integration-s0-r1` 因相对 config path 无法冻结 source snapshot 而
  `rejected`，未物化资产、未启动推理；r2=`20260810T134128Z__...` 已物化资产，但在首个 forward 发现
  DriveStudio 必须显式 `set_eval()`，故 `rejected`；r3=`20260810T134421Z__...` 的混合精度与 exact chunk
  均通过，但误把冻结 `MAE<=1` 写成 `max_error<=1`，保持 `rejected`，不事后改写；
- canonical r4=`20260810T134658Z__r0-final-integration-s0-r1`，config/runner/integration/test SHA-256=
  `7011d99f70fc59835569c43bd7e750a5e1981ea67843ef08873bfe4707deb624` /
  `deb1a82f8d60eb659acf1237482ffff26a6d47d615c3eeb50df75d18f0c3c97c` /
  `5ca86b4170d6990bd8b54e15033c090ccc19675b6d8d5340b1bd22ec0eded1f1` /
  `b9cdaab156c6bf3bec19111a22f43764d64926d23bc96e9633a0c073440700d2`；
- S2 generated-background provenance 1,896/1,896 行连续 exact；cross-view true/false=`1,835/61`，target
  code 0/1=`1,824/72`，confidence 与 observed contract exact；S1 high/boundary sidecar 扩展后 SHA-256=
  `7caae12fdfb92f15ae02f5f7fc6f5c8111236f18632516a128b09960b6d79b26` /
  `74dd3679b58423c6e752cd3441a347d8f3f3f1add1e5ce748e75150eb510185b`，旧 prefix/rigid suffix exact；
- P2-style mixed checkpoint bytes/SHA-256=`432,347,490 /
  6d4e4c489f53bf4e7de3f5c405ec37dc63d3f79155aad5237fe175ce0fcd7e5d`，较 S2 FP32 source 减少
  `146,922,064 bytes / 25.363333%`；十个转换 tensor 的 FP16 值、未转换 tensor、schema 与重载全部 exact；
- registry bytes/SHA-256=`3,589 / 6633af150baa4b5adda143b2037091e7647f85966490de5d660fa74968ab6c57`；
  high-support rigid index `5` 使用 S3 99,045-Gaussian `GENERATED_ACTOR`，boundary 和其余 actor 用 V3.1
  fallback；S4 excluded diagnostic、S5 blocked 均为显式字段；
- P3-style package 为 `133 static + 24 actor + skeleton`，manifest bytes/SHA-256=`141,427 /
  af7b402e0b171b11f8c22e4123002f4f844db746ea72f53b77c3de878bf0947d`，payload=`444,282,102 bytes`；
  Background/RigidNodes=`1,207,060/104,704` 行均 covered once，missing/duplicated=`0/0`，85 tensor paths、
  recursive schema 与 non-tensor signature exact；
- fixed views `(31,0)/(51,1)/(91,1)` 的 source→mixed PSNR=
  `68.299304/67.239933/68.432160 dB`，MAE=`0.009614/0.012271/0.009330` uint8；mixed 与 reassembled
  三个 RGB SHA exact，FP16 persistent/FP32 renderer adapter exact；
- wall=`103.099s`，peak NVIDIA=`8,362 MiB`，CUDA allocated/reserved=`7,729.707/8,020 MiB`，cgroup=
  `48,169,205,760 bytes`，run=`948,244,397 bytes`，OOM/kill=`0/0`；input immutability exact，训练/optimizer
  steps=`0`；V3.2 定向测试=`36 passed`；
- summary/manifest/status/report SHA-256=
  `40624cbc79a004e9e07e57b00cebc535b900297a10f0d070fb4e9305a5f7937a` /
  `358d9fc7fde6a535c2ffb0bb2ff34cf1f9df3c151066f3051e24859a5d73a27e` /
  `d31a4f8e62f31dbbf6bbf2520243f5061c68e6682ea5011ef8c64a8dbb541617` /
  `b5397f555270a901013f0a6ce82ba20c8a868d9e22039ba1d9cc2066adf20913`；R0=`done`。

### `WS-V32-S1-SEMANTIC-LIFT-01` r5 历史运行（identity-invalid）

- run=`20260810T093248Z__s1-semantic-lift-s0-r5`，config=
  `configs/worldsim_v32/s1_semantic_lift_v2.yaml`，config SHA-256=
  `ecb6c2bc6f68376c9cd81e3e2a362a30506edfba5772226ad27125f0dcbad706`；
- prompt v2 SHA-256=`771817828acb689e8cab19c4f4c368d8ead24c0d1c154bd1d8bcc283a9b6c071`，使用相对
  video block、逐帧投影 box、双向传播、源图到 800×450 的 exact 坐标映射和冻结 fail-closed QC；
- 263 masks=`212 accepted / 51 rejected`，heldout leaks=`0`；SAM2 wall=`58.337s`，peak
  allocated/reserved=`1,895,410,176 / 2,197,815,296 bytes`；
- semantic lift 263/263 views，wall=`770.733s`，peak allocated/reserved=
  `15,726,013,440 / 24,536,678,400 bytes`，无 CUDA/cgroup OOM；
- high-support sidecar SHA-256=`85ef3c1473b19c6fd5c46ab92d27f78e873e68066dde21fb25beab64ec19e103`，
  labels=`1,295,141 / 4,525 / 3,927 / 6,275`（negative/core/semantic/ambiguous）；
- boundary-support sidecar SHA-256=`983cffe338caa602b8d347e347b95950ec8d2f5d5568ac82dda0a180b9dfca81`，
  labels=`1,276,895 / 3,728 / 21,043 / 8,202`；
- 6 个 original/delete/lateral smoke 完整且每个 actor 的三个 SHA 互异；D2 checkpoint before/after SHA exact；
- final summary SHA-256=`84fbf086dfe8a171b7b4025aceff078bf84b8674f2feb902cd43fe694d112408`；
  旧 finalizer 当时返回 `done`，但 identity 审计后该 run 已失效，不得作为 S1 selected 或 S3 输入。

## 2. V3.1 冻结注册表

| Task ID | 状态 | 目标 | 完成门禁 |
|---|---|---|---|
| `WS-V3-P0-ROUTE-01` | done | 单一 V3 权威计划与 V2 事实冻结 | `076ebdc`；文档一致，链接与 Git diff 校验通过 |
| `WS-V3-A0-NATIVE-BASELINE-01` | done | 三场景原生 StreetGS 基线 | `20260805T175000Z__a0-three-scene-finalize-s0-r2`；3/3 完整矩阵 |
| `WS-V3-F0-FEEDFORWARD-AUDIT-01` | done | Instant NuRec 官方本地能力审计 | canonical audit done；4/11 prerequisites；inference not-run；F1 未解锁 |
| `WS-V3-A1-CALIBRATION-01` | done_off | 成像、位姿和 LiDAR 初始化消融 | 10/10 逻辑项、8/8 唯一训练；C*=C0；finalizer done |
| `WS-V3-A2-ACTOR-DENSIFY-01` | done | actor-aware densification/pruning | D1/D2 formal 完成；A2*=D2 boundary-priority，D1 fallback；D3/D4 未启动 |
| `WS-V3-A3-LOCAL-REFINE-01` | done | 编辑区域局部 Gaussian 精修 | R1 rejected；A3*=R0/D2 exact alias；formal、R2–R4 未授权 |
| `WS-V3-A4-DEPLOYMENT-01` | done | pruning/precision/chunk/LOD 与资产注册 | P0/P5/P1/P2/P3 complete；P1 rejected；P2 mixed checkpoint + P3 exact package selected |
| `WS-V3-R0-INTEGRATION-01` | done | 完整 A0–A4/F0 结论与复现包 | canonical done；63 inputs/23 decisions/12 deliverables/26 manifest files/no-launch exact |

### `WS-V3-A0-NATIVE-BASELINE-01` 完成证据

- fix commit：`436cfc1`；DriveStudio upstream：`e59bda4`；compatibility patch SHA-256：
  `54e7584b6d74431e58f626dfaadd69812d4058d54f82c7941e75aa11f5f94619`；
- 完整 A0 定向测试：`16 passed`；
- canonical smoke：
  `/root/autodl-tmp/runs/worldsim_v3/WS-V3-A0-NATIVE-BASELINE-01/20260805T161656Z__scene0255-catfix-s0-r2`；
- terminal=`done`；torch=`2.1.2+cu118`；原生 mixed-empty cat 复现
  `invalid configuration argument`，patched output=`[59,3] / 177 numel / exact point-color pairing`；
- 真实 1-step DriveStudio 路径完成数据集、LiDAR 实例初始化、一次优化和 checkpoint；controller
  `72.1 s`，peak GPU sample `8,388 MiB`，peak cgroup `5,971,820,544` bytes，无 OOM；
- smoke checkpoint `320,832,362` bytes 只证明执行路径，不注册为 A0 最终模型。

正式三场景矩阵：

| scene | checkpoint / step | global PSNR / SSIM / LPIPS | high actor PSNR / SSIM / LPIPS | high boundary PSNR / SSIM | bg / rigid GS | train s / peak MiB |
|---|---|---:|---:|---:|---:|---:|
| 0230 | `24a39f…e49` / 30k | 24.934 / .740 / .169 | 21.728 / .596 / .121 | 20.165 / .603 | 1,152,614 / 167,299 | 3014.5 / 23,799 |
| 0242 | `16179d…fda` / 30k | 29.107 / .906 / .113 | 19.788 / .665 / .153 | 23.277 / .795 | 843,756 / 86,255 | 2006.2 / 12,783 |
| 0255 | `f8c81c…ef9` / 30k | 25.230 / .743 / .192 | 23.531 / .665 / .058 | 22.991 / .656 | 1,510,936 / 40,447 | 2739.4 / 24,057 |

- scene-0255 正式训练：`20260805T162355Z__scene0255-native30k-s0-r1`；0230/0242 通过 config normalized
  SHA、checkpoint bytes/SHA/step 和同实现合同注册复用；
- actor evaluator 提交 `01cd303`；counterfactual mask 明示不是真值分割，并记录 coverage。actor runs：
  `20260805T173900Z__scene0230-actor-metrics-s0-r1`、`20260805T174100Z__scene0242-actor-metrics-s0-r1`、
  `20260805T174300Z__scene0255-actor-metrics-s0-r1`；peak GPU `8,455 / 7,905 / 8,685 MiB`；
- 0242 boundary role 按注册表保留 `ABSTAIN`；其余 boundary actor 区域/边界带均有正式指标；
- finalizer r1 `20260805T174700Z__a0-three-scene-finalize-s0-r1` 因 native/reuse training resource 字段差异
  `blocked`；`00ba4e8` 归一化后 r2 `20260805T175000Z__a0-three-scene-finalize-s0-r2`=`done`，产出
  `a0_matrix.json/csv` 与 `a0_report.md`；
- A0 只支持固定三场景的描述性结论。跨场景 GS 数与质量不可作因果归因；A1/A2 必须做同场景受控消融。

工作树准备脚本首次创建旧候选 worktree 后，因 `git status --short` 的输出已被 `.strip()` 去除前导空格，
verification literal 写成带前导空格而失败；修正为 `M datasets/driving_dataset.py` 后 verify-only 通过。canonical
patch 为 r2 worktree 和上述 SHA；旧候选只解释首次 smoke，不进入 formal training。

### `WS-V3-A1-CALIBRATION-01` 当前证据

开发场景 `scene-0230` 已完成 C0–C3 30k formal；初始化 provenance SHA 均为
`8951543c33f72f439068237f1a552fae660895f8906afbf4651f5f580981b898`。固定 step 结果：

| variant | global PSNR / SSIM / LPIPS | boundary actor PSNR / SSIM / LPIPS | high actor PSNR / SSIM / LPIPS | total GS | train min |
|---|---:|---:|---:|---:|---:|
| C0-off | 27.746 / .851 / .176 | 27.756 / .892 / .069 | 25.358 / .844 / .094 | 1,360,649 | 52.05 |
| C1-native | 24.979 / .743 / .169 | 22.549 / .700 / .103 | 21.696 / .602 / .120 | 1,316,421 | 53.69 |
| C2-factorized-isp | 25.011 / .743 / .168 | 22.583 / .705 / .104 | 21.779 / .608 / .117 | 1,322,979 | 52.26 |
| C3-bounded-pose | 28.109 / .862 / .167 | 28.169 / .897 / .066 | 25.137 / .846 / .094 | 1,363,040 | 56.14 |

A1-E0 实现提交为 `20c4276`，相机映射修复为 `d85ef27`。冻结配置
`configs/worldsim_v3/a1_endpoints_v1.yaml` 的 SHA-256 为
`60c211625860c25edf92842b88bdb040ea8c180b12fe0fa78f2fc1c342bc4051`。相机对只使用 DriveStudio 权威映射
`0=FRONT / 1=FRONT_LEFT / 2=FRONT_RIGHT` 下的相邻对，支持双向投影、静态/可见/遮挡/深度边缘 mask、
near/far、coverage 与 `ABSTAIN`。

有效正式回填：

| variant / run | E1 valid/candidate/coverage | E1 median/P90 ↓ | E2 high mean/P90/coverage ↓ | E2 boundary mean/P90/coverage ↓ |
|---|---:|---:|---:|---:|
| C0 `20260806T141409Z__scene0230-c0-a1-e0-formal-full-camera-map-fix-s0-r2` | 28,744/266,631/10.780% | .05951/.14719 | .004813/.010895/26.316% | .003547/.006353/35.294% |
| C1 `20260806T141623Z__scene0230-c1-a1-e0-formal-full-camera-map-fix-s0-r1` | 29,151/274,658/10.614% | .06289/.15623 | .004751/.010895/28.070% | .004450/.007626/35.294% |
| C2 `20260806T154541Z__scene0230-c2-a1-e0-formal-full-s0-r1` | 31,299/275,877/11.345% | .06544/.16160 | .004844/.011734/28.070% | .003346/.005447/35.294% |
| C3 `20260806T164852Z__scene0230-c3-a1-e0-formal-full-s0-r1` | 29,846/268,826/11.102% | .06309/.15448 | .004930/.011734/26.316% | .003592/.006537/35.294% |

C2 只改善 boundary role E2，high role E2 与 actor/boundary LPIPS 退化；C3 全图、boundary actor 与位姿稳定性
最好，但 E1 和两个 E2 role 均未严格优于 C0。四次有效评估 checkpoint SHA 前后相同。QA panel 只确认投影落在
真实相邻视野和 actor 支持边界，不能替代人工质量裁决。

首次 formal `20260806T140703Z__scene0230-c0-a1-e0-formal-full-s0-r1` 继承了错误相机 ID 标签，实际把
非相邻画面当成预注册相机对，已保留为 `rejected / INVALID_CAMERA_ID_LABEL_MAPPING`；修复后 run 是唯一有效证据。

最小 LiDAR provenance 实现提交为 `14bc3c2`，冻结配置 SHA-256
`f2fd1712cf4ddd75c1c4d1da4a426dcf7e1340a5fd943066401ba881f51c5639`。正式 run
`20260806T143644Z__scene0230-a1-lidar-provenance-formal-full-witness-s0-r1`=`done`：196 个 LiDAR/pose block、
6,804,832 raw points、24 actor/75,002 actor points；记录的 LiDAR 与 actor tensors 全部 exact match，RigidNodes
初始计数 75,002 exact。held-out sparse depth 172,844/172,844，绝对 median/P90=`7.679/35.958 m`，相对
median/P90=`.6649/.9077`，checkpoint SHA 未变。

背景随机 near/far 点的 CUDA visibility filter 不提供跨初始化 exact replay：源背景初始计数 946,484，三次
replay 分别为 946,597、946,309、946,291。首次 strict smoke
`20260806T142900Z__scene0230-a1-lidar-provenance-smoke1-s0-r1` 因 exact SHA 门禁 `blocked`；协议在查看
正式 depth 结果前冻结为“LiDAR/actor tensor exact 是 gate，随机背景 exact 仅 report”，成功 smoke 和 formal 的
初始 depth 都明确标为 reconstructed witness。没有使用事后计数容差。逐 Gaussian ancestry 留到 A2 instrumentation。

A1-D0 配置 SHA-256 为 `a445078d3bea89a78a0c9e6544a94a2be4c9c2e71f45aec4a9d8878b4c6593c1`；
`20260806T170219Z__scene0230-a1-diagnostics-c0-c3-formal-s0-r1`=`done`。输入速度层为
near-static/low/normal=`2/18/176` 帧；near-static 只有 2 帧，不承担统计结论。C1 位姿修正 translation
median/P90=`7.256/12.215 mm`、rotation=`0.1660/0.35465°`；C3 为 `1.703/2.338 mm`、
`0.02553/0.03337°`。这些是学习修正幅值，不是独立 pose accuracy。

选择协议提交 `60ef079`；配置 `configs/worldsim_v3/a1_dev_selection_v1.yaml` SHA-256 为
`a45699ebf696c875a18832f8db920a6106837a1e4f235dcd9036eff48dfbc609`。协议如实披露结果访问，且不引入
数值容差。正式 run `20260806T171417Z__scene0230-a1-dev-selection-formal-s0-r1`=`done`，输出
`C*=C0-off / done_off`。若 C* 为 C0/C1，确认矩阵保留三个逻辑项但用 source run/checkpoint exact alias。
开发场景冻结时进度为 `4/10` 逻辑项、`4/8` 唯一训练；后续确认矩阵结果如下。

确认配置提交为 `198a681`，SHA-256 为
`63a3cc607ccfddbb714cc81d0570da356263c01c5a68880345953023d2d6a8cd`。正式结果：

| scene / variant | training run | endpoint run | global PSNR / LPIPS | E1 median / P90 | E2 high mean / P90 |
|---|---|---|---:|---:|---:|
| 0242 C0 | `20260806T172514Z__scene0242-c0-confirm-formal30k-s0-r1` | `20260806T181834Z__scene0242-c0-a1-e0-confirm-formal-full-s0-r1` | 30.064 / .1108 | .03147 / .08826 | .008264 / .020697 |
| 0242 C1 | `20260806T181957Z__scene0242-c1-confirm-formal30k-s0-r1` | `20260806T191202Z__scene0242-c1-a1-e0-confirm-formal-full-s0-r1` | 29.161 / .1122 | .03333 / .08971 | .008660 / .021708 |
| 0255 C0 | `20260806T191340Z__scene0255-c0-confirm-formal30k-s0-r1` | `20260806T200907Z__scene0255-c0-a1-e0-confirm-formal-full-s0-r1` | 27.255 / .2086 | .04348 / .14248 | .004772 / .009805 |
| 0255 C1 | `20260806T201041Z__scene0255-c1-confirm-formal30k-s0-r1` | `20260806T210645Z__scene0255-c1-a1-e0-confirm-formal-full-s0-r1` | 25.240 / .1921 | .04277 / .13626 | .003715 / .007704 |

0242 的 boundary role 继续 `ABSTAIN`。0255 C1 虽降低 E1/E2 error，但 high E2 coverage 从 `23.529%` 降至
`21.569%`，boundary/high actor LPIPS 也全部退化，故不通过冻结合同。两个 C* alias run 为
`20260806T211000Z__scene0242-cstar-c0-exact-alias-s0-r1`、
`20260806T211100Z__scene0255-cstar-c0-exact-alias-s0-r1`，不含新训练/评测。finalizer
`20260806T211248Z__a1-three-scene-finalize-s0-r1`=`done`，正式终态 `done_off`。A1 完成 `10/10` 逻辑项、
`8/8` 唯一训练；原始端点方向必须报告为 scene-dependent。

### `WS-V3-A2-ACTOR-DENSIFY-01` I0 完成证据

- canonical r3 项目基线 `70cf2b2` + run 内 source snapshot；当前实现提交 `271d876`；
  DriveStudio upstream `e59bda4`，patched worktree
  `/root/autodl-tmp/third_party/drivestudio-worldsim-v3-a2-r5`；
- 配置 `configs/worldsim_v3/a2_instrumentation_v1.yaml` SHA-256：
  `bac1ec5b3642470a999e7f0cf8ddc9cf5b4d9a1445029c43ae92601929f4bfce`；
- instrumentation patch SHA-256：`87c084f77ed5d6395acce95abb992ca86004bdc47b68154878bf462a0fb345b0`；
- canonical run：
  `/root/autodl-tmp/runs/worldsim_v3/WS-V3-A2-ACTOR-DENSIFY-01/20260809T071500Z__a2-i0-ancestry-formal-s0-r3`；
  terminal=`done`，split=`synthetic RigidNodes deterministic refinement contract`；
- module-off/on 原生 tensor 逐位一致；off 无 ancestry checkpoint key，on 有 key 且 checkpoint round-trip；
- 8 个初始 Gaussian 经 1 split、1 clone、1 prune 后为 10 live / 11 allocated，来源计数
  LiDAR/split/clone=`7/2/1`；parent、lineage、actor ID 与 prune 索引一致；
- r3 duration=`10.10 s`，peak cgroup=`1,377,320,960 bytes`，GPU sample=`0 MiB`；
- patched worktree verify 与当前 working-tree diff 门禁通过；WorldSim 定向测试 `66 passed`。

边界：这是 I0 工程 instrumentation 证据，不是 scene-0230 真实训练或质量结论。boundary/photometric/depth/normal
当前只冻结 update API，normal 无可靠输入时保持 schema-only，background nearest-LiDAR 保持 deferred。D1 必须只增加
actor/background threshold 和 per-actor min/max quota，先做 D0/D1 配对短步 smoke，不得合并 D2–D4。

### `WS-V3-A2-ACTOR-DENSIFY-01` D1 quota-only paired smoke

- 实现提交：`c9b2422`；配置 SHA-256=`6895370625080ccab327e731264e9ebb0f980499b8fec87d02d9efb2e56b14af`；
- DriveStudio worktree=`/root/autodl-tmp/third_party/drivestudio-worldsim-v3-a2-d1-r5`；quota patch SHA-256=
  `c232af2c5fa532016943f399830c85ebba612078871b7c1a296bda816ae7bb1b`；
- canonical run：
  `/root/autodl-tmp/runs/worldsim_v3/WS-V3-A2-ACTOR-DENSIFY-01/20260809T081330Z__a2-d1-paired-smoke1k-s0-r4`，
  terminal=`done`，summary SHA-256=`ec219bb567799d4d84252e86bd4194620f6b5563d6032c43067ff8e155d3b8bd`；
- scene-0230 / seed 0 / 1000 step；D0 后 D1 顺序执行，配置预算匹配，initialization provenance 完全相同；
- D1 policy：actor grad threshold=`0.00025`，Background native threshold=`0.0005`；24 actor 初始/min/max 总量=
  `75,002 / 37,504 / 180,013`；
- D0 final：Background/Rigid=`1,144,022 / 125,915`；D1 final：`1,144,598 / 152,830`；
- D1 5 次 quota event，accepted children=`93,057`，rejected parents=`30,171`，24/24 actor 未超过上限；
- D0/D1 原生 tensor finite；module-off bitwise=true；D1 quota/ancestry checkpoint round-trip=true；
- D0/D1 duration=`110.91 / 110.97 s`，peak GPU=`12,807 / 12,795 MiB`，peak cgroup=
  `5,392,334,848 / 5,661,368,320 bytes`，无 OOM；
- r2 terminal=`blocked / MANUAL_RESTART_IN_TMUX`；r3 terminal=`blocked / GPU is not idle`，遵循 `PIVOT-F22`
  精确回收独立 session 子进程后新建 r4，未覆盖旧 terminal。

裁决：工程 smoke=`done`，只解锁 D1 formal 协议冻结。当前没有冻结 held-out actor/boundary 质量证据，D1 比 D0
多 `26,915` 个 Rigid Gaussian，不能登记为质量改进或直接进入 D2。

### `WS-V3-A2-ACTOR-DENSIFY-01` D1 formal 协议冻结

- 提交=`387dd50`；配置 `configs/worldsim_v3/a2_d1_formal_v1.yaml` SHA-256=
  `ad77db41d9d8c5172804a20b38a2dd92173c3639398d8abc24dc6f4799e8f8e7`；
- frozen pair=scene-0230 / seed 0 / D0→D1 / 每臂 30k；5k candidate grid 只读，不改变训练；
- matched-Gaussian-budget 目标为 D0 最终 RigidNodes 数；D1 选绝对差最小、并列更早的 checkpoint，2% 之外
  `ABSTAIN_BUDGET_NOT_MATCHED`；不做事后 pruning、重训或 quota retune；
- held-out 报告 global、high/boundary actor region/boundary band、两 actor 反事实 mask 并集之外的 non-target、
  24 actor GS、Background/total GS、训练时间、peak VRAM/cgroup；matched 中间 step 的峰值只报完整 30k 上界；
- `80 passed`；read-only preflight=`done`：GPU=`0 MiB`、free disk=`58.39 GiB`、cgroup memory.max=`90 GiB`，
  canonical r4 summary SHA 与 compatibility/instrumentation/quota patch SHA 均匹配；
- 协议冻结提交时 formal 尚未启动；本条本身不构成 fixed-step 或 matched-budget 质量结果。

### `WS-V3-A2-ACTOR-DENSIFY-01` D1 formal 正式结果

- run=`20260809T085400Z__a2-d1-paired-formal30k-s0-r1`；source commit=`f32f96b`；tmux=`ws_a2_d1_f1`；
- terminal=`done`；summary SHA-256=`e3b194c2ed0563385df70ca2043dbc791bedb21068d28dc9d75fb59984c166ac`；
  manifest SHA-256=`f10e6e654ab27289ccb1c995ebbe1ffde913009dbfb3eae0ab4c6414de18a560`；
- materialized configs normalized match=true；D0/D1 初始化 provenance SHA=`8951543c...b898`，Background/RigidNodes
  均为 `946,484 / 75,002`；6×2 checkpoint 网格有效且同源，评测前后 checkpoint SHA 不变；
- D1 quota 158 次 event；24/24 actor 不超过冻结上限；D0/D1 native tensor finite；无 OOM。

Fixed-step（30k）：

| arm | global PSNR / SSIM / LPIPS | high actor PSNR / SSIM / LPIPS | boundary actor PSNR / SSIM / LPIPS | non-target PSNR / SSIM / LPIPS | bg / rigid / total GS | train s / peak MiB |
|---|---:|---:|---:|---:|---:|---:|
| D0 native | 27.7481 / .851207 / .176319 | 24.9965 / .838813 / .094204 | 27.1783 / .882177 / .068895 | 26.8707 / .848887 / .057715 | 1,182,619 / 177,628 / 1,360,247 | 2883.08 / 23,867 |
| D1 quota | 27.7700 / .850915 / .177704 | 25.1238 / .840230 / .096602 | 28.4658 / .899698 / .063419 | 26.8901 / .848493 / .058316 | 1,201,057 / 105,412 / 1,306,469 | 2099.33 / 23,989 |

- quality 轴 D1/D0 更优=`12/7`；quality-cost 轴=`15/9`；两者均 `tradeoff_non_dominated`；
- peak cgroup D0/D1=`10,350,350,336 / 16,012,115,968 bytes`。

Matched-RigidNodes-budget：

| arm | checkpoint | global PSNR / SSIM / LPIPS | high actor PSNR / SSIM / LPIPS | boundary actor PSNR / SSIM / LPIPS | non-target PSNR / SSIM / LPIPS | bg / rigid / total GS |
|---|---:|---:|---:|---:|---:|---:|
| D0 native | 30k exact alias | 27.7481 / .851207 / .176319 | 24.9965 / .838813 / .094204 | 27.1783 / .882177 / .068895 | 26.8707 / .848887 / .057715 | 1,182,619 / 177,628 / 1,360,247 |
| D1 quota | 15k | 25.9290 / .825381 / .217941 | 25.7705 / .829707 / .109637 | 29.2937 / .902828 / .061463 | 24.3371 / .822724 / .090772 | 2,432,701 / 176,741 / 2,609,442 |

- D1 15k 是 5k 网格绝对差最小候选：距 D0 Rigid target `887 / 0.499%`，通过 2% 门；checkpoint SHA-256=
  `b864e5ff772777108fcf2214c0548fd4fdc243c360c79890018d7e0d213a9f58`，elapsed=`1127.66 s`；
- quality 轴 D1/D0 更优=`9/10`；quality-cost 轴=`11/13`；两者均 `tradeoff_non_dominated`；
- D1 在 boundary-support actor 与其 boundary band 多数指标改善，但 global、non-target 与部分 high-support 指标退化；
  RigidNodes 匹配不等于 total GS 匹配，D1 total GS 多 `1,249,195`；
- 裁决：`d2_unlocked=true`，只解锁 D2 boundary/residual ordering + boundary scale cap 协议冻结。证据仅限
  scene-0230 / seed 0；不宣称 D1 全面优于 D0，也不以更多 Gaussian 冒充改进。

### `WS-V3-A2-ACTOR-DENSIFY-01` D2 协议冻结

- 配置=`configs/worldsim_v3/a2_d2_protocol_v1.yaml`；SHA-256=
  `acceb7f4ce0f8dc3745de2fcaca51659891cfd82e4175f5a0e5765d77a01e567`；
- D1 formal summary SHA=`e3b194c2...66ac` 与 closeout commit=`f380dd2` 为冻结前置；
- attribution：dynamic mask 3px morphological boundary band + detached RGB channel-mean L1 residual；用 gsplat
  projected center 做 nearest pixel sampling，按 Gaussian 记录 running mean/count；
- ordering：boundary observed/mean、residual observed/mean、screen-grad 全部降序，Gaussian index 升序作稳定 tie-break；
- cap：复用 native `densify_size_thresh × scene_scale`；pre-cap scale 决定 geometry，随后三轴同比缩放并清零 cap 行
  Adam moments；不新增 RNG；
- D1 eligibility/quota/native cull/Background 精确继承；depth/normal、LiDAR/visibility/provenance pruning 与
  Background intervention 禁止；
- 工程提交=`1065264762569c9832219936ddae6f063d6eaf07`；canonical worktree=
  `/root/autodl-tmp/third_party/drivestudio-worldsim-v3-a2-d2-r8`；D2 patch SHA-256=
  `80fef55195906808d74394af0b997cfccbdb88fd7cb356b45240473e55f357cc`；四层 patch replay/reverse-check 通过；
- D1/D2 materializer normalized-match、真实 `RigidNodes` synthetic integration 与联合回归=`29 passed`；boundary/residual
  各 6 次观测、1 次排序/refinement、6 个 cap、配额上限、optimizer moments、checkpoint round-trip 和 module-off
  native state/RNG bitwise 全部通过；
- paired smoke r1 见下一节；工程门禁本身不能登记 D2 方法结果。

### `WS-V3-A2-ACTOR-DENSIFY-01` D2 配对 smoke

- run=`20260809T111304Z__a2-d2-paired-smoke1k-s0-r1`，terminal=`done`；summary SHA=
  `749c7d15c27cc0798c267aa8af12857f3bea52a52ea9d00f7617a3b3edda3136`，manifest SHA=
  `5cb7879d898839b88a46c8ec7ec34141f3402245490416d589938658f33b4c8d`，source=`c594e0c`；
- normalized configs、initialization provenance 与 frozen initial quota 全匹配；D1/D2 step 1000 totals 分别为
  `Background 1,141,192 / Rigid 152,733` 与 `Background 1,144,988 / Rigid 152,807`；
- D2 记录 `1001` 个 observation event，boundary/residual 各 `10,846,748` 个观测，5 个 ordering/refinement event，
  365 个 cap；cap/quota/finite/checkpoint round-trip 均通过；
- D1/D2 duration=`142.17/141.99 s`，torch peak GPU=`9,615/9,620 MiB`，cgroup peak=
  `16,473,858,048/16,667,971,584 bytes`，无 OOM；
- 裁决=`d2_formal_unlocked=true`；只解锁 formal 协议，不把 1k 规模/训练指标登记为质量结论。

### `WS-V3-A2-ACTOR-DENSIFY-01` D2 formal 协议

- config=`configs/worldsim_v3/a2_d2_formal_v1.yaml`，SHA-256=
  `b66cf795c55dfe65315ecf49c09951482d8d6809ce7d001b901942a6bd9a05bc`，commit=`20b3f4d`，39 tests passed；
- D1 formal r1 作为 immutable exact alias：summary SHA=`e3b194c2...66ac`，provenance SHA=`8951543c...b898`，
  fixed checkpoint SHA=`c9d2a052...af52`，Rigid target=`105,412`；不重训、不改写；
- 新训练仅 D2 30k / seed 0，每 5k checkpoint；fixed D1→D2，matched 从 D2 网格匹配 D1 fixed Rigid，
  maximum gap=2%，无 pruning/retrain/retune/mutation；
- 完整复用 held-out/high/boundary/non-target、checkpoint immutable 与 exact quality/quality-cost Pareto；
- read-only preflight=`done`，SHA=`9cf49af0be9a2676c6c113bee963efb79704bb9434083857684f97bd19caaa28`，
  GPU 0 MiB、free disk 47.92 GiB，D2 smoke/D1 alias/r8 patch/cgroup 门禁通过。

### `WS-V3-A2-ACTOR-DENSIFY-01` D2 formal 正式结果与收口

- run=`20260809T113230Z__a2-d2-formal30k-s0-r1`，terminal=`done`，source=`482fba0`，summary SHA-256=
  `9c41dfc83c9da0a14201e1c719fb3d0e2cf59dd1ad20cd279c6e1a9a1c97de7d`，manifest SHA-256=
  `260af5d99f3d3ece4f2c178f8c18385338432da9fbf94b7d8a4603163db20926`；
- D2 final checkpoint SHA-256=`1a061247...e7c`，30k counts=`Background 1,205,164 / Rigid 104,704`；
  D1 checkpoint SHA 在运行前后保持 `c9d2a052...af52`；D1/D2 initialization provenance 相同；
- 六个 5k grid checkpoint 均 finite 且通过 quota/cap；matched 目标=`105,412` Rigid，选中 D2 30k，
  gap=`708 / 0.67165%`，所以 fixed 与 matched D2 是 exact alias；
- D2/D1 global PSNR=`27.703188/27.770024`、SSIM=`.850333/.850915`、LPIPS=`.178344/.177704`；
  boundary-support boundary-band PSNR=`26.171399/25.770024`、SSIM=`.828868/.821572`、
  LPIPS=`.044568/.048382`；局部边界改善与 global/其他局部轴退化并存；
- fixed/matched strict-quality exact Pareto 都为 `tradeoff_non_dominated`（D1/D2/equal=`11/8/0`）；
  quality-cost 也为 `tradeoff_non_dominated`（`14/9/1`）。D2/D1 wall=`2720.82/2099.33 s`；
- D2 audit=`30,001` observation events、boundary/residual 各 `591,405,097` observations、`295`
  refinement events、capped Gaussian=`70,764`；297 条资源记录、四个 stage completed、无 OOM；
- 状态=`done`。冻结 D2 boundary-residual 为 A3 的 boundary-priority research asset，D1 quota-only 为 fallback；
  不宣称 D2 dominance。`d3_unlocked=false`，D4 未启动。

### `WS-V3-A3-LOCAL-REFINE-01` I0 语义协议

- config=`configs/worldsim_v3/a3_local_refine_protocol_v1.yaml`，SHA-256=
  `03fbf632645326692bbcf18ab18a08b5440c7733c709f925945c78018bb272d0`；A2 closeout=`2246693`；
- input 固定为 D2 scene-0230 30k checkpoint SHA=`1a061247...e7c` 和 actor registry SHA=
  `ed57764e...0c68`；D1 SHA=`c9d2a052...af52` 只作 fallback；
- scene/seed/cameras=`scene-0230/0/[0,1,2]`；actors=`high-support/boundary-support`；edits=
  `lateral +1m/delete`；stride-10 的 19 个 held-out frames 禁止进入优化或支持选择；
- paired footprint threshold/dilation=`2/2px`，affected union dilation=`3px`，depth-order tolerance=`0.05m`；
  source/edited mask 仍是 counterfactual diagnostic；
- S-A/S-B/S-C 严格分层；S-B 禁止 RGB loss，S-C 禁止更新/seed/loss；expected/first-hit/measured depth=
  `diagnostic/T1/T0`，ancestry 不冒充 measured depth；
- R0 为 immutable exact alias；R1 只允许 affected S-A/S-B 的 Background opacity/scale；outside parameter 与
  optimizer、RigidNodes/trajectory/registry exact。R2 appearance、R3 evidence seed、R4 temporal 继续锁定；
- I0 冻结时 stage=`semantic_protocol_and_synthetic_contract_only`、`formal_training_authorized=false`；V2 M5 未提交文件
  不得成为 A3 dependency；新增 `12 passed`，联合 WorldSim V3/materializer 回归 `98 passed`。后续工程门见 I1/I2。

### `WS-V3-A3-LOCAL-REFINE-01` I1 engineering guard / synthetic closeout

- implementation=`9c639dd5a0adcd1f8b5126f7f20d836815b127a6`；DriveStudio patch SHA=
  `155ec58fd2bfdc2e40357035dc20800bf2340b0c1c9ac5972c7c78efbd8cb69b`；独立 A3 工作树通过
  apply/reverse、`py_compile` 与 import；
- run=`20260809T132133Z__a3-r0-r1-synthetic-s0-r1`；summary SHA=
  `2ac123f0603120a103743e59680a31dd4cdf5b6d5fa45605d7c84d36ec337ada`，manifest SHA=
  `8ffa697e15d8a97108d8281a51313119c304fbf0f245d88bfbd127663fde27c4`；联合回归 `110 passed`；
- R0=`immutable_exact_alias_no_optimizer_no_new_checkpoint`，重新命中 D2 checkpoint/config/protocol SHA；
- R1 synthetic 只改变 affected S-A/S-B Background opacity/scale；outside parameter/Adam state、position/color、
  RigidNodes/trajectory、tensor shape/order exact；原 D2 与 A3 module-off RGB/SSIM loss tensor exact；
- 缺 paired provenance/masks 时 DriveStudio fail closed；实际 checkpoint Background/Rigid rows=
  `1,205,164/104,704`，trajectory=`196×24`；
- 状态=`done synthetic_contract_only`，不是 paired/质量证据；该 run 自身记录
  `paired_engineering_smoke_complete=false`。后续 I2 已关闭 paired 工程门，但 `formal_training_authorized=false` 不变。

### `WS-V3-A3-LOCAL-REFINE-01` I2 real S-B/T0 paired smoke / numeric freeze

- sidecar materializer/controller commits=`3b8526af6e3ffb53362ec6641d6f280862ad1cb8 / aac521328eb38e8367e4071601443c0c45086a39`；
  run=`20260809T133911Z__a3-sb-sidecar-s0-r3`，manifest/rows SHA=
  `42474f73fc563a2bba4c52cbec029bb4c28d33a21ca5f3d83ad4311bb7957273 / c5756ecbc0eabee9a576a55297a1739aa20e2af578aa4a5a92e727701b5138fc`；
- high actor 选择 frame 0，boundary actor 选择 frame 31，均为 camera 0 且不是 19 个 heldout frames；四个
  lateral/delete units 冻结 affected/S-B mutable/S-C rows=`16,502/51/16,451`，共 8 个 T0 geometry pixels；
  S-A/RGB=`0/ABSTAIN`，first-hit alpha=`0.5` 仅作 visibility diagnostic；
- paired implementation/fix=`d89e0ace37eda22434470849ec9940360c0e9251 / 78741b3abee07b2c39be6646c63928e8212b6a6b`；
  native Gaussian/trainer regularizer 被双重关闭，S-B occupancy target 只使用 measured T0 LiDAR；delete 使用临时
  visibility 关闭而不删除 Gaussian，lateral 在每 unit 后精确恢复 actor trajectory；
- canonical paired run=`20260809T135921Z__a3-r1-sb-paired4-s0-r2`，summary/manifest SHA=
  `ba4e2b853690f0b9c9bb7bfe039b4571db16c020ce726768a1ff884b09b3557d / de717ba0a5adb1afeb416a15a53ec55f471a8eb841882f784012b04ac86b596c`；
  step `30001–30004`，每 unit 一步；opacity/scale 均有行内 finite nonzero gradient 和变化，outside parameter/Adam、
  Rigid/trajectory/registry、shape/dtype/order exact；checkpoint SHA=
  `e995e7c266d9fed4e64c86813718e46ab4576bbfdf60500a637bdaeaaba78cd1`；
- numeric freeze commit/config SHA=`c02c8c74c671362e86269bd7e00980bfa75ae1c9 / d9289df0b2ac7df7a7c408b5cb1601bc5f874e2922ebc9cb87961aacee43b3e3`；
  steps=`4`，opacity/scaling LR=`0.05/0.005`，affected/mutable cap=`16,502/51`，seed cap=`0`，alpha=`0.5`；
- frozen replay run=`20260809T140534Z__a3-r1-sb-frozen-replay4-s0-r1`，summary/manifest SHA=
  `7d820a53de21f505a5c56043d56556edb8d3a86510488ea3956b7cfa159187c6 / 393e65d5f91c0e2072eebd7c23a1161d46422502220ceeeaa18c04905fec646d`；
  四 unit loss 与 checkpoint SHA 逐位重现；wall/GPU/cgroup=`50.68s/8,286.86MiB/22,631,796,736B`，OOM delta 0；
- 当前定向/联合回归=`26/119 passed`。本节只登记工程可重放性，不登记 heldout/S-B RGB 质量改进；
  `quality_claim_authorized=false`、`formal_training_authorized=false`、R2–R4 未授权。

### `WS-V3-A3-LOCAL-REFINE-01` I3 heldout read-only evaluation / negative closeout

- frozen protocol=`configs/worldsim_v3/a3_r1_eval_protocol_v1.yaml`，SHA-256=
  `eb87a9f2ea7df9bdc050a8d4e4f3cdc7c6a1115ea6f4f69e2fd3c8011904b05a`；protocol/evaluator commits=
  `42508fb810bad5bcd9bd16f6386465c4b9fc8c95 / c8fc5603b471588a5b4d3e54199f4b44b5cf1752`；固定
  19 heldout frames × 3 cameras、R0-derived masks、T0/T1 geometry、non-target/global RGB safeguard、exact Pareto、
  checkpoint immutability 与只读执行；
- observability/memory diagnostic commits=`05cee1ed0ea6144dd2a8bb95a04e9457b1ac64c1 / c9e3df4654933b2e8a21fcd10dc37f1acea9efd8 /
  ef74622382dd2c2b96090282db80f4eb8c315077 / c2eb14f635b5ddee362f584b20bea967dafcfaf2`；当前联合回归=
  `139 passed`。这些提交不改变 protocol SHA、端点、mask、阈值或 checkpoint；
- failed closeout run=
  `/root/autodl-tmp/runs/worldsim_v3/WS-V3-A3-LOCAL-REFINE-01/20260809T144037Z__a3-r1-heldout-eval-s0-r5`；
  exit=`1`，terminal SHA=`eabe266bc3b4e7917391789e95e8cff771c6cac7d2b0567e60205f9cd8a7aad9`；resource audit
  SHA=`d9536f4ec937bee0694a754038b22ab75a4b6b028f20e1e6f42e38e4db9a6280`；wall/GPU/cgroup/run bytes=
  `117.983202636 s / 14,241.398926 MiB / 23,749,709,824 / 299,910`，GPU ceiling=`12,288 MiB`，OOM/OOM-kill
  delta=`0/0`；
- metric/global rows count=`432/114`，SHA=
  `04da7a2503460c075a3164c90d6c08436bbea9f4ec5560ea0417ee40e91aa939 / 04bf741e1da6cfe845b5ee6c9d4cccede54d79a1c8f7178e00abcf737ff7245e`；
  checkpoint before/after SHA=`1a061247...e7c / e995e7c2...8cd1`，run 中无 `.pth`；
- resource-invalid diagnostic primary axes（R0→R1）：coverage=`1.0→1.0`，depth violation=
  `.9157920573→.9081725143`，non-target RGB MSE=`.002095031327→.002095032019`，original-global RGB MSE=
  `.002104032262→.002104032654`；exact Pareto=`tradeoff_non_dominated`。该重算不得登记为合格 heldout 质量证据；
- r2/r4 同样完成 432/114 rows 后只失败 GPU ceiling，峰值=`14,241.777/14,244.924 MiB`；r3 在指标前因
  Rigid quota device mismatch 失败并已修复。失败 run 不覆盖、不删除、不改写为 done；
- final decision：R1 arm=`rejected_resource_gate_and_diagnostic_tradeoff`，A3 task=`done`，`A3*=R0-off`
  （D2 immutable exact alias）。S-A 未物化，formal/R2–R4 未授权；下一阶段为 A4-P0 profile 协议冻结。

### `WS-V3-A4-DEPLOYMENT-01` P0 end-to-end profile protocol freeze

- protocol=`configs/worldsim_v3/a4_p0_profile_protocol_v1.yaml`，SHA-256=
  `8ba96278b7f65957480a343a21977e2e24a537462b7a0b042a3268684d27d9a4`；A3 closeout=
  `10eee3ad30c3729532afecdcc520c1ef542e0210`；selected asset=`A3*=R0-off/D2 exact alias`，checkpoint/config/
  registry SHA=`1a061247...e7c / 115deaf...5e68 / ed57764e...0c68`；R1 输入禁止；
- A2-D2 train、render/eval、registry、actor metrics stage JSON 及 manifest/summary/resource log 均固定 hash，P0
  只读复用，不重跑训练或质量评测；convert 只盘点 inventory，不做参数转换；
- new probe=`process-cold/warm load + 2 warm-up + frames 10/100/190 × cameras 0/1/2 original render`，原生
  `1600×900`、CUDA 同步、P50/P95 nearest-rank、只存 RGB hash/JSON；OS cache eviction 禁止且 filesystem cache
  明示 uncontrolled；
- frozen ceilings=`600 s / 16,384 MiB allocated / 24,576 MiB reserved / 24,000 MiB NVIDIA sampled / 32 GiB
  cgroup / 50 MB run`，OOM delta=`0/0`；recovery=`inventory→runtime_probe→aggregate→resume_audit`，completed
  stages 不覆盖，resume audit 不启动 GPU；
- validator preflight 核对 10 个 immutable path/hash/bytes；协议单测=`4 passed`，联合 WorldSim V3=`143 passed`。
  本条写入时尚未读取新 A4 runtime/load 结果；P1/P2/P3/P5 未授权。

### `WS-V3-A4-DEPLOYMENT-01` P0 v1 resolution block / v2 correction freeze

- runner 实现提交=`199abd99d642747241b79ce543c8eb9096553a1d`；v1 formal r1=
  `20260809T151539Z__a4-p0-profile-s0-r1` 完成 inventory/runtime/aggregate/resume 四个 stage，resume audit 未导入
  torch、前后均无 GPU compute process、三个 completed stage hash 全部复用；
- r1 终态=`blocked`，唯一失败 audit=`native_resolution_exact`。11 个 runtime rows 均为 `800×450`；warm-up 两行
  RGB hash exact，rows SHA=`fbf801eb...1e1a`，runtime stage SHA=`d86ec9fa...9b7`，terminal SHA=
  `9084e49c...e5ed`；run 内无 checkpoint/render media；
- source config SHA=`115deaf...5e68` 明确冻结三路 `downscale_when_loading=[2,2,2]`。因此 v1 的 `1600×900`
  是传感器原始尺寸，不是当前 checkpoint 的模型原生 render 尺寸；v1 不改写，也不登记为正式 P0 性能结果；
- r1 resource diagnostic 自身 passed：wall=`62.144008 s`，prepare=`52.069271 s`，cold/warm load=
  `.341640/.343181 s`，render P50/P95=`.029865/.092485 s`，FPS=`20.501837`，torch allocated/reserved=
  `7,913.31/8,232 MiB`，NVIDIA sampled=`8,574 MiB`，cgroup peak=`24,775,639,040 bytes`，OOM=`0/0`；这些值
  只用于 v2 风险审计；
- v2 protocol=`configs/worldsim_v3/a4_p0_profile_protocol_v2.yaml`，SHA=`43db7182...3f18`；只把模型原生尺寸纠正
  为 `800×450`，冻结 v1 protocol/manifest/runtime stage/rows/resource audit/terminal 六项证据，其余输入、矩阵、
  ceiling、recovery 和 claim boundary 不变。v2 必须新目录完整 rerun，不复用 v1 measured runtime stage；
  validator exact 核对 16 个 inputs，协议测试=`7 passed`，联合 WorldSim V3=`152 passed`；P1/P2/P3/P5 仍未授权。

### `WS-V3-A4-DEPLOYMENT-01` P0 v2 canonical profile result

- canonical=`20260809T152923Z__a4-p0-profile-v2-s0-r2`，source commit=`b191afaa...8897`，exit=`0`，
  terminal=`done`；summary/manifest/resource/rows SHA=`0278a320...e92 / 12df93b3...a0f5 / b89c93bb...5fac /
  4a94b1fb...a934`；13/13 audits true，checkpoint/registry before-after exact，无训练、optimizer、checkpoint 或媒体输出；
- inventory：checkpoint/config/registry=`578,819,674 / 4,661 / 3,721,428 bytes`，checkpoint+registry=
  `582,541,102 bytes`；static block=`1 monolithic`，actor assets=`24 total / 23 available / 1 unavailable`，Gaussian=
  `1,205,164 Background / 104,704 RigidNodes`；convert=`inventory_only_no_parameter_conversion`；
- performance：wall=`60.784519 s`，prepare=`50.420569 s`（82.95%），trainer construction=`1.885644 s`，
  process-cold/warm load=`.391351/.397158 s`；9-view render P50/P95=`.068017/.127388 s`，FPS=`16.377547`；
  filesystem cache=`uncontrolled_report_explicitly`；
- resources=`passed`：allocated/reserved/NVIDIA sampled=`7,913.31/8,232/8,574 MiB`，cgroup peak=
  `24,474,128,384 bytes`，run bytes=`85,169`，disk free=`45,292,818,432 bytes`，OOM/kill=`0/0`；
- recovery：no-torch dry-run=`.160304 s`，复用 3 个 completed stage，输入/输出=`16,256/919 bytes`，无 GPU launch；
  summary 的全部性能值只适用于 scene-0230/seed-0/800×450/单进程，不产生 concurrency 或质量 claim；
- P0=`done`。prepare 是明显主导项，而 load/runtime 未触发冻结资源门，故不先承担 P1/P2/P3 的质量或数值风险；
  下一门禁只冻结 reference-only P5 registry/resume 协议，再决定是否执行。P1/P2/P3 仍未授权。

### `WS-V3-A4-DEPLOYMENT-01` P5 registry/resume protocol freeze

- protocol=`configs/worldsim_v3/a4_p5_registry_resume_protocol_v1.yaml`，SHA=`51acb935...5874`，P0 closeout=
  `9811c03c...e0b5`；validator exact 核对 P0 protocol/manifest/summary/resource/rows/terminal 与 checkpoint/config/
  actor registry 共 9 项 path/hash/bytes，P0 terminal/13 audits 必须 done/true；
- output=`artifacts/deployment_registry.json`，schema=`worldsim-v3-deployment-registry-v1`，mode=
  `reference_only_immutable_manifest`；checkpoint copy/rewrite 禁止，registry ceiling=`2,000,000 bytes`；
- static 固定 `models.Background / 1 asset / 1,205,164 GS / monolithic reference / not independently extractable`；
  actors 固定 `models.RigidNodes / 24 assets / 104,704 GS / 23 available / 1 unavailable`，compact entry 只保留身份、
  selector、count、flat-index hash 与 source registry hash，不复制大段 ranges；
- reload=`fresh DriveStudio / load_only_model / exactly one checkpoint load / 0 render`；核对模型总量与全部 actor
  count/index hash，不构造 optimizer、不训练；filesystem cache 仍 uncontrolled；
- recovery=`input_audit→registry_materialize→reload_smoke→aggregate→resume_audit`，completed stage 不覆盖，resume=
  no-torch/no-GPU；ceilings=`180 s / 16,384 MiB allocated / 24,576 MiB reserved / 24,000 MiB NVIDIA / 32 GiB
  cgroup / 5 MB run / OOM 0`；required audits=`14`；
- 协议测试=`6 passed`，联合 WorldSim V3=`158 passed`。本条写入时尚未执行任何 P5 formal measurement；下一动作
  只实现并提交 runner。P1/P2/P3 未授权。

### `WS-V3-A4-DEPLOYMENT-01` P5 formal registry/resume result

- runner=`4de2126e...d8078`。formal r1=`20260809T155209Z__a4-p5-registry-resume-s0-r1` 完成 input audit 与
  registry materialize，产出 `14,729-byte` compact registry 后在 reload 读取 `RigidNodes.points_ids` 时报错；
  terminal=`blocked`，SHA=`61d30a11...773e`，registry SHA=`e48bccdf...9039d`，旧证据不覆盖；
- root cause 是 DriveStudio checkpoint state key=`points_ids`、loaded runtime attribute=`point_ids`。fix=
  `0e899b2e6dcf7d5a091a0a4092ea99767c982357`，只修运行时读取并锁定别名拒绝；protocol SHA 保持
  `51acb935...5874`。聚焦测试=`15 passed`，联合 WorldSim V3=`167 passed`；
- canonical r2=`20260809T155753Z__a4-p5-registry-resume-s0-r2`，source=`0e899b2...2357`，exit=`0`，
  summary/manifest/resource/registry SHA=`0c86ff68...8744 / 78830d74...58bd / f6c06df0...3ac4 / e48bccdf...9039d`；
  required audits=`14/14 true`，source checkpoint/registry before-after exact；
- compact registry=`14,729 bytes`，canonical content SHA=`02467963...cb6`；`1 static / 1,205,164 GS`，
  `24 actors / 104,704 GS / 23 available / 1 unavailable`，fresh reload 后 24/24 actor count/index hashes exact；
- reload=`52.320687 s`，prepare/trainer/load=`49.631015/1.987488/.445515 s`，checkpoint load count=`1`，render=`0`，
  无 optimizer/training/checkpoint copy；resources=`passed`：allocated/reserved/NVIDIA=`7,188.73/7,226/7,564 MiB`，
  cgroup=`24,498,089,984 bytes`，wall=`60.437454 s`，run=`102,229 bytes`，OOM/kill=`0/0`；
- no-torch resume=`.127572 s`，torch 未导入、GPU launch=false、四个 completed stage 按 hash 全复用；P5=`done`。
  结论只覆盖 reference-only packaging 与单实例 recovery。A4 仍需 P1/P2/P3；下一步只冻结 P1 protocol，P2/P3
  继续未授权。

### `WS-V3-A4-DEPLOYMENT-01` P1 contribution-prune protocol freeze

- protocol=`configs/worldsim_v3/a4_p1_contribution_prune_protocol_v1.yaml`，SHA=`4f893c09...429b`，P5 closeout=
  `4db43ddc...1779b`；full validator exact 核对 source checkpoint/config/registry、D2 quality、P0 profile、P5
  registry/recovery 共 13 files + 1 directory，33 个 frozen actor masks digest=`429f3693...43cf`；
- ranking 只用 train frames `[5,45,85,125,165,195] × 3 cameras`，从 gsplat near→far intersections 稳定计算
  occlusion-aware `T_before×alpha`；`[10,50,90,130,170,190] × 3` heldout contribution 只作 audit，不参与排名；
- arms=`source/b05/b10/b20`。Background 与 23 个 available actors 分资产按冻结 key 排名并删除固定 fraction，
  unavailable actor 保持空；所有 Gaussian/`points_ids`/ancestry row 同 mask，Sky/LPIPS/trajectory/step exact；
- full quality=`19 heldout frames × 3 cameras`，复用 source 的 33 mask bytes，禁止 candidate 重生成。global
  PSNR/SSIM/LPIPS 最大退化=`.10 dB/.002/.002`；actor/boundary=`.20 dB/.005/.005`、MAE `+.002`；non-target=
  `.10 dB/.002/.002`、MAE `+.001`；缺失或 nonfinite 即 reject；
- selection 固定为通过 reload/count/invariant、质量、compression 与资源门的最大 prune fraction；若全失败，P1 method
  rejected 且生产资产 exact fallback 到 source。不得看结果新增 arm/阈值，也不把 bounded loss 写成质量提升；
- recovery=`11 stages`，ceilings=`1,800 s / 20,480 MiB allocated / 24,576 MiB reserved / 24,000 MiB NVIDIA /
  48 GiB cgroup / 2.5 GB run / 30 GB disk floor / OOM 0`，required audits=`21`；协议测试=`11 passed`，联合
  WorldSim V3=`178 passed`。本条写入时尚无 P1 新 measurement；下一动作只实现并提交 runner，P2/P3 未授权。

### `WS-V3-A4-DEPLOYMENT-01` P1 formal contribution-prune result

- runner=`19cab2cf40b8ed8ef9a4ad1ba8cce4cc8cf67163`；正式运行前聚焦测试=`23 passed`，实现提交后完整
  `tests/*worldsim_v3*.py` 回归=`190 passed`。canonical r1=
  `/root/autodl-tmp/runs/worldsim_v3/WS-V3-A4-DEPLOYMENT-01/20260809T165058Z__a4-p1-contribution-prune-s0-r1`，
  exit=`0`、terminal=`done`、21/21 audits true；summary/manifest/resource/terminal SHA=
  `7c5347e3...7119 / 486342ba...61ac / 8b6073ed...4b7c / 80dd8178...c645`；
- contribution scan=`198.712 s`，完整覆盖 `18 train ranking + 18 heldout audit-only` views；score NPZ=
  `30,376,517 bytes`、SHA=`0165401a...69a9`，15 个数组的 dtype/shape/content hash exact。峰值 allocated/
  reserved/NVIDIA/cgroup=`14,342.71 MiB / 14,892 MiB / 15,234 MiB / 24,710,811,648 bytes`；
- materialization 全部 exact：

| arm | checkpoint bytes | byte reduction | Background / RigidNodes GS | reload/count/invariant |
|---|---:|---:|---:|---|
| source | 578,819,674 | 0 | 1,205,164 / 104,704 | exact |
| b05 | 554,938,306 | 23,881,368 | 1,144,906 / 99,480 | exact |
| b10 | 531,056,962 | 47,762,712 | 1,084,648 / 94,246 | exact |
| b20 | 483,292,674 | 95,527,000 | 964,132 / 83,773 | exact |

- source replay 与冻结 historical global/actor/boundary/non-target 逐端点 exact。候选门禁结果：

| arm | failed safeguards / 31 | 关键失败 | quality decision |
|---|---:|---|---|
| b05 | 3 | occupied PSNR `-0.117684 dB`；global PSNR `-0.110926 dB`；non-target PSNR `-0.125462 dB` | rejected |
| b10 | 12 | global PSNR `-0.364281 dB`、LPIPS `+.004091`，并含 non-target/动态端点失败 | rejected |
| b20 | 15 | global PSNR `-0.682975 dB`、LPIPS `+.007970`，并含 human/non-target 端点失败 | rejected |

  三臂 actor/boundary 的部分指标不退化，但不能覆盖 all-endpoint 合同；禁止事后增加 `<5%` arm 或放宽门限；
- 9-view runtime 只作报告：source/b05/b10/b20 load=`.365/.387/.365/.356 s`，P50=
  `.0447/.0329/.0700/.0399 s`，P95=`.1402/.0785/.1538/.1008 s`，FPS=`18.11/23.64/14.54/22.73`。
  cache 未控制且结果非单调，不用于选择；
- final resources=`passed`：wall=`605.281 s`，allocated/reserved/NVIDIA=`14,342.71/14,892/15,234 MiB`，
  cgroup=`26,264,842,240 bytes`，run=`1,610,165,885 bytes`，disk free=`43,679,989,760 bytes`，OOM/kill=`0/0`；
  no-torch resume=`2.316 s`，10/10 completed stages 复用、GPU launch=false；
- `method_state=rejected_quality_or_integrity_gate`，selected=`p1-source` immutable exact alias，P1 experiment=`done`。
  这是 scene-0230/seed-0 的冻结负结果，不是所有贡献度剪枝或跨场景剪枝不可行的结论。A4 仍需 P2/P3；下一步
  只冻结 P2 FP16 协议。

### `WS-V3-A4-DEPLOYMENT-01` P2 mixed-precision protocol freeze

- protocol=`configs/worldsim_v3/a4_p2_mixed_precision_protocol_v1.yaml`，SHA=`6558fb3f...6d4e`，P1 closeout=
  `e733cbed...283d`；9 个 exact files + 33-mask directory 锁定 P1-selected source 与 P1 canonical 21/21 evidence，
  不允许将 b05/b10/b20 rejected checkpoint 输入 P2；
- arms 固定 `p2-source` 与 `p2-gs-param-fp16`。候选只把两 Gaussian 模型的 `_scales/_quats/_features_dc/
  _features_rest/_opacities` 共 10 个 float32 tensors 转为 float16；候选必须 bitwise 等于 `source.to(float16)`，
  checkpoint schema、Gaussian/actor count/index 保持 exact；
- source dtype audit 在新 P2 measurement 前确认 Background means 范围 `[-686.0377,2996.3384] m`，若 FP16
  roundtrip 最大误差=`.999267578125 m`；故 Background/Rigid means、Sky、LPIPS、trajectory、points_ids、ancestry/
  quota/boundary state 与 step 全部保留 FP32/原 dtype exact；
- runtime policy 是 persistent converted parameters FP16、`collect_gaussians` 后 renderer inputs FP32、autocast=false。
  本实验不声称 FP16 gsplat kernel 或 Tensor Core speedup，checkpoint bytes 下降也不自动等于 peak VRAM 下降；
- quality 仍为 57 views/33 frozen masks/31 endpoints；global 门=`.05 dB/.001/.001`，actor/boundary=
  `.10 dB/.0025/.0025/+ .001 MAE`，non-target=`.05 dB/.001/.001/+ .0005 MAE`，必须 31/31；
- runtime=`2 arms × frames 10/100/190 × cameras 0/1/2`、2 warm-up、800×450、sync/nearest-rank；recovery=
  7 stages；resources=`900 s / 16,384 MiB allocated / 24,576 MiB reserved / 24,000 MiB NVIDIA / 48 GiB cgroup /
  1 GB run / 30 GB disk floor / OOM 0`；required audits=`19`；
- full validator exact 通过 10 项输入记录与 source 10-field dtype audit；协议测试=`9 passed`，联合 WorldSim V3=
  `199 passed`。本条没有 P2 conversion/render measurement；下一动作只实现并提交 runner，P3 未授权。

### `WS-V3-A4-DEPLOYMENT-01` P2 mixed-precision formal

- protocol/runner/fix=`6558fb3f...6d4e / 1cd9a6e / dcf2822`。r1=
  `20260809T174337Z__a4-p2-mixed-precision-s0-r1` 完成 candidate、quality、runtime、aggregate 与 resume，但 mapped
  Gaussian Parameters 未进入 persistent-byte inventory，finalizer 唯一 audit 失败；terminal SHA=`5ef3dab6...74c0`，
  run 保留 `blocked`。修复只补账本遍历与回归，不改协议或测量合同；
- canonical r2=`20260809T174850Z__a4-p2-mixed-precision-s0-r2`，source=`dcf2822...9860`，exit=`0`、terminal=
  `done`、summary/manifest/resource SHA=`980f9b0f...1103 / bed45626...98cb / 221d5e82...0df5`，19/19 audits；
- candidate checkpoint=`7be87e8b...7448 / 432,111,754 bytes`，相对 source `578,819,674` 减少
  `146,707,920 / 25.346049%`；registry=`69c4f38a...8a27 / 3,721,277 bytes`。10-field half bytes、75 preserved
  tensor、schema/count/index/reload 与 renderer FP32 输入全部 exact；persistent parameter bytes=
  `394,641,424→247,936,208 / -37.174307%`；
- source 31-endpoint replay 最大绝对差=`0`；candidate 31/31 safeguards 通过。最大门限消耗为 high-support boundary
  LPIPS `+0.000036410 / 1.4564% budget`；global human SSIM=`-0.000009515`，boundary-support actor PSNR=
  `-0.000755311 dB`。这是 bounded-loss pass，不登记质量提升；
- runtime 只报告 source/candidate load=`.33669/.47407 s`、P50=`.04583/.08721 s`、P95=`.13170/.09750 s`、
  FPS=`17.256/13.065`；filesystem cache 未控制且 P50/FPS 退化，不登记 speedup；
- resource=`passed`：wall=`206.548 s`，allocated/reserved/NVIDIA=`7,754.05/8,072/8,426 MiB`，cgroup=
  `29,673,631,744 bytes`，run=`436,430,167 bytes`，disk free=`42,806,071,296 bytes`，OOM/kill=`0/0`；resume=
  `1.217 s`、6/6 stages、no torch/no GPU；
- selection=`p2-gs-param-fp16`，method=`selected_mixed_precision_parameter_storage_fp32_render`，P2=`done`。结论只
  覆盖 scene-0230/seed-0，且不声称 FP16 renderer/Tensor Core/VRAM 或 load/render speedup；下一步只冻结 P3。

### `WS-V3-A4-DEPLOYMENT-01` P3 exact chunk package protocol freeze

- protocol=`configs/worldsim_v3/a4_p3_chunk_protocol_v1.yaml`，SHA=`dfaaba79...1b41`，P2 closeout=
  `e954e23c...ebe7`；9 个 exact files + 33-mask directory 锁定 P2-selected mixed checkpoint 与 P2 canonical
  19/19 evidence，禁止接入原 FP32 source 或 P1 rejected prune candidates；
- static contract 是 origin `[0,0] m`、50 m XY half-open cells、数值 `(ix,iy)` 排序；source inventory=
  `133 chunks / 1,205,164 rows / count 1..330,169 / 98 sparse<100 / 7 dense>=10,000 / 69,393 rows within
  .25 m of cell edge`。稀疏和离群 chunk 全保留，不设 min count、merge 或 post-hoc cell-size search；
- Background/Rigid 分别冻结 25/26 row tensors。每个 asset 保存全部 row fields 与升序 `int64 source_flat_indices`；
  24 个 actor assets 中 23 个非空 source rows 全部 interleaved，actor 14 仍输出完整 zero-row asset。static/actor
  source inventory SHA=`d78fa6e...27cae / 384870e6...f23a`；
- package=`manifest + skeleton + 133 static + 24 actor = 159 files`；shared skeleton 对 row tensors 使用唯一 sentinel，
  其余 state exact。row values 不重复；manifest 绑定逐文件 path/SHA/bytes/count/bounds/index digest；候选只在内存
  scatter 重组并要求 recursive schema、tensor shape/dtype/value、reload/registry 与 P2 FP16-persistent/FP32-renderer
  adapter exact；禁止复制 source 或持久写出 reassembled checkpoint；
- quality=`57 views + 33 masks`，source 先 exact replay P2 selected quality，chunk 再要求 57 个 RGB SHA 和 31 个
  endpoints exact；runtime=`2 arms × 9 views`、2 warm-up、800×450、读取全部 package、cache uncontrolled，仅报告。
  选择成功为 `selected_exact_chunk_package`；任一 integrity/quality/resource 门失败则 exact fallback P2；
- recovery=8 stages，resources=`900 s / 16,384 MiB allocated / 24,576 MiB reserved / 24,000 MiB NVIDIA /
  48 GiB cgroup / 1 GB run / 30 GB disk floor / OOM 0`，required audits=`21`。full validator passed，协议测试=
  `12 passed`，联合 WorldSim V3=`222 passed`。本条没有 P3 materialization/render/formal measurement；下一动作只
  实现并提交 runner，P4 继续未授权。

### `WS-V3-A4-DEPLOYMENT-01` P3 exact chunk package formal

- runner commit=`aba55777f38a3d8e4363d2ff7d546d412214b481`；focused=`23 passed`，WorldSim V3 full=
  `233 passed`。canonical r1=
  `/root/autodl-tmp/runs/worldsim_v3/WS-V3-A4-DEPLOYMENT-01/20260809T184240Z__a4-p3-chunk-s0-r1`，
  exit=`0`、terminal=`done`、21/21 audits passed；summary/manifest/resource/terminal SHA=
  `f8e6e166...a293 / 8b79d355...bd7af / 55ee6f0b...55e81 / 80dd8178...c645`；
- package manifest=`35a3f1fe...64b8 / 143,913 bytes`；133 static、24 actor、1 skeleton 加 manifest 共
  `159 files / 444,177,055 bytes`，158 payload=`444,033,142 bytes`。source checkpoint=
  `432,111,754 bytes`，package 开销=`12,065,301 bytes / +2.792171%`；skeleton=
  `101,176,684 bytes / 51 row sentinels`，没有 source copy 或 persistent reassembled checkpoint；
- 85 个 tensor path 的 recursive schema/shape/dtype/value SHA 与 non-tensor signature 全部 exact；Background=
  `1,205,164`、RigidNodes=`104,704` rows 均 covered once，missing/duplicated=`0/0`；24 actor assets 完整，actor 14
  为显式 zero-row asset。source checkpoint/registry SHA 前后保持 `7be87e8b...7448 / 69c4f38a...48a27`；
- source replay 31 endpoints max abs diff=`0`；chunk 相对 source 的 57 RGB SHA、31 quality endpoints 和 masks
  exact。quality adapter 两臂均为 `57 renderer / 114 SH` observations，runtime 为 `11/22`，全部 FP32 且
  autocast=false；P2 mixed-persistent/FP32-renderer 合同 exact；
- 9-view runtime（2 warm-up、800×450、sync、cache uncontrolled）：source/chunk load=`.907071/4.177543 s`，
  P50=`.030126/.039505 s`，P95=`.094460/.105862 s`，FPS=`21.2783/20.4471`。package 没有缩小，load/reassembly
  与 render 均未加速；结果只支持 exact spatial/actor asset separation，不支持 streaming、speedup 或 concurrency claim；
- resource passed：wall=`221.786 s`、allocated/reserved/NVIDIA=`7,614.99/8,066/8,420 MiB`、cgroup=
  `32,689,958,912 bytes`、run=`444,885,133 bytes`、disk free=`42,359,705,600 bytes`、OOM/kill=`0/0`；resume=
  `1.104 s`、7 actions、159 artifacts、no torch/no GPU；
- selected=`p3-chunk-package`，method=`selected_exact_chunk_package`，P3=`done`。P0/P5/P1/P2/P3 最低完成集
  全部满足，A4=`done`；P4 保持 optional，下一任务是 R0 前置的 F0 官方能力审计。

### `WS-V3-F0-FEEDFORWARD-AUDIT-01` Instant NuRec protocol freeze

- protocol=`configs/worldsim_v3/f0_instant_nurec_audit_v1.yaml`，SHA=`2004a029...fd611`；runner SHA=
  `249f26d5...8e4a`；official checkout revision/tree=`1ce2288e...8d0 / 96e36fa4...5dc0`，16 个关键文件 hash
  exact 且 clean。协议/源码指纹测试=`8 passed`、WorldSim V3 联合回归=`241 passed`；
- 三份正式 weights-only PTH 的 profile/path/HF commit/bytes/SHA-256/Xet hash 全冻结；code/model/dataset license
  分层为 Apache-2.0 / NVIDIA Open Model License / NVIDIA Autonomous Vehicle Dataset License，NCore gated 与
  terms acceptance 不由 source license 代替；
- paper/model-card contract=`calibrated multi-camera RGB + 2–4 Hz + per-image pose/intrinsics + optional cuboids →
  static/dynamic 3DGS + sky + per-camera ISP`；standalone CLI contract=`NCore V4/FTheta/CUDA → static PLY only`。
  CLI 不读 LiDAR，不导出 dynamic、sky、ISP、actor registry/trajectory 或 depth/point map；
- local inference gate=11 条全合取前置：Python 3.11、uv、CC≥8.0、VRAM≥30,720 MiB、RAM≥32 GB、disk free≥
  100 GB、exact weight、licensed NCore input、terms record、exact clean source 与 CLI help。任一失败时禁止构造
  inference command，也不安装依赖、下载权重/gated data 或启动 GPU；
- F1 默认裁决=`conditional_not_unlocked`；static PLY 不是 exact StreetGS checkpoint，且当前无 scene-0230
  nuScenes→NCore exact converter/actor registry。该边界不等于宣称 upstream 永久不可用，DGGT fallback 也不阻塞 R0；
- 本条没有 formal run、inference wall/VRAM 或质量测量。下一动作只运行已提交协议的 canonical read-only audit。

### `WS-V3-F0-FEEDFORWARD-AUDIT-01` canonical local audit

- canonical=
  `/root/autodl-tmp/runs/worldsim_v3/WS-V3-F0-FEEDFORWARD-AUDIT-01/20260809T192139Z__f0-instant-nurec-audit-s0-r1`，
  source=`ab76f1901b60491af7bc8355589e149fcb69fe04`，exit=`0`、terminal=`done`；summary/manifest/terminal SHA=
  `d111c457...be37 / f1c76fdd...6a11 / 207758b9...15c6`；run=`65,917 bytes`，9/9 manifest artifacts 的
  path/bytes/SHA 复核 exact；
- official checkout revision/tree、16 file hashes、code signatures、no-LiDAR-reader 与 clean status 全通过；CLI help
  exit=`0`。官方 `ncore_input+pretrained` tests=`33 passed`；`cli+ncore_input`=`37 passed / 15 failed`，15 项全部为
  当前未配置环境缺 `shortuuid`，是 environment preflight，不是 upstream 质量结论；
- prerequisites=`4/11 passed`：CC 8.6、system/cgroup RAM、source exact、CLI help 通过；Python 3.11、uv、VRAM
  `24,576<30,720 MiB`、free disk `42,327,777,280<100,000,000,000 bytes`、exact weight、licensed NCore input、
  terms record 失败；HF token 也不存在但未单独作为第 12 条门；
- `inference_smoke=not_run_prerequisites_failed`、`inference_command_constructed=false`；audit wall=`.991914 s`，
  torch/GPU/training/dependency install/weight-or-dataset download 全 false，GPU compute process 为空，OOM/kill=`0/0`；
- F0 outcome=`done_local_inference_not_executable_on_current_host`；F1=`conditional_not_unlocked`、未启动。CLI static PLY
  不含 dynamic/sky/ISP/actor registry/depth，且不是 exact StreetGS checkpoint；未来兼容硬件/合法输入/converter
  的窄 static-background pilot 不被永久否定。当前任务切换为 R0 integration，DGGT fallback 不阻塞 R0。

### `WS-V3-R0-INTEGRATION-01` formal 前协议冻结

- protocol=`configs/worldsim_v3/r0_integration_protocol_v1.yaml`，SHA=`4fe20c3197...7575`；runner SHA=
  `d58c4008...c5ce`；前置 closeout commit=`80b4f983b665d7bb3e4d73fb6d9531f1adbbe901`；
- 授权边界为 read/hash/derive reports/document snapshot/chunk-payload verification；training/inference/GPU/install/
  download/checkpoint-or-registry mutation 与 F1/P4/D3/D4/A3 extra 均为 false；
- 冻结 `5 protocols + 51 canonical evidence files + 3 selected production files + 4 offline MP4 = 63` 个输入审计行；
  11 组 terminal status、23/23 decision checks 均 exact；
- P3 package=`159 files / 444,177,055 bytes / 133 static / 24 actor`；manifest 和 158 payload 的 path/bytes/SHA
  全 exact；12 个最终 deliverable 路径与 claim boundary 全冻结；
- selected chain=`A1-C0-off__A2-D2-boundary-priority__A3-R0-off__A4-P2-mixed__A4-P3-exact-chunk`；
  production checkpoint/registry/config SHA=`7be87e8b...7448 / 69c4f38a...8a27 / 115deaf8...5e68`；
- 定向测试 `11 passed`，WorldSim V3 联合回归 `252 passed`。本条未运行 canonical、未生成 terminal，也没有新训练、
  推理、GPU measurement 或质量结论；下一动作只提交冻结协议后运行 canonical read-only integration。

### `WS-V3-R0-INTEGRATION-01` canonical closeout

- canonical=
  `/root/autodl-tmp/runs/worldsim_v3/WS-V3-R0-INTEGRATION-01/20260809T194625Z__r0-integration-s0-r1`；
  source=`64e3d15ca30de44088c2f6fbfb6da048a31a4acf`，terminal=`done`，summary/manifest/terminal SHA=
  `3ffe99ea...15a7 / a9b052a6...1d90 / 207758b9...15c6`；28 files=`1,117,645 bytes`；
- input/decision/deliverable/manifest=`63/63 / 23/23 / 12/12 / 26/26 exact`；11 组 canonical terminal、5 份
  documentation snapshot 与 P3 159-file/444,177,055-byte package 全 exact；
- selected chain=
  `A1-C0-off__A2-D2-boundary-priority__A3-R0-off__A4-P2-mixed__A4-P3-exact-chunk`；五个 final conclusion
  tokens 与 12 条 claim boundary 完整落盘；
- resource/no-launch passed：wall=`1.678173 s`、cgroup current=`30,389,452,800 bytes`、disk free=
  `42,325,843,968 bytes`、OOM/kill=`0/0`；torch/GPU/training/inference/install/download 全未启动；
- `next_action=none_plan_complete`。F1/P4/D3/D4/A3 R2–R4 仍未启动，不冒充 R0 交付或后续缺口。

## 3. V2 冻结注册表

| Task ID | 状态 | 目标 | 当前输入事实 | 解锁条件 |
|---|---|---|---|---|
| `DR-V2-M0-BOOTSTRAP-01` | done | 事实源、分支、镜像与 bootstrap | 正式 run 完成；历史失败实例保留 | README/STATUS/PLAN 一致，bootstrap smoke 通过 |
| `DR-V2-M1-DGGT-REPAIR-01` | done | 修复 pointops2 并做 18-window inference | 1/3-view、common、regional 全部完成 | 18/18 + 216/216 + 完整运行合同 |
| `DR-V2-M2-ACTOR-EVAL-01` | done | nuScenes 真值 actor 评测适配器 | raw 2Hz 轨迹、4,356 exact mappings、6/6 cohort | 三 scene eligible 16/20/6，visual QA 通过 |
| `DR-V2-M3-EDIT-BASELINE-01` | done | DriveStudio/StreetGS 可编辑基线 | 30k checkpoint、registry、27-image smoke 完成 | scene-0230 remove/lateral/3-camera smoke |
| `DR-V2-M4-EDIT-PILOT-01` | done | scene-0230 真实编辑闭环 | 1,764 RGB、9 MP4、1,176 paired rows、16/16 checks | 两种编辑真实执行且证据可审计 |
| `DR-V2-M5-STRESS-3SCENE-01` | pending | 三场景编辑/去遮挡压力测试 | 0230/0242 checkpoint 与 0255 诊断已生成；任务未闭环并冻结 | 历史门禁未满足；V3 不再授权继续 |
| `DR-V2-M6-HYPOTHESIS-01` | pending | 基于真实失败做 novelty gate | 未生成 | V3 路线不再授权 |
| `DR-V2-M7-METHOD-01` | pending | 最小方法与 matched ablation | 未生成 | V3 路线不再授权 |
| `DR-V2-M8-HUMAN-01` | pending | 人工盲审与终局 | 未生成 | V3 路线不再授权 |

## 4. V2 启动前维护记录

2026-08-02 的文档归档和存储清理属于 maintenance，不冒充 `DR-V2-M0-BOOTSTRAP-01`：

- V1 当前态、实验台账、环境与报告已移入命名归档；
- V2 计划已按实际 checkpoint、DriveStudio 缺口和用户镜像偏好校准；
- 可再生中间产物的精确路径、字节数与恢复方式见
  [`archive/2026-08/v2-preflight/CLEANUP_MANIFEST.md`](archive/2026-08/v2-preflight/CLEANUP_MANIFEST.md)；
- AD-GS 六场景最终 60k checkpoint/render/metrics、processed 输入、raw subset、DGGT 完整预下载候选均受保护。

## 5. V1 冻结输入

| 资产 | 终态 | V2 用法 |
|---|---|---|
| AD-GS 六场景 exact reproduction | done，6/6 | 只读 checkpoint/render/metrics；不重复训练 |
| DGGT V1 run | blocked，未 inference | 只作为原始失败证据；V2 新环境重做 |
| V1 pseudo identity audit | 0/12 slots | 失败边界；不得当作真实编辑结果 |
| V1 候选 A novelty | rejected | 禁止复活“补身份 + 基础轨迹编辑”作为贡献 |

## 6. `DR-V2-M0-BOOTSTRAP-01`

### 工程失败实例

- run：
  `/root/autodl-tmp/runs/dynamic_editing_v2/DR-V2-M0-BOOTSTRAP-01/20260802T114342Z__bootstrap-s0`；
- terminal：`blocked / empty_shell_python_not_on_path`；
- 网络四源已可达，但非登录空 shell 中裸 `python` 不在 PATH，导致 `source_resolution.json` 未生成；
- 修复仅显式选择 `/root/miniconda3/bin/python`，没有安装依赖或改写全局环境。

### 正式完成实例

- 验证实例：
  `/root/autodl-tmp/runs/dynamic_editing_v2/DR-V2-M0-BOOTSTRAP-01/20260802T114453Z__bootstrap-s0-r2`
  为 `done`；其后只为让 source snapshot 覆盖相对 `HEAD` 的 staged/unstaged 全部 M0 文件创建 r3，未改变
  bootstrap、资产或测试协议；
- run：
  `/root/autodl-tmp/runs/dynamic_editing_v2/DR-V2-M0-BOOTSTRAP-01/20260802T115419Z__bootstrap-s0-r3`；
- terminal：`done`；
- branch：`research/dynamic-editing-v2`；source commit：`09fbb55` + 本 M0 工作树快照；seed：`0`；
- empty shell/tmux shell：`PASS/PASS`；TUNA Conda/PyPI、HF mirror、GitHub：`4/4 HTTP 200`；
- AD-GS `model_60000`、42 test renders、138 train renders 与 processed 输入：`6/6`；
- DGGT preload：`5,411,266,466` bytes，SHA-256
  `fd15644b3a878849470cbf5f0f9eae39167cfec1b853092898ae754c4f3acde9`；
- DriveStudio commit `e59bda4fa681f829dbb1d65f0de582b0f633c450` 与 env 可用，pilot assets `missing`；
- 测试：
  `python -m pytest -q tests/test_dr_pseudo_tracks.py tests/test_v71_actor_registry.py` → `7 passed`；
- `shellcheck` 未安装，按计划未为此污染环境；`bash -n` 通过。

## 7. `DR-V2-M1-DGGT-REPAIR-01`

### 冻结实现

- repo `a3276d2b`；model revision `735ac9a6`；checkpoint bytes/SHA-256 通过并 hardlink 复用；
- `/root/autodl-tmp/envs/dggt-v2`：Python 3.10 / torch 2.4.1+cu121 / 固定 NVIDIA CUDA 12.1
  compiler+runtime+headers；
- pointops2 upstream `python setup.py install`；CUDA forward/backward=`PASS`；
- compatibility patch 仅 `args.difix -> args.diffusion`，untouched 错误单独保留。

### 正式运行

| 证据 | 终态 | 覆盖/结果 |
|---|---|---|
| `20260802T125138Z__native-nusc-s0-r6` | native done；后续 common import blocked | 1-view 18/18；3-view 18/18；原生输出不受 common 失败影响 |
| `20260802T133151Z__common-retry-s0-r8` | done | AD-GS common target 216/216；GT 像素身份 216/216 |
| `20260802T133912Z__regional-s0-r9` | done | AD-GS 216 + DGGT 1-view 72 + DGGT 3-view 216 = 504 rows |

M1 均值：

| 协议 | PSNR | SSIM | LPIPS(Alex) | inference s |
|---|---:|---:|---:|---:|
| DGGT 1-view | 20.707359 | 0.856031 | 0.135780 | 1.785527 |
| DGGT 3-view | 21.165262 | 0.771051 | 0.165553 | 4.517659 |
| AD-GS same-target 1-view | 34.581860 | 0.951918 | 0.062490 | n/a |
| AD-GS same-target 3-view | 34.894344 | 0.951711 | 0.061447 | n/a |

区域诊断的动态区/边界带 PSNR 分别为：AD-GS `29.640118/29.480968`、DGGT 1-view
`22.999911/22.017347`、DGGT 3-view `22.902139/21.810579`。边界带固定为二值动态区
7x7 dilation XOR erosion。这些数值仅用于 failure characterization，不得解释为同预算排行。

### 失败实例

`r1–r5`分别固定了 pip backtracking、CUDA 11.8/cu121 compiler mismatch、缺 cusparse headers、
transformers 5.x/DTensor 和 diffusers 0.39/torch schema 不兼容；r6 common 固定 `flow_vis`
缺失；r7 固定重试封装字段错误。全部为独立 `blocked` run，没有覆盖原运行。

## 8. `DR-V2-M2-ACTOR-EVAL-01`

### 正式运行

`/root/autodl-tmp/runs/dynamic_editing_v2/DR-V2-M2-ACTOR-EVAL-01/20260802T140312Z__actor-eval-s0-r5`
=`done`。

| scene | raw actors | eligible | high-support | boundary-support |
|---|---:|---:|---|---|
| scene-0230 | 58 | 16 | `af663976db5e...` | `18c7f0c5fa6b...` |
| scene-0242 | 53 | 20 | `40f087d8d9d7...` | `2c820a798ad9...` |
| scene-0255 | 56 | 6 | `f4aa30b8d0b4...` | `80c08b992f1d...` |

- 预注册 support score 和字典序 tie-break 未调节，冻结时尚无 M3/M4 编辑输出；
- 4,356/4,356 observations 使用 timestamp+exact `sample_token`；无效投影不从分母中静默删除；
- raw 2 Hz 与 interpolated visualization 字段物理分离，本运行插值列表为空；
- 11 个输入 metadata 哈希、167 actor metrics、3 份 cohort CSV、6 组投影 panel、6 张 raw
  轨迹图与视觉 QA 齐全。

### 失败实例

- r1：错误假设磁盘 `sample.json` 含 devkit 运行时 `anns` 反向索引；
- r2：`ijson` Decimal 进入 JSON 运行合同；
- r3：近平面后的 invalid projection 没有统一零面积 schema；
- r4：只选最近 timestamp sweep 导致 raw sample token 不精确，protocol QA 失败；
- r5：改为 exact token 内再做 timestamp 最近选择，不改 actor 门槛，通过。

## 9. `DR-V2-M3-EDIT-BASELINE-01`

### 正式训练与恢复

- 原生训练 run `20260802T152252Z__native-train-s0-r8` 完成 100/1,000-step profile 和 30k
  训练，checkpoint=`386,398,646 bytes / step 30000 / SHA-256 8ed40576...a73f9e`；
- 训练后上游 full render 把帧累计在内存中，`577/588` 时 cgroup memory 连续两次超过 90%，
  守卫返回 `-15`；峰值 GPU `23,873 MiB`，峰值 cgroup `89,836,462,080 bytes`，
  `oom=0 / oom_kill=0`；r8 保持 `blocked`；
- r12 对 checkpoint step/bytes/hash、r8 formal stage 和 blocked terminal 做窄范围复核，未复制或修改
  checkpoint；训练语义完成与上游 post-render 未完成分开记录。

### 正式编辑 smoke

完成 run：
`/root/autodl-tmp/runs/dynamic_editing_v2/DR-V2-M3-EDIT-BASELINE-01/20260802T163930Z__formal-checkpoint-recovery-s0-r12`。

- registry 24 个 model：23 non-empty，1 个被原生训练裁剪为空并显式不可用；
- 冻结 actor `af663976... → true id 13 → column 13 → model 5 → 2,683 Gaussians`；
- `3 frames × 3 cameras × 3 variants = 27` PNG；original/remove/lateral 均非空且时间同步；
- lateral/remove mean absolute RGB diff 分别为 `0.0003448362 / 0.0002196175`；两者均在 2 个
  frame-camera 上非零；这只是 effect smoke，不是质量指标；
- checkpoint SHA、非目标参数、reload 后完整 RigidNodes state 均精确不变；
- post peak GPU `8,241 MiB`，peak cgroup `58,291,757,056 bytes`；readiness 11/11 available。

### 独立失败实例

- r4/r6/r7：旧 `gsplat/nvdiffrast` CUDA binary 不含 RTX 3090 SM 8.6；固定源码重建后通过；
- r5：tmux 非登录环境没有裸 `python`；改为前缀解释器；
- r8：训练完成后的累积 full render 触发内存守卫；
- r9：恢复 probe provenance 字段错误嵌套；
- r10：registry helper 在 DriveStudio 环境误依赖未声明 `ijson`；改为读取 16 MB 标准 JSON；
- r11：一个非目标 model 的 Gaussian slice 被训练裁剪为空；registry v2 显式标记 unavailable，
  仍要求所选 actor slice 非空。

## 10. `DR-V2-M4-EDIT-PILOT-01`

### 正式运行

`/root/autodl-tmp/runs/dynamic_editing_v2/DR-V2-M4-EDIT-PILOT-01/20260802T171000Z__scene0230-pilot-s0-r7`
=`done`。

- 固定 scene-0230 high-support actor `af663976...`，196 帧、三相机、original/lateral +1m/delete
  共 `1,764` 张 RGB，所有配套 depth/opacity/dynamic/target mask/footprint 与 9 个 MP4 完整；
- paired metrics=`1,176` rows；16/16 协议、不变量和产物检查通过；
- lateral/delete non-target PSNR=`93.394483/95.598042`，LPIPS(Alex, 256px)=
  `5.260851e-09/3.052960e-09`，source effect energy=`0.055526/0.031926`；
- actor-local 位移最大误差 `3.814697e-06 m`，rotation/size/canonical drift 和 multi-camera
  world mismatch 均为 `0`；
- 自动检查不冒充质量门禁；人工抽检只确认非黑、非重复 original、footprint 和目标差分可见。

### 独立失败实例

- `smoke_frame1_s0_r1`：float32 transform 往返最大误差高于不现实的 `1e-6 m`，其余检查通过；
  r2 将协议容差固定为 `1e-4 m` 后 16/16 通过，正式实测误差为 `3.814697e-06 m`；
- `debug_controller_s0_r5`：外层诊断 `timeout` 中断 controller，但 child 使用
  `start_new_session=True`，故需按精确 PGID 回收；未覆盖；
- `20260802T170600Z__...-r6`：调试 tmux 生命周期中断后保留 running terminal 证据；
  正式 r7 改用 nohup controller，资源守卫和 terminal 均闭环。

## 11. `DR-V2-M5-STRESS-3SCENE-01` 部分执行后冻结

该任务保持 `pending`，因为没有满足 V2 预注册完成门禁。2026-08-05 路线切换后不再继续扩建其大型评测链，
但已生成资产和失败诊断作为 V3 A0/A3 的输入保留。

### 已有证据

- scene-0230 held-out：checkpoint `398,652,534` bytes，SHA-256
  `24a39f27dfeed36bbdb01ee14211aec51b414e6ab0e61915b71c1dddcdf61e49`；
  high/boundary actor 均可用，分别 `4,747/1,914` GS；
- scene-0242 held-out：checkpoint `306,034,934` bytes，SHA-256
  `16179d8f99becb86b6893a18ff036af72d78c9897f7aa2b0e297b735dd6c5fda`；high actor `6,939` GS，
  boundary actor 显式不可用；
- scene-0255：数据准备与 sky 产物已落盘；r8 的 90% cgroup memory stop 与其后的 cache recovery 分开保留；
- r25–r27 把训练失败定位到 DriveStudio 实例点聚合的 CUDA `torch.cat`；r27 输入 166 个 CUDA float32
  tensor，其中 152 个为空 `(0, 3)`，总计 177 scalars，terminal=`done` 表示诊断完成，不表示训练完成；
- 当前无 M5 控制器、tmux 或 GPU 进程。r16/r18 的 `running` terminal 是容器生命周期中断证据，不改写。

### 未完成

- scene-0255 原生完整 checkpoint 与 actor registry；
- 三场景 × 两 actor × 四编辑的 24 条有效序列；
- pseudo-hole、perception 与跨场景 final matrix；
- M5 单独实现提交和 V2 M6 novelty gate。

未提交的 M5 config/scripts/tests 属于保留工作树，V3 P0 文档提交不得 stage 它们。scene-0255 修复在 V3 A0
以新 task、新 run 和最小 compatibility patch 执行，不能倒写旧 M5 terminal。

## 12. 计划终态

`WS-V3-A1-CALIBRATION-01` 已 `done_off`，`WS-V3-A2-ACTOR-DENSIFY-01` 已
`done / tradeoff_non_dominated`，`WS-V3-A3-LOCAL-REFINE-01` 已 `done`：R1 因 frozen GPU resource gate
失败且 resource-invalid diagnostic 为 tradeoff 被 rejected，`A3*=R0/D2 exact alias`。A4-P0 v1 r1 已因
resolution contract blocked，v2 r2 已 13/13 audits passed 并登记 `done`。P5 r1 保留 runtime attribute blocked，
r2 已 14/14 audits passed 并登记 `done`。P1 canonical r1 已 21/21 audits passed；b05/b10/b20 分别失败
3/12/15 个冻结质量门，method rejected、selected=p1-source exact alias，P1 experiment=`done`。P2 r1 因 evidence
ledger 漏项保留 blocked，canonical r2 已 19/19 audits、31/31 safeguards 并选择 `p2-gs-param-fp16`，P2=`done`。
P3 protocol SHA=`dfaaba79...1b41` 已冻结且 validator/12 项协议测试/222 项联合回归通过。下一动作只实现并提交
`P3-chunk` runner；该冻结动作随后由 runner `aba5577` 和 canonical r1 完成：21/21 audits、57 RGB/31 endpoints、
85 tensor paths 与 source/registry immutable 全部 exact，selected=`p3-chunk-package`。package 比 source 大
`2.792171%` 且 load/reassembly 更慢，只登记 exact asset separation；A4=`done`。

F0 canonical audit 已 `done`，inference=`not_run_prerequisites_failed`，F1=`conditional_not_unlocked`。R0 canonical
已 `done`：63 inputs、23 decisions、12 deliverables、26 manifest files 与 P3 package 全 exact；A0→A4 主表、
Pareto、负结果/适用边界、复现 manifest 和最小离线可视化索引均已生成。V3.1 当前为 `none_plan_complete`。
P4、A3 formal/R2–R4、D3/D4 与新训练/推理仍未授权，除非未来以独立任务重新预注册。

## 13. WorldSim V6 selector 研究族最终注册表

2026-08-22 在 `WS-V6-R140-CROSS-FRONTEND-END-TO-END-UTILITY-01` 后完成收口。R141 未执行；selector 研究族冻结，不再继续 threshold 13/45、新 actor、新方向或新 selector 机制。

| Task | 状态 | Canonical evidence |
| --- | --- | --- |
| `WS-V6-R134-ADGS-CROSS-FRONTEND-THRESHOLD13-01` | `rejected` | threshold 13 在 AD-GS frame 13 出现 1 FN；V6-F94。 |
| `WS-V6-R136-ADGS-HELDOUT-POLICY-CONFIRMATION-01` | `rejected` | threshold 1 在唯一 heldout confirmation 出现 1 FP；V6-F95。 |
| `WS-V6-R137-EXACT-INPUT-REUSE-CROSS-FRONTEND-01` | `done` | 157 帧、0 false reuse、调用减少 16.56%、628 hashes exact。 |
| `WS-V6-R138-ADGS-ANTITHETIC-EXACT-INPUT-CONFIRMATION-01` | `rejected` | CLI infrastructure failure consumed；无方法结论；V6-F96。 |
| `WS-V6-R139-ADGS-ORTHOGONAL-EXACT-INPUT-CONFIRMATION-01` | `done` | 39 帧 exact-once、0 false reuse、调用减少 17.95%、156 hashes exact。 |
| `WS-V6-R140-CROSS-FRONTEND-END-TO-END-UTILITY-01` H001/H002 | `rejected` | 小写 Python boolean 导致 formal closeout 失败；V6-F97/V6-F98。 |
| `WS-V6-R140-CROSS-FRONTEND-END-TO-END-UTILITY-01` H003 | `done` | 只做 `false → False` recovery；macro end-to-end reduction 8.78024%、worst 1.66365%、0 reconstruction errors。 |
| `WS-V6-SELECTOR-RESEARCH-FAMILY-CLOSEOUT-01` | `done` | Selector family frozen；active hypothesis none；R141 not executed。 |

R140 H003 canonical run：

```text
run://worldsim_v6/WS-V6-R140-CROSS-FRONTEND-END-TO-END-UTILITY-01/20260822T063937Z__end-to-end-utility-s20260821-r1
```

Scientific certificate SHA256=`913833af47e4171e27707f71418b6625ed358b538d1c8a5a18bca5ac7f585363`；gate SHA256=`ac3c79c0e93f2932a076da8323b89a210ff2cbaac27ffa13079ce89ae9d07b51`。Failure ledger delta 是 V6-F97/V6-F98 的 recovery closeout 注记；没有新增 failure ID。完整证据入口为 `docs/autoresearch/worldsim_v6/SELECTOR_RESEARCH_FAMILY_CLOSEOUT.md`。
V3.1 计划和本文件的 R0 收口快照见
[`archive/2026-08/worldsim-v3.1/`](archive/2026-08/worldsim-v3.1/README.md)；归档内容不构成新的执行入口。
