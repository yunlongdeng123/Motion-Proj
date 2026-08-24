# Research Status

## WorldSim V6.2 CPSC P2 formal 完成，P4 IR-WM sidecar 启动（2026-08-24）

状态：`p2_formal_materialization_passed / p4_irwm_sidecar_active`；active task=
`WS-V62-P4-IRWM-PRIOR-SIDECAR-01`。

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
