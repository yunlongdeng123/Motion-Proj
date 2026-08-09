# Motion-Proj 当前研究风险与防重复账本

> **最后更新**：2026-08-09
> **当前范围**：V3 WorldSim 模型链直接约束，以及 V1–V7.1、N1/cut-in、V2 的完整防重复结论。
> **历史账本**：完整 `RF-01`–`RF-18` 原文见
> [`archive/2026-07/v7-feasibility/RESEARCH_FAILURES_RF01_RF18.md`](archive/2026-07/v7-feasibility/RESEARCH_FAILURES_RF01_RF18.md)。
> **事实源**：[`EXPERIMENTS.md`](EXPERIMENTS.md) 和实际 run 产物。

本文件保留仍约束后续路线的历史结论，并把 H1-11D 的失败严格分为“观察到的事实、合理推断、尚未知、
复开条件”。归档不会使旧失败失效；任何新计划复用旧机制时仍须满足原 RF 的重开条件。

## V3 启动时必须先读的结论（2026-08-05）

- `V3-F01`：M4 的 non-target PSNR 93/95 dB 是硬局部编辑的构造/保持性证据，不是编辑后视觉质量。
- `V3-F02`：DriveStudio 已有 Affine、CamPose 与 LiDAR 初始化；A1 必须做 off/native/enhanced 消融，
  不得把上游能力改名为新增模块。
- `V3-F03`：V2 M5 未完成。0230/0242 checkpoint、Tier A/B/C 和 0255 诊断可复用，但不得把部分资产
  写成三场景压力测试通过。
- `V3-F04`：scene-0255 是小输入 CUDA `torch.cat` 工程阻塞且无 OOM 证据，不能写成 3DGS 方法失败。
- `V3-F05`：三个 scene 只支撑模型消融和工程判断；不得外推 trainval、夜间、长时或复杂交互。
- `V3-F06`：Instant NuRec 等工作已经改变前馈基线边界；DGGT 只作历史范式对照，不做跨分辨率、跨输入、
  跨训练预算的 leaderboard。
- `V3-F07`：persistent identity、actor binding、scene graph 和基础 trajectory edit 已由上游与 V2 覆盖，
  不能作为 V3 模型贡献。
- `V3-F08`：rolling shutter 需要真实 readout direction/time；没有 metadata 时必须 `not_supported`，不得
  从帧时间或相机顺序推测行曝光时间。
- `V3-F09`：actor-aware densification 必须分 D0–D3 小步消融；不得一次加入 boundary、LiDAR、visibility、
  residual 后只报告一个合并结果。
- `V3-F10`：编辑后 local refinement 的 unknown background 仍是 unknown；只允许 Tier-A、多视图或 LiDAR
  支持监督，Tier B/C 不得当伪真值回传。
- `V3-F11`：全图 PSNR/SSIM 不能代替 actor/边界质量；counterfactual mask 也不是真值分割，必须同时报告
  visible-image/pixel coverage，避免目标未渲染时通过缩小分母得到虚高指标。
- `V3-F12`：nuScenes processed camera ID 必须以数据加载器事实源映射；显示标签写错会把非相邻相机当成
  预注册相机对，已有 formal 必须 rejected 后重跑，不能只改图标题。
- `V3-F13`：seed=0 不保证 CUDA visibility filter 后的随机背景初始化逐点/逐计数复现。记录的 LiDAR/actor
  tensor exact 可作门禁，重建初始化 depth 只能作 witness，不能冒充源训练初始化 exact residual。
- `V3-F14`：局部 role、全图画质或 learned correction 稳定性改善不能替代预注册阶段主端点。C2/C3 未通过
  E1/E2 合同就不能为了保留增强模块而成为 C*。
- `V3-F15`：确认场景的原始端点方向可以与开发场景相反。不得把完整 Pareto 合同的 `done_off` 改写成
  “C0 在所有场景、所有指标都最好”，也不得只挑 0255 E1/E2 error 改写 C*。
- `V3-F16`：A2-D2 的边界改善、global/non-target 退化与更高训练成本构成严格 Pareto tradeoff。不得用新增
  事后标量权重把它改写成 D2 dominance；后续采用 D2 必须同时登记 D1 fallback 和完整退化轴。

### V3-F01：局部保持不等于编辑质量

M4 的 lateral/delete non-target PSNR=`93.394483/95.598042`，主要来自编辑器只改变目标 actor 并保留其他
Gaussian。它证明实现没有意外改动非目标区域，但不能证明 source footprint 后方背景正确、actor 边界自然或
连续帧无闪烁。V3 必须把 outside preservation 与 Tier-A hole、depth ordering、boundary 和 temporal 指标分开。

### V3-F02：原生校准和初始化不能重复发明

DriveStudio `e59bda4` 的 `AffineTransform` 已输出 RGB affine，`CameraOptModule` 已学习 3D 平移和 6D 旋转
残差，数据集也已从 LiDAR 初始化背景/实例。A1 的合法动作是关闭/原生/增强的受控消融，以及 support provenance
审计；不能把启用原生 config 写成新成像、位姿或 LiDAR 模块。

### V3-F03/F04：V2 M5 部分证据与 scene-0255 工程阻塞必须分开

V2 M5 没有生成预注册的 24 条有效序列和 final matrix。scene-0230/0242 checkpoint 是有效训练资产；
scene-0255 训练则阻塞于 `datasets/driving_dataset.py` 实例点列表的 CUDA `torch.cat`。r27 观察到 166 个
CUDA float32 tensors、152 个 `(0, 3)` 空 tensor、177 scalars，且 `oom/oom_kill=0`。V3 A0 可以基于此做
最小 compatibility fix，但必须使用新 task/run，不能改写 M5 terminal，也不能由诊断完成推断训练完成。

V3 A0 已用 `436cfc1` 实现配对过滤：点与颜色按同一个 empty-row 条件过滤，全空时返回 prototype view。
canonical smoke `20260805T161656Z__scene0255-catfix-s0-r2` 在原生错误复现后完成真实 dataset init、1-step
优化与 checkpoint，说明该工程阻塞已在 smoke 范围解除。随后新 30k run
`20260805T162355Z__scene0255-native30k-s0-r1` 完成 checkpoint、registry 与 held-out 评估；0230/0242 通过
严格等价合同复用。该兼容问题现已闭环，但只证明工程修复和 A0 基线成立，不证明任何 A1/A2 方法提升。

### V3-F11：全图质量与模型差分 mask 都有明确边界

A0 中 scene-0242 全图 PSNR=`29.107`，高于 0230/0255，但其 high actor 区域 PSNR=`19.788`，反而是三场景
最低。scene-0255 boundary actor 区域 SSIM=`0.526`，也没有被全图 SSIM=`0.743` 反映。后续 A1/A2 不得只用
全图指标判断动态对象提升。

A0 actor mask 来自同一 checkpoint 的 original 与 actor-delete 配对渲染差分，是模型 counterfactual
diagnostic，不是 nuScenes 真值 segmentation。如果模型没有画出 actor，mask 会缩小；因此每个结果必须同时报告
candidate/visible image、effect pixel coverage 和 `ABSTAIN`。tight-crop LPIPS 用固定 8px padding 与 256px 输出，
不能和全图 DriveStudio LPIPS 混为同一指标。

A0 finalizer r1 因复用 checkpoint run 使用 `source_training_resources`、原生 run 使用 `train_resources` 而
`blocked`。这是汇总 schema 兼容失败，不是模型失败；`00ba4e8` 增加显式 provenance 归一化，r2 为唯一完成矩阵。

### V3-F05/F06/F07：结论规模与研究边界

三个固定 scene 足以比较相同数据、预算和实现下的 A0–A4，但不构成数据规模、天气、城市或交互分布覆盖。
Instant NuRec、OmniRe、IDSplat、SplatAD、ADGaussian、Real2Sim、RoVES 等工作分别覆盖前馈分层重建、
实例场景图、传感器和物理方向；V3 的价值来自完整复现、窄模型改动、负结果和工程 Pareto，而不是重新命名
已公开能力。只有 A2/A3 在至少 2/3 场景方向一致且资源稳定，才讨论扩展场景。

### V3-F08/F09/F10：禁止不可归因或无真值捷径

rolling shutter 没有 row timing 就不能实现；actor-aware densification 必须从 actor/background threshold 与
quota 开始，再分别增加 boundary/residual 和 LiDAR/visibility；local refinement 必须冻结 affected set 外参数，
并区分 expected/first-hit/measured depth。不得用 hard-composition outside=0、原图 actor 像素或未知区域的
自洽渲染作为方法成功证据。

### V3-F12：相机标签错误会污染跨相机端点

A1-E0 初版沿用了错误的显示顺序 `0=FRONT_LEFT / 1=FRONT / 2=FRONT_RIGHT`，但 DriveStudio nuScenes
事实源明确为 `0=FRONT / 1=FRONT_LEFT / 2=FRONT_RIGHT`。结果是名义上的相邻相机对可能实际落到
左右两侧非相邻画面，零支持也会被错误解释为模型现象。首次 formal
`20260806T140703Z__scene0230-c0-a1-e0-formal-full-s0-r1` 因此已标记
`rejected / INVALID_CAMERA_ID_LABEL_MAPPING`，原 terminal/manifest/summary 以 `*.original_done.json` 保留；
`d85ef27` 修复后 C0/C1 使用新唯一 run 回填。

防重复门禁：相机 ID/name 映射必须来自训练数据加载器或预处理权威列表，写入 resolved config 并纳入 hash；
QA 必须验证投影落在实际重叠的建筑/路面。若映射错误，所有受影响正式结果必须 rejected，不得通过重命名
已有 JSON、图片或曲线继续使用。

### V3-F13：随机 CUDA 可见性筛选不等于 exact 初始化 replay

A1 最小 LiDAR provenance 的 strict smoke
`20260806T142900Z__scene0230-a1-lidar-provenance-smoke1-s0-r1` 观察到：800,000 个背景 LiDAR 点、全部 24 个
actor point/color tensor、75,002 个 RigidNodes 初始点均 exact match，但随机 near/far 球面候选经过 CUDA
visibility filter 后，背景初始 Gaussian 数从源运行 946,484 变为 replay 的 946,597；后续 replay 又得到
946,309 和 946,291。这不是 LiDAR 输入变化，也没有训练或 checkpoint 修改。

防重复门禁：冻结的 `a1_lidar_provenance_v1.yaml` 要求记录 LiDAR/actor tensor exact match，并记录随机球面
候选、visibility mask SHA 和计数；背景 exact replay 固定为 `report_not_gate`，不允许事后设置“接近即可”的计数
容差。正式初始 depth residual 必须标为
`seed0_reconstructed_initialization_witness_not_exact_source_initialization`。要获得源训练初始化的 exact depth，未来
必须在训练创建时直接持久化 post-filter 初始化 tensors；A2 的逐 Gaussian ancestry 仍需独立 instrumentation。

### V3-F14：局部改善不能替代阶段主端点

scene-0230 中，C2 的 boundary-support E2 mean/P90 从 C0 的 `0.003547/0.006353` 改善为
`0.003346/0.005447`，但 high-support E2 P90 退化到 `0.011734`，actor/boundary LPIPS 也整体退化；因此不能把
单个 role 的改善提升为整个 E2 端点改善。C3 的全图 PSNR/LPIPS、boundary actor 质量和 learned pose correction
稳定性均最好，但 E1 median/P90 与两个 E2 role 仍未严格优于 C0。

A1-S0-v1 在结果已可见后、确认场景前把 V3.1 7.5 操作化为无容差严格 Pareto，并如实披露该时点；没有新增
事后数值阈值。正式结论必须是 `C*=C0-off / done_off`。不得更换 role、放宽端点、只引用 C3 全图画质或把
learned correction 幅值写成 pose GT，以强行保留增强模块。

### V3-F15：完整合同通过不等于每项指标方向一致

scene-0242 的 C0 在 global、E1 和 high E2 上优于 C1；scene-0255 则相反，C1 的 E1 median/P90 和两个 E2
role error 都更低。但 0255 C1 的 high E2 coverage 从 `23.529%` 降至 `21.569%`，boundary/high actor LPIPS
均退化，因而仍未通过冻结的“主端点改善、另一端点不退化、appearance LPIPS 可接受”完整合同。

A1 finalizer 的合法表述是：C1 在两个确认场景均不 eligible，C*=C0 保持 `done_off`，同时原始端点方向具有
scene dependence。禁止写成“C0 普遍校准更优”，也禁止忽略 coverage/appearance 只引用 0255 error 重选 C1。

### V3-F16：边界优先分支选择不等于全面方法提升

A2 formal 中，D2 相对 D1 的 boundary-support boundary-band PSNR/SSIM/LPIPS 从
`25.770024/.821572/.048382` 改善到 `26.171399/.828868/.044568`；但 global 从
`27.770024/.850915/.177704` 退化到 `27.703188/.850333/.178344`，non-target PSNR/SSIM 也下降，训练
wall time 从 `2099.33 s` 增至 `2720.82 s`。fixed 与 matched strict-quality、quality-cost 裁决都为
`tradeoff_non_dominated`，且 matched D2 只是 fixed 30k 的 exact alias，不是独立复现。

A3 采用 D2 是因为 A2 的预注册靶点包含 actor boundary，并且 D2 在该边界带三项指标同时改善；这是完整结果
可见后的工程资产路由，不是新增数值门槛、统计显著性或 D2 对 D1 的支配结论。任何后续报告都必须同时保留
D1 quota-only fallback，披露 D2 的 global/部分 actor/non-target/cost 退化，并禁止只摘录边界带结果宣称 A2
“全面提升”。单场景 scene-0230 也不能支持跨场景泛化结论。

## V2 启动时必须先读的结论（2026-08-02）

- `PIVOT-F03`：AD-GS exact reproduction 已完成；V2 只读最终 checkpoint/render/metrics，不重复训练。
- `PIVOT-F04`：可见性建模不等于未观测背景真值；M5 必须保留 Tier A/B/C。
- `PIVOT-F05/F14`：资源、外部实例与方法失败分开；OOM/重启不能写成模型质量结论。
- `PIVOT-F06/F07/F08`：换机、非登录 shell 与浮动权重必须重新审计；镜像不能改变固定版本。
- `PIVOT-F10/F11/F12/F13`：PNG/JPEG、COLMAP 并发、cgroup 90% 和合法空占位均已有失败证据。
- `PIVOT-F14B`：V1 pointops2 的直接根因是 PEP 517 隔离构建缺少 torch；V2 先按 upstream
  `python setup.py install`，不重复原 `pip install .`。
- `PIVOT-F15`：AD-GS camera-local pseudo ID 与二值 `obj` 不能支持对象级编辑；V2 以 nuScenes
  `instance_token` 只做评测真值，不注入 AD-GS 训练。
- `PIVOT-F16`：持久身份、actor binding 与基础轨迹编辑本身已不新；V2 必须先产生跨三场景真实失败，
  再做新的 novelty gate。

存储清理只使历史环境和中间 checkpoint non-resident，不撤销上述失败，也不允许重新运行已关闭路线。

## N1 kinematics-first 第三次 reject 与第四版约束（2026-07-25）

### N1-F12：地图分支收敛不是车辆横向机动

**观察**

- 第三次人审文件：
  `/root/autodl-tmp/runs/event_first/N1-EVENT-KINEMATIC-01/v71_n1-event-kinematic-01__kinematic-v1__s0__20260725T092427030639Z__8c2247b6/audit/review_working.jsonl`；
- review SHA256：
  `005cd74b874833808435fd2f47387d1d8e446cdea2d3a5cae6146e34bf331e96`；
- 12/12 已审，`TRUE_POSITIVE=0`、`FALSE_POSITIVE=12`、`UNCERTAIN=0`，precision=`0`；
- subject maneuver 为 `INVALID` 12/12；failure code 为
  `SUBJECT_NO_LATERAL_MANEUVER=12`、`ROUTE_CONTINUATION=11`、`NORMAL_TURN=1`、
  `MAP_MATCH_JITTER=1`；
- 第三版 12/12 机器候选都是 `converging_branch_merge`。规则只验证 source/target
  地图分支在几何上汇合，却没有验证车辆中心/车身相对接收车道发生 outside→inside 横移；
- target corridor 人审 12/12 为 `VALID` 并不能挽救 subject maneuver。地图画对了，不等于事件成立。

**根因**

第三版仍把“actor 沿一条会汇入 target 的道路行驶”当成“actor 主动切入 target 车流”。车辆可以保持正常
转向/道路中心线跟随，而道路本身向另一分支收敛；仅比较 source/target approach heading 或地图 token
变化仍会把路形变化误写成车辆运动学。

**防重复**

- 必须直接从原始 2 Hz annotation 计算 subject 相对接收 corridor 的连续横向状态；
- 至少观察目标车道中心外→中心内，并在进入后保持名义 1 s；10 Hz 插值不参与物理门；
- 进入前还必须与接收 corridor 近似同向，避免把大角度路口/主路续接的几何距离收敛当作 cut-in；
- 不再把 `merge` 地图类别、multiple incoming、route token change 或道路弯曲本身当正例。

### N1-F13：接收车必须来自独立目标车流，不能复用 subject 后车

**观察**

- 第三次 review 中 rear 为 `INVALID` 2/12、front 为 `INVALID` 1/12；
- 第三版 corridor 构造会贪心选择与 subject source 最顺的 incoming，再在该 corridor 上找 rear；
- 因而所谓 rear 往往就是 subject 原队列中的后车，而不是被切入目标车流的接收车；
- K3-004 选错 front branch；K3-007、K3-010 选错 rear branch。其余多项虽被人审写成 corridor
  `VALID`，也只说明地图链连续，不证明 receiver 角色语义成立。

**第四版硬约束**

1. parallel lane change 的 target chain 显式排除 source token；
2. merge 只枚举 `target` 的 direct incoming 中不同于 subject source 的分支；
3. RECEIVER 必须在进入前后保持同一 identity、同向、最近后车次序与 `[0.5,40] m` bumper gap；
4. subject/receiver 之间不得遗漏更近同 corridor 车辆；
5. negative control 也必须存在持续 receiver，不能用孤车普通直行冒充交互密度等价 control。

### N1-F14：第三次裁决的研究失败与工程失败必须分开

**研究裁决**

- clean adjudication commit：`1fbbbc1`；
- 成功 run：
  `/root/autodl-tmp/runs/event_first/N1-EVENT-KINEMATIC-AUDIT-01/v71_n1-event-kinematic-audit-01__human-audit-reject-v1__s0__20260725T155754010881Z__4c51f0d9`；
- 唯一终态 `REJECTED`，`n2_authorized=false`。

**保留的工程失败**

第一次 formal adjudication 使用了错误的 audit-manifest 指纹键，在写入研究产物前失败：
`.../v71_n1-event-kinematic-audit-01__human-audit-reject-v1__s0__20260725T155523736677Z__4c51f0d9/`。
该目录保留 `FAILED/failure.json`，原因是 `engineering_manifest_key_mismatch`。修复将
`artifact_set_sha256` 更正为实际 schema 的 `immutable_artifact_set_sha256`，并把所有输入校验提前到
run 目录创建之前。不得删除失败尝试或把它统计成 research reject。

### N1-F15：四图全常驻与重型 map API 会触发 2 GiB cgroup 峰值

**观察**

- 第四版首个 development smoke 在算法开始前以 `RC=137` 被杀；
- 容器 `memory.max=2147483648`，当时常驻服务已占约 `1.85 GiB`；
- 官方 `nuscenes.map_expansion.map_api` 的导入会连带 OpenCV、Matplotlib、Shapely 和渲染 API；
  单是 import probe RSS 就从约 `58 MiB` 增至约 `212 MiB`；
- 同时常驻四张 `NuScenesMap`、完整 sample/instance JSON 行和 128-scene dense batch 会进一步放大峰值。

**工程修复**

- 新增只读取 `lane`、`lane_connector`、`arcline_path_3`、`connectivity` 的轻量 map reader；
- arcline 离散化与官方 devkit reference 在单测中逐点一致；
- map index 改为一次只缓存一个 location，calibration/evaluation 按 location 排序；
- `sample.json`、`instance.json` 改为 ijson 流式最小字段投影，scene builder 复用同一 metadata source；
- scene batch 冻结为 32；不得通过杀死用户编辑器进程、修改容器上限或跳过地图证据来“解决”。

### N1-F16：负对照配置契约缺项导致首个正式 K4 工程失败

**失败事实**

- 失败 run：
  `/root/autodl-tmp/runs/event_first/N1-EVENT-CUTIN-01/v71_n1-event-cutin-01__receiver-cutin-v1__s0__20260725T170948229629Z__46186120`；
- 失败代码提交：`f5c9bbe4c819abce42e1cca0b8800e16a77af680`；
- 正式日志：
  `/root/autodl-tmp/runs/event_first/n1k4_formal_f5c9bbe.log`，SHA256
  `b9ac6d3cce2e731f16aad7bc6a068eaf09c54439ad654bb0a3a9d0c58f63a487`；
- calibration 已运行，进入首个 formal evaluation batch 后，在构造 same-actor 30-frame
  lane-keeping negative control 时抛出 `KeyError: min_median_speed_mps`；
- `lane_keeping_features` 实际接收 `kinematics_control`，但该字段只写在 `cutin` 下；同一调用下一步还会需要
  `max_acceleration_mps2`，原 YAML 也遗漏；
- 失败发生在 `event_pool.json`、`summary.json` 与任何研究裁决写入前。因此它是工程失败，不是机器 gate
  reject，更不是第四次人工评测结论；`n2_authorized=false`。

**保留与修复**

- 旧目录不删除、不改造成成功 run；写入结构化 `failure.json` 与 `FAILED`，原 `RUNNING` 被保留为
  `RUNNING.invalidated`；
- 修复提交 `8581d4dcd1bf9a4f92b426c601e1149c804afc5a` 同时补入
  `kinematics_control.min_median_speed_mps=0.5` 和 `max_acceleration_mps2=12.0`；
- 新增启动前 `_validate_config_contract`，在加载 nuScenes metadata 前检查 delayed runtime dependency、
  `receiver_cutin` 审核 schema 与 `never_start_n2_from_this_run=true`；
- 新增 post-run-directory 异常处理：后续未捕获异常自动写 `FAILED/failure.json`、清除活动
  `RUNNING`，并强制 `n2_authorized=false`；
- 27 项相关测试通过后才从新 run ID 完整重跑。

**防重复**

1. development pilot 必须覆盖至少一个 positive actor 的 negative-control 搜索；“positive=1、
   negative=0”不能被误读为该分支已执行；
2. 所有按候选稀疏触发的配置依赖必须在启动时校验，不能等全量运行数分钟后才由 `KeyError` 暴露；
3. 任何残留 `RUNNING` 的异常目录必须先结构化归档，再开始新 run；禁止覆盖、续跑或统计为 research
   reject；
4. 修复配置/异常落盘不授权改变冻结 K4 阈值、评估 scene 或候选排序。

### N1-F17：重复扫描 583 MB 标注文件产生 cgroup 页缓存压力与外部 SIGKILL

**失败事实**

- 失败 run：
  `/root/autodl-tmp/runs/event_first/N1-EVENT-CUTIN-01/v71_n1-event-cutin-01__receiver-cutin-v1__s0__20260725T171746938858Z__5b1634e3`；
- 失败代码提交：`8581d4dcd1bf9a4f92b426c601e1149c804afc5a`；
- 正式日志：
  `/root/autodl-tmp/runs/event_first/n1k4_formal_8581d4d.log`，SHA256
  `7a89e5f5ab88c53a6d9531dedc56a9db302cef8dab144ade7da75a91f3c09191`；
- calibration 与前 96/685 个 evaluation scenes 已执行，随后 shell 报 `Killed`；没有
  `event_pool.json`、`summary.json` 或研究裁决；
- 本层 cgroup `memory.max=2147483648`，事件计数仍为 `oom=0`、`oom_kill=0`，因此不能把信号来源伪写成
  kernel OOM；终态登记为 `external_sigkill_under_cgroup_memory_pressure`；
- `sample_annotation.json` 大小为 `583417244` bytes。失败后进程已消失时，
  `memory.current=1704148992`、file cache=`639545344` bytes；
- 对该只读标注文件执行 `POSIX_FADV_DONTNEED` 后，在没有停止编辑器/Jupyter/TensorBoard 等用户服务的前提下，
  `memory.current` 立即降为 `1169817600`、file cache 降为 `102739968` bytes。

**根因边界**

证据支持以下工程推断：每个 32-scene batch 都顺序扫描 583 MB 标注表，读页长期计入 2 GiB cgroup；
进程 RSS、既有服务和文件页缓存共同逼近硬上限，外部管理层随后发送 SIGKILL。由于无内核日志权限且
`oom_kill=0`，不能声称已证明具体 killer；但页缓存释放的前后差值直接证明了主要可控压力源。

**修复与复验**

- 修复提交 `f13eb0f1e39b608de1c5e698cd678c2dfd8365a4`；
- 所有大型顺序输入在读取前标记 `POSIX_FADV_SEQUENTIAL`，读取后标记
  `POSIX_FADV_DONTNEED`；
- per-scene JSON 改为 `json.load(file_handle)`，不再由 `read_text` 同时常驻字符串和解析对象；
- 每批显式删除 dense scene payload、执行 `gc.collect` 与 glibc `malloc_trim`；
- 每批日志新增 process RSS 与 cgroup current；正式启动若缺 POSIX page-cache control 则 fail closed；
- 30 项相关测试通过。新正式 run 在 96 scenes 的同一死亡点记录
  RSS `602673152`、cgroup current `1707110400` bytes，且继续运行，证明修复覆盖了原路径；
- 成功 run 最终完成 685/685 scenes；最后一批 RSS `510734336`、cgroup current
  `1612763136` bytes，`oom=0`、`oom_kill=0`。它以独立 run ID
  `...T173015103731Z__5b1634e3` 和唯一 `AWAITING_HUMAN_REVIEW` 结束。

**防重复**

1. Python RSS 不是 2 GiB 容器的完整内存分母；必须同时记录 anon、file cache、cgroup current 与
   `memory.events`；
2. 流式解析只限制 Python 对象，不自动释放内核页缓存；反复全表扫描必须有 cache-pressure 策略；
3. 不得以杀死用户服务、跳过正式场景、降低地图分辨率或减少校准标签来换取“成功”；
4. SIGKILL 无法触发 Python exception handler，因此监控器必须把残留 `RUNNING` 另行结构化封存；
5. 该修复只改变 I/O/内存生命周期，不改变 K4 候选、阈值、排序、scene split 或人工门槛。

### 第四版 calibration 冻结结果与禁止矩阵

第四版只用第二、三次全部 49 条人工标签调试阈值，所有 26 个已审 scene 从 formal evaluation 排除。
截至冻结前 development replay：

- 第三次 FP 拒绝 `12/12`；
- 第二次 FP 拒绝 `35/35`；
- 第二次 TP 保留 `1/2`；
- 被保留真例同时满足目标车道中心外→中心内、进入后稳定、进入前近似同向和 RECEIVER 前后身份连续；
- 另一个旧 TP 因没有独立 RECEIVER 的 pre identity support 被拒绝，不用旧 overall 标签覆盖新事件定义。

| 快捷做法 | 为什么无效 | 第四版合法替代 |
|---|---|---|
| 降低分支/多 incoming 门槛 | 仍把地图属性当车辆行为 | 原始 2 Hz center/box outside→inside |
| 复用 subject source-stream rear | 重演 K3 rear 污染 | 独立 direct incoming / target lane RECEIVER |
| 只要求进入后有 rear | 无法证明被切入车流在事件前已存在 | 同一 RECEIVER pre/post identity |
| 因 0.999 s 拒绝名义三帧 1 s | nuScenes 时间戳有毫秒抖动 | 冻结 20 ms timestamp tolerance，仍需 3 个 2 Hz 帧 |
| 在正式 train 结果上再调阈值/scene | evaluation 泄漏 | 阈值只由 49 条旧审标签冻结 |
| 缩短 30-frame negative 或允许 overlap | 改写 matched-control 问题 | physical event window + 0.5 s guard，control 仍 30 frames |
| 自动启动 N2 | 三次 reject 后边界更严格 | 第四次用户裁决 + 新授权前 `n2_authorized=false` |

## N1 kinematics-first 第三版（2026-07-25，已人工 REJECTED）

### N1-F09：候选真实性与 matched-control 支持是两个独立门槛

**观察**

- clean commit `aa162ef4dea808ad28ca7e56f1273f106e9c0e49` 上的 official train 694-scene
  formal run 完成 8,631 transitions → 1,879 topology-pass → 244 physical-motion-pass →
  12 interaction candidates；
- 12 candidates 覆盖 9 scenes，达到 candidate `≥12` 与 scene `≥6`；
- same-actor lane-keeping negative 只有 2，same-actor pair 只有 2，均低于冻结阈值 4；
- 因此 parent `machine_gate_passed=false`；parent 的唯一 terminal 保持
  `AWAITING_HUMAN_REVIEW`，后续独立 adjudication 已按 12/12 FP 写成 `REJECTED`；
- `AWAITING_HUMAN_REVIEW` 只是当时的审计就绪状态，从未表示 machine pass 或 N1/N2 授权。

**Pair 失败的冻结诊断**

对 12 个 positive actor 重放原 30-frame negative 搜索，不改 event pool：

| 主阻塞 | actor 数 | 观察 |
|---|---:|---|
| paired | 2 | 仅 `scene-0870` 两个 actor |
| 无 30-frame stable run | 1 | actor 轨迹支持太短 |
| 所有窗口与 positive overlap | 5 | 4–27 个候选窗口全部重叠 |
| non-overlap lane-keeping 存在，但 interaction 全失败 | 4 | 共 25 个 lane-keeping PASS windows，全部缺 center front/rear |

其中 6/10 未配对 actor 没有可用的非重叠长控制窗口；另外 4/10 没有等价的双侧 interaction control。
这不是把 gap 或速度阈值稍微放宽就能解决的问题。

**禁止快捷修补**

- 不把 30-frame 缩短到刚好得到 4 pairs；
- 不允许 negative 与 positive event overlap；
- 不用不同 actor 冒充 same-actor control，也不只挑有 pair 的两个 actor 报告；
- 不把普通 lane-keeping 但缺 front/rear 的窗口当成与正例等价的 interaction negative；
- 不因人工可能判真而把 `machine_research_support` 改成 true。

**可能突破**

第三次人工已表明 12/12 merge 候选均不真实，因此先修 subject/receiver 语义，再谈 control 扩展。
若第四版人审真实性通过但 same-actor control 仍不足，可新预注册“更长日志中的同 actor control”或
“matched-other-actor control”；后者必须显式匹配 scene、类别、速度、道路与交互密度。二者都不能回写
第三版 run。

### N1-F10：第三版最终候选只覆盖 converging-branch merge

**观察**

- 244 个 physical-motion-pass 包含 181 merge、63 parallel lane change；
- interaction 层有 215 个在中心关键帧缺 front/rear、17 个 temporal identity/bumper-gap 失败；
- 最终 12/12 candidates 全是 `converging_branch_merge`，parallel lane-change 为 0。

**能下的结论**

第三次人审只能估计这 12 个 converging-branch merge 的真实性。即使全部为真，也不能声称第三版已经覆盖
一般 lane change/cut-in；同时也不能断言 63 个 physical lane-change 都是假事件，因为它们是在更严格的
双侧 interaction 层失败。

**复开条件**

先把 `subject maneuver authenticity` 与 `front+rear gap-insertion interaction` 拆成两个预注册层。
可对 63 个 physical lane-change 建独立 diagnostic audit，但不得事后补进当前 12 条、降低当前 machine gate
或把 subject-only event 当 interaction positive。

### N1-F11：完整审核包不等于每项都有前视相机可见证据

**观察**

- 正式包包含 12/12 panels、evidence、topdown、checklist、prompt 和逐文件 SHA256；
- 本机 full train 数据完整覆盖 CAM_FRONT，但其他五个相机目录只有 mini 规模，不能为这些 formal scenes
  提供稳定六相机视图；
- 首尾面板 QA 正常；部分 subject/front/rear 不在 CAM_FRONT 视野，但 2 Hz annotation topdown 仍存在；
- 40 个 immutable audit files 复算 0 hash mismatch；空白 review validator 按预期 fail closed。

**审核边界**

不在 CAM_FRONT 中的角色不能被猜测为 VALID。评审先使用 topdown、vector centerline 和跨时刻 identity；
若相机与 annotation 冲突或证据仍不足，必须判 `UNCERTAIN` 并记录
`INSUFFICIENT_VISUAL_EVIDENCE`。补六相机或 raw LiDAR 需要独立资产/用途授权，不能偷偷进入本轮或 N2。

### N1 第三版历史禁止重试矩阵

| 快捷做法 | 为什么无效 | 合法后续 |
|---|---|---|
| 把 parent `AWAITING_HUMAN_REVIEW` 写成 pass | negative/pair 失败且人工 12/12 FP | 独立 adjudication 已 `REJECTED` |
| 用人工结果覆盖 pair gate | authenticity 与 comparison support 是不同问题 | 两类 gate 均保留 |
| 缩短/重叠 negative window | 事后改变 matched-control 定义 | 新任务、新 split、新预注册 |
| 用其他 actor 补足 same-actor pair | 混入 actor/scene confound | 预注册 matched-other-actor 设计 |
| 把 63 个 physical lane changes 加入 positive | 它们没有通过冻结 interaction | 单独 subject-only diagnostic |
| 用单侧 front 或 rear 算 interaction | 改变“插入双侧 gap”的研究对象 | 另立事件 subtype |
| 没有相机框也猜 TRUE/FALSE | 把证据缺失转成标签 | `UNCERTAIN` |
| 审核后自动启动 N2 | 本 run 明确 `n2_authorized=false` | 新授权 + 新 gate |

## N1 full-domain 第二次 reject（2026-07-25）

### N1-F05：把 target 多 incoming 的地图类别误当成 subject 行为

**观察**

- 父机器 run `N1-EVENT-FULL-01` 在 val 146 上报 37 个 positive，其中 topology 为 35 merge + 2 lane change；
- 完成人审文件 SHA256 为
  `ae71b31e02faf1d783c36748e629e85acf32a132f35ce2f98102a5f62201dd05`；
- 用户确认的逐项结果为 `TRUE_POSITIVE=2`、`FALSE_POSITIVE=35`、`UNCERTAIN=0`，机器候选精度
  `2/37=0.054054`；
- 多数 reviewer notes 明确指出：subject 沿与 target 共线的主路 lane/connector 正常直行，真正汇入
  target 的是另一条 incoming branch；旧规则却只因 `target_incoming_count>=2` 就把 subject 标为 merge；
- 独立 audit adjudication：
  `/root/autodl-tmp/runs/event_first/N1-EVENT-FULL-AUDIT-01/v71_n1-event-full-audit-01__human-audit-reject-v1__s0__20260725T083929632491Z__6507cbac/`，
  唯一终态 `REJECTED`。

**能下的结论**

graph-corridor 修复了邻车跨 token fragmentation，却没有证明 subject 本身执行了 lateral maneuver。
“target 有多个 incoming”是地图节点属性，不是 actor-specific merge 证据；第二次 N1 不能进入 N2。

**不能下的结论**

不能据此断言 full nuScenes 没有真实 lane change/merge，也不能把 2 个标注 TP 当成已验证的完整事件池。
旧 audit panel 没有把 subject/front/rear 的 3D identity 投影到图像，且 reviewer 字段包含多个来源；
用户已整体确认 reject，但单条 TP 仍只能作为第三版 calibration 标签，不得直接进入 formal evaluation。

### N1-F06：10 Hz 插值 cadence 被错误提升为物理证据

**观察**

nuScenes `sample` 是 2 Hz 标注关键帧。第二版把 2 Hz box 线性/SLERP 插值到 10 Hz 后，用连续 lane-token run
寻找 transition；该做法对齐了 DriveStudio cadence，却没有产生新的物理观测。第二次人审 notes 多次指出
短 token 切换、轨迹插值或 map assignment 假象。

**防重复**

- 第三版速度、加速度、yaw-rate、lane preference、front/rear persistence 只能用原始 2 Hz keyframe；
- 10 Hz 只用于 frame 对齐、可视化和复现旧 transition 候选，不得计算导数或宣称 0.1 s 观测；
- 至少 3 个 pre 和 3 个 post keyframes；不足时为 `UNKNOWN`，不得靠插值补齐。

### N1-F07：单时刻中心距不是持续物理交互

**观察**

第二版在单一 relation frame 上用中心线 `s` 与中心距 `[2,60] m` 选择 front/rear，没有扣除 box extent，
也不要求同一 front/rear identity 跨时刻持续。37 个 machine positive 中 36 个至少依赖一个跨-token 邻车，
因此 branch 选择错误会直接翻转结果。

**第三版解除条件**

1. target corridor 每个 graph edge 同时满足方向连续和 endpoint 连续，并只选单一最连续分支；
2. 使用 oriented box 在 lane tangent 上的投影半长，报告 bumper gap 与 center gap；
3. 至少 2/3 个连续 2 Hz keyframe 保持同一 front/rear identity、方向和次序；
4. 同时报告 longitudinal speed、closing speed、headway/TTC；它们是诊断，不得替代人审。

### N1-F08：第二版审核合同与 provenance 不足

**观察**

- 父机器 run 诚实记录 `code_dirty=true`；它可定位但不是 clean-commit formal baseline；
- 旧人审清单给了逐项 verdict 定义，却未预注册聚合阈值；
- 因此第二次 reject adjudication 没有查看结果后补造阈值，只登记用户明确决定；
- 旧 CAM_FRONT 清单没有身份 box overlay，容易把画面中“真正并道的另一辆车”认成 subject。

**复开条件**

第三版必须在 clean commit 上运行；正式 audit pack 同时提供盲序、subject/front/rear 颜色框、2 Hz 俯视轨迹、
逐项 component verdict、failure codes、完整提示词、immutable file hashes、预注册统计阈值和独立
adjudication 命令。Agent 不得填写人工 verdict。

### N1 full-domain 禁止重试矩阵

| 快捷做法 | 为什么无效 | 第三版允许替代 |
|---|---|---|
| 继续调 `graph_hops` 或 gap | 35/37 误报的主体事件本身不成立 | actor-specific 2 Hz kinematics 先行 |
| target 有多个 incoming 就叫 merge | 把地图节点类别当行为 | 比较 source 与主路 incoming 的 approach geometry |
| 用已审 val 37 条挑最终阈值并在同一 split 报结果 | calibration/evaluation 泄漏 | val 只 calibration；official train formal evaluation |
| 从 10 Hz 插值计算速度/横移 | 人造高频证据 | 原始 sample timestamp + 2 Hz boxes |
| 单帧 front/rear 中心距 | branch/identity 易跳变，忽略车长 | branch-safe corridor + temporal identity + bumper gap |
| 把 2 个旧 TP 直接当第三版正例 | panel identity 仍有未决风险 | 只作 calibration；第三版候选重新盲审 |
| 机器候选一出现就启动 N2 | 人工真实性与样本支持尚未通过 | `AWAITING_HUMAN_REVIEW`，`n2_authorized=false` |

## N1 mini event-pool reject（2026-07-24）

### N1-F01：interaction-support failure

**观察**

- N0 map-expansion、scene→map 与 pose contract 已通过，不再是资产缺失；
- 45 个 source-only eligible actors 产生 71 个 stable token transitions；
- topology taxonomy：39 route continuations、19 merges、3 lane changes、10 unresolved；
- 19 merges + 3 lane changes 共 22 个 topology-pass candidates；
- 22/22 的 exact-target-token front/rear relation 为 FAIL；
- 18 个没有 target-token 邻车，4 个只有 front、没有 rear；0 个同时满足 2–60 m front/rear；
- positive=0、negative pairing=0、same-actor pair=0、positive scenes=0，唯一终态 `REJECTED`。

**能下的结论**

冻结 mini split 不支持可比较 interaction event pool，N2–N5 不触发。地图缺失不是旧 H1 的唯一根因；
补地图后 mini interaction support 仍为零。

**不能下的结论**

不能写成“人类绝对看不到任何交互”或“full nuScenes 也没有事件”。exact target token 可能把同一
longitudinal corridor 上的 actor 分到相邻 lane/connector token；该表示风险尚未独立校准。

**复开条件**

mini run 不复开。新的路线必须：

1. 使用不同 run/task ID；
2. 以 22 topology-pass mini cases 仅作 calibration/audit，不作 formal evaluation；
3. 在 graph corridor 上定义 route-aligned curvilinear front/rear，而非后验放宽欧氏半径；
4. calibration 与 evaluation scenes 分离；
5. 优先在 full nuScenes trainval annotations/metadata 上冻结并评估。

### N1-F02：exact-token corridor fragmentation

**观察**

71 transitions 中 39 个只是 directed route continuation，说明官方 lane graph 将连续道路划分为多个
lane/lane_connector token。当前 interaction 只接受 relation frame 上与 subject 完全相同的 target token。

**推断**

该规则高精度但可能低 recall，尤其在 lane→connector→lane 或短 lane segment 附近。它是 0 interaction
PASS 的一个可能贡献因素，但不是已证实的唯一原因；mini 本身也可能确实缺少前后车。

**禁止快捷修补**

- 不把“相邻 token”全部并入；
- 不把只有 front 或只有 rear 改成 positive；
- 不把 82–89 m front 后验纳入 60 m；
- 不在同一 22 cases 上调 graph hops、gap 或 heading 直到出现 positive。

允许的修复是先定义有向 corridor、route-aligned `s` 和 branch disambiguation，再由独立 calibration
审计冻结；formal evaluation 必须 scene-disjoint。

### N1-F03：mini scale 与静止对象密度

**观察**

- 003/005/004 eligible actors 为 7/22/16；
- 因首尾位移不足 5 m 被拒的 actor 为 107/17/5；
- eligible pose map-match coverage 为 88.89% / 95.60% / 93.36%；
- 官方 full nuScenes 有 1,000 个约 20 秒 scenes，850 个为 train/val，而当前 formal pool 只有 3 scenes。

**结论**

mini 三场景对多 scene interaction event pool 的统计支持不足。下一步应扩数据底座，不应换 actor 或删场景。
优先同域 `v1.0-trainval` annotations/metadata，只有其 event gate 仍失败才评估 nuPlan/Waymo。

### N1-F04：negative=0 的语义

N1 只为已经有 positive 的 actor 构造 same-actor comparable negative。因此 `negative=0` 是
`positive actor set=∅` 的结构结果，不证明没有稳定非事件窗口。后续报告必须同时给出 positive actor 分母，
不得把 negative=0 解释为数据中全是事件或完全无普通驾驶。

### N1 禁止重试矩阵

| 快捷做法 | 为什么无效 | 允许替代 |
|---|---|---|
| 删除 rear requirement | 改变冻结 interaction claim | corridor calibration + scene-disjoint evaluation |
| 扩大 60 m 到覆盖 82–89 m | 看结果后调阈值 | 在新 calibration pool 依据任务时间窗冻结 |
| exact token 改成任意相邻 token | 可能跨 branch/对向车道误配 | directed corridor + route-aligned `s` |
| 从 22 cases 挑“看起来像”的 positive | 人工/后验标签泄漏 | 完整盲审协议；calibration 不进入 eval |
| 在 005 单 scene 继续 | 删除失败 scene、失去多 scene gate | full trainval scene-disjoint split |
| 直接启动 N2/N3/render | 没有 comparable event | 新 N1 先通过 |

## 0. H1 reject 执行摘要

### 0.1 为什么 reject

| ID | 层级 | 观察到的事实 | 能下的结论 | 不能下的结论 |
|---|---|---|---|---|
| `H1-F01` | 事件存在性 | 30 proposals：0 positive、25 negative、5 source-positive/non-event、0 same-actor pair | 冻结 proposal bank 不支持 H3 或配对因果比较 | “occupancy 一定无效”或“换几个 actor 就会成功” |
| `H1-F02` | certificate 精度 | D1 TP=15、FP=5、precision=0.75 < 0.80 | H1-CERT 按预注册 reject | 仅因 recall=0.8824 就称 certificate 通过 |
| `H1-F03` | certificate 覆盖 | D1 UNKNOWN=10/30、PASS=0、PASS coverage=0 | 当前证据无法给出足够确定的正判定 | 把 UNKNOWN 排除或并入 PASS 后重算 |
| `H1-F04` | repair 吞吐 | D2 reject=30/30、export=0、usable yield=0 | H1-PROJ 按预注册 reject；外部 rate 不可定义 | “导出集 0/0 违规，所以修复完美” |
| `H1-F05` | 数据效用 | 无 positive pair，H1 已拒绝 | H3 不触发 | 以 RGB 差分、accept rate 或 proxy 代替下游任务 |
| `H1-F06` | 高成本阶段 | H1 前置 gate 失败 | H2/render audit/blind pack 不实例化是正确停止 | “没跑 H2，所以 H1 结论不完整” |
| `H1-F07` | 统计实现 | 首版 aggregate 把 rejection 计成零违例 | 聚合 bug 已修复且不影响方法输出 | 用首版 aggregate 支持方法 claim |
| `H1-F08` | 资产/证据 | 本机 map 只有 raster PNG；base UNKNOWN 约 96–98% | lane/road support 与独立覆盖存在硬缺口 | 从 raster 或 learned occupancy 静默补成真值 |

### 0.2 冻结证据

- 正式 run：
  `/root/autodl-tmp/runs/occgs_resim/v71/V7-H1-11D/v71_v7-h1-11d__pilot-3-matched__s0__20260723T155755269940Z__cf8d5ebc/`；
- proposal-bank SHA256：
  `f8986915f8d2be0cddddfa6be86f4d2d1ece456c12bf9a962cafec78fd058cd7`；
- config SHA256：
  `cf8d5ebc1429e076fc5142aa6a759a18f54b7f3f937c8423d51505a094bc9fe3`；
- C/D1 realized trajectory 30/30 identity；
- C external hard violation 17/30：003=5/10、005=7/10、004=5/10；
- D1：15 TP、5 FP、2 FN（含 abstention）、20 FAIL、10 UNKNOWN、0 PASS；
- D2：0 accept/export、0 usable yield；
- 唯一 terminal marker：`REJECTED`。

### 0.3 逐失败点根因、卡点与复开要求

#### `H1-F01`：proposal-support failure

**观察**

- source-only eligibility 固定为 3 scenes × 2 actors；
- 每 actor 固定 P1–P5，共 30 个 proposal；
- scenario-effect 没有产生任何 `0→1` positive；
- 5 个 source-positive case 在 proposal 后成为 non-event，25 个为 negative；
- 没有 same-actor positive/negative pair。
- 后续只读 continuity 审计发现，冻结 actor 003:38、003:35、005:23 在完整连续 track 内的 world
  displacement 仅 0.88 / 0.29 / 0.76 m；原 source-only 排序偏好长、清晰 track，但没有事件相关性。

**推断**

固定横向位移只满足几何“移动过”，没有以 lane topology、corridor crossing、target-lane front/rear gap、
duration 或 interaction 定义事件。该设计与“cut-in/merge 正例”的目标错位。这是由结果和 schema 支持的
最强解释，但尚未通过 vector map 重标注，所以不能断言每个 case 的唯一失败原因。

**卡点**

- 本机缺 nuScenes map-expansion vector JSON；
- mini 三场景的真实事件上限未知；
- 没有先冻结 natural-event pool，就不能知道是 proposal family 失败还是场景本身无事件。

**复开条件**

不是改 P1–P5。必须新建 event-first 路线：先冻结 map/track 事件定义和 actor pool，证明存在预定数量的
positive/negative 与 same-actor pair，然后才允许生成候选。若 mini 事件池不足，应 reject mini pool 或
请求新数据授权。

#### `H1-F02`：certificate precision failure

**观察**

5 个 FP 全来自 scene 004 actor 8；certificate 报告 5 个 static-overlap voxels，而独立 raw LiDAR
检查为 0 points。D1 precision 为 0.75，低于冻结门槛 0.80。

**推断**

结果与 coarse voxel quantization、box-to-voxel 接触或证据层不一致相符；尚不能证明是哪一个机制，也不能
从“0 raw points”推出空间一定安全，因为 LiDAR 可能受遮挡和采样稀疏影响。

**卡点**

- `0.4m` 离散 grid 将连续几何压成二值接触；
- 单一 voxel overlap 缺少距离、置信度和观测支持；
- static/dynamic 分层仍可能受历史 sweep 和运动补偿影响；
- raw point absence 不是 free-space ground truth。

**复开条件**

在 scene-disjoint calibration pool 上比较 coarse voxel 与 motion-compensated raw sweeps 的连续
point-to-OBB/swept-volume distance；逐类报告量化、动态残影、遮挡、地图边界和标注误差。门槛必须在
冻结评估前预注册，不能用 actor 004:8 调到通过。

#### `H1-F03`：coverage/abstention failure

**观察**

三场景 base unknown 约为 97.10% / 96.04% / 97.57%；D1 10/30 UNKNOWN，PASS coverage 为 0。
两个 FN 位于 005 的 P3/P5，D1 known fraction 为 0，而 raw LiDAR 只有 3/2 points。

**推断**

当前 single/coarse observation 无法支持大部分 free/occupied 判定。两个 FN 说明“极少 raw points”
也不能自动解决判定；具体是遮挡、采样、时序或标注问题仍未知。

**卡点**

- raw LiDAR 稀疏；
- 缺 vector drivable/lane polygons；
- 多 sweep 若不做动态/ego motion compensation 会制造 ghost；
- learned completion 会提高表面 coverage，却失去独立真值身份。

**复开条件**

增加独立 evidence，而不是调低 known-fraction：官方 vector map、ego/dynamic compensated sweeps、
显式 truth tier 与 uncertainty。继续报告 PASS/FAIL coverage 和 abstention；任何 learned occupancy
只能是附加证据层，不能作为外部 evaluator。

#### `H1-F04`：repair all-reject failure

**观察**

D2 没有接受或导出任何 proposal；usable yield=0，外部 violation rate 无分母。

**推断**

当前 projection/repair 约束组合没有可用工作区，或者 proposal 全都离可行域过远。因为 0 export，无法
区分“repair 算法差”与“输入候选全不可修复”各自贡献。

**卡点**

- 没有成功样本用于 paired outcome；
- 先验 proposal 不由 lane-reachable set 生成；
- 二值 certificate 既可能过严又可能不准；
- H2/H3 都依赖 D2 产出，故被同时锁死。

**复开条件**

先通过 N1 证明事件存在，再以 lane graph/target state 生成 reachable proposal；冻结 minimum usable yield、
comparable export 数和外部 evaluator。若仍为 all-reject，直接 reject proposal/repair family。

#### `H1-F05`：metric aggregation bug

**观察**

首版 summary 把 rejection 计为零违例，使 0 export 看起来像 0% external violation。唯一允许的
`metric_aggregation_bug` 修复保留了旧 aggregate；修复后无 export 时 fail closed。修复提交为
`b82c540`，不改变 proposal、trajectory、certificate 或 D2 输出。

**防重复**

- 所有 rate 必须同时报告 numerator、denominator、rejected、unknown；
- denominator=0 时写 `undefined`，不能写 0；
- terminal decision 必须读取 comparable export 和 usable yield；
- 原始 aggregate 不覆盖，修复生成新版本并记录 migration。

#### `H1-F06`：地图资产与证据缺口

**观察**

`/root/autodl-tmp/data/nuscenes/maps/` 只有 4 个 PNG，没有 vector JSON；本机没有 Waymo/nuPlan 数据。
`/root/autodl-tmp` 约有 65G 可用空间。

**卡点**

官方 lane graph/drivable polygon 暂不可查询；不能可靠地定义 target lane、connectivity、off-road 或
corridor crossing。DriveStudio adapter 代码的存在不等于数据和许可就绪。

**复开条件**

先生成最小资产清单并取得下载授权；保存来源、许可、大小、SHA256 和 scene→map 映射。不得从 raster PNG
反推正式 lane graph，也不得静默下载全量 Waymo/nuPlan。

### 0.4 禁止重试矩阵

| 快捷做法 | 为什么无效 | 允许的替代 |
|---|---|---|
| 降 known-fraction / coverage | 把无证据改名为有证据 | 增加独立 map/raw evidence |
| UNKNOWN 并入 PASS/FAIL | 改变预注册语义和分母 | 继续三态并单列 coverage |
| 删除 S1、005 或 004 actor 8 | 后验删难例 | scene-disjoint 新 pool |
| 换 actor、方向、P1–P5 幅度 | 用结果挑 proposal | 先冻结 event definition 与 actor pool |
| 0 export 报 0 violation | denominator=0 | 报 undefined + yield=0 |
| multi-sweep 直接堆叠 | 动态物体会 ghost | ego/dynamic motion compensation |
| learned occupancy 当 GT | 方法与 evaluator 循环 | raw/map 独立 evaluator + calibration |
| GS floaters/画质当安全证据 | renderer 不是物理传感器 | GS 只在 N4 导出 |
| 先做 H2/H3/scale | 没有 comparable positive | N1–N3 先过门 |
| 重命名 OccGS 复开 | 没有解除原失败 | 新路线必须满足复开条件 |

### 0.5 可复用资产

失败不否定以下工程资产：

- coordinate contract、`WorldState`、typed label/depth；
- run contract、artifact index、terminal marker 和 fail-closed aggregate；
- object-centric GS reconstruction/renderer；
- D1/D2 接口和 `PASS/FAIL/UNKNOWN` schema；
- 冻结 H1 bank 作为负对照与回归 fixture。

复用这些资产不能继承 H1 claim；新路线必须有新 preregistration、独立 event pool 和 evaluator。

## 1. 仍直接约束 V7 的历史结论

| ID | 状态 | 对 V7 的约束 |
|---|---|---|
| `RF-05` | rejected | 合法轨迹/点或局部像素变化不等于 RGB、遮挡、source removal、depth、identity 与标签都合法 |
| `RF-06` | rejected | 局部 loss 或 mask 不保证参数/输出只在局部改变；必须测 outside、boundary、frame-0 与 held-out |
| `RF-08` | limitation | 可复现的机器 evaluator 不等于绝对物理真值，更不能替代人工 verdict |
| `RF-09` | rejected | same-scene、shared identity 或结构合法不等于人类能辨别方法收益 |
| `RF-16` | limitation | layout/trajectory controllability 不等于 action-disentangled actor physics 或数据效用 |
| `RF-18` | rejected | ReSim `exp0_no_carla` 的 E-vs-F action response 不足；V7 不得借归档重开 C1P/C1S |

其他 RF 仍完整有效，但当前 OccGS 计划不直接复用对应的 SVD projection/preference 配方。

## 2. V7 风险索引

| ID | 状态 | 风险 | 禁止的快捷修补 |
|---|---|---|---|
| `V7-RISK-01` | rejected_v71 | occupancy 已接入 11D，但 certificate precision 与 repair yield 均未过预注册 gate | 因为 occupancy 文件存在或 D2 无 export 就宣称 H1 通过 |
| `V7-RISK-02` | limitation | C0 24/24 是按效应 top-k 的机器筛选，不是用户人工评测 | 写成 human pass，或只报 top-k 隐藏 46/62 全分布 |
| `V7-RISK-03` | open_risk | L0 mask 来自 RGB 差分，outside=0 由 hard composition 构造保证 | 用 0 leakage 宣称 occupancy-guided completion 有质量收益 |
| `V7-RISK-04` | open_risk | U0 以极端 V4 为 naive 对照且没有下游任务 | 把 accept rate / RGB signal 写成优于 naive GS 或 mAP 收益 |
| `V7-RISK-05` | legacy_limitation | V7 既有 run 缺正式 manifest、resolved config 与终态标记；V7.1 新 run 已由 EV-10 fail closed | 事后猜 seed/fingerprint 或伪造 immutable provenance |
| `V7-RISK-06` | open_risk | 只覆盖 mini 三场景，S1 held-out 质量偏弱 | 先扩规模、只筛容易场景或把三场景外推为论文结论 |
| `V7-RISK-07` | interface_mitigated_v71 | 11C 已闭合 WorldState→renderer→typed-label 工程链；occupancy repair 的方法增益仍未验证 | 把 label-sync 工程通过写成 occupancy certificate/projection 通过 |
| `V7-RISK-08` | legacy_risk_mitigated_v71 | O0 坐标注释、metadata 与实际变换含义不一致；11A 已冻结显式 frame 合同 | 沿用含义不明的 `pose/T`，或在 round-trip 前计算 H1 指标 |
| `V7-RISK-09` | confirmed_mitigated_v71 | 旧 rotated-corner AABB 使 PILOT-3 动态体素量膨胀 1.72–2.83 倍；扁平语义不能诚实移除 actor | 把旧 O0 AABB 当正式安全几何，或移除 actor 后把体积恢复为 free |
| `V7-RISK-10` | confirmed_failure_v71 | 高 UNKNOWN 在 11D 导致 10/30 D1 abstain、D2 30/30 拒绝与 0 usable yield | 把 UNKNOWN 并入 PASS/FAIL，或降低观测门槛追求 yield |
| `V7-RISK-15` | architecture_mitigated_v71 | certificate detection 与 trajectory projection 若混组会混淆检测和修复收益 | D1 修改 C trajectory，或把 D1/D2 合成单一 validity 数字 |
| `V7-RISK-16` | confirmed_failure_v71 | 冻结 30-proposal bank 得到 0 个 0→1 positive 和 0 个 same-actor pair | 用位移幅度或 RGB 差分代替 scenario-effect gate，或事后换 actor |
| `V7-RISK-17` | confirmed_mitigated_v71 | 单一 `depth` 名称会混淆 expected、first-hit 与 LiDAR measured truth tier；11C 已强制分名和 sidecar | 把 expected depth 登记为 measured GT，或省略 validity/truth-tier |

## 3. 风险详情与解除条件

### V7-RISK-01：occupancy 尚未进入方法

**观察**

- `occupancy/build_scene_occupancy.py` 独立写出 per-frame grid；
- `resim/s0_trajectory_editor.py` 只检查横向运动学、yaw、actor/ego 距离和粗横向范围；
- `resim/c0_counterfactual_render.py` 改写 RigidNodes pose，但没有查询 occupancy；
- `resim/l0_local_completion.py` 用 V0/edited RGB 差分构 mask。

**边界**

O0 是有用的世界状态基础设施，但当前不能支持“occupancy 提高合法性”或“occupancy-guided completion”主张。

**解除条件**

按 `V7-H1-11` 建立统一 actor/state mapping，让 occupancy 进入 edit certificate、visibility 与标签重生，并对
matched kinematic-only/naive baselines 做非循环消融。只添加一次 occupancy lookup 或 post-hoc filter 不足以解除。

### V7-RISK-02：机器 top-k 不等于人工合法率

**观察**

- C0 全部可见 case 为 46/62 machine legal；
- 24/24 是按 mean edit effect 排序后的 top-24；
- 当前 `reviews/` 目录是机器面板与机器 JSON，没有用户填写的 verdict。

**边界**

可表述为“机器筛选 top-24 均满足当前规则”，不得表述为“24/24 人工合法”或用其估计全候选分布。

**解除条件**

先冻结 blind sample、逐项 rubric、失败优先级、JSONL schema 与聚合阈值，再由用户或指定评审者完成 verdict。
agent 不代填，也不以机器字段映射成人工答案。

### V7-RISK-03：L0 primary metric 目前是构造不变量

**观察**

hard composition 直接复制 mask 外的 edited GS，因此 outside-mask L1 必然为 0；当前 12 帧结果只验证实现遵守
公式。mask 由 RGB 差分阈值和膨胀获得，不包含 ray visibility、unknown/free 或 source footprint geometry。

**边界**

L0 只证明 local composition 工程可行。没有证据表明 Telea 改善视觉、时序、depth 或 identity。

**解除条件**

使用 geometry-derived disocclusion mask，并在有真值的 pseudo-hole 上比较 no completion、Telea 与局部生成；
primary 必须包含 inside quality、boundary、temporal、depth/instance，而不是继续调阈值追 outside=0。

### V7-RISK-04：U0 proxy 不识别数据效用

**观察**

`naive_V4` 是约 39–50 m 的强制横移负例；它被拒绝只能证明 validator 能识别一个极端错误。当前没有训练
detector、occupancy model 或 event classifier，JSON 明确记录 `u0_full_map_pass=false`。

**边界**

不能声称 OccGS 优于 matched naive GS、real-only 或提供下游增益。

**解除条件**

对相同 proposal、相同样本量和相同训练预算比较 R / R+naive / R+OccGS / R+OccGS+completion，并使用
scene-disjoint split、至少 3 seeds 和任务指标。三场景只可用于 pipeline smoke。

### V7-RISK-05：既有 run provenance 不完整

**观察**

`runs/occgs_resim/` 现有 B0/C0/L0/U0 目录未发现 `manifest.json`、`resolved.yaml` 或终态标记。B0 仍有
`config.yaml`、metrics、checkpoint；其他阶段有 JSON 报告，但不足以满足正式 run contract。

**边界**

现有数值可作为 retrospective evidence，不能声称是完整、不可变、可从 manifest 一键复现的正式 run。

**解除条件**

`V7-EV-10` 为既有证据生成显式缺失项索引；所有新 run 通过 fail-closed wrapper 产生完整协议。禁止事后补造
未知字段或覆盖旧目录。

**2026-07-23 缓解结果**

- `V7_EVIDENCE_INDEX.json` 已逐文件索引 B0/O0/S0/C0/L0/U0 的 1,610 个文件，并保留正式字段的
  `missing/unknown_not_inferred`；
- V7.1 run contract 对 run ID 复用、三层 hash、artifact bytes、summary、冲突终态标记和 optional
  `not_triggered` 分支 fail closed；
- 正式 smoke 在 commit `3590558` 上以唯一 `COMPLETE` 结束，25 项相关测试通过。

该缓解只约束 V7.1 新 run；V7 旧 run 的 provenance 缺口不可逆，仍保持 retrospective/legacy limitation。

### V7-RISK-06：场景覆盖与质量

**观察**

本机只有 mini 10 scenes 具备前向完整 sweep；feasibility 只使用 3 scenes。S1 test PSNR/SSIM 为 20.18/0.472，
明显弱于 S0/S2。

**边界**

当前结果不能外推到 trainval、长时、多相机、夜间或复杂交互；也不能只删掉 S1 后报告更好均值。

**解除条件**

H1 先在冻结三场景与 worst-case 上通过，再审计可获得的 scene-disjoint 数据。扩展必须保留困难场景分层、
真实/插值 provenance 与相同门禁。

### V7-RISK-07：标签链未闭环

**观察**

C0 已改写 RigidNodes pose 并输出 RGB/depth/rigid 分量，但尚未形成统一的 semantic、instance、2D/3D box、
occupancy 与 visibility regeneration 流水线。

**边界**

“label synchronization”当前只可称 proxy/interface 可行，不是完整传感器与标签一致性。

**解除条件**

同一 world-state record 驱动 renderer 与所有标签 writer，逐帧验证 pose、depth、mask、box 和 occupancy 共位；
对缺失/不可见标签 fail closed。

**2026-07-23 缓解结果**

- 11C 在 PILOT-3 的 V0/V1、三场景、三前向相机上生成 18 个样本和 432 个 typed sidecar；
- 独立审计验证 18/18 样本、6/6 WorldState hash、temporal identity、三相机覆盖、instance-depth z-order 与
  state-specific safety/observation/render-support 引用；
- expected、first-hit、LiDAR measured depth 分名，有限 semantic scope 和 visibility provenance 均写入 sidecar；
- S1 保留，正式 run 以唯一 `COMPLETE` 结束。

该结果只解除 renderer/label 工程接口风险；11D 之前仍不能声称 occupancy certificate 或 repair 有方法收益。

### V7-RISK-08：O0 坐标框架歧义已确认

**观察**

- `occupancy/build_scene_occupancy.py` 文件头将 grid 描述为首帧 ego-centric；
- `meta.json` 将同一产物描述为 per-frame ego-centric；
- 实际实现每帧读取 `lidar_pose/{t}.txt`，以其逆矩阵把 world box 变换到 grid，同时直接使用 sensor-local
  LiDAR 点。因此产物实际是 per-frame LiDAR-sensor grid，而不是首帧固定 grid，也不能在未审计 LiDAR-to-ego
  外参前简称 ego frame；
- DriveStudio 则以起始 `CAM_FRONT` 的 `camera_to_world` 逆矩阵定义 model frame。

**边界**

现有 O0 数值仍可作为 coarse retrospective evidence，但在显式记录 `T_grid_world`、`T_model_world`、
`T_world_camera` 并通过 world→model/grid→world round trip 前，不得用于 H1 合法性指标。

**解除条件**

`V7-H1-11A` 统一使用 `T_dst_src` 命名，修正新 schema/adapter 的 frame 声明，以 synthetic fixtures 和
PILOT-3 原始标定验证 translation、yaw、box corners、camera projection 及 checkpoint pose round trip。
旧 O0 文件不原地改写；正式 H1 evidence 产生新版本与新 fingerprint。

**2026-07-23 缓解结果**

- 11A 将 annotation/model/grid/camera/LiDAR frame 分别冻结为 world、start-CAM_FRONT、per-frame-LiDAR、
  `T_world_camera` 与 `T_world_lidar`；
- 三场景 1,679 个 actor poses 的 translation、rotation、box 和三前向相机投影 round-trip gate 通过；
- registry 跨独立进程重建 hash 完全一致，正式 run 以唯一 `COMPLETE` 结束。

旧 O0 metadata 不原地改写，故该风险仍是 retrospective artifact 的 legacy limitation；V7.1 后续模块必须引用
11A coordinate contract 和新 fingerprint。

### V7-RISK-09/10：AABB 膨胀与高 UNKNOWN 已确认

**观察**

- 在完全相同的 PILOT-3 raw annotation、grid 和 240 帧上，旧 rotated-corner AABB 相对 oriented-box
  center-inclusion 的动态体素量比分别为 003 `1.721×`、005 `2.249×`、004 `2.833×`；
- 分离 dynamic instance layers 后，base unknown 比例仍为 `97.10% / 96.04% / 97.57%`；
- source actor removal 后原体积恢复 UNKNOWN，不会恢复 FREE；edited layer 可独立 remove/insert，三场景未出现
  layer overlap；
- 缺少 nuScenes map-expansion polygons 时 road-support 与 off-road control 保持 UNKNOWN。

**边界**

11B 已消除 AABB 作为正式动态几何和扁平 layer 删除污染，但没有降低 observation sparsity。30 条可测真实
controls 的 retention 为 100%，collision/teleport 可检测负例为 2/2；然而加入 road-support 后 32 条完整
certificate 全为 UNKNOWN。这是诚实 abstention，不是 H1-CERT pass。

**后续约束**

D1 必须报告 precision、recall、abstention 和 PASS coverage；UNKNOWN 不进入 TP/FP/FN。只有独立观测或 map
证据能把 UNKNOWN 变为可判定状态，禁止通过调大 unknown threshold、把 box 当 background surface 或用 Gaussian
floaters 补 safety evidence。

### V7-RISK-15/16：certificate/projector 与 scenario effect 必须继续拆分

11B 已冻结 `scenario-effect-v1` 的纯 3D 0→1/0→0 gate、same-actor pair schema 和
`certificate-calibration-v1` 三态接口。11D 必须让 D1 逐字节复用 C trajectory，D2 才允许修改轨迹；位移 proposal
若未形成冻结的 corridor crossing、duration、gap 与 TTC/headway 条件，只能标为 non-event，不能靠命名成为
cut-in/merge positive。

### V7-RISK-17：typed depth 语义混淆已缓解

11C 把 depth 冻结为三个不同产品：diagnostic expected depth、T1 Gaussian first-hit depth、T0 LiDAR measured
depth；每个产品有独立 validity、definition、truth tier 与 artifact sidecar。独立审计确认三类各 18 个，且没有
expected-as-measured 混写。后续 export/evaluator 必须继续按产品名和 truth tier 消费，不能重新折叠成无类型
`depth`。

### V7-H1-11D：H1-CERT / H1-PROJ 预注册拒绝

**冻结事实**

- source-only eligibility 覆盖 3 scenes × 2 actors，P1–P5 共 30 proposals；S1 未删除；
- C/D1 realized trajectory hash 30/30 完全相同；
- D1：precision `0.75`、recall `0.8824`、abstention `0.3333`、PASS coverage `0`；
- C external hard violation `17/30`；
- D2：0/30 export、0 usable yield，external rate 不可定义；
- scenario-effect：0 positive、25 negative、5 source-positive/non-event，0 same-actor pair。

**裁决**

H1-CERT 因 precision 低于 `0.80` 拒绝；H1-PROJ 因拒绝全部 proposal、无 comparable export、usable yield
低于 `70%` 拒绝。按路线转向规则停止 OccGS 方法 claim，只保留 object-centric GS、WorldState、typed label、
certificate/evaluator 与 run-contract 基础设施。

**唯一修复与防重复**

首版聚合把 rejection 计成零违规，已作为 `metric_aggregation_bug` 唯一修复，旧 aggregate 保留。修复未改变
方法输出；第二版对无 export 的 rate fail closed。不得继续：

- 调低 known-evidence/coverage 门槛把 UNKNOWN 改成 PASS；
- 删除 005/S1 或 004 actor 8；
- 根据现有结果重选 actor、方向、proposal 或 event threshold；
- 用固定-pool `0/30 violation` 隐藏 D2 的 `30/30 reject`；
- 因 recall 达标而隐藏 precision fail，或把 UNKNOWN 排除后重算；
- 在当前配方上继续 H2/H3/scale。

## N1 receiver-centric cut-in final：第四轮后新增防重复项（2026-07-26）

### N1-F18：receiver branch merge 的 13/13 历史假阳性不能靠阈值微调挽回

第四版旧 parent 的 18 个 machine candidates 中有 13 个 receiver-branch merge；第四轮人工裁决表明这类
历史 branch-merge 候选均为 `FALSE_POSITIVE`。因此 branch topology、`target_incoming_count`、shared successor
或 token change 永远不能单独证明 cut-in。final v2 把该类别固定为
`ABSTAIN/UNSUPPORTED_BRANCH_MERGE_MODE`；不得为了候选数量重新放宽它。

### N1-F19：support-count 不能替代完整 receiver identity 时序

K4-012 暴露了“support 数量足够”仍可能跨 raw 帧切换接收车身份的问题。legacy fixture 的 `1→38` 与 v2
raw map 重放中观测到的 `9→1→9` 是不同窗口/枚举证据，二者都不能被静默等同为连续 receiver。final v2
要求 required raw frame 全窗唯一 non-null identity、last-post anchor 和每帧 rank/gap/path-clear；任一身份
切换必须 FAIL 或 ABSTAIN，不能被总 support count 抵消。

### N1-F20：弯道 map jitter 的 post heading 不能借由宽松窗口穿透

K4-015 证明 source/target 局部不平行或 post heading 过大时，几何横向收敛可以伪装成切入。final v2 使用
local parallel overlap、raw post-heading、累计 yaw 和 raw-only provenance；它不是针对 scene/token 的黑名单。
禁止为了保留 K4-015 或增加 PASS 数而放宽这些 hard gate。

### N1-F21：CAM_FRONT 的五帧截图不能承担角色/时序的完整证明

单相机可见性和五帧页面截断无法可靠展示 SUBJECT、RECEIVER、source/target corridor 的完整 raw 窗口。
审核 V2 因而以逐 raw-frame topdown、2 Hz signals、actor-ID switch 标注和固定 camera-unavailable 警告为主；
相机只作可选证据。看不清必须 `UNCERTAIN`，不得肉眼猜身份或通过下载未授权传感器补洞。

### N1-F22：final v2 不是旧阈值的第五次微调

本轮只吸收第四轮已完成的校准信息，变更的是事件语义和证据链：parallel-only subject body entry、独立
receiver 全时序、raw 2 Hz hard evidence、三态 first-failure、streaming worker 与 blind/debug 分离。K4 只做
固定 regression；Resource Contract V1 在任何 final scene 前失败，用户复开后的 V2 已按 N1-F25 修正为
675 scenes 并完整运行，但同样没有用于调参。后续若研究 branch merge 或新资产，必须是新的任务 ID、
预注册与 scene-disjoint 评估。

### N1-F23：共享 cgroup 的 start 合同是研究终止门，而非可绕过的工程告警

final formal 在 clean commit `7104f5c` 的 preflight 已将 runner 自身 RSS 降至 `20,705,280` bytes，仍记录
`cgroup_memory_current_bytes=1,523,929,088`，超过冻结上限 `1,350,000,000`。它在任何 evaluation scene 前
安全失败；独立裁决冻结证据并以 `REJECTED/stop_nuscenes_cutin_mining` 结束。development override 的 32/96
smoke、清页缓存或 K4 回归均不能替代正式 start 合同。禁止杀死 Cursor/Jupyter/TensorBoard 等用户服务、修改
正式阈值、截断正式 split（当时 expected 常量误写为 669，见 N1-F25）或把这次结果说成“nuScenes 没有
cut-in”；`n2_authorized=false` 保持不变。

**2026-07-26 用户复开授权**

上述 `REJECTED` 仍是 Resource Contract V1 下不可改写的历史裁决，但不再代表任务永久停滞。用户随后显式
扩大容器内存并授权继续本次 final：现场复核 `memory.max=128,849,018,880` bytes（120 GiB），原
`2,147,483,648` bytes（2 GiB）资源前提已改变。因此必须保留失败 parent 与独立拒绝裁决，同时使用新的
Resource Contract V2、全新 config fingerprint 和不可复用 run ID 恢复 scene-disjoint formal
（经 N1-F25 确认为 675 scenes）；不得覆盖或续写
V1 失败目录，也不得把 V2 成功倒写成 V1 当时没有失败。

### N1-F24：内存不足时停止并等待资源授权，不继续死磕

**新执行规则**

1. 任何正式或开发任务若触发启动/运行 stop 阈值、`RC=137`、SIGKILL，或观察到持续逼近 cgroup
   `memory.max`，立即停止启动新 batch，并尽最大可能写入结构化 `FAILED/failure.json`、最后完成 scene、
   process RSS、cgroup current、anon/file cache 与 `memory.events`；
2. 不通过反复重跑、杀死 Cursor/Jupyter/TensorBoard 等用户服务、缩短正式 split、降低证据质量、跳过
   audit、修改研究阈值或清理不属于本任务的缓存来争抢资源；
3. 把失败点、最低所需资源和恢复命令回报用户，然后等待用户开放资源；没有新的明确授权时不得自行恢复；
4. 用户开放资源后，先记录新的 `memory.max/current` 和授权时间，版本化 resource contract，使用新 run ID
   从冻结研究配置重新运行。资源合同变化只允许调整资源阈值，不允许调整 cut-in taxonomy、hard gate、
   calibration/evaluation split、抽样或人工聚合门槛；
5. 资源暂停与研究拒绝分开登记。未读取 prospective evaluation scene 的资源失败不能被写成方法精度失败，
   后续在新授权下完成的结果也不能删除或覆盖先前工程失败证据。

### N1-F25：final 的 669-scene 预期是 split 算术错误，不是 evaluation 集合定义

Resource Contract V2 的首次 clean formal run：
`/root/autodl-tmp/runs/event_first/N1-EVENT-CUTIN-FINAL-01/v71_n1-event-cutin-final-01__receiver-cutin-final-v1__s0__20260726T142634031503Z__5c8c65d7`
在 K4 regression 通过后、任何 evaluation scene 或 candidate 读取前 fail closed，错误为
`final evaluation scene 数不匹配: 675 != 669`。

独立复算表明：nuScenes official `train` 为 700 scenes；冻结的 42 个 calibration scenes 中，25 个属于
`train`、17 个属于 `val`，且没有 split 外 scene。因此 scene-disjoint evaluation 的确定数量是
`700 - 25 = 675`。`_resolve_evaluation_scenes` 已正确执行
`set(train) - set(all_calibration_scenes)`；错误只在 YAML 的 `expected_scene_count` 常量把不属于 train 的
calibration scene 也错误计入了减法。

合法修复仅为把 Resource Contract V2 配置中的 assertion 从 669 改为 675，并生成新 config fingerprint、
新 clean commit 和新 run ID。不得借此增删 calibration scene、显式挑选 evaluation scene、查看 candidate 后
改 split，或修改 taxonomy、strict gate、K4、抽样与人工门槛。失败 run 必须保留为工程契约失败，不能统计成
research reject。

### N1-F26：strict v2 在 675 scenes 上只有 1 个 PASS，不能靠人审单例或放宽规则扩池

Resource Contract V2 的 clean formal run：
`/root/autodl-tmp/runs/event_first/N1-EVENT-CUTIN-FINAL-01/v71_n1-event-cutin-final-01__receiver-cutin-final-v1__s0__20260726T142941598714Z__883fae9a`
在 commit `beee1de`、seed 0、config fingerprint
`883fae9a6514c0bff5bba8bcaf81a22c79e6d719586221596a7d4b5364c337da` 上完成 675/675 scenes。

结果为 `ABSTAIN=1,556`、`FAIL=200`、`PASS=1`，唯一 PASS 只覆盖 1 scene；冻结 machine-readiness 要求
至少 3 candidates / 3 scenes，故 parent 以
`REJECTED / stop_nuscenes_cutin_mining_too_sparse` 结束。K4、raw-only 和资源合同检查均通过，峰值 batch
process RSS 为 `337,154,048` bytes、cgroup current 为 `4,556,898,304` bytes；这次拒绝不再是资源失败。

独立稀疏终局人工包保留唯一 PASS 和 3 条 diagnostic，但人工结果不能改变数量门失败：即使唯一 PASS 被判为
TP，仍只有 1 TP / 1 scene，低于 sparse 的 3/3。禁止为了形成池而把 ABSTAIN 提升为 PASS、恢复
`receiver_branch_merge`、放宽 raw/parallel/receiver 时序门、事后改 scene 或把单例人工真实性外推为总体
precision。准确结论是“当前冻结 strict v2 的 prospective pool 过稀”，不是“nuScenes 没有 cut-in”。

## 4. 跨路线必须保留的原则

1. 先证明监督/比较对象存在，再训练或扩量。
2. occupancy、编辑、渲染和标签必须共享同一显式状态，不允许旁路文档绑定。
3. matched baseline 使用相同 proposal、scene、actor、幅度、seed 与预算。
4. top-k 只用于诊断，不替代全分布、coverage 与 worst-case。
5. machine pass 只解锁下一门禁，不自动成为 human verdict、论文 claim 或 scale 授权。
6. hard composition 的局部性与 completion 的质量是两个独立门禁。
7. 下游效用必须由任务指标证明，不能由约束 accept rate、RGB 差分或 PSNR 代替。
8. 工程失败与 research reject 分开登记；既有 provenance 缺失必须诚实标记。
9. 失败范围不能过度外推，但也不能通过改名、放宽阈值或只挑成功场景重复旧问题。

## 5. 新实验防重复检查表

- [ ] 是否明确引用了相关 `RF-*` 与 `V7-RISK-*`？
- [ ] occupancy 是否真正进入决策/状态链，而非只在磁盘上存在？
- [ ] baseline 是否 matched，而非故意构造的极端负例？
- [ ] primary endpoint 是否避免“方法规则自己定义方法成功”的循环论证？
- [ ] 是否同时报告全分布、coverage、per-scene 与 worst case？
- [ ] completion 是否测 inside quality/temporal/depth，而非只测 outside exact？
- [ ] human verdict 是否只由用户/指定评审者填写？
- [ ] run 是否有唯一 ID、resolved config、fingerprint、metrics、summary 与终态标记？
- [ ] 哪个单卡门禁失败时停止，什么条件才允许 scale？

## 6. 2026-07-26 路线转向新增失败与防重复项

### PIVOT-F01：nuScenes cut-in 没有可验证的召回率分母

**观察**

nuScenes 官方公开的是场景、样本、对象实例、类别、属性、3D 框、传感器与地图等结构；`scene.description`
是自由文本，不是事件级 cut-in 真值。官方没有发布 cut-in 场景占比、逐事件标签或可直接用于召回率计算的全集分母。
四轮挖掘和最终 675-scene prospective run 最多只能测量“冻结规则产出的候选质量”，不能测量“数据集中所有
cut-in 被召回了多少”。

**最终证据**

- `N1-F26`：strict v2 在 675 个 scene 上只有 `1 PASS / 1 scene`；
- 最终人工稀疏包即使把唯一 PASS 判为真阳性，也仍低于预注册的 `3 candidates / 3 scenes`；
- 该结果不能外推为“nuScenes 没有 cut-in”，也不能用事后放宽规则伪造召回。

**裁决**

`cut-in mining` 状态固定为 `rejected / frozen`。以后 cut-in 只允许作为已经具备重建与编辑能力后的可选演示，
不再承担数据集入口、方法定义、训练前置条件或论文成立条件。

**解除条件**

只有新的、独立的数据源提供事件级真值及明确分母，或新的任务本身不需要宣称事件召回率，才允许创建全新任务 ID
重新讨论；不得恢复当前 strict-v2 阈值调参。

### PIVOT-F02：贡献漂移——工程系统吞噬了重建与编辑研究

**观察**

过去路线的主要投入逐步变成事件挖掘、地图匹配、接收车身份、规则校准、候选审核与资源合同。它们改善了审计性，
却没有自然回答动态对象几何、连续运动表示、遮挡/去遮挡、反事实轨迹编辑或下游感知一致性。

**边界**

这不否定已形成的 WorldState、typed label、run contract、审计与人工审核基础设施；它只否定把“更好的 cut-in
挖掘器”作为 3DGS/4D 重建论文的核心贡献。

**后续约束**

新路线必须先复现公开强基线，再通过重建/编辑压力测试选择创新点。数据工程模块只能服务于冻结实验，不得重新成为
论文主任务。每个里程碑都必须说明它直接回答的重建或编辑问题。

### PIVOT-F03：未完成 exact reproduction 前禁止集成式“改进”

**观察**

`RF-05/06/08/09/16/18` 与 `V7-RISK-03/04/06/07/10/16/17` 共同表明：输入、状态、比较对象、覆盖率和真值定义
未冻结时，模块堆叠会把工程可运行误当成方法收益。AD-GS 的公开 nuScenes 协议提供了固定 scene、帧区间、预处理、
训练和评测入口，适合作为新的事实锚点。

**禁令**

在 AD-GS exact reproduction 门禁通过前，不得：

- 合并 Motion-Proj/StreetGS/OccGS 模块；
- 加 occupancy、物理约束、扩散补全、感知损失或轨迹编辑；
- 更换为自选事件场景、调低分辨率后对齐论文指标或只展示成功帧；
- 把兼容性补丁、预处理修复或运行成功表述为方法改进。

任何 unavoidable compatibility patch 必须独立提交、最小化、附 upstream diff 与消融；原始基线结果必须保留。

### PIVOT-F04：不能把“可见性建模”泛化成未观测背景已经解决

**观察**

AD-GS 的双向时间可见性用于动态对象生命周期和已观察运动建模；VAD-GS（CVPR 2026）的 visibility-aware
densification 已覆盖稀疏观测下的几何补密；DrivingEditor 支持对象删除/添加；Real2Sim 进一步展示对象级编辑与
物理交互。这意味着“增加一个 visibility 模块”或“支持平移对象”本身已经不足以构成新意。

**仍未闭合的问题**

反事实轨迹编辑会同时制造原位置去遮挡、新位置遮挡、跨相机深度排序和证据外外观。当前项目只有在下列内容形成
联合、可验证方案时才可提出方法 claim：

- 编辑诱发的显式可见性重计算；
- 未观测区域的真实性/置信度与拒绝机制；
- 跨视角、跨时间一致的背景恢复；
- 目标区预期变化与非目标区感知保持。

**防重复**

创新选择前必须把 AD-GS、VAD-GS、DrivingEditor、DGGT/ReconDrive 和当时最新工作重新做一次代码可用性与
claim 边界审计；不得把已有 visibility-aware densification 或基本对象变换重新命名为贡献。

### PIVOT-F05：资源不足时研究停机规则跨路线继续生效

`N1-F24` 是项目级规则，不属于 cut-in 专属逻辑。本轮 cgroup 为 `memory.max=2,147,483,648` bytes，轻量元数据
审计后 `memory.current` 一度达到 `2,129,526,784` bytes，因此立即停止 Python 扫描、conda 求解、下载、
预处理和训练，只继续轻量文本/文件操作。后续任何新路线任务遇到相同条件时，必须保存失败/现场证据并等待用户开放
资源；不得反复重跑、杀用户服务或偷偷缩减正式协议。

### PIVOT-F06：旧机器 smoke 证据不能替代新实例复验

迁移到 RTX 4080 SUPER 新容器后，已有环境目录和旧 RTX 4090 日志仍然存在，但它们不能证明当前 driver、CUDA、
扩展 ABI、显存与 cgroup 合同可用。M2 因此在新机器上重新执行 AD-GS forward/backward、DPT、SAM2、
Grounding DINO HF 和 CoTracker3 smoke，并为每项保存独立退出码。

后续换机或容器重建时，即使复用同一 env/checkpoint，也必须生成新的 instance 级环境证据；旧日志只能作为历史，
不能复制为当前 PASS。

### PIVOT-F07：非 login shell 的 PATH 不能作为 CUDA provenance

M2 首次当前机器采集因非 login shell 找不到 `nvcc` 提前失败，而 `/usr/local/cuda/bin/nvcc` 实际存在。环境报告
已改为显式设置 `CUDA_HOME` 并调用绝对路径，同时传播 smoke 的真实退出码。

后续自动任务必须显式记录并使用 toolkit 路径；“命令不在 PATH”与“机器没有 CUDA toolkit”必须分开裁决。

### PIVOT-F08：在线浮动模型不能进入 exact reproduction

upstream 的 CoTracker `torch.hub` 在线 `main` 与未固定 revision 的 Hugging Face 模型都会随时间变化。M2 将
CoTracker repo、离线 checkpoint、Grounding DINO HF revision 与 snapshot fingerprint 全部固定并哈希，
运行时使用 offline mode。

后续 baseline 不得在正式 run 中联网追随 `main`、latest 或未固定 snapshot；若必须升级，使用新 config
fingerprint 和新 run instance。

### PIVOT-F09：tar 页缓存与 nuScenes auxiliary 都属于资源/资产合同

并行流式扫描约 294 GB tar 时，文件页缓存计入本容器 cgroup，首个扫描实例峰值达到 `57,001,484,288` bytes。
本任务只对自己已读过的 tar 文件区间调用 `POSIX_FADV_DONTNEED`，没有全局 `drop_caches`、杀用户服务或清理
其他任务缓存。

第一次结构审计还发现 1,440 个 RGB/LiDAR payload 齐全并不足以初始化 nuScenes devkit；`map.json` 引用的
4 个静态 map masks 同样是必需资产。失败实例
`20260727T165549__e49a4e-4080s-r2` 保留为 `blocked`，补齐并哈希登记 maps 后由新实例
`20260727T180733__e49a4e-4080s-r3` 通过。以后 selective extraction 必须同时审计运行库隐式依赖的 auxiliary
文件，不能只按训练脚本直接打开的传感器路径计数。

### PIVOT-F10：AD-GS 的 PNG 输出与 SAM2 的 JPEG-only 枚举不兼容

M3 首个 scene-0230 实例完成 `prepare_raw` 和 180 张 depth 后，在 sky mask 初始化时报
`no images found`。AD-GS `scripts/nuscene/nuscene.py` 固定写 `000000.png`，其 `semantic.py` 自己也会枚举
PNG；但 Grounded-SAM-2 `load_video_frames_from_jpg_images` 只按 `.jpg/.jpeg` 扩展名建立 video frame 列表。
这不是空数据、模型失败或资源 OOM。

失败实例 `20260727T181617__scene0230__s0` 保留为 `blocked`。最小兼容性修复只在 instance work dir 中为每个
PNG 建立相同字节内容的 `.jpg` 硬链接，跨文件系统时复制原始字节；PIL 已验证按内容可无损读取，不做 JPEG 转码，
不改 Grounding/SAM 模型、box/text 阈值、mask、帧序或评测。修复后的 AD-GS patch SHA-256 为
`114c3976af2c80d1da5581b401b3a099f22a7483347fc401113c8439bc991eb9`，必须由新 M3 instance 复验。

### PIVOT-F11：COLMAP 默认全核并发会越过本机 cgroup 内存门禁

M3 第二个实例 `20260727T182247__scene0230__s0-r2` 已完成 180 张 depth/object/sky/semantic 与
138/138 flow，在 COLMAP feature extraction 阶段发现 upstream 未指定线程数，COLMAP 自动使用容器可见的
128 个 CPU threads。cgroup memory 峰值达到 `62,265,835,520` bytes，并连续两个采样超过
`memory.max=66,571,993,088` 的 90% 停止线；runner 只终止本 run 的进程组，`oom=0 / oom_kill=0`，
失败实例与部分 COLMAP 目录均保留。

这不是图像、SIFT、匹配或几何协议失败。最小资源兼容修复显式传入
`SiftExtraction.num_threads=16` 与 `SiftMatching.num_threads=16`；不改分辨率、相机、帧、SIFT 参数、
exhaustive matching 或评测。r3 的 COLMAP 已完成 138/138 图像注册与 70,933 points，阶段峰值降至
`35,117,174,784` bytes。当前完整 compatibility patch SHA-256 为
`49b4c06ecec6c30f1e80b5abf4d46970920f9d71952acbda273774d9b5b34f48`。

以后在 CPU 核数远大于内存预算的容器内运行 COLMAP，必须显式登记并发数并纳入资源合同；不得把减少图像、
降低分辨率或删相机伪装成等价的资源修复。

### PIVOT-F12：跨场景连续执行时，official render 可先于 OOM 触发 cgroup 90% 合同

M3 scene-0230 的 60k render 峰值已达到 `59,530,678,272` bytes，距离注册的 90% 停止线仅
384,115,507 bytes。M4 scene-0242 严格串行完成全部 preprocess 后，100-step train 峰值为
`59,359,428,608` bytes；随后的 official render 在第 2/138 帧连续两个采样达到 90%，峰值
`59,996,393,472` bytes，比停止线高约 81.6 MB。runner 只向本 stage 进程组发送 `SIGTERM`，
stage `rc=-15`、runner `rc=1`，`oom=0 / oom_kill=0`，没有影响其他服务。

该结果说明当前 `memory.max=66,571,993,088` bytes 对六场景连续 exact reproduction 没有足够安全余量；
“尚未 OOM”不能用来绕过预注册停止线。blocked 实例
`/root/autodl-tmp/runs/dynamic_recon/DR-M4-ADGS-6SCENE-01/20260727T235743__scene0242__s0/`
必须保留，不在同一资源合同下立即重跑，也不得降分辨率、删相机、调模型或全局 `drop_caches`。

恢复时应先提高 cgroup 内存额度，建议至少 80 GiB、推荐 96 GiB，再创建新 instance；允许复用逐文件冻结的
processed scene，但必须记录来源、哈希和新资源合同。若无法增加资源，则 M4 保持 `blocked`，不得将只有
scene-0230 的结果写成六场景论文复现。

### PIVOT-F13：processed scene 复用校验必须区分关键产物与合法空占位

RTX 3090 换机后的首个 scene-0242 复用实例
`/root/autodl-tmp/runs/dynamic_recon/DR-M4-ADGS-6SCENE-01/20260728T131533__scene0242__s0-r2-wm3090/`
在训练前 fail closed：新增递归哈希校验把 COLMAP 的合法 0-byte `created/sparse/model/points3D.txt`
占位文件误判为损坏。旧实例以 `blocked` 保留，没有修改 processed scene 或启动训练。

合法修复允许 COLMAP 非关键占位文件为 0 bytes，同时继续强制 `database.db`、`cameras.txt`、
`images.txt`、`colmap.ply` 和所有训练直接消费的 image/depth/mask/flow/meta/point cloud 非空；
复用后必须重新运行独立 processed audit。修复后的新实例 output fingerprint 为
`32bf9ccaa108273b69286625a0c7aaacb04fd9d76f243daff976206d0b7ef4f6`，138/138 registered images
审计通过。不得删除空占位文件、伪造非空内容或因此重跑昂贵预处理。

### PIVOT-F14：容器实例重建必须与 OOM/方法失败分开

M5 首个正式 DGGT 实例
`/root/autodl-tmp/runs/dynamic_recon/DR-M5-DGGT-NUSC-01/20260729T094923__native-nusc-s0-wm3090/`
在 `env_torch` 下载期间停止。日志没有 stage 终态、OOM 或 `oom_kill` 增量；当前容器 PID 1 的启动时间为
2026-07-29 13:13:43 +08:00，晚于日志停止时间，故裁决为外部容器实例重建，而不是 DGGT 精度、显存或方法失败。

旧 run 与旧 controller 已原子标为 `blocked`，部分环境移动到
`/root/autodl-tmp/envs/dggt.interrupted-20260729T094923/`，没有覆盖或删除。恢复使用新 run
`/root/autodl-tmp/runs/dynamic_recon/DR-M5-DGGT-NUSC-01/20260729T133346__native-nusc-s0-wm3090/`；以后同类中断必须先核对 PID 1 启动时间、stage marker、launcher 与 OOM 证据，再决定是否创建
新实例，禁止把 stale `running` 当成任务仍存活。

### PIVOT-F14B：pointops2 的 PEP 517 build isolation 没有继承已安装 torch

恢复实例完成 Python 3.10、torch 2.4.1 和全部 requirements；resolver 最终选择
`rerun-sdk 0.23.1 / opencv-python 4.11.0.86 / numpy 1.26.4`。随后 upstream pointops2 执行普通
`pip install .` 时，PEP 517 临时 build env 在读取 setup requirements 阶段报
`ModuleNotFoundError: No module named 'torch'`。正式 stage `rc=1`，峰值 cgroup memory
`16,839,843,840` bytes、GPU 0 MiB，`oom=0 / oom_kill=0`。

正式 blocked 证据：
`/root/autodl-tmp/runs/dynamic_recon/DR-M5-DGGT-NUSC-01/20260729T133346__native-nusc-s0-wm3090/`。
checkpoint、native inference 和 common-observation metrics 均未启动；没有在同一实例事后加入
`--no-build-isolation` 覆盖失败。该结果属于明确 upstream packaging blocked，不是 DGGT 质量、显存或方法裁决，
并按权威计划第 15.1 节满足继续 M6 的替代前置证据。

### PIVOT-F15：AD-GS 冻结 pseudo ID 与 checkpoint 都不能支持单对象编辑

M6 直接审计训练前冻结的 `semantic/mask_*.npy`。按 camera-local ID 统计，六个官方场景最长支持帧为
`1 / 6 / 1 / 1 / 2 / 1`，全部低于预注册 `≥20/60`；processed scene 也没有冻结的 vehicle track artifact。
与此同时，六个 60k checkpoint 的 `point_cloud.ply` 均只有二值 `obj∈{0,1}`，没有持久 instance ID。

正式证据：
`/root/autodl-tmp/runs/dynamic_recon/DR-M6-STRESS-01/20260729T145645__identity-audit-s0-wm3090/`。稳定失败
`persistent_object_identity_unavailable` 在 6/6 scenes 重复。对象编辑、pseudo-hole 和噪声行全部保留为
`ABSTAIN`，0/12 object slots 没有从 coverage 分母删除。禁止在看到 checkpoint/场景结果后用几何 Hungarian
轨迹回填 M6 baseline；这类重关联只能作为新方法候选，并必须先过 novelty。

### PIVOT-F16：instance-aware 与 driving edit 已被 2025–2026 工作直接覆盖

M7 只沿决策表考察 A“可编辑运动表示与轨迹不确定性”。重新核对官方来源后：InstDrive 已用 SAM pseudo masks
学习动态驾驶场景 2D/3D instance identity；Director 已做 4D Gaussian identity consistency；OmniRe 已用
actor scene graph/canonical vehicle nodes 做仿真；HorizonForge 与 G²Editor 已覆盖车辆轨迹操作、删除和遮挡区
恢复。

正式 evidence：
`/root/autodl-tmp/runs/dynamic_recon/DR-M7-HYPOTHESIS-01/20260729T145748__novelty-audit-s0-wm3090/`。候选的持久身份、actor-centric Gaussian binding、时序一致性和轨迹/对象编辑
核心机制均为 direct overlap；confidence/ABSTAIN 是评测与安全护栏，不构成独立技术 delta，剩余差异只是
AD-GS 适配工程。因此 M7=`rejected`，不注册事后 primary endpoint；M8/M9 均
`rejected / not authorized`，不得通过改名、挑场景或把 0 coverage 写成改进继续。

### PIVOT-F17：DGGT 扩展构建必须同时固定 compiler、headers 和 Python 依赖上界

V2 M1 表明，“已安装 torch cu121”并不足以证明 CUDA extension 可构建。宿主只有 CUDA 11.8
toolkit，会在 pointops2 编译时与 torch 2.4.1+cu121 硬失配；只补 `nvcc` 又会缺
cusparse 等 headers。正确合同是在前缀环境固定 NVIDIA CUDA 12.1 compiler/runtime/headers，
传播 `CUDA_HOME/CPATH/LD_LIBRARY_PATH`，再按 upstream `python setup.py install`。

同一里程碑还暴露了浮动 Python 树的独立风险：transformers 5.x 使用 torch 2.4.1 未提供的
DTensor API，diffusers 0.39 触发 torch schema 不兼容。最终固定
`transformers 4.48.3 / tokenizers 0.21.0 / diffusers 0.32.2 / numpy 1.26.4 /
opencv-python 4.11.0.86 / rerun-sdk 0.23.1 / flow-vis 0.1`。

对应 blocked runs 为
`20260802T120027Z__native-nusc-s0`、`120943Z__...-r2`、`122213Z__...-r3`、
`122904Z__...-r4`、`124347Z__...-r5`。这些失败是构建/依赖证据，不能推断 DGGT
方法质量。

### PIVOT-F18：原生阶段完成不应被后续评估依赖失败覆盖

M1 r6 已完成 18/18 1-view 和 18/18 3-view，但 common evaluator 导入 AD-GS 冻结
`loss_utils` 时因 `flow_vis` 未安装而 blocked。原生输出本身未损坏，但主 terminal 已转为
blocked，禁止为了“好看的 done”改写。

恢复方式是新建 r8，对 r6 `native_summary.json/metrics.json`、每个 stage 和输出哈希做
fail-closed 引用后只执行 common diagnostic。r7 中重试封装自身的 `KeyError` 也以新的
blocked run 保留，再由 r8 完成。后续所有 multi-stage run 必须把“可复用的完成阶段”与
“整个 instance 的 terminal 终态”分开；重试不得修改旧 terminal。

### PIVOT-F19：nuScenes devkit 反向索引与磁盘 metadata 不是同一 schema

M2 r1 直接读取官方磁盘 `sample.json` 时发现其中没有 `anns`；该字段是 nuScenes devkit
初始化后才注入的反向索引，不是原始 JSON 合同。正式适配器改为流式扫描
`sample_annotation.sample_token`；由于这个外键非唯一，不得用单值 dict 覆盖同一 sample
的多个 annotation。`ijson` 还必须以 `use_float=True` 读取，否则 Decimal 会污染严格 JSON
运行合同。

同一里程碑还表明，“时间最近”不足以建立 raw annotation 到 camera sweep 的真值映射。r4
中 scene-0242 boundary actor 命中更近的 sweep，但 sweep 所属 `sample_token` 与 raw 2 Hz annotation
不同，因此在 QA 前即 blocked。正确规则是先限定 exact sample token，再在候选内最小化
timestamp delta；正式 r5 达到 `4356/4356` exact mappings。后续不得仅按文件名或时间
猜测 raw/processed/render 映射。

### PIVOT-F20：CUDA 扩展 import 成功不等于包含当前 GPU 架构

M3 的 DriveStudio 环境能正常 import `gsplat` 和 `nvdiffrast`，但旧二进制没有 RTX 3090 的
SM 8.6 kernel：前者在 SH rasterization 报 `no kernel image`，后者在 EnvLight 路径报 CUDA 209。
只做 import smoke 无法发现此类错误。恢复时分别固定官方源码 commit，以
`TORCH_CUDA_ARCH_LIST=8.6+PTX` 重建，并执行真实 CUDA forward/backward；旧 `.so` 先备份，
没有修改算子语义或模型配置。

对应 blocked runs 为 r4、r6、r7；正式 binary SHA-256 为 gsplat
`6d7c8e5a...dd6131`、nvdiffrast `0d18f767...96499`。以后 CUDA 扩展 readiness 必须包含
目标 GPU 上的实际 kernel forward/backward，不能只看包版本和 import。

### PIVOT-F21：训练完成 checkpoint 与累积式 post-render 必须分开裁决

M3 r8 的 30k 原生训练已保存 `step=30000` checkpoint，但上游随后将 588 个 full-render 结果累积在
内存中；在 `577/588` 时 cgroup memory 连续两次超过 90%，资源守卫发送 SIGTERM。`oom=0 /
oom_kill=0`，checkpoint 字节数与 step 完整。r8 仍保持 `blocked`，不得改写为 done；r12 通过新的
不可变 run 对 checkpoint step/bytes/hash 和原失败 terminal 做窄范围复核，再执行流式 27-image
edit smoke 完成 M3。

同一恢复链还发现，正式训练会把某个非目标 rigid model 的全部 Gaussian 裁剪掉。token、dataset column
和 model index 仍是一一映射，但 checkpoint slice 为空。registry v2 因此将其明确标成
`unavailable_empty_checkpoint_slice`，同时对正式选中 actor 继续要求非空。禁止为了全 registry 看起来
完整而伪造 slice，也禁止因一个非目标空 slice 丢弃 23 个真实非空映射。

### PIVOT-F22：外层 timeout 不会自动回收独立 session 的 GPU 子进程

M4 controller 用 `subprocess.Popen(..., start_new_session=True)` 隔离正式渲染，使 SSH/tmux 断开不应
误杀长任务；相应地，用外层 `timeout` 调试 controller 时，SIGINT 只终止父进程，子进程会以 PPID 1
继续占用 GPU。`debug_controller_s0_r5` 复现了该行为；残留子进程通过已核实的精确 PGID 发送 SIGTERM
回收，GPU 从约 `8.1 GiB` 回到 `0 MiB`，没有终止用户服务。

以后不得用外层 timeout 探测会派生独立 session 的 controller。正式运行应直接由 nohup/tmux 托管，
同时监控 controller PID、child PID、terminal 和 resource.jsonl；确需中止时必须核实 process tree 后
显式回收 child process group。`r5/r6` 的 running terminal 保留为中断证据，不改写成 done。

### PIVOT-F23：SE(3) 一致性容差必须覆盖 float32 往返误差

M4 单帧 r1 的 actor transform 先由 checkpoint float32 tensor 变换，再写入 JSON 并读回，最大平移误差
略高于 `1e-6 m`；其余 15 项检查均通过。把该值当几何失败会制造假阴性。协议在查看正式全量结果前
固定为 `1e-4 m`，r2/r3 冒烟通过，正式 196 帧实测最大误差为
`3.814697265625e-06 m`，rotation/size/canonical drift 均为零。容差变更只反映数值精度，不降低
1 m 编辑幅度，也不得据此为真正的轨迹偏差放宽门禁。

## 7. 历史新路线启动前附加检查

- [ ] 是否明确说明该步骤直接服务于重建、编辑或可信评测，而不是重新做事件挖掘？
- [ ] AD-GS exact reproduction 是否已经通过冻结门禁？
- [ ] 是否把 upstream 原始结果与 compatibility patch 结果分开？
- [ ] 是否对 VAD-GS 等已公开的 visibility/completion 工作做 novelty 边界核对？
- [ ] 反事实无真值指标是否有真实 held-out/pseudo-hole 证据，而不是自洽规则？
- [ ] 是否同时评估目标区变化、非目标区保持、几何/时序一致性和下游感知？
- [ ] 遇到内存/GPU不足时是否按 `N1-F24/PIVOT-F05` 停机并等待授权？

## 8. V3 每个正式消融前检查

- [ ] 是否使用 V3 task ID、新 run 和冻结 config/source hash，而不是续写 V2 terminal？
- [ ] 是否保持 scene-0230/0242/0255、split、seed、相机、步数和 actor cohort 不变？
- [ ] 是否把原生 Affine/CamPose/LiDAR init 与新增实现分开？
- [ ] rolling-shutter 路径是否有真实 row timing；没有时是否显式 `not_supported`？
- [ ] actor-aware 变化是否只增加一个可归因因子，并保留 module-off 原生等价测试？
- [ ] 是否同时报告 actor/boundary 质量、GS 数、训练时间、VRAM 和 non-target 保持？
- [ ] local refinement 是否冻结 affected set 外参数，并只使用 Tier-A/多视图/LiDAR 可观测证据？
- [ ] expected/first-hit/measured depth 是否继续保持 typed separation？
- [ ] 工程 `blocked`、方法负结果 `rejected` 和任务完成 `done` 是否没有混写？
- [ ] 结论是否明确限制在三场景消融，不写成大规模泛化或闭环安全结论？
