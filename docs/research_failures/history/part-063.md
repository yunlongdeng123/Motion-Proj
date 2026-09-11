# 历史原始记录 063

> 冻结历史，不是当前执行规则。当前规则见 [scaling law](../../../auto-research_scaling_law.md) 与 [AGENTS](../../../AGENTS.md)。
> 原文件：`docs/RESEARCH_FAILURES.md`；迁移前提交 `abbd431c`。原有相对链接按原目录解释，证据路径原样保留。
> 按索引读取相关小节；旧哈希、过度门控和资源指令均不重新生效。

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
- `V3-F17`：Gaussian ancestry、counterfactual footprint 和未提交 V2 M5 产物都不能自动升级为 A3 真值。
  ancestry 只证明来源，paired mask 只是模型诊断；A3 必须使用已提交输入和 typed support，S-C 保持 unsupported。

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

### V3-F17：来源账本与 paired mask 不等于局部精修监督真值

D2 final checkpoint 的 Background ancestry 完整对齐 `1,205,164` 个 Gaussian，其中 `240,528` 个
`init_source=LIDAR` direct roots，其余含 random、split 与 clone；`nearest_lidar_distance` 对部分 lineage 有限，
但它是出生/父子来源记录，不是当前 target ray 的 T0 measured depth。A3 只能把 calibrated LiDAR projection 的
`depth_lidar_measured` 当 T0，把 first-hit 当 T1 ordering，把 expected depth 保持 diagnostic。

同样，source/edited footprint 来自同一 checkpoint 的 paired RGB difference，只能定位干预区域，不能充当真值
segmentation 或删除后的背景 RGB。S-A RGB 监督必须来自排除 target view 的 alternate camera/time 真实观测并有
calibrated reprojection；S-B 只使用 measured LiDAR 或至少两视图 geometry，禁止 RGB loss；S-C 不更新、不 seed、
不进入 loss，只报告 coverage/uncertainty/ABSTAIN。

当前工作树中的 V2 M5 protocol、`stress_metrics.py` 和 stress runner 均未提交且属于被冻结的用户工作，A3 不得
通过 import 或复制其结果建立隐式依赖。只能复用已提交并按 SHA 冻结的 M4 edit、paired mask、typed-depth 与
registry 接口；否则无法形成 clean source commit，也会把 V2 未闭环事实倒写成 V3 证据。

### V3-F18：A3 工程可重放不等于局部精修可晋级

A3 R1 已证明四个 S-B/T0 unit 的 opacity/scale 更新可以逐位重放，但可变集合只有 `51` 个 Background rows、
四个 unit 合计只有 `8` 个 T0 geometry pixels，S-A/RGB 为 `0/ABSTAIN`。heldout r2/r4/r5 的单 view 峰值稳定在
`14,241–14,245 MiB`，超过结果前冻结的 `12,288 MiB` ceiling；r5 的资源无效 diagnostic 同时出现 depth-order
改善和 non-target/original-global RGB MSE 严格退化，exact Pareto 为 `tradeoff_non_dominated`。

因此后续若复开局部精修，研究变量必须先变为“可观测支持如何获得、分层和拒绝”，而不是继续调 R1 的 step、
LR、alpha、mask dilation 或旧 renderer。合法复开需要新任务、新协议、与 heldout 隔离的支持审计，以及冻结前
证明 S-A 或更充分 T0/多视图证据确实存在；否则保持 `A3*=R0/D2 exact alias`。

