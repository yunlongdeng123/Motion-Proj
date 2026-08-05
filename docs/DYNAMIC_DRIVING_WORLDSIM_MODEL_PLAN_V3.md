# 面向世界仿真的动态驾驶 3DGS 复现、模型增强与工程化计划 V3

- **版本**：V3
- **日期**：2026-08-05
- **项目根目录**：`/root/autodl-tmp/motion_proj`
- **执行环境**：AutoDL，单卡 NVIDIA GeForce RTX 3090 24 GiB，cgroup memory 90 GiB
- **V3 启动基线**：`research/dynamic-editing-v2@e691c1f`
- **当前任务**：`WS-V3-A0-NATIVE-BASELINE-01`（`running`）
- **唯一当前计划**：本文件
- **历史前序**：[`DYNAMIC_DRIVING_EDITING_DIAGNOSTIC_PLAN_V2.md`](DYNAMIC_DRIVING_EDITING_DIAGNOSTIC_PLAN_V2.md)

## 0. 权威边界

从 V3 生效起，本文件是唯一具有当前执行授权的研究计划。V2 保留为历史事实与实现来源，不再更新，也不再以
M5/M6 的大型评测和 novelty gate 驱动项目。V2 的 `M0–M4` 保持 `done`；M5 已执行但未闭环的训练、检查点、
诊断与 dirty worktree 保持为“冻结的部分证据”，不得追记为 `done`、`rejected` 或方法结论。

V3 的定位固定为：

> **面向世界仿真的动态驾驶 3DGS 复现、模型增强与工程化研究。**

本项目不承诺提出“大创新”，也不把身份绑定、基础轨迹编辑、场景图或评测框架包装成贡献。目标是完成一条
有技术判断、有实际模型改动、有完整消融并可交付的 WorldSim 模型链：

```text
多相机日志 / 位姿 / LiDAR / 实例标注
                ↓
校准、初始化与可审计数据合同
                ↓
静态背景 + 动态实例 + 天空 3DGS
                ↓
对象轨迹编辑
                ↓
受影响区域的局部 3D Gaussian 精修
                ↓
剪枝、精度压缩、分块 / LOD 与实时渲染
```

研究问题只保留两个：

1. 原生的统一梯度驱动增密/剪枝规则是否适合动态驾驶 actor；
2. 轨迹编辑后，只优化受影响的 3D Gaussian，能否改善去遮挡空洞、深度/透明度顺序和时间稳定性。

## 1. 已核实的起点

### 1.1 V2 可直接继承的资产

- DGGT 已完成 18/18 单视图、18/18 三视图推理，以及 common/regional 诊断；它只保留为前馈范式的历史
  对照，不再和逐场景优化方法做非等价排行榜。
- nuScenes `scene-0230/0242/0255` 已建立 persistent `instance_token`、raw 2 Hz 轨迹、三相机投影、
  actor cohort 和 4,356/4,356 exact token mappings。
- scene-0230 原生 StreetGS 30k checkpoint、actor registry 和 `original/lateral/delete` 全 196 帧闭环已完成；
  1,764 张 RGB、9 个视频和成对指标可复用。
- `motion_proj/resim/` 已有 WorldState、actor registry、轨迹编辑和 typed render 基础设施。V3 直接复用，
  不重写身份与编辑 API。
- V2 的 Tier A/B/C 去遮挡证据、held-out stride 和失败分层可以缩减为模型消融指标，不再扩张为独立
  benchmark。

### 1.2 V2 M5 冻结事实

| 场景 | 已有证据 | 缺口 |
|---|---|---|
| scene-0230 | held-out checkpoint `398,652,534` bytes；high/boundary actor 均有非空 Gaussian slice | 未完成 V2 四编辑全量压力测试 |
| scene-0242 | checkpoint `306,034,934` bytes，SHA-256 `16179d8f...c5fda`；high actor `6,939` GS | boundary actor 在 checkpoint 中不可用，必须 `ABSTAIN` |
| scene-0255 | raw/processed/sky 资产已推进到训练前后诊断 | 原生 `torch.cat` CUDA invalid configuration 阻塞完整 checkpoint |

scene-0255 的 r27 已把错误缩小到 DriveStudio `datasets/driving_dataset.py` 的实例点聚合：输入为 166 个
CUDA float32 tensor，其中 152 个是 `(0, 3)`，总计只有 177 个 scalar，且没有 OOM 证据。这是明确的
工程兼容性阻塞，不是 3DGS 方法失败。V3 A0 先用最小、可测试、可回退的兼容修复闭合它。

### 1.3 上游源码审计

DriveStudio 固定为 `e59bda4fa681f829dbb1d65f0de582b0f633c450`。已核实：

- `models.modules.AffineTransform` 已按 image embedding 输出 3×4 RGB affine，可选 pixel affine；
- `models.modules.CameraOptModule` 已学习每图像 3D 平移 + 6D 旋转残差；
- `datasets/driving_dataset.py` 已从 LiDAR 初始化背景和实例点、颜色与尺度；
- `models.nodes.RigidNodes.refinement_after` 对所有刚体 Gaussian 使用统一梯度、尺度、屏幕尺寸、透明度
  和越界规则，没有 actor support、边界、LiDAR 支持或重投影残差的差异化决策。

因此 A1 必须是“核实并增强已有校准”，不能把现有 `Affine/CamPose/LiDAR init` 重命名为新模块；A2 的
首要实现入口是 `RigidNodes` 的增密/剪枝控制链。

## 2. 事实源与研究边界

| 工作 | 已公开能力 | V3 的用法与不主张内容 |
|---|---|---|
| [Instant NuRec](https://research.nvidia.com/labs/sil/projects/instant-nurec/) | 短多相机日志到静态/动态/天空分层 Gaussian 和逐相机 ISP 的前馈重建 | 作为当前前馈范式上界；不再把 DGGT 当唯一前沿。先审计输入、权重与导出协议，不承诺直接替换主链 |
| [Instant NuRec 官方代码](https://github.com/NVIDIA/instant-nurec) | 开源推理/导出入口；standalone CLI 的公开导出能力与完整研究系统有边界 | 明示静态 PLY 导出限制，禁止把网页系统能力等同于本地 CLI 能力 |
| [OmniRe](https://arxiv.org/abs/2408.16760) / [IDSplat](https://arxiv.org/abs/2511.19235) | 动态场景图、实例分解、轨迹与 LiDAR 锚定 | 证明 identity/scene graph/basic edit 已拥挤；V3 只复用，不宣称发明 |
| [SplatAD](https://openaccess.thecvf.com/content/CVPR2025/html/Hess_SplatAD_Real-Time_Lidar_and_Camera_Rendering_with_3D_Gaussian_Splatting_CVPR_2025_paper.html) | 相机与 LiDAR 实时渲染、rolling shutter 与 LiDAR 特性 | 支持传感器建模动机；没有 readout timing 时不得伪造 rolling-shutter 实验 |
| [ADGaussian](https://arxiv.org/abs/2504.00437) | 泛化式、多模态驾驶 Gaussian 重建 | 只作为 generalizable reconstruction 方向参考，不做不等价数值排行 |
| [Real2Sim](https://arxiv.org/abs/2605.13591) / [RoVES](https://arxiv.org/abs/2605.25373) | 物理交互、道路几何和车辆动力学 | 定义未来接口；V3 不实现 MPM、车辆动力学或道路生成 |
| [RealityBridge](https://arxiv.org/abs/2606.16278) / [Difix3D+](https://arxiv.org/abs/2503.01774) | 生成式伪影、光照与长视频修复 | 只作为后续上界；A3 先做轻量 3D Gaussian 精修，不把 2D diffusion 作为主方法 |
| [Speedy-Splat](https://openaccess.thecvf.com/content/CVPR2025/html/Hanson_Speedy-Splat_Fast_3D_Gaussian_Splatting_with_Sparse_Pixels_and_Sparse_CVPR_2025_paper.html) | 稀疏像素/稀疏 primitive 的训练和渲染加速 | 为 A4 剪枝与稀疏化提供工程参照，不直接移植其结论 |

V3 不做 Occupancy→video 大模型，不恢复 cut-in 挖掘，不从零实现基础 scene graph，不把三场景结果外推为
大规模泛化结论，也不以“从未有人命名过”作为方法成立条件。

## 3. 固定实验矩阵

### 3.1 主消融

| 实验 | 固定含义 | 主要变化 |
|---|---|---|
| A0 | 原生 DriveStudio/StreetGS | 修复 scene-0255 工程兼容性；冻结三场景原生基线 |
| A1 | A0 + 成像/位姿校准增强 | 对已有 Affine/CamPose 做 off/native/enhanced 消融；有真实 timing 才做 rolling shutter |
| A2 | A1 + actor-aware densification/pruning | 对动态 actor 的分裂、复制、重置和剪枝使用实例感知统计 |
| A3 | A2 + local Gaussian refinement | 编辑后只重建受影响的局部 3D Gaussian 与时序窗口 |
| A4 | A3 + deployment optimization | contribution pruning、FP16/量化、chunk/LOD、资产注册与运行时加载 |

主表必须报告完整 A0→A4 链。若某一模块没有改善预注册主要端点，仍保留该行并写 `rejected` 或负结果；
最终推荐系统可以采用 Pareto 最优祖先，但不能从消融表中删除失败模块。

### 3.2 公平性合同

- 正式场景固定为 `scene-0230/0242/0255`，不得按结果删场景。
- 原始图像、相机集合、训练/held-out split、seed、优化步数、分辨率和 actor cohort 在 A0 冻结后不变。
- 调参只允许使用独立 smoke fixture 或 scene-0230 预留的 development frames；正式 held-out frames 不用于
  选择阈值。所有 A0–A4 共享最终冻结配置。
- 每场景至少评估 frozen high-support actor；boundary-support slice 不存在时写 `ABSTAIN`，不换更容易的
  actor，也不从均值分母静默删除。
- 保留 raw 2 Hz 轨迹与插值可视化的物理分离；nuScenes GT 只用于选择、诊断和 held-out 指标，不注入
  训练以制造 oracle 优势。
- 所有 formal run 使用唯一 task ID、timestamp、seed、source commit/config hash、资源记录和 terminal。

### 3.3 指标合同

指标服务模型消融，不建立新的大型 benchmark。

| 维度 | 必报指标 | 解释边界 |
|---|---|---|
| 原始重建 | held-out PSNR/SSIM/LPIPS；相机间 photometric residual；LiDAR/渲染 depth residual | global 指标为次要，必须同时报告动态区 |
| actor/边界 | actor mask PSNR/SSIM/LPIPS；固定宽度 boundary band；轮廓重投影误差；每 actor GS 数 | 不用单一全图均值掩盖小 actor |
| 编辑后 | source residual、Tier-A pseudo-hole 误差、depth ordering violation、hole/opacity coverage、temporal flicker | Tier B/C 分开，未知不得并入通过 |
| 保持性 | non-target region 差异、非目标参数 drift、registry/trajectory invariants | M4 的 93/95 dB 只证明硬局部编辑保持，不是画质 |
| 工程效率 | total/background/per-actor GS、训练 wall time、peak VRAM、peak cgroup RAM、checkpoint bytes、FPS | 质量、规模和速度必须组成 Pareto 报告 |

各阶段主要端点：

- A1：动态边界重投影误差与跨相机 photometric residual；
- A2：actor/boundary LPIPS 与 per-actor GS/训练代价 Pareto；
- A3：Tier-A hole error、depth-order violations 与 temporal flicker；
- A4：checkpoint bytes、峰值 VRAM、FPS，以及相对 A3 的质量下降。

所有其他指标是诊断项，不能在看到结果后替换主要端点。

## 4. 任务与执行顺序

```text
P0 路线切换与事实冻结
 → A0 三场景原生基线
 → A1 校准/初始化消融
 → A2 actor-aware 增密与剪枝
 → A3 编辑后局部 Gaussian 精修
 → A4 部署优化
 → R0 集成、复现包与结论
```

`F0` 前馈基线审计可在 A0 后与短 CPU/source audit 交错执行，但不得抢占 A1–A4 主链 GPU 预算。

| Task ID | 状态 | 交付物 | 完成门禁 |
|---|---|---|---|
| `WS-V3-P0-ROUTE-01` | done | V3 plan、状态/实验/失败/README 同步 | `076ebdc`；单一权威计划、V2 M5 冻结、链接与 Git 校验通过 |
| `WS-V3-A0-NATIVE-BASELINE-01` | running | 三场景原生 checkpoint、held-out render、actor registry、资源基线 | scene-0255 最小修复有回归测试；3/3 terminal；不丢场景 |
| `WS-V3-F0-FEEDFORWARD-AUDIT-01` | pending | Instant NuRec 本地可运行性/输入/输出/license 审计 | 官方 revision 固定；unsupported 能力明示；不要求 GPU 全量跑通 |
| `WS-V3-A1-CALIBRATION-01` | pending | off/native/enhanced 校准消融 | 三场景同协议；rolling shutter 有 metadata 或显式 `not_supported` |
| `WS-V3-A2-ACTOR-DENSIFY-01` | pending | actor-aware densification/pruning 模块与子消融 | 模块可关闭；完成 D0–D3；Gaussian/质量/代价齐全 |
| `WS-V3-A3-LOCAL-REFINE-01` | pending | 局部 Gaussian affected-set、短步优化与时序约束 | outside frozen；Tier-A/深度/时序指标齐全；无大视频 diffusion |
| `WS-V3-A4-DEPLOYMENT-01` | pending | pruning/precision/chunk/LOD、转换器、asset registry | pruning + 数值压缩 + chunk 完成；不变量和质量-大小-速度 Pareto 通过 |
| `WS-V3-R0-INTEGRATION-01` | pending | A0–A4 最终表、可复现命令、模型链文档 | 所有 terminal 和负结果可审计；结论不超出三场景证据 |

## 5. P0：路线切换与冻结

### 实施

1. 新建本计划，V2 计划保持原文件不改；
2. 在 STATUS/EXPERIMENTS 中把 V2 M5 登记为“部分执行后冻结”，列出已有 checkpoint、缺失产物和
   scene-0255 阻塞；
3. 把 M5 dirty files 标为保留资产，不在 P0 清理、覆盖或混入提交；
4. 在失败账本增加 V3 约束；
5. 更新根 README 和 docs 导航，使新对话只从 V3 恢复。

### 完成后下一步

P0 提交后只授权 `WS-V3-A0-NATIVE-BASELINE-01`。不得跳到 A2，也不得继续扩大 V2 M5 evaluator。

## 6. A0：三场景原生基线

### 6.1 scene-0255 最小兼容修复

先用 r27 固定的 tensor 形状构造 CUDA 回归测试。修复边界优先为：在聚合前过滤空 tensor，并在所有输入为空时
返回显式空 `(0, 3)` 或跳过该实例；不得改变非空点的数值、顺序、颜色配对或 instance ID。若过滤空 tensor
仍触发 upstream CUDA kernel 错误，才允许把该次小规模聚合转到 CPU 后一次性搬回 device，并记录额外 wall time。

修复必须放在 Motion-Proj compatibility layer 或可重放 patch 中；不得无备份地修改第三方源码。测试至少覆盖：

- mixed empty/non-empty CUDA tensors；
- all-empty instance；
- points/colors 长度与顺序一致；
- 0230/0242 既有聚合不变；
- 新容器 GPU smoke 与资源合同。

### 6.2 A0 冻结产物

- 三场景原生 StreetGS 配置、完整 config hash 与 source patch hash；
- 30k checkpoint 或经审计等价的既有 checkpoint；
- train/held-out render、per-scene/per-actor Gaussian inventory；
- Affine/CamPose 是否启用、参数量、残差分布；
- 训练时间、峰值 VRAM/RAM、checkpoint bytes 和渲染 FPS；
- high/boundary actor registry 及不可用原因。

已有 0230/0242 checkpoint 可在 hash、split、config 与实现一致时注册复用，不为形式完整而重训。scene-0255
必须在修复后创建新 formal run，旧 blocked run 不覆盖。

## 7. F0：Instant NuRec 前馈范式审计

本任务只回答“当前官方代码能否在本项目输入上形成可审计基线”。固定检查：

1. 官方仓库 revision、license、checkpoint provenance 和硬件要求；
2. 相机模型、图像 cadence、pose/LiDAR/实例输入与 nuScenes 三场景的映射；
3. 本地 CLI 实际导出的 static/dynamic/sky/ISP 能力；
4. 是否能保留 actor registry 与编辑接口；
5. 1-window smoke 的 wall time、VRAM 和输出 schema。

若权重、输入许可、动态层导出或算力不满足，任务可 `blocked`，但必须产出事实审计；这不阻塞 A1–A4。不得用
网页演示指标填充本地结果，也不得把 DGGT 与 Instant NuRec 混成同一模型。

## 8. A1：成像、位姿与初始化消融

### 8.1 消融单元

- `C0-off`：关闭原生 Affine/CamPose，仅作诊断下界；
- `C1-native`：原生 per-image Affine + 9-DoF CamPose，等同 A0；
- `C2-factorized-isp`：camera embedding + time/exposure embedding 的轻量分解，输出受限 RGB affine；
- `C3-bounded-pose`：在 C2 上对平移/旋转残差加幅度先验与时间平滑；
- `C4-rolling-shutter`：仅在 processed data 含可验证 readout direction/time 时加入 row-time pose interpolation，
  否则状态固定为 `not_supported`，不伪造 timing。

A1 主行是 C2/C3/C4 中预注册主要端点的 Pareto 选择，但 C0/C1/C2/C3 的结果全部保留。

### 8.2 LiDAR 初始化边界

原生背景和实例已经由 LiDAR 初始化。A1 不重写点云预处理；只审计并落盘每个 Gaussian/actor 的 LiDAR
support provenance、初始点密度、可见帧数和深度 residual，供 A2 使用。若增加地面/动态框过滤，必须作为
独立 `L1` 子消融，不能与 ISP/pose 同时启用后归因。

## 9. A2：实例感知的动态 Gaussian 增密与剪枝

### 9.1 最小模型改动

对每个 rigid Gaussian (i) 维护可审计统计：

- 平均屏幕空间梯度 `grad_i` 与观察次数 `vis_i`；
- 所属 `actor_id`、actor 可见帧数和速度区间；
- 固定 mask boundary band 内的贡献率 `boundary_i`；
- 训练帧重投影/光度 residual `residual_i`；
- 初始化 LiDAR support 或最近点距离 `lidar_i`。

统计先在 actor 内做 robust normalization，再用于分裂/复制候选排序和剪枝保护。所有权重、quota、阈值和
normalization 写入配置；模块关闭时必须逐位退化为 native `RigidNodes` 行为。

### 9.2 子消融

| 子实验 | 改动 |
|---|---|
| D0 | 原生统一阈值 |
| D1 | actor/background 分离阈值 + 每 actor 最小/最大 Gaussian quota |
| D2 | D1 + boundary/reprojection residual 加权候选排序 |
| D3 | D2 + LiDAR support/visibility-aware pruning protection |

只先实现 D1，smoke 和同预算单场景通过后再加 D2；D3 只在 LiDAR support 字段经过 A1 provenance 审计后实现。
禁止把 D1–D3 一次性合并为不可归因模块。

### 9.3 判定

- 质量比较同时给 fixed-step 与 matched-Gaussian-budget 两种视图；
- 主要判断使用 actor/boundary LPIPS 与 per-actor GS、wall time 的 Pareto，不以更多 GS 自动算提升；
- 若 D1–D3 都不能改善 Pareto，A2 记 `rejected`，结论为当前原生梯度规则在本协议下更合适；A3 仍在固定
  A2 产物上运行，以保持主消融完整，最终系统可回退到 A1。

## 10. A3：编辑区域局部 Gaussian 精修

### 10.1 affected set

对 lateral/delete 编辑，以 source footprint、edited footprint、深度前后关系和固定像素 dilation 构造受影响视锥；
选择：

- 目标 actor Gaussian；
- source footprint 后方、历史上被遮挡但在其他帧/相机可见的 static Gaussian；
- 与局部 hole 相交且有 LiDAR/多视图支持的 static Gaussian。

affected set 外的 Gaussian 参数和 optimizer state 全部冻结，并用 hash/drift 测试验证。

### 10.2 短步精修

按独立开关依次增加：

1. `R1-reactivate`：恢复局部低 opacity 静态 Gaussian，优化 opacity/scale；
2. `R2-appearance`：在可见证据支持下优化局部 color/SH；
3. `R3-hole-seed`：只从 LiDAR 或多视图三角化支持位置补局部 Gaussian，不从单张 RGB 幻觉 3D 点；
4. `R4-temporal`：相邻帧共享的静态 Gaussian 加颜色/opacity/scale 一致性，动态 actor 保留轨迹平滑约束。

loss 只使用未被编辑 actor 污染的可观测像素、typed depth 与现有多视图证据。Tier B/C 不作为伪真值回传。
固定短步上限和局部 Gaussian 上限在 smoke 后预注册；不得用整场景重训冒充 local refinement。

### 10.3 安全边界

- 不用 hard composition 的 outside=0 作为成功指标；
- 不把原图中仍有 actor 的像素当删除后的背景真值；
- expected depth、first-hit depth 与 measured LiDAR depth 分开；
- 不接大型视频 diffusion；需要生成式上界时另开独立后续任务。

## 11. A4：部署优化

### 11.1 顺序

1. `P1-contribution-prune`：按跨训练/held-out 视图贡献、opacity、visibility 做可回退剪枝；
2. `P2-fp16`：验证 FP16 参数/渲染路径；量化只在逐字段误差可审计时进入；
3. `P3-chunk`：按空间块 + actor registry 拆分 checkpoint，静态块与动态 actor 资产分离；
4. `P4-lod`：按距离/投影大小选择层级，并支持静态 chunk 与动态 actor 加载/卸载；
5. `P5-registry`：实现 checkpoint converter、asset manifest、bytes/hash/schema 与 reload smoke。

### 11.2 必须保持的不变量

- actor `instance_token → model/slice/asset` 可追踪；
- original/lateral/delete 在转换后仍可执行；
- 非目标参数 drift、轨迹和相机标定不因转换改变；
- chunk 边界不产生不可接受的裂缝/闪烁；
- 每个性能数字同时写分辨率、相机数、硬件、warm-up、同步方式和统计窗口。

量化、chunk 和 LOD 是三个独立子消融。若量化或 LOD 不支持或质量损失过大，保持 `rejected` 结果，不用
未实现项填充 A4。A4 的最低完成集是 contribution pruning + FP16 数值压缩 + spatial chunk + asset
registry；整数/低比特量化和 LOD 是在该最低集之上的独立实验。

## 12. 资源、提交与停止规则

### 12.1 资源合同

- 新容器/新 GPU 先执行 torch/CUDA/renderer smoke；旧实例日志不能复制为当前 PASS。
- cgroup memory 连续两次达到 90% 时停止当前 stage，保留 checkpoint/terminal，不伪装 OOM 或方法失败。
- 正式训练、评测和视频渲染分 stage；累积 render 失败不能覆盖已完成 checkpoint 的事实。
- 只管理本任务创建且已核实 PID/PGID 的进程；不得终止用户 Cursor/Jupyter/TensorBoard 服务。
- 大型数据、env、checkpoint 和 run 只放 `/root/autodl-tmp`。

### 12.2 提交合同

每个 P0/A0/A1/A2/A3/A4/R0 至少一个独立 Conventional Commit。提交前：

1. `git status --short`，确认没有混入其他 task 的 dirty files；
2. 定向测试、配置解析、`git diff --check`；
3. 检查 staged diff、run terminal、config/source hash；
4. 同步本计划、STATUS、EXPERIMENTS、FAILURES；
5. 提交信息正文写 Task ID、测试、正式证据和下一步。

### 12.3 停止与扩展

- 三场景只支持模型消融和工程结论，不支持 trainval/夜间/长时/复杂交互泛化主张。
- 只有 A2 或 A3 在至少 2/3 场景的主要端点方向一致、无关键保持性退化且资源稳定，才讨论扩展到 6 场景；
  扩场景不是 R0 完成条件。
- 任何新 physics、occupancy、diffusion、downstream perception 或 closed-loop 安全主张都需要新的计划和用户授权。
- 不因“提升不显著”更换指标、actor 或场景；负结果是合法交付物。

## 13. 恢复顺序

清空上下文或换实例后，依次读取：

1. `AGENTS.md`；
2. [`RESEARCH_STATUS.md`](RESEARCH_STATUS.md)；
3. [`RESEARCH_FAILURES.md`](RESEARCH_FAILURES.md)；
4. [`EXPERIMENTS.md`](EXPERIMENTS.md)；
5. 本计划；
6. 当前 task 的 formal run terminal、Git status 和资源状态。

不得从 V2 计划、归档报告或旧 tmux terminal 恢复“下一步”。当前动作永远以 STATUS 和本计划的任务表为准。

## 14. 更新记录

### 2026-08-05 — `WS-V3-P0-ROUTE-01` done

- 用户授权从 V2 大型失败诊断路线切换到 WorldSim 模型链与 A0–A4 消融；
- 核实 V2 M0–M4 完成、M5 部分执行、scene-0255 空 CUDA tensor 聚合阻塞和 dirty worktree；
- 核实 DriveStudio 已有 Affine/CamPose/LiDAR init，原生 RigidNodes 仍为统一增密/剪枝；
- 核对 Instant NuRec、OmniRe、IDSplat、SplatAD、ADGaussian、Real2Sim、RoVES、RealityBridge、
  Difix3D+ 与 Speedy-Splat 的一手事实源；
- 创建 V3 唯一权威计划，Markdown local links 与 `git diff --check` 通过；
- 路线注册提交：`076ebdc`（`docs(worldsim): 注册 V3 模型增强路线`）；
- 当前任务切换为 `WS-V3-A0-NATIVE-BASELINE-01`，先固化 scene-0255 空 CUDA tensor 回归。
