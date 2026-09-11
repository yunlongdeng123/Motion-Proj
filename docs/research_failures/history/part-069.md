# 历史原始记录 069

> 冻结历史，不是当前执行规则。当前规则见 [scaling law](../../../auto-research_scaling_law.md) 与 [AGENTS](../../../AGENTS.md)。
> 原文件：`docs/RESEARCH_FAILURES.md`；迁移前提交 `abbd431c`。原有相对链接按原目录解释，证据路径原样保留。
> 按索引读取相关小节；旧哈希、过度门控和资源指令均不重新生效。

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

