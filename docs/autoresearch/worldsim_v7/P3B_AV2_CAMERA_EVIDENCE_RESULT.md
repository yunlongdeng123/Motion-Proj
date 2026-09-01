# V7 P3-B AV2 Camera Evidence Result

## Canonical result

- run：`run://worldsim_v7/WS-V7-P3B-AV2-CAMERA-EVIDENCE-01/20260902T150000Z__camera-evidence-s0-r1`；
- verdict：`supported_frozen_av2_camera_evidence_package`；
- outputs：10 logs，30 frozen panels，8 main cases，30 supplement cases，30 MP4；
- resources：43.380s，GPU 0.066319GiB，RSS 1.339GiB，46MiB。
- paper：首个 frozen case 进入 Figure 2；主稿 5 pages / 928,786 bytes，PDF 视觉检查无 overlap/clipping。

## Camera coverage

相机只由 query-point visibility 与固定 camera order 决定，选完才解码 RGB。30 cases 无 zero-visible：visible query
points min/median=`14/82`，visibility fraction min/median=`.727829/1.0`，每 crop 稀疏 camera-depth points 最少 `870`。
8 个 main panels 均完成视觉检查；未因质量、遮挡、hazard 或方法结果替换。

## Exposed boundary

第 8 main case 的 Chamfer 为 `.527→.967m`，作为预注册失败例保留。全量 P3-A 634 Actors 的 secondary tail read：

- Chamfer non-worse/worse=`525/109`（`82.81/17.19%`）；
- Chamfer ratio median/p90/p95/max=`.7569/1.0795/1.2034/1.8725`；
- LiDAR depth worsened=`271/634`；
- ray consistency worsened=`253/634`；
- hazard Actor Chamfer worsened=`5/230`。

这不撤销预冻结 aggregate verdict，但明确否定 universal per-Actor improvement。P3 的 direct-hit PROJECT 证书只覆盖
paired matched-ray free-space，不自动覆盖组合 surface 的 target geometry。

## Research response

下一阶段迁移 SelectiveNet 的 risk--coverage/reject interface，并参考 Conformal Risk Control 定义 monotone repair loss。
低容量 validity head 只读取 method-visible geometry/provenance，hazard head 只读取 dynamics/interaction；train、label、
calibration、threshold selection 均限 nuScenes。由于 nuScenes 与 AV2 的 cross-dataset shift 不满足已知 exchangeability，
AV2 只报告冻结 zero-shot risk--coverage 与 failure retention，不主张 distribution-free conformal guarantee。
