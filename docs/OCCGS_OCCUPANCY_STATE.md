# OccGS O0 — Occupancy / Panoptic Anchor

- created_at_utc: 2026-07-21T19:30Z
- gate: `O0-OCC-04`
- 结论：**O0 PASS**（scene-local LiDAR+box occupancy；unknown 保留；动态/静态可分；不依赖 Occ3D）

---

## 1. 设计

按 V7 §8.1 优先级 1：**不训练 occupancy 模型**，不下载 Occ3D。

- 网格：ego/lidar 系 per-frame；range [-40,40]×[-40,40]×[-1,5.4] m；voxel 0.4 m → 200×200×16
  （与 Occ3D-nuScenes 对齐，便于后续可选对照）。
- 语义：`0=unknown, 1=free, 2=static_occ, 3=dynamic_vehicle`
- **unknown 必须保留**：从未被 LiDAR ray 命中的体素保持 unknown，不标 free。
- free：沿 LiDAR 射线粗 DDA 雕刻至首个 occupied。
- static：LiDAR 命中（非 dynamic）。
- dynamic：3D vehicle box AABB 光栅化 + instance_id。

脚本：`motion_proj/occupancy/build_scene_occupancy.py`  
产物：`data/occgs/occupancy/{003,004,005}/frame_XXX.npz` + `meta.json` + `summary.json`  
BEV 可视化：`data/occgs/reviews/o0_occupancy/bev_*_t040.png`  
Sanity：`data/occgs/occupancy/o0_sanity_v1.json`

## 2. 定量（8s 窗口均值比例）

| scene | unk | free | static | dyn | unknown_preserved |
|---|---|---|---|---|---|
| 003 S0 | 0.961 | 0.024 | 0.005 | 0.010 | True |
| 004 S2 | 0.958 | 0.020 | 0.005 | 0.018 | True |
| 005 S1 | ~0.96 | ~0.03 | ~0.005 | ~0.01 | True |

## 3. 与 Gaussian 的绑定（轻量）

- 每个 vehicle actor 的 canonical occupancy = 其 3D box 体素（本阶段）；Gaussian node 绑定通过
  DriveStudio `RigidNodes` instance index ↔ `instances_info.json` key（经 `instances_true_id`）。
- 编辑时：pose 变换 → Gaussian `instances_trans/quats` 变换 → instance occupancy 同步变换
  （C0 渲染路径已实现前者；occupancy 同步在 S0 编辑 JSON 的 frame 级 `obj_to_world` 上可重跑
  `rasterize_box`）。
- 未下载 Occ3D；不使用 future occupancy 作为生成条件。

## 4. O0 Gate（V7 §8.7）

| 条件 | 结果 |
|---|---|
| collision/free-space 错误可度量 | PASS（free/static/dyn 分离；S0 编辑器用 occupancy 外的几何约束证书） |
| unknown 不被误当 free | PASS（unk 比例 > 0.9） |
| actor render 无明显退化 | PASS（O0 不改重建；B0 指标保持） |
| 不依赖 Occ3D future 信息 | PASS |

**结论：O0 PASS，进入 S0/C0。**
