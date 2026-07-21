# OccGS D0 — nuScenes 三场景数据准备

- created_at_utc: 2026-07-21T17:45Z
- gate: `D0-DATA-02`
- 结论：**D0 PASS**（完整性 + 磁盘门槛均过；选择标准在查看训练结果前冻结）

---

## 1. 约束与场景池

G0 本地 raw 审计：`/root/autodl-tmp/data/nuscenes` 仅 **v1.0-mini 的 10 个 scene** 具备
前向 3 相机 + LIDAR_TOP 完整 sweep。10Hz 插值依赖 sweep → D0 场景池锁定为 mini。

预处理：

```bash
python datasets/preprocess.py \
  --data_root /root/autodl-tmp/data/nuscenes \
  --dataset nuscenes --split v1.0-mini \
  --target_dir /root/autodl-tmp/data/occgs/processed \
  --scene_ids 3 4 5 --interpolate_N 4 --workers 3 \
  --process_keys images lidar calib dynamic_masks objects objects_vis
```

输出根：`/root/autodl-tmp/data/occgs/processed_10Hz/mini/{003,004,005}/`
（DriveStudio 会把 `processed` 自动改写为 `processed_10Hz`）。

## 2. 冻结选择标准（v2，训练前）

脚本：`motion_proj/resim/d0_scene_scan.py`  
扫描产物：`data/occgs/scene_specs/d0_scene_scan_v1.json`  
冻结清单：`data/occgs/scene_specs/d0_frozen_picks_v2.json`

| Role | mini idx | nuScenes name | 关键统计 | 理由 |
|---|---|---|---|---|
| **S0** static-heavy | 003 | scene-0655 | dist=163m, yaw=11°, moving=7, night=False | 小 yaw + 足够视差 + 最少动态车 |
| **S1** vehicle-dynamic | 005 | scene-0796 | dist=237m, yaw=8.6°, persistent_moving=22 | 最多持续可见运动车辆 |
| **S2** cut-in/merge | 004 | scene-0757 | dist=27m, yaw=12°, cutin_score=49.7, persistent=15 | 日间最高 cut-in 代理分（排除零视差 scene-0553） |

Cut-in 代理：前向窗口 `0<x<40m` 内横向扫幅 `|Δy|>1.5m` 且连续可见 ≥8 keyframes。

v1 选择器曾因“只最小化 moving”选中 yaw=218° 的 scene-0916，违反计划 §6.3 “小 ego rotation”；
已在训练前修正为 v2 并重新冻结。

## 3. 时间窗与相机

- 处理频率：10 Hz（`interpolate_N=4`）；约 196–201 帧 / scene（全长 ~20s）。
- **训练时间窗**：`start_timestep=0, end_timestep=79` → 前 8 秒（符合 V7 §6.4）。
- **训练相机**：`cameras=[0,1,2]` = CAM_FRONT / CAM_FRONT_LEFT / CAM_FRONT_RIGHT
  （`configs/datasets/nuscenes/3cams.yaml`）。预处理仍落盘 6 相机（DriveStudio 硬编码），
  训练只读前向 3 相机。
- 插值 box provenance：一律视为 `interpolated`，不冒充人工 GT（V7 §6.2）。

## 4. Sky mask（§5.5 替代路径）

官方 mmseg/SegFormer 栈与适配后的 torch 2.1.2 不兼容，按计划启用现代替代：

- 脚本：`motion_proj/resim/d0_sky_mask_segformer.py`
- 模型：`nvidia/segformer-b5-finetuned-cityscapes-1024-1024`（transformers，HF mirror）
- Cityscapes class 10 = sky；输出 PNG `{0,255}` 至 `sky_masks/{t:03d}_{cam}.png`
- 仅生成前向 3 相机（与训练相机一致）：003/004 各 603 张，005 588 张
- 版本化：本文件 + 脚本即为替代 mask 的版本记录；**未静默改变 baseline**

粗 dynamic mask 由预处理 bbox 投影生成（`dynamic_masks/{all,human,vehicle}`）；
fine_dynamic_masks 本阶段不生成（StreetGS 可用粗 mask）。

## 5. 完整性（`d0_integrity_v1.json`）

| scene | frames | front3 imgs | sky | lidar | veh inst | cont≥4s | dyn cov | sky cov | disk | gate |
|---|---|---|---|---|---|---|---|---|---|---|
| 003 S0 | 201 | 603 | 603 | 201 | 110 | 91 | 0.061 | 0.179 | 439 MB | OK |
| 005 S1 | 196 | 588 | 588 | 196 | 39 | 26 | 0.009 | 0.053 | 541 MB | OK |
| 004 S2 | 201 | 603 | 603 | 201 | 22 | 19 | 0.067 | 0.066 | 452 MB | OK |

- 时间戳单调：三 scene 均 True
- S1/S2 `n_vehicle_cont4s ≥ 2`：通过
- 可视化面板（12 时刻 × 3 相机，sky=蓝 / vehicle_dyn=红）：
  `data/occgs/reviews/d0_integrity/{S0_003,S1_005,S2_004}_panel.png`

## 6. 磁盘

- 写盘前 avail ≈ 36 GiB；处理后 ≈ 33 GiB（逼近 30 GiB 门槛）。
- 清理动作（可重建、属已关闭 V6 路线）：删除
  `/root/autodl-tmp/third_party/ReSim/checkpoints/CogVideoX-2b-sat`（37 GiB）及 pip/conda/sandbox cache。
  ReSim **源码保留**；权重可按 `C1_V6_FINAL_REPORT.md` / 原下载脚本重建。正式 run 的
  manifest/metrics/reviews **未动**。
- 清理后 avail ≈ **70 GiB**，满足 B0 写盘门槛。

## 7. D0 Gate 核对（V7 §6.8）

| 条件 | 结果 |
|---|---|
| 每 scene 3 相机数据完整 | PASS |
| 时间窗有效（≥8s @10Hz） | PASS（~20s 全长，训练裁 8s） |
| 至少 1 个 vehicle 连续轨迹 | PASS |
| S1/S2 ≥2 moving actors | PASS |
| mask 与 box 抽查可视化 | PASS（面板已落盘） |
| 处理后保留 ≥30 GiB | PASS（清理后 70 GiB） |

**结论：D0 PASS，进入 B0。**
