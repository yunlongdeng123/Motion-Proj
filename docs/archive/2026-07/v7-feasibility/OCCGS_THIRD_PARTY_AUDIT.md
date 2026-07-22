# OccGS G0 — 第三方审计（DriveStudio / SplatAD / Occ3D）

- created_at_utc: 2026-07-21T16:58Z
- gate: `G0-THIRDPARTY-00`
- 审计方式：本地仓库一手代码审计（DriveStudio，由 explore subagent 完成）+ 官方页面核查（SplatAD / Occ3D）
- 结论：**G0 PASS**（详见文末 Gate 核对表）

---

## 1. DriveStudio（主框架）

- 本地路径：`/root/autodl-tmp/third_party/drivestudio`
- 固定 commit：`e59bda4fa681f829dbb1d65f0de582b0f633c450`（2025-08-27，与 E0 manifest 一致）
- License：**MIT**（`LICENSE` 全文核对，允许学术使用/修改）

### 1.1 nuScenes 预处理（datasets/preprocess.py）

- CLI：`--data_root --dataset nuscenes --split --target_dir --workers --scene_ids/--split_file/--start_idx+--num_scenes --interpolate_N --process_keys`
- **scene 限定：可行**。`--scene_ids` 优先，其次 `--split_file`，再次 `--start_idx + --num_scenes`。
- **相机限定：预处理阶段不可行**（`NuScenesProcessor.cam_list` 硬编码 6 相机，`datasets/nuscenes/nuscenes_preprocess.py` L135-143）；
  训练阶段用 `data.pixel_source.cameras=[0,1,2]`（`configs/datasets/nuscenes/3cams.yaml`）选前向 3 相机。
  → 磁盘影响：预处理会落 6 相机图像，预算按 6 相机估算，训练只读 3 相机。
- `--interpolate_N 4`：keyframe 2Hz → (4+1)×2 = **10Hz**；输出目录自动改名 `processed_10Hz/`；代码 `assert N<=4`。
- `process_keys`：`images, calib, lidar, dynamic_masks, objects(, objects_vis)`；nuScenes 处理器不消费 `pose`
  （相机位姿在 `extrinsics/`，LiDAR 位姿在 `lidar_pose/`）。
- 输出结构（每 scene `{idx:03d}/`）：`images/ sky_masks/ extrinsics/ intrinsics/ lidar/ lidar_pose/
  dynamic_masks/{all,human,vehicle}/ instances/{instances_info.json,frame_instances.json}`。
  注意：文档写 `objects/`，代码实际是 `instances/`。
- sky mask：**必需**，由 `datasets/tools/extract_masks.py` + SegFormer
  （`segformer.b5.1024x1024.city.160k.pth`，mmseg 栈）生成；fine dynamic mask 可选（同脚本 `--process_dynamic_mask`）。
- split：`--split` 直接传 `NuScenes(version=...)`，`v1.0-mini` / `v1.0-trainval` 均可。

### 1.2 训练 / 渲染

- 入口：`tools/train.py --config_file configs/streetgs.yaml dataset=nuscenes/3cams data.scene_idx=... data.start_timestep=... data.end_timestep=...`
- Street-Gaussians-style = `configs/streetgs.yaml`（Background + RigidNodes + Sky/Affine/CamPose，**无 SMPL**）；
  OmniRe = `configs/omnire.yaml`；450+ 图像建议 `omnire_extended_cam.yaml`。
- **单卡默认**：`train.py` 无 DDP 启动逻辑；`num_iters` 默认 30000；checkpoint 每 15000 步 + final，
  `--resume_from` 可恢复；`tools/eval.py` 提供 render-only。
- 渲染输出：`rgb / depth / opacity`（+ 按类分解 `Background_* / RigidNodes_* / Dynamic_*`）。
  **无 semantic 渲染通道**——C0 所需 semantic/instance mask 须经由 per-instance gaussian 渲染
  （`get_instance_activated_gs_dict` / `point_ids` mask）自行组合，为 O0/C0 已知工作量。

### 1.3 Actor 编辑接口（C0 切入点）

- per-frame actor pose：数据侧 `instances_pose (num_frames, num_instances, 4, 4)` ←
  `instances_info.json.obj_to_world`；模型侧 `RigidNodes.instances_quats / instances_trans`（nn.Parameter）。
- 渲染时经 `transform_means/transform_quats` 应用（`models/nodes/rigid.py` L315-384）。
- 编辑 = 直接改写 `instances_trans/quats` 后 forward；已有 `remove_instances` / `replace_instances` API。
- **无官方 edit/trajectory 脚本**（tools/ 仅 train.py/eval.py）——编辑器由本项目实现（S0/C0 范围内，符合预期）。

## 2. SplatAD / neurad-studio（仅接口参考，不安装）

- `carlinds/splatad`：**Apache-2.0**；是 gsplat 的 fork，提供 `rasterization`（相机）与
  `lidar_rasterization`（LiDAR 球坐标 tile rasterizer + rolling shutter），入口 `gsplat/rendering.py`。
- 完整模型（dataloader/decoder）在 `georghess/neurad-studio`（Apache-2.0，nerfstudio 系）。
- 第一阶段用途：多传感器 renderer 的 API 参考与后续 LiDAR 输出基线；**不下载 PandaSet、不另起训练路线**。

## 3. Occ3D-nuScenes（仅格式审计，暂不下载）

- License：**MIT**（原始 nuScenes 为 CC BY-NC-SA 4.0，学术使用无冲突）。
- 格式：每 keyframe 一个 `gts/[scene_name]/[frame_token]/labels.npz`，含 `semantics`（200×200×16，
  0.4m voxel，range [-40,-40,-1, 40,40,5.4]，18 类含 free=17）、`mask_lidar`、`mask_camera`。
- 体积：全量 40k 帧（Google Drive 托管）；**按 scene 子集下载可行**（gts 按 scene_name 目录组织）。
  单帧 labels.npz 压缩后 ~数百 KB，3 scene × ~40 keyframe ≈ 数十 MB，可接受。
- 决策：O0 主路仍是 LiDAR+map+box 自建 scene-local occupancy（V7 §8.1 优先级 1）；
  Occ3D 子集仅作可选验证，且只在 O0 需要时下载对应 3 个 scene 的 gts。

## 4. 本地 raw nuScenes 覆盖审计（影响 D0 选 scene）

- `/root/autodl-tmp/data/nuscenes`：35 GB；`v1.0-mini`（10 scenes）与 `v1.0-trainval`
  （850 scenes 元数据）并存，samples/（keyframe）34,149 帧/相机通道完整。
- **sweep（非 keyframe）覆盖**：逐文件核查结果——只有 v1.0-mini 的 10 个 scene
  （0061/0103/0553/0655/0757/0796/0916/1077/1094/1100）具备前向 3 相机 + LIDAR_TOP 的完整 sweep；
  其余 trainval scene sweep 缺失（仅少量 CAM_FRONT 残留）。
- 10Hz 插值需要 sweep 相机图像 → **D0 的 S0/S1/S2 必须从这 10 个 mini scene 中选取**，
  预处理 `--split v1.0-mini`。此约束已在选择标准冻结前记录。

## 5. G0 Gate 核对表（V7 §4.4）

| 条件 | 结果 |
|---|---|
| DriveStudio license 允许学术研究 | PASS（MIT） |
| nuScenes processing 可限定 scene/camera/time range | PASS（scene=预处理 CLI；camera/time=训练 config） |
| 单卡可配置 | PASS（默认单卡，无 DDP 强制；E0 smoke 已过） |
| 无需提前下载另一完整数据集 | PASS（本地 nuScenes 足够；Occ3D 仅可选 scene 子集） |
| 不要求 future actor 状态输入自由生成模型 | PASS（重建-编辑-重渲染路线，无自由生成条件） |

**结论：G0 PASS，进入 D0。** 附加风险记录：
1. semantic/instance 渲染需自行实现（O0/C0 工作量，非 blocker）；
2. sky mask 依赖 mmseg/SegFormer 旧栈，若安装失败按 V7 §5.5 用现代分割模型替代（单独版本化）；
3. D0 场景池被 sweep 覆盖约束在 10 个 mini scene。
