# WorldSim V4 一手文献与 Baseline 矩阵

- 日期：2026-08-11
- Task：`WS-V4-P0-SCOPE-PAPER-FREEZE-01`
- 状态：`done`
- 原则：只把论文官方页、CVF Open Access、arXiv 作者记录或作者官方仓库作为事实源

## 1. Driving reconstruction / simulation

| 工作 | 一手来源 | 与 V4 的关系 | V4 执行裁决 |
|---|---|---|---|
| Street Gaussians | [arXiv 2401.01339](https://arxiv.org/abs/2401.01339) | 动态 foreground/background 3DGS 与编辑 base | Tier A；复用已冻结 DriveStudio/StreetGS 资产 |
| DrivingGaussian | [CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Zhou_DrivingGaussian_Composite_Gaussian_Splatting_for_Surrounding_Dynamic_Autonomous_Driving_Scenes_CVPR_2024_paper.html) | 多相机 composite dynamic Gaussian graph | paper comparison；不因存在项目页自动进入 matched 表 |
| NeuRAD | [CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Tonderski_NeuRAD_Neural_Rendering_for_Autonomous_Driving_CVPR_2024_paper.html) | camera+LiDAR sensor rendering、跨数据集 | 官方源码存在；本轮不安装，不是默认 3DGS baseline |
| HUGS | [CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Zhou_HUGS_Holistic_Urban_3D_Scene_Understanding_via_Gaussian_Splatting_CVPR_2024_paper.html) | semantics/motion/geometry 联合优化 | paper comparison；若未来 adapter 单卡可行再单独预注册 |
| AD-GS | [arXiv 2507.12137](https://arxiv.org/abs/2507.12137) | locality-aware B-spline motion、visibility | Tier A dynamic baseline；同 scene/split/resolution，不作为 delta baseline |
| SplatAD | [CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Hess_SplatAD_Real-Time_Lidar_and_Camera_Rendering_with_3D_Gaussian_Splatting_CVPR_2025_paper.html) | camera+LiDAR real-time 3DGS | Tier B；adapter/单卡 preflight 前为 `blocked_baseline_adapter` |
| IDSplat | [官方仓库](https://github.com/zenseact/idsplat) | instance-decomposed driving 3DGS | Tier B；只有 adapter 成本合理且单卡可跑才进表 |

## 2. Closed-loop / editing

| 工作 | 一手来源 | 与 V4 的关系 | V4 执行裁决 |
|---|---|---|---|
| UniSim | [CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Yang_UniSim_A_Neural_Closed-Loop_Sensor_Simulator_CVPR_2023_paper.html) | editable closed-loop sensor simulation | related work；协议和表示不同，不填非 matched 数值 |
| HorizonForge | [CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Wang_HorizonForge_Driving_Scene_Editing_with_Any_Trajectories_and_Any_Vehicles_CVPR_2026_paper.html) | Gaussian/mesh + video diffusion 任意轨迹/车辆编辑 | paper-only；不下载大模型，不手写复现 |
| RecEdit-Drive | [CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Wu_RecEdit-Drive_3D_Reconstruction-Guided_Spatiotemporal_Video_Editing_for_Autonomous_Driving_Scenes_CVPR_2026_paper.html) | 3D reconstruction-guided 时空视频编辑 | paper-only；无 matched runnable 合同时不填数值 |

## 3. Maintainable / evidence-aware Gaussian assets

| 工作 | 一手来源 | 与 V4 的关系 | V4 执行裁决 |
|---|---|---|---|
| OP2GS | [arXiv 2605.20044](https://arxiv.org/abs/2605.20044) | dual-opacity object-aware primitive | `OP2GS-inspired` evidence 初始化；无官方 runnable source 时不称复现 |
| GaME | [官方仓库](https://github.com/VladimirYugay/GaME) | evolving scene Gaussian mapping | source reference；本轮不接新系统 |
| GOR-IS | [CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Zhao_GOR-IS_3D_Gaussian_Object_Removal_In_the_Intrinsic_Space_CVPR_2026_paper.html) | intrinsic-space 3DGS object removal | paper-only/audit-only；继承非商业许可、无冻结 runtime/weights 风险 |
| FocusGS | [arXiv 2607.28834](https://arxiv.org/abs/2607.28834) | local spatial delta、deterministic editing | 只吸收 asset-state delta 接口思想；不声称图像线性可加 |
| LiDAR-EVS | [arXiv 2603.14763](https://arxiv.org/abs/2603.14763) | pseudo-LiDAR extrapolated-view supervision | future sensor robustness；无官方 runnable source 时不进入当前主线 |

## 4. 由来源事实得到的边界

- Street Gaussians、AD-GS、SplatAD 分别覆盖 object-composed reconstruction、B-spline motion 和 camera/LiDAR
  real-time rendering，但都不直接提供 V4 的 evidence-calibrated reversible repair compiler。
- HorizonForge 和 RecEdit-Drive 证明视频扩散与 3D 引导编辑是重要强比较，但 V4 的单卡、exact rollback、provenance
  和 selective abstention 合同不同；没有官方 matched execution 时只作 paper comparison。
- OP2GS/FocusGS/LiDAR-EVS 的公式或接口只能在本仓库存在对应实现、配置、ablation 和指标后进入 Method claim。
- “官方论文/源码存在”不等于“已在本机单卡执行”；所有数值 baseline 仍需 B0 独立 preflight 和 same-split replay。
