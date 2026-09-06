# WorldSim V7.2 外部基线 capability 审计

日期：2026-09-06

任务：`WS-V72-P0-BASELINE-CAPABILITY-01`

本审计只读取论文、官方仓库和本地数据可用性，不运行质量评测。源码快照位于 `/root/autodl-tmp/external/worldsim_v72/`，不修改上游仓库。

## 结论

路线 A 可以把 AdaPoinTr 作为第一个强学习基线，但不能直接拿 ShapeNet/PCN checkpoint 在驾驶 Actor 上失败后称其无效。需要使用相同 Actor build/target 划分明确微调，并统一尺度、输入点数和输出点数。官方 PCN checkpoint 与 593/66 Actor 适配缓存现已就绪，CUDA capability 等待开卡。

路线 B 当前主要受数据和 CUDA 环境阻塞。DyNFL 与 V7.2 的动态对象/给定轨迹设定最接近，但官方流程依赖 Waymo preprocessing、Nerfstudio 0.3.4 和编译扩展；当前没有已授权 Waymo 数据。LiDAR4D 的 KITTI-360 流程更容易独立跑通，但远端只有 KITTI tracking smoke，没有其完整 sequence 数据。

## 官方代码对照

| 方法 | 固定源码 | 原生输入/任务 | 环境与计算 | V7.2 处理 |
|---|---|---|---|---|
| AdaPoinTr | `4603257e`；MIT | partial→complete 点集；PCN 配方 2048 输入、16384 输出、600 epochs | Chamfer、PointNet++、kNN CUDA 扩展；官方支持单 GPU 训练 | 第一 A 强基线；Actor-uniform normalization 后训练，原生与点数匹配结果并报 |
| Object-Centric Occ Completion | `0ca9883d`；MIT | Waymo tracklet 与隐式对象占据，连接检测/跟踪下游 | 官方标注示例 8 GPU；依赖 SST/MMDetection3D、检测与 ImmortalTracker | A 入选后再接；当前不为跑 baseline 搭完整检测系统 |
| DyNFL | `b6d03de5` | Waymo LiDAR + tracked dynamic objects 的组合神经场 | Python 3.10.10、Torch 1.13.1、CUDA 11.6、Nerfstudio 0.3.4；raymarch/chamfer 编译；官方测试 RTX 3090/1080Ti | B 首选；Waymo access 未具备时状态为 `blocked_access`，不能记方法失败 |
| LiDAR4D | `4d6abbd9`；Apache-2.0 | KITTI-360 逐序列 4D 场，depth/intensity/ray-drop 与模拟 | 默认 30k iters、768 samples/ray；tiny-cuda-nn 与 Chamfer 扩展 | B 可访问备选；必须下载完整 KITTI-360 sequence，tracking smoke 不足 |
| LiDAR-RT | `a411b756`；训练/评测代码已发布 | 动态 Gaussian ray tracing LiDAR 重建 | 含 submodules/编译组件，尚未做本机安装 | B 入选后作为第二种神经 LiDAR 对照 |

## 不能直接公平比较的地方

- AdaPoinTr 的 ShapeNet/PCN 预训练包含额外形状数据；从头训练与预训练必须分表。
- Object-Centric Occ 的官方结果包含 Waymo、检测与跟踪链；给 GT 轨迹的适配版不能冒充原表复现。
- DyNFL 和 LiDAR4D 是逐场景重建；若 V7.2 在目标场景 build scans 上优化，也必须固定步数并留出 target scans，不能称零样本。
- LiDAR4D 的官方 depth 误差在 GT return mask 下计算，同时另评 ray-drop；V7.2 主表需要固定全部有效发射分母，避免漏报返回从距离指标消失。
- 当前 nuScenes/AV2 返回点文件不足以证明真实硬件 no-return；若没有 firing validity，只能称 projected range-image 空 bin 协议。

## 当前资源和阻塞

| 项目 | 实测 |
|---|---:|
| Motion-Proj 仓库 | 352 MiB |
| `/root/autodl-tmp` 可用空间 | 约 82 GiB |
| AV2 已下载 | 80 logs / 83 GiB，全部历史已消费 |
| V7.1 Actor cache | 1004 Actors / 56 MiB 数据目录 |
| PoinTr/DyNFL/LiDAR4D 源码快照 | 约 79 MiB |
| AdaPoinTr 官方 PCN checkpoint | 389,745,620 bytes；SHA-256 `f58a5650...64fa1` |
| AdaPoinTr legacy adapter cache | 659 Actors / 约 15.6 MiB；593 train / 66 holdout |
| GPU | 不可用；`nvidia-smi` 被拒绝 |

当前 PyTorch 为 `2.4.1+cu121`，但 `torch.cuda.is_available()=false` 且无 `nvcc`；因此不创建 CUDA 环境、不编译扩展、不下载场景级大数据。GPU 恢复时先用 1×RTX 3090 24 GiB 做 capability；正式 B 路线还需 Waymo 授权数据，或为 LiDAR4D 准备完整 KITTI-360 sequence。

## 一手来源

- PoinTr/AdaPoinTr：https://github.com/yuxumin/PoinTr ，https://arxiv.org/abs/2301.04545
- Object-Centric Occupancy Completion：https://github.com/Ghostish/ObjectCentricOccCompletion ，https://arxiv.org/abs/2412.05154
- DyNFL：https://github.com/prs-eth/Dynamic-LiDAR-Resimulation ，https://arxiv.org/abs/2312.05247
- LiDAR4D：https://github.com/ispc-lab/LiDAR4D ，https://arxiv.org/abs/2404.02742
- LiDAR-RT：https://github.com/zju3dv/LiDAR-RT ，https://arxiv.org/abs/2412.15199
