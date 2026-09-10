> 历史归档（2026-09-10）：以下为原版本事实与指令快照；当前执行以 docs/RESEARCH_STATUS.md 为准。

# WorldSim V7.2 外部基线 capability 审计

日期：2026-09-06

任务：`WS-V72-P0-BASELINE-CAPABILITY-01`

本文件先记录 P0 的论文、官方仓库与本地数据能力审计；2026-09-07 已追加 AdaPoinTr 的本机 capability 与 D0 质量运行结果。源码快照位于 `/root/autodl-tmp/external/worldsim_v72/`，上游 checkout 保持不改动；兼容修改位于独立 build copy。

## 结论

路线 A 已完成 AdaPoinTr 的官方 zero-shot、预训练适配 600 epochs 与 scratch 600 epochs。zero-shot 失败显示严重域差；预训练适配稳定优于 scratch，证明外部初始化有价值；但在 64/128/256/512 固定点数下仍未超过 G0/G1 简单前沿。详细结果见 `docs/WORLDSIM_V7_2_D0_GPU_RESULTS.md`。

路线 B 当前主要受数据与独立环境阻塞。DyNFL 与 V7.2 的动态对象/给定轨迹设定最接近，但官方流程依赖 Waymo preprocessing、Nerfstudio 0.3.4 和编译扩展；当前没有已授权 Waymo 数据。LiDAR4D 的 KITTI-360 流程更容易独立跑通，但远端只有 KITTI tracking smoke，没有其完整 sequence 数据。

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
| GPU | 1×RTX 3090 24 GiB；AdaPoinTr AMP batch 64 峰值 `19.96GiB` |

AdaPoinTr 独立环境使用 PyTorch `2.4.1+cu121` 与 CUDA toolkit `12.1.105`；Chamfer 和 PointNet++ 已按 RTX 3090 的 `sm_86` 编译并完成 forward/backward 与正式运行。正式 B 路线仍需 Waymo 授权数据，或为 LiDAR4D 准备完整 KITTI-360 sequence；GPU 已可用，不再列为其外部阻塞。

## 2026-09-07 capability 与质量结果

| 配方 | 训练 | 256 点 CD / F / early / hit | 512 点 CD / F / early / hit | 结论 |
|---|---:|---|---|---|
| official zero-shot | 0 | `812.90mm / 3.33% / 2.75% / 1.60%` | `810.00mm / 3.46% / 2.85% / 1.68%` | 域差严重，低 early 由几乎不命中造成 |
| pretrained-adapted | 600 epochs | `194.00mm / 69.25% / 37.63% / 55.07%` | `170.49mm / 74.19% / 43.91% / 52.09%` | 明显恢复，但被 G0/G1 固定密度结果压过 |
| scratch | 600 epochs | `237.19mm / 60.44% / 36.72% / 54.14%` | `206.49mm / 67.54% / 44.10% / 51.68%` | 同预算弱于预训练适配版 |

checkpoint 原生 16384 输出 strict load 335/335 tensors；适配 4096 输出时载入 333 tensors，只重置 decoder 最后一层 2 个 tensors。训练配方为 AMP、TF32、batch 64、593 Actors、600 epochs；holdout 只在最终评估读取，不参与 checkpoint 选择。

## 一手来源

- PoinTr/AdaPoinTr：https://github.com/yuxumin/PoinTr ，https://arxiv.org/abs/2301.04545
- Object-Centric Occupancy Completion：https://github.com/Ghostish/ObjectCentricOccCompletion ，https://arxiv.org/abs/2412.05154
- DyNFL：https://github.com/prs-eth/Dynamic-LiDAR-Resimulation ，https://arxiv.org/abs/2312.05247
- LiDAR4D：https://github.com/ispc-lab/LiDAR4D ，https://arxiv.org/abs/2404.02742
- LiDAR-RT：https://github.com/zju3dv/LiDAR-RT ，https://arxiv.org/abs/2412.15199
