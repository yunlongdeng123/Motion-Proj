> 历史归档（2026-09-10）：以下为原版本事实与指令快照；当前执行以 docs/RESEARCH_STATUS.md 为准。

# WorldSim V7.2 B0 LiDAR4D 原生能力结果

## 结论

LiDAR4D 官方 KITTI-360 sequence 00 / frames 4950--5000 协议已经在单张 RTX 3090 上完整执行。运行覆盖官方 30k 训练、1000-step ray-drop refinement、4 帧最终评测和扫描结果导出，证明路线 B 已具备真实 full-return、depth、intensity 与 ray-drop 的场景级执行能力。

canonical run：

`run://worldsim_v72/WS-V72-B0-LIDAR4D-CAPABILITY-01/20260906T192557Z__lidar4d-kitti360-f4950-s0-r2`

该运行只证明外部场景级方法、数据 I/O、CUDA 环境和完整扫描协议可执行。它使用 KITTI-360 公共序列，不能与 legacy Actor completion 的绝对数值横比，也不解锁 D1 选路。

## 冻结协议

| 维度 | 实际执行 |
|---|---|
| LiDAR4D | 官方代码 `4d6abbd9e0c49744f5fb724a467ab0d8983f23fa` |
| tiny-cuda-nn | `749dd70c5afc5a9dadb85e5652ed65d55e0ba187`，sm86 |
| 数据 | KITTI-360 `2013_05_28_drive_0000_sync`，frames 4950--5000 |
| split | 47 train；4 validation/test |
| 训练 | seed 0；30,000 nominal iterations；实际 639 epochs × 47 steps = 30,033 steps |
| 采样 | 1024 LiDAR rays；768 samples/ray；FP16；数据预载 GPU |
| 损失权重 | depth 1.0；intensity 0.1；ray-drop 0.01 |
| refinement | 官方 UNet ray-drop refinement，1000 steps |
| 数据角色 | `external_public_capability`；`route_selection_eligible=false` |

## 数据处理与 I/O

官方序列压缩包约 16.1 GB，但本协议只使用 51 帧。下载器读取远程 ZIP central directory，并通过 HTTP Range 提取所需 `.bin`，没有下载或解压其余约 1.1 万帧。

- 51 个 Velodyne 文件：`93,389,072` bytes；每帧记录 bytes、点数与 SHA-256。
- calibration、poses、timestamps 使用官方小型 archive，并记录 URL、archive bytes 与 SHA-256。
- 预处理输出 51 个 `66×1030×3` range view，以及官方 `47/4/4` transforms JSON。
- 数据 manifest：`/root/autodl-tmp/datasets/kitti360_2013_05_28_drive_0000_sync_4950_5000_manifest.json`。

## 最终评测

| 输出 | 指标 | 结果 |
|---|---|---:|
| ray-drop | RMSE | 0.21856689 |
| ray-drop | accuracy | 0.93833113 |
| ray-drop | F1 | 0.95533785 |
| intensity | RMSE / MedAE | 0.10640751 / 0.02615141 |
| intensity | LPIPS / SSIM / PSNR | 0.15639430 / 0.61158503 / 19.46094179 |
| depth | RMSE / MedAE（m） | 2.94718367 / 0.03123474 |
| depth | LPIPS / SSIM / PSNR | 0.07972384 / 0.85347614 / 28.67598783 |
| point cloud | CD | 0.11732581 |
| point cloud | F-score | 0.92080630 |

第 100→600 epoch 的 point CD 从 `0.19769783` 降到 `0.12444092`，F-score 从 `0.89189100` 升到 `0.91757846`。最终 refinement 后 ray-drop F1 从 epoch 600 的 `0.89606319` 提高到 `0.95533785`；这说明完整输出改善包含独立 ray-drop refinement 的贡献，不能全部归因于主场模型。

## 资源与产物

| 项目 | 结果 |
|---|---:|
| wall time | 8,791.62 s（2 h 26 min 32 s） |
| GPU 样本 | 1,729 |
| 平均 GPU utilization | 71.91% |
| 峰值 GPU memory | 22,506 MiB |
| 模型参数 | 53,056,673 |
| 主 checkpoint | 1,009,061,493 bytes；SHA-256 `289e2953...34557f` |
| refined checkpoint | 212,319,394 bytes；SHA-256 `a8d7b89f...e2a2a65` |

运行目录包含 resolved config、dataset manifest、upstream config、environment freeze、fingerprint、stdout、GPU resource JSONL、7 组评测 JSONL、两个 checkpoint、summary 和 final manifest。

## 失败边界

r1=`20260906T192156Z__lidar4d-kitti360-f4950-s0-r1` 在构造 LPIPS meter 时遇到 AlexNet 权重的 Python 下载连接停滞，发生在 dataloader 和训练之前，见 `V71-F61`。r2 只预取相同官方 URL、bytes 与 SHA-256 的权重，模型、数据、split、seed 和训练协议没有改变。

canonical r2 的 `failure_ledger_delta=none`。Motion-Proj `source_test` 与 `external_test` 始终未读。
