# Motion-Proj：面向驾驶世界模型的动力学投影蒸馏

对研究方案 `../motion_proj_cvpr_plan.md` 的工程实现（V1）。

Motion-Proj 通过把干净层级（clean-level）生成/受扰的视频投影到由几何定义的驾驶动力学流形上，
并将投影诱导的局部去噪分数蒸馏进低噪声去噪器，从而对齐视频扩散驾驶世界模型。
运动审计器（motion auditor）与动力学投影器（dynamics projector）均为 **无梯度 / 离线 / 缓存**，
因此训练只在视频去噪器上做反向传播。

## 流水线

```
NuScenesFutureVideoDataset
  -> Motion Auditor (no-grad): RAFT flow + ego-induced static flow + GT box tracks + depth
  -> Dynamics Projector: energies + RTS smoothing + support filter + warper Gamma
  -> Projection Cache: (y, x_dagger, mask M_y, context, metadata) in latent space
  -> Projection Distillation: L_real + lambda_proj * L_proj + beta * L_anchor (low-noise tube)
  -> Eval: static drift / object track smoothness / LPIPS / FVD
```

## 目录结构

```
configs/                      # OmegaConf yaml 配置
motion_proj/
  config.py                   # 配置加载 + dataclass schema
  utils/                      # 几何、io、可视化、日志
  data/                       # NuScenesFutureVideoDataset
  backbones/                  # DiffusionBackbone 接口 + SVD 适配器
  auditor/                    # 无梯度运动审计器 -> MotionState
  projector/                  # 动力学投影器 -> x_dagger + mask
  cache/                      # 投影缓存 writer/reader + 构建 CLI
  losses/                     # tube 采样 + L_real / L_proj / L_anchor
  train/                      # trainer + 训练 CLI
  eval/                       # 指标 + 诊断 + 评估 CLI
  replay/                     # replay 挖掘 CLI
scripts/                      # 数据抽取 + 权重下载说明
tests/                        # 几何 / 投影损失 / 数据集冒烟测试
```

## 快速开始

```bash
conda activate motionproj

# 0. （一次性）把 nuScenes mini 抽取到数据盘
bash scripts/extract_nuscenes_mini.sh

# 1. 下载 SVD 权重（见 scripts/download_weights.md）

# 2. 在 mini 切分上构建投影缓存
python -m motion_proj.cache.build_cache --config configs/train/motionproj_v1.yaml

# 3. 投影蒸馏微调
python -m motion_proj.train.train_motionproj --config configs/train/motionproj_v1.yaml

# 4. 评估动态一致性指标
python -m motion_proj.eval.evaluate --config configs/train/motionproj_v1.yaml
```

## 状态（V1）

代码完整、模块化。较重的外部步骤（SVD 权重下载、mini 抽取、完整训练）已延后处理，
并在上文中记录说明。每个模块均可独立导入，且多数模块提供 `__main__` 自检或在
`tests/` 中有对应测试。

conda 环境、数据集路径与网络说明见 `../ENVIRONMENT.md`。
