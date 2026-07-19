# Motion-Proj 运行环境速记

本文件记录 Motion-Proj 项目的 conda 环境与服务器关键路径，方便后续会话快速接续。
对应方案见 `motion_proj_cvpr_plan.md`。

## 1. 环境激活

环境建在数据盘（系统盘可写层仅 30G，不够放 cache/checkpoint）：

```bash
# Motion-Proj 主环境（审计 / CoTracker / 本仓库测试）
conda activate motionproj
# 等价于
conda activate /root/autodl-tmp/envs/motionproj

# ReSim C1（V6）独立环境——推理/训练用，勿与 motionproj 混装
conda activate /root/autodl-tmp/envs/resim
```

- Python: 3.10
- `motionproj` 路径: `/root/autodl-tmp/envs/motionproj`（约 7.8G）
- `resim` 路径: `/root/autodl-tmp/envs/resim`（约 6.4G；`torch 2.4.0+cu121` + vendored SAT）
- `envs_dirs` 已注册 `/root/autodl-tmp/envs`，故可直接按名 `motionproj` 激活；`resim` 建议用绝对 `-p` 路径激活。
- ReSim 权重: `/root/autodl-tmp/third_party/ReSim/checkpoints/CogVideoX-2b-sat`（含合成 T5 `t5-v1_1-xxl/`）。
- 数据盘 128G **不可扩容**；可用空间硬门槛 ≥30 GiB，训练/大批候选中间产物预算见
  [`MOTION_RESIM_C1_AUTORESEARCH_PLAN_V6.md`](MOTION_RESIM_C1_AUTORESEARCH_PLAN_V6.md) §1.3。

## 2. 硬件与底座

- GPU: 单卡 RTX 4090 24GB（sm_89）
- PyTorch: `2.4.1+cu121`，torchvision `0.19.1`，torchaudio `2.4.1`
- 加速算子: `xformers==0.0.28.post1`，`flash-attn==2.6.3`（cu123/torch2.4/cp310/abiFALSE 预编译 wheel）
- 三者在 bf16 下已实测可用（matmul / flash_attn_func / memory_efficient_attention 均通过）

## 3. 已安装的主要包

- 扩散训练栈: diffusers 0.31.0（注: 必须用 0.31.0，0.38 在 torch2.4+flash-attn 下导入 SVD 会因 attention_dispatch 的 FA3 schema 报错）, transformers 4.46.3, accelerate 1.14, peft 0.19.1, safetensors, einops, timm, bitsandbytes, omegaconf, sentencepiece, ftfy
- 感知（no-grad/离线用）:
  - 光流: torchvision 内置 RAFT（`from torchvision.models.optical_flow import raft_large`），无额外依赖
  - 深度: Depth-Anything V2 → 通过 transformers + timm 加载（首次需联网下权重）
  - 2D 检测/跟踪: ultralytics 8.4 + lap（ByteTrack 类）
  - 几何/warp 辅助: kornia 0.8.2
  - 3D 框 / ego pose: V1 直接用 nuScenes GT 标注
- 数据工具: nuscenes-devkit 1.2, av2 0.3.6, opencv-python 4.11, pyquaternion, shapely, imageio(+ffmpeg), decord
- 评测/指标: lpips, clean-fid, torchmetrics, scikit-image, scipy
- 训练辅助: tensorboard, wandb, tqdm, rich

> 注: `pip check` 会提示 "decord 0.6.0 is not supported on this platform"，这只是 wheel 平台标签问题，`import decord` 实测正常；若遇异常可改用 torchvision/av 读视频。

## 4. 数据集路径

- nuScenes 全量（只读共享盘，无需拷贝）:
  `/autodl-pub/data/nuScenes/Fulldatasetv1.0`
  - Map expansion: `/autodl-pub/data/nuScenes/Mapexpansion`
  - CAN bus expansion: `/autodl-pub/data/nuScenes/CANbusexpansion`
- 当前本地子集：`/root/autodl-tmp/data/nuscenes`，`du -sh` 约 35G。它包含当前研究所需的
  CAM_FRONT、LIDAR_TOP 和 `v1.0-trainval` metadata，不等同于复制完整公共盘数据集。
- Argoverse 2（方案主数据集）: 公共盘暂无，需自行下载到数据盘后用 av2 API 读取。

## 5. 网络 / 模型下载

- pip 走阿里云镜像、conda 走清华镜像（国内快），直连可用。
- HuggingFace 直连超时，下权重前二选一：
  ```bash
  source /etc/network_turbo          # 学术代理（172.29.51.4:12798）
  # 或
  export HF_ENDPOINT=https://hf-mirror.com
  ```
- 注意: 开启学术代理后访问 pip/aliyun 源会变慢，装包时记得 `unset http_proxy https_proxy`。

## 6. 磁盘现状与策略（重要）

2026-07-19 清理登记前的只读快照：

```text
系统盘 /               : 30G，可写约 26G（勿在此放大文件）
数据盘 /root/autodl-tmp : 128G，总计 137,438,953,472 字节
                          已用 92,389,117,952 字节，可用 45,049,835,520 字节（df -h 显示 42G）
本地 nuScenes 子集      : du -sh 约 35G
motionproj 环境         : du -sh 约 7.8G
SVD-XT 完整快照         : 32,608,949,417 字节；已登记为可恢复清理对象
```

清理完成后，数据盘曾可用约 85 GiB；随后装入 ReSim 权重/环境/T5 后余量下降。2026-07-19 环境就绪后快照约
**86G used / 43G avail**（以实时 `df` 为准）。SVD-XT 为 `non-resident`；`/root/autodl-tmp/weights/` 只保留
CoTracker3。

- 正式 run ID、manifest、resolved config、metrics、summary、终止标记和人工 verdict 不得覆盖；已经固化结论的
  checkpoint、candidate 视频、adapter 和中间 tensor 可按
  [`ARTIFACT_RETENTION.md`](ARTIFACT_RETENTION.md) 的逐路径批次瘦身。
- **禁止**为腾盘删除 `envs/motionproj` 或 `envs/resim`；ReSim 正式 evidence 与受保护 review 材料同样不可擅自删。
- 任何训练/候选生成写盘前检查：`avail - 预估峰值 ≥ 30 GiB`；否则先缩协议或等用户授权清理临时产物。
- 大权重下载前重新检查 `df`；历史 V5/C0 文档中的 42G/85G 数字不回写覆盖当前事实。

## 7. 可选后续环境: motionproj-mm（重型 3D 感知）

仅当进入 replay-mining 且需在生成帧上做学习式 3D 检测时再建（mmlab 系列对 torch 版本锁定严，不与主 env 混装）:

```bash
conda create -p /root/autodl-tmp/envs/motionproj-mm python=3.10 -y
# torch 2.1.2 cu121 + mmcv 2.1 / mmdet 3.x / mmdet3d 1.4，离线生成 cache
```

## 8. 快速自检

```bash
conda activate motionproj
python -c "import torch,diffusers,transformers,peft,xformers,flash_attn,kornia,nuscenes,av2; \
print('torch',torch.__version__,'cuda',torch.cuda.is_available(),torch.cuda.get_device_name(0))"
```
