# Motion-Proj 运行环境速记

本文件记录 Motion-Proj 项目的 conda 环境与服务器关键路径，方便后续会话快速接续。
对应方案见 `motion_proj_cvpr_plan.md`。

## 1. 环境激活

环境建在数据盘（系统盘可写层仅 30G，不够放 cache/checkpoint）：

```bash
conda activate motionproj
# 等价于
conda activate /root/autodl-tmp/envs/motionproj
```

- Python: 3.10
- 路径: `/root/autodl-tmp/envs/motionproj`（约 7.8G）
- `envs_dirs` 已注册 `/root/autodl-tmp/envs`，故可直接按名 `motionproj` 激活。

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

```
系统盘 /              : 30G，可写约 26G（勿在此放大文件）
数据盘 /root/autodl-tmp: 50G，env 占 7.8G，余约 43G
```

- 数据盘仅 50G，env + cache + checkpoint 会很快吃紧。建议:
  - 尽早在 AutoDL 控制台扩容 autodl-tmp；
  - 优先用 latent-space 投影 cache，RGB 中间产物验证后即删；
  - 按方案 §11 的磁盘策略管理 cache 版本。

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
