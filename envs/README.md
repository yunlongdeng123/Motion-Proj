# M2 环境说明（AD-GS / DPT / SAM）

## 快速激活

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate adgs        # AD-GS 主训练环境
conda activate adgs-dpt    # Depth Anything V2 深度先验
conda activate adgs-sam    # Grounded-SAM-2 语义分割先验
```

## 环境一览

| 环境 | 路径 | Python | PyTorch | 用途 |
|------|------|--------|---------|------|
| `adgs` | `/root/autodl-tmp/envs/adgs` | 3.7.16 | 1.13.1+cu117 | AD-GS 训练/渲染 |
| `adgs-dpt` | `/root/autodl-tmp/envs/adgs-dpt` | 3.11 | 2.4.1+cu121 | 单目深度先验 |
| `adgs-sam` | `/root/autodl-tmp/envs/adgs-sam` | 3.10 | 2.4.1+cu118 | 文本引导分割先验 |

## 仓库与 commit

| 仓库 | 路径 | Commit |
|------|------|--------|
| AD-GS | `/root/autodl-tmp/third_party/AD-GS` | `9a208512e49c8ddbaa20387921d9648adcd21cb4` |
| pytorch3d | `/root/autodl-tmp/third_party/pytorch3d-v0.7.2` | `v0.7.2` |
| Depth-Anything-V2 | `/root/autodl-tmp/third_party/Depth-Anything-V2` | `a561b84` |
| Grounded-SAM-2 | `/root/autodl-tmp/third_party/Grounded-SAM-2` | `b7a9c29` |

## 冒烟测试

```bash
# AD-GS（需先 conda activate adgs）
cd /root/autodl-tmp/third_party/AD-GS
python /root/autodl-tmp/motion_proj/scripts/smoke_adgs_env.py

# DPT
conda activate adgs-dpt
cd /root/autodl-tmp/third_party/Depth-Anything-V2
python /root/autodl-tmp/motion_proj/scripts/smoke_dpt_env.py

# SAM（必须在 Grounded-SAM-2 根目录）
conda activate adgs-sam
cd /root/autodl-tmp/third_party/Grounded-SAM-2
python /root/autodl-tmp/motion_proj/scripts/smoke_sam_env.py
```

## 重建环境（如需）

```bash
# 1. micromamba 在 /root/autodl-tmp/bin/micromamba
export MAMBA_ROOT_PREFIX=/root/autodl-tmp/micromamba
export PATH=/root/autodl-tmp/bin:$PATH

# 2. AD-GS conda 层
micromamba create -y -p /root/autodl-tmp/envs/adgs \
  -f /root/autodl-tmp/motion_proj/envs/adgs_conda.yaml

# 3. AD-GS pip 层 + CUDA 扩展
conda activate adgs
pip install -r /root/autodl-tmp/motion_proj/envs/adgs_pip.txt
cd /root/autodl-tmp/third_party/AD-GS
pip install --no-build-isolation -e ./submodules/simple-knn
pip install --no-build-isolation -e ./submodules/depth-diff-gaussian-rasterization
cd /root/autodl-tmp/third_party/pytorch3d-v0.7.2
FORCE_CUDA=1 pip install --no-build-isolation --no-deps .
```

## 注意事项

- **不要**在 base conda 上装 `conda-libmamba-solver`（会触发大规模依赖升级并可能破坏 base）
- 下载 GitHub/HuggingFace 时开 `source /etc/network_turbo`；装 conda/pip 时**关掉**代理
- `adgs` 激活后自动设置 `OMP_NUM_THREADS=16`、`CUDA_HOME=/usr/local/cuda-11.8`、`TORCH_CUDA_ARCH_LIST=8.6+PTX`
- pip 缓存目录：`/root/autodl-tmp/pip_cache`（配置在 `~/.config/pip/pip.conf`）
