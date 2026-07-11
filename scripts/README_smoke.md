# Motion-Proj smoke 跑通保障脚本

这组脚本用于在正式训练前做最小真实链路验证：检查资产、构建 1 个 latent 投影缓存、跑 2 step LoRA 训练、再读取缓存元数据做评估。

今天没有 GPU 时，不要运行 `smoke_pipeline.sh`。可以只阅读本文档，或在有环境但无 GPU 时运行不带 `--require-gpu` 的资产检查。

## 文件

- `scripts/check_assets.py`：检查 Python 环境、核心依赖、CUDA、nuScenes mini、SVD 权重、输出目录可写性和磁盘余量。
- `scripts/smoke_pipeline.sh`：串联 `check_assets -> build_cache -> train -> eval`，全部参数通过命令行 override 传入，不修改主配置。
- smoke 运行报告会写到 `/root/autodl-tmp/runs/motionproj_smoke/reports/`。

## 明天有 GPU 后的推荐命令

```bash
cd /root/autodl-tmp/motion_proj
source /root/miniconda3/etc/profile.d/conda.sh
conda activate motionproj


# HF 镜像下载 Depth-Anything V2 Small
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/hf_cache

bash scripts/smoke_pipeline.sh
```

默认假设 SVD 权重在：

```bash
/root/autodl-tmp/weights/svd-xt
```

如果你的权重路径不同：

```bash
MODEL_PRETRAINED=/path/to/svd-xt bash scripts/smoke_pipeline.sh
```

## 只做资产检查

无 GPU 时可运行：

```bash
python scripts/check_assets.py --config configs/train/motionproj_v1.yaml \
  model.pretrained=/root/autodl-tmp/weights/svd-xt
```

有 GPU 且准备跑 smoke 前可运行更严格版本：

```bash
python scripts/check_assets.py --config configs/train/motionproj_v1.yaml --require-gpu \
  model.pretrained=/root/autodl-tmp/weights/svd-xt
```

## 常用参数

```bash
MAX_SAMPLES=1 TRAIN_STEPS=2 bash scripts/smoke_pipeline.sh
RUN_TRAIN=0 bash scripts/smoke_pipeline.sh
CACHE_DIR=/root/autodl-tmp/cache/projection_smoke_v2 \
WORK_DIR=/root/autodl-tmp/runs/motionproj_smoke_v2 \
bash scripts/smoke_pipeline.sh
```

`RUN_TRAIN=0` 只跳过训练；缓存构建仍会加载 RAFT/SVD，通常仍需要 GPU。

## 预期产物

- `/root/autodl-tmp/cache/projection_smoke/<sample_id>/metadata.json`
- `/root/autodl-tmp/cache/projection_smoke/<sample_id>/y.pt`
- `/root/autodl-tmp/cache/projection_smoke/<sample_id>/x_dagger.pt`
- `/root/autodl-tmp/cache/projection_smoke/<sample_id>/mask.pt`
- `/root/autodl-tmp/cache/projection_smoke/<sample_id>/context.pt`
- `/root/autodl-tmp/runs/motionproj_smoke/ckpts/adapter_final.safetensors`
- `/root/autodl-tmp/runs/motionproj_smoke/reports/smoke_*.md`

这些输出目录都在 `.gitignore` 覆盖范围内，不应加入 Git。
