# 下载 SVD backbone 权重

Motion-Proj V1 的 backbone 是 **Stable Video Diffusion (img2vid)**。权重*不*随项目打包。
在构建 latent 缓存或训练之前，请先下载一次。

## 方式 A：通过 AutoDL 学术加速使用 HuggingFace（推荐）

```bash
conda activate motionproj
source /etc/network_turbo                 # 为 huggingface.co 启用学术代理
export HF_HOME=/root/autodl-tmp/hf_cache   # 把大缓存放在数据盘上

python - <<'PY'
from huggingface_hub import snapshot_download
# ~10GB；xt 变体最多可生成 25 帧。
snapshot_download("stabilityai/stable-video-diffusion-img2vid-xt",
                  local_dir="/root/autodl-tmp/weights/svd-xt")
PY

unset http_proxy https_proxy                # 关闭代理，以便 pip/aliyun 正常使用
```

然后把模型配置指向本地目录：

```bash
python -m motion_proj.cache.build_cache --config configs/train/motionproj_v1.yaml \
    model.pretrained=/root/autodl-tmp/weights/svd-xt
```

## 方式 B：hf-mirror.com（无需代理）

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/hf_cache
huggingface-cli download stabilityai/stable-video-diffusion-img2vid-xt \
    --local-dir /root/autodl-tmp/weights/svd-xt
```

## 说明

- 仓库：<https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt>
  （非 xt 的 14 帧变体：`stabilityai/stable-video-diffusion-img2vid`）。
- 磁盘：SVD-xt 约 10GB；数据盘只有 50GB。请把 `HF_HOME` 保持在
  `/root/autodl-tmp`，并考虑在同时下载权重和大型投影缓存之前先扩容磁盘。
- 可选的 Depth-Anything 模型（`depth-anything/Depth-Anything-V2-Small-hf`）
  会在审计器首次运行时自动拉取（同样需要代理 / 镜像）。若缺失，审计器会退化为
  使用恒定深度平面。
