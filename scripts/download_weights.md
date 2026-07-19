# 下载 SVD backbone 权重

Motion-Proj 历史 V1 的 backbone 是 **Stable Video Diffusion (img2vid)**。权重*不*随项目打包。
2026-07-19 的保留策略批次已将本地 SVD-XT 清理为 `non-resident` 可恢复资产；驻留状态以
[`docs/ARTIFACT_RETENTION.md`](../docs/ARTIFACT_RETENTION.md) 为准。只有经当前研究计划明确授权、且确实需要
复现历史 SVD 路线时才重新下载；C1 的 ReSim feasibility 不以该目录为依赖。

## 方式 A：通过 AutoDL 学术加速使用 HuggingFace（推荐）

```bash
conda activate motionproj
source /etc/network_turbo                 # 为 huggingface.co 启用学术代理
export HF_HOME=/root/autodl-tmp/hf_cache   # 把大缓存放在数据盘上

python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("stabilityai/stable-video-diffusion-img2vid-xt",
                  revision="9e43909513c6714f1bc78bcb44d96e733cd242aa",
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
    --revision 9e43909513c6714f1bc78bcb44d96e733cd242aa \
    --local-dir /root/autodl-tmp/weights/svd-xt
```

## 说明

- 仓库：<https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt>
  （非 xt 的 14 帧变体：`stabilityai/stable-video-diffusion-img2vid`）。
- 固定 revision：`9e43909513c6714f1bc78bcb44d96e733cd242aa`。2026-07-19 清理前的完整本地快照为
  `32,608,949,417` 字节（32.61 GB），同时包含两个 monolithic safetensors，以及 Diffusers
  `full`/`fp16` 的 UNet、VAE 和 image encoder；旧文档中的“约 10GB”只接近单个 monolithic 文件，不能代表
  完整目录。重建完整快照前应至少预留 33 GB，并另留下载临时空间。
- `HF_HOME` 保持在 `/root/autodl-tmp/hf_cache`，不要把大缓存写入系统盘。
- 可选的 Depth-Anything 模型（`depth-anything/Depth-Anything-V2-Small-hf`）
  会在审计器首次运行时自动拉取（同样需要代理 / 镜像）。若缺失，审计器会退化为
  使用恒定深度平面。
