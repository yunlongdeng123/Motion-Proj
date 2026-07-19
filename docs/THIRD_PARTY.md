# 第三方依赖

Motion-Proj 主仓库不包含大型第三方源码与模型权重。换机后按需准备以下路径。

## CoTracker3（E0 独立 evaluator）

- **用途：** Autoresearch E0 冻结 CoTracker3 offline evaluator
- **固定 commit：** `82e02e8029753ad4ef13cf06be7f4fc5facdda4d`
- **预期路径：** `/root/autodl-tmp/third_party/co-tracker`
- **一键克隆：**

```bash
bash scripts/setup_third_party.sh
```

- **权重：** `scaled_offline.pth` 需放到 `third_party/co-tracker/checkpoints/`。首次 E0 若缺失会 blocked；下载前可 `source /etc/network_turbo` 或 `export HF_ENDPOINT=https://hf-mirror.com`。
- **配置引用：** `configs/diagnostics/autoresearch_e0_evaluator*.yaml` 中 `repository_path` 指向上述目录。

## SVD-XT backbone

见 [`scripts/download_weights.md`](../scripts/download_weights.md)。固定 Hugging Face revision 为
`9e43909513c6714f1bc78bcb44d96e733cd242aa`。2026-07-19 清理前的完整本地快照为 32.61 GB，同时包含
monolithic 与 Diffusers `full`/`fp16` 权重，而不是旧文档所写的约 10 GB。该资产已按
[`ARTIFACT_RETENTION.md`](ARTIFACT_RETENTION.md) 清理，当前为 `non-resident`；如获授权，可从固定 revision
重建。不能仅凭历史配置假定 `/root/autodl-tmp/weights/svd-xt` 驻留。

## nuScenes

完整数据集在 AutoDL 公共盘；当前数据盘另有约 35G 的 CAM_FRONT/LIDAR_TOP 与 metadata 本地子集。
路径见 [`ENVIRONMENT.md`](ENVIRONMENT.md) §4，抽取脚本在 `scripts/extract_nuscenes_*.sh`。

## 不入 Git 的运行产物

以下目录保留在数据盘，换机若只 `git clone` 需重新生成或从旧机拷贝：

| 路径 | 内容 |
|---|---|
| `/root/autodl-tmp/runs/` | 实验 run；正式结论保留轻量证据，checkpoint/candidate 等载荷可按保留策略瘦身 |
| `/root/autodl-tmp/cache/` | 投影 / replay cache |
| `/root/autodl-tmp/weights/` | SVD、CoTracker 等权重 |
| `/root/autodl-tmp/envs/motionproj` | Conda 环境（可用 `requirements.lock.txt` 重建） |

轻量 run 摘要已归档到 `docs/run_manifests/`，供对照 commit 与 resolved config。正式 run ID 不得复用或覆盖，
但这不要求永久保留每个 checkpoint、candidate 视频或中间 tensor；实际驻留范围与受保护人工材料以
[`ARTIFACT_RETENTION.md`](ARTIFACT_RETENTION.md) 为准。
