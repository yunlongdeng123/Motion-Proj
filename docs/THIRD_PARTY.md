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

见 `scripts/download_weights.md`。本地目录默认 `/root/autodl-tmp/weights/svd-xt`（约 10GB，不入 Git）。

## nuScenes

数据集在 AutoDL 公共盘，路径见 `docs/ENVIRONMENT.md` §4。抽取脚本在 `scripts/extract_nuscenes_*.sh`。

## 不入 Git 的运行产物

以下目录保留在数据盘，换机若只 `git clone` 需重新生成或从旧机拷贝：

| 路径 | 内容 |
|---|---|
| `/root/autodl-tmp/runs/` | 完整实验 run（metrics、checkpoint、panel） |
| `/root/autodl-tmp/cache/` | 投影 / replay cache |
| `/root/autodl-tmp/weights/` | SVD、CoTracker 等权重 |
| `/root/autodl-tmp/envs/motionproj` | Conda 环境（可用 `requirements.lock.txt` 重建） |

轻量 run 摘要已归档到 `docs/run_manifests/`，供对照 commit 与 resolved config；完整证据仍以 run 目录为准。
