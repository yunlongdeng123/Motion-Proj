# 换机 / 新实例接续指南

系统盘若通过 AutoDL 克隆，conda 环境与数据盘通常可以直接沿用。若只拉取 Git 代码，按本指南恢复
代码、环境和研究上下文；run 载荷、cache、权重与 checkpoint 不在 Git 中，且可能已按保留策略清理。

## 1. 拉取代码并核对状态

```bash
cd /root/autodl-tmp
git clone https://github.com/yunlongdeng123/Motion-Proj.git motion_proj
cd motion_proj
git status --short --branch
git log -3 --oneline
```

不得在不知道原机器 commit、dirty 状态和 run fingerprint 的情况下续跑正式实验。

## 2. 恢复环境

克隆系统盘时通常可跳过安装，但每个新 shell 仍需加载 conda：

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/envs/motionproj
```

若环境不存在：

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda create -p /root/autodl-tmp/envs/motionproj python=3.10 -y
conda activate /root/autodl-tmp/envs/motionproj
pip install -r requirements.lock.txt
```

完整版本约束与网络说明见 [`ENVIRONMENT.md`](ENVIRONMENT.md)。

## 3. 第三方、权重与数据

```bash
bash scripts/setup_third_party.sh
# 历史 SVD-XT 当前 non-resident，仅在获授权复现时按固定 revision 重建，见 scripts/download_weights.md
```

- nuScenes 全量：`/autodl-pub/data/nuScenes/Fulldatasetv1.0`
- 当前本地子集：`/root/autodl-tmp/data/nuscenes`（约 35G）
- Mini 开发切分：`bash scripts/extract_nuscenes_mini.sh`
- Hugging Face cache：`/root/autodl-tmp/hf_cache`

第三方 commit 与离线资源见 [`THIRD_PARTY.md`](THIRD_PARTY.md)。

## 4. 研究上下文阅读顺序

不要仅依赖聊天记录，也不要从归档计划恢复“下一步”。按以下顺序阅读：

1. `AGENTS.md`：环境、研究连续性和提交约定；
2. [`RESEARCH_STATUS.md`](RESEARCH_STATUS.md)：唯一当前状态与执行授权入口；
3. [`RESEARCH_FAILURES.md`](RESEARCH_FAILURES.md)：已验证 research 负结论、禁止重复项和未决风险；
4. [`EXPERIMENTS.md`](EXPERIMENTS.md)：实验事实源；
5. 相关正式 run 的 `manifest.json`、`resolved.yaml`、`summary.json` 和指标文件；
6. [`archive/2026-07/README.md`](archive/2026-07/README.md)：仅在追查历史方案时阅读。

`docs/archive/` 中的旧计划、报告和提示词不再授权任何执行。新的研究动作必须由新的当前计划明确解锁。

## 5. 迁移运行产物

如果要复核已有实验，先查 [`ARTIFACT_RETENTION.md`](ARTIFACT_RETENTION.md) 判断资产是否仍驻留。需要迁移的
现存目录通常包括：

```text
/root/autodl-tmp/runs/
/root/autodl-tmp/cache/
/root/autodl-tmp/weights/
/root/autodl-tmp/hf_cache/
```

用 `docs/run_manifests/` 的轻量副本和 [`EXPERIMENTS.md`](EXPERIMENTS.md) 核对 commit、config
fingerprint、数据 split、seed 与终止标记。正式 run ID 不得复用或覆盖；但已按策略移除的 checkpoint、
candidate pool 或中间 tensor 不应被误报为迁移丢失。正式 48 条人工 verdict、R1/A0 待评审包和 UPO
common-support 证据属于受保护资产，迁移时必须核对其 SHA-256。

## 6. 快速自检

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/envs/motionproj
cd /root/autodl-tmp/motion_proj
pytest -q
```

如果只检查关键基础设施，可先运行：

```bash
pytest -q \
  tests/test_svd_conditioning_parity.py \
  tests/test_independent_evaluator.py \
  tests/test_selective_partial_order.py \
  tests/test_physics_preference_candidate_fallback.py
```

## 7. 当前研究状态

截至 2026-07-19，V1 projection、SVD common-prefix sibling preference 以及 V5 route-pivot 均已按门禁收口；
V5 选择 `C1 = ReSim exp0_no_carla feasibility`，但没有授权下载、推理或训练。R1 的 32 个 pair 与 A0 v3 的
12 个 panel 仍待人工 review，并受保留策略保护。准确任务状态只以
[`RESEARCH_STATUS.md`](RESEARCH_STATUS.md) 为准；V5 计划只保留为已完成协议快照。
