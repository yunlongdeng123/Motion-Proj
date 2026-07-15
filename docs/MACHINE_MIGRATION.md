# 换机 / 新实例接续指南

系统盘若已通过 AutoDL 克隆，conda 环境与 `/root/autodl-tmp/envs/motionproj` 通常可直接沿用。若只拉 Git 代码，按下列顺序恢复可跑实验的状态。

## 1. 拉代码

```bash
cd /root/autodl-tmp
git clone https://github.com/yunlongdeng123/Motion-Proj.git motion_proj
cd motion_proj
git log -3 --oneline   # 确认 HEAD 与旧机一致
```

## 2. 环境

克隆系统盘时跳过此步。否则：

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda create -p /root/autodl-tmp/envs/motionproj python=3.10 -y
conda activate motionproj
pip install -r requirements.lock.txt
```

细节与包版本约束见 `docs/ENVIRONMENT.md`。

## 3. 第三方与权重

```bash
bash scripts/setup_third_party.sh          # CoTracker3 @ 固定 commit
# SVD-XT：见 scripts/download_weights.md
```

## 4. 数据

- nuScenes 全量：AutoDL 公共盘 `/autodl-pub/data/nuScenes/Fulldatasetv1.0`
- Mini 开发切分：`bash scripts/extract_nuscenes_mini.sh`

## 5. 研究上下文（必读）

按顺序阅读，不要仅依赖对话历史：

1. `AGENTS.md` — shell/conda/HF 约定
2. `docs/MOTION_PROJ_CVPR_PLAN.md` — 原始研究方案
3. `docs/CVPR2027_PLAN.md` — 当前里程碑与 blocked 状态
4. `docs/EXPERIMENTS.md` — 实验事实源
5. `docs/AUTORESEARCH_RETROSPECTIVE_2026-07.md` — 2026-07 复盘与停止理由
6. `docs/AUTORESEARCH_ROUTE_DECISION.md` — Phase 2 路线决策
7. `docs/PHYSICS_DPO_AUTORESEARCH_PLAN.md` — 2026-07-14 用户授权的新偏好对齐计划；先读完前述失败边界，再以该计划作为未来任务顺序

Autoresearch 自主研究续跑提示词：`docs/prompts/AUTORESEARCH_PHASE2.prompt.md`。

## 6. 运行产物迁移（可选）

完整 run / cache / checkpoint 不在 Git 中。若需在新机继续同一实验而非重跑：

- 从旧机 rsync `/root/autodl-tmp/runs/`、`/root/autodl-tmp/cache/`、`/root/autodl-tmp/weights/`
- 用 `docs/run_manifests/` 中的 `resolved.yaml` / `manifest.json` 核对 commit 与 fingerprint

## 7. 快速自检

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate motionproj
cd /root/autodl-tmp/motion_proj
pytest -q tests/test_independent_evaluator.py tests/test_svd_conditioning_parity.py
```

## 当前研究状态（2026-07-14）

- **旧 explicit projection：** P1 target legality failed — 当前 RGB/VAE counterfactual 构造不合法；该 endpoint 训练链永久保持 blocked。
- **P-UNC / E0 CoTracker3：** 均有机器门禁证据但人工 review 尚未完成，不能作旧路线的 rollout 改善结论。
- **当前后续：** 新建的 offline Physics-DPO 是独立问题，不使用 P1 target。必须先执行 `docs/PHYSICS_DPO_AUTORESEARCH_PLAN.md` 的 PA0 人审与 PA1/PA2 pair 合法性门槛；在 PA4 单卡 screening 通过前，不启动长训或切双卡。
