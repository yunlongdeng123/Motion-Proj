# WS-V75-DOWNSTREAM-CFBENCH-PREP-02

> 本页是 2026-09-23 的准备阶段快照。后续 r9 人工批准、HUGSIM/DriveEditor 实际执行数量与资产退役边界见 [r9 收尾](r9-closeout/README.md)；下文的 `manual_pending`、`execution_enabled=false` 只描述当时状态。

本目录保存下游反事实 pilot 的轻量、可审计证据；大型模型、数据和逐 case pickle 不进入 Git。

## 当前证据

- `cases.json`：24 个冻结 paired edits，四类各 6 个；
- `qualification.json`：24/24 通过 CPU 几何 gate，全部仍为 `geometry_pass_manual_pending`；
- `review_sheets/`：每个 case 一张人工道路/语义审阅图；
- `driveeditor-reference-contact-sheet.jpg`：18 个 non-ego case 的参考 crop 快速审阅图；
- `assets.json`：下载/解压/软链接物化状态，断点文件不会记作 ready；
- `environment-smoke.json`：六套环境的无 GPU import 检查；
- `gaussiandwm-sample-contract.json`：官方裸 Tensor 被 loader 拒绝、兼容包装被接受的实测；
- `preflight.json`：source/environment/weights/data/prior-evidence gate；
- `run-plan.json`：固定模型顺序、角色和 case 能力路由，`execution_enabled=false`。

大型输入位于：

- `/root/autodl-tmp/data/worldsim_v75_downstream_bench/adapter_inputs/`
- `/root/autodl-tmp/models/worldsim_v75_downstream_bench/`
- `/root/autodl-tmp/external/worldsim_v75_downstream_bench/`

## GPU-free 重建命令

```bash
cd /root/autodl-tmp/motion_proj

/root/autodl-tmp/envs/motionproj/bin/python \
  scripts/qualify_worldsim_v75_downstream_bench.py \
  --manifest docs/autoresearch/worldsim_v75/downstream_bench/cases.json \
  --dataset-root /root/autodl-tmp/data/dynamic_editing_v2/drivestudio_processed_10Hz/trainval \
  --output docs/autoresearch/worldsim_v75/downstream_bench/qualification.json \
  --review-root docs/autoresearch/worldsim_v75/downstream_bench/review_sheets

/root/autodl-tmp/envs/motionproj/bin/python \
  scripts/materialize_worldsim_v75_downstream_assets.py \
  --asset-root /root/autodl-tmp/models/worldsim_v75_downstream_bench \
  --gaussian-archive-root /root/autodl-tmp/data/worldsim_v75_downstream_bench/gaussiandwm/sampled_archives \
  --inventory docs/autoresearch/worldsim_v75/downstream_bench/assets.json

/root/autodl-tmp/envs/motionproj/bin/python \
  scripts/preflight_worldsim_v75_downstream_bench.py \
  --registry configs/worldsim_v75_downstream_bench/models.json \
  --output docs/autoresearch/worldsim_v75/downstream_bench/preflight.json

/root/autodl-tmp/envs/motionproj/bin/python \
  scripts/plan_worldsim_v75_downstream_bench.py \
  --manifest docs/autoresearch/worldsim_v75/downstream_bench/cases.json \
  --registry configs/worldsim_v75_downstream_bench/models.json \
  --preflight docs/autoresearch/worldsim_v75/downstream_bench/preflight.json \
  --qualification docs/autoresearch/worldsim_v75/downstream_bench/qualification.json \
  --output docs/autoresearch/worldsim_v75/downstream_bench/run-plan.json
```

人工 reviewer 只能在看完 review sheet 后填写道路与最终 verdict；任何自动脚本都不得把 `null` 改成通过。正式推理仍需 GPU 恢复，且每种方法先做一个合法 paired smoke。
