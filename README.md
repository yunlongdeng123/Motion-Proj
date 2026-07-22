# Motion-Proj

Motion-Proj 当前研究主线是 **OccGS-Resim V7**：基于 object-centric Gaussian scene representation、显式
occupancy/world state 和约束轨迹编辑，构建可验证的驾驶反事实重仿真数据。

## 当前状态

V7 已在单张 RTX 4090 上完成三场景 feasibility 闭环：

```text
nuScenes preparation
→ StreetGS reconstruction
→ actor trajectory edit
→ counterfactual RGB/depth render
→ local hard composition
```

当前决策为 `modify_method_then_scale`。这表示路线保留，但核心研究假设仍未通过：

- occupancy 已生成，尚未真正进入 editor、visibility 与 completion mask；
- C0 只有机器 screen，没有用户人工 verdict，也未完成所有标签重生；
- L0 的 outside-mask exact 由 hard composition 构造保证，不代表补全质量提升；
- U0 未运行正式下游任务，不能声称合成数据有效。

因此下一步是证据契约与 occupancy matched ablation，不是直接扩场景、切双卡或回到旧扩散路线。

## 文档入口

从 [`docs/README.md`](docs/README.md) 开始。核心阅读顺序：

1. [`docs/RESEARCH_STATUS.md`](docs/RESEARCH_STATUS.md)：唯一当前状态与执行授权入口；
2. [`docs/OCCGS_RESIM_AUTORESEARCH_PLAN_V7.md`](docs/OCCGS_RESIM_AUTORESEARCH_PLAN_V7.md)：V7 当前计划与门禁；
3. [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md)：V7 实验事实；
4. [`docs/OCCGS_FINAL_REPORT.md`](docs/OCCGS_FINAL_REPORT.md)：feasibility 收口审计；
5. [`docs/RESEARCH_FAILURES.md`](docs/RESEARCH_FAILURES.md)：当前风险、历史约束与防重复条件。

V1–V7 的旧计划、完整旧实验账本、逐 Gate 报告和评审材料索引位于
[`docs/archive/2026-07/`](docs/archive/2026-07/)。归档中的“当前任务”“下一步”只描述当时状态，不构成执行授权。

## 当前 V7 代码

```text
occupancy/
  build_scene_occupancy.py   LiDAR + box scene-local occupancy
  o0_sanity_vis.py           occupancy sanity / BEV review material
resim/
  d0_scene_scan.py           scene scan 与冻结选择
  d0_integrity_check.py      数据完整性检查
  d0_sky_mask_segformer.py   sky mask 替代路径
  s0_trajectory_editor.py    actor 轨迹编辑原型
  c0_counterfactual_render.py RigidNodes counterfactual render
  c0_legality_panel.py       机器合法性 screen 与材料生成
  l0_local_completion.py     local hard-composition feasibility
  u0_utility_screen.py       轻量 proxy（非下游 utility）
```

历史 SVD projection/preference/ReSim 工程仍保留用于追溯，但其研究路线已经关闭；接口存在不代表允许续跑。

## 环境

V7 使用隔离的 DriveStudio 环境：

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/envs/drivestudio
export CUDA_HOME=/usr/local/cuda-11.8
export PATH=$CUDA_HOME/bin:$PATH
export PYTHONPATH=/root/autodl-tmp/third_party/drivestudio:$PYTHONPATH
cd /root/autodl-tmp/motion_proj
```

完整环境、依赖、迁移与存储规则见：

- [`docs/ENVIRONMENT.md`](docs/ENVIRONMENT.md)
- [`docs/THIRD_PARTY.md`](docs/THIRD_PARTY.md)
- [`docs/MACHINE_MIGRATION.md`](docs/MACHINE_MIGRATION.md)
- [`docs/ARTIFACT_RETENTION.md`](docs/ARTIFACT_RETENTION.md)

## 证据与产物

- 当前实验登记：`docs/EXPERIMENTS.md`
- V7 reconstruction：`/root/autodl-tmp/runs/occgs_resim/b0_recon/occgs_b0/`
- V7 occupancy：`/root/autodl-tmp/data/occgs/occupancy/`
- V7 counterfactual：`/root/autodl-tmp/runs/occgs_resim/c0_cf/`
- V7 completion / proxy：`/root/autodl-tmp/runs/occgs_resim/{l0_comp,u0_screen}/`
- 历史轻量 Git 证据：`docs/run_manifests/`

既有 V7 run 为 retrospective evidence，缺少完整 formal manifest/terminal marker；后续新 run 必须先完成
`V7-EV-10` 的运行契约，不得覆盖旧目录或事后伪造 provenance。
