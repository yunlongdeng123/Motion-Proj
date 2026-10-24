# V4.2 design and read-only gate audit

[中文设计与现有方法比较](V42_DESIGN.md) · [组件图](architecture.svg)

This package contains CPU diagnostics and a design, **not an implemented V4.2 training engine**. It does not alter V4.1 inputs, output eligibility, checkpoints, or optimizer state. GPU runs are not started by any tool here.

Starting commit: `2706091d53d77b3dac7898c81e907b189bd80e44`.

Read-only audit on the existing AutoDL environment:

```bash
CUDA_VISIBLE_DEVICES='' /root/autodl-tmp/envs/drivingeditor/bin/python tools/audit_source_gates.py \
  --repo /root/autodl-tmp/motion_proj_v82_gn11_v42 \
  --campaign /root/autodl-tmp/drivingeditor_repro/campaign/r15_gn11_stage1_v41_20261023 \
  --output /root/autodl-tmp/drivingeditor_repro/campaign/r15_gn11_stage1_v42_design_20261024/gate_audit_NEW

/root/autodl-tmp/envs/drivingeditor/bin/python tools/extract_point_examples.py \
  --audit /root/autodl-tmp/drivingeditor_repro/campaign/r15_gn11_stage1_v42_design_20261024/gate_audit_NEW
```

The output directory must be new. The script uses the original V4.1 campaign's frozen `tools/prepare_v41_dense.py`, `v41_dense_geometry.py`, `prepare_v41_revision.py`, the r7 dense manifest, the V3 bundle and original frames. Do not replace those dependencies with unrelated current inputs when reproducing this audit. `--repo` supplies the inherited V3 geometry helper. Nine gate arrays must replay exactly before interpreting sensitivity results.

Actual outputs: `.../r15_gn11_stage1_v42_design_20261024/gate_audit_r2/`. It includes the full 26,088-unit arrays, actor query unions, threshold variants, real point examples and numeric summary. These data, original RGB and model/log assets stay outside Git. The script prints diagnostic upper bounds; none of its drop-gate variants is an approved training policy. Disjoint temporal color checks are proxies, not independent ground truth.

Local report: `outputs/V42_Design_Audit_20261024/index.html` in the current Codex workspace. The local report builder expects `work/v42/results/` and the existing V4.1 source-preview images under `work/v41/report_assets_r5/original/`. `tools/build_report.py` is retained as the exact generator; for another checkout, stage those inputs or adapt only the artifact paths.

Implementation milestones, not completed claims:

- Unmodified V1 master and independently loaded edit copy; explicit complete/edit requests.
- Unified source ledger and consistent provenance/eligibility/dense-domain/export rules.
- Native multiview geometry proposals with conflict-aware visibility checks.
- Actual-contribution-driven paired road insertion and local static suppression.
- GPU identity/gradient checks and real ~10-minute save/evaluate/restore/next-stage smoke before any newly authorized long run.

No additional formal experiment budget has been assigned in this design. Default-output isolation is an engineering contract; scientific novelty and editing quality require the comparisons in the design document.

## GaussianPro机制CPU实证

已完成49帧完整车辆轮廓检验及四项几何/信息隔离检查；关键a37/a6未解决，不据此启动GPU。见[CPU_PROPAGATION.md](CPU_PROPAGATION.md)及 tools/cpu_plane_propagation.py。首轮及有界深度修正回退均保留。完整GaussianPro GPU基线尚未复现，候选文件未写入原V1/R/B。

## 孔洞内部道路教师

随后完成独立的CPU内部伪真值原型，见[INTERIOR_ROAD_TEACHER.md](INTERIOR_ROAD_TEACHER.md)。tools/road_teacher_cpu.py输出伪RGB/深度/权重与薄高斯补点；tools/road_teacher_bridge.py提供主副本隔离检查、真实标签优先的编辑损失和近似CPU光栅。整块撤源验证与5项检查已完成；最终编辑训练未接入，生成式模型仅完成调研和输入准备。

## 当前优先级：预训练模型直接补中心

用户已明确a37/a6车底中心是主要目标。外部预训练几何／生成先验成为中心补全主线，已有教师保留为边界约束与控制。近一年候选、公开资源、组件图和核心验收见[PRETRAINED_CENTER_COMPLETION.md](PRETRAINED_CENTER_COMPLETION.md)。首个可运行候选FreeInpaint，GPGS作学习式几何对照，WINGS作原生三维生成方向参考；模型尚未运行，未声称中心已解决。
