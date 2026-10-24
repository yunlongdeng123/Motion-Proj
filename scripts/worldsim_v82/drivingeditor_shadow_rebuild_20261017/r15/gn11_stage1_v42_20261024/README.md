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
