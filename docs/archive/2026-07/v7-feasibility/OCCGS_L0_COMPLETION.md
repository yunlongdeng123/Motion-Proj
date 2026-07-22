# OccGS L0 — Localized Completion Feasibility

- created_at_utc: 2026-07-21T19:42Z
- gate: `L0-COMP-07`
- 结论：**协议 PASS / 视觉弱基线**（hard composition 保证 outside-mask 不变；Telea 非 SOTA）

## 方法

`motion_proj/resim/l0_local_completion.py`

1. 用 V0 vs edited GS 的 |\ΔRGB| 构 disocclusion/source-footprint 代理 mask（膨胀）。
2. OpenCV Telea 仅在 mask 内补全。
3. Hard composition：`I = (1-M) I_GS + M I_gen`。

## 结果

| scene/variant | outside-mask L1 | inside-mask L1 | mask frac |
|---|---:|---:|---:|
| s2 / V3 | **0.0** | ~26 | ~2.0% |
| s0 / V3 | **0.0** | ~22 | ~1.3% |

证据：`runs/occgs_resim/l0_comp/{s0,s2}_v3/`

## Gate

| 条件 | 结果 |
|---|---|
| outside-mask 变化≈0 | **PASS**（构造保证） |
| 不回整视频扩散 | PASS |
| 视觉质量 > 纯 GS | **未宣称**（Telea 弱） |

**结论：L0 可行性（locality 协议）PASS；生产级 completion 留给后续。**
