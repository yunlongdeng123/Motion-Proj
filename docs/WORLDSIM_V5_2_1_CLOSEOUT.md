# WorldSim V5.2.1 Closeout

## Terminal

- badcase basis frozen：yes
- coverage：`complete_full`
- problem definition：RGB global/actor/boundary 已冻结；geometry/occlusion/identity/true-temporal 是明确 evidence gaps
- M1/M2/M3：`M1_EVIDENCE_INSUFFICIENT_KEEP_PENDING / M2_EVIDENCE_INSUFFICIENT / M3_EVIDENCE_INSUFFICIENT_KEEP_PENDING`
- algorithm candidate：`0`
- fresh validation/test/KITTI：`unread`
- Stage H/BKI：`not executed`

## Go / NoGo

当前对直接进入 V5.2.2 算法结构设计为 **NoGo**。下一步应先建立 base RGB 与 M1 ownership 的 exact same-view overlap，
并冻结 actor identity、visibility/occlusion、可比 depth 与 correspondence evidence；完成前不得再次更换传播核或图扩散方式。

最终机器证据位于 `/root/autodl-tmp/runs/worldsim_v521/20260820T113000Z__p10-base-badcase-closeout-s0-r001`。

## P11 人工归因补充（2026-08-20）

P10 原始 closeout 与 registries 保持不可变。18-case 人工复核在其上新增 P11 attribution layer，冻结
`9 BASE_FAILURE + 8 M123_ELIGIBLE + 1 ATTRIBUTION_UNRESOLVED`，并将 eligible 分成 `5 Discovery design + 3 one-shot
Confirmation`。该层使“编写 V5.2 因果桥/自动研究计划”转为 **Go**，但不把“直接训练完整 TrackBayes”改成 Go。

下一执行入口是 [`WORLDSIM_V5_2_M123_AUTORESEARCH_PLAN.md`](WORLDSIM_V5_2_M123_AUTORESEARCH_PLAN.md) 的 R0/R1：
先通过 Base Validity 与 exact pixel→Gaussian/temporal correspondence bridge，再按门禁自动进入 M1/M3；M2 只作 safety。
完整归因机器证据位于 `docs/run_manifests/worldsim-v5.2.1-human-review-attribution-v1/`。
