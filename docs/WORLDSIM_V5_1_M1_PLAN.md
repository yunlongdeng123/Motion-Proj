# WorldSim V5.1 M1 执行登记

本文件是 V5.1 M1 的短执行入口；完整规范、方法树、门槛与第一轮约束以
`docs/WORLDSIM_V5_1_M1_TOPCONF_PLAN.md` 为唯一 normative plan。这里不复制长计划，只维护当前阶段、授权和证据，
避免两份计划发生漂移。

## 当前阶段（2026-08-17）

| Task ID | 状态 | 当前证据/下一门 |
|---|---|---|
| `WS-V51-P0-M1-SCOPE-FREEZE-01` | done | r001 start audit；scope/授权/quality locks exact |
| `WS-V51-D0-DEV-ROLE-FREEZE-01` | done | r001 start audit；H/S/C=`3/2/3` 与原 cohort exact |
| `WS-V51-M1-A-UNARY-OBSERVABILITY-01` | running | Stage A H closed；S screening frozen；下一门 one-shot S SAM/evaluation |
| `WS-V51-M1-B-LUDVIG-UPLIFT-01` | pending/locked | Stage A 收口前禁止启动 |
| `WS-V51-M2` | pending | 未授权 |
| `WS-V51-M3` | pending | 未授权 |

## 第一轮授权

只允许 P0、development role freeze 与 Stage A。Historical diagnostic=`0471/1087/0379`，
screening=`0998/0359`，development confirmation=`0875/0535/0436`。V5 的 8-scene validation 与
20-scene test 继续不可读；KITTI 不用于方法调参。

## Failure ledger 绑定

- scope/data/protocol：`V5-F09`、`V5-F11`–`V5-F14`、`V5-F18`；
- unary/evaluation：`V5-F20`–`V5-F26`、`V5-F29`–`V5-F33`；
- 本轮 freeze 实现尚无新增 failure，`failure_ledger_delta=none`；每个正式 run 收口时重新复核。

## 配置与入口

- `configs/worldsim_v51/p0_m1_scope_v1.yaml`
- `configs/worldsim_v51/development_roles_v1.yaml`
- `configs/worldsim_v51/m1_unary_baselines_v1.yaml`
- `configs/worldsim_v51/m1_unary_visibility_v1.yaml`
- `configs/worldsim_v51/m1_unary_unknown_v1.yaml`
- `configs/worldsim_v51/m1_effective_count_audit_v1.yaml`
- `configs/worldsim_v51/m1_effective_count_audit_v2.yaml`
- `configs/worldsim_v51/m1_cif_decoupling_audit_v1.yaml`
- `configs/worldsim_v51/stage_a_screening_freeze_v1.yaml`
- `configs/worldsim_v51/m1_sam_screening_scene0998_v1.yaml`
- `configs/worldsim_v51/m1_sam_screening_scene0359_v1.yaml`
- `scripts/audit_worldsim_v51_start.py`
- `scripts/replay_worldsim_v51_v5_unary.py`
- `scripts/run_worldsim_v51_unary_visibility.py`
- `scripts/run_worldsim_v51_unary_unknown.py`
- `scripts/audit_worldsim_v51_effective_count.py`
- `scripts/audit_worldsim_v51_cif_decoupling.py`

正式状态、实验事实和失败事实仍分别以 `docs/RESEARCH_STATUS.md`、`docs/EXPERIMENTS.md` 与
`docs/RESEARCH_FAILURES.md` 为准。

## P0/D0 canonical evidence

- run：`/root/autodl-tmp/runs/worldsim_v51/WS-V51-P0-M1-SCOPE-FREEZE-01/20260817T101000Z__p0-start-audit-s0-r001`
- source commit：`58953a57557b97f449c4d83db7d11132ddda5e73`
- conclusion：`v51_m1_scope_roles_and_v5_inputs_frozen`
- summary/manifest SHA：`6d495ce26c211843e69dd9034dccfc916f17311dc59edaf5e7115ed32723ef9c /`
  `8ab0ad66eddedece7cfe6db4871172b07ae2c80430c8ddba156df76ce2941dc5`
- canonical inventory：r037/r042/r043=`65/44/50 files`，总计 `680,254,598 bytes`，逐文件 exact。

## Stage A A0 canonical evidence

- run：`/root/autodl-tmp/runs/worldsim_v51/WS-V51-M1-A-UNARY-OBSERVABILITY-01/20260817T102000Z__m1-a0-v5-unary-replay-s20260814-r001`
- source commit：`1e2361658b85e1f12145867164238ce81ecb55ea`
- conclusion：`a0_v5_b0_b1_b3_posterior_and_gaussian_metrics_bit_exact`
- exact 分母：45 observation files；9 arm×scene；54 posterior/statistic arrays；54 Gaussian metric values。
- exact 结果：array mismatch=`0`；metric delta=`0.0`；每场 12 个核心 generation source SHA exact。
- 边界：GPU renderer 未重跑；2D evaluation 只做 canonical artifact/source identity 复核。
- 下一门：A1 visibility mask。A2/A3/A4、Stage B/C 与 Graph 均保持锁定。

## Stage A A1 canonical evidence

- run：`/root/autodl-tmp/runs/worldsim_v51/WS-V51-M1-A-UNARY-OBSERVABILITY-01/20260817T104000Z__m1-a1-visibility-h-s20260814-r002`
- source commit：`38bc9b44c6c86d58173930aa019745b8a9a8e00b`
- B3 replay：12/12 GPU renders byte exact，3/3 checkpoints exact。
- A1 H gate：BF1 positive scenes=`2/3`；mean ΔBF1/IoU/FN/Brier/ECE=
  `+0.001155713/+0.000460310/+0.001105687/-0.000013972/-0.000144854`；五项 gate 全通过。
- 边界：1087 只有 1 个 accepted view，且整体效应小；A1 只是未经过 S 的 candidate。
- 下一门：只做 A2 UNKNOWN/ABSTAIN；A3/A4、S 与后续 Stage 保持锁定。

## Stage A A2 canonical evidence

- run：`/root/autodl-tmp/runs/worldsim_v51/WS-V51-M1-A-UNARY-OBSERVABILITY-01/20260817T113000Z__m1-a2-unknown-h-s20260814-r003`
- source commit：`7e783f1fe04cc05cdd206b56086ef0f02a4215ee`
- threshold freeze：positive-count H/A1 pooled Gaussian=`944,443`；Q25 count/Q75 entropy/Q75 disagreement exact。
- parent replay：A1 conditional render=`12/12 byte exact`；conditional metric delta A2−A1=`0`；checkpoint=`3/3 exact`。
- selective H gate：scene-balanced coverage=`0.7199625`；accepted/abstained error=`0.0148914/0.164250`；
  error separation=`+0.149358`；全部 checks PASS。
- 边界：0471 coverage=`0.464865`，低于逐场 60%；冻结 gate 是 scene-balanced mean，A2 只作 H candidate。
- 下一门：只做 A3 correlation-aware effective count；A4、S 与后续 Stage 保持锁定。

## Stage A A3 canonical mechanism audit

- r004：relative change 被 subnormal reliability denominator 污染，保留 `done/inconclusive`，见 `V51-F06`。
- canonical r005：`/root/autodl-tmp/runs/worldsim_v51/WS-V51-M1-A-UNARY-OBSERVABILITY-01/20260817T120000Z__m1-a3-effective-count-audit-v2-s20260814-r005`
- source commit：`150b0721cb0e0acf630846d815a7ab2f287ceea8`
- parent exact：45 observations；3/3 A2 effective-count arrays float32 exact；observed Gaussian=`944,443`。
- mechanism result：no-epsilon Kish below fractional mass=`0`；meaningful absolute cap change=`0`；replacement
  amplification=`940,762/944,443`。公式没有 correlation observable，A3 rejected，不启动 quality/GPU arm。
- 下一门：只做 A4 CIF-style visibility/occupancy/conditional-identity decoupling；S 与后续 Stage 保持锁定。

## Stage A A4 canonical identifiability audit

- canonical r006：`/root/autodl-tmp/runs/worldsim_v51/WS-V51-M1-A-UNARY-OBSERVABILITY-01/20260817T122000Z__m1-a4-cif-identifiability-audit-s20260814-r006`
- source commit：`cee8b66849e5c556a79e05c813d45f225efa7814`
- A2 occupancy field=`0/3`；constant occupancy=1 与现有 renderer=`3/3 bit exact`；appearance opacity reuse=
  `3/3 non-exact`。无独立 occupancy observable，A4 在 quality read/GPU/training 前 rejected。
- Stage A H retained candidates=`A1,A2`；A3/A4 不进入候选。下一步必须 freeze-only 绑定 S screening，再一次性读取
  S=`0998/0359`；C/validation/test/KITTI 继续锁定。

## Stage A S screening freeze

- candidates=`A1/A2`；S=`0998/0359`；seed=`20260814`；SAM/split/checkpoint/operator 与 H exact inherited。
- gate=`2/2 BF1 nonnegative + >=1/2 delta>=0.001 + mean BF1>0 + mean FN<=+0.02 + calibration caps`；A2
  另需 mean coverage>=0.60 与 error concentration。
- S 后最多保留一臂；优先 A2 selective PASS，否则 A1 conditional PASS，否则回退 U2。
- r047/r048 与 one-shot read policy 见 `docs/WS_V51_STAGE_A_SCREENING_FREEZE.md`；执行前 S quality unread。
