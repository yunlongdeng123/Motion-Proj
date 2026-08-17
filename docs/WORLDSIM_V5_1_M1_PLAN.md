# WorldSim V5.1 M1 执行登记

本文件是 V5.1 M1 的短执行入口；完整规范、方法树、门槛与第一轮约束以
`docs/WORLDSIM_V5_1_M1_TOPCONF_PLAN.md` 为唯一 normative plan。这里不复制长计划，只维护当前阶段、授权和证据，
避免两份计划发生漂移。

## 当前阶段（2026-08-17）

| Task ID | 状态 | 当前证据/下一门 |
|---|---|---|
| `WS-V51-P0-M1-SCOPE-FREEZE-01` | done | r001 start audit；scope/授权/quality locks exact |
| `WS-V51-D0-DEV-ROLE-FREEZE-01` | done | r001 start audit；H/S/C=`3/2/3` 与原 cohort exact |
| `WS-V51-M1-A-UNARY-OBSERVABILITY-01` | running | 159 个 canonical 文件 exact，下一门是 B0/B1/B3 posterior/metric replay |
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
- `scripts/audit_worldsim_v51_start.py`

正式状态、实验事实和失败事实仍分别以 `docs/RESEARCH_STATUS.md`、`docs/EXPERIMENTS.md` 与
`docs/RESEARCH_FAILURES.md` 为准。

## P0/D0 canonical evidence

- run：`/root/autodl-tmp/runs/worldsim_v51/WS-V51-P0-M1-SCOPE-FREEZE-01/20260817T101000Z__p0-start-audit-s0-r001`
- source commit：`58953a57557b97f449c4d83db7d11132ddda5e73`
- conclusion：`v51_m1_scope_roles_and_v5_inputs_frozen`
- summary/manifest SHA：`6d495ce26c211843e69dd9034dccfc916f17311dc59edaf5e7115ed32723ef9c /`
  `8ab0ad66eddedece7cfe6db4871172b07ae2c80430c8ddba156df76ce2941dc5`
- canonical inventory：r037/r042/r043=`65/44/50 files`，总计 `680,254,598 bytes`，逐文件 exact。
