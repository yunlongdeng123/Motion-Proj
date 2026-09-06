# WorldSim V5.1 P0 M1 Scope 与 Development Roles Freeze

## 结论

`WS-V51-P0-M1-SCOPE-FREEZE-01` 与 `WS-V51-D0-DEV-ROLE-FREEZE-01` 已完成。V5.1 从 immutable V5
canonical `44d0e4a2468112b89a454992ecd9177d65184067` 分支启动，只授权 M1 P0/D0/Stage A；M2/M3 和
Stage B 以后路线均未解锁。

## Canonical run

```text
/root/autodl-tmp/runs/worldsim_v51/WS-V51-P0-M1-SCOPE-FREEZE-01/
20260817T101000Z__p0-start-audit-s0-r001
```

- source commit：`58953a57557b97f449c4d83db7d11132ddda5e73`
- conclusion：`v51_m1_scope_roles_and_v5_inputs_frozen`
- normative plan SHA：`3d7f74813db79b00b71b522391ff2d953bb480ef6a25179c1919ef3acc54d733`
- V5 cohort file/cohort SHA：`6d8caf1973f090248de237b01788d086bf52ef41c294726b80fd6590a357e7f3 /`
  `553373159023218b44615be27aeeb5533a6c585be276e06425235fe09b6b48b1`

## Development roles

| Role | Frozen scenes | 使用规则 |
|---|---|---|
| H historical diagnostic | 0471, 1087, 0379 | debug、机制诊断、消融与阈值冻结；不能单独宣称 generalization |
| S screening | 0998, 0359 | frozen candidate exact-once；不得回头调参 |
| C development confirmation | 0875, 0535, 0436 | family/config 全冻结后 exact-once |

V5 8-scene validation 与 20-scene test 保持 quality unread；KITTI method tuning=false。

## V5 canonical 输入审计

| Scene / Run | Manifest files | Bytes | Checkpoint SHA | Summary / Manifest SHA |
|---|---:|---:|---|---|
| 0471 / r037 | 65 | 235,584,726 | `496356ca...` | `dd8b2a9e... / 80ff775d...` |
| 1087 / r042 | 44 | 184,193,494 | `84c34b83...` | `d19cabd9... / 77b3965b...` |
| 0379 / r043 | 50 | 260,476,378 | `d77fa13f...` | `1beff3d9... / 4cee48fd...` |

总计 `159 files / 680,254,598 bytes`，全部逐 bytes/SHA exact。r001 自身
summary/status/fingerprint/manifest SHA：

```text
6d495ce26c211843e69dd9034dccfc916f17311dc59edaf5e7115ed32723ef9c
8a724b06563ff1cc4181f0760db9dc0013fc9897d7a38a3d3bdc08005fd1bd93
b52b63d342034fa9c2fabe858ad0f1d18d5ee6d67e9c67472e82c725aa643958
8ab0ad66eddedece7cfe6db4871172b07ae2c80430c8ddba156df76ce2941dc5
```

## Failure ledger 与边界

- `failure_ledger_refs`：`V5-F09/F11–F14/F18/F20–F26/F29–F33`。
- formal start audit 的 `failure_ledger_delta=none`。
- 实现窄测曾在 collection 阶段触发 repo-root import blocked；已修复并登记 `V51-F01=resolved`。
- 没有方法推理、训练、参数搜索、validation/test quality read 或 KITTI tuning。

## 下一门

只执行 Stage A A0：从 r037/r042/r043 的冻结 observation 重算 B0/B1/B3 posterior 与统计，要求 canonical
Gaussian table bit-exact，并复核 Gaussian metrics 与 evaluation artifact identity。A0 通过前不实现 A1，不接触 Graph。
