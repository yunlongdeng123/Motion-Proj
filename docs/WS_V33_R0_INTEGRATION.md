# WorldSim V3.3 R0：完整集成与内容寻址发布包

- Task：`WS-V33-R0-INTEGRATION-01`
- 终态：`done / v33_supported`
- canonical：`20260810T222701Z__r0-integration-canonical-s0-r7`
- source baseline：`research/worldsim-v3.3-object-maintenance@1dff896`
- 训练/optimizer：`0`
- 新 checkpoint：`0`

## 1. 最终生产链

```text
V3.2 D2 immutable base
→ S1 O1 dual instance-opacity field
→ S2 B1 RoadPatch-Lite
→ S3 A4 auto-4view high-support actor
→ S4 posterior-gated spatial delta
→ S5 G0 raw-3D fail-safe renderer
→ V3.2 mixed persistent-storage reference
→ V3.3 exact content-addressed release
```

R0 验证 44 个 canonical 输入的 path/bytes/SHA/terminal/decision，随后再次执行 O1 instance-field、
RoadPatch delta 与 A4 actor-asset schema validator。D2 checkpoint=`578,819,674 bytes` 和 actor registry 只作
external exact reference；发布包不复制 `.pth/.pt/.ckpt/.safetensors`。

## 2. 十项最终回答

| # | 问题 | R0 回答 |
|---:|---|---|
| 1 | object field 是否改善 | 是。O1 相对 heuristic 的 boundary F1/IoU 相对提高 `387.47%/422.87%`，NBD/FP mass 相对下降 `27.37%/30.77%`；FN mass 增加 `0.048078`，明确披露，不声明全面支配 |
| 2 | RoadPatch 是否优于 V3.2 Telea | 不作 matched 排名。B1 在冻结 heldout 门内成立并取代 Telea 成为 V3.3 主方法，但两者 base/协议不同，没有伪造 head-to-head 胜负 |
| 3 | Inpaint360GS 能否单卡运行 | 当前官方合同不能。`blocked_single_3090`、`official_execution_attempted=false`；这是环境/权重/adapter 前置阻塞，不是质量失败 |
| 4 | view selection 是否改善 | high-support 是。A4 heldout IoU/boundary F1=`+0.023490/+0.059889`，四项 retention gate 通过；boundary actor 失败并 `ABSTAIN_GENERATED_OVERRIDE` |
| 5 | delta 是否 exact rollback | 是。5 views×4 stacks=`20/20` rollback exact，full replay exact，base/registry SHA exact，package 无完整 checkpoint 副本 |
| 6 | 是否避免 delete semantic reintroduction | 是，通过 fail-safe。unconstrained candidate 在 1/5 view 被 SAM2 标记；生产 G0 delete 5/5 pixel/SAM mass exact safe |
| 7 | provenance 是否完整 | 是。original base、learned occupancy、real patch reuse、generated actor、runtime erase、raw-3D insertion/delete 均有 typed provenance |
| 8 | 单卡 wall/VRAM/disk | S1–S5 选定链累计实测 wall=`379.552 s`；峰值为 S3 `20,137 MiB`；各阶段 OOM/kill=`0/0`；R0 release/run 见第 5 节 |
| 9 | rejected/blocked 是否完整 | 是。O3、dense RoadPatch、boundary A4、all-hard erase、S5 G1 均保留 rejected；SAM3.1/R3D2/Inpaint360GS/GOR-IS 均按外部状态记录 |
| 10 | unavailable 是否冒充算法失败 | 否。所有缺 source/weights/runtime 的路线都显式写成 `blocked/source_not_released/audit_only`，没有质量结论 |

## 3. 成功标准

计划的四个必须项全部满足：

1. object-aware field：O1 selected，base RGB checkpoint bitwise exact；
2. 3D-native background repair：104-row RoadPatch delta、native donor provenance、heldout gate accepted；
3. automatic actor view selection：1/2/4-view 全运行，development 选择 A4，heldout 只确认；
4. spatial delta：ERASE/INSERT_BACKGROUND/INSERT_ACTOR/RENDER_ONLY、20/20 rollback exact、base immutable。

因此整体终态为 `v33_supported`。S5 G1 未获 production 增益、Inpaint360GS/R3D2/SAM3.1 未解锁等加分项
不改变这四个必须项，也不会被隐藏。

## 4. 发布包

```text
/root/autodl-tmp/runs/worldsim_v33/WS-V33-R0-INTEGRATION-01/
20260810T222701Z__r0-integration-canonical-s0-r7/artifacts/
├── worldsim_v33_release/
├── worldsim_v33_release.zip
└── worldsim_v33_release_replay/
```

release 包含：

- O1 `instance_field.npz`（`1,309,868` Gaussian）；
- B1 `roadpatch_delta.npz`（`104` rows）；
- A4 actor asset（`99,241` rows）；
- S4 14-file spatial-delta package；
- S5 5 views×insertion/delete production PNG；
- V3.2 exact chunk manifest reference；
- 39 份 canonical JSON evidence；
- asset/decision/provenance/resource/claims ledgers；
- 可独立运行的 `tools/verify_release.py`。

内容清单=`76 files / 18,432,994 payload bytes`，manifest SHA=
`e386c14b6b29c74bd1316a31a3abefedf10a74530cfe3149cf9e040eb78a6c53`，完整 checkpoint copy=`0`。
归档在同一 run 内独立构建两次，SHA 都为：

```text
cffaad16e2d14e8274c41bb48b24be64c73d9fb6f41d1fe4792934adeab244a7
```

archive bytes=`13,760,114`。解包后再执行 file-set/bytes/SHA/forbidden-suffix 检查，得到相同 manifest SHA；
formal 前 diagnostic 已冻结同一 archive SHA。

离线复验：

```bash
/root/autodl-tmp/envs/drivestudio/bin/python \
  artifacts/worldsim_v33_release/tools/verify_release.py \
  verify-archive artifacts/worldsim_v33_release.zip
```

## 5. R0 自身资源与终端证据

| 项目 | 数值 |
|---|---:|
| R0 wall | `2.721847 s` |
| R0 GPU compute processes max | `0` |
| R0 cgroup peak（含容器既有 page cache） | `39,614,062,592 bytes` |
| R0 run bytes | `50,851,476` |
| OOM / oom_kill delta | `0 / 0` |
| source snapshots | `7/7` exact |
| gates | `10/10` passed |

| 产物 | SHA-256 |
|---|---|
| config | `4b4a20b95c2cd9803d2087128dca4942344e7e0a6ac1669b71e108c0e11273a9` |
| summary | `c19032559796377d28073ce14584ce086a0d6ec8b20c598069fe15ae391ca2b2` |
| status | `0a1396f45a063df6ae60bc8ba56378d89df20651a4074c157a5babbc18f09aa4` |
| release archive | `cffaad16e2d14e8274c41bb48b24be64c73d9fb6f41d1fe4792934adeab244a7` |
| content manifest | `e386c14b6b29c74bd1316a31a3abefedf10a74530cfe3149cf9e040eb78a6c53` |

## 6. diagnostic 失败与 canonical 选择

- `222435`：S2 empty heldout schema 被 R0 config 误写成数值 0；
- `222453`：S3 heldout phase 枚举误写为 `heldout_confirmation`，真实冻结值为 `heldout`；
- `222511`：S4 status stage 枚举误写为 `evaluation`，真实值为 `real_renderer_evaluation`；
- `222526`：R0 报告层错误要求 instance field 必含 `schema_version`，而正式 validator 已通过且该 NPZ 无此字段；
- `222549`：release builder 未先创建 `tools/` 目录；
- `222610`：diagnostic 首次 10/10 gates 通过，冻结 archive SHA；
- `222701`：formal config 带 expected archive SHA，canonical 通过。

所有失败 run 都是独立 terminal=`failed`，没有覆盖或续写。

## 7. 适用边界

- 主证据为 scene-0230、冻结 actor/视图与单 RTX 3090；
- scene-0242/0255 没有同协议完整 V3.3 S1–S5 资产链，不外推三场景 production；
- A4 生成背面只支持 completeness/consistency，不是 GT；
- S5 冻结视图不相邻，temporal consistency 未评估；
- 没有闭环控制、安全、传感器仿真真实性或大规模 benchmark claim；
- LiDAR-EVS 仍是 R0 后 conditional audit，不属于本次交付。
