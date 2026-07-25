# N1 kinematics-first 第三次人工 reject 归档

> 日期：2026-07-25  
> 结论：第三版 N1 `REJECTED`；第四版 receiver-centric N1 另立预注册；N2 未授权。

## 1. 不可变事实入口

Parent machine run：

`/root/autodl-tmp/runs/event_first/N1-EVENT-KINEMATIC-01/v71_n1-event-kinematic-01__kinematic-v1__s0__20260725T092427030639Z__8c2247b6/`

用户完成的 review：

`/root/autodl-tmp/runs/event_first/N1-EVENT-KINEMATIC-01/v71_n1-event-kinematic-01__kinematic-v1__s0__20260725T092427030639Z__8c2247b6/audit/review_working.jsonl`

Review SHA256：

`005cd74b874833808435fd2f47387d1d8e446cdea2d3a5cae6146e34bf331e96`

成功 adjudication：

`/root/autodl-tmp/runs/event_first/N1-EVENT-KINEMATIC-AUDIT-01/v71_n1-event-kinematic-audit-01__human-audit-reject-v1__s0__20260725T155754010881Z__4c51f0d9/`

- code commit：`1fbbbc1`；
- terminal：唯一 `REJECTED`；
- `n2_authorized=false`。

保留的工程失败：

`/root/autodl-tmp/runs/event_first/N1-EVENT-KINEMATIC-AUDIT-01/v71_n1-event-kinematic-audit-01__human-audit-reject-v1__s0__20260725T155523736677Z__4c51f0d9/`

- terminal：`FAILED`；
- reason：`engineering_manifest_key_mismatch`；
- 没有写入研究裁决；
- 修复后使用 `immutable_artifact_set_sha256`，并把输入校验前移。

## 2. 人工结果

- reviewed：12/12；
- TRUE_POSITIVE：0；
- FALSE_POSITIVE：12；
- UNCERTAIN：0；
- determinate precision：0；
- subject maneuver：INVALID 12/12；
- target corridor：VALID 12/12；
- front：INVALID 1；
- rear：INVALID 2。

Failure codes：

- `SUBJECT_NO_LATERAL_MANEUVER=12`；
- `ROUTE_CONTINUATION=11`；
- `NORMAL_TURN=1`；
- `REAR_INVALID=2`；
- `FRONT_INVALID=1`；
- `MAP_MATCH_JITTER=1`。

## 3. 根因

第三版 12/12 候选均为 `converging_branch_merge`。机器验证的是地图 branch 的收敛，不是 subject 车身的
横向 outside→inside。正常主路续接、道路弯曲、正常转弯和 map-match jitter 因而被误报。

target corridor 本身连续不代表角色语义正确。第三版会选择 subject source 最顺的 branch，再从相同队列取 rear：

- K3-004：front branch 错；
- K3-007、K3-010：rear branch 错；
- K3-012：map-match jitter；
- 其余主导模式：同一车流正常排队/路线续接。

machine support 也独立失败：negative=2、same-actor pair=2，低于冻结 4/4。人工真实性失败不能隐藏 pair
失败，pair 失败也不能代替人工真实性裁决。

## 4. 第四版接续

当前合同：

- [`../../../N1_RECEIVER_CUTIN_PREREGISTRATION.md`](../../../N1_RECEIVER_CUTIN_PREREGISTRATION.md)
- [`../../../RESEARCH_FAILURES.md`](../../../RESEARCH_FAILURES.md) 的 N1-F12–F15；
- [`../../../RESEARCH_STATUS.md`](../../../RESEARCH_STATUS.md)。

核心变化：

1. 原始 2 Hz subject center outside→post full-box inside；
2. pre heading alignment 排除大角度 route continuation；
3. parallel target 排除 source，merge 只用独立 direct incoming；
4. 同一最近 RECEIVER pre/post identity 与 bumper gap；
5. 30-frame negative 不缩短、不 overlap，并要求 receiver-matched control；
6. 49 条历史人审只作 calibration，全部 26 个 scene 从 formal train 排除；
7. 第四次用户裁决前 `n2_authorized=false`。

本归档不把第四版写成已通过；它只冻结第三次 reject 事实和防重复约束。
