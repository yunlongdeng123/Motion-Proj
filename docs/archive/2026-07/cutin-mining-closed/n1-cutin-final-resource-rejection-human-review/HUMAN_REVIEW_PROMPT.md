# Human Review Prompt：N1 cut-in final 资源拒绝

## 目的与非目标

请确认 final cut-in plan 是否因**实际、冻结且可复核的资源合同失败**而正确终止。

这不是 cut-in candidate 人工标注：没有 prospective evaluation scene、machine candidate 或待填的 TP/FP verdict。
请勿根据旧第四轮 panel、K4 calibration 或个人判断补写 candidate 结论；本复核也不会授权 N2。

## 冻结判断

应被复核的终局是：

```text
REJECTED / stop_nuscenes_cutin_mining
```

前提是 final formal parent 在 clean code 上先完成 K4 regression，随后于任何 evaluation scene 前因：

```text
cgroup_memory_current_bytes = 1,523,929,088
max_start_cgroup_current_bytes = 1,350,000,000
excess = 173,929,088 bytes
```

而失败。它不应被误述为数据集没有 cut-in，也不应被 development override 或 K4 regression 覆盖。

## 盲法与材料边界

本包不隐藏 machine/资源状态，因为复核对象正是资源合同；它也不提供 candidate label、K4 标签或旧 reviewer
notes 作为本次 verdict 输入。请只核对下列文件及其哈希。

| 文件 | SHA256 |
|---|---|
| parent `preflight.json` | `974e80e78fe1bde1be014faea42549f066beb8b0856fbefb907fae7e8ee3eff3` |
| parent `failure.json` | `352daccce72a63f38b344f5e43a4fb8b6c208be4cfa7b7c683732292e1e9fbb8` |
| parent `stages/K4_REGRESSION.json` | `f15caefe193a582727537c266be57c7d851cf08699e6951ee5d8b92218189599` |
| rejection `summary.json` | `db9b9e8768a9df6637cb2c5d6a25506af9cf75d7e73c5aee76073512c9f38ff9` |
| rejection `manifest.json` | `1777cf295dd2b2449b8c3a003142544bf4955c05a970544cfad5e728672dcac4` |

## 复核步骤

1. 打开 parent formal 的 `preflight.json`，确认 `formal=true`、`code_dirty=false`、commit 为
   `7104f5c…`，并核对 runner RSS、cgroup current 与冻结 start 上限；
2. 打开 parent 的 `stages/K4_REGRESSION.json`，确认 K4 regression 为通过且 `n2_authorized=false`；
3. 打开 parent 的 `failure.json`，确认 worker 在 mining 前失败，且不存在 `scene_metrics.jsonl`、
   `strict_candidates.jsonl` 或 audit 产物；
4. 打开 independent rejection 的 `parent_evidence.json`、`summary.json` 和 `manifest.json`，确认它固定了父
   run 的 SHA256，并给出 `REJECTED/stop_nuscenes_cutin_mining`、0 evaluation scene、0 candidate、
   `n2_authorized=false`；
5. 在 `review_working.jsonl` 中仅填写资源复核的 reviewer、时间、结论、五项检查与备注。

可在远端运行的只读核验命令：

```bash
PARENT=/root/autodl-tmp/runs/event_first/N1-EVENT-CUTIN-FINAL-01/v71_n1-event-cutin-final-01__receiver-cutin-final-v1__s0__20260726T121042935837Z__a6b12de0
REJECTION=/root/autodl-tmp/runs/event_first/N1-EVENT-CUTIN-FINAL-RESOURCE-AUDIT-01/v71_n1-event-cutin-final-resource-audit-01__resource-contract-reject-v1__s0__20260726T121624740059Z__025850f8
sha256sum "$PARENT/preflight.json" "$PARENT/failure.json" "$PARENT/stages/K4_REGRESSION.json" "$REJECTION/summary.json" "$REJECTION/manifest.json"
sed -n '1,220p' "$PARENT/preflight.json" "$PARENT/failure.json" "$REJECTION/summary.json"
```

## 可填写结论

- `CONFIRM_RESOURCE_REJECTION`：证据证明冻结 start 合同失败，且终止没有绕过计划；
- `REQUEST_NEW_AUTHORITY`：证据无法复核，或人类希望在不同资源条件下提出一项**新的**研究任务；
- `UNABLE_TO_VERIFY`：当前材料无法完成资源证据核验。

`REQUEST_NEW_AUTHORITY` 不会自动重开本计划、改写本次 `REJECTED`，也不会授权 N2。未来重跑需要用户明确批准新的
资源方案、预注册、run ID 和 scene-disjoint 评估。

## JSONL 合同

每行必须是一个 JSON object，保留 `review_id` 和所有五项检查。`reviewer`、`reviewed_at_utc`、
`resource_contract_verdict` 和 `notes` 必须由人类填写。不要删除冻结 SHA256，不要改变 `n2_authorized`。

完成后请将填写后的 `review_working.jsonl` 连同本包路径交还给用户/项目负责人；它是人工确认记录，不是自动
adjudication 或 N2 授权。
