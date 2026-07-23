# N1 mini natural-event pool 正式报告

> **日期**：2026-07-24
> **任务**：`N1-EVENT-01`
> **终态**：`REJECTED / reject_mini_event_pool`
> **结论范围**：nuScenes mini processed scenes 003/005/004；不外推到 full nuScenes、nuPlan 或 Waymo。

## 1. 一句话结论

冻结的 actor、map matching、topology 与 exact-target-token front/rear interaction 定义在三个 mini scenes
中得到 71 个稳定 token transition、22 个 topology-pass event candidates，但 22/22 都没有同时满足冻结
front/rear relation，因此没有 positive、可配对 negative 或 same-actor pair。N1 按预注册立即 reject，
N2/N3/render/training 均未触发。

## 2. Provenance

| 项目 | 值 |
|---|---|
| run | `v71_n1-event-01__mini-event-v1__s0__20260723T232920917536Z__cd56b326` |
| 路径 | `/root/autodl-tmp/runs/event_first/N1-EVENT-01/v71_n1-event-01__mini-event-v1__s0__20260723T232920917536Z__cd56b326/` |
| code commit | `82117c7ec58db9bbe7e26d0f866442b620b617f6` |
| code dirty | `false` |
| config fingerprint | `cd56b326cd38ecda3ab6dd36bb31a38ce03f14aacbef7e199ff8f073558f5cf3` |
| data fingerprint | `919b08593a5fdf13e668714865cc2f1d2129f5bf221a3d7c3ad54af80ccbc0a3` |
| event-pool SHA256 | `6f39cc8b917c277adfc9a8b17130c4d5d1e762beb862cbaf51c92b61727dc792` |
| split / seed | `nuscenes-mini-scenes-003-004-005` / `0` |
| terminal marker | 唯一 `REJECTED` |

父资产 run：

`/root/autodl-tmp/runs/event_first/N0-ASSET-01/v71_n0-asset-01__map-v1-3__s0__20260723T232427413355Z__e250cccd/`

## 3. 冻结定义

结果查看前，commit `82117c7` 已冻结：

- source-only eligibility：`vehicle.*`、track ≥20 frames、0 gaps、首尾位移 ≥5 m；
- map matching：官方 v1.3 centerline、0.5 m 离散、最近 16 点、距离 ≤3 m、heading error ≤45°；
- source/target 各稳定 ≥10 frames，transition gap ≤20 frames；
- lane change：1.5–6 m parallel corridor shift、heading error ≤30°，且 2-hop graph 有共享前驱/后继；
- merge：target 在 source 的 2-hop directed closure 内，target incoming lanes ≥2；
- interaction：稳定进入 target 后，在**精确 target token**上同时存在 2–60 m 的 front 与 rear actor；
- gate：每 scene ≥2 eligible actors；positive/negative 各 ≥2；same-actor pairs ≥2；positive scenes ≥2；
- UNKNOWN、单侧邻车和 noninteractive transition 都不能算 positive。

失败恢复明确禁止降低阈值、删除 scene、后验换 actor、把 UNKNOWN 并入 positive 或继续 N2–N5。

## 4. 数值结果

### 4.1 Eligibility 与 map matching

| scene | eligible actors | matched poses | eligible poses | match fraction |
|---|---:|---:|---:|---:|
| 003 | 7 | 464 | 522 | 0.8889 |
| 005 | 22 | 1,976 | 2,067 | 0.9560 |
| 004 | 16 | 1,238 | 1,326 | 0.9336 |
| total | 45 | 3,678 | 3,915 | 0.9395 |

Eligibility 拒绝原因不是事件标签：003/005/004 分别有 107/17/5 个 actor 因位移不足 5 m 未进入；
class、短 track 也分别触发。它说明 mini 特别是 003 含大量静止/近静止标注对象。

### 4.2 Transition taxonomy

| 类型 | 003 | 005 | 004 | total | topology pass |
|---|---:|---:|---:|---:|---:|
| route continuation | 6 | 18 | 15 | 39 | 0 |
| merge | 1 | 12 | 6 | 19 | 19 |
| lane change | 1 | 1 | 1 | 3 | 3 |
| unresolved | 2 | 6 | 2 | 10 | 0 |
| total | 10 | 37 | 24 | 71 | 22 |

所有 71 个 transition 的 interaction verdict 都是 FAIL。22 个 topology-pass cases 中：

- 18 个在 relation frame 的精确 target token 上没有邻车；
- 4 个只有 front、没有 rear；其中两个 front gap 约 82.66/89.33 m，超过冻结 60 m，另外两个约
  16.00/13.94 m；
- 0 个同时有合格 front 和 rear。

### 4.3 Gate

| gate | threshold | observed | result |
|---|---:|---:|---|
| eligible actors / scene | ≥2 | 7 / 22 / 16 | pass |
| positive events | ≥2 | 0 | fail |
| negative events | ≥2 | 0 | fail |
| same-actor pairs | ≥2 | 0 | fail |
| positive scenes | ≥2 | 0 | fail |
| UNKNOWN/noninteractive 不计 positive | required | enforced | pass |

Negative pool 只为已经有 positive 的 actor 构造 comparable stable-interaction window；所以
`negative=0` 是“没有正例 actor 可配对”的结构结果，不等于三场景不存在任何普通行驶片段。

## 5. 能下与不能下的结论

能下：

1. 该 mini split 不支持冻结的交互事件池和 same-actor comparison；
2. 旧 H1 的 0 positive 不是仅缺 map 文件；补足官方地图后，interaction support 仍为零；
3. N2 independent evidence、N3 proposal、GS render 和 utility training 没有合法比较对象，必须停止；
4. 事件池规模/结构是当前第一瓶颈，不是 GPU、renderer 或 occupancy 实现。

不能下：

1. “mini 中绝对没有任何人类可识别 cut-in/merge”；
2. “exact target token 是唯一正确的 longitudinal relation”；
3. “只要把 rear 要求删掉就能形成有效正例”；
4. “full nuScenes、nuPlan 或 Waymo 也会失败”；
5. “0 positive 证明生成方法无效”——N1 尚未运行生成方法。

## 6. 下一条可突破方向

首选不是复开本 run，而是 `full-domain event-pool`：

1. 获取 nuScenes `v1.0-trainval` annotations/metadata；先不下载 sweeps、相机数据或模型权重；
2. 把 22 个 topology-pass mini candidates 只作为 calibration/audit 池，检查 exact-token 是否把同一
   longitudinal corridor 的邻车切到相邻 token；不得回填本次 verdict；
3. 实现 graph-corridor curvilinear coordinate：沿 directed lane/connector chain 计算 front/rear，而非
   后验扩大欧氏半径；
4. 将 calibration scenes 与 formal evaluation scenes 严格分离，再冻结 corridor relation 与事件 gate；
5. 在 full trainval 先跑 annotation+map-only N1；只有 positive、negative、same-actor pair 和多 scene
   coverage 都通过，才请求 sensor sweeps 并进入 N2。

官方 nuScenes 提供 1,000 个约 20 秒 scenes，其中 850 个为 train/val，规模远高于当前 3 scenes，且保持
同一 schema/map/devkit（[nuScenes 官方数据说明](https://www.nuscenes.org/nuscenes)）。它比立即切换
nuPlan/Waymo 更少引入 domain、许可和适配变量。若 full nuScenes 仍不足，再评估
nuPlan/Waymo/ScenarioNet 的最小 shard。

## 7. 证据文件

- `resolved.yaml`：冻结配置；
- `manifest.json`：commit、fingerprint、split、seed；
- `actor_eligibility.json`：所有 actor 的 source-only eligibility；
- `map_match_audit.json`：逐 scene coverage；
- `event_pool.json`：71 transitions、event labels 与 hashes；
- `metrics.jsonl`、`summary.json`、`REJECTED`：汇总和唯一终态。
