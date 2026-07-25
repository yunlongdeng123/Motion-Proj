# N1 full-domain natural-event pool 正式报告

> **日期**：2026-07-24
> **任务**：`N1-EVENT-FULL-01`
> **终态**：`COMPLETE / n1_fulldomain_event_pool_pass`
> **结论范围**：nuScenes 官方 `val` split（146 scenes，排除全部 10 个 mini scene）；
> calibration 为 mini 三场景，仅作机制复核，不改写其 `REJECTED` 终态。

## 1. 一句话结论

在 scene-disjoint 的官方 val split（146 scenes、1,361 eligible vehicle actors）上，沿用 mini N1 的
全部冻结阈值、并把 interaction relation 从 exact-target-token 升级为预注册的 **graph-corridor
curvilinear coordinate** 后，得到 1,898 个稳定 transition、396 个 topology-pass candidate、
**37 个 positive、7 个 negative、7 个 same-actor pair、17 个 positive scenes**，四项样本量 gate 全部通过，
`N1-EVENT-FULL-01` 唯一终态为 `COMPLETE`。同一套代码在 calibration（mini）上**逐位复现** mini N1 的
71 transitions / 22 topology-pass / 0 positive，证明 corridor 升级没有把已 reject 的 mini pool 翻案。

## 2. Provenance

| 项目 | 值 |
|---|---|
| run | `v71_n1-event-full-01__fulldomain-v1__s0__20260724T081214528945Z__f12c886c` |
| 路径 | `/root/autodl-tmp/runs/event_first/N1-EVENT-FULL-01/v71_n1-event-full-01__fulldomain-v1__s0__20260724T081214528945Z__f12c886c/` |
| code commit | `2bac4c9`（dirty=true，diff hash `9b15fd2b…`） |
| config fingerprint | `f12c886cb6bb58171e7c5bf358809d88e3a15e1424005ce05d5fd335c561fcea` |
| event-pool SHA256 | `9635b5148ae58933db0c594605c1ae104fa891f065d565bef789a32139ba708c` |
| 父 N0 run | `…/N0-ASSET-01/v71_n0-asset-01__map-v1-3__s0__20260723T232427413355Z__e250cccd`（asset manifest `48e8ace8…`） |
| seed / interpolate_N | `0` / `4`（10 Hz） |
| terminal marker | 唯一 `COMPLETE` |

## 3. 数据底座与 provenance 一致性

- 轨迹来源不再是 DriveStudio 预处理，而是 `motion_proj/resim/nuscenes_trainval_tracks.py` 从
  `v1.0-trainval` 标注**流式**（ijson，2 GB cgroup 内）复刻 `save_objects` + `interpolate_boxes`
  (interpolate_N=4) 的 10 Hz 轨迹，**annotation-only、不触碰任何相机/LiDAR sweep**；
- 该 loader 已对 mini 三场景与 DriveStudio `processed_10Hz/mini` 逐位对拍：117/24/50 个 actor 全部按
  instance_token 命中，`frame_idx` 完全一致，`obj_to_world` 最大绝对误差 ~1e-15；
- 因此 mini 的帧级冻结阈值（`min_track_frames=20`、`min_stable=10`、`max_transition_gap=20`、
  `negative_window=30`）在 full-domain 上**同 cadence 原样沿用**，没有做任何阈值改动。

## 4. 冻结定义

结果查看前冻结（复用 mini commit `82117c7` 的 eligibility/map-matching/topology/gate，数值见
`configs/resim/event_first_n1_fulldomain_v1.yaml`），唯一预注册变更：

- **interaction relation：graph-corridor curvilinear coordinate**。沿 subject 所在 target lane 的
  有向 lane/connector 链，向 outgoing/incoming 各贪心展开 `graph_hops=2` 跳（每步取 min heading
  discontinuity），以累积弧长构造 corridor s 轴；邻车只要匹配到 corridor 链上的任一 token 即按其
  curvilinear 位置计入 front/rear，gap 阈值仍为冻结的 `[2, 60] m`。
- 它是 mini `exact-target-token` relation 的直接推广（`hops=0` 即退化为 exact token），针对 mini
  报告 §6 指出的“exact-token 把同一 longitudinal corridor 邻车切到相邻 token”的 fragmentation。
- calibration / evaluation 严格 scene-disjoint；formal verdict 只由 evaluation gate 决定。

## 5. 数值结果

### 5.1 Calibration（mini 三场景，只复核不翻案）

| scene | map | eligible actors | matched frac | topology pass | positive |
|---|---|---:|---:|---:|---:|
| scene-0655 (003) | boston-seaport | 7 | 0.8889 | 2 | 0 |
| scene-0757 (004) | boston-seaport | 16 | 0.9336 | 7 | 0 |
| scene-0796 (005) | singapore-queenstown | 22 | 0.9560 | 13 | 0 |
| total | | 45 | | 22 | 0 |

与 [`N1_MINI_EVENT_POOL_REPORT.md`](N1_MINI_EVENT_POOL_REPORT.md) 完全一致（45 eligible、71 transition、
22 topology-pass、0 positive）。**即使启用 graph-corridor relation，mini 仍 0 positive**——corridor
升级不会凭空制造正例，mini 的 `REJECTED` 终态不受影响。

### 5.2 Evaluation（val 146 scenes）

| 指标 | 值 |
|---|---:|
| eligible vehicle actors | 1,361 |
| median matched-pose fraction | 0.975 |
| transition candidates | 1,898 |
| topology-pass candidates | 396 |
| positive events | 37 |
| negative events | 7 |
| same-actor pairs | 7 |
| positive scenes | 17 |

scenes by map：boston-seaport 75、singapore-onenorth 35、singapore-queenstown 21、
singapore-hollandvillage 15。positive 分布在 17 个 scene（scene-0093 与 scene-0794 各 5–6 个）。
topology 类型：35 merge + 2 lane_change。

### 5.3 Gate

| gate | threshold | observed | result |
|---|---:|---:|---|
| positive events | ≥2 | 37 | pass |
| negative events | ≥2 | 7 | pass |
| same-actor pairs | ≥2 | 7 | pass |
| positive scenes | ≥2 | 17 | pass |
| UNKNOWN 不计 positive | required | enforced | pass |
| noninteractive 不计 positive | required | enforced | pass |

四项样本量 gate 全部通过 → `COMPLETE`。

## 6. 关键归因与诚实警示

- **corridor relation 是决定性变量**：37 个 positive 中 **36 个至少依赖 1 个 corridor 挽回的跨-token
  邻车**（front 或 rear 不在精确 target token 上），只有 1 个在 exact-token 下也成立。即：若在同一 val
  上改用 mini 的 exact-token relation，positive 数量约为 1，仍不过 gate。
- 因此本次 pass 是“**full-domain 规模 + graph-corridor relation**”两者共同作用：corridor 单独在 mini
  上仍 0（§5.1），大域单独在 exact-token 下也过不了 gate。二者缺一不可。
- 由此产生的风险：整个 verdict 高度依赖 corridor relation 的正确性与不过度宽松。缓解证据是它在 mini
  上仍产 0 positive，且入选 gap 均落在物理合理的 `[2, 60] m`；但这**不足以**替代人工核验。
- 已知局限：negative pool 仅对 positive actor 构造（7 个），规模小；positive 目前以 merge 为主
  （35/37），lane_change 仅 2；10 Hz 轨迹在关键帧间为线性/SLERP 插值，非真实观测。

## 7. 能下与不能下的结论

能下：

1. 在同域 full nuScenes（val）上，冻结定义 + 预注册 corridor relation 能形成含 positive、negative、
   same-actor pair 且跨多 scene 的可比较事件池；mini 的第一瓶颈（pool 规模/结构）在 full-domain 解除；
2. corridor 升级正是 mini 报告 §6 预注册的修复方向，且被证明不会翻案 mini。

不能下：

1. “37 个 positive 已经过人工核验为真实 cut-in/merge”——尚未；
2. “corridor relation 的宽松度已被证明恰当”——需人工审计与敏感性分析；
3. “可以直接进入 N2/N3/render/training”——按停止规则，N1 通过只解锁**请求 raw sweeps**，且本项目把
   人工审核列为停止条件。

## 8. 下一步（不在本 run 内执行）

1. **人工审计**：盲评抽样 positive（尤其依赖跨-token 邻车者），确认 corridor s 轴与 front/rear 判定
   物理正确；必要时对 `graph_hops` 做 scene-disjoint 敏感性分析。审计 verdict 只能由用户/指定评审填写。
2. **N2 raw 资产**：审计通过并获得传感器授权后，用 `scripts/extract_nuscenes_scenes.py` 按
   `configs/resim/n2_extract_scene_list.txt`（17 个 positive scene）只抽 CAM_FRONT + LIDAR_TOP 的
   samples+sweeps，再冻结独立 raw/map evaluator 运行 N2。
3. 不得以本 run 的 positive 直接充当 N2 的独立证据；N2 evaluator 必须独立于事件挖掘。

## 9. 证据文件

- `resolved.yaml`：冻结配置；
- `manifest.json`：commit、config fingerprint、calibration/evaluation scene 名单、seed；
- `calibration_audit.json`：mini 三场景复核（含逐场景 topology-pass 与 0 positive）；
- `evaluation_map_audit.json`：val 逐 scene eligible/matched/topology-pass/positive；
- `event_pool.json`：1,898 transitions、37 positive、7 negative、7 pair 与逐事件 hash；
- `metrics.jsonl`、`summary.json`、`COMPLETE`：汇总与唯一终态。
