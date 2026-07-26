# N1-KINEMATIC-01 第三版预注册：先运动学、后交互

> **预注册日期**：2026-07-25
> **任务 ID**：`N1-EVENT-KINEMATIC-01`
> **当前状态**：`pending formal evaluation`
> **硬边界**：只执行 N1；无论结果如何均 `n2_authorized=false`
> **冻结配置**：`configs/resim/event_first_n1_kinematic_v1.yaml`

## 1. 为什么第三版不是重复调参

第二次 full-domain N1 的 parent machine run 在 val 146 上给出 37 个 positive，但用户确认的人审结果为
2 TP / 35 FP，独立 adjudication 已唯一 `REJECTED`。失败不是 gap 或 graph hops 数值差一点，而是判定对象错位：

- `N1-F05`：target 有多个 incoming 是地图节点属性，不能证明当前 subject 正在 merge；
- `N1-F06`：2 Hz annotation 插值到 10 Hz 只增加采样点，不增加物理观测；
- `N1-F07`：单 relation frame 的中心距不能证明 front/rear identity 持续；
- `N1-F08`：旧 panel 缺 subject/front/rear identity overlay，且没有预注册聚合阈值；
- `N1-F01/F02`：exact-token fragmentation 的确存在，但 graph-corridor 只能修邻车表示，不能替代主体行为；
- `H1-F01/V7-RISK-16`：先证明真实事件与可比较对象存在，再进入后续证据或生成。

第三版改变的是因果顺序和 evidence contract：

```text
2 Hz subject kinematics
  → actor-specific lane crossing / converging branch
    → branch-safe target corridor
      → temporally persistent front + rear
        → blind human verdict
```

它不通过调低旧阈值、删 scene、挑 actor 或扩大 gap 翻案。旧 val 人审只用于设计/校准；formal evaluation 改用
scene-disjoint official train。

## 2. 数据与 split

### 2.1 Calibration（已查看标签）

- parent：
  `/root/autodl-tmp/runs/event_first/N1-EVENT-FULL-01/v71_n1-event-full-01__fulldomain-v1__s0__20260724T081214528945Z__f12c886c/`
- audit reject：
  `/root/autodl-tmp/runs/event_first/N1-EVENT-FULL-AUDIT-01/v71_n1-event-full-audit-01__human-audit-reject-v1__s0__20260725T083929632491Z__6507cbac/`
- completed review SHA256：
  `ae71b31e02faf1d783c36748e629e85acf32a132f35ce2f98102a5f62201dd05`
- 37 events / 17 val scenes，只用于冻结第三版表示和阈值，不进入 formal metric。

冻结前的 calibration 结果诚实记录为：

- 35/35 已标 FP 被第三版 subject-motion gate 拒绝；
- 2 个已标 TP 中保留 1 个、拒绝 1 个；
- calibration TP recall=0.5、FP rejection=1.0。

拒绝一个旧 TP 是刻意的 precision-first 取舍：其记录 transition 在原始 2 Hz world track 上呈主路正常续接，
旧无身份框 panel 可能把画面中另一辆变道车认成 subject。第三版不以保住旧 TP 为优化目标。

### 2.2 Formal evaluation（未查看人工标签）

- 官方 nuScenes `train` split；
- 排除全部 10 个 mini scene；train 与第二次 val 人审天然分离；
- 按 scene 名排序，全部 evaluation scenes 进入正式汇总，不做 top-k scene/actor 选择；
- annotation-only + map-expansion v1.3；不读取或下载 sweeps，不执行 N2。

## 3. 冻结事件合同

### 3.1 Eligibility 与基础 map match

沿用第一/二版，不通过改变 actor pool 取得通过：

- `vehicle.*`；
- 10 Hz 对齐 track ≥20 frames、无 frame gap、首尾位移 ≥5 m；
- centerline resolution 0.5 m，distance ≤3 m，heading error ≤45°；
- source/target stable run 各 ≥10 aligned frames，transition gap ≤20。

10 Hz 只用于复现旧 cadence 与定位候选。所有物理导数和持续性只使用原始 2 Hz keyframe。

### 3.2 原始 keyframe 运动学

每个候选必须有至少 3 个 source-side 与 3 个 target-side annotation keyframes；时间间隔读取官方
`sample.timestamp`，不假设完美 0.5 s。

逐事件保存：

- median speed、vector acceleration、yaw-rate；
- pre/post course 与 object yaw 变化；
- 世界坐标横向位移、span、peak lateral speed、单调性；
- source/target continuous polyline projection、cross-track distance 与 lane-preference margin；
- source branch 在汇合前 8 m 的 approach heading、target 离开后 8 m 的 heading；
- target 各 incoming branch 的相对 alignment。

两个合法模式：

1. `parallel_lane_change`：source/target 非 directed continuation，旧 topology 满足平行/共享 corridor，
   且 2 Hz pre/post 对 source/target 的距离偏好稳定翻转；
2. `converging_branch_merge`：source directed-connected 到 target，但 source 在 target incoming 中不是最顺直
   主路；其 approach error 相对最佳 alternative 至少差 3°，且 convergence angle 为 3°–30°。

明确拒绝：

- source 是 target 的最顺直 incoming（route continuation）；
- 正常转弯、course/yaw/approach 超过冻结上限；
- source/post 距离不支持 lane membership；
- 静止/低速 token jitter；
- annotation keyframe 不足或运动学异常。

### 3.3 Branch-safe interaction

以 target 为中心向前/后各最多 2 graph hops：

- 每条 edge heading discontinuity ≤20°、endpoint gap ≤3 m；
- 有分支时只取最连续的一条，不把所有 adjacent token 合并；
- 邻车 heading 相对 lane tangent ≤25°；
- 使用 box length/width 在 tangent 上的投影半长，把 center gap 转为 bumper gap；
- center keyframe 必须同时有 front/rear，bumper gap 均在 `[0.5,60] m`；
- 至少 2/3 个连续 2 Hz keyframes 保持同一 front/rear identity 与前后次序；
- 记录 subject/front/rear longitudinal speed、closing speed 与 TTC；这些只作诊断，不替代逐项 gate。

### 3.4 Same-actor negative

对 machine positive actor 搜索不与事件重叠的 30 aligned-frame lane-keeping window：

- 至少 5 个 2 Hz keyframes；
- centerline distance ≤2 m、lateral span ≤0.6 m、heading error ≤20°；
- moving 且 acceleration sanity 通过；
- 使用同一 temporal front/rear contract。

negative 不存在时不得用别的 actor 补 pair，也不得把普通窗口数量写成 same-actor pair。

## 4. 机器研究支持门槛

formal evaluation 同时要求：

| 指标 | 冻结阈值 |
|---|---:|
| machine positive candidates | ≥12 |
| negative windows | ≥4 |
| same-actor pairs | ≥4 |
| candidate scenes | ≥6 |

用户明确要求第三次人审，因此只要有至少 1 个完整 machine positive，就仍生成诊断/盲审材料并停在
`AWAITING_HUMAN_REVIEW`；但人工 verdict 不能覆盖上述 machine support gate 的失败。0 candidate 才直接
`REJECTED / reject_n1_kinematic_no_candidate`。

## 5. 人工盲审合同

最多按 `SHA256(blind_seed:event_id)` 均匀选择 40 个；不足 40 时全审。每项包括：

- 最多 5 张原始 CAM_FRONT 2 Hz keyframe；
- subject=洋红、front=绿、rear=蓝的 annotation 3D box 投影（若在相机视野）；
- 2 Hz 三车轨迹、source/target 与附近 vector centerlines；
- 运动学、branch、bumper gap、identity persistence 只读证据；
- component verdict、overall verdict、failure codes、reviewer 与 notes。

冻结 human gate：

| 指标 | 阈值 |
|---|---:|
| 完整审核 | ≥12 |
| TRUE_POSITIVE | ≥8 |
| TRUE_POSITIVE scenes | ≥4 |
| `TP/(TP+FP)` | ≥0.80 |
| Wilson 95% precision lower bound | ≥0.60 |
| UNCERTAIN fraction | ≤0.10 |

machine support 与 human gate 必须全部通过。完整、带实际 run/hash/命令的提示词由正式 run 写入
`audit/HUMAN_REVIEW_PROMPT.md`；agent 不能填写或推断 verdict。

## 6. Run contract 与停止条件

正式 run 必须：

- clean git commit；
- 唯一 run ID、resolved config、code/config/data fingerprint；
- calibration/evaluation scene list；
- event pool、per-scene audit、artifact hashes；
- 唯一 `REJECTED` 或 `AWAITING_HUMAN_REVIEW` marker；
- audit evidence/template/panels/prompt 的逐文件 SHA256。

无论 machine/human 结果：

- 不启动 N2/N3/render/training；
- 不恢复已失效的 `n2_extract_scene_list.txt`；
- 不下载或抽取 sweeps；
- 若失败，记录具体分母、UNKNOWN、无 pair 原因，不通过降低阈值重跑同 split。

## 7. 技术依据与适用边界

- nuScenes 官方 schema 明确 `sample` 是 2 Hz annotated keyframe：
  [nuScenes schema](https://github.com/nutonomy/nuscenes-devkit/blob/master/docs/schema_nuscenes.md)；
- 官方 devkit 的 `box_velocity` 优先用相邻 annotation 的 centered difference，并限制时间间隔：
  [nuScenes box_velocity](https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/nuscenes.py)；
- nuPlan 官方 metrics 只在 lane/lane-connector occupancy、intersection/merge 等明确语境下计算 lateral/cross
  traffic TTC，并沿 expert lane/connector route 计算 progress：
  [nuPlan metrics](https://github.com/motional/nuplan-devkit/blob/master/docs/metrics_description.md)。

这些来源支持“用原始时间戳、lane occupancy、route 与物理 box 关系”的设计，不证明本项目会通过。
阈值和最终结论仍只由本预注册 split/run 与用户人审裁决。
