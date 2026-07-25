# N1-EVENT-CUTIN-01 第四版预注册：receiver-centric 真实 cut-in

> 冻结日期：2026-07-25  
> 任务：`N1-EVENT-CUTIN-01`  
> 配置：`configs/resim/event_first_n1_cutin_v1.yaml`  
> 范围：只挖掘 N1 natural event、matched control 与第四次人工审核材料。  
> 禁止：N2 raw evidence、N3 proposal、render、training、数据下载和任何自动下阶段。

## 1. 为什么必须新建第四版

第二版的 37 个 machine positives 经人工审核仅 2 TP / 35 FP；第三版虽然加入 2 Hz kinematics，
12 个最终候选仍全部是 `converging_branch_merge`，第三次人审得到 0 TP / 12 FP。第三版失败集中在：

1. 把道路分支/中心线的几何收敛当成车辆横向机动；
2. target corridor 贪心选择 subject source branch，再把 subject 原队列后车命名为 rear；
3. 正常 lane→connector→lane 续接、正常转弯和 map-match jitter 被写成 cut-in；
4. negative/pair 只有 2/2，候选真实性和 matched-control 支持同时失败。

因此第四版不再问“subject 是否经过 multiple-incoming target”，而问：

> 某辆已经存在于独立目标车流中的 RECEIVER，是否被一个原本位于该车流之外、随后真实横向进入并在其前方
> 稳定的 SUBJECT 切入？

该定义是 receiver-centric，不要求可选 FRONT 才算事件；FRONT 只用于上下文。

## 2. 参考规则与可迁移部分

本地参考包：`/root/autodl-tmp/third_party/data_mining/cutin_rules_package`。

采用的结构：

- 相邻/独立目标车流的前置条件；
- 同一 subject identity 的 outside→inside 车道边界状态；
- 至少名义 1 s 的进入后稳定；
- 接收车/自车视角，而不是只看发起车的地图 token；
- 进入前后同一接收车 identity；
- subject 与 receiver 之间没有被忽略的更近车辆；
- 排除正常转弯、路口续接和自身车道变化混淆；
- gap、TTC、THW 作为诊断量。

不复制的部分：

- `CutIn_HighRecall.lua` 中把 `abs(diff)` 当单调性的实现；绝对值不能证明横向距离单调收敛；
- 依赖单帧 lane ID 或宽松 fallback 的规则；
- 用插值频率冒充物理观测；
- 只要“进入某 lane”就判事件，而不验证独立 receiver。

其他数据集提供的是设计启发，不参与第四版训练或 evaluation：

- highD 的 cut-in 研究把 lane-change 与基于 gap 的 cut-in 分类拆分：
  <https://arxiv.org/abs/2402.08289>；
- Waymo Open Dataset 同时提供 motion trajectories 与 3D maps，支持 road-relative trajectory
  思路：<https://github.com/waymo-research/waymo-open-dataset>；
- nuPlan devkit 的 map/trajectory 场景接口说明了路线与 actor motion 应分层：
  <https://github.com/motional/nuplan-devkit>。

这些来源不能替代 nuScenes 本轮人工真值，也不授权下载或引入新数据集。

## 3. 数据分层与泄漏隔离

### 3.1 Calibration

只使用已经完成的人审：

- 第二次：37 条，review SHA256
  `ae71b31e02faf1d783c36748e629e85acf32a132f35ce2f98102a5f62201dd05`；
- 第三次：12 条，review SHA256
  `005cd74b874833808435fd2f47387d1d8e446cdea2d3a5cae6146e34bf331e96`。

49 条标签只允许用于第四版事件定义、阈值和回归门。所有涉及的 26 个 scene 从 formal evaluation 排除。
冻结 calibration gate：

- 拒绝第三次 FP `≥12/12`；
- 拒绝第二次 FP `≥34/35`；
- 保留第二次 TP `≥1/2`。

冻结时实际 replay 为 12/12、35/35、1/2，三项均通过。未保留的旧 TP 不视为算法 FN：它没有满足第四版
新增的独立 RECEIVER pre/post identity 定义。

### 3.2 Formal evaluation

- split：nuScenes official `train`；
- 排除全部 calibration scenes 与配置中的 10 个 mini scenes；
- scene list 在查看第四版候选 panel/人工结果前由 split 规则确定；
- 只报告 machine counts、门状态和盲审材料，不根据 evaluation scene、候选类型或数量改阈值；
- 不读取 evaluation 的任何人工标签；第四次审核是唯一新的人工真实性证据。

## 4. 冻结候选流水线

### 4.1 候选与地图

- vehicle track：连续 10 Hz 对齐帧 `≥20`、无 frame gap、位移 `≥5 m`；
- 10 Hz 轨迹只用于 lane-token 候选和显示；
- map match：centerline distance `≤3 m`、heading error `≤45°`；
- source/target stable run 各 `≥10` dense frames，transition gap `≤20`；
- topology 只允许 parallel lane change 或有至少两个 incoming 的 merge，topology 不是最终事件证据。

地图工程只加载 `lane`、`lane_connector`、`arcline_path_3`、`connectivity`；centerline 离散化为
`0.5 m`。轻量实现必须与官方 nuScenes arcline reference 逐点回归一致，一次只常驻一个 location。

### 4.2 原始 2 Hz physical window

对每个 topology candidate：

1. 在 source 边界之前最多 9 个原始 keyframes、target 边界之后最多 9 个原始 keyframes内搜索；
2. pre/post 状态窗各为连续 3 个 2 Hz keyframes；
3. pre 窗最后一帧到 post 窗第一帧的 physical entry interval 必须在 `(0,4] s`；
4. 不要求 physical window 与 map token 边界完全同刻，因为 map match 可先于/晚于车身越线；
5. 所有速度、横移、box 和 identity 量只由 annotation keyframe 派生。

SUBJECT 必须同时满足：

- pre 中车身中心在 target lane band 外 `≥2/3` 帧；
- pre 中心绝对横距中位数 `≥1.85 m`；
- post 完整 oriented box 位于 target band 内 `≥2/3` 帧；
- post 中心绝对横距中位数 `≤0.80 m`；
- pre→post 横向收敛 `≥1.40 m`；
- 收敛一致率 `≥0.60`，pre 横向符号一致率 `≥2/3`；
- pre 相对 receiver corridor 航向误差中位数 `≤30°`；
- post 航向误差最大值 `≤30°`；
- 三个 post keyframes 覆盖名义 `≥1.0 s`；只允许 `20 ms` timestamp jitter tolerance；
- 运动速度中位数 `≥0.5 m/s`。

完整车身是否在 pre 全部离开 target band 继续报告为诊断，不作为硬门。原因是 2 Hz 标注可能在车身已经部分压线后
才出现稳定 source token；硬门使用车辆中心越过 lane boundary，panel 仍向人工展示 oriented box 四角。

### 4.3 独立 RECEIVER corridor

- parallel lane change：target 的上下游单分支 chain 明确排除 subject source；
- merge：只枚举 target 的 direct incoming 中不同于 subject source 的分支；
- graph edge heading error `≤30°`、endpoint gap `≤4 m`、上下游最多 2 hops；
- RECEIVER 在关系帧必须是 corridor 上最近、同向的后车；
- receiver centerline distance `≤1.5 m`、heading error `≤25°`；
- 同一 RECEIVER identity 在 pre `≥2`、post `≥2` 个 keyframes 保持最近后车；
- 所有支持帧 bumper gap 均在 `[0.5,40] m`；
- subject source token 不得出现在 receiver corridor；
- FRONT 可选，不进入 machine pass 的必要条件。

subject/receiver longitudinal speed、closing speed 与 TTC 只用于审核诊断，不替代上述几何和 identity 门。

### 4.4 Matched negative

- 只为已有 positive 的同一 actor 搜索；
- lane-keeping window 固定 30 dense frames（3 s），不得缩短；
- negative 不得与 physical event `[pre 第一帧, post 最后一帧]` 重叠，并额外留 5 dense frames（0.5 s）；
- 2 Hz lane-keeping：至少 5 keyframes、centerline distance `≤2 m`、lateral span `≤0.6 m`、
  heading error `≤20°`；
- control 中也必须有同一最近 RECEIVER，3 个关键帧中支持 `≥2`，gap 使用同一 `[0.5,40] m`。

这保证 negative 不再是“孤车普通直行”，而是同 actor、相近 interaction density 的非切入窗口。

## 5. 冻结机器与人工门

Machine gates：

- positive candidates `≥8`；
- negative windows `≥4`；
- same-actor pairs `≥4`；
- candidate scenes `≥5`。

只要有 `≥1` positive，就生成完整第四次 blind audit pack；machine gate 失败仍会原样写入 prompt 和 validator，
不能由人工真实性覆盖。

Audit：

- 最多 40 项，按 `n1-receiver-cutin-audit-v1:event_id` 的 SHA256 盲序；
- 每项包含最多 5 个 CAM_FRONT 2 Hz keyframes、S/R/F identity box、2 Hz topdown、原始 evidence JSON；
- `review_template.jsonl`、panel、evidence、prompt、checklist 全部进入 immutable hash；
- 只有 `review_working.jsonl` 可编辑。

Human gates：

- reviewed `≥8`；
- TP `≥6`，覆盖 `≥4` scenes；
- determinate precision `≥0.80`；
- Wilson 95% precision lower bound `≥0.50`；
- uncertain fraction `≤0.10`；
- machine gates 必须同时通过。

## 6. 终态与禁止恢复

- calibration gate 失败：开发停止，不能创建 formal research run；
- formal 无审核候选：`REJECTED`；
- formal 有候选：`AWAITING_HUMAN_REVIEW`，无论 machine gate 是否通过都完整披露；
- 第四次 review 只能由用户/指定评审填写，agent 只校验和聚合；
- parent run 永远 `n2_authorized=false`；
- 即使第四次 machine + human gates 全通过，也只获得“可请求下一阶段授权”的资格，不自动进入 N2。

禁止：

- 复用 source-stream rear；
- 把 map token change 当车身进入；
- 用 10 Hz 插值计算物理量；
- 缩短 30-frame negative；
- negative 与 physical event overlap；
- 查看 formal 结果后改阈值/scene/sampling；
- 启动 N2/N3/render/training。
