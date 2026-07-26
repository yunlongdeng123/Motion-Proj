# nuScenes Cut-in 挖掘与人工标注系统：最后一轮迭代计划

> **计划版本**：V3（在 V2 终止证据上按用户新资源授权复开）
> **计划状态**：`completed / rejected / stop_nuscenes_cutin_mining_too_sparse`
> **计划日期**：2026-07-26
> **目标任务 ID**：`N1-EVENT-CUTIN-FINAL-01`
> **适用仓库**：`/root/autodl-tmp/motion_proj`
> **运行环境**：无 GPU、cgroup 内存上限 120 GiB、CPU-only；Resource Contract V2
> **数据范围**：nuScenes `v1.0-trainval`；硬判定只使用官方 2 Hz annotation、calibration、ego pose 和 vector-map centerline/connectivity
> **最终边界**：这是 cut-in 挖掘器和人工标注系统的最后一次规则迭代。无论最终得到高精度池、稀疏可信池或拒绝结论，均停止继续调阈值或开启下一版挖掘器。

> **复开授权（2026-07-26）**：Resource Contract V1 的失败 parent 与独立 `REJECTED` 裁决保持不可变。
> 用户随后把容器 `memory.max` 扩大为 `128,849,018,880` bytes（120 GiB），并明确授权继续本 final。
> 当前只版本化资源阈值并使用新 config fingerprint / run ID；taxonomy、hard gate、K4、675-scene split、
> 抽样和人工聚合门槛全部保持 V2 冻结值。若 V2 再遇资源 stop，立即保全证据、停止并等待用户开放资源，
> 不反复重跑、不杀用户服务、不缩短正式 split。

> **复开执行收口（2026-07-26）**：M0 资源核对、M1 失败账本/Resource Contract V2、M2 675-scene
> formal、M3 稀疏终局人工审核包、M4 事实源/报告/回归均已完成。最终 parent 为
> `REJECTED / stop_nuscenes_cutin_mining_too_sparse`；人工包只作 1 primary + 3 diagnostic 复核，
> 不能改变数量门失败或授权 N2。

---

## 0. 给清空 context 后 Codex 的执行指令

本文件是独立执行入口。执行者不得仅依赖对话摘要，也不得跳过研究事实源。

开始时依次读取：

```text
/root/autodl-tmp/motion_proj/AGENTS.md
/root/autodl-tmp/motion_proj/docs/RESEARCH_STATUS.md
/root/autodl-tmp/motion_proj/docs/RESEARCH_FAILURES.md
/root/autodl-tmp/motion_proj/docs/EXPERIMENTS.md
/root/autodl-tmp/motion_proj/docs/NUSCENES_CUTIN_FINAL_ITERATION_PLAN_V1.md
```

然后只读检查：

```bash
cd /root/autodl-tmp/motion_proj
git status --short --branch
git log --oneline -20
git cat-file -t f13eb0f1e39b608de1c5e698cd678c2dfd8365a4
```

必须保留现有 dirty worktree 中的用户修改。不得 `git reset --hard`、不得覆盖第四轮 run、不得修改人工填写的 verdict。

第四轮真实代码已定位，不再从旧脚本另起一套：

```text
入口：
  resim/event_first_n1_cutin.py

核心判定：
  motion_proj/resim/cutin_receiver.py
  motion_proj/resim/event_kinematics.py

轻量数据与地图：
  motion_proj/resim/nuscenes_trainval_tracks.py
  motion_proj/resim/lightweight_nuscenes_map.py
  motion_proj/resim/io_memory.py

审核与校验：
  motion_proj/resim/n1_kinematic_audit.py
  scripts/validate_n1_cutin_review.py

配置与测试：
  configs/resim/event_first_n1_cutin_v1.yaml
  tests/test_cutin_receiver.py
  tests/test_n1_cutin_config.py
  tests/test_n1_cutin_review.py
```

第四轮成功 run 的源码基线为 clean commit：

```text
f13eb0f1e39b608de1c5e698cd678c2dfd8365a4
```

禁止事项：

1. 不修改第四轮 parent run 或其中的 `review_working.jsonl`；
2. 不让 agent、规则或 LLM 代填人工 verdict；
3. 不下载 K-Risk 数据集、额外 nuScenes 相机包、LiDAR sweeps 或模型权重；
4. 不启动 N2、3DGS、renderer、训练或下游数据抽取；
5. 不以 K4 单例结果继续逐 case 调阈值；
6. 不把 `receiver_branch_merge` 改名后继续进入 machine-positive；
7. 不为了省内存杀死用户的编辑器、Jupyter、TensorBoard 或其他服务；
8. 不在正式结果出来后改变 split、scene、阈值、抽样顺序或统计分母。

---

# 1. 当前事实与第四轮正式裁决

## 1.1 权威输入

第四轮 parent run：

```text
/root/autodl-tmp/runs/event_first/N1-EVENT-CUTIN-01/
v71_n1-event-cutin-01__receiver-cutin-v1__s0__20260725T173015103731Z__5b1634e3
```

关键 provenance：

```text
code commit:
  f13eb0f1e39b608de1c5e698cd678c2dfd8365a4

config fingerprint:
  5b1634e3347c81ca8d6c7a1b6b3d5a737b092732a3ff9b5c79b093fccfd846c5

event-pool canonical SHA256:
  850434a349c65e2f8fc9ece98357e3a0a2f94afcd55d544e7648b47e44affe7f

completed review:
  audit/review_working.jsonl

completed review SHA256:
  983e4b7a4160ff7aec127343b5ca3e1e9a1f07f06d799f4db9695fa241851321

reviewer:
  yunlong.deng
```

冻结 validator 的正确只读命令必须带 `PYTHONPATH=.`：

```bash
cd /root/autodl-tmp/motion_proj
source /root/miniconda3/etc/profile.d/conda.sh
conda activate motionproj
PYTHONPATH=. python scripts/validate_n1_cutin_review.py \
  --run-dir /root/autodl-tmp/runs/event_first/N1-EVENT-CUTIN-01/v71_n1-event-cutin-01__receiver-cutin-v1__s0__20260725T173015103731Z__5b1634e3 \
  --review-file /root/autodl-tmp/runs/event_first/N1-EVENT-CUTIN-01/v71_n1-event-cutin-01__receiver-cutin-v1__s0__20260725T173015103731Z__5b1634e3/audit/review_working.jsonl
```

2026-07-26 已只读复验，结果：

```text
reviewed = 18
TRUE_POSITIVE = 3
FALSE_POSITIVE = 15
UNCERTAIN = 0
determinate precision = 0.1666666667
Wilson 95% lower bound = 0.0583657681
TP scenes = 3
all_human_gates_passed = false
recommended verdict = reject_n1_receiver_cutin_after_human_audit
n2_authorized = false
```

因此第四轮必须在独立 adjudication run 中登记为 `REJECTED`。parent run 继续保持不可变的 `AWAITING_HUMAN_REVIEW` 历史终态，不能原地改写。

## 1.2 第四轮人工结果分解

第四轮 machine-positive 共 18 条：

```text
parallel_lane_change = 5
receiver_branch_merge = 13
```

按模式分解：

| 模式 | TP | FP | 人工精度 | 最终决策 |
|---|---:|---:|---:|---|
| `receiver_branch_merge` | 0 | 13 | 0.000 | 最终版不再支持 machine-positive |
| `parallel_lane_change` | 3 | 2 | 0.600 | 保留，但必须增加严格后验验证 |
| 全部 | 3 | 15 | 0.167 | 第四轮正式拒绝 |

人工 failure code 计数：

| failure code | 数量 |
|---|---:|
| `RECEIVER_INVALID` | 11 |
| `NORMAL_TURN` | 8 |
| `ROUTE_CONTINUATION` | 6 |
| `WRONG_BRANCH` | 5 |
| `OPPOSITE_OR_CROSS_TRAFFIC` | 2 |
| `IDENTITY_NOT_PERSISTENT` | 1 |
| `MAP_MATCH_JITTER` | 1 |

component 结果：

```text
subject_maneuver INVALID = 14
receiver_corridor INVALID = 12
receiver_relation INVALID = 12
temporal_persistence INVALID = 1
```

## 1.3 K4 回归分层

人工真值必须原样冻结：

```yaml
human_true_positive:
  - K4-009
  - K4-010
  - K4-011

human_false_positive:
  - K4-001
  - K4-002
  - K4-003
  - K4-004
  - K4-005
  - K4-006
  - K4-007
  - K4-008
  - K4-012
  - K4-013
  - K4-014
  - K4-015
  - K4-016
  - K4-017
  - K4-018
```

发布回归策略与人工真值分开：

```yaml
release_blocking_tp:
  - K4-010
  - K4-011

boundary_tp_non_blocking:
  - K4-009

required_not_pass:
  - K4-001
  - K4-002
  - K4-003
  - K4-004
  - K4-005
  - K4-006
  - K4-007
  - K4-008
  - K4-012
  - K4-013
  - K4-014
  - K4-015
  - K4-016
  - K4-017
  - K4-018
```

`K4-009` 的人工标签仍是 `TRUE_POSITIVE`，不得改成 UNCERTAIN。它之所以不作为 release blocker，是因为其进入前车身已经部分压入目标带，审核窗口只完整证明“中心在外→后窗车身稳定进入”，没有完整展示整车最初位于目标带外。最终 verifier 若对它输出 `PASS`，必须符合冻结的 center-based 事件定义；若输出 `ABSTAIN`，必须使用明确的边界证据 reason；不得为保留它放宽其他样本的规则。

## 1.4 两个平行车道误报的直接代码根因

### K4-012：身份支持计数不等于身份持续

当前 `_evaluate_corridor`：

1. 在 post 中间 `relation_frame` 选 `receiver_id`；
2. 只统计与该 ID 相同的 pre/post 支持帧；
3. 只要 pre 和 post 各达到最小数量即 PASS；
4. 没有拒绝其他原始帧出现不同的最近后车。

K4-012 的最近后车序列为：

```text
frame 75/80/85/115/120/125
actor 1/1/1/1/1/38
```

最后一帧已切换为 actor 38，gap 约 `51.42 m`，但旧实现仍因 actor 1 获得 3 个 pre、2 个 post 支持而 PASS。

最终修复必须检查：

```text
所有非空最近后车 ID 的唯一性
末个 post 原始帧的 anchor identity
每帧最近后车 rank
所有支持帧 gap
不同 ID 的显式 switch
```

### K4-015：弯道 token 切换通过了过宽航向阈值

K4-015：

```text
source/target local heading error = 11.23°
post subject heading error = 20.9° / 21.9° / 18.0°
```

旧配置允许：

```text
max_parallel_heading_error_deg = 30
max_post_heading_error_deg = 30
```

这使弯道地图匹配切换被解释为平行车道进入。最终版的相邻平行车道和 post 稳定航向都必须收紧到有明确交通语义的 `10°` 级别，并使用原始 2 Hz 多帧而非单点 token。

## 1.5 结论边界

能下的结论：

- 第四轮 receiver branch merge 规则在本次 scene-disjoint 人审中为 13/13 FP；
- 当前目标是高精度 cut-in seed pool，因此停止支持该模式具有充分证据；
- 平行车道剩余误报可对应到两个明确的实现缺陷，而不是继续泛化地调 gap；
- 第四轮人工结果可在排除其 16 个 scenes 后作为最终版 calibration/regression；
- 第四轮本身不通过人工门槛，不能进入 N2。

不能下的结论：

- 不能声称 nuScenes 不存在真实 branch merge；
- 不能用 K4 修复后的 3/3 回放宣称 prospective precision 为 100%；
- 不能把人工 TP 直接当最终 event pool；
- 不能因 TTC 很小、gap 很近或画面“看起来危险”而跳过 cut-in 几何定义；
- 不能把机器 ABSTAIN 算作正确负例或排除出 coverage 报告。

---

# 2. 本轮目标、非目标和事件定义

## 2.1 核心目标

构建一个 CPU-only、2 GiB 内存可运行、可追溯、允许弃权的高精度 receiver-centric cut-in 挖掘与人工标注系统。

成功优先级：

```text
人工精度 > 证据可审计性 > 确定性与资源稳定性 > 样本数 > 召回率
```

最终 machine-positive 定义：

> SUBJECT 在原始 2 Hz 观测中原先位于一条相邻、近似平行的 source lane；随后其中心和定向车身相对 target corridor 发生持续横向收敛并稳定进入。进入前后，同一辆动态、同向 RECEIVER 已在 target corridor 上行驶，并持续是 SUBJECT 后方最近车辆；二者之间没有另一辆 target-corridor 车辆。

## 2.2 最终 taxonomy

```yaml
parallel_lane_change:
  supported: true
  may_be_machine_positive: true

receiver_branch_merge:
  supported: false
  may_be_machine_positive: false
  status: ABSTAIN
  reason: UNSUPPORTED_BRANCH_MERGE_MODE
```

`receiver_branch_merge` 仍可计入宽松候选漏斗和 reason 分布，但不得进入 PASS pool、人工精度主样本或最终 seed pool。

## 2.3 明确非目标

本轮不做：

- 不解决路口汇入、环岛、正常转弯或复杂多分支 merge；
- 不做学习式分类器、VLM 自动裁决或 LLM 代标；
- 不下载 K-Risk 数据或其他外部轨迹数据；
- 不使用 LiDAR sweeps、3DGS 或 learned occupancy 作为事件真值；
- 不把风险等级、TTC、THW 当 cut-in 成立条件；
- 不追求高召回；
- 不要求 same-actor negative/pair 作为本轮 event-pool 通过门槛；
- 不自动启动任何下游阶段；
- 不在最终盲测后继续第五轮规则修补。

## 2.4 matched control 与事件真实性解耦

根据 `N1-F09`，事件真实性和 matched-control 支持是两个独立问题。

本轮只冻结高精度事件池。因为 N2 仍锁定，`negative_window_count` 和 `same_actor_pair_count`：

- 可作为诊断输出；
- 不作为 final machine readiness gate；
- 不得通过缩短 30-frame window、允许 event overlap 或换 actor 来补数量；
- 若以后需要 matched control，必须由用户另行授权新任务。

---

# 3. 证据分层与第三方规则吸收方式

## 3.1 硬证据优先级

| 层级 | 数据 | 用途 | 能否单独使 PASS |
|---|---|---|---|
| T0 | 原始 nuScenes 2 Hz annotation box、instance、timestamp | 主体横移、航向、速度、接收车身份和 gap | 是 |
| T0 | 轻量 vector-map centerline/connectivity | 相邻平行拓扑、corridor `(s,d)` | 是，但必须与 T0 轨迹共同成立 |
| T1 | 原始 CAM_FRONT keyframe | 人工辅助、角色可见性 | 否 |
| T2 | 10 Hz 插值轨迹 | 宽松候选、时间对齐、显示 | 否 |
| T2 | TTC/THW/risk rank | 审核排序和描述 | 否 |

任何只由插值帧满足的 hard check：

```text
ABSTAIN / INTERPOLATION_ONLY
```

## 3.2 不引入重型 map API

最终 mining 进程继续使用：

```text
motion_proj/resim/lightweight_nuscenes_map.py
```

不得在 mining 进程导入官方 `nuscenes.map_expansion.map_api`、OpenCV、Matplotlib、Shapely 或完整渲染 API。

原因见 `N1-F15`：

- 重型 map API import 曾使 RSS 从约 58 MiB 增至约 212 MiB；
- 四图常驻与完整元数据会在 2 GiB cgroup 下触发 `RC=137`；
- 当前本轮只需 lane/lane_connector centerline 和 connectivity。

车道边界继续使用明示 fallback：

```text
nominal lane half width = 1.75 m
```

这与参考规则在感知车道线缺失时的 `±1.75 m` fallback 一致。evidence 必须记录：

```yaml
lane_width_source: configured_nominal_fallback
lane_half_width_m: 1.75
```

不得伪装成官方 polygon overlap。若关键 map centerline 缺失或 corridor 不能构造，输出 `ABSTAIN/MAP_GEOMETRY_UNAVAILABLE`。

## 3.3 cut-in 规则包中保留的语义

参考路径：

```text
/root/autodl-tmp/third_party/data_mining/cutin_rules_package/README.md
/root/autodl-tmp/third_party/data_mining/cutin_rules_package/
  01_themis_roadmerge/fft_standard_OD/cut_in_highway_rule.md
/root/autodl-tmp/third_party/data_mining/cutin_rules_package/
  01_themis_roadmerge/fft_standard_OD/cut_in_urban_rule.md
/root/autodl-tmp/third_party/data_mining/cutin_rules_package/
  02_Dionysus/atomic_checkers/ObjAttr_EventMotion/CutIn.lua
/root/autodl-tmp/third_party/data_mining/cutin_rules_package/
  02_Dionysus/Scenario_checkers/KeyScenario_ObjAttr_EventMotion/CutinCutout.lua
```

吸收以下语义，不复制其传感器字段或阈值：

| 参考规则语义 | 本项目实现 |
|---|---|
| 目标先在相邻车道，后进入接收车道 | 原始 2 Hz pre center outside + post oriented box inside |
| 同一 `obj_id` 跨帧追踪 | SUBJECT/RECEIVER instance token 和 actor ID 持续 |
| 航向过滤横穿目标 | corridor 有符号纵向速度 + 多帧 heading error |
| 进入后需持续确认 | 至少两个 post raw keyframe，名义 1 s settle |
| 目标与 receiver 之间路径清晰 | 全支持窗最近后车 rank 和 intermediate actor 列表 |
| 状态机区分候选、确认、退出 | `PASS/FAIL/ABSTAIN` 与 first-failure reason |
| clip 边界不足需显式处理 | 不静默 PASS；证据不足为 ABSTAIN |

不吸收以下做法：

- 不采用 clip 起点已在车道内的高召回豁免作为 machine-positive；
- 不用单个当前帧 path-clear 代替全窗验证；
- 不把产品规则的 ego 车速度阈值直接套到任意 annotation receiver；
- 不用高召回版 `0.2 m` 横移阈值；
- 不用在线感知目标类别枚举代替 nuScenes `vehicle.*`。

## 3.4 K-Risk 中保留的系统设计

参考路径：

```text
/root/autodl-tmp/third_party/K-Risk/README.md
/root/autodl-tmp/third_party/K-Risk/generate_AV.py
/root/autodl-tmp/third_party/K-Risk/compute_statistics.py
/root/autodl-tmp/third_party/K-Risk/prompts.py
```

只吸收：

1. 每个 event 使用稳定 ID，同步保存轨迹 JSON、符号 metadata、自然语言说明和审核记录；
2. source-specific adapter 不可混用；nuScenes 必须保留自己的字段和坐标合同；
3. per-frame 轨迹、速度、lane、spacing、TTC/THW 分字段保存；
4. risk/trajectory-conflict 只作为事件成立后的二级属性；
5. 统计时同时报告样本分母、source/scene 分层和 coverage。

明确不吸收：

- 不调用 K-Risk LLM annotation notebook；
- 不用 LLM 输出替代人工 verdict；
- 不下载 Figshare 数据集；
- 不把 K-Risk 的 risk level 当本项目 cut-in label。

---

# 4. 资源合同：V1 历史证据与 V2 当前授权

## 4.1 启动环境

所有开发、测试和正式命令统一：

```bash
cd /root/autodl-tmp/motion_proj
source /root/miniconda3/etc/profile.d/conda.sh
conda activate motionproj

export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=""
export MPLBACKEND=Agg
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MALLOC_ARENA_MAX=2
```

正式 preflight 必须记录：

```bash
cat /sys/fs/cgroup/memory.max
cat /sys/fs/cgroup/memory.current
cat /sys/fs/cgroup/memory.events
nvidia-smi 2>/dev/null || true
```

无 GPU 是正常状态，不能因为 `nvidia-smi` 不可用而失败。

## 4.2 内存门槛

Resource Contract V1 的冻结值如下，仅用于解释已保留的历史失败与独立拒绝裁决：

```yaml
runtime:
  require_posix_page_cache_control: true
  scene_batch_size: 32
  max_live_map_locations: 1
  keep_all_scene_results_in_memory: false
  max_start_cgroup_current_bytes: 1350000000
  warn_process_rss_bytes: 700000000
  stop_process_rss_bytes: 900000000
  warn_cgroup_current_bytes: 1750000000
  stop_cgroup_current_bytes: 1950000000
```

Resource Contract V2 是当前唯一执行入口：

```text
configs/resim/event_first_n1_cutin_final_resource_v2.yaml
```

```yaml
resource_contract:
  version: resource-contract-v2-user-authorized-20260726
  observed_memory_max_bytes: 128849018880
  policy_on_resource_stop: stop_and_wait_for_user_resource_authorization
runtime:
  max_start_cgroup_current_bytes: 8589934592
  warn_process_rss_bytes: 4294967296
  stop_process_rss_bytes: 8589934592
  warn_cgroup_current_bytes: 17179869184
  stop_cgroup_current_bytes: 25769803776
```

V2 的 start 上限为 8 GiB，现场当前 cgroup 使用约 4.2 GB；运行 stop 上限保守冻结为 process RSS 8 GiB、
cgroup current 24 GiB，显著低于 120 GiB 硬上限并给用户服务留出余量。资源 V2 只改变资源可用性，不改变
任何研究判据。若达到 stop 阈值或再次出现 `RC=137` / SIGKILL，不继续试探：写失败证据、停止新 batch并等待
用户新的资源授权。

达到 stop 阈值时：

1. 不继续处理新 batch；
2. 写 `FAILED/failure.json`；
3. reason 为 `resource_pressure_preemptive_stop`；
4. 保存最后完成 scene、RSS、cgroup current、anon、file cache 和 `memory.events`；
5. 不杀其他服务；
6. 修复后使用新 run ID，不覆盖或续写失败 run。

## 4.3 mining 与 audit 必须分进程

当前第四轮在同一 Python 进程完成：

```text
全量 mining → 保留 scene_results → 导入 Matplotlib/PIL → build audit
```

最终版改为一个轻量 orchestrator 启动两个子进程：

```text
orchestrator
  ├─ worker A: annotation/map-only mining
  └─ worker B: audit rendering
```

要求：

- worker A 退出后，操作系统释放其 Python heap 和 map cache；
- worker B 才允许导入 Matplotlib/PIL；
- parent orchestrator 独占 run terminal marker；
- worker stage 失败时 parent 写结构化 `FAILED`；
- 外部用户只执行一个 formal command，不手工拼接或复用 run ID。

## 4.4 流式产物

不得继续把所有 scene 的完整 transition dict 保存在 `scene_results`。

改为：

```text
transition_diagnostics.jsonl  # 流式写出所有宽松 transition 的精简诊断
strict_candidates.jsonl       # 只写 PASS/ABSTAIN 候选
strict_event_pool.json        # 小型 PASS pool 和记录 hash
scene_metrics.jsonl           # 每 scene 漏斗与内存
```

内存中只保留：

- Counter；
- 当前 scene；
- 当前 map location；
- 最终 PASS/ABSTAIN 的小型索引；
- record SHA256 列表。

每个 batch 后：

```text
del dense scene payload
gc.collect()
malloc_trim
POSIX_FADV_DONTNEED
```

## 4.5 相机资产现实边界

当前本地样本数：

```text
CAM_FRONT       34149
CAM_FRONT_LEFT    404
CAM_FRONT_RIGHT   404
CAM_BACK           404
CAM_BACK_LEFT      404
CAM_BACK_RIGHT     404
```

因此取消 V1 的“正式审核必须六相机”要求。最终审核：

- CAM_FRONT 文件存在且角色投影可见时才显示；
- 其他五相机只在本地文件实际存在时作为可选增强；
- 不因缺多相机下载新资产；
- 角色不可见时以逐帧俯视 box 和原始信号为主；
- 页面明确写“相机不提供双角色直接证据”，不得用肉眼猜角色。

---

# 5. 最终系统架构

```text
official nuScenes metadata + vector-map centerlines
        │
        ├─ 10 Hz interpolation：仅宽松 token transition 候选
        │
        └─ raw 2 Hz annotations：所有 hard evidence
                 ↓
        broad candidate miner
                 ↓
        final strict verifier v2
        ├─ parallel topology
        ├─ subject raw body entry
        ├─ post heading stability
        ├─ receiver dynamic/direction
        ├─ receiver identity/rank full-window
        └─ corridor ambiguity policy
                 ↓
          PASS / FAIL / ABSTAIN
                 ↓
        primary PASS audit + diagnostic abstain audit
                 ↓
        immutable evidence + editable JSONL
                 ↓
        human validator + independent adjudication
                 ↓
      cutin_pool_pass / usable_but_sparse / rejected
```

PASS 的唯一含义：

```text
machine-positive，可进入人工精度审核
```

PASS 不等于人工 TP，不等于 N1 最终通过，更不授权 N2。

---

# 6. 工作包与验收

## P0：第四轮独立 adjudication 和校准冻结

**状态**：`done`

**实施记录（2026-07-26）**：已在提交 `a74c55a` 固化独立裁决入口和离线 K4 fixture；正式裁决使用干净
worktree，生成
`/root/autodl-tmp/runs/event_first/N1-EVENT-CUTIN-AUDIT-01/v71_n1-event-cutin-audit-01__human-audit-reject-v1__s0__20260726T102715433654Z__ee795332`。
该 run 的唯一终态为 `REJECTED`，review SHA、18 条计数和 parent immutable audit-set 均已复验，
`n2_authorized=false`。

### P0.1 新增第四轮独立裁决入口

新增：

```text
scripts/adjudicate_n1_cutin_audit.py
configs/resim/n1_cutin_audit_rejection_v1.yaml
tests/test_adjudicate_n1_cutin_audit.py
```

输入：

```text
第四轮 parent run
completed review SHA256 = 983e4b7a...
validator aggregate
用户已完成的逐项 verdict
```

输出独立不可复用 run：

```text
task_id: N1-EVENT-CUTIN-AUDIT-01
terminal: REJECTED
exit_reason: reject_n1_receiver_cutin_after_human_audit
n2_authorized: false
```

parent run 不修改。

### P0.2 冻结 K4 calibration fixture

新增：

```text
tests/fixtures/n1_cutin_k4/
  review_working.jsonl
  audit_manifest_minimal.json
  evidence/K4-001.json ... K4-018.json
  expected_strict_status.yaml
  README.md
```

不复制 panel PNG。README 记录：

- parent run；
- review SHA256；
- evidence hash；
- 人工标签来源；
- 16 个 K4 scene；
- 仅用于 calibration/regression，不进入 final evaluation metric。

### P0.3 scene-disjoint 更新

第二、三次共有：

```text
49 labels / 26 scenes
```

加入第四次：

```text
18 labels / 16 new scenes
```

最终 calibration：

```text
67 human labels / 42 scenes
```

最终 official train evaluation 必须排除全部 K4 16 scenes。基于第四轮 685-scene evaluation，预期剩余：

```text
675 scenes
```

实现必须计算而不是只相信硬编码，并断言：

```text
calibration_scenes ∩ evaluation_scenes = ∅
K4_scenes ∩ evaluation_scenes = ∅
```

### P0 验收

- 第四轮独立 run 唯一终态为 `REJECTED`；
- review hash 与 18 条计数一致；
- parent run 逐文件 hash 未变化；
- K4 fixture 可离线读取；
- final evaluation scene 预期为 675，交集为 0；
- N2 仍为 false。

---

## P1：strict verifier v2 schema 与 taxonomy 收敛

**状态**：`done`

**实施记录（2026-07-26）**：提交 `3aa7038` 已提供 `receiver-centric-cutin-strict-v2` 三态
schema、冻结的 first-failure reason 顺序和 v1 只读 diagnostic adapter。`receiver_branch_merge` 固定
为 `ABSTAIN/UNSUPPORTED_BRANCH_MERGE_MODE`，任何插值标记均不得形成 `machine_positive`；K4 v1 evidence
读取与 schema 单元测试已通过。

### P1.1 三态 schema

升级为：

```yaml
schema_version: receiver-centric-cutin-strict-v2
status: PASS | FAIL | ABSTAIN
primary_reason: string | null
all_reasons: [string]
maneuver_mode: parallel_lane_change | receiver_branch_merge
machine_positive: bool
hard_evidence_source: raw_2hz_annotations
uses_interpolated_physics: false

provenance:
  source_event_record_sha256: string
  config_fingerprint: string
  map_version: string
  lane_width_source: configured_nominal_fallback

subject:
  actor_id: int
  instance_token: string
  source_token: string
  target_token: string
  pre_frames: [int]
  post_frames: [int]
  per_frame:
    - frame: int
      observation_source: raw_2hz
      source_d_m: float | null
      target_d_m: float
      target_s_m: float
      target_heading_error_deg: float
      speed_mps: float | null
      center_outside_target_band: bool
      box_inside_target_band: bool

receiver:
  selected_actor_id: int | null
  actor_id_by_frame: [int | null]
  nearest_rear_rank_by_frame: [int | null]
  gap_m_by_frame: [float | null]
  longitudinal_speed_mps_by_frame: [float | null]
  heading_error_deg_by_frame: [float | null]
  identity_switch_frames: [int]
  missing_frames: [int]
  intermediate_actor_ids_by_frame: object
  identity_persistent: bool
  nearest_rear_persistent: bool
  path_clear: bool

checks:
  supported_mode: bool
  source_target_parallel: bool
  raw_pre_outside: bool
  raw_post_inside: bool
  lateral_convergence: bool
  post_heading_stable: bool
  subject_dynamic: bool
  receiver_dynamic: bool
  receiver_same_direction: bool
  receiver_identity_persistent: bool
  receiver_nearest_rear_persistent: bool
  path_clear: bool
  corridor_unambiguous: bool
```

字段名可按代码风格调整，但信息能力不得减少。

### P1.2 reason 优先级

冻结 first-failure 顺序：

```text
UNSUPPORTED_BRANCH_MERGE_MODE
INSUFFICIENT_RAW_SUPPORT
MAP_GEOMETRY_UNAVAILABLE
SOURCE_TARGET_NOT_PARALLEL
NO_RAW_LATERAL_ENTRY
POST_HEADING_UNSTABLE
SUBJECT_NOT_DYNAMIC
AMBIGUOUS_RECEIVER_CORRIDOR
RECEIVER_NOT_DYNAMIC
RECEIVER_WRONG_DIRECTION
RECEIVER_IDENTITY_SWITCH
RECEIVER_SUPPORT_INSUFFICIENT
RECEIVER_GAP_INVALID
PATH_NOT_CLEAR
INTERPOLATION_ONLY
```

政策：

- 定义明确不满足：`FAIL`；
- 数据缺失、地图歧义、证据被 clip 截断：`ABSTAIN`；
- 只有全部 hard checks 通过：`PASS`；
- `receiver_branch_merge` 固定 `ABSTAIN/UNSUPPORTED_BRANCH_MERGE_MODE`；
- 所有 reason 都必须能追溯到帧或 topology 字段。

### P1.3 旧 schema 兼容

旧 `receiver-centric-cutin-v1` evidence：

- validator 可继续读取；
- 通过显式 adapter 转成 v2 diagnostic；
- 不原地迁移第四轮文件；
- 新 run 只写 v2。

### P1 验收

- schema 有版本；
- PASS 必定 `machine_positive=true`；
- FAIL/ABSTAIN 必定 `machine_positive=false`；
- 插值证据不能改变 hard verdict；
- reason 顺序和字段有单元测试；
- 旧 K4 evidence 可读取。

---

## P2：平行车道 SUBJECT verifier

**状态**：`done`

**实施记录（2026-07-26）**：提交 `32ee843` 已实现局部相邻平行车道 geometry、raw 2 Hz
center/box outside→inside、yaw unwrap 和严格 post-heading 检查；route continuation 与插值 hard-support
均 fail-closed。synthetic standard cut-in、正常续接、弯道 heading jitter 与插值误用回归已通过。

### P2.1 只接受局部相邻平行车道

hard checks：

1. source 与 target token 不同；
2. 不是 source→target directed route continuation；
3. crossing 局部 source/target tangent heading error ≤ `10°`；
4. 局部中心线横向间隔在 `[2.0, 6.0] m`；
5. source/target 在 crossing 前存在足够的并行纵向覆盖；
6. 不以 `shared_successor` 或 token 改变单独判定；
7. 几何缺失时 ABSTAIN，不回退到 `target_incoming_count`。

新增纯函数：

```text
local_parallel_lane_geometry(...)
parallel_overlap_length(...)
raw_lane_preference_sequence(...)
```

### P2.2 原始 2 Hz 车身进入

保留冻结的 center-based 语义：

```text
pre:
  至少 3 个 raw keyframe；
  至少 2 帧 subject center 位于 target nominal band 外；
  pre median |d_target| ≥ 1.85 m；

post:
  至少 3 个 raw keyframe；
  至少 2 帧 oriented box 位于 target nominal band 内；
  post median |d_target| ≤ 0.80 m；
  settle duration 名义 ≥ 1.0 s，timestamp tolerance 0.02 s；

motion:
  pre→post median lateral convergence ≥ 1.40 m；
  convergence consistency ≥ 0.80；
  pre side consistency ≥ 0.80。
```

`pre_box_fully_outside` 保留为诊断，不作为 PASS 必需条件；否则会错误拒绝已人工确认的 K4-009/K4-010。若未来业务定义要求整车完全在外，那是新 taxonomy，不得在本最后迭代中临时改写。

### P2.3 航向与弯道稳定性

```yaml
subject:
  max_source_target_heading_error_deg: 10.0
  max_pre_heading_error_deg: 15.0
  max_post_heading_error_deg: 10.0
  max_accumulated_yaw_change_deg: 15.0
  min_median_speed_mps: 1.0
```

`max_pre_heading_error_deg` 使用稳健中位数；`max_post_heading_error_deg` 使用 raw post 最大值，防止 K4-015 一类后窗持续未对齐。

累计 yaw 使用 unwrap 后的相邻 raw yaw 绝对变化和，不从 10 Hz 插值计算。

### P2.4 K4 预期

```text
K4-010: PASS（blocking）
K4-011: PASS（blocking）
K4-009: 目标 PASS；若为 ABSTAIN，必须有明确边界证据 reason，且不得调阈值挽救
K4-015: FAIL / SOURCE_TARGET_NOT_PARALLEL 或 POST_HEADING_UNSTABLE
所有 receiver_branch_merge: 不进入本模块 PASS
```

### P2 验收

- K4-015 不再 PASS；
- K4-010/K4-011 保留；
- synthetic 正常弯道、route continuation、token jitter 均不 PASS；
- synthetic 标准相邻 lane change PASS；
- 删除 raw 支持帧后即使 10 Hz 插值完整也不能 PASS。

---

## P3：RECEIVER、corridor 和完整时序

**状态**：`done`

**实施记录（2026-07-26）**：提交 `b61b191` 已改为对每个 required raw frame 重算 rank=1
最近后车，并验证唯一 non-null identity、末个 post anchor、raw 动态/方向、gap、target-stream 建立和
path-clear。多个 corridor 若导出不同 receiver 则 `ABSTAIN/AMBIGUOUS_RECEIVER_CORRIDOR`；identity
switch 保持首要失败原因。相关 synthetic 与旧 v1 回归共 20 项通过。

### P3.1 receiver 类别与局部动态性

RECEIVER 必须：

```yaml
class_prefix: vehicle.
min_raw_pre_support: 2
min_raw_post_support: 2
min_total_non_null_support: 5
max_missing_required_frames: 1
min_local_displacement_m: 1.0
min_median_longitudinal_speed_mps: 1.0
max_heading_error_deg: 15.0
max_centerline_distance_m: 1.5
min_bumper_gap_m: 0.5
max_bumper_gap_m: 40.0
```

速度使用 raw 2 Hz 局部线性拟合或相邻 keyframe 的稳健中位数。不得使用 track 全局位移证明事件窗内动态，也不得用 10 Hz 插值制造速度。

有符号纵向速度：

- `≤ 0`：`FAIL/RECEIVER_WRONG_DIRECTION`；
- `(0, 1.0)`：`FAIL/RECEIVER_NOT_DYNAMIC`；
- 缺少足够 raw 帧：`ABSTAIN/RECEIVER_SUPPORT_INSUFFICIENT`。

### P3.2 最近后车身份的正确语义

最终算法：

1. 对每个 required raw frame 重新排序 target corridor 后方 actor；
2. 记录 rank=1 actor 或 null；
3. 从全部非空 rank=1 ID 取唯一集合；
4. 集合大小必须为 1；
5. 至少 2 个 pre、2 个 post，且总支持 ≥5；
6. 最后一个 post raw frame 必须仍为同一 actor，或在没有任何后车时明确 ABSTAIN；
7. 任一帧出现另一个 rank=1 actor，立即 `FAIL/RECEIVER_IDENTITY_SWITCH`；
8. gap 对所有同一 receiver 支持帧检查，而不是只检查被选 ID 的子集；
9. 记录所有 intermediate actor，确保 path clear。

允许 K4-011：

```text
113 / 113 / null / 113 / 113 / 113
```

因为非空最近后车集合唯一、pre=2、post=3、总支持=5、末帧仍为 113。

拒绝 K4-012：

```text
1 / 1 / 1 / 1 / 1 / 38
```

因为非空 ID 集合包含两个 actor，且末帧发生 switch。

### P3.3 receiver 本身不能是并入者

在 pre/post raw window 计算 RECEIVER 相对 target corridor 的：

- signed lateral；
- heading error；
- longitudinal speed；
- lane token chain；
- lateral span。

RECEIVER 必须在 pre 时已经是 target stream 的稳定成员。若 receiver 自己从侧支路并入 SUBJECT 后方，输出：

```text
FAIL / RECEIVER_NOT_ESTABLISHED_ON_TARGET_STREAM
```

这条规则防止 K4-007/K4-014 类型的交互方向反转在未来以平行模式重新出现。

### P3.4 corridor 分支歧义

旧 `_best_chain` 在候选数 >1 时按 heading/gap/token 贪心选第一条，且相同 heading 时由 token 字典序决定。

最终版：

1. 在 `graph_hops≤2` 内枚举小型候选 chain；
2. 分别评估 receiver raw identity 支持；
3. 若只有一个 chain PASS，使用该 chain；
4. 若多个 chain PASS 但得到同一 receiver、同一纵向次序且投影差在容差内，可折叠为等价；
5. 若多个 chain 对应不同 receiver 或次序，`ABSTAIN/AMBIGUOUS_RECEIVER_CORRIDOR`；
6. evidence 保存每条 chain 的 edge、candidate count、receiver ID 和首个失败原因。

不得简单规定 `candidate_count==1`，因为 K4-010/K4-011 的合法 corridor 也存在等航向上游/下游分支。

### P3 验收

- K4-012 固定不 PASS，首要 reason 为 identity switch；
- 静止、低速、对向、横穿 receiver 不 PASS；
- K4-011 的一个 null 帧不被误杀；
- 中间车辆使 path-clear FAIL；
- 多 chain 不同 receiver 时 ABSTAIN；
- lane→connector→lane token 变化但同一 receiver 可持续通过。

---

## P4：CPU/2 GiB 运行时重构

**状态**：`done / resource_contract_v2_authorized`

**实施记录（2026-07-26）**：提交 `7f35a9c` 已落地流式 worker A、独立 audit worker 编排、clean-worktree
code-state 检查、每 batch 的 RSS/cgroup stop check，以及 K4 读取缓存的安全释放。K4 16-scene 原始重放与
审核渲染均在 2 GiB cgroup 下完成。为消除编排器的隐藏启动开销，提交 `7104f5c` 将
`motion_proj.runtime` 改为保持原 API 的按需导入：`runtime.atomic` 导入后 RSS 从约 378 MB 降至约 18 MB，
且公共 API 回归测试通过。

在干净 worktree `/root/autodl-tmp/motion_proj_final_clean_7104f5c`、commit `7104f5c` 上，冻结正式配置运行
`/root/autodl-tmp/runs/event_first/N1-EVENT-CUTIN-FINAL-01/v71_n1-event-cutin-final-01__receiver-cutin-final-v1__s0__20260726T121042935837Z__a6b12de0`
先完成 K4 regression，随后 mining 启动前记录 `process_rss_bytes=20,705,280`、
`cgroup_memory_current_bytes=1,523,929,088`，仍高于冻结的 `1,350,000,000` start 合同，因而安全 `FAILED`
于任何 evaluation scene 之前。32/96-scene development smoke 虽在开发启动阈值下完成 stop-threshold 检查，
但不满足正式 start 合同，不能替代此结论。已仅对本任务读过的 nuScenes/QA 文件执行
`POSIX_FADV_DONTNEED`，未杀死 Cursor、Jupyter、TensorBoard 或其他用户服务，也未改写正式阈值；按 P7.5C，
这是关键资源合同失败，后续只可做独立拒绝裁决与报告，不得启动 675-scene mining。

**复开记录（2026-07-26）**：上段结论保持为 V1 历史事实。用户已改变外部资源前提并明确授权继续，
`memory.max` 现为 120 GiB；P4 不再修改实现，只新增 Resource Contract V2 配置并恢复 675-scene formal。

### P4.1 保留现有有效修复

必须保留 `N1-F15`、`N1-F16`、`N1-F17` 已验证机制：

- lightweight map reader；
- 单 location map cache；
- `ijson` 元数据投影；
- 启动前 config contract；
- batch=32；
- `POSIX_FADV_SEQUENTIAL/DONTNEED`；
- file-handle JSON parsing；
- batch 后 `gc.collect` 和 `malloc_trim`；
- RSS + cgroup current 双监控；
- 异常结构化 `FAILED`。

### P4.2 去除内存常驻

修改 `resim/event_first_n1_cutin.py`：

- 不保留全量 `scene_results`；
- scene 结果立即写 JSONL；
- aggregate 使用 Counter 和小型 PASS/ABSTAIN 索引；
- 不在 worker A 导入 audit module；
- event pool 不再嵌入全部 8k+ transition detail；
- calibration 也按 scene 及时释放；
- 旧 30 MB K4 `event_pool.json` 使用 `ijson`/流式 adapter 读取。

### P4.3 一个逻辑 run、两个 worker

建议实现：

```text
scripts/run_n1_cutin_final.py            # 轻量 orchestrator
resim/event_first_n1_cutin.py            # worker A，支持 internal stage
scripts/build_n1_cutin_audit.py           # worker B
```

正式 run 只允许 orchestrator 创建目录和终态。worker 不可单独把 run 标成研究 COMPLETE。

中间 marker：

```text
RUNNING
stages/MINING_COMPLETE
stages/AUDIT_COMPLETE
```

最终唯一 marker：

```text
AWAITING_HUMAN_REVIEW
或 REJECTED
或 FAILED
```

### P4.4 资源 smoke

按顺序：

1. synthetic/unit；
2. K4 16-scene replay；
3. 32-scene development；
4. 96-scene development，覆盖上次 SIGKILL 点；
5. 只有前四项资源合同通过才允许 675-scene formal。

development run 可以 dirty，但必须新 ID 且 `formal=false`。正式 run 必须 clean git。

### P4 验收

- worker A 不导入 Matplotlib、PIL、OpenCV、Shapely；
- 96 scenes 后 RSS/cgroup 未越 stop 阈值；
- 全量 transition 不常驻；
- mining 完成后 worker A 进程已退出；
- 失败 run 不留活动 `RUNNING`；
- 不改第四轮 cache/run。

---

## P5：审核页面 V2 与 JSONL validator

**状态**：`done`

**实施记录（2026-07-26）**：提交 `7ef2d00` 已提供逐 raw-frame topdown、source/target centerline、
SUBJECT/RECEIVER 定向 box 与轨迹、raw 2 Hz signals、actor-ID switch 彩色标注、相机不可用固定警告和
严格 JSONL validator。K4 visual QA-only package 位于
`/tmp/n1-cutin-k4-visual-qa-20260726201000`：7 条 case、30 张 topdown、7 张 signals；blind 页只显示
`primary`/`diagnostic` 中性层名，不泄露 `PASS/FAIL/ABSTAIN` 或 machine reason。validator 的重排拒绝和
hash 保持测试均已通过。

目标是让评审直接回答四个 component，不要求手工分析原始大 JSON。

### P5.1 每条 case 的 raw 时间轴

至少覆盖：

```text
3 个 pre raw keyframe
crossing 最近 raw keyframe
3 个 post raw keyframe
```

若 crossing 与 pre/post 重复，仍保留顺序和角色说明。每帧显示：

- frame；
- 相对时间；
- `raw_2hz` 标志；
- SUBJECT/RECEIVER actor ID；
- source/target/corridor token 简写；
- 当前 `d_target`、heading、gap、receiver rank。

### P5.2 逐帧俯视证据

不再只有一张聚合轨迹图。每个 raw frame 的 topdown 至少显示：

- SUBJECT 和 RECEIVER 定向 box；
- 可选 FRONT 和所有 SUBJECT–RECEIVER 中间车辆；
- source/target centerline；
- target corridor 方向箭头；
- 轨迹历史和未来段；
- 当前 receiver rank；
- actor ID 简写。

另生成 corridor-aligned `(s,d)` 图，直接显示：

```text
SUBJECT 是否从侧向进入
RECEIVER 是否始终在后
是否发生 ID/rank 切换
```

### P5.3 信号曲线

每条 case 输出：

```text
signals/<audit_id>.png
```

包含：

1. subject signed `d_target`；
2. subject box-inside 状态；
3. subject heading error；
4. subject/receiver longitudinal speed；
5. bumper gap；
6. nearest rear actor ID/rank；
7. raw/interpolated provenance；
8. first hard failure frame。

### P5.4 相机只作可选证据

对 CAM_FRONT：

1. 检查文件存在；
2. 投影 SUBJECT/RECEIVER box；
3. 计算 box 是否落在图内、面积和截断；
4. 只有角色可见才在页面称其为相机证据；
5. 无双角色可见时显示固定警告。

其他相机仅在本地文件存在时使用，不设 completeness gate。

### P5.5 blind/debug 分离

```text
audit/index.html        # 人工盲审，不显示聚合 machine status/reason
audit/debug_index.html  # 工程调试，显示 PASS/FAIL/ABSTAIN 与 reason
```

盲审页可显示原始信号，不显示：

- machine overall status；
- first failure reason；
- K4 预期；
- calibration label；
- 旧 reviewer notes。

### P5.6 分层审核

主审核：

```text
primary_pass
```

只从 strict PASS population 抽样，用于 precision。

诊断审核：

```text
diagnostic_abstain
```

按 ABSTAIN reason 分层最多抽 10 条，用于 coverage/证据检查，不混入 PASS precision。

不要求对所有 FAIL 做人工审核；可从每个主要 FAIL reason 抽 1 条 engineering sanity，但单独报告。

### P5.7 validator 修复

现有 validator 比较 audit ID 集合，但没有真正拒绝重排。v2 必须校验：

```text
[row.audit_id for row in reviewed]
==
[row.audit_id for row in template]
```

同时校验：

- 行数、顺序、重复、字段集合；
- evidence/panel/signal hash；
- 非 review 字段逐字节不变；
- TP 四项全 VALID、failure_codes 为空；
- FP 至少一个 INVALID、failure_codes 非空；
- UNCERTAIN 至少一项 UNCERTAIN 且无 INVALID；
- reviewer/notes 非空；
- primary 与 diagnostic 分开聚合；
- validator 不写 verdict、不启动 N2。

### P5.8 audit 构建的内存约束

worker B：

- 一次只加载一个 scene、一个 case；
- 每张图保存后 `plt.close`、关闭 PIL handle；
- 不使用保留全部 selected scenes 的 `scene_cache`；
- 逐 case 写 evidence 和 hash；
- HTML 只引用静态文件，不嵌入 base64 大图；
- 默认不运行后端服务。

### P5 验收

用 K4-009～015 目视 QA：

- K4-009 能看到边界进入证据；
- K4-010/K4-011 crossing 不再被五帧抽样跳过；
- K4-012 的 `1→38` switch 显式变色和标注；
- K4-015 的 post heading 18°～22° 清楚显示；
- CAM_FRONT 无角色时页面明确警告；
- blind 页不泄漏 machine verdict；
- validator 确实拒绝重排。

---

## P6：测试、K4 回归和配置冻结

**状态**：`done`

**实施记录（2026-07-26）**：提交 `7f35a9c` 已冻结 final config、narrow tests 与两个 K4 replay。
快速 evidence regression 18/18 通过、15 个 human FP 的 PASS 数为 0、K4-010/011 均 PASS；原始
annotation/map 16-scene replay 也通过，报告为
`/tmp/n1-cutin-k4-scene-replay-20260726200800/K4_SCENE_REPLAY.json`，`uses_interpolated_physics=false`。
K4-009 按既定边界政策为 `ABSTAIN/BOUNDARY_RAW_ENTRY_EVIDENCE`；K4-012 的 raw map 先证明
`SOURCE_TARGET_NOT_PARALLEL`，同时 `all_reasons` 保留 `RECEIVER_IDENTITY_SWITCH`。旧 v1 的 `1→38`
序列与 v2 枚举窗口不同，两条序列均写入重放报告，未被静默等同或伪造。

### P6.1 单元测试

至少覆盖：

```text
test_branch_merge_is_abstain_and_never_machine_positive
test_route_continuation_not_parallel_lane_change
test_parallel_lane_geometry_requires_local_overlap
test_post_heading_instability_rejects_map_jitter
test_raw_pre_outside_and_post_inside_pass_standard_cutin
test_interpolation_cannot_supply_hard_support
test_raw_accumulated_yaw_uses_unwrapped_angles
test_stationary_receiver_rejected
test_negative_longitudinal_receiver_rejected
test_single_missing_receiver_frame_allowed_with_unique_identity
test_receiver_identity_switch_rejected_even_when_support_count_passes
test_last_post_receiver_anchor_required
test_intermediate_actor_breaks_path_clear
test_multiple_corridors_different_receivers_abstain
test_equivalent_corridors_same_receiver_can_collapse
test_old_v1_evidence_adapter
test_review_validator_rejects_reordering
test_review_validator_preserves_all_hashes
test_mining_worker_does_not_import_heavy_render_dependencies
test_streaming_aggregate_matches_reference
```

### P6.2 K4 快速 evidence 回归

```bash
PYTHONPATH=. python scripts/replay_n1_cutin_k4_evidence.py \
  --fixture tests/fixtures/n1_cutin_k4 \
  --output-root /tmp/n1-cutin-k4-evidence-v2
```

要求：

| case | 要求 |
|---|---|
| K4-010 | PASS |
| K4-011 | PASS |
| K4-009 | 目标 PASS；ABSTAIN 可接受但必须登记非阻塞 false-negative 风险 |
| K4-012 | 不 PASS，identity switch reason |
| K4-015 | 不 PASS，parallel/post-heading reason |
| 其余 13 | 不 PASS；branch mode 固定 unsupported |
| 全部 15 human FP | 0 个 PASS |

K4 修复回放只能称 calibration regression，不得称 final precision。

### P6.3 K4 16-scene 全链重放

从原始 nuScenes annotation 和 map 重新构造，不只读取旧 evidence：

```bash
PYTHONPATH=. python scripts/replay_n1_cutin_k4_scenes.py \
  --config configs/resim/event_first_n1_cutin_final_v1.yaml \
  --fixture tests/fixtures/n1_cutin_k4 \
  --output-root /tmp/n1-cutin-k4-scene-replay
```

比较：

- event ID；
- source/target token；
- raw frame；
- status/reason；
- receiver ID sequence；
- record hash；
- uses_interpolated_physics=false。

### P6.4 统计测试

测试分母：

```text
precision = TP / (TP + FP)
uncertain 不进入 determinate，但必须进入 uncertain_fraction
ABSTAIN machine population 不得静默丢弃
denominator=0 时 precision/Wilson 为 null，不是 0
scene coverage 按唯一 scene 计算
```

### P6 验收

- narrow tests 全通过；
- K4 blocking TP 保留；
- K4 15 FP 无 PASS；
- 全链重放和 evidence 快速回归一致；
- 流式与 reference aggregate 一致；
- 配置在查看 final evaluation 前冻结并提交。

---

## P7：全新 675-scene formal、人工审核和三态终局

**状态**：`done / rejected / stop_nuscenes_cutin_mining_too_sparse`

**实施记录（2026-07-26）**：P4 的 clean-worktree formal run 已因冻结 start 资源合同失败，未读取任何
evaluation scene、未生成候选或 prospective 人工审核包。按 P7.5C 已在干净 worktree、commit `d88d5e2` 生成独立
resource-contract rejection adjudication：
`/root/autodl-tmp/runs/event_first/N1-EVENT-CUTIN-FINAL-RESOURCE-AUDIT-01/v71_n1-event-cutin-final-resource-audit-01__resource-contract-reject-v1__s0__20260726T121624740059Z__025850f8`。
该 run 固定父 preflight/failure/resolved/K4 的 SHA256，记录 `1,523,929,088 - 1,350,000,000 = 173,929,088` bytes
启动合同超额，终态 `REJECTED/stop_nuscenes_cutin_mining`、`evaluation_scene_count_started=0`、
`human_audit_created=false`、`n2_authorized=false`。K4 regression 通过仅为 calibration regression，不能替代
prospective formal/human review。

**复开记录（2026-07-26）**：V1 的 parent 与 resource rejection run 均保持不可变；用户已开放 120 GiB
内存并授权继续。本次使用 `event_first_n1_cutin_final_resource_v2.yaml`、clean commit、新 config fingerprint
与不可复用 run ID 重启 P7.1。只有资源阈值变化；evaluation scenes、strict 判据、K4 和人工审核合同不变。

**scene 计数修复记录（2026-07-26）**：V2 首次 clean formal 在 K4 后、任何 evaluation scene 前因
`675 != 669` fail closed。独立复算为 official train 700 scenes，42 个 calibration scenes 中仅 25 个属于
train、另 17 个属于 val，所以冻结集合差实际为 675。当前仅把 `expected_scene_count` assertion 修正为 675；
`set(train) - set(all_calibration_scenes)` 的实际 split、所有 scene identity 和研究门槛均未改变。失败 run
`...T142634031503Z__5c8c65d7` 保留；修复后必须使用新 commit、fingerprint 和 run ID。

**最终实施记录（2026-07-26）**：clean commit `beee1de` 的新 formal run
`/root/autodl-tmp/runs/event_first/N1-EVENT-CUTIN-FINAL-01/v71_n1-event-cutin-final-01__receiver-cutin-final-v1__s0__20260726T142941598714Z__883fae9a`
完成 675/675 scenes，calibration 42 scenes、交集 0；strict status 为 `PASS=1`、`FAIL=200`、
`ABSTAIN=1,556`，PASS 只覆盖 1 scene。K4、raw-only、资源合同均通过，但冻结 3 candidates / 3 scenes
machine-readiness 失败，故唯一终态为 `REJECTED/stop_nuscenes_cutin_mining_too_sparse`。

### P7.1 final formal

配置：

```text
configs/resim/event_first_n1_cutin_final_v1.yaml
```

正式运行前必须：

- clean git；
- K4 adjudication 已 `REJECTED`；
- K4 regression 全通过；
- 96-scene resource smoke 通过；
- config fingerprint 冻结；
- calibration/evaluation scene 交集为 0；
- 不读取 final 人工结果进行阈值调整。

正式产物：

```text
resolved.yaml
manifest.json
calibration_audit.json
scene_metrics.jsonl
transition_diagnostics.jsonl
strict_candidates.jsonl
strict_event_pool.json
summary.json
metrics.jsonl
audit/index.html
audit/debug_index.html
audit/evidence/*.json
audit/topdown/*
audit/signals/*
audit/camera/*
audit/review_template.jsonl
audit/review_working.jsonl
audit/HUMAN_REVIEW_PROMPT.md
AWAITING_HUMAN_REVIEW | REJECTED | FAILED
```

### P7.2 machine readiness

```yaml
machine_readiness:
  min_strict_pass_candidates: 3
  min_strict_pass_scenes: 3
  require_k4_regression: true
  require_raw_evidence_only: true
  max_resource_contract_violations: 0
```

matched negative/pair 不在本 gate。

若 PASS <3 或 scenes <3：

```text
REJECTED / stop_nuscenes_cutin_mining_too_sparse
```

不为数量放宽规则。

### P7.3 primary PASS 抽样

```yaml
review:
  primary_target_count: 30
  primary_max_count: 40
  max_primary_per_scene: 1
  selection: deterministic_sha256
  abstain_diagnostic_max_count: 10
```

政策：

- PASS population ≤40：审核全部；
- PASS population >40：按 scene 去重后 SHA256 抽 30；
- scene 不足时可允许第二条/scene，但必须报告 clustered count；
- diagnostic ABSTAIN 单独模板、单独聚合；
- FAIL 不混入 primary precision。

### P7.4 完整人工提示词

在交给用户前，必须生成独立可执行的 `HUMAN_REVIEW_PROMPT.md`，完整包含：

- 目的与非目标；
- receiver-centric 定义；
- blind/debug 入口；
- raw 2 Hz 与插值边界；
- 四个 component verdict；
- failure codes；
- 页面操作；
- K4 不可读取的 calibration 信息；
- JSONL 填写格式；
- hash 和顺序保护；
- primary/diagnostic 分层；
- 聚合阈值；
- 完成后的精确 validator 命令；
- 不启动 N2 的影响。

agent 到 `AWAITING_HUMAN_REVIEW` 后停止。人工 verdict 只能由用户或指定评审填写。

### P7.5 最终终局

三种终局都关闭本计划。

#### A. `done / cutin_pool_pass`

全部满足：

```text
primary determinate reviewed ≥ 15
TRUE_POSITIVE ≥ 8
TRUE_POSITIVE scenes ≥ 5
determinate precision ≥ 0.80
Wilson 95% precision lower bound ≥ 0.50
UNCERTAIN fraction ≤ 0.15
K4 blocking regression 通过
15 个 K4 human FP 无 PASS
资源合同通过
```

动作：

- 最终 seed pool 只包含人工 TP；
- machine PASS 未审核部分不得当真值；
- 冻结系统；
- N2 仍不自动启动，由用户另行决定。

#### B. `done / usable_but_sparse`

全部满足：

```text
已审核全部 primary PASS population
未满足 A 的正式统计规模门槛
TRUE_POSITIVE ≥ 3
TRUE_POSITIVE scenes ≥ 3
determinate precision ≥ 0.80
UNCERTAIN fraction ≤ 0.25
K4 blocking regression 通过
资源合同通过
```

动作：

- 冻结少量人工确认 seed；
- 明确“不支持总体 precision 的强统计声明”；
- 不为增加数量再调规则；
- 本计划关闭。

#### C. `rejected / stop_nuscenes_cutin_mining`

任一：

```text
不满足 A，也不满足 B
primary precision < 0.80
TRUE_POSITIVE < 3
K4-010 或 K4-011 不 PASS
任一 K4 human FP 重新 PASS
hard evidence 依赖插值
关键资源合同失败
人工证据包无法可靠辨认角色/时序
```

动作：

- 写独立 adjudication run；
- 更新失败账本；
- 停止 nuScenes cut-in 挖掘；
- 不启动下一轮阈值优化。

---

## P8：事实源、报告和提交

**状态**：`done / final_sparse_rejection_reported`

**实施记录（2026-07-26）**：已更新 `RESEARCH_STATUS.md`、`RESEARCH_FAILURES.md`、`EXPERIMENTS.md`、
`README.md`，并新增 `N1_CUTIN_FINAL_BASELINE.md`、`N1_CUTIN_FINAL_REPORT.md` 和
`docs/n1-cutin-final-resource-rejection-human-review/`。后者提供完整 `HUMAN_REVIEW_PROMPT.md`、空白
`review_template.jsonl` / `review_working.jsonl` 和不可变 SHA256；它只请人工复核资源终止证据，明确不是
不存在的 prospective candidate 标注包。相关回归 `25 passed`，parent/rejection 的 preflight、failure、K4、
summary、manifest SHA256 已逐项复核。代码提交为 `7104f5c` 与 `d88d5e2`；现有文档在开始前已由用户暂存，
为保留这些内容及其索引状态，文档改动未被 agent 自动 commit，留作人工审核/提交。

**复开记录（2026-07-26）**：上述交付是 V1 资源拒绝终局的历史包，继续保留但不作为本次最终候选人工审核包。
P8 将在 V2 formal 到达 `AWAITING_HUMAN_REVIEW` 或新的真实终局后重新更新，并交付对应的完整审核材料。

**最终实施记录（2026-07-26）**：已更新计划、`RESEARCH_STATUS.md`、`RESEARCH_FAILURES.md`、
`EXPERIMENTS.md`、`README.md`、baseline 与 final report。独立稀疏终局人工包位于
`/root/autodl-tmp/runs/event_first/N1-EVENT-CUTIN-FINAL-SPARSE-AUDIT-01/v71_n1-event-cutin-final-sparse-audit-01__human-review-v1__s0__20260726T145456566329Z__d3ceeef5`：
1 primary + 3 diagnostic、18 PNG，immutable set SHA256
`949aed9405721643613a72f9947cbea1a47e94caec4f8f14bc5e1d491b41ec7a`。空白 validator 按预期
fail closed，所有 immutable/source hash、PNG 解码、盲法与代表图视觉 QA 已通过；人工 verdict 未代填。

### P8.1 必须更新

```text
docs/RESEARCH_STATUS.md
docs/RESEARCH_FAILURES.md
docs/EXPERIMENTS.md
docs/README.md
```

新增：

```text
docs/N1_CUTIN_FINAL_BASELINE.md
docs/N1_CUTIN_FINAL_REPORT.md
```

第四轮 adjudication 后应登记新的防重复项，编号按当时账本顺序分配，至少覆盖：

- branch merge 13/13 FP；
- support-count 漏掉 receiver identity switch；
- post heading 过宽导致弯道 map jitter；
- CAM_FRONT 五帧页面证据截断；
- 最终版为何不是旧阈值微调。

### P8.2 报告边界

最终报告必须区分：

```text
观察事实
人工标签
机器回归
prospective final 结果
工程资源结果
尚未知
最终终止原因
```

禁止：

- 把 K4 3/3 修复回放写成 final precision；
- 把 `usable_but_sparse` 写成高召回；
- 把 `REJECTED` 写成 nuScenes 没有 cut-in；
- 把 ABSTAIN 隐藏；
- 把无人审 machine PASS 写成 seed pool。

### P8.3 推荐 commit 切分

```text
research(event): 固化第四轮 cut-in 人审拒绝
refactor(event): 提取 cut-in strict v2 证据与流式聚合
fix(event): 收紧平行主体与接收车完整时序
feat(audit): 增加逐帧俯视与信号审核页
test(event): 固化 K4 cut-in 人工回归
docs(research): 登记 cut-in 最终迭代与终止规则
```

每个 commit：

```bash
git diff --cached --check
git diff --cached
```

研究 commit 正文写明：

- task/run ID；
- split；
- seed；
- commit/config/data fingerprint；
- evidence 路径；
- 测试命令和真实结果；
- `n2_authorized=false`。

---

# 7. 建议冻结配置

新配置，不覆盖第四轮：

```text
configs/resim/event_first_n1_cutin_final_v1.yaml
```

建议：

```yaml
schema_version: receiver-centric-cutin-final-v1
task_id: N1-EVENT-CUTIN-FINAL-01
seed: 0
repo_root: /root/autodl-tmp/motion_proj
dataset_root: /root/autodl-tmp/data/nuscenes
cache_dir: /root/autodl-tmp/data/occgs/processed_10Hz/n1_receiver_cutin_final_v1
run_root: /root/autodl-tmp/runs/event_first/N1-EVENT-CUTIN-FINAL-01
interpolate_n: 4
require_clean_git: true

calibration:
  include_second_review: true
  include_third_review: true
  include_fourth_review: true
  fourth_review_sha256: 983e4b7a4160ff7aec127343b5ca3e1e9a1f07f06d799f4db9695fa241851321
  expected_label_count: 67
  expected_scene_count: 42
  exclude_all_calibration_scenes_from_evaluation: true

evaluation:
  split_name: train
  expected_scene_count: 675
  max_scenes: null

taxonomy:
  allowed_positive_modes:
    - parallel_lane_change
  unsupported_modes:
    receiver_branch_merge: UNSUPPORTED_BRANCH_MERGE_MODE

strict:
  hard_evidence_source: raw_2hz_annotations
  interpolation_can_pass_hard_gate: false
  ambiguity_policy: ABSTAIN
  lane_half_width_m: 1.75
  lane_width_source: configured_nominal_fallback

  subject:
    raw_pre_keyframes: 3
    raw_post_keyframes: 3
    min_pre_center_outside_keyframes: 2
    min_post_box_inside_keyframes: 2
    min_pre_center_lateral_m: 1.85
    max_post_center_lateral_m: 0.80
    min_lateral_convergence_m: 1.40
    min_lateral_convergence_consistency: 0.80
    min_pre_side_consistency: 0.80
    min_source_target_shift_m: 2.0
    max_source_target_shift_m: 6.0
    max_source_target_heading_error_deg: 10.0
    max_pre_heading_error_deg: 15.0
    max_post_heading_error_deg: 10.0
    max_accumulated_yaw_change_deg: 15.0
    min_median_speed_mps: 1.0
    min_settle_duration_s: 1.0
    timestamp_tolerance_s: 0.02

  receiver:
    class_prefix: vehicle.
    min_raw_pre_support: 2
    min_raw_post_support: 2
    min_total_non_null_support: 5
    max_missing_required_frames: 1
    require_unique_non_null_nearest_rear_id: true
    require_last_post_anchor: true
    require_nearest_rear_persistence: true
    require_path_clear: true
    min_local_displacement_m: 1.0
    min_median_longitudinal_speed_mps: 1.0
    max_heading_error_deg: 15.0
    max_centerline_distance_m: 1.5
    min_bumper_gap_m: 0.5
    max_bumper_gap_m: 40.0

  corridor:
    graph_hops: 2
    enumerate_candidate_chains: true
    max_edge_heading_error_deg: 15.0
    max_edge_endpoint_gap_m: 4.0
    different_receiver_policy: ABSTAIN

runtime:
  cpu_only: true
  scene_batch_size: 32
  max_live_map_locations: 1
  streaming_aggregate: true
  split_mining_and_audit_processes: true
  require_posix_page_cache_control: true
  max_start_cgroup_current_bytes: 1350000000
  warn_process_rss_bytes: 700000000
  stop_process_rss_bytes: 900000000
  warn_cgroup_current_bytes: 1750000000
  stop_cgroup_current_bytes: 1950000000

machine_readiness:
  min_strict_pass_candidates: 3
  min_strict_pass_scenes: 3
  require_k4_regression: true
  require_raw_evidence_only: true

audit:
  primary_target_count: 30
  primary_max_count: 40
  max_primary_per_scene: 1
  abstain_diagnostic_max_count: 10
  raw_pre_keyframes: 3
  raw_post_keyframes: 3
  include_crossing_keyframe: true
  include_framewise_topdown: true
  include_corridor_aligned_view: true
  include_signal_timeseries: true
  use_camera_only_when_file_exists: true
  require_all_six_cameras: false
  build_debug_page: true
  build_blind_page: true

human_gates:
  pass_min_reviewed_determinate: 15
  pass_min_true_positive: 8
  pass_min_positive_scenes: 5
  pass_min_precision: 0.80
  pass_min_wilson_lower_bound: 0.50
  pass_max_uncertain_fraction: 0.15
  sparse_min_true_positive: 3
  sparse_min_positive_scenes: 3
  sparse_min_precision: 0.80
  sparse_max_uncertain_fraction: 0.25
  sparse_require_all_primary_pass_reviewed: true

stop_rule:
  final_iteration: true
  no_further_threshold_iteration: true
  never_start_n2_from_this_run: true
```

阈值只允许因以下原因在 final formal 前修改：

- 单位错误；
- raw frame stride 错误；
- yaw unwrap 实现错误；
- 配置字段未被实际消费；
- synthetic/K4 fixture 证明实现未满足本文语义。

任何修改必须在 commit 和 baseline 文档中说明。不得根据 final 675-scene candidate 数或人工结果修改。

---

# 8. 未来执行命令合同

以下命令是实现后的目标接口；若实现时调整参数名，必须同步更新本文件、README 和测试，不能只在对话里说明。

## 8.1 第四轮只读复验

```bash
PYTHONPATH=. python scripts/validate_n1_cutin_review.py \
  --run-dir /root/autodl-tmp/runs/event_first/N1-EVENT-CUTIN-01/v71_n1-event-cutin-01__receiver-cutin-v1__s0__20260725T173015103731Z__5b1634e3 \
  --review-file /root/autodl-tmp/runs/event_first/N1-EVENT-CUTIN-01/v71_n1-event-cutin-01__receiver-cutin-v1__s0__20260725T173015103731Z__5b1634e3/audit/review_working.jsonl
```

## 8.2 测试

```bash
PYTHONPATH=. pytest -q \
  tests/test_adjudicate_n1_cutin_audit.py \
  tests/test_cutin_receiver.py \
  tests/test_n1_cutin_config.py \
  tests/test_n1_cutin_review.py \
  tests/test_n1_cutin_streaming.py \
  tests/test_n1_cutin_audit_v2.py
```

## 8.3 K4 回归

```bash
PYTHONPATH=. python scripts/replay_n1_cutin_k4_evidence.py \
  --fixture tests/fixtures/n1_cutin_k4 \
  --output-root /tmp/n1-cutin-k4-evidence-v2

PYTHONPATH=. python scripts/replay_n1_cutin_k4_scenes.py \
  --config configs/resim/event_first_n1_cutin_final_v1.yaml \
  --fixture tests/fixtures/n1_cutin_k4 \
  --output-root /tmp/n1-cutin-k4-scene-replay
```

## 8.4 资源 smoke

```bash
PYTHONPATH=. python scripts/run_n1_cutin_final.py \
  --config configs/resim/event_first_n1_cutin_final_v1.yaml \
  --allow-dirty-development \
  --max-evaluation-scenes 96 \
  --output-root /tmp/n1-cutin-final-resource-smoke
```

## 8.5 正式运行

```bash
PYTHONPATH=. python scripts/run_n1_cutin_final.py \
  --config configs/resim/event_first_n1_cutin_final_v1.yaml
```

正式命令返回后必须从 stdout/manifest 复制真实 run path，不得手工猜 run ID。

## 8.6 人工完成后的校验

```bash
PYTHONPATH=. python scripts/validate_n1_cutin_review.py \
  --run-dir <FINAL_RUN_DIR> \
  --review-file <FINAL_RUN_DIR>/audit/review_working.jsonl
```

validator 只聚合，不写终态。用户确认后再运行独立 adjudication：

```bash
PYTHONPATH=. python scripts/adjudicate_n1_cutin_audit.py \
  --parent-run <FINAL_RUN_DIR> \
  --review-file <FINAL_RUN_DIR>/audit/review_working.jsonl \
  --config configs/resim/n1_cutin_final_adjudication_v1.yaml
```

---

# 9. 与历史失败账本的逐项对应

| 失败/风险 | 本计划如何解除 | 为什么不是旧路线重跑 |
|---|---|---|
| `N1-F05` target 多 incoming 被当行为 | branch merge 不再 machine-positive | 删除错误事件族，不调 graph gap |
| `N1-F06` 插值被当物理证据 | hard checks 只读 raw 2 Hz，schema 写 provenance | 不增加人造观测 |
| `N1-F07` 单时刻关系 | 全窗 receiver ID/rank/gap/path-clear | 修复身份语义，不调单帧距离 |
| `N1-F08` provenance/audit 不足 | K4 独立 adjudication、v2 hash、blind/debug 分离 | 不改 parent run |
| `N1-F09` authenticity/control 混淆 | matched negative 从 final event gate 移除 | 不缩短 control，只改变本轮目标边界 |
| `N1-F11` CAM_FRONT 不等于完整证据 | 逐帧 topdown/曲线为主，相机可选 | 不下载或猜测角色 |
| `N1-F12` 地图分支收敛不是横移 | parallel-only + raw body entry + post heading | 不再用 branch topology |
| `N1-F13` receiver 污染 | receiver 预先在 target stream、唯一 ID、方向和速度 | 不复用 source rear |
| `N1-F15` 重型 map API OOM | lightweight map、单 location、无 polygon API | 不以删地图换内存 |
| `N1-F16` 延迟配置缺项 | v2 config contract 启动前检查 | 不等全量运行后才失败 |
| `N1-F17` 583 MB 页缓存压力 | 保留 fadvise、流式 aggregate、分进程 audit | 不杀服务、不减少正式 scenes |
| K4-012 新证据 | unique non-null receiver + last-post anchor | 修复 support-count 漏洞 |
| K4-015 新证据 | parallel/post heading ≤10° | 针对弯道语义，不调 gap |

---

# 10. 最终禁令

以下任一发生，视为计划实施失败：

1. `receiver_branch_merge` 再次进入 machine-positive；
2. 用 `target_incoming_count`、`shared_successor` 或 token change 单独证明 cut-in；
3. 用 10 Hz 插值补齐 raw pre/post、速度、yaw 或 receiver identity；
4. 只在 relation frame 选择 receiver；
5. 允许 `1/1/1/1/1/38` 因支持数够而 PASS；
6. 静止、负纵向速度或横穿对象被当 receiver；
7. 多 corridor 得到不同 receiver 时按 token 字典序选一个；
8. 为 K4-009 放宽全部 pre-entry 规则；
9. 为拒绝 K4-012 事后设置专属 gap 或 actor 条件；
10. 为拒绝 K4-015 使用 scene/token 黑名单；
11. mining 进程导入重型 map/render 依赖；
12. 正式 run 保留全量 scene result 于内存；
13. 强制要求当前不存在的六相机资产；
14. CAM_FRONT 无角色框时靠肉眼猜身份；
15. blind 页泄漏 machine verdict、K4 标签或旧 notes；
16. validator 接受 item 重排；
17. agent 或 LLM 自动填写 verdict；
18. 把 K4 修复回放写成 prospective precision；
19. 把 ABSTAIN 从 coverage 分母静默删除；
20. precision 失败后继续下一轮阈值优化；
21. machine PASS 未经人工确认就进入 seed pool；
22. 自动启动 N2、渲染、训练或下载。

---

# 11. 最终完成定义

> **最终收口（2026-07-26）**：Resource Contract V1 的失败证据保持不变；用户开放资源后，V2 formal 已完成
> 全部 675 scenes，并因 1 PASS / 1 scene 低于冻结 3/3 machine-readiness 而按 P7.5C 关闭。人工审核包和事实源
> 已交付；人工可复核唯一 PASS 与 diagnostics，但结果不能改变 parent 终态、恢复规则迭代或授权 N2。

- [ ] 第四轮 review SHA、18 条结果和 validator 输出已复验；
- [ ] 第四轮独立 adjudication run 为 `REJECTED`；
- [ ] 67 条人工标签/42 scenes 已冻结为 calibration；
- [ ] final evaluation 与全部 calibration scenes 分离，预期 675 scenes；
- [ ] strict v2 三态 schema 完成；
- [ ] branch merge 永不 machine-positive；
- [ ] K4-015 弯道/map jitter 不 PASS；
- [ ] K4-012 receiver switch 不 PASS；
- [ ] K4-010/K4-011 blocking TP PASS；
- [ ] K4 15 个 human FP 无 PASS；
- [ ] 所有 hard evidence 只来自 raw 2 Hz；
- [ ] receiver 动态性、方向、唯一 ID、last-post anchor、rank、gap、path-clear 完整；
- [ ] mining 流式 aggregate 且不导入重型依赖；
- [ ] mining/audit 分进程，96-scene resource smoke 通过；
- [ ] 审核 V2 提供逐帧 box、corridor `(s,d)` 和信号曲线；
- [ ] 相机缺失或角色不可见被显式标记；
- [ ] blind/debug 页面分离；
- [ ] validator 拒绝 hash、字段、行数、顺序或 verdict 语义异常；
- [ ] final formal run 使用 clean commit、seed 0、冻结 config 和不可复用 run ID；
- [ ] 完整人工提示词与 JSONL 模板已生成；
- [ ] 用户人工完成后产生独立 adjudication；
- [ ] 最终只形成 `cutin_pool_pass`、`usable_but_sparse` 或 `stop_nuscenes_cutin_mining` 之一；
- [ ] `RESEARCH_STATUS`、`RESEARCH_FAILURES`、`EXPERIMENTS` 和 final report 已更新；
- [ ] `n2_authorized=false`；
- [ ] 无第五轮 cut-in 规则迭代。

完成后，本计划永久关闭。后续若用户希望研究复杂 branch merge、补六相机、构建 matched controls 或进入 N2，必须作为新的研究问题、使用新的任务 ID、数据授权、预注册和 scene-disjoint 评估，不得继续修改本 final run。
