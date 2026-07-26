# N1-EVENT-KINEMATIC-01 正式结果与第三次人工审核交付

> **运行日期**：2026-07-25
> **Parent 终态**：`AWAITING_HUMAN_REVIEW`（历史不可变）
> **最终研究裁决**：第三次人审 0 TP / 12 FP；独立 adjudication `REJECTED`
> **硬边界**：`n2_authorized=false`，N2 未启动、未解锁

## 1. 结论先行

第三版在 scene-disjoint official train 的 694 scenes 上完成全量筛选：

```text
8,631 stable transitions
  → 1,879 topology-pass
    → 244 original-2-Hz physical-motion-pass
      → 12 persistent front+rear interaction candidates
        → 2 same-actor lane-keeping negatives / 2 pairs
```

12 个候选覆盖 9 scenes，达到候选数 `≥12` 与场景数 `≥6`；negative/pair 只各 2，未达到冻结的
`≥4`。因此：

- `machine_gate_passed=false`；
- parent 按预注册停在唯一 `AWAITING_HUMAN_REVIEW`，没有提前替用户作最终裁决；
- 12/12 候选均已生成完整盲审材料；
- 用户随后完成 12/12：TP=0、FP=12、UNCERTAIN=0；
- 独立 adjudication 已 `REJECTED`；parent 文件和 terminal 没有被回写；
- machine support failure 与 12/12 人工真实性 failure 同时保留。

### 1.1 第三次人审追加事实

- review：
  `/root/autodl-tmp/runs/event_first/N1-EVENT-KINEMATIC-01/v71_n1-event-kinematic-01__kinematic-v1__s0__20260725T092427030639Z__8c2247b6/audit/review_working.jsonl`；
- SHA256：
  `005cd74b874833808435fd2f47387d1d8e446cdea2d3a5cae6146e34bf331e96`；
- subject maneuver `INVALID=12/12`；
- `SUBJECT_NO_LATERAL_MANEUVER=12`、`ROUTE_CONTINUATION=11`、`NORMAL_TURN=1`；
- rear `INVALID=2`、front `INVALID=1`、`MAP_MATCH_JITTER=1`；
- clean adjudication：
  `/root/autodl-tmp/runs/event_first/N1-EVENT-KINEMATIC-AUDIT-01/v71_n1-event-kinematic-audit-01__human-audit-reject-v1__s0__20260725T155754010881Z__4c51f0d9/`；
- adjudication commit `1fbbbc1`，唯一 `REJECTED`，`n2_authorized=false`。

准确根因是：12/12 最终候选验证了地图分支收敛，却没有验证 subject 车身相对独立接收车流发生真实
outside→inside 横移；第三版 corridor 还可能复用 subject 原队列后车。第四版接续见
[`N1_RECEIVER_CUTIN_PREREGISTRATION.md`](N1_RECEIVER_CUTIN_PREREGISTRATION.md)。

## 2. 正式 provenance

| 项目 | 值 |
|---|---|
| Task | `N1-EVENT-KINEMATIC-01` |
| Run ID | `v71_n1-event-kinematic-01__kinematic-v1__s0__20260725T092427030639Z__8c2247b6` |
| Run 根 | `/root/autodl-tmp/runs/event_first/N1-EVENT-KINEMATIC-01/v71_n1-event-kinematic-01__kinematic-v1__s0__20260725T092427030639Z__8c2247b6/` |
| 代码 commit | `aa162ef4dea808ad28ca7e56f1273f106e9c0e49` |
| `code_dirty` | `false` |
| 配置 SHA256 | `8c2247b6e968aa792fec1dfc475a232f3497e379445ca04f3a38fd12254b3b1b` |
| 数据 fingerprint | `4f914ac88d2927f78baa67b91b77eab48c6bb56d87acddfae8580acead1befe3` |
| event-pool canonical SHA256 | `4778abadfc44c830f815efd4c52e544bf23e67f975d0b7df44e52e742289d6bf` |
| event-pool file SHA256 | `c29adfe87ce4b476240a199e0816b7172166bdf5cadef397bd679060ca42727d` |
| artifact-set SHA256 | `5742231a88da08ed73d6cf156913084c2a32ad771b16e27f063eae9b1eb70c6b` |
| audit immutable-set SHA256 | `8696e66dcb5764b414b7e3cf74e89261fff63f137197fe13bf6617724180e168` |
| 开始/结束 UTC | `2026-07-25T09:24:26.534639Z` / `2026-07-25T09:45:01.234559Z` |
| 唯一 terminal | `AWAITING_HUMAN_REVIEW` |
| N2 | `n2_authorized=false` |

数据 fingerprint 覆盖 N0 asset manifest、trainval 的 `scene.json`、`sample.json`、
`sample_annotation.json`、`instance.json`、`category.json`、`log.json`、calibration review SHA
和完整 evaluation scene list。正式运行未读取 sweeps，也未使用 N2 入口。

## 3. Calibration 与 split 隔离

第二次 val 37 条人审只用于第三版设计与校准：

- review SHA256：
  `ae71b31e02faf1d783c36748e629e85acf32a132f35ce2f98102a5f62201dd05`；
- 35/35 旧 `FALSE_POSITIVE` 被第三版拒绝；
- 2 个旧 `TRUE_POSITIVE` 保留 1 个、拒绝 1 个；
- calibration TP recall=`0.5`，FP rejection=`1.0`；
- calibration 共 17 val scenes，与 formal official train evaluation scene-disjoint；
- formal 另排除全部 10 个 mini scenes，最终 evaluation 为 694 scenes。

这些 calibration 数字不是 formal precision，也不能与 train 的未审 12 条混合估计准确率。

## 4. 全量漏斗

### 4.1 Transition 与主体运动学

| 层 | 数量 |
|---|---:|
| 全部 transition candidates | 8,631 |
| route continuation | 5,759 |
| merge topology | 1,660 |
| lane-change topology | 219 |
| unresolved | 993 |
| topology-pass | 1,879 |
| 原始 2 Hz physical-motion-pass | 244 |
| physical merge | 181 |
| physical lane change | 63 |

topology-pass 中，362 条因原始 annotation keyframe 不足为 `UNKNOWN`，1 条因静止/退化 course 为
`UNKNOWN`，1,272 条明确 motion `FAIL`，244 条 `PASS`。速度、加速度、yaw rate 与前后轨迹全部用
`sample.timestamp` 和原始 2 Hz box 计算；10 Hz 只用于候选定位与展示。

### 4.2 Interaction 瓶颈

| 244 个 physical-motion-pass 的 interaction 结果 | 数量 |
|---|---:|
| 中心关键帧缺 front 或 rear | 215 |
| temporal identity / bumper-gap gate 失败 | 17 |
| persistent interaction PASS | 12 |

12 个最终候选全部为 `converging_branch_merge`；63 个 physical `parallel_lane_change` 没有任何一个同时
通过 front+rear 持续交互。这是正式覆盖边界，不能把第三次 audit 外推到一般 lane-change。

候选 scene 分布：

- `scene-0870`：3；
- `scene-0855`：2；
- `scene-0005`、`scene-0669`、`scene-0750`、`scene-0853`、`scene-0860`、
  `scene-1007`、`scene-1018`：各 1。

### 4.3 Machine gate

| 冻结检查 | 结果 | 阈值 | 通过 |
|---|---:|---:|---|
| positive candidates | 12 | ≥12 | 是 |
| candidate scenes | 9 | ≥6 | 是 |
| same-actor negative windows | 2 | ≥4 | 否 |
| same-actor pairs | 2 | ≥4 | 否 |

`audit_ready=true` 只表示有材料可交给用户，不表示 research support pass。

## 5. Pair 失败诊断

在不改阈值、不改正式 event pool 的前提下，对 12 个 positive actor 重放冻结的 30-frame negative 搜索：

| 主阻塞 | positive actors | 机制 |
|---|---:|---|
| 成功找到 negative/pair | 2 | 都在 `scene-0870` |
| 没有 30-frame stable run | 1 | actor 可见/稳定轨迹太短 |
| 所有 30-frame window 与 positive event overlap | 5 | 20 秒 scene 内没有独立同 actor control window |
| lane-keeping window 存在，但 front/rear interaction 全失败 | 4 | 所有合格窗口在中心关键帧缺 front 或 rear |

逐 actor blocker：

| Event | Blocker |
|---|---|
| `scene-0005:22:K3:11:13` | no 30-frame stable run |
| `scene-0669:33:K3:93:95` | 10 个 lane-keeping windows 全部缺 front/rear |
| `scene-0750:14:K3:89:102` | 23/23 windows overlap |
| `scene-0853:11:K3:40:52` | 27/27 windows overlap |
| `scene-0855:19:K3:137:138` | 27/27 windows overlap |
| `scene-0855:1:K3:16:24` | 11 个 lane-keeping PASS windows 全部缺 front/rear |
| `scene-0860:13:K3:63:64` | 2 个 lane-keeping PASS windows 全部缺 front/rear |
| `scene-0870:1:K3:172:173` | 5/5 windows overlap |
| `scene-0870:20:K3:101:102` | paired |
| `scene-0870:5:K3:122:123` | paired |
| `scene-1007:15:K3:46:49` | 4/4 windows overlap |
| `scene-1018:20:K3:128:132` | 2 个 lane-keeping PASS windows 全部缺 front/rear |

所以 pair 失败不是简单的“negative 阈值差一点”：6/10 未配对 actor 根本没有非重叠长窗口，
另外 4/10 缺同等 front+rear interaction。直接把 window 缩短、允许 overlap、改成单侧邻车或换 actor
都会改变预注册研究问题，不能用于本 run 翻案。

## 6. 第三次人工审核包

审核根目录：

`/root/autodl-tmp/runs/event_first/N1-EVENT-KINEMATIC-01/v71_n1-event-kinematic-01__kinematic-v1__s0__20260725T092427030639Z__8c2247b6/audit/`

入口与职责：

| 文件 | 用途 |
|---|---|
| `index.html` | 12 项连续浏览入口 |
| `REVIEW_CHECKLIST.md` | Markdown 清单与逐项 panel |
| `HUMAN_REVIEW_PROMPT.md` | 完整盲审合同、字段与阈值 |
| `review_template.jsonl` | 不可修改的空白模板基准 |
| `review_working.jsonl` | 用户唯一应编辑的文件 |
| `evidence/K3-001.json` … `K3-012.json` | 逐事件只读数据 |
| `panels/K3-001.png` … `K3-012.png` | CAM_FRONT + topdown + metrics 合成面板 |
| `topdown/K3-001.png` … `K3-012.png` | 原始 2 Hz 三车轨迹和 vector map |
| `audit_manifest.json` | 40 个 immutable files 的逐文件 hash |

交付时材料 QA（随后 `review_working.jsonl` 已由用户填写，immutable 文件未改）：

- candidate population=12，audit items=12，全量审核，没有 top-k 丢弃；
- 40 个 immutable file 的 SHA256 已全部复算，0 mismatch；
- 交付时 `review_working.jsonl` 与模板逐字节相同，SHA256
  `77e68c2e9dea76113148dade3f176ef3365f3c4a3162b8b54d8247f222bfc9aa`；
- 空白 review 运行 validator 按预期 fail closed：
  `ValueError: component verdict 非法或缺失: K3-001`；
- `K3-001` 与 `K3-012` 首尾 panel 已目检，图像尺寸、CAM_FRONT、topdown、颜色与 metrics 布局正常；
- 非全部角色都会进入 CAM_FRONT 视野；提示词要求此时使用 topdown，证据仍不足必须判 `UNCERTAIN`。

仓库中的同版提示词快照：
[`N1_KINEMATIC_HUMAN_REVIEW_PROMPT.md`](N1_KINEMATIC_HUMAN_REVIEW_PROMPT.md)。

## 7. 已执行的审核汇总命令

只编辑 `audit/review_working.jsonl` 的 component verdict、overall verdict、failure codes、reviewer 和 notes，
不要改 hash、audit ID 或模板：

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate motionproj
cd /root/autodl-tmp/motion_proj
PYTHONPATH=. python scripts/validate_n1_kinematic_review.py \
  --run-dir /root/autodl-tmp/runs/event_first/N1-EVENT-KINEMATIC-01/v71_n1-event-kinematic-01__kinematic-v1__s0__20260725T092427030639Z__8c2247b6 \
  --review-file /root/autodl-tmp/runs/event_first/N1-EVENT-KINEMATIC-01/v71_n1-event-kinematic-01__kinematic-v1__s0__20260725T092427030639Z__8c2247b6/audit/review_working.jsonl
```

validator 只校验与汇总，没有启动 N2。用户已确认并另建 immutable adjudication run；parent 未改写。

## 8. 下一步突破方向

第三次人审已选择“主体 merge 真实性低”分支并正式 reject。第四版不再调第三版 branch angle，而是另立
receiver-centric 事件：

1. 原始 2 Hz subject center outside→post full-box inside；
2. pre heading alignment 排除主路/路口几何收敛；
3. target corridor 排除 source，RECEIVER 必须来自独立目标车流并保持 pre/post identity；
4. 30-frame control 不缩短且同样要求 receiver；
5. 49 条旧审标签只作 calibration，全部已审 scene 从新 formal train 排除。

完整冻结合同见
[`N1_RECEIVER_CUTIN_PREREGISTRATION.md`](N1_RECEIVER_CUTIN_PREREGISTRATION.md)。

技术设计依据：

- nuScenes schema 将 `sample` 定义为 2 Hz annotated keyframe：
  [nuScenes schema](https://github.com/nutonomy/nuscenes-devkit/blob/master/docs/schema_nuscenes.md)；
- 官方 devkit 的 `box_velocity` 使用相邻 annotation 与实际时间差：
  [nuScenes devkit](https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/nuscenes.py)；
- nuPlan 官方 devkit公开 lane-change/interaction scenario 类型：
  [nuPlan devkit](https://github.com/motional/nuplan-devkit)；
- nuPlan metrics按 lane/lane_connector occupancy、route 与 projected boxes 限定 TTC：
  [nuPlan metrics](https://github.com/motional/nuplan-devkit/blob/master/docs/metrics_description.md)。

这些来源只支持下一版数据与指标设计，不构成启动外部数据、N2 或任何下载的授权。
