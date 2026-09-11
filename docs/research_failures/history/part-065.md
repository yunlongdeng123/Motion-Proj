# 历史原始记录 065

> 冻结历史，不是当前执行规则。当前规则见 [scaling law](../../../auto-research_scaling_law.md) 与 [AGENTS](../../../AGENTS.md)。
> 原文件：`docs/RESEARCH_FAILURES.md`；迁移前提交 `abbd431c`。原有相对链接按原目录解释，证据路径原样保留。
> 按索引读取相关小节；旧哈希、过度门控和资源指令均不重新生效。

### N1-F16：负对照配置契约缺项导致首个正式 K4 工程失败

**失败事实**

- 失败 run：
  `/root/autodl-tmp/runs/event_first/N1-EVENT-CUTIN-01/v71_n1-event-cutin-01__receiver-cutin-v1__s0__20260725T170948229629Z__46186120`；
- 失败代码提交：`f5c9bbe4c819abce42e1cca0b8800e16a77af680`；
- 正式日志：
  `/root/autodl-tmp/runs/event_first/n1k4_formal_f5c9bbe.log`，SHA256
  `b9ac6d3cce2e731f16aad7bc6a068eaf09c54439ad654bb0a3a9d0c58f63a487`；
- calibration 已运行，进入首个 formal evaluation batch 后，在构造 same-actor 30-frame
  lane-keeping negative control 时抛出 `KeyError: min_median_speed_mps`；
- `lane_keeping_features` 实际接收 `kinematics_control`，但该字段只写在 `cutin` 下；同一调用下一步还会需要
  `max_acceleration_mps2`，原 YAML 也遗漏；
- 失败发生在 `event_pool.json`、`summary.json` 与任何研究裁决写入前。因此它是工程失败，不是机器 gate
  reject，更不是第四次人工评测结论；`n2_authorized=false`。

**保留与修复**

- 旧目录不删除、不改造成成功 run；写入结构化 `failure.json` 与 `FAILED`，原 `RUNNING` 被保留为
  `RUNNING.invalidated`；
- 修复提交 `8581d4dcd1bf9a4f92b426c601e1149c804afc5a` 同时补入
  `kinematics_control.min_median_speed_mps=0.5` 和 `max_acceleration_mps2=12.0`；
- 新增启动前 `_validate_config_contract`，在加载 nuScenes metadata 前检查 delayed runtime dependency、
  `receiver_cutin` 审核 schema 与 `never_start_n2_from_this_run=true`；
- 新增 post-run-directory 异常处理：后续未捕获异常自动写 `FAILED/failure.json`、清除活动
  `RUNNING`，并强制 `n2_authorized=false`；
- 27 项相关测试通过后才从新 run ID 完整重跑。

**防重复**

1. development pilot 必须覆盖至少一个 positive actor 的 negative-control 搜索；“positive=1、
   negative=0”不能被误读为该分支已执行；
2. 所有按候选稀疏触发的配置依赖必须在启动时校验，不能等全量运行数分钟后才由 `KeyError` 暴露；
3. 任何残留 `RUNNING` 的异常目录必须先结构化归档，再开始新 run；禁止覆盖、续跑或统计为 research
   reject；
4. 修复配置/异常落盘不授权改变冻结 K4 阈值、评估 scene 或候选排序。

### N1-F17：重复扫描 583 MB 标注文件产生 cgroup 页缓存压力与外部 SIGKILL

**失败事实**

- 失败 run：
  `/root/autodl-tmp/runs/event_first/N1-EVENT-CUTIN-01/v71_n1-event-cutin-01__receiver-cutin-v1__s0__20260725T171746938858Z__5b1634e3`；
- 失败代码提交：`8581d4dcd1bf9a4f92b426c601e1149c804afc5a`；
- 正式日志：
  `/root/autodl-tmp/runs/event_first/n1k4_formal_8581d4d.log`，SHA256
  `7a89e5f5ab88c53a6d9531dedc56a9db302cef8dab144ade7da75a91f3c09191`；
- calibration 与前 96/685 个 evaluation scenes 已执行，随后 shell 报 `Killed`；没有
  `event_pool.json`、`summary.json` 或研究裁决；
- 本层 cgroup `memory.max=2147483648`，事件计数仍为 `oom=0`、`oom_kill=0`，因此不能把信号来源伪写成
  kernel OOM；终态登记为 `external_sigkill_under_cgroup_memory_pressure`；
- `sample_annotation.json` 大小为 `583417244` bytes。失败后进程已消失时，
  `memory.current=1704148992`、file cache=`639545344` bytes；
- 对该只读标注文件执行 `POSIX_FADV_DONTNEED` 后，在没有停止编辑器/Jupyter/TensorBoard 等用户服务的前提下，
  `memory.current` 立即降为 `1169817600`、file cache 降为 `102739968` bytes。

**根因边界**

证据支持以下工程推断：每个 32-scene batch 都顺序扫描 583 MB 标注表，读页长期计入 2 GiB cgroup；
进程 RSS、既有服务和文件页缓存共同逼近硬上限，外部管理层随后发送 SIGKILL。由于无内核日志权限且
`oom_kill=0`，不能声称已证明具体 killer；但页缓存释放的前后差值直接证明了主要可控压力源。

**修复与复验**

- 修复提交 `f13eb0f1e39b608de1c5e698cd678c2dfd8365a4`；
- 所有大型顺序输入在读取前标记 `POSIX_FADV_SEQUENTIAL`，读取后标记
  `POSIX_FADV_DONTNEED`；
- per-scene JSON 改为 `json.load(file_handle)`，不再由 `read_text` 同时常驻字符串和解析对象；
- 每批显式删除 dense scene payload、执行 `gc.collect` 与 glibc `malloc_trim`；
- 每批日志新增 process RSS 与 cgroup current；正式启动若缺 POSIX page-cache control 则 fail closed；
- 30 项相关测试通过。新正式 run 在 96 scenes 的同一死亡点记录
  RSS `602673152`、cgroup current `1707110400` bytes，且继续运行，证明修复覆盖了原路径；
- 成功 run 最终完成 685/685 scenes；最后一批 RSS `510734336`、cgroup current
  `1612763136` bytes，`oom=0`、`oom_kill=0`。它以独立 run ID
  `...T173015103731Z__5b1634e3` 和唯一 `AWAITING_HUMAN_REVIEW` 结束。

**防重复**

1. Python RSS 不是 2 GiB 容器的完整内存分母；必须同时记录 anon、file cache、cgroup current 与
   `memory.events`；
2. 流式解析只限制 Python 对象，不自动释放内核页缓存；反复全表扫描必须有 cache-pressure 策略；
3. 不得以杀死用户服务、跳过正式场景、降低地图分辨率或减少校准标签来换取“成功”；
4. SIGKILL 无法触发 Python exception handler，因此监控器必须把残留 `RUNNING` 另行结构化封存；
5. 该修复只改变 I/O/内存生命周期，不改变 K4 候选、阈值、排序、scene split 或人工门槛。

### 第四版 calibration 冻结结果与禁止矩阵

第四版只用第二、三次全部 49 条人工标签调试阈值，所有 26 个已审 scene 从 formal evaluation 排除。
截至冻结前 development replay：

- 第三次 FP 拒绝 `12/12`；
- 第二次 FP 拒绝 `35/35`；
- 第二次 TP 保留 `1/2`；
- 被保留真例同时满足目标车道中心外→中心内、进入后稳定、进入前近似同向和 RECEIVER 前后身份连续；
- 另一个旧 TP 因没有独立 RECEIVER 的 pre identity support 被拒绝，不用旧 overall 标签覆盖新事件定义。

| 快捷做法 | 为什么无效 | 第四版合法替代 |
|---|---|---|
| 降低分支/多 incoming 门槛 | 仍把地图属性当车辆行为 | 原始 2 Hz center/box outside→inside |
| 复用 subject source-stream rear | 重演 K3 rear 污染 | 独立 direct incoming / target lane RECEIVER |
| 只要求进入后有 rear | 无法证明被切入车流在事件前已存在 | 同一 RECEIVER pre/post identity |
| 因 0.999 s 拒绝名义三帧 1 s | nuScenes 时间戳有毫秒抖动 | 冻结 20 ms timestamp tolerance，仍需 3 个 2 Hz 帧 |
| 在正式 train 结果上再调阈值/scene | evaluation 泄漏 | 阈值只由 49 条旧审标签冻结 |
| 缩短 30-frame negative 或允许 overlap | 改写 matched-control 问题 | physical event window + 0.5 s guard，control 仍 30 frames |
| 自动启动 N2 | 三次 reject 后边界更严格 | 第四次用户裁决 + 新授权前 `n2_authorized=false` |

## N1 kinematics-first 第三版（2026-07-25，已人工 REJECTED）

### N1-F09：候选真实性与 matched-control 支持是两个独立门槛

**观察**

- clean commit `aa162ef4dea808ad28ca7e56f1273f106e9c0e49` 上的 official train 694-scene
  formal run 完成 8,631 transitions → 1,879 topology-pass → 244 physical-motion-pass →
  12 interaction candidates；
- 12 candidates 覆盖 9 scenes，达到 candidate `≥12` 与 scene `≥6`；
- same-actor lane-keeping negative 只有 2，same-actor pair 只有 2，均低于冻结阈值 4；
- 因此 parent `machine_gate_passed=false`；parent 的唯一 terminal 保持
  `AWAITING_HUMAN_REVIEW`，后续独立 adjudication 已按 12/12 FP 写成 `REJECTED`；
- `AWAITING_HUMAN_REVIEW` 只是当时的审计就绪状态，从未表示 machine pass 或 N1/N2 授权。

**Pair 失败的冻结诊断**

对 12 个 positive actor 重放原 30-frame negative 搜索，不改 event pool：

| 主阻塞 | actor 数 | 观察 |
|---|---:|---|
| paired | 2 | 仅 `scene-0870` 两个 actor |
| 无 30-frame stable run | 1 | actor 轨迹支持太短 |
| 所有窗口与 positive overlap | 5 | 4–27 个候选窗口全部重叠 |
| non-overlap lane-keeping 存在，但 interaction 全失败 | 4 | 共 25 个 lane-keeping PASS windows，全部缺 center front/rear |

其中 6/10 未配对 actor 没有可用的非重叠长控制窗口；另外 4/10 没有等价的双侧 interaction control。
这不是把 gap 或速度阈值稍微放宽就能解决的问题。

**禁止快捷修补**

- 不把 30-frame 缩短到刚好得到 4 pairs；
- 不允许 negative 与 positive event overlap；
- 不用不同 actor 冒充 same-actor control，也不只挑有 pair 的两个 actor 报告；
- 不把普通 lane-keeping 但缺 front/rear 的窗口当成与正例等价的 interaction negative；
- 不因人工可能判真而把 `machine_research_support` 改成 true。

**可能突破**

第三次人工已表明 12/12 merge 候选均不真实，因此先修 subject/receiver 语义，再谈 control 扩展。
若第四版人审真实性通过但 same-actor control 仍不足，可新预注册“更长日志中的同 actor control”或
“matched-other-actor control”；后者必须显式匹配 scene、类别、速度、道路与交互密度。二者都不能回写
第三版 run。

### N1-F10：第三版最终候选只覆盖 converging-branch merge

**观察**

- 244 个 physical-motion-pass 包含 181 merge、63 parallel lane change；
- interaction 层有 215 个在中心关键帧缺 front/rear、17 个 temporal identity/bumper-gap 失败；
- 最终 12/12 candidates 全是 `converging_branch_merge`，parallel lane-change 为 0。

**能下的结论**

第三次人审只能估计这 12 个 converging-branch merge 的真实性。即使全部为真，也不能声称第三版已经覆盖
一般 lane change/cut-in；同时也不能断言 63 个 physical lane-change 都是假事件，因为它们是在更严格的
双侧 interaction 层失败。

**复开条件**

先把 `subject maneuver authenticity` 与 `front+rear gap-insertion interaction` 拆成两个预注册层。
可对 63 个 physical lane-change 建独立 diagnostic audit，但不得事后补进当前 12 条、降低当前 machine gate
或把 subject-only event 当 interaction positive。

