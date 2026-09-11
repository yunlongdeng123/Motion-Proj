# 历史原始记录 064

> 冻结历史，不是当前执行规则。当前规则见 [scaling law](../../../auto-research_scaling_law.md) 与 [AGENTS](../../../AGENTS.md)。
> 原文件：`docs/RESEARCH_FAILURES.md`；迁移前提交 `abbd431c`。原有相对链接按原目录解释，证据路径原样保留。
> 按索引读取相关小节；旧哈希、过度门控和资源指令均不重新生效。

### V3-F19：部署尺寸必须分为 sensor、model-native 与 evaluation 三层

A4-P0 v1 把 nuScenes sensor `1600×900` 写成 checkpoint 原生分辨率，但 source config 已固定
`downscale_when_loading=[2,2,2]`，真实模型加载和 render 均为 `800×450`。v1 保持 `blocked`，v2 只纠正输入
合同并在新 run 完整重跑；不能使用 v1 的性能数字关闭 P0。

后续所有 runtime、质量和资源协议必须显式记录 sensor resolution、source-config downscale、model-native
resolution 和最终 evaluation/output resolution。任何一层变化都是新的实验因子，不能用“native”一词静默折叠。

### V3-F20：序列化 schema 不能替代 live runtime API 审计

A4-P5 r1 已生成合法 registry，却把 checkpoint key `points_ids` 当作加载后的 `RigidNodes.points_ids`；实际
`load_state_dict` 会把它写入 `self.point_ids`。r1 因此保留 `blocked`，修复后的 r2 才通过 14/14 audits。

后续做 lazy loading、分块恢复、资产注册或 checkpoint 迁移时，必须分别验证 state key、load hook、live attribute
和调用方，并用真实 fresh-process reload 测试锁定。不能因为 checkpoint 中存在字段，就推断运行时对象暴露同名接口。

### V3-F21：预注册最小剪枝臂失败后不能事后缩小 fraction

A4-P1 的 b05/b10/b20 均通过结构、reload、count 和资源审计，但最小 b05 已使 global occupied PSNR、global PSNR
和 non-target PSNR 分别退化 `0.117684/0.110926/0.125462 dB`，超过冻结 `0.10 dB` 门；更大 fraction 失败更多
端点。局部 actor/boundary 指标保持或改善不能覆盖全局与非目标区失败。

后续不得在同一 ranking、视图和结果上新增 b01/b02、改变阈值或只报局部轴。若重新研究压缩，必须有不同的、
结果前可解释的结构假设与新预注册；单纯把 fraction 调小不是新的研究问题。

### V3-F22：FP16 存储压缩不等于端到端加速

A4-P2 把 Background/RigidNodes 的 10 个 scale/quat/feature/opacity tensors 转为 FP16，checkpoint 从
`578,819,674` 降到 `432,111,754` bytes，31/31 quality safeguards 通过。但 candidate 的 load、P50 和 FPS 没有
形成一致加速，renderer 输入仍显式转回 FP32；source audit 还表明 Background means 若做 FP16 roundtrip，最大
空间误差接近 `1 m`。

因此当前合法 claim 仅为 `mixed_precision_parameter_storage_fp32_render`。后续若研究低精度执行，必须单独冻结
renderer dtype、数值误差、质量、peak resident memory 和 latency 合同；不得从文件变小推断 Tensor Core、VRAM
或实时收益，也不得把 means、trajectory 或 provenance 一并降精度。

### V3-F23：exact chunk package 不等于 streaming/LOD 系统

A4-P3 的 133 static + 24 actor + skeleton + manifest 共 159 files 可 exact 重组，57 RGB SHA、31 endpoints、
85 tensor paths 和 source/registry immutability 全通过。但 package 比 source checkpoint 大 `2.792171%`，全量读取的
load/reassembly 与 render 均未加速，filesystem cache 也未控制。

后续只有在实现真实 demand loading、明确 working set/cache/eviction、记录首帧与稳态 latency、peak resident bytes、
I/O bytes 和 exact fallback 后，才可研究 streaming/LOD。继续把同一 159-file package 全量读入内存，只能叫资产
分离，不能叫部署加速。

### V3-F24：F0 前置失败不是前馈方法质量失败

Instant NuRec canonical audit 只通过 4/11 prerequisites；Python 3.11、uv、30 GiB VRAM、100 GB free disk、exact
weights、licensed NCore input 与 terms record 未同时满足，所以 `inference_command_constructed=false`。官方 standalone
CLI 的实际输出又只含 static PLY，不含 dynamic/sky/ISP/actor registry/trajectory/depth。

后续不得把“本机未运行”写成 upstream quality reject，也不得用 static PLY 与 StreetGS 完整 checkpoint 做假等价
比较。只有硬件、许可、数据和 converter 接口全部独立满足后，才能用新任务做窄范围前馈 pilot。

### V3-F25：证据链 exact 不等于研究主张自动成立

R0 canonical 的 63/63 inputs、23/23 decisions、12/12 deliverables、26/26 manifest files 与 P3 159-file package
全部 exact，证明 V3.1 可以从冻结事实恢复同一结论与生产链。它没有增加场景、seed、真值、闭环控制或新的方法臂。

后续研究必须从一个明确、可证伪的新问题出发，并说明新增证据解除哪一条失败约束；不能把 R0 的可复现性重新命名为
完整 world model、跨场景泛化、物理真实性或安全性。若主张涉及 A2/A3 方法，至少需要独立场景确认；若主张涉及
部署收益，必须直接测量对应 runtime/working-set 端点。

<a id="detail-v2"></a>

## V2 启动时必须先读的结论（2026-08-02）

- `PIVOT-F03`：AD-GS exact reproduction 已完成；V2 只读最终 checkpoint/render/metrics，不重复训练。
- `PIVOT-F04`：可见性建模不等于未观测背景真值；M5 必须保留 Tier A/B/C。
- `PIVOT-F05/F14`：资源、外部实例与方法失败分开；OOM/重启不能写成模型质量结论。
- `PIVOT-F06/F07/F08`：换机、非登录 shell 与浮动权重必须重新审计；镜像不能改变固定版本。
- `PIVOT-F10/F11/F12/F13`：PNG/JPEG、COLMAP 并发、cgroup 90% 和合法空占位均已有失败证据。
- `PIVOT-F14B`：V1 pointops2 的直接根因是 PEP 517 隔离构建缺少 torch；V2 先按 upstream
  `python setup.py install`，不重复原 `pip install .`。
- `PIVOT-F15`：AD-GS camera-local pseudo ID 与二值 `obj` 不能支持对象级编辑；V2 以 nuScenes
  `instance_token` 只做评测真值，不注入 AD-GS 训练。
- `PIVOT-F16`：持久身份、actor binding 与基础轨迹编辑本身已不新；V2 必须先产生跨三场景真实失败，
  再做新的 novelty gate。

存储清理只使历史环境和中间 checkpoint non-resident，不撤销上述失败，也不允许重新运行已关闭路线。

<a id="detail-legacy"></a>

## N1 kinematics-first 第三次 reject 与第四版约束（2026-07-25）

### N1-F12：地图分支收敛不是车辆横向机动

**观察**

- 第三次人审文件：
  `/root/autodl-tmp/runs/event_first/N1-EVENT-KINEMATIC-01/v71_n1-event-kinematic-01__kinematic-v1__s0__20260725T092427030639Z__8c2247b6/audit/review_working.jsonl`；
- review SHA256：
  `005cd74b874833808435fd2f47387d1d8e446cdea2d3a5cae6146e34bf331e96`；
- 12/12 已审，`TRUE_POSITIVE=0`、`FALSE_POSITIVE=12`、`UNCERTAIN=0`，precision=`0`；
- subject maneuver 为 `INVALID` 12/12；failure code 为
  `SUBJECT_NO_LATERAL_MANEUVER=12`、`ROUTE_CONTINUATION=11`、`NORMAL_TURN=1`、
  `MAP_MATCH_JITTER=1`；
- 第三版 12/12 机器候选都是 `converging_branch_merge`。规则只验证 source/target
  地图分支在几何上汇合，却没有验证车辆中心/车身相对接收车道发生 outside→inside 横移；
- target corridor 人审 12/12 为 `VALID` 并不能挽救 subject maneuver。地图画对了，不等于事件成立。

**根因**

第三版仍把“actor 沿一条会汇入 target 的道路行驶”当成“actor 主动切入 target 车流”。车辆可以保持正常
转向/道路中心线跟随，而道路本身向另一分支收敛；仅比较 source/target approach heading 或地图 token
变化仍会把路形变化误写成车辆运动学。

**防重复**

- 必须直接从原始 2 Hz annotation 计算 subject 相对接收 corridor 的连续横向状态；
- 至少观察目标车道中心外→中心内，并在进入后保持名义 1 s；10 Hz 插值不参与物理门；
- 进入前还必须与接收 corridor 近似同向，避免把大角度路口/主路续接的几何距离收敛当作 cut-in；
- 不再把 `merge` 地图类别、multiple incoming、route token change 或道路弯曲本身当正例。

### N1-F13：接收车必须来自独立目标车流，不能复用 subject 后车

**观察**

- 第三次 review 中 rear 为 `INVALID` 2/12、front 为 `INVALID` 1/12；
- 第三版 corridor 构造会贪心选择与 subject source 最顺的 incoming，再在该 corridor 上找 rear；
- 因而所谓 rear 往往就是 subject 原队列中的后车，而不是被切入目标车流的接收车；
- K3-004 选错 front branch；K3-007、K3-010 选错 rear branch。其余多项虽被人审写成 corridor
  `VALID`，也只说明地图链连续，不证明 receiver 角色语义成立。

**第四版硬约束**

1. parallel lane change 的 target chain 显式排除 source token；
2. merge 只枚举 `target` 的 direct incoming 中不同于 subject source 的分支；
3. RECEIVER 必须在进入前后保持同一 identity、同向、最近后车次序与 `[0.5,40] m` bumper gap；
4. subject/receiver 之间不得遗漏更近同 corridor 车辆；
5. negative control 也必须存在持续 receiver，不能用孤车普通直行冒充交互密度等价 control。

### N1-F14：第三次裁决的研究失败与工程失败必须分开

**研究裁决**

- clean adjudication commit：`1fbbbc1`；
- 成功 run：
  `/root/autodl-tmp/runs/event_first/N1-EVENT-KINEMATIC-AUDIT-01/v71_n1-event-kinematic-audit-01__human-audit-reject-v1__s0__20260725T155754010881Z__4c51f0d9`；
- 唯一终态 `REJECTED`，`n2_authorized=false`。

**保留的工程失败**

第一次 formal adjudication 使用了错误的 audit-manifest 指纹键，在写入研究产物前失败：
`.../v71_n1-event-kinematic-audit-01__human-audit-reject-v1__s0__20260725T155523736677Z__4c51f0d9/`。
该目录保留 `FAILED/failure.json`，原因是 `engineering_manifest_key_mismatch`。修复将
`artifact_set_sha256` 更正为实际 schema 的 `immutable_artifact_set_sha256`，并把所有输入校验提前到
run 目录创建之前。不得删除失败尝试或把它统计成 research reject。

### N1-F15：四图全常驻与重型 map API 会触发 2 GiB cgroup 峰值

**观察**

- 第四版首个 development smoke 在算法开始前以 `RC=137` 被杀；
- 容器 `memory.max=2147483648`，当时常驻服务已占约 `1.85 GiB`；
- 官方 `nuscenes.map_expansion.map_api` 的导入会连带 OpenCV、Matplotlib、Shapely 和渲染 API；
  单是 import probe RSS 就从约 `58 MiB` 增至约 `212 MiB`；
- 同时常驻四张 `NuScenesMap`、完整 sample/instance JSON 行和 128-scene dense batch 会进一步放大峰值。

**工程修复**

- 新增只读取 `lane`、`lane_connector`、`arcline_path_3`、`connectivity` 的轻量 map reader；
- arcline 离散化与官方 devkit reference 在单测中逐点一致；
- map index 改为一次只缓存一个 location，calibration/evaluation 按 location 排序；
- `sample.json`、`instance.json` 改为 ijson 流式最小字段投影，scene builder 复用同一 metadata source；
- scene batch 冻结为 32；不得通过杀死用户编辑器进程、修改容器上限或跳过地图证据来“解决”。

