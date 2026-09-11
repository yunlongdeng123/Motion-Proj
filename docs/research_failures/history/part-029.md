# 历史原始记录 029

> 冻结历史，不是当前执行规则。当前规则见 [scaling law](../../../auto-research_scaling_law.md) 与 [AGENTS](../../../AGENTS.md)。
> 原文件：`docs/RESEARCH_FAILURES.md`；迁移前提交 `abbd431c`。原有相对链接按原目录解释，证据路径原样保留。
> 按索引读取相关小节；旧哈希、过度门控和资源指令均不重新生效。

### V7-F10 — P4 adapter 误把 devkit reverse-index shortcut 当作 raw sample 字段

- run：`run://worldsim_v7/WS-V7-P4-NUSCENES-SELECTIVE-FACTORIZE-01/20260902T160000Z__selective-factor-s70401-r1`；
  `nuScenes_index` 刚开始即因 `sample.json["anns"]` 抛 `KeyError`。
- exposure：run 只读取 scene/sample metadata；未读取 sample_data/ego_pose/sample_annotation 全表、LIDAR、Actor geometry、
  repair label、model score 或 AV2；training=false，formal method-quality read=false，r1=`no_verdict`。
- root cause：nuScenes raw schema 的 sample 仅有 token/timestamp/scene/prev/next；`anns`/`data` 是官方 devkit
  `NuScenes.__make_reverse_index__()` 初始化时添加的 shortcut，不存在于原始 JSON。
- literature/open-source response：nuScenes 官方 schema 规定 annotation 以 `sample_token` 外键指向 sample；官方
  `nuscenes.py` 也显示 reverse-index 才执行 `sample_record['anns'].append(...)`。
- resolution：删除完全未使用的 raw `anns` 读取与排序，保留既有 `sample_annotation.sample_token` streaming join；
  r2 不改冻结 cohort、label、feature、model、seed、threshold、gate 或 claim。状态=`resolved_before_quality_read`。

下一可用编号：`V7-F11`。

### P4 freeze prevention note — clean-query fallback 与外域风险边界

- P3 的 `before` 包含 synthetic ghost/duplicate/flicker，不能用于证明逐 Actor 修复比原始观测更安全；P4 单独记录
  `query_only`，label 和 selective fallback 都以 untouched clean query 为基线，target 仅在标签/评价阶段读取。
- SelectiveNet/Conformal Risk Control 的迁移限于 reject/risk--coverage 结构；standardizer/model/threshold 只允许读取
  nuScenes train/calibration。AV2 未知 dataset shift 不满足已知 exchangeability 前提，只作冻结 zero-shot 描述。
- hazard 分支不接收任何 validity feature，validity 分支不接收 TTC/clearance 等 hazard feature；paired swap 直接报告
  cross-input score shift，防止“高风险 Actor 被自动拒绝”成为隐蔽策略。
- 本项不是新 failure；后续已使用 `V7-F10`，下一可用编号为 `V7-F11`。

### V7-F09 — aggregate hard-geometry gates 掩盖 per-Actor repair degradation

- exposure：P3-B 不挑图的第 8 main case 显示 Chamfer `.527→.967m`。对 P3-A 全 634 Actors 作一次全量 tail read，
  `109/634=17.19%` individual Chamfer worsened；depth error 与 ray termination 分别有 `271/253` Actors 方向变差。
  P3-A role mean 与预冻结 gates 仍有效，但不能解释为每个 Actor 都被安全修复。
- root cause：aggregate mean gate 允许收益大的 Actor 抵消长尾退化；ray-certified PROJECT 只证明 paired matched-ray
  free interval，不为 KEEP/COMPLETE/UNKNOWN 组合后的 target geometry 提供 per-Actor non-worsening certificate。
- literature/open-source response：ICML 2019 SelectiveNet 把 reject option 明确为 risk--coverage 问题；ICLR 2024
  Conformal Risk Control 对 monotone loss 提供校准风险控制。V7 迁移“repair or abstain”接口，但 calibration/train 只允许
  nuScenes；nuScenes→AV2 未知 shift 不满足 exchangeability，因此 AV2 只报告 zero-shot risk--coverage，不包装成 formal guarantee。
- resolution plan：下一 candidate 用 method-visible geometry/provenance 特征学习低容量 repairability score，并与 hazard
  encoder 输入隔离；label/threshold 仅来自 nuScenes build/query/target split。AV2 全 30 logs exact frozen evaluation，
  失败 Actor 不删除。status=`open_p4_selective_certificate_required`。

下一可用编号：`V7-F10`。

### P3-B freeze prevention note — camera/crop 不读取 RGB 外观或结果质量

- P3-B 逐项复用 P3-A 冻结 `VISUAL_CASES.jsonl`，没有新的案例 admission。相机只按 official AV2 calibration 下
  query LiDAR points 的 in-frame count 选取，平局按预先写入 config 的 ring-camera 顺序；RGB 在选择完成后才 decode。
- crop 仅由同一 query projection 的包围盒和固定 padding/minimum size 决定；不按画面美观、遮挡、hazard、Chamfer、
  free-space 或 action 成功率调整。全部 30 cases 都写 panel/video，不删除难例。
- paired ghost/duplicate/flicker 仍是 synthetic contract overlay；真实主张来自 P3-A target-only LiDAR depth/ray/surface。
  P3-B 明示不是 photorealistic reconstruction。本项本身不是新 failure；formal tail read 随后暴露 `V7-F09`，
  下一可用编号为 `V7-F10`。

### V7-F08 — ray-correct 评价确认 nearest-canonical PROJECT 违反 observed-free interval

- run：`run://worldsim_v7/WS-V7-P3-AV2-HARD-EVIDENCE-01/20260902T140000Z__hard-evidence-s0-r2`；
  r2 修复 V7-F07 后仍为 quantitative/qualitative `6/8`。
- scientific result：逐 primitive ray 评价得到 free-space after=`.849111/.828770`、ghost-component ratio=
  `.916598/.900591`；PROJECT 输出平均仍在 observed hit 前 `.09178/.08881m`。因此 P2 的 Euclidean
  nearest-canonical PROJECT 在 zero-level/Chamfer 上改善，却不满足同 ray line-of-sight 物理约束。
- root cause：三维最近点不保序；canonical surfel 可在欧氏距离很近、但沿对应 beam 位于真实 termination 之前。
  这不是调 metric/gate 能解决的评测问题，而是 projection operator 的真实表示边界。
- literature/open-source response：NeuRAD/SplatAD 将 LiDAR depth/line-of-sight/ray-drop 与显式 ray 绑定，LiDAR-RT
  通过 range rays 渲染 depth/mask。V7 新候选据此只允许有 direct observed-hit provenance 的 ghost 投回同 ray
  实测 termination；没有 matched hit 的 primitive 必须 `UNKNOWN`，禁止投向附近 surface。
- resolution boundary：r2 verdict 永久保持 rejected；新 hypothesis `WS-V7-H-P3-002` 的 r3=
  `20260902T143000Z__ray-certified-s0-r3` 保留 30 logs、8 gates、视觉案例、target isolation 和除 PROJECT
  output 外的全部实现，最终 quantitative/qualitative=`8/8`、`8/8`。状态=`resolved_by_ray_certified_projection`。

下一可用编号：`V7-F09`。

### V7-F07 — P3 ghost residual 评价丢失逐 primitive 射线 provenance

- run：`run://worldsim_v7/WS-V7-P3-AV2-HARD-EVIDENCE-01/20260902T133000Z__hard-evidence-s0-r1`；30/30 logs、
  30 frozen panels 正常完成，但两个 role 均只有 6/8 gates。
- symptom：quantitative/qualitative 的 free-space after=`.6777/.6493`、ghost-component ratio=`.4542/.4565`，与
  ghost PROJECT=`.9934/.9911` 及 SDF ratio=`.3034/.2999` 同时出现。
- root cause：r1 用 ghost 到“任意 compiled point”的 Euclidean 最近邻判断残留；Actor 曲面上邻近但不属于该 ghost
  primitive 的合法 surface 会被错误计作同一 LiDAR ray 上的提前 termination，破坏了 P3 要证明的 provenance 边界。
- literature/open-source response：CVPR 2024 NeuRAD 显式建模 LiDAR rays/beam/ray drop；CVPR 2025 SplatAD 同时使用
  depth、line-of-sight 与 ray-drop 对象；CVPR 2025 LiDAR-RT 官方实现从 `ray_o, ray_d` 产生 range-ray depth/mask。
  V7 迁移其最小共同边界：保存每个 ghost 的 `ray_o/ray_d`、真实 hit 与对齐 PROJECT output，只在同一 beam tube
  内统计提前 termination；UNKNOWN 无输出，不允许用附近无关 surface 代替该 primitive。
- resolution/impact：r1=`implementation_rejected/no_valid_free_space_or_ghost_verdict`；depth/ray/SDF/jitter/Chamfer/
  Actor-state 六门和冻结 panels 可保留为 descriptive evidence，但 P3 总 verdict 维持 rejected。r2 不改 cohort、案例、
  compiler action、阈值或 gates，只修 metric provenance；r2 有效暴露 V7-F08，状态=`resolved_by_ray_provenance_metric`。

后续已使用 `V7-F08`；下一可用编号：`V7-F09`。

### P3 freeze prevention note — visual selection 不读取质量排序

- P3 继续使用冻结 30 logs；每个 qualitative log 只取按 UUID 词法序排列的前三个 eligible Actor，不按 Chamfer、
  hazard、action 成功率或视觉观感排名。主文固定为前 8 个 qualitative logs 的首 Actor；supplement 固定保留
  10 logs 的全部前三 Actor，预计 30 cases。
- 新硬证据只复用 P2 已冻结配置；P3 overlay 通过 `p2_config` 引用，避免再次复制 Actor/hazard 参数。target ray
  仅做评价，仍不参与 action。8 个 role gates 是非退化/物理边界，不扩成 smoke/regression 矩阵。
- paired ghost component/free-space 仍是合成合同；Actor trajectory/speed/acceleration/TTC 零 shift 来自 compiler-external
  state immutable，不冒充感知或预测准确率。本项不是新 failure；后续已使用 `V7-F07`、`V7-F08`，下一可用编号为 `V7-F09`。

### V7-F06 — P2 配置遗漏复用模块所需的冻结 hazard 段

- run：`run://worldsim_v7/WS-V7-P2-AV2-FOUR-ACTION-COMPILE-01/20260902T120000Z__four-action-s0-r1`。
- symptom/exposure：首个 log 的 annotation/pose metadata 加载后，`_track_geometries` 访问 `config["hazard"]`
  报 `KeyError`；点云关联、canonical surface、action decision、target metric 与任何 gate 均未发生，r1=`implementation_failed_before_quality_read/no_verdict`。
- root cause：P2 复用 P1 Actor/hazard extraction，却在独立 YAML 中漏复制冻结的 hazard 参数组；入口只做了解析/import，未执行重复 smoke/regression。
- literature/open-source response：Hydra 官方 Structured Config schema/Defaults List 建议以共享 schema 组合必需字段并用 mandatory
  value 阻止缺字段运行；本阶段不扩框架，只恢复与 P1 完全相同的已冻结 10 项 hazard 配置，避免改变 P2 动作阈值、
  cohort、seed、gates 或 claim boundary。
- resolution：r1 状态明确写为 failed；同一合同 r2=`20260902T125000Z__four-action-s0-r2` 在 30/30 logs 上完成，
  quantitative/qualitative 各 13/13 gates 通过。后续配置族应复用公共 Actor/hazard defaults，但不为本次恢复增加
  校验矩阵。status=`resolved_before_scientific_trial`。

### V7-F05 — GitHub 直连 git operations 挂起

- 2026-09-07 复发与恢复：EAS-VGGT 文档提交 `7538f38f` 的 direct push 以 `GnuTLS recv error (-110)` 退出。核对 [Git 官方代理/HTTP 配置](https://git-scm.com/docs/git-config) 后，读取当前 LocalTUN session，并仅为该命令设置代理和 HTTP/1.1；同一提交普通 push 成功，状态=`resolved`。这是既有出口传输问题的恢复，未改写历史、关闭 TLS 校验、修改全局 Git 配置或读取科学数据；task=`WS-V72-E0-EAS-VGGT-REPLAN-01`，不新增 failure ID。

- symptom：提交 `e985e59` 后，远端 `git push` 与 `git ls-remote` 均长时间无返回；没有改写 commit、branch 或工作树。
- diagnosis：GitHub 官方状态页显示 Git Operations operational；故障限于当前 AutoDL 出口路由，而非仓库或 GitHub 服务端事故。
- resolution：终止精确识别的悬挂 git 进程，读取当前 LocalTUN session 的 remote proxy 后仅为该次 git command 设置
  `HTTP(S)_PROXY`，push 成功；没有启动第二个研究 run，0 scientific impact。LocalTUN 端口是 session-specific，后续不硬编码复用。

后续已使用 `V7-F07`、`V7-F08`；下一可用编号：`V7-F09`。

### P2 freeze prevention note — completion 不读取 held-out target

- P1 已证明 build-side canonical surfel support，但 paired hole 的满分不能直接当真实 completion 质量；P2 因此把
  第一个 held-out frame 固定为 query、其余 held-out frames 固定为 target-only。
- `COMPLETE` 候选只由 build surface 的 temporal/view support 与 query-space hole 产生；target 只在全部坐标编译完成后
  计算 support/recall/precision/Chamfer。禁止用 target 删除 completion、调 hole radius 或换 Actor。
- quantitative 20 logs 与 qualitative 10 logs 在同一冻结实现中一次执行，避免 development 结果后再修改 confirmation。
- 本项不是新 failure；后续已使用 `V7-F05`--`V7-F08`，下一可用编号为 `V7-F09`。

### V7-F04 — background launcher 将准备链与训练命令一起放入 subshell，PID sidecar 未写出

- symptom：formal P1 进程已正常启动，但父 shell 在 background subshell 创建 `launch_logs/` 前写 PID 文件，报告
  `No such file or directory`；实际 Python PID=`3035`，log 与 canonical run 均持续正常写入。
- root cause：Bash `&` 与 `;` 的 list 结合使前置 `cd && mkdir && nohup ...` 作为异步 list 在 subshell 执行；父 shell
  的 sidecar write 与目录创建发生竞争，同时 PowerShell/SSH 转义让显示 PID 不可信。
- literature/open-source response：GNU Bash Reference Manual 规定以 `&` 终止的 list 在 subshell 异步执行，且 `;`/`&`
  具有相同的低优先级；后续 launcher 必须先同步创建目录，再只对圆括号分组后的 Python command 后台化。
- resolution/impact：没有重启或并发启动第二个 run；用 `pgrep` 与 canonical `status/summary` 只读监控原进程。
  20/20 logs、summary 与 10/10 gates 正常完成；仅 monitoring sidecar 缺失，0 scientific impact。

后续已使用 `V7-F05`--`V7-F08`；下一可用编号：`V7-F09`。

### P1 freeze prevention note — 不把外观补点当物理证据

- NeuRAD/SplatAD 官方实现会在 Actor-local seed 初始化中镜像一份点以帮助外观重建；该策略适合可渲染先验，
  但镜像点没有独立射线或多帧 provenance，不能进入 V7 collision surface。
- V7 只迁移其 Actor cuboid 入盒和 `ego/world→box` canonical transform；不镜像、不神经补全、不把 box shell
  当占据真值。远侧缺证据仍为 `UNKNOWN`，只有真实多视角支持的预注册 hole probe 才允许 `COMPLETE`。
- 本项为 formal run 前的方法边界，不是新 failure；后续已使用 `V7-F04`--`V7-F08`，当前下一可用编号为 `V7-F09`。

### V7-F03 — 非登录 shell 的定向 pytest 未包含仓库 import path

- symptom：首次运行单个坐标合同测试时，collection 阶段报 `ModuleNotFoundError: motion_proj`。
- root cause：远端非登录 shell 激活环境后未把仓库根加入 Python module search path；测试与任何数据读取均未开始。
- response：不改代码和测试合同，仅按项目既有入口约定设置 `PYTHONPATH=.` 原样重跑，结果 `1 passed`；实际首 log
  坐标读取得到合理 ego range。
- claim impact：纯入口环境失败，0 scientific quality read，不增加 smoke/regression 矩阵。

后续已使用 `V7-F04`--`V7-F08`；当前下一可用编号：`V7-F09`。

### V7-F02 — AV2 annotation frame 被误标为 city frame

- symptom：P0 adapter 将 `annotations.feather` 的 cuboid pose 直接命名为 `city_se3_actor`，再用
  `ego_se3_city` 计算 `center_ego_m`；首个冻结 log 会产生公里级 Actor center 错位。
- root cause：混淆 AV2 的 `egovehicle_SE3_actor` annotation 与独立的 `city_SE3_egovehicle` pose；P0 metadata
  smoke 只统计 Actor/state 数量，没有检查物理位置。
- literature/open-source response：AV2 官方 Sensor Dataset 文档与官方 API 均规定 cuboid pose 位于 ego frame，
  LiDAR return 也已 egomotion-compensated 到 ego reference timestamp。
- resolution：显式存储 ego-frame Actor pose，以 city/ego pose composition 派生 city-frame pose，并把
  `center_ego_m` 固定为 cuboid ego translation；冻结 cohort 与 scientific protocol 不变。
- exposure/claim impact：在任何 V7 method-quality、artifact/hazard 或 surface metric read 前发现；不产生科学负结果。

后续已使用 `V7-F03`--`V7-F08`；当前下一可用编号：`V7-F09`。

## WorldSim V7 P0 failures（2026-09-01）

