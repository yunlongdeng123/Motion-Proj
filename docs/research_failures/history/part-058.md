# 历史原始记录 058

> 冻结历史，不是当前执行规则。当前规则见 [scaling law](../../../auto-research_scaling_law.md) 与 [AGENTS](../../../AGENTS.md)。
> 原文件：`docs/RESEARCH_FAILURES.md`；迁移前提交 `abbd431c`。原有相对链接按原目录解释，证据路径原样保留。
> 按索引读取相关小节；旧哈希、过度门控和资源指令均不重新生效。

## V6.2 CPSC 防重复结论（2026-08-24）

- `V62-F01`（`algorithm/evaluation`, `active`）：V6.1 oracle Occupancy 在 legacy28 得到 `10/28 ACCEPT` 且
  `0 false-safe`，说明物理状态补全存在真实上界；GaussianWorld 和 IR-WM 的 learned argmax Occupancy 都得到同一
  `10/28` 表面支持，但各自接受项全部为 `10/10 false-safe`。已确认的共同根因是 dense learned prior 能覆盖 proposal，
  却没有把真实 observed FREE 当作不可违反的前向约束；这推翻“感知型 argmax Occupancy 可直接成为 world compiler
  物理权威”，不推翻 learned prior 作为软信息源。防重复：不得以第三 backend、confidence threshold、entropy、grid、
  history window、checkpoint、verifier 放宽或 observed-FREE 事后全 veto 复开 V6.1。合法复开仅限 V6.2 CPSC：方法输入内
  hard FREE/OCC、contradiction→UNKNOWN、可推翻 prior、anti-trivial coverage 和独立 false-safe 评测；若 B1 hard clip
  已达 `>=5/28, 0 false-safe`，应诚实转为 projection-only compiler。证据=`V61-F11,V61-F13`、
  `docs/autoresearch/worldsim_v61/V61_MINIMUM_EXPERIMENT_CLOSEOUT.md`、
  `docs/WORLDSIM_V6_2_CONSTRAINT_AWARE_PHYSICAL_STATE_COMPLETION_PLAN.md`。
- `V62-F02`（`data/protocol`, `resolved`）：P2 r1 query probe 的 `method/target_state` 沿用 V6.1 evidence 编码
  `UNKNOWN/FREE/OCCUPIED=0/1/2`，而 P3 model distribution 固定 `FREE/OCCUPIED/UNKNOWN=0/1/2`；字段名未显式区分，
  若直接训练会静默互换 UNKNOWN/FREE/OCC 标签。r1 只做 CPU 资源/池探针，未训练、未产出科学结果或 formal dataset。
  r2 在任何 formal materialization 前把字段拆成 `*_evidence_state` 与 remapped `*_class_index`，并确认两者范围0..2；
  canonical probe=`20260824T082318Z__query-probe-s20260824-r2`。防重复：loader 只能把 `target_class_index` 送入
  three-state loss，把 `*_evidence_state` 作为 hard-evidence feature；禁止依赖裸整数碰巧相同或在 loader 中无名 remap。

- `V62-F03`（`data/algorithm`, `resolved`）：P2 formal r1 在 `scene-1012/f152` 因 instantaneous
  `actor_envelope` pool=`0` 停止；该帧并非没有 actor，而是4个当前 actor 全在冻结 ROI 外，其中一个 actor 在可见
  method sweep `f146` 仍穿过 ROI。只按 target-frame box 构造 actor pool 推翻了“每个冻结 target 当前 ROI 都含 actor”
  的隐含假设，也会诱使实现删除固定的15k actor query。参考 QueryOcc 的相邻时刻独立4D查询以及动态稀疏 query 的
  时序传播，恢复方案把 actor query support 固定为 `current target envelope ∪ visible method-sweep envelopes`；时序包络
  只定义 query support，不升级为 hard OCC evidence、不读取 dropout/target evidence，也不挪用 actor quota。定点复现
  `20260824T083403Z__actor-sweep-repro-s20260824-r5`：current=`0`、visible swept=`450` voxels、actor-type query=
  `15000/15000`、total=`100000`、exit=`0`。防重复：不得因某个 target 当前 ROI 无 actor 而删 unit、删 actor query、
  改 ROI 或把 target evidence 当 method input；若 visible method sweep 也无 actor support，必须登记新的 cohort-level
  事实并重新审视 actor-query定义，不能静默转采 EASY FREE。formal r1=
  `20260824T082601Z__query-dataset-s20260824-r1`，未完成 manifest、未用于训练或质量结论。恢复后的 formal r2=
  `20260824T083654Z__query-dataset-s20260824-r2` 已完成72/72 units、7.2M queries，combined actor pool 0空、
  source-role overlap=0；该成功不新增 failure ID。

- `V62-F04`（`engineering`, `resolved`）：P4 probe r1 在 official IR-WM plugin import 阶段、GPU forward 与任何
  sidecar 写入前失败；隔离 Python 可执行文件虽来自 `worldsim-v61-irwm`，controller 却继承外层 shell PATH，导致
  PyTorch `cpp_extension.load()` 的 `verify_ninja_availability()` 找不到 env 内已安装的 `bin/ninja`。这不是缺依赖、
  CUDA 不兼容、IR-WM 方法失败或数据失败。PyTorch 官方实现明确通过 PATH 调用 `ninja --version`，V6.1 已成功的
  IR-WM controller 也显式 prepend env `bin` 并冻结 `TORCH_CUDA_ARCH_LIST=8.6`。恢复仅复用同一环境合同：prepend
  env bin，设置 `PYTHONNOUSERSITE=1`、OMP/MKL threads、CUDA device与SM 8.6；不安装包、不改模型/输入/query或门槛。
  failed probe=`20260824T085711Z__prior-sidecar-probe-s1-r1`，无科学输出；防重复：用隔离 Python 启动 native/CUDA
  worker时不得假设其 bin 自动进入 PATH，也不得把 loader import failure记成方法 rejection。恢复后的同输入 r2=
  `20260824T085956Z__prior-sidecar-probe-s1-r2` 已产生100k query-aligned sidecar，peak=`4.05GiB`、target evidence
  read=`false`；因此 F04 保持 resolved，不新增 recovery。

- `V62-F05`（`data/protocol`, `resolved for artifact-bounded P6`）：P6 接口审计推翻“V6.1 已冻结P5可直接消费的
  prior sidecar”。canonical ME3R 的四个IR-WM输出仅有argmax `class_label[200,200,16]`、occupied mask和网格/pose，
  没有17 logits或256D BEV；禁止重跑IR-WM，故逐cell uncertainty/latent不可精确恢复。阶段表还有第二处冲突：B2需要
  Tier-C校准阈值、B4需要未训练的no-evidence-dropout checkpoint、full M0需要P8 grouped conformal，三者在P6均不存在。
  防重复：不得从硬label伪造逐cell置信度、用legacy O_eval拟合adapter/threshold、重跑backbone、补训多臂或把B5冒充
  conformal M0。参考CVPR 2022 ProtoSeg的非参数训练特征均值，恢复只用P5 train split按17 class求query-weighted
  logits/BEV prototype并查表，P5保持frozen；24-unit只读失真审计agreement=`0.896898`、bridge hidden-FREE=
  `0.399349`、safe-OCC=`0.872897`、hard violation=`0`。合法P6只比较B0/B1/B3/B5，明确B2/B4 unavailable、M0 defer
  P8；bridge claim始终是lossy artifact transfer，不是native sidecar parity。证据=`P6_LEGACY_INTERFACE.md`、
  `configs/worldsim_v62/p6_legacy28_v1.yaml`、`motion_proj/worldsim_v62/legacy_bridge.py`。

- `V62-F06`（`algorithm/evaluation`, `active; recovery exhausted`）：P6 canonical=`20260824T095529Z__legacy28-s0-r1` 在同一28-case上
  B5仅`4/28 ACCEPT`且`4/4 false-safe`，mask-area=`0.09402`、R10=`2/3`、Actor新增=`0`；source-valid UNKNOWN=
  `0.82735` 超过0.50。B3与B5 case decision完全相同；B5 hard projection仍是`0/939206`违规，oracle accepted surface
  safe-OCC retention=`1.0`。B1虽把accepted FREE conflict从B0 mean/worst=`0.26748/0.57057`降到
  `0.05058/0.11722`，但仍`10/10 false-safe`，所以简单hard clip既不安全也未触发Stop 1。根因边界：argmax-only
  prototype input造成严重missing-feature shift，evidential head高UNKNOWN仍保留4个hidden-unsafe route surface；局部
  projection只能保证观测cell，不能恢复丢失特征或证明隐藏表面。禁止用本次O_eval调threshold/prototype、改grid/window、
  重跑IR-WM、删case、放松UNKNOWN/FREE gate或改选第二backend。唯一合法复开=`P6R evidence dropout`：依据CVPR 2022
  Modality-Agnostic Learning，只用P2/P4 train模拟`p=0.5` prototype feature loss，并由frozen full-view P5 teacher做
  `0.25 KL`一致性；相同P6 gate一次性复评。若P6R失败，CPSC-Lite关闭，不再换projection/set-valued recovery。
  P6R formal r2=`20260824T101705Z__feature-dropout-train-s0-r2` 已按冻结复合目标选best epoch2；objective改善但
  prototype hidden-FREE false-OCC为`0.41441`，尚未解除本条。只有未改门槛的legacy28 recovery可以裁决F06，训练
  selection不能替代false-safe结果。
  唯一P6R legacy recovery=`20260824T102709Z__feature-dropout-legacy28-s0-r1` 仍为`4/28 ACCEPT,4/4 false-safe`，
  接受集合完全相同；UNKNOWN虽从`0.827351`降到`0.638518`，仍超过0.50，mask-area=`0.094024`、R10=`2/3`、Actor
  gain=`0`、worst FREE conflict=`0.087379`。因此missing-feature exposure缓解abstention但没有建立hidden-surface
  authority，本条从“允许唯一recovery”更新为“recovery exhausted / family closed”。后续不得选择projection architecture
  或set-valued head作为第二recovery，也不得绕行P7/P8 calibration。未来新版本复开至少要求native per-voxel logits/features、
  独立calibration cohort与直接hidden-surface false-safe risk supervision，并在任何legacy评分前重新scope-freeze。

- `V62-F07`（`engineering`, `resolved`）：P6R首次formal入口
  `20260824T101047Z__feature-dropout-train-s0-r1` 从source=`d8f69d0`创建run后，在pure-prototype baseline selection
  的首个`compute_cpsc_losses`调用触发`KeyError: prior_tristate`；optimizer steps=`0`、checkpoint=`0`、legacy O_eval
  read=`0`，因此不是训练不稳定或机制rejection。根因是recovery runner自定义mapping batch漏传原P5 prior-preserve
  loss需要的字段。PyTorch官方`torch.utils.data`说明mapping sample/batch由collation保留键，调用方必须完整提供约定字段；
  恢复前已核对loss的全部batch访问，确认没有第二个遗漏键。修复不能把full-view先验静态塞入所有路径：selection传
  `bridge_prior[:,18:21]`，训练传逐query混合后的`corrupt_prior[:,18:21]`，使loss与student实际证据视图一致。
  failed r1保持不可变；revision 2只改batch合同与run revision，不改模型、数据、p/KL、loss权重、seed、资源或legacy
  gate，不增probe/smoke/回归矩阵。证据：PyTorch DataLoader官方文档
  `https://docs.pytorch.org/docs/stable/data.html`、P6R terminal/config/runner。

<a id="detail-v61"></a>

## V6.1 Occupancy-verified world compiler 防重复结论（2026-08-22）

- `V61-F01`（`engineering/protocol`, `resolved`）：`WS-V61-H-P0-001` 首次正式启动在创建 run
  directory、读取 R9/R10/raw evidence、GPU、训练或生成器之前，对尚不存在的
  `/root/autodl-tmp/runs/worldsim_v61` 调用 `shutil.disk_usage`，触发 `FileNotFoundError`。没有 canonical run，
  也没有方法结果；不得把它记成 Occupancy/SceneIR-O rejection。`WS-V61-H-P0-002` 只在资源审计前以
  `mkdir(parents=True, exist_ok=True)` 创建精确 namespace，并增加缺失目录单测；R9/R10 hashes、28-case、scene mapping、
  truth tiers、threshold/stop rules、资源门与 confirmation lock 全部不变。H002 从干净提交 `6247fd8` 运行并使全部
  P0 gate PASS；canonical=`run://worldsim_v61/WS-V61-P0-SCOPE-FREEZE-01/20260822T100812Z__scope-freeze-s20260822-r1`，
  gate SHA=`fb2a416a...ae40`。仍然成立的边界：任何新路线 runner 都必须先创建自己的精确 namespace，禁止把父目录的
  可用空间当成子 namespace 已存在。

<a id="detail-v6"></a>

## V6 可验证世界编译器新增防重复结论（2026-08-21）

- `V6-F01`（`governance/research-direction`, `active`）：V5.2.1 的 badcase census、人工归因、Base Validity、
  immutable run、exact-once、UNKNOWN/abstention 与 failure-ledger 纪律继续有效，但它们没有建立 TrackBayes/M3 的
  causal bridge，也未解决偏离 logged trajectory 后的大场景扩展、生成内容可信固化和闭环复用。Stage H/BKI 从未执行，
  不得写成算法 reject；V5.2 M123 autoresearch 主线状态为 `superseded_by_v6_direction_reset`，M1/M2/M3 只作为
  SceneIR provenance、factorized validity 和 dynamics verifier 的子系统证据。V6 的合法复开边界是跨 frontend 的
  SceneIR/support/provenance/verify/bake/deterministic-runtime 问题，禁止把 V6 再退化为 StreetGS repair、TrackBayes-only、
  KNN/Graph/BKI 或 cut-in mining 主线。证据=`WS-V6-G0-REPO-CONVERGENCE-01`、
  `docs/WORLDSIM_V6_VERIFIABLE_WORLD_COMPILER_AUTORESEARCH_PLAN.md`、
  `docs/autoresearch/worldsim_v6/governance/REPO_PREFLIGHT.json`；方法质量 run=`0`。
- `V6-F02`（`engineering/protocol`, `resolved`）：G2 首次裸 `pytest -q` 在 collection 阶段出现 `12` 个
  `motion_proj/scripts` import error，显式 `PYTHONPATH=$PWD` 后才执行测试；随后 motionproj interpreter 下的 4 个
  V5.1 frozen-runtime tests 因预期 `/root/autodl-tmp/envs/drivestudio/bin/python / torch 2.1.2+cu118` 而失败，使用合同
  指定解释器后对应四文件 `15 passed`。根因是 integration runner 把 repo import root 与历史 runtime profile 当成单一
  环境默认值，不是 merge、算法或冻结结果漂移。以后全量 gate 必须显式注入 repo root，并按 config runtime 分组；不得
  删除 exact-runtime tests 或放宽版本字段。证据=`WS-V6-G2-BRANCH-CONVERGENCE-01`、最终 motionproj
  `1443 passed / 1 skipped` + DriveStudio `15 passed`。
- `V6-F03`（`engineering/asset-integrity`, `resolved with retained asset boundary`）：G2 frozen-asset regression 发现
  Instant NuRec official checkout、P2 selected checkpoint 和 P3 的 `158` 个 chunk payload 缺失，但 manifest、source
  checkpoint 与冻结 hash 仍在；这会造成 5 个 asset-dependent tests 失败，不能倒写历史方法 reject。Instant NuRec 按 exact
  public commit/tree 恢复；P2 从 immutable source 确定性重建并命中 `432,111,754 bytes / 7be87e8b...7448`；P3 重建
  payload 对旧 manifest `158/158` bytes/hash exact 后只补缺失文件，未覆盖旧 manifest。P2 recovery r1 因传相对 protocol
  path 在 snapshot 前 blocked，r2 改为绝对路径后成功；旧 r1 保留。以后清理 canonical selected asset 必须同步保留可执行
  materializer 与 exact source，恢复只能新 run→逐 hash 比较→补缺失字节，禁止生成近似资产或改旧 manifest。证据=
  `20260821T073335Z__g2-p2-asset-recovery-s0-r1`、`20260821T073353Z__g2-p2-asset-recovery-s0-r2`、
  `20260821T073459Z__g2-p3-asset-recovery-s0-r1`、R0/P3 `23 passed`。
- `V6-F04`（`protocol/governance`, `resolved`）：V5.1 `_validate_normative_plan_binding` 只允许 P0 base hash 与
  `b359541` Stage-B append hash，但当前 canonical 文档已在 `a9dede0` 冻结 terminal closeout，SHA-256=
  `a0e764f3...fe1d`；因此 4 个 protocol tests 在到达各自语义断言前统一被旧 allowlist 拦截。修复不改历史 config、plan
  或 gate，只把该 exact terminal commit/hash 加入 fail-closed allowlist，并把回归期望指向 terminal hash；任何未知第四种
  状态仍拒绝。禁止用“任意后继 commit”或跳过 hash 来救测试。证据=`WS-V6-G2-BRANCH-CONVERGENCE-01`、
  `tests/test_worldsim_v51_protocol.py=9 passed`。
- `V6-F05`（`engineering/provenance`, `resolved with noncanonical run retained`）：R1 首个实例
  `20260821T081500Z__r1-capability-s0-r1` 在 capability runner/config/tests 仍未提交时执行，summary 如实记录
  `source_dirty=true`；手工指定的目录时间标签还晚于真实完成时间。其能力事实虽通过，但不能作为 canonical closeout，旧目录
  保留且不覆盖。修复是在 runner 创建 run 前读取 `git status --porcelain`，dirty 即 fail-closed；先提交
  `d981df7fdde5458eb3878193c4a76f6dcf926ad4`，再由 runner 自动生成真实 UTC 标签的新实例
  `20260821T080610Z__r1-capability-s0-r1`，其 `source_dirty=false`、gate PASS。以后不得用“内容看起来正确”绕过
  source cleanliness 或手工修正旧 terminal；工程实例与 canonical evidence 必须分开登记。

<a id="detail-v52"></a>

## V5.2 人工归因与 M123 causal bridge 防重复结论（2026-08-20）

- `V52-F01`（`evaluation/attribution`, `active`）：V5.2.1 的 `GLOBAL_RGB / ACTOR_RGB / BOUNDARY` failure label 是合法
  census 结果，但不能自动解释为 M1/M2/M3 的模块失败。用户指定评审者对代表性 18-case package 完成 `18/18` 逐图复核后，
  冻结 `9 BASE_FAILURE + 8 M123_ELIGIBLE + 1 ATTRIBUTION_UNRESOLVED`：AD-GS 的多条 actor/boundary case 实际由白屏、
  单色、全局 smear 主导；即使 ownership 完美也无法恢复这些画面。不得用 BASE_FAILURE case 评价 TrackBayes、M3 delta 或
  M2 router，也不得删除这些 case 来改善基座 aggregate。所有后续 M123 run 必须先执行 P0 Base Validity Gate，并在完整报告中
  单独保留 base sentinels。证据=`WS-V521-P11-HUMAN-ATTRIBUTION-01`、canonical run=
  `/root/autodl-tmp/runs/worldsim_v521/20260820T130000Z__p11-human-review-attribution-s0-r001`、cases SHA=
  `d89f4a4b...381f`。
- `V52-F02`（`evaluation/causal-identification`, `active`）：8 个 StreetGS eligible case 的视觉症状与 M1 observation scarcity、
  M3 actor trajectory/visibility 高度相容，但 panel 不能证明失败 pixel 恰好来自 low-observation/uncertain Gaussian，也不能证明
  ghost 会被 actor-pose warp 解释。因此状态只能是 `DIRECTION_SUPPORTED_CAUSAL_BRIDGE_PENDING` /
  `SYMPTOM_OVERLAP_STRONG_EXACT_TEMPORAL_BRIDGE_PENDING`，不得从人工诊断直接晋级 TrackBayes 或修改 M3。合法复开必须保持
  Discovery design `5`（#05/#10/#11/#16/#17）与 one-shot Confirmation `3`（#06/#12/#18）分离，先冻结并执行 exact
  pixel→Gaussian/U2-B3 observability bridge 与 `unwarped/flow-warped/pose-warped` temporal bridge；Confirmation 不得选 arm、
  threshold 或 metric。M2 只消费已通过 candidate 的 uncertainty/validity 并决定 execute/abstain；geometry undefined 时不得
  写 geometry-safe。证据=`docs/run_manifests/worldsim-v5.2.1-human-review-attribution-v1/` 与
  `configs/worldsim_v52/m123_autoresearch_v1.yaml`。

<a id="detail-v51"></a>

