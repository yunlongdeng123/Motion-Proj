# 历史原始记录 014

> 冻结历史，不是当前执行规则。当前规则见 [scaling law](../../../auto-research_scaling_law.md) 与 [AGENTS](../../../AGENTS.md)。
> 原文件：`docs/RESEARCH_FAILURES.md`；迁移前提交 `abbd431c`。原有相对链接按原目录解释，证据路径原样保留。
> 按索引读取相关小节；旧哈希、过度门控和资源指令均不重新生效。

## V71-F60 — 单个截断 canonical 数组保留了截断前 sidecar 索引（2026-09-07）

- category=`legacy_sidecar_index_provenance`；status=`resolved_same_protocol_r3`；task=`WS-V72-D0-W0-W4-G0-01`。
- failed runs=`20260906T185100Z__w0-w4-g0-s7205-r1`,`20260906T185400Z__w0-w4-g0-s7205-r2`；1/659 个 eligible Actor 的 `canonical` 已截断到 4096 点，但 sidecar 仍含最大值 4960 的截断前 `input_canonical_surface_indices`。r1 在 G0 build-only 证据 `index_add_` 越界；r2 已修 build 权重映射，但辅助 F/O/U 标签仍误用旧索引。
- exposure：r1 未挂载任何 target；r2 完成 659 个 target-free geometry 后开始首个 legacy train Actor，读取训练 target 并在任何优化步骤前失败。两者均无 holdout read、无 quality metric、无模型训练；source/external final read=false。
- resolution：有效索引继续使用显式 provenance；仅对越界 Actor 将 sidecar anchors 映射到当前 canonical 的最近点，同一个已校验映射同时用于 build 权重与训练辅助标签，并在 manifest/summary 记录 fallback Actor 数和最大映射距离。全量静态审计确认只有该 1 个 Actor 越界；sidecar anchors 与 cache anchors 最大坐标误差为 `1.09e-6m`。
- prevention：任何由 capped/sampled 数组消费旧 sidecar 索引的 runner，必须在 GPU scatter 前检查索引范围；fallback 必须可计数，不得静默删 Actor 或 clip 索引。

下一可用统一失败编号：`V71-F61`。

## V71-F59 — TorchVersion 子类不能直接写入 resolved YAML（2026-09-07）

- category=`run_metadata_serialization`；status=`resolved_same_protocol_r3`；task=`WS-V72-D0-G3-ADAPOINTR-OFFICIAL-ZS-01`。
- failed run=`20260906T164500Z__g3-adapointr-official-zs-r2`；模型 335/335 tensors strict load 后，PyYAML 拒绝序列化
  `torch.__version__` 的 `TorchVersion` 子类。
- exposure：载入官方模型与 66 个 target-free partial；发生在 forward 和 target metric 前；无 quality read，
  source/external final read=false。
- resolution：只将 Torch/CUDA 版本元数据显式转换为普通字符串；模型、权重、partial、seed、batch、density 和
  evaluator 不变，r3 继续同一协议。
- prevention：第三方库返回的字符串子类在写 resolved manifest 前统一转为 JSON/YAML 原生 scalar。

## V71-F58 — G3 runner 未在 PoinTr 根目录解析相对 base config（2026-09-07）

- category=`external_config_working_directory`；status=`resolved_same_protocol_r2`；task=`WS-V72-D0-G3-ADAPOINTR-OFFICIAL-ZS-01`。
- failed run=`20260906T164200Z__g3-adapointr-official-zs-r1`；官方 config 的 `_base_` 使用相对路径，runner 从
  Motion-Proj 根目录解析时找不到 `cfgs/dataset_configs/PCN.yaml`。
- exposure：只读 adapter/cohort metadata；发生在模型构造、点数组加载、forward 和 target metric 之前；
  source/external final read=false。
- resolution：只在官方 config 解析期间切换到固定 PoinTr build root，并在 `finally` 恢复工作目录；模型、checkpoint、
  data、seed、batch、density 和 evaluator 均不变，r2 继续同一协议。
- prevention：带相对 `_base_` 的外部 config 必须以其仓库根目录为解析边界。

## V71-F57 — PoinTr 顶层导入要求未使用架构的 CUDA 扩展（2026-09-07）

- category=`external_baseline_import_scope`；status=`resolved_capability`；task=`WS-V72-D0-G3-ADAPOINTR-*`。
- symptom：AdaPoinTr capability 在模型构造前因 `models/__init__.py` 同时导入 GRNet，继而要求未编译的
  `gridding` 扩展；没有模型 forward、target metric 或 source/external final read。
- root cause：官方包入口注册所有架构，而本任务只使用 AdaPoinTr；该依赖与 G3 graph 无关。
- resolution：保持官方 `4603257` snapshot 不改，在独立 build copy 中只注册 AdaPoinTr；模型源码、checkpoint、
  forward 和 loss 不变。真实 512→4096 forward/backward 随后通过。
- prevention：外部基线只编译实际执行图依赖；build delta 与 upstream commit 同时记录，不能冒充 clean upstream wheel。

## V71-F56 — PointNet++ 上游固定旧架构且最小 CUDA 环境缺开发头文件（2026-09-07）

- category=`cuda_extension_build_contract`；status=`resolved_sm86_build`；task=`WS-V72-D0-G3-ADAPOINTR-*`。
- symptom：第一次 PointNet++ wheel 构建同时包含上游硬编码 `sm_37...sm_75`，且编译器找不到 `cusparse.h`；
  发生在任何 G3 模型/数据读取前。
- resolution：安装官方 NVIDIA CUDA 12.1 development libraries，在独立 build copy 将架构收窄为 RTX 3090
  的 `sm_86`；PointNet++ 3.0.0 与 Chamfer CUDA 均完成 forward/backward smoke。
- prevention：扩展构建必须记录 toolkit、GPU capability、upstream commit、build delta 和 wheel hash。

## V71-F55 — NVIDIA label 被全局 conda mirror 重写为不存在路径（2026-09-07）

- category=`environment_channel_resolution`；status=`resolved_explicit_official_channel`；task=`WS-V72-D0-G3-ADAPOINTR-*`。
- symptom：`nvidia/label/cuda-12.1.1` 被映射到 TUNA mirror 后返回 HTTP 404；无模型、数据或 target read。
- resolution：保持全局 conda 配置不变，仅对隔离环境使用完整官方 channel URL 与 `--override-channels`，成功安装
  CUDA 12.1.105 compiler/runtime/development libraries。
- prevention：带 label 的 vendor channel 使用 manifest 中的完整 URL，不依赖全局 channel alias。

## V71-F54 — G1 r1 假定 legacy cohort 全部保留 raw LiDAR（2026-09-06）

- category=`data_provenance_dispatch`；status=`resolved_same_protocol_r3`；task=`WS-V72-D0-G1-ACTOR-TSDF-01`。
- failed run=`20260906T155422Z__g1-actor-tsdf-cpu-r1`；在第一个 processed-recovery scene 读取已缺失的 raw
  `.pcd.bin` 时失败，发生在任何 Actor metric/summary 之前。
- root cause：V7.1 的 66-Actor M8 holdout 混合两个合法来源：44 Actors 来自 raw nuScenes，22 Actors 来自
  DriveStudio processed recovery；r1 只按 scene metadata 建 raw index，忽略了 cache 的恢复来源。
- resolution：读取冻结 `ADDED_ACTOR_INDEX.json`，按 identity 将 43 raw scenes 与 20 processed scenes 分派到
  原编译路径；拒绝同 scene 混合来源、缺 root 或缺 Actor，不做 fallback/删样本。r2 完整重建 66/66 Actors，
  但 manifest 的 `failure_ledger_delta` 误写为 `none`，故原样保留为非 canonical。r3 只修正运行元数据并按相同
  协议重跑；r2/r3 的 `ACTORS.jsonl`、`LOGS.jsonl` SHA-256 和全部 metrics 均完全相同。G0/G1 identity set 相同。
- exposure：r1 target metric read=0、source/external final read=false；r2/r3 只读既有 `legacy_diagnostic`。
- evidence：source-dispatch 修复 commit=`691619c5`；r1/r2/r3 `status.json`，r3 manifest/summary；
  prevention：所有 legacy cache 重建必须携带 per-Actor source provenance，禁止从目录存在推断 raw 可恢复。

下一可用统一失败编号（该段记录时）：`V71-F60`；当前编号以文首为准。

## V7.2 D0 G0 outcome note — density is a required matched factor（2026-09-06）

G0 在同一 66-Actor legacy cohort 上把平均点数从 `60.9` 增至 `453.1` 时，CD-L1 改善 `53.49mm`，但
early 恶化 `25.67pp`，hit 在 128--256 点达到平台后下降。该结果不是新 failure，也不选择路线；它将“点数/密度
解释”从可选消融升级为 G0--G3 主表的强制匹配因素。禁止用 native-density CD 改善直接宣称回波改善，禁止在
最终曲线上事后挑点。canonical=`20260906T153519Z__g0-raw-fusion-cpu-r1`；failure_ledger_delta=none；
下一可用统一失败编号仍为 `V71-F55`。

## V71-F53 — 文档首页保留了过期的当前路线与授权（2026-09-06）

- category=`documentation_navigation_stale_authorization`；status=`resolved_navigation`；base commit=`ec9c2e08`。
- observation：根 README 仍以 V6 为当前路线，docs 索引仍将 V5.2 的旧执行许可写作当前授权，与 V7.1 状态不一致。
- cause：研究状态已推进，入口索引未同步；历史计划与当前入口混放增加误读风险。
- exposure：本次仅只读审计及文档整理，无实验启动、GPU 调用或 target 读取，研究结论不受影响。
- resolution：重写两级导航，中文稿归位；77 份历史文档分类归档，保留 18 份机器兼容文件与 8 份链接依赖。
- prevention：授权只读 `docs/RESEARCH_STATUS.md`，旧计划仅供追溯；未来移动文件先检查代码引用与冻结证据路径。
- evidence：`docs/archive/2026-09/root-docs-20260906/MANIFEST.json`；task=`docs-reorganization-20260906`；new runs=0。

下一可用编号：`V71-F54`。下方历史条目的 next-ID 属于当时状态。

## V71-F52 — M43 描述性 point-surface 汇总覆盖了基线算子计数（2026-09-05）

- category=`evaluation_operator_accounting`；代码与科学证据基线 commit=`1913ab0e`。
- discovery：论文证据审计只读检查 `scripts/run_worldsim_v71_m43_m39_av2_zero_shot.py::_evaluate_bundle`
  与已完成 canonical `20260905T091500Z__m43-m39-av2-zero-shot-r1/summary.json`；没有重评测或读取新 target。
- root cause：`evaluate_actor_surface` 先产生 literal point-surface baseline/output；随后 `row.update`
  用 unit categorical 的计数覆盖 `baseline_early_count` 和 `baseline_hit_count`，但保留 literal output 计数。
  描述性 `m8_point_surface` 的 early/hit 差值因此不是同算子比较。
- affected interpretation：原 all/hazard `16.907/16.556% -> 45.271/50.149%` 与 hit `+0.079pp`
  不再作为几何迁移效果；撤回“这些差值已证明 geometry/sensor coupling”这一归因。原 artifact 不改写。
- unaffected：M39 正式 categorical baseline/output 使用匹配算子，`+0.229/+0.542/-0.036pp` early、
  `+5.888/+6.302/+5.537pp` hit 与既有拒绝结论不变；描述性 Chamfer `+0.162mm` 仍是合法配对。
- resolution：main 移除混算子归因；本地 supplement 明示 erratum；同步 STATUS、EXPERIMENTS 与 synthesis。
- remaining boundary：本次论文任务不修改 runner、不重读 target。未来复用 runner 前必须为 literal/categorical
  分别命名 baseline/output 字段，并以固定合成计数测试聚合。当前状态=`documentation_corrected_runner_risk_open`。
- V7.2 resolution update（2026-09-06）：`WS-V72-P0-OPERATOR-ERRATUM-01` 已将 runner 的 categorical baseline
  改为独立 `categorical_baseline_*` 字段，并为 M43 接入输入／输出 Actor 状态实测。已有 canonical JSONL 未保存
  被覆盖前的 literal baseline，故当前状态=`runner_fixed_legacy_recompute_pending_gpu`；未来重算只可使用原 20-log
  cohort、冻结模型与匹配 literal 算子，结果只作 `legacy_diagnostic`，不得改变 `V71-F43`。

下一可用编号：`V71-F53`。下方历史条目中的 next-ID 属于当时状态。

## V71-F43 — M39 categorical authority does not preserve early-return direction across sensors（2026-09-05）

- run=`run://worldsim_v71/WS-V71-M43-M39-AV2-ZERO-SHOT-01/20260905T091500Z__m43-m39-av2-zero-shot-r1`；
  complete=`20/20` logs、352 Actors、1,016,652 rays，final summary只在`ALL_COMPLETE`且三类进程退出后读取；
- symptom=M39相对unit categorical baseline的all/hazard/clear early delta为`+0.229/+0.542/-0.036pp`，仅clear
  不增；hit delta为`+5.888/+6.302/+5.537pp`，因此冻结判定=`false/false/true`、仅1/3通过；
- retained evidence=跨传感器上 M39 恢复 hit，但 hazard early 恶化；描述性 M8 early/hit 混算子差值不构成
  几何迁移证据（更正见 V71-F52），其 Chamfer 配对仍为 `+0.162mm`；
- root cause boundary=正式同算子比较拒绝 source-domain 改善方向的直接跨传感器迁移，但不能单独识别
  几何、证据与传感器分布的因果贡献。categorical normalization 仅具有全局权重缩放不变性，不消除
  不同 primitive family 相对数量或质量的影响；source-domain M39 mechanism 保留；
- resolution=按事前协议保留M39为development-exposed mechanism result，停止AV2 fine-tuning、calibration、threshold/
  margin/bin/scale sweep、log replacement或回选descriptive arm；论文明确报告负迁移与hit--early trade-off；
- integrity=no partial quality read、no target adaptation、no failed-log deletion；状态=`scientific_rejection`。

下一可用编号仍为：`V71-F52`。

## V71-F51 — all-in-one单实例guard自匹配未来launch参数（2026-09-05）

- symptom：将`pgrep -f`检查与nohup launch写在同一远端shell命令时，shell自身command line包含后半段runner/
  downloader名称，guard误判“already running”并在启动前退出；
- exposure：evaluator/downloader/s5cmd均未启动，M43仍`16/20`，0新target read、0partial quality、0状态覆盖；
- resolution：用独立只读`ps`调用确认三类进程为空，再执行无内嵌字符串guard的单次launch；随后核对PID=
  `1751/1752/1759`、status waiting、GPU `258MiB`、无traceback；
- prevention：长命令中的`pgrep -f`会匹配同一shell后续参数；以后把read-only process audit与launch拆成两次调用，
  不叠加脆弱门控。状态=`resolved_pre_launch`，next ID=`V71-F52`。

## V71-F50 — M43 runner缺少同run断点续跑入口（2026-09-05）

- symptom：用户重新开机要求继续M43时，原进程已退出，而runner固定
  `run_dir.mkdir(parents=True, exist_ok=False)`；直接复用run ID会在加载模型/外测前报目录已存在，新run ID则会
  重读前16个target logs并破坏exact-once语义；
- exposure：恢复诊断只读`status.json`、进程、marker和日志计数；M43保持`16/20`、276 Actors、final summary=false、
  partial quality human read=false；0新target evaluation、0模型/阈值/cohort/decision变化；
- root cause：最初runner只覆盖不中断的一次性执行，没有把长IO与关机后的程序级continuation建模为合法状态；
- resolution：增加显式`--resume`，从`status.completed_logs`恢复冻结cohort前缀，程序内部载入自产partial rows用于最终
  拼接，并从下一log继续；既有summary存在时拒绝resume，row log只能属于已完成前缀；
- prevention：长时exact-once external runner必须区分fresh launch与explicit resume；恢复不创建新run、不重算已完成log、
  不输出partial aggregate，downloader/evaluator均保持单实例；状态=`resolved_pre_quality`，next ID=`V71-F51`。

## V7.1 final handoff prevention note — incomplete M43 is not a scientific failure（2026-09-05）

- M43 handoff state=`running/waiting_fresh_av2`, `16/20` logs、276 Actors；current log=
  `c85a88a8-c916-30a7-923c-0c66bd3ebbd3`；唯一evaluator/downloader健康，最近日志无error/retry，磁盘余量
  `86GiB`；
- run没有`summary.json`，未读取`EXTERNAL_ACTORS.partial.jsonl`或任何partial quality；因此不得登记`V71-F43`、
  不得把进度或Actor count解释成method quality，也不得用旧nonlearned AV2结果替代learned transfer；
- 本次由用户明确要求在docs/paper推送后shutdown。该中断是资源生命周期动作，不是method/engineering failure，
  next failure ID仍=`V71-F50`；
- 合法恢复只允许复用同一cohort/config/run和已有下载、保持单一downloader；20/20且evaluator正常退出后才读取一次
  aggregate。若三项冻结判定任一失败，届时登记`V71-F43`并关闭AV2 adaptation。

## V7.1 paper object boundary — classifier/reliability variables are legacy（2026-09-05）

V7.1主问题不得重新以artifact probability、hazard classifier或task-reliability density替代GT surface与first-return
supervision。旧变量仅作supplement中的历史路径；主claim对象是surface geometry、return evidence和typed state。
纯写作重构无新failure；next ID仍=`V71-F50`。

