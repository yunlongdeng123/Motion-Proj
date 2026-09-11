# 历史原始记录 059

> 冻结历史，不是当前执行规则。当前规则见 [scaling law](../../../auto-research_scaling_law.md) 与 [AGENTS](../../../AGENTS.md)。
> 原文件：`docs/RESEARCH_FAILURES.md`；迁移前提交 `abbd431c`。原有相对链接按原目录解释，证据路径原样保留。
> 按索引读取相关小节；旧哈希、过度门控和资源指令均不重新生效。

## V5.1 M1-only 新增防重复结论（2026-08-17）

- `V51-F01`（`engineering`, `resolved`）：首轮执行
  `pytest -q tests/test_worldsim_v51_protocol.py tests/test_audit_worldsim_v51_start.py` 在 collection 阶段因测试文件直接
  import `motion_proj.worldsim_v51`、但未显式把仓库根加入 `sys.path` 而报 `ModuleNotFoundError: motion_proj`；同轮
  `python scripts/audit_worldsim_v51_start.py --help` 已通过，所以该 terminal 推翻的是“pytest 启动环境总会自动注入
  repo root”的工程假设，不是 P0 协议或算法失败。修复在测试 import 前按绝对 `ROOT` 注入路径并以原命令回归；失败时
  没有运行方法、读取 validation/test/KITTI quality 或产生质量数字。后续直接脚本与测试入口都必须有独立 import smoke，
  禁止把 collection error 计入方法分母。证据：`WS-V51-P0-M1-SCOPE-FREEZE-01`、
  `tests/test_worldsim_v51_protocol.py`、`tests/test_audit_worldsim_v51_start.py`。
- `V51-F02`（`engineering/evaluation`, `resolved`）：A0 runner 的 metric 单测把输入 `float32(0.1)` 产生的
  `0.10000000149011612` 与 Python 十进制 `0.1` 做严格相等，导致 `1 failed / 6 passed`；这推翻的是“人工十进制常数
  可以作为 bit-exact 浮点 oracle”的测试假设，不是 frozen metric 定义或 A0 canonical replay 失败。修复只把人工常数
  断言改为 `pytest.approx`；正式 A0 仍把同一实现重算值与 canonical JSON float 做 `delta == 0.0`，posterior/statistics
  仍逐 bit 比较。禁止为了让 exact gate 通过而对 canonical metric 使用容差、舍入或字符串截断。失败时未启动正式 run、
  GPU renderer、方法推理或 validation/test/KITTI quality read。证据：`WS-V51-M1-A-UNARY-OBSERVABILITY-01`、
  `tests/test_replay_worldsim_v51_v5_unary.py`。
- `V51-F03`（`engineering/evaluation`, `resolved`）：A1 规定 `visibility >= 0.01` 为 inclusive gate，但冻结 NPZ 中
  visibility 是 `float32`；若直接与 Python double `0.01` 比较，存储的 `float32(0.01)` 会因表示略小而被误判为
  false，首轮测试得到 `[False,False,False]` 而非 `[True,False,False]`。这会真实改变 observation denominator，不能靠
  放宽单测解决。修复是在比较前把配置阈值量化为 observation dtype，同时在诊断中分别记录 configured/applied value；
  门仍是 inclusive，未读取 evaluation quality 或搜索阈值。禁止用 epsilon、容差或事后改 threshold 隐式改变分母。
  证据：`WS-V51-M1-A-UNARY-OBSERVABILITY-01`、`motion_proj/worldsim_v51/evidence/visibility.py`、
  `tests/test_worldsim_v51_visibility.py`。
- `V51-F04`（`algorithm/protocol`, `resolved before A2 quality read`）：A2 首轮只读 evidence-statistics 检查发现，
  若直接在“全部 Gaussian”上取 effective-count 下分位数、entropy/disagreement 上分位数并用 OR 组成 UNKNOWN，
  scene-0379/1087 的 count 与 disagreement 分位数会同时退化为 `0`；inclusive `disagreement >= 0` 会把全部 Gaussian
  判为 UNKNOWN，Gaussian coverage 直接变成 `0`。根因是未观测但由冻结 base-model prior 明确赋类的 Gaussian 占
  `67.39%/97.20%`，它们不能和真正有 semantic observation 的校准总体混在一起。A2 在任何 evaluation artifact 或
  quality metric 读取前，把阈值总体冻结为“三个 H scene 中 effective-count>0 的 A1 Gaussian pooled population”，并用
  `high entropy AND (low count OR high disagreement)` 保留 entropy 作为必要条件；三个阈值固定为该总体的
  `Q25(count)=0.19274792820215225`、`Q75(entropy)=0.005402358970383594`、
  `Q75(disagreement)=8.494543610182426e-12`。禁止把全量分位数退化误写成 A2 方法负结果，也禁止在看到 A2 evaluation
  quality 后改总体、分位点、布尔规则或图像 abstain threshold。证据：
  `configs/worldsim_v51/m1_unary_unknown_v1.yaml` 与 A1 posterior SHA binding；S/validation/test/KITTI 仍未读取。
- `V51-F05`（`algorithm/protocol`, `rejected by A3 r005`）：计划 A3-1 提议以
  `n_eff=(sum r)^2/(sum r^2+epsilon)` 限制 A3-0 的 fractional concentration `sum r`，并解释为
  correlation-aware。但 A1 reliability 逐 observation 严格在 `[0,1]`，因此忽略仅用于数值稳定的 epsilon 时恒有
  `sum r^2 <= sum r`，进而 `n_eff >= sum r`：作为上限必然是 no-op，直接替换则会提高而非降低 posterior
  concentration。更根本的是该式只见单 observation 权重及其平方和，没有任何 view-pair correlation observable；
  调换 view 顺序或相关结构而保持权重集合时结果不变，不能支持“10 个高度相关 view 不等于 10 个独立证据”的命题。
  r005 用 v2 evidence-only audit 对 `944,443` 个 positive-count Gaussian 复现：无 epsilon 时
  `n_eff<sum(r)` 为 `0`，absolute cap change>`1e-9` 为 `0`；若直接 replacement，`940,762/944,443=99.6102%`
  Gaussian concentration 被放大。结论=`a3_kish_cap_rejected_structural_noop_not_correlation_aware`；A3 不启动 GPU
  quality arm，按原计划只解锁独立 A4。禁止为挽救 A3 事后加入相关系数/时间核/feature similarity；那将是新机制，
  不是原 A3-1 的修复。r005 未读 evaluation artifact/quality、validation/test/KITTI，failure delta=`V51-F05/F06`。
- `V51-F06`（`engineering/evaluation`, `resolved without quality read`）：A3 audit v1/r004 用相对 cap change 判断
  epsilon 是否产生“有意义修正”，但 0471/0379 存在 reliability=`1.401298464324817e-45` 的 float32 最小次正规数；
  `epsilon=1e-12` 使这些近零 mass 的相对变化达到 `1.0`，尽管三个场景最大绝对 cap reduction 仅约
  `2.5e-13`。r004 因此合法保留为 `done/inconclusive`，不是 A3 得到有效 concentration reduction，也不是方法质量
  失败；它没有读取 evaluation artifact/quality、启动 GPU renderer 或改变 posterior。v2 新配置绑定 r004 与 v1 hash，
  在新结果前把 meaningful gate 改为 absolute cap change>`1e-9`，同时继续报告相对量作诊断。禁止用近零分母的巨大
  相对数宣称机制有效，也禁止覆盖 r004 terminal；只能用新 r005 重放同一 45 份 evidence observation。
- `V51-F07`（`algorithm/novelty`, `rejected by A4 r006`）：CIF 原论文将 occupancy probability 与 conditional
  instance distribution 分开，并明确针对 appearance opacity 与 occupancy 混淆；其完整方法还包含 learned deformable
  Gaussian instance field、identity calibration 与 semantic resampling。V5.1 计划明确不引入后三类机制，而当前 renderer
  已把 appearance `base_opacity` 与 conditional ownership sidecar 相乘，A1 已分离 visibility eligibility，A2 已分离
  UNKNOWN。因而 A4 若把 `base_opacity` 当 occupancy，会在 renderer 中二次乘 alpha 且违反参考机制；若用
  visibility/effective-count，会再次把不可见误作不存在；若对已实例化 Gaussian 设 occupancy=1，则与现有 renderer
  bit-exact no-op。r006 绑定三个 A2 posterior 与 renderer/visibility/abstention 源码，确认 occupancy field=`0/3`、
  constant-one 对现有 renderer=`3/3 bit exact`，而复用 appearance opacity=`3/3 non-exact` 且会二次缩放。结论=
  `a4_cif_decoupling_rejected_no_independent_occupancy_observable`；未读 evaluation artifact/quality、未启动 GPU/training。
  禁止把已有 A1/A2 分解重新命名为 CIF 增益，或在结果后偷偷解锁完整 CIF 训练、校准/重采样。参考：
  [CVPR 2026 official paper](https://openaccess.thecvf.com/content/CVPR2026/html/Wu_Consistent_Instance_Field_for_Dynamic_Scene_Understanding_CVPR_2026_paper.html)。
- `V51-F08`（`engineering/resource`, `resolved without duplicate run`）：首次尝试并行启动 S unary materialization 时，
  PowerShell→SSH→远端 Bash 的多层转义把 `$!/$?` 保留成字面量，导致外层 `wait/test` 解析失败。只读 PID、GPU、run 目录
  与 status 审计确认 scene-0998 只有一个正式进程，scene-0359 根本没有启动；因此保留 r049 单实例自然完成，再以独立前台
  命令串行执行 r050。两者均为 `done`、checkpoint 前后 SHA exact，且无重复 scene/candidate 分母。3090 实测后段显存保留
  分别达到约 `22 GiB/20 GiB`，也推翻了“两个 unary materialization 可安全共卡并发”的资源假设。以后 Windows 发起的远端
  后台编排不得内联依赖 `$!/$?`；优先每个长 run 使用独立前台 SSH session，确需后台时使用远端脚本/控制器并单独审计
  PID、日志与 terminal。wrapper 失败不得写成方法失败，也不得因外层 exit code 重跑已封口的 immutable run。证据：
  r049/r050、source commit `6950597`、`configs/worldsim_v51/stage_a_screening_v1.yaml`。
- `V51-F09`（`algorithm/evaluation`, `rejected by Stage A r007`）：A1 visibility 在 H 上通过的 scene-balanced
  `ΔBoundary-F1=+0.001155713` 没有在预注册 S=`0998/0359` 上复现。S 两场 delta 分别为
  `-0.0000904944/+0.0000574359`，只有 `1/2` 非负、`0/2` 达到 clearly-positive `+0.001`，均值为
  `-0.0000165293`；尽管 mean IoU/Brier/ECE 略改善且 FN 增量仍在门内，冻结 gate 是合取，A1 必须 rejected。
  这推翻“hard visibility eligibility 的 H 小效应可跨开发场景稳定复制”，不是 Bayesian U2/B3 基线失败。禁止根据 IoU
  或 calibration 的微小正向分量保留 A1、删除 0998、放宽 clearly-positive 门或在同一 S 上重选 visibility threshold。
  合法复开需要新机制、新任务和未读场景；V5.1 当前冻结 U2/B3，不再继续复杂化 Bayesian family。证据：r007、
  `configs/worldsim_v51/stage_a_closeout_v1.yaml`。
- `V51-F10`（`algorithm/evaluation`, `rejected by Stage A r007`）：A2 UNKNOWN 在 S 上仍能集中错误：scene-balanced
  accepted/abstained error=`0.0148416/0.134393`，两场均有非空 denominator；但 coverage 在 0998/0359 为
  `0.250105/0.864765`，scene-balanced mean=`0.557435<0.60`，未通过冻结 selective gate。0998 的 UNKNOWN Gaussian
  ratio=`34.5512%`，0359 仅=`0.7891%`，说明 H 分位数规则的场景依赖很强；同时 A2 conditional posterior 与 A1 相同，
  继承 `V51-F09` 的 conditional gate 失败。因此 A2 rejected，不能用较高 unknown recall 或 error separation 掩盖可用覆盖率
  不足，也不能在看到 S 后调整 Q25/Q75、布尔规则或 image threshold。证据：r007；failure delta=`V51-F09/F10`。
- `V51-F11`（`protocol/governance`, `resolved by explicit user authorization on 2026-08-17`）：normative plan 对 Stage A 全失败后的解锁规则内部冲突。§10.8
  明写“所有 Stage A arm 都失败”时保留 U1/U2 并进入 Stage B；附录“八、Stage A 后如何解锁”却只在 Stage A
  candidate 通过 S 时允许 `WS-V51-M1-B-LUDVIG-UPLIFT-01`。r007 的真实状态正是 A1–A4 全 rejected、fallback=
  `U2/B3`，因此执行者不能静默挑选有利条款，也不能把“进入 Stage B”的研究顺序当成独立授权。Stage B 保持
  `pending/locked`；合法复开必须由用户明确选择“授权 U2/B3 fallback 进入 Stage B”或“按 candidate-pass 条款关闭
  M1”，再用 freeze-only commit 统一 normative/short plan/config。用户于 2026-08-17 明确选择“授权 U2/B3 fallback
  继续 M1”，并要求单 arm/scene/工程/paper failure 留档后自动进入下一冻结路线；因此本治理阻塞解除，但原条款冲突和
  r007 结论不删除。解法采用 executable authorization overlay 绑定原 normative/P0/Stage A/proposal SHA，不原地改写
  冻结字节；M2/M3 与 validation/test/KITTI 锁保持。该问题不是算法负结果，解除时仍未读取 C/validation/test/KITTI
  quality。证据：`docs/WORLDSIM_V5_1_M1_TOPCONF_PLAN.md`、`configs/worldsim_v51/stage_a_closeout_v1.yaml`、
  `configs/worldsim_v51/stage_b_authorization_v1.yaml`、`docs/WS_V51_STAGE_B_PREFLIGHT.md`。
- `V51-F12`（`engineering/resource/governance`, `resolved by r003 asset + r004 resource smoke`）：Stage B 的 faithful 第一版要求官方 DINOv2
  ViT-g/14 registers，但 2026-08-17 只读审计只找到 Depth-Anything-V2 内部 DINOv2 模块，torch/HuggingFace cache
  均无对应官方 checkpoint；“存在 DINOv2 源文件”不能写成 LUDVIG 资产已冻结。官方 LUDVIG README 记录的测试平台是
  A6000 48GB，当前 RTX 3090 只有 24576 MiB，且 Stage A 单个 unary materialization 已实测约 20–22 GiB，因此
  DINO extraction 与 DriveStudio renderer 禁止同进程或同卡并发常驻。授权后必须先冻结 upstream commit、官方模型来源、
  checkpoint SHA/license、preprocess 与 PCA population/seed，再采用“离线 DINO sidecar→释放进程→renderer uplift”分段
  执行。官方 checkpoint HEAD bytes=`4,546,140,349`，multipart ETag 不是 SHA；官方 extractor 又会把全 camera raw
  feature 预分配在 GPU。V5.1 只允许先下载后全 SHA，再用 CPU memmap/40-D patch-grid streaming 与 dense-parity test
  保持语义；缺资产或 OOM 只记工程/resource terminal，不得写成 feature uplift 失败，也不得临时换小模型、降分辨率后
  仍称 faithful port。
  资产缺失子项由 r003 解除：official bytes=`4,546,140,349`、SHA-256=`746ecb8c...a283`，本地重算
  8 MiB×542-part ETag exact。r004 随后在 clean source=`935d2b2` 上以官方 commit=`7764ea0f...25fc8`、原始分辨率
  预处理、ViT-g/14 registers 与 strict state dict 完成 one-image forward：params=`1,136,486,912`、keys=`568`、
  missing/unexpected=`0/0`，4 个输出 shape 全 exact；GPU sampled/Torch reserved peak=`6,702/6,376 MiB`、cgroup peak=
  `15,701,860,352 bytes`，显著低于预注册门，资源风险因此 resolved。该解除不代表 feature uplift 有效，也不解除
  DINO→释放进程→renderer 的顺序合同；后续仍禁止同卡并发、临时换小模型/分辨率，且必须先过 operator parity。
  证据：`configs/worldsim_v51/stage_b_preflight_v1.yaml`、`docs/WS_V51_STAGE_B_PREFLIGHT.md`、
  `configs/worldsim_v51/stage_b_dinov2_resource_freeze_v1.yaml`、r004 canonical run。
- `V51-F13`（`engineering/protocol`, `resolved without method execution`）：P0 scope config 已把 normative plan
  SHA-256 冻结为 `3d7f7481...`，但 Stage A closeout commit `3d33262` 曾直接向该长计划加入 5 行执行进展，使当前
  HEAD SHA 漂到 `b119cd56...`。Stage B preflight 运行 `pytest -q tests/test_worldsim_v51_protocol.py` 时因此得到
  `2 failed / 1 passed`，即使本轮新增注记撤回后仍复现，证明这是 inherited drift。这推翻“冻结后的 normative plan
  仍可作普通活文档窄改”的工程假设，不是 P0、Stage B 算法或数据失败。修复用新提交移除这 5 行，把当前状态继续保留
  在 short plan/status/experiments，恢复原 plan SHA exact；不改写历史，也不只改 expected SHA 掩盖漂移。若用户授权后
  确需统一解锁规则，必须建立显式 supersession/migration 并同步 P0 binding。失败时没有下载 checkpoint、启动
  method/GPU run 或读取 C/validation/test/KITTI quality。证据：`3d33262`、
  `configs/worldsim_v51/p0_m1_scope_v1.yaml`、`tests/test_worldsim_v51_protocol.py`、
  `docs/WS_V51_STAGE_B_PREFLIGHT.md`。
- `V51-F14`（`engineering/protocol`, `resolved by r010`）：LUDVIG DINO extractor 的 PCA 路径不是天然确定性合同。
  `PCA(n_components=40)` 没有 `random_state`，大矩阵会走 randomized solver；当 patch 数超过 500,000 时还用未设 seed
  的 `np.random.choice` subsample。更隐蔽的是 GPU path 用 PyTorch `std`（默认 correction=1），CPU path 用 NumPy
  `std`（correction=0），所以为省显存切到 CPU 会改变标准化与全部 feature。V5.1 proposal 冻结 H evidence=
  `45 views×7,296 patches=328,320`，明确不触发 subsample；固定 std correction=1、randomized PCA
  random_state=`20260814`、40-D、whiten=false，并把 scaler/PCA state 持久化后只 transform S/C。不得把 solver/seed/std
  差异当作 backbone 增益或在 S/C refit；这是 reproducibility hardening，不是参数搜索。本轮未下载模型、提取 feature 或
  读取质量。证据：LUDVIG `predictors/dino.py`、`configs/worldsim_v51/stage_b_freeze_proposal_v1.yaml`、
  `docs/WS_V51_STAGE_B_FREEZE_PROPOSAL.md`。r010 已对 45 H views exact 执行该 hardened contract：首图 raw feature
  repeat bit-exact，PCA state deterministic NPZ repeat byte-exact，45 个 sidecar 的 file/content SHA 全 exact，raw memmap
  成功后删除，PCA state SHA=`fe9eea72...3231c8`；因此本条 resolved。该解决只证明 feature/PCA 可复现，不证明
  LUDVIG uplift 或方法质量有效，S/C 仍只准 transform、不得 refit。
- `V51-F15`（`evaluation/governance`, `resolved by r015 without promoting proxy to method input`）：Stage B 的 same-actor/actor-background metric 可从
  frozen `RigidNodes.points_ids[:,0]` 与 Background row 构造，但这是 base-model membership proxy，不是真实 ownership GT。
  若把该 proxy 输入 DINO/PCA/uplift/权重会形成标签泄漏；若只凭 proxy margin 解锁 Graph，则会把模型自身表示循环证明为
  语义正确。proposal 将其限制为 evaluation-only stratum，强制写
  `model_membership_proxy_not_ground_truth`，并同时报告不消费 membership 的 same-Gaussian repeatability 与 heldout DINO
  reprojection。无 eligible actor 的 scene 必须保留 abstain；不得降低 32-Gaussian eligibility、删 1087/0379 或只报大
  Rigid 场景。Stage B 未获授权，本轮没有产生 metric。证据：V5 formal30k r027–r034 metadata、
  `configs/worldsim_v51/stage_b_freeze_proposal_v1.yaml`、`docs/WS_V51_STAGE_B_FREEZE_PROPOSAL.md`。r015 已严格按
  evaluation-only 声明执行：proxy 未进入 method/PCA/uplift，1087 因无 eligible actor 保留 abstain，同时报告不消费
  membership 的 repeatability 与 heldout reprojection；治理风险因此 resolved，但方法因 margin 失败另记 `V51-F31`。
- `V51-F16`（`engineering/resource`, `resolved by parallel r003`）：DINOv2 asset r002 在已 source network turbo、official URL、
  fixed target 与 curl resume 合同下运行约 `106 s`，连续 prefix 仅增长到 `26,566,656 / 4,546,140,349 bytes`；
  按稳定窗口外推需数小时。执行者精确核对并 `TERM` 唯一 curl PID，runner 以 `exit=-15` 写入 blocked terminal，
  final asset 不存在，`.partial` 及其 SHA=`934ef5aa...e2265` 保留。该事实推翻“代理单连接足以在合理实验窗口完成 4.5 GB
  official asset”的工程假设，不是 checkpoint 损坏、DINO/LUDVIG 方法或 GPU 失败。合法恢复必须新 run ID，冻结 prefix
  bytes/SHA，以互不重叠的 fixed HTTP ranges 并行下载；每段验证 range bytes/SHA，assembly 后同时通过 full SHA-256 与
  S3 multipart ETag=`3d1b...-542`（8 MiB×542 parts）才可原子发布。禁止覆盖 r002、删除 prefix 后假装首次下载、
  使用镜像/不同权重或只凭 total bytes/remote ETag 宣称完成。证据：
  `20260817T141600Z__m1-stage-b-dinov2-asset-s20260814-r002`、
  `configs/worldsim_v51/stage_b_dinov2_download_parallel_v1.yaml`。r003 以 14 ranges 在 `1504.935 s` 完成；逐段
  bytes/SHA、assembled full SHA=`746ecb8c...a283`、multipart ETag=`3d1b...-542` 与 terminal/manifest 二次复核全 exact，
  final 原子发布后精确删除 `15 files / 4,546,140,349 bytes` staging。本条工程恢复因此 resolved；当时仍 active 的
  ViT-g 24GB resource smoke 风险 `V51-F12` 后由 r004 独立解除。
- `V51-F17`（`engineering/protocol`, `resolved before formal r005`）：synthetic operator parity 首轮 unit regression=
  `2 failed / 8 passed`，两处 failure 都来自同一个 below-Gaussian-view-mass 夹具：它把 24 个 intersection 全设为
  minimum contribution=`1e-4`，所以聚合 mass 实际为 `0.0024≥0.001`，被测算子正确保留该 group，而测试错误地要求
  drop。这推翻的是“逐 intersection 在 floor 上就能构造低于 group floor”的夹具假设，不是 B0/B1 公式、dense oracle、
  lazy bilinear 或 LUDVIG 方法失败。修复只把该 group 改成 5 个 `1e-4`、其余为 0，使 mass=`0.0005`；不改阈值、
  operator、seed 或通过标准。失败发生在 formal r005 创建前，未加载 DINO/renderer、未读真实 feature 或质量；禁止用
  降低 group floor 掩盖夹具错误。证据：`tests/test_worldsim_v51_feature_uplift.py`、
  `scripts/audit_worldsim_v51_stage_b_operator_parity.py` 的 pre-formal regression。修正后 19/19 regression PASS；formal
  r005 又以 11/11 checks PASS，并真实观测 `8 Gaussian-view → 7 kept + 1 dropped`，确认本条 resolved。
- `V51-F18`（`engineering`, `resolved before r005 result-freeze commit`）：新增 result-freeze test 先通过 canonical run
  文件存在/SHA、summary status/checks/checkpoint immutable，随后因把局部变量 `validate_freeze` 简化为 `freeze` 时漏改
  两条 parity 断言，得到 `NameError` 与 `1 failed / 19 passed`。这推翻“机械重命名后所有引用自然一致”的测试维护假设，
  不推翻 r005 artifact、operator parity 或任一质量结论。修复只替换两处旧变量名，并重跑同一 20-test regression；
  禁止重跑/覆盖 r005 或修改 freeze 数字来绕过测试。证据：`tests/test_worldsim_v51_feature_uplift.py`、
  `configs/worldsim_v51/stage_b_operator_parity_freeze_v1.yaml`。
- `V51-F19`（`engineering/protocol`, `resolved by v2/r007 reaching renderer`）：one-H-view formal r006 在
  `_build_runtime()` 导入 DriveStudio `models.gaussians.basics` 时因 `ModuleNotFoundError: pytorch3d` blocked；v1 config 错把
  入口冻结为 motionproj Python，而历史 DriveStudio 运行合同使用独立 `/root/autodl-tmp/envs/drivestudio/bin/python`。
  terminal 发生在 dataset/trainer 构造和 renderer 启动前，0 intersection、0 denominator、0 quality；这推翻“主项目环境可
  直接承载 DriveStudio CUDA 依赖”的工程假设，不是 renderer/LUDVIG/资源失败。合法恢复必须保留 r006，以 v2 + 新 r007
  只替换 interpreter，并在 formal 内 exact 核对 executable、torch=`2.1.2+cu118`、CUDA=`11.8`、`pytorch3d/gsplat`
  imports；不得安装包污染 motionproj env 或改 view/floor/resource gate。证据：r006 status SHA=`06b74ec9...b4be3`、
  `configs/worldsim_v51/stage_b_one_view_contribution_v1.yaml` 与 v2 recovery config。v2/r007 已 exact 使用该环境并完成
  dataset/trainer/checkpoint/renderer 启动，本条因此 resolved；r007 的后续尺寸阻塞另记 `V51-F20`。
- `V51-F20`（`engineering/protocol`, `resolved by v3/r008 reaching post-render resource gate`）：r007 到达真实单视图 renderer 后，v2 把 sensor
  JPEG `1600×900` 错冻为 renderer width/height，触发 fail-closed。冻结 r027 source config 已明确三路
  `downscale_when_loading=[2,2,2]`，现有 V5 SAM/actor configs 与历史 `V3-F18` 也记录 model-native=`800×450`；这是重复
  违反三层尺寸合同，不是 renderer 或 contribution 质量失败。r007 在 intersection inventory 前停止，且旧错误文本未写出
  observed/expected 数值。合法恢复必须保留 r007，以 v3/r008 显式同时冻结 sensor/downscale/model-native 三层尺寸并增强
  错误文本；不得改 checkpoint、view、support floor 或把 800×450 写成降分辨率调参。loader 会基础设施性物化
  image/mask/LiDAR，但 runner 不消费其值；二者须分开记录。证据：r007 status SHA=`da279515...8d3c9`、r027
  `config.yaml` SHA=`eb22faea...9c6d`、`configs/worldsim_v51/stage_b_one_view_contribution_v3.yaml`。v3/r008 已按
  `800×450` 完成 renderer 并进入 post-render 资源门，因此本条 resolved；后续资源阻塞另记 `V51-F21`。
- `V51-F21`（`engineering/resource/protocol`, `resolved by v4/r009`）：r008 在真实单视图 renderer 和
  contribution 汇总完成后，NVIDIA peak=`14,234 MiB` 超过预注册 ceiling=`12,288 MiB`，故 status=`blocked`；
  cgroup peak 仅 `9,598,074,880 bytes`，89 个采样无错误，进程正常退出且 GPU 已释放，不能误写成 OOM、renderer
  或算法质量失败。原 runner 又在资源门通过后才写 contribution/resource artifact，使 blocked run 只保留
  status/events/resolved/resource-samples；这会降低失败诊断可审计性。禁止覆盖 r008，合法恢复只能新建 v4/r009：
  保持 scene/view/checkpoint/renderer/two-floor/quality locks 不变，仅把 NVIDIA/Torch ceiling 提升为 `16,384 MiB`
  （仍低于 24 GiB），并在资源判定前先持久化只读 denominator/resource 诊断。r008 status/resource-samples SHA=
  `8b8ebe17...b2118bf / fc0f9788...a90932`；不得用这次工程资源事实选择或调节算法质量。v4/r009 已在
  `14,234 MiB NVIDIA / 13,882 MiB Torch reserved / 9,593,946,112 bytes cgroup` 下通过，诊断 artifacts 也在 gate 前
  持久化，因此本条 resolved；冻结结果见 `stage_b_one_view_contribution_freeze_v1.yaml`。
- `V51-F22`（`engineering/audit`, `resolved immediately`）：r009 完成后的第一次独立逐文件 verifier 把 manifest
  payload key 硬编码为 `files`，但该 runner 的冻结 schema 使用 `inventory`，只读命令因此 `KeyError: 'files'`；
  formal r009 及任何 artifact 均未改变。修正 verifier 按冻结 schema 读取 `inventory` 后，8/8 manifest entries 的
  SHA/bytes exact，run=`10 files / 28,156 bytes`。后续 verifier 必须先读 schema/key，不能跨 runner 猜测 manifest 字段。
- `V51-F23`（`engineering/resource/protocol`, `resolved by v2/r012`）：H 45-view sparse uplift r011 已完整处理
  3 scenes×15 views、写出 6 个 Gaussian feature sidecar 并证明 3 个 base checkpoint before/after exact，但 post-compute
  NVIDIA peak=`20,554 MiB` 与 Torch reserved=`20,202 MiB` 超过 v1 预注册的共同 ceiling=`18,432 MiB`，故 formal status
  必须保持 `blocked`。cgroup peak 仅 `13,328,011,264 bytes`，799 个 resource samples 无 monitor error，runner 正常释放 GPU；
  这推翻的是“单-view 14,234 MiB 足以外推三场景 streaming full-run 低于 18 GiB”的资源假设，不推翻 sparse transpose、
  B0/B1、DINO/PCA 或任何 method-quality 结论。合法恢复必须保留 r011 及其 gate 前诊断，以 v2 + 新 r012 从冻结原输入
  完整重跑；唯一改变为 NVIDIA/Torch ceiling=`22,528 MiB`（仍低于 24 GiB），不得复用 blocked sidecar、改 scene/view/floor、
  调整 operator 或读取 membership/quality。r011 失败证据已独立验证 6/6 NPZ file/content identity、manifest chain 与
  checkpoint immutability exact；status/resources/report/manifest/resource-samples SHA=
  `a450cdaf...0eee6/0312e190...8a37/98140571...99c9/88956448...dae6/76422821...ae9c`。配置证据为
  `configs/worldsim_v51/stage_b_h_uplift_v1.yaml` 与 v2 recovery config。v2/r012 从原冻结输入完整重跑，在相同 observed
  NVIDIA/Torch reserved=`20,554/20,202 MiB` 下通过 22 GiB 门；6/6 sidecar、19/19 manifest、3 checkpoint 与全部 locks
  独立审计 exact，因此本条 resolved。该解决不证明 B0/B1 方法质量有效，`V51-F15` 仍须由预注册 evaluation-only 门处理。
- `V51-F24`（`engineering/protocol`, `resolved before formal r013`）：H heldout feature 预注册回归中，新 runner/sidecar
  测试已在 frozen motionproj interpreter 得到 `8 passed`，但随后一次聚合命令又在同一 interpreter 调用依赖 DriveStudio
  runtime 的 `test_worldsim_v51_h_uplift.py`，该测试按合同报告 runtime mismatch，汇总为 `1 failed / 9 passed`。这只是测试
  调用环境错误，未创建 formal run、未启动 GPU、未读 membership/uplift quality，不能写成 H uplift 或 heldout transform 失败。
  合法修复是按 `V51-F19` 的环境分层分别执行：DINO/sidecar/heldout tests 使用 motionproj Python，renderer/uplift tests
  使用 drivestudio Python；禁止为让聚合命令通过而改任一冻结 runtime config 或向主环境安装 DriveStudio CUDA 依赖。
  r013 config 将本条纳入 `failure_ledger_refs`，并须在 clean prereg commit 前保留两个解释器各自的 PASS 证据。
- `V51-F25`（`engineering`, `resolved before formal r013`）：从 Windows PowerShell 发出的第二次聚合 SSH 命令在
  双引号内包含 `$(find ...)`；本地 shell 在 SSH 前提前解释该命令替换，并把远端 `find -name` 片段误当成 PowerShell
  命令，最终本地报 `-name is not recognized`、远端 bash 报 unmatched quote。测试没有启动、GPU 未使用、仓库未修改、
  quality 未读。防重复门禁：跨 PowerShell→SSH 的命令不得嵌套未转义命令替换；本次改为独立执行测试、`git diff --check`
  与只读 `find`，不使用 shell substitution 控制流。不得把 launcher quoting 失败计入算法或测试 verdict。
- `V51-F26`（`engineering`, `resolved during r013 post-run audit`）：首次只读 r013 inspection 又把多语句 Python
  `-c` 嵌入 PowerShell 的双引号 SSH 字符串，本地 parser 在远端执行前把 Python 括号误解释为 PowerShell，报
  `An expression was expected after '('`；没有命令抵达远端、没有 artifact/repo/GPU/quality 状态变化。该问题与
  `V51-F25` 同属跨 shell 引号边界，但触发面是嵌套 Python source。合法修复为用 `apply_patch` 创建独立只读 auditor、
  `scp` 到精确 `/tmp` 路径后以固定参数执行；后续禁止在 PowerShell→SSH 双层命令中内嵌多语句 Python source。
- `V51-F27`（`engineering`, `resolved during r013 post-run audit`）：同步 auditor 与三份台账时，命令工作目录是
  local staging 根而三份 source path 误写成 `docs/...`，因此 auditor 已成功传到精确 `/tmp/audit_r013.py`，随后三个
  docs scp 在本地以 `stat local ... No such file` 失败；远端 repo 和 run 均未被部分修改。修复只把三份 source path
  写成 `motion_proj/docs/...` 并逐项同步；防重复要求多 source scp 前先按当前 workdir 解析 source，且不能把部分成功
  的前序传输误当成整条命令成功。
- `V51-F28`（`engineering/resource/protocol`, `resolved by v2/r015`）：formal H evaluation r014 已完整处理
  3 scenes、90/90 evidence/evaluation views并先持久化 3 scene reports 与完整只读 report，但 terminal resource gate
  观测 NVIDIA peak=`22,570 MiB > 22,528 MiB`、Torch reserved=`23,354 MiB > 22,528 MiB`，因此 status 必须保持
  `blocked`。cgroup peak=`14,305,161,216 bytes`，1,208 samples、0 monitor errors、duration=`897.647 s`，GPU 已释放；
  这推翻的是“r012 的 22 GiB uplift ceiling 足以覆盖双向 evaluation sparse projection”的资源外推，不是 H gate verdict。
  禁止读取 blocked r014 的 scene/aggregate quality、覆盖 run 或据其数值改 metric/pair/proxy/gate。合法恢复只允许新 v2/r015
  把 NVIDIA/Torch ceiling 同时提高到 `24,000 MiB`（仍低于 24,576 MiB 卡容量），其余 base config 逐字继承，并从
  r012/r010/r013 原冻结输入完整重跑。r014 status/resources/report/progress/resource-samples SHA=
  `6409545b...f6d1/ffc98a00...674e/510f82ec...227c/61475cb1...06ed/8fae05eb...7cf`；10 files，无 partial。
  v2/r015 从原 freeze 完整重跑，在同一 NVIDIA/Torch=`22,570/23,354 MiB` 下通过 `24,000 MiB` 门，故本条 resolved；
  r015 的 H verdict 由独立质量门决定，不能倒写 r014 为成功。
- `V51-F29`（`engineering/audit`, `resolved immediately`）：r014 blocked metadata 首次只读 hash 命令把实际
  `events.jsonl` 误写成 `events.json`，`sha256sum` 因该单项不存在返回非零，使后续 `&& find` 没有执行；前面其余 hash
  已正常输出，run/repo/quality 均未改变。修正为冻结 schema 的 `events.jsonl` 后 SHA=`31fff013...a7ce`，并完成 10-file
  bytes inventory。后续 verifier 必须从 runner/schema 读取精确 artifact 名，不凭相邻 runner 猜扩展名。
- `V51-F30`（`engineering/audit`, `resolved during r015 closeout`）：独立 auditor 首先要求 blocked r014 与 recovery r015
  report 在删除 `seconds` 后逐 Python float exact，得到 assertion failure。递归定位显示差异均为并行 CPU sparse/BLAS reduction
  的末位浮点扰动；离散字段、denominator、checkpoint、gate verdict 全 exact。修正审计合同为离散字段 exact、float
  absolute tolerance=`1e-12`，共 241 个差异，最大仅 `4.9760e-13`，复核 PASS。禁止把非 bit-exact reduction 误写成方法
  不可复现，也不得用宽松相对容差掩盖 gate 翻转；任何离散/gate 差异或 float 超过 `1e-12` 仍须 fail。
- `V51-F31`（`algorithm/evaluation`, `active rejected-route prevention`）：canonical r015 的 H gate 为 rejected。3 scenes
  中仅 0471/0379 evaluable，B1 actor-background margins=`-0.121280/-0.098618`，正场景=`0/2`，scene-balanced=
  `-0.109949<0`；1087 按冻结 32-Gaussian rule 无 eligible active actor，必须 abstain。Rigid coverage mean=`0.842910`
  与 heldout reprojection `B1-B0=+0.022777` 均过门，说明 normalized transpose 能改善 2D feature reconstruction，
  但不能证明 actor 内 feature 比最近 Background 更紧致；0471/0379 的 B0 margin 也已为负，且 B1 没有救回该前提。
  这推翻 LUDVIG uplift 可直接支撑 driving Gaussian semantic graph 的核心假设，因此按预注册同时 reject raw LUDVIG graph，
  不得降低 actor minimum、删除 1087、只报 reprojection 或先加 Bayesian/SAM/motion edge 救 graph。下一合法路线是独立
  faithful progressive propagation；S/C/validation/test/KITTI 仍未读。
- `V51-F32`（`engineering/launcher`, `resolved before D0 preregistration`）：首次 D0 只读 inventory 把含
  `U2|B3|Bayesian|...` 的正则直接嵌入 PowerShell→SSH 命令，外层 shell 抢先解释 `|`，导致远端 `sed` address
  截断并把各 regex 分支误当命令；没有文件、run、GPU 或 quality 状态变化。后续统一把多行只读脚本 UTF-8 base64
  编码后交给远端 `bash`，避免跨 shell parser 改写。不得把 launcher quoting failure 计入 D0 方法 verdict。
- `V51-F33`（`engineering/runtime`, `resolved before D0 preregistration`）：第二次 artifact inventory 在远端使用裸
  `python`，但主机 PATH 按合同没有该命令；三个 YAML path 已被 `rg` 只读打印，内嵌解析均未运行，也未改状态。
  修正为显式 `/root/autodl-tmp/envs/motionproj/bin/python` 后完成 NPZ identity/count/quantile 审计。后续所有 V5.1
  runner/auditor 必须使用冻结解释器绝对路径，禁止把 shell PATH 差异写成数据或算法失败。2026-08-20 收尾清理预审
  首次又假设 `/usr/bin/python3` 存在并在任何 inventory 写入前失败；随后显式使用 `/root/miniconda3/bin/python` 运行同一
  审计，候选集合未变化。该复发没有研究资产状态变化，进一步要求一次性维护的 cleanup 工具也必须绑定已探测的绝对解释器。
- `V51-F34`（`engineering/runtime`, `resolved before D0 preregistration`）：D0 扩大回归时又用 motionproj Python
  调用了依赖 DriveStudio runtime identity 的 H evaluation config test，得到该项 runtime drift；D0 新测试本身已
  `4/4 PASS`，没有 formal run、GPU 或 quality read。这是 `V51-F24` 的重复触发，说明仅在文档记环境分层不足。
  r020 freeze 的扩大回归再次用 motionproj 聚合 26 个 V5.1 test files，结果 `95 passed / 3 runtime-drift failed`；失败项
  正是三个已冻结为 DriveStudio Python 的旧 H tests，本次 E0a 定向 8 项均 PASS。按 suite 拆分后再用 frozen DriveStudio
  interpreter 补跑 3 项通过。后续回归命令必须在生成 file list 时就按 runtime 分组并分别记录结果；不得改 runtime
  freeze 来迁就聚合命令，也不得把这一环境错误写成算法 regression。
- `V51-F35`（`engineering/protocol`, `resolved before D0 preregistration`）：同一扩大回归发现 P0 scope 与 Stage-B
  authorization 仍保留最初 top-plan SHA=`3d7f7481...`，而 commit `b359541` 为预注册 H heldout contract 对计划做了
  17 行 append-only 更新，当前 SHA=`b4888476...`；旧 validator 只允许单 hash，导致 4 个既有 protocol tests 在真实
  route assertion 前 fail。禁止改写两份历史 freeze 的 recorded SHA。修复在 validator 中显式固定
  `base hash → authorized append hash` 两状态链，第三种状态仍 fail-closed，并增加 current/historical 双 hash 回归。
- `V51-F36`（`engineering/test`, `resolved before formal D0 operator run`）：首个 progressive expansion unit fixture
  预期 `p=0.5` 节点最终 UNKNOWN，却把该节点直接连到 `p=0.01` Background seed。两者 L2-normalized binary
  distribution cosine 约 `0.714`，高于冻结最低 threshold=`0.5`，故算子把它合法扩张为 Background；失败的是 fixture
  的“无支持”假设，不是算法。修复只删除这条边，使该节点真正孤立；其余阈值/公式/实现不变，5/5 operator tests PASS。
  禁止为满足错误预期而提高阈值、加入 confidence gate 或改变 UNKNOWN 语义。
- `V51-F37`（`algorithm/evaluation`, `active; D0 rejected by r018`）：faithful SAI3D-style raw-Gaussian progressive
  propagation 在 frozen H matched 12 views 上只通过 BF1 两项门：positive scene=`2/3`、scene-balanced BF1=
  `+0.0002196`；IoU=`-0.0714543<0` 与 FN semantic mass=`+0.1694766>+0.02` 同时 FAIL。0471 的
  BF1/IoU 有改善，但 FN 仍 `+0.080830`；1087/0379 的 IoU=`-0.159417/-0.220397`、FN=
  `+0.218146/+0.209454`，说明减少 FP/提高部分 calibration 并不能补偿 actor 漏检和跨场不稳定。这推翻“在 raw
  Gaussian 上按冻结 KNN 与多视图 SAM affinity 做 progressive growing 可稳定优于 U2/B3”的假设，不推翻所有 graph
  或 super-primitive 路线。D1 永久跳过；禁止按 r018 调 thresholds/hops/seeds/affinity 或重读 H。合法后续只有按冻结
  顺序进入 Stage E，先以 no-quality E0 检查 node elevation 是否提高 observation density，再按其门禁决定 E1/E2。
  证据：r018，source=`2cd98b3`，summary/manifest=`b08c7276...62d6/792660e3...010c`，independent metric/gate
  replay=`18c12f4d...0d2`，freeze=`configs/worldsim_v51/stage_d_progressive_h_evaluation_freeze_v1.yaml`。
- `V51-F38`（`engineering/shell`, `resolved during r019 monitoring`）：旁路进度查询再次在 PowerShell→SSH 边界使用
  `$run`，本地 shell 先展开变量并破坏远端引号，得到 `unexpected EOF`；正式 r019 进程独立运行、未受影响，也没有
  repo/run 写入。后续监控只能使用绝对字面路径或仓库内 CLI，禁止跨 shell 传未编码变量。这是 `V51-F32` 的复发，
  说明“已知坑”仍需由可执行入口而非记忆约束。
- `V51-F39`（`engineering/data-contract`, `resolved by v2/r020`）：formal r019 在 0471/1087 完成后，
  因 0379 frozen KNN 含 `34/7,123,746` zero-length edges 而 blocked；v1 把“用于 voxel scale 的 edge length”错误
  写成全量严格正值。三场 nonfinite edge 均为 0，0379 仍有 `7,123,712` positive edges，因此这是 quantile 输入合同
  过强，不是算法质量、OOM 或 corrupted geometry。r019 terminal/13 files 保留，partial assignments 禁止晋级/复用。
  合法 recovery v2 只排除零长 edge 的 scale statistic，保留全部 Gaussian，其他 quantiles/gate/views/locks byte-semantic
  继承，并以新 r020 完整重跑；r020 三场/九档 assignment、metrics 与 gate 独立复算 exact，report=`8df03b2a...5d34`。
  防重复边界仍成立：不得把 zero edge 直接删除出后续 topology，也不得借 recovery 改 voxel level。
- `V51-F40`（`engineering/shell`, `resolved with enforced command boundary during r020 freeze`）：量化 F39 时首次 base64 远端 Python 命令仍错误嵌入双引号，
  PowerShell 把 `base64.b64decode(...)` 当作本地命令，远端再次未执行。修复不是继续堆转义，而是新增可测试的只读
  `scripts/audit_worldsim_v51_e0a_edges.py`；CLI test PASS，三场 edge identity/zero/nonfinite/positive quantile 审计完成，
  report=`30493d5d...bc5`。r020 freeze 时又误用一次跨 PowerShell/SSH inline `python -c`，只产生远端 SyntaxError、未写入
  run；随即改为本地解析 YAML、远端只运行仓库 CLI/pytest。后续需要多语句远端分析时必须先落地仓库内 auditor，
  禁止临时内嵌脚本；单语句也不得跨两层 shell 手写嵌套引号。r022 审计前的旁路摘要查询再次因 heredoc 嵌套引号
  得到 `unexpected EOF`，随后发现远端没有 `jq`；两者均未写 run。改为 scp 冻结 JSON 后在本地只读解析，并由仓库
  auditor 完成正式审计。进入 Stage F 后又有一次含 `$f` 的远端 loop 被 PowerShell 提前展开，以及一次嵌套
  `python -c` 验证命令 SyntaxError；均在 formal r023 前、无 run/asset/repo 状态变化。该复发进一步说明远端临时解析
  不是证据入口；正式 source audit 必须由仓库 runner 完成。r025 freeze 上传后的 YAML smoke 又因 PowerShell 中手写
  `python -c` 反斜杠转义得到 `unterminated string literal`，紧接着尝试 stdin 单层命令仍被本地 quote stripping 破坏；
  两次测试链都在该点停止且未修改 run/repo 文件。最终不再修补 shell quoting，改为仓库 pytest 直接加载 freeze YAML，
  并配合 `git diff --check` 复核。该 recurrence 不影响此前已 PASS 的 r025 独立 auditor。2026-08-20 cleanup inventory 又有
  两次 inline `awk`/shell-loop 因 PowerShell→SSH quoting 失败；两次均为只读、没有创建/修改/删除研究资产。最终把审计和
  fail-closed deletion 放入固定 Python 文件，以 exact JSON plan 执行。禁止再为临时汇总跨两层 shell 拼循环、变量或 awk。
- `V51-F41`（`engineering/environment`, `resolved during r020 audit`）：本地 `autodl-stage/motion_proj` 只是按约束用于
  `apply_patch` 的 partial staging tree，不包含完整 `motion_proj.worldsim_v5` package；误在该目录收集 E0a 联合测试时触发
  `ModuleNotFoundError`。这不是 canonical repo、r020 或 auditor 失败。修复为只在本地做语法检查/编辑，将新增文件同步到
  远端完整 clean checkout 后运行同一测试，结果 `8 passed`；随后 r020 独立审计通过。后续不得把 partial staging 当作
  可运行 checkout，也不得为迎合该环境复制缺失 package 或修改 import path。r022 审计阶段在同一 staging tree 误跑
  `git diff --check`，因它不含 `.git` 只打印 usage；命令没有修改文件，正式 CLI test 与审计仍在远端完整 checkout PASS。
- `V51-F42`（`algorithm/evaluation`, `active; E0B rejected by r022`）：simple voxel super-primitive control 的
  `fine_q50 + member-unary mean + visibility-weighted SAM mean + max visibility + frozen D0 propagation` 在 frozen H matched
  12 views 上未能优于 U2/B3，也未能稳定优于 raw D0。相对 U2/B3 虽有 BF1 positive scenes=`2/3`，scene-balanced
  BF1=`-0.0002566`、IoU=`-0.0925468`、FN=`+0.1899473` 全部 FAIL；相对 D0 的 BF1 nonnegative scenes 仅
  `1/3`，mean BF1=`-0.0004762`、IoU=`-0.0210926`、FN=`+0.0204707`，四项机制门全 FAIL。0379 相对 D0
  `ΔIoU=-0.064618 / ΔFN=+0.067752`，说明确定性 voxel 合并与 member evidence 平均会扩散弱/错误证据，结构密度提升
  不能推出语义质量提升；1087 的近 no-op 也未形成可泛化收益。该结果只推翻这套 simple node-elevation 实例，不推翻
  faithful Gaussian Grouping 或所有 graph/node 方法。E1 PanoGS 与 E2 AG²aussian 按预注册停止，禁止依据 r022 调
  voxel level、node aggregation、seed/threshold/hop、删除 0379 或重读 H；下一合法路线是 Gaussian Grouping faithful
  source audit 与 no-quality preflight。证据：r022 summary/manifest=`4964a2f0...3d4/3c5a2fbe...7aa`，independent dual-gate
  replay=`5ced73db...104f`，freeze=`configs/worldsim_v51/stage_e_e0b_h_evaluation_freeze_v1.yaml`。
- `V51-F43`（`engineering/tooling`, `resolved before F0 preregistration`）：下载并哈希 Gaussian Grouping official PDF 后，
  远端没有 `pdfinfo/pdftotext`；桌面依赖清单给出的 Poppler override/fallback 也不可执行。改用 bundled Python 的
  `pdfplumber` 读 18 页，但首次输出受 Windows GBK 限制，遇到作者脚注符号触发 `UnicodeEncodeError`；只设置任务级
  `PYTHONIOENCODING=utf-8` 后完成全文提取，并用已存在的 `pypdfium2` 渲染方法第 6–8 页检查公式与图示。没有安装
  系统包、没有改 PDF、也未触及方法数据。后续 PDF source audit 优先复用 bundled Python 并显式 UTF-8，不假设远端或
  dependency catalog 中声明的 Poppler binary 实际存在；工具缺失不得写成论文或算法证据。
- `V51-F44`（`engineering/runner`, `resolved by r024`）：F0 source preflight runner 复用 Stage-B `_git`
  helper 时写成 `_git("rev-parse", "HEAD")`，但 helper 签名是 `_git(project, *args)`，实际调用变成
  `git -C rev-parse HEAD` 并在 source commit 读取处失败。r023 此时只创建 run 目录和 `resolved_config.yaml=7,796`
  bytes，尚未写 status、启动 resource monitor、读取 source/data/schema、运行 CUDA smoke 或读取任何 quality；它是不可晋级
  的 incomplete shell。合法 recovery 只新增 `repository_source_identity(PROJECT)` 显式绑定与参数顺序回归，用新 clean
  commit/r024 从头运行；r024 已越过 source identity 并执行到 adapter smoke 前，证明本项修复有效。不得删除 r023、
  手补 terminal 或借机改变 F0 source/method/data contract。
- `V51-F45`（`engineering/runtime`, `resolved by r025`）：r024 完成 official source identity、代码语义与三场
  train-only metadata/observation schema 的内存检查后，在 16D adapter smoke 前直接调用
  `torch.cuda.reset_peak_memory_stats(torch.device("cuda:0"))`；当前进程尚未初始化 CUDA context，PyTorch 返回
  `Invalid device argument 0: did you call init?`。r024 status=`blocked`，只有 resolved/status/events/resource samples
  四文件=`9,449 bytes`，没有 source report、CUDA render、SAM/DEVA/identity training 或 quality read。合法 recovery 只按
  `set_device → one-scalar allocation/context init → reset peak → smoke` 顺序执行并加入 call-order regression，新 r025 从头
  重跑；r025 与独立 replay 均得到 `[1,32,32,16]` render、`189` positive-alpha pixels、`48/48` identity gradients、
  base gradients absent，GPU peak=`310 MiB`，证明初始化顺序修复有效。禁止放宽 resource ceiling 或把已在 blocked run
  内存检查过的数据结果晋级；canonical 只认 r025。
- `V51-F46`（`data-contract/algorithm`, `active prerequisite after r025`）：Gaussian Grouping official identity mechanism
  需要 SAM everything masks 经 DEVA semionline 关联后的跨视图一致 short IDs；r025 独立核对三场各 15 个 train-only
  observation 后，45/45 只有 binary actor-union probability 等同一套 23 fields，没有任何
  `instance/identity/object_id/mask_id/class_id` label。`instances_info/frame_instances` 中的 stable actor token 只描述场景级
  track metadata，不提供每像素 mask；三个 formal checkpoint 也只能提供已训练 Gaussian state，不能反推出监督标签。
  同时 official DEVA propagation 与 SAM ViT-H weights 均 absent，现有 SAM2.1 Hiera Large 虽有 checkpoint，但不符合上游
  SAM-v1 everything-mode source contract。该缺口不否定 source core 或 frozen-base 16D adapter（两者 r025 PASS），但
  `current_training_input_ready=false /identity_training_authorized=false`。合法下一步只能先预注册并执行 train-only F0a
  asset acquisition + SAM/DEVA identity-mask materialization，冻结 URL/SHA、输出 schema、确定性、资源与 partial recovery；
  禁止用 metadata、binary U2/B3、SAM2 或 evaluation target 代替，禁止在 materialization 冻结前启动 F0 training。
  证据：r025 summary=`da4890d...988`、audit=`14d2b78b...8b64`、
  `configs/worldsim_v51/stage_f_f0_source_preflight_freeze_v1.yaml`。
- `V51-F47`（`engineering/orchestration/tooling`, `resolved during r026`）：首次启动 r026 时把远端 Linux command 与
  `/root/autodl-tmp/motion_proj` workdir 直接交给桌面本地 `bash` tool，Windows 在建立 SSH 前返回
  `CreateProcess ... 目录名称无效`；远端 run/path/assets 均尚不存在，因此这不是 r026 blocked terminal，修复为从本地
  PowerShell 显式 `ssh wm-3090-0811` 后按原 prereg run ID 启动。r026 完成后只读汇总文件字节时又假设远端存在 `bc`，
  在已打印 auditor hash 与 status/manifest size 后报 `bc: command not found`；它没有修改 run，完整字节数改由已审计的
  manifest inventory 加 status/manifest exact size 得到 `51,021`。这两次都推翻“tool shell/workdir 与常用 CLI 可跨本地/
  远端默认存在”的工程假设，不影响 r026 的 `done` 或 asset hashes。后续远端命令必须从 Windows 使用 SSH alias，run
  证据计算由仓库 runner/auditor 完成，禁止临时依赖未冻结的 `bc/jq/python -c`。证据：r026 audit=`5a360f42...817c`，
  freeze=`configs/worldsim_v51/stage_f_f0a_asset_source_acquisition_freeze_v1.yaml`。
- `V51-F48`（`engineering/data-contract`, `resolved before formal r027`）：r027 prereg config 首稿手录
  scene-0471/frame0/camera0 SHA 时，把 canonical `093d38e8d8d8f12...819e` 漏写一个 `d8` 成
  `093d38e8d8f12...819e`。formal-config pytest 在任何 r027 run 目录、wheel/env mutation、GPU/model/image decode 前
  fail-closed；远端 `sha256sum` 与 r026 已独立审计的 selected manifest 一致，证明是配置转录错误而非图像漂移。修复只从
  r026 canonical record 恢复完整 SHA，并以原测试重跑；不得改图、重选 view、放宽 hash 或把该失败写成 SAM/DEVA 结果。
  后续长 identity 必须由 manifest 机器传递并保留 config-validation test，禁止凭聊天摘要/截断 hash 手工补全。
- `V51-F49`（`engineering/license/runtime`, `resolved by r028/r029`）：r027 已 atomic 构建 isolated venv 并安装
  exact `supervision=0.14.0/PuLP=2.7.0/gurobipy=10.0.3`，但第一个 Gurobi tiny MILP 在创建 model 时报告
  `License expired 2024-10-28`，因此 status=`blocked`。失败发生在 one-view upstream CLI 前：没有加载 DEVA/SAM 权重、
  没有 GPU model forward、没有 decode input/mask、没有 quality；4 files=`12,861 bytes`，不得把它写成 identity mechanism
  或 association 失败。根因是把上游 `gurobipy>=10.0.3` 的最低版本误冻结成 exact 10.0.3，而该 wheel 的内置 restricted
  runtime 已过期；服务器没有另一个 `gurobi.lic` 可续用。合法 recovery v2 只将 Gurobi 提升到当前 index 可得的
  `12.0.3`（仍满足上游版本下界），使用全新 wheelhouse/venv/r028 重跑，其他 source/assets/input/CLI/resource/locks
  不变；仍要求 Gurobi tiny optimum，禁止直接跳过 gate 或静默采用 PuLP 后声称 faithful。r028/r029 的 Gurobi 12.0.3
  tiny model 均得到 status=`2`、solution=`1.0`，因此 license 前置已解除；后续 stdout 与 GPU failures 分别独立记账。
- `V51-F50`（`engineering/source-provenance`, `resolved before r028`）：r027 environment path verification 用子 Python
  import frozen DEVA source，默认 bytecode policy 在 Gaussian Grouping checkout 内生成 4 个未跟踪 `__pycache__` 目录；
  tracked diff 为 0，但后续 v2 config 的 clean-source gate 正确 fail-closed。该污染不是 upstream 修改、算法失败或 r028 run；
  在确认 exact paths 后只删除这 4 个由本任务生成的 cache，并让 runner 的全部子进程继承
  `PYTHONDONTWRITEBYTECODE=1`。第一次清理 wrapper 又因 PowerShell 提前处理 `$p` 而出现 quote EOF，未删除或修改任何
  文件；随后改用 4 个显式绝对 target 完成清理，两个 external repo 恢复 clean。禁止把 `git status` 放宽为忽略 untracked，
  也不得把 source tree 内 cache 纳入冻结；后续 import smoke 必须同时核对 commit/tree/clean。
- `V51-F51`（`engineering/runner`, `resolved by r029`）：v2/r028 成功获取并安装
  `gurobipy=12.0.3`，Gurobi 不再抛 license-expired/CalledProcessError；但 restricted-license banner 与 runner 自己打印的
  JSON 同时写 stdout，v2 对整段 `json.loads` 而得到 `JSONDecodeError line 1 column 1`。r028 因此在 solver output parse
  blocked，4 files=`10,642 bytes`；one-view CLI、DEVA/SAM load、input/mask decode 与 quality 仍全部未发生，不能把它写成
  solver、association 或算法失败。v3/r029 唯一修复是解析最后一个非空 stdout 行为 JSON，并把前置 banner 原样存档；
  solver status=`OPTIMAL` 与 solution=`1` 的门不放宽，环境/权重/view/CLI/resource/locks 全继承 v2。禁止粗暴丢弃全部
  stdout、用正则猜数字或因 parser failure 绕过 Gurobi gate。r029 已越过两种 solver gate 并启动 official CLI，证明
  terminal-line parser 恢复有效；r029 后续 OOM 另记 `V51-F52`，不得倒写本项未解决。
- `V51-F52`（`engineering/resource`, `active; allocator-only recovery disproved by r030`）：r029 首次完整越过 environment、Gurobi/PuLP 和
  official model-load gate，在唯一 scene-0471/frame0/camera0 输入上执行 SAM ViT-H everything mask；上游默认
  points-per-side/batch=`64/64`，在 `BatchMaskData.cat` 尝试再分配 `6.74 GiB` 时 CUDA OOM。错误现场为 GPU total/free=
  `23.56/6.72 GiB`、process=`16.83 GiB`、PyTorch allocated/reserved-unallocated=`10.74/5.77 GiB`，独立 resource samples
  peak=`17,246 MiB`。r029 status=`blocked`，6 files=`22,458 bytes`，没有 mask、`pred.json`、identity quality 或 cross-view
  association denominator；这推翻“24GB 可直接运行 official default one-view”的资源假设，不推翻 Gaussian Grouping 算法。
  第一合法 recovery/r030 只设置 allocator `max_split_size_mb:128`，source/view/size/IoU/grid/batch/ceilings 均不变；若仍
  OOM，必须保留 r030 后另行预注册 `SAM_NUM_POINTS_PER_BATCH` 单变量 batching adaptation，并补 parity/repeatability，禁止
  同轮缩图、改 grid/阈值或把工程失败写成 algorithm reject。r030 已证明该 allocator-only recovery 不足，后续事实续记
  `V51-F55`。证据：r029 stderr=`205266c9...403`、resource=`a0d825a8...01a`、status=`ff75badc...f66`。
- `V51-F53`（`engineering/source-provenance`, `resolved by r030 asset publish`）：r029 模型构造首次触发冻结 DEVA source
  `deva/model/resnet.py` 中的 `model_zoo.load_url`，隐式从 PyTorch URL 下载 ResNet50/18 到用户级 `/root/.cache`。两份
  资产分别为 `102,502,400 bytes /19c8e357...097` 与 `46,827,520 bytes /5c106cde...13f8`；此前 F0a asset freeze 未枚举
  这项 transitive dependency，故“官方 DEVA+SAM 两权重已经覆盖全部模型资产”的来源合同被推翻。合法 recovery 固定 source
  literal URL、bytes/full SHA，把 exact cache 原子复制到专用
  `/root/autodl-tmp/models/gaussian_grouping_v51_stage_f/torch_home/hub/checkpoints`，并让 subprocess 固定 `TORCH_HOME`；禁止
  继续依赖用户 cache、重新下载未验 hash、修改 frozen upstream 或把 ResNet 权重混称 DEVA checkpoint。canonical 目标独立
  审计前不得删除原 cache；审计后只允许精确清理由本次生成且 hash 匹配的两个源文件。r030 前置已把两份资产原子发布到
  dedicated target；独立复核 bytes/full SHA、无 `.partial`，且 official CLI stderr 没有 download 行，证明 `TORCH_HOME`
  生效。本项来源缺口已解除。r032 canonical audit PASS 后，两份原 cache 源副本已按 exact path 精确删除；dedicated
  `TORCH_HOME` copies full SHA 保持，后续不再依赖用户 cache 或网络。
- `V51-F54`（`engineering/orchestration`, `resolved before prereg commit`）：第一次提交 v4 时把包含括号与多段正文的
  `git commit -m` 放进 Windows PowerShell→SSH→bash 双层命令；本地层提前剥掉远端参数引号，bash 在 Conventional Commit
  标题的 `(` 前直接 syntax error。失败发生在 Git 创建 commit、push、r030 run 或 canonical asset publish 之前；staged diff
  保持不变，不是 config、test 或算法失败。这是 `V51-F47` 跨 shell 合同的复发：恢复改为在本地用 patch 生成独立 commit
  message 文件、scp 到远端临时路径，再以 `git commit -F` 单参数读取；禁止继续手工嵌套长 `-m`、省略正文或覆盖 staged
  内容。提交后必须精确删除临时 message，并重查 branch/status/commit。
- `V51-F55`（`engineering/resource`, `active; batch-only recovery disproved by r031`）：r030 在 source=`33c013d` 上保持同一 input、官方
  point grid/batch=`64/64` 与全部方法参数，仅增加 `max_split_size_mb:128`。reserved-unallocated 已由 r029 的 `5.77 GiB`
  降到 `578.72 MiB`，说明碎片明显减少，但同一 `BatchMaskData.cat` 仍尝试分配 `9.49 GiB`，free=`9.16 GiB` 而 OOM；
  sampled GPU peak=`24,098 MiB`，超过 prereg `24,000 MiB`，cgroup=`18,035,429,376 bytes`，101 samples/0 errors。
  6 files=`25,873 bytes`，没有 mask、metadata 或 quality。该结果推翻 allocator-only 足以让 official default batch 在 3090
  上运行的假设，也提示主要约束已不是 allocator fragmentation；仍不构成算法 reject。下一合法 recovery/r031 只把上游
  文档明确称为 parallel point prompts 的 `SAM_NUM_POINTS_PER_BATCH=64→32`，保留 points-per-side=`64`，并要求成功后另做
  batch parity/repeatability；禁止同轮改 grid、size、IoU、模型或资源 gate。证据：r030 stderr=`15d9bd12...bef`、resource=
  `39060722...6ad`、status=`d38fd753...0f7`。r031 已证明 batch32 仍不足，后续累计规模事实续记 `V51-F57`。
- `V51-F56`（`engineering/orchestration`, `resolved before r031 prereg`）：为核对 batch 参数来源所发的只读 `rg` 命令在
  Windows PowerShell→SSH→bash 双层字符串内包含 alternation `|`，引号被提前剥离后 bash 把后半段当命令，返回
  `points_per_batch: command not found`。没有 repo/run/asset 状态变化；恢复为单关键词、无 pipe 的 `rg`，定位到 DEVA
  `docs/DEMO.md`、`ext_eval_args.py` 和 `automatic_sam.py`：参数默认 64，定义为每批并行 point prompts，并直接传给 SAM
  `points_per_batch`。今后临时只读 SSH 查询也必须避开嵌套 alternation/pipe，复杂查询落到本地或正式 auditor。
- `V51-F57`（`engineering/resource`, `resolved for 24GB by r032 grid32; default-grid boundary remains`）：r031 精确执行 v5 的唯一变化
  `SAM_NUM_POINTS_PER_BATCH=64→32`，stdout 确认 side/batch=`64/32`，但仍在同一 `MaskData.cat` 累积点 OOM：request/free=
  `9.32/9.31 GiB`、allocated/reserved-unallocated=`13.34 GiB/599.11 MiB`，GPU peak=`24,066 MiB`、cgroup peak=
  `18,052,734,976 bytes`、119 samples/0 errors。6 files=`28,677 bytes`，mask/metadata/quality 均 absent。相对 r030 的
  `9.49 GiB` request 仅下降约 `0.17 GiB`，推翻“缩并行 batch 可解决最终累计 masks 峰值”的假设；继续 batch16 是重复
  调参，未授权。源码表明每批 full-resolution masks 在 NMS 前累积，规模主要受 points-per-side² 控制；DEVA 官方文档又明确
  建议降低 `SAM_NUM_POINTS_PER_SIDE` 来减少 automatic queries。下一合法 recovery/r032 只设 side=`32`（1024 prompts），
  batch=`32` 与其他参数/门禁不动；它必须标作 documented resource adaptation，成功后需同-grid batch parity、3-view
  association/repeatability 和后续质量门，禁止把 resource PASS 冒充 default-grid parity。证据：r031 stderr=
  `b822aab6...692`、resource=`0a06475d...af4`、status=`99d081ee...e23`。r032 以 side/batch=`32/32` 在 GPU peak
  `23,954 MiB` 内完成 output schema，解除当前 24GB execution prerequisite；但 default grid64 仍不可运行，grid32 quality/
  association 尚未证明，不能删除 r029–r031 或声称 exact-default parity。
- `V51-F58`（`engineering/orchestration`, `resolved after r032 audit`）：按 `V51-F53` 清理门禁删除两份用户 cache 前，首次
  wrapper 用嵌套 `$(sha256sum ... | cut -d " " -f1)`；PowerShell/SSH/bash 再次破坏 delimiter 引号，`cut` 在第一个
  `&&` 前退出，两份文件均未删除。已有 r032 independent audit 保存 source/canonical full SHA，恢复改为无 pipe/无命令替换
  的两个显式 `rm -f`，随后验证源路径 absent 且 canonical SHA 分别保持 `5c106cde...13f8/19c8e357...0097`。禁止在双层
  shell 中拼 checksum parser；以后先由 auditor 落证据，再用 exact path 单动作清理并独立验证。
- `V51-F59`（`protocol/data-contract`, `resolved by r033 association subgate; one-view boundary remains`）：r032 的 mask 是合法 `900×1600 uint8`，但 histogram=
  `{0:1,440,000}`。这不是 SAM grid32 quality reject：唯一输入少于 semionline `num_voting_frames=3`，upstream flush 没有形成
  cross-view consensus，因此 all-background 正是预先声明的 one-view 边界。它同时推翻“one-view resource PASS 可证明
  identity masks ready”的隐含推断。下一步必须在冻结 grid32 上做 same-grid batch parity，并用至少 3 个按时序排序的
  train-only views 检查 non-empty masks、stable short IDs、repeatability 与资源；在此之前禁止 full materialization、identity
  training 或把 annotation_count=1 当成实例覆盖。证据：r032 mask=`0bf854a1...59d`、audit=`cebe07fd...cd5`、freeze=
  `configs/worldsim_v51/stage_f_f0a_environment_one_view_smoke_freeze_v1.yaml`。r033 在同一 grid32/batch32 上用冻结的
  frame=`0/40/80` 完成 3-frame voting：三张 mask 全 non-empty，19 个 positive short IDs 至少跨 2 帧，且 batch32 repeat
  三 mask/metadata bit-exact，因此“one-view 不能证明 identity input”的前置边界已解除；该证据不读质量，也不解除 r033
  的 batch/resource 失败（`V51-F60/F61`）。
- `V51-F60`（`algorithm/implementation-contract`, `resolved for current method selection by r034; sensitivity boundary remains`）：r033 预注册把同 grid32 的
  `SAM_NUM_POINTS_PER_BATCH=32→16` 视为 execution-memory parity 臂，但 batch16 与 batch32 的三张短 ID mask 和 `pred.json`
  全不 exact。逐帧不同 label pixels=`208,647/288,527/244,696`，exact fraction=`0.855106/0.799634/0.830072`，binary
  foreground IoU=`0.961177/0.995201/0.969622`；batch32 IDs 含 `36/62/95`，batch16 含 `13/63/96`。与此同时
  batch32 primary↔repeat 的三张 mask 和 metadata 均 bit-exact，association/non-empty 也通过，故差异不能归因于无约束随机
  重跑。已确认的推翻项是“batch 只改变显存、不改变输出”；更深机制可能涉及 AMP batch-shape 数值路径与候选/NMS 边界，
  但 r033 没有证明具体源码根因。禁止放宽 exact 门、用高 foreground IoU 冒充 identity parity，或从 batch16/32 中按结果
  挑一臂。合法恢复只允许在新 run 恢复 upstream default batch64、保持 grid32/输入/阈值不变并独立检查 repeatability、
  association 与资源；若 batch64 不可运行，则该 faithful-input 路径必须保持 blocked 而不是继续调 batch。证据：r033 source=
  `191d3e4...12f`、parity=`7a6db15f...7ae`、audit=`a5a7d5c8...fa7d`。r034 恢复 upstream default batch64，保持
  grid32/三帧/其他参数，primary↔repeat 三 mask/metadata bit-exact、association PASS；当前方法选择因此固定 batch64，不再
  依赖被证伪的 batch32 execution-only 解释。本条的 batch-sensitive 事实仍是 active anti-regression boundary：后续任何
  batch 改动都是方法变化，必须新协议且不能用 r034 质量作等价担保。
- `V51-F61`（`engineering/resource/protocol`, `resolved by r034 physical-headroom contract; r033 remains failed`）：r033 三臂串行而非并发，independent audit 从 241 条
  resource samples 重算 NVIDIA peak=`24,116 MiB > 24,000 MiB`，超过预注册门 116 MiB；cgroup peak=
  `17,956,044,800 bytes`、event wall=`78.917s`、monitor errors=`0`。runner 先因 parity fail-closed，故没有执行后置 resource
  adjudication 或发布 `resources.json/summary/manifest`；这不能让已记录的 GPU 越门消失。它推翻“r032 one-view peak
  23,954 MiB 可直接外推到三视图三臂合同”的资源假设，不是 mask quality reject。禁止倒写 r033 为资源 PASS、事后把旧门
  提到 24,116，或把无 OOM 等同有安全余量。新 batch64 smoke 若修改 ceiling，必须在启动前绑定卡总显存、明确保留 headroom
  与复开理由；若 OOM/越新门则停止该 recovery，不得继续缩 batch 回到已证伪的 execution-only 解释。证据：r033 resource=
  `db4e6d17...8e9c`、status=`e027888e...7234`、audit=`22,939 bytes /a5a7d5c8...fa7d`。r034 没有修改 r033 门，
  而是新预注册 card total=`24,576 MiB`、minimum headroom=`256 MiB`、peak ceiling=`24,320 MiB`；upstream batch64 两臂
  实测 peak=`24,092 MiB`、headroom=`484 MiB`、cgroup=`17,957,322,752 bytes`、142 samples/0 errors，全门通过并由
  audit=`e0988f50...5258` 重放。因此当前三视图 upstream-batch64 resource prerequisite resolved；45-view materialization
  仍需独立总时长/磁盘/输出分母门，不能从两臂 smoke 外推。
- `V51-F62`（`engineering/CUDA-runtime/resource-boundary`, `resolved_recovery; root_cause_unproven`）：r035 依冻结顺序串行做三场 45-view train-only
  materialization；0471 已完成 `15 masks +pred.json`，但 1087 official grid32/upstream-batch64/AMP subprocess 处理前两张
  后，在第三张触发 three-frame vote，于 `consensus_associated.py:58 spatial_alignment` 的 `value @ affinity` 返回
  `CUDA CUBLAS_STATUS_INTERNAL_ERROR / cublasGemmStridedBatchedExFix`，因此 1087 没有 canonical mask/pred/report，0379
  未启动。该错误不是显式 PyTorch OOM；resource samples 重放 peak=`24,124/24,576 MiB`、headroom=`452 MiB`、cgroup=
  `17,961,271,296 bytes`、174 samples/0 errors，仍通过 r035 预注册数值门，所以现阶段既不能武断归因 OOM，也不能用
  headroom 数值排除 allocator/CUBLAS workspace/driver 异常。它推翻了“r034 同 batch64 三视图 PASS 可直接外推任意场景
  的 45-view execution stability”，但没有推翻 Gaussian Grouping identity 算法或证明 mask quality 失败。禁止把 0471 的
  `15/45` partial 写成 full materialization、原地续跑/覆盖 r035、跳过 1087、改场景顺序、缩 batch 回到已证明会改变输出
  的配置，或读取 partial quality 再选 recovery。合法下一步只能新预注册 exact 1087 `000_0/000_1/000_2` 三视图，保持
  grid32/batch64/AMP/size480/thresholds 并启用 `CUDA_LAUNCH_BLOCKING=1` 定位是否可重放；diagnostic 输出不得进入质量或
  training。证据：r035 source=`e4d64d3...1424`、status/events/resource-samples=`c3f917bd...f61/7d2221b5...0b7/
  d46d632d...4e2`、stderr=`f626efc6...8a5`、audit=`25,311 bytes /6d217a7e...13e1 /PASS`。r036 在 exact
  1087 首三视图、相同 method 且 `CUDA_LAUNCH_BLOCKING=1` 下串行 fresh-process replay 两次：第一次在相同 GEMM
  位置复现、第二次成功并生成 3 个 schema-valid mask/pred，故 deterministic input/shape 必现假设被推翻；与此同时一成一败
  证明 runtime 还不具备 materialization 所需的 repeatable execution。r036 resource 仍 PASS，成功输出未读质量且不能补写
  r035。下一步先做预注册 runtime health/control-vs-target reproducibility gate；禁止用第二次偶然成功直接重启 45-view。
  r036 evidence=`summary 32e59c85...3ea /audit 5,077 bytes, ec7cfa36...34f6 /freeze
  configs/worldsim_v51/stage_f_f0e_scene1087_cuda_fault_localization_freeze_v1.yaml`。r037 用同卡、同 frozen method、同
  `CUDA_LAUNCH_BLOCKING=1` 做 A–B–A–B：两次 0471 known-good control 均成功、mask/pred 彼此 exact 且与 r034 SHA
  相同；夹在其间的两次 1087 target 均在相同 CUBLAS site 失败。故“整张 GPU/所有三视图已普遍失效”不成立，失败收窄为
  target-path process instability；但 r036 曾有一次 target success，仍不能倒写 deterministic data failure。ECC/page/row 对
  RTX3090 为 N/A，dmesg 又无权限，二者都不能冒充健康证明。合法下一步是 source-neutral trace，在不改 upstream 文件/
  tensor 内容/方法参数的前提下记录 control/target matmul tensor metadata 与 allocator 状态；任何 trace 输出仍不得参与质量或
  training。r037 evidence=`summary 5fd4a4e8...df8 /audit 8,245 bytes, 2fb76f32...d50d /freeze
  configs/worldsim_v51/stage_f_f0f_cuda_runtime_health_reproducibility_freeze_v1.yaml`。r038 source-neutral trace 的 control/
  target 都成功且输出分别 exact 对齐 r034/r036 success；control 两个 matmul 是 `26/36 objects`，target 是 `3/52`，两侧
  affinity 都为 `[1,1620,1620]`，故“target 首个 matmul 更大所以必败”被直接推翻。更关键的 allocator observation 是：
  control pre-matmul driver-free 仅约 `35/55 MiB`、allocator retry=`0`；target success process 已发生一次 allocator retry，
  cache 被释放后 pre-matmul free 约 `18.15/17.23 GiB`。这使 allocator-cache/CUBLAS-workspace state 成为有证据的 active
  hypothesis，但不是根因证明：trace timing 可能扰动执行，且 control 在低 free 下仍成功。合法 recovery 只允许预注册在
  frozen line58 matmul 前执行 `torch.cuda.empty_cache()`，不改 tensor/operator/grid/batch/AMP，并要求 control/target 双 repeat
  对既有 success hashes bit-exact；禁止把 cache observation 写成 OOM 或跳过 parity 直接 full materialization。r038 evidence=
  `summary e9db6152...8f46 /audit 16,025 bytes, a8cbdb5b...4047 /freeze
  configs/worldsim_v51/stage_f_f0g_target_tensor_allocator_instrumentation_freeze_v1.yaml`。r039 在每个 frozen matmul 前调用
  `torch.cuda.empty_cache()`，A–B–A–B 四个 fresh process 全部成功；8 次 intervention 均有 before/after allocator 证据，
  control/target 双 repeat 与 r034、r036/r038 success hashes 全部 exact，且资源门 PASS。因此 empty-cache 是当前合法的
  execution recovery candidate，并推翻“释放 cache 必然改变 identity 输出”的担忧；但三视图 parity 不能外推 1087
  15-view/全 45-view，`V51-F62` 仍 active。下一步只允许单场 1087 15-view recovery，禁止直接把 r039 写成 full
  materialization ready。r039 evidence=`summary d720af4e...9505 /audit 8,625 bytes, fda57ee4...88ab /freeze
  configs/worldsim_v51/stage_f_f0h_pre_matmul_empty_cache_parity_freeze_v1.yaml`。r040 把同一 intervention 扩到 exact
  scene-1087 15-view：单 fresh process 完成 `15 uint8 900×1600 masks +pred.json`，6/6 observed matmul 均有
  empty-cache before/after 证据，resource gate 与独立审计全部 PASS；未读取 mask 内容质量。因此“该 recovery 只对三视图
  probe 有效、扩到 1087 15-view 必然失败”已被推翻，但它仍不能外推 fresh 三场 45-view，`V51-F62` 保持 active。
  下一步只允许预注册新目录、按 `0471→1087→0379` 串行的 45-view recovery；禁止续写 r035、复用 r035 partial、先读
  r040 quality，或直接放行 identity training。r040 evidence=`summary 312a0277...a65 /audit 9,254 bytes,
  1393c664...67c /freeze configs/worldsim_v51/stage_f_f0i_scene1087_15_view_empty_cache_recovery_freeze_v1.yaml`。
  r041 再从 exact r026 manifest 新建三组 scene-local 输入，按 `0471→1087→0379` 三个 fresh process 串行执行；三场
  全部成功，45/45 schema masks、3/3 pred、18/18 pre-matmul empty-cache evidence、output record chain 与资源门均由
  独立审计重放。因此 r035 暴露的 full-materialization execution failure 已在 frozen empty-cache intervention 下解除，
  `V51-F62` 改为 resolved recovery；但 trace timing/allocator-cache/CUBLAS workspace 中哪一个是唯一根因仍未证明，不能
  写成 OOM root cause。该结论也完全不包含 mask quality、actor identity alignment 或 training readiness；下一步必须新预
  注册质量/对齐门，禁止把“45 个文件存在”写成算法有效。r041 evidence=`summary f3ee3ad1...c183 /materialization
  32b5d8d3...1b7f /audit 18,462 bytes, acd5a91b...31d2 /freeze configs/worldsim_v51/stage_f_f0j_fresh_45_view_
  empty_cache_materialization_freeze_v1.yaml`。
- `V51-F63`（`algorithm/instance-quality-alignment`, `rejected`）：faithful Gaussian Grouping 的 45-view materialization 在
  F0l 首次读取 frozen train-only weak support 后，只通过 scene-1087，scene-0471/0379 均失败。0471 的 foreground
  coverage/one-to-one identity recall/persistent-track fraction=`0.122784/0.080747/0`，0379=`0.238278/0.202933/0`，分别远低于
  读像素前冻结的 `0.70/0.35/0.50`；两场 assignment efficiency=`0.937/1.0`，说明问题不是主要由 short-ID collision
  造成，而是 actor support 大量未覆盖且同一 3D track 的 assigned ID 没有跨两个 eligible views 持续。1087 的
  `0.859091/0.505009/0.5` 全门通过不能覆盖 all-three-scene contract。该负结论推翻“只要 full materialization 稳定，
  faithful DEVA short IDs 就足以作为当前 driving identity training 输入”；它不是 CUDA/资源 blocked，也不允许训练后再救。
  限制：DriveStudio dynamic union 只作 foreground weak support，3D projected boxes 只作 track attribution，不是真值 instance
  segmentation；因此结论严格限于当前三场、frame→camera view order 与 adapter，不能外推为 Gaussian Grouping 普遍无效。
  F1/F2/identity training 关闭，下一步按冻结路线进入 Trace3D source/method/immutable-base adapter preflight。证据：r043
  summary=`f13c8094...da8a`、report=`b1e4bb40...95ed`、audit=`4,210 bytes /f478fbd9...4320`、freeze=
  `configs/worldsim_v51/stage_f_f0l_train_only_quality_identity_alignment_freeze_v1.yaml`。
- `V51-F64`（`engineering/tool-availability`, `resolved`）：Trace3D G0 r044 已成功原子发布 official PDF 与 exact
  repo commit/tree，随后仅在读取 PDF 页数时因系统没有 `pdfinfo` executable 抛出 `FileNotFoundError`。异常发生在任何
  repository semantic report、source execution、submodule initialization、model download 或 image/mask/quality read 前；run
  只留下 running status 与 start event，必须保留且不能事后补成 blocked/done。该失败推翻“AutoDL 默认具备 Poppler CLI”
  的工程假设，不涉及 Trace3D 方法可行性。合法恢复只在新 run exact 复核现有 paper=`2,390,825 bytes /d50eda07...47e4`
  与 repo=`7465ad94...c442/tree 22d30d19...a05d/clean`，用标准库 PDF `/Type /Page` marker 计数替代 external tool；不得
  删除或重下已发布资产，也不得借恢复 init submodules/执行源码/读质量。closeout=
  `configs/worldsim_v51/stage_g_g0_trace3d_source_method_preflight_r044_closeout_v1.yaml`。r045 以原资产 exact reuse、标准库
  page-marker count=`11` 完成恢复；独立 audit `053cf574...d3b` PASS，故该工程失败关闭。r044 历史终态不倒写；本次修复不构成
  Trace3D 方法成功或训练授权。
- `V51-F65`（`engineering/algorithm-determinism`, `resolved_method_rejection`）：Trace3D exact unpatched CUDA extension 在 r046 的
  preregistered synthetic class-response gates 上 PASS，但相同 config/input/extension 的独立 fresh-process audit 将 foreground
  alpha weight 从 `0.0267562941` 重放为 `0.0056084292`（absolute diff=`0.0211478649`）；hard class vector 均为 `[0,1]`。
  official `id_trace.cu` 在多个 pixel threads 面向同一 per-Gaussian/class global weight 时使用普通 `+=` 而不是原子累加，
  这是与漂移相容的 source-level hazard，但一次跨进程差异尚不足以把根因写死。不得因 hard argmax 一致就进入真实 U2/B3
  adapter，也不得事后给 r046 增加失败门或直接 patch upstream。合法下一步是在全新预注册 run 中以多个 fresh processes 同时
  冻结 hard/alpha exact determinism；FAIL 时 faithful Trace3D operator rejected 并按路线转 BKI/graph-free，PASS 才允许预注册
  real tensor/camera adapter。r046 capability PASS 与本风险并存，且都不构成质量证据。
  r047 按上述规则执行 8 个 fresh processes、每进程 hard 两次与 alpha 两次；16 个 hard vectors 均为 `[0,1]`，但
  alpha exact vectors 为 `0.0056084292/0.0267562941` 两种，故唯一数 `2 > 1` 并被独立 audit `98c72ba7...d31`
  重算确认。该结果足以拒绝当前 exact unpatched faithful operator，但仍不把普通 `+=` 写成已证明的唯一根因，也不外推
  所有 Trace3D 实现。按预注册 failover 不 patch、不进入 real adapter，路线转 `WS-V51-M1-H-GRAPHFREE-01`。
- `V51-F66`（`algorithm/governance`, `superseded`）：Stage H 的 faithful BKI/graph-free fallback 在 V5.1 内没有启动，
  task status 保持 `pending`、execution=`false`，并由 V5.2 scope 取代；因此它既不是 done，也不是 empirical rejected。
  收口依据是累计而非新增质量读数：r018 的 progressive propagation 只有 ΔBoundary-F1=`+0.0002196`，同时
  ΔIoU=`-0.0714543`、ΔFN=`+0.1694766`；r022 的 simple voxel node 虽提高 observation density，相对 U2/B3 的
  BF1/IoU/FN 仍为 `-0.0002566/-0.0925468/+0.1899473`；r043 的 0471/0379 identity recall 仅
  `0.080747/0.202933` 且 persistence=`0/0`；r047 的 faithful Trace3D alpha 又不能跨 fresh process exact 重放。
  这些事实共同推翻“在不改变 evidence source 的情况下继续替换传播器仍有较高边际收益”的 V5.1 资源分配假设；它们
  支持当前瓶颈为 effective observation structural missing，但不证明 BKI 或所有空间 kernel 普遍无效。禁止在 V5.1
  继续做 BKI source preflight、kernel/threshold 调参，或把 `superseded` 改写成 BKI reject。合法复开只能在 V5.2 先引入
  独立新观测源，并在首次方法质量读取前冻结 coverage、identity persistence、fresh-process reproducibility 与跨场分母；
  证据=`configs/worldsim_v51/m1_closeout_v1.yaml`、`docs/archive/2026-08/worldsim-v51-m1-closeout/README.md`，
  authoring base=`fc07b99`，failure delta=`V51-F66`。

<a id="detail-v5"></a>

