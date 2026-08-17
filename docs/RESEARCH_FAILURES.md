# Motion-Proj 统一失败、风险与防重复账本

> **最后更新**：2026-08-17
> **唯一活跃失败事实源**：本文件 `docs/RESEARCH_FAILURES.md`
> **覆盖范围**：V1–V5.1、V7/V7.1、N1/cut-in 与跨路线工程/资源/协议教训
> **事实边界**：失败事实以 canonical run、`docs/EXPERIMENTS.md`、`docs/RESEARCH_STATUS.md` 和冻结证据为准

本文件是仓库中唯一持续维护的 failure ledger。`docs/archive/**/RESEARCH_FAILURES*.md` 只是对应 commit 的不可变
历史快照；`WS_*_FAILURE_FORENSICS.md` 是专项诊断报告，不是第二本账。新路线、新版本和新实验不得再创建并行的
`*_FAILURES.md` 事实源。

## 0. 使用合同与渐进式导航

### 0.1 渐进式读取

1. 每次研究任务先读本节和“V1–V5 版本总览”，确认当前问题属于算法、数据、评测、协议、工程、资源还是治理风险。
2. 根据 task、模块和关键词用 `rg` 定位 failure ID；只展开命中的完整条目及其相邻门禁，不默认一次加载全部长账本。
3. 新计划至少引用一个直接相关 failure ID；涉及跨路线复用时，再读取对应版本详细章和 canonical evidence。
4. 只有在做全局路线审计、迁移或报告附录时才通读全文件；归档快照仅在核对当时字节或历史编号时读取。

### 0.2 渐进式写入

1. 新坑先在当前版本详细章追加一个唯一 ID，再更新本文件顶部版本总览；不要复制整章或另建版本 failure 文档。
2. 旧结论被新证据解除时不得删除原条目；在原条目追加 `resolved/superseded`、新证据、仍然成立的边界和新 ID。
3. 写入只包含可复用的失败事实与防重复门禁；逐步日志、完整 stdout 和大表留在 run，实验结果表留在
   `EXPERIMENTS.md`。
4. 每个正式实验在启动前登记 `failure_ledger_refs`，收口时登记 `failure_ledger_delta`。任何 `blocked/rejected`、
   前提被推翻、工程恢复、门禁失败或旧风险解除，都必须在同一逻辑提交中更新本文件；若确无新增，只在实验台账写
   `failure_ledger_delta=none`，避免向失败账本灌入成功流水账。

### 0.3 单条记录最小 schema

| 字段 | 必填内容 |
|---|---|
| ID / 分类 | 唯一 `<路线>-FNN`；分类为 `algorithm/data/evaluation/protocol/engineering/resource/governance` 之一或组合 |
| 状态 | `active/resolved/superseded`；实验 task 状态仍只用 `pending/running/blocked/done/rejected` |
| 观察事实 | 分母、错误、指标、资源或 terminal；不把推断写成事实 |
| 根因与推翻项 | 已确认根因，以及它推翻了哪个假设、实现合同或旧结论 |
| 防重复与复开 | 禁止的事后调参/删分母/覆盖 run，以及合法复开需要的新证据 |
| 证据 | task/run ID、commit、summary/manifest/文档路径；工程失败与算法 reject 分开 |

### 0.4 目录

- [V1–V5.1 版本总览与 V1/V2 汇总](#1-v1v51-版本总览与-v1v2-汇总)
- [V5.1 详细账本](#detail-v51)
- [V5 详细账本](#detail-v5)
- [V4 详细账本](#detail-v4)
- [V3.3/V3.2/V3.1 详细账本](#detail-v3)
- [V2 继承门禁](#detail-v2)
- [V7/V7.1、N1/cut-in 与历史路线](#detail-legacy)
- [跨路线原则与新实验检查表](#detail-cross-route)

## 1. V1–V5.1 版本总览与 V1/V2 汇总

| 版本 | 终态/核心推翻 | 主要工程坑 | 详细证据入口 |
|---|---|---|---|
| V1 | AD-GS 六场景复现成立，但 persistent identity 不存在；唯一候选与已有工作直接重合，M7 rejected，M8/M9 未授权 | DGGT `pointops2` PEP 517 隔离缺 torch；训练完成、评估依赖和容器生命周期必须分开 | 下方 `V1-F01`–`V1-F06`、`PIVOT-F14B/F15/F16`、V1 frozen archive |
| V2 | M0–M4 闭环；M5 三场景压力测试只完成部分资产，不能写成路线完成；局部保持不等于删除后背景真实 | 空 shell PATH、CUDA/headers/包版本、devkit schema、SM 8.6、render 累积内存、子进程回收、空 actor slice | 下方 `V2-F01`–`V2-F09`、V2 继承门禁、`EXPERIMENTS.md` V2 注册表 |
| V3/V3.1 | A1 保持 off；A2 是 boundary/全局/成本 tradeoff；A3 局部精修不晋级；P1 pruning rejected；P2/P3 只支持存储/资产拆分 claim | 相机标签、随机 CUDA 初始化、分辨率三层合同、runtime state key、资源 ceiling、前馈前置条件 | `V3-F01`–`V3-F25` |
| V3.2/V3.3 | S4 temporal 未完成、S5 语义生产链回退；RoadPatch/asset/release 只在冻结场景和协议成立，不构成跨场景 dominance | frozen base identity、empty-target、模型/视图可用性、类型/枚举严格比较、确定性 archive | `V32-*`、`V33-*` 详细章 |
| V4 | M1 scene-disjoint validation rejected；M2 selective routing 成立但 geometry MAE 退化 `+3.3908 m`；M3 仅在冻结 18-scene exact-once test confirmed | cohort 非确定性、split leak、SSH 断管、解释器分层、CUDA arch、immutable run/staging、完整 denominator | live canonical `V4-F01`–`V4-F49` |
| V5 | M1/M2/M3 全部 rejected；structured graph 不稳定、无 absolute geometry-safe candidate、constraint projection 信号不足 | KITTI calib/OXTS 语义、缺 LiDAR 帧、provenance enum、launcher 原子目录、heading metric 和 long-run stdout | `V5-F01`–`V5-F59` |
| V5.1 | M1-only 正在推进；Stage A 冻结 U2/B3；Stage B official ViT-g resource/shape 门通过 | H→S 小效应未复现、UNKNOWN coverage 不达门；PCA/proxy leakage 仍 active，24GB/下载/fixture 风险已解除 | `V51-F01`–`V51-F17` |

### 1.1 V1 汇总条目

- `V1-F01`（`algorithm/evaluation`, `active`）：AD-GS 六场景 exact reproduction 只证明 frozen baseline 可复现，
  不证明对象级编辑、未知背景恢复或新方法成立。证据为 V1 M4 aggregate；禁止把复现分数重命名为贡献。
- `V1-F02`（`engineering`, `resolved in V2`）：DGGT V1 在 input staging 前被 `pointops2` 的 PEP 517 隔离构建
  缺少 torch 阻塞，没有质量/速度数字。V2 用固定 compiler/runtime/headers 和 upstream 非隔离安装解除工程前置；
  V1 terminal 仍保持 blocked，详见 `PIVOT-F14B`。
- `V1-F03`（`algorithm/data`, `active`）：六场景 pseudo ID 最长支持仅 `1/6/1/1/2/1` 帧，checkpoint 只有
  二值 `obj`，`0/12` object slots 可评；`persistent_object_identity_unavailable` 不能靠事后几何关联回填。
- `V1-F04`（`algorithm/governance`, `active`）：候选“恢复持久身份并绑定 actor 后做轨迹编辑”与 InstDrive、Director、
  OmniRe、HorizonForge、G²Editor 直接重合；适配 AD-GS 是工程，不足以通过 novelty gate。M7 保持 rejected。
- `V1-F05`（`protocol/evaluation`, `active`）：M7 拒绝后 M8/M9 未授权，0 seeds、0 proposed metrics、human verdict
  为 `null`。禁止事后补 endpoint、把 0 coverage 写成提升或由 Codex 代填人评。
- `V1-F06`（`data/governance`, `active`）：早期 cut-in 路线没有官方召回率分母，strict-v2 在 675 scenes 仅
  `1 PASS / 1 scene`；这说明当前可验证事件池过稀，不说明 nuScenes 没有 cut-in。cut-in 只能作可选演示，不能再
  承担主数据入口或论文成立条件。

V1 canonical 状态、实验和专项报告保存在 `docs/archive/2026-07/dynamic-reconstruction-v1/`；其中的
`RESEARCH_FAILURES.md` 是冻结快照，不再单独维护。

### 1.2 V2 汇总条目

- `V2-F01`（`engineering`, `resolved`）：非登录 shell 中裸 `python` 不在 PATH；runner 必须显式绑定解释器，
  不运行 `conda init`，也不能把 PATH 错误写成网络/依赖失败。
- `V2-F02`（`engineering`, `resolved`）：DGGT r1–r7 依次暴露 pip backtracking、CUDA compiler/runtime mismatch、
  cusparse headers、transformers/diffusers/torch schema、`flow_vis` 和 retry schema；每次修复必须新 run，native 已完成
  的阶段不被后续 common-eval blocked 覆盖。
- `V2-F03`（`data/protocol`, `resolved`）：磁盘 `sample.json` 没有 devkit runtime `anns` 反向索引，Decimal、
  invalid projection schema 和 nearest-sweep 也会破坏 exact mapping；最终必须以 exact `sample_token` 为主键。
- `V2-F04`（`resource/protocol`, `active`）：30k checkpoint 完成不等于累积 full render 完成；后者在 577/588 时
  越过 90% cgroup 合同并安全停止。训练与 post-render 必须分开裁决，不得删 checkpoint 或写成 OOM/方法失败。
- `V2-F05`（`engineering`, `resolved`）：CUDA 扩展 import 成功不代表包含 RTX 3090 SM 8.6 kernel；必须从冻结源码
  按目标 arch 重编并运行真实 forward/backward，不能只做 import smoke。
- `V2-F06`（`data/evaluation`, `active`）：训练可把非目标 actor Gaussian slice 裁为空；registry 必须显式 unavailable，
  选定 actor 必须非空，禁止静默删 denominator 或把空 slice 当成功删除。
- `V2-F07`（`engineering/resource`, `resolved`）：外层 timeout 不会回收 `start_new_session=True` 的子进程；必须按
  精确 PGID 清理并保留 interrupted terminal，长任务使用 detached controller 和独立日志。
- `V2-F08`（`algorithm/protocol`, `active`）：M5 只完成 0230/0242 checkpoint 与 0255 诊断；三场景×两 actor×四编辑、
  pseudo-hole/perception/final matrix 未完成。空 tensor `torch.cat` 是工程阻塞，不是 3DGS 方法失败，也不允许把部分资产
  写成 M5 done。
- `V2-F09`（`evaluation/algorithm`, `active`）：lateral/delete non-target PSNR 93/95 dB 主要是硬局部保持构造，
  不能证明 source footprint 后背景、边界或时序真实；后续必须把 outside preservation 与 hole/depth/boundary/temporal
  指标分开。

### 1.3 V4 历史编号冲突校正

2026-08-17 统一账本时发现，旧追加段落重复使用了 `V4-F30`–`V4-F33`。为保证后续引用唯一，本文件将 live canonical
编号校正如下；archive 快照保持原字节和旧编号，不回写：

- B0/D0 段的 `V4-F17`–`V4-F33` 保持不变；
- 历史 M1 development/validation `V4-F30`–`V4-F34` → live `V4-F34`–`V4-F38`；
- 历史 M1 rejection/M2 validation `V4-F35`–`V4-F39` → live `V4-F39`–`V4-F43`；
- 历史 M3 `V4-F40`–`V4-F45` → live `V4-F44`–`V4-F49`。

新文档、代码和 run manifest 只引用 live canonical ID；核对旧 commit/归档时同时记录“historical ID → live ID”。

<a id="detail-v51"></a>

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
- `V51-F14`（`engineering/protocol`, `active pre-quality risk`）：LUDVIG DINO extractor 的 PCA 路径不是天然确定性合同。
  `PCA(n_components=40)` 没有 `random_state`，大矩阵会走 randomized solver；当 patch 数超过 500,000 时还用未设 seed
  的 `np.random.choice` subsample。更隐蔽的是 GPU path 用 PyTorch `std`（默认 correction=1），CPU path 用 NumPy
  `std`（correction=0），所以为省显存切到 CPU 会改变标准化与全部 feature。V5.1 proposal 冻结 H evidence=
  `45 views×7,296 patches=328,320`，明确不触发 subsample；固定 std correction=1、randomized PCA
  random_state=`20260814`、40-D、whiten=false，并把 scaler/PCA state 持久化后只 transform S/C。不得把 solver/seed/std
  差异当作 backbone 增益或在 S/C refit；这是 reproducibility hardening，不是参数搜索。本轮未下载模型、提取 feature 或
  读取质量。证据：LUDVIG `predictors/dino.py`、`configs/worldsim_v51/stage_b_freeze_proposal_v1.yaml`、
  `docs/WS_V51_STAGE_B_FREEZE_PROPOSAL.md`。
- `V51-F15`（`evaluation/governance`, `active pre-quality risk`）：Stage B 的 same-actor/actor-background metric 可从
  frozen `RigidNodes.points_ids[:,0]` 与 Background row 构造，但这是 base-model membership proxy，不是真实 ownership GT。
  若把该 proxy 输入 DINO/PCA/uplift/权重会形成标签泄漏；若只凭 proxy margin 解锁 Graph，则会把模型自身表示循环证明为
  语义正确。proposal 将其限制为 evaluation-only stratum，强制写
  `model_membership_proxy_not_ground_truth`，并同时报告不消费 membership 的 same-Gaussian repeatability 与 heldout DINO
  reprojection。无 eligible actor 的 scene 必须保留 abstain；不得降低 32-Gaussian eligibility、删 1087/0379 或只报大
  Rigid 场景。Stage B 未获授权，本轮没有产生 metric。证据：V5 formal30k r027–r034 metadata、
  `configs/worldsim_v51/stage_b_freeze_proposal_v1.yaml`、`docs/WS_V51_STAGE_B_FREEZE_PROPOSAL.md`。
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
  `scripts/audit_worldsim_v51_stage_b_operator_parity.py` 的 pre-formal regression。

<a id="detail-v5"></a>

## V5 M3 constraint projection 新增防重复结论（2026-08-14）

- `V5-F52`：V4 M3 canonical r238/r335 的 baseline 是 `FRAME_INDEPENDENT`，不是 V5 要求的 `T2_V4_FROZEN_SE3_BSPLINE`。V4 的 `30.41%/34.39%` warp 改善不能直接成为 V5 T2 comparator 证据；V5 必须在同一 fresh clip 上重跑 T2–T5。
- `V5-F53`：M2 rejected 后，M3 REMOVE 不能隐式复用 geometry-safe repair。REMOVE 只保留 exact bypass、semantic reintroduction 与 rollback checks，不进入 trajectory physics denominator；M3 正结果永远不得倒写 M2 成功。
- `V5-F54`：r002 是 config identity blocked，不是 clip inventory 质量失败。缺少 `protocol_audit.conclusion` 在 annotation streaming 前触发 KeyError；无图片/LiDAR blob/quality/GPU 读数。terminal 保留，只以新 r003 修复。
- `V5-F55`：低速位置抖动不能定义 velocity heading，倒车也不是 heading inconsistency。r004 的 T2 38 项违例全部来自旧 heading metric；禁止用 speed>`0.1m/s` 且 forward-only 的结果宣称 projection 有大幅物理改善。当前修正规则是 speed>`1m/s` 且 forward/reverse mismatch 取小者。
- `V5-F56`：POCS 更新量固定点不等于物理可行。r004 T5 的逐帧 heading correction 产生 `20` yaw-rate + `14` heading violations，却曾报告 converged；现在只有剩余 total violations=`0` 才允许 convergence。禁止用旧 converged flag 支持方法。
- `V5-F57`：r005 exact replay 后 T2=`15/16 safe`，仅 `1/16` request 有 `2` 项违例；T5 降到 `1` 项，但预注册最小 evaluable 是 `8`。当前结论必须是 insufficient signal，不能因相对 reduction=`50%` 而解锁 renderer、选择 arm 或进入 validation。
- `V5-F58`：不得通过降低 heading speed floor、取消 reverse、降低物理 caps、删除 T2-safe requests 或改 minimum-evaluable gate 来“修复”r005。复开只能注册新的 desired-motion hypothesis 与新 run，明确保留 result-aware development 身份；collision/render/validation 仍需独立门。
- `V5-F59`：r006 已把 V5 constraint-projection M3 正式收口为 `rejected`。禁止事后扩大 lane shift/acceleration stress template 来人为制造 T2 violations；V4 M3 时序正结果继续成立，但只能写作历史 baseline，不能倒写 V5 constraint projection 成功。未来复开必须是独立新路线与新冻结协议。

## V5 M2 cross-view scaffold 与拒绝收口新增防重复结论（2026-08-14）

- `V5-F46`：repair asset provenance 必须来自已有不可变枚举。r012 使用新字符串 `cross_view_background_depth_scaffold`，在首个 asset、GPU 与方法质量读取前被拒绝；该 terminal 是工程 blocked，不是 G4 质量结果。修复只改为既有 `native_scene_donor` 并新建 r013，不覆盖 r012。
- `V5-F47`：G4 的 Gaussianization 后改善 `17/22` 不能替代 raw 相对门。r013 raw 仅 `12/22`，低于冻结的 `14/22`，且 raw/post absolute-safe 均为 `0/22`；G4 必须保持 rejected。其 direct projection mean/median 仅约 `4.73%/0.78%`，不得靠 Gaussianization 或 fallback 隐藏覆盖不足。
- `V5-F48`：G5 的相对支持不能覆盖绝对几何安全失败。r014 raw/post 改善=`15/22`、`19/22`，mean delta=`-3.270320/-4.023966m`，但 raw/post absolute-safe 仅 `1/22`、`0/22`；正式写法只能是“model proxy 上相对改善，未形成 geometry-safe candidate”，不得写成 M2 成功或跨视角恢复真实背景。
- `V5-F49`：多相机投影覆盖不是独立真值。r014 any/direct/extrapolation/fallback mean=`60.40%/15.57%/47.99%/36.44%`，LiDAR projected mean≈`0.8%`；大量信息来自 bounded extrapolation 与 G0 fallback。禁止把 source 数量、投影覆盖率或 proxy MAE 当作 same-view hidden-background GT confidence。
- `V5-F50`：G5 绝对门失败触发结果前 hard stop。禁止事后搜索 absolute threshold、camera/time source grid、fusion、disagreement、extrapolation radius、Gaussian stride/opacity，亦不得自动解锁神经 surface；任何复开必须是新科研假设、新 task、新冻结协议与独立 evidence source。
- `V5-F51`：r015 已把 `WS-V5-M2-GEOMETRY-FIRST-REPAIR-01` 正式收口为 `rejected`，method/router/validation 均未解锁。后续 M3 是独立任务，任何 M3 正结果不得倒写 M2 成功；`WS-V5-M2-GEOMETRY-FEASIBLE-ROUTER-01` 保持 locked，不能在没有 geometry-safe candidate 时单独运行。

## V5 M2 geometry-first 新增防重复结论（2026-08-14）

- `V5-F34`：同一 view 的 actor-union mask 不是合法的一次修复 request。r002/r003 把多个 actor 合成一个 hole，最大 union 达 `152,410` pixels，导致大洞支配均值；r004 已恢复 `one_actor_one_view_one_hole`，得到 `23=22 accepted+1 rejected` 且 union pixel-exact replay。后续方法选择不得复用 union-mask 分母。
- `V5-F35`：base renderer `background_depth` 是维护一致性 proxy，不是 same-view hidden-background GT。r005 的 reference confidence mean/median 只有 `0.0585/0.0582`，范围 `0–0.1557`；任何 raw/post MAE 必须与 `model_proxy_not_ground_truth` 同表，不得写成真实道路深度恢复。
- `V5-F36`：G0 在逐 actor r005 上 raw absolute fail=`22/22`，raw MAE mean/median=`8.5872/8.7151m`；同时 Gaussianization primary=`16/22`。不得因为 G0 是最简单 arm 或相对复杂 surface 较稳，就把它写成 geometry-safe candidate。
- `V5-F37`：G1 piecewise plane 只有 `5/22` 请求改善 `>=0.5m`，candidate−G0 mean/median=`+3.658565/+2.300282m`。它在当前 model proxy 上正式 rejected；禁止根据局部 side 拟合直觉继续调 piece 分区后复活同一 arm。
- `V5-F38`：G2 MLS 的收益由少数洞驱动。它在一个 `118,022`-pixel actor 上改善 `11.194128m`，但仅 `8/22` 请求达到改善门，mean/median delta=`+3.005506/+1.620793m`。不得引用最大改善或自适应带宽完成度掩盖广泛稳定性失败。
- `V5-F39`：G3 quadratic 的 median delta=`-1.489037m` 不能覆盖 mean=`+0.103693m` 与 improvement count=`11/22<14/22`。r009 已按冻结 gate rejected；不得事后将判据改成 median-only，也不得沿用 request-unit 错误的 r003 作为相反证据。
- `V5-F40`：blocked terminal 必须与方法失败分离。r001 是 unavailable-view denominator 合同错误；r007 是 artifact serializer 局部变量遮蔽导致 `KeyError: 0`。两者没有可用质量 summary，不能进入 arm 均值，也不能覆盖目录或改 terminal；修复只允许新 run ID。
- `V5-F41`：G1/G2/G3 全部 rejected 且 G0 raw `22/22` fail，当前不存在 safe candidate。按照 V5 causal order，feasibility-first router、validation 和神经 surface 都不得解锁；下一步只能先诊断/修复 Gaussianization representation 与 alpha compositing，之后重新经过独立 development gate。
- `V5-F42`：formal run 的目标目录必须由 runner 原子创建，不能为了 stdout 重定向提前 `mkdir`。r010 在任何模型加载/GPU/质量读取前被 overwrite guard 拒绝；其 blocked terminal、events 和 run.log 保留。修复只改变 launcher，把日志放到 run 外部并用新 r011；不得删除 r010 或把它计入方法分母。
- `V5-F43`：提高 Gaussian asset opacity 不能修复当前 representation gap。r011 的 OPAQUE−BASE 在 `0/22` 请求改善 `>=0.1m`，mean/median post-MAE delta=`+0.059686/+0.065773m`；在 dense 条件下继续提高 opacity 也退化 `+0.035533m`。禁止继续提高 opacity、改 alpha 阈值或把 background mixing 写成已支持机制。
- `V5-F44`：stride `2→1` 的 DENSE arm 在 `20/22` 请求改善，mean/median delta=`-0.424179/-0.480927m`，但这是 frozen scene0471 model proxy 上的机制取证，不是 geometry-safe method selection。G0 raw 仍 `22/22` absolute fail，validation/KITTI/独立 GT 均未读取；不得直接把 stride-1 送入 validation、改 router，或把 post-render 改善写成真实道路恢复。
- `V5-F45`：factorial arm 通过数必须解释为因素对比，不能按臂名投票。r011 中 DENSE 与 DENSE_OPAQUE 都过门，正式 summary 因此保守写作 `multiple_gaussianization_factors_have_broad_mechanism_support`；但 OPAQUE 在 sparse/dense 两个条件都退化，描述性 density/opacity main effect=`-0.436256/+0.047609m`。后续只允许冻结 density representation repair 并重新过独立 gate，不能把组合臂通过误写成 opacity 或 interaction 获支持。

## V5 M1 structured unary 新增防重复结论（2026-08-14）

- `V5-F20`：renderer 的逐 pixel intersection 不是独立多视角证据；同一 Gaussian 在同一 view 覆盖更多像素时，若逐行更新 Beta，会把屏幕面积伪装成 view count。V5 必须先按 `Gaussian × view` 聚合 contribution，再以 `1-exp(-mass)` 形成饱和 visibility，B0/B1 每 Gaussian 每 view 最多一票；不得以原始 intersection 行数扩大 evidence denominator。
- `V5-F21`：只改 arm 名称不能形成 Bayesian ablation。若 B0/B1/B3 共享同一 soft signal 与同一 reliability，它们在代数上会退化为同一方法。结果读取前已冻结为 `B0=hard unweighted`、`B1=hard reliability-weighted`、`B3=soft SAM probability × reliability fractional count`；B2 继续延后，禁止在读到 scene0471 指标后改定义。
- `V5-F22`：ownership posterior 不是新的 Gaussian opacity。若直接把 posterior 当 opacity，会给原 base 中近透明的 Gaussian 注入虚假 semantic mass。所有 2D ownership evaluation 必须渲染 `immutable base opacity × ownership probability`，并在运行前后复核 base checkpoint SHA；不得用 posterior-only rasterization 形成虚假 IoU/Boundary F1 改善。
- `V5-F23`：scene0471 的 annotation prompt 集与 checkpoint RigidNodes 表示没有天然一一对应。冻结规则选出 `17` 个轨迹大于 1 m 的非自行车 vehicle（`15 car + 2 construction`），而 formal checkpoint 只有 `15` 个 RigidNodes instances；差异可能来自无足够 LiDAR 的标注 actor。2D SAM union 与 3D base-model proxy 必须分别报告，不得把表示缺口静默算成 unary FN、删除 prompt 或补造 Gaussian。
- `V5-F24`：直接调用 `process_camera/collect_gaussians` 会绕过 `SceneGraphTrainer.forward()` 中的 timeline 设置；若不显式按 `normed_time` 更新 `trainer.cur_frame` 与各 Gaussian model 的 `set_cur_frame`，动态 actor 会被错误地固定在旧帧。V5 sidecar runner 必须复现该状态迁移并做 nearest-timestamp 回归测试；仅图像 frame ID 正确不足以证明动态 Gaussian pose 正确。
- `V5-F25`：scene0471 r037 的 B1/B3 虽同时改善 IoU、Boundary F1、Brier、ECE、NLL 与 FP semantic mass，但 2D FN semantic mass 相对 B0 分别增加 `+0.0915315/+0.0954773`，明显超过计划 validation gate 的 `+0.01` 容忍量。不得只摘录正指标把 r037 写成 M1 成功，也不得以 aggregate calibration 改善掩盖漏检代价；后续 graph 诊断必须逐臂保留 FN、per-view denominator 与 abstain。
- `V5-F26`：r037 是单个 development scene、固定 `0.5` 阈值、`8 accepted + 7 abstained` evaluation views 上的 SAM-proxy 机制诊断；它能推翻“reliability-aware unary 完全无方向信号”，但不能证明 topology graph 必要、不能选择 B1/B3、不能代表 8-scene validation。graph 只能在单独预注册协议后启动，禁止根据 r037 结果补调 unary 参数、读取 validation 或直接扩展 Transformer/semantic split。
- `V5-F27`：r038 的 G3 在 scene0471 2D SAM-proxy 上只带来 `+0.008585/+0.006245` Boundary F1（B1/B3），虽然方向一致且 FN 增量小于 `0.002`，但 Gaussian membership proxy 的 IoU 与 Boundary F1 同时退化。不得只摘录 2D 正指标把 graph 写成已通过，也不得只看 proxy 负指标否定全部图机制；两套口径都必须保留，并在 result-blind development replication 后再决定 formal arm。
- `V5-F28`：r038 的 `cross_proxy_affinity_ratio` 使用 Background/RigidNodes membership 仅做事后 leakage 审计；graph candidate/affinity 明确不消费该字段。G1→G3 从 `0.0083646` 降到 `0.0040198` 证明物理 affinity 更少跨越 base proxy，不等于真实语义边界 GT 或 graph 必要性。禁止把 proxy 反馈进建图、据此调 k/扩散率，或直接解锁 semantic split/validation。
- `V5-F29`：长时 sidecar 不能依赖 SSH stdout 生命周期。scene1087 unary r041 已完成主要计算，却在关闭的输出管道上写日志触发 `BrokenPipeError` 并合法标记 `blocked`；不得把它写成方法失败、覆盖目录或复用编号。正式长任务必须在启动时脱离 SSH 并把 stdout/stderr 重定向到独立日志；同一冻结配置以新编号 r042 完成，r041 继续作为基础设施失败证据。
- `V5-F30`：scene0471 的 `8 accepted + 7 abstain` 不能硬编码成 graph 的全场景分母。r044 在读取 scene1087 的绑定 unary 后被该常量 fail-closed；修复 `d55a067` 改为 summary/diagnostics 双重验证 accepted、abstain、总分母与 B1/B3 `(frame,camera)` 键，并要求 accepted>0。r044 是通用化合同失败，不是数据或 graph 质量失败；修复后只能使用新 run r045。
- `V5-F31`：三场景 frozen SAM 的可用视图为 scene0471/1087/0379=`18/2/6`（各 30），对应 unary 可评估分母=`8+7 / 1+14 / 3+12`。scene1087 的负方向只来自 1 个可评估视图，不能扩大为总体失败；但 result-blind replication 必须保留该稀疏场景和全部 abstain，禁止删场景、补 prompt、补 mask 或只报告可评估视图较多的场景。
- `V5-F32`：G3 三场景复制门正式失败。六个 `scene × unary` 单元只有 `3/6` 个 Boundary F1 为正（门槛 `>=4/6`）；虽然 mean ΔBoundary-F1=`+0.0016107723`、mean ΔFN-mass=`+0.0025676789` 单项通过，但 scene1087 的 G1 cross-proxy affinity 已为 `0`，G3=`1.2800523e-29`，逐场严格下降也失败。不得用正均值覆盖稳定性门、选择 G3、读取 validation 或直接堆 Transformer。semantic split 仍是条件任务，必须先用独立 boundary-residual forensic 证明 boundary ambiguity 是主要残差；当前不自动解锁。
- `V5-F33`：boundary error enrichment 高不等于 boundary 是主要残差。r001 六个单元的 enrichment=`3.83×–280.98×`，但 boundary-primary=`0/6`，mean boundary classification/semantic-error share 只有 `0.402095/0.248353`。尤其 scene0379 虽约 68% threshold error 位于极小边界带，boundary semantic-error mass 仍只有 26%–36%；不得只引用 enrichment 解锁 split。M1B 条件未成立，semantic split/Transformer/validation 继续禁止，M1 structured ownership 收口为 rejected。

## V5 KITTI archive / adapter 新增防重复结论（2026-08-14）

- `V5-F01`：官方 KITTI Tracking calibration 不是统一的 `key: values` 语法。实际 `P0`–`P3` 行带冒号，`R_rect`、`Tr_velo_cam`、`Tr_imu_velo` 行不带冒号；V4 `_read_numeric_table()` 会静默忽略后三类行，真实 adapter 将缺失 rectification/extrinsic。V5 必须同时解析 colon/whitespace 两种格式，并对矩阵 shape、finite、handedness 和投影做 2-sequence smoke；不得把 zip layout ready 写成 calibration gate 已通过。
- `V5-F02`：官方 tracking OXTS 每行是 `30` 个导航/IMU 字段，不是 12-value `3×4` world pose。V4 `_load_pose_matrices()` 会截取前 12 个值并错误解释为位姿，行数与 sensor frame 相等也不能证明 object/world/camera chain 正确。V5 必须按官方语义从 latitude/longitude/altitude/roll/pitch/yaw 构造 pose，并结合 `Tr_imu_velo` 验证坐标链；禁止直接复用 V4 OXTS path 当 pose matrix。
- `V5-F03`：central directory 可读、成员集合对齐和全 archive SHA-256 只证明压缩包可冻结、可进入 staging，不等于真实 adapter 已完成。合法晋级仍需要独立 `.partial` 解压、post-extract member/frame audit、2-sequence 坐标/pose/track-ID smoke 和新 manifest；不得从 archive metadata 直接写 `WS-V5-D1-KITTI-ADAPTER-01=done`。
- `V5-F04`：KITTI Tracking 官方 testing split 没有 `label_02`。testing sequences 可用于无标签 adapter/engineering smoke，但不能进入需要 track/box GT 的 cross-domain 质量主表。V5 10-sequence formal 必须从 21 个 labeled training sequences 中在结果前冻结；不得用 testing split 扩大带 GT denominator。
- `V5-F05`：实际 archive 的 `training/0001` 不是三传感器全帧严格对齐：`image_02/image_03/velodyne=447/447/443`，LiDAR 缺 `000177`–`000180`。这不是 ZIP 损坏或全 KITTI 缺失，但会使“每帧都有 stereo+LiDAR”的 adapter 合同失败。V5 必须在结果前冻结 common-frame/abstain 与 coverage denominator，逐序列记录被排除帧；不得静默 `set` 取交集、补造 LiDAR、删掉 0001 或写成 447/447 完整 multimodal coverage。
- `V5-F06`：V4 M2 geometry risk 使用 `clip(hole_geometry_mae_m / 0.5, 0, 1)`；r219–r221 的 `214` 个 candidates 中 `192` 个饱和为 1，且所有 MAE `>=0.5 m` 的 `192/192` candidates 都相同。`57/130` 个有候选 request 存在“未归一化 rendered MAE 不同、geometry risk 全相同”的碰撞。V5 不得只改 geometry 权重或阈值后宣称解决；任何 mapping 必须保持 tail rank，并报告 saturation ratio、unique risk、rank correlation 与 bad-tail distinguishability。
- `V5-F07`：M2 `+3.3908096237 m` 是保留 abstain 的 policy-level scene-balanced delta，不等于 83 个 accepted repair candidate 相对 TELEA 退化。accepted-only 同请求 router/TELEA=`1.62295/2.01453 m`，而 47 个 risk-abstain 的 atomic no-op/TELEA request mean=`16.58283/2.60817 m`。V4 caveat 不得删除，但后续必须同时报告 accepted geometry、abstain geometry、coverage 和 full-denominator valid yield；不得把两种口径互相替代。
- `V5-F08`：V4 M1 canonical state 已把 observation 压成正/负 count 与乘积 weight，未持久化 per-view observation、投影 boundary distance、Gaussian center/covariance 或 neighborhood/topology disagreement。仅凭 r200 state 不能证明 SAM 错、graph 必然有效或 topology 是唯一根因。M1-D0 必须先生成带 provenance 的 per-Gaussian/per-view diagnostic；缺字段时保持 `running/blocked_evidence_missing`，不能从 aggregate Boundary F1 直接跳到完整 graph 实现。
- `V5-F09`：`WS-V5-M1-D0-BAYES-FORENSICS-01=done` 只表示历史分母已机器重算、缺失字段采集合约已冻结。canonical conclusion=`blocked_evidence_missing_contract_frozen`；不得把 task done 改写成 evidence complete、graph 已验证、M1 rejection 被推翻或可直接训练 full structured ownership。
- `V5-F10`：M2 的 retrospective geometry oracle 只按现有 rendered `hole_geometry_mae_m` 相对排序；其 reference 是 immutable base `Background_depth`，不是 same-view hidden-background GT。即使 `62/83` accepted 与该 oracle 一致，也不能证明 candidate 物理正确。必须先补 `reference_source/confidence` 与 raw→pre-Gaussian→post-render 三段误差，随后才允许在 fresh development 拟合 non-saturating mapping。
- `V5-F11`：P0 freeze-only commit=`dfe7526c7a83ca12d7fa9f6c5a11a29ea7b27b19` 只冻结 scope、historical bindings、missing-evidence schema 与审计器。它不包含 fresh scene selection、模型实现或质量结果；任何后续工作必须通过 P0 formal audit，并继续保持 fresh/test/KITTI quality 未读与 parameter search=false。
- `V5-F12`：fresh 8/8/20 的冻结只使用官方 split、scene context、actor annotation/LiDAR-count metadata proxy 与 sensor-keyframe completeness；没有展开图像/LiDAR blob，也没有读取 reconstruction/edit/M1/M2/M3 quality。20 个 test scene 的身份出现在 freeze manifest 不等于 test quality 已读；`V5_TEST_FREEZE.json` 与 exact-once ledger 形成前，禁止加载其内容或指标。
- `V5-F13`：fresh development cohort 冻结后，8 个 scene 的 processed 为 `0/8`；三前向相机+LiDAR keyframe 的 `0/1280` 只是早期粗审计，不能作为 DriveStudio 10Hz preprocess 的完整分母。核对上游后，真实合同是六相机+`LIDAR_TOP` 的完整 keyframe/sweep 时间链，metadata-only 精确分母为 `14,220` files、当前 `0` present。这是 selective extraction/preprocess 工程前置，不是 M1 质量失败；不得退回 V4 scenes、替换 frozen cohort、提前读 validation，或为省事解压全部约 294 GB blobs。必须一次扫描 metadata、按 member→archive 选择性抽取，并保留逐 scene/sensor/file denominator 与内容哈希。
- `V5-F14`：V4 semantic mask NPZ 实际保留 SAM2 `logits/raw_binary/binary`，因此 V5 不得把最终 binary 当作唯一 confidence，也不得为补 confidence 重新运行或更换 SAM。V5 使用 frozen logit 的 sigmoid 作为 observation probability；quality gate rejected mask 仍保留 raw logit 供诊断，但必须把 positive/negative/reliability 全部置零并显式记录 availability，禁止把拒绝样本误当作背景负证据。
- `V5-F15`：Gaussian 最小 covariance 主轴只能作为 renderer-native surface-normal proxy，不是 LiDAR/mesh ground-truth normal；其符号还具有本征向量二义性。V5 必须用 reference camera 定向、验证 covariance 正定与 available normal 单位范数，并把 `normal_is_ground_truth=false` 固化进 config。若 graph 改善，不能据此宣称已恢复真实表面法线。
- `V5-F16`：DriveStudio 原生 nuScenes preprocess 完成不等于 StreetGS 训练输入已闭合；它生成 images/calibration/LiDAR/object/dynamic masks，但不生成 StreetGS loader 必需的三训练相机 `sky_masks`。V5 首个 profile r003 在任何训练迭代前因 `sky_masks/000_0.png` 缺失合法 `blocked`，summary SHA=`a2802430984ab369143be609088df514e3ed0943563b23ee0a5b3bee02e214f7`。这不是 reconstruction 质量失败，不得覆盖 r003、伪造空 mask 或把 preprocess 8/8 改写成失败；必须先用已冻结本地 SegFormer revision、offline/atomic 协议派生每 scene `frames×3` masks，绑定独立 manifest/SHA，再以新 run ID 重跑 profile。

- `V5-F19`：Python 包内单测通过不等于脚本可从仓库根直接启动。KITTI audit r002 attempt 在读取任何 payload 前因 `ModuleNotFoundError: motion_proj` 失败；import 发生在 runner main 前，因此没有生成 run 目录。不得伪造 r002 terminal、复用该 ID 或把它写成数据/坐标质量失败。修复必须在 package import 前显式加入 project root，并增加 `script.py --help` 直接入口回归测试；提交 `43fe090...` 后以 r003 新 ID 完成真实 smoke。
- `V5-F18`：单场或部分场景的 100-step 成功不能解锁全量 formal，也不能解释成 reconstruction 质量成功。必须保留 8-scene denominator，每场验证 step-100 checkpoint、finite means、summary/status/fingerprint/run-manifest、clean source 与 checkpoint bytes/SHA；formal runner 必须再次读取已提交的 cohort binding。r019–r026 的 `8/8 done` 只证明训练链路与资源门可用，尚未读取 development quality，也不允许跳过 30k base、改用 profile checkpoint 做 structured ownership 结论。
- `V5-F17`：sky mask 文件存在不等于训练输入已合法绑定。V5 必须同时验证 8 个独立 run 的 summary、run manifest、sky-mask manifest、冻结 SegFormer revision、`frames×3` denominator 与全部 PNG bytes/SHA，并把这些 identity 通过新 overlay 绑定到不可变 reconstruction base 配置；不得回写被 r003 引用的 base 配置、只数文件名后开训，或把 segmentation inference 误写成 method inference。r011–r018 已按该协议闭合 `4704/4704`，只解锁 `profile100`，不直接解锁 30k formal 或质量结论。

<a id="detail-v4"></a>

## V4 M3 / 18-scene exact-once 防重复结论（2026-08-13）

- `V4-F44`：r258 因 18 场 sky masks 尚未齐全而 fail-closed，r277 因上游假定 instance timeline 稠密而在 scene-0919 暴露稀疏时间轴合同错误；两者都是资产/兼容性失败，不是模型质量失败。只允许以提交 `d5a4794e` 的稀疏 timeline 兼容修复及 r278 100-step smoke 解锁正式训练，不得覆盖失败 run 或提前读 test quality。
- `V4-F45`：M3 validation r238 的完整 denominator 是 `3 evaluable + 3 abstain = 6`。30.4106% warp L1 改善与 2.6470% temporal LPIPS 改善只来自可评场景；不得删除 abstain、写成 6/6 质量成功，或外推到长时序/非三前向相机。
- `V4-F46`：REMOVE 使用 exact bypass，零时序增益是冻结组合合同的结果；M3 通过依赖预注册的 across-operation temporal gate，不代表每个 operation 都严格改善。不得事后取消 bypass、改 operation 权重或只报告 LATERAL/INSERT。
- `V4-F47`：M2 晋级不消除 geometry 风险。hole geometry MAE 的 signed improvement 为 `-3.3908096237 m`（即误差退化 `+3.3908096237 m`）；18-scene 时序结论无论为 `confirmed`，都不得改写成 repair geometry dominance。
- `V4-F48`：18-scene test 使用 committed freeze 与 exact-once ledger；每场 attempt marker 在任何 test content/quality read 前以 exclusive create 写入，已消费 attempt 禁止重跑。canonical ledger=`/root/autodl-tmp/runs/worldsim_v4/WS-V4-M3-TEMPORAL-DELTA-01/20260813T222011Z__m3-test-exact-once-ledger-s0`，attempt/completion=`18/18`；聚合器只读 run evidence，未重读 test source content。
- `V4-F49`：test 的 abstain 必须留在 18-scene denominator。canonical `20260813T225624Z__m3-test-aggregate18-s0-r335` 为 `12 evaluable + 6 abstain`、conclusion=`confirmed`；不得把 evaluable-only gate 写成全 18 场成功，也不得因 `not_confirmed` 复用同一 test 调参或因 `confirmed` 扩大声明边界。

## V4 M1 rejection / M2 validation 新增防重复结论（2026-08-13）

- `V4-F39`：M1 的 development 正结果不能覆盖 scene-disjoint validation 负结果。validation 只有
  `3/6` scenes 可评，方向支持=`0/6`，Boundary F1/Brier/ECE 均反向；base/checkpoint exact 且没有 validation
  重搜。M1 必须保持 `rejected`，不得继续加 feature、transformer、改 threshold，或把 M2 成功倒写成 M1 成功。
- `V4-F40`：M2 validation 的完整 denominator 是 `6 scenes / 154 requests`。scene-1089/0862/1012 的
  `ABSTAIN_NO_ACTOR` 和 scene-0317 的 24 个 `ABSTAIN_NO_ROLE_MATCHED_ERASE_PACKAGE` 都必须保留；不得只用
  130 个具备 role asset 的请求或 3 个可评场景改写 coverage。canonical coverage 固定为 `83/154=0.5389610390`。
- `V4-F41`：validation 不允许重新选择 baseline、risk weights 或 threshold。matched baseline 必须沿用 development
  冻结的 `TELEA`，router 必须沿用 `uncertainty_forward/threshold=1.0`。即使 validation 上其他 arm 的 composite
  error 更低，也不得事后改 comparator 或路由 operating point。
- `V4-F42`：M2 通过的是预注册的合取门，不是所有 repair 轴支配。相对 TELEA，router 的 global PSNR/SSIM/LPIPS、
  hole PSNR、static LiDAR 和 selective-risk separation 通过，但 hole geometry MAE 从 `2.1435024986 m` 退化到
  `5.5343121223 m`。不得把 `hole_any_endpoint` 通过写成 geometry 改善、真值背景恢复或全面优于 Telea。
- `V4-F43`：selective-risk 成立只表示 frozen uncertainty 排序在当前 validation 请求上有误差分离：abstained
  counterfactual error 比 accepted 高 `0.1241311528`。它不证明 71 个 abstain 已被成功修复，也不允许把 abstain
  从 usable-yield 分母删除。M3 与 18-test 必须继续同时报告 coverage、abstain、blocked 和 worst-case。

## V4 M1 / validation 新增防重复结论（2026-08-12）

- `V4-F34`：同一 scene 的历史 V3.3 train mask 不能自动视为符合 V4 冻结的
  `sample_index mod 5` partition。scene-0230 的 development target 审计发现真实 train/evaluation
  frame overlap，因此必须 `ABSTAIN_LEGACY_SPLIT_LEAK`；不得放宽 split、删除该 scene denominator，
  或把旧 heldout 结果改名为 development。
- `V4-F35`：M1 六场景质量均值只允许在可评 scenes 上计算，但 coverage denominator 必须保持全部六场。
  r124 的 `2 evaluable + 4 abstain` 是协议事实，不得把 2/2 改写成 6/6 成功或静默删除 abstain。
- `V4-F36`：validation 只能复用 development 冻结的 evidence arm、calibrator、mask threshold 与 temporal
  retention；禁止在六个 validation scenes 上再次执行 arm search、calibration fit 或 threshold search。
- `V4-F37`：长时间 archive scan 不能依附会超时断开的 SSH stdout。r128 的 10 个 worker 在扫描约
  58 分钟后因外层 SSH 断管触发 `BrokenPipeError`；这不是数据缺失，也不得覆盖该 run。重试必须使用
  stdin=`/dev/null`、stdout/stderr 文件重定向、parent PID=1 的 detached 进程，并复用已提取的非空文件。
- `V4-F38`：Python 环境必须按 stage 显式区分。validation raw extraction 需要
  `/root/autodl-tmp/envs/motionproj/bin/python`（含 `ijson`）；StreetGS/V3.3 GPU runtime 使用
  `/root/autodl-tmp/envs/drivestudio/bin/python`。r127/r129 分别保留缺依赖与错误解释器路径证据，
  不通过临时安装或删除失败记录掩盖环境错误。


> **历史合并注记（2026-08-12）**：以下 V4 D0/B0 与更早内容在当日从旧账本合入；当前权威元数据、目录和写入合同
> 以上方 2026-08-17 统一入口为准。完整 `RF-01`–`RF-18` 原文见
> [`archive/2026-07/v7-feasibility/RESEARCH_FAILURES_RF01_RF18.md`](archive/2026-07/v7-feasibility/RESEARCH_FAILURES_RF01_RF18.md)。

本文件保留仍约束后续路线的历史结论，并把 H1-11D 的失败严格分为“观察到的事实、合理推断、尚未知、
复开条件”。归档不会使旧失败失效；任何新计划复用旧机制时仍须满足原 RF 的重开条件。

## V4 D0 防重复结论（2026-08-11）

- `V4-F08`：nuScenes metadata 的实际可用入口是 `/root/autodl-pub/nuScenes` 中的官方 archive，不是历史代码里的
  `/root/autodl-tmp/data/nuscenes` 或空的 DriveStudio data 目录。D0 只展开 `v1.0-trainval_meta.tgz`，不得为 cohort
  选择展开 549 GB sensor blobs、下载副本或把空目录写成数据集失败。
- `V4-F09`：nuScenes log filename 不能用未锚定的三组 `NN-NN-NN` 正则取小时。首版把
  `n015-2018-11-14-19-09-14+0800` 的月份 `11` 当成小时，测试将夜间误分为白天。修复必须解析完整
  `YYYY-MM-DD-HH-MM-SS±ZZZZ` 尾部；不得以放宽测试或手工改 scene 标签绕过。
- `V4-F10`：三个既有 processed scene 是 infrastructure anchors，不是结果前随机样本，也不能用其 V3/V3.3 质量选择
  test。D0 只把 0230/0242/0255 固定在 development，并以 metadata-only diversity 补足其余 scenes；后续不得把
  anchor smoke 写成 baseline 或方法质量结论。
- `V4-F11`：只冻结 scene name 列表不足以复验 cohort。D0 config 同时冻结 30 scene 的 actor/edit/clip/frame/sensor
  完整记录，formal builder 逐字段比对并锁 cohort SHA；任何 metadata 更新或构建逻辑变化都必须新建 run，不能静默
  沿用 `eda9f684...44578`。
- `V4-F12`：metadata donor support 是场景级 proxy，不是重建后 donor 的图像/几何质量。D0 可以用它做结果前分层，
  但 B0/M2 不得把 strong/medium/weak 标签直接当作 repair 成功、真实性或 quality ground truth。
- `V4-F13`：确定性 greedy selection 不能对无序 `set` 直接做普通浮点求和。r1/r2 恰好重建一致，r3 在另一
  `PYTHONHASHSEED` 下因同分 score 的末位舍入选择了不同 scene，freeze gate 以 `af36f51a...3447 !=
  eda9f684...44578` 拦截。r4 必须对 tag 排序并用 `math.fsum`，测试须跨多个 hash seed；不得固定环境变量掩盖算法
  非确定性，也不得把 r3 候选倒写成新 cohort。

## V4 D1 防重复结论（2026-08-11）

- `V4-F14`：requested dataset path 与 symlink-resolved physical path 必须同时记录。D1 r1 只写了
  `/autodl-pub/data/KITTI`，容易被误读为审计了不同根；r2 明确 `/root/autodl-pub/KITTI ->
  /autodl-pub/data/KITTI` 且两者均缺失。不得用 `Path.resolve()` 后的单一路径抹掉用户合同。
- `V4-F15`：synthetic 12-gate pass 只证明 adapter schema/坐标/投影/track-ID 检查可执行，不等于真实 KITTI adapter
  smoke。真实目录缺失时任务必须保持 `blocked`，不能将 unit fixture 写成两 sequence evidence 或 10-sequence
  cross-domain 结果。
- `V4-F16`：KITTI 原生合同只有 `image_02/image_03` 两路彩色相机。不得为了匹配 nuScenes 三相机伪造第三视角；
  tracking 缺失时才审计 raw，且 pose/tracklet/calibration 任一门失败都必须 `blocked_dataset_adapter`。

## V4 B0 防重复结论（2026-08-11）

- `V4-F17`：历史 summary/metrics 记录 checkpoint 曾经存在，不等于当前 checkpoint 可执行。B0 首次磁盘审计中，
  scene-0230/0242/0255 的历史 StreetGS 文件均已不在路径；只能保留 bytes/hash provenance，不能把历史
  `exists=true` 或旧质量数值登记为当前 executable。正式 matrix 必须在每次 run 重新检查真实文件。
- `V4-F18`：AD-GS historical aggregate 的 6-scene metrics 只覆盖旧 cohort，且其历史 source/env/checkpoint 路径已缺失。
  与 V4 development 重叠的 0230/0242/0255 也只能记 `historical_metrics_only`；2026-08-12 新恢复的 exact official
  source/env 只解除执行前置，不使历史 metric executable。不得把旧三场景数值拼接新三场景、用 aggregate mean 代替
  scene rows，或把环境 smoke 写成 V4 same-split checkpoint。
- `V4-F19`：baseline inventory 的 `blocked` terminal 是当前资产前置条件未齐，不是 B0 task 的永久 blocked 或方法失败。
  B0 继续 `running`，通过新 run 补齐资产；旧 inventory 不覆盖。只有 V3.3/StreetGS/AD-GS 各 6/6 且统一 evaluator
  完整后才可收口，不能因 M1 实现更有趣而跳过 matched baseline。
- `V4-F20`：DriveStudio preprocess 会把输出根再追加 `_10Hz`，并按零填充 scene index 写目录。r4 的上游命令成功
  不等于 runner 目录合同成功；必须验证真实输出 root、scene dir、`1,176 RGB / 196 LiDAR` 后才能登记 done，不能移动
  或重命名一个未审计路径来掩盖合同错误。
- `V4-F21`：远端网络不可达不能用镜像、floating revision 或未校验模型绕过。sky-model r10 因 `Errno 101` 保留
  blocked；本机只从官方固定 revision URL 获取三文件，传输前后均按 bytes/SHA256 exact 校验，远端恢复经临时目录
  原子发布且 generation 保持 offline。任何 staging 漂移都必须 fail-closed。
- `V4-F22`：preprocess 预建的空 `sky_masks/` 与已有推理产物不是同一状态。r12 因旧 runner 只看目录存在而 blocked；
  修复只允许非 symlink 的空目录，发布前再次 `rmdir`，已有 mask、非目录或 partial 一律拒绝，不能覆盖正式产物。
- `V4-F23`：StreetGS 100-step profile 只证明训练链、checkpoint schema 和单卡资源门可执行。r16 不能计入 6-scene
  30k formal coverage，也不能读取/登记质量或据此宣称 baseline 已完成；每场 formal 必须新建不可变 run。
- `V4-F24`：六场景 StreetGS 的 Gaussian 数、wall 与 peak GPU 差异很大；scene-0255/0048 sampled peak 达
  `24,092/24,000 MiB`，scene-0994 final RigidNodes 仅 `1,029`。不能用一个 scene 的 profile 外推所有资源，不能因
  actor 稀疏补点或删 scene，也不能把无 OOM 的近上限运行倒写成资源失败。主表保留每场分母与工程行。
- `V4-F25`：StreetGS 原生 `test_image_stride=10` 不是冻结的 `sample_index mod 5` 三分区。r17/r20/r22/r24/r26/r28
  即使完成 30k、checkpoint finite 且未主动读 test quality，余数 4 的 heldout 输入仍可能进入训练，因此只能保留为
  protocol-mismatch provenance；r29 的 `StreetGS=6` 被 corrected inventory r33 明确推翻。不得用“训练成功”替代
  matched-contract 合规，也不得覆盖旧 run 来修正历史。
- `V4-F26`：不读取 test quality 还不够，训练进程也必须在 I/O 层隔离 development/heldout。AD-GS adapter 正式训练
  只物化 `train` 的 354 张图；兼容补丁增加 `--disable_test_evaluation`，避免上游在 final iteration 自动将 test
  iterations 加入评测。审计/统一 evaluator 可以显式物化三分区，但训练 runner 不得复用该全量目录。
- `V4-F27`：source checkout、权重和 Python 包必须分别固定 commit/bytes/SHA，环境恢复成功不等于 baseline scene
  executable。r34 从冻结本地环境离线复制，编译 `simple_knn` 与 `diff_gaussian_rasterization` 并通过真实 CUDA
  forward/backward smoke；在 strict preprocess + checkpoint 完成前，AD-GS coverage 仍为 `0/6`。
- `V4-F28`：传输/命令包装失败与模型失败必须分开。CoTracker/plyfile 的首次远端校验受 shell quoting 影响，DPT 下载
  曾出现两个进程指向同一 partial，发布后的附带 `stat` 也曾因 quoting 失败；这些尝试均未被登记为 canonical。
  只有停止冲突进程、临时路径原子发布并对最终 bytes/SHA 做独立复验后才可使用，且不得把 wrapper failure 写成
  权重、CUDA 或算法失败。
- `V4-F29`：preprocess 失败后的目标目录不是可静默复用的 canonical。r35 因启动包装器预建 run 目录而被不可变门拒绝；
  r36 因环境构建留下的未跟踪目录被 source-audit 拒绝；r37 完成 adapter/depth/segment 后因可选诊断依赖
  `flow_vis` 缺失而在 flow 启动时 blocked。三者均未训练或读取 dev/heldout；r37 partial 移入
  `work/codex-backups/2026-08-12-adgs-r37-partial-scene0230`，不覆盖、不伪装 resume。正式 flow 通过显式
  no-visualization 合同移除诊断视频依赖，CUDA extension 后续从 run-local source copy 构建，避免再次污染 official checkout。
- `V4-F30`：修掉 preprocess 可视化依赖不等于训练 import graph 已解除同名依赖。r39 尚未进入 iteration，
  `loss_utils -> flow_utils` 就因全局 `flow_vis` import blocked；正确修复是只在 TensorBoard flow 图真正调用时 lazy import，
  并把 `utils/flow_utils.py` 纳入 exact compatibility patch。不得安装非必要诊断包来掩盖正式无评测训练合同。
- `V4-F31`：Python import 成功不证明 CUDA extension 包含当前 GPU kernel。r41 能加载 PyTorch3D 0.7.5，
  但 inherited `_C.so` 在 RTX3090 KNN 首次执行时报 `no kernel image`；必须从 clean frozen source 在 run-local
  目录以 `TORCH_CUDA_ARCH_LIST=8.6` 重编，并在环境 smoke 中真实调用 `knn_points`。r42 完成该合同后 r43 才进入
  100/100 iteration；r39/r41 继续保留 blocked，不倒写为成功。
- `V4-F32`：进度条达到 60k 或路径上存在 checkpoint 不等于 scene executable。r44 只有在 formal step、run 内
  `point_cloud/deform/env` 三文件 bytes/SHA、fingerprint/manifest、source HEAD、六修改文件与兼容补丁全部精确后，
  才由 r45 从 `AD-GS 0/6` 更新为 `1/6`；StreetGS 同样从“存在即计数”收紧为 runtime+bytes+SHA 精确。
  `AD-GS-2026-07-27.patch` 是 zero-context patch，reverse-check 必须显式传 `--unidiff-zero`，否则会产生审计假阴性。
- `V4-F33`：单个方法达到 6/6 不等于 B0 完成。StreetGS r32/r46/r48/r50/r52/r54 已按 strict mod5、
  checkpoint bytes/SHA 与 clean r55 inventory 收口为 6/6，但 V3.3/AD-GS 仍各为 1/6，统一 evaluator 也尚未生成
  完整 scene rows；不得据此启动 M1、读取 test quality，或把 inventory 的 `matched_baseline_assets_incomplete`
  倒写成 StreetGS 失败。后续只补缺失方法/场景并保留旧 inventory。

## V4 P0 防重复结论（2026-08-11）

- `V4-F01`：计划草案记录的 HEAD 不是执行时事实。草案写 `main@144ed19`，P0 实查为 `main@2108430`，且 V3.3
  收口 `e6663e1` 已进入其历史。V4 必须从真实 `main` 建分支，不得回退旧 HEAD 或把草案 provenance 写成 canonical。
- `V4-F02`：计划写“`/root/autodl-pub/KITTI` 已在公共盘”不等于当前机器已挂载。P0 实查目录不存在，状态固定为
  `blocked_local_dataset_missing`。不得创建空目录、下载 KITTI、借其他 layout 冒充，或把外部缺盘写成 adapter/算法失败。
- `V4-F03`：H0 在计划表中写了 `conditional`，但同一计划规定任务状态只允许
  `pending/running/blocked/done/rejected`。V4 注册表将 H0 规范化为 `pending`，条件授权单独记录；不得引入第六种状态。
- `V4-F04`：一手论文、项目页或官方源码存在不等于 baseline 已在本机 single RTX 3090 + same split 执行。SplatAD、
  IDSplat、HorizonForge、RecEdit-Drive 等必须分开记录 paper/source/executable 状态；没有 matched run 不填数值。
- `V4-F05`：KITTI 缺失不阻塞 D0 nuScenes cohort，但会阻塞 single-card closure 的 KITTI adapter smoke。不得因此提前
  读取 nuScenes test、跳过跨数据集条件，或用多卡/新下载掩盖外部前提。
- `V4-F06`：V4 的公式必须同时对应 config、代码、ablation 和可计算指标。P0 只冻结 schema，不代表 M1/M2/M3
  已实现或有效；后续失败必须按预注册早停，不得继续堆 evidence feature、diffusion 或数学包装。
- `V4-F07`：formal run 通过不代表可跳过提交前 whitespace gate。P0 r1 使用的方法合同正确，但未跟踪计划的参考文献
  含 Markdown 行尾空格，`git diff --check` 拒绝提交；规范引用格式后 plan/config SHA 改变。r1 保留 noncanonical
  done，r2 对最终字节重新审计并成为 canonical；不得倒写或覆盖 r1。

<a id="detail-v3"></a>

## V3.3 R0 防重复结论（2026-08-11）

- `V33-F41`：R0 不能把 JSON 中“语义相近”的类型或枚举视为相同。diagnostic 前三次分别把 S2 的空列表写成
  数值 0、把 S3 `heldout` 写成 `heldout_confirmation`、把 S4 `real_renderer_evaluation` 写成
  `evaluation`，均 fail-closed。以后 verifier 必须比较原始类型/枚举；不得用字符串归一化掩盖 schema 漂移。
- `V33-F42`：正式 instance-field validator 通过不代表 NPZ 必须有未约定的 `schema_version`。r4 在报告层
  冗余读取该字段而 failed；修复只移除报告假设，仍执行完整 validator。以后不能把“自己希望存在的字段”变成
  canonical 资产失败，也不能因此跳过正式 schema 校验。
- `V33-F43`：RoadPatch 成为 V3.3 主方法不等于已在 matched 协议下胜过 V3.2 Telea。两者 base、空间语义与
  评测协议不同；R0 答案固定为 `not_directly_ranked`。不得用 B1 相对 B0 的 heldout gate 写成 Telea head-to-head，
  也不得因缺直接排名否定 RoadPatch 的 3D-native/provenance/heldout 成立结论。
- `V33-F44`：内容寻址 release 不允许为“完整”而复制 579 MB base checkpoint。R0 package 含 O1 sidecar、
  RoadPatch/A4/S4 delta、production renders 与 external reference，forbidden model suffix count=`0`；离线 verifier
  同时锁 file set/bytes/SHA。任何新增 `.pth/.pt/.ckpt/.safetensors` 或未登记文件都必须拒绝。
- `V33-F45`：deterministic archive 不能包含当前 run timestamp、绝对输出路径或可变 ZIP metadata。R0 release
  ledger 只引用固定 canonical 输入，ZIP entry 排序/1980 timestamp/permission/compression 固定；diagnostic、
  formal 和同 run replay SHA 均为 `cffaad16...44a7`。以后新增 release 字段必须先证明跨 run byte-exact。
- `V33-F46`：R0 的 `v33_supported` 只覆盖 scene-0230 主链、冻结确认视图和单 RTX 3090。它不证明
  scene-0242/0255 完整 V3.3 transfer、生成 actor GT、相邻视频时序、闭环安全或传感器真实性。F0 LiDAR-EVS
  仍是 conditional 新任务，不能倒写进 R0 的 4/4 success criteria。

## V3.3 S5 防重复结论（2026-08-11）

- `V33-F35`：unconstrained Harmonizer 不能因“只做视觉润色”进入删除生产链。canonical r4 的 edit target
  delete candidate 让冻结 SAM2 semantic mass/fraction 增加 `+0.126399/+0.133885`；production raw fallback
  两项均为 `0`。以后不得关闭 detector、改用候选图作 delete，或只展示另外四个未触发视图。
- `V33-F36`：跨视图平均改善不能掩盖单视图确认失败。五视图 contact 平均为改善，但 heldout f060/c1 的
  contact L1 delta=`+0.422686`，超过冻结逐视图上限 `+0.25`；因此 G1 必须 rejected、production=G0。
  不得改成 aggregate gate、放宽上限、删除该视图或用 edit target 的强改善抵消它。
- `V33-F37`：heldout confirmation 不是第二个开发集。S5 先只用 f091/c1、f005/c0、f065/c1 选择 G1，随后才
  读取 f020/c0、f060/c1；看到 F36 后不得调 contact/shadow 区域、权重、cap 或阈值并继续在同一 heldout 上
  宣称泛化。合法复开需要新假设、新 task 和未读 confirmation 数据；R0 只登记当前负结果。
- `V33-F38`：冻结推理环境不应为共享模块的无关顶层依赖而污染。r1 的 Harmonizer 已完成，SAM2 启动时因
  `semantic_gate.py` 顶层 SciPy import failed；SAM2 只消费 semantic mass/decision，并不构建 gate。修复是将
  SciPy 限在 builder 内 lazy import，不是往冻结 SAM2 环境临时安装包；r1 terminal 保持 failed。
- `V33-F39`：R3D2 official code、Apache license 与 clean commit 不等于存在作者 pretrained pipeline。
  canonical S5 只登记 `blocked_pretrained_model_unavailable`，`model_loaded=false/training=false`。不得拿
  SD-Turbo/TAESD base、Harmonizer 或自训权重冒充 R3D2-fast。
- `V33-F40`：S5 的五个冻结视图不是相邻视频帧，不能从 deterministic image SHA、跨 run exact 或五视图
  quality 推出 temporal consistency。canonical 明确 `not_evaluated_non_temporal_frozen_five_view_protocol`；
  时序 claim 需要独立连续视频协议、新指标与新 run。

## V3.3 S4 防重复结论（2026-08-11）

- `V33-F30`：S1 `hard_instance_id` 是候选身份集合，不等于所有 Background 候选都应被硬 ERASE。r2 将
  high actor 的 `36,736 Background + 4,525 Rigid` 全部设为零，虽然 target coverage=`0.999741`，但目标外
  L1=`0.821965>0.5`，方法按冻结门 rejected。不得通过放宽 L1 门、扩大 target mask 或只报告 coverage 复活该臂。
  合法修复是使用 S1 已训练 instance opacity 的 MAP 正类 `p>=0.5` 选择 Background，同时保留全部物理 Rigid core；
  r8 以 `1,614+4,525` 行将目标外 L1 降到 `0.225349`，门与视角未变。
- `V33-F31`：S2 canonical `roadpatch_delta.npz` 的 104 行同时服务 high/boundary 两个 actor，不是任一单 edit
  都应加载的整体。S4 high package 必须按冻结 `target_role=high_support` 只取 25 行，并保留 parent delta SHA；
  把 boundary 的 79 行混入 high edit 会产生无关场景变化，不能用“同属 Background repair”掩盖。
- `V33-F32`：恢复同一批 Python/Parameter 对象是必要条件，但不足以证明可回滚。renderer 可能持有缓存或顺序状态；
  每个 stack 卸载后必须重新执行 source render 并比较 tensor schema+bytes SHA。canonical r8 为 5 视角×4 stack=
  `20/20` exact，另有 full deterministic replay 和 replay rollback；以后不能只比较 checkpoint SHA 或 object id。
- `V33-F33`：immutable base + delta package 不能把 579 MB checkpoint 复制到 base 目录后仍称“小型 delta”。
  canonical package 的 base 只含 checkpoint/registry reference descriptor，完整 checkpoint copy=`0`，最大 payload
  `3,942,422 bytes`；任何 materialized deployment checkpoint 必须作为独立 deployment factor，不得改写 authoring state。
- `V33-F34`：S4 r3/r4 的方法和指标有效，但最终核心增加 fail-closed policy validator 后其 source snapshot 不再是
  提交态；r5/r6 与最终方法一致，但 builder 后续补齐异常 terminal。最终只认完整重跑 r7/r8 为 canonical；不得以
  “只是校验/失败处理代码”绕过源码 byte-exact 合同。

## V3.3 S3 防重复结论（2026-08-11）

- `V33-F22`：DriveStudio `instances_info.obj_to_world` 在冻结数据中可直接是 list，也可能是历史 JSON string。
  selector r0 只接受 string，在首个候选 fail-closed，未发布 selection。修复必须对两种 schema 都验证 4×4 shape
  与 finite，不得无校验 `json.loads` 或续写 r0；r1/r2 已以新 run 收口。
- `V33-F23`：view selector 的确定性不能用复制 manifest 冒充。high r1→r2、boundary r7→r8 都重新执行全部
  D2 original/delete 候选渲染，formal 传入 diagnostic 的 expected selection/input SHA 并得到 byte-exact；候选池
  必须继续排除 heldout 与 reserved development，不能让评估帧参与 view selection。
- `V33-F24`：更多输入视图不自动等于更好 actor。high development 中 A4 被选择是因为冻结 metric order 下
  IoU/boundary 改善且 LPIPS/PSNR/背景漂移/横移碎裂 retention 全过；heldout 只允许 A0 与 frozen A4 一次确认。
  不得从 heldout 重选 A1/A2/A4，或把生成背面写成 GT accuracy。
- `V33-F25`：V3.2 manual A0 只绑定 high-support token `af663...c5c29`，不是 boundary actor 基线。boundary
  评估必须使用同 identity 的 immutable D2 native actor，不能复用 high A0、换 actor 或仅凭 class 相同跳过三元核对。
- `V33-F26`：Asset Harvester 对 boundary A4 成功生成 PLY/NPZ 不等于 production override 可接受。r12 中 A4
  相对 native 的 IoU/boundary F1 从 `0.666562/0.555343` 降到 `0.624832/0.492141`，LPIPS/PSNR 也失败；
  决策为 `ABSTAIN_GENERATED_OVERRIDE`，未读 boundary heldout。不得通过放宽门、读取 heldout 或只展示 orbit
  render 复活该资产。
- `V33-F27`：CLI 编排提供错误 inference manifest SHA 时，importer 必须在物化前 fail-closed。boundary r10
  因传错 SHA 保留失败证据；正确 SHA 只用于新 r11。不得删除 r10 后续写、绕过 SHA，或把编排错误写成模型失败。
- `V33-F28`：canonical eval 的 source snapshot 必须等于最终提交源码。high r5/r6 的指标与 r13/r14 完全相同，
  但 evaluator 后来增加 native baseline 支持，旧快照不再等于提交态；因此只把 r13/r14 作为 canonical。不得用
  “逻辑没变”绕过 byte-exact 合同。
- `V33-F29`：scene-0242/0255 有旧 V3 checkpoint 不等于存在本任务冻结的 V3.3 S1/S2 mask/actor 输入链。
  在 boundary transfer 已拒绝的情况下更不能混用旧资产补齐跨场景表。合法复开需要新 task、每场景 exact identity/
  mask/checkpoint 协议、冻结 high actor policy 和新 run；旧 S3 terminal 不续写。

## V3.3 S2 防重复结论（2026-08-11）

- `V33-F15`：正式 run 目录不能在 runner 注册前由 `nohup ... > run.log` 预创建。r0 因 shell 先创建
  `run.log` 而触发 non-empty run-directory fail-closed；不得删除日志后续写。正式托管应把 launcher 日志写到
  run 目录外，或由 runner 创建目录后再写；r10/r11 使用新 run 收口。
- `V33-F16`：V3.1 P3 package 的空间网格是 `(x,y)` 且绑定 V3.2 P2 FP16 mixed checkpoint，不是当前 D2 FP32
  原生 Background 的道路索引。DriveStudio 首个 CAM_FRONT 是 OpenCV `x-right/y-down/z-forward`，道路 BEV
  必须使用 `(x,z)`；r1 因错误要求 P3 manifest 绑定 D2 exact SHA 而失败。不得为复用旧索引而放宽 checkpoint hash。
- `V33-F17`：相机内参在当前 DriveStudio 输入中是 9 个值（`fx/fy/cx/cy + 5 distortion`），不是只含 4 个值；
  r2 在正式物化前 fail closed。所有 adapter 必须显式接受已冻结 schema、校验前四项和 distortion 长度，不能静默切片
  后假称已适配其他相机模型。
- `V33-F18`：对整格直接取 `max_scale/max_plane_residual` 会被一个天空、立面或跨层 Gaussian 污染。r3 得到
  `53,541` patches 但 `0` valid；这不证明场景没有道路 donor。修复应先逐行排除 actor/generated/低 support/
  scale outlier，再确定性选择 `<=0.75 m` 的 densest vertical slab，最后做 plane/normal 门。r10 由此得到
  `822` valid patches；不得回到 whole-cell 放宽阈值。
- `V33-F19`：cross-view sidecar 的 `visible_view_count` 与 front-camera frustum observation 是两个合同。
  r4/r5 已有 valid 4 m patch，但把 `minimum_multi_camera_count=2` 当作当前六相机逐相机观测计数，导致两个真实
  target 的 top-5 手工门失败；当前实现明确要求 sidecar `visible_view_count>=5` 且至少一个 front-camera frustum
  observation，不虚构不可得的逐相机 visibility。
- `V33-F20`：donor 几何合格不等于新增 Gaussian 数量可以无限。r8 的 2,150-row dense delta 在 development
  selection 中可见，但 heldout PSNR/SSIM 退化 `-0.8553 dB/-0.00619`，保持 rejected。修复不是从 heldout 选
  top-K，而是在候选资格阶段冻结 `maximum_rows_per_target=512`，让搜索选择最小可见 delta；r11 最终为
  `25+79=104` rows，并通过全部 heldout 门。不得复活 r8 或事后改写其 terminal。
- `V33-F21`：官方 Inpaint360GS source clean、Apache-2.0 不等于当前 StreetGS/3090 上已复现。官方声明
  RTX 4090/CUDA 11.8，并需要主环境、独立 LaMa 环境、CropFormer/Big-LaMa/SAM/DeAOT/GroundingDINO 权重；
  官方代码没有 DriveStudio/StreetGS checkpoint adapter。r12 因这些前置条件 fail-closed 为
  `blocked_single_3090`、`official_execution_attempted=false`。该状态不是 B2 质量负结论，也不得用 Telea、base
  SAM 或自写 RoadPatch 输出冒充官方 Inpaint360GS。

## V3.3 S1 防重复结论（2026-08-11）

- `V33-F07`：磁盘可用空间在 P0 后被外部流程扩大，同时 V3.2 SAM2 checkout/weight/runtime 被删除；不能把
  canonical train masks 仍存在误判为 heldout 推理环境仍可用。普通 clone 又被大型 demo checkout 拖住，已终止本任务
  PID 并保留 `sam2.incomplete-20260811T0138`。恢复只能 sparse checkout exact commit、下载 exact weight，另建
  隔离环境并冻结 package list；不得修改 DriveStudio 环境或复用不完整 checkout。
- `V33-F08`：heldout-target r2 在旧 `/root/autodl-tmp/envs/worldsim-v32-sam/bin/python` 不存在时 exit=`127`；
  prompts 虽生成但 run terminal=`failed`，不得续写。新环境必须复原 V3.2 记录的 Python/torch/torchvision 版本并
  新建 run；r4 已按此收口。
- `V33-F09`：SAM2 singleton predictor 在当前 exact runtime 返回 `[object,1,H,W]`，而旧兼容路径也可能给
  `[object,H,W]`。r3 无条件 `unsqueeze` 产生 5D interpolation 错误；修复必须显式接受 rank 3/4、拒绝其他 rank，
  并新建 r4。r3 未发布正式 mask，不得当质量证据。
- `V33-F10`：更宽的 O3 ambiguous reassignment 并不自动改善边界。100-step development smoke 中 O3 的
  boundary F1/IoU=`0.123499/0.160504`，低于 O1 的 `0.149382/0.181752`，且 FP 更高；O3 已排除。除非提出
  新的几何邻域/身份证据并使用新 task/run，否则不得因“候选更多”重开。
- `V33-F11`：O1 在 heldout 显著改善 boundary/IoU/NBD/FP，但 FN mass 从 `0.061278` 增到 `0.109356`，
  identity presence 仍为 `0.972973`。因此只能声明对象边界与 false-positive 抑制突破，不能声明全面支配或完整
  召回；S2 delete mask 必须继续报告 FN/残留语义，不能用 O1 的高 precision 掩盖漏删。
- `V33-F12`：`np.savez_compressed` 默认把当前时间写入 ZIP entry header；即使 r6/r7 的 O0 全部数组 exact，文件
  SHA 也会漂移。r6 因此保留为 done noncanonical。r7 writer 固定 entry 排序、1980 ZIP timestamp、权限与压缩
  参数，并用同一 field 二次写入 byte-exact 测试锁定容器确定性。该合同不等于宣称 CUDA 训练位级确定性：
  r6→r7 O1 最大 logit/opacity 漂移为 `0.001357 / 8.918e-05`，但 heldout aggregate exact。
- `V33-F13`：正式 run 的 source snapshot 必须与最终提交源码 byte exact，纯 EOF 空白也不能例外。r7 方法与门禁
  均通过，但提交前 `git diff --check` 清理了 4 个新文件的多余 EOF 空行，导致其中 2 个冻结快照不再与待提交源码
  exact；r7 因此降为 done noncanonical，不能仅凭“空白不影响算法”继续引用为 canonical。
- `V33-F14`：长 GPU 任务不能把前台 SSH 生命周期当作任务托管。r8 已完成模型与 finalizer，但 124 秒调用超时关闭
  stdout，外层 `tee` 收到 SIGPIPE，terminal 按预注册 trap 写成 `failed / exit 141`；不得把已有 summary 反推成 done。
  r9 改用 `nohup` 后台托管并以只读 SSH 轮询，最终正常 `done`、GPU 释放、9 个 source snapshots 全 exact。

## V3.3 P0 防重复结论（2026-08-11）

- `V33-F01`：官方 SAM3.1 source 可 checkout 不等于 checkpoint 可执行。当前代码固定为 `96914d2`，但
  `hf auth whoami` 为未登录且 cache 无 SAM3.1；不得绕过 gated access、猜权重 revision/hash，或让该门阻塞
  dual-opacity 主假设。S1 必须 exact fallback 到 V3.2 SAM2.1 canonical masks；未来解锁需新 task/protocol/run。
- `V33-F02`：论文写 code available、GitHub 仓库存在或项目页可访问，不等于存在 runnable implementation。
  GS-RoadPatching `468f812` 只有 HTML/CSS/JS/图片、无算法源码和根 LICENSE；OP2GS、3D-GIMP、FocusGS、
  LiDAR-EVS 也没有可固定官方 runnable source。后续只能称 `*-inspired` 或 `audit_only`，不得写 reproduction。
- `V33-F03`：R3D2 `3fc6e31` 已公开 Apache-2.0 训练/export/eval 代码，但仓库只声明下载 `sd-turbo/taesd`
  base，没有作者训练并导出的 R3D2 pipeline。单卡从零训练不是“补齐 inference”，且被计划禁止；S5 保持
  `weights_blocked`，不能拿 base diffusion 输出冒充 R3D2。
- `V33-F04`：GOR-IS source release 不消除许可与运行合同。根许可证只允许 non-commercial research/evaluation，
  torch/CUDA 未 pin，且要求 nvdiffrast、CUDA rasterizer 和 OptiX gtracer；没有 pretrained manifest。它只能作
  optional audit，不能抢占 RoadPatch 主线或被写成单卡已验证 baseline。
- `V33-F05`：Inpaint360GS 的官方 source/Apache-2.0 只支持进入 adapter/preflight。上游验证环境是 RTX 4090 /
  CUDA 11.8，并依赖外部 CropFormer/LaMa 权重；在 StreetGS split、相机、分辨率和输入 schema 冻结前，不得
  安装/训练。若 24 GiB 下必须静默降正式分辨率、改 heldout 或改相机数，必须 `blocked_single_3090`。
- `V33-F06`：P0 重新 hash 的 D2/S2/S3/mixed/chunk 五资产 exact 与 V3.2 `36 passed` 只证明 immutable baseline
  仍成立，不证明 V3.3 方法有效。P0 全程无训练/模型推理；S1 必须另建 protocol、run 和指标证据。

## V3.2 终局处置与复开门禁（2026-08-11）

V3.2 已以 `WS-V32-R0-INTEGRATION-01=done`、整体 `none_plan_complete` 收口。归档位于
[`archive/2026-08/worldsim-v3.2/`](archive/2026-08/worldsim-v3.2/README.md)。下面的 `V3-F34`–`V3-F46`
继续约束任何后续路线，但不构成继续执行 V3.2 的任务清单。

| 分支 | 终局处置 | 禁止的延续方式 | 合法复开条件 |
|---|---|---|---|
| S1 semantic lift | `done`，canonical r6 | 复用 identity-invalid r5；绕过 ID/token/rigid 三元核对 | 新数据或新语义假设；新 task/protocol/run；继续 fail-closed identity |
| S2 background inpaint | `done`，canonical r3 | 把 Telea unseen RGB 当作 geometry/GT；复活退化的 r2 | 独立深度或多视图证据；预注册未观测 3D 门；新 run |
| S3 actor harvest | `done`，canonical r3 | 把生成背面写成 GT；倒写 CUDA preflight 失败的 r2 | 新 actor/方法假设；固定 source/weight/license；新 task/run |
| S4 harmonizer | task `done`，non-temporal `excluded diagnostic`；temporal `blocked` | 仅凭锐化或全图指标把删除区重生车辆纳入生产链；绕过 gated 权重 | 合法取得 gated base；显式 semantic preservation + temporal gate；新 task/protocol/run |
| S5 multiview upper bound | `blocked`，未授权 | 猜测许可证、移植无根许可证代码或把未执行写成质量结论 | 明确可执行许可证与权重；独立资源审计；用户重新授权的新 task |
| R0 integration | `done`，canonical r4 | 从 exact package 外推 streaming、跨场景、闭环安全或 GT correctness | 为对应 claim 增加独立数据、协议、测量与 run；不得续写 r4 |

统一复开规则：外部门禁解除只改变“是否可提出新任务”，不会把 S4 temporal 或 S5 自动变成当前任务。任何复开都
必须引用相关失败 ID，使用新 task ID、新冻结 protocol 和不可复用 run ID；旧 `blocked/rejected/done` terminal
保持不可改写。当前 `next_action=none_plan_complete`。

## V3.2 防重复结论（2026-08-10）

- `V3-F34`：actor role 必须同时绑定 dataset instance ID、`instances_info.id`/instance token 与 checkpoint
  rigid model index。只分别验证 class、token registry 和 core count 会允许“2D mask 属于 actor A、D2 core 属于 actor B”
  的静默错配。所有 prompt、semantic lift、asset generation 在运行前必须 fail-closed 核对三元 identity；旧 r5
  因 ID `5`/token `af663…` 错配而失效，不得通过后续 adapter 或人工选图补救。

- `V3-F35`：AutoDL 根 `.condarc` 可把 `nvidia` channel 重写到缺包镜像；第三方官方 setup 的 CUDA channel
  不能只凭 channel 名复现。Asset Harvester 必须使用 `--override-channels` 和明确的官方 NVIDIA/defaults URL，
  同时记录 setup 日志；TUNA 对应 404 不能被误判成官方包不存在。
- `V3-F36`：复制第三方 setup 脚本到 `/tmp` 后，基于脚本位置计算的 `REPO_DIR` 会静默变成 `/tmp`。
  transport-only patch 必须冻结真实 checkout 绝对路径，并让恢复 wrapper 显式接收 formal `RUN_DIR`；不得把
  环境完成结果写入已拒绝的旧 run。
- `V3-F37`：`gsplat` 的浅克隆或 transport-only 复制不会自动带上 GLM submodule。Asset Harvester
  环境不能只以 `pip install` 成功为准；必须固定 `gsplat` commit、初始化 GLM 到 exact commit，
  并在当前 GPU 上 import CUDA extension。
- `V3-F38`：第三方 setup 子进程里的 conda activation 不会传回父 wrapper。恢复脚本后续记录、
  校验或 formal 推理必须使用明确的环境 Python 绝对路径，不得依赖裸 `python`
  或父 shell 的隐式 PATH。
- `V3-F39`：PyTorch 2.10 下 `torch.cuda.manual_seed_all` 不会立即建立 CUDA context；在此前调用
  `reset_peak_memory_stats(0)` 可以在模型加载前报 `Invalid device argument`。资源监控 runner 必须先
  `set_device` + `cuda.init`，再清零峰值计数器。S3 r2 因此在 GPU peak=`0 MiB`、无部分输出时
  `rejected`；修复后必须新建 formal run，不得改写 r2。
- `V3-F40`：边界目标的时间邻近帧不等于存在静态世界几何重叠。S2 r1 的固定支持集合跨过 frame `31` 后，
  boundary mask 的有效跨视图覆盖低于 `32` 像素；放宽深度门也不能修复零几何重叠。后续只能在 train-only
  视图上冻结 exhaustive camera/frame geometry audit，再新建 run；不得读取 held-out 来选支持帧，也不得单纯
  放宽深度容差掩盖视锥不重叠。
- `V3-F41`：2D unseen completion 可生成不等于其深度足以作为全时段静态 Background。S2 r2 把全部未观测
  Telea 区域写入 3D checkpoint 后，四路 held-out 平均 PSNR/SSIM 退化 `0.495842 dB / 0.007160`，形成后续帧
  灰色遮挡；候选必须拒绝。S2 r3 保留完整 2D unseen artifact/provenance，但高支持 checkpoint 只持久化
  cross-view observed geometry，并重新通过未放宽的 held-out 门。后续不得把 inpainted RGB 自动升级为
  geometry-grounded world state；若要持久化 unseen 3D，必须增加独立深度/多视图证据和新预注册门。
- `V3-F42`：Harmonizer 导出 JIT 不是脱离官方 NGC runtime 即可原样执行的普通 TorchScript。当前权重包含
  `tex_ts::rmsnorm_fwd_inf_ts`，PyTorch 2.10 还会把两个 einops shape scalar 随 `map_location` 移到 CUDA，
  造成 shape tensor CPU/CUDA 冲突。当前适配只允许使用独立公式验证为 BF16 exact 的 RMSNorm 回退，并将整数
  1/2 shape scalar 放回 CPU；必须记录 runtime deviation 和测试，不能写成 untouched official runtime。
- `V3-F43`：生成式 final-render enhancer 可以恢复外观，同时破坏明确的 counterfactual 语义。S4 r2/r3 在
  G1 remove+inpaint 区域重新生成 actor-like 黑色车辆外观；r3 的 mask 内 L1=`14.217278`、changed fraction
  `0.541750`，失败冻结 `12.0 / 0.40` 门。全图 PSNR、outside drift 或“看起来更锐利”都不能覆盖 actor deletion
  失败；non-temporal Harmonizer 仅保留 optional diagnostic，不得默认进入 remove 输出链。未来复开必须增加
  显式 semantic conditioning/preservation 和连续帧 temporal gate，并取得 gated Cosmos base 的合法授权。
- `V3-F44`：DriveStudio 只读 forward 使用 `torch.inference_mode()` 不等于 trainer 已切换到 eval。训练态
  renderer 会对 `means2d` 调用 `retain_grad()`，与 inference tensor 冲突；R0 r2 因此在首个 forward 明确
  `rejected`。所有只读 replay 必须在每次 state load 后显式 `trainer.set_eval()`；失败 run 不得补写结果，
  只能修复代码后新建唯一 run。
- `V3-F45`：质量门必须冻结指标、区域、单位和范数；`MAE<=1 uint8` 不能被实现成逐像素 L∞/max-error
  `<=1`。R0 r3 的 source→mixed PSNR=`67.24–68.43 dB`、MAE=`0.0093–0.0123`，但两视角存在极少量
  max error=`2`，因错误合同仍保持 `rejected`。修复必须更新 protocol/runner hash 后新建 run，不能在旧结果上
  改 gate 或把 max error 隐藏。
- `V3-F46`：R0 `done` 只证明当前 scene 的 V3.2 资产可追踪集成与固定三视角 storage/package 等价。
  `GENERATED_BACKGROUND` 和 `GENERATED_ACTOR` 仍不是 GT；S4 仍被排除；S5 仍阻塞。432 MB mixed checkpoint、
  444 MB chunk payload 与 8.36 GiB 峰值也不证明 streaming、load、render、跨场景泛化或闭环安全收益；未来相关
  claim 必须有独立 protocol、数据与测量，不能由 R0 exact reassembly 外推。

- `V3-F18`：A3 R1 的工程链可逐位重放，但 heldout 评测越过冻结 GPU ceiling，且资源无效 diagnostic 为
  geometry 改善与 RGB safeguard 退化并存。后续不得提高旧 ceiling、替换旧 renderer 或继续调同一四步配方。
- `V3-F19`：传感器原始尺寸、checkpoint 原生加载尺寸和评测输出尺寸是三个不同合同。A4-P0 v1 的
  `1600×900` 误记不能在后续路线重现；每次 profile 必须同时记录三层尺寸和 downscale 来源。
- `V3-F20`：checkpoint `state_dict` 键不等于加载后的 runtime attribute。任何新资产注册、恢复或 streaming
  代码必须同时审计保存端、加载端赋值和 live object，不能从序列化 schema 猜运行时 API。
- `V3-F21`：A4-P1 最小预注册剪枝臂 b05 已违反 global/non-target 质量门。不得事后增加 b01/b02、放宽
  `0.10 dB` 或只挑 actor/boundary 指标，把同一结果改写成剪枝成功。
- `V3-F22`：A4-P2 只证明选择性 FP16 参数存储在冻结质量门内可把 checkpoint 减少 `25.35%`；它没有证明
  renderer、load、FPS 或 peak VRAM 加速，且 Gaussian means 不能安全降为 FP16。
- `V3-F23`：A4-P3 只证明 159-file static/actor chunk package 可 exact 重组。package 比 source 大
  `2.79%`、load/reassembly 更慢；没有 demand loading、cache policy 和驻留集测量时，不能声称 streaming/LOD 收益。
- `V3-F24`：F0 只完成 Instant NuRec 官方能力与本机前置审计；本机没有执行 inference，standalone CLI 只导出
  static PLY。不得把它写成前馈基线质量失败，也不得把 static PLY 当完整 dynamic WorldSim checkpoint。
- `V3-F25`：R0 的 63 inputs、23 decisions、12 deliverables 与 P3 package exact 只证明 V3.1 证据链闭环。
  它不证明 D2 dominance、R1/P1 有效、P2/P3 加速、完整 world model、跨场景泛化或闭环安全。

## V3 启动时必须先读的结论（2026-08-05）

- `V3-F01`：M4 的 non-target PSNR 93/95 dB 是硬局部编辑的构造/保持性证据，不是编辑后视觉质量。
- `V3-F02`：DriveStudio 已有 Affine、CamPose 与 LiDAR 初始化；A1 必须做 off/native/enhanced 消融，
  不得把上游能力改名为新增模块。
- `V3-F03`：V2 M5 未完成。0230/0242 checkpoint、Tier A/B/C 和 0255 诊断可复用，但不得把部分资产
  写成三场景压力测试通过。
- `V3-F04`：scene-0255 是小输入 CUDA `torch.cat` 工程阻塞且无 OOM 证据，不能写成 3DGS 方法失败。
- `V3-F05`：三个 scene 只支撑模型消融和工程判断；不得外推 trainval、夜间、长时或复杂交互。
- `V3-F06`：Instant NuRec 等工作已经改变前馈基线边界；DGGT 只作历史范式对照，不做跨分辨率、跨输入、
  跨训练预算的 leaderboard。
- `V3-F07`：persistent identity、actor binding、scene graph 和基础 trajectory edit 已由上游与 V2 覆盖，
  不能作为 V3 模型贡献。
- `V3-F08`：rolling shutter 需要真实 readout direction/time；没有 metadata 时必须 `not_supported`，不得
  从帧时间或相机顺序推测行曝光时间。
- `V3-F09`：actor-aware densification 必须分 D0–D3 小步消融；不得一次加入 boundary、LiDAR、visibility、
  residual 后只报告一个合并结果。
- `V3-F10`：编辑后 local refinement 的 unknown background 仍是 unknown；只允许 Tier-A、多视图或 LiDAR
  支持监督，Tier B/C 不得当伪真值回传。
- `V3-F11`：全图 PSNR/SSIM 不能代替 actor/边界质量；counterfactual mask 也不是真值分割，必须同时报告
  visible-image/pixel coverage，避免目标未渲染时通过缩小分母得到虚高指标。
- `V3-F12`：nuScenes processed camera ID 必须以数据加载器事实源映射；显示标签写错会把非相邻相机当成
  预注册相机对，已有 formal 必须 rejected 后重跑，不能只改图标题。
- `V3-F13`：seed=0 不保证 CUDA visibility filter 后的随机背景初始化逐点/逐计数复现。记录的 LiDAR/actor
  tensor exact 可作门禁，重建初始化 depth 只能作 witness，不能冒充源训练初始化 exact residual。
- `V3-F14`：局部 role、全图画质或 learned correction 稳定性改善不能替代预注册阶段主端点。C2/C3 未通过
  E1/E2 合同就不能为了保留增强模块而成为 C*。
- `V3-F15`：确认场景的原始端点方向可以与开发场景相反。不得把完整 Pareto 合同的 `done_off` 改写成
  “C0 在所有场景、所有指标都最好”，也不得只挑 0255 E1/E2 error 改写 C*。
- `V3-F16`：A2-D2 的边界改善、global/non-target 退化与更高训练成本构成严格 Pareto tradeoff。不得用新增
  事后标量权重把它改写成 D2 dominance；后续采用 D2 必须同时登记 D1 fallback 和完整退化轴。
- `V3-F17`：Gaussian ancestry、counterfactual footprint 和未提交 V2 M5 产物都不能自动升级为 A3 真值。
  ancestry 只证明来源，paired mask 只是模型诊断；A3 必须使用已提交输入和 typed support，S-C 保持 unsupported。

### V3-F01：局部保持不等于编辑质量

M4 的 lateral/delete non-target PSNR=`93.394483/95.598042`，主要来自编辑器只改变目标 actor 并保留其他
Gaussian。它证明实现没有意外改动非目标区域，但不能证明 source footprint 后方背景正确、actor 边界自然或
连续帧无闪烁。V3 必须把 outside preservation 与 Tier-A hole、depth ordering、boundary 和 temporal 指标分开。

### V3-F02：原生校准和初始化不能重复发明

DriveStudio `e59bda4` 的 `AffineTransform` 已输出 RGB affine，`CameraOptModule` 已学习 3D 平移和 6D 旋转
残差，数据集也已从 LiDAR 初始化背景/实例。A1 的合法动作是关闭/原生/增强的受控消融，以及 support provenance
审计；不能把启用原生 config 写成新成像、位姿或 LiDAR 模块。

### V3-F03/F04：V2 M5 部分证据与 scene-0255 工程阻塞必须分开

V2 M5 没有生成预注册的 24 条有效序列和 final matrix。scene-0230/0242 checkpoint 是有效训练资产；
scene-0255 训练则阻塞于 `datasets/driving_dataset.py` 实例点列表的 CUDA `torch.cat`。r27 观察到 166 个
CUDA float32 tensors、152 个 `(0, 3)` 空 tensor、177 scalars，且 `oom/oom_kill=0`。V3 A0 可以基于此做
最小 compatibility fix，但必须使用新 task/run，不能改写 M5 terminal，也不能由诊断完成推断训练完成。

V3 A0 已用 `436cfc1` 实现配对过滤：点与颜色按同一个 empty-row 条件过滤，全空时返回 prototype view。
canonical smoke `20260805T161656Z__scene0255-catfix-s0-r2` 在原生错误复现后完成真实 dataset init、1-step
优化与 checkpoint，说明该工程阻塞已在 smoke 范围解除。随后新 30k run
`20260805T162355Z__scene0255-native30k-s0-r1` 完成 checkpoint、registry 与 held-out 评估；0230/0242 通过
严格等价合同复用。该兼容问题现已闭环，但只证明工程修复和 A0 基线成立，不证明任何 A1/A2 方法提升。

### V3-F11：全图质量与模型差分 mask 都有明确边界

A0 中 scene-0242 全图 PSNR=`29.107`，高于 0230/0255，但其 high actor 区域 PSNR=`19.788`，反而是三场景
最低。scene-0255 boundary actor 区域 SSIM=`0.526`，也没有被全图 SSIM=`0.743` 反映。后续 A1/A2 不得只用
全图指标判断动态对象提升。

A0 actor mask 来自同一 checkpoint 的 original 与 actor-delete 配对渲染差分，是模型 counterfactual
diagnostic，不是 nuScenes 真值 segmentation。如果模型没有画出 actor，mask 会缩小；因此每个结果必须同时报告
candidate/visible image、effect pixel coverage 和 `ABSTAIN`。tight-crop LPIPS 用固定 8px padding 与 256px 输出，
不能和全图 DriveStudio LPIPS 混为同一指标。

A0 finalizer r1 因复用 checkpoint run 使用 `source_training_resources`、原生 run 使用 `train_resources` 而
`blocked`。这是汇总 schema 兼容失败，不是模型失败；`00ba4e8` 增加显式 provenance 归一化，r2 为唯一完成矩阵。

### V3-F05/F06/F07：结论规模与研究边界

三个固定 scene 足以比较相同数据、预算和实现下的 A0–A4，但不构成数据规模、天气、城市或交互分布覆盖。
Instant NuRec、OmniRe、IDSplat、SplatAD、ADGaussian、Real2Sim、RoVES 等工作分别覆盖前馈分层重建、
实例场景图、传感器和物理方向；V3 的价值来自完整复现、窄模型改动、负结果和工程 Pareto，而不是重新命名
已公开能力。只有 A2/A3 在至少 2/3 场景方向一致且资源稳定，才讨论扩展场景。

### V3-F08/F09/F10：禁止不可归因或无真值捷径

rolling shutter 没有 row timing 就不能实现；actor-aware densification 必须从 actor/background threshold 与
quota 开始，再分别增加 boundary/residual 和 LiDAR/visibility；local refinement 必须冻结 affected set 外参数，
并区分 expected/first-hit/measured depth。不得用 hard-composition outside=0、原图 actor 像素或未知区域的
自洽渲染作为方法成功证据。

### V3-F12：相机标签错误会污染跨相机端点

A1-E0 初版沿用了错误的显示顺序 `0=FRONT_LEFT / 1=FRONT / 2=FRONT_RIGHT`，但 DriveStudio nuScenes
事实源明确为 `0=FRONT / 1=FRONT_LEFT / 2=FRONT_RIGHT`。结果是名义上的相邻相机对可能实际落到
左右两侧非相邻画面，零支持也会被错误解释为模型现象。首次 formal
`20260806T140703Z__scene0230-c0-a1-e0-formal-full-s0-r1` 因此已标记
`rejected / INVALID_CAMERA_ID_LABEL_MAPPING`，原 terminal/manifest/summary 以 `*.original_done.json` 保留；
`d85ef27` 修复后 C0/C1 使用新唯一 run 回填。

防重复门禁：相机 ID/name 映射必须来自训练数据加载器或预处理权威列表，写入 resolved config 并纳入 hash；
QA 必须验证投影落在实际重叠的建筑/路面。若映射错误，所有受影响正式结果必须 rejected，不得通过重命名
已有 JSON、图片或曲线继续使用。

### V3-F13：随机 CUDA 可见性筛选不等于 exact 初始化 replay

A1 最小 LiDAR provenance 的 strict smoke
`20260806T142900Z__scene0230-a1-lidar-provenance-smoke1-s0-r1` 观察到：800,000 个背景 LiDAR 点、全部 24 个
actor point/color tensor、75,002 个 RigidNodes 初始点均 exact match，但随机 near/far 球面候选经过 CUDA
visibility filter 后，背景初始 Gaussian 数从源运行 946,484 变为 replay 的 946,597；后续 replay 又得到
946,309 和 946,291。这不是 LiDAR 输入变化，也没有训练或 checkpoint 修改。

防重复门禁：冻结的 `a1_lidar_provenance_v1.yaml` 要求记录 LiDAR/actor tensor exact match，并记录随机球面
候选、visibility mask SHA 和计数；背景 exact replay 固定为 `report_not_gate`，不允许事后设置“接近即可”的计数
容差。正式初始 depth residual 必须标为
`seed0_reconstructed_initialization_witness_not_exact_source_initialization`。要获得源训练初始化的 exact depth，未来
必须在训练创建时直接持久化 post-filter 初始化 tensors；A2 的逐 Gaussian ancestry 仍需独立 instrumentation。

### V3-F14：局部改善不能替代阶段主端点

scene-0230 中，C2 的 boundary-support E2 mean/P90 从 C0 的 `0.003547/0.006353` 改善为
`0.003346/0.005447`，但 high-support E2 P90 退化到 `0.011734`，actor/boundary LPIPS 也整体退化；因此不能把
单个 role 的改善提升为整个 E2 端点改善。C3 的全图 PSNR/LPIPS、boundary actor 质量和 learned pose correction
稳定性均最好，但 E1 median/P90 与两个 E2 role 仍未严格优于 C0。

A1-S0-v1 在结果已可见后、确认场景前把 V3.1 7.5 操作化为无容差严格 Pareto，并如实披露该时点；没有新增
事后数值阈值。正式结论必须是 `C*=C0-off / done_off`。不得更换 role、放宽端点、只引用 C3 全图画质或把
learned correction 幅值写成 pose GT，以强行保留增强模块。

### V3-F15：完整合同通过不等于每项指标方向一致

scene-0242 的 C0 在 global、E1 和 high E2 上优于 C1；scene-0255 则相反，C1 的 E1 median/P90 和两个 E2
role error 都更低。但 0255 C1 的 high E2 coverage 从 `23.529%` 降至 `21.569%`，boundary/high actor LPIPS
均退化，因而仍未通过冻结的“主端点改善、另一端点不退化、appearance LPIPS 可接受”完整合同。

A1 finalizer 的合法表述是：C1 在两个确认场景均不 eligible，C*=C0 保持 `done_off`，同时原始端点方向具有
scene dependence。禁止写成“C0 普遍校准更优”，也禁止忽略 coverage/appearance 只引用 0255 error 重选 C1。

### V3-F16：边界优先分支选择不等于全面方法提升

A2 formal 中，D2 相对 D1 的 boundary-support boundary-band PSNR/SSIM/LPIPS 从
`25.770024/.821572/.048382` 改善到 `26.171399/.828868/.044568`；但 global 从
`27.770024/.850915/.177704` 退化到 `27.703188/.850333/.178344`，non-target PSNR/SSIM 也下降，训练
wall time 从 `2099.33 s` 增至 `2720.82 s`。fixed 与 matched strict-quality、quality-cost 裁决都为
`tradeoff_non_dominated`，且 matched D2 只是 fixed 30k 的 exact alias，不是独立复现。

A3 采用 D2 是因为 A2 的预注册靶点包含 actor boundary，并且 D2 在该边界带三项指标同时改善；这是完整结果
可见后的工程资产路由，不是新增数值门槛、统计显著性或 D2 对 D1 的支配结论。任何后续报告都必须同时保留
D1 quota-only fallback，披露 D2 的 global/部分 actor/non-target/cost 退化，并禁止只摘录边界带结果宣称 A2
“全面提升”。单场景 scene-0230 也不能支持跨场景泛化结论。

### V3-F17：来源账本与 paired mask 不等于局部精修监督真值

D2 final checkpoint 的 Background ancestry 完整对齐 `1,205,164` 个 Gaussian，其中 `240,528` 个
`init_source=LIDAR` direct roots，其余含 random、split 与 clone；`nearest_lidar_distance` 对部分 lineage 有限，
但它是出生/父子来源记录，不是当前 target ray 的 T0 measured depth。A3 只能把 calibrated LiDAR projection 的
`depth_lidar_measured` 当 T0，把 first-hit 当 T1 ordering，把 expected depth 保持 diagnostic。

同样，source/edited footprint 来自同一 checkpoint 的 paired RGB difference，只能定位干预区域，不能充当真值
segmentation 或删除后的背景 RGB。S-A RGB 监督必须来自排除 target view 的 alternate camera/time 真实观测并有
calibrated reprojection；S-B 只使用 measured LiDAR 或至少两视图 geometry，禁止 RGB loss；S-C 不更新、不 seed、
不进入 loss，只报告 coverage/uncertainty/ABSTAIN。

当前工作树中的 V2 M5 protocol、`stress_metrics.py` 和 stress runner 均未提交且属于被冻结的用户工作，A3 不得
通过 import 或复制其结果建立隐式依赖。只能复用已提交并按 SHA 冻结的 M4 edit、paired mask、typed-depth 与
registry 接口；否则无法形成 clean source commit，也会把 V2 未闭环事实倒写成 V3 证据。

### V3-F18：A3 工程可重放不等于局部精修可晋级

A3 R1 已证明四个 S-B/T0 unit 的 opacity/scale 更新可以逐位重放，但可变集合只有 `51` 个 Background rows、
四个 unit 合计只有 `8` 个 T0 geometry pixels，S-A/RGB 为 `0/ABSTAIN`。heldout r2/r4/r5 的单 view 峰值稳定在
`14,241–14,245 MiB`，超过结果前冻结的 `12,288 MiB` ceiling；r5 的资源无效 diagnostic 同时出现 depth-order
改善和 non-target/original-global RGB MSE 严格退化，exact Pareto 为 `tradeoff_non_dominated`。

因此后续若复开局部精修，研究变量必须先变为“可观测支持如何获得、分层和拒绝”，而不是继续调 R1 的 step、
LR、alpha、mask dilation 或旧 renderer。合法复开需要新任务、新协议、与 heldout 隔离的支持审计，以及冻结前
证明 S-A 或更充分 T0/多视图证据确实存在；否则保持 `A3*=R0/D2 exact alias`。

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

### N1-F11：完整审核包不等于每项都有前视相机可见证据

**观察**

- 正式包包含 12/12 panels、evidence、topdown、checklist、prompt 和逐文件 SHA256；
- 本机 full train 数据完整覆盖 CAM_FRONT，但其他五个相机目录只有 mini 规模，不能为这些 formal scenes
  提供稳定六相机视图；
- 首尾面板 QA 正常；部分 subject/front/rear 不在 CAM_FRONT 视野，但 2 Hz annotation topdown 仍存在；
- 40 个 immutable audit files 复算 0 hash mismatch；空白 review validator 按预期 fail closed。

**审核边界**

不在 CAM_FRONT 中的角色不能被猜测为 VALID。评审先使用 topdown、vector centerline 和跨时刻 identity；
若相机与 annotation 冲突或证据仍不足，必须判 `UNCERTAIN` 并记录
`INSUFFICIENT_VISUAL_EVIDENCE`。补六相机或 raw LiDAR 需要独立资产/用途授权，不能偷偷进入本轮或 N2。

### N1 第三版历史禁止重试矩阵

| 快捷做法 | 为什么无效 | 合法后续 |
|---|---|---|
| 把 parent `AWAITING_HUMAN_REVIEW` 写成 pass | negative/pair 失败且人工 12/12 FP | 独立 adjudication 已 `REJECTED` |
| 用人工结果覆盖 pair gate | authenticity 与 comparison support 是不同问题 | 两类 gate 均保留 |
| 缩短/重叠 negative window | 事后改变 matched-control 定义 | 新任务、新 split、新预注册 |
| 用其他 actor 补足 same-actor pair | 混入 actor/scene confound | 预注册 matched-other-actor 设计 |
| 把 63 个 physical lane changes 加入 positive | 它们没有通过冻结 interaction | 单独 subject-only diagnostic |
| 用单侧 front 或 rear 算 interaction | 改变“插入双侧 gap”的研究对象 | 另立事件 subtype |
| 没有相机框也猜 TRUE/FALSE | 把证据缺失转成标签 | `UNCERTAIN` |
| 审核后自动启动 N2 | 本 run 明确 `n2_authorized=false` | 新授权 + 新 gate |

## N1 full-domain 第二次 reject（2026-07-25）

### N1-F05：把 target 多 incoming 的地图类别误当成 subject 行为

**观察**

- 父机器 run `N1-EVENT-FULL-01` 在 val 146 上报 37 个 positive，其中 topology 为 35 merge + 2 lane change；
- 完成人审文件 SHA256 为
  `ae71b31e02faf1d783c36748e629e85acf32a132f35ce2f98102a5f62201dd05`；
- 用户确认的逐项结果为 `TRUE_POSITIVE=2`、`FALSE_POSITIVE=35`、`UNCERTAIN=0`，机器候选精度
  `2/37=0.054054`；
- 多数 reviewer notes 明确指出：subject 沿与 target 共线的主路 lane/connector 正常直行，真正汇入
  target 的是另一条 incoming branch；旧规则却只因 `target_incoming_count>=2` 就把 subject 标为 merge；
- 独立 audit adjudication：
  `/root/autodl-tmp/runs/event_first/N1-EVENT-FULL-AUDIT-01/v71_n1-event-full-audit-01__human-audit-reject-v1__s0__20260725T083929632491Z__6507cbac/`，
  唯一终态 `REJECTED`。

**能下的结论**

graph-corridor 修复了邻车跨 token fragmentation，却没有证明 subject 本身执行了 lateral maneuver。
“target 有多个 incoming”是地图节点属性，不是 actor-specific merge 证据；第二次 N1 不能进入 N2。

**不能下的结论**

不能据此断言 full nuScenes 没有真实 lane change/merge，也不能把 2 个标注 TP 当成已验证的完整事件池。
旧 audit panel 没有把 subject/front/rear 的 3D identity 投影到图像，且 reviewer 字段包含多个来源；
用户已整体确认 reject，但单条 TP 仍只能作为第三版 calibration 标签，不得直接进入 formal evaluation。

### N1-F06：10 Hz 插值 cadence 被错误提升为物理证据

**观察**

nuScenes `sample` 是 2 Hz 标注关键帧。第二版把 2 Hz box 线性/SLERP 插值到 10 Hz 后，用连续 lane-token run
寻找 transition；该做法对齐了 DriveStudio cadence，却没有产生新的物理观测。第二次人审 notes 多次指出
短 token 切换、轨迹插值或 map assignment 假象。

**防重复**

- 第三版速度、加速度、yaw-rate、lane preference、front/rear persistence 只能用原始 2 Hz keyframe；
- 10 Hz 只用于 frame 对齐、可视化和复现旧 transition 候选，不得计算导数或宣称 0.1 s 观测；
- 至少 3 个 pre 和 3 个 post keyframes；不足时为 `UNKNOWN`，不得靠插值补齐。

### N1-F07：单时刻中心距不是持续物理交互

**观察**

第二版在单一 relation frame 上用中心线 `s` 与中心距 `[2,60] m` 选择 front/rear，没有扣除 box extent，
也不要求同一 front/rear identity 跨时刻持续。37 个 machine positive 中 36 个至少依赖一个跨-token 邻车，
因此 branch 选择错误会直接翻转结果。

**第三版解除条件**

1. target corridor 每个 graph edge 同时满足方向连续和 endpoint 连续，并只选单一最连续分支；
2. 使用 oriented box 在 lane tangent 上的投影半长，报告 bumper gap 与 center gap；
3. 至少 2/3 个连续 2 Hz keyframe 保持同一 front/rear identity、方向和次序；
4. 同时报告 longitudinal speed、closing speed、headway/TTC；它们是诊断，不得替代人审。

### N1-F08：第二版审核合同与 provenance 不足

**观察**

- 父机器 run 诚实记录 `code_dirty=true`；它可定位但不是 clean-commit formal baseline；
- 旧人审清单给了逐项 verdict 定义，却未预注册聚合阈值；
- 因此第二次 reject adjudication 没有查看结果后补造阈值，只登记用户明确决定；
- 旧 CAM_FRONT 清单没有身份 box overlay，容易把画面中“真正并道的另一辆车”认成 subject。

**复开条件**

第三版必须在 clean commit 上运行；正式 audit pack 同时提供盲序、subject/front/rear 颜色框、2 Hz 俯视轨迹、
逐项 component verdict、failure codes、完整提示词、immutable file hashes、预注册统计阈值和独立
adjudication 命令。Agent 不得填写人工 verdict。

### N1 full-domain 禁止重试矩阵

| 快捷做法 | 为什么无效 | 第三版允许替代 |
|---|---|---|
| 继续调 `graph_hops` 或 gap | 35/37 误报的主体事件本身不成立 | actor-specific 2 Hz kinematics 先行 |
| target 有多个 incoming 就叫 merge | 把地图节点类别当行为 | 比较 source 与主路 incoming 的 approach geometry |
| 用已审 val 37 条挑最终阈值并在同一 split 报结果 | calibration/evaluation 泄漏 | val 只 calibration；official train formal evaluation |
| 从 10 Hz 插值计算速度/横移 | 人造高频证据 | 原始 sample timestamp + 2 Hz boxes |
| 单帧 front/rear 中心距 | branch/identity 易跳变，忽略车长 | branch-safe corridor + temporal identity + bumper gap |
| 把 2 个旧 TP 直接当第三版正例 | panel identity 仍有未决风险 | 只作 calibration；第三版候选重新盲审 |
| 机器候选一出现就启动 N2 | 人工真实性与样本支持尚未通过 | `AWAITING_HUMAN_REVIEW`，`n2_authorized=false` |

## N1 mini event-pool reject（2026-07-24）

### N1-F01：interaction-support failure

**观察**

- N0 map-expansion、scene→map 与 pose contract 已通过，不再是资产缺失；
- 45 个 source-only eligible actors 产生 71 个 stable token transitions；
- topology taxonomy：39 route continuations、19 merges、3 lane changes、10 unresolved；
- 19 merges + 3 lane changes 共 22 个 topology-pass candidates；
- 22/22 的 exact-target-token front/rear relation 为 FAIL；
- 18 个没有 target-token 邻车，4 个只有 front、没有 rear；0 个同时满足 2–60 m front/rear；
- positive=0、negative pairing=0、same-actor pair=0、positive scenes=0，唯一终态 `REJECTED`。

**能下的结论**

冻结 mini split 不支持可比较 interaction event pool，N2–N5 不触发。地图缺失不是旧 H1 的唯一根因；
补地图后 mini interaction support 仍为零。

**不能下的结论**

不能写成“人类绝对看不到任何交互”或“full nuScenes 也没有事件”。exact target token 可能把同一
longitudinal corridor 上的 actor 分到相邻 lane/connector token；该表示风险尚未独立校准。

**复开条件**

mini run 不复开。新的路线必须：

1. 使用不同 run/task ID；
2. 以 22 topology-pass mini cases 仅作 calibration/audit，不作 formal evaluation；
3. 在 graph corridor 上定义 route-aligned curvilinear front/rear，而非后验放宽欧氏半径；
4. calibration 与 evaluation scenes 分离；
5. 优先在 full nuScenes trainval annotations/metadata 上冻结并评估。

### N1-F02：exact-token corridor fragmentation

**观察**

71 transitions 中 39 个只是 directed route continuation，说明官方 lane graph 将连续道路划分为多个
lane/lane_connector token。当前 interaction 只接受 relation frame 上与 subject 完全相同的 target token。

**推断**

该规则高精度但可能低 recall，尤其在 lane→connector→lane 或短 lane segment 附近。它是 0 interaction
PASS 的一个可能贡献因素，但不是已证实的唯一原因；mini 本身也可能确实缺少前后车。

**禁止快捷修补**

- 不把“相邻 token”全部并入；
- 不把只有 front 或只有 rear 改成 positive；
- 不把 82–89 m front 后验纳入 60 m；
- 不在同一 22 cases 上调 graph hops、gap 或 heading 直到出现 positive。

允许的修复是先定义有向 corridor、route-aligned `s` 和 branch disambiguation，再由独立 calibration
审计冻结；formal evaluation 必须 scene-disjoint。

### N1-F03：mini scale 与静止对象密度

**观察**

- 003/005/004 eligible actors 为 7/22/16；
- 因首尾位移不足 5 m 被拒的 actor 为 107/17/5；
- eligible pose map-match coverage 为 88.89% / 95.60% / 93.36%；
- 官方 full nuScenes 有 1,000 个约 20 秒 scenes，850 个为 train/val，而当前 formal pool 只有 3 scenes。

**结论**

mini 三场景对多 scene interaction event pool 的统计支持不足。下一步应扩数据底座，不应换 actor 或删场景。
优先同域 `v1.0-trainval` annotations/metadata，只有其 event gate 仍失败才评估 nuPlan/Waymo。

### N1-F04：negative=0 的语义

N1 只为已经有 positive 的 actor 构造 same-actor comparable negative。因此 `negative=0` 是
`positive actor set=∅` 的结构结果，不证明没有稳定非事件窗口。后续报告必须同时给出 positive actor 分母，
不得把 negative=0 解释为数据中全是事件或完全无普通驾驶。

### N1 禁止重试矩阵

| 快捷做法 | 为什么无效 | 允许替代 |
|---|---|---|
| 删除 rear requirement | 改变冻结 interaction claim | corridor calibration + scene-disjoint evaluation |
| 扩大 60 m 到覆盖 82–89 m | 看结果后调阈值 | 在新 calibration pool 依据任务时间窗冻结 |
| exact token 改成任意相邻 token | 可能跨 branch/对向车道误配 | directed corridor + route-aligned `s` |
| 从 22 cases 挑“看起来像”的 positive | 人工/后验标签泄漏 | 完整盲审协议；calibration 不进入 eval |
| 在 005 单 scene 继续 | 删除失败 scene、失去多 scene gate | full trainval scene-disjoint split |
| 直接启动 N2/N3/render | 没有 comparable event | 新 N1 先通过 |

## 0. H1 reject 执行摘要

### 0.1 为什么 reject

| ID | 层级 | 观察到的事实 | 能下的结论 | 不能下的结论 |
|---|---|---|---|---|
| `H1-F01` | 事件存在性 | 30 proposals：0 positive、25 negative、5 source-positive/non-event、0 same-actor pair | 冻结 proposal bank 不支持 H3 或配对因果比较 | “occupancy 一定无效”或“换几个 actor 就会成功” |
| `H1-F02` | certificate 精度 | D1 TP=15、FP=5、precision=0.75 < 0.80 | H1-CERT 按预注册 reject | 仅因 recall=0.8824 就称 certificate 通过 |
| `H1-F03` | certificate 覆盖 | D1 UNKNOWN=10/30、PASS=0、PASS coverage=0 | 当前证据无法给出足够确定的正判定 | 把 UNKNOWN 排除或并入 PASS 后重算 |
| `H1-F04` | repair 吞吐 | D2 reject=30/30、export=0、usable yield=0 | H1-PROJ 按预注册 reject；外部 rate 不可定义 | “导出集 0/0 违规，所以修复完美” |
| `H1-F05` | 数据效用 | 无 positive pair，H1 已拒绝 | H3 不触发 | 以 RGB 差分、accept rate 或 proxy 代替下游任务 |
| `H1-F06` | 高成本阶段 | H1 前置 gate 失败 | H2/render audit/blind pack 不实例化是正确停止 | “没跑 H2，所以 H1 结论不完整” |
| `H1-F07` | 统计实现 | 首版 aggregate 把 rejection 计成零违例 | 聚合 bug 已修复且不影响方法输出 | 用首版 aggregate 支持方法 claim |
| `H1-F08` | 资产/证据 | 本机 map 只有 raster PNG；base UNKNOWN 约 96–98% | lane/road support 与独立覆盖存在硬缺口 | 从 raster 或 learned occupancy 静默补成真值 |

### 0.2 冻结证据

- 正式 run：
  `/root/autodl-tmp/runs/occgs_resim/v71/V7-H1-11D/v71_v7-h1-11d__pilot-3-matched__s0__20260723T155755269940Z__cf8d5ebc/`；
- proposal-bank SHA256：
  `f8986915f8d2be0cddddfa6be86f4d2d1ece456c12bf9a962cafec78fd058cd7`；
- config SHA256：
  `cf8d5ebc1429e076fc5142aa6a759a18f54b7f3f937c8423d51505a094bc9fe3`；
- C/D1 realized trajectory 30/30 identity；
- C external hard violation 17/30：003=5/10、005=7/10、004=5/10；
- D1：15 TP、5 FP、2 FN（含 abstention）、20 FAIL、10 UNKNOWN、0 PASS；
- D2：0 accept/export、0 usable yield；
- 唯一 terminal marker：`REJECTED`。

### 0.3 逐失败点根因、卡点与复开要求

#### `H1-F01`：proposal-support failure

**观察**

- source-only eligibility 固定为 3 scenes × 2 actors；
- 每 actor 固定 P1–P5，共 30 个 proposal；
- scenario-effect 没有产生任何 `0→1` positive；
- 5 个 source-positive case 在 proposal 后成为 non-event，25 个为 negative；
- 没有 same-actor positive/negative pair。
- 后续只读 continuity 审计发现，冻结 actor 003:38、003:35、005:23 在完整连续 track 内的 world
  displacement 仅 0.88 / 0.29 / 0.76 m；原 source-only 排序偏好长、清晰 track，但没有事件相关性。

**推断**

固定横向位移只满足几何“移动过”，没有以 lane topology、corridor crossing、target-lane front/rear gap、
duration 或 interaction 定义事件。该设计与“cut-in/merge 正例”的目标错位。这是由结果和 schema 支持的
最强解释，但尚未通过 vector map 重标注，所以不能断言每个 case 的唯一失败原因。

**卡点**

- 本机缺 nuScenes map-expansion vector JSON；
- mini 三场景的真实事件上限未知；
- 没有先冻结 natural-event pool，就不能知道是 proposal family 失败还是场景本身无事件。

**复开条件**

不是改 P1–P5。必须新建 event-first 路线：先冻结 map/track 事件定义和 actor pool，证明存在预定数量的
positive/negative 与 same-actor pair，然后才允许生成候选。若 mini 事件池不足，应 reject mini pool 或
请求新数据授权。

#### `H1-F02`：certificate precision failure

**观察**

5 个 FP 全来自 scene 004 actor 8；certificate 报告 5 个 static-overlap voxels，而独立 raw LiDAR
检查为 0 points。D1 precision 为 0.75，低于冻结门槛 0.80。

**推断**

结果与 coarse voxel quantization、box-to-voxel 接触或证据层不一致相符；尚不能证明是哪一个机制，也不能
从“0 raw points”推出空间一定安全，因为 LiDAR 可能受遮挡和采样稀疏影响。

**卡点**

- `0.4m` 离散 grid 将连续几何压成二值接触；
- 单一 voxel overlap 缺少距离、置信度和观测支持；
- static/dynamic 分层仍可能受历史 sweep 和运动补偿影响；
- raw point absence 不是 free-space ground truth。

**复开条件**

在 scene-disjoint calibration pool 上比较 coarse voxel 与 motion-compensated raw sweeps 的连续
point-to-OBB/swept-volume distance；逐类报告量化、动态残影、遮挡、地图边界和标注误差。门槛必须在
冻结评估前预注册，不能用 actor 004:8 调到通过。

#### `H1-F03`：coverage/abstention failure

**观察**

三场景 base unknown 约为 97.10% / 96.04% / 97.57%；D1 10/30 UNKNOWN，PASS coverage 为 0。
两个 FN 位于 005 的 P3/P5，D1 known fraction 为 0，而 raw LiDAR 只有 3/2 points。

**推断**

当前 single/coarse observation 无法支持大部分 free/occupied 判定。两个 FN 说明“极少 raw points”
也不能自动解决判定；具体是遮挡、采样、时序或标注问题仍未知。

**卡点**

- raw LiDAR 稀疏；
- 缺 vector drivable/lane polygons；
- 多 sweep 若不做动态/ego motion compensation 会制造 ghost；
- learned completion 会提高表面 coverage，却失去独立真值身份。

**复开条件**

增加独立 evidence，而不是调低 known-fraction：官方 vector map、ego/dynamic compensated sweeps、
显式 truth tier 与 uncertainty。继续报告 PASS/FAIL coverage 和 abstention；任何 learned occupancy
只能是附加证据层，不能作为外部 evaluator。

#### `H1-F04`：repair all-reject failure

**观察**

D2 没有接受或导出任何 proposal；usable yield=0，外部 violation rate 无分母。

**推断**

当前 projection/repair 约束组合没有可用工作区，或者 proposal 全都离可行域过远。因为 0 export，无法
区分“repair 算法差”与“输入候选全不可修复”各自贡献。

**卡点**

- 没有成功样本用于 paired outcome；
- 先验 proposal 不由 lane-reachable set 生成；
- 二值 certificate 既可能过严又可能不准；
- H2/H3 都依赖 D2 产出，故被同时锁死。

**复开条件**

先通过 N1 证明事件存在，再以 lane graph/target state 生成 reachable proposal；冻结 minimum usable yield、
comparable export 数和外部 evaluator。若仍为 all-reject，直接 reject proposal/repair family。

#### `H1-F05`：metric aggregation bug

**观察**

首版 summary 把 rejection 计为零违例，使 0 export 看起来像 0% external violation。唯一允许的
`metric_aggregation_bug` 修复保留了旧 aggregate；修复后无 export 时 fail closed。修复提交为
`b82c540`，不改变 proposal、trajectory、certificate 或 D2 输出。

**防重复**

- 所有 rate 必须同时报告 numerator、denominator、rejected、unknown；
- denominator=0 时写 `undefined`，不能写 0；
- terminal decision 必须读取 comparable export 和 usable yield；
- 原始 aggregate 不覆盖，修复生成新版本并记录 migration。

#### `H1-F06`：地图资产与证据缺口

**观察**

`/root/autodl-tmp/data/nuscenes/maps/` 只有 4 个 PNG，没有 vector JSON；本机没有 Waymo/nuPlan 数据。
`/root/autodl-tmp` 约有 65G 可用空间。

**卡点**

官方 lane graph/drivable polygon 暂不可查询；不能可靠地定义 target lane、connectivity、off-road 或
corridor crossing。DriveStudio adapter 代码的存在不等于数据和许可就绪。

**复开条件**

先生成最小资产清单并取得下载授权；保存来源、许可、大小、SHA256 和 scene→map 映射。不得从 raster PNG
反推正式 lane graph，也不得静默下载全量 Waymo/nuPlan。

### 0.4 禁止重试矩阵

| 快捷做法 | 为什么无效 | 允许的替代 |
|---|---|---|
| 降 known-fraction / coverage | 把无证据改名为有证据 | 增加独立 map/raw evidence |
| UNKNOWN 并入 PASS/FAIL | 改变预注册语义和分母 | 继续三态并单列 coverage |
| 删除 S1、005 或 004 actor 8 | 后验删难例 | scene-disjoint 新 pool |
| 换 actor、方向、P1–P5 幅度 | 用结果挑 proposal | 先冻结 event definition 与 actor pool |
| 0 export 报 0 violation | denominator=0 | 报 undefined + yield=0 |
| multi-sweep 直接堆叠 | 动态物体会 ghost | ego/dynamic motion compensation |
| learned occupancy 当 GT | 方法与 evaluator 循环 | raw/map 独立 evaluator + calibration |
| GS floaters/画质当安全证据 | renderer 不是物理传感器 | GS 只在 N4 导出 |
| 先做 H2/H3/scale | 没有 comparable positive | N1–N3 先过门 |
| 重命名 OccGS 复开 | 没有解除原失败 | 新路线必须满足复开条件 |

### 0.5 可复用资产

失败不否定以下工程资产：

- coordinate contract、`WorldState`、typed label/depth；
- run contract、artifact index、terminal marker 和 fail-closed aggregate；
- object-centric GS reconstruction/renderer；
- D1/D2 接口和 `PASS/FAIL/UNKNOWN` schema；
- 冻结 H1 bank 作为负对照与回归 fixture。

复用这些资产不能继承 H1 claim；新路线必须有新 preregistration、独立 event pool 和 evaluator。

## 1. 仍直接约束 V7 的历史结论

| ID | 状态 | 对 V7 的约束 |
|---|---|---|
| `RF-05` | rejected | 合法轨迹/点或局部像素变化不等于 RGB、遮挡、source removal、depth、identity 与标签都合法 |
| `RF-06` | rejected | 局部 loss 或 mask 不保证参数/输出只在局部改变；必须测 outside、boundary、frame-0 与 held-out |
| `RF-08` | limitation | 可复现的机器 evaluator 不等于绝对物理真值，更不能替代人工 verdict |
| `RF-09` | rejected | same-scene、shared identity 或结构合法不等于人类能辨别方法收益 |
| `RF-16` | limitation | layout/trajectory controllability 不等于 action-disentangled actor physics 或数据效用 |
| `RF-18` | rejected | ReSim `exp0_no_carla` 的 E-vs-F action response 不足；V7 不得借归档重开 C1P/C1S |

其他 RF 仍完整有效，但当前 OccGS 计划不直接复用对应的 SVD projection/preference 配方。

## 2. V7 风险索引

| ID | 状态 | 风险 | 禁止的快捷修补 |
|---|---|---|---|
| `V7-RISK-01` | rejected_v71 | occupancy 已接入 11D，但 certificate precision 与 repair yield 均未过预注册 gate | 因为 occupancy 文件存在或 D2 无 export 就宣称 H1 通过 |
| `V7-RISK-02` | limitation | C0 24/24 是按效应 top-k 的机器筛选，不是用户人工评测 | 写成 human pass，或只报 top-k 隐藏 46/62 全分布 |
| `V7-RISK-03` | open_risk | L0 mask 来自 RGB 差分，outside=0 由 hard composition 构造保证 | 用 0 leakage 宣称 occupancy-guided completion 有质量收益 |
| `V7-RISK-04` | open_risk | U0 以极端 V4 为 naive 对照且没有下游任务 | 把 accept rate / RGB signal 写成优于 naive GS 或 mAP 收益 |
| `V7-RISK-05` | legacy_limitation | V7 既有 run 缺正式 manifest、resolved config 与终态标记；V7.1 新 run 已由 EV-10 fail closed | 事后猜 seed/fingerprint 或伪造 immutable provenance |
| `V7-RISK-06` | open_risk | 只覆盖 mini 三场景，S1 held-out 质量偏弱 | 先扩规模、只筛容易场景或把三场景外推为论文结论 |
| `V7-RISK-07` | interface_mitigated_v71 | 11C 已闭合 WorldState→renderer→typed-label 工程链；occupancy repair 的方法增益仍未验证 | 把 label-sync 工程通过写成 occupancy certificate/projection 通过 |
| `V7-RISK-08` | legacy_risk_mitigated_v71 | O0 坐标注释、metadata 与实际变换含义不一致；11A 已冻结显式 frame 合同 | 沿用含义不明的 `pose/T`，或在 round-trip 前计算 H1 指标 |
| `V7-RISK-09` | confirmed_mitigated_v71 | 旧 rotated-corner AABB 使 PILOT-3 动态体素量膨胀 1.72–2.83 倍；扁平语义不能诚实移除 actor | 把旧 O0 AABB 当正式安全几何，或移除 actor 后把体积恢复为 free |
| `V7-RISK-10` | confirmed_failure_v71 | 高 UNKNOWN 在 11D 导致 10/30 D1 abstain、D2 30/30 拒绝与 0 usable yield | 把 UNKNOWN 并入 PASS/FAIL，或降低观测门槛追求 yield |
| `V7-RISK-15` | architecture_mitigated_v71 | certificate detection 与 trajectory projection 若混组会混淆检测和修复收益 | D1 修改 C trajectory，或把 D1/D2 合成单一 validity 数字 |
| `V7-RISK-16` | confirmed_failure_v71 | 冻结 30-proposal bank 得到 0 个 0→1 positive 和 0 个 same-actor pair | 用位移幅度或 RGB 差分代替 scenario-effect gate，或事后换 actor |
| `V7-RISK-17` | confirmed_mitigated_v71 | 单一 `depth` 名称会混淆 expected、first-hit 与 LiDAR measured truth tier；11C 已强制分名和 sidecar | 把 expected depth 登记为 measured GT，或省略 validity/truth-tier |

## 3. 风险详情与解除条件

### V7-RISK-01：occupancy 尚未进入方法

**观察**

- `occupancy/build_scene_occupancy.py` 独立写出 per-frame grid；
- `resim/s0_trajectory_editor.py` 只检查横向运动学、yaw、actor/ego 距离和粗横向范围；
- `resim/c0_counterfactual_render.py` 改写 RigidNodes pose，但没有查询 occupancy；
- `resim/l0_local_completion.py` 用 V0/edited RGB 差分构 mask。

**边界**

O0 是有用的世界状态基础设施，但当前不能支持“occupancy 提高合法性”或“occupancy-guided completion”主张。

**解除条件**

按 `V7-H1-11` 建立统一 actor/state mapping，让 occupancy 进入 edit certificate、visibility 与标签重生，并对
matched kinematic-only/naive baselines 做非循环消融。只添加一次 occupancy lookup 或 post-hoc filter 不足以解除。

### V7-RISK-02：机器 top-k 不等于人工合法率

**观察**

- C0 全部可见 case 为 46/62 machine legal；
- 24/24 是按 mean edit effect 排序后的 top-24；
- 当前 `reviews/` 目录是机器面板与机器 JSON，没有用户填写的 verdict。

**边界**

可表述为“机器筛选 top-24 均满足当前规则”，不得表述为“24/24 人工合法”或用其估计全候选分布。

**解除条件**

先冻结 blind sample、逐项 rubric、失败优先级、JSONL schema 与聚合阈值，再由用户或指定评审者完成 verdict。
agent 不代填，也不以机器字段映射成人工答案。

### V7-RISK-03：L0 primary metric 目前是构造不变量

**观察**

hard composition 直接复制 mask 外的 edited GS，因此 outside-mask L1 必然为 0；当前 12 帧结果只验证实现遵守
公式。mask 由 RGB 差分阈值和膨胀获得，不包含 ray visibility、unknown/free 或 source footprint geometry。

**边界**

L0 只证明 local composition 工程可行。没有证据表明 Telea 改善视觉、时序、depth 或 identity。

**解除条件**

使用 geometry-derived disocclusion mask，并在有真值的 pseudo-hole 上比较 no completion、Telea 与局部生成；
primary 必须包含 inside quality、boundary、temporal、depth/instance，而不是继续调阈值追 outside=0。

### V7-RISK-04：U0 proxy 不识别数据效用

**观察**

`naive_V4` 是约 39–50 m 的强制横移负例；它被拒绝只能证明 validator 能识别一个极端错误。当前没有训练
detector、occupancy model 或 event classifier，JSON 明确记录 `u0_full_map_pass=false`。

**边界**

不能声称 OccGS 优于 matched naive GS、real-only 或提供下游增益。

**解除条件**

对相同 proposal、相同样本量和相同训练预算比较 R / R+naive / R+OccGS / R+OccGS+completion，并使用
scene-disjoint split、至少 3 seeds 和任务指标。三场景只可用于 pipeline smoke。

### V7-RISK-05：既有 run provenance 不完整

**观察**

`runs/occgs_resim/` 现有 B0/C0/L0/U0 目录未发现 `manifest.json`、`resolved.yaml` 或终态标记。B0 仍有
`config.yaml`、metrics、checkpoint；其他阶段有 JSON 报告，但不足以满足正式 run contract。

**边界**

现有数值可作为 retrospective evidence，不能声称是完整、不可变、可从 manifest 一键复现的正式 run。

**解除条件**

`V7-EV-10` 为既有证据生成显式缺失项索引；所有新 run 通过 fail-closed wrapper 产生完整协议。禁止事后补造
未知字段或覆盖旧目录。

**2026-07-23 缓解结果**

- `V7_EVIDENCE_INDEX.json` 已逐文件索引 B0/O0/S0/C0/L0/U0 的 1,610 个文件，并保留正式字段的
  `missing/unknown_not_inferred`；
- V7.1 run contract 对 run ID 复用、三层 hash、artifact bytes、summary、冲突终态标记和 optional
  `not_triggered` 分支 fail closed；
- 正式 smoke 在 commit `3590558` 上以唯一 `COMPLETE` 结束，25 项相关测试通过。

该缓解只约束 V7.1 新 run；V7 旧 run 的 provenance 缺口不可逆，仍保持 retrospective/legacy limitation。

### V7-RISK-06：场景覆盖与质量

**观察**

本机只有 mini 10 scenes 具备前向完整 sweep；feasibility 只使用 3 scenes。S1 test PSNR/SSIM 为 20.18/0.472，
明显弱于 S0/S2。

**边界**

当前结果不能外推到 trainval、长时、多相机、夜间或复杂交互；也不能只删掉 S1 后报告更好均值。

**解除条件**

H1 先在冻结三场景与 worst-case 上通过，再审计可获得的 scene-disjoint 数据。扩展必须保留困难场景分层、
真实/插值 provenance 与相同门禁。

### V7-RISK-07：标签链未闭环

**观察**

C0 已改写 RigidNodes pose 并输出 RGB/depth/rigid 分量，但尚未形成统一的 semantic、instance、2D/3D box、
occupancy 与 visibility regeneration 流水线。

**边界**

“label synchronization”当前只可称 proxy/interface 可行，不是完整传感器与标签一致性。

**解除条件**

同一 world-state record 驱动 renderer 与所有标签 writer，逐帧验证 pose、depth、mask、box 和 occupancy 共位；
对缺失/不可见标签 fail closed。

**2026-07-23 缓解结果**

- 11C 在 PILOT-3 的 V0/V1、三场景、三前向相机上生成 18 个样本和 432 个 typed sidecar；
- 独立审计验证 18/18 样本、6/6 WorldState hash、temporal identity、三相机覆盖、instance-depth z-order 与
  state-specific safety/observation/render-support 引用；
- expected、first-hit、LiDAR measured depth 分名，有限 semantic scope 和 visibility provenance 均写入 sidecar；
- S1 保留，正式 run 以唯一 `COMPLETE` 结束。

该结果只解除 renderer/label 工程接口风险；11D 之前仍不能声称 occupancy certificate 或 repair 有方法收益。

### V7-RISK-08：O0 坐标框架歧义已确认

**观察**

- `occupancy/build_scene_occupancy.py` 文件头将 grid 描述为首帧 ego-centric；
- `meta.json` 将同一产物描述为 per-frame ego-centric；
- 实际实现每帧读取 `lidar_pose/{t}.txt`，以其逆矩阵把 world box 变换到 grid，同时直接使用 sensor-local
  LiDAR 点。因此产物实际是 per-frame LiDAR-sensor grid，而不是首帧固定 grid，也不能在未审计 LiDAR-to-ego
  外参前简称 ego frame；
- DriveStudio 则以起始 `CAM_FRONT` 的 `camera_to_world` 逆矩阵定义 model frame。

**边界**

现有 O0 数值仍可作为 coarse retrospective evidence，但在显式记录 `T_grid_world`、`T_model_world`、
`T_world_camera` 并通过 world→model/grid→world round trip 前，不得用于 H1 合法性指标。

**解除条件**

`V7-H1-11A` 统一使用 `T_dst_src` 命名，修正新 schema/adapter 的 frame 声明，以 synthetic fixtures 和
PILOT-3 原始标定验证 translation、yaw、box corners、camera projection 及 checkpoint pose round trip。
旧 O0 文件不原地改写；正式 H1 evidence 产生新版本与新 fingerprint。

**2026-07-23 缓解结果**

- 11A 将 annotation/model/grid/camera/LiDAR frame 分别冻结为 world、start-CAM_FRONT、per-frame-LiDAR、
  `T_world_camera` 与 `T_world_lidar`；
- 三场景 1,679 个 actor poses 的 translation、rotation、box 和三前向相机投影 round-trip gate 通过；
- registry 跨独立进程重建 hash 完全一致，正式 run 以唯一 `COMPLETE` 结束。

旧 O0 metadata 不原地改写，故该风险仍是 retrospective artifact 的 legacy limitation；V7.1 后续模块必须引用
11A coordinate contract 和新 fingerprint。

### V7-RISK-09/10：AABB 膨胀与高 UNKNOWN 已确认

**观察**

- 在完全相同的 PILOT-3 raw annotation、grid 和 240 帧上，旧 rotated-corner AABB 相对 oriented-box
  center-inclusion 的动态体素量比分别为 003 `1.721×`、005 `2.249×`、004 `2.833×`；
- 分离 dynamic instance layers 后，base unknown 比例仍为 `97.10% / 96.04% / 97.57%`；
- source actor removal 后原体积恢复 UNKNOWN，不会恢复 FREE；edited layer 可独立 remove/insert，三场景未出现
  layer overlap；
- 缺少 nuScenes map-expansion polygons 时 road-support 与 off-road control 保持 UNKNOWN。

**边界**

11B 已消除 AABB 作为正式动态几何和扁平 layer 删除污染，但没有降低 observation sparsity。30 条可测真实
controls 的 retention 为 100%，collision/teleport 可检测负例为 2/2；然而加入 road-support 后 32 条完整
certificate 全为 UNKNOWN。这是诚实 abstention，不是 H1-CERT pass。

**后续约束**

D1 必须报告 precision、recall、abstention 和 PASS coverage；UNKNOWN 不进入 TP/FP/FN。只有独立观测或 map
证据能把 UNKNOWN 变为可判定状态，禁止通过调大 unknown threshold、把 box 当 background surface 或用 Gaussian
floaters 补 safety evidence。

### V7-RISK-15/16：certificate/projector 与 scenario effect 必须继续拆分

11B 已冻结 `scenario-effect-v1` 的纯 3D 0→1/0→0 gate、same-actor pair schema 和
`certificate-calibration-v1` 三态接口。11D 必须让 D1 逐字节复用 C trajectory，D2 才允许修改轨迹；位移 proposal
若未形成冻结的 corridor crossing、duration、gap 与 TTC/headway 条件，只能标为 non-event，不能靠命名成为
cut-in/merge positive。

### V7-RISK-17：typed depth 语义混淆已缓解

11C 把 depth 冻结为三个不同产品：diagnostic expected depth、T1 Gaussian first-hit depth、T0 LiDAR measured
depth；每个产品有独立 validity、definition、truth tier 与 artifact sidecar。独立审计确认三类各 18 个，且没有
expected-as-measured 混写。后续 export/evaluator 必须继续按产品名和 truth tier 消费，不能重新折叠成无类型
`depth`。

### V7-H1-11D：H1-CERT / H1-PROJ 预注册拒绝

**冻结事实**

- source-only eligibility 覆盖 3 scenes × 2 actors，P1–P5 共 30 proposals；S1 未删除；
- C/D1 realized trajectory hash 30/30 完全相同；
- D1：precision `0.75`、recall `0.8824`、abstention `0.3333`、PASS coverage `0`；
- C external hard violation `17/30`；
- D2：0/30 export、0 usable yield，external rate 不可定义；
- scenario-effect：0 positive、25 negative、5 source-positive/non-event，0 same-actor pair。

**裁决**

H1-CERT 因 precision 低于 `0.80` 拒绝；H1-PROJ 因拒绝全部 proposal、无 comparable export、usable yield
低于 `70%` 拒绝。按路线转向规则停止 OccGS 方法 claim，只保留 object-centric GS、WorldState、typed label、
certificate/evaluator 与 run-contract 基础设施。

**唯一修复与防重复**

首版聚合把 rejection 计成零违规，已作为 `metric_aggregation_bug` 唯一修复，旧 aggregate 保留。修复未改变
方法输出；第二版对无 export 的 rate fail closed。不得继续：

- 调低 known-evidence/coverage 门槛把 UNKNOWN 改成 PASS；
- 删除 005/S1 或 004 actor 8；
- 根据现有结果重选 actor、方向、proposal 或 event threshold；
- 用固定-pool `0/30 violation` 隐藏 D2 的 `30/30 reject`；
- 因 recall 达标而隐藏 precision fail，或把 UNKNOWN 排除后重算；
- 在当前配方上继续 H2/H3/scale。

## N1 receiver-centric cut-in final：第四轮后新增防重复项（2026-07-26）

### N1-F18：receiver branch merge 的 13/13 历史假阳性不能靠阈值微调挽回

第四版旧 parent 的 18 个 machine candidates 中有 13 个 receiver-branch merge；第四轮人工裁决表明这类
历史 branch-merge 候选均为 `FALSE_POSITIVE`。因此 branch topology、`target_incoming_count`、shared successor
或 token change 永远不能单独证明 cut-in。final v2 把该类别固定为
`ABSTAIN/UNSUPPORTED_BRANCH_MERGE_MODE`；不得为了候选数量重新放宽它。

### N1-F19：support-count 不能替代完整 receiver identity 时序

K4-012 暴露了“support 数量足够”仍可能跨 raw 帧切换接收车身份的问题。legacy fixture 的 `1→38` 与 v2
raw map 重放中观测到的 `9→1→9` 是不同窗口/枚举证据，二者都不能被静默等同为连续 receiver。final v2
要求 required raw frame 全窗唯一 non-null identity、last-post anchor 和每帧 rank/gap/path-clear；任一身份
切换必须 FAIL 或 ABSTAIN，不能被总 support count 抵消。

### N1-F20：弯道 map jitter 的 post heading 不能借由宽松窗口穿透

K4-015 证明 source/target 局部不平行或 post heading 过大时，几何横向收敛可以伪装成切入。final v2 使用
local parallel overlap、raw post-heading、累计 yaw 和 raw-only provenance；它不是针对 scene/token 的黑名单。
禁止为了保留 K4-015 或增加 PASS 数而放宽这些 hard gate。

### N1-F21：CAM_FRONT 的五帧截图不能承担角色/时序的完整证明

单相机可见性和五帧页面截断无法可靠展示 SUBJECT、RECEIVER、source/target corridor 的完整 raw 窗口。
审核 V2 因而以逐 raw-frame topdown、2 Hz signals、actor-ID switch 标注和固定 camera-unavailable 警告为主；
相机只作可选证据。看不清必须 `UNCERTAIN`，不得肉眼猜身份或通过下载未授权传感器补洞。

### N1-F22：final v2 不是旧阈值的第五次微调

本轮只吸收第四轮已完成的校准信息，变更的是事件语义和证据链：parallel-only subject body entry、独立
receiver 全时序、raw 2 Hz hard evidence、三态 first-failure、streaming worker 与 blind/debug 分离。K4 只做
固定 regression；Resource Contract V1 在任何 final scene 前失败，用户复开后的 V2 已按 N1-F25 修正为
675 scenes 并完整运行，但同样没有用于调参。后续若研究 branch merge 或新资产，必须是新的任务 ID、
预注册与 scene-disjoint 评估。

### N1-F23：共享 cgroup 的 start 合同是研究终止门，而非可绕过的工程告警

final formal 在 clean commit `7104f5c` 的 preflight 已将 runner 自身 RSS 降至 `20,705,280` bytes，仍记录
`cgroup_memory_current_bytes=1,523,929,088`，超过冻结上限 `1,350,000,000`。它在任何 evaluation scene 前
安全失败；独立裁决冻结证据并以 `REJECTED/stop_nuscenes_cutin_mining` 结束。development override 的 32/96
smoke、清页缓存或 K4 回归均不能替代正式 start 合同。禁止杀死 Cursor/Jupyter/TensorBoard 等用户服务、修改
正式阈值、截断正式 split（当时 expected 常量误写为 669，见 N1-F25）或把这次结果说成“nuScenes 没有
cut-in”；`n2_authorized=false` 保持不变。

**2026-07-26 用户复开授权**

上述 `REJECTED` 仍是 Resource Contract V1 下不可改写的历史裁决，但不再代表任务永久停滞。用户随后显式
扩大容器内存并授权继续本次 final：现场复核 `memory.max=128,849,018,880` bytes（120 GiB），原
`2,147,483,648` bytes（2 GiB）资源前提已改变。因此必须保留失败 parent 与独立拒绝裁决，同时使用新的
Resource Contract V2、全新 config fingerprint 和不可复用 run ID 恢复 scene-disjoint formal
（经 N1-F25 确认为 675 scenes）；不得覆盖或续写
V1 失败目录，也不得把 V2 成功倒写成 V1 当时没有失败。

### N1-F24：内存不足时停止并等待资源授权，不继续死磕

**新执行规则**

1. 任何正式或开发任务若触发启动/运行 stop 阈值、`RC=137`、SIGKILL，或观察到持续逼近 cgroup
   `memory.max`，立即停止启动新 batch，并尽最大可能写入结构化 `FAILED/failure.json`、最后完成 scene、
   process RSS、cgroup current、anon/file cache 与 `memory.events`；
2. 不通过反复重跑、杀死 Cursor/Jupyter/TensorBoard 等用户服务、缩短正式 split、降低证据质量、跳过
   audit、修改研究阈值或清理不属于本任务的缓存来争抢资源；
3. 把失败点、最低所需资源和恢复命令回报用户，然后等待用户开放资源；没有新的明确授权时不得自行恢复；
4. 用户开放资源后，先记录新的 `memory.max/current` 和授权时间，版本化 resource contract，使用新 run ID
   从冻结研究配置重新运行。资源合同变化只允许调整资源阈值，不允许调整 cut-in taxonomy、hard gate、
   calibration/evaluation split、抽样或人工聚合门槛；
5. 资源暂停与研究拒绝分开登记。未读取 prospective evaluation scene 的资源失败不能被写成方法精度失败，
   后续在新授权下完成的结果也不能删除或覆盖先前工程失败证据。

### N1-F25：final 的 669-scene 预期是 split 算术错误，不是 evaluation 集合定义

Resource Contract V2 的首次 clean formal run：
`/root/autodl-tmp/runs/event_first/N1-EVENT-CUTIN-FINAL-01/v71_n1-event-cutin-final-01__receiver-cutin-final-v1__s0__20260726T142634031503Z__5c8c65d7`
在 K4 regression 通过后、任何 evaluation scene 或 candidate 读取前 fail closed，错误为
`final evaluation scene 数不匹配: 675 != 669`。

独立复算表明：nuScenes official `train` 为 700 scenes；冻结的 42 个 calibration scenes 中，25 个属于
`train`、17 个属于 `val`，且没有 split 外 scene。因此 scene-disjoint evaluation 的确定数量是
`700 - 25 = 675`。`_resolve_evaluation_scenes` 已正确执行
`set(train) - set(all_calibration_scenes)`；错误只在 YAML 的 `expected_scene_count` 常量把不属于 train 的
calibration scene 也错误计入了减法。

合法修复仅为把 Resource Contract V2 配置中的 assertion 从 669 改为 675，并生成新 config fingerprint、
新 clean commit 和新 run ID。不得借此增删 calibration scene、显式挑选 evaluation scene、查看 candidate 后
改 split，或修改 taxonomy、strict gate、K4、抽样与人工门槛。失败 run 必须保留为工程契约失败，不能统计成
research reject。

### N1-F26：strict v2 在 675 scenes 上只有 1 个 PASS，不能靠人审单例或放宽规则扩池

Resource Contract V2 的 clean formal run：
`/root/autodl-tmp/runs/event_first/N1-EVENT-CUTIN-FINAL-01/v71_n1-event-cutin-final-01__receiver-cutin-final-v1__s0__20260726T142941598714Z__883fae9a`
在 commit `beee1de`、seed 0、config fingerprint
`883fae9a6514c0bff5bba8bcaf81a22c79e6d719586221596a7d4b5364c337da` 上完成 675/675 scenes。

结果为 `ABSTAIN=1,556`、`FAIL=200`、`PASS=1`，唯一 PASS 只覆盖 1 scene；冻结 machine-readiness 要求
至少 3 candidates / 3 scenes，故 parent 以
`REJECTED / stop_nuscenes_cutin_mining_too_sparse` 结束。K4、raw-only 和资源合同检查均通过，峰值 batch
process RSS 为 `337,154,048` bytes、cgroup current 为 `4,556,898,304` bytes；这次拒绝不再是资源失败。

独立稀疏终局人工包保留唯一 PASS 和 3 条 diagnostic，但人工结果不能改变数量门失败：即使唯一 PASS 被判为
TP，仍只有 1 TP / 1 scene，低于 sparse 的 3/3。禁止为了形成池而把 ABSTAIN 提升为 PASS、恢复
`receiver_branch_merge`、放宽 raw/parallel/receiver 时序门、事后改 scene 或把单例人工真实性外推为总体
precision。准确结论是“当前冻结 strict v2 的 prospective pool 过稀”，不是“nuScenes 没有 cut-in”。

<a id="detail-cross-route"></a>

## 4. 跨路线必须保留的原则

1. 先证明监督/比较对象存在，再训练或扩量。
2. occupancy、编辑、渲染和标签必须共享同一显式状态，不允许旁路文档绑定。
3. matched baseline 使用相同 proposal、scene、actor、幅度、seed 与预算。
4. top-k 只用于诊断，不替代全分布、coverage 与 worst-case。
5. machine pass 只解锁下一门禁，不自动成为 human verdict、论文 claim 或 scale 授权。
6. hard composition 的局部性与 completion 的质量是两个独立门禁。
7. 下游效用必须由任务指标证明，不能由约束 accept rate、RGB 差分或 PSNR 代替。
8. 工程失败与 research reject 分开登记；既有 provenance 缺失必须诚实标记。
9. 失败范围不能过度外推，但也不能通过改名、放宽阈值或只挑成功场景重复旧问题。

## 5. 新实验防重复检查表

- [ ] 是否明确引用了相关 `RF-*` 与 `V7-RISK-*`？
- [ ] occupancy 是否真正进入决策/状态链，而非只在磁盘上存在？
- [ ] baseline 是否 matched，而非故意构造的极端负例？
- [ ] primary endpoint 是否避免“方法规则自己定义方法成功”的循环论证？
- [ ] 是否同时报告全分布、coverage、per-scene 与 worst case？
- [ ] completion 是否测 inside quality/temporal/depth，而非只测 outside exact？
- [ ] human verdict 是否只由用户/指定评审者填写？
- [ ] run 是否有唯一 ID、resolved config、fingerprint、metrics、summary 与终态标记？
- [ ] 哪个单卡门禁失败时停止，什么条件才允许 scale？

## 6. 2026-07-26 路线转向新增失败与防重复项

### PIVOT-F01：nuScenes cut-in 没有可验证的召回率分母

**观察**

nuScenes 官方公开的是场景、样本、对象实例、类别、属性、3D 框、传感器与地图等结构；`scene.description`
是自由文本，不是事件级 cut-in 真值。官方没有发布 cut-in 场景占比、逐事件标签或可直接用于召回率计算的全集分母。
四轮挖掘和最终 675-scene prospective run 最多只能测量“冻结规则产出的候选质量”，不能测量“数据集中所有
cut-in 被召回了多少”。

**最终证据**

- `N1-F26`：strict v2 在 675 个 scene 上只有 `1 PASS / 1 scene`；
- 最终人工稀疏包即使把唯一 PASS 判为真阳性，也仍低于预注册的 `3 candidates / 3 scenes`；
- 该结果不能外推为“nuScenes 没有 cut-in”，也不能用事后放宽规则伪造召回。

**裁决**

`cut-in mining` 状态固定为 `rejected / frozen`。以后 cut-in 只允许作为已经具备重建与编辑能力后的可选演示，
不再承担数据集入口、方法定义、训练前置条件或论文成立条件。

**解除条件**

只有新的、独立的数据源提供事件级真值及明确分母，或新的任务本身不需要宣称事件召回率，才允许创建全新任务 ID
重新讨论；不得恢复当前 strict-v2 阈值调参。

### PIVOT-F02：贡献漂移——工程系统吞噬了重建与编辑研究

**观察**

过去路线的主要投入逐步变成事件挖掘、地图匹配、接收车身份、规则校准、候选审核与资源合同。它们改善了审计性，
却没有自然回答动态对象几何、连续运动表示、遮挡/去遮挡、反事实轨迹编辑或下游感知一致性。

**边界**

这不否定已形成的 WorldState、typed label、run contract、审计与人工审核基础设施；它只否定把“更好的 cut-in
挖掘器”作为 3DGS/4D 重建论文的核心贡献。

**后续约束**

新路线必须先复现公开强基线，再通过重建/编辑压力测试选择创新点。数据工程模块只能服务于冻结实验，不得重新成为
论文主任务。每个里程碑都必须说明它直接回答的重建或编辑问题。

### PIVOT-F03：未完成 exact reproduction 前禁止集成式“改进”

**观察**

`RF-05/06/08/09/16/18` 与 `V7-RISK-03/04/06/07/10/16/17` 共同表明：输入、状态、比较对象、覆盖率和真值定义
未冻结时，模块堆叠会把工程可运行误当成方法收益。AD-GS 的公开 nuScenes 协议提供了固定 scene、帧区间、预处理、
训练和评测入口，适合作为新的事实锚点。

**禁令**

在 AD-GS exact reproduction 门禁通过前，不得：

- 合并 Motion-Proj/StreetGS/OccGS 模块；
- 加 occupancy、物理约束、扩散补全、感知损失或轨迹编辑；
- 更换为自选事件场景、调低分辨率后对齐论文指标或只展示成功帧；
- 把兼容性补丁、预处理修复或运行成功表述为方法改进。

任何 unavoidable compatibility patch 必须独立提交、最小化、附 upstream diff 与消融；原始基线结果必须保留。

### PIVOT-F04：不能把“可见性建模”泛化成未观测背景已经解决

**观察**

AD-GS 的双向时间可见性用于动态对象生命周期和已观察运动建模；VAD-GS（CVPR 2026）的 visibility-aware
densification 已覆盖稀疏观测下的几何补密；DrivingEditor 支持对象删除/添加；Real2Sim 进一步展示对象级编辑与
物理交互。这意味着“增加一个 visibility 模块”或“支持平移对象”本身已经不足以构成新意。

**仍未闭合的问题**

反事实轨迹编辑会同时制造原位置去遮挡、新位置遮挡、跨相机深度排序和证据外外观。当前项目只有在下列内容形成
联合、可验证方案时才可提出方法 claim：

- 编辑诱发的显式可见性重计算；
- 未观测区域的真实性/置信度与拒绝机制；
- 跨视角、跨时间一致的背景恢复；
- 目标区预期变化与非目标区感知保持。

**防重复**

创新选择前必须把 AD-GS、VAD-GS、DrivingEditor、DGGT/ReconDrive 和当时最新工作重新做一次代码可用性与
claim 边界审计；不得把已有 visibility-aware densification 或基本对象变换重新命名为贡献。

### PIVOT-F05：资源不足时研究停机规则跨路线继续生效

`N1-F24` 是项目级规则，不属于 cut-in 专属逻辑。本轮 cgroup 为 `memory.max=2,147,483,648` bytes，轻量元数据
审计后 `memory.current` 一度达到 `2,129,526,784` bytes，因此立即停止 Python 扫描、conda 求解、下载、
预处理和训练，只继续轻量文本/文件操作。后续任何新路线任务遇到相同条件时，必须保存失败/现场证据并等待用户开放
资源；不得反复重跑、杀用户服务或偷偷缩减正式协议。

### PIVOT-F06：旧机器 smoke 证据不能替代新实例复验

迁移到 RTX 4080 SUPER 新容器后，已有环境目录和旧 RTX 4090 日志仍然存在，但它们不能证明当前 driver、CUDA、
扩展 ABI、显存与 cgroup 合同可用。M2 因此在新机器上重新执行 AD-GS forward/backward、DPT、SAM2、
Grounding DINO HF 和 CoTracker3 smoke，并为每项保存独立退出码。

后续换机或容器重建时，即使复用同一 env/checkpoint，也必须生成新的 instance 级环境证据；旧日志只能作为历史，
不能复制为当前 PASS。

### PIVOT-F07：非 login shell 的 PATH 不能作为 CUDA provenance

M2 首次当前机器采集因非 login shell 找不到 `nvcc` 提前失败，而 `/usr/local/cuda/bin/nvcc` 实际存在。环境报告
已改为显式设置 `CUDA_HOME` 并调用绝对路径，同时传播 smoke 的真实退出码。

后续自动任务必须显式记录并使用 toolkit 路径；“命令不在 PATH”与“机器没有 CUDA toolkit”必须分开裁决。

### PIVOT-F08：在线浮动模型不能进入 exact reproduction

upstream 的 CoTracker `torch.hub` 在线 `main` 与未固定 revision 的 Hugging Face 模型都会随时间变化。M2 将
CoTracker repo、离线 checkpoint、Grounding DINO HF revision 与 snapshot fingerprint 全部固定并哈希，
运行时使用 offline mode。

后续 baseline 不得在正式 run 中联网追随 `main`、latest 或未固定 snapshot；若必须升级，使用新 config
fingerprint 和新 run instance。

### PIVOT-F09：tar 页缓存与 nuScenes auxiliary 都属于资源/资产合同

并行流式扫描约 294 GB tar 时，文件页缓存计入本容器 cgroup，首个扫描实例峰值达到 `57,001,484,288` bytes。
本任务只对自己已读过的 tar 文件区间调用 `POSIX_FADV_DONTNEED`，没有全局 `drop_caches`、杀用户服务或清理
其他任务缓存。

第一次结构审计还发现 1,440 个 RGB/LiDAR payload 齐全并不足以初始化 nuScenes devkit；`map.json` 引用的
4 个静态 map masks 同样是必需资产。失败实例
`20260727T165549__e49a4e-4080s-r2` 保留为 `blocked`，补齐并哈希登记 maps 后由新实例
`20260727T180733__e49a4e-4080s-r3` 通过。以后 selective extraction 必须同时审计运行库隐式依赖的 auxiliary
文件，不能只按训练脚本直接打开的传感器路径计数。

### PIVOT-F10：AD-GS 的 PNG 输出与 SAM2 的 JPEG-only 枚举不兼容

M3 首个 scene-0230 实例完成 `prepare_raw` 和 180 张 depth 后，在 sky mask 初始化时报
`no images found`。AD-GS `scripts/nuscene/nuscene.py` 固定写 `000000.png`，其 `semantic.py` 自己也会枚举
PNG；但 Grounded-SAM-2 `load_video_frames_from_jpg_images` 只按 `.jpg/.jpeg` 扩展名建立 video frame 列表。
这不是空数据、模型失败或资源 OOM。

失败实例 `20260727T181617__scene0230__s0` 保留为 `blocked`。最小兼容性修复只在 instance work dir 中为每个
PNG 建立相同字节内容的 `.jpg` 硬链接，跨文件系统时复制原始字节；PIL 已验证按内容可无损读取，不做 JPEG 转码，
不改 Grounding/SAM 模型、box/text 阈值、mask、帧序或评测。修复后的 AD-GS patch SHA-256 为
`114c3976af2c80d1da5581b401b3a099f22a7483347fc401113c8439bc991eb9`，必须由新 M3 instance 复验。

### PIVOT-F11：COLMAP 默认全核并发会越过本机 cgroup 内存门禁

M3 第二个实例 `20260727T182247__scene0230__s0-r2` 已完成 180 张 depth/object/sky/semantic 与
138/138 flow，在 COLMAP feature extraction 阶段发现 upstream 未指定线程数，COLMAP 自动使用容器可见的
128 个 CPU threads。cgroup memory 峰值达到 `62,265,835,520` bytes，并连续两个采样超过
`memory.max=66,571,993,088` 的 90% 停止线；runner 只终止本 run 的进程组，`oom=0 / oom_kill=0`，
失败实例与部分 COLMAP 目录均保留。

这不是图像、SIFT、匹配或几何协议失败。最小资源兼容修复显式传入
`SiftExtraction.num_threads=16` 与 `SiftMatching.num_threads=16`；不改分辨率、相机、帧、SIFT 参数、
exhaustive matching 或评测。r3 的 COLMAP 已完成 138/138 图像注册与 70,933 points，阶段峰值降至
`35,117,174,784` bytes。当前完整 compatibility patch SHA-256 为
`49b4c06ecec6c30f1e80b5abf4d46970920f9d71952acbda273774d9b5b34f48`。

以后在 CPU 核数远大于内存预算的容器内运行 COLMAP，必须显式登记并发数并纳入资源合同；不得把减少图像、
降低分辨率或删相机伪装成等价的资源修复。

### PIVOT-F12：跨场景连续执行时，official render 可先于 OOM 触发 cgroup 90% 合同

M3 scene-0230 的 60k render 峰值已达到 `59,530,678,272` bytes，距离注册的 90% 停止线仅
384,115,507 bytes。M4 scene-0242 严格串行完成全部 preprocess 后，100-step train 峰值为
`59,359,428,608` bytes；随后的 official render 在第 2/138 帧连续两个采样达到 90%，峰值
`59,996,393,472` bytes，比停止线高约 81.6 MB。runner 只向本 stage 进程组发送 `SIGTERM`，
stage `rc=-15`、runner `rc=1`，`oom=0 / oom_kill=0`，没有影响其他服务。

该结果说明当前 `memory.max=66,571,993,088` bytes 对六场景连续 exact reproduction 没有足够安全余量；
“尚未 OOM”不能用来绕过预注册停止线。blocked 实例
`/root/autodl-tmp/runs/dynamic_recon/DR-M4-ADGS-6SCENE-01/20260727T235743__scene0242__s0/`
必须保留，不在同一资源合同下立即重跑，也不得降分辨率、删相机、调模型或全局 `drop_caches`。

恢复时应先提高 cgroup 内存额度，建议至少 80 GiB、推荐 96 GiB，再创建新 instance；允许复用逐文件冻结的
processed scene，但必须记录来源、哈希和新资源合同。若无法增加资源，则 M4 保持 `blocked`，不得将只有
scene-0230 的结果写成六场景论文复现。

### PIVOT-F13：processed scene 复用校验必须区分关键产物与合法空占位

RTX 3090 换机后的首个 scene-0242 复用实例
`/root/autodl-tmp/runs/dynamic_recon/DR-M4-ADGS-6SCENE-01/20260728T131533__scene0242__s0-r2-wm3090/`
在训练前 fail closed：新增递归哈希校验把 COLMAP 的合法 0-byte `created/sparse/model/points3D.txt`
占位文件误判为损坏。旧实例以 `blocked` 保留，没有修改 processed scene 或启动训练。

合法修复允许 COLMAP 非关键占位文件为 0 bytes，同时继续强制 `database.db`、`cameras.txt`、
`images.txt`、`colmap.ply` 和所有训练直接消费的 image/depth/mask/flow/meta/point cloud 非空；
复用后必须重新运行独立 processed audit。修复后的新实例 output fingerprint 为
`32bf9ccaa108273b69286625a0c7aaacb04fd9d76f243daff976206d0b7ef4f6`，138/138 registered images
审计通过。不得删除空占位文件、伪造非空内容或因此重跑昂贵预处理。

### PIVOT-F14：容器实例重建必须与 OOM/方法失败分开

M5 首个正式 DGGT 实例
`/root/autodl-tmp/runs/dynamic_recon/DR-M5-DGGT-NUSC-01/20260729T094923__native-nusc-s0-wm3090/`
在 `env_torch` 下载期间停止。日志没有 stage 终态、OOM 或 `oom_kill` 增量；当前容器 PID 1 的启动时间为
2026-07-29 13:13:43 +08:00，晚于日志停止时间，故裁决为外部容器实例重建，而不是 DGGT 精度、显存或方法失败。

旧 run 与旧 controller 已原子标为 `blocked`，部分环境移动到
`/root/autodl-tmp/envs/dggt.interrupted-20260729T094923/`，没有覆盖或删除。恢复使用新 run
`/root/autodl-tmp/runs/dynamic_recon/DR-M5-DGGT-NUSC-01/20260729T133346__native-nusc-s0-wm3090/`；以后同类中断必须先核对 PID 1 启动时间、stage marker、launcher 与 OOM 证据，再决定是否创建
新实例，禁止把 stale `running` 当成任务仍存活。

### PIVOT-F14B：pointops2 的 PEP 517 build isolation 没有继承已安装 torch

恢复实例完成 Python 3.10、torch 2.4.1 和全部 requirements；resolver 最终选择
`rerun-sdk 0.23.1 / opencv-python 4.11.0.86 / numpy 1.26.4`。随后 upstream pointops2 执行普通
`pip install .` 时，PEP 517 临时 build env 在读取 setup requirements 阶段报
`ModuleNotFoundError: No module named 'torch'`。正式 stage `rc=1`，峰值 cgroup memory
`16,839,843,840` bytes、GPU 0 MiB，`oom=0 / oom_kill=0`。

正式 blocked 证据：
`/root/autodl-tmp/runs/dynamic_recon/DR-M5-DGGT-NUSC-01/20260729T133346__native-nusc-s0-wm3090/`。
checkpoint、native inference 和 common-observation metrics 均未启动；没有在同一实例事后加入
`--no-build-isolation` 覆盖失败。该结果属于明确 upstream packaging blocked，不是 DGGT 质量、显存或方法裁决，
并按权威计划第 15.1 节满足继续 M6 的替代前置证据。

### PIVOT-F15：AD-GS 冻结 pseudo ID 与 checkpoint 都不能支持单对象编辑

M6 直接审计训练前冻结的 `semantic/mask_*.npy`。按 camera-local ID 统计，六个官方场景最长支持帧为
`1 / 6 / 1 / 1 / 2 / 1`，全部低于预注册 `≥20/60`；processed scene 也没有冻结的 vehicle track artifact。
与此同时，六个 60k checkpoint 的 `point_cloud.ply` 均只有二值 `obj∈{0,1}`，没有持久 instance ID。

正式证据：
`/root/autodl-tmp/runs/dynamic_recon/DR-M6-STRESS-01/20260729T145645__identity-audit-s0-wm3090/`。稳定失败
`persistent_object_identity_unavailable` 在 6/6 scenes 重复。对象编辑、pseudo-hole 和噪声行全部保留为
`ABSTAIN`，0/12 object slots 没有从 coverage 分母删除。禁止在看到 checkpoint/场景结果后用几何 Hungarian
轨迹回填 M6 baseline；这类重关联只能作为新方法候选，并必须先过 novelty。

### PIVOT-F16：instance-aware 与 driving edit 已被 2025–2026 工作直接覆盖

M7 只沿决策表考察 A“可编辑运动表示与轨迹不确定性”。重新核对官方来源后：InstDrive 已用 SAM pseudo masks
学习动态驾驶场景 2D/3D instance identity；Director 已做 4D Gaussian identity consistency；OmniRe 已用
actor scene graph/canonical vehicle nodes 做仿真；HorizonForge 与 G²Editor 已覆盖车辆轨迹操作、删除和遮挡区
恢复。

正式 evidence：
`/root/autodl-tmp/runs/dynamic_recon/DR-M7-HYPOTHESIS-01/20260729T145748__novelty-audit-s0-wm3090/`。候选的持久身份、actor-centric Gaussian binding、时序一致性和轨迹/对象编辑
核心机制均为 direct overlap；confidence/ABSTAIN 是评测与安全护栏，不构成独立技术 delta，剩余差异只是
AD-GS 适配工程。因此 M7=`rejected`，不注册事后 primary endpoint；M8/M9 均
`rejected / not authorized`，不得通过改名、挑场景或把 0 coverage 写成改进继续。

### PIVOT-F17：DGGT 扩展构建必须同时固定 compiler、headers 和 Python 依赖上界

V2 M1 表明，“已安装 torch cu121”并不足以证明 CUDA extension 可构建。宿主只有 CUDA 11.8
toolkit，会在 pointops2 编译时与 torch 2.4.1+cu121 硬失配；只补 `nvcc` 又会缺
cusparse 等 headers。正确合同是在前缀环境固定 NVIDIA CUDA 12.1 compiler/runtime/headers，
传播 `CUDA_HOME/CPATH/LD_LIBRARY_PATH`，再按 upstream `python setup.py install`。

同一里程碑还暴露了浮动 Python 树的独立风险：transformers 5.x 使用 torch 2.4.1 未提供的
DTensor API，diffusers 0.39 触发 torch schema 不兼容。最终固定
`transformers 4.48.3 / tokenizers 0.21.0 / diffusers 0.32.2 / numpy 1.26.4 /
opencv-python 4.11.0.86 / rerun-sdk 0.23.1 / flow-vis 0.1`。

对应 blocked runs 为
`20260802T120027Z__native-nusc-s0`、`120943Z__...-r2`、`122213Z__...-r3`、
`122904Z__...-r4`、`124347Z__...-r5`。这些失败是构建/依赖证据，不能推断 DGGT
方法质量。

### PIVOT-F18：原生阶段完成不应被后续评估依赖失败覆盖

M1 r6 已完成 18/18 1-view 和 18/18 3-view，但 common evaluator 导入 AD-GS 冻结
`loss_utils` 时因 `flow_vis` 未安装而 blocked。原生输出本身未损坏，但主 terminal 已转为
blocked，禁止为了“好看的 done”改写。

恢复方式是新建 r8，对 r6 `native_summary.json/metrics.json`、每个 stage 和输出哈希做
fail-closed 引用后只执行 common diagnostic。r7 中重试封装自身的 `KeyError` 也以新的
blocked run 保留，再由 r8 完成。后续所有 multi-stage run 必须把“可复用的完成阶段”与
“整个 instance 的 terminal 终态”分开；重试不得修改旧 terminal。

### PIVOT-F19：nuScenes devkit 反向索引与磁盘 metadata 不是同一 schema

M2 r1 直接读取官方磁盘 `sample.json` 时发现其中没有 `anns`；该字段是 nuScenes devkit
初始化后才注入的反向索引，不是原始 JSON 合同。正式适配器改为流式扫描
`sample_annotation.sample_token`；由于这个外键非唯一，不得用单值 dict 覆盖同一 sample
的多个 annotation。`ijson` 还必须以 `use_float=True` 读取，否则 Decimal 会污染严格 JSON
运行合同。

同一里程碑还表明，“时间最近”不足以建立 raw annotation 到 camera sweep 的真值映射。r4
中 scene-0242 boundary actor 命中更近的 sweep，但 sweep 所属 `sample_token` 与 raw 2 Hz annotation
不同，因此在 QA 前即 blocked。正确规则是先限定 exact sample token，再在候选内最小化
timestamp delta；正式 r5 达到 `4356/4356` exact mappings。后续不得仅按文件名或时间
猜测 raw/processed/render 映射。

### PIVOT-F20：CUDA 扩展 import 成功不等于包含当前 GPU 架构

M3 的 DriveStudio 环境能正常 import `gsplat` 和 `nvdiffrast`，但旧二进制没有 RTX 3090 的
SM 8.6 kernel：前者在 SH rasterization 报 `no kernel image`，后者在 EnvLight 路径报 CUDA 209。
只做 import smoke 无法发现此类错误。恢复时分别固定官方源码 commit，以
`TORCH_CUDA_ARCH_LIST=8.6+PTX` 重建，并执行真实 CUDA forward/backward；旧 `.so` 先备份，
没有修改算子语义或模型配置。

对应 blocked runs 为 r4、r6、r7；正式 binary SHA-256 为 gsplat
`6d7c8e5a...dd6131`、nvdiffrast `0d18f767...96499`。以后 CUDA 扩展 readiness 必须包含
目标 GPU 上的实际 kernel forward/backward，不能只看包版本和 import。

### PIVOT-F21：训练完成 checkpoint 与累积式 post-render 必须分开裁决

M3 r8 的 30k 原生训练已保存 `step=30000` checkpoint，但上游随后将 588 个 full-render 结果累积在
内存中；在 `577/588` 时 cgroup memory 连续两次超过 90%，资源守卫发送 SIGTERM。`oom=0 /
oom_kill=0`，checkpoint 字节数与 step 完整。r8 仍保持 `blocked`，不得改写为 done；r12 通过新的
不可变 run 对 checkpoint step/bytes/hash 和原失败 terminal 做窄范围复核，再执行流式 27-image
edit smoke 完成 M3。

同一恢复链还发现，正式训练会把某个非目标 rigid model 的全部 Gaussian 裁剪掉。token、dataset column
和 model index 仍是一一映射，但 checkpoint slice 为空。registry v2 因此将其明确标成
`unavailable_empty_checkpoint_slice`，同时对正式选中 actor 继续要求非空。禁止为了全 registry 看起来
完整而伪造 slice，也禁止因一个非目标空 slice 丢弃 23 个真实非空映射。

### PIVOT-F22：外层 timeout 不会自动回收独立 session 的 GPU 子进程

M4 controller 用 `subprocess.Popen(..., start_new_session=True)` 隔离正式渲染，使 SSH/tmux 断开不应
误杀长任务；相应地，用外层 `timeout` 调试 controller 时，SIGINT 只终止父进程，子进程会以 PPID 1
继续占用 GPU。`debug_controller_s0_r5` 复现了该行为；残留子进程通过已核实的精确 PGID 发送 SIGTERM
回收，GPU 从约 `8.1 GiB` 回到 `0 MiB`，没有终止用户服务。

以后不得用外层 timeout 探测会派生独立 session 的 controller。正式运行应直接由 nohup/tmux 托管，
同时监控 controller PID、child PID、terminal 和 resource.jsonl；确需中止时必须核实 process tree 后
显式回收 child process group。`r5/r6` 的 running terminal 保留为中断证据，不改写成 done。

### PIVOT-F23：SE(3) 一致性容差必须覆盖 float32 往返误差

M4 单帧 r1 的 actor transform 先由 checkpoint float32 tensor 变换，再写入 JSON 并读回，最大平移误差
略高于 `1e-6 m`；其余 15 项检查均通过。把该值当几何失败会制造假阴性。协议在查看正式全量结果前
固定为 `1e-4 m`，r2/r3 冒烟通过，正式 196 帧实测最大误差为
`3.814697265625e-06 m`，rotation/size/canonical drift 均为零。容差变更只反映数值精度，不降低
1 m 编辑幅度，也不得据此为真正的轨迹偏差放宽门禁。

### PIVOT-F24：冻结 heldout 资源门失败不能靠事后更换 renderer 或提高阈值挽回

A3 R1 在结果前冻结 `12,288 MiB` PyTorch allocated GPU ceiling。三条完成全部 R0/R1 指标计算的只读路径
`r2/r4/r5` 分别达到 `14,241.777 / 14,244.924 / 14,241.399 MiB`；wall、cgroup、run bytes 与 OOM delta
均通过。资源审计、CPU checkpoint staging、Rigid quota device 兼容和逐 view `trainer.info` 释放都没有改变这一
单 view 峰值。继续把 renderer 改为 `packed=true`、分块/降分辨率，或把 ceiling 提高到观测值以上，会在看到结果后
改变 source-render 路径或预算，不再是预注册评测。

r5 的资源无效 diagnostic 也不能救回方法：S-B depth-order violation 从 `0.915792` 降至 `0.908173`，但
non-target 与 original-global RGB MSE 都按 exact comparator 严格变差，故为 `tradeoff_non_dominated`。正确分层是：
r5 run 保留 `blocked`，R1 方法臂登记 `rejected_resource_gate_and_diagnostic_tradeoff`，A3 任务以负结果 `done`；
生产路由回退到 R0/D2 immutable exact alias。以后若研究 packed/分块渲染，只能在 A4 作为新的部署因子另行冻结，
不得倒写 A3 heldout 结论或解锁 R2–R4。

### PIVOT-F25：部署 profile 必须区分传感器原始尺寸与 checkpoint 原生加载尺寸

A4-P0 v1 在新测量前把 scene-0230 的 nuScenes 传感器尺寸 `1600×900` 冻结为“原生分辨率”。formal r1 实际
完成 2 次 warm-up 与 9 次 measured render 后，11 行全部为 `800×450`，因此只在 finalize 的
`native_resolution_exact` audit 失败；资源、输入 hash、无训练/无 checkpoint、同步矩阵与无 torch resume audit
均通过。source config 事后审计确认三路相机已冻结 `data.pixel_source.downscale_when_loading=[2,2,2]`，故当前
checkpoint 的模型原生加载/渲染尺寸本来就是 `800×450`。

不得修改 v1 或把 r1 改写为 done，也不得用 r1 性能数字关闭 P0。正确处理是保留 r1=`blocked`，冻结其 protocol、
manifest、runtime stage/rows、resource audit 与 terminal hash；创建 v2 只纠正分辨率语义，再从新目录完整重跑。
后续协议必须同时记录 sensor resolution、source-config downscale 与 model-native render resolution；“native”一词
不能在这三层之间无来源转换。该纠错不授权降低分辨率、切换 renderer、改变资源 ceiling 或开启 P1/P2/P3/P5。

### PIVOT-F26：checkpoint state key 不能冒充加载后的模型运行时属性

A4-P5 formal r1 已通过 9 项输入审计并生成 `14,729-byte` reference-only deployment registry，但 fresh DriveStudio
worker 在 checkpoint 成功加载后读取 `RigidNodes.points_ids` 时报 `AttributeError`。源码事实是：checkpoint
`state_dict` 以 `points_ids` 为序列化键，`load_state_dict` 将它弹出并写入运行时属性 `self.point_ids`。两者语义
相关但接口层不同；直接把 checkpoint 键拼成对象属性会使恢复链在资产已经物化后失败。

r1=`20260809T155209Z__a4-p5-registry-resume-s0-r1` 保留 `blocked`，terminal SHA=`61d30a11...773e`；其已生成
registry SHA=`e48bccdf...9039d` 不覆盖。修复 `0e899b2` 只通过 fail-closed helper 读取 runtime `point_ids`，并用
两条测试分别锁定有效属性和拒绝 `points_ids` 别名；没有修改 P5 protocol、输入、资源 ceiling 或审计口径。
新目录 r2 以相同 registry SHA 完成 14/14 audits，证明问题属于 runner/runtime contract，不是资产或方法失败。

以后凡从 checkpoint 结构推断 live module API，必须同时核对 `state_dict` 保存端、加载端赋值和加载后的真实对象，
并用回归测试锁定层级；旧失败 run 维持 `blocked`，不得因修复后的新 run 成功而倒写。

### PIVOT-F27：最小预注册剪枝臂失败后不能事后补更小 fraction 或放宽质量门

A4-P1 在结果前固定 source/b05/b10/b20 四臂，并要求 global、actor、boundary 与 non-target 的全部 safeguard 同时
通过。canonical r1=`20260809T165058Z__a4-p1-contribution-prune-s0-r1` 完成 36-view contribution、三臂物化、
四臂 57-view 质量和 9-view runtime，21/21 audits 全 true，资源门通过；因此它不是工程或资源 `blocked`。

最小候选 b05 已将 checkpoint 减少 `23,881,368 bytes`，全部 row/invariant/reload/count 审计 exact，但 global
occupied PSNR、global PSNR 与 non-target PSNR 分别退化 `0.117684/0.110926/0.125462 dB`，超过冻结的
`0.10 dB` 上限；b10/b20 分别失败 12/15 个端点。局部 actor/boundary 指标的保持或改善不能覆盖全局与非目标区
失败。运行时 P50/FPS 也非随 fraction 单调，且 filesystem cache 未控制，只能报告，不能充当事后选择理由。

正确裁决是 P1 experiment=`done`、method=`rejected_quality_or_integrity_gate`、生产资产 exact fallback 到 source。
不能在看到 b05 失败后新增 b01/b02、改排名视图、放宽 `0.10 dB` 或只保留通过的局部端点；这些都属于新的预注册
实验，而不是当前 P1 的修复。该负结果只约束 scene-0230/seed-0/冻结视图矩阵，不外推为所有贡献度剪枝均失败。

### PIVOT-F28：顶层 `named_parameters()` 不保证覆盖普通映射中的子模型参数

A4-P2 formal r1=`20260809T174337Z__a4-p2-mixed-precision-s0-r1` 已成功完成 10-field checkpoint conversion、
source/candidate 57-view quality、两臂 runtime、aggregate 与 no-torch resume；aggregate 也按冻结门选择 mixed arm。
但 finalizer 的 `checkpoint_reduction_and_runtime_matrix_exact` 唯一失败：账本只调用
`trainer.named_parameters()`，而 DriveStudio 把 Gaussian 子模型保存在普通 `trainer.models` 映射中，并未注册成
顶层 `ModuleDict`。因此账本只看见 LPIPS 等 `9,883,392 bytes`，把 source/candidate 错记为相同总量且没有 FP16
bucket，尽管候选 checkpoint 已实际从 `578,819,674` 降到 `432,111,754 bytes`。

这属于 evidence collection defect，不是 conversion、质量、资源或方法失败。r1 保留 `blocked`，terminal SHA=
`5ef3dab60ff934af19ff547c0f7e7cd0fe74b83000476888a630341ee39474c0`；不得手改 r1 runtime stage 或把它倒写为
done。修复 `dcf2822` 显式遍历 `trainer.models` 中每个 module 的 parameters，并按 Parameter identity 去重；回归 fixture
故意使用未注册的普通映射，锁定 `models.Background._scales` 等字段必须进入账本。协议、字段、阈值、renderer 与
selection 均未改变。

新目录 canonical r2=`20260809T174850Z__a4-p2-mixed-precision-s0-r2` 完成 19/19 audits，正确记录 source/candidate
persistent bytes=`394,641,424 / 247,936,208` 与 candidate FP16 bucket=`146,705,216 bytes`，并选择 mixed arm。
以后审计复合模型时，必须同时核对容器是否已注册为 `nn.Module`、顶层 traversal 覆盖范围与逐字段预期集合；
证据账本缺失不得用 checkpoint 文件变小或 runtime 成功来推断补齐。

### PIVOT-F29：无卡实例必须以 cgroup 内存为资源合同，不能读取宿主机 `free`

V3.2 S0 在 AutoDL 无卡开机模式中观察到 `free -b` 暴露宿主机约 810 GB 内存，但
`/sys/fs/cgroup/memory.max=2,147,483,648`，实际只有 2 GiB。CoIn 的完整 partial-clone checkout 在大量 blob
物化时把 `memory.current` 推近上限且长时间无进展；继续并发创建环境或校验大权重会把可回收 page cache 与真实
匿名内存混在一起，增加无意义的 OOM 风险。

正确处理是停止该 checkout，把残留移到仓库外备份，并改用 `--no-checkout --filter=blob:none` 固定 commit/tree；
所有下载使用流式落盘，依赖安装和 hash 校验串行执行。GPU、VRAM 与 driver 则必须在有卡重启后重新审计，不能把
无卡模式的 `nvidia-smi: permission denied` 写成硬件不存在。后续任何资源 preflight 都必须同时记录
`memory.max`、`memory.current`、`memory.events` 和数据盘余量；宿主机 `free` 只作诊断，不作授权。

### PIVOT-F30：原子发布目录不能把 `.partial` 绝对路径写进 manifest

S1 prompt preparer 首次在 `s1_prompt_v1.partial` 内生成绝对 `video_dir`，发布时将目录改名为
`s1_prompt_v1`，manifest 内部却仍指向已经不存在的 `.partial` 路径。SAM2 因此把它判定为既非 MP4 也非 JPEG
目录，r1 在真实 GPU 启动后立即失败。修复是 manifest 只保存相对 `sam_inputs/...`，消费者相对 manifest
父目录解析；原子 rename 后重新验证每个目录和 JPEG 链接。任何会整体 rename 的 staged asset 都不得在内部保存
staging 绝对路径。

### PIVOT-F31：SAM2 `reverse=True` 默认从最早 prompt 开始，可能合法地产生零帧

首次双向传播实现只设置 `reverse=True`，但官方 predictor 默认 `start_frame_idx=min(condition frames)`；train-only
block 的首个 prompt 常在 local frame 0，反向 processing order 因此为空。调用成功和进度条 `0it` 不能证明反向
覆盖。正确做法是显式用 block 内最晚 prompt 作为 reverse start，并按每个 object 自己的 prompt frame 过滤输出；
r5 中实际产生 13 个 prompt 之前的 mask，才构成双向证据。

### PIVOT-F32：mask QC 必须在同一像素坐标系比较

r4 将 SAM logits 从源图 `1600×900` resize 到模型原生 `800×450`，却直接与源图坐标的逐帧 3D box 比较，
造成 `235/263` 假拒绝。修复后 box 按 exact x/y 比例映射到 800×450，r5 为 `212 accepted / 51
fail-closed`，其中 43 个是近空 mask。以后任何 IoU、centroid、boundary 或 area ratio 门禁都必须同时记录 source
size、target size 与变换；不同尺度间的数值不得直接进入裁决。

### PIVOT-F33：大规模 Gaussian 重复索引累加不得使用逐元素 `np.add.at`

S1 r2 在每个视图的数百万 ray/Gaussian intersections 上多次使用 `np.add.at`，CPU 单核成为瓶颈；同时该版本
仍缺计划要求的 negative views、depth-consistency rate 和 boundary score，因此保留 250 个 mask 后以 exit 143
终止，不得作为完成证据。r5 改用 `np.bincount(minlength=total)` 和向量化 view count，263-view lift wall
`770.733s`，并保存完整 posterior schema。研究 runner 必须输出阶段进度；“CPU 持续运行”不能替代复杂度审计。

## 7. 历史新路线启动前附加检查

- [ ] 是否明确说明该步骤直接服务于重建、编辑或可信评测，而不是重新做事件挖掘？
- [ ] AD-GS exact reproduction 是否已经通过冻结门禁？
- [ ] 是否把 upstream 原始结果与 compatibility patch 结果分开？
- [ ] 是否对 VAD-GS 等已公开的 visibility/completion 工作做 novelty 边界核对？
- [ ] 反事实无真值指标是否有真实 held-out/pseudo-hole 证据，而不是自洽规则？
- [ ] 是否同时评估目标区变化、非目标区保持、几何/时序一致性和下游感知？
- [ ] 遇到内存/GPU不足时是否按 `N1-F24/PIVOT-F05` 停机并等待授权？

## 8. WorldSim 后续正式消融前检查

- [ ] 是否使用 V3 task ID、新 run 和冻结 config/source hash，而不是续写 V2 terminal？
- [ ] 是否保持 scene-0230/0242/0255、split、seed、相机、步数和 actor cohort 不变？
- [ ] 是否把原生 Affine/CamPose/LiDAR init 与新增实现分开？
- [ ] rolling-shutter 路径是否有真实 row timing；没有时是否显式 `not_supported`？
- [ ] actor-aware 变化是否只增加一个可归因因子，并保留 module-off 原生等价测试？
- [ ] 是否同时报告 actor/boundary 质量、GS 数、训练时间、VRAM 和 non-target 保持？
- [ ] local refinement 是否冻结 affected set 外参数，并只使用 Tier-A/多视图/LiDAR 可观测证据？
- [ ] expected/first-hit/measured depth 是否继续保持 typed separation？
- [ ] 工程 `blocked`、方法负结果 `rejected` 和任务完成 `done` 是否没有混写？
- [ ] 结论是否明确限制在三场景消融，不写成大规模泛化或闭环安全结论？
- [ ] 新路线是否只选择一个 primary hypothesis，并说明它具体解除 `V3-F18`–`V3-F25` 中哪一项？
- [ ] 是否在任何训练、推理或新结果读取前冻结 matched baseline、主端点、资源门、停止条件和确认场景？
- [ ] 是否避免把更小剪枝 fraction、提高旧资源 ceiling、全量读取 chunk 或继续调 R1 配方伪装成新研究？
