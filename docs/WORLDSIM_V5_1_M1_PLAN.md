# WorldSim V5.1 M1 执行登记

## 2026-08-18 Stage F F0a environment smoke v3 recovery

r028 已证明 Gurobi 12.0.3 不再触发 expired-license，但 license banner 破坏了整段 JSON parser；仍未运行模型。r029 只把
stdout 合同改为“最后非空行 JSON + 前置行留档”，不改 solver gate、环境或 one-view。见 `V51-F51`。

## 2026-08-18 Stage F F0a environment smoke v2 recovery

r027 因 exact Gurobi 10.0.3 内置 license 过期，在任何模型/input 执行前 blocked。r028 只把版本改为 upstream 下界内的
12.0.3，并使用 fresh wheelhouse/venv 完整重跑；仍要求 Gurobi/PuLP tiny solver gate，不允许直接把 fallback 当 faithful。
同时禁止子 Python 在 frozen source 写 bytecode；其余 one-view、official CLI、资源与 read locks 不变；见
`V51-F49/F50`。

## 2026-08-18 Stage F F0a isolated environment/one-view smoke 预注册

r027 只建立隔离、可追溯的 inference 环境并在唯一 0471/0/0 输入上运行 official automatic demo。依赖使用 pinned wheels
与 frozen source path，Gurobi/PuLP 均需 tiny solver PASS；单图输出只做 schema/资源核对，不读 quality。由于 one-view 不足
semionline 三帧 voting，本轮不允许声称 DEVA association 成立，也不解锁 45-view materialization；下一门必须是 3-view
association+repeatability smoke。环境失败按工程 failure 留档后恢复，禁止换 SAM2、MobileSAM、GroundingDINO 或改 CLI 参数。

## 2026-08-18 Stage F F0a r026 收口与 environment smoke 解锁

r026 已 exact 获取 DEVA/SAM-v1 权重、Grounded-Segment-Anything source，并冻结 45-view train-only input chain；独立审计
full-hash/replay PASS。该结果只解除“缺 source/assets”前置，不解除 materialization：当前 runtime 缺 6 个相关模块，且尚未
验证 SAM ViT-H+DEVA 在 3090 上的单视图显存、输出 schema 或 short-ID determinism。下一步必须新预注册隔离 venv 的
可复现安装来源/版本，再只用 scene-0471/frame0/camera0 做 one-view SAM/DEVA resource/schema smoke；不得直接跑 45 views，
不得据 smoke 调 `size=480/IoU=0.7/chunk=4`，不得读取 mask quality。freeze=
`stage_f_f0a_asset_source_acquisition_freeze_v1.yaml`，failure delta=`V51-F47 resolved`，M2/M3=pending。

## 2026-08-18 Stage F F0a asset/source acquisition 预注册

F0a 先拆出一个不可混入方法质量的 acquisition gate。r026 只冻结 upstream-declared DEVA/SAM-v1 权重的 full SHA、
Grounded-Segment-Anything fork exact source，以及 45 张 train-only 原图的 inherited SHA denominator；下载 sequential、
partial-resumable、exact-byte 后 atomic publish，不 decode image/mask、不运行模型。当前 DriveStudio 环境缺失的自动分割依赖
只报告，不原地安装；r026 PASS 后下一门是 isolated environment + one-view resource smoke 的新预注册，仍不直接全量
materialize。CLI 参数、scene/frame/camera order 和 short-ID PNG/pred.json schema 已提前冻结，禁止借环境恢复换 SAM2、
调 IoU/size/chunk 或读取 quality。M2/M3=pending。

## 2026-08-18 Stage F F0 r025 收口与 F0a 解锁

r025 已在 source=`8d68cad1` 上完成并由独立 auditor 重放通过。Gaussian Grouping official source、faithful identity
mechanism 与 frozen-base 16D gsplat adapter 都可执行；45 个 train-only observation 则一致缺少跨视图 instance identity
labels，且 official DEVA/SAM-v1 权重尚不存在。该结论只解锁 F0a 的预注册：固定 train-only 45-view 输入、官方权重
provenance/hash、SAM everything 与 DEVA semionline short-ID 输出 schema、确定性/资源/失败恢复合同，再单独执行资产获取与
identity-mask materialization。不得把 stable actor metadata、binary U2/B3 evidence、SAM2 或 evaluation target 作为替代；
不得直接开始 identity training。freeze=`stage_f_f0_source_preflight_freeze_v1.yaml`，failure delta=
`V51-F45 resolved /V51-F46 active`；S/C/validation/test/KITTI 继续锁定，M2/M3=pending。

## 2026-08-18 Stage F F0 source/adapter preflight 预注册

F0 不直接复用二值 ownership。官方 Gaussian Grouping 固定为 `SAM everything + DEVA cross-view short IDs + 16D`
`identity encoding + differentiable identity rendering + shared classifier CE + k5 KNN KL`；source=
`lkeab/gaussian-grouping@0ab60afe` 与 ECCV 2024 official PDF。由于 normative plan 要求 immutable base，本项目适配仅冻结
geometry/appearance/opacity/actor pose，而保留 identity branch 的原机制；必须明确这不是上游 joint-reconstruction exact
reproduction。r023 只核对 source、15 train-only views/scene 的 schema/instance metadata、上游 asset presence，并做小型
16D renderer gradient smoke；不读 image/mask pixels 或 quality，不运行 SAM/DEVA/identity training。现有 observation 没有
associated instance IDs 时，只解锁 F0a train-only mask materialization 的预注册。r023 在 status 前因 shared git helper
调用遗漏 project 参数而终止；只保留 resolved config。r024 修复后在 CUDA context 初始化前 reset peak counter 而 blocked；
r025 只增加显式 device init 顺序并完整重跑，不改变 source/method/data 门。

## 2026-08-18 Stage E r022 收口与 Stage F 解锁

r022 已在 frozen H 12 views 上独立复算并拒绝 E0B：相对 U2/B3 的 scene-balanced BF1/IoU/FN=
`-0.0002566/-0.0925468/+0.1899473`，相对 raw D0=`-0.0004762/-0.0210926/+0.0204707`；primary 与
mechanism gate 均 FAIL。E1 PanoGS/E2 AG²aussian 永久停止，禁止根据 r022 回调 voxel level、aggregation 或 propagation。
结果 freeze=`stage_e_e0b_h_evaluation_freeze_v1.yaml`，failure=`V51-F42`。下一任务严格使用 normative task id
`WS-V51-M1-F-IDENTITY-EMBEDDING-01`，先冻结 Gaussian Grouping 官方论文/代码并做 no-quality source/adapter preflight；
不得把现有 binary actor-union SAM evidence 冒充上游要求的 cross-view associated instance-ID masks。

## 2026-08-18 Stage E E0b H evaluation 预注册

r022 在冻结 12 个 H views 上比较 `U2_B3_G0/D0/E0B`；只新渲染 E0B，U2/B3 与 D0 复用 frozen float16 artifacts。
primary gate 继承 Stage D，mechanism gate 额外要求 E0B 相对 D0 在至少两场 BF1 不退、scene-balanced BF1 严格为正、
IoU 不退、FN 不增；两组 gate 必须同时 PASS。PASS 才解锁 E1 PanoGS preregistration；FAIL 后 E1/E2 永久锁定并转
Gaussian Grouping。r022 后禁止回看 fine/medium/coarse 或调 aggregation/propagation。

## 2026-08-18 Stage E E0b r021 operator 收口

r021 在 `fine_q50` node 上完成 no-quality same-propagation，full operator 独立 replay exact；三场相对 D0 改变
`4,065/1/475` 个 Gaussian posterior，说明 1087 几乎 no-op，但这不是 H quality 结论，禁止改 level/aggregation/threshold。
下一门只能先冻结三臂 matched H evaluator：主门继承 D0 对 U2/B3 的四项门，机制门同时要求 E0B 相对 D0 的 BF1/IoU/FN
不退化并有 scene-balanced BF1 正增益。H PASS 才能预注册 E1 PanoGS；FAIL 则收口 E0、锁定 E1/E2 并自动进入
Gaussian Grouping。freeze=`stage_e_e0b_same_propagation_freeze_v1.yaml`。

## 2026-08-18 Stage E E0b same-propagation 预注册

E0b 的 level 在任何新 H quality read 前冻结为 `fine_q50`：对 E0a 三场共同 PASS levels 按原 edge-length quantile 升序
取第一档，执行 minimum intervention；density gain 与 seed conflict 均不参与排序。将 frozen raw KNN quotient 到 voxel
node，node unary 取 member mean，逐 view probability 取 visibility-weighted mean、visibility 取 maximum；其余 D0 seed、
cosine affinity、exact two-hop、decay、threshold、fixed-point 与 UNKNOWN 全部不变。r021 只做 no-quality operator；完成并
冻结前不得读取 H 或执行 E1/E2。future H 同时对 U2/B3 主基线和 D0 mechanism comparator 设门，失败即收口 simple node
elevation、停止 E1/E2、转 Gaussian Grouping。

## 2026-08-18 Stage E E0a r020 收口

r020 已在三场完成：三档 voxel level 均通过 no-quality observation-density gate，9 份 assignment、45 个训练视图分母、
18-entry manifest 与 gate 由独立 auditor 复算 exact。该结果仅证明结构分组能增加 member-view union，不证明 BF1/IoU
改善；尤其 1087 增益很小，0471 coarse seed mixing 明显增大，禁止按事后 density 最大值选 coarse。E0b 下一步只能先
冻结 H-independent level selection 与 same-propagation raw-vs-voxel 合同，再运行新 matched A/B；E1/E2 仍锁定。
`V51-F39 resolved by r020`，result freeze=`stage_e_e0a_superprimitive_probe_freeze_v1.yaml`。

## 2026-08-18 Stage E E0a r019 recovery

r019 在 0379 edge-length quantile 前因 `34` 条 zero-length KNN edges blocked；0471/1087 partial 产物不作为方法证据。
v2/r020 只将 voxel size 明确为 positive-edge `q50/q75/q90`，所有 Gaussian 仍参与 voxel assignment，其他输入、门禁和
quality locks 不变。r020 必须完整重跑；在它 terminal/freeze 前，E0b same-propagation、E1 PanoGS 与 E2 AG²aussian
继续锁定。工程 delta=`V51-F38/F40 resolved`、`V51-F39 recovery pending`。

## 2026-08-18 Stage E E0a 分相预注册

为避免 observation-density 结构证据与传播质量混在一次运行中，E0 被拆成两相。E0a 只将 frozen raw Gaussian 按
KNN edge-length `q50/q75/q90` 三档做 world-origin voxel grouping，检查 member-view union 是否在每场严格提高且能救回
raw zero-observation Gaussian；不选 voxel level、不运行 propagation、不读任何 quality。PASS 后才允许另行冻结 E0b
的 same-edge/same-propagation raw-vs-voxel 对照；FAIL 直接停止 E1/E2 并转 Gaussian Grouping。PanoGS 论文/代码仅作为
后续 E1 provenance 冻结，E0a 不得称为 PanoGS faithful port。

## 2026-08-18 Stage D 收口与 Stage E 解锁

r018 在冻结 H 12 views 上被正式拒绝：BF1 positive scenes=`2/3`、mean BF1=`+0.0002196`，但 mean IoU=
`-0.0714543`、FN semantic mass=`+0.1694766`，后两门 FAIL。D0 progressive 收口、D1 永久跳过，不允许 H recovery
tuning。按固定路线自动解锁 `WS-V51-M1-E-NODE-ELEVATION-01`；下一步只做 E0 simple multi-resolution voxel
super-primitive 的结构/observation-density preflight，不读 S/C/validation/test/KITTI quality，也不启动 E1/E2。

## 2026-08-18 Stage D D0 H matched evaluation 预注册

r016 source/input preflight 与 r017 full-H no-quality operator 已冻结。下一门固定为 r018 H matched evaluation：只在
`0471/1087/0379` 的 frozen 12 views 上比较 `U2/B3 G0`、frozen V5 `U2/B3+G3` 与 D0，三臂统一用持久化
float16 probability 计算质量指标；SAM mask 是 evaluation-only proxy，不是 ground truth 或方法输入。H gate 严格继承
normative plan §23.1。若 PASS，只解锁 frozen D0 的 S exact-once；若 FAIL，立即 reject progressive、skip D1 并按路线
顺序进入 super-primitive/anchor，不得在 H 上调参。S/C/validation/test/KITTI 继续锁定，M2/M3 继续 pending。

## 2026-08-18 Stage B H evaluation-only 分相预注册

为防止 heldout feature extraction 与方法质量读取发生隐式耦合，H evaluation-only 被拆成两个不可合并的阶段：

1. r013 只抽取 `frame % 5 == 2` 的 45 个 H evaluation DINO features，并用 r010 PCA state 做 frozen transform；
   它不读取 membership、uplift 或 quality，也不启动 renderer。
2. 只有 r013 terminal evidence 冻结后，才能另行提交 evaluation runner/config。该提交必须在第一次读取 r012
   B0/B1 feature quality 前冻结 proxy 声明、pair denominator、abstention 和 H gate。

最终 heldout remainder=`4`、S/C、validation/test、KITTI 与 M2/M3 不因本分相解锁。

r013 已冻结；r014 evaluation-only config/module/runner/test 现已按上述顺序完成预注册。r014 的 clean commit 是第一次
读取 r012/r013 数值的必要前置；H pass 只解锁 S exact-once，H fail 按冻结路线跳过 raw LUDVIG graph。

本文件是 V5.1 M1 的短执行入口；完整规范、方法树、门槛与第一轮约束以
`docs/WORLDSIM_V5_1_M1_TOPCONF_PLAN.md` 为唯一 normative plan。这里不复制长计划，只维护当前阶段、授权和证据，
避免两份计划发生漂移。

## 当前阶段（2026-08-18）

| Task ID | 状态 | 当前证据/下一门 |
|---|---|---|
| `WS-V51-P0-M1-SCOPE-FREEZE-01` | done | r001 start audit；scope/授权/quality locks exact |
| `WS-V51-D0-DEV-ROLE-FREEZE-01` | done | r001 start audit；H/S/C=`3/2/3` 与原 cohort exact |
| `WS-V51-M1-A-UNARY-OBSERVABILITY-01` | done | r007 S screening：A1/A2 rejected；freeze U2/B3 |
| `WS-V51-M1-B-LUDVIG-UPLIFT-01` | rejected | r015 H gate：reprojection PASS 但 actor margin FAIL；uplift/raw graph 均收口 |
| `WS-V51-M1-D-PROGRESSIVE-01` | rejected | r018：BF1 两门 PASS，但 IoU/FN FAIL；D1 skipped，freeze + `V51-F37` |
| `WS-V51-M1-E-NODE-ELEVATION-01` | running | E0a r020 + E0b r021 frozen；三臂 r022 matched H 已预注册，E1/E2 locked |
| `WS-V51-M2` | pending | 未授权 |
| `WS-V51-M3` | pending | 未授权 |

## 授权历史与当前边界

第一轮只允许 P0、development role freeze 与 Stage A；这些历史 artifact 保持不变。2026-08-17 用户明确授权
`U2/B3 fallback` 进入 Stage B，并授权 M1 在单 arm/scene/工程/paper failure 后留档并按冻结顺序自动换路线。
Historical diagnostic=`0471/1087/0379`，screening=`0998/0359`，development confirmation=
`0875/0535/0436` 不变。V5 的 8-scene validation 与 20-scene test 继续不可读；KITTI 不用于方法调参；M2/M3
保持 `pending`。

M1 路线顺序固定为 LUDVIG uplift/semantic graph→progressive propagation→super-primitive/anchor→Gaussian
Grouping→Trace3D→BKI/graph-free。每条路线先原样迁移论文方法；无效则 rejected 并进入下一条，有效后才允许创新；
所有 matched A/B 始终保留 U2/B3。只有 development confirmation 上稳定优于 U2/B3 的冻结 candidate 才能一次性
进入 fresh validation，test 仅供最终候选 exact-once 使用。

## Failure ledger 绑定

- scope/data/protocol：`V5-F09`、`V5-F11`–`V5-F14`、`V5-F18`；
- unary/evaluation：`V5-F20`–`V5-F26`、`V5-F29`–`V5-F33`；
- V5.1 新增 failure=`V51-F01`–`V51-F41`；Stage A closeout delta=`V51-F09/F10`；Stage B preflight
  delta=`V51-F11/F12/F13`；freeze proposal delta=`V51-F14/F15`；asset recovery delta=`V51-F16 resolved`；
  operator pre-formal fixture/result-freeze test delta=`V51-F17/F18 resolved`；r006/r007 recovery delta=
  `V51-F19/F20/F21/F22 resolved`；H full-run resource delta=`V51-F23 resolved by v2/r012`；Stage D rejection
  delta=`V51-F37 active`；Stage E recovery/audit delta=`V51-F38/F40/F41 resolved, V51-F39 resolved by r020`。

## 配置与入口

- `configs/worldsim_v51/p0_m1_scope_v1.yaml`
- `configs/worldsim_v51/development_roles_v1.yaml`
- `configs/worldsim_v51/m1_unary_baselines_v1.yaml`
- `configs/worldsim_v51/m1_unary_visibility_v1.yaml`
- `configs/worldsim_v51/m1_unary_unknown_v1.yaml`
- `configs/worldsim_v51/m1_effective_count_audit_v1.yaml`
- `configs/worldsim_v51/m1_effective_count_audit_v2.yaml`
- `configs/worldsim_v51/m1_cif_decoupling_audit_v1.yaml`
- `configs/worldsim_v51/stage_a_screening_freeze_v1.yaml`
- `configs/worldsim_v51/stage_a_screening_v1.yaml`
- `configs/worldsim_v51/stage_a_closeout_v1.yaml`
- `configs/worldsim_v51/stage_b_preflight_v1.yaml`
- `docs/WS_V51_STAGE_B_PREFLIGHT.md`
- `configs/worldsim_v51/stage_b_freeze_proposal_v1.yaml`
- `docs/WS_V51_STAGE_B_FREEZE_PROPOSAL.md`
- `configs/worldsim_v51/stage_b_authorization_v1.yaml`
- `configs/worldsim_v51/stage_b_input_freeze_v1.yaml`
- `configs/worldsim_v51/stage_b_dinov2_download_v1.yaml`
- `configs/worldsim_v51/stage_b_dinov2_download_parallel_v1.yaml`
- `configs/worldsim_v51/stage_b_dinov2_asset_freeze_v1.yaml`
- `configs/worldsim_v51/stage_b_dinov2_resource_smoke_v1.yaml`
- `configs/worldsim_v51/stage_b_dinov2_resource_freeze_v1.yaml`
- `configs/worldsim_v51/stage_b_operator_parity_v1.yaml`
- `configs/worldsim_v51/stage_b_operator_parity_freeze_v1.yaml`
- `configs/worldsim_v51/stage_b_one_view_contribution_v1.yaml`
- `configs/worldsim_v51/stage_b_one_view_contribution_v2.yaml`
- `configs/worldsim_v51/stage_b_one_view_contribution_v3.yaml`
- `configs/worldsim_v51/stage_b_one_view_contribution_v4.yaml`
- `configs/worldsim_v51/stage_b_one_view_contribution_freeze_v1.yaml`
- `configs/worldsim_v51/stage_b_h_feature_pca_v1.yaml`
- `configs/worldsim_v51/stage_b_h_feature_pca_freeze_v1.yaml`
- `configs/worldsim_v51/stage_b_h_uplift_v1.yaml`
- `configs/worldsim_v51/stage_b_h_uplift_v2.yaml`
- `configs/worldsim_v51/stage_b_h_uplift_freeze_v1.yaml`
- `scripts/freeze_worldsim_v51_stage_b.py`
- `scripts/fetch_worldsim_v51_dinov2_asset.py`
- `scripts/fetch_worldsim_v51_dinov2_asset_parallel.py`
- `scripts/smoke_worldsim_v51_dinov2_resource.py`
- `scripts/audit_worldsim_v51_stage_b_operator_parity.py`
- `scripts/smoke_worldsim_v51_one_view_contribution.py`
- `scripts/run_worldsim_v51_h_feature_pca.py`
- `scripts/run_worldsim_v51_h_uplift.py`
- `configs/worldsim_v51/m1_sam_screening_scene0998_v1.yaml`
- `configs/worldsim_v51/m1_sam_screening_scene0359_v1.yaml`
- `scripts/audit_worldsim_v51_start.py`
- `scripts/replay_worldsim_v51_v5_unary.py`
- `scripts/run_worldsim_v51_unary_visibility.py`
- `scripts/run_worldsim_v51_unary_unknown.py`
- `scripts/audit_worldsim_v51_effective_count.py`
- `scripts/audit_worldsim_v51_cif_decoupling.py`
- `scripts/run_worldsim_v51_stage_a_screening.py`

正式状态、实验事实和失败事实仍分别以 `docs/RESEARCH_STATUS.md`、`docs/EXPERIMENTS.md` 与
`docs/RESEARCH_FAILURES.md` 为准。

## P0/D0 canonical evidence

- run：`/root/autodl-tmp/runs/worldsim_v51/WS-V51-P0-M1-SCOPE-FREEZE-01/20260817T101000Z__p0-start-audit-s0-r001`
- source commit：`58953a57557b97f449c4d83db7d11132ddda5e73`
- conclusion：`v51_m1_scope_roles_and_v5_inputs_frozen`
- summary/manifest SHA：`6d495ce26c211843e69dd9034dccfc916f17311dc59edaf5e7115ed32723ef9c /`
  `8ab0ad66eddedece7cfe6db4871172b07ae2c80430c8ddba156df76ce2941dc5`
- canonical inventory：r037/r042/r043=`65/44/50 files`，总计 `680,254,598 bytes`，逐文件 exact。

## Stage A A0 canonical evidence

- run：`/root/autodl-tmp/runs/worldsim_v51/WS-V51-M1-A-UNARY-OBSERVABILITY-01/20260817T102000Z__m1-a0-v5-unary-replay-s20260814-r001`
- source commit：`1e2361658b85e1f12145867164238ce81ecb55ea`
- conclusion：`a0_v5_b0_b1_b3_posterior_and_gaussian_metrics_bit_exact`
- exact 分母：45 observation files；9 arm×scene；54 posterior/statistic arrays；54 Gaussian metric values。
- exact 结果：array mismatch=`0`；metric delta=`0.0`；每场 12 个核心 generation source SHA exact。
- 边界：GPU renderer 未重跑；2D evaluation 只做 canonical artifact/source identity 复核。
- 下一门：A1 visibility mask。A2/A3/A4、Stage B/C 与 Graph 均保持锁定。

## Stage A A1 canonical evidence

- run：`/root/autodl-tmp/runs/worldsim_v51/WS-V51-M1-A-UNARY-OBSERVABILITY-01/20260817T104000Z__m1-a1-visibility-h-s20260814-r002`
- source commit：`38bc9b44c6c86d58173930aa019745b8a9a8e00b`
- B3 replay：12/12 GPU renders byte exact，3/3 checkpoints exact。
- A1 H gate：BF1 positive scenes=`2/3`；mean ΔBF1/IoU/FN/Brier/ECE=
  `+0.001155713/+0.000460310/+0.001105687/-0.000013972/-0.000144854`；五项 gate 全通过。
- 边界：1087 只有 1 个 accepted view，且整体效应小；A1 只是未经过 S 的 candidate。
- 下一门：只做 A2 UNKNOWN/ABSTAIN；A3/A4、S 与后续 Stage 保持锁定。

## Stage A A2 canonical evidence

- run：`/root/autodl-tmp/runs/worldsim_v51/WS-V51-M1-A-UNARY-OBSERVABILITY-01/20260817T113000Z__m1-a2-unknown-h-s20260814-r003`
- source commit：`7e783f1fe04cc05cdd206b56086ef0f02a4215ee`
- threshold freeze：positive-count H/A1 pooled Gaussian=`944,443`；Q25 count/Q75 entropy/Q75 disagreement exact。
- parent replay：A1 conditional render=`12/12 byte exact`；conditional metric delta A2−A1=`0`；checkpoint=`3/3 exact`。
- selective H gate：scene-balanced coverage=`0.7199625`；accepted/abstained error=`0.0148914/0.164250`；
  error separation=`+0.149358`；全部 checks PASS。
- 边界：0471 coverage=`0.464865`，低于逐场 60%；冻结 gate 是 scene-balanced mean，A2 只作 H candidate。
- 下一门：只做 A3 correlation-aware effective count；A4、S 与后续 Stage 保持锁定。

## Stage A A3 canonical mechanism audit

- r004：relative change 被 subnormal reliability denominator 污染，保留 `done/inconclusive`，见 `V51-F06`。
- canonical r005：`/root/autodl-tmp/runs/worldsim_v51/WS-V51-M1-A-UNARY-OBSERVABILITY-01/20260817T120000Z__m1-a3-effective-count-audit-v2-s20260814-r005`
- source commit：`150b0721cb0e0acf630846d815a7ab2f287ceea8`
- parent exact：45 observations；3/3 A2 effective-count arrays float32 exact；observed Gaussian=`944,443`。
- mechanism result：no-epsilon Kish below fractional mass=`0`；meaningful absolute cap change=`0`；replacement
  amplification=`940,762/944,443`。公式没有 correlation observable，A3 rejected，不启动 quality/GPU arm。
- 下一门：只做 A4 CIF-style visibility/occupancy/conditional-identity decoupling；S 与后续 Stage 保持锁定。

## Stage A A4 canonical identifiability audit

- canonical r006：`/root/autodl-tmp/runs/worldsim_v51/WS-V51-M1-A-UNARY-OBSERVABILITY-01/20260817T122000Z__m1-a4-cif-identifiability-audit-s20260814-r006`
- source commit：`cee8b66849e5c556a79e05c813d45f225efa7814`
- A2 occupancy field=`0/3`；constant occupancy=1 与现有 renderer=`3/3 bit exact`；appearance opacity reuse=
  `3/3 non-exact`。无独立 occupancy observable，A4 在 quality read/GPU/training 前 rejected。
- Stage A H retained candidates=`A1,A2`；A3/A4 不进入候选。下一步必须 freeze-only 绑定 S screening，再一次性读取
  S=`0998/0359`；C/validation/test/KITTI 继续锁定。

## Stage A S screening freeze

- candidates=`A1/A2`；S=`0998/0359`；seed=`20260814`；SAM/split/checkpoint/operator 与 H exact inherited。
- gate=`2/2 BF1 nonnegative + >=1/2 delta>=0.001 + mean BF1>0 + mean FN<=+0.02 + calibration caps`；A2
  另需 mean coverage>=0.60 与 error concentration。
- S 后最多保留一臂；优先 A2 selective PASS，否则 A1 conditional PASS，否则回退 U2。
- r047/r048 与 one-shot read policy 见 `docs/WS_V51_STAGE_A_SCREENING_FREEZE.md`；执行前 S quality unread。

## Stage A S canonical result / closeout

- canonical r007：`/root/autodl-tmp/runs/worldsim_v51/WS-V51-M1-A-UNARY-OBSERVABILITY-01/20260817T140000Z__m1-stage-a-s-screening-s20260814-r007`
- source=`dc24f28e1de21b0fb5d1cbb41c959c3d51624a38`；status=`done`；conclusion=`stage_a_screening_selected_u2_b3`。
- A1：BF1 nonnegative=`1/2`、clearly positive=`0/2`、scene-balanced mean=`-0.0000165293`，S gate FAIL。
- A2：mean coverage=`0.557435<0.60`；error separation=`+0.119551`，但 selective gate FAIL，且 conditional 继承 A1 FAIL。
- Stage A 新 arm 全部 rejected；冻结 `U2/B3`，不再继续 Bayesian family。closeout=
  `configs/worldsim_v51/stage_a_closeout_v1.yaml`；failure delta=`V51-F09/F10`。
- 第一轮授权到此完成。Stage B 仍 `pending/locked`，C/validation/test/KITTI 继续不可读；不得把 plan 中“进入 Stage B”
  解释为自动授权执行。

## Stage B 独立授权前预检

- normative plan §10.8 与附录“八、Stage A 后如何解锁”对“Stage A 全失败”给出不兼容规则；当前 r007 正好命中
  该分支。`V51-F11` 要求用户明确选择 U2/B3 fallback 授权或关闭 M1，执行者不得自行挑选条款。
- upstream LUDVIG 只读冻结到 `4461fc515439bb498a75d71738a1e73cf7a452ed`；faithful 第一版必须使用官方
  DINOv2 ViT-g/14 registers 语义，而服务器当前没有对应 checkpoint。24GB 3090 也不能让 DINO 与 renderer 同进程/
  同卡并发常驻，详见 `V51-F12`。
- 若获独立授权，下一次提交仍只做 freeze-only：统一解锁规则，冻结 DINO checkpoint SHA/license/preprocess/PCA、
  Stage B H/S/C denominator、feature gate 与分阶段资源合同；冻结前不下载权重、不实现 method、不读质量。
- normative plan 已由 P0 按 SHA 冻结；preflight 发现 Stage A closeout 的 5 行进展造成 inherited SHA drift，见
  `V51-F13`。纠正后长计划恢复原始 byte exact；进展留在本 short plan/status/experiments。授权后若改规范，需显式
  supersede/migrate P0 binding，不能静默改 expected SHA。

## Stage B freeze-only proposal

- draft config=`configs/worldsim_v51/stage_b_freeze_proposal_v1.yaml`，明确 `executable=false`；未获独立授权前不得
  下载 checkpoint、实现 operator 或读取 feature quality。
- H/S/C 沿用 `0471/1087/0379`、`0998/0359`、`0875/0535/0436`；每场固定 15 uplift +15 evaluation views，
  240/240 image header exact、尺寸统一 1600×900，remainder 4 不读。
- DINO identity、40-D H-only PCA、B0/B1 matched formula、sidecar/resource contract、proxy leakage 与 H→S→C gate 已
  proposal 化；PCA 确定性见 `V51-F14`，membership proxy 边界见 `V51-F15`。
- 授权后的下一提交仍只做 freeze-only/P0 supersession 与 image SHA，不直接启动模型；完整序列见
  `docs/WS_V51_STAGE_B_FREEZE_PROPOSAL.md`。

## Stage B 授权迁移与当前执行

- explicit authorization overlay=`configs/worldsim_v51/stage_b_authorization_v1.yaml`；它选择 §10.8 的 U2/B3 fallback，
  显式 supersede 第一轮 stage lock 与 appendix candidate-pass-only 分支，但保持 normative plan、P0、Stage A 和 proposal
  原字节。`V51-F11=resolved by explicit user authorization`。
- canonical r001=`20260817T141000Z__m1-stage-b-input-freeze-s20260814-r001`，source=`2214961`；240 张图=
  `39,747,172 bytes`、8 个 checkpoint/Gaussian counts、terminal/manifest 均 exact。质量读取、feature extraction、
  checkpoint download 均为 false；machine freeze=`configs/worldsim_v51/stage_b_input_freeze_v1.yaml`。
- r002 single-connection 因持续低吞吐被精确停止并保留 blocked terminal/prefix，见 `V51-F16`；r003 以 14 ranges
  完成并通过 full SHA-256=`746ecb8c...a283` + S3 multipart ETag=`3d1b...-542`，`V51-F16 resolved`。当前下一门只冻结
  official DINOv2 source 并做 one-image resource smoke；asset 成功本身不解锁 H quality，之后仍需 operator parity。
- official DINOv2 external checkout 已固定为 origin=`https://github.com/facebookresearch/dinov2.git`、commit/tree=
  `7764ea0f...25fc8 / 2a27257b...12b3f43`、worktree clean。resource smoke 已在 quality read 前冻结唯一输入、
  official ViT-g/14 registers 构造、strict state-dict、last-four `[1,1536,64,114]` 输出、FP32/FP16 策略以及
  `22,528 MiB GPU / 80 GiB cgroup / 900 s` 门；禁止 smaller-model/resolution fallback，所有质量锁仍为 false。
- canonical r004=`20260817T150400Z__m1-stage-b-dinov2-resource-smoke-s20260814-r004` 在 source=`935d2b2` PASS：
  strict keys=`568`、missing/unexpected=`0/0`，4 层输出均为 `[1,1536,64,114]`；GPU sampled/Torch reserved peak=
  `6,702/6,376 MiB`，cgroup peak=`15,701,860,352 bytes`，manifest 二次复核 exact。`V51-F12=resolved`；下一门只做
  synthetic B0/B1 operator parity，H/S/C quality 继续锁定。
- LUDVIG external source 已冻结到 commit/tree=`4461fc5 / 4d1287b5...fb70d`，non-commercial license 且不 vendor。
  r005 在真实数据前只验证 normalized transpose、view-saturated B0、共同 support、duplicate/zero/chunk-order 与 lazy
  bilinear dense parity；checkpoint 前后 SHA exact，所有 quality lock 保持 false。
- canonical r005=`20260817T151900Z__m1-stage-b-operator-parity-s20260814-r005` 在 source=`1efa7dd` 以 11/11
  checks PASS；B0/B1 dense error=`0/0`、lazy error=`1.192e-7`、order bit-exact、arm difference L2=`0.0829221`，
  checkpoint immutable。下一门只做一个 H view contribution denominator smoke，仍不读取 feature/quality。
- r006 已在结果前冻结 scene-0471/frame-0/camera-0、V5 r027 checkpoint、`1e-4/1e-3` 两级 support、12 GiB GPU/
  48 GiB cgroup 门与 payload-consumption locks；只统计 renderer denominator，不保存 intersection rows、不消费 RGB/LiDAR/
  membership、不计算 feature/quality。
- r006 因 v1 错绑 motionproj interpreter、缺 `pytorch3d` 在 dataset/renderer 前 blocked，见 `V51-F19`；v2/r007 只改用
  已验证的 frozen DriveStudio Python/torch/CUDA/import contract，其他输入、门和 locks 不变。
- r007 证明 environment recovery 成功但因 v2 把 sensor `1600×900` 误作 model-native render 尺寸而 blocked；v3/r008
  exact 冻结 sensor/downscale/renderer=`1600×900 / [2,2,2] / 800×450`，见 `V51-F20`，其余合同不变。
- r008 已按正确分辨率完成 renderer/contribution 汇总，但 observed NVIDIA peak=`14,234 MiB` 越过预注册
  `12,288 MiB` ceiling 而 blocked；v4/r009 仅把 NVIDIA/Torch ceiling 提升到 `16,384 MiB` 并把失败诊断写入
  移到资源 gate 前，见 `V51-F21`。不得借该工程恢复修改算法、数据、floor 或读取质量。
- r009 已通过：raw/supported intersections=`47,378,525/32,030,248`，`1e-3` 后 Gaussian=`313,764/859,613`
  （coverage=`0.3650061132`），GPU/Torch reserved=`14,234/13,882 MiB`；checkpoint exact，manifest 8/8 exact。
  result freeze 已绑定；r010 又完成 45-view DINO sidecar/seeded-PCA，首图 repeat、PCA state 和 45 sidecar identity exact，
  `V51-F14 resolved`。r011 完成 45-view uplift 计算后因 observed NVIDIA/Torch reserved=`20,554/20,202 MiB`
  越过 `18,432 MiB` 资源门而 blocked；v2/r012 只把两项 ceiling 提升为 `22,528 MiB`，见 `V51-F23`，quality 继续锁定。
- r012 已从冻结输入完整重跑并通过：45/45 views、3/3 scenes、6 sidecars、3 checkpoint identities、19/19 manifest
  entries exact；GPU/Torch reserved=`20,554/20,202 MiB`，cgroup=`14,450,888,704 bytes`。`V51-F23 resolved`，结果已冻结。
  下一门只允许预注册 H evaluation-only proxy/repeatability/heldout reprojection，`V51-F15` 未解决前不得直接读 quality。
- 后续仍严格执行 resource smoke→operator parity→H→S→C；H 失败只拒绝当前 faithful route 并进入下一条冻结 M1
  路线，不停止整个 M1。validation/test/KITTI 与 M2/M3 不因本次授权解锁。
