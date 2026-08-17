# WorldSim V5.1 M1 执行登记

本文件是 V5.1 M1 的短执行入口；完整规范、方法树、门槛与第一轮约束以
`docs/WORLDSIM_V5_1_M1_TOPCONF_PLAN.md` 为唯一 normative plan。这里不复制长计划，只维护当前阶段、授权和证据，
避免两份计划发生漂移。

## 当前阶段（2026-08-17）

| Task ID | 状态 | 当前证据/下一门 |
|---|---|---|
| `WS-V51-P0-M1-SCOPE-FREEZE-01` | done | r001 start audit；scope/授权/quality locks exact |
| `WS-V51-D0-DEV-ROLE-FREEZE-01` | done | r001 start audit；H/S/C=`3/2/3` 与原 cohort exact |
| `WS-V51-M1-A-UNARY-OBSERVABILITY-01` | done | r007 S screening：A1/A2 rejected；freeze U2/B3 |
| `WS-V51-M1-B-LUDVIG-UPLIFT-01` | running | r004 official ViT-g resource/shape PASS；下一门 synthetic B0/B1 operator parity |
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
- V5.1 新增 failure=`V51-F01`–`V51-F16`；Stage A closeout delta=`V51-F09/F10`；Stage B preflight
  delta=`V51-F11/F12/F13`；freeze proposal delta=`V51-F14/F15`；asset recovery delta=`V51-F16 resolved`。

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
- `scripts/freeze_worldsim_v51_stage_b.py`
- `scripts/fetch_worldsim_v51_dinov2_asset.py`
- `scripts/fetch_worldsim_v51_dinov2_asset_parallel.py`
- `scripts/smoke_worldsim_v51_dinov2_resource.py`
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
- 后续仍严格执行 resource smoke→operator parity→H→S→C；H 失败只拒绝当前 faithful route 并进入下一条冻结 M1
  路线，不停止整个 M1。validation/test/KITTI 与 M2/M3 不因本次授权解锁。
