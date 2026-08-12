# Experiments

## V4 当前注册表补充（2026-08-12）

| Task ID | 状态 | 当前证据 / 下一门禁 |
|---|---|---|
| `WS-V4-B0-MATCHED-BASELINES-01` | done | 六场 strict matched baseline 与最终审计 r117 已冻结；commit `55232a4/9e322fa` |
| `WS-V4-M1-EVIDENCE-FIELD-01` | running | smoke r121 与 development r124 gate 均通过；r126 已冻结选择，当前只允许六场 validation confirmation |
| `WS-V4-M2-REPAIR-ROUTER-01` | pending | 等 M1 validation 闭环后执行 development matched router；不得提前读取 test quality |
| `WS-V4-M3-TEMPORAL-DELTA-01` | pending | 等 M2 development 冻结后执行 2–4 s nuScenes clips |
| `WS-V4-E0-NUSCENES-SCALE-01` | pending | 先完成 6 validation；18 test 仍封存且只能在 test freeze commit 后读取一次 |
| `WS-V4-D1-KITTI-ADAPTER-01` | blocked | 等用户自行复制真实 KITTI；禁止下载、禁止用 synthetic fixture 冒充数据集完成 |

### M1 development canonical

- run：`/root/autodl-tmp/runs/worldsim_v4/WS-V4-M1-EVIDENCE-FIELD-01/20260812T121734Z__m1-development6-s0-r124`；
  source=`451073a0ff592d2bd69240d0f33f9124c567eeac`，seed=`0`，单卡 RTX 3090。
- frozen arm=`raw__risk_100`，calibration=`raw`，threshold=`0.5`，temporal retention=`0.75`；
  scene accounting=`6 required / 2 evaluable / 4 abstain`。
- 相对 V3.3 O1：Boundary F1=`+0.1255247811`、FN semantic mass=`+0.0054849633`、
  Brier=`-0.0115803990`、ECE=`-0.0311158595`；M1 预注册 gate=`pass`。
- base RGB 与 checkpoint before/after exact；peak CUDA allocated/reserved=
  `8,802,204,672 / 8,927,576,064 bytes`；heldout/test quality 均未读。
- freeze audit r126=`done`；配置冻结提交=`b39d49b`。validation 数据/reconstruction 配置提交=`06d56ee`。

### M1 validation 数据提取

- r127：错误使用 DriveStudio Python，因缺 `ijson` 在提取前 fail-closed。
- r128：本地官方 10-shard 扫描期间 SSH 超时断管，terminal=`blocked/BrokenPipeError`；已写入约
  `6.7 GiB` raw union 的非空文件保留复用。
- r129：detached launcher 使用了不存在的 `/root/miniconda3/envs/motionproj/bin/python`，未形成正式 run；
  正确环境位于 `/root/autodl-tmp/envs/motionproj`。
- r130：detached parent PID=1，`done`；10-shard scan wall=`3,521.8796 s`，精确绑定
  `10,647` 个 sensor members，六场 required count=`1773/1747/1778/1784/1783/1782`；
  summary/inventory SHA=`6f0b0933...76854c / 41b2d5eb...ee5c`，Git=`06d56ee` clean，
  `no_download=true`、test quality 未读。`extracted_this_run=0` 表示完整复用 r128 已原子写入的非空文件。


- 更新时间：2026-08-12
- 当前路线：WorldSim V4 / EviDelta-GS
- 当前执行授权：`WS-V4-B0-MATCHED-BASELINES-01` 6-development-scene evaluator/baseline；不得读取 test quality
- 当前方案：[`WORLDSIM_V4_EVIDELTA_GS_PLAN.md`](WORLDSIM_V4_EVIDELTA_GS_PLAN.md)
- V3.2 终局归档：[`archive/2026-08/worldsim-v3.2/`](archive/2026-08/worldsim-v3.2/README.md)
- V3.1 终局归档：[`archive/2026-08/worldsim-v3.1/`](archive/2026-08/worldsim-v3.1/README.md)
- V2 历史方案：[`DYNAMIC_DRIVING_EDITING_DIAGNOSTIC_PLAN_V2.md`](DYNAMIC_DRIVING_EDITING_DIAGNOSTIC_PLAN_V2.md)
- V1 最终台账：
  [`archive/2026-07/dynamic-reconstruction-v1/EXPERIMENTS.md`](archive/2026-07/dynamic-reconstruction-v1/EXPERIMENTS.md)

本文件保留 V2 完整执行证据，并从 2026-08-05 起登记 V3。V2 M0–M4 已完成；M5 部分执行后停止扩张，
保持 `pending` 历史终态；M6–M8 不再授权。A0、A1 与 A2 已完成；A2 fixed/matched 正式裁决为
`tradeoff_non_dominated`。`WS-V3-A3-LOCAL-REFINE-01` 已以 R1 资源门失败和 diagnostic tradeoff 的负结果
`done`，`A3*=R0-off`；A4 已 `done`，P0、P5、P1、P2 与 P3 全部闭环；P1 候选被质量门拒绝并回退到 source，
P2 canonical r2 选择 mixed checkpoint，P3 canonical r1 选择 exact chunk package。F0 canonical audit 已 `done`，
本机 inference 前置条件未通过，F1=`conditional_not_unlocked`；R0 canonical 已 `done`，V3.1=`none_plan_complete`。
V3.1 终态保持冻结；V3.2 S1 canonical r6 已完成，S3 canonical r3 已生成并选定 Asset Harvester
2-view actor asset，S2 canonical r3 已选定 3DGIC-adapted generated-background checkpoint。S4 canonical r3
完成 non-temporal JIT，但删除语义保持门失败，只保留 optional diagnostic；temporal arm 受 gated Cosmos base
阻塞。R0 canonical r4 已完成语义扩展、mixed storage、S3 actor override registry、exact chunk package 与三视角
单卡验证；全部单卡 RTX 3090 可执行环节终态为 `done`，S5 继续 `blocked`。V3.2 整体终态为
`none_plan_complete`；本注册表自 2026-08-11 起冻结为历史事实，不授权续跑 S4 temporal、S5 或旧 rejected run。

## V4 注册表

| Task ID | 状态 | 目标 | 当前证据/门禁 |
|---|---|---|---|
| `WS-V4-P0-SCOPE-PAPER-FREEZE-01` | done | 冻结 claim、HEAD、数学、baseline、数据、指标与一手来源 | `main@2108430`；15 sources；KITTI missing 如实阻塞；P0 不训练 |
| `WS-V4-D0-NUSCENES-COHORT-01` | done | 结果前冻结 6 dev + 6 val + 18 test | canonical r4；850 candidates；2-scene preprocess smoke passed；无训练/test quality |
| `WS-V4-D1-KITTI-ADAPTER-01` | blocked | 本地 tracking/raw adapter | code + 12-gate synthetic contract done；canonical r2=`blocked_local_dataset_missing`；禁止下载 |
| `WS-V4-B0-MATCHED-BASELINES-01` | running | V3.3/StreetGS/AD-GS same-split replay | 当前唯一授权：6 development scenes evaluator；M1 仍未授权 |
| `WS-V4-M1-EVIDENCE-FIELD-01` | pending | Bayesian/calibrated temporal evidence | B0 未闭环前未授权 |
| `WS-V4-M2-REPAIR-ROUTER-01` | pending | Bayes-risk repair compiler | M1 未收口前未授权 |
| `WS-V4-M3-TEMPORAL-DELTA-01` | pending | `SE(3)` B-spline temporal delta | M2 未收口前未授权 |
| `WS-V4-E0-NUSCENES-SCALE-01` | pending | 6 dev + 6 val + 18 test | test freeze 提交前不得读取 test quality |
| `WS-V4-E1-KITTI-CROSSDATA-01` | pending | 2 smoke + 10 formal | D1 exact 且 nuScenes 参数冻结后解锁 |
| `WS-V4-E2-ENGINEERING-BENCH-01` | pending | 生产效率 benchmark | 方法冻结后执行 |
| `WS-V4-E3-DOWNSTREAM-GAP-01` | pending | perception real-to-sim gap | 方法冻结后执行 |
| `WS-V4-H0-HUMAN-STUDY-01` | pending | 条件式人评 | 只有自动/定量门完成后单独解锁 |
| `WS-V4-R0-RELEASE-01` | pending | exact paper release | 前序收口后解锁 |
| `WS-V4-W0-PAPER-01` | pending | 技术报告 / paper | 只写已有证据 |

### `WS-V4-P0-SCOPE-PAPER-FREEZE-01` closeout

- 实际起点为 `main@21084309480895f5541196a06191a5dffb4e30c1`；计划草案里的 `144ed19` 保留为历史点；
  `e6663e1` 和 `144ed19` 均在当前 `main` 历史中；
- 创建 `research/worldsim-v4-evidelta`，冻结 EviDelta-GS 三模块、30-scene nuScenes 6/6/18 split、
  KITTI frozen cross-domain、Tier A/B baseline、PSNR/SSIM/LPIPS-Alex、scene-level statistics 与工程指标；
- 审计 15 个一手工作；“官方论文/源码存在”与“本机 executable”分开记录，不可执行路线不填数值；
- `/root/autodl-pub/KITTI` 当前不存在，登记 `blocked_local_dataset_missing`；没有创建目录或网络下载；
- P0 时 GPU=`RTX 3090 24,576 MiB`、GPU process=`0`、cgroup OOM/kill=`0/0`、disk free 约 `193 GiB`；
- training/model inference/weight download 均为 false；P0 只解锁 D0。
- canonical=`20260811T080636Z__p0-scope-formal-s0-r2`；config/summary/manifest/status SHA=
  `248bde62...aaa8 / aba1fbcf...1283 / ec32e983...3970 / b3941601...272a`；5 份 source snapshot exact，
  run=`101,624 bytes`；r1 因提交前只规范计划参考文献 whitespace 导致 source/config SHA 改变，保留为 noncanonical
  done，r2 对最终字节重新审计。

### `WS-V4-D0-NUSCENES-COHORT-01` closeout

- 从官方 nuScenes `v1.0-trainval` 元数据枚举 850 个候选，只按结果前 metadata 分层；development/validation
  全来自官方 train，18 个 test 全来自官方 val，scene name/token 全局互斥；
- 冻结 6 development、6 validation、18 test；完整配置保存 30 scene 的 high/difficult actor、remove/lateral/insert、
  2–4 秒 continuous clip、3-front-camera + LIDAR_TOP 与逐帧 train/development/heldout 划分；
- cohort SHA=`eda9f684...44578`；formal 构建会逐字段比较 `scene_records`，不只校验名单；30/30 scene 均找到
  high/difficult actor，连续片段范围=`2.899163–3.150218 s`；
- scene-0230/0242 只复用已有 preprocess 产物做 smoke：每 scene `196` 帧、`1,176` 图像、`196` LiDAR，实例 JSON
  与抽样 artifact SHA 全通过；没有训练、模型推理或质量读取；
- canonical=`20260811T084108Z__d0-cohort-formal-s40117-r4`；config/summary/manifest/status/cohort SHA=
  `ed47c0da...667a1 / ec96970d...f276 / 3349a636...6b30 / 1dfd5db4...4461 / eda9f684...44578`；
  12 files=`1,221,825 bytes`，11 个 metadata table fingerprint exact；diagnostic r1 保留为 noncanonical；r2 因
  CRLF→LF source bytes 改变降为 noncanonical；r3 被 cohort SHA freeze gate 拦截并记录 `blocked` terminal，定位为
  set 浮点求和顺序受 Python hash seed 影响；r4 使用排序 + `math.fsum` 并恢复原 cohort SHA；
- D0 定向测试=`8 passed`，D0/P0 + V3.3/V3.2 联合回归=`106 passed`。下一任务 D1 必须在 public KITTI
  缺失时输出合法 blocked terminal，禁止下载；随后才进入 6-development-scene B0 matched baseline，M1/M2/M3 与
  test quality 仍未授权。

### `WS-V4-D1-KITTI-ADAPTER-01` blocked closeout

- 实现 tracking-first / raw-fallback 自动 layout discovery、KITTI 原生 `image_02/image_03` 双相机合同、calibration、
  track ID/3D box、LiDAR/box projection、pose/timestamp、stereo 与 frame-leak 检查；KITTI threshold search=false；
- synthetic tracking/raw fixtures 的 12 项 adapter gates 全通过，定向测试=`7 passed`；D1/D0/P0 + V3.3/V3.2
  联合回归=`113 passed`；这些测试不冒充真实 KITTI smoke 或 cross-domain quality；
- `/root/autodl-pub -> /autodl-pub/data`；requested `/root/autodl-pub/KITTI` 与 resolved
  `/autodl-pub/data/KITTI` 均不存在，没有创建目录或下载；terminal=`blocked_local_dataset_missing`；
- canonical blocked run=`20260811T085210Z__d1-kitti-layout-formal-s0-r2`；
  config/summary/manifest/status/fingerprint SHA=`bffe6eaa...93d4 / 05eeb265...9ca9 / 7c9ca256...f582 /
  598eac51...d54b / 233e464e...fdc5`；14 files=`45,214 bytes`；r1 只记 physical path，r2 补齐 requested +
  symlink-resolved provenance；
- D1 只在真实数据挂载后以新 run 复开；E1 继续 pending。当前转入 B0 nuScenes development，不因此宣称
  single-card closure，也不读取 test quality。

### `WS-V4-B0-MATCHED-BASELINES-01` evaluator + inventory milestone

- 冻结 `metrics_v1.yaml`：PSNR/SSIM/LPIPS-Alex、五类区域、无 GT/空区域 undefined、scene-level
  mean/median/std/IQR/bootstrap CI 与 paired bootstrap/sign-flip/Wilcoxon；failed/blocked/abstain 保留 denominator；
- baseline 区域协议显式冻结为 actor=`dynamic_masks/all>0`、static=`not actor and not egocar`、boundary=L1
  半径 3 px 形态学带，未编辑 baseline 的 edit_roi 为空；development-only scene evaluator 已加入文件/partition/重复键门；
- 实现统一 evaluator、scene statistics 和 engineering raw-row 派生，定向测试 `9 passed`；该里程碑未启动训练、
  模型推理或 test quality 读取；
- `baseline_matrix_v1.yaml` 精确锁 D0 六个 development scenes、same split/resolution、DriveStudio
  `e59bda4`、V3.3 `e6663e1` 与 AD-GS `9a20851`；分辨率层级固定为 sensor `1600×900`、source downscale=2、
  model/metric `800×450`；historical metric/provenance 不算 executable；
- diagnostic inventory=`20260811T090951Z__b0-inventory-diagnostic-s0-r1`，terminal=
  `blocked / matched_baseline_assets_incomplete`，coverage=`V3.3 1/6、StreetGS 0/6、AD-GS 0/6`；run 只说明
  当前磁盘资产缺口，B0 task 继续 `running`；
- 三个历史 StreetGS checkpoint 的 summary/bytes provenance 仍可审计，但文件当前不在磁盘；AD-GS historical
  六场景 metrics 存在，而 source/env/checkpoint 不存在。后续必须新 run 重训/恢复，不能把历史数值填入 matched 主表。
- raw extraction canonical r3 从官方 10 个 nuScenes tar shard 物化缺失 scene-0048/0139/0994，`5,264/5,264`
  members，fingerprint=`c51f4162...e38`；没有下载数据或读取 test quality；
- preprocess r5/r6/r7 均 done；六个 scene index `045/110/179/191/204/752` 现均为 `1,176 RGB / 196 LiDAR`。
  r4 的真实上游输出保留，但因 `_10Hz` root 合同不匹配为 blocked；
- remote sky-model r10 因网络不可达 blocked；官方 fixed-revision 本机 exact staging 由 r11 离线恢复完成，fingerprint=
  `15c200fd...90bfd`。sky r12 因预建空目录误拒绝 blocked；r13/r14/r15 分别为三个新 scene 原子生成
  `588/588` masks，fingerprint=`c35a9dbe...c3eb / 5f6a2fe7...c702 / 455dcb20...e6c7`；
- StreetGS r9 在 iteration 前因缺 sky mask blocked；r16 profile100 完成，wall=`90.8965 s`、checkpoint=
  `340,298,602 bytes / SHA 446297b8...3af`、peak GPU=`9,004 MiB`、OOM/kill=`0/0`。该 profile 只解锁
  六场景 30k formal，B0 coverage 仍为 `0/6 formal`，M1 与 test quality 继续未授权。
- StreetGS r17/r20/r22/r24/r26/r28 均完成 30k、means finite、OOM/kill=`0/0`，但 2026-08-12 复核确认其
  `test_image_stride=10` 不满足冻结的 `sample_index mod 5` 三分区合同；六个 run 与 r29=`6/1/0` 只保留为
  protocol-mismatch provenance，不计 strict matched coverage；
- strict scene-0230 r32=`20260811T154831Z__streetgs-scene0230-matched-formal30k-s0-r32` 30k done，wall=
  `3,200.0184 s`，checkpoint=`386,410,166 bytes / SHA 766648bf...af97cd1`，Background/RigidNodes=
  `1,095,606/172,264`，peak GPU=`23,892 MiB`、peak cgroup=`15,281,917,952 bytes`，OOM/kill=`0/0`，
  test quality 未读；
- corrected inventory r33=`20260811T165009Z__baseline-matched-correction-s0-r33`，terminal blocked，strict
  coverage=`StreetGS/V3.3/AD-GS 1/1/0`，inventory/fingerprint SHA=
  `73d36544...bea9e / c19fba13...e285853`；它覆盖当前结论但不覆盖旧 run；
- strict scene-0242 r46=`20260811T210253Z__streetgs-scene0242-matched-formal30k-s0-r46` 30k done，
  wall=`1,998.0482 s`，checkpoint=`302,953,462 bytes / dd41a34d...52bc0`，Background/Rigid=
  `824,583/92,170`，peak GPU/cgroup=`17,530 MiB / 23,842,824,192 bytes`，OOM/kill=`0/0`，
  无 test/full render；r47 内容寻址 inventory=`StreetGS/V3.3/AD-GS 2/1/1`，
  inventory/fingerprint=`89c72659...eafad / b91f7c76...712d6`；
- strict scene-0255 r48=`20260811T214009Z__streetgs-scene0255-matched-formal30k-s0-r48` 30k done，
  wall=`2,392.0649 s`，checkpoint=`444,340,086 bytes / dba24982...cb2d2`，Background/Rigid=
  `1,478,401/38,721`，peak GPU/cgroup=`23,932 MiB / 24,132,476,928 bytes`，OOM/kill=`0/0`，
  无 test/full render；r49 inventory=`StreetGS/V3.3/AD-GS 3/1/1`，
  inventory/fingerprint=`79b6b1d0...c86c85f / bd822e61...9641a`；
- strict scene-0048 r50、scene-0994 r52、scene-0139 r54 均 30k done，wall=
  `2,179.8120/1,806.3743/2,069.0543 s`；checkpoint bytes=`332,725,750/279,185,462/314,307,830`，
  SHA=`70d02a0b...b00d2 / 3e2b2534...3aea / 4fff4452...8dfe`；Background/Rigid=
  `1,030,993/15,717 / 819,952/932 / 962,074/7,219`，peak GPU=
  `23,694/20,970/23,056 MiB`，资源事件与 test/full render 均为 `0`；
- StreetGS 六场 registry 提交=`a4ee23a`；r55=`20260812T001330Z__baseline-streetgs-sixscene-registration-s0-r55`
  在 clean HEAD 得到 coverage=`StreetGS/V3.3/AD-GS 6/1/1`，inventory/fingerprint=
  `8bc62596...be3a1 / 4f12c1d2...32372`；
- AD-GS official exact source=`9a208512`、DPT/CoTracker weights 与 train-only adapter 已恢复；环境 r34=
  `20260811T165030Z__adgs-environment-restore-r34` 离线完成，torch=`2.1.2+cu118`，两个 CUDA 扩展真实
  forward/backward smoke passed，OOM/kill=`0/0`。该结果只解锁 strict preprocess/profile，不计 scene coverage；
- AD-GS preprocess r35/r36/r37 分别被 run-dir 不可变门、official source 未跟踪 build 目录、可选 `flow_vis`
  诊断依赖 fail-closed；均未训练或读取 dev/heldout。r37 已完成的 adapter/depth/segment partial 移入规定 backup，
  no-visualization flow 合同与 run-local extension build source 作为新提交修复，不覆盖旧 run；
- AD-GS scene-0230 canonical train-only preprocess r38=
  `20260811T171507Z__adgs-scene0230-preprocess-s0-r38` done，文件计数=`354/354/354/354/285`，flow wall=
  `3,046.9930 s`，peak GPU/cgroup=`20,112 MiB / 22,384,893,952 bytes`，OOM/kill=`0/0`，无可视化视频且
  development/heldout/test quality 均未读；fingerprint/manifest=`d44a0530...fbf2c37 / cefae230...c05873`；
- profile r39 因训练 import 全局依赖可选 `flow_vis`、r41 因 inherited PyTorch3D binary 无 sm86 KNN kernel
  分别在 iteration 前 blocked；均未读 dev/heldout。r40 exact `roma 1.5.7` runtime restore done；r42 从 clean
  `pytorch3d@2f11ddc5` run-local 重编 sm86，build=`1,151.39 s`，KNN + Gaussian CUDA smoke passed，
  新 binary=`10,313,184 bytes / eca71e2c...e3084`，fingerprint=`03ec74e8...05f7fb`；
- profile r43=`20260811T185145Z__adgs-scene0230-profile100-s0-r43` done，100 steps train stage=`40.3595 s`，
  peak GPU/cgroup=`6,012 MiB / 30,991,519,744 bytes`，三文件 checkpoint SHA=
  `bc364930...16c5 / 8205e276...e5ab / 45644cb4...8e77`，test quality 未读；profile 不计 coverage；
- formal r44=`20260811T185600Z__adgs-scene0230-formal60k-s0-r44` done，60k stage=`7,054.6221 s`，
  peak GPU/cgroup=`16,692 MiB / 33,680,572,416 bytes`，OOM/kill=`0/0`；三文件 bytes=
  `413,905,347/435,921,657/805,307,528`，SHA=`f17ed27f...a0cbb / c725f952...c84b0 /
  c3233b71...e4d34`，train-only partition 且 development/heldout/test quality 均未读；
- fail-closed registry 提交=`904e395`；r45=`20260811T205840Z__baseline-adgs-formal-registration-s0-r45`
  重新校验 runtime/source/patch、formal step、run 内三文件 bytes/SHA 与 fingerprint/manifest，得到 strict coverage=
  `StreetGS/V3.3/AD-GS 1/1/1`，inventory/fingerprint=`4bf7cf68...ad6b / 3db524d2...49e5`；
- run-local extension build source=`9f839fd`，baseline region/evaluator=`abd82d8`，runtime import/sm86 fixes=
  `c3600be/eed00cb`；联合定向测试=`50 passed`；
- B0 继续 running；StreetGS strict 六场已齐，下一步补齐 AD-GS/V3.3 其余五场 same-split，
  再运行统一 evaluator。M1 与 test quality 继续未授权。

## V3.3 注册表

| Task ID | 状态 | 目标 | 当前证据/门禁 |
|---|---|---|---|
| `WS-V33-P0-ROUTE-SOTA-AUDIT-01` | done | 冻结 V3.2、切换分支、审计 source/license/weights/hardware | canonical r2；10 sources；5 个 V3.2 大资产 SHA exact；36+4 tests |
| `WS-V33-S1-OBJECT-AWARE-GS-01` | done | SAM2.1 fallback + dual instance-opacity field | canonical formal r9；O1 selected；base exact；51 tests |
| `WS-V33-S2-ROADPATCH-INPAINT-01` | done | RoadPatch-Lite + Inpaint360GS baseline | r10 index + r11 B1 canonical；r12 B2 blocked_single_3090 |
| `WS-V33-S3-ASSET-VIEWSELECT-01` | done | Asset Harvester auto 1/2/4-view | high A4 heldout accepted；boundary override ABSTAIN；63 tests |
| `WS-V33-S4-SPATIAL-DELTA-01` | done | immutable base + erase/insert delta | canonical r7/r8；20 rollback exact；posterior-gated erase；9 tests |
| `WS-V33-S5-SEMANTIC-RENDER-01` | done | semantic-gated render；R3D2 conditional | canonical r4；G1 heldout rejected；G0 production；delete 5/5 safe |
| `WS-V33-R0-INTEGRATION-01` | done | 单卡完整集成与 exact package | canonical r7；44 inputs；10/10 gates；76-file release；v33_supported |
| `WS-V33-F0-LIDAR-EVS-AUDIT-01` | pending | 条件式未来 LiDAR 扩展 | 不阻塞 R0，当前未授权 |

### `WS-V33-P0-ROUTE-SOTA-AUDIT-01` canonical closeout

- branch=`research/worldsim-v3.3-object-maintenance`，baseline=`a055fc6727dddacd194665d5c997a1fe47c2d2f4`；
- canonical=
  `/root/autodl-tmp/runs/worldsim_v33/WS-V33-P0-ROUTE-SOTA-AUDIT-01/20260810T171744Z__p0-source-audit-s0-r2`，
  terminal=`done`；config/summary/manifest/status SHA-256=
  `29c167fe...9ae2 / 08806b5f...d68a / 2603ff0e...a12d / 91096d0e...072d`；
- r1 的审计门全部通过，但 runner 未把 auditor/module/test 快照进 run，故保留为 noncanonical `done` 证据；r2 新增
  三份 source snapshot（SHA=`96024691...c065 / 5760318f...fec3 / 0fe51b7c...a778`），不改写 r1；
- 10 个 source=`2 executable / 2 weights_blocked / 5 source_not_released / 1 audit_only`；5 个有 Git checkout 的新 source
  与继承 Asset Harvester 均 commit/tree clean exact，存在的 5 份 root license SHA exact；
- SAM3.1=`96914d2`，当前 `hf auth whoami=Not logged in` 且无 cached checkpoint，故 SAM3 arms `weights_blocked`；
  OP2GS/3D-GIMP/FocusGS/LiDAR-EVS 无官方 runnable source；GS-RoadPatching=`468f812` 只有 project-page assets；
- Inpaint360GS=`d54c893 / Apache-2.0 / source executable`，只在 S2 adapter 后做最小单卡 preflight；R3D2=
  `3fc6e31 / Apache-2.0 / exported author model absent`，不从零训练；GOR-IS=`eb36acc / noncommercial audit_only`；
- D2/S2/S3/mixed/chunk 五资产重新 hash 全 exact；V3.2 R0 8/8 gates 与 `36 passed` 回归复核通过；P0 auditor=
  `4 passed`；第一次无 `PYTHONPATH=.` 的 pytest 只在 collection 阶段失败，按仓库入口重跑无逻辑失败；
- GPU=`RTX 3090 24,576 MiB`、cgroup max=`96,636,764,160 bytes`、OOM/kill=`0/0`、disk free 约 `40 GiB`；
  P0 training/model inference/install/large-weight download/DriveStudio mutation 均为 false；
- P0 关闭时只解锁 S1；该历史门禁已由下节 S1 closeout 满足，当前授权以文件头与 STATUS 为准。

### `WS-V33-S1-OBJECT-AWARE-GS-01` canonical closeout

- diagnostic r0 完整贯通 O0/O1/O3，但 `diagnostic_steps_override=1`，只作工程证据；development smoke r1
  使用固定 10 个 development frames、零 heldout，O1 相对 O0 同时改善 boundary F1、NBD、IoU 与 FP mass，
  因而冻结 `formal_selected_arm=O1_dual_opacity`；O3 未入选；
- heldout-target r2 因外部清理后的旧 SAM Python 路径不存在而 fail closed；r3 在首个 block 暴露 logits rank
  兼容错误，未正式发布 mask；修复后 canonical r4=`20260810T180231Z__s1-heldout-targets-s0-r4`，
  31 blocks / 37 accepted / 0 rejected，summary/manifest/status SHA=
  `686c7100...20ff / d2b2ab14...3050 / b186664e...b4d1`；
- r4 使用 SAM2 source=`2b90b9f`、checkpoint=`898,083,611 bytes / 2647878d...318`；隔离 runtime=
  Python `3.10.20`、torch `2.5.1+cu124`、torchvision `0.20.1+cu124`，peak reserved=
  `2,099,249,152 bytes`；19 个 heldout frames 从文件级标记 `optimization_forbidden=true`；
- canonical formal r9=
  `/root/autodl-tmp/runs/worldsim_v33/WS-V33-S1-OBJECT-AWARE-GS-01/20260810T183154Z__s1-instance-field-formal-s0-r9`，
  terminal=`done`；config/summary/manifest/status SHA=
  `9afa48aa...3150 / 4ab311a6...8c4a / e1b858fd...584e / 9394d15e...03b9`；
- O0 heldout：boundary F1=`0.068960`、IoU=`0.063253`、NBD=`0.144958`、FP/FN mass=
  `0.900308 / 0.061278`；O1：`0.336158 / 0.330727 / 0.105280 / 0.623276 / 0.109356`；
  对应 boundary F1 `+387.47%`、IoU `+422.87%`、NBD `-27.37%`、FP mass `-30.77%`，但 FN mass
  增加，按范围限定为 precision/boundary breakthrough，不宣称所有指标全面支配；
- O1 field=`5,882,296 bytes`、SHA=`23b2403ccb47e2e2c6b5fa3d22a9a6d93815d9f9bcbc6d11b66f035831adc8d7`，
  包含 `1,309,868` 行、`65,989` assigned/trainable Gaussian；11 个近似身份冲突背景点 fail closed 未分配；
- formal optimization=`300 steps / 15.115s`，peak allocated/reserved=
  `8,001,482,240 / 8,084,520,960 bytes`；D2 checkpoint SHA 前后 exact，base means/scales/quats/SH/RGB
  opacity 均未进入 optimizer；
- r5 为输入正确但缺 runner 入口 identity 三元重核的 noncanonical formal；r6 补齐重核，但暴露
  `np.savez_compressed` ZIP timestamp 导致相同 O0 数组文件 SHA 漂移；r7 改为固定排序/timestamp/权限/压缩参数，
  同一数组重写后 O0/O1 SHA exact，但其 source snapshot 与提交前 EOF 空白清理不再 exact；r8 在同步 SSH 超时后
  以 SIGPIPE/141 failed；r9 由后台托管重跑，9 个源码快照与待提交文件 SHA exact。r7→r9 O1 有小幅 CUDA
  数值漂移，但两 arm aggregate 指标 exact；
- 定向回归=`51 passed`，py_compile、bash syntax、git diff check 通过；下一步只解锁 S2。

### `WS-V33-S2-ROADPATCH-INPAINT-01` canonical closeout

- diagnostic r0 因外层 `nohup` 在注册前创建 run.log 而拒绝 non-empty run 目录；r1 识别出旧 P3 绑定 P2 FP16
  且使用 `(x,y)` 网格；r2 识别出内参为 9 值；三者均保留 failed，未续写；
- r3 的 whole-cell geometry gate 被天空/立面 outlier 污染，`53,541` patches 中 `0` valid；r4–r6 引入
  row-level fail-closed、densest vertical slab 与明确的 sidecar/front-camera support，得到 `822` valid patches；
- canonical index r10=
  `/root/autodl-tmp/runs/worldsim_v33/WS-V33-S2-ROADPATCH-INPAINT-01/20260810T193004Z__s2-patch-index-formal-s0-r10`，
  terminal=`done`；config/index/manifest/summary/status SHA-256=
  `34022784...af79 / 51561eec...a4c / 565741c5...8845 / 4216c652...f1b / 754ab982...063`；
- r10 从 D2 FP32 `1,205,164` 个 native Background rows 中保留 `702,506` eligible rows，建立
  `15,591` 个 1/2/4 m patches，valid=`617/160/45`；index=`4,146,483 bytes`，generated donors=`0`；
- high/boundary 两个 target anchors 由 S1 object-aware delete mask ∩ accepted SAM2 mask、target-view first-hit
  depth 与 cross-view support 冻结，均选择最小覆盖的 4 m patch，且各自 top-5 通过 geometry/visibility/separation；
- r7 完成真实 GPU anchor/top-5 preflight；r8 用 2,150-row dense delta 时 heldout PSNR/SSIM delta=
  `-0.8553 dB/-0.00619`，保持 rejected；r9 在冻结 `maximum_rows_per_target=512` 后以 104 rows 通过门，
  但在 formal index manifest 前，仅作 diagnostic；
- canonical RoadPatch r11=
  `/root/autodl-tmp/runs/worldsim_v33/WS-V33-S2-ROADPATCH-INPAINT-01/20260810T193140Z__s2-roadpatch-formal-s0-r11`，
  terminal=`done`；summary/acceptance/selection/delta/status SHA-256=
  `5de28a02...17d / 9be39845...cd0 / da54ad6d...0f0 / a3105313...014 / f4780ce1...c87`；
- r11 选择 high=`p4-x-000008-z+000009 / 25 rows`、boundary=`p4-x-000009-z+000009 / 79 rows`；
  delta=`24,557 bytes`，保留 exact native row/chunk provenance、rigid `(x,z)` transform、opacity feather、bounded
  RGB affine、scale clamp；candidate 只临时挂载 Background 后渲染并恢复同一对象，base checkpoint 不变；
- heldout B0→B1 mean：PSNR `28.1571546740→28.0731240061`、SSIM
  `0.87145031937→0.87054233897`、LPIPS `0.14966575801→0.15152705088`；冻结门分别为
  `-0.1 dB/-0.005/+0.01`，全部通过；static PSNR `+0.00286484865 dB`，static LiDAR MAE
  `-0.00525204837 m`；
- r11 wall=`69.335 s`，peak CUDA allocated/reserved=
  `8,337,670,144 / 8,420,065,280 bytes`；D2 checkpoint before/after SHA exact；
- Inpaint360GS canonical preflight r12=
  `/root/autodl-tmp/runs/worldsim_v33/WS-V33-S2-ROADPATCH-INPAINT-01/20260810T193426Z__s2-inpaint360gs-preflight-s0-r12`；
  source=`d54c893285c6cb27788e05cce607e7d3cca6388a`、tree clean、Apache-2.0 license
  SHA=`41d80577...5a10`；config/preflight/summary/status SHA=
  `f6dc0291...1845 / 91b5c6a0...30b / 263f336b...8e3 / 292e90df...ec9`；
- 官方要求 RTX 4090/CUDA 11.8、main+LaMa 隔离环境以及 CropFormer/Big-LaMa/SAM/DeAOT/GroundingDINO
  权重；当前 RTX 3090 24,576 MiB 上上述环境/权重与 StreetGS adapter 均缺失，故
  `blocked_single_3090`、`official_execution_attempted=false`；不生成 B2 质量指标；
- RoadPatch 专项=`6 passed`，V3.3/V3.2 定向回归=`52 passed`，py_compile 与 r10/r11/r12 的 8 个
  canonical source snapshots byte-exact；
- S2 以 B1 RoadPatch 为 canonical，B2 作为可审计外部阻塞保留；下一步只解锁 S3。

### `WS-V33-S3-ASSET-VIEWSELECT-01` canonical closeout

- high selector diagnostic r1 与 formal r2 的 selection/input manifest byte-exact；r2=
  `/root/autodl-tmp/runs/worldsim_v33/WS-V33-S3-ASSET-VIEWSELECT-01/20260810T201345Z__s3-viewselect-high-formal-s0-r2`；
  `130` candidates / `119` eligible，heldout/development read=false；selection/input/summary/status SHA=
  `192e5035...b7be9 / 34b1e09e...ef2e7 / 65485a63...0311 / b4c0af50...376c`；
- high 自动集合：A1=`f091 c1`，A2=`f000 c0 + f091 c1`，A4=`f011 c0 + f083/f089/f094 c1`；
  selector 使用 frozen area/mask/sharpness/D2 visibility/occlusion/truncation Q_view 与 yaw/time/camera Q_set；
- high AH r3=`20260810T201830Z__s3-asset-high-formal-s0-r3`，official source=`767b243` clean、
  三权重和 VAE/C-RADIO exact、HF offline；PLY SHA=`13ff42b6...299b / 9bb7f925...8c4a / 9e9875b8...6ca0`，
  inference manifest=`e33fcc65...2d90`，wall=`160.189 s`，peak=`20,137 MiB`，OOM/high=`0`；
- importer r4=`20260810T202210Z__s3-import-high-formal-s0-r4`；A1/A2/A4 Gaussian=
  `101,988/100,783/99,241`，asset SHA=`00397310...5290 / 1a6d9300...6859 / 06d5db85...ec13`，
  deterministic reserialization/reload exact；
- high development canonical r13=`20260810T205300Z__s3-eval-high-development-formal-s0-r13`；
  A0/A1/A2/A4 IoU=`.669876/.658477/.664463/.701490`，boundary F1=`.517563/.497629/.544166/.604799`，
  LPIPS=`.090755/.092720/.095948/.098533`，PSNR=`17.718824/18.480375/17.980733/17.694031`；
  三 auto arm 的六项 retention gate 全过，A4 由冻结 metric order 选中，decision=`28d4f75c...82bf`；
- high heldout canonical r14=`20260810T205600Z__s3-eval-high-heldout-formal-s0-r14`；只比较 A0/A4，
  IoU=`.704974→.728464`、boundary F1=`.505017→.564906`、LPIPS=`.094170→.102697`、
  PSNR=`17.025697→17.009936`；四项 gate 全过，decision=`795ecbc5...8032`，无 optimization，checkpoint exact；
- boundary selector diagnostic/formal exact；formal r8=`20260810T203700Z__s3-viewselect-boundary-formal-s0-r8`，
  `135/127` candidates/eligible，A4=`f039 c0 + f146/f151/f156 c1`，yaw=`17.95°→84.53°`；selection/input=
  `ba05b224...505f / 5cb0ab6c...5856`；
- boundary AH r9/import r11 完成，inference=`2bd2d011...54bd3`，A4=`94,835 Gaussian / 3,632,764 bytes /
  9b2295e5...5dd1f`；错误 CLI SHA 的 importer r10 fail-closed 并保留，未续写；
- V3.2 无 boundary manual asset；r12 使用 immutable D2 native baseline。native vs A4 IoU=
  `.666562/.624832`、boundary F1=`.555343/.492141`、LPIPS=`.015414/.111043`、PSNR=
  `31.006882/16.396747`；A4 retention 失败，生产 override=`ABSTAIN_GENERATED_OVERRIDE`，boundary heldout 未读；
- scene-0242/0255 缺少同协议冻结的 V3.3 S1/S2 输入链，且 boundary transfer 已拒绝，条件确认未执行；
- selector r0 的 `obj_to_world` list schema 与 importer r10 的错误 CLI hash 都按新 run 修复；high r5/r6 因最终
  evaluator snapshot 漂移降为有效 noncanonical，r13/r14 重跑后与提交态 byte-exact；
- S3 专项=`11 passed`，V3.3/V3.2 定向回归=`63 passed`；py_compile/diff check/source snapshot exact；
  下一步只解锁 S4，且只允许 high A4=`06d5db85...ec13` 进入 production delta。

### `WS-V33-S4-SPATIAL-DELTA-01` canonical closeout

- initial all-hard package r1 completed；real-render r2=`rejected`，唯一失败门为 target 外 L1
  `0.8219646>0.5`；其余 exact rollback、toggle、resource、checkpoint/registry integrity 均通过；
- r2 暴露 S1 的 36,736 个 high Background hard assignments 大多是低 instance-opacity 候选；最终 protocol
  保持原视角/门不变，以概率 MAP 边界 `p(instance)>=0.5` 选择 Background，Rigid core 4,525 行仍全部 ERASE；
- canonical package r7=
  `/root/autodl-tmp/runs/worldsim_v33/WS-V33-S4-SPATIAL-DELTA-01/20260810T221300Z__s4-package-canonical-s0-r7`；
  config/package/summary/status SHA=`4b318a67...508a / 3be8ce88...ee43 / cbde9600...0c63 / 4c8332d6...375f`；
- package 只含 base checkpoint/registry reference descriptor，delta/inventory/stacks 共 `4,007,120 bytes`，
  最大 payload=`3,942,422 bytes`，完整 checkpoint copy=`0`；r3/r5/r7 package manifest byte-exact；
- ERASE=`1,614 Background + 4,525 Rigid`，base row deletion=`0`、runtime effective opacity exact zero；
  high RoadPatch=`25` rows（不混入 boundary 79 rows），A4 actor=`99,241` rows，insert provenance 逐行完整；
- canonical evaluation r8=
  `/root/autodl-tmp/runs/worldsim_v33/WS-V33-S4-SPATIAL-DELTA-01/20260810T221700Z__s4-eval-canonical-s0-r8`；
  summary/status/decision SHA=`6f143040...9085 / 87d33d95...c5e5 / 19e3aba6...db9`；
- edit target f091/c1 的 erase/background/actor/full effect pixels=`27,000/6,663/14,844/28,218`，
  erase/actor mask coverage=`0.999741/0.849298`，outside L1=`0.225349<=0.5`；
- 5 个固定视角、4 个 overlay stack 均在卸载后重新 source-render，`20/20` SHA exact；full target render
  二次 replay SHA=`451ae330...50bff` exact，replay rollback exact；duplicate insert index=`0`；
- wall=`66.181 s`、peak CUDA allocated/reserved=`8,433,577,472/8,527,020,032 bytes`，run bytes=
  `11,744,674`，OOM/kill=`0/0`，无训练/optimizer；S4 专项=`9 passed`、V3.3/V3.2 定向回归=`72 passed`，
  py_compile、bash syntax 与 source snapshot exact；
- r3/r4 为有效 noncanonical（最终 validator 后 snapshot 漂移），r5/r6 与最终方法一致但 builder fail-terminal
  尚未进入 snapshot；r7/r8 才是 canonical。下一步只解锁 S5。

### `WS-V33-S5-SEMANTIC-RENDER-01` canonical closeout

- r1=`failed`：input 与 Harmonizer 已完成，但 SAM2 冻结环境没有 SciPy，共享 semantic module 顶层导入形态学
  依赖；修复为 gate builder 内 lazy import，没有安装包或修改冻结环境；
- r2 首次完整完成；development 选择 G1，heldout f060/c1 contact delta=`+0.422686>+0.25`，final arm
  自动回退 G0；r3 仅消除只读 NumPy view warning；r4 只清理 input-prep EOF 空白，协议/阈值/选择均未改；
- canonical r4=
  `/root/autodl-tmp/runs/worldsim_v33/WS-V33-S5-SEMANTIC-RENDER-01/20260810T220500Z__s5-semantic-gate-canonical-s0-r4`；
  config/input/Harmonizer/SAM2 SHA=`b3848289...8c89 / 939a829e...635c / 1da253d8...3863 / c03fe7c9...8b19`；
  summary/status/decision SHA=`1e0bfb59...0761 / 969bb009...fb97 / 988b6647...6159d`；
- semantic gate residual cap=`12/255`、far weight=`0`；五视图 far changed pixels=`0`、actor interior delta=`0`；
  development boundary/contact delta=`-1.837229/-2.771866`；
- delete production 全部 raw 3D exact copy，SAM2 production mass/fraction delta 5/5=`0/0`；unconstrained
  candidate 在 f091/c1 的 mass/fraction=`+0.126399/+0.133885`，1/5 被标记；
- Harmonizer/SAM2 wall=`30.180697/5.701133 s`、peak NVIDIA sampled=`3,553/2,399 MiB`、peak torch
  reserved=`3,940/2,070 MiB`，run bytes=
  `34,548,858`，OOM/kill=`0/0`；r2–r4 的 30 个 RGB 产物跨三次 run SHA 全 exact，decision SHA exact；
- R3D2 commit/tree/license exact，但作者 exported pretrained model 不存在；状态
  `blocked_pretrained_model_unavailable`，无模型加载/训练；temporal 因非相邻五视图显式 not-evaluated；
- S5 专项=`8 passed`，V3.3/V3.2 定向回归=`80 passed`。S5 task=`done`，G1 method=`rejected`，
  production=`G0_raw_3d`；下一步只解锁 R0。

### `WS-V33-R0-INTEGRATION-01` canonical closeout

- diagnostic `222435/222453/222511` 分别因 S2 empty-list、S3 heldout phase、S4 real-render stage 的 exact
  schema 枚举误写而 failed；`222526` 移除报告层冗余 `schema_version` 假设，正式 instance-field validator 已通过；
  `222549` 修复 release `tools/` 目录创建；旧 terminal 均未覆盖；
- diagnostic r6=`20260810T222610Z__r0-integration-diagnostic-s0-r1` 首次 10/10 gates 通过，并冻结 archive SHA；
  formal config 将 expected SHA 固定后执行 canonical r7=
  `/root/autodl-tmp/runs/worldsim_v33/WS-V33-R0-INTEGRATION-01/20260810T222701Z__r0-integration-canonical-s0-r7`；
- config/summary/status SHA=`4b4a20b9...73a9 / c1903255...a2b2 / 0a1396f4...aa4`；44 inputs
  exact，O1/RoadPatch/A4 schema validator 通过，S4 rollback=`20/20`，S5 production exact-safe=`5/5`；
- selected chain=`D2→O1→B1→A4→posterior-gated delta→S5 G0→persistent storage reference→exact release`；
  4/4 required success criteria，overall=`v33_supported`；
- release=`76 files / 18,432,994 payload bytes`、39 JSON evidence、full checkpoint copy=`0`；archive=
  `13,760,114 bytes / cffaad16...44a7`，双构建 exact；解包 content manifest=`e386c14b...a6c53` exact；
- standalone verifier 的 directory/archive 两种模式均返回 valid；R0 wall=`2.721847 s`、GPU compute max=`0`、
  run bytes=`50,851,476`、OOM/kill=`0/0`；S1–S5 selected wall=`379.552 s`、peak=`20,137 MiB`；
- 对 RoadPatch vs Telea 结论为 `not_directly_ranked`；blocked SOTA 不作质量失败；R0 专项=`6 passed`、
  V3.3/V3.2 定向回归=`86 passed`；当前无下一执行授权，F0 LiDAR-EVS 保持 conditional。

## 1. 状态词

只使用：

```text
pending | running | blocked | done | rejected
```

`done` 表示预注册门禁满足；`blocked` 表示工程、资源或外部依赖阻塞；`rejected` 表示研究门禁失败。

## V3.2 注册表

| Task ID | 状态 | 目标 | 当前证据/门禁 |
|---|---|---|---|
| `WS-V32-S0-ROUTE-AND-SOTA-AUDIT-01` | done | source/license/weight/hardware audit | 11 个 official HEAD exact；SAM2.1 large weight bytes/SHA exact；V3.1 immutable |
| `WS-V32-S1-SEMANTIC-LIFT-01` | done | SAM2 temporal mask + Gaussian semantic posterior | r6 identity-aware finalizer 通过；398 masks、334 accepted、0 heldout leak、6 smoke |
| `WS-V32-S2-BACKGROUND-INPAINT-01` | done | background 3D inpainting | canonical r3；1,896 generated rows；两目标与四路 held-out 全部门禁通过 |
| `WS-V32-S3-ASSET-HARVEST-01` | done | complete dynamic actor asset | canonical r3；1/2-view 皆完成，选定 2-view；4 个真实视角回注渲染与资源门通过 |
| `WS-V32-S4-HARMONIZER-01` | done | visual harmonization | canonical r3；non-temporal 执行完成但语义删除门失败，optional diagnostic；temporal externally blocked |
| `WS-V32-S5-MULTIVIEW-UPPERBOUND-01` | blocked | multi-view editing upper bound | 未授权；许可证门未通过 |
| `WS-V32-R0-INTEGRATION-01` | done | V3.2 integration | canonical r4；8/8 gates，mixed checkpoint、extended semantics、actor override 与 exact chunk package selected |

### `WS-V32-S0-ROUTE-AND-SOTA-AUDIT-01` 完成证据

- branch=`research/worldsim-v3.2-semantic-repair`，baseline=`d91e80e`；
- source truth：`configs/worldsim_v32/s0_sources_v1.yaml`；报告：`docs/WS_V32_S0_SOTA_AUDIT.md`；
- SAM2 checkout=`2b90b9f5ceec907a1c18123530e92e794ad901a4`；SAM2.1 large HF revision=
  `665f8e2ad61cf5f53d65644ff27c8ee525124610`；checkpoint=`898,083,611 bytes`，SHA-256=
  `2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318`；
- 无卡模式 cgroup max=`2,147,483,648 bytes`；未运行 torch GPU、模型推理或训练；
- D2 checkpoint/A3*=R0/P2/P3/V3.1 terminal 未 mutation；S1 为唯一 next authorization。

### `WS-V32-S1-SEMANTIC-LIFT-01` 无卡准备（未启动 GPU 任务）

- 冻结配置：`configs/worldsim_v32/s1_semantic_lift_v1.yaml`；
- 无卡验收事实：`configs/worldsim_v32/s1_no_gpu_preflight_v1.yaml`；独立环境为 torch
  `2.5.1+cu124` / torchvision `0.20.1+cu124`，`pip check` 无破损依赖；
- prompt asset：`/root/autodl-tmp/assets/worldsim_v32/s1_prompt_v1/prompt_manifest.json`，SHA-256
  `f60131687e5b4e814dd41a47e69e4b6d4d83d626e20221c43462332d03d9e69d`；
- split：177 个 train frames / 19 个 heldout frames，manifest 检查为零 heldout 泄漏；
- 静态验收：Python compile、4 个 schema/projection/split 单测、runner `bash -n`、`git diff --check` 均通过；
- 未执行 SAM2 mask、Gaussian lift、original/delete/lateral smoke，故 S1 保持 `pending`。

### `WS-V32-S1-SEMANTIC-LIFT-01` r5 invalidation 与 canonical r6

- r5 的旧 finalizer 只核对字段、SHA 与 rigid core count，没有核对 `dataset_instance_id` 对应的
  `instances_info.id` 是否等于同一 role 的 `instance_token`；
- high-support 旧配置为 dataset ID `5` / token `af663…` / rigid index `5`，实际 ID `5` 的 token 是
  `bf9a…`，`af663…` 的真实 dataset ID 是 `13`；因此 r5 的高支持 SAM mask 与 D2 actor core 不是同一对象；
- 新 validator 同时核对 dataset ID ↔ token ↔ actor registry rigid index；v2 负测 exit=`1`，v3 正测通过；
- v3 config SHA-256=`377cd95999dcc02d15782fce06940952826c410d5f4df13846e5dd4c58304960`，
  prompt v3 SHA-256=`8c43b59175da1598b9720bb71d35d573647651ee4075c44ac7b0e265931f6ccf`；
- r6=`20260810T101739Z__s1-semantic-lift-s0-r6` 已 `done`；actor identity contract=`validated`；
  final summary SHA-256=`482dcd067ee91952536e863cded1e18cffa1003bbd3f1b0caa9a18380e93bb4a`；
- 398 masks=`334 accepted / 64 rejected`，heldout leaks=`0`；SAM2 wall=`81.605s`，peak
  allocated/reserved=`1,908,027,904 / 2,204,106,752 bytes`；
- semantic lift wall=`919.436s`，peak allocated/reserved=`15,723,618,816 / 24,683,479,040 bytes`；
  high-support labels=`1,230,548 / 4,525 / 36,767 / 38,028`，boundary-support labels=
  `1,276,927 / 3,728 / 21,033 / 8,180`（negative/core/semantic/ambiguous）；
- 6 个 original/delete/lateral smoke 均非空；D2 checkpoint before/after SHA-256 均为
  `1a061247a753c0d8c9aa7835a52efa2ab1ddc79141a6168adc18b9748de66e7c`；S1=`done`；
- 以下 r5 数值只保留为失效证据，不再支撑任何 selected/canonical 结论。

### `WS-V32-S2-BACKGROUND-INPAINT-01` canonical r3

- 官方 source 审计：3DGIC commit=`0fdbaed680264c02d6222c573434618eb21a44a1`、license SHA-256=
  `c2297cb5b2dd996979a6031ae7c5e112be310f87595c1dc40340be820e0d67e5`；Inpaint360GS commit=
  `d54c893285c6cb27788e05cce607e7d3cca6388a`、Apache-2.0 license SHA-256=
  `41d805773f2aa0b36c2fb69491f64c3079fe3e0671c9848680645fc9e65d5a10`。当前 StreetGS checkpoint 与两者
  原生 schema 都不相同；正式实现明确称为 3DGIC depth-guided cross-view 原理适配，不冒充上游原生运行；
- r1=`20260810T120554Z__s2-3dgic-adapted-s0-r1` 在生成任何候选 checkpoint 前因 boundary-support
  跨视图像素低于 `32` 而 `rejected`。保留的 train-only exhaustive diagnostic 显示目标 frame `31` 之后三相机
  均无几何重叠，重新冻结 frame `24..29` / CAM_FRONT，不使用 held-out；
- r2=`20260810T121342Z__s2-3dgic-adapted-s0-r2` 完成 3,940 行候选与严格重载，但把 high-support
  未观测 Telea 像素持久化进静态背景，导致四路 held-out 平均 PSNR/SSIM delta=
  `-0.495842 dB / -0.007160`，候选为 `candidate_selected=false`；
- canonical r3=`20260810T121829Z__s2-3dgic-adapted-s0-r3`，config/runner SHA-256=
  `8ad962d84009b44464ca70347fec8c935b012ad727e8e233e03648dc41defe29` /
  `8a3c9dae6c937828ff4193bcfc64bda09e0720ba64632c1bdb5848ad6b3de93c`；基础 projection/adapter 测试
  `6 passed`；
- high-support mask=`15,461` pixels，train-only cross-view observed=`7,189`、multi-support=`1,788`、
  unseen Telea=`8,272`；checkpoint 仅持久化 observed geometry。boundary-support mask=`288`，observed=`46`、
  multi-support=`44`、unseen=`242`，小目标保留完整补全。所有 unseen 只声明 provenance、view consistency 与
  artifact，不声明 GT accuracy；
- append=`1,896` rows，Background=`1,205,164 → 1,207,060`；旧 means exact，候选 strict reload、V3.1
  ancestry 对齐，权威 sidecar 对每个新行记录 `GENERATED_BACKGROUND`、confidence、observed flag、target code 与
  source pixel；
- high/boundary candidate effect=`9,928 / 176` pixels，outside L1=`0.042503 / 0.005122`；candidate 对
  completion-reference mask PSNR=`17.127871 / 21.363783 dB`，均高于 source delete 的
  `15.969350 / 19.123467 dB`；
- 四路 held-out 只在生成后读取；平均 PSNR/SSIM/LPIPS delta=
  `-0.022958 dB / -0.000528 / +0.000301`，通过冻结 `-0.1 dB / -0.005 / +0.01` 门；
- selected checkpoint/summary/provenance SHA-256=
  `3d6e13d47291f5b5949ff3adf5598b6e0cffb930c4cbff2200c6e708d82e6e0f` /
  `a07bbf7a1b160d352fd0d3d08be9e217a3d27648eeffec7841f443b5bc871407` /
  `1baf73b81205f66cfe30a6ea3385cdf960b3d8952648031fb34be26a7ef758cc`；D2 source before/after SHA exact；
  wall=`63.908s`，peak CUDA allocated/reserved=`8,051,344,384 / 8,141,144,064 bytes`，NVIDIA sampled=
  `8,125 MiB`，cgroup peak=`39,369,183,232 bytes`，OOM=`0`。S2=`done`，selected=canonical r3。

### `WS-V32-S3-ASSET-HARVEST-01` canonical r3

- r2=`20260810T103527Z__s3-asset-harvest-s0-r2` 因在 CUDA context 显式初始化前调用 PyTorch 2.10
  峰值显存计数器而 `rejected`；实际模型推理未启动、GPU peak=`0 MiB`、无部分输出；
- canonical r3=`20260810T112505Z__s3-asset-harvest-s0-r3`，官方 source commit=
  `767b2439ce47a8b2513038ae0fb2073026f89ee8`，config SHA-256=
  `72a5901348265369e14636090ca02b1c61b10b454bfa76e869188036aefc1cdb`；
- actor identity=dataset ID `13` / token `af663…` / rigid index `5`；CAM_FRONT_LEFT frame `91/51` 均为
  直接 prompt、非 heldout，SAM 被 D2 actor-effect 完整覆盖；input manifest SHA-256=
  `35555c431c44754b0d6fc2a019d7ef9ccf4d47cd7ea2cc8733189ebe2e6cf2dd`；
- Asset Harvester 1/2-view 各生成 `16` 个新视角和非空 `gaussians.ply`；inference wall=
  `113.981s`，peak CUDA allocated/reserved=`15,522,251,776 / 20,772,290,560 bytes`，NVIDIA sampled=
  `20,137 MiB`，cgroup peak=`48,426,651,648 bytes`，OOM=`0`；
- 导入后 1-view/2-view 分别为 `102,303 / 99,045` Gaussians，exact reload，3σ bounds 与冻结
  LWH=`4.51 / 1.76 / 1.65 m` 的最大误差为 `2.22e-16 m`；
- 两资产各在 frame `91/51` 完成 original/lateral/delete，四个 formal render 的 D2 checkpoint
  before/after SHA exact；
- 1-view mean IoU/F1/PSNR/LPIPS=`0.723918 / 0.499815 / 16.078961 dB / 0.104843`；
  2-view=`0.733945 / 0.459813 / 16.671399 dB / 0.094894`；综合指标与前/侧/后目视 QA，
  selected=`high_support_2view`，同时保留 boundary F1 较低的限制；
- summary/evaluation SHA-256=
  `8dc4fc930229fbb17343b0bbcf9ccda632ac54b2e5301d4ca6448bda0d99c2d1` /
  `224eda1a6480941592cef843a685b7c73a70833cbf0a81e0618385466c180a3e`；S3=`done`，生成背面不声明 GT correctness。

### `WS-V32-S4-HARMONIZER-01` canonical r3

- 官方 source commit=`dd5799e50855c5bcb1f6ef52a77b5b644b4798c0`；non-temporal JIT model bytes/SHA-256=
  `1,448,843,112 / ece8e2daa914e8c2a027a2da94e0eb2064491d5b3fd8514009fae9a442e06e90`；code 与 model license
  分别为 Apache-2.0 / NVIDIA Open Model License；
- temporal checkpoint 的官方链还要求 gated Cosmos 0.6B base。固定 revision=`dd55b6858b22ad569976bff207880b8fea839da7`，
  当前无 HF 授权，403 证据保留且未绕过；因此不做 temporal consistency claim；
- 当前 PyTorch 2.10+cu128 环境通过公式等价 RMSNorm 回退和两个 shape scalar device 修复运行导出图；
  BF16 operator 对独立参考式 exact、max abs error=`0`，适配器测试 `4 passed`；
- r1=`20260810T131510Z__s4-harmonizer-nontemporal-s0-r1` 在模型前向前因 device 初始化顺序错误
  `rejected`；r2 工程门通过但视觉 QA 发现 G1 删除区被生成式模型重新解释成车辆，失去候选资格；
- canonical r3=`20260810T131909Z__s4-harmonizer-nontemporal-s0-r3` 覆盖 5 张同 camera 输入：
  G0 original ×2、G1 semantic remove+inpaint ×1、G2 selected Asset Harvester lateral ×2；
  800×450→1024×576→800×450，direct bilinear、无 crop/pad；
- G0/G1/G2 mean outside-mask L1=`3.543039 / 3.831952 / 3.640835` uint8；G1 inside L1=
  `14.217278`、inside changed fraction >8=`0.541750`，同时失败冻结 `<=12.0 / <=0.40` 门；
  `non_temporal_candidate_selected=false`，`final_disposition=optional_diagnostic`，不得默认串入删除输出；
- 5/5 输出均有 `HARMONIZED_2D` provenance，无 3D 写回；D2、S2 selected checkpoint、S3 selected actor
  before/after SHA exact。wall=`35.048s`，常态 median inference=`0.3386s`，peak NVIDIA=`4,077 MiB`，
  peak CUDA reserved=`4,131,389,440 bytes`，OOM=`0`；
- summary/status/grid SHA-256=`4543b5fa2543f6f42aa65f0dbc17f11899de1cc7ebad4aed653200e881f1ba39` /
  `42465759974c60f0fa5407969b12ccf8aeb5952ed7c36904b378a86163b78e51` /
  `086b08b7ab57de7a27d28dda28a84109579ff8cfae15216f88533085e19f3cbf`；S4 task=`done`。

### `WS-V32-R0-INTEGRATION-01` canonical r4

- r1=`20260810T134019Z__r0-final-integration-s0-r1` 因相对 config path 无法冻结 source snapshot 而
  `rejected`，未物化资产、未启动推理；r2=`20260810T134128Z__...` 已物化资产，但在首个 forward 发现
  DriveStudio 必须显式 `set_eval()`，故 `rejected`；r3=`20260810T134421Z__...` 的混合精度与 exact chunk
  均通过，但误把冻结 `MAE<=1` 写成 `max_error<=1`，保持 `rejected`，不事后改写；
- canonical r4=`20260810T134658Z__r0-final-integration-s0-r1`，config/runner/integration/test SHA-256=
  `7011d99f70fc59835569c43bd7e750a5e1981ea67843ef08873bfe4707deb624` /
  `deb1a82f8d60eb659acf1237482ffff26a6d47d615c3eeb50df75d18f0c3c97c` /
  `5ca86b4170d6990bd8b54e15033c090ccc19675b6d8d5340b1bd22ec0eded1f1` /
  `b9cdaab156c6bf3bec19111a22f43764d64926d23bc96e9633a0c073440700d2`；
- S2 generated-background provenance 1,896/1,896 行连续 exact；cross-view true/false=`1,835/61`，target
  code 0/1=`1,824/72`，confidence 与 observed contract exact；S1 high/boundary sidecar 扩展后 SHA-256=
  `7caae12fdfb92f15ae02f5f7fc6f5c8111236f18632516a128b09960b6d79b26` /
  `74dd3679b58423c6e752cd3441a347d8f3f3f1add1e5ce748e75150eb510185b`，旧 prefix/rigid suffix exact；
- P2-style mixed checkpoint bytes/SHA-256=`432,347,490 /
  6d4e4c489f53bf4e7de3f5c405ec37dc63d3f79155aad5237fe175ce0fcd7e5d`，较 S2 FP32 source 减少
  `146,922,064 bytes / 25.363333%`；十个转换 tensor 的 FP16 值、未转换 tensor、schema 与重载全部 exact；
- registry bytes/SHA-256=`3,589 / 6633af150baa4b5adda143b2037091e7647f85966490de5d660fa74968ab6c57`；
  high-support rigid index `5` 使用 S3 99,045-Gaussian `GENERATED_ACTOR`，boundary 和其余 actor 用 V3.1
  fallback；S4 excluded diagnostic、S5 blocked 均为显式字段；
- P3-style package 为 `133 static + 24 actor + skeleton`，manifest bytes/SHA-256=`141,427 /
  af7b402e0b171b11f8c22e4123002f4f844db746ea72f53b77c3de878bf0947d`，payload=`444,282,102 bytes`；
  Background/RigidNodes=`1,207,060/104,704` 行均 covered once，missing/duplicated=`0/0`，85 tensor paths、
  recursive schema 与 non-tensor signature exact；
- fixed views `(31,0)/(51,1)/(91,1)` 的 source→mixed PSNR=
  `68.299304/67.239933/68.432160 dB`，MAE=`0.009614/0.012271/0.009330` uint8；mixed 与 reassembled
  三个 RGB SHA exact，FP16 persistent/FP32 renderer adapter exact；
- wall=`103.099s`，peak NVIDIA=`8,362 MiB`，CUDA allocated/reserved=`7,729.707/8,020 MiB`，cgroup=
  `48,169,205,760 bytes`，run=`948,244,397 bytes`，OOM/kill=`0/0`；input immutability exact，训练/optimizer
  steps=`0`；V3.2 定向测试=`36 passed`；
- summary/manifest/status/report SHA-256=
  `40624cbc79a004e9e07e57b00cebc535b900297a10f0d070fb4e9305a5f7937a` /
  `358d9fc7fde6a535c2ffb0bb2ff34cf1f9df3c151066f3051e24859a5d73a27e` /
  `d31a4f8e62f31dbbf6bbf2520243f5061c68e6682ea5011ef8c64a8dbb541617` /
  `b5397f555270a901013f0a6ce82ba20c8a868d9e22039ba1d9cc2066adf20913`；R0=`done`。

### `WS-V32-S1-SEMANTIC-LIFT-01` r5 历史运行（identity-invalid）

- run=`20260810T093248Z__s1-semantic-lift-s0-r5`，config=
  `configs/worldsim_v32/s1_semantic_lift_v2.yaml`，config SHA-256=
  `ecb6c2bc6f68376c9cd81e3e2a362a30506edfba5772226ad27125f0dcbad706`；
- prompt v2 SHA-256=`771817828acb689e8cab19c4f4c368d8ead24c0d1c154bd1d8bcc283a9b6c071`，使用相对
  video block、逐帧投影 box、双向传播、源图到 800×450 的 exact 坐标映射和冻结 fail-closed QC；
- 263 masks=`212 accepted / 51 rejected`，heldout leaks=`0`；SAM2 wall=`58.337s`，peak
  allocated/reserved=`1,895,410,176 / 2,197,815,296 bytes`；
- semantic lift 263/263 views，wall=`770.733s`，peak allocated/reserved=
  `15,726,013,440 / 24,536,678,400 bytes`，无 CUDA/cgroup OOM；
- high-support sidecar SHA-256=`85ef3c1473b19c6fd5c46ab92d27f78e873e68066dde21fb25beab64ec19e103`，
  labels=`1,295,141 / 4,525 / 3,927 / 6,275`（negative/core/semantic/ambiguous）；
- boundary-support sidecar SHA-256=`983cffe338caa602b8d347e347b95950ec8d2f5d5568ac82dda0a180b9dfca81`，
  labels=`1,276,895 / 3,728 / 21,043 / 8,202`；
- 6 个 original/delete/lateral smoke 完整且每个 actor 的三个 SHA 互异；D2 checkpoint before/after SHA exact；
- final summary SHA-256=`84fbf086dfe8a171b7b4025aceff078bf84b8674f2feb902cd43fe694d112408`；
  旧 finalizer 当时返回 `done`，但 identity 审计后该 run 已失效，不得作为 S1 selected 或 S3 输入。

## 2. V3.1 冻结注册表

| Task ID | 状态 | 目标 | 完成门禁 |
|---|---|---|---|
| `WS-V3-P0-ROUTE-01` | done | 单一 V3 权威计划与 V2 事实冻结 | `076ebdc`；文档一致，链接与 Git diff 校验通过 |
| `WS-V3-A0-NATIVE-BASELINE-01` | done | 三场景原生 StreetGS 基线 | `20260805T175000Z__a0-three-scene-finalize-s0-r2`；3/3 完整矩阵 |
| `WS-V3-F0-FEEDFORWARD-AUDIT-01` | done | Instant NuRec 官方本地能力审计 | canonical audit done；4/11 prerequisites；inference not-run；F1 未解锁 |
| `WS-V3-A1-CALIBRATION-01` | done_off | 成像、位姿和 LiDAR 初始化消融 | 10/10 逻辑项、8/8 唯一训练；C*=C0；finalizer done |
| `WS-V3-A2-ACTOR-DENSIFY-01` | done | actor-aware densification/pruning | D1/D2 formal 完成；A2*=D2 boundary-priority，D1 fallback；D3/D4 未启动 |
| `WS-V3-A3-LOCAL-REFINE-01` | done | 编辑区域局部 Gaussian 精修 | R1 rejected；A3*=R0/D2 exact alias；formal、R2–R4 未授权 |
| `WS-V3-A4-DEPLOYMENT-01` | done | pruning/precision/chunk/LOD 与资产注册 | P0/P5/P1/P2/P3 complete；P1 rejected；P2 mixed checkpoint + P3 exact package selected |
| `WS-V3-R0-INTEGRATION-01` | done | 完整 A0–A4/F0 结论与复现包 | canonical done；63 inputs/23 decisions/12 deliverables/26 manifest files/no-launch exact |

### `WS-V3-A0-NATIVE-BASELINE-01` 完成证据

- fix commit：`436cfc1`；DriveStudio upstream：`e59bda4`；compatibility patch SHA-256：
  `54e7584b6d74431e58f626dfaadd69812d4058d54f82c7941e75aa11f5f94619`；
- 完整 A0 定向测试：`16 passed`；
- canonical smoke：
  `/root/autodl-tmp/runs/worldsim_v3/WS-V3-A0-NATIVE-BASELINE-01/20260805T161656Z__scene0255-catfix-s0-r2`；
- terminal=`done`；torch=`2.1.2+cu118`；原生 mixed-empty cat 复现
  `invalid configuration argument`，patched output=`[59,3] / 177 numel / exact point-color pairing`；
- 真实 1-step DriveStudio 路径完成数据集、LiDAR 实例初始化、一次优化和 checkpoint；controller
  `72.1 s`，peak GPU sample `8,388 MiB`，peak cgroup `5,971,820,544` bytes，无 OOM；
- smoke checkpoint `320,832,362` bytes 只证明执行路径，不注册为 A0 最终模型。

正式三场景矩阵：

| scene | checkpoint / step | global PSNR / SSIM / LPIPS | high actor PSNR / SSIM / LPIPS | high boundary PSNR / SSIM | bg / rigid GS | train s / peak MiB |
|---|---|---:|---:|---:|---:|---:|
| 0230 | `24a39f…e49` / 30k | 24.934 / .740 / .169 | 21.728 / .596 / .121 | 20.165 / .603 | 1,152,614 / 167,299 | 3014.5 / 23,799 |
| 0242 | `16179d…fda` / 30k | 29.107 / .906 / .113 | 19.788 / .665 / .153 | 23.277 / .795 | 843,756 / 86,255 | 2006.2 / 12,783 |
| 0255 | `f8c81c…ef9` / 30k | 25.230 / .743 / .192 | 23.531 / .665 / .058 | 22.991 / .656 | 1,510,936 / 40,447 | 2739.4 / 24,057 |

- scene-0255 正式训练：`20260805T162355Z__scene0255-native30k-s0-r1`；0230/0242 通过 config normalized
  SHA、checkpoint bytes/SHA/step 和同实现合同注册复用；
- actor evaluator 提交 `01cd303`；counterfactual mask 明示不是真值分割，并记录 coverage。actor runs：
  `20260805T173900Z__scene0230-actor-metrics-s0-r1`、`20260805T174100Z__scene0242-actor-metrics-s0-r1`、
  `20260805T174300Z__scene0255-actor-metrics-s0-r1`；peak GPU `8,455 / 7,905 / 8,685 MiB`；
- 0242 boundary role 按注册表保留 `ABSTAIN`；其余 boundary actor 区域/边界带均有正式指标；
- finalizer r1 `20260805T174700Z__a0-three-scene-finalize-s0-r1` 因 native/reuse training resource 字段差异
  `blocked`；`00ba4e8` 归一化后 r2 `20260805T175000Z__a0-three-scene-finalize-s0-r2`=`done`，产出
  `a0_matrix.json/csv` 与 `a0_report.md`；
- A0 只支持固定三场景的描述性结论。跨场景 GS 数与质量不可作因果归因；A1/A2 必须做同场景受控消融。

工作树准备脚本首次创建旧候选 worktree 后，因 `git status --short` 的输出已被 `.strip()` 去除前导空格，
verification literal 写成带前导空格而失败；修正为 `M datasets/driving_dataset.py` 后 verify-only 通过。canonical
patch 为 r2 worktree 和上述 SHA；旧候选只解释首次 smoke，不进入 formal training。

### `WS-V3-A1-CALIBRATION-01` 当前证据

开发场景 `scene-0230` 已完成 C0–C3 30k formal；初始化 provenance SHA 均为
`8951543c33f72f439068237f1a552fae660895f8906afbf4651f5f580981b898`。固定 step 结果：

| variant | global PSNR / SSIM / LPIPS | boundary actor PSNR / SSIM / LPIPS | high actor PSNR / SSIM / LPIPS | total GS | train min |
|---|---:|---:|---:|---:|---:|
| C0-off | 27.746 / .851 / .176 | 27.756 / .892 / .069 | 25.358 / .844 / .094 | 1,360,649 | 52.05 |
| C1-native | 24.979 / .743 / .169 | 22.549 / .700 / .103 | 21.696 / .602 / .120 | 1,316,421 | 53.69 |
| C2-factorized-isp | 25.011 / .743 / .168 | 22.583 / .705 / .104 | 21.779 / .608 / .117 | 1,322,979 | 52.26 |
| C3-bounded-pose | 28.109 / .862 / .167 | 28.169 / .897 / .066 | 25.137 / .846 / .094 | 1,363,040 | 56.14 |

A1-E0 实现提交为 `20c4276`，相机映射修复为 `d85ef27`。冻结配置
`configs/worldsim_v3/a1_endpoints_v1.yaml` 的 SHA-256 为
`60c211625860c25edf92842b88bdb040ea8c180b12fe0fa78f2fc1c342bc4051`。相机对只使用 DriveStudio 权威映射
`0=FRONT / 1=FRONT_LEFT / 2=FRONT_RIGHT` 下的相邻对，支持双向投影、静态/可见/遮挡/深度边缘 mask、
near/far、coverage 与 `ABSTAIN`。

有效正式回填：

| variant / run | E1 valid/candidate/coverage | E1 median/P90 ↓ | E2 high mean/P90/coverage ↓ | E2 boundary mean/P90/coverage ↓ |
|---|---:|---:|---:|---:|
| C0 `20260806T141409Z__scene0230-c0-a1-e0-formal-full-camera-map-fix-s0-r2` | 28,744/266,631/10.780% | .05951/.14719 | .004813/.010895/26.316% | .003547/.006353/35.294% |
| C1 `20260806T141623Z__scene0230-c1-a1-e0-formal-full-camera-map-fix-s0-r1` | 29,151/274,658/10.614% | .06289/.15623 | .004751/.010895/28.070% | .004450/.007626/35.294% |
| C2 `20260806T154541Z__scene0230-c2-a1-e0-formal-full-s0-r1` | 31,299/275,877/11.345% | .06544/.16160 | .004844/.011734/28.070% | .003346/.005447/35.294% |
| C3 `20260806T164852Z__scene0230-c3-a1-e0-formal-full-s0-r1` | 29,846/268,826/11.102% | .06309/.15448 | .004930/.011734/26.316% | .003592/.006537/35.294% |

C2 只改善 boundary role E2，high role E2 与 actor/boundary LPIPS 退化；C3 全图、boundary actor 与位姿稳定性
最好，但 E1 和两个 E2 role 均未严格优于 C0。四次有效评估 checkpoint SHA 前后相同。QA panel 只确认投影落在
真实相邻视野和 actor 支持边界，不能替代人工质量裁决。

首次 formal `20260806T140703Z__scene0230-c0-a1-e0-formal-full-s0-r1` 继承了错误相机 ID 标签，实际把
非相邻画面当成预注册相机对，已保留为 `rejected / INVALID_CAMERA_ID_LABEL_MAPPING`；修复后 run 是唯一有效证据。

最小 LiDAR provenance 实现提交为 `14bc3c2`，冻结配置 SHA-256
`f2fd1712cf4ddd75c1c4d1da4a426dcf7e1340a5fd943066401ba881f51c5639`。正式 run
`20260806T143644Z__scene0230-a1-lidar-provenance-formal-full-witness-s0-r1`=`done`：196 个 LiDAR/pose block、
6,804,832 raw points、24 actor/75,002 actor points；记录的 LiDAR 与 actor tensors 全部 exact match，RigidNodes
初始计数 75,002 exact。held-out sparse depth 172,844/172,844，绝对 median/P90=`7.679/35.958 m`，相对
median/P90=`.6649/.9077`，checkpoint SHA 未变。

背景随机 near/far 点的 CUDA visibility filter 不提供跨初始化 exact replay：源背景初始计数 946,484，三次
replay 分别为 946,597、946,309、946,291。首次 strict smoke
`20260806T142900Z__scene0230-a1-lidar-provenance-smoke1-s0-r1` 因 exact SHA 门禁 `blocked`；协议在查看
正式 depth 结果前冻结为“LiDAR/actor tensor exact 是 gate，随机背景 exact 仅 report”，成功 smoke 和 formal 的
初始 depth 都明确标为 reconstructed witness。没有使用事后计数容差。逐 Gaussian ancestry 留到 A2 instrumentation。

A1-D0 配置 SHA-256 为 `a445078d3bea89a78a0c9e6544a94a2be4c9c2e71f45aec4a9d8878b4c6593c1`；
`20260806T170219Z__scene0230-a1-diagnostics-c0-c3-formal-s0-r1`=`done`。输入速度层为
near-static/low/normal=`2/18/176` 帧；near-static 只有 2 帧，不承担统计结论。C1 位姿修正 translation
median/P90=`7.256/12.215 mm`、rotation=`0.1660/0.35465°`；C3 为 `1.703/2.338 mm`、
`0.02553/0.03337°`。这些是学习修正幅值，不是独立 pose accuracy。

选择协议提交 `60ef079`；配置 `configs/worldsim_v3/a1_dev_selection_v1.yaml` SHA-256 为
`a45699ebf696c875a18832f8db920a6106837a1e4f235dcd9036eff48dfbc609`。协议如实披露结果访问，且不引入
数值容差。正式 run `20260806T171417Z__scene0230-a1-dev-selection-formal-s0-r1`=`done`，输出
`C*=C0-off / done_off`。若 C* 为 C0/C1，确认矩阵保留三个逻辑项但用 source run/checkpoint exact alias。
开发场景冻结时进度为 `4/10` 逻辑项、`4/8` 唯一训练；后续确认矩阵结果如下。

确认配置提交为 `198a681`，SHA-256 为
`63a3cc607ccfddbb714cc81d0570da356263c01c5a68880345953023d2d6a8cd`。正式结果：

| scene / variant | training run | endpoint run | global PSNR / LPIPS | E1 median / P90 | E2 high mean / P90 |
|---|---|---|---:|---:|---:|
| 0242 C0 | `20260806T172514Z__scene0242-c0-confirm-formal30k-s0-r1` | `20260806T181834Z__scene0242-c0-a1-e0-confirm-formal-full-s0-r1` | 30.064 / .1108 | .03147 / .08826 | .008264 / .020697 |
| 0242 C1 | `20260806T181957Z__scene0242-c1-confirm-formal30k-s0-r1` | `20260806T191202Z__scene0242-c1-a1-e0-confirm-formal-full-s0-r1` | 29.161 / .1122 | .03333 / .08971 | .008660 / .021708 |
| 0255 C0 | `20260806T191340Z__scene0255-c0-confirm-formal30k-s0-r1` | `20260806T200907Z__scene0255-c0-a1-e0-confirm-formal-full-s0-r1` | 27.255 / .2086 | .04348 / .14248 | .004772 / .009805 |
| 0255 C1 | `20260806T201041Z__scene0255-c1-confirm-formal30k-s0-r1` | `20260806T210645Z__scene0255-c1-a1-e0-confirm-formal-full-s0-r1` | 25.240 / .1921 | .04277 / .13626 | .003715 / .007704 |

0242 的 boundary role 继续 `ABSTAIN`。0255 C1 虽降低 E1/E2 error，但 high E2 coverage 从 `23.529%` 降至
`21.569%`，boundary/high actor LPIPS 也全部退化，故不通过冻结合同。两个 C* alias run 为
`20260806T211000Z__scene0242-cstar-c0-exact-alias-s0-r1`、
`20260806T211100Z__scene0255-cstar-c0-exact-alias-s0-r1`，不含新训练/评测。finalizer
`20260806T211248Z__a1-three-scene-finalize-s0-r1`=`done`，正式终态 `done_off`。A1 完成 `10/10` 逻辑项、
`8/8` 唯一训练；原始端点方向必须报告为 scene-dependent。

### `WS-V3-A2-ACTOR-DENSIFY-01` I0 完成证据

- canonical r3 项目基线 `70cf2b2` + run 内 source snapshot；当前实现提交 `271d876`；
  DriveStudio upstream `e59bda4`，patched worktree
  `/root/autodl-tmp/third_party/drivestudio-worldsim-v3-a2-r5`；
- 配置 `configs/worldsim_v3/a2_instrumentation_v1.yaml` SHA-256：
  `bac1ec5b3642470a999e7f0cf8ddc9cf5b4d9a1445029c43ae92601929f4bfce`；
- instrumentation patch SHA-256：`87c084f77ed5d6395acce95abb992ca86004bdc47b68154878bf462a0fb345b0`；
- canonical run：
  `/root/autodl-tmp/runs/worldsim_v3/WS-V3-A2-ACTOR-DENSIFY-01/20260809T071500Z__a2-i0-ancestry-formal-s0-r3`；
  terminal=`done`，split=`synthetic RigidNodes deterministic refinement contract`；
- module-off/on 原生 tensor 逐位一致；off 无 ancestry checkpoint key，on 有 key 且 checkpoint round-trip；
- 8 个初始 Gaussian 经 1 split、1 clone、1 prune 后为 10 live / 11 allocated，来源计数
  LiDAR/split/clone=`7/2/1`；parent、lineage、actor ID 与 prune 索引一致；
- r3 duration=`10.10 s`，peak cgroup=`1,377,320,960 bytes`，GPU sample=`0 MiB`；
- patched worktree verify 与当前 working-tree diff 门禁通过；WorldSim 定向测试 `66 passed`。

边界：这是 I0 工程 instrumentation 证据，不是 scene-0230 真实训练或质量结论。boundary/photometric/depth/normal
当前只冻结 update API，normal 无可靠输入时保持 schema-only，background nearest-LiDAR 保持 deferred。D1 必须只增加
actor/background threshold 和 per-actor min/max quota，先做 D0/D1 配对短步 smoke，不得合并 D2–D4。

### `WS-V3-A2-ACTOR-DENSIFY-01` D1 quota-only paired smoke

- 实现提交：`c9b2422`；配置 SHA-256=`6895370625080ccab327e731264e9ebb0f980499b8fec87d02d9efb2e56b14af`；
- DriveStudio worktree=`/root/autodl-tmp/third_party/drivestudio-worldsim-v3-a2-d1-r5`；quota patch SHA-256=
  `c232af2c5fa532016943f399830c85ebba612078871b7c1a296bda816ae7bb1b`；
- canonical run：
  `/root/autodl-tmp/runs/worldsim_v3/WS-V3-A2-ACTOR-DENSIFY-01/20260809T081330Z__a2-d1-paired-smoke1k-s0-r4`，
  terminal=`done`，summary SHA-256=`ec219bb567799d4d84252e86bd4194620f6b5563d6032c43067ff8e155d3b8bd`；
- scene-0230 / seed 0 / 1000 step；D0 后 D1 顺序执行，配置预算匹配，initialization provenance 完全相同；
- D1 policy：actor grad threshold=`0.00025`，Background native threshold=`0.0005`；24 actor 初始/min/max 总量=
  `75,002 / 37,504 / 180,013`；
- D0 final：Background/Rigid=`1,144,022 / 125,915`；D1 final：`1,144,598 / 152,830`；
- D1 5 次 quota event，accepted children=`93,057`，rejected parents=`30,171`，24/24 actor 未超过上限；
- D0/D1 原生 tensor finite；module-off bitwise=true；D1 quota/ancestry checkpoint round-trip=true；
- D0/D1 duration=`110.91 / 110.97 s`，peak GPU=`12,807 / 12,795 MiB`，peak cgroup=
  `5,392,334,848 / 5,661,368,320 bytes`，无 OOM；
- r2 terminal=`blocked / MANUAL_RESTART_IN_TMUX`；r3 terminal=`blocked / GPU is not idle`，遵循 `PIVOT-F22`
  精确回收独立 session 子进程后新建 r4，未覆盖旧 terminal。

裁决：工程 smoke=`done`，只解锁 D1 formal 协议冻结。当前没有冻结 held-out actor/boundary 质量证据，D1 比 D0
多 `26,915` 个 Rigid Gaussian，不能登记为质量改进或直接进入 D2。

### `WS-V3-A2-ACTOR-DENSIFY-01` D1 formal 协议冻结

- 提交=`387dd50`；配置 `configs/worldsim_v3/a2_d1_formal_v1.yaml` SHA-256=
  `ad77db41d9d8c5172804a20b38a2dd92173c3639398d8abc24dc6f4799e8f8e7`；
- frozen pair=scene-0230 / seed 0 / D0→D1 / 每臂 30k；5k candidate grid 只读，不改变训练；
- matched-Gaussian-budget 目标为 D0 最终 RigidNodes 数；D1 选绝对差最小、并列更早的 checkpoint，2% 之外
  `ABSTAIN_BUDGET_NOT_MATCHED`；不做事后 pruning、重训或 quota retune；
- held-out 报告 global、high/boundary actor region/boundary band、两 actor 反事实 mask 并集之外的 non-target、
  24 actor GS、Background/total GS、训练时间、peak VRAM/cgroup；matched 中间 step 的峰值只报完整 30k 上界；
- `80 passed`；read-only preflight=`done`：GPU=`0 MiB`、free disk=`58.39 GiB`、cgroup memory.max=`90 GiB`，
  canonical r4 summary SHA 与 compatibility/instrumentation/quota patch SHA 均匹配；
- 协议冻结提交时 formal 尚未启动；本条本身不构成 fixed-step 或 matched-budget 质量结果。

### `WS-V3-A2-ACTOR-DENSIFY-01` D1 formal 正式结果

- run=`20260809T085400Z__a2-d1-paired-formal30k-s0-r1`；source commit=`f32f96b`；tmux=`ws_a2_d1_f1`；
- terminal=`done`；summary SHA-256=`e3b194c2ed0563385df70ca2043dbc791bedb21068d28dc9d75fb59984c166ac`；
  manifest SHA-256=`f10e6e654ab27289ccb1c995ebbe1ffde913009dbfb3eae0ab4c6414de18a560`；
- materialized configs normalized match=true；D0/D1 初始化 provenance SHA=`8951543c...b898`，Background/RigidNodes
  均为 `946,484 / 75,002`；6×2 checkpoint 网格有效且同源，评测前后 checkpoint SHA 不变；
- D1 quota 158 次 event；24/24 actor 不超过冻结上限；D0/D1 native tensor finite；无 OOM。

Fixed-step（30k）：

| arm | global PSNR / SSIM / LPIPS | high actor PSNR / SSIM / LPIPS | boundary actor PSNR / SSIM / LPIPS | non-target PSNR / SSIM / LPIPS | bg / rigid / total GS | train s / peak MiB |
|---|---:|---:|---:|---:|---:|---:|
| D0 native | 27.7481 / .851207 / .176319 | 24.9965 / .838813 / .094204 | 27.1783 / .882177 / .068895 | 26.8707 / .848887 / .057715 | 1,182,619 / 177,628 / 1,360,247 | 2883.08 / 23,867 |
| D1 quota | 27.7700 / .850915 / .177704 | 25.1238 / .840230 / .096602 | 28.4658 / .899698 / .063419 | 26.8901 / .848493 / .058316 | 1,201,057 / 105,412 / 1,306,469 | 2099.33 / 23,989 |

- quality 轴 D1/D0 更优=`12/7`；quality-cost 轴=`15/9`；两者均 `tradeoff_non_dominated`；
- peak cgroup D0/D1=`10,350,350,336 / 16,012,115,968 bytes`。

Matched-RigidNodes-budget：

| arm | checkpoint | global PSNR / SSIM / LPIPS | high actor PSNR / SSIM / LPIPS | boundary actor PSNR / SSIM / LPIPS | non-target PSNR / SSIM / LPIPS | bg / rigid / total GS |
|---|---:|---:|---:|---:|---:|---:|
| D0 native | 30k exact alias | 27.7481 / .851207 / .176319 | 24.9965 / .838813 / .094204 | 27.1783 / .882177 / .068895 | 26.8707 / .848887 / .057715 | 1,182,619 / 177,628 / 1,360,247 |
| D1 quota | 15k | 25.9290 / .825381 / .217941 | 25.7705 / .829707 / .109637 | 29.2937 / .902828 / .061463 | 24.3371 / .822724 / .090772 | 2,432,701 / 176,741 / 2,609,442 |

- D1 15k 是 5k 网格绝对差最小候选：距 D0 Rigid target `887 / 0.499%`，通过 2% 门；checkpoint SHA-256=
  `b864e5ff772777108fcf2214c0548fd4fdc243c360c79890018d7e0d213a9f58`，elapsed=`1127.66 s`；
- quality 轴 D1/D0 更优=`9/10`；quality-cost 轴=`11/13`；两者均 `tradeoff_non_dominated`；
- D1 在 boundary-support actor 与其 boundary band 多数指标改善，但 global、non-target 与部分 high-support 指标退化；
  RigidNodes 匹配不等于 total GS 匹配，D1 total GS 多 `1,249,195`；
- 裁决：`d2_unlocked=true`，只解锁 D2 boundary/residual ordering + boundary scale cap 协议冻结。证据仅限
  scene-0230 / seed 0；不宣称 D1 全面优于 D0，也不以更多 Gaussian 冒充改进。

### `WS-V3-A2-ACTOR-DENSIFY-01` D2 协议冻结

- 配置=`configs/worldsim_v3/a2_d2_protocol_v1.yaml`；SHA-256=
  `acceb7f4ce0f8dc3745de2fcaca51659891cfd82e4175f5a0e5765d77a01e567`；
- D1 formal summary SHA=`e3b194c2...66ac` 与 closeout commit=`f380dd2` 为冻结前置；
- attribution：dynamic mask 3px morphological boundary band + detached RGB channel-mean L1 residual；用 gsplat
  projected center 做 nearest pixel sampling，按 Gaussian 记录 running mean/count；
- ordering：boundary observed/mean、residual observed/mean、screen-grad 全部降序，Gaussian index 升序作稳定 tie-break；
- cap：复用 native `densify_size_thresh × scene_scale`；pre-cap scale 决定 geometry，随后三轴同比缩放并清零 cap 行
  Adam moments；不新增 RNG；
- D1 eligibility/quota/native cull/Background 精确继承；depth/normal、LiDAR/visibility/provenance pruning 与
  Background intervention 禁止；
- 工程提交=`1065264762569c9832219936ddae6f063d6eaf07`；canonical worktree=
  `/root/autodl-tmp/third_party/drivestudio-worldsim-v3-a2-d2-r8`；D2 patch SHA-256=
  `80fef55195906808d74394af0b997cfccbdb88fd7cb356b45240473e55f357cc`；四层 patch replay/reverse-check 通过；
- D1/D2 materializer normalized-match、真实 `RigidNodes` synthetic integration 与联合回归=`29 passed`；boundary/residual
  各 6 次观测、1 次排序/refinement、6 个 cap、配额上限、optimizer moments、checkpoint round-trip 和 module-off
  native state/RNG bitwise 全部通过；
- paired smoke r1 见下一节；工程门禁本身不能登记 D2 方法结果。

### `WS-V3-A2-ACTOR-DENSIFY-01` D2 配对 smoke

- run=`20260809T111304Z__a2-d2-paired-smoke1k-s0-r1`，terminal=`done`；summary SHA=
  `749c7d15c27cc0798c267aa8af12857f3bea52a52ea9d00f7617a3b3edda3136`，manifest SHA=
  `5cb7879d898839b88a46c8ec7ec34141f3402245490416d589938658f33b4c8d`，source=`c594e0c`；
- normalized configs、initialization provenance 与 frozen initial quota 全匹配；D1/D2 step 1000 totals 分别为
  `Background 1,141,192 / Rigid 152,733` 与 `Background 1,144,988 / Rigid 152,807`；
- D2 记录 `1001` 个 observation event，boundary/residual 各 `10,846,748` 个观测，5 个 ordering/refinement event，
  365 个 cap；cap/quota/finite/checkpoint round-trip 均通过；
- D1/D2 duration=`142.17/141.99 s`，torch peak GPU=`9,615/9,620 MiB`，cgroup peak=
  `16,473,858,048/16,667,971,584 bytes`，无 OOM；
- 裁决=`d2_formal_unlocked=true`；只解锁 formal 协议，不把 1k 规模/训练指标登记为质量结论。

### `WS-V3-A2-ACTOR-DENSIFY-01` D2 formal 协议

- config=`configs/worldsim_v3/a2_d2_formal_v1.yaml`，SHA-256=
  `b66cf795c55dfe65315ecf49c09951482d8d6809ce7d001b901942a6bd9a05bc`，commit=`20b3f4d`，39 tests passed；
- D1 formal r1 作为 immutable exact alias：summary SHA=`e3b194c2...66ac`，provenance SHA=`8951543c...b898`，
  fixed checkpoint SHA=`c9d2a052...af52`，Rigid target=`105,412`；不重训、不改写；
- 新训练仅 D2 30k / seed 0，每 5k checkpoint；fixed D1→D2，matched 从 D2 网格匹配 D1 fixed Rigid，
  maximum gap=2%，无 pruning/retrain/retune/mutation；
- 完整复用 held-out/high/boundary/non-target、checkpoint immutable 与 exact quality/quality-cost Pareto；
- read-only preflight=`done`，SHA=`9cf49af0be9a2676c6c113bee963efb79704bb9434083857684f97bd19caaa28`，
  GPU 0 MiB、free disk 47.92 GiB，D2 smoke/D1 alias/r8 patch/cgroup 门禁通过。

### `WS-V3-A2-ACTOR-DENSIFY-01` D2 formal 正式结果与收口

- run=`20260809T113230Z__a2-d2-formal30k-s0-r1`，terminal=`done`，source=`482fba0`，summary SHA-256=
  `9c41dfc83c9da0a14201e1c719fb3d0e2cf59dd1ad20cd279c6e1a9a1c97de7d`，manifest SHA-256=
  `260af5d99f3d3ece4f2c178f8c18385338432da9fbf94b7d8a4603163db20926`；
- D2 final checkpoint SHA-256=`1a061247...e7c`，30k counts=`Background 1,205,164 / Rigid 104,704`；
  D1 checkpoint SHA 在运行前后保持 `c9d2a052...af52`；D1/D2 initialization provenance 相同；
- 六个 5k grid checkpoint 均 finite 且通过 quota/cap；matched 目标=`105,412` Rigid，选中 D2 30k，
  gap=`708 / 0.67165%`，所以 fixed 与 matched D2 是 exact alias；
- D2/D1 global PSNR=`27.703188/27.770024`、SSIM=`.850333/.850915`、LPIPS=`.178344/.177704`；
  boundary-support boundary-band PSNR=`26.171399/25.770024`、SSIM=`.828868/.821572`、
  LPIPS=`.044568/.048382`；局部边界改善与 global/其他局部轴退化并存；
- fixed/matched strict-quality exact Pareto 都为 `tradeoff_non_dominated`（D1/D2/equal=`11/8/0`）；
  quality-cost 也为 `tradeoff_non_dominated`（`14/9/1`）。D2/D1 wall=`2720.82/2099.33 s`；
- D2 audit=`30,001` observation events、boundary/residual 各 `591,405,097` observations、`295`
  refinement events、capped Gaussian=`70,764`；297 条资源记录、四个 stage completed、无 OOM；
- 状态=`done`。冻结 D2 boundary-residual 为 A3 的 boundary-priority research asset，D1 quota-only 为 fallback；
  不宣称 D2 dominance。`d3_unlocked=false`，D4 未启动。

### `WS-V3-A3-LOCAL-REFINE-01` I0 语义协议

- config=`configs/worldsim_v3/a3_local_refine_protocol_v1.yaml`，SHA-256=
  `03fbf632645326692bbcf18ab18a08b5440c7733c709f925945c78018bb272d0`；A2 closeout=`2246693`；
- input 固定为 D2 scene-0230 30k checkpoint SHA=`1a061247...e7c` 和 actor registry SHA=
  `ed57764e...0c68`；D1 SHA=`c9d2a052...af52` 只作 fallback；
- scene/seed/cameras=`scene-0230/0/[0,1,2]`；actors=`high-support/boundary-support`；edits=
  `lateral +1m/delete`；stride-10 的 19 个 held-out frames 禁止进入优化或支持选择；
- paired footprint threshold/dilation=`2/2px`，affected union dilation=`3px`，depth-order tolerance=`0.05m`；
  source/edited mask 仍是 counterfactual diagnostic；
- S-A/S-B/S-C 严格分层；S-B 禁止 RGB loss，S-C 禁止更新/seed/loss；expected/first-hit/measured depth=
  `diagnostic/T1/T0`，ancestry 不冒充 measured depth；
- R0 为 immutable exact alias；R1 只允许 affected S-A/S-B 的 Background opacity/scale；outside parameter 与
  optimizer、RigidNodes/trajectory/registry exact。R2 appearance、R3 evidence seed、R4 temporal 继续锁定；
- I0 冻结时 stage=`semantic_protocol_and_synthetic_contract_only`、`formal_training_authorized=false`；V2 M5 未提交文件
  不得成为 A3 dependency；新增 `12 passed`，联合 WorldSim V3/materializer 回归 `98 passed`。后续工程门见 I1/I2。

### `WS-V3-A3-LOCAL-REFINE-01` I1 engineering guard / synthetic closeout

- implementation=`9c639dd5a0adcd1f8b5126f7f20d836815b127a6`；DriveStudio patch SHA=
  `155ec58fd2bfdc2e40357035dc20800bf2340b0c1c9ac5972c7c78efbd8cb69b`；独立 A3 工作树通过
  apply/reverse、`py_compile` 与 import；
- run=`20260809T132133Z__a3-r0-r1-synthetic-s0-r1`；summary SHA=
  `2ac123f0603120a103743e59680a31dd4cdf5b6d5fa45605d7c84d36ec337ada`，manifest SHA=
  `8ffa697e15d8a97108d8281a51313119c304fbf0f245d88bfbd127663fde27c4`；联合回归 `110 passed`；
- R0=`immutable_exact_alias_no_optimizer_no_new_checkpoint`，重新命中 D2 checkpoint/config/protocol SHA；
- R1 synthetic 只改变 affected S-A/S-B Background opacity/scale；outside parameter/Adam state、position/color、
  RigidNodes/trajectory、tensor shape/order exact；原 D2 与 A3 module-off RGB/SSIM loss tensor exact；
- 缺 paired provenance/masks 时 DriveStudio fail closed；实际 checkpoint Background/Rigid rows=
  `1,205,164/104,704`，trajectory=`196×24`；
- 状态=`done synthetic_contract_only`，不是 paired/质量证据；该 run 自身记录
  `paired_engineering_smoke_complete=false`。后续 I2 已关闭 paired 工程门，但 `formal_training_authorized=false` 不变。

### `WS-V3-A3-LOCAL-REFINE-01` I2 real S-B/T0 paired smoke / numeric freeze

- sidecar materializer/controller commits=`3b8526af6e3ffb53362ec6641d6f280862ad1cb8 / aac521328eb38e8367e4071601443c0c45086a39`；
  run=`20260809T133911Z__a3-sb-sidecar-s0-r3`，manifest/rows SHA=
  `42474f73fc563a2bba4c52cbec029bb4c28d33a21ca5f3d83ad4311bb7957273 / c5756ecbc0eabee9a576a55297a1739aa20e2af578aa4a5a92e727701b5138fc`；
- high actor 选择 frame 0，boundary actor 选择 frame 31，均为 camera 0 且不是 19 个 heldout frames；四个
  lateral/delete units 冻结 affected/S-B mutable/S-C rows=`16,502/51/16,451`，共 8 个 T0 geometry pixels；
  S-A/RGB=`0/ABSTAIN`，first-hit alpha=`0.5` 仅作 visibility diagnostic；
- paired implementation/fix=`d89e0ace37eda22434470849ec9940360c0e9251 / 78741b3abee07b2c39be6646c63928e8212b6a6b`；
  native Gaussian/trainer regularizer 被双重关闭，S-B occupancy target 只使用 measured T0 LiDAR；delete 使用临时
  visibility 关闭而不删除 Gaussian，lateral 在每 unit 后精确恢复 actor trajectory；
- canonical paired run=`20260809T135921Z__a3-r1-sb-paired4-s0-r2`，summary/manifest SHA=
  `ba4e2b853690f0b9c9bb7bfe039b4571db16c020ce726768a1ff884b09b3557d / de717ba0a5adb1afeb416a15a53ec55f471a8eb841882f784012b04ac86b596c`；
  step `30001–30004`，每 unit 一步；opacity/scale 均有行内 finite nonzero gradient 和变化，outside parameter/Adam、
  Rigid/trajectory/registry、shape/dtype/order exact；checkpoint SHA=
  `e995e7c266d9fed4e64c86813718e46ab4576bbfdf60500a637bdaeaaba78cd1`；
- numeric freeze commit/config SHA=`c02c8c74c671362e86269bd7e00980bfa75ae1c9 / d9289df0b2ac7df7a7c408b5cb1601bc5f874e2922ebc9cb87961aacee43b3e3`；
  steps=`4`，opacity/scaling LR=`0.05/0.005`，affected/mutable cap=`16,502/51`，seed cap=`0`，alpha=`0.5`；
- frozen replay run=`20260809T140534Z__a3-r1-sb-frozen-replay4-s0-r1`，summary/manifest SHA=
  `7d820a53de21f505a5c56043d56556edb8d3a86510488ea3956b7cfa159187c6 / 393e65d5f91c0e2072eebd7c23a1161d46422502220ceeeaa18c04905fec646d`；
  四 unit loss 与 checkpoint SHA 逐位重现；wall/GPU/cgroup=`50.68s/8,286.86MiB/22,631,796,736B`，OOM delta 0；
- 当前定向/联合回归=`26/119 passed`。本节只登记工程可重放性，不登记 heldout/S-B RGB 质量改进；
  `quality_claim_authorized=false`、`formal_training_authorized=false`、R2–R4 未授权。

### `WS-V3-A3-LOCAL-REFINE-01` I3 heldout read-only evaluation / negative closeout

- frozen protocol=`configs/worldsim_v3/a3_r1_eval_protocol_v1.yaml`，SHA-256=
  `eb87a9f2ea7df9bdc050a8d4e4f3cdc7c6a1115ea6f4f69e2fd3c8011904b05a`；protocol/evaluator commits=
  `42508fb810bad5bcd9bd16f6386465c4b9fc8c95 / c8fc5603b471588a5b4d3e54199f4b44b5cf1752`；固定
  19 heldout frames × 3 cameras、R0-derived masks、T0/T1 geometry、non-target/global RGB safeguard、exact Pareto、
  checkpoint immutability 与只读执行；
- observability/memory diagnostic commits=`05cee1ed0ea6144dd2a8bb95a04e9457b1ac64c1 / c9e3df4654933b2e8a21fcd10dc37f1acea9efd8 /
  ef74622382dd2c2b96090282db80f4eb8c315077 / c2eb14f635b5ddee362f584b20bea967dafcfaf2`；当前联合回归=
  `139 passed`。这些提交不改变 protocol SHA、端点、mask、阈值或 checkpoint；
- failed closeout run=
  `/root/autodl-tmp/runs/worldsim_v3/WS-V3-A3-LOCAL-REFINE-01/20260809T144037Z__a3-r1-heldout-eval-s0-r5`；
  exit=`1`，terminal SHA=`eabe266bc3b4e7917391789e95e8cff771c6cac7d2b0567e60205f9cd8a7aad9`；resource audit
  SHA=`d9536f4ec937bee0694a754038b22ab75a4b6b028f20e1e6f42e38e4db9a6280`；wall/GPU/cgroup/run bytes=
  `117.983202636 s / 14,241.398926 MiB / 23,749,709,824 / 299,910`，GPU ceiling=`12,288 MiB`，OOM/OOM-kill
  delta=`0/0`；
- metric/global rows count=`432/114`，SHA=
  `04da7a2503460c075a3164c90d6c08436bbea9f4ec5560ea0417ee40e91aa939 / 04bf741e1da6cfe845b5ee6c9d4cccede54d79a1c8f7178e00abcf737ff7245e`；
  checkpoint before/after SHA=`1a061247...e7c / e995e7c2...8cd1`，run 中无 `.pth`；
- resource-invalid diagnostic primary axes（R0→R1）：coverage=`1.0→1.0`，depth violation=
  `.9157920573→.9081725143`，non-target RGB MSE=`.002095031327→.002095032019`，original-global RGB MSE=
  `.002104032262→.002104032654`；exact Pareto=`tradeoff_non_dominated`。该重算不得登记为合格 heldout 质量证据；
- r2/r4 同样完成 432/114 rows 后只失败 GPU ceiling，峰值=`14,241.777/14,244.924 MiB`；r3 在指标前因
  Rigid quota device mismatch 失败并已修复。失败 run 不覆盖、不删除、不改写为 done；
- final decision：R1 arm=`rejected_resource_gate_and_diagnostic_tradeoff`，A3 task=`done`，`A3*=R0-off`
  （D2 immutable exact alias）。S-A 未物化，formal/R2–R4 未授权；下一阶段为 A4-P0 profile 协议冻结。

### `WS-V3-A4-DEPLOYMENT-01` P0 end-to-end profile protocol freeze

- protocol=`configs/worldsim_v3/a4_p0_profile_protocol_v1.yaml`，SHA-256=
  `8ba96278b7f65957480a343a21977e2e24a537462b7a0b042a3268684d27d9a4`；A3 closeout=
  `10eee3ad30c3729532afecdcc520c1ef542e0210`；selected asset=`A3*=R0-off/D2 exact alias`，checkpoint/config/
  registry SHA=`1a061247...e7c / 115deaf...5e68 / ed57764e...0c68`；R1 输入禁止；
- A2-D2 train、render/eval、registry、actor metrics stage JSON 及 manifest/summary/resource log 均固定 hash，P0
  只读复用，不重跑训练或质量评测；convert 只盘点 inventory，不做参数转换；
- new probe=`process-cold/warm load + 2 warm-up + frames 10/100/190 × cameras 0/1/2 original render`，原生
  `1600×900`、CUDA 同步、P50/P95 nearest-rank、只存 RGB hash/JSON；OS cache eviction 禁止且 filesystem cache
  明示 uncontrolled；
- frozen ceilings=`600 s / 16,384 MiB allocated / 24,576 MiB reserved / 24,000 MiB NVIDIA sampled / 32 GiB
  cgroup / 50 MB run`，OOM delta=`0/0`；recovery=`inventory→runtime_probe→aggregate→resume_audit`，completed
  stages 不覆盖，resume audit 不启动 GPU；
- validator preflight 核对 10 个 immutable path/hash/bytes；协议单测=`4 passed`，联合 WorldSim V3=`143 passed`。
  本条写入时尚未读取新 A4 runtime/load 结果；P1/P2/P3/P5 未授权。

### `WS-V3-A4-DEPLOYMENT-01` P0 v1 resolution block / v2 correction freeze

- runner 实现提交=`199abd99d642747241b79ce543c8eb9096553a1d`；v1 formal r1=
  `20260809T151539Z__a4-p0-profile-s0-r1` 完成 inventory/runtime/aggregate/resume 四个 stage，resume audit 未导入
  torch、前后均无 GPU compute process、三个 completed stage hash 全部复用；
- r1 终态=`blocked`，唯一失败 audit=`native_resolution_exact`。11 个 runtime rows 均为 `800×450`；warm-up 两行
  RGB hash exact，rows SHA=`fbf801eb...1e1a`，runtime stage SHA=`d86ec9fa...9b7`，terminal SHA=
  `9084e49c...e5ed`；run 内无 checkpoint/render media；
- source config SHA=`115deaf...5e68` 明确冻结三路 `downscale_when_loading=[2,2,2]`。因此 v1 的 `1600×900`
  是传感器原始尺寸，不是当前 checkpoint 的模型原生 render 尺寸；v1 不改写，也不登记为正式 P0 性能结果；
- r1 resource diagnostic 自身 passed：wall=`62.144008 s`，prepare=`52.069271 s`，cold/warm load=
  `.341640/.343181 s`，render P50/P95=`.029865/.092485 s`，FPS=`20.501837`，torch allocated/reserved=
  `7,913.31/8,232 MiB`，NVIDIA sampled=`8,574 MiB`，cgroup peak=`24,775,639,040 bytes`，OOM=`0/0`；这些值
  只用于 v2 风险审计；
- v2 protocol=`configs/worldsim_v3/a4_p0_profile_protocol_v2.yaml`，SHA=`43db7182...3f18`；只把模型原生尺寸纠正
  为 `800×450`，冻结 v1 protocol/manifest/runtime stage/rows/resource audit/terminal 六项证据，其余输入、矩阵、
  ceiling、recovery 和 claim boundary 不变。v2 必须新目录完整 rerun，不复用 v1 measured runtime stage；
  validator exact 核对 16 个 inputs，协议测试=`7 passed`，联合 WorldSim V3=`152 passed`；P1/P2/P3/P5 仍未授权。

### `WS-V3-A4-DEPLOYMENT-01` P0 v2 canonical profile result

- canonical=`20260809T152923Z__a4-p0-profile-v2-s0-r2`，source commit=`b191afaa...8897`，exit=`0`，
  terminal=`done`；summary/manifest/resource/rows SHA=`0278a320...e92 / 12df93b3...a0f5 / b89c93bb...5fac /
  4a94b1fb...a934`；13/13 audits true，checkpoint/registry before-after exact，无训练、optimizer、checkpoint 或媒体输出；
- inventory：checkpoint/config/registry=`578,819,674 / 4,661 / 3,721,428 bytes`，checkpoint+registry=
  `582,541,102 bytes`；static block=`1 monolithic`，actor assets=`24 total / 23 available / 1 unavailable`，Gaussian=
  `1,205,164 Background / 104,704 RigidNodes`；convert=`inventory_only_no_parameter_conversion`；
- performance：wall=`60.784519 s`，prepare=`50.420569 s`（82.95%），trainer construction=`1.885644 s`，
  process-cold/warm load=`.391351/.397158 s`；9-view render P50/P95=`.068017/.127388 s`，FPS=`16.377547`；
  filesystem cache=`uncontrolled_report_explicitly`；
- resources=`passed`：allocated/reserved/NVIDIA sampled=`7,913.31/8,232/8,574 MiB`，cgroup peak=
  `24,474,128,384 bytes`，run bytes=`85,169`，disk free=`45,292,818,432 bytes`，OOM/kill=`0/0`；
- recovery：no-torch dry-run=`.160304 s`，复用 3 个 completed stage，输入/输出=`16,256/919 bytes`，无 GPU launch；
  summary 的全部性能值只适用于 scene-0230/seed-0/800×450/单进程，不产生 concurrency 或质量 claim；
- P0=`done`。prepare 是明显主导项，而 load/runtime 未触发冻结资源门，故不先承担 P1/P2/P3 的质量或数值风险；
  下一门禁只冻结 reference-only P5 registry/resume 协议，再决定是否执行。P1/P2/P3 仍未授权。

### `WS-V3-A4-DEPLOYMENT-01` P5 registry/resume protocol freeze

- protocol=`configs/worldsim_v3/a4_p5_registry_resume_protocol_v1.yaml`，SHA=`51acb935...5874`，P0 closeout=
  `9811c03c...e0b5`；validator exact 核对 P0 protocol/manifest/summary/resource/rows/terminal 与 checkpoint/config/
  actor registry 共 9 项 path/hash/bytes，P0 terminal/13 audits 必须 done/true；
- output=`artifacts/deployment_registry.json`，schema=`worldsim-v3-deployment-registry-v1`，mode=
  `reference_only_immutable_manifest`；checkpoint copy/rewrite 禁止，registry ceiling=`2,000,000 bytes`；
- static 固定 `models.Background / 1 asset / 1,205,164 GS / monolithic reference / not independently extractable`；
  actors 固定 `models.RigidNodes / 24 assets / 104,704 GS / 23 available / 1 unavailable`，compact entry 只保留身份、
  selector、count、flat-index hash 与 source registry hash，不复制大段 ranges；
- reload=`fresh DriveStudio / load_only_model / exactly one checkpoint load / 0 render`；核对模型总量与全部 actor
  count/index hash，不构造 optimizer、不训练；filesystem cache 仍 uncontrolled；
- recovery=`input_audit→registry_materialize→reload_smoke→aggregate→resume_audit`，completed stage 不覆盖，resume=
  no-torch/no-GPU；ceilings=`180 s / 16,384 MiB allocated / 24,576 MiB reserved / 24,000 MiB NVIDIA / 32 GiB
  cgroup / 5 MB run / OOM 0`；required audits=`14`；
- 协议测试=`6 passed`，联合 WorldSim V3=`158 passed`。本条写入时尚未执行任何 P5 formal measurement；下一动作
  只实现并提交 runner。P1/P2/P3 未授权。

### `WS-V3-A4-DEPLOYMENT-01` P5 formal registry/resume result

- runner=`4de2126e...d8078`。formal r1=`20260809T155209Z__a4-p5-registry-resume-s0-r1` 完成 input audit 与
  registry materialize，产出 `14,729-byte` compact registry 后在 reload 读取 `RigidNodes.points_ids` 时报错；
  terminal=`blocked`，SHA=`61d30a11...773e`，registry SHA=`e48bccdf...9039d`，旧证据不覆盖；
- root cause 是 DriveStudio checkpoint state key=`points_ids`、loaded runtime attribute=`point_ids`。fix=
  `0e899b2e6dcf7d5a091a0a4092ea99767c982357`，只修运行时读取并锁定别名拒绝；protocol SHA 保持
  `51acb935...5874`。聚焦测试=`15 passed`，联合 WorldSim V3=`167 passed`；
- canonical r2=`20260809T155753Z__a4-p5-registry-resume-s0-r2`，source=`0e899b2...2357`，exit=`0`，
  summary/manifest/resource/registry SHA=`0c86ff68...8744 / 78830d74...58bd / f6c06df0...3ac4 / e48bccdf...9039d`；
  required audits=`14/14 true`，source checkpoint/registry before-after exact；
- compact registry=`14,729 bytes`，canonical content SHA=`02467963...cb6`；`1 static / 1,205,164 GS`，
  `24 actors / 104,704 GS / 23 available / 1 unavailable`，fresh reload 后 24/24 actor count/index hashes exact；
- reload=`52.320687 s`，prepare/trainer/load=`49.631015/1.987488/.445515 s`，checkpoint load count=`1`，render=`0`，
  无 optimizer/training/checkpoint copy；resources=`passed`：allocated/reserved/NVIDIA=`7,188.73/7,226/7,564 MiB`，
  cgroup=`24,498,089,984 bytes`，wall=`60.437454 s`，run=`102,229 bytes`，OOM/kill=`0/0`；
- no-torch resume=`.127572 s`，torch 未导入、GPU launch=false、四个 completed stage 按 hash 全复用；P5=`done`。
  结论只覆盖 reference-only packaging 与单实例 recovery。A4 仍需 P1/P2/P3；下一步只冻结 P1 protocol，P2/P3
  继续未授权。

### `WS-V3-A4-DEPLOYMENT-01` P1 contribution-prune protocol freeze

- protocol=`configs/worldsim_v3/a4_p1_contribution_prune_protocol_v1.yaml`，SHA=`4f893c09...429b`，P5 closeout=
  `4db43ddc...1779b`；full validator exact 核对 source checkpoint/config/registry、D2 quality、P0 profile、P5
  registry/recovery 共 13 files + 1 directory，33 个 frozen actor masks digest=`429f3693...43cf`；
- ranking 只用 train frames `[5,45,85,125,165,195] × 3 cameras`，从 gsplat near→far intersections 稳定计算
  occlusion-aware `T_before×alpha`；`[10,50,90,130,170,190] × 3` heldout contribution 只作 audit，不参与排名；
- arms=`source/b05/b10/b20`。Background 与 23 个 available actors 分资产按冻结 key 排名并删除固定 fraction，
  unavailable actor 保持空；所有 Gaussian/`points_ids`/ancestry row 同 mask，Sky/LPIPS/trajectory/step exact；
- full quality=`19 heldout frames × 3 cameras`，复用 source 的 33 mask bytes，禁止 candidate 重生成。global
  PSNR/SSIM/LPIPS 最大退化=`.10 dB/.002/.002`；actor/boundary=`.20 dB/.005/.005`、MAE `+.002`；non-target=
  `.10 dB/.002/.002`、MAE `+.001`；缺失或 nonfinite 即 reject；
- selection 固定为通过 reload/count/invariant、质量、compression 与资源门的最大 prune fraction；若全失败，P1 method
  rejected 且生产资产 exact fallback 到 source。不得看结果新增 arm/阈值，也不把 bounded loss 写成质量提升；
- recovery=`11 stages`，ceilings=`1,800 s / 20,480 MiB allocated / 24,576 MiB reserved / 24,000 MiB NVIDIA /
  48 GiB cgroup / 2.5 GB run / 30 GB disk floor / OOM 0`，required audits=`21`；协议测试=`11 passed`，联合
  WorldSim V3=`178 passed`。本条写入时尚无 P1 新 measurement；下一动作只实现并提交 runner，P2/P3 未授权。

### `WS-V3-A4-DEPLOYMENT-01` P1 formal contribution-prune result

- runner=`19cab2cf40b8ed8ef9a4ad1ba8cce4cc8cf67163`；正式运行前聚焦测试=`23 passed`，实现提交后完整
  `tests/*worldsim_v3*.py` 回归=`190 passed`。canonical r1=
  `/root/autodl-tmp/runs/worldsim_v3/WS-V3-A4-DEPLOYMENT-01/20260809T165058Z__a4-p1-contribution-prune-s0-r1`，
  exit=`0`、terminal=`done`、21/21 audits true；summary/manifest/resource/terminal SHA=
  `7c5347e3...7119 / 486342ba...61ac / 8b6073ed...4b7c / 80dd8178...c645`；
- contribution scan=`198.712 s`，完整覆盖 `18 train ranking + 18 heldout audit-only` views；score NPZ=
  `30,376,517 bytes`、SHA=`0165401a...69a9`，15 个数组的 dtype/shape/content hash exact。峰值 allocated/
  reserved/NVIDIA/cgroup=`14,342.71 MiB / 14,892 MiB / 15,234 MiB / 24,710,811,648 bytes`；
- materialization 全部 exact：

| arm | checkpoint bytes | byte reduction | Background / RigidNodes GS | reload/count/invariant |
|---|---:|---:|---:|---|
| source | 578,819,674 | 0 | 1,205,164 / 104,704 | exact |
| b05 | 554,938,306 | 23,881,368 | 1,144,906 / 99,480 | exact |
| b10 | 531,056,962 | 47,762,712 | 1,084,648 / 94,246 | exact |
| b20 | 483,292,674 | 95,527,000 | 964,132 / 83,773 | exact |

- source replay 与冻结 historical global/actor/boundary/non-target 逐端点 exact。候选门禁结果：

| arm | failed safeguards / 31 | 关键失败 | quality decision |
|---|---:|---|---|
| b05 | 3 | occupied PSNR `-0.117684 dB`；global PSNR `-0.110926 dB`；non-target PSNR `-0.125462 dB` | rejected |
| b10 | 12 | global PSNR `-0.364281 dB`、LPIPS `+.004091`，并含 non-target/动态端点失败 | rejected |
| b20 | 15 | global PSNR `-0.682975 dB`、LPIPS `+.007970`，并含 human/non-target 端点失败 | rejected |

  三臂 actor/boundary 的部分指标不退化，但不能覆盖 all-endpoint 合同；禁止事后增加 `<5%` arm 或放宽门限；
- 9-view runtime 只作报告：source/b05/b10/b20 load=`.365/.387/.365/.356 s`，P50=
  `.0447/.0329/.0700/.0399 s`，P95=`.1402/.0785/.1538/.1008 s`，FPS=`18.11/23.64/14.54/22.73`。
  cache 未控制且结果非单调，不用于选择；
- final resources=`passed`：wall=`605.281 s`，allocated/reserved/NVIDIA=`14,342.71/14,892/15,234 MiB`，
  cgroup=`26,264,842,240 bytes`，run=`1,610,165,885 bytes`，disk free=`43,679,989,760 bytes`，OOM/kill=`0/0`；
  no-torch resume=`2.316 s`，10/10 completed stages 复用、GPU launch=false；
- `method_state=rejected_quality_or_integrity_gate`，selected=`p1-source` immutable exact alias，P1 experiment=`done`。
  这是 scene-0230/seed-0 的冻结负结果，不是所有贡献度剪枝或跨场景剪枝不可行的结论。A4 仍需 P2/P3；下一步
  只冻结 P2 FP16 协议。

### `WS-V3-A4-DEPLOYMENT-01` P2 mixed-precision protocol freeze

- protocol=`configs/worldsim_v3/a4_p2_mixed_precision_protocol_v1.yaml`，SHA=`6558fb3f...6d4e`，P1 closeout=
  `e733cbed...283d`；9 个 exact files + 33-mask directory 锁定 P1-selected source 与 P1 canonical 21/21 evidence，
  不允许将 b05/b10/b20 rejected checkpoint 输入 P2；
- arms 固定 `p2-source` 与 `p2-gs-param-fp16`。候选只把两 Gaussian 模型的 `_scales/_quats/_features_dc/
  _features_rest/_opacities` 共 10 个 float32 tensors 转为 float16；候选必须 bitwise 等于 `source.to(float16)`，
  checkpoint schema、Gaussian/actor count/index 保持 exact；
- source dtype audit 在新 P2 measurement 前确认 Background means 范围 `[-686.0377,2996.3384] m`，若 FP16
  roundtrip 最大误差=`.999267578125 m`；故 Background/Rigid means、Sky、LPIPS、trajectory、points_ids、ancestry/
  quota/boundary state 与 step 全部保留 FP32/原 dtype exact；
- runtime policy 是 persistent converted parameters FP16、`collect_gaussians` 后 renderer inputs FP32、autocast=false。
  本实验不声称 FP16 gsplat kernel 或 Tensor Core speedup，checkpoint bytes 下降也不自动等于 peak VRAM 下降；
- quality 仍为 57 views/33 frozen masks/31 endpoints；global 门=`.05 dB/.001/.001`，actor/boundary=
  `.10 dB/.0025/.0025/+ .001 MAE`，non-target=`.05 dB/.001/.001/+ .0005 MAE`，必须 31/31；
- runtime=`2 arms × frames 10/100/190 × cameras 0/1/2`、2 warm-up、800×450、sync/nearest-rank；recovery=
  7 stages；resources=`900 s / 16,384 MiB allocated / 24,576 MiB reserved / 24,000 MiB NVIDIA / 48 GiB cgroup /
  1 GB run / 30 GB disk floor / OOM 0`；required audits=`19`；
- full validator exact 通过 10 项输入记录与 source 10-field dtype audit；协议测试=`9 passed`，联合 WorldSim V3=
  `199 passed`。本条没有 P2 conversion/render measurement；下一动作只实现并提交 runner，P3 未授权。

### `WS-V3-A4-DEPLOYMENT-01` P2 mixed-precision formal

- protocol/runner/fix=`6558fb3f...6d4e / 1cd9a6e / dcf2822`。r1=
  `20260809T174337Z__a4-p2-mixed-precision-s0-r1` 完成 candidate、quality、runtime、aggregate 与 resume，但 mapped
  Gaussian Parameters 未进入 persistent-byte inventory，finalizer 唯一 audit 失败；terminal SHA=`5ef3dab6...74c0`，
  run 保留 `blocked`。修复只补账本遍历与回归，不改协议或测量合同；
- canonical r2=`20260809T174850Z__a4-p2-mixed-precision-s0-r2`，source=`dcf2822...9860`，exit=`0`、terminal=
  `done`、summary/manifest/resource SHA=`980f9b0f...1103 / bed45626...98cb / 221d5e82...0df5`，19/19 audits；
- candidate checkpoint=`7be87e8b...7448 / 432,111,754 bytes`，相对 source `578,819,674` 减少
  `146,707,920 / 25.346049%`；registry=`69c4f38a...8a27 / 3,721,277 bytes`。10-field half bytes、75 preserved
  tensor、schema/count/index/reload 与 renderer FP32 输入全部 exact；persistent parameter bytes=
  `394,641,424→247,936,208 / -37.174307%`；
- source 31-endpoint replay 最大绝对差=`0`；candidate 31/31 safeguards 通过。最大门限消耗为 high-support boundary
  LPIPS `+0.000036410 / 1.4564% budget`；global human SSIM=`-0.000009515`，boundary-support actor PSNR=
  `-0.000755311 dB`。这是 bounded-loss pass，不登记质量提升；
- runtime 只报告 source/candidate load=`.33669/.47407 s`、P50=`.04583/.08721 s`、P95=`.13170/.09750 s`、
  FPS=`17.256/13.065`；filesystem cache 未控制且 P50/FPS 退化，不登记 speedup；
- resource=`passed`：wall=`206.548 s`，allocated/reserved/NVIDIA=`7,754.05/8,072/8,426 MiB`，cgroup=
  `29,673,631,744 bytes`，run=`436,430,167 bytes`，disk free=`42,806,071,296 bytes`，OOM/kill=`0/0`；resume=
  `1.217 s`、6/6 stages、no torch/no GPU；
- selection=`p2-gs-param-fp16`，method=`selected_mixed_precision_parameter_storage_fp32_render`，P2=`done`。结论只
  覆盖 scene-0230/seed-0，且不声称 FP16 renderer/Tensor Core/VRAM 或 load/render speedup；下一步只冻结 P3。

### `WS-V3-A4-DEPLOYMENT-01` P3 exact chunk package protocol freeze

- protocol=`configs/worldsim_v3/a4_p3_chunk_protocol_v1.yaml`，SHA=`dfaaba79...1b41`，P2 closeout=
  `e954e23c...ebe7`；9 个 exact files + 33-mask directory 锁定 P2-selected mixed checkpoint 与 P2 canonical
  19/19 evidence，禁止接入原 FP32 source 或 P1 rejected prune candidates；
- static contract 是 origin `[0,0] m`、50 m XY half-open cells、数值 `(ix,iy)` 排序；source inventory=
  `133 chunks / 1,205,164 rows / count 1..330,169 / 98 sparse<100 / 7 dense>=10,000 / 69,393 rows within
  .25 m of cell edge`。稀疏和离群 chunk 全保留，不设 min count、merge 或 post-hoc cell-size search；
- Background/Rigid 分别冻结 25/26 row tensors。每个 asset 保存全部 row fields 与升序 `int64 source_flat_indices`；
  24 个 actor assets 中 23 个非空 source rows 全部 interleaved，actor 14 仍输出完整 zero-row asset。static/actor
  source inventory SHA=`d78fa6e...27cae / 384870e6...f23a`；
- package=`manifest + skeleton + 133 static + 24 actor = 159 files`；shared skeleton 对 row tensors 使用唯一 sentinel，
  其余 state exact。row values 不重复；manifest 绑定逐文件 path/SHA/bytes/count/bounds/index digest；候选只在内存
  scatter 重组并要求 recursive schema、tensor shape/dtype/value、reload/registry 与 P2 FP16-persistent/FP32-renderer
  adapter exact；禁止复制 source 或持久写出 reassembled checkpoint；
- quality=`57 views + 33 masks`，source 先 exact replay P2 selected quality，chunk 再要求 57 个 RGB SHA 和 31 个
  endpoints exact；runtime=`2 arms × 9 views`、2 warm-up、800×450、读取全部 package、cache uncontrolled，仅报告。
  选择成功为 `selected_exact_chunk_package`；任一 integrity/quality/resource 门失败则 exact fallback P2；
- recovery=8 stages，resources=`900 s / 16,384 MiB allocated / 24,576 MiB reserved / 24,000 MiB NVIDIA /
  48 GiB cgroup / 1 GB run / 30 GB disk floor / OOM 0`，required audits=`21`。full validator passed，协议测试=
  `12 passed`，联合 WorldSim V3=`222 passed`。本条没有 P3 materialization/render/formal measurement；下一动作只
  实现并提交 runner，P4 继续未授权。

### `WS-V3-A4-DEPLOYMENT-01` P3 exact chunk package formal

- runner commit=`aba55777f38a3d8e4363d2ff7d546d412214b481`；focused=`23 passed`，WorldSim V3 full=
  `233 passed`。canonical r1=
  `/root/autodl-tmp/runs/worldsim_v3/WS-V3-A4-DEPLOYMENT-01/20260809T184240Z__a4-p3-chunk-s0-r1`，
  exit=`0`、terminal=`done`、21/21 audits passed；summary/manifest/resource/terminal SHA=
  `f8e6e166...a293 / 8b79d355...bd7af / 55ee6f0b...55e81 / 80dd8178...c645`；
- package manifest=`35a3f1fe...64b8 / 143,913 bytes`；133 static、24 actor、1 skeleton 加 manifest 共
  `159 files / 444,177,055 bytes`，158 payload=`444,033,142 bytes`。source checkpoint=
  `432,111,754 bytes`，package 开销=`12,065,301 bytes / +2.792171%`；skeleton=
  `101,176,684 bytes / 51 row sentinels`，没有 source copy 或 persistent reassembled checkpoint；
- 85 个 tensor path 的 recursive schema/shape/dtype/value SHA 与 non-tensor signature 全部 exact；Background=
  `1,205,164`、RigidNodes=`104,704` rows 均 covered once，missing/duplicated=`0/0`；24 actor assets 完整，actor 14
  为显式 zero-row asset。source checkpoint/registry SHA 前后保持 `7be87e8b...7448 / 69c4f38a...48a27`；
- source replay 31 endpoints max abs diff=`0`；chunk 相对 source 的 57 RGB SHA、31 quality endpoints 和 masks
  exact。quality adapter 两臂均为 `57 renderer / 114 SH` observations，runtime 为 `11/22`，全部 FP32 且
  autocast=false；P2 mixed-persistent/FP32-renderer 合同 exact；
- 9-view runtime（2 warm-up、800×450、sync、cache uncontrolled）：source/chunk load=`.907071/4.177543 s`，
  P50=`.030126/.039505 s`，P95=`.094460/.105862 s`，FPS=`21.2783/20.4471`。package 没有缩小，load/reassembly
  与 render 均未加速；结果只支持 exact spatial/actor asset separation，不支持 streaming、speedup 或 concurrency claim；
- resource passed：wall=`221.786 s`、allocated/reserved/NVIDIA=`7,614.99/8,066/8,420 MiB`、cgroup=
  `32,689,958,912 bytes`、run=`444,885,133 bytes`、disk free=`42,359,705,600 bytes`、OOM/kill=`0/0`；resume=
  `1.104 s`、7 actions、159 artifacts、no torch/no GPU；
- selected=`p3-chunk-package`，method=`selected_exact_chunk_package`，P3=`done`。P0/P5/P1/P2/P3 最低完成集
  全部满足，A4=`done`；P4 保持 optional，下一任务是 R0 前置的 F0 官方能力审计。

### `WS-V3-F0-FEEDFORWARD-AUDIT-01` Instant NuRec protocol freeze

- protocol=`configs/worldsim_v3/f0_instant_nurec_audit_v1.yaml`，SHA=`2004a029...fd611`；runner SHA=
  `249f26d5...8e4a`；official checkout revision/tree=`1ce2288e...8d0 / 96e36fa4...5dc0`，16 个关键文件 hash
  exact 且 clean。协议/源码指纹测试=`8 passed`、WorldSim V3 联合回归=`241 passed`；
- 三份正式 weights-only PTH 的 profile/path/HF commit/bytes/SHA-256/Xet hash 全冻结；code/model/dataset license
  分层为 Apache-2.0 / NVIDIA Open Model License / NVIDIA Autonomous Vehicle Dataset License，NCore gated 与
  terms acceptance 不由 source license 代替；
- paper/model-card contract=`calibrated multi-camera RGB + 2–4 Hz + per-image pose/intrinsics + optional cuboids →
  static/dynamic 3DGS + sky + per-camera ISP`；standalone CLI contract=`NCore V4/FTheta/CUDA → static PLY only`。
  CLI 不读 LiDAR，不导出 dynamic、sky、ISP、actor registry/trajectory 或 depth/point map；
- local inference gate=11 条全合取前置：Python 3.11、uv、CC≥8.0、VRAM≥30,720 MiB、RAM≥32 GB、disk free≥
  100 GB、exact weight、licensed NCore input、terms record、exact clean source 与 CLI help。任一失败时禁止构造
  inference command，也不安装依赖、下载权重/gated data 或启动 GPU；
- F1 默认裁决=`conditional_not_unlocked`；static PLY 不是 exact StreetGS checkpoint，且当前无 scene-0230
  nuScenes→NCore exact converter/actor registry。该边界不等于宣称 upstream 永久不可用，DGGT fallback 也不阻塞 R0；
- 本条没有 formal run、inference wall/VRAM 或质量测量。下一动作只运行已提交协议的 canonical read-only audit。

### `WS-V3-F0-FEEDFORWARD-AUDIT-01` canonical local audit

- canonical=
  `/root/autodl-tmp/runs/worldsim_v3/WS-V3-F0-FEEDFORWARD-AUDIT-01/20260809T192139Z__f0-instant-nurec-audit-s0-r1`，
  source=`ab76f1901b60491af7bc8355589e149fcb69fe04`，exit=`0`、terminal=`done`；summary/manifest/terminal SHA=
  `d111c457...be37 / f1c76fdd...6a11 / 207758b9...15c6`；run=`65,917 bytes`，9/9 manifest artifacts 的
  path/bytes/SHA 复核 exact；
- official checkout revision/tree、16 file hashes、code signatures、no-LiDAR-reader 与 clean status 全通过；CLI help
  exit=`0`。官方 `ncore_input+pretrained` tests=`33 passed`；`cli+ncore_input`=`37 passed / 15 failed`，15 项全部为
  当前未配置环境缺 `shortuuid`，是 environment preflight，不是 upstream 质量结论；
- prerequisites=`4/11 passed`：CC 8.6、system/cgroup RAM、source exact、CLI help 通过；Python 3.11、uv、VRAM
  `24,576<30,720 MiB`、free disk `42,327,777,280<100,000,000,000 bytes`、exact weight、licensed NCore input、
  terms record 失败；HF token 也不存在但未单独作为第 12 条门；
- `inference_smoke=not_run_prerequisites_failed`、`inference_command_constructed=false`；audit wall=`.991914 s`，
  torch/GPU/training/dependency install/weight-or-dataset download 全 false，GPU compute process 为空，OOM/kill=`0/0`；
- F0 outcome=`done_local_inference_not_executable_on_current_host`；F1=`conditional_not_unlocked`、未启动。CLI static PLY
  不含 dynamic/sky/ISP/actor registry/depth，且不是 exact StreetGS checkpoint；未来兼容硬件/合法输入/converter
  的窄 static-background pilot 不被永久否定。当前任务切换为 R0 integration，DGGT fallback 不阻塞 R0。

### `WS-V3-R0-INTEGRATION-01` formal 前协议冻结

- protocol=`configs/worldsim_v3/r0_integration_protocol_v1.yaml`，SHA=`4fe20c3197...7575`；runner SHA=
  `d58c4008...c5ce`；前置 closeout commit=`80b4f983b665d7bb3e4d73fb6d9531f1adbbe901`；
- 授权边界为 read/hash/derive reports/document snapshot/chunk-payload verification；training/inference/GPU/install/
  download/checkpoint-or-registry mutation 与 F1/P4/D3/D4/A3 extra 均为 false；
- 冻结 `5 protocols + 51 canonical evidence files + 3 selected production files + 4 offline MP4 = 63` 个输入审计行；
  11 组 terminal status、23/23 decision checks 均 exact；
- P3 package=`159 files / 444,177,055 bytes / 133 static / 24 actor`；manifest 和 158 payload 的 path/bytes/SHA
  全 exact；12 个最终 deliverable 路径与 claim boundary 全冻结；
- selected chain=`A1-C0-off__A2-D2-boundary-priority__A3-R0-off__A4-P2-mixed__A4-P3-exact-chunk`；
  production checkpoint/registry/config SHA=`7be87e8b...7448 / 69c4f38a...8a27 / 115deaf8...5e68`；
- 定向测试 `11 passed`，WorldSim V3 联合回归 `252 passed`。本条未运行 canonical、未生成 terminal，也没有新训练、
  推理、GPU measurement 或质量结论；下一动作只提交冻结协议后运行 canonical read-only integration。

### `WS-V3-R0-INTEGRATION-01` canonical closeout

- canonical=
  `/root/autodl-tmp/runs/worldsim_v3/WS-V3-R0-INTEGRATION-01/20260809T194625Z__r0-integration-s0-r1`；
  source=`64e3d15ca30de44088c2f6fbfb6da048a31a4acf`，terminal=`done`，summary/manifest/terminal SHA=
  `3ffe99ea...15a7 / a9b052a6...1d90 / 207758b9...15c6`；28 files=`1,117,645 bytes`；
- input/decision/deliverable/manifest=`63/63 / 23/23 / 12/12 / 26/26 exact`；11 组 canonical terminal、5 份
  documentation snapshot 与 P3 159-file/444,177,055-byte package 全 exact；
- selected chain=
  `A1-C0-off__A2-D2-boundary-priority__A3-R0-off__A4-P2-mixed__A4-P3-exact-chunk`；五个 final conclusion
  tokens 与 12 条 claim boundary 完整落盘；
- resource/no-launch passed：wall=`1.678173 s`、cgroup current=`30,389,452,800 bytes`、disk free=
  `42,325,843,968 bytes`、OOM/kill=`0/0`；torch/GPU/training/inference/install/download 全未启动；
- `next_action=none_plan_complete`。F1/P4/D3/D4/A3 R2–R4 仍未启动，不冒充 R0 交付或后续缺口。

## 3. V2 冻结注册表

| Task ID | 状态 | 目标 | 当前输入事实 | 解锁条件 |
|---|---|---|---|---|
| `DR-V2-M0-BOOTSTRAP-01` | done | 事实源、分支、镜像与 bootstrap | 正式 run 完成；历史失败实例保留 | README/STATUS/PLAN 一致，bootstrap smoke 通过 |
| `DR-V2-M1-DGGT-REPAIR-01` | done | 修复 pointops2 并做 18-window inference | 1/3-view、common、regional 全部完成 | 18/18 + 216/216 + 完整运行合同 |
| `DR-V2-M2-ACTOR-EVAL-01` | done | nuScenes 真值 actor 评测适配器 | raw 2Hz 轨迹、4,356 exact mappings、6/6 cohort | 三 scene eligible 16/20/6，visual QA 通过 |
| `DR-V2-M3-EDIT-BASELINE-01` | done | DriveStudio/StreetGS 可编辑基线 | 30k checkpoint、registry、27-image smoke 完成 | scene-0230 remove/lateral/3-camera smoke |
| `DR-V2-M4-EDIT-PILOT-01` | done | scene-0230 真实编辑闭环 | 1,764 RGB、9 MP4、1,176 paired rows、16/16 checks | 两种编辑真实执行且证据可审计 |
| `DR-V2-M5-STRESS-3SCENE-01` | pending | 三场景编辑/去遮挡压力测试 | 0230/0242 checkpoint 与 0255 诊断已生成；任务未闭环并冻结 | 历史门禁未满足；V3 不再授权继续 |
| `DR-V2-M6-HYPOTHESIS-01` | pending | 基于真实失败做 novelty gate | 未生成 | V3 路线不再授权 |
| `DR-V2-M7-METHOD-01` | pending | 最小方法与 matched ablation | 未生成 | V3 路线不再授权 |
| `DR-V2-M8-HUMAN-01` | pending | 人工盲审与终局 | 未生成 | V3 路线不再授权 |

## 4. V2 启动前维护记录

2026-08-02 的文档归档和存储清理属于 maintenance，不冒充 `DR-V2-M0-BOOTSTRAP-01`：

- V1 当前态、实验台账、环境与报告已移入命名归档；
- V2 计划已按实际 checkpoint、DriveStudio 缺口和用户镜像偏好校准；
- 可再生中间产物的精确路径、字节数与恢复方式见
  [`archive/2026-08/v2-preflight/CLEANUP_MANIFEST.md`](archive/2026-08/v2-preflight/CLEANUP_MANIFEST.md)；
- AD-GS 六场景最终 60k checkpoint/render/metrics、processed 输入、raw subset、DGGT 完整预下载候选均受保护。

## 5. V1 冻结输入

| 资产 | 终态 | V2 用法 |
|---|---|---|
| AD-GS 六场景 exact reproduction | done，6/6 | 只读 checkpoint/render/metrics；不重复训练 |
| DGGT V1 run | blocked，未 inference | 只作为原始失败证据；V2 新环境重做 |
| V1 pseudo identity audit | 0/12 slots | 失败边界；不得当作真实编辑结果 |
| V1 候选 A novelty | rejected | 禁止复活“补身份 + 基础轨迹编辑”作为贡献 |

## 6. `DR-V2-M0-BOOTSTRAP-01`

### 工程失败实例

- run：
  `/root/autodl-tmp/runs/dynamic_editing_v2/DR-V2-M0-BOOTSTRAP-01/20260802T114342Z__bootstrap-s0`；
- terminal：`blocked / empty_shell_python_not_on_path`；
- 网络四源已可达，但非登录空 shell 中裸 `python` 不在 PATH，导致 `source_resolution.json` 未生成；
- 修复仅显式选择 `/root/miniconda3/bin/python`，没有安装依赖或改写全局环境。

### 正式完成实例

- 验证实例：
  `/root/autodl-tmp/runs/dynamic_editing_v2/DR-V2-M0-BOOTSTRAP-01/20260802T114453Z__bootstrap-s0-r2`
  为 `done`；其后只为让 source snapshot 覆盖相对 `HEAD` 的 staged/unstaged 全部 M0 文件创建 r3，未改变
  bootstrap、资产或测试协议；
- run：
  `/root/autodl-tmp/runs/dynamic_editing_v2/DR-V2-M0-BOOTSTRAP-01/20260802T115419Z__bootstrap-s0-r3`；
- terminal：`done`；
- branch：`research/dynamic-editing-v2`；source commit：`09fbb55` + 本 M0 工作树快照；seed：`0`；
- empty shell/tmux shell：`PASS/PASS`；TUNA Conda/PyPI、HF mirror、GitHub：`4/4 HTTP 200`；
- AD-GS `model_60000`、42 test renders、138 train renders 与 processed 输入：`6/6`；
- DGGT preload：`5,411,266,466` bytes，SHA-256
  `fd15644b3a878849470cbf5f0f9eae39167cfec1b853092898ae754c4f3acde9`；
- DriveStudio commit `e59bda4fa681f829dbb1d65f0de582b0f633c450` 与 env 可用，pilot assets `missing`；
- 测试：
  `python -m pytest -q tests/test_dr_pseudo_tracks.py tests/test_v71_actor_registry.py` → `7 passed`；
- `shellcheck` 未安装，按计划未为此污染环境；`bash -n` 通过。

## 7. `DR-V2-M1-DGGT-REPAIR-01`

### 冻结实现

- repo `a3276d2b`；model revision `735ac9a6`；checkpoint bytes/SHA-256 通过并 hardlink 复用；
- `/root/autodl-tmp/envs/dggt-v2`：Python 3.10 / torch 2.4.1+cu121 / 固定 NVIDIA CUDA 12.1
  compiler+runtime+headers；
- pointops2 upstream `python setup.py install`；CUDA forward/backward=`PASS`；
- compatibility patch 仅 `args.difix -> args.diffusion`，untouched 错误单独保留。

### 正式运行

| 证据 | 终态 | 覆盖/结果 |
|---|---|---|
| `20260802T125138Z__native-nusc-s0-r6` | native done；后续 common import blocked | 1-view 18/18；3-view 18/18；原生输出不受 common 失败影响 |
| `20260802T133151Z__common-retry-s0-r8` | done | AD-GS common target 216/216；GT 像素身份 216/216 |
| `20260802T133912Z__regional-s0-r9` | done | AD-GS 216 + DGGT 1-view 72 + DGGT 3-view 216 = 504 rows |

M1 均值：

| 协议 | PSNR | SSIM | LPIPS(Alex) | inference s |
|---|---:|---:|---:|---:|
| DGGT 1-view | 20.707359 | 0.856031 | 0.135780 | 1.785527 |
| DGGT 3-view | 21.165262 | 0.771051 | 0.165553 | 4.517659 |
| AD-GS same-target 1-view | 34.581860 | 0.951918 | 0.062490 | n/a |
| AD-GS same-target 3-view | 34.894344 | 0.951711 | 0.061447 | n/a |

区域诊断的动态区/边界带 PSNR 分别为：AD-GS `29.640118/29.480968`、DGGT 1-view
`22.999911/22.017347`、DGGT 3-view `22.902139/21.810579`。边界带固定为二值动态区
7x7 dilation XOR erosion。这些数值仅用于 failure characterization，不得解释为同预算排行。

### 失败实例

`r1–r5`分别固定了 pip backtracking、CUDA 11.8/cu121 compiler mismatch、缺 cusparse headers、
transformers 5.x/DTensor 和 diffusers 0.39/torch schema 不兼容；r6 common 固定 `flow_vis`
缺失；r7 固定重试封装字段错误。全部为独立 `blocked` run，没有覆盖原运行。

## 8. `DR-V2-M2-ACTOR-EVAL-01`

### 正式运行

`/root/autodl-tmp/runs/dynamic_editing_v2/DR-V2-M2-ACTOR-EVAL-01/20260802T140312Z__actor-eval-s0-r5`
=`done`。

| scene | raw actors | eligible | high-support | boundary-support |
|---|---:|---:|---|---|
| scene-0230 | 58 | 16 | `af663976db5e...` | `18c7f0c5fa6b...` |
| scene-0242 | 53 | 20 | `40f087d8d9d7...` | `2c820a798ad9...` |
| scene-0255 | 56 | 6 | `f4aa30b8d0b4...` | `80c08b992f1d...` |

- 预注册 support score 和字典序 tie-break 未调节，冻结时尚无 M3/M4 编辑输出；
- 4,356/4,356 observations 使用 timestamp+exact `sample_token`；无效投影不从分母中静默删除；
- raw 2 Hz 与 interpolated visualization 字段物理分离，本运行插值列表为空；
- 11 个输入 metadata 哈希、167 actor metrics、3 份 cohort CSV、6 组投影 panel、6 张 raw
  轨迹图与视觉 QA 齐全。

### 失败实例

- r1：错误假设磁盘 `sample.json` 含 devkit 运行时 `anns` 反向索引；
- r2：`ijson` Decimal 进入 JSON 运行合同；
- r3：近平面后的 invalid projection 没有统一零面积 schema；
- r4：只选最近 timestamp sweep 导致 raw sample token 不精确，protocol QA 失败；
- r5：改为 exact token 内再做 timestamp 最近选择，不改 actor 门槛，通过。

## 9. `DR-V2-M3-EDIT-BASELINE-01`

### 正式训练与恢复

- 原生训练 run `20260802T152252Z__native-train-s0-r8` 完成 100/1,000-step profile 和 30k
  训练，checkpoint=`386,398,646 bytes / step 30000 / SHA-256 8ed40576...a73f9e`；
- 训练后上游 full render 把帧累计在内存中，`577/588` 时 cgroup memory 连续两次超过 90%，
  守卫返回 `-15`；峰值 GPU `23,873 MiB`，峰值 cgroup `89,836,462,080 bytes`，
  `oom=0 / oom_kill=0`；r8 保持 `blocked`；
- r12 对 checkpoint step/bytes/hash、r8 formal stage 和 blocked terminal 做窄范围复核，未复制或修改
  checkpoint；训练语义完成与上游 post-render 未完成分开记录。

### 正式编辑 smoke

完成 run：
`/root/autodl-tmp/runs/dynamic_editing_v2/DR-V2-M3-EDIT-BASELINE-01/20260802T163930Z__formal-checkpoint-recovery-s0-r12`。

- registry 24 个 model：23 non-empty，1 个被原生训练裁剪为空并显式不可用；
- 冻结 actor `af663976... → true id 13 → column 13 → model 5 → 2,683 Gaussians`；
- `3 frames × 3 cameras × 3 variants = 27` PNG；original/remove/lateral 均非空且时间同步；
- lateral/remove mean absolute RGB diff 分别为 `0.0003448362 / 0.0002196175`；两者均在 2 个
  frame-camera 上非零；这只是 effect smoke，不是质量指标；
- checkpoint SHA、非目标参数、reload 后完整 RigidNodes state 均精确不变；
- post peak GPU `8,241 MiB`，peak cgroup `58,291,757,056 bytes`；readiness 11/11 available。

### 独立失败实例

- r4/r6/r7：旧 `gsplat/nvdiffrast` CUDA binary 不含 RTX 3090 SM 8.6；固定源码重建后通过；
- r5：tmux 非登录环境没有裸 `python`；改为前缀解释器；
- r8：训练完成后的累积 full render 触发内存守卫；
- r9：恢复 probe provenance 字段错误嵌套；
- r10：registry helper 在 DriveStudio 环境误依赖未声明 `ijson`；改为读取 16 MB 标准 JSON；
- r11：一个非目标 model 的 Gaussian slice 被训练裁剪为空；registry v2 显式标记 unavailable，
  仍要求所选 actor slice 非空。

## 10. `DR-V2-M4-EDIT-PILOT-01`

### 正式运行

`/root/autodl-tmp/runs/dynamic_editing_v2/DR-V2-M4-EDIT-PILOT-01/20260802T171000Z__scene0230-pilot-s0-r7`
=`done`。

- 固定 scene-0230 high-support actor `af663976...`，196 帧、三相机、original/lateral +1m/delete
  共 `1,764` 张 RGB，所有配套 depth/opacity/dynamic/target mask/footprint 与 9 个 MP4 完整；
- paired metrics=`1,176` rows；16/16 协议、不变量和产物检查通过；
- lateral/delete non-target PSNR=`93.394483/95.598042`，LPIPS(Alex, 256px)=
  `5.260851e-09/3.052960e-09`，source effect energy=`0.055526/0.031926`；
- actor-local 位移最大误差 `3.814697e-06 m`，rotation/size/canonical drift 和 multi-camera
  world mismatch 均为 `0`；
- 自动检查不冒充质量门禁；人工抽检只确认非黑、非重复 original、footprint 和目标差分可见。

### 独立失败实例

- `smoke_frame1_s0_r1`：float32 transform 往返最大误差高于不现实的 `1e-6 m`，其余检查通过；
  r2 将协议容差固定为 `1e-4 m` 后 16/16 通过，正式实测误差为 `3.814697e-06 m`；
- `debug_controller_s0_r5`：外层诊断 `timeout` 中断 controller，但 child 使用
  `start_new_session=True`，故需按精确 PGID 回收；未覆盖；
- `20260802T170600Z__...-r6`：调试 tmux 生命周期中断后保留 running terminal 证据；
  正式 r7 改用 nohup controller，资源守卫和 terminal 均闭环。

## 11. `DR-V2-M5-STRESS-3SCENE-01` 部分执行后冻结

该任务保持 `pending`，因为没有满足 V2 预注册完成门禁。2026-08-05 路线切换后不再继续扩建其大型评测链，
但已生成资产和失败诊断作为 V3 A0/A3 的输入保留。

### 已有证据

- scene-0230 held-out：checkpoint `398,652,534` bytes，SHA-256
  `24a39f27dfeed36bbdb01ee14211aec51b414e6ab0e61915b71c1dddcdf61e49`；
  high/boundary actor 均可用，分别 `4,747/1,914` GS；
- scene-0242 held-out：checkpoint `306,034,934` bytes，SHA-256
  `16179d8f99becb86b6893a18ff036af72d78c9897f7aa2b0e297b735dd6c5fda`；high actor `6,939` GS，
  boundary actor 显式不可用；
- scene-0255：数据准备与 sky 产物已落盘；r8 的 90% cgroup memory stop 与其后的 cache recovery 分开保留；
- r25–r27 把训练失败定位到 DriveStudio 实例点聚合的 CUDA `torch.cat`；r27 输入 166 个 CUDA float32
  tensor，其中 152 个为空 `(0, 3)`，总计 177 scalars，terminal=`done` 表示诊断完成，不表示训练完成；
- 当前无 M5 控制器、tmux 或 GPU 进程。r16/r18 的 `running` terminal 是容器生命周期中断证据，不改写。

### 未完成

- scene-0255 原生完整 checkpoint 与 actor registry；
- 三场景 × 两 actor × 四编辑的 24 条有效序列；
- pseudo-hole、perception 与跨场景 final matrix；
- M5 单独实现提交和 V2 M6 novelty gate。

未提交的 M5 config/scripts/tests 属于保留工作树，V3 P0 文档提交不得 stage 它们。scene-0255 修复在 V3 A0
以新 task、新 run 和最小 compatibility patch 执行，不能倒写旧 M5 terminal。

## 12. 计划终态

`WS-V3-A1-CALIBRATION-01` 已 `done_off`，`WS-V3-A2-ACTOR-DENSIFY-01` 已
`done / tradeoff_non_dominated`，`WS-V3-A3-LOCAL-REFINE-01` 已 `done`：R1 因 frozen GPU resource gate
失败且 resource-invalid diagnostic 为 tradeoff 被 rejected，`A3*=R0/D2 exact alias`。A4-P0 v1 r1 已因
resolution contract blocked，v2 r2 已 13/13 audits passed 并登记 `done`。P5 r1 保留 runtime attribute blocked，
r2 已 14/14 audits passed 并登记 `done`。P1 canonical r1 已 21/21 audits passed；b05/b10/b20 分别失败
3/12/15 个冻结质量门，method rejected、selected=p1-source exact alias，P1 experiment=`done`。P2 r1 因 evidence
ledger 漏项保留 blocked，canonical r2 已 19/19 audits、31/31 safeguards 并选择 `p2-gs-param-fp16`，P2=`done`。
P3 protocol SHA=`dfaaba79...1b41` 已冻结且 validator/12 项协议测试/222 项联合回归通过。下一动作只实现并提交
`P3-chunk` runner；该冻结动作随后由 runner `aba5577` 和 canonical r1 完成：21/21 audits、57 RGB/31 endpoints、
85 tensor paths 与 source/registry immutable 全部 exact，selected=`p3-chunk-package`。package 比 source 大
`2.792171%` 且 load/reassembly 更慢，只登记 exact asset separation；A4=`done`。

F0 canonical audit 已 `done`，inference=`not_run_prerequisites_failed`，F1=`conditional_not_unlocked`。R0 canonical
已 `done`：63 inputs、23 decisions、12 deliverables、26 manifest files 与 P3 package 全 exact；A0→A4 主表、
Pareto、负结果/适用边界、复现 manifest 和最小离线可视化索引均已生成。V3.1 当前为 `none_plan_complete`。
P4、A3 formal/R2–R4、D3/D4 与新训练/推理仍未授权，除非未来以独立任务重新预注册。
V3.1 计划和本文件的 R0 收口快照见
[`archive/2026-08/worldsim-v3.1/`](archive/2026-08/worldsim-v3.1/README.md)；归档内容不构成新的执行入口。
