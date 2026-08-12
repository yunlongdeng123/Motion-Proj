# Research Status

## V4 当前状态（2026-08-12，M1 development freeze 后）

- 当前路线：`WorldSim V4 / EviDelta-GS paper-first`，分支
  `research/worldsim-v4-evidelta`，当前已登记实现提交 `06d56ee`。
- 最新有效完成任务：`WS-V4-M1-EVIDENCE-FIELD-01` 的
  `6-scene development freeze`；当前任务为同一任务的 6-scene nuScenes validation
  只读确认。KITTI 继续等待用户自行复制，禁止下载、禁止质量运行。
- B0 已在六个 development scenes 完整收口并冻结；最终只读审计：
  `/root/autodl-tmp/runs/worldsim_v4/WS-V4-B0-MATCHED-BASELINES-01/20260812T112848Z__b0-final-audit-s0-r117`。
- M1 两场景 smoke canonical：
  `/root/autodl-tmp/runs/worldsim_v4/WS-V4-M1-EVIDENCE-FIELD-01/20260812T115302Z__m1-smoke2-risk-s0-r121`；
  选择 `raw__risk_100 + raw calibration + threshold=0.5`，gate=`pass`。
- M1 六场景 development canonical：
  `/root/autodl-tmp/runs/worldsim_v4/WS-V4-M1-EVIDENCE-FIELD-01/20260812T121734Z__m1-development6-s0-r124`；
  全六场保留 denominator，`scene-0994/0139` 可评，`scene-0230/0242/0255/0048`
  显式 abstain，Boundary F1 scene mean delta=`+0.1255247811`、FN semantic mass
  delta=`+0.0054849633`、Brier delta=`-0.0115803990`、ECE delta=`-0.0311158595`，
  base RGB/checkpoint exact，heldout/test quality 未读。
- M1 development freeze 审计：
  `/root/autodl-tmp/runs/worldsim_v4/WS-V4-M1-EVIDENCE-FIELD-01/20260812T122636Z__m1-development-freeze-audit-s0-r126`；
  frozen selection=`risk_100/raw/0.5/temporal_retention_0.75`，后续 validation 禁止重搜、拟合或改阈值。
- validation 数据准备由 `06d56ee` 冻结六场景配置；r127 因错误 Python 环境缺 `ijson` 失败，
  r128 因 SSH 超时断管触发 `BrokenPipeError`，两者均保留。detached 重试
  `/root/autodl-tmp/runs/worldsim_v4/WS-V4-M1-EVIDENCE-FIELD-01/20260812T134400Z__m1-validation-data-extract6-s0-r130`，
  已完成本地 10 个官方 nuScenes blobs shard 扫描并精确绑定 `10,647` 个成员，
  `status/summary/manifest/inventory SHA=fac0a587...3476e / 6f0b0933...76854c /
  eb6eeb9a...76d4 / 41b2d5eb...ee5c`，`no_download=true`、test quality 未读。
- 下一步：逐场景 preprocess → StreetGS profile/30k → V3.3 evidence chain →
  使用 frozen M1 参数做 validation confirmation；validation 完成前不读取 18-scene test quality。

## V4 启动期历史快照（以下进度行已由上方当前状态取代）

- 更新时间：2026-08-12
- 当前路线：WorldSim V4 / EviDelta-GS paper-first 扩展
- 最新有效完成任务：`WS-V4-D0-NUSCENES-COHORT-01`
- 当前任务：`WS-V4-B0-MATCHED-BASELINES-01`
- 路线状态：`active / d0_done / d1_blocked_external / b0_streetgs_3of6_v33_adgs_1of6`
- 当前门禁：继续补齐 V3.3/StreetGS/AD-GS 6-development-scene strict matched baseline；StreetGS 旧 stride=10 六场景只作 provenance，D1 因公共 KITTI 缺失保持 blocked，M1 与 test quality 尚未授权
- 当前计划：[`WORLDSIM_V4_EVIDELTA_GS_PLAN.md`](WORLDSIM_V4_EVIDELTA_GS_PLAN.md)
- V4 分支：`research/worldsim-v4-evidelta`
- V4 起始 HEAD：`main@21084309480895f5541196a06191a5dffb4e30c1`
- V4 P0：[`WS_V4_P0_SCOPE.md`](WS_V4_P0_SCOPE.md)
- V4 P0 canonical：`20260811T080636Z__p0-scope-formal-s0-r2`
- V4 P0 config/summary/manifest/status SHA-256：
  `248bde621343597196c1a608ce8674a0c4a1f974d38abc70710c7783d8ecaaa8` /
  `aba1fbcffbe89e7b992bb1d0c691f398423c143319628b52cff7f7f3d0b51283` /
  `ec32e983ad48e6ed415906562c90338844bfc47ec305afa950bb4a99f1543970` /
  `b39416015b1d6275dd3b8bfefa74c7aa45d4ceee790fdeab4d72b5e3baca272a`
- V4 D0：[`WS_V4_D0_NUSCENES_COHORT.md`](WS_V4_D0_NUSCENES_COHORT.md)
- V4 D0 canonical：`20260811T084108Z__d0-cohort-formal-s40117-r4`
- V4 D0 config/summary/manifest/status/cohort SHA-256：
  `ed47c0da2c76e14b3b0a0e7a8b4d9b580bdf37e4c69a1d5a389b965e88c667a1` /
  `ec96970d8733e99b206048baf463fce69cae99db916c15b1e4fd777a74d4f276` /
  `3349a63667988c61596494506c67cf4d3b7f36e934ab4fac5d0935974c0d6b30` /
  `1dfd5db4e71566c344aa382e9f8e464c0b512cb01ff8a6053a03123bd3cb4461` /
  `eda9f6847d2d9d01ce813c06f550aa2a0f5cf9a23ee8ab3ba766911acb144578`
- V4 一手文献矩阵：[`WS_V4_LITERATURE_MATRIX.md`](WS_V4_LITERATURE_MATRIX.md)
- KITTI P0 审计：[`WS_V4_KITTI_AUDIT.md`](WS_V4_KITTI_AUDIT.md)，当前 `blocked_local_dataset_missing`
- KITTI D1 审计：[`KITTI_LAYOUT_AUDIT.md`](KITTI_LAYOUT_AUDIT.md)，canonical blocked run=
  `20260811T085210Z__d1-kitti-layout-formal-s0-r2`
- V4 B0 盘点：[`WS_V4_B0_BASELINE_AUDIT.md`](WS_V4_B0_BASELINE_AUDIT.md)；6-scene DriveStudio 输入与 sky masks
  已齐；当前 strict executable coverage=`V3.3 1/6 / StreetGS 6/6 / AD-GS 1/6`，B0 仍为 running
- B0 StreetGS profile：`20260811T111810Z__streetgs-scene0048-profile100-s0-r16`，100 steps done，checkpoint
  SHA=`446297b8...3af`，peak GPU=`9,004 MiB`，30k formal 已解锁；test quality 未读
- B0 StreetGS 协议纠错：r17/r20/r22/r24/r26/r28 虽均为 30k finite 且 OOM/kill=`0/0`，但使用
  `test_image_stride=10`，不满足冻结的 `sample_index mod 5` 三分区合同，全部降为 protocol-mismatch provenance；
- B0 StreetGS strict canonical：scene-0230 r32 30k done，checkpoint=`386,410,166 bytes /
  766648bf...af97cd1`，peak GPU=`23,892 MiB`，OOM/kill=`0/0`，test quality 未读；corrected inventory
  r33=`StreetGS/V3.3/AD-GS 1/1/0`，fingerprint=`c19fba13...e285853`；
- B0 StreetGS strict scene-0242：r46 30k done，wall=`1,998.0482 s`，checkpoint=`302,953,462 bytes /
  dd41a34d...52bc0`，Background/Rigid=`824,583/92,170`，peak GPU/cgroup=`17,530 MiB /
  23,842,824,192 bytes`，OOM/kill=`0/0`，无 test/full render；r47 inventory=`StreetGS/V3.3/AD-GS 2/1/1`，
  inventory/fingerprint=`89c72659...eafad / b91f7c76...712d6`；
- B0 StreetGS strict scene-0255：r48 30k done，wall=`2,392.0649 s`，checkpoint=`444,340,086 bytes /
  dba24982...cb2d2`，Background/Rigid=`1,478,401/38,721`，peak GPU/cgroup=`23,932 MiB /
  24,132,476,928 bytes`，OOM/kill=`0/0`，无 test/full render；r49 inventory=`StreetGS/V3.3/AD-GS 3/1/1`，
  inventory/fingerprint=`79b6b1d0...c86c85f / bd822e61...9641a`；
- B0 StreetGS strict scene-0048/0994/0139：r50/r52/r54 均 30k done；checkpoint bytes=
  `332,725,750 / 279,185,462 / 314,307,830`，SHA=`70d02a0b...b00d2 / 3e2b2534...3aea /
  4fff4452...8dfe`；Background/Rigid=`1,030,993/15,717 / 819,952/932 / 962,074/7,219`，峰值 GPU=
  `23,694/20,970/23,056 MiB`，三场 OOM/kill/max/high 均为 `0`，无 test/full render；
- B0 StreetGS 六场登记提交=`a4ee23a`；clean r55 inventory=`StreetGS/V3.3/AD-GS 6/1/1`，
  inventory/fingerprint=`8bc62596...be3a1 / 4f12c1d2...32372`；StreetGS 已闭环，但 V3.3/AD-GS 各缺五场且统一评测未执行，
  因而 B0/M1 门禁保持不变；
- B0 AD-GS 恢复：official `9a208512` + exact DPT/CoTracker weights 已审计；离线环境 r34 done，
  r42 又从 clean PyTorch3D v0.7.5 source 离线重编 sm86 并通过真实 KNN kernel smoke；scene-0230
  train-only preprocess r38 done，`image/semantic/sky/depth/flow=354/354/354/354/285`，峰值 GPU=`20,112 MiB`、
  峰值 cgroup=`22,384,893,952 bytes`、OOM/kill=`0/0`；profile100 r43 done，peak GPU=`6,012 MiB`；
  formal60k r44 done，stage=`7,054.6221 s`，peak GPU/cgroup=`16,692 MiB / 33,680,572,416 bytes`，三文件 SHA=
  `f17ed27f...a0cbb / c725f952...c84b0 / c3233b71...e4d34`，development/heldout/test quality 均未读；
  r45 内容寻址 inventory=`V3.3/StreetGS/AD-GS 1/1/1`，inventory/fingerprint=`4bf7cf68...ad6b / 3db524d2...49e5`
- B0 统一评测：PSNR/SSIM/LPIPS-Alex + global/static/actor/boundary/edit_roi；scene bootstrap/paired tests 与
  engineering timing/yield/recovery 派生已实现；baseline/AD-GS/region/evaluator 联合定向单测=`50 passed`
- V3.3 终态：`v33_supported`，全部 canonical 资产只读
- V3.3 历史计划：[`DYNAMIC_DRIVING_WORLDSIM_MODEL_PLAN_V3_3.md`](DYNAMIC_DRIVING_WORLDSIM_MODEL_PLAN_V3_3.md)
- P0 审计：[`WS_V33_P0_SOTA_AUDIT.md`](WS_V33_P0_SOTA_AUDIT.md)
- S1 对象场：[`WS_V33_S1_OBJECT_AWARE_GS.md`](WS_V33_S1_OBJECT_AWARE_GS.md)
- S2 道路修复：[`WS_V33_S2_ROADPATCH_INPAINT.md`](WS_V33_S2_ROADPATCH_INPAINT.md)
- S3 Actor 视图选择：[`WS_V33_S3_ASSET_VIEW_SELECTION.md`](WS_V33_S3_ASSET_VIEW_SELECTION.md)
- S4 Spatial Delta：[`WS_V33_S4_SPATIAL_DELTA.md`](WS_V33_S4_SPATIAL_DELTA.md)
- S5 语义门控渲染：[`WS_V33_S5_SEMANTIC_RENDER.md`](WS_V33_S5_SEMANTIC_RENDER.md)
- R0 完整集成：[`WS_V33_R0_INTEGRATION.md`](WS_V33_R0_INTEGRATION.md)
- V3.2 终局归档：[`archive/2026-08/worldsim-v3.2/`](archive/2026-08/worldsim-v3.2/README.md)
- V3.1 终局归档：[`archive/2026-08/worldsim-v3.1/`](archive/2026-08/worldsim-v3.1/README.md)
- V3 启动 Git 基线：`research/dynamic-editing-v2@e691c1f`
- V3.3 历史分支：`research/worldsim-v3.3-object-maintenance`
- V3.3 P0 canonical run：`20260810T171744Z__p0-source-audit-s0-r2`
- V3.3 P0 config/summary/manifest/status SHA-256：
  `29c167fe050d074f626884c0eba7b67fd6fd56c8493adc4c6be0d390f09b9ae2` /
  `08806b5f197d524207aa5d527b9976a993042b6451fa0cc9b0458a20b3a1d68a` /
  `2603ff0e037931aef8f8c84606038bd748600c99cef1a2a29cc82c621c51a12d` /
  `91096d0eae7616f2c68d133796922b49d475220c0c18fb7c438ca3655a32072d`
- V3.3 S1 canonical formal run：`20260810T183154Z__s1-instance-field-formal-s0-r9`
- V3.3 S1 config/summary/manifest/status SHA-256：
  `9afa48aa1ff5ebbb290da564e901f25c48ac1f6ee16f97379f936457acdc3150` /
  `4ab311a64437202ecdd5fa915c4bd528543cdc6040a12df54a5183a39bdf8c4a` /
  `e1b858fd505c65e41dc3272137e355ad36b4e45bc18b350c3140bcbde1ef584e` /
  `9394d15e935285955812e9a4502ffa0f4029ca4c3be9c535b249ecc93e7303b9`
- V3.3 S2 canonical index / RoadPatch / Inpaint360GS preflight：
  `20260810T193004Z__s2-patch-index-formal-s0-r10` /
  `20260810T193140Z__s2-roadpatch-formal-s0-r11` /
  `20260810T193426Z__s2-inpaint360gs-preflight-s0-r12`
- V3.3 S2 index/manifest/RoadPatch delta/acceptance/preflight SHA-256：
  `51561eecf66ac20f38d139abd9738c970cefe686f40ba9ae787ea62be74a1a4c` /
  `565741c5b92c60a4a75552b71ff6c24758db605425618adefd0d0209f42d8845` /
  `a31053137e37bb36eb7f59d0250d525a9ebe274caf2903f5dd92a47063289014` /
  `9be398450e34a5b5a4f43dcfccd562b42439a4735a7efc9faaf97b59afa43cd0` /
  `91b5c6a04cefc6086e4695584f57c0497bc9985ba36e874336b85cc4a11a830b`
- V3.3 S3 high selector / AH / import / development / heldout canonical runs：
  `20260810T201345Z__s3-viewselect-high-formal-s0-r2` /
  `20260810T201830Z__s3-asset-high-formal-s0-r3` /
  `20260810T202210Z__s3-import-high-formal-s0-r4` /
  `20260810T205300Z__s3-eval-high-development-formal-s0-r13` /
  `20260810T205600Z__s3-eval-high-heldout-formal-s0-r14`
- V3.3 S3 high selection/input/inference/A4 asset/dev decision/heldout decision SHA-256：
  `192e5035ad9697f70a14c47ecdcdc3bc37c3cc1435633e83e06649aac53b7be9` /
  `34b1e09e6f8e7fbd8ed47a64e3140aa3b1158a66efea3c3784da0693dfdef2e7` /
  `e33fcc650848de868c76ea7f0d54b4c73e51df42bc90b562481ed3606a7f2d90` /
  `06d5db8599624f2f067c4065f53aad1828ca42c946becfb037a9e24c3cf7ec13` /
  `28d4f75c8778e179b13a235a574e868be60d5539db7f3399d31d426dcd0d82bf` /
  `795ecbc5852c4cfeb2df9e18d803a14c6e79a5845c5053aaaceb127ce83d8032`
- V3.3 S4 canonical package / real-render evaluation：
  `20260810T221300Z__s4-package-canonical-s0-r7` /
  `20260810T221700Z__s4-eval-canonical-s0-r8`
- V3.3 S4 config/package manifest/package summary/eval summary/decision SHA-256：
  `4b318a67786e576d56b6ea57d91528252fa290f0a53bd3a2f5d45dbae1c3508a` /
  `3be8ce88764b8261740ced82a460e0109f2ce04a29c1c343c9d97ca3152bee43` /
  `cbde96004e81a6f1f0e37b7ccdd095fed364482a9754d0704052973caeda0c63` /
  `6f143040177cc251317328e8574ad12047803c710289159ca6eaaf5ca3c79085` /
  `19e3aba6d65479701d7eef296730d974a3032c6dec13c2368ddc325547c30db9`
- V3.3 S5 canonical run：`20260810T220500Z__s5-semantic-gate-canonical-s0-r4`
- V3.3 S5 config/input/Harmonizer/SAM2/summary/status/decision SHA-256：
  `b3848289add5e0f401d7386abf3e72caed80d3fa126b63a34694787463b18c89` /
  `939a829eac74014ff913eb8d02058ef83166a576c8b93e89d5b7689bd58a635c` /
  `1da253d85e98babc1a8b33187f48cfe4b1a7a6c712cacc5cc25886e836913863` /
  `c03fe7c9c4c25d56fc256d9c3328ecc70453b2daef05f4a61f2ed76da3c58b19` /
  `1e0bfb59602a012c799c94d2c18e9e0a35bfa09ecc3c05adbce2e22c37160761` /
  `969bb00995b592889803b9b8a147096ddde61037c250e4608d609d05cbe6fb97` /
  `988b6647a0d2a17a58d82b53b0c54c5e9854ba37a9ec8c4511f4d2b2cde6159d`
- V3.3 R0 canonical run：`20260810T222701Z__r0-integration-canonical-s0-r7`
- V3.3 R0 config/summary/status/release/content-manifest SHA-256：
  `4b4a20b95c2cd9803d2087128dca4942344e7e0a6ac1669b71e108c0e11273a9` /
  `c19032559796377d28073ce14584ce086a0d6ec8b20c598069fe15ae391ca2b2` /
  `0a1396f45a063df6ae60bc8ba56378d89df20651a4074c157a5babbc18f09aa4` /
  `cffaad16e2d14e8274c41bb48b24be64c73d9fb6f41d1fe4792934adeab244a7` /
  `e386c14b6b29c74bd1316a31a3abefedf10a74530cfe3149cf9e040eb78a6c53`
- V3.1 F0 审计协议 SHA-256：`2004a0294cc4adb9750dd3bc78aac0b650c99338f761697c14afd8e71a6fd611`
- V3.1 R0 集成协议/runner SHA-256：`7011d99f70fc59835569c43bd7e750a5e1981ea67843ef08873bfe4707deb624` /
  `deb1a82f8d60eb659acf1237482ffff26a6d47d615c3eeb50df75d18f0c3c97c`
- V3.1 R0 canonical summary/manifest/status SHA-256：`40624cbc79a004e9e07e57b00cebc535b900297a10f0d070fb4e9305a5f7937a` /
  `358d9fc7fde6a535c2ffb0bb2ff34cf1f9df3c151066f3051e24859a5d73a27e` /
  `d31a4f8e62f31dbbf6bbf2520243f5061c68e6682ea5011ef8c64a8dbb541617`

## V3.3 R0 canonical 收口

- R0 对 44 个 canonical inputs 做 path/bytes/SHA 与嵌套 terminal/decision 枚举 exact 检查；再次验证
  O1=`1,309,868` Gaussian、RoadPatch=`104` rows、A4=`99,241` rows 的正式 schema；
- selected chain 固定为 `D2 immutable base→O1→B1→A4→posterior-gated spatial delta→S5 G0→
  V3.2 persistent storage reference→V3.3 exact release`；四个必须成功标准 `4/4`，overall=`v33_supported`；
- R0 对问题 2 明确 `not_directly_ranked`：B1 通过冻结 heldout 且成为 V3.3 主方法，但不声称在非 matched
  协议下优于 V3.2 Telea；Inpaint360GS/SAM3.1/R3D2 的 blocked 也不写成质量失败；
- release 物化 O1 field、完整 RoadPatch delta、A4 asset、14-file S4 delta package、S5 5×2 production
  PNG、V3.2 chunk manifest、39 JSON evidence 与五类 ledger；`76 files / 18,432,994 bytes`，full checkpoint copy=`0`；
- archive 在 diagnostic/formal 与同 run 双构建均 SHA exact=`cffaad16...44a7`；directory/archive replay 的
  content manifest SHA=`e386c14b...a6c53`，standalone verifier 两种模式均 passed；
- R0 wall=`2.721847 s`、GPU compute max=`0`、cgroup peak（含既有 page cache）=`39,614,062,592 bytes`、
  run=`50,851,476 bytes`、OOM/kill=`0/0`；S1–S5 selected wall 累计=`379.552 s`、peak=`20,137 MiB`；
- 前五个 diagnostic 分别修复 exact list/enum、无冗余 schema 假设和 tools directory 创建，terminal 均 failed；
  r6 冻结 archive，formal r7 以 expected SHA 通过；
- R0 专项=`6 passed`、V3.3/V3.2 定向回归=`86 passed`、7 个 source snapshots 与提交候选 exact；
  当前无下一执行授权，F0 LiDAR-EVS 保持 conditional。

## V3.3 S5 收口与 R0 授权

- 实现 semantic gate 核心、五视图输入冻结、Harmonizer/SAM2 分环境串行 runner、development→heldout
  finalizer 与新 run 不覆盖 launcher；删除 production 固定为 raw 3D render exact copy；
- gate 只覆盖 actor boundary、ground contact、shadow/seam support，residual cap=`12/255`，far weight=`0`；
  五视图 far changed pixels=`0`、actor interior L1 delta=`0`；
- development 三视图的 boundary/contact L1 delta=`-1.837229/-2.771866`，故预注册选择 G1；只有此后才读取
  heldout；f060/c1 contact delta=`+0.422686>+0.25`，G1 被确认门拒绝，生产回退 G0；
- unconstrained delete 在 edit target 的 SAM2 mass/fraction delta=`+0.126399/+0.133885` 并被标记；production
  delete 5/5 pixel SHA exact、semantic mass/fraction delta=`0/0`，安全门 5/5；
- R3D2 source/commit/tree/license exact，但没有作者 exported pretrained model；状态固定
  `blocked_pretrained_model_unavailable`，model loaded/training 均为 false；
- canonical r4 的 Harmonizer/SAM2 wall=`30.180697/5.701133 s`、peak NVIDIA sampled=`3,553/2,399 MiB`、
  peak torch reserved=`3,940/2,070 MiB`、
  run bytes=`34,548,858`、OOM/kill=`0/0`；r2–r4 的 30 个 RGB SHA 跨三次 run exact、decision SHA exact；
- r1 因 SAM2 环境无 SciPy 而 failed，修复为形态学依赖延迟导入，不安装新包；r3 消除 r2 的 NumPy warning，
  r4 清理 input-prep EOF 空白并使 source snapshot 与提交态 exact；S5 专项=`8 passed`，V3.3/V3.2 定向回归=`80 passed`；
- 当前唯一 next action 是 `WS-V33-R0-INTEGRATION-01`；R0 必须登记 G1 负结果、G0 production、R3D2 外部
  阻塞与 temporal not-evaluated，不得把可回退工程合同写成增强泛化成功。

## V3.3 S4 收口与 S5 授权

- S4 将 D2 checkpoint/registry 表达为 external exact reference；package 不复制 `.pth/.ckpt`，总量
  `4,007,120 bytes`、最大文件 `3,942,422 bytes`，完整 checkpoint copy=`0`；
- composition 固定为 `base→ERASE→INSERT_BACKGROUND→INSERT_ACTOR→RENDER_ONLY`；ERASE 只创建临时
  opacity Parameter，`sigmoid` 精确为零，不删除 base 行；INSERT 逐行保留 patch reuse / generated actor provenance；
- S2 的 104-row combined delta 按 `target_role=high_support` 只取 25 行，不把 boundary 的 79 行混入当前 edit；
  S3 high A4 以 99,241-row actor-local delta 装载到 rigid index 5，原 point-id prefix 不变；
- r2 把 S1 的 36,736 个 Background hard assignments 全部擦除，目标外 L1=`0.821965>0.5`，按冻结门
  rejected；没有放宽门。最终用 S1 已学得 instance opacity 的 MAP 边界 `p>=0.5`，保留
  `1,614 Background + 4,525 Rigid core` erase rows；
- canonical r8 的 edit target f091/c1：erase/background/actor/full effect pixels=
  `27,000/6,663/14,844/28,218`，erase/actor mask coverage=`0.999741/0.849298`，目标外 L1=`0.225349`；
- 五视角 aggregate erase/background/actor/full effect=`51,218/14,147/28,688/54,519`；20/20 逐栈
  rollback render SHA exact，full stack 二次重放 SHA exact，额外 replay rollback exact；
- checkpoint/registry before-after SHA exact，base row deletion/nonzero erased opacity/duplicate insert index 均为 `0`；
  wall=`66.181 s`、peak CUDA reserved=`8,132 MiB`、run bytes=`11,744,674`、OOM/kill=`0/0`；
- S4 专项=`9 passed`，V3.3/V3.2 定向回归=`72 passed`；真实 GPU source snapshots 与最终
  核心/config/builder/evaluator byte-exact；r7/r8 为 canonical；
- 当前唯一 next action 是 `WS-V33-S5-SEMANTIC-RENDER-01`；通用 unconstrained Harmonizer 继续禁止，
  R3D2 仍受 pretrained 缺失约束，R0 未授权。

## V3.3 S3 收口与 S4 授权

- selector 只枚举 train frames，并排除 19 heldout + 10 reserved development frames；每个候选使用真实 D2
  original/delete counterfactual effect，保存 area/mask/sharpness/visibility/occlusion/truncation/yaw 全量证据；
- high diagnostic/formal selection/input SHA byte-exact；r2 candidates/eligible=`130/119`，A1=`91L`、
  A2=`0F+91L`、A4=`11F+83L+89L+94L`，heldout/development read 均为 false；
- 官方 Asset Harvester source=`767b243` clean，三套权重与 VAE/C-RADIO revision exact，HF offline；r3
  wall=`160.189 s`、peak=`20,137 MiB`、OOM/high=`0`；三份 PLY 均非空；
- importer r4 的 A4=`99,241 Gaussian / 3,791,327 bytes / 06d5db85...ec13`，deterministic
  reserialization 与 reload exact，enriched manifest=`4590c1bd...7343`；
- high development r13 的 A0/A1/A2/A4 IoU=`0.669876/0.658477/0.664463/0.701490`、boundary F1=
  `0.517563/0.497629/0.544166/0.604799`；三条 auto arm retention gates 全过，冻结选择 A4；
- high heldout r14 只比较 A0/A4；A4 相对 A0 的 IoU/boundary F1=`+0.023490/+0.059889`、
  PSNR/LPIPS=`-0.015760 dB/+0.008527`，四项 gate 全过；checkpoint exact、无 heldout 优化；
- boundary formal selector r8/AH r9/import r11 执行成功，A4=`94,835 Gaussian / 9b2295e5...5dd1f`；
  r12 使用 immutable D2 native actor 作诚实基线，不复用错误角色的 manual A0；
- boundary A4 相对 native 的 IoU/boundary F1=`0.624832/0.492141` vs `0.666562/0.555343`，LPIPS/PSNR
  也失败，故 `ABSTAIN_GENERATED_OVERRIDE`；没有读取 boundary heldout；
- scene-0242/0255 没有本协议冻结的 V3.3 S1/S2 输入链，且 boundary transfer 已拒绝，不混用旧 V3 资产；
- S3 专项=`11 passed`，V3.3/V3.2 定向回归=`63 passed`，py_compile、diff check 与最终 source snapshots exact；
- S3 production 输出只包含 high-support A4；generated backside 仍只作 completeness/consistency claim；
- 当前唯一 next action 是 `WS-V33-S4-SPATIAL-DELTA-01`；S5/R0 仍未授权。

## V3.3 S2 收口与 S3 授权

- RoadPatch-Lite 明确标记为 `GS-RoadPatching-inspired`；没有把仅含项目页静态文件的上游仓库写成官方复现；
- DriveStudio 首个 CAM_FRONT 使用 OpenCV `x-right/y-down/z-forward`，道路 BEV 为 `(x,z)`；V3.1 P3 的 `(x,y)`
  网格和 V3.2 P2 FP16 checkpoint 仅为历史 package schema，不再误用为原生 donor truth；
- canonical index r10 从 D2 FP32 Background 的 `1,205,164` 个 native rows 建立 1/2/4 m 静态索引；先做
  row-level actor/generated/scale/support fail-closed，再取 `<=0.75 m` densest vertical slab，避免一个天空/立面点污染整格；
- index 共 `15,591` patches，`822` valid（1/2/4 m=`617/160/45`），eligible native rows=`702,506`，
  generated donor=`0`；index=`4,146,483 bytes / 51561eec...a4c`；
- S1∩SAM2 delete mask、target-view first-hit depth 与 cross-view support 形成两个真实 4 m hole anchors；两者的
  5 个候选均满足冻结的几何/可见性/分离门禁；
- development-only r8 的 `2,150`-row dense delta 造成 heldout PSNR/SSIM=`-0.8553 dB/-0.00619`，保持
  rejected；新增 `maximum_rows_per_target=512` 作为候选资格门，不事后修改 r8；
- canonical RoadPatch r11 自动选择 `25 + 79 = 104` 个 native donor rows，delta=`24,557 bytes / a3105313...9014`；
  authoring state 是 immutable D2 base + deterministic delta，不另造完整 checkpoint；
- heldout B0→B1：PSNR `28.157155→28.073124`（`-0.084031 dB`）、SSIM
  `0.871450→0.870542`、LPIPS `0.149666→0.151527`；static PSNR `+0.002865 dB`，static LiDAR MAE
  `0.895636→0.890384 m`，全部通过冻结门；checkpoint before/after SHA exact；
- r11 wall=`69.335 s`，peak CUDA allocated/reserved=`8,337,670,144 / 8,420,065,280 bytes`；
- 官方 Inpaint360GS r12 固定 source=`d54c893`、Apache-2.0；其 RTX 4090/CUDA 11.8 双环境与
  CropFormer/LaMa/SAM/DeAOT/GroundingDINO 权重、StreetGS adapter 在当前 3090 主机均不齐，故
  `blocked_single_3090`，`official_execution_attempted=false`，不伪造 B2 质量结果；
- V3.3/V3.2 定向回归=`52 passed`，RoadPatch 专项=`6 passed`，py_compile 与 8 个 canonical source
  snapshot byte-exact 均通过；
- S2 canonical 以 B1 RoadPatch 收口，当前唯一 next action 为 `WS-V33-S3-ASSET-VIEWSELECT-01`；
  S4–S5/R0 仍未授权。

## V3.3 S1 收口与 S2 授权

- development smoke canonical r1 在未读取 heldout 的前提下比较 O0/O1/O3，冻结选择
  `O1_dual_opacity`；O3 的宽 ambiguous reassignment 未入选；
- heldout target canonical r4 固定 19 帧、31 个可见 block、37/37 accepted SAM2.1 masks，
  `optimization_forbidden=true`；source commit=`2b90b9f`、checkpoint SHA=`2647878d...318`；
- SAM2 隔离环境恢复为 Python `3.10.20`、torch `2.5.1+cu124`、torchvision `0.20.1+cu124`，
  conda explicit/pip freeze SHA=`c9294494...713 / aded7fb5...d69`；未修改 DriveStudio 环境；
- canonical formal r9 只运行 `O0 + frozen O1`，O1 相对 O0 的 heldout boundary F1=
  `0.068960→0.336158`、IoU=`0.063253→0.330727`、normalized boundary distance=
  `0.144958→0.105280`、false-positive semantic mass=`0.900308→0.623276`；
- false-negative semantic mass=`0.061278→0.109356`，因此不宣称全面支配；identity presence 两 arm 同为
  `0.972973`，全局共享 instance logit 的参数稳定性为 `1.0`；
- selected field=`5,882,296 bytes`、SHA=`23b2403ccb47e2e2c6b5fa3d22a9a6d93815d9f9bcbc6d11b66f035831adc8d7`；
  D2 checkpoint before/after SHA=`1a061247...e7c` exact；peak CUDA reserved=`8,084,520,960 bytes`；
- instance-field writer 固定 entry 排序、ZIP timestamp、权限与压缩参数；同一数组二次写入的 SHA exact。r7→r9
  的 O0 数组 exact，O1 CUDA 优化存在最大 `0.001357` logit / `8.918e-05` opacity 浮点漂移，但 heldout aggregate exact；
- V3.3 P0+S1 与 V3.2 定向回归=`51 passed`，py_compile、bash syntax、diff check 均通过；
- 当前唯一 next action 是 `WS-V33-S2-ROADPATCH-INPAINT-01`；S3–S5/R0 仍未授权。

## V3.3 P0 收口与 S1 授权

- 新分支从 `a055fc6727dddacd194665d5c997a1fe47c2d2f4` 建立；V2/M5 dirty files 原样保留且未纳入 V3.3；
- canonical P0 run=
  `/root/autodl-tmp/runs/worldsim_v33/WS-V33-P0-ROUTE-SOTA-AUDIT-01/20260810T171744Z__p0-source-audit-s0-r2`，
  terminal=`done`；10 个 source 裁决为 `2 executable / 2 weights_blocked / 5 source_not_released / 1 audit_only`；
- SAM3.1 官方 source 固定为 `96914d2`，但当前 HF 未登录且无 cached checkpoint，故 `weights_blocked`；S1 exact fallback 到 V3.2 SAM2.1 canonical masks；
- OP2GS、GS-RoadPatching、3D-GIMP、FocusGS、LiDAR-EVS 没有可执行官方 source 时只允许 inspired/audit-only；
  GS-RoadPatching 官方仓库 `468f812` 当前只有项目页静态文件、无算法源码和根 LICENSE；
- Inpaint360GS `d54c893`/Apache-2.0 可进入 S2 独立 adapter/preflight；R3D2 `3fc6e31` 虽有 Apache-2.0 代码，
  但没有作者导出的 R3D2 model，保持 `weights_blocked` 且禁止从零训练；GOR-IS 只作非商业研究 audit；
- 五个 V3.2 canonical 大资产重新计算 SHA 全 exact，R0 terminal 仍 `done`、8/8 gates；V3.2 定向回归=`36 passed`；
- P0 新增 source auditor 回归=`4 passed`；P0 未训练、未运行模型推理、未安装依赖、未下载大型权重、未修改 DriveStudio；
- P0 关闭时唯一解锁的是 `WS-V33-S1-OBJECT-AWARE-GS-01`；该任务现已由上节 canonical r9 收口。

## V3.2 终局处置

- S0–S4 与 R0 的全部单卡 RTX 3090 可执行工作已终结；S1/S2/S3/R0 形成 production candidate；
- 最终链固定为 S1 extended semantic sidecars + S2 generated-background mixed scene + S3 generated-actor
  override + R0 exact chunk package；
- S4 non-temporal 因删除语义重生成被排除，仅保留 optional diagnostic；S4 temporal 受 gated Cosmos base
  权重阻塞，S5 受许可证门阻塞；
- 外部条件未来变化时，也必须先建立新 task ID、冻结新 protocol 并创建新 run；不得续写 V3.2 terminal；
- `next_action=none_plan_complete`，当前无训练、评测、下载或第三方接入授权。

## V3.2 S0–S3 收口

- project baseline=`d91e80eea33a1bf8b6596d2357ee0ccf357691cc`；
- V3.1 D2 checkpoint SHA-256=`1a061247a753c0d8c9aa7835a52efa2ab1ddc79141a6168adc18b9748de66e7c`，
  A3*=R0-off，P2/P3 与 V3.1 terminal 全部保持只读；
- 11 个公开官方 source 已固定 commit，并验证本地 checkout 与审计时 upstream HEAD exact；
- MV-SAM 未找到可固定官方代码/checkpoint；VISTA、Omni-3DEdit、CoIn 因无明确根许可证保持执行阻塞；
- SAM2.1 Hiera Large revision=`665f8e2ad61cf5f53d65644ff27c8ee525124610`，checkpoint
  bytes=`898,083,611`、SHA-256=`2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318`；
- 当前为单卡 RTX 3090 24 GiB；cgroup memory max=`96,636,764,160 bytes`，S1 GPU 推理与 lift 已完成且
  `memory.events oom/oom_kill=0/0`；
- r5 复核发现 high-support 配置把 token `af663…` / rigid index `5` 错配到 dataset instance ID `5`；
  数据事实中该 token 的 ID 是 `13`，ID `5` 属于 token `bf9a…`，故 r5 不再是 canonical 证据；
- 新增 dataset ID ↔ instance token ↔ rigid index fail-closed 合同；旧 v2 配置现会以 exit=`1` 明确拒绝；
- 修正配置：`configs/worldsim_v32/s1_semantic_lift_v3.yaml`，SHA-256=
  `377cd95999dcc02d15782fce06940952826c410d5f4df13846e5dd4c58304960`；prompt v3 SHA-256=
  `8c43b59175da1598b9720bb71d35d573647651ee4075c44ac7b0e265931f6ccf`；
- canonical run=`20260810T101739Z__s1-semantic-lift-s0-r6` 已 `done`，final summary SHA-256=
  `482dcd067ee91952536e863cded1e18cffa1003bbd3f1b0caa9a18380e93bb4a`；398 个 train-only masks 中
  `334 accepted / 64 rejected`，heldout leaks=`0`；
- high-support labels=`1,230,548 / 4,525 / 36,767 / 38,028`，boundary-support labels=
  `1,276,927 / 3,728 / 21,033 / 8,180`（negative/core/semantic/ambiguous）；6 个真实
  original/delete/lateral smoke 非空，D2 checkpoint before/after SHA exact；
- SAM2 wall=`81.605s`、semantic lift wall=`919.436s`；peak allocated 分别为
  `1,908,027,904 / 15,723,618,816 bytes`，单卡 RTX 3090 无 OOM；
- S3 r2=`20260810T103527Z__s3-asset-harvest-s0-r2` 在模型加载前因 PyTorch 2.10 CUDA context
  未显式初始化而 `rejected`；GPU 峰值 `0 MiB`、无部分模型输出，不得作为质量证据；
- S3 canonical r3=`20260810T112505Z__s3-asset-harvest-s0-r3` 已 `done`；输入冻结为 high-support
  actor 的 CAM_FRONT_LEFT frame `91`（1-view）和 `51`（2-view 补充），两帧皆为直接 prompt、
  非 heldout，SAM mask 均被 D2 counterfactual actor-effect 完整覆盖；
- 官方 Asset Harvester 1-view/2-view 均生成 `16` 个新视角与非空 Gaussian PLY；推理
  wall=`113.981s`，peak NVIDIA VRAM=`20,137 MiB`，cgroup peak=`48,426,651,648 bytes`，
  `oom/oom_kill=0/0`，磁盘停止门通过；
- 两份资产均已 exact reload 并匹配 actor LWH，各在 frame `91/51` 完成 original/lateral/delete
  回注渲染，D2 checkpoint before/after SHA exact；2-view 的 mean IoU/PSNR/LPIPS 为
  `0.733945 / 16.671399 dB / 0.094894`，优于 1-view 的 `0.723918 / 16.078961 dB / 0.104843`，
  但 boundary F1 较低（`0.459813` vs `0.499815`）；综合多视角目视 QA 选定 `high_support_2view`；
- S3 final summary SHA-256=`8dc4fc930229fbb17343b0bbcf9ccda632ac54b2e5301d4ca6448bda0d99c2d1`；
  背面只声明生成完整性/一致性，不声明 GT correctness。
- S2 r1=`20260810T120554Z__s2-3dgic-adapted-s0-r1` 因 boundary-support 原支持视图几何重叠不足而
  `rejected`；其 train-only 全量视图/相机搜索证明只有目标帧之前的 CAM_FRONT 有重叠，诊断保留在 r1；
- S2 r2=`20260810T121342Z__s2-3dgic-adapted-s0-r2` 完成方法链但候选未选定：把全部未观测 Telea
  补全写入静态 Background 后，held-out PSNR/SSIM 平均退化 `0.495842 dB / 0.007160`；该候选保持
  `candidate_selected=false`，不得用于集成；
- S2 canonical r3=`20260810T121829Z__s2-3dgic-adapted-s0-r3` 已 `done`；3DGIC 官方深度引导跨视图原则与
  RGB-D unprojection 被显式适配到 StreetGS，2D unseen 补全用确定性 OpenCV Telea；不声明是未修改的上游
  3DGIC checkpoint 运行；
- high-support 的 `15,461` 个目标像素中 `7,189` 个有 train-only 跨视图观测，unseen=`8,272`；
  boundary-support 的 `288` 个目标像素中 `46` 个有观测，unseen=`242`。完整 unseen 2D 资产均保留，但高支持
  checkpoint 只持久化有几何观测的点，小 boundary 目标保留完整补全；
- r3 向 Background 追加 `1,896` 个 `GENERATED_BACKGROUND` 行（`1,205,164 → 1,207,060`），旧行保持
  exact，候选落盘后严格重载，V3.1 ancestry 兼容账本与权威 V3.2 provenance sidecar 对齐；
- 目标视图 candidate effect=`9,928 / 176` 像素，outside L1=`0.042503 / 0.005122`；四路只读 held-out
  平均 PSNR/SSIM/LPIPS delta=`-0.022958 dB / -0.000528 / +0.000301`，通过冻结
  `-0.1 dB / -0.005 / +0.01` 门；unseen completion 不声明 accuracy；
- r3 checkpoint/summary/provenance SHA-256=`3d6e13d47291f5b5949ff3adf5598b6e0cffb930c4cbff2200c6e708d82e6e0f` /
  `a07bbf7a1b160d352fd0d3d08be9e217a3d27648eeffec7841f443b5bc871407` /
  `1baf73b81205f66cfe30a6ea3385cdf960b3d8952648031fb34be26a7ef758cc`；wall=`63.908s`，
  peak NVIDIA=`8,125 MiB`，cgroup peak=`39,369,183,232 bytes`，OOM=`0`，D2 source before/after SHA exact。

## V3.2 S4 收口与 R0 入口

- 官方 Harmonizer source commit=`dd5799e50855c5bcb1f6ef52a77b5b644b4798c0`，Apache-2.0 code license；
  `harmonizer_nontemporal.pt` revision=`20ca33d4612b1e98e0526b3a7ee604af5b289f58`，bytes=`1,448,843,112`，
  SHA-256=`ece8e2daa914e8c2a027a2da94e0eb2064491d5b3fd8514009fae9a442e06e90`；模型受 NVIDIA Open Model License 管理；
- temporal 分支所需 `nvidia/Cosmos-Predict2-0.6B-Text2Image@dd55b685...` 为 gated repo，当前账号无授权、
  HF token 不存在；4,324,256,313-byte 文件清单可审计但下载返回 403，不绕过门禁；
- 官方 JIT 在非 NGC 环境依赖 `tex_ts::rmsnorm_fwd_inf_ts`，并把两个 einops shape scalar 随
  `map_location` 移到 CUDA。适配器使用公式等价 RMSNorm（BF16 exact，max error=`0`）并把整数 1/2 shape scalar
  放回 CPU；4 个单测通过；
- r1=`20260810T131510Z__s4-harmonizer-nontemporal-s0-r1` 在推理前因未显式 `set_device` 就重置峰值计数器而
  `rejected`；r2 完成 5 图但视觉 QA 发现 G1 删除区重生成黑色车辆外观，不能作为候选；
- canonical r3=`20260810T131909Z__s4-harmonizer-nontemporal-s0-r3` 已 `done`，覆盖同一
  CAM_FRONT_LEFT 的 G0 original、G1 remove+inpaint、G2 Asset Harvester lateral；全部输出仅标记
  `HARMONIZED_2D`，D2/S2/S3 assets before/after SHA exact，无 Gaussian 写回；
- G0/G1/G2 mean outside-mask L1 分别约 `3.543 / 3.832 / 3.641` uint8，常态 inference median=`0.3386s`；
  G1 删除区 inside L1=`14.2173`、`>8` changed fraction=`0.54175`，违反冻结 `12.0 / 0.40` 语义保持门，
  因此 `non_temporal_candidate_selected=false`、`final_disposition=optional_diagnostic`；
- r3 summary/status/grid SHA-256=`4543b5fa2543f6f42aa65f0dbc17f11899de1cc7ebad4aed653200e881f1ba39` /
  `42465759974c60f0fa5407969b12ccf8aeb5952ed7c36904b378a86163b78e51` /
  `086b08b7ab57de7a27d28dda28a84109579ff8cfae15216f88533085e19f3cbf`；wall=`35.048s`，
  peak NVIDIA=`4,077 MiB`、CUDA reserved=`4,131,389,440 bytes`、OOM=`0`；S4 task=`done`，生产集成时保持 excluded diagnostic。

## V3.2 R0 canonical 收口

- 失败尝试全部保留为 `rejected`：r1 在 source snapshot 前发现相对 config path；r2 在首个 forward 发现
  DriveStudio 未 `set_eval()` 会与 `inference_mode` 的 `retain_grad` 冲突；r3 把冻结的 MAE 门误实现为
  L∞/max-error 门。三者均未被改写，修复后使用新的唯一 run；
- canonical r4=`20260810T134658Z__r0-final-integration-s0-r1` 已 `done`，8/8 gates 通过：generated-background
  provenance、semantic extension、mixed precision、actor registry、chunk reassembly、render validation、input
  immutability 与 resource ceiling 全为 `true`；
- S1 两份 sidecar 在旧 Background 与 RigidNodes 之间插入 1,896 行 actor-negative/zero evidence；旧 Background
  prefix 与 Rigid suffix exact。high/boundary V3.2 sidecar SHA-256=
  `7caae12fdfb92f15ae02f5f7fc6f5c8111236f18632516a128b09960b6d79b26` /
  `74dd3679b58423c6e752cd3441a347d8f3f3f1add1e5ce748e75150eb510185b`；
- S2 selected checkpoint 仅将 Background/RigidNodes 的 scales、quats、features 与 opacities 转为 FP16；means
  与其余 state 保持原 dtype/value。candidate bytes=`432,347,490`、SHA-256=
  `6d4e4c489f53bf4e7de3f5c405ec37dc63d3f79155aad5237fe175ce0fcd7e5d`，较 FP32 source 减少
  `146,922,064 bytes / 25.363333%`；
- V3.2 registry SHA-256=`6633af150baa4b5adda143b2037091e7647f85966490de5d660fa74968ab6c57`；
  rigid index `5` 绑定 S3 99,045-Gaussian `GENERATED_ACTOR`，boundary 与其他 actor 明确回退 V3.1 native registry；
  S4 non-temporal 明确 excluded，S5 保持 `blocked`；
- 动态 row schema 物化 `133 static + 24 actor + 1 skeleton` payload，chunk manifest SHA-256=
  `af7b402e0b171b11f8c22e4123002f4f844db746ea72f53b77c3de878bf0947d`；payload bytes=`444,282,102`。
  Background `1,207,060` 与 RigidNodes `104,704` 行均 covered once，missing/duplicated=`0/0`；85 个 tensor
  path、容器 schema 与 non-tensor state 全部 exact；
- 三个固定视角 source→mixed PSNR=`68.2993 / 67.2399 / 68.4322 dB`，MAE=
  `0.009614 / 0.012271 / 0.009330` uint8；mixed 与内存重组 checkpoint 的三个 RGB SHA 逐视角 exact；
- resource audit：wall=`103.099s`，peak NVIDIA=`8,362 MiB`，peak CUDA allocated/reserved=
  `7,729.707 / 8,020 MiB`，cgroup peak=`48,169,205,760 bytes`，run bytes=`948,244,397`，
  OOM/kill=`0/0`；所有 D2/S1/S2/S3/S4 输入 before/after SHA exact，无训练或 optimizer step；
- V3.2 定向回归=`36 passed`。production chain 固定为 S1 extended semantic sidecars + S2 generated-background
  mixed scene + S3 generated-actor override + R0 exact chunk package；当前单卡工作终态=`done`，无下一执行授权。

## V3.1 终局裁决

项目不再以“提出新的可编辑 3DGS”或 V2 M5/M6 大型失败评测为主线。V3 的交付目标是完整的 WorldSim
模型链和 A0–A4 消融：原生 StreetGS → 校准增强 → actor-aware 增密/剪枝 → 编辑后局部 Gaussian 精修
→ 部署优化。

核心模型问题固定为：

1. 动态 actor 是否应使用区别于静态背景的 Gaussian 增密与剪枝规则；
2. 对象移动/删除后，局部 3D Gaussian 短步精修是否能改善空洞、深度/透明度排序和时序闪烁。

A3 已给出受冻结合同约束的负答案：R1 S-B 四步工程链可重放，但 heldout evaluator 连续越过 GPU ceiling，且
资源无效 diagnostic 是 geometry 改善与 RGB safeguard 退化并存的 tradeoff。当前生产路由使用 R0/D2 exact alias，
不把 R1 checkpoint 升级为方法或部署基线。

A4-P0 v1 formal r1 已完成 probe 与无 torch resume audit，但 source config 的三路相机实际按 2 倍下采样加载，
模型原生 render 为 `800×450`，不是 v1 误写的传感器尺寸 `1600×900`。r1 因唯一 audit
`native_resolution_exact` 保留为 `blocked`；v2 只纠正该输入合同并冻结 r1 证据，不把 r1 性能登记为正式结果。
v2 validator 已核对 16 个 exact inputs，协议测试 7 passed，联合 WorldSim V3 回归 152 passed。

A4-P0 v2 formal r2 已以 `done` 关闭：13/13 audits 全 true，prepare 占 60.78 s wall 的 82.95%，cold/warm load
约 `.39/.40 s`，9 个模型原生 view 为 P50/P95 `.068/.127 s` 与 `16.38 FPS`；资源峰值为 `8,574 MiB`
NVIDIA sampled / `22.79 GiB` cgroup，OOM=0。P0 不证明并发或质量改进；它只支持先冻结无模型变异的 P5
registry/resume，而不先启动 prune、FP16 或 chunk。

P5 protocol SHA=`51acb935...5874` 已在新 P5 测量前冻结。r1 在成功生成 `14,729-byte` registry 后，因把 checkpoint
key `points_ids` 当作 runtime attribute 而 blocked；旧 terminal 保留。修复提交=`0e899b2`，未改变协议与测量合同。
canonical r2=`20260809T155753Z__a4-p5-registry-resume-s0-r2` 已 14/14 audits passed：reference-only registry
保持 `1 static / 24 actors（23 available / 1 unavailable）/ 1,309,868 total GS`，全部 actor count/index hash 与
source before/after SHA exact。reload=`52.321 s / one load / zero render`，资源门通过；no-torch resume=`.128 s`，
无 GPU launch并复用四个 completed stage。P5=`done`，不产生 chunk、filesystem-cold、concurrency 或质量 claim。

P1 完成时 A4 最低完成集仍要求 P2/P3，因此 task 当时保持 `running`。P1 protocol SHA=`4f893c09...429b` 在测量前冻结，runner=
`19cab2cf...7163`。canonical r1=`20260809T165058Z__a4-p1-contribution-prune-s0-r1` exit=`0`、21/21 audits
passed、summary SHA=`7c5347e3...7119`。36-view contribution score、b05/b10/b20 原子 checkpoint/registry、四臂
57-view global/actor/boundary/non-target 质量、9-view runtime 与 no-torch resume 均完成；source replay exact。
b05/b10/b20 分别减少 checkpoint `23,881,368 / 47,762,712 / 95,527,000 bytes`，但最小 b05 已使 global
occupied PSNR/global PSNR/non-target PSNR 退化 `0.117684/0.110926/0.125462 dB`，超过冻结 `0.10 dB` 门；
b10/b20 分别失败 12/15 项。全部候选因质量而 rejected，P1 method=`rejected_quality_or_integrity_gate`，生产路由
exact fallback 到 p1-source/A3*=R0-D2，实验终态=`done`。resource audit passed：wall=`605.281 s`、allocated/
reserved/NVIDIA=`14,342.71/14,892/15,234 MiB`、cgroup=`26,264,842,240 bytes`、run=`1,610,165,885 bytes`、
OOM=0；resume=`2.316 s`/10 stages/no torch/no GPU。

P2 protocol SHA=`6558fb3f...6d4e` 已在任何 P2 conversion/render 前冻结。输入 exact 指向 P1-selected source 与 P1
canonical evidence，不允许使用 rejected prune checkpoint。候选只转换 Background/RigidNodes 的 scales/quats/
features/opacities 共 10 tensors；source audit 显示 Background means 若 FP16 roundtrip 最大空间误差近 `1 m`，因此
means、Sky、LPIPS、trajectory 与 provenance 保留 FP32/原 dtype exact。runtime persistent parameters 为 FP16，
但进入 gsplat 前显式转 FP32、autocast=false，不宣称 FP16 renderer。57-view 31 项质量门、9-view runtime、
7-stage recovery、900 s/16 GiB torch/48 GiB cgroup/1 GB run ceiling 与 19 audits 已固定；full validator passed，
协议测试 9 passed、联合 WorldSim V3 199 passed。该冻结点的下一步只实现并提交 runner，P3 当时仍未授权。

P2 runner/fix=`1cd9a6e / dcf2822`。r1=`20260809T174337Z__a4-p2-mixed-precision-s0-r1` 的 conversion、quality、
runtime、aggregate 与 resume 均完成，但参数账本未遍历普通 `trainer.models` 映射，finalizer 唯一 audit 失败；r1
保持 `blocked`，terminal SHA=`5ef3dab6...74c0`。只修账本后的 canonical r2=
`20260809T174850Z__a4-p2-mixed-precision-s0-r2` exit=`0`、19/19 audits、31/31 safeguards、source replay exact，
summary SHA=`980f9b0f...1103`。candidate checkpoint=`7be87e8b...7448 / 432,111,754 bytes`，较 source 减少
`146,707,920 bytes / 25.346049%`；persistent parameters=`394,641,424→247,936,208 bytes / -37.174307%`。
runtime 只报告 source/candidate load=`.33669/.47407 s`、P50=`.04583/.08721 s`、P95=`.13170/.09750 s`、
FPS=`17.256/13.065`，不支持 speedup claim。resource passed：wall=`206.548 s`、allocated/reserved/NVIDIA=
`7,754.05/8,072/8,426 MiB`、cgroup=`29,673,631,744 bytes`、run=`436,430,167 bytes`、OOM=0；resume=
`1.217 s`/6 stages/no torch/no GPU。P2=`done`，selected=`p2-gs-param-fp16`；该时点 A4 仍缺 P3。

P3 protocol SHA=`dfaaba79...1b41` 已在任何 chunk materialization/render 前冻结，输入 exact 接 P2-selected mixed
checkpoint 与 P2 19/19 canonical evidence。static 使用原点 `[0,0] m` 的 50 m XY 半开网格，source-only audit
固定 `133` 个 occupied chunks（count `1..330,169`，98 个 `<100`，7 个 `>=10,000`），不允许稀疏/离群块丢弃、
merge 或 cell-size search。Background/Rigid row tensor schema=`25/26`；24 个 actor 均使用显式升序 source flat
indices，23 个非空 actor 全部 interleaved，actor 14 输出 zero-row asset。package 固定 manifest+skeleton+133 static+
24 actor=`159 files`，仅内存 scatter 重组，recursive tensor 必须 bitwise exact，禁止复制 source 或落盘重组
checkpoint。质量要求 source 回放 P2 exact、chunk 的 57 RGB SHA 与 31 endpoints exact；9-view runtime 读取全部
assets，只报告、不做 streaming/load/render speedup claim。8-stage recovery、900 s/16 GiB torch/48 GiB cgroup/
1 GB run ceiling、21 audits 与 P2 exact fallback 已固定；full validator passed，协议测试 12 passed、联合 WorldSim
V3 222 passed。本冻结点未创建 package/render/formal run；下一步只实现并提交 runner。

P3 runner=`aba55777...b481`。canonical r1=
`20260809T184240Z__a4-p3-chunk-s0-r1` exit=`0`、terminal=`done`、21/21 audits passed，summary SHA=
`f8e6e166...a293`。package manifest=`35a3f1fe...64b8`，`133 static + 24 actor + skeleton + manifest = 159 files`；
package=`444,177,055 bytes`，比 `432,111,754-byte` source checkpoint 大 `12,065,301 bytes / 2.792171%`。
85 个 tensor path 和 non-tensor state exact reassembly，Background/Rigid `1,205,164/104,704` rows covered once、
missing/duplicated=`0/0`，actor 14 显式为空；source checkpoint/registry SHA 前后不变。source replay 31 endpoints
max abs diff=`0`，chunk 的 57 RGB SHA、31 endpoints 与 masks 全 exact，P2 FP16-persistent/FP32-renderer adapter exact。
runtime 只报告：source/chunk load=`.9071/4.1775 s`、P50=`.03013/.03950 s`、P95=`.09446/.10586 s`、
FPS=`21.278/20.447`，filesystem cache uncontrolled；不支持 package size、load、render、streaming 或 concurrency
收益 claim。resource passed：wall=`221.786 s`，allocated/reserved/NVIDIA=`7,614.99/8,066/8,420 MiB`，cgroup=
`32,689,958,912 bytes`，run=`444,885,133 bytes`，disk free=`42,359,705,600 bytes`，OOM/kill=`0/0`；resume=
`1.104 s`/7 actions/159 artifacts/no torch/no GPU。selected=`p3-chunk-package`，method=
`selected_exact_chunk_package`，P3=`done`；A4 最低完成集满足，A4=`done`。

三场景是模型消融场，不是新 benchmark。结果只支持当前数据、实现和资源合同下的模型/工程结论，不外推为
大规模泛化、物理真实性或闭环安全结论。

## V2 继承与冻结

### 已完成并继承

| Task | 终态 | V3 用法 |
|---|---|---|
| `DR-V2-M0-BOOTSTRAP-01` | done | 环境、资产、网络与 source provenance |
| `DR-V2-M1-DGGT-REPAIR-01` | done | 历史前馈范式对照；不再做非等价排行榜 |
| `DR-V2-M2-ACTOR-EVAL-01` | done | persistent actor、raw 轨迹、三相机投影和 frozen cohort |
| `DR-V2-M3-EDIT-BASELINE-01` | done | StreetGS checkpoint、actor registry、基础轨迹编辑 |
| `DR-V2-M4-EDIT-PILOT-01` | done | scene-0230 全序列编辑闭环和可复用指标设施 |

### M5 部分执行后冻结

`DR-V2-M5-STRESS-3SCENE-01` 没有完成，也没有产生 V2 预注册的 24 条序列、pseudo-hole/perception 全量结果
或三场景 final matrix。它不记为 `done` 或 `rejected`，只保留下列事实：

- scene-0230 held-out checkpoint：`398,652,534` bytes，SHA-256
  `24a39f27dfeed36bbdb01ee14211aec51b414e6ab0e61915b71c1dddcdf61e49`；high/boundary actor 分别
  `4,747/1,914` GS；
- scene-0242 checkpoint：`306,034,934` bytes，SHA-256
  `16179d8f99becb86b6893a18ff036af72d78c9897f7aa2b0e297b735dd6c5fda`；high actor `6,939` GS，
  boundary actor 为显式 `ABSTAIN`；
- scene-0255 数据准备和 sky 阶段已有产物，但原生训练阻塞于
  `datasets/driving_dataset.py` 的 CUDA `torch.cat(instance_dict[ins_id]["pts"], dim=0)`；
- r27 诊断输入为 166 个 CUDA float32 tensors，其中 152 个 `(0, 3)`，总计 177 scalars；无 OOM 证据；
- evaluation sequencer r16/r18 的 `running` terminal 属于容器中断遗留；现场无对应进程或 tmux，不得改写终态；
- M5 未提交的脚本、配置和测试保留在工作树中，P0 不清理、不覆盖、不混入 V3 文档提交。

V2 M6–M8 不再授权。V2 计划原文件保持不改，只作历史执行合同。

## V3 源码事实

DriveStudio 固定 commit `e59bda4fa681f829dbb1d65f0de582b0f633c450`。源码审计确认：

- 原生 `AffineTransform` 已提供 per-image RGB affine；
- 原生 `CameraOptModule` 已提供平移和旋转位姿残差；
- 原生数据链已用 LiDAR 初始化背景和动态实例；
- `RigidNodes` 仍对所有 actor Gaussian 使用统一的 gradient/scale/screen-size/opacity 阈值。

因此 A1 是已有校准能力的 off/native/enhanced 消融；A2 才是 V3 的首要模型新增。rolling shutter 只有在
processed data 存在真实 readout direction/time 后才可实现，否则必须报告 `not_supported`。

## A0 完成证据

- 实现提交：`436cfc1`（`fix(drivestudio): 过滤空 LiDAR 实例块`）；
- patch SHA-256：`54e7584b6d74431e58f626dfaadd69812d4058d54f82c7941e75aa11f5f94619`；
- frozen DriveStudio：`e59bda4`，实际训练使用独立 patched worktree
  `/root/autodl-tmp/third_party/drivestudio-worldsim-v3-r2`，原始上游保持 clean；
- 定向测试：`16 passed`；patch apply/reverse-check 与 `git diff --check` 通过；
- scene-0255 canonical smoke：
  `/root/autodl-tmp/runs/worldsim_v3/WS-V3-A0-NATIVE-BASELINE-01/20260805T161656Z__scene0255-catfix-s0-r2`
  =`done`；原生 r27 mixed-empty CUDA cat 错误被复现，修复后为 `59×3 / 177 numel` 且点/颜色 exact pairing；
- 1-step 真实训练完成 dataset init、`966,259` background GS、`27,894` rigid GS、优化和 checkpoint 保存；
  controller duration `72.1 s`，peak GPU sample `8,388 MiB`，peak cgroup `5,971,820,544` bytes，
  `invalid_configuration=false`。

该 smoke 只解释兼容修复。A0 正式冻结还包括：

- scene-0255 新 30k run：`20260805T162355Z__scene0255-native30k-s0-r1`；
- scene-0230/0242 等价 checkpoint 复用 run：`20260805T171624Z__scene0230-reuse-eval-s0-r1` 与
  `20260805T171914Z__scene0242-reuse-eval-s0-r1`；
- 全图 PSNR（0230/0242/0255）：`24.934 / 29.107 / 25.230`；总 GS：
  `1,319,913 / 930,011 / 1,551,383`；训练时间：`3014.5 / 2006.2 / 2739.4 s`；
- high actor 区域 PSNR/SSIM/tight-crop LPIPS：`21.728/0.596/0.121`、
  `19.788/0.665/0.153`、`23.531/0.665/0.058`；scene-0242 boundary role 为预注册 `ABSTAIN`；
- actor mask 为 paired original/delete render 的模型 counterfactual diagnostic，不是真值分割；每场记录
  visible image 和 pixel coverage，checkpoint 评估前后哈希一致；
- 唯一汇总：
  `/root/autodl-tmp/runs/worldsim_v3/WS-V3-A0-NATIVE-BASELINE-01/20260805T175000Z__a0-three-scene-finalize-s0-r2`
  =`done`。r1 的训练资源 schema 字段差异已作为 `blocked` 保留，`00ba4e8` 修复后 r2 通过。

A0 的核心判断是：全图重建质量不能替代 actor/边界质量。scene-0242 全图 PSNR 最高，但 high actor PSNR
最低；scene-0255 boundary actor 区域 SSIM 仅 `0.526`。这为 A1/A2 提供目标，不构成跨场景因果结论。

## A1 开发场景完成证据

- 端点提交：`20c4276`；权威相机映射修复：`d85ef27`；LiDAR provenance：`14bc3c2`；
- 冻结 E1/E2 配置 SHA-256：
  `60c211625860c25edf92842b88bdb040ea8c180b12fe0fa78f2fc1c342bc4051`；
- C0/C1 有效正式端点 run：
  `20260806T141409Z__scene0230-c0-a1-e0-formal-full-camera-map-fix-s0-r2`、
  `20260806T141623Z__scene0230-c1-a1-e0-formal-full-camera-map-fix-s0-r1`，均为 `done` 且 checkpoint SHA 未变；
- E1 median/P90：C0 `0.05951/0.14719`，C1 `0.06289/0.15623`；coverage 为 `10.780%/10.614%`；
- E2 high actor mean/P90：C0 `0.004813/0.010895`，C1 `0.004751/0.010895`；boundary actor：
  C0 `0.003547/0.006353`，C1 `0.004450/0.007626`；
- 错误相机标签 run `20260806T140703Z__scene0230-c0-a1-e0-formal-full-s0-r1` 已显式
  `rejected / INVALID_CAMERA_ID_LABEL_MAPPING`，不得进入结果；
- 最小 LiDAR provenance 正式 run：
  `20260806T143644Z__scene0230-a1-lidar-provenance-formal-full-witness-s0-r1`=`done`；配置 SHA-256
  `f2fd1712cf4ddd75c1c4d1da4a426dcf7e1340a5fd943066401ba881f51c5639`；196 个 block、6,804,832 raw
  points、24 actor/75,002 actor points 均入账；
- 记录的 LiDAR/actor tensors exact match，但 CUDA visibility filter 使随机背景初始 GS 从源运行 946,484
  变为正式 witness 946,291。初始深度 median/P90=`7.679/35.958 m` 仅为
  `seed0_reconstructed_initialization_witness_not_exact_source_initialization`，不是源初始化 exact replay；
- A1 定向测试 `23 passed`；逐 Gaussian ancestry/parent-child/split-clone lineage 按 V3.1 后移至 A2。

scene-0230 四个配对 30k 训练均已完成；共同 initialization provenance SHA 为
`8951543c33f72f439068237f1a552fae660895f8906afbf4651f5f580981b898`：

| variant | global PSNR / LPIPS | boundary actor PSNR / LPIPS | high actor PSNR / LPIPS | total GS | train min |
|---|---:|---:|---:|---:|---:|
| C0-off | 27.746 / .1764 | 27.756 / .0687 | 25.358 / .0943 | 1,360,649 | 52.05 |
| C1-native | 24.979 / .1694 | 22.549 / .1033 | 21.696 / .1201 | 1,316,421 | 53.69 |
| C2-factorized-isp | 25.011 / .1677 | 22.583 / .1043 | 21.779 / .1174 | 1,322,979 | 52.26 |
| C3-bounded-pose | 28.109 / .1666 | 28.169 / .0657 | 25.137 / .0938 | 1,363,040 | 56.14 |

- C2/C3 训练 run：`20260806T144938Z__scene0230-c2-factorized-isp-formal30k-s0-r1`、
  `20260806T154834Z__scene0230-c3-bounded-pose-formal30k-s0-r1`；
- C2/C3 端点 run：`20260806T154541Z__scene0230-c2-a1-e0-formal-full-s0-r1`、
  `20260806T164852Z__scene0230-c3-a1-e0-formal-full-s0-r1`；均保持 checkpoint SHA 不变；
- 冻结 A1-D0 配置 SHA-256 为
  `a445078d3bea89a78a0c9e6544a94a2be4c9c2e71f45aec4a9d8878b4c6593c1`；正式诊断
  `20260806T170219Z__scene0230-a1-diagnostics-c0-c3-formal-s0-r1`=`done`；
- 输入速度层为 near-static/low/normal=`2/18/176` 帧；near-static 仅 2 帧，只作低支持描述；
- C3 学习位姿修正 translation median/P90=`1.703/2.338 mm`、rotation=`0.02553/0.03337°`，明显小于
  C1 的 `7.256/12.215 mm`、`0.1660/0.35465°`；这只是学习修正幅值，不是独立 pose GT；
- 选择实现提交 `60ef079`，无容差选择配置 SHA-256 为
  `a45699ebf696c875a18832f8db920a6106837a1e4f235dcd9036eff48dfbc609`；明确披露其在开发结果可见后、
  确认场景前操作化；
- 正式选择 run `20260806T171417Z__scene0230-a1-dev-selection-formal-s0-r1`=`done`，冻结
  `C*=C0-off / done_off`：C2 只改善 boundary role E2，high role 与 LPIPS 退化；C3 画质和位姿稳定性最好，
  但 E1/E2 均未严格改善。确认场景 C* 项登记为 C0 exact alias，10 个逻辑矩阵项对应 8 个唯一训练。

## A1 确认与正式终态

冻结确认配置 `configs/worldsim_v3/a1_confirmation_v1.yaml` SHA-256 为
`63a3cc607ccfddbb714cc81d0570da356263c01c5a68880345953023d2d6a8cd`，实现提交 `198a681`。四个确认训练和
端点 run 均 `done`、每场景 C0/C1 initialization SHA 相同、所有端点评估前后 checkpoint SHA 不变：

| scene / variant | global PSNR / LPIPS | E1 median / P90 / coverage | E2 high mean / P90 / coverage | E2 boundary mean / P90 / coverage |
|---|---:|---:|---:|---:|
| 0242 C0 | 30.064 / .1108 | .03147 / .08826 / 6.491% | .008264 / .020697 / 42.857% | `ABSTAIN` |
| 0242 C1 | 29.161 / .1122 | .03333 / .08971 / 6.423% | .008660 / .021708 / 42.857% | `ABSTAIN` |
| 0255 C0 | 27.255 / .2086 | .04348 / .14248 / 6.710% | .004772 / .009805 / 23.529% | .004032 / .009308 / 41.176% |
| 0255 C1 | 25.240 / .1921 | .04277 / .13626 / 6.751% | .003715 / .007704 / 21.569% | .003923 / .008784 / 41.176% |

- 0242 原始端点与全图指标偏向 C0；boundary role 按预注册继续 `ABSTAIN`；
- 0255 的 C1 E1/E2 error 较低，但 high-role coverage 降低，且 boundary/high actor LPIPS 均退化，未通过完整合同；
- exact alias run：`20260806T211000Z__scene0242-cstar-c0-exact-alias-s0-r1`、
  `20260806T211100Z__scene0255-cstar-c0-exact-alias-s0-r1`，均明确无新训练/评测；
- A1 finalizer `20260806T211248Z__a1-three-scene-finalize-s0-r1`=`done`：10/10 逻辑项、8/8 唯一训练，
  `C*=C0-off / done_off`。该结论是完整冻结合同下的 Pareto 选择，不是“所有场景每项指标 C0 都最好”。

## A2-I0 ancestry instrumentation 完成证据

- canonical r3 项目基线：`research/worldsim-v3@70cf2b2` + formal run 内不可变 source snapshot；当前实现提交：
  `271d876`；
- DriveStudio upstream：`e59bda4fa681f829dbb1d65f0de582b0f633c450`；patched worktree：
  `/root/autodl-tmp/third_party/drivestudio-worldsim-v3-a2-r5`；
- 配置 `configs/worldsim_v3/a2_instrumentation_v1.yaml` SHA-256：
  `bac1ec5b3642470a999e7f0cf8ddc9cf5b4d9a1445029c43ae92601929f4bfce`；
- instrumentation patch SHA-256：`87c084f77ed5d6395acce95abb992ca86004bdc47b68154878bf462a0fb345b0`；
- canonical formal run：
  `/root/autodl-tmp/runs/worldsim_v3/WS-V3-A2-ACTOR-DENSIFY-01/20260809T071500Z__a2-i0-ancestry-formal-s0-r3`=`done`；
- module-off/on 原生 checkpoint tensor 逐位一致、无 mismatch；off 不增加 ancestry key，on 增加且 round-trip；
- 8 个初始 Gaussian 经 split/clone/prune 后保留 10 个、累计分配 11 个 ID；来源计数 LiDAR/split/clone=`7/2/1`；
- actor/parent/lineage root、prune 后索引与 checkpoint 恢复通过；`nearest_lidar_distance` 对 actor 做 exact offline，
  background 因无有界参考集保持 deferred；
- boundary/photometric/depth/normal 在 I0 只冻结 attributed update API；无可靠 normal 时保持 schema-only；
- patched worktree verify、patch reverse-check、当前 working-tree `git diff --check` 和 WorldSim 定向测试
  `66 passed`。

该结果只关闭 deterministic synthetic `RigidNodes` instrumentation 门禁，不是 scene-0230 真实质量证据，
不授权直接启动 D1 formal。本次 source commit 只包含 A2-I0 代码、测试与直接相关文档，不混入保留的 V2 M5 文件。

## A2-D1 formal 协议冻结证据

- clean 协议/控制器/评测提交：`387dd501cd931b632ca4fd9950ee40b14bac6fce`；
- formal 配置：`configs/worldsim_v3/a2_d1_formal_v1.yaml`，SHA-256=
  `ad77db41d9d8c5172804a20b38a2dd92173c3639398d8abc24dc6f4799e8f8e7`；
- scene-0230 / seed 0 / D0→D1 / 每臂 30k；5k 保存只读 candidate checkpoint；
- matched-GS 只匹配干预域 `RigidNodes`：目标为 D0 30k 最终计数，D1 按绝对差最小、并列更早 step 选择；
  相对差 `<=2%` 才登记 done，否则 `ABSTAIN_BUDGET_NOT_MATCHED`；禁止事后 pruning、重训或 quota retune；
- held-out 端点为 global、high/boundary actor region 与 boundary band，以及两 actor 反事实 mask 并集之外的 non-target；
  counterfactual mask 明示不是 GT segmentation；
- `80 passed`；只读 preflight=`done`：GPU=`0 MiB`、free disk=`58.39 GiB`、memory.max=`90 GiB`、
  canonical r4 summary SHA 与三层 DriveStudio patch SHA 全部匹配；
- 协议冻结提交时 formal 尚未启动。该证据本身只解除启动门，不构成 D1 质量结论。

## A2-D1 formal 完成证据

- canonical run：
  `/root/autodl-tmp/runs/worldsim_v3/WS-V3-A2-ACTOR-DENSIFY-01/20260809T085400Z__a2-d1-paired-formal30k-s0-r1`；
- source commit=`f32f96b47619e05066d2ee11c899e38d07398e11`；terminal=`done`；summary SHA-256=
  `e3b194c2ed0563385df70ca2043dbc791bedb21068d28dc9d75fb59984c166ac`；manifest SHA-256=
  `f10e6e654ab27289ccb1c995ebbe1ffde913009dbfb3eae0ab4c6414de18a560`；
- D0/D1 物化配置配对，初始化 provenance SHA 均为 `8951543c...b898`，初始 Background/RigidNodes=
  `946,484 / 75,002`；6×2 checkpoint 网格、quota/ancestry、native finite 与 24/24 actor 上限均通过；
- fixed 30k D0/D1：Background/Rigid/total GS=`1,182,619/177,628/1,360,247` 与
  `1,201,057/105,412/1,306,469`；global PSNR/SSIM/LPIPS=`27.7481/.851207/.176319` 与
  `27.7700/.850915/.177704`；质量轴更优数 D1/D0=`12/7`，裁决=`tradeoff_non_dominated`；
- matched 选中 D1 15k：Rigid=`176,741`，与 D0 target 差 `887 / 0.499%`；D0 视图为 fixed final exact alias；
  D1 Background/total=`2,432,701/2,609,442`，global=`25.9290/.825381/.217941`；质量轴更优数 D1/D0=
  `9/10`，裁决仍为 `tradeoff_non_dominated`；
- matched D1 boundary-support actor PSNR/SSIM/LPIPS=`29.2937/.902828/.061463`，优于 D0 的
  `27.1783/.882177/.068895`；但 non-target PSNR/SSIM/LPIPS=`24.3371/.822724/.090772`，劣于 D0 的
  `26.8707/.848887/.057715`。这是局部—全局 tradeoff，不是 D1 全面改进；
- D0/D1 train duration=`2883.08/2099.33 s`，peak GPU=`23,867/23,989 MiB`，peak cgroup=
  `10,350,350,336/16,012,115,968 bytes`；matched 15k elapsed=`1127.66 s`，资源按完整 D1 臂上界报告；
- fixed D0、fixed D1、matched D1 三次评测前后 checkpoint SHA 均不变，high/boundary/non-target 均 `done`，
  `oom=0 / oom_kill=0`。控制器登记 `d2_unlocked=true`，仅解锁 D2 协议冻结。

## A2-D2 协议冻结证据

- 配置：`configs/worldsim_v3/a2_d2_protocol_v1.yaml`，SHA-256=
  `acceb7f4ce0f8dc3745de2fcaca51659891cfd82e4175f5a0e5765d77a01e567`；
- immutable prerequisite：D1 canonical summary SHA=`e3b194c2...66ac`，D1 closeout commit=`f380dd2`；
- 真实信号只用训练帧 dynamic mask 的 3px 形态学轮廓带与 projected-center RGB channel-mean L1 residual；
  gsplat `means2d` 按像素坐标 nearest-center 采样，跳过不可见、非有限和中心出界项；
- per-actor quota 内排序为 boundary observed/mean → residual observed/mean → screen-grad → Gaussian index；
  D1 gradient eligibility、minimum recovery、maximum quota、split/clone cost、Background 与 native cull 全部不变；
- boundary scale cap 复用 native densify size threshold，pre-cap scale 先决定 split/clone geometry，再在原生 refinement
  前同比缩放三轴、保持 anisotropy，并清零 cap 行的 Adam moments；不新增 RNG draw；
- D3 depth/normal、D4 LiDAR/visibility/provenance pruning、非原生 cull 与 Background 干预明确禁止；
- 工程提交=`1065264762569c9832219936ddae6f063d6eaf07`；canonical worktree=
  `/root/autodl-tmp/third_party/drivestudio-worldsim-v3-a2-d2-r8`；D2 patch SHA-256=
  `80fef55195906808d74394af0b997cfccbdb88fd7cb356b45240473e55f357cc`；replay/reverse-check 与六文件状态通过；
- D1/D2 materializer normalized-match、真实 `RigidNodes` synthetic integration 与联合回归=`29 passed`；boundary/residual
  各 6 次观测、1 次排序/refinement、6 个 cap、quota maximum、optimizer moments、checkpoint round-trip 和 module-off
  native state/RNG bitwise 均通过；
- paired smoke r1 见下一节；协议/工程门禁通过本身不等于 D2 方法通过。

## A2-D2 配对工程 smoke 完成证据

- canonical run=`20260809T111304Z__a2-d2-paired-smoke1k-s0-r1`，terminal=`done`；summary SHA=
  `749c7d15c27cc0798c267aa8af12857f3bea52a52ea9d00f7617a3b3edda3136`，manifest SHA=
  `5cb7879d898839b88a46c8ec7ec34141f3402245490416d589938658f33b4c8d`，source=`c594e0c`；
- D1/D2 configs normalized match，initialization provenance 与 frozen initial quota 精确匹配；两臂 step=1000，
  D1=`Background 1,141,192 / Rigid 152,733`，D2=`Background 1,144,988 / Rigid 152,807`；
- D2 observation event=`1001`，boundary/residual observations 各 `10,846,748`，refinement/ordering event=`5`，
  capped Gaussian=`365`，boundary-observed live=`56,732`；cap/quota/finite/checkpoint round-trip 全通过；
- D1/D2 duration=`142.17/141.99 s`，torch peak GPU=`9,615/9,620 MiB`，cgroup peak=
  `16,473,858,048/16,667,971,584 bytes`，`oom=0 / oom_kill=0`；
- 裁决=`d2_formal_unlocked=true`，仅解锁 formal 协议冻结；1k smoke 不登记质量改进。

## A2-D2 formal 协议冻结证据

- formal config=`configs/worldsim_v3/a2_d2_formal_v1.yaml`，SHA-256=
  `b66cf795c55dfe65315ecf49c09951482d8d6809ce7d001b901942a6bd9a05bc`；提交=`20b3f4d`；39 tests passed；
- D1 baseline 使用 formal r1 immutable exact alias，不重训：summary SHA=`e3b194c2...66ac`，provenance SHA=
  `8951543c...b898`，fixed checkpoint SHA=`c9d2a052...af52`，target Rigid=`105,412`；
- 唯一新训练为 D2 30k / seed 0 / 5k checkpoint grid；fixed 比较 D1 alias 与 D2 30k，matched 从 D2
  grid 匹配 D1 fixed Rigid target，最大 relative gap=2%，无 pruning/retrain/retune/mutation；
- held-out/high/boundary/non-target、checkpoint immutability、quality 与 quality-cost exact Pareto 完整继承 D1；
- read-only preflight=`done`，输出 SHA=`9cf49af0be9a2676c6c113bee963efb79704bb9434083857684f97bd19caaa28`；
  project=`20b3f4d`、GPU=`0 MiB`、free disk=`47.92 GiB`，所有依赖与资源门禁通过。

## A2-D2 formal 完成与 A2 裁决

- canonical run=
  `/root/autodl-tmp/runs/worldsim_v3/WS-V3-A2-ACTOR-DENSIFY-01/20260809T113230Z__a2-d2-formal30k-s0-r1`，
  terminal=`done`，source=`482fba0`，summary SHA-256=`9c41dfc83c9da0a14201e1c719fb3d0e2cf59dd1ad20cd279c6e1a9a1c97de7d`；
- D2 final checkpoint SHA-256=`1a061247...e7c`，counts=`Background 1,205,164 / Rigid 104,704`；D1 reference
  checkpoint 运行前后 SHA 都是 `c9d2a052...af52`，初始化 provenance SHA 精确匹配；
- 5k–30k 六个 checkpoint 全部通过 finite/quota/cap 审计；matched 选中 30k，Rigid gap=`708 / 0.67165%`，
  matched D2 因而是 fixed D2 exact alias；
- D1→D2 global PSNR/SSIM/LPIPS 从 `27.770024/.850915/.177704` 变为
  `27.703188/.850333/.178344`；boundary-support boundary-band 从 `25.770024/.821572/.048382` 变为
  `26.171399/.828868/.044568`。边界三项改善与 global/部分 actor/non-target 退化并存；
- fixed/matched strict-quality Pareto 都为 `tradeoff_non_dominated`（D1/D2/equal=`11/8/0`），quality-cost
  也为 `tradeoff_non_dominated`（`14/9/1`）；D2/D1 wall=`2720.82/2099.33 s`；
- 297 条资源记录、四个 stage 全部 completed，peak GPU=`23,989 MiB`，full-run peak cgroup=
  `25,837,490,176 bytes`，`oom=0 / oom_kill=0`，终态 GPU=`0 MiB`；
- A2 状态冻结为 `done`。A3 使用 D2 boundary-residual 作为 boundary-priority research asset，D1 quota-only
  作为低成本/全局质量 fallback；这不是 dominance 或跨场景结论。`d3_unlocked=false`，D4 未启动。

## A3-I0 语义协议冻结证据

- config=`configs/worldsim_v3/a3_local_refine_protocol_v1.yaml`，SHA-256=
  `03fbf632645326692bbcf18ab18a08b5440c7733c709f925945c78018bb272d0`；依赖 A2 closeout=`2246693`、
  D2 checkpoint SHA=`1a061247...e7c`、summary SHA=`9c41dfc8...de7d`、registry SHA=`ed57764e...0c68`；
- 固定 scene-0230 / seed 0 / cameras 0–2、high/boundary 两 actor、lateral/delete 与 19 个只读 held-out frames；
  D1 checkpoint `c9d2a052...af52` 只作 fallback；
- affected set 冻结为 paired source/edited footprint（threshold 2、2px dilation）、supported hole、first-hit conflict
  的并集，再做 3px dilation；target actor 只作冻结 context；
- S-A 要求排除 target view 的 alternate observed RGB + calibrated reprojection；S-B 只接受 T0 LiDAR measured 或
  至少两视图 geometry，禁止 RGB loss；S-C 不更新、不 seed、不进 loss；
- depth 产品继续分为 expected=`diagnostic`、first-hit=`T1`、measured LiDAR=`T0`；D2 Background ancestry 的
  `240,528` 个 direct LiDAR roots 只证明 provenance，不是 measured-depth GT；
- R0 为 D2 immutable exact alias；首个工程门 R1 仅允许 affected S-A/S-B Background opacity/scale，outside
  参数与 optimizer state、RigidNodes、trajectory、registry 全部 exact；
- `formal_training_authorized=false`。未提交 V2 M5 config/metrics/runner 明确排除为依赖；I0 当时要求 paired smoke
  后再冻结数值合同，该门已由下方 real paired/frozen replay 证据关闭；新增 `12 passed`，联合回归 `98 passed`。

## A3 R0/R1 engineering guard 与 synthetic closeout

- implementation=`9c639dd5a0adcd1f8b5126f7f20d836815b127a6`；DriveStudio patch SHA-256=
  `155ec58fd2bfdc2e40357035dc20800bf2340b0c1c9ac5972c7c78efbd8cb69b`；独立工作树=
  `/root/autodl-tmp/third_party/drivestudio-worldsim-v3-a3-r1-r1`，apply/reverse、`py_compile`、import 均通过；
- canonical synthetic run=
  `/root/autodl-tmp/runs/worldsim_v3/WS-V3-A3-LOCAL-REFINE-01/20260809T132133Z__a3-r0-r1-synthetic-s0-r1`，
  terminal=`done`，summary SHA-256=`2ac123f0603120a103743e59680a31dd4cdf5b6d5fa45605d7c84d36ec337ada`，
  manifest SHA-256=`8ffa697e15d8a97108d8281a51313119c304fbf0f245d88bfbd127663fde27c4`；
- R0 materializer 重新命中 checkpoint/config/protocol SHA，只生成 immutable exact alias；optimizer steps=`0`，
  无新 checkpoint/key；
- R1 guard 在 Adam step 前只保留 affected S-A/S-B Background opacity/scale 行梯度，step 后逐位审计参数与 moments；
  synthetic 中授权行变化，outside、position/color、RigidNodes/trajectory、shape/order exact；
- 原 D2 与 A3 module-off 的 RGB/SSIM loss tensor 逐位相等；缺少 paired provenance/masks 会拒绝；
  checkpoint 实际布局为 Background=`1,205,164` 行、RigidNodes=`104,704` 行、trajectory=`196×24`；
- 联合 WorldSim V3/materializer 回归=`110 passed`。该 synthetic run 自身仍为 `synthetic_contract_only` 且记录
  `paired_engineering_smoke_complete=false`；后续 paired 门见下一节，`formal_training_authorized` 始终为 false。

## A3 R1 真实 paired smoke、数值冻结与 replay 证据

- heldout-safe sidecar run=
  `/root/autodl-tmp/runs/worldsim_v3/WS-V3-A3-LOCAL-REFINE-01/20260809T133911Z__a3-sb-sidecar-s0-r3`；
  manifest/rows SHA=`42474f73fc563a2bba4c52cbec029bb4c28d33a21ca5f3d83ad4311bb7957273 / c5756ecbc0eabee9a576a55297a1739aa20e2af578aa4a5a92e727701b5138fc`；
  frame `0/31` 与 heldout 交集为空；affected/S-B mutable/S-C=`16,502 / 51 / 16,451` rows，四 unit 共 8 个
  S-B/T0 geometry pixels，S-A/RGB=`0/ABSTAIN`；
- paired implementation=`d89e0ace37eda22434470849ec9940360c0e9251`，CUDA init fix=`78741b3abee07b2c39be6646c63928e8212b6a6b`；
  当前 DriveStudio patch SHA=`f1732f63ae38f9298cdbd45d38e91bbd9fb5d3dec46e4b96c647ef14db3c588a`，
  materializer 会移除 native regularizer，trainer 再次 fail-closed 校验；S-B occupancy 只认 T0 LiDAR；
- canonical paired run=`20260809T135921Z__a3-r1-sb-paired4-s0-r2`，summary/manifest SHA=
  `ba4e2b853690f0b9c9bb7bfe039b4571db16c020ce726768a1ff884b09b3557d / de717ba0a5adb1afeb416a15a53ec55f471a8eb841882f784012b04ac86b596c`；
  step `30001–30004` 的 opacity/scale 授权行均有 finite nonzero gradient/变化，outside parameter/Adam、
  Rigid/trajectory/registry、shape/order exact；checkpoint SHA=`e995e7c266d9fed4e64c86813718e46ab4576bbfdf60500a637bdaeaaba78cd1`；
- numeric freeze implementation=`c02c8c74c671362e86269bd7e00980bfa75ae1c9`；config SHA=
  `d9289df0b2ac7df7a7c408b5cb1601bc5f874e2922ebc9cb87961aacee43b3e3`，冻结 4 steps、LR `0.05/0.005`、
  affected/mutable cap `16,502/51`、seed cap 0、alpha 0.5 与资源 ceiling；联合回归=`119 passed`；
- frozen replay run=`20260809T140534Z__a3-r1-sb-frozen-replay4-s0-r1`，summary/manifest SHA=
  `7d820a53de21f505a5c56043d56556edb8d3a86510488ea3956b7cfa159187c6 / 393e65d5f91c0e2072eebd7c23a1161d46422502220ceeeaa18c04905fec646d`；
  四 unit loss 逐值一致并重现同一 checkpoint SHA；wall/GPU/cgroup=`50.68 s / 8,286.86 MiB / 22,631,796,736 bytes`，OOM delta 0；
- 结论仅为 `real_paired_engineering_and_bitwise_replay_done`。S-A 未物化，S-B pixel quality claim 禁止，
  `formal_training_authorized=false`，R2–R4 未授权。

## A3 R1 heldout 只读评测负结果与任务收口

- heldout protocol=`configs/worldsim_v3/a3_r1_eval_protocol_v1.yaml`，SHA-256=
  `eb87a9f2ea7df9bdc050a8d4e4f3cdc7c6a1115ea6f4f69e2fd3c8011904b05a`；冻结/评测器提交=
  `42508fb / c8fc560`；资源审计与内存诊断提交=`05cee1e / c9e3df4 / ef74622 / c2eb14f`；联合回归
  `139 passed`；
- closeout run=
  `/root/autodl-tmp/runs/worldsim_v3/WS-V3-A3-LOCAL-REFINE-01/20260809T144037Z__a3-r1-heldout-eval-s0-r5`，
  exit=`1`，terminal=`blocked / peak_gpu_memory_mib`；resource audit SHA=
  `d9536f4ec937bee0694a754038b22ab75a4b6b028f20e1e6f42e38e4db9a6280`；wall/GPU/cgroup/run bytes=
  `117.983 s / 14,241.399 MiB / 23,749,709,824 / 299,910`，冻结 GPU ceiling=`12,288 MiB`，OOM delta=`0/0`；
- r2/r4 的完整指标路径分别为 `14,241.777 / 14,244.924 MiB`，同样只失败 GPU ceiling；r3 在指标前
  失败于 Rigid quota CPU/CUDA validator，已修复且不作结果。未提高 ceiling，也未换 packed/分块 renderer；
- r5 metric/global rows SHA=
  `04da7a2503460c075a3164c90d6c08436bbea9f4ec5560ea0417ee40e91aa939 / 04bf741e1da6cfe845b5ee6c9d4cccede54d79a1c8f7178e00abcf737ff7245e`；
  R0/R1 checkpoint SHA 前后保持 `1a061247...e7c / e995e7c2...8cd1`，run 内无 `.pth`；
- 资源无效 diagnostic：coverage `1.0→1.0`，depth violation `0.915792→0.908173`，non-target RGB MSE
  `0.002095031327→0.002095032019`，original-global RGB MSE `0.002104032262→0.002104032654`；exact Pareto=
  `tradeoff_non_dominated`。该数值只刻画失败，不登记为合格 heldout 证据；
- 状态分层：r5 run=`blocked`，R1 arm=`rejected_resource_gate_and_diagnostic_tradeoff`，A3 task=`done`。
  `A3*=R0-off`，即 D2 checkpoint immutable exact alias；formal、R2–R4 与独立 S-A 训练未授权。

## A2-D1 quota-only 配对 smoke 完成证据

- 工程提交：`c9b2422af637370ca90f48b42a7d0131f458f96d`；配置 SHA-256：
  `6895370625080ccab327e731264e9ebb0f980499b8fec87d02d9efb2e56b14af`；
- DriveStudio upstream=`e59bda4`，canonical worktree=`/root/autodl-tmp/third_party/drivestudio-worldsim-v3-a2-d1-r5`，
  quota patch SHA-256=`c232af2c5fa532016943f399830c85ebba612078871b7c1a296bda816ae7bb1b`；
- canonical run：
  `/root/autodl-tmp/runs/worldsim_v3/WS-V3-A2-ACTOR-DENSIFY-01/20260809T081330Z__a2-d1-paired-smoke1k-s0-r4`，
  terminal=`done`，summary SHA-256=`ec219bb567799d4d84252e86bd4194620f6b5563d6032c43067ff8e155d3b8bd`；
- D0/D1 均为 scene-0230、seed 0、1000 step，顺序执行；配置除 quota enable/variant 外匹配，初始化 provenance 相同；
- actor threshold=`0.00025`，Background 保持原生 `0.0005`；初始/min/max actor 总量=`75,002 / 37,504 / 180,013`；
- D1 quota 5 次 event 接受 `93,057` children、拒绝 `30,171` parent；最终 `152,830` Rigid，24/24 actor
  不超过最大值；D0 最终 `125,915` Rigid；
- module-off tensor 逐位等价；D1 quota/ancestry checkpoint round-trip，D0/D1 原生 tensor finite；
- D0/D1 peak GPU=`12,807 / 12,795 MiB`，peak cgroup=`5,392,334,848 / 5,661,368,320 bytes`，
  duration=`110.91 / 110.97 s`，无 OOM；
- patch replay/reverse-check、synthetic integration 与 WorldSim 定向回归通过；当前回归为 `75 passed`；
- noncanonical r2 因前台 SSH 转 tmux 显式中止，r3 因 r2 遗留独立 session GPU 子进程被 idle preflight 拒绝；
  遵循 `PIVOT-F22` 精确回收后，r4 才作为 canonical，旧 terminal 不改写为 done。

该证据只授权冻结 D1 formal 协议；1000-step smoke 未执行冻结 held-out actor/boundary 质量合同，且 D1 Gaussian
更多，不能登记为方法改进或直接解锁 D2。

## F0 Instant NuRec canonical 审计收口

- official source checkout=`/root/autodl-tmp/third_party/instant-nurec-worldsim-v3-f0`，revision/tree=
  `1ce2288e646548e61fea6100bc58de3acd4bc8d0 / 96e36fa4772f5ddada37dc3decb1be9d2e595dc0`；16 个关键文件
  hash exact、git clean，协议/源码指纹测试 `8 passed`、WorldSim V3 联合回归 `241 passed`；
- 代码 Apache-2.0；权重 NVIDIA Open Model License；gated NCore 数据为 NVIDIA Autonomous Vehicle Dataset
  License 且需单独接受 terms。三个当前 weights-only PTH 已固定各自 HF commit、bytes、SHA-256 与 Xet hash；
- 论文/模型卡描述完整 static+dynamic+sky+per-camera ISP 研究模型；当前 standalone CLI 的实际输入为
  NCore V4 `.json/.lst`、FTheta camera、RGB/pose/intrinsics/mask/optional cuboids，不读 LiDAR；输出仅 static PLY，
  不导出 dynamic/sky/ISP/actor registry/trajectory/depth。网页 demo 不作本地 CLI 证据；
- formal smoke gate 固定 Python 3.11、uv、CC≥8.0、VRAM≥30,720 MiB、RAM≥32 GB、free disk≥100 GB、精确
  权重、合法 NCore input/terms、exact clean checkout 与 CLI help 全合取。任一失败时不得构造 inference command，
  不安装依赖、不下载权重/gated 数据、不启动 GPU；
- protocol/runner SHA=`2004a029...fd611 / 249f26d5...8e4a`；canonical=
  `20260809T192139Z__f0-instant-nurec-audit-s0-r1`，source=`ab76f19`，terminal=`done`，summary/manifest/terminal
  SHA=`d111c457...be37 / f1c76fdd...6a11 / 207758b9...15c6`；9/9 manifest artifacts exact；
- exact source、CC/system memory、CLI help 共形成 4/11 passed；失败项为 Python 3.11、uv、≥30,720 MiB VRAM、
  ≥100 GB free disk、exact weight、licensed NCore input、terms record。官方 focused tests=`33 passed` 与
  `37 passed / 15 failed`，15 项均因当前未配置环境缺 `shortuuid`；
- `inference_command_constructed=false`，torch/GPU/training/install/download 全未启动，OOM/kill=`0/0`。F0=
  `done_local_inference_not_executable_on_current_host`；F1=`conditional_not_unlocked`，当前转 R0。

## R0 formal 前协议冻结

- task/profile=`WS-V3-R0-INTEGRATION-01 / R0-INTEGRATION-v1`，seed=`0`；前置 closeout commit=`80b4f98`，
  protocol/runner SHA=`4fe20c31...7575 / d58c4008...c5ce`；
- exact 输入为 `5` 份协议、`51` 个 canonical evidence files、`3` 个 selected production files 和 `4` 个
  scene-0230 D2 held-out MP4，共 `63` 行；11 组 terminal status 与 23/23 frozen decisions 均 exact；
- selected P3 package=`159 files / 444,177,055 bytes`：133 static、24 actor、1 skeleton、1 manifest；全部
  158 payload 的 path/bytes/SHA-256 逐项通过；
- 12 项交付冻结为文档 snapshot、A0/A1/F0、`actor_quality.json`、A3 support、A4 deployment、A0→A4 主表、
  质量—规模—时间—显存 Pareto、负结果/边界、复现 manifest 与现有离线可视化索引；
- 最终 conclusion vocabulary 固定为 `calibration_native_or_off_preferred / actor_aware_supported /
  local_refine_limited_to_observed_support / deployment_pareto_supported / engineering_blocked`，并保留三场景、
  scene-0230 method evidence、D2 tradeoff、R1/P1 rejected、P2/P3 性能边界及 F0 no-inference 边界；
- R0 定向测试=`11 passed`，WorldSim V3 联合回归=`252 passed`。训练、推理、GPU、安装、下载、源 checkpoint/
  registry mutation、F1/P4/D3/D4/A3 追加实验均未授权。本条尚无 canonical R0 terminal。

## R0 canonical 收口

- canonical=
  `/root/autodl-tmp/runs/worldsim_v3/WS-V3-R0-INTEGRATION-01/20260809T194625Z__r0-integration-s0-r1`；
  source=`64e3d15ca30de44088c2f6fbfb6da048a31a4acf`，terminal=`done`，summary/manifest/terminal SHA=
  `3ffe99ea...15a7 / a9b052a6...1d90 / 207758b9...15c6`；28 files=`1,117,645 bytes`；
- 63/63 inputs、11/11 terminal states、23/23 decisions、12/12 deliverables 与 26/26 manifest files 全 exact；
  documentation snapshot 5/5 exact，P3 package 159/159 files、444,177,055 bytes 再次 exact；
- final chain=
  `A1-C0-off__A2-D2-boundary-priority__A3-R0-off__A4-P2-mixed__A4-P3-exact-chunk`；
- final conclusions 为 `calibration_native_or_off_preferred / actor_aware_supported /
  local_refine_limited_to_observed_support / deployment_pareto_supported / engineering_blocked`；12 条 claim boundary
  全 true，故这些结论不表示 D2 dominance、R1/P1 selected、P2/P3 render speedup、Instant NuRec local quality、完整
  world model、安全闭环或跨场景泛化；
- resource/no-launch 全通过：wall=`1.678173 s`、cgroup current=`30,389,452,800 bytes`、disk free=
  `42,325,843,968 bytes`、OOM/kill=`0/0`；torch 未导入，GPU/训练/推理/安装/下载均未启动；
- `next_action=none_plan_complete`。F1/P4/D3/D4/A3 R2–R4 保留未启动终态，不构成 V3.1 主计划缺口。

## V3.1 冻结任务状态

| Task ID | 状态 | 当前结论/门禁 |
|---|---|---|
| `WS-V3-P0-ROUTE-01` | done | `076ebdc`；单一 V3 计划、V2 冻结边界、链接与 Git 校验通过 |
| `WS-V3-A0-NATIVE-BASELINE-01` | done | 3/3 30k/等价 checkpoint、held-out、registry、actor/boundary、GS 与资源矩阵完成 |
| `WS-V3-F0-FEEDFORWARD-AUDIT-01` | done | canonical audit done；4/11 prerequisites；inference not-run；F1 conditional_not_unlocked |
| `WS-V3-A1-CALIBRATION-01` | done_off | 10/10 逻辑项、8/8 唯一训练；C*=C0；确认原始端点方向存在场景依赖 |
| `WS-V3-A2-ACTOR-DENSIFY-01` | done | D1/D2 fixed/matched 均为 tradeoff；A2*=D2 boundary-priority，D1 fallback；D3/D4 未启动 |
| `WS-V3-A3-LOCAL-REFINE-01` | done | R1 resource gate failed，diagnostic tradeoff；R1 rejected，A3*=R0/D2 exact alias；formal、R2–R4 未授权 |
| `WS-V3-A4-DEPLOYMENT-01` | done | P0/P5/P1/P2/P3 complete；P1 rejected；P2 mixed checkpoint + P3 exact chunk package selected |
| `WS-V3-R0-INTEGRATION-01` | done | canonical done；63 inputs、23 decisions、12 deliverables、26 manifest files、P3 package 与 no-launch exact |

## 机器与工作树

- GPU：NVIDIA GeForce RTX 3090，24,576 MiB；driver `580.105.08`；最近审计 0 MiB；
- cgroup memory：90 GiB，`oom=0 / oom_kill=0`；
- 数据盘：F0 canonical preflight free=`42,327,777,280 bytes`；
- A3 heldout r5、A4-P0 v1 r1、A4-P5 r1 与 A4-P2 r1 均保留 blocked；P0/P5/P2 canonical r2 与 P3 canonical r1 exit=`0`，GPU 无遗留进程；
- 当前非 V3 文档 dirty files 属于 V2 M5，必须保留。

## 计划终态与归档

`WS-V3-R0-INTEGRATION-01` 已 `done`，V3.1 当前为 `none_plan_complete`。F1、P4、D3/D4 与 A3 formal/R2–R4
保持未解锁；除非未来以新任务、新协议和新授权启动，否则不得恢复为当前动作，也不得改写既有 terminal。
V3.1 的计划与 R0 收口快照已归档至
[`archive/2026-08/worldsim-v3.1/`](archive/2026-08/worldsim-v3.1/README.md)；当前没有新的研究计划或实验授权。
