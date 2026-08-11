# WorldSim V4 B0 Matched Baseline Audit

## 当前结论

`WS-V4-B0-MATCHED-BASELINES-01` 保持 `running`。统一评测、scene-level 统计、工程指标、6 个 development
scenes 的 DriveStudio 输入与 sky masks 已经落地。2026-08-12 复核发现：早先 r17/r20/r22/r24/r26/r28
使用的是 `test_image_stride=10`，不能满足冻结的 `sample_index mod 5` 三分区合同；这些 run 只保留为
native/provenance，不再计入 matched coverage。当前严格 coverage 为 `V3.3 1/6 / StreetGS 3/6 / AD-GS 1/6`。

| baseline | 当前 executable scenes | 需要 | 事实边界 |
|---|---:|---:|---|
| V3.3 frozen | 1 | 6 | scene-0230 canonical release 与其外部 base 仍可解析；其余五 scene 未物化完整链 |
| Native StreetGS | 3 | 6 | scene-0230/0242/0255 strict mod5 30k r32/r46/r48 可执行；旧 stride=10 runs 只作协议不匹配 provenance |
| AD-GS | 1 | 6 | scene-0230 strict train-only 60k r44 已由内容寻址审计登记；其余五 scene 尚待重建 |

## 冻结协议

- development scenes：`scene-0230/0242/0255/0048/0994/0139`，精确来自 D0 cohort；
- 同 scene、同 `sample_index mod 5` train/development/heldout；sensor `1600×900`、source downscale=2、
  model/metric `800×450`；不得读取 test quality；
- 图像主指标固定 `PSNR/SSIM/LPIPS-Alex`，区域固定 global/static/actor/boundary/edit_roi；
- baseline 区域生成固定为 actor=`dynamic_masks/all>0`、static=`not actor and not egocar`、boundary=dynamic mask
  的 L1 半径 3 px 形态学带；无编辑 baseline 的 edit_roi 固定为空并返回 `undefined`；
- 无 GT 或空区域返回 `undefined`，禁止以生成结果自身作 GT；
- 主统计单位固定为 scene；failed/blocked/abstain 保留 denominator；
- 工程指标由 raw timing/count/resource rows 派生，不允许手填 success/yield/retry 比率。

## Diagnostic run

- run：`/root/autodl-tmp/runs/worldsim_v4/WS-V4-B0-MATCHED-BASELINES-01/20260811T090951Z__b0-inventory-diagnostic-s0-r1`
- terminal：`blocked / matched_baseline_assets_incomplete`
- matrix/summary/manifest/status/fingerprint/inventory SHA-256：
  `8f5d31e65f0710cfdc5751aada3381ee48eed42736d59403227b628fecbf9eea` /
  `b25a9c35b726f9aa1a5b75ed3ccf84998e8703b99a05f9c548fb8619a0df8e93` /
  `ce7e437fea0951292e5ef2b69d4d84d1ffff767e80b176b8724cc87e4f93ba49` /
  `1c1e702f38fe7860d798ccb8bc7fa4217b903e411407e1d46af93549de77fadb` /
  `3bbbaf960e4072b846028f5293db2ee6ee9bc9f7b5bef047ff4d0df878f7145b` /
  `40a322c6ee8190823bba3cbfa7339fc4ffd66b00d492d5cacc57aa0c26e31a0b`
- run size：`53,158 bytes`；training/model inference/test quality 均为 `false`。

## 六场景数据与 sky-mask 里程碑

- 公共 nuScenes archive 缺失三场景的 canonical extraction run：
  `20260811T092132Z__b0-data-extract-missing3-s0-r3`，terminal=`done`，`5,264/5,264` members，
  fingerprint=`c51f41622ab4023ccaea656836576855d717be2b93c2d4a8692eba2562379e38`；只扫描官方 10 个 tar shard，
  不下载数据、不读取 test quality；
- preprocess canonical runs：scene-0048/0139/0994 分别为
  `20260811T102622Z__b0-preprocess-scene0048-s0-r5`、
  `20260811T102857Z__b0-preprocess-scene0139-s0-r6`、
  `20260811T103142Z__b0-preprocess-scene0994-s0-r7`，三者均 `done`；r4 因上游 `_10Hz` 输出目录合同
  未对齐而 blocked，未覆盖；
- canonical processed root=
  `/root/autodl-tmp/data/worldsim_v4/drivestudio_processed_10Hz/trainval`；scene index
  `045/110/179/191/204/752` 均精确为 `1,176 RGB / 196 LiDAR / 588 sky masks`；
- SegFormer 固定 `nvidia/segformer-b5-finetuned-cityscapes-1024-1024@2c6f153e...`。远端官方网络恢复 r10
  因 `Errno 101` blocked；本机从官方固定 URL 下载后按 bytes/SHA256 exact 校验并传入远端 staging，r11
  `20260811T110746Z__sky-model-restore-staging-exact-s0-r11` 以
  `restore_mode=codex_local_official_staging_exact / network_attempted=false` 完成，fingerprint=
  `15c200fdd8bd6eeb5999d639076b3c5d0d38c76523f1c7f187181ec7b4790bfd`；
- sky-mask runs：r12 因 preprocess 预建空 target 被旧合同拒绝并保留 blocked；修复只接受空目录、任何已有
  mask/partial 仍 fail-closed。scene-0048/0139/0994 canonical r13/r14/r15 均完成 `588/588`，fingerprint=
  `c35a9dbe...c3eb / 5f6a2fe7...c702 / 455dcb20...e6c7`，峰值 GPU 均 `2,876 MiB`，全程离线且
  test quality 未读。

## StreetGS profile 门

- r9 已真实启动训练，但在 iteration 之前因 scene-0048 缺 `sky_masks/000_0.png` blocked；GPU 子进程已退出；
- 修复数据链后 canonical profile=
  `20260811T111810Z__streetgs-scene0048-profile100-s0-r16`，terminal=`done`；100 steps / `90.8965 s`，
  checkpoint=`340,298,602 bytes / SHA256 446297b897024b326def46c66f05e7e30cf2736dcc767789180652e8498203af`，
  Gaussian=`Background 1,074,426 / RigidNodes 4,746`，峰值 GPU=`9,004 MiB`、cgroup=`10,867,355,648 bytes`，
  fingerprint=`62ebeb1ef11dc6785e1031f1555188c97e3e17d426101a316584abf3765c2eda`；
- profile 只证明 30k 训练链和资源前置可执行，不计入 formal coverage，不登记图像质量改进。

## StreetGS 旧六场景 native provenance（不计 matched）

| scene | run | wall s | checkpoint bytes | SHA256 | Background / Rigid | peak GPU MiB |
|---|---|---:|---:|---|---:|---:|
| scene-0230 | r17 | 3,118.35 | 396,257,270 | `fba28355...0162f` | 1,140,862 / 168,849 | 23,720 |
| scene-0242 | r20 | 2,069.20 | 306,226,038 | `3c292d74...413b` | 844,231 / 86,579 | 12,692 |
| scene-0255 | r22 | 2,442.84 | 451,821,046 | `ee5450b4...428d` | 1,507,559 / 41,179 | 24,092 |
| scene-0048 | r24 | 2,250.89 | 340,189,878 | `dede872b...cf3d` | 1,059,206 / 19,020 | 24,000 |
| scene-0994 | r26 | 1,877.13 | 297,410,742 | `f7965426...b7a0` | 897,077 / 1,029 | 16,744 |
| scene-0139 | r28 | 2,129.88 | 325,005,110 | `cb6d4254...a8f2` | 1,005,857 / 8,713 | 23,754 |

- 六个 run 均为独立原生初始化、step=`30000`、means finite、OOM/kill=`0/0`；但配置
  `test_image_stride=10` 未显式物化冻结的 mod5 分区，余数 4 的 heldout 帧可能进入训练；
- scene-0255 最高 sampled GPU=`24,092 MiB`，没有提高门槛或发生 OOM；scene-0994 actor 很稀疏但 final
  RigidNodes=`1,029` 非空，按预注册合同保留场景差异，不补点、不拒绝；
- r29 的 `StreetGS=6` 是被后续复核推翻的旧 inventory 结论，不覆盖或删除；matrix 已把六个 run 移入
  `protocol_mismatch_runs`，理由统一登记为 stride=10 不满足 strict mod5。

## StreetGS strict matched 重跑

- scene-0230 canonical strict run=
  `20260811T154831Z__streetgs-scene0230-matched-formal30k-s0-r32`，30k done，wall=`3,200.0184 s`；
- checkpoint=`386,410,166 bytes / SHA256 766648bf954142dc6f4cac8b767623fdc5bff4e6eed766cc45d2a0680af97cd1`，
  Gaussian=`Background 1,095,606 / RigidNodes 172,264`，means finite；peak GPU=`23,892 MiB`、peak cgroup=
  `15,281,917,952 bytes`、OOM/kill=`0/0`；fingerprint=
  `8b1a43b74f727658b0f2d9d1d00e72a2cc62fdeed886fbb0ab870b579788dde3`；
- run 配置 `render_test=false / render_full=false`，无 Test/Full Set 评测行或 render 文件，summary 明确
  `test_quality_read=false`；训练结束日志的通用 Pixels 聚合只包含 train metrics；
- corrected inventory=`20260811T165009Z__baseline-matched-correction-s0-r33`，terminal=
  `blocked / matched_baseline_assets_incomplete`，coverage=`StreetGS/V3.3/AD-GS=1/1/0`；inventory/fingerprint SHA=
  `73d3654450621d48a60a18ae296e61e1fb5ced5f211763ff3473bcb14bbbea9e /
  c19fba13a7f8143608c56a318533d23d2894e888bbf87d8a5652ccaf0e285853`。
- scene-0242 strict run=`20260811T210253Z__streetgs-scene0242-matched-formal30k-s0-r46`，30k done，wall=
  `1,998.0482 s`；checkpoint=`302,953,462 bytes / SHA256 dd41a34d877f64abf39c50ecf78d04fcccf238f4a2898cb45e5de5bba5452bc0`，
  Gaussian=`Background 824,583 / RigidNodes 92,170`，means finite；peak GPU/cgroup=
  `17,530 MiB / 23,842,824,192 bytes`，200 个资源采样 OOM/kill/max/high 均为 `0`；无 test/full render，
  `test_quality_read=false`，fingerprint/manifest/summary=`9c193ddf...2236c / 55c88461...3cade / cfb4308d...2c503`；
- registry 提交=`9d587de`；r47=`20260811T213829Z__baseline-streetgs-scene0242-registration-s0-r47` 得到 coverage=
  `StreetGS/V3.3/AD-GS=2/1/1`，inventory/fingerprint SHA=`89c72659...eafad / b91f7c76...712d6`，
  terminal 仍为预期的 `blocked / matched_baseline_assets_incomplete`，project clean。
- scene-0255 strict run=`20260811T214009Z__streetgs-scene0255-matched-formal30k-s0-r48`，30k done，wall=
  `2,392.0649 s`；checkpoint=`444,340,086 bytes / SHA256 dba24982a3f25e162b5e293165258a588cf9bd7a49e54e05d0d052de703cb2d2`，
  Gaussian=`Background 1,478,401 / RigidNodes 38,721`，means finite；peak GPU/cgroup=
  `23,932 MiB / 24,132,476,928 bytes`，239 个资源采样 OOM/kill/max/high 均为 `0`，无 test/full render；
  fingerprint/manifest/summary=`c5c3ebf4...163f1 / 7325ab82...b4a31 / 7e3636b6...73834`；
- registry 提交=`af8efb9`；r49=`20260811T222138Z__baseline-streetgs-scene0255-registration-s0-r49` 得到 coverage=
  `StreetGS/V3.3/AD-GS=3/1/1`，inventory/fingerprint SHA=`79b6b1d0...c86c85f / bd822e61...9641a`，project clean。

## AD-GS exact source、权重与环境恢复

- official source=`/root/autodl-tmp/third_party/AD-GS@9a208512e49c8ddbaa20387921d9648adcd21cb4`；兼容补丁只含
  `scripts/colmap.py/scripts/flow.py/scripts/run-dpt.py/scripts/semantic.py/train.py/utils/flow_utils.py`，patch SHA=
  `b2614c8bf720e041b2d5abaeb20305dbc30057b690d567a3ff646d5d9719513b`；其中 flow preprocess 支持正式
  无可视化模式、训练 import 不再强依赖可选诊断包，`train.py` 增加
  `--disable_test_evaluation`，strict 训练不触碰 development/heldout 质量；
- Depth-Anything-V2=`a561b849...c71bf`，DPT weight=`1,341,395,338 bytes /
  a7ea19fa0ed99244e67b624c72b8580b7e9553043245905be58796a608eb9345`；CoTracker=`82e02e80...dda4d`，
  weight=`101,890,938 bytes / 2670d4562ed69326dda775a26e54883925cd11b6fc9b24cb7aa9f8078bce7834`；
- environment restore=
  `20260811T165030Z__adgs-environment-restore-r34`，terminal=`done`；从冻结本地环境离线复制并构建
  `simple_knn` 与 `diff_gaussian_rasterization`，CUDA forward/backward smoke=`passed`，torch=`2.1.2+cu118`、
  device=`RTX 3090`、visible Gaussians=`1,024`、smoke peak GPU=`12.63 MiB`；扩展 SHA=
  `019c87b6...15190 / be64aef4...b0ed99`，OOM/kill=`0/0`，fingerprint=
  `c7e0d5be3a066416a8fde8ccaaca1ab7a45252a6b8c3ffb3b98b19dfb5693b46`；
- r34 记录的兼容补丁 SHA 为前一版 `01145883...3d8e4`；后续 `b2614c8b...9513b` 改变 Python flow/训练
  的诊断依赖与 runner 命令，不改变 r34 已编译的两个 Gaussian CUDA extension；新的 runtime 证据由 r40/r42 独立补齐；
- 环境成功只解锁 preprocess/profile，不计入 executable scene coverage；historical aggregate 仍不得写入 matched 主表。
- preprocess r35 因包装器预建 run 目录、r36 因环境构建遗留 untracked build 目录、r37 因可选 `flow_vis`
  诊断依赖分别 fail-closed；三者均未训练或读 dev/heldout。r37 partial 已移入规定 backup，旧 run/partial 均未覆盖；
  后续正式 flow 固定 `--disable-visualization`，CUDA extension 恢复改为 run-local source copy。
- scene-0230 canonical train-only preprocess=
  `20260811T171507Z__adgs-scene0230-preprocess-s0-r38`，terminal=`done`；adapter/depth/segment/flow wall=
  `80.7806/100.9956/10.0506/3,046.9930 s`，文件计数=`image/semantic/sky/depth/flow=354/354/354/354/285`；
  peak GPU=`20,112 MiB`、peak cgroup=`22,384,893,952 bytes`、OOM/kill=`0/0`，无 `flow.mp4`；fingerprint/manifest SHA=
  `d44a053039f5b0899ab44ab6e52cb311630e22c6e319c1ccaca279809fbf2c37 /
  cefae2306265806043314cc4bf77dc83d657e5d72efa1f2d176f6e7763c05873`；项目在 terminal 时
  clean=`54ca265723d55226c0af4e634370a612c06c10a8`，development/heldout/test quality 均未读；
- run-local extension source 修复=`9f839fd`，冻结区域协议与 development-only scene evaluator=`abd82d8`；
  baseline/AD-GS/region/evaluator 联合定向测试在内容寻址 registry 纳入后=`50 passed`。
- profile r39 在 iteration 前因 `utils/flow_utils.py` 全局导入可选 `flow_vis` blocked；r40 从 exact 本地
  `roma-1.5.7-py3-none-any.whl`（`25,627 bytes / a322b032...f5a2`）离线补齐 Python runtime，terminal done；
  r41 随后在 iteration 前暴露 inherited PyTorch3D binary 缺少 sm86 KNN kernel，保持 blocked；两者均未读 dev/heldout；
- r42=`20260811T183048Z__adgs-environment-pytorch3d-sm86-r42` 从 clean
  `pytorch3d@2f11ddc5 (v0.7.5)` 复制到 run-local source，离线重编 `sm_86`；build wall=`1,151.39 s`，
  新 `_C.so=10,313,184 bytes / eca71e2c...e3084`，真实 `knn_points` + Gaussian forward/backward smoke passed，
  OOM/kill=`0/0`，fingerprint=`03ec74e888265ee97974e4be6a021d4988f3b71acc274e4db296f4f12205f7fb`；
- scene-0230 profile100 r43=`20260811T185145Z__adgs-scene0230-profile100-s0-r43` done，train stage=
  `40.3595 s`、peak GPU=`6,012 MiB`、peak cgroup=`30,991,519,744 bytes`、OOM/kill=`0/0`；checkpoint
  `point_cloud/deform/env=82,578,770/114,955,577/805,307,528 bytes`，SHA=
  `bc364930...16c5 / 8205e276...e5ab / 45644cb4...8e77`；fingerprint/manifest=
  `1a6e32673e365d9be6aeea2782f86706c836256f8749af550bd8ea3991ff268d /
  080ca14d444f4104841776fa3255fff7e9cae44d8f6917cf7381a5501aa58396`，development/heldout/test quality 均未读；
  profile 只解锁 formal，不计 executable coverage。
- scene-0230 formal60k r44=`20260811T185600Z__adgs-scene0230-formal60k-s0-r44` done，train stage=
  `7,054.6221 s`、peak GPU=`16,692 MiB`、peak cgroup=`33,680,572,416 bytes`、701 个资源采样的
  OOM/kill/max/high 均为 `0`；checkpoint `point_cloud/deform/env=413,905,347/435,921,657/805,307,528 bytes`，SHA=
  `f17ed27f...a0cbb / c725f952...c84b0 / c3233b71...e4d34`；fingerprint/manifest/summary SHA=
  `0e69e2b9...e8d28 / 9fa1f2c1...d8f42 / 7774d712...b9a8f`。训练只物化 train remainder，
  `development_content_read=false / heldout_content_read=false / test_quality_read=false`，项目 terminal HEAD=
  `7001b5c` 且 clean；
- fail-closed checkpoint registry=`904e395`：StreetGS/AD-GS 只有在 runtime/source/patch 精确、formal step 精确、
  checkpoint bytes/SHA 精确且位于登记 run 内、fingerprint/manifest SHA 精确时才计 coverage；联合定向测试=`50 passed`；
- corrected inventory r45=`20260811T205840Z__baseline-adgs-formal-registration-s0-r45`，terminal=
  `blocked / matched_baseline_assets_incomplete`，coverage=`StreetGS/V3.3/AD-GS=1/1/1`；inventory/fingerprint SHA=
  `4bf7cf68213ab068370a3251ee76005156631912b63d673769bcdde63333ad6b /
  3db524d235496d68945223fcc223e57e5b0eb8197a8afe6f310ef5a79ee349e5`，project HEAD=`904e395` 且 clean。

## 下一动作

依次完成 StreetGS 其余三场 strict mod5 重跑，再补齐 AD-GS 与 V3.3 其余五场景 same-split。
B0 只有在三种方法均达到 6/6 且统一 evaluator 生成完整 scene rows 后才可
`done`，M1 在此之前不启动。
