# Research Status

## V5.2.1 Base Badcase Census 已授权、待执行（2026-08-20）

- task family=`WS-V521-P0..P10`，status=`pending`，executed=`false`；执行计划=
  `docs/WORLDSIM_V5_2_1_BASE_BADCASE_CENSUS_PLAN.md`（`54,123 bytes / sha256 ce332dea...3c56`），
  authoring source base=`a9dede0b327c20d744a1dea7d5d6eb99b4203012`。实际执行 HEAD/branch 必须由 P0 重新冻结，
  不得 reset 到 V5.1 authoring base `fc07b99...`。
- scope=exact AD-GS/StreetGS asset audit → quality-blind shared sample partition freeze → Discovery-only frame/camera/actor/
  temporal census → axis-wise deterministic badcase registry/panels → M1/M2/M3 historical re-audit → one-shot internal
  Confirmation → closeout。V5.2.1 是诊断/问题复核，不提交算法 candidate。
- 数据锁：Stage H/BKI 与 V5.1 rejected arms 不复开；fresh validation/test/KITTI method-tuning quality 全程 unread；
  Confirmation 来自 development 内部且不冒充 scene-disjoint heldout。P0 只冻结分区算法，P1 在 asset/split 审计后、
  任何 Tier D quality read 前冻结具体 membership。
- 核心防重复引用=`V1-F01/F03, V4-F34/F39/F42/F45/F47/F49, V5-F07/F31-F33/F47/F48/F51/F52/F57/F59,
  V51-F31/F37/F42/F63/F65/F66`；`BC-*`/`BCE-*` 只作 badcase 个案/事件 ID，不是新的 failure-ledger ID。
- 当前唯一下一步=`WS-V521-P0-BASE-CENSUS-FREEZE-01`：先渐进式读取事实源、审计真实 repo/GPU/disk/cgroup、
  冻结计划/质量锁/分区协议并建立 V5.2.1 执行分支；不得直接跳到 renderer、P2 quality 或 V5.2.2 方法设计。

## V5.1 M1 已收尾；Stage H pending 但由 V5.2 scope 取代（2026-08-20）

- terminal=`closed_without_promoted_candidate`；U2/B3 仅保留为 V5.2 matched comparator。M2/M3、validation、test 与
  KITTI method tuning 在 V5.1 全程未解锁；authoring base=`fc07b99`，closeout freeze=
  `configs/worldsim_v51/m1_closeout_v1.yaml`。
- 累计实证指向 `effective_observations_are_structurally_missing`：progressive r018 的 scene-balanced
  ΔBoundary-F1/ΔIoU/ΔFN=`+0.0002196/-0.0714543/+0.1694766`；simple voxel super-primitive 虽提高
  observation density，r022 相对 U2/B3 仍为 `-0.0002566/-0.0925468/+0.1899473`；Gaussian Grouping r043
  仅 1087 通过，0471/0379 identity recall=`0.080747/0.202933` 且 persistence=`0/0`；Trace3D r047 的 16 次
  alpha 调用出现两个 exact 值。
- Stage H task status=`pending`、executed=`false`、disposition=`superseded_by_v5.2_scope`。BKI 改变传播方式但不引入
  新证据源，因此当前 expected gain=`low`；这是基于累计证据的 scope 决策，不是 BKI empirical rejection。
- 下一研究入口只能是 V5.2 的新 observation-source scope：先冻结 coverage、identity persistence 与 fresh-process
  reproducibility，再决定是否重开 propagation/completion。收尾归档=`docs/archive/2026-08/worldsim-v51-m1-closeout/`；
  failure delta=`V51-F66`。

## V5.1 Stage G Trace3D faithful operator 已拒绝并关闭（2026-08-18）

- canonical r047=`20260819T000000Z__m1-stage-g-g0b-trace3d-determinism-s20260814-r047`，source/tree=
  `e427022...7ec1/2b26037a...934e`；8 个 fresh processes 共 16 次 foreground alpha 调用出现两个 exact vector，
  values=`0.0056084292/0.0267562941`，unique=`2 > gate 1`。hard 16 次均为 `[0,1]`，其余 gates 全 PASS。
- official source hazard=`plain global += 2 /atomicAdd 0`；这支持 data-race 风险但不证明唯一根因。没有 threshold search、
  source patch、真实 checkpoint/camera/image/mask/quality/training。
- outcome=`rejected`；独立 audit=`20260819T003000Z__stage-g-g0b-r047-audit.json /3,880 bytes /
  98c72ba7...d31 /PASS`，逐一重哈希 8 个 process rows 并重算裁决。`V51-F65` 关闭为 faithful operator rejection。
- freeze=`configs/worldsim_v51/stage_g_trace3d_faithful_operator_rejection_freeze_v1.yaml`。不进入 real adapter、split/prune
  或 patched Trace3D；r047 当时的 failover 指向 `WS-V51-M1-H-GRAPHFREE-01`，其后已由 2026-08-20 V5.1 closeout
  冻结为 `pending + superseded_by_v5.2_scope`，未执行 BKI preflight。

## V5.1 Stage G G0b r047 cross-process determinism forensic 已预注册（2026-08-18）

- formal target=`20260819T000000Z__m1-stage-g-g0b-trace3d-determinism-s20260814-r047`；auth=r046 freeze
  `3,896 bytes /79717939...1ba`，exact extension=`1,386,928 bytes /f81ef6d6...f53`。
- 固定 `8` 个 sequential fresh Python/CUDA processes；每个进程对同一 synthetic input 做 background hard×1、foreground
  hard×2、foreground alpha×2。hard 与 alpha 分别要求跨进程/进程内唯一 vector 数=`1`，同时要求 label response、finite/
  bounded、input immutability；不调 scale/mask/threshold/process count。
- source hazard 同步冻结：`id_trace.cu=12,695 bytes /7d8ef4a5...630`，`trace_renderCUDA` 内 global weight
  plain `+=` 两处、`atomicAdd` 零处；该事实只作根因候选，不预判 forensic 结果。
- PASS 才进入 real tensor/camera adapter preflight；FAIL 则 faithful Trace3D operator=`rejected`，不 patch 上游，自动转
  `WS-V51-M1-H-GRAPHFREE-01` 的 faithful BKI source/method preflight。checkpoint/camera/image/mask/quality/training=false。

## V5.1 Stage G G0a r046 capability PASS；跨进程确定性 forensic 待执行（2026-08-18）

- canonical r046=`20260818T230000Z__m1-stage-g-g0a-trace3d-capability-s20260814-r046`，source/tree=
  `7c7ee17...33bd/4987f46d...d6e`；official source 1,490 entries clean/unpatched，wheel=`378,727 bytes /
  471786c6...287`，extension=`1,386,928 bytes /f81ef6d6...f53`。
- frozen synthetic gate 全 PASS：background/foreground hard=`[1,0]/[0,1]`，hard repeat exact，alpha finite/positive/bounded，
  input tensors bitwise immutable；wall/GPU/cgroup=`73.872497 s /47,616 /13,375,209,472 bytes`。
- independent audit=`20260818T233000Z__stage-g-g0a-r046-audit.json /3,149 bytes /ecc3d061...81c /PASS`；它重放
  hard 结果一致，但 alpha 从原 run `0.0267562941` 变为 `0.0056084292`。该项不在 r046 的跨进程 exact gate 中，故不
  倒写 capability PASS；但它阻止真实 adapter 自动解锁。
- official `id_trace.cu` 对 global weights 使用普通 `+=`，与漂移相容但根因尚未证明；登记 `V51-F65`。freeze=
  `stage_g_g0a_trace3d_reverse_tracing_capability_freeze_v1.yaml`。下一步只预注册多 fresh-process determinism forensic；
  不 patch source、不读 checkpoint/camera/quality。FAIL 则 faithful Trace3D 关闭并转 `WS-V51-M1-H-GRAPHFREE-01`。

## V5.1 Stage G G0a r046 official reverse-tracing capability 已预注册（2026-08-18）

- formal target=`20260818T230000Z__m1-stage-g-g0a-trace3d-capability-s20260814-r046`；auth=r045 freeze
  `3,207 bytes /3b4e9504...0dc1`。从 official repo `7465ad94...c442` 以 `git archive` 导出 exact
  `diff-id-rasterization` subtree（`1,490` tracked entries /ls-tree SHA `8e37e910...9ee`），禁止 source patch。
- 仅在隔离 target 构建：DriveStudio Python 3.9、torch `2.1.2+cu118`、CUDA/nvcc `11.8`、GCC `11.3`、
  Ninja `1.13`、RTX3090 `sm_86`、`MAX_JOBS=12`；pip `--no-index --no-deps`，不改原环境或 official checkout。
- synthetic operator gate 只有 1 个中心 2D Gaussian 与 `32×32` 全背景/全前景 label maps；要求 class response、hard-count
  repeat exact、alpha-weight finite/bounded 与所有 input tensor bitwise immutable。它只证明官方 kernel 在本机可构建并有基本
  reverse-tracing 能力，不证明真实 WorldSim tensor/camera adapter 或质量。
- real checkpoint/camera metadata/image/mask/quality/training/parameter mutation/validation/test/KITTI=false。PASS 后仅允许预注册
  frozen U2/B3 real tensor/camera adapter preflight，仍不读质量。

## V5.1 Stage G G0 r045 source/method preflight PASS 并冻结（2026-08-18）

- canonical r045=`20260818T220000Z__m1-stage-g-g0-trace3d-source-recovery-s20260814-r045`，source/tree=
  `ed39600...5651/75932029...52ef`；exact reuse paper/repo，stdlib page markers=`11`，network/redownload=false。
- repo=`7465ad94...c442/tree 22d30d19...a05d/clean`，2 个 submodule pointers 均未初始化；5 组 tracing/
  CUDA accumulation/patch repair/ambiguous refinement/density-control 源码合同共复核 `57,321` bytes，marker/hash 全通过。
- summary/report/manifest SHA=`5e15c3df...b13/b0ae58cc...0fd/70329fab...dfd`；独立 audit=
  `20260818T223000Z__stage-g-g0-r045-audit.json /2,311 bytes /053cf574...d3b /PASS`。资源门全过，wall=
  `0.051832 s`、cgroup=`12,470,280,192 bytes`、disk free=`79,850,524,672 bytes`。
- `V51-F64` 已解决；r044 仍保留 running 中断证据。freeze=
  `configs/worldsim_v51/stage_g_g0_trace3d_source_method_preflight_freeze_v1.yaml`。PASS 只允许预注册 immutable-base
  reverse-tracing operator capability；不等于上游训练复现，split/prune/density-control/training/quality/validation/test/KITTI=false。

## V5.1 Stage G G0 r045 exact-asset recovery 已预注册（2026-08-18）

- formal target=`20260818T220000Z__m1-stage-g-g0-trace3d-source-recovery-s20260814-r045`；auth=r044 closeout
  `2,211 bytes /19026cdb...15dd`。只复核并复用 paper=`2,390,825 bytes /d50eda07...47e4` 与 clean repo
  `7465ad94...c442/tree 22d30d19...a05d`，network/redownload/delete=false。
- 唯一实现变化是以 Python 标准库 regex `/Type\\s*/Page\\b` 统计 PDF page markers，替代缺失的 `pdfinfo`；接受范围
  预先冻结为 `8..30`，不按结果修改。
- source semantic audit 显式绑定 `gaussian_renderer/trace.py`、CUDA `id_trace.cu`、`merge_patches.py`、
  `remove_ab_gaus.py` 与 `train_gaus.py` 的 exact markers/hash，区分 reverse tracing、patch reduction/repair、ambiguous
  split/prune 与 ordinary density control。该轮仍不执行 upstream source、不 init submodules、不读 image/mask/quality。
- immutable-base 边界、H/S/C/validation/test/KITTI=false 与 M2/M3=pending 原样保持；成功只解锁新的 frozen-base
  reverse-tracing operator capability preflight 预注册，不解锁 split/prune/training。

## V5.1 Stage G G0 r044 因缺少 pdfinfo 中断（2026-08-18）

- r044 在 official PDF 与 repo=`7465ad94...c442/tree 22d30d19...a05d` 已原子发布后，调用不存在的 `pdfinfo`
  时报 `FileNotFoundError`；run 仅有 resolved/status(running)/start-event，无 source report/summary，保持原状不补写。
- published paper=`2,390,825 bytes /d50eda07...47e4`；repo clean。没有执行 Trace3D source、init submodules、下载模型
  或读取 image/mask/quality；这是 `V51-F64` 工具可用性 blocked，不是 Trace3D 方法失败。
- recovery 只允许新目录 exact 复核并复用已发布资产、以标准库 PDF page marker 计数替代 external `pdfinfo`；禁止删除/
  重下资产或扩大读集。closeout=`stage_g_g0_trace3d_source_method_preflight_r044_closeout_v1.yaml`。

## V5.1 Stage G G0 Trace3D source/method preflight 已预注册（2026-08-18）

- task=`WS-V51-M1-G-AMBIGUITY-01`，formal target=`20260818T210000Z__m1-stage-g-g0-trace3d-source-s20260814-r044`；
  auth=r043 rejected closeout `3,645 bytes /27b744e7...1a0e`。official repo=`trace-3d/Trace3D@7465ad94...c442`，
  official ICCV 2025 paper 从 CVF URL sequential/partial/atomic 获取。
- 本轮只冻结 paper/repo/license/files/submodule pointers/dependency 与方法语义：GIT reverse rasterization、跨共视图
  majority-vote patch merge、ambiguous Gaussian 与原论文 split/prune density-control 边界；不 init submodules、不执行源码/
  下载模型/读 image-mask-quality。
- normative immutable base 初始只允许 reverse-tracing capability 与 no-quality disagreement diagnostic；原论文 split/prune/
  density control、geometry/appearance/opacity/pose update 全锁定。因此后续是 Trace3D-style diagnostic adapter，不冒充 full
  upstream training reproduction；training/H/S/C/validation/test/KITTI=false，M2/M3=pending。

## V5.1 Stage F F0l r043 Gaussian Grouping 已拒绝并关闭（2026-08-18）

- canonical r043=`20260818T200000Z__m1-stage-f-f0l-quality-alignment-s20260814-r043`，source/tree=
  `70293de...e4f0/d1d6ddfb...349c`；outcome=`rejected`，独立审计 scene pass vector=`[FAIL,PASS,FAIL]`。
- 0471 metrics=`8 tracks/19 actor-views/428,240 support；coverage 0.122784；one-to-one recall 0.080747；
  efficiency 0.937354；persistence 0`。1087=`2/4/16,869；0.859091；0.505009；1.0；0.5`，全门 PASS。
  0379=`3/7/80,041；0.238278；0.202933；1.0；0`，FAIL。
- audit=`20260818T203000Z__stage-f-f0l-r043-audit.json /4,210 bytes /f478fbd9...4320 /PASS`；freeze=
  `configs/worldsim_v51/stage_f_f0l_train_only_quality_identity_alignment_freeze_v1.yaml`；failure=`V51-F63`。
- 该结论是当前三场、冻结 view order/weak-support adapter 上的算法 quality/alignment reject，不是工程 blocked，也不外推
  普遍 Gaussian Grouping 失效；F1/F2 与 identity training 永久停止。下一任务=`WS-V51-M1-G-AMBIGUITY-01`，只预注册
  Trace3D official source/method/immutable-base adapter preflight；H/S/C/validation/test/KITTI=false，M2/M3=pending。

## V5.1 Stage F F0l train-only quality/identity-alignment gate 已预注册（2026-08-18）

- formal target=`20260818T200000Z__m1-stage-f-f0l-quality-alignment-s20260814-r043`；auth=r042 freeze
  `2,922 bytes /e6b54e94...b9c4`，input manifest=`45 views /90 projections /6640b5e1...6817`。
- 首次且仅按冻结分母读取 `45 candidate +45 dynamic-union` masks；不读 RGB。box rasterization=`floor low/ceil high/
  half-open/clipped`；重叠 actor boxes 像素排除。dynamic union 只给 actor foreground support，不提供 instance identity。
- 每场以 3D instance token 对 positive DEVA short ID 做最大权一对一匹配；冻结门为 eligible tracks/views≥`1/2`、
  foreground coverage≥0.70、one-to-one recall≥0.35、assignment efficiency≥0.75、persistence≥0.50，三场全过才 PASS。
- PASS 只解锁 frozen-base identity-training smoke 预注册；任一门 FAIL 则 faithful Gaussian Grouping 在当前 train-only
  质量/对齐证据上 rejected，并关闭该路线、转 Trace3D source preflight。threshold search/training/H/S/C/validation/test/
  KITTI/F1/F2=false，M2/M3=pending。

## V5.1 Stage F F0k r042 quality/alignment denominator PASS 已冻结（2026-08-18）

- canonical r042=`20260818T190000Z__m1-stage-f-f0k-quality-input-freeze-s20260814-r042`，source/tree=
  `858e0d1...64f0c/8df9f291...996d`；45-view exact assets 与 90 个 projected actor boxes 通过独立 hash/geometry replay。
- input manifest=`103,846 bytes /6640b5e1...6817`；audit=`1,937 bytes /108bb60a...082b /PASS`；verified
  asset logical bytes=`8,111,447`。candidate/dynamic/image pixels 与 quality metrics 全部保持 unread。
- freeze=`configs/worldsim_v51/stage_f_f0k_quality_alignment_input_freeze_freeze_v1.yaml`；后续 F0l 必须逐场使用既定
  `0.70/0.35/0.75/0.50` coverage/one-to-one recall/assignment efficiency/persistence 门。dynamic union 仅为 weak
  foreground support，不是 instance truth；F0l 结果前不授权 training/H/S/C/validation/test/KITTI/F1/F2，M2/M3=pending。

## V5.1 Stage F F0k quality/identity-alignment 输入冻结已预注册（2026-08-18）

- formal target=`20260818T190000Z__m1-stage-f-f0k-quality-input-freeze-s20260814-r042`；auth=r041 freeze
  `3,811 bytes /12692b2e...bd0c`。本轮只冻结 45 candidate masks、45 DriveStudio dynamic-union masks、相机矩阵、
  instance metadata 与 3D-box projected denominator 的路径/bytes/SHA；不 decode image 或任何 mask。
- dynamic mask 明确只作 train-only actor-foreground weak support，绝不当 instance-ID 真值。重叠 projected boxes 的像素
  在后续归因中排除；actor-view 至少 64 support pixels，track 至少 2 views；scene-local track↔short-ID 用最大权一对一匹配。
- 后续门已在读像素前冻结：每场至少 `1 track/2 actor-views`，foreground coverage≥0.70、one-to-one assignment recall
  ≥0.35、assignment efficiency≥0.75、persistent-track fraction≥0.50，三场必须全门 PASS。F0k 只建立 denominator；
  quality/training/H/S/C/validation/test/KITTI/F1/F2=false，M2/M3=pending。

## V5.1 Stage F F0j r041 fresh 45-view materialization PASS 已冻结（2026-08-18）

- canonical r041=`20260818T180000Z__m1-stage-f-f0j-fresh-45-view-recovery-s20260814-r041`，source/tree=
  `27dfaa8...150a/9f433d16...abd2`。三个 scene-local fresh subprocess 全部 success；exact 45 inputs、`45 uint8
  900×1600 masks +3 pred.json` 与 output chain 均通过独立重放。
- 18/18 observed line58 matmul 均有 empty-cache before/after evidence；source/operator/tensor-content/grid32/batch64/AMP
  未变。audit=`20260818T183000Z__stage-f-f0j-r041-audit.json /18,462 bytes /acd5a91b...31d2 /PASS`；
  manifest=`108 entries /8,735,373 logical /1,205,363 regular`。
- resources=`GPU 1→24,118/24,576 MiB，headroom 458 MiB /cgroup 17,981,091,840 bytes /104.677s /324 samples /
  0 errors`。freeze=`configs/worldsim_v51/stage_f_f0j_fresh_45_view_empty_cache_materialization_freeze_v1.yaml`。
- `V51-F62` 对 frozen empty-cache 45-view execution 已解除，但 allocator/CUBLAS 唯一根因未证明；该解除不包含 mask
  quality、actor identity alignment 或 training readiness。下一步只预注册 train-only quality/alignment gate；validation/test/
  KITTI/F1/F2=false，M2/M3=pending。

## V5.1 Stage F F0j fresh 45-view empty-cache materialization 已预注册（2026-08-18）

- formal target=`20260818T180000Z__m1-stage-f-f0j-fresh-45-view-recovery-s20260814-r041`；authorization 仅来自 r040
  freeze=`3,252 bytes /ac390557...baca`。新目录从 exact r026 45-record manifest 重建输入，不续写或复用 r035 partial。
- 三个 scene-local fresh subprocess 严格串行 `0471→1087→0379`；grid32/upstream-batch64/AMP/size480/thresholds
  不变，唯一 recovery intervention 仍是每次 frozen line58 matmul 前 `torch.cuda.empty_cache()`。
- PASS 分母=`45 uint8 900×1600 schema masks +3 pred.json +all observed matmul intervention evidence +output chain +
  resource gate`。只读 schema/hash，不读 nonzero、mask quality 或 actor identity alignment。
- resources=`24,576 total /256 headroom /24,320 peak MiB /cgroup60 GiB /1,200s /disk40 GiB`；失败立即保留 blocked
  terminal 并更新 V51-F62。PASS 也只授权预注册 train-only quality/identity-alignment gate；training/validation/test/KITTI/
  F1/F2=false，M2/M3=pending。

## V5.1 Stage F F0i r040 scene-1087 15-view recovery PASS 已冻结（2026-08-18）

- canonical r040=`20260818T170000Z__m1-stage-f-f0i-scene1087-recovery-s20260814-r040`，source/tree=
  `9c8c503...9003/ff47087a...965e`；exact 15 inputs、`15 uint8 900×1600 masks +1 pred.json` 均通过独立重放。
- frozen intervention 仍仅为每次 line58 matmul 前 `torch.cuda.empty_cache()`；6/6 matmul 有 before/after allocator
  evidence，未修改 source/operator/tensor-content/method，未读 nonzero/quality。
- audit=`20260818T173000Z__stage-f-f0i-r040-audit.json /9,254 bytes /1393c664...67c /PASS`；manifest=
  `39 entries /3,779,879 logical /436,594 regular`。resources=`GPU 1→24,118/24,576 MiB，headroom 458 MiB /
  cgroup 17,966,829,568 bytes /29.030s /89 samples /0 errors`。
- `V51-F62` 仍 active：r040 建立 1087 单场 15-view recovery，不证明 fresh 三场 45-view 稳定。下一步只预注册 fresh
  `0471→1087→0379` 三场 45-view empty-cache materialization；quality/training/validation/test/KITTI/F1/F2=false，M2/M3=pending。

## V5.1 Stage F F0i scene-1087 15-view recovery 已预注册（2026-08-18）

- formal target=`20260818T170000Z__m1-stage-f-f0i-scene1087-recovery-s20260814-r040`；auth=r039 freeze
  `3,294 bytes /c83cd169...d72a`。exact scene-1087 15 train-only views，grid32/batch64/AMP 与 pre-matmul empty-cache 不变。
- 本轮只验证 `15 uint8 masks +1 pred.json` schema、每次 matmul intervention evidence 与资源；不读 nonzero/quality，
  不续写 r035。PASS 后只预注册 fresh 三场 45-view recovery，full materialization 仍 false。

## V5.1 Stage F F0h r039 empty-cache parity PASS 已冻结（2026-08-18）

- canonical r039=`20260818T160000Z__m1-stage-f-f0h-empty-cache-parity-s20260814-r039`，source/tree=
  `ba2f24f...a6b7/64b4c992...a8e5`，四臂全部 success；control/target 双 repeat 与 r034、r036/r038 reference
  全部 bit-exact，8/8 pre-matmul empty-cache evidence 成立。
- intervention 仅为 `torch.cuda.empty_cache()`，无 source/operator/tensor-content/method 变化。audit=
  `20260818T163000Z__stage-f-f0h-r039-audit.json /8,625 bytes /fda57ee4...88ab /PASS`；manifest=
  `39 entries /1,555,158 logical /518,169 regular`。
- resources=`GPU 1→24,118/24,576 MiB，headroom 458 MiB /cgroup 17,972,154,368 bytes /96.057s /301 samples /
  0 errors`。freeze=`configs/worldsim_v51/stage_f_f0h_pre_matmul_empty_cache_parity_freeze_v1.yaml`。
- `V51-F62` 仍未解除：r039 只建立 recovery candidate，不证明 1087 15-view 或 45-view 稳定。下一步只预注册 1087
  单场 15-view empty-cache recovery；full materialization/quality/training/validation/test/KITTI/F1/F2=false，M2/M3=pending。

## V5.1 Stage F F0h pre-matmul empty-cache parity 已预注册（2026-08-18）

- formal target=`20260818T160000Z__m1-stage-f-f0h-empty-cache-parity-s20260814-r039`；authorization=r038 freeze=
  `4,289 bytes /ff3e8692...03ca`。唯一 intervention 是在 frozen `memory_readout=value@affinity` 前调用
  `torch.cuda.empty_cache()`；不改 tensor 内容、算子、upstream source、grid32/batch64/AMP/size480/阈值。
- 四臂 A–B–A–B=`control_cache_1→target_cache_1→control_cache_2→target_cache_2`，均 fresh process。每个 trace 必须
  证明两次 matmul 前都调用 empty-cache 且 driver-free 不下降；四臂必须全部成功。
- control 三 mask/pred 必须同时 exact 对齐 r034 reference=`cbfc00d5...226/c11db011...f18/7ffe5683...593 /
  f5491453...156`；target 必须 exact 对齐 r036/r038 success=`6d679a37...e1c/c9768bbd...f02/1d46fa81...a6d /
  10d55216...650`。failure 或 nonexact 均拒绝 recovery；PASS 也只允许预注册 1087 15-view recovery。
- resources=`24,576 total /256 headroom /24,320 peak MiB /cgroup60 GiB /900s`；full materialization/quality/training/
  validation/test/KITTI/F1/F2=false，M2/M3=pending，failure delta=`none`。

## V5.1 Stage F F0g r038 tensor/allocator trace 已冻结（2026-08-18）

- canonical r038=`20260818T150000Z__m1-stage-f-f0g-tensor-trace-s20260814-r038`，source/tree=
  `da2169d...a5f3/9a00a267...50a0`，control/target trace 都 success；两侧成功 mask/pred 分别与 r034、r036 success
  bit-exact。trace source 未修改、无 operator monkeypatch、未读 tensor content。
- control 两个 matmul objects=`26/36`、value shapes=`[1,13312,1620]/[1,18432,1620]`，pre-matmul driver-free=
  `36,765,696/57,737,216 bytes`，allocator retries=`0/0`；target objects=`3/52`、value shapes=
  `[1,1536,1620]/[1,26624,1620]`，free=`19,494,141,952/18,502,189,056`，allocator retries=`1/1`。
  两侧 affinity 都是 `[1,1620,1620]`，tensor contiguous/dimension valid。
- 因 target 首次 matmul 只有 3 objects，远小于 control 的 26，“target 首个 shape 更大所以确定性失败”被推翻。target
  success 与一次 allocator retry/cache release 同时出现，支持 cache/CUBLAS workspace state hypothesis；但 trace timing 本身
  可能扰动 runtime，且 control 在极低 driver-free 下也成功，因此 `root_cause_proven=false`。
- audit=`20260818T153000Z__stage-f-f0g-r038-audit.json /16,025 bytes /a8cbdb5b...4047 /PASS`；freeze=
  `configs/worldsim_v51/stage_f_f0g_target_tensor_allocator_instrumentation_freeze_v1.yaml`。resources=`peak 24,118 MiB /
  headroom 458 MiB /cgroup 17,968,336,896 bytes /48.698s /151 samples /0 errors`。
- 下一步只预注册 pre-matmul `torch.cuda.empty_cache()` execution recovery parity：不改 tensor/operator/方法参数，control 与
  target 各两次，必须对已有 success hashes bit-exact；单次 recovery 不授权 full materialization。`V51-F62` 仍 active，
  quality/alignment/training/validation/test/KITTI/F1/F2=false，M2/M3=pending。

## V5.1 Stage F F0g source-neutral tensor/allocator instrumentation 已预注册（2026-08-18）

- formal target=`20260818T150000Z__m1-stage-f-f0g-tensor-trace-s20260814-r038`；authorization=r037 freeze=
  `4,391 bytes /18fd02be...55b7`。顺序为 0471 known-good control trace→1087 target trace，各一次 fresh process；输入、
  grid32/batch64/AMP/size480/thresholds 与 blocking 全冻结。
- 新 launcher 通过 Python trace 在 frozen `consensus_associated.py` line58/59 的 matmul 前/后或异常时记录局部 tensor
  `shape/dtype/device/stride/contiguous/numel/logical-bytes` 与 CUDA allocator counters。traced source=`5,985 bytes /
  a1b86e65...c5a6`；不读 tensor 内容、不 monkeypatch 算子、不写第三方 source，run 后再次检查 checkout clean。
- 这是 root-cause instrumentation，不是 recovery。无论 target 成败，trace/output 都不进 quality/materialization/training；
  下一步只依据 control-target shape/allocator delta 判断是否可预注册代数等价的 memory-readout chunking，或关闭 faithful 路线。
- denominator=`6 input decodes /<=6 schema reads`；resources 沿用 `24,576 total /256 headroom /24,320 peak MiB /
  cgroup60 GiB /600s`。failure delta=`none`，M2/M3=pending。

## V5.1 Stage F F0f r037 control 稳定、target 不稳定，已冻结（2026-08-18）

- canonical r037=`20260818T140000Z__m1-stage-f-f0f-runtime-repro-s20260814-r037`，source/tree=
  `3c692f4...cc3/ad8d798b...89c`，status=`done`，outcome=`control_stable_target_failure`。A–B–A–B 中两次
  0471 control 均 success，三 mask/pred pair exact 且与 r034 SHA 完全相同；两次 1087 target 均在冻结 CUBLAS site
  失败、0 mask/pred、非显式 OOM。
- control mask SHA=`cbfc00d5...226/c11db011...f18/7ffe5683...593`、pred=`f5491453...156`；target stderr=
  `35c3494d...c58/f6378ead...a5a`。结合 r036 的 target 一败一成，只能写作 target path process instability；不能写成
  全局 GPU 故障，也不能写成 target deterministic 必败。
- health probe：identity/temp/P-state 与 NVIDIA ECC/page/row 命令成功；RTX3090 ECC 等字段 N/A，只表示不可用，不算健康
  PASS；dmesg 因权限拒绝不可观测。没有 GPU reset/driver mutation。resources=`peak/headroom 24,124/452 MiB /cgroup
  17,969,901,568 bytes /86.805s /271 samples /0 errors`，全门 PASS。
- audit=`20260818T143000Z__stage-f-f0f-r037-audit.json /8,245 bytes /2fb76f32...d50d /PASS`；freeze=
  `configs/worldsim_v51/stage_f_f0f_cuda_runtime_health_reproducibility_freeze_v1.yaml`。`V51-F62` 保持 active，并细化为
  control stable/target unstable。下一步只预注册 source-neutral target tensor/allocator instrumentation；full materialization/
  quality/alignment/training/validation/test/KITTI/F1/F2=false，M2/M3=pending。

## V5.1 Stage F F0f CUDA runtime health/reproducibility 已预注册（2026-08-18）

- formal target=`20260818T140000Z__m1-stage-f-f0f-runtime-repro-s20260814-r037`，authorization 只来自 r036 mixed freeze=
  `4,338 bytes /207b28f5...b62f`。执行顺序严格 A–B–A–B：A=0471 camera0 frames0/40/80（r034 known-good control），
  B=1087 frame0 cameras0/1/2（r035/r036 target）。每臂 fresh process、串行、grid32/batch64/AMP 与
  `CUDA_LAUNCH_BLOCKING=1` 全相同。
- 同时只读采集 GPU identity/temperature/P-state、ECC/page-retirement/row-remapper 与 dmesg 访问结果；RTX3090 的 ECC N/A
  不冒充 PASS，dmesg 若因权限拒绝只记录不可观测边界。不 reset GPU、不改 driver/runtime/source。
- outcome 已冻结：A/B 各两次均 success+exact→先做 1087 15-view blocking recovery；A exact 而 B 仍失败→进入 target
  tensor/allocator source-neutral instrumentation；A 也失败→runtime unhealthy 并暂停该 GPU 路线；任一成功 pair nonexact→
  faithful identity input 以不可复现收口。任何成功输出都只做 schema/hash，不读质量。
- denominator=`12 input decodes /<=12 schema mask reads`；resources=`24,576 total /256 headroom /24,320 peak MiB /
  cgroup<=60 GiB /wall<=900s /disk>=40 GiB`。full materialization/quality/alignment/training/validation/test/KITTI/F1/F2=false，
  M2/M3=pending；failure delta=`none_at_preregistration`。

## V5.1 Stage F F0e r036 mixed CUDA replay 已冻结（2026-08-18）

- canonical r036=`20260818T130000Z__m1-stage-f-f0e-cuda-localization-s20260814-r036`，source/tree=
  `223f943...6e0/bb8166df...7ae`，status=`done`，outcome=`mixed`。同一组 1087 frame0 cameras0/1/2、同一
  grid32/batch64/AMP 和 `CUDA_LAUNCH_BLOCKING=1` 下，replay1 在原 GEMM 位置返回 CUBLAS internal error，
  replay2 则成功产出 `3` 张 `900×1600 uint8` mask 与 `pred.json`。
- replay1 stderr/stdout=`56bec693...ea2/45785f73...be2`，不是显式 OOM且无 partial output；replay2 mask SHA=
  `6d679a37...e1c/c9768bbd...f02/1d46fa81...a6d`，metadata=`10d55216...650`。这些成功输出只做 schema/hash，
  未读 nonzero/quality，不能补写 r035。
- runtime=`driver 595.58.03 /torch 2.1.2+cu118 /torch CUDA 11.8 /cuDNN 8700`；resources=`GPU 1→24,124/
  24,576 MiB，headroom 452 MiB /cgroup 17,964,371,968 bytes /43.107s /134 samples /0 errors`，全部门 PASS。
- audit=`20260818T133000Z__stage-f-f0e-r036-audit.json /5,077 bytes /ec7cfa36...34f6 /PASS`；freeze=
  `configs/worldsim_v51/stage_f_f0e_scene1087_cuda_fault_localization_freeze_v1.yaml`。`V51-F62` 细化为当前 probe 下
  process-to-process nondeterministic，仍 active；下一步只预注册 CUDA runtime health/reproducibility gate，full
  materialization/quality/training/H/S/C/validation/test/KITTI/F1/F2=false，M2/M3=pending。

## V5.1 Stage F F0e scene-1087 CUDA fault localization 已预注册（2026-08-18）

- task/run=`WS-V51-M1-F-IDENTITY-EMBEDDING-01 /20260818T130000Z__m1-stage-f-f0e-cuda-localization-s20260814-r036`；
  authorization 只来自 r035 blocked closeout=`4,271 bytes /3cb38341...93da /V51-F62 active`。
- 输入锁为 r035 首次出错前的 exact scene-1087 frame0 cameras=`0/1/2`，逐文件 SHA/bytes 取 r026 train-only manifest；
  两次独立 official subprocess 串行 replay。方法继续 grid32/upstream-batch64/AMP/size480/IoU0.7/semionline，唯一新增
  execution diagnostic 是 `CUDA_LAUNCH_BLOCKING=1`；不改 upstream source，也不做参数搜索。
- outcome matrix 预先冻结为：两次同位 CUBLAS failure→进入 source-neutral tensor/allocator instrumentation；两次 success 且
  output exact→先做 1087 15-view blocking recovery；一成一败→CUDA runtime health/reproducibility gate；两次成功但不 exact→
  faithful identity input 以 nonrepeatable 收口。unrecognized failure 直接 protocol blocked。
- 本轮最多 schema-read `6` 张成功 mask，不读 nonzero/quality/actor alignment；任何输出均不补写 r035 或进入训练。
  resources 仍锁 total/headroom/peak=`24,576/256/24,320 MiB`、cgroup<=60 GiB、wall<=600s、disk>=40 GiB。
  full materialization/quality/training/H/S/C/validation/test/KITTI/F1/F2=false，M2/M3=pending；failure delta=`none`。

## V5.1 Stage F F0d r035 因 CUDA/CUBLAS 中断，已独立审计收口（2026-08-18）

- canonical r035=`20260818T120000Z__m1-stage-f-f0d-train-materialization-s20260814-r035`，source/tree=
  `e4d64d3...1424/25590428...8a2`，terminal=`blocked`。0471 完成 `15 masks +1 pred.json`，其中 `14` 张
  non-empty、`1` 张全背景，总 nonzero pixels=`6,527,167`、stable short IDs=`16`；这些只证明输出 schema/关联存在，
  没有读取 mask quality。
- 1087 的 15 个输入均 exact staged；official grid32/upstream-batch64/AMP subprocess 处理完前两张后，在第三张进入
  three-frame vote，于 `consensus_associated.py:58 spatial_alignment` 的 `value @ affinity` 返回
  `CUDA CUBLAS_STATUS_INTERNAL_ERROR`。它没有留下 mask/pred/report；0379 未启动。错误不是显式 PyTorch OOM，当前也没有
  证据把它定为质量拒绝、确定性数据故障或具体 driver/resource 根因。
- resource replay=`GPU peak 24,124/24,576 MiB，headroom 452 MiB /cgroup 17,961,271,296 bytes /174 samples /
  0 errors /55.993866s`，仍在 r035 预注册 peak/headroom 门内；因此不能把 CUBLAS internal error 直接改写成 OOM，也不能
  因“资源门数值通过”倒写 full materialization 成功。
- independent audit=`20260818T123000Z__stage-f-f0d-r035-audit.json /25,311 bytes /6d217a7e...13e1 /PASS`，重放
  30 个 staged symlink、0471 全部 mask/pred/report、1087 failure signature、资源和 `55 entries /5,320,231 logical bytes`
  inventory。closeout=`configs/worldsim_v51/stage_f_f0d_train_only_identity_mask_materialization_closeout_v1.yaml`；
  `V51-F62=active`。
- r035 只形成 `15/45 masks、1/3 pred.json`，partial 不是 canonical full materialization。下一合法动作仅为预注册 1087
  exact 首三视图的 `CUDA_LAUNCH_BLOCKING=1` 故障定位，保持 grid32/batch64/AMP/阈值；禁止缩 batch、删/换/重排场景、
  覆盖/续跑 r035、读取质量或按输出选臂。identity training/quality/H/S/C/validation/test/KITTI/F1/F2=false，M2/M3=pending。

## V5.1 Stage F F0d 45-view train-only materialization 已预注册（2026-08-18）

- task=`WS-V51-M1-F-IDENTITY-EMBEDDING-01`；formal target=
  `20260818T120000Z__m1-stage-f-f0d-train-materialization-s20260814-r035`，source=本预注册提交，seed=`20260814`。
  authorization 只来自 r034 freeze；输入严格复用 r026 manifest=`45 records /7,530,010 bytes /record-chain
  b3458c27...4d95`，仅 train-only 三场，每场 frames=`0/40/80/120/160` × cameras=`0/1/2`。
- 三场存在同名 `000_0.jpg` 等文件，且 short-ID namespace 必须 scene-local；因此每场独立 input/output directory、独立
  official CLI subprocess，按 `scene-0471→1087→0379` 串行。场内 filename lexicographic order 恰为 frame→camera；禁止
  把 45 张图放进同一目录覆盖文件或让 identity 跨 scene 传播。
- method 固定 r034 的 grid32/upstream-batch64/size480/IoU0.7/semionline；输出分母锁为 `45 uint8 masks +3 pred.json`，
  每场至少一张 non-empty mask、至少一个 short ID 跨两 views，并发布逐 record output chain。资源仍锁 card total/headroom/
  peak=`24,576/256/24,320 MiB`、cgroup<=60 GiB、wall<=1,200s、disk>=40 GiB、0 monitor errors。
- 本轮只执行 materialization，不读质量、不对齐 actor identity、不训练。`failure_ledger_refs=[V51-F31,F37,F42,F46,
  F52,F55,F57,F59,F60,F61,PIVOT-F05]`，delta=`none`；若 PASS，下一步只允许预注册 train-only mask-quality 与
  identity-alignment gate；H/S/C/validation/test/KITTI/F1/F2=false，M2/M3=pending。

## V5.1 Stage F F0c r034 upstream batch64 PASS 已冻结（2026-08-18）

- canonical r034=`20260818T110000Z__m1-stage-f-f0c-upstream-batch-s20260814-r034`，source/tree=
  `27b1958...b48/552ba018...58e`，status=`done`，conclusion=
  `f0c_upstream_batch64_three_view_association_repeatability_resource_smoke_done_full_materialization_preregistration_required`。
  grid/batch=`32/64`，恢复 upstream default batch；primary/repeat 的三张 mask 与 metadata 全部 exact。
- 三张 nonzero pixels=`1,183,290/1,257,333/954,829`，19 个 stable short IDs；mask SHA=
  `cbfc00d5...226/c11db011...f18/7ffe5683...593`，metadata=`f5491453...156`。这建立 no-quality association/
  repeatability contract，不建立 mask quality、grid64 quality parity 或 U2/B3 uplift。
- resources=`GPU 1→24,092/24,576 MiB，headroom 484>=256 MiB /cgroup 17,957,322,752 bytes /45.917s /
  142 samples /0 errors /disk 79,934,353,408 bytes`，全门 PASS。manifest=`23 entries /483,584 logical bytes`；
  independent audit=`4,243 bytes /e0988f50...5258 /PASS`，完整重放 source/config/inputs/outputs/resources/locks。
- freeze=`configs/worldsim_v51/stage_f_f0c_upstream_batch_association_repeatability_freeze_v1.yaml`；`V51-F60` 以固定
  upstream batch64 解除当前方法选择但保留 batch-sensitive 边界，`V51-F61` 由 r034 新 headroom 合同解除，r033 旧失败
  不倒写。下一步只允许预注册 r026 45-view train-only full identity-mask materialization；materialization/training 尚未授权，
  quality/H/S/C/validation/test/KITTI/F1/F2=false，M2/M3=pending。

## V5.1 Stage F F0c upstream batch64 recovery 已预注册（2026-08-18）

- task=`WS-V51-M1-F-IDENTITY-EMBEDDING-01`；formal target=
  `20260818T110000Z__m1-stage-f-f0c-upstream-batch-s20260814-r034`，source=本预注册提交，seed=`20260814`。
  r033 closeout 被固定为 blocked authorization；data/source/assets/三帧/official CLI 与 r033 相同，唯一方法恢复是把
  `SAM_NUM_POINTS_PER_BATCH=32→64` 恢复为 upstream default，grid32 作为 `V51-F57` 已记录的 24GB 必要适配保留。
- r034 只串行执行 `primary_batch64→repeat_batch64`；要求三张 mask 与 `pred.json` repeat exact、至少一张 non-empty、
  至少一个 short ID 跨至少两帧，并锁 exact input/output read denominator=`6/6`。不再把 batch16/32 当 execution-only
  选项，也不按 r033 结果挑选较好输出。
- 新 run 在启动前锁 `RTX 3090 total=24,576 MiB`，预注册 `256 MiB` headroom，因此 peak ceiling=`24,320 MiB`；
  这是 r034 的物理资源合同，不倒改 r033 的 24,000 门。若 batch64 OOM、peak>24,320、headroom<256 或 repeat/
  association 失败，则关闭该 faithful recovery，不再缩 batch 重试。
- `failure_ledger_refs=[V51-F31,F37,F42,F46,F52,F55,F57,F59,F60,F61,PIVOT-F05]`，delta=`none`；
  materialization/training/quality/H/S/C/validation/test/KITTI/F1/F2=false，M2/M3=pending。全部 gate PASS 也只授权
  预注册 train-only full identity-mask materialization。

## V5.1 Stage F F0b r033 blocked：batch 改变输出且显存越门（2026-08-18）

- canonical r033=`20260818T100000Z__m1-stage-f-f0b-association-parity-s20260814-r033`，source/tree=
  `191d3e4...12f/c4f32ec...e36`，terminal=`blocked`。primary batch32 与 repeat batch32 的三张 mask 和 `pred.json`
  全部 exact SHA；三张均 non-empty，19 个 positive short IDs 至少跨 2 帧，故 r032 的 one-view identity-input 边界
  `V51-F59` 已被三视图 association 子门解除，但不含任何质量结论。
- batch16 与 batch32 的三帧 exact-label 差异分别为 `208,647/288,527/244,696 pixels`，exact fraction=
  `0.855106/0.799634/0.830072`；binary foreground IoU=`0.961177/0.995201/0.969622`。batch32 IDs 含
  `36/62/95`，batch16 则含 `13/63/96`，metadata SHA=`e841df5d...12c5` vs `89fde3ad...59cb`。由于同 batch
  repeat bit-exact，这不是随机重跑抖动；`SAM_NUM_POINTS_PER_BATCH` 不能再被当成 execution-only knob，见 `V51-F60`。
- independent audit=`22,939 bytes /a5a7d5c8...fa7d /PASS`，重算 9 masks、3 metadata、输入/source/config、parity、
  resources 与锁；run inventory=`29 logical entries /544,368 bytes`。审计另发现 sampled GPU peak=`24,116 MiB`
  超过 prereg `24,000 MiB`，cgroup=`17,956,044,800 bytes`、event wall=`78.917s`、241 samples/0 errors，见
  `V51-F61`；runner 因 parity 先 fail-closed，未发布 done summary/manifest。
- full materialization/identity training 仍未授权。下一合法步骤只允许预注册 grid32 + upstream-default batch64 的
  同三视图 association/repeatability/resource smoke，以移除已证伪的 batch32 execution adaptation；不得倒写 r033、
  事后放宽其 parity/24,000 门或读取 quality/H/S/C/validation/test/KITTI。F1/F2=false，M2/M3=pending。

## V5.1 Stage F F0b 三视图 association/batch parity 已预注册（2026-08-18）

- task=`WS-V51-M1-F-IDENTITY-EMBEDDING-01`；formal target=
  `20260818T100000Z__m1-stage-f-f0b-association-parity-s20260814-r033`，source=本预注册提交，seed=`20260814`。
  授权只来自 r032 freeze；输入只取 r026 已冻结 45-view train-only manifest 中 `scene-0471/index382/camera0`
  的 frame=`0/40/80`，逐文件 bytes/full SHA 锁定，不读 H/S/C/validation/test/KITTI。
- 三臂在同一个不可复用 run 内串行执行：primary=`grid32/batch32`、parity=`grid32/batch16`、repeat=
  `grid32/batch32`。除 batch parity 臂的 `SAM_NUM_POINTS_PER_BATCH` 外，official source/assets、1024 prompts、size480、
  IoU0.7、semionline 与输入完全相同；要求三张 mask 与 `pred.json` 在 primary↔parity、primary↔repeat 间 exact SHA 一致。
- association 结构门要求 primary 至少 1 张 non-empty mask，且至少 1 个 positive short ID 跨至少 2 帧出现；这只检验
  3-frame voting/association 是否产生可重复的 identity input，不读 mask quality，也不证明 grid32 与 upstream grid64
  的质量等价。GPU/cgroup/wall/disk/monitor 门沿用 r032，三臂串行避免同时占卡。
- `failure_ledger_refs=[V51-F31,F37,F42,F43,F46,F49,F52,F53,F55,F57,F59]`，预注册 delta=`none`；
  `full_materialization/identity_training/F1/F2=false`，M2/M3=`pending`。仅当 exact parity、repeatability、non-empty、
  stable short-ID 与资源门全部 PASS，下一步才允许预注册 train-only full identity-mask materialization。

## V5.1 Stage F F0a r032 resource/schema PASS 已冻结（2026-08-18）

- canonical r032=`20260818T090000Z__m1-stage-f-f0a-environment-one-view-s20260814-r032`，source/tree=
  `29a160a...b52/4681c265...10d`，status=`done`，conclusion=
  `f0a_environment_and_one_view_resource_smoke_done_grid_quality_batch_parity_and_association_smoke_required`。grid/batch=
  `32/32`、1024 prompts；环境、两种 solver、DEVA/SAM/ResNet assets 与 output schema 全通过。
- resource=`GPU 1→23,954 MiB /cgroup 18,044,903,424 bytes /wall 22.494s /71 samples /0 errors`，所有 prereg gate
  PASS。manifest=`13 logical entries /157,563 bytes`；mask=`900×1600 uint8 /0bf854a1...59d`、metadata=
  `35fbd75a...5e8/1 annotation`。mask histogram=`{0:1,440,000}` 是单视图短于 3-frame voting 的预期边界，只证明
  resource/schema，不证明 non-empty mask、association 或 quality（`V51-F59 active`）。
- independent audit=`1,687 bytes /cebe07fd...cd5`，重放 manifest/source/assets/wheels/solvers/output/resources/locks
  全 PASS；freeze=`configs/worldsim_v51/stage_f_f0a_environment_one_view_smoke_freeze_v1.yaml`。审计后已精确删除 r029
  生成的两份 `/root/.cache` 源副本，canonical `TORCH_HOME` full SHA 保持；首次清理 wrapper 的 `cut` 引号失败未删除任何
  文件，随后无管道恢复完成（`V51-F58 resolved`）。
- 下一步只允许预注册 same-grid batch32↔16 parity + 至少 3-view semionline association/repeatability；在 non-empty、
  stable short-ID 与资源门完成前，full materialization/identity training=false，quality/H/S/C/validation/test/KITTI=false，
  F1/F2=false，M2/M3=pending。

## V5.1 Stage F F0a r031 batch32 blocked；v6/r032 grid32 recovery 已预注册（2026-08-18）

- r031=`20260818T083000Z__m1-stage-f-f0a-environment-one-view-s20260814-r031`，source=`2e96f05`，official CLI 确认
  batch=`32`，但同一 `BatchMaskData.cat` 仍 OOM：request/free=`9.32/9.31 GiB`，allocated/reserved-unallocated=
  `13.34 GiB/599.11 MiB`；sampled GPU/cgroup peak=`24,066 MiB/18,052,734,976 bytes`，119 samples/0 errors。没有
  mask/metadata/quality，batch-only 不足且不构成算法 reject，见 `V51-F57`。
- r030/r031 的最终 allocation 仅 `9.49→9.32 GiB`，说明峰值由 64×64=`4096` prompts 的累计 masks 主导，而不是 batch
  staging。继续 batch16 属于重复调参，未授权。DEVA 官方文档明确建议降低 `SAM_NUM_POINTS_PER_SIDE` 以减少 automatic SAM
  queries，因此 v6/r032 只把 grid=`64→32`（4096→1024），batch 固定 32；其他 source/assets/view/size/IoU/resource/locks
  不变。
- grid32 是有来源的资源适配，不冒充 exact-default 复现；PASS 后必须在同一 grid 做 batch32↔16 parity，并做 3-view
  association/repeatability 与后续质量门。materialization/training、H/S/C/validation/test/KITTI=false，F1/F2=false，
  M2/M3=pending。

## V5.1 Stage F F0a r030 allocator recovery blocked；v5/r031 batch recovery 已预注册（2026-08-18）

- r030=`20260818T080000Z__m1-stage-f-f0a-environment-one-view-s20260814-r030`，source=`33c013d`，status=`blocked`。
  dedicated `TORCH_HOME` 中 ResNet18/50 exact bytes/full SHA 与 prereg 一致、无 `.partial`，stderr 无 download，故
  `V51-F53 resolved`；Gurobi/PuLP、source/view/official CLI 均未漂移。
- allocator 已把 reserved-unallocated 从 r029 的 `5.77 GiB` 降到 `578.72 MiB`，但 SAM `64×64 / batch64` 在同一
  `BatchMaskData.cat` 仍需 `9.49 GiB`、当时只有 `9.16 GiB` free；sampled GPU peak=`24,098 MiB`（超过 prereg
  `24,000 MiB` gate），cgroup peak=`18,035,429,376 bytes`，101 samples/0 monitor errors。仍无 mask/`pred.json`/quality；
  这证明 allocator-only 不足，见 `V51-F55`，仍非算法 reject。
- v5/r031 只把上游明确暴露的 `SAM_NUM_POINTS_PER_BATCH=64→32`；point grid=`64`、size=`480`、IoU=`0.7`、其余
  source/assets/view/CLI/resource/locks 全不变。该 knob 只降低并行批量，不减少 prompt denominator；PASS 也必须补
  batch parity + 3-view association/repeatability，materialization/training 与 quality/H/S/C/validation/test/KITTI 继续 false，
  F1/F2=false，M2/M3=pending。

## V5.1 Stage F F0a r029 resource blocked；v4/r030 recovery 已预注册（2026-08-18）

- r029=`20260818T073000Z__m1-stage-f-f0a-environment-one-view-s20260814-r029`，source=`3e87323`。v3 已正确
  解析并保留 Gurobi restricted-license banner，Gurobi/PuLP optimum gate 均越过；official DEVA+SAM CLI 随后加载模型并
  decode 唯一输入，但 SAM ViT-H everything mask 在首图尝试再分配 `6.74 GiB` 时 OOM。allocator 报进程占用
  `16.83 GiB`、PyTorch allocated/reserved-unallocated=`10.74/5.77 GiB`，resource monitor peak=`17,246 MiB`；没有
  mask、`pred.json` 或 quality，故是 `V51-F52` 工程资源 blocked，不是 Gaussian Grouping/association 算法 reject。
- 首次模型加载还暴露 DEVA `resnet.py` 的隐式 `model_zoo.load_url`：ResNet50/18 分别下载到 `/root/.cache`，exact
  bytes/SHA=`102,502,400/19c8e357...097` 与 `46,827,520/5c106cde...13f8`，见 `V51-F53`。v4/r030 先把两份
  权重原子发布到专用 `TORCH_HOME` 并冻结 URL/bytes/full SHA，再令 upstream 子进程只读该 cache。
- r030 的唯一资源恢复是 `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128`；source、DEVA/SAM 权重、唯一 view、official
  CLI、`size=480`、IoU=`0.7`、SAM grid/batch=`64/64`、resource/locks 全不变。仍不授权 materialization/training，
  quality/H/S/C/validation/test/KITTI=false，F1/F2=false，M2/M3=pending。预提交双层 shell 引号复发已在 Git/run 前
  fail-closed，并切换为文件化 commit message（`V51-F54 resolved`）。

## V5.1 Stage F F0a r028 stdout blocked；v3/r029 recovery 已预注册（2026-08-18）

- r028 已用 `gurobipy=12.0.3` 越过 expired-license 异常，但 Gurobi 把 restricted-license banner 写在 JSON 前，runner
  对整段 stdout 做 `json.loads` 而 blocked。4 files=`10,642 bytes`；没有 one-view CLI、DEVA/SAM load、input/mask decode
  或 quality，见 `V51-F51`。
- v3/r029 只改 solver stdout contract：最后一个非空行必须是 JSON，前置 banner 逐行保留在 report；Gurobi/PuLP 的
  optimal status 与 solution=1 门不变，复用 exact v2 wheelhouse/venv。source/assets/view/CLI/resource/locks 不变，
  materialization/training 仍未授权，M2/M3=pending。

## V5.1 Stage F F0a r027 license blocked；v2/r028 recovery 已预注册（2026-08-18）

- r027 在 isolated venv 与三份 pinned wheels 完成后，于 Gurobi tiny model 创建处报 `License expired 2024-10-28`；
  status=`blocked`，4 files=`12,861 bytes`。one-view CLI 尚未启动，DEVA/SAM 未加载，input/mask pixels=`0/0`，GPU/
  quality 均未读；见 `V51-F49`。
- 上游约束是 `gurobipy>=10.0.3`，v1 错把最低版本固定为 exact 10.0.3。v2 只换为当前 package index 可得的
  `12.0.3` 并使用 fresh `python_wheels_v2`、`deva-v51-stage-f-v2`；source/assets/view/CLI/resource/locks 全部不变。
  r027 import path probe 另产生 4 个 source-tree `__pycache__`，已精确清除并在 v2 禁止子进程写 bytecode（`V51-F50`）。
  r028 仍必须先过 Gurobi 与 PuLP tiny optimum，不能绕过 license gate。materialization/training/quality=false，M2/M3=pending。

## V5.1 Stage F F0a isolated environment/one-view smoke 已预注册（2026-08-18）

- planned r027=`20260818T063000Z__m1-stage-f-f0a-environment-one-view-s20260814-r027`。用 DriveStudio runtime 创建
  `/root/autodl-tmp/envs/deva-v51-stage-f` 的 `--system-site-packages` 隔离 venv；只从独立 wheelhouse 安装 pinned/no-deps
  `supervision=0.14.0/PuLP=2.7.0/gurobipy=10.0.3`，DEVA 与 Segment Anything 用冻结绝对 source path 注入，不改上游。
- wheel 第一次下载后必须保存 filename/bytes/full SHA，再从本地 wheelhouse `--no-index` 安装；同时要求 Gurobi 与 PuLP
  各完成一个 tiny binary optimum，不能静默把 Gurobi 不可用当成 faithful association 已就绪。
- 唯一输入冻结为 scene-0471/frame0/camera0=`99,906 bytes /093d38e8...819e`；以 symlink staging 调 official
  `demo_automatic.py`，完整保持 `chunk4/amp/semionline/size480/short-ID/suppress-small/IoU0.7`，只验一张 uint8 PNG、
  `pred.json`、label range 与资源，不评价 mask quality。
- 该视图少于 semionline `num_voting_frames=3`，因此 PASS 只证明 environment/model load/SAM forward/resource/output schema，
  明确不证明 cross-view association。PASS 后仍需另行预注册 3-view association+repeatability smoke；full materialization/
  identity training=false，H/S/C/validation/test/KITTI=false，F1/F2=false，M2/M3=pending。首稿手录 input SHA 漏一个
  `d8`，formal-config test 在任何 run/env/GPU 前捕获并按 r026 manifest 恢复，见 `V51-F48 resolved`。

## V5.1 Stage F F0a r026 asset/source acquisition 已冻结（2026-08-18）

- canonical r026=`20260818T053000Z__m1-stage-f-f0a-asset-source-s20260814-r026`，source/tree=
  `a77f458c...e7a1/1df5c234...aaa3`，status=`done`，conclusion=
  `f0a_assets_and_sources_frozen_environment_setup_required`。独立审计重新 full-hash 两份权重、核对 external source
  commit/tree/license/clean state、重建 45-view train-only manifest、重放 153 个 resource samples 与全部 locks，结果 PASS。
- assets：DEVA=`276,911,801 bytes /52737482...5e48`，SAM ViT-H=`2,564,550,879 bytes /a7bf3b02...262e`；
  都从 upstream-declared URL 顺序下载、exact bytes 后 atomic publish，`.partial` 已消失。Grounded-Segment-Anything=
  `99fbbe78 /tree 89c82ae8...c97 /Apache-2.0`；Gaussian Grouping checkout 仍 clean。
- train-only input=`45 files /7,530,010 bytes /chain b3458c27...4d95`，按 scene→frame→camera 排列并逐文件 SHA exact；
  image pixels decoded=false、staged=false。environment mutated=false；DriveStudio runtime 的预计缺失模块集合
  `segment_anything/supervision/hickle/pulp/gurobipy/thinplate` 与实测一致。
- resource=`GPU 1 MiB /cgroup 11,761,352,704 bytes /wall 166.742s /153 samples /0 errors`；disk free=
  `83,420,561,408 → 80,406,933,504 bytes`。manifest=`7 entries /49,468 bytes`，full run=`9 files /51,021 bytes`；
  audit=`1,396 bytes /5a360f42...817c`。
- `V51-F47 resolved` 记录本地/远端执行边界与缺失 `bc` 只读工具，不影响 run。下一步只允许预注册 isolated DEVA
  environment + one-view resource smoke；SAM/DEVA materialization、identity training、quality/S/C/validation/test/KITTI
  继续 false，F1/F2=false，M2/M3=pending。freeze=
  `configs/worldsim_v51/stage_f_f0a_asset_source_acquisition_freeze_v1.yaml`。

## V5.1 Stage F F0a asset/source acquisition 已预注册（2026-08-18）

- planned r026=`20260818T053000Z__m1-stage-f-f0a-asset-source-s20260814-r026`。本轮只顺序获取并 full-SHA
  DEVA propagation (`276,911,801 bytes`) 与 SAM ViT-H (`2,564,550,879 bytes`)；URL 必须逐字存在于 frozen upstream
  `download_models.sh`，下载使用 `.partial + resume + exact-byte gate + atomic publish`，不写入 clean Gaussian Grouping repo。
- 同轮把上游安装说明要求的 `hkchengrex/Grounded-Segment-Anything@99fbbe78` clone 到独立 third-party 路径，并要求
  clean commit、`segment_anything/segment_anything` subtree 与 Apache-2.0 license。vendored DEVA source 保持
  `CC-BY-NC-SA-4.0`，仅用于本学术研究流程；不安装或修改 upstream source。
- 45-view 输入继承 Stage B frozen image manifest：只选 historical-diagnostic 的
  `0471/1087/0379 × frames 0/40/80/120/160 × cameras 0/1/2`，按 scene→frame→camera 排序，逐文件验证 size/SHA，
  不 decode pixels、不 staging images、不运行 SAM/DEVA。
- future faithful CLI 已冻结 `chunk=4/amp/semionline/size=480/use_short_id/suppress_small_objects/SAM IoU=0.7` 及所有
  relevant upstream defaults；未来输出必须每场 15 张 uint8 short-ID PNG、input/output stem 一一对应，并保存 label histogram/
  mask SHA 与 `pred.json`。本次只探测当前环境，明确不安装依赖；预计缺 `segment_anything/supervision/hickle/pulp/gurobipy/
  thinplate`，下一步必须另行预注册 isolated environment + one-view resource smoke。
- materialization/identity training=false，quality/H/S/C/validation/test/KITTI=false，F1/F2=false，M2/M3=pending；
  `failure_ledger_refs` 到 `V51-F46`，prereg delta=`none`。config=
  `configs/worldsim_v51/stage_f_f0a_asset_source_acquisition_v1.yaml`。

## V5.1 Stage F F0 r025 source/adapter preflight 已冻结（2026-08-18）

- canonical r025=`20260818T045000Z__m1-stage-f-f0-source-preflight-s20260814-r025`，source/tree=
  `8d68cad1...15c/bb73109b...6cd`，status=`done`，conclusion=
  `f0_source_adapter_preflight_done_input_materialization_required`。独立审计重新核对 official PDF/repo、10 个 source files
  与关键代码语义、45 个 train-only observation schema、三场 instance metadata、checkpoint identity、asset presence、
  16D gsplat backward、resource samples 和全部 read/training locks，结果 exact/pass。
- faithful source 核实为 `SAM everything → DEVA semionline cross-view short IDs → 16D identity encoding → alpha`
  `compositing → shared 1×1 classifier + normalized CE + k=5/sample=1000 forward-KL`。本项目只允许在 frozen
  geometry/appearance/opacity/dynamic poses 上学习 identity+classifier，因此登记为“faithful identity mechanism under
  frozen-base adaptation”，不冒充上游 joint-reconstruction exact reproduction。
- 三场各 `15` views、共 `45` 个现有 observation 都只有同一套 23 fields；没有 `instance/identity/object_id/mask_id/class_id`
  字段。train-frame stable metadata 可用于核对分母，却不是 pixel identity labels。三个 formal checkpoint 分别只做
  `288,937,974 /305,219,702 /365,700,214 bytes` 全 SHA 核对，均未 load；现有 SAM2 明确不是 SAM-v1/DEVA 的替代。
- adapter smoke=`[1,32,32,16]`、positive-alpha pixels=`189`、identity nonzero gradients=`48/48`、base gradients absent；
  GPU start/peak=`1/310 MiB`，cgroup=`8,810,483,712 bytes`，9 samples/0 errors，wall=`3.154 s`。manifest=
  `6 entries /29,935 bytes`，完整 run=`8 files /31,328 bytes`；audit=`1,602 bytes /14d2b78b...8b64`。
- `V51-F45 resolved`；新增 `V51-F46 active`：source/adapter 可行，但当前 binary actor-union evidence 与缺失的 upstream
  DEVA/SAM-v1 权重使 identity training 未就绪。下一步只允许预注册 F0a train-only SAM+DEVA identity-mask materialization；
  不下载即训练、不用 metadata/SAM2/quality target 冒充输入。H/S/C/validation/test/KITTI=false，F1/F2=false，
  M2/M3=pending。freeze=`configs/worldsim_v51/stage_f_f0_source_preflight_freeze_v1.yaml`。

## V5.1 Stage F F0 source/adapter preflight 已预注册（2026-08-18）

- task=`WS-V51-M1-F-IDENTITY-EMBEDDING-01`。Gaussian Grouping 官方 source 已冻结：ECCV 2024 paper=
  `10,225,908 bytes /61e82145...823`，official repo=`lkeab/gaussian-grouping@0ab60afe`、tree=
  `036936e1...fd16`、Apache-2.0。论文方法页与代码共同固定 identity dim=`16`、SH degree=`0`、alpha-composited
  identity render、共享 1x1 classifier、2D CE、Euclidean `k=5` KNN KL、sample=`1000`、3D loss weight=`2`。
- 忠实输入必须是 SAM everything masks 经 DEVA semionline 关联后的跨视图一致 short IDs。现有 V5 observation 只保存
  binary moving-actor-union probability，instance IDs 明确未保留，不能拿 U2/B3 或 H target 代替 identity label。
  预检只读取 `0/40/80/120/160 × cameras 0/1/2` 的 train-only 文件存在性、NPZ schema 与 instances metadata，
  不读取 image/mask pixels 或任何 quality。
- normative frozen-base adaptation 明确区别于上游 joint reconstruction：geometry/appearance/opacity/dynamic poses 全冻结，
  只允许学习 16D identity encoding 与 shared classifier；上游 SAM+DEVA association、2D CE、3D KL 保持。禁止 Bayesian
  init、DINO、anchor、UNKNOWN、evaluation labels 和 binary-union substitute。
- preflight 将做 16-channel differentiable gsplat gradient smoke，核对 frozen base 无梯度；同时只报告上游 DEVA/SAM-v1
  checkpoint 是否存在，不下载、不运行。预期当前 input-ready=false 时正常收口到 F0a train-only SAM+DEVA identity-mask
  materialization 预注册，不得直接启动 identity training。H/S/C/validation/test/KITTI=false，F1/F2=false，M2/M3=pending。
- 首次 r023 在创建 status 前因 shared `_git(project, *args)` 调用漏传 `PROJECT` 而只留下 resolved config；无 source/data/
  GPU/quality read。r024 已证明 `V51-F44` 修复有效并完成 source/train-only schema 内存检查，但在 16D smoke 前对尚未
  初始化的 CUDA device 调 `reset_peak_memory_stats` 而 blocked；没有 report 或训练。`V51-F45` recovery 只先显式绑定并
  初始化 device，再重置计数；r025 从头运行。

## V5.1 Stage E r022 H rejected；已进入 Gaussian Grouping（2026-08-18）

- canonical r022=`20260818T020000Z__m1-stage-e-e0b-h-evaluation-s20260814-r022`，source/tree=
  `3a84be68...ade/1bc14e34...12`，status=`rejected`，conclusion=
  `e0b_rejected_stop_e1_e2_advance_gaussian_grouping`。12 个 frozen H views 上的 U2/B3、D0、E0B 三臂 target、
  persisted float16 precision、逐 view metrics、equal-view/equal-scene aggregates 与 checkpoint immutability 均独立复算 exact。
- primary gate vs U2/B3：BF1 positive scenes=`2/3` PASS，但 scene-balanced BF1=`-0.0002566`、IoU=
  `-0.0925468`、FN=`+0.1899473` 全 FAIL。机制门 vs D0 更直接显示 node elevation 没有稳定增益：BF1 nonnegative
  scenes=`1/3`，mean BF1=`-0.0004762`、IoU=`-0.0210926`、FN=`+0.0204707`，四项全部 FAIL。
- 逐场 `(ΔBF1, ΔIoU, ΔFN)` 相对 U2/B3：0471=`(+0.124700,+0.166308,+0.075097)`、1087=
  `(+0.056490,-0.158934,+0.217539)`、0379=`(-0.181960,-0.285014,+0.277206)`；相对 D0：0471=
  `(+0.001927,+0.000857,-0.005733)`、1087=`(-0.000410,+0.000483,-0.000607)`、0379=
  `(-0.002946,-0.064618,+0.067752)`。结论是否定 `fine_q50 + mean/max node evidence + 原 D0 propagation` 的稳定性，
  不是否定一切节点或图方法。
- manifest=`21 entries /1,197,572 bytes`，完整 run=`23 files /1,201,649 bytes`；audit=`6,314 bytes /
  5ced73db...104f`。GPU start/peak=`4/10,724 MiB`，cgroup peak=`11,474,419,712 bytes`，124 samples、
  0 errors、wall=`151.543 s`。`V51-F42 active`；E1 PanoGS/E2 AG²aussian 停止，禁止依据 r022 回调 voxel
  level/aggregation/threshold 或重读 H。S/C/validation/test/KITTI 未读，M2/M3=pending；下一步只允许 Gaussian Grouping
  faithful source audit 与 no-quality preflight。freeze=`configs/worldsim_v51/stage_e_e0b_h_evaluation_freeze_v1.yaml`。

## V5.1 Stage E E0b r021 no-quality operator 已冻结（2026-08-18）

- canonical r021=`20260818T014000Z__m1-stage-e-e0b-operator-s20260814-r021`，source/tree=
  `e573fe4f...2b74/d47edb41...d802`，status=`done`，conclusion=
  `e0b_fine_q50_same_propagation_sidecars_ready_without_quality_read`。三场 node/quotient directed edges=
  `561,618/2,991,329`、`620,540/3,289,464`、`764,752/4,037,917`。
- 相对 frozen raw D0，E0B posterior 改变的 Gaussian count/fraction 为 0471=`4,065/0.472887%`、
  1087=`1/0.000107%`、0379=`475/0.040007%`。这证明 operator 不是全局 no-op，但不含质量含义；1087 几乎不变是
  后续 matched H 的重要风险，禁止据此改 coarse level、aggregation 或 threshold。
- 独立 auditor 从 frozen B3/45 observations/raw KNN/fine assignment 重建 node evidence、quotient、1/2-hop affinity、
  progressive result 与 Gaussian broadcast，三场 full arrays exact。manifest=`13 entries / 18,222,912 bytes`，完整 run=
  `15 files / 18,225,498 bytes`；audit=`2,635 bytes / 1e5a0564...650c`。
- resource=`1 MiB GPU / 9,532,755,968 bytes cgroup /45 samples /0 errors /92.608 s`；quality/E1/E2=false，
  M2/M3=pending。下一门只允许先预注册 H matched `U2/B3 G0 vs D0 vs E0B`，freeze=
  `configs/worldsim_v51/stage_e_e0b_same_propagation_freeze_v1.yaml`。

## V5.1 Stage E E0b same-propagation 已预注册（2026-08-18）

- E0b 冻结使用各场 `fine_q50`：选择规则是在 E0a 三场共同通过的 level 中按 frozen edge-length quantile 升序取第一档，
  即最小结构干预；不按 density gain、seed conflict 或任何 H quality 排名，也不搜索 resolution。
- 唯一机制变化是 raw Gaussian→voxel node：node unary 为 member U2/B3 posterior 算术均值；逐 view SAM probability
  为有效 member 的 visibility-weighted mean，node visibility 取有效 member maximum；frozen directed KNN 映射到 node 后
  去 self-loop/deduplicate，再按 D0 原算子 symmetrize、构造 exact 2-hop。
- seeds=`0.9/0.1`、soft binary cosine、visibility-product affinity、distance decay=`0.5`、progressive thresholds=
  `[0.9,0.8,0.7,0.6,0.5]`、fixed-point schedule、UNKNOWN=`0.5` 全部与 D0 相同，node 结果常量广播回 member
  Gaussians。r021 只生成 full-H no-quality operator sidecar，不渲染、不读 H/S/C/validation/test/KITTI quality。
- r021 PASS 后仍须先冻结 operator，再另行预注册 H matched `U2/B3 G0 vs D0 vs E0B`；E1 PanoGS/E2 AG²aussian
  继续锁定，M2/M3=pending。config=`configs/worldsim_v51/stage_e_e0b_same_propagation_v1.yaml`。

## V5.1 Stage E E0a r020 结构门通过并冻结（2026-08-18）

- canonical r020=`20260818T012000Z__m1-stage-e-e0a-density-s20260814-r020`，source/tree=
  `4342ddb3...0c13/2a264757...e4645`，status=`done`，conclusion=
  `e0a_structural_density_gate_pass_preregister_e0b_same_propagation`。三场 `q50/q75/q90` 全部通过 node reduction、
  Gaussian-weighted observation union 严格增加与 raw-zero Gaussian rescue 三项结构门。
- fine/medium/coarse node count：0471=`561,618/363,221/171,742`，1087=`620,540/414,262/230,295`，
  0379=`764,752/497,260/257,892`；对应 union gain 分别为 0471=`0.133301/0.265919/0.507044`、
  1087=`0.000771/0.002053/0.004459`、0379=`0.039214/0.094326/0.187572` views/Gaussian。
- 独立审计逐项重建 9 份 world-origin voxel assignment 与 45 个 train-only visibility denominators，复算全部 density
  metrics/gate、18-entry manifest 与 repeatability exact；report=`6,130 bytes / 8df03b2a...5d34`。run inventory=
  `18 entries / 44,321,779 bytes`，完整 run=`20 files / 44,325,325 bytes`；GPU peak=`1 MiB`、cgroup peak=
  `9,423,683,584 bytes`、159 samples、0 errors、wall=`182.733 s`。
- 边界：E0a 只证明结构层能合并观测，不是质量提升。1087 即使 coarse gain 也仅 `0.004459`；0471 coarse seed-conflict
  Gaussian fraction=`3.4661%`，因此不能按最大 density gain 事后挑 coarse。E0b 只能先用 H-independent、最小结构改动规则
  冻结 level，再做 same-propagation matched A/B；E1 PanoGS、E2 AG²aussian、S/C/validation/test/KITTI 仍锁定，
  M2/M3=pending。`V51-F39` 已由 v2/r020 完整重跑解决；result freeze=
  `configs/worldsim_v51/stage_e_e0a_superprimitive_probe_freeze_v1.yaml`。

## V5.1 Stage E r019 blocked；E0a v2 recovery 已预注册（2026-08-18）

- r019=`20260818T010000Z__m1-stage-e-e0a-density-s20260814-r019` 在 0471/1087 完成后、0379 voxel-size
  quantile 前 fail-closed：frozen 0379 KNN 的 `7,123,746` edges 中有 `34` 条零长度边，v1 错误要求所有 edge
  length 严格为正。status=`blocked`，13 files / `28,114,119 bytes`；没有 terminal density gate、传播或 quality read，
  两场 partial sidecar 不得晋级或复用为 canonical。
- 独立 edge audit：0471/1087/0379 zero edges=`0/0/34`、nonfinite=`0/0/0`；positive q50/q75/q90 meters=
  `0.177945/0.314480/0.626550`、`0.143529/0.233341/0.379943`、`0.147677/0.264845/0.487519`。
  v2/r020 唯一变化是从 scale quantile 排除零长边，仍保留全部 Gaussian；levels/gate/inputs/locks 全部继承 v1。
- `V51-F38/F40` 记录两次 PowerShell→SSH 旁路查询 quoting 复发，均未触及 run；现统一改为仓库内 CLI auditor。
  `V51-F39` 记录 r019 工程阻塞。r020 必须从 frozen input 完整重跑，不读取 r019 partial method evidence；
  E0b/E1/E2、H/S/C/validation/test/KITTI 仍锁定，M2/M3=pending。

## V5.1 Stage E E0a no-quality structural probe 已预注册（2026-08-18）

- Stage E 第一小步固定为 `E0a simple voxel structural observation-density probe`，不是 PanoGS 复现，也不声称方法
  质量。三档 voxel size 只由各场 frozen directed-KNN 正边长的 `q50/q75/q90` 导出，world-origin grid、无 level
  选择、无 learned anchor、无 DINO/motion/SAM probability/base-model membership/quality target、无传播或参数搜索。
- raw comparator 为一 Gaussian 一 node；voxel node 的 observation 仅取 15 个 frozen train-only views 中 member
  availability/reliability/visibility 的 union。逐场至少一档同时满足 node count 严格减少、Gaussian-weighted union views
  严格增加、至少救回一个 raw-zero-observation Gaussian 才 PASS；seed conflict 只报告，不能用于选 level。
- E0a PASS 只解锁 E0b `same propagation, raw Gaussian vs voxel super-primitive` 的预注册；FAIL 则 reject node
  elevation、停止 E1/E2 并进入 Gaussian Grouping。PanoGS official paper/repo 已绑定 CVPR 2025、commit=`8dfb69b`、
  Apache-2.0，但 E1 execution 仍锁定；AG²aussian E2 同样锁定。H/S/C/validation/test/KITTI 未读，M2/M3=pending。

## V5.1 Stage D r018 H rejected；已自动进入 Stage E（2026-08-18）

- canonical r018=`20260818T003000Z__m1-stage-d-d0-h-evaluation-s20260814-r018`，source=`2cd98b3`，
  status=`rejected`，conclusion=`d0_progressive_rejected_skip_d1_advance_super_primitive_or_anchor`。三场
  `8/1/3=12` matched views、三臂 float16 metric replay、checkpoint/layout、21-entry manifest 均独立审计 exact。
- 相对 U2/B3 G0，D0 的 scene-balanced delta：BF1=`+0.0002196`、IoU=`-0.0714543`、FN semantic mass=
  `+0.1694766`、FP=`-0.0335564`、Brier=`-0.0064368`、ECE=`-0.0091008`、NLL=`+0.1051994`。BF1 正向
  scene=`2/3`，但 IoU 与 FN 两个安全门 FAIL；0471/1087/0379 的 BF1 delta=`+0.122773/+0.056899/-0.179014`。
- 结论只是否定 raw-Gaussian faithful progressive propagation 的跨场稳定性：它能减少 FP/部分 calibration error，
  但在 1087/0379 明显牺牲 IoU 并增加漏检；不得由此声称 graph 整体无效，也不得在 H 调 threshold/seed/edge。
  D1 禁止，`V51-F37 active`；路线已自动切到 `WS-V51-M1-E-NODE-ELEVATION-01 / E0`。
- resource PASS：GPU start/peak=`4/10,724 MiB`，Torch allocated/reserved=`10,062.119/10,372 MiB`，cgroup=
  `11,357,159,424 bytes`，108 samples、0 errors、wall=`124.897 s`。freeze=
  `stage_d_progressive_h_evaluation_freeze_v1.yaml`；S/C/validation/test/KITTI 未读，M2/M3=pending。

## V5.1 Stage D D0 H matched evaluation 已预注册（2026-08-18）

- 新冻结 `stage_d_progressive_h_evaluation_v1.yaml`：H 只使用 V5 已接受的 0471/1087/0379、共 `8/1/3=12`
  个 evaluation rows；主比较器为 immutable `U2/B3 G0`，同时报告 frozen V5 `U2/B3+G3`，候选只读取 r017 D0
  sidecar。三臂均从持久化 `float16` probability 重新计算同一套 BF1/IoU/FN/FP/Brier/ECE/NLL，target 明确为
  `frozen_v5_sam_binary_mask_proxy_not_ground_truth`，仅用于评测，绝不回流方法。
- H gate 原样绑定 top-confidence plan §23.1：至少 `2/3` scene 的 BF1 delta 为正、scene-balanced BF1 delta `>0`、
  IoU delta `>=0`、FN semantic-mass delta `<=+0.02`。PASS 才可冻结 D0 并 exact-once 读取 S；FAIL 必须
  reject progressive、跳过 D1、自动进入 super-primitive/anchor，禁止按 H 结果调 threshold/edge/seed。
- runner 对 graph manifest/NPZ、D0 freeze/sidecar、checkpoint before-after、实时 `(base_model,base_index)` 布局、
  12-view denominator 和 float16 metric source 全部 fail-closed；GPU start `<=512 MiB`、peak `<=24,000 MiB`、
  cgroup `<=80 GiB`。S/C/validation/test/KITTI 未读，M2/M3=pending；须在 clean prereg commit 后运行 r018。

## V5.1 Stage D r017 D0 full-H operator 已冻结（2026-08-18）

- canonical r017=`20260818T001000Z__m1-stage-d-d0-operator-s20260814-r017`，source=`d3321bb`，status=`done`，
  conclusion=`d0_full_h_operator_sidecars_ready_without_quality_read`；13/13 manifest entries、15 run files、三场 sidecar
  arrays/seed immutability/UNKNOWN mapping/report denominators 全部独立审计 exact。4,096-node repeat NPZ byte-exact，SHA=
  `f1c6157a...23c8`。
- 0471/1087/0379 exact 1-hop neighbors=`6,641,790/7,150,770/9,155,680`，2-hop=
  `14,363,054/15,744,408/19,063,808`；actor seeds=`47,369/236/607`，final actor=
  `52,764/244/621`，final UNKNOWN=`1,085/4/85`。这些是方法结构/coverage 证据，不是质量 verdict。
- resource PASS：cgroup peak=`9,381,789,696 bytes`，GPU=`1 MiB`，65 samples、0 errors、memory events exact，
  wall=`132.908 s`。summary/status/manifest SHA=`febeef0d...d05f/ac7e7548...db15/ec10bab2...93ed`。
- freeze=`stage_d_progressive_operator_freeze_v1.yaml`。H quality/render、S/C/validation/test/KITTI 未读，M2/M3=pending；
  下一步必须先预注册 H matched evaluation，对比 U2/B3 G0 与 frozen V5 G3，不能先看质量或调 D0。

## V5.1 Stage D D0 sparse operator 已实现并预注册 full-H no-quality run（2026-08-18）

- 新增 clean-room `progressive_propagation.py`：冻结 directed KNN 先对称化，再构造 exact 1/2-hop CSR；每个 Gaussian
  的逐视图 SAM soft probability 映射为 L2-normalized `[1-p,p]`，pair affinity 为共同有效 visibility-product 加权
  cosine；region affinity 按 member count 与 distance decay=`0.5` 聚合。
- U2/B3 posterior 只产生 immutable actor/background high-confidence seeds；每级 threshold 到 fixed point 后再降阈值，
  actor/background 同时过门时取更高 region affinity，exact tie 与最终无支持节点保持 UNKNOWN=`0.5`。没有 one-shot
  smoothing、DINO uplift、learned/gated/motion edge、node change 或参数搜索。
- pure regression=`9 passed`，覆盖公式、visibility、exact distance-2、strict→relaxed expansion、UNKNOWN/tie、边顺序
  确定性、manifest validity/neutral fill、future-quality fail-closed 和 CLI。首个 UNKNOWN fixture 误把可由 Background
  合法扩张的节点当孤立点，登记 `V51-F36 resolved` 后用真正无支持节点修正。
- full-H operator config=`stage_d_progressive_operator_v1.yaml`：只读 15 train-only views/scene，生成三场 D0 sidecar 与
  4,096-node byte-exact repeat probe；70 GiB cgroup、5,400 s total、GPU<=1 GiB fail-closed。H quality/render、S/C、
  validation/test/KITTI 未读，M2/M3=pending；须先 clean commit 再运行。

## V5.1 Stage D r016 D0 preflight 已冻结（2026-08-18）

- canonical r016=`20260818T000000Z__m1-stage-d-d0-preflight-s20260814-r016`，source=`99a626b`，status=`done`，
  conclusion=`d0_faithful_port_inputs_and_source_ready_without_quality_read`。29 个 project/upstream/input identities 全部
  exact；独立 replay report byte-exact，SHA=`b84cb719...4b7b`，bytes=`10,433`。
- scene 0471/1087/0379 Gaussian=`859,613/931,223/1,187,291`、directed KNN edges=
  `5,157,678/5,587,338/7,123,746`、matched evaluation views=`8/1/3`，均与冻结 V5 输入一致。
- SAI3D upstream commit/tree/source hashes 与“无显式 LICENSE”inventory 通过；实现继续执行 clean-room policy。
  本 run 只 hash quality-bearing diagnostics，没有解析其 payload；H/S/C/validation/test quality 均未读，M2/M3=pending。
  freeze=`configs/worldsim_v51/stage_d_progressive_preflight_freeze_v1.yaml`。下一步只实现/测试 D0 sparse operator。

## V5.1 Stage D D0 faithful progressive preflight 已预注册（2026-08-18）

- 路线已按冻结顺序从 rejected LUDVIG uplift/raw graph 自动推进到
  `WS-V51-M1-D-PROGRESSIVE-01 / D0`；U2/B3 继续作为 immutable matched baseline，C0 因上一路线拒绝不存在，
  strong external baseline 保留 V5 `U2/B3+G3`。D0 若 H gate 失败，将 reject progressive、禁止 D1，并自动进入
  super-primitive/anchor。
- 官方 SAI3D source 固定为 commit=`1d9a6a2`、tree=`7320924`；上游仓库未提供显式 LICENSE，因此只按论文公式做
  clean-room 实现，不复制上游源码。faithful contract 固定 raw Gaussian node、冻结 KNN geometry 邻接、共同可见的
  多视图 SAM soft binary distribution cosine、逻辑距离 `<=2`、distance decay=`0.5`、progressive thresholds=
  `[0.9,0.8,0.7,0.6,0.5]`，以及未充分支持节点最终 `UNKNOWN`，不做 one-shot smoothing 或参数搜索。
- 三个 H historical inputs 已逐 SHA 绑定：Gaussian=`859,613/931,223/1,187,291`，directed KNN edges=
  `5,157,678/5,587,338/7,123,746`，matched evaluation views=`8/1/3`。当前只允许 source/input/license/hash preflight
  与 pure operator tests，不读取任何 quality payload。
- H gate 预注册为 `>=2/3` scene BF1 positive、scene-balanced BF1 `>0`、IoU delta `>=0`、FN semantic mass delta
  `<=+0.02`；通过才可冻结 D0 并 exact-once 读取 S。S/C/validation/test/KITTI quality 仍锁定，M2/M3=`pending`。
  config/auditor/test=`stage_d_progressive_preflight_v1.yaml`、`audit_worldsim_v51_d0_preflight.py`、
  `test_worldsim_v51_d0_preflight.py`。launcher/runtime/protocol 小坑登记 `V51-F32–F35 resolved`；其中旧 P0/Stage-B
  plan hash 采用显式 `3d7f7481… → b4888476…` authorized append chain，历史 freeze 不改写。

## V5.1 Stage B r015 H gate rejected；LUDVIG uplift/raw graph 收口（2026-08-18）

- canonical r015=`20260817T173940Z__m1-stage-b-h-evaluation-s20260814-r015`，source=`0a79a56`，status=`rejected`。
  90/90 views、3 checkpoints before/after exact；membership 明确为 evaluation-only model proxy，未回流方法输入。
- H gate：evaluable=`2/3` PASS；Rigid coverage mean=`0.8429098` PASS；heldout scene-balanced `B1-B0=+0.0227772`
  PASS；但 positive B1 margin scenes=`0/2` FAIL，scene-balanced B1 actor-background margin=`-0.1099492` FAIL。
  0471/0379 B1 margin=`-0.1212799/-0.0986184`；1087 无 >=32 covered active actor，按合同 abstain。
- 关键结论：B1 在三场 heldout reprojection 都改善（`+0.026951/+0.023009/+0.018372`），但 actor feature 不比最近
  Background 更紧致；因此不能用 2D 重投影提升冒充 semantic ownership，LUDVIG uplift 与 raw graph 同时 rejected。
- resource PASS：NVIDIA/Torch reserved=`22,570/23,354 MiB <=24,000`，cgroup=`14,221,561,856 bytes`，
  1,211 samples、0 errors、duration=`896.320 s`。独立 audit 12/12 manifest、14 files、gate/checkpoint/locks exact；
  r014↔r015 离散 exact，241 个末位 float 差异 max=`4.976e-13<=1e-12`。
- freeze=`stage_b_h_evaluation_freeze_v1.yaml`；`V51-F15/F28 resolved`，新增 `V51-F30 resolved`、`V51-F31`。
  下一步跳过 raw LUDVIG graph，先预注册 faithful progressive propagation；S/C/validation/test/KITTI 未读，M2/M3 pending。

## V5.1 Stage B r014 resource blocked；r015 24,000 MiB recovery 待预注册（2026-08-18）

- r014=`20260817T172012Z__m1-stage-b-h-evaluation-s20260814-r014`，source=`9b151c8`；90/90 views、3 scene reports
  和 aggregate report 均已计算并在 gate 前持久化，但 NVIDIA/Torch reserved peak=`22,570/23,354 MiB` 超过共同
  ceiling=`22,528 MiB`，故 formal status=`blocked`，禁止读取或引用其中 quality verdict。
- cgroup=`14,305,161,216 bytes`，1,208 samples、0 monitor errors、duration=`897.647 s`，GPU 已释放。r014 共
  `10 files`、无 partial；status/resources/report/progress/resource-samples SHA=`6409545b...f6d1/ffc98a00...674e/`
  `510f82ec...227c/61475cb1...06ed/8fae05eb...7cf`。登记 `V51-F28`。
- recovery 只能把 NVIDIA/Torch 两项 ceiling 提高到 `24,000 MiB`，其他 inputs/views/operator/proxy/pairs/gate/locks
  全部继承 v1，并以新 r015 从原冻结输入完整重跑。第一次 blocked inventory 把 `events.jsonl` 误写为 `events.json`，
  不影响任何 artifact；修正后登记 `V51-F29 resolved`。

## V5.1 Stage B H evaluation-only gate 已机器预注册（2026-08-18）

- 计划 formal r014 只读 H：r012 frozen B0/B1 Gaussian features、r010 evidence features、r013 evaluation features，
  并用相同 base checkpoint 重渲染 `45 evidence + 45 evaluation` views；最终 heldout remainder=`4` 不可读。
- membership 固定声明为 `model_membership_proxy_not_ground_truth`，只在 evaluation 使用 `RigidNodes.point_ids`、
  reference frame 80 的 active flag 与 world means；不得回流为 method/PCA/uplift 输入。每 actor 至少 32 covered
  Gaussian，最多 4,096 deterministic unordered pairs；最近 Background 用 frozen world geometry 的 cKDTree workers=1。
- 指标冻结为 aggregate→single-view same-Gaussian cosine repeatability、same-actor cosine、nearest-background cosine/
  margin、heldout renderer reprojection cosine 和 Background/Rigid coverage；B0/B1 reprojection 使用 exact common pixels。
- H gate：至少 2/3 scenes evaluable、至少 2 scene B1 margin>0、scene-balanced B1 margin>0、mean Rigid
  coverage>=0.60、scene-balanced heldout `B1-B0>=-0.01`。未过门则同时 reject uplift/raw LUDVIG graph 并自动转下条
  frozen route；过门也只解锁 S exact-once。
- config/module/runner/test=`stage_b_h_evaluation_v1.yaml`、`feature_evaluation.py`、
  `run_worldsim_v51_h_evaluation.py`、`test_worldsim_v51_feature_evaluation.py`。在 clean prereg commit 前仍未读取
  r012/r013 quality；S/C/validation/test/KITTI 锁定，M2/M3=`pending`。

## V5.1 Stage B r013 H heldout feature transform 已冻结（2026-08-18）

- canonical r013=`20260817T170028Z__m1-stage-b-h-eval-feature-s20260814-r013`，source=`b359541`，
  status=`done`，conclusion=`h_heldout_dinov2_features_transformed_with_frozen_pca_without_quality_read`。
  45/45 evaluation views、328,320 patches；official ViT-g params/keys=`1,136,486,912/568`，checkpoint before/after exact。
- r010 frozen PCA 只做 transform、`fit=false`；首图 raw repeat 与首图 transform repeat 均 bit-exact。45 个
  `[40,64,114] float32` sidecars 逐文件 SHA、逐数组 content SHA、shape/dtype/finite 独立审计 exact；feature bytes=
  `48,452,027`，record chain=`2ca3f8bc...9d50`。
- resource PASS：GPU start/peak=`1/6,702 MiB`，Torch allocated/reserved=`6,070.182/6,376 MiB`，cgroup=
  `17,320,468,480 bytes`，123 samples、0 errors、duration=`70.806 s`；run=`57 files / 48,547,857 bytes`，无 partial/scratch。
- summary/status/manifest/fingerprint/feature-manifest SHA=`8d630c35...e262/7742ec49...d7d8/15625e56...fbf3/`
  `7613db20...501c/8824a8dc...f73c`。result freeze=`stage_b_h_eval_feature_freeze_v1.yaml`。
- 本门未读 membership/uplift/method quality，未启动 renderer；S/C、validation/test/KITTI 未读，M2/M3=`pending`。
  下一步只能先提交 H evaluation-only runner/config，再第一次读取 r012/r013 quality；`V51-F15` 仍 active。

## V5.1 Stage B H heldout feature transform 已预注册（2026-08-18）

- 下一正式 run 计划为 `m1-stage-b-h-eval-feature-s20260814-r013`。它只从冻结 image manifest 精确选择
  `0471/1087/0379 × frames 2/42/82/122/162 × cameras 0/1/2 = 45` 个 H evaluation views；这些 frame
  全部满足 `frame % 5 == 2`，并显式排除 `frame % 5 == 4` 的最终 heldout remainder。
- 使用 r010 已冻结的 official DINOv2 ViT-g/14-reg4 与 PCA state；只执行 transform，`fit=false`，不得重估
  mean/std/PCA。首个 raw feature 重复推理和首个 PCA transform 均须 bit-exact；输出 45 个
  `[40,64,114] float32` deterministic sidecars，逐项绑定 image/model/checkpoint/PCA/content identity。
- 本门只授权为随后 H evaluation-only runner 准备 heldout feature：不读取 r012 uplift 数值、不读取
  Background/Rigid membership proxy、不启动 renderer、不计算或查看任何 method quality。S/C、validation/test/KITTI
  继续锁定，M2/M3=`pending`；`V51-F15` 仍 active。
- config/runner/test=`configs/worldsim_v51/stage_b_h_eval_feature_v1.yaml`、
  `scripts/run_worldsim_v51_h_eval_feature.py`、`tests/test_worldsim_v51_h_eval_feature.py`。只有 clean prereg commit
  与 regression 通过后才能创建 r013；本门结果还不能判定 H gate。
- 首次聚合 regression 把 DriveStudio uplift test 误放入 motionproj interpreter，得到 `1 failed / 9 passed`；新 heldout
  tests 本身为 `8/8 PASS`。该调用错误登记为已解决 `V51-F24`，后续按冻结双环境分别复跑，不改 runtime contract。
- 第二次聚合命令在 PowerShell→SSH 边界被本地提前解释 `$(find ...)`，测试未启动且无状态变化；登记已解决
  `V51-F25`，后续检查不再使用嵌套 command substitution。
- r013 完成后的首次只读 inspection 又因双层 shell 内嵌多语句 Python `-c` 被本地 parser 拒绝；远端未执行且无
  状态变化，登记已解决 `V51-F26`，改用独立只读 auditor 文件。
- auditor/docs 首次 scp 又因 staging workdir 下漏写 `motion_proj/` 前缀而仅 auditor 成功、docs 未传；run/repo 未改，
  登记已解决 `V51-F27`，按解析后的精确 source path 重传。

## V5.1 Stage B r012 H 45-view B0/B1 uplift 门通过（2026-08-18）

- canonical r012=`20260817T163100Z__m1-stage-b-h-uplift-s20260814-r012`，source=`4fc07cb`，status=`done`，
  conclusion=`h_45_view_b0_b1_gaussian_sidecars_ready_without_quality_read`。45/45 train-only views、3/3 H scenes、
  6 个 `[N_gaussian,40]` B0/B1 sidecars 均完成；3 个 base checkpoint before/after SHA exact。
- scene 0471/1087/0379 coverage=`0.8986823140/0.8479816328/0.8529442234`，covered Gaussian=
  `772,519/789,660/1,012,693`，supported intersections=`368,806,013/580,912,738/360,850,116`，B0/B1 L2=
  `1826.1010/2077.5490/1875.4642`；这只证明 uplift/denominator 与两臂 non-alias，不构成方法质量结论。
- resource PASS：GPU start/peak=`1/20,554 MiB`，Torch allocated/reserved=`19,314.634/20,202 MiB`，cgroup=
  `14,450,888,704 bytes`，846 samples、0 error、duration=`621.170 s`，成功后无活跃 GPU。`V51-F23 resolved`。
- 独立审计 6/6 NPZ file/content SHA、dtype/shape/finite/nonnegative、19/19 manifest entries、3 checkpoint identities、
  summary/status/fingerprint/locks 均 exact；sidecar=`811,046,036 bytes`，run=`21 files / 811,273,469 bytes`。
  summary/status/manifest/fingerprint/sidecar-manifest SHA=`156d1a34...046e2/4574725d...7c34/752ade66...d0b/`
  `6f1d5e7d...8e7f/77477253...5c78`，record chain=`bb7563ad...69d4`。
- result freeze=`configs/worldsim_v51/stage_b_h_uplift_freeze_v1.yaml`。membership proxy/method quality 未读；S/C、
  validation/test/KITTI 未读，M2/M3=`pending`。下一步只允许先预注册 H evaluation-only proxy/repeatability/heldout
  reprojection 合同以处理 `V51-F15`，不得直接查看或调节质量结果。

## V5.1 Stage B r011 resource blocked，r012 22 GiB recovery 已预注册（2026-08-18）

- canonical blocked r011=`20260817T161351Z__m1-stage-b-h-uplift-s20260814-r011`，source=`40f4d64`；45/45 H views、
  3/3 scenes、6 个 B0/B1 Gaussian feature sidecar 和 3 个 checkpoint immutability audit 均已完成，随后仅因
  NVIDIA peak=`20,554 MiB > 18,432 MiB` 而 fail-closed。Torch allocated/reserved=`19,314.634/20,202 MiB`，
  cgroup peak=`13,328,011,264 bytes`，799 samples、0 monitor error、duration=`588.750 s`；不是 OOM、renderer、
  sparse transpose、checkpoint 或算法质量失败。r011 保留 `blocked`，不得改写或复用。
- r011 的只读 post-compute 证据已在资源门前持久化：processed=`45 views / 3 scenes`、sidecar=`6 / 811,045,640 bytes`；
  Gaussian coverage 0471/1087/0379=`0.8986823140/0.8479816328/0.8529442234`，B0/B1 L2 difference=
  `1826.1010/2077.5490/1875.4642`。独立审计 6/6 NPZ 的 file/content SHA、dtype/shape/finite/nonnegative、
  manifest chain、3 个 checkpoint before/after 均 exact；这只封存 blocked 诊断，不能把 r011 冻结为 canonical 结果。
- r011 status/resources/report/sidecar-manifest/resource-samples SHA=`a450cdaf...0eee6/0312e190...8a37/`
  `98140571...99c9/88956448...dae6/76422821...ae9c`；sidecar record chain=`5339b880...8e12`，GPU 已释放。
- 登记 `V51-F23`。v2/r012 仅把 NVIDIA/Torch reserved ceiling 从 `18,432` 提升到 `22,528 MiB`，仍低于
  24 GiB 单卡容量；scene/view/order/checkpoint、r010 PCA/sidecars、两级 floor、B0/B1、sparse operator、cgroup/disk/timeout
  与全部 quality/validation/test/KITTI/M2/M3 locks 原字继承。新 run ID 必须完整重跑，不读取 r011 产物作为输入。

## V5.1 Stage B H 45-view B0/B1 uplift 已预注册（2026-08-18）

- planned r011=`m1-stage-b-h-uplift-s20260814-r011`；3 H scenes 各固定 train-only `5 frames×3 cameras=15 views`，
  逐 scene 构造只读 DriveStudio trainer，renderer 使用 model-native `800×450`；base checkpoints、source configs、
  r010 45 feature sidecars、PCA state 与 r005 operator freeze 全部逐 SHA 绑定。
- 为避免把单 view 约 3,200 万 supported intersections 展开为 `[N,40]`，新增 faithful sparse transpose：先以
  SciPy CSR float64 聚合 Gaussian×pixel contribution，再与 dense pixel feature 相乘得到 per-view numerator/mass；
  两级 floor、B0 saturation、B1 normalization 与 r005 数学合同不变。synthetic streaming vs dense parity 必须通过。
- 输出每 scene×arm=`6` 个 deterministic Gaussian sidecar：feature=`[N_gaussian,40] float32`、weight=float64、
  supported-view-count=int32；保留共同 coverage、逐 view denominator、B0/B1 non-alias 和 checkpoint immutability。
- 本门只计算 feature uplift 和结构 denominator，不消费 RGB/LiDAR/membership proxy，不读取 method quality；S/C、
  validation/test/KITTI 未读，M2/M3=`pending`。`V51-F15` 继续 active，必须在后续独立 evaluation-only 合同处理。

## V5.1 Stage B r010 H feature-sidecar/PCA 门通过（2026-08-18）

- canonical r010=`20260817T155859Z__m1-stage-b-h-feature-pca-s20260814-r010`，source=`11c35fd`，status=`done`，
  conclusion=`h_dinov2_feature_sidecars_and_seeded_pca_ready_without_quality_read`。45/45 H views、328,320 patches，
  official ViT-g params/keys=`1,136,486,912/568`，strict missing/unexpected=`0/0`，checkpoint before/after exact。
- 首图 full raw feature 重复推理 bit-exact；两遍 correction=1 stats + seeded randomized 40-D PCA 完成，PCA state SHA=
  `fe9eea72...3231c8`，deterministic NPZ 二次写入 byte-exact。45 个 `[40,64,114] float32` sidecar file/content
  identity 全部复核 exact，总 bytes=`48,447,248`；raw `2,017,198,080-byte` memmap 成功后已删除，scratch 不存在。
- resource：GPU start/peak=`1/6,702 MiB`，Torch allocated/reserved=`6,070.182/6,376 MiB`，cgroup=
  `15,635,017,728 bytes`，172 samples、0 error，duration=`104.472 s`。manifest=`57 files / 48,826,634 bytes`，
  run=`59 files / 48,837,077 bytes`，逐文件 SHA/bytes exact。
- summary/status/manifest/fingerprint/feature-manifest/PCA-state SHA=`c6b81374.../1e8b78da.../160efe34.../c5cacfc4.../`
  `c9b4f669.../fe9eea72...`；result freeze=`configs/worldsim_v51/stage_b_h_feature_pca_freeze_v1.yaml`，`V51-F14 resolved`。
- H RGB 仅用于 feature extraction；membership proxy、renderer/uplift 和 method quality 未读，S/C/validation/test/KITTI 未读，
  M2/M3=`pending`。下一步仅预注册 H 45-view B0/B1 uplift，不得直接评价质量。

## V5.1 Stage B H DINO feature-sidecar/PCA 已预注册（2026-08-17）

- planned r010=`m1-stage-b-h-feature-pca-s20260814-r010`；只读取 historical diagnostic H=
  `0471/1087/0379 × frames 0/40/80/120/160 × cameras 0/1/2 =45 views`，冻结 image manifest 240 records 中
  exact 选择，patch population=`45×7,296=328,320`，低于 500,000 cap，禁止 subsample。
- faithful contract：official DINOv2 ViT-g/14 registers exact checkpoint，`1596×896 → 1536×64×114`，取最近 4 层
  的最后一层；逐图 fp16 autocast，raw float32 写 `2,017,198,080-byte` CPU memmap。首图重复推理必须 bit-exact。
- `V51-F14` reproducibility hardening：固定两遍 float64 statistics、sample std correction=`1`、标准化存储 float32、
  randomized PCA=`40D / random_state=20260814 / whiten=false / sklearn=1.7.2`；state 用 deterministic NPZ 二次写入
  byte-exact，S/C 后续只能 transform、禁止 refit。
- 每 view 持久化 `[40,64,114] float32` patch-grid sidecar 与 image/model/checkpoint/PCA/content 全身份；成功后删除
  raw memmap。runner 不启动 renderer/uplift，不读 membership proxy 或任何 quality，S/C/validation/test/KITTI 未读，
  M2/M3=`pending`。config/module/runner/test=`stage_b_h_feature_pca_v1.yaml`、`feature_sidecar.py`、
  `run_worldsim_v51_h_feature_pca.py`、`test_worldsim_v51_feature_sidecar.py`。

## V5.1 Stage B r009 one-H-view denominator/resource 门通过（2026-08-17）

- canonical r009=`20260817T154359Z__m1-stage-b-one-view-contribution-s20260814-r009`，source=`7f0c6c9`，
  status=`done`，conclusion=`one_h_view_renderer_contribution_denominator_ready`；checkpoint before/after SHA=
  `496356ca...3cfa` exact。正确三层尺寸=`1600×900 / [2,2,2] / 800×450`。
- 真实 denominator：raw/supported intersections=`47,378,525 / 32,030,248`，raw/supported/dropped mass=
  `299,051.805624 / 298,668.303850 / 383.501774`；Gaussian support before/after `1e-3` mass floor=
  `355,759 / 313,764`，drop=`41,995`，全局 coverage=`41.385949% → 36.500611%`；全部 360,000 pixels 有支持。
- 资源：GPU start/peak=`1/14,234 MiB`，Torch allocated/reserved peak=`13,389.991/13,882 MiB`，cgroup=
  `9,593,946,112 bytes`，88 samples、0 error、61.109 s，cleanup 后无活跃 GPU 进程；`V51-F21 resolved`。
- summary/status/manifest/fingerprint/inventory/resources SHA=`b1e2282a.../c0bd3501.../6439a1a9.../4376c139.../`
  `b29f200e.../092be4bf...`；manifest=`8 files / 26,331 bytes`，run=`10 files / 28,156 bytes`，逐文件二次
  SHA/bytes exact。首次独立 verifier 误读 manifest key=`files`，实际 schema 为 `inventory`，立即修正并登记已解决 `V51-F22`。
- result freeze=`configs/worldsim_v51/stage_b_one_view_contribution_freeze_v1.yaml`。loader materialize image/mask/LiDAR=
  true，但 RGB/LiDAR/membership consumption=false；DINO/PCA/uplift/quality/validation/test/KITTI 未读，M2/M3=`pending`。
  下一步只允许预注册 H DINO feature sidecar 与 PCA 合同，不得直接读 quality。

## V5.1 Stage B r008 资源上限 blocked，r009 16 GiB recovery 已预注册（2026-08-17）

- r008=`20260817T153826Z__m1-stage-b-one-view-contribution-s20260814-r008`，source=`eb334fa`，已越过
  DriveStudio import、dataset/trainer/checkpoint、`800×450` renderer 和 contribution 汇总；最终仅因 NVIDIA peak=
  `14,234 MiB > 12,288 MiB` 而 fail-closed。cgroup peak=`9,598,074,880 bytes`，89 个采样、0 monitor error，
  不是 OOM、renderer、LUDVIG 或质量失败。`V51-F20` 因正确尺寸已进入 post-render 资源门而 resolved；新增 `V51-F21`。
- r008 status/events/resolved/resource-samples SHA=`8b8ebe17.../ed216a5b.../c284d64d.../fc0f9788...`；run 仅保留
  4 个 terminal/config/resource 文件，未生成 summary，禁止覆盖或续写。
- v4/r009 只把 NVIDIA/Torch reserved ceiling 从 `12,288` 提升到 `16,384 MiB`，仍低于 24 GiB 单卡容量；
  同时把只读 denominator/resource 诊断移到资源门前持久化。scene/view/checkpoint、两级 floor、renderer、消费合同和
  所有 quality/validation/test/KITTI/M2/M3 locks 均逐字继承；没有读取或据此调节任何方法质量。

## V5.1 Stage B r007 三层分辨率 blocked，r008 model-native recovery 已预注册（2026-08-17）

- r007=`20260817T153300Z__m1-stage-b-one-view-contribution-s20260814-r007`，source=`e06b5ff`，DriveStudio env、dataset、
  trainer/checkpoint 与单视图 renderer 均已启动，随后因预注册错误要求 renderer=`1600×900` 而 blocked；冻结 source
  config 实际 `downscale_when_loading=[2,2,2]`，model-native renderer=`800×450`。0 intersection inventory/quality。
- r007 status/events/resolved/resource-samples SHA=`da279515.../365d212b.../d08a5f96.../8b2ca135...`；`V51-F19` 因已越过
  import 并到达 renderer 而 resolved，新增 `V51-F20` 记录重复踩中既有三层分辨率合同。
- v3/r008 显式分开 sensor=`1600×900`、source downscale=`[2,2,2]`、model-native renderer=`800×450`，只修改尺寸
  身份与错误诊断文本，view/checkpoint/floor/resource/locks 不变。DriveStudio loader 会基础设施性物化 image/mask/LiDAR，
  但 runner 不消费这些值、不计算 feature/quality；该事实在 v3 config/report 中显式记录。

## V5.1 Stage B r006 interpreter blocked，r007 DriveStudio-env 恢复已预注册（2026-08-17）

- canonical blocked r006=`20260817T152900Z__m1-stage-b-one-view-contribution-s20260814-r006`，source=`5e59443`；
  在 `_build_runtime()` import `pytorch3d` 时 `ModuleNotFoundError`，发生于 dataset/trainer 构造和 renderer 启动前。
  status/events/resolved/resource-samples SHA=`06b74ec9.../914fa591.../1b2cb043.../f6157f3f...`。
- 根因是 v1 config 错绑 `/root/autodl-tmp/envs/motionproj/bin/python`；冻结 DriveStudio runner 实际需要独立
  `/root/autodl-tmp/envs/drivestudio/bin/python`。后者已只读确认 torch=`2.1.2+cu118`、CUDA=`11.8`、
  `pytorch3d` 与 `gsplat` 可 import。登记 `V51-F19`，不是 renderer/LUDVIG/资源或质量失败。
- recovery config=`configs/worldsim_v51/stage_b_one_view_contribution_v2.yaml`；只改变 interpreter/env contract，
  scene/view/checkpoint、operator freeze、两级 floor、资源门与所有 consumption/quality locks 原字继承。formal run 还会核对
  `sys.executable` 与 torch/CUDA exact；新 run ID=r007，禁止覆盖或续写 r006。

## V5.1 Stage B one-H-view contribution denominator smoke 已预注册（2026-08-17）

- planned r006 suffix=`m1-stage-b-one-view-contribution-s20260814-r006`；唯一 view=
  scene-0471/index-382/H/frame-0/camera-0，沿用 V5 r027 immutable checkpoint/source config，预期 Gaussian=
  `809,902 Background +49,711 Rigid =859,613`。operator freeze SHA=`d523b0d8...19e25e`。
- runner=`scripts/smoke_worldsim_v51_one_view_contribution.py`，只构造 read-only DriveStudio trainer、执行一次
  `renderer_intersections()` 并统计 raw/`≥1e-4` intersection、`≥1e-3` Gaussian-view mass、全局 Gaussian/pixel
  coverage 与资源。完整 intersection rows 不落盘，checkpoint 前后 SHA 必须 exact。
- 数据集基础设施会物化 image tensor，但 runner 只保留 `normed_time/img_idx`，不访问 RGB/LiDAR 数值；不消费
  Background/Rigid membership proxy，不加载 DINO、不做 PCA/uplift feature、不计算任何 method/quality metric。该门只能
  裁决真实 contribution denominator/资源是否 ready。
- resource gate 预先固定为 NVIDIA/Torch reserved peak `≤12,288 MiB`、cgroup `≤48 GiB`、timeout=`900 s`；DINO 与
  renderer 不并发。validation/test/KITTI 锁定，M2/M3=`pending`。通过后才允许冻结 H feature-sidecar/PCA 执行合同。

## V5.1 Stage B r005 synthetic operator parity 全通过，下一门仅一张 H view denominator smoke（2026-08-17）

- canonical r005=`/root/autodl-tmp/runs/worldsim_v51/WS-V51-M1-B-LUDVIG-UPLIFT-01/20260817T151900Z__m1-stage-b-operator-parity-s20260814-r005`，
  source=`1efa7dd321f7318d10f257d2a4ff2333d357b131`，status=`done`，conclusion=
  `synthetic_b0_b1_and_lazy_bilinear_operator_parity_passed`。
- 11/11 checks PASS：B0/B1 对独立 dense oracle max error=`0/0`，constant error=`0`，lazy bilinear 对
  `align_corners=False` dense max error=`1.1920929e-7`，row/chunk permutation bit-exact；duplicate、intersection floor、
  view-mass floor、zero denominator、float32 output、pruning disabled 全 exact。
- synthetic denominator=`240 intersections → 173 supported`，Gaussian-view=`8 → 7 supported +1 dropped`，covered
  Gaussian=`4/5`；B0/B1 L2 difference=`0.0829221`，排除两臂实现别名。checkpoint 前后完整 SHA 均=
  `746ecb8c...a283`。
- r005 summary/status/manifest/fingerprint/parity-report SHA=`d15b82d1.../5683bf42.../0fc3fe51.../340c6b83.../`
  `c0a4319c...`；manifest=`6 files / 12,521 bytes`、run=`14,025 bytes`，逐文件二次 SHA/bytes exact。machine freeze=
  `configs/worldsim_v51/stage_b_operator_parity_freeze_v1.yaml`；failure delta=`none`。
- 该 PASS 只证明纯 operator 数学/采样合同，不证明 renderer contribution inventory 或 feature quality。下一步只预注册并
  执行一个 H view 的 contribution denominator smoke；DINO/PCA/真实 feature/quality/validation/test/KITTI 仍不可读，
  M2/M3=`pending`。
- result-freeze 首轮 regression=`1 failed / 19 passed`：新增测试把局部变量从 `validate_freeze` 简化为 `freeze` 后，两条
  parity 断言漏改旧名而 `NameError`；此前 run SHA/assert 已通过。只修变量名并登记 `V51-F18`，不改变 r005 结果。

## V5.1 Stage B LUDVIG source/operator parity 已预注册，H quality 仍锁定（2026-08-17）

- external LUDVIG checkout=`/root/autodl-tmp/third_party/ludvig-v51-stage-b`，origin=
  `https://github.com/naver/ludvig.git`，commit/tree=`4461fc515439bb498a75d71738a1e73cf7a452ed /`
  `4d1287b5a8c86b75e67358d3f03cc22d442fb70d`，worktree clean。`LICENSE.txt/NOTICE.txt/solver.py/apply_weights.cu`
  SHA 已冻结；许可证为 non-commercial，源码不 vendor 到项目。
- faithful audit 确认上游核心是逐 intersection 累加 `feature×alpha×T` 与 `alpha×T`，最终统一除以
  `weights+1e-8`；optional pruning 保持禁用。本地 `motion_proj/worldsim_v51/feature_uplift.py` 只重实现冻结数学合同并
  直接消费现有 `renderer_intersections` 的 contribution，不复制上游 CUDA，也不使用 membership proxy。
- operator config/runner=`configs/worldsim_v51/stage_b_operator_parity_v1.yaml` /
  `scripts/audit_worldsim_v51_stage_b_operator_parity.py`。B0/B1 共同 support 固定为 intersection `≥1e-4`、
  Gaussian-view mass `≥1e-3`、epsilon=`1e-8`；内部 float64 canonical reduction、输出 float32、Gaussian row immutable。
- formal r005 前冻结 9 个 synthetic cases：独立 dense reference、重复 index、两级 floor、zero denominator、constant
  feature、row/chunk bit-exact、lazy bilinear 对 `align_corners=False` dense parity、B0/B1 non-alias。该门不加载 DINO、
  不启动 renderer、不读真实 image feature/PCA/任何 quality，validation/test/KITTI 与 M2/M3 均锁定。
- 首轮 pre-formal unit regression=`2 failed / 8 passed`：两处 failure 都来自同一个 synthetic fixture，把 24 个
  intersection 全设为 `1e-4` 后 group mass=`0.0024`，并未低于 `0.001` view floor；其余检查已过。夹具改为仅 5 个
  `1e-4`（mass=`0.0005`）后重跑，登记 `V51-F17`；未创建 formal run，也没有方法/质量失败。

## V5.1 Stage B ViT-g r004 资源/张量门通过，下一门仅 operator parity（2026-08-17）

- canonical r004=`/root/autodl-tmp/runs/worldsim_v51/WS-V51-M1-B-LUDVIG-UPLIFT-01/20260817T150400Z__m1-stage-b-dinov2-resource-smoke-s20260814-r004`，
  source=`935d2b27176d40b8e566c29c23a8f4122f97d5bd`，status=`done`，conclusion=
  `official_dinov2_vitg14_reg4_one_image_resource_and_shape_gate_passed`。
- official ViT-g=`1,136,486,912` params；checkpoint=`568` keys，strict missing/unexpected=`0/0`。last-four output 均为
  `[1,1536,64,114]`、dtype=`float32`、selected finite；这只证明官方 checkpoint/source/preprocess/forward 合同成立，
  不构成 LUDVIG uplift 或任何 paper-method quality 结论。
- resource gate PASS：GPU start/peak=`1/6,702 MiB`，Torch peak allocated/reserved=`6,067.956/6,376 MiB`，
  cgroup peak/max=`15,701,860,352/96,636,764,160 bytes`，49 samples、0 monitor error；cleanup 后 Torch=
  `8.125/48 MiB`。因此 `V51-F12` 的官方资产及 24GB faithful one-image resource 风险均 resolved，但 DINO 与 renderer
  分进程/不并发合同继续有效。
- r004 summary/status/manifest/fingerprint/resource-samples SHA=`27ae3bd2.../97f4dccc.../fc8cf1ab.../d99e0590.../`
  `d569c278...`；manifest=`8 files / 23,039 bytes`、run=`24,854 bytes`，逐文件二次 SHA/bytes 复核 exact。machine freeze=
  `configs/worldsim_v51/stage_b_dinov2_resource_freeze_v1.yaml`。
- PCA/feature sidecar/renderer/method/H/S/C quality/validation/test/KITTI 均未读，M2/M3=`pending`。下一步只允许先冻结并
  通过 synthetic B0/B1 operator parity；`V51-F14/F15` 保持 active，H quality 仍锁定。

## V5.1 Stage B official DINOv2 source 已冻结，ViT-g 单图资源门已预注册（2026-08-17）

- official source checkout=`/root/autodl-tmp/third_party/dinov2-v51-stage-b`，origin=
  `https://github.com/facebookresearch/dinov2.git`，commit=`7764ea0f912e53c92e82eb78a2a1631e92725fc8`，tree=
  `2a27257b79b0633b027a21014bc9360e3c1b3f43`，worktree clean；LICENSE/hubconf SHA-256=
  `600cc67c...b7b2 / c1f5090e...a6f64`。源码仅放在 external third-party checkout，不 vendor 到项目仓库。
- checkpoint 继续继承 r003 machine freeze：bytes=`4,546,140,349`，SHA-256=`746ecb8c...a283`。资源冒烟配置/runner=
  `configs/worldsim_v51/stage_b_dinov2_resource_smoke_v1.yaml` / `scripts/smoke_worldsim_v51_dinov2_resource.py`；
  输入唯一固定为 scene-0471/index-382/frame-0/camera-0，JPEG bytes/SHA=`99,906 / 093d38e8...5819e`。
- faithful contract 固定为 official `dinov2_vitg14_reg`、ViT-g/14 +4 register tokens、raw dim=`1536`、
  `1600×900→1596×896` bilinear + ImageNet normalization、last-four normalized reshape output 均为
  `[1,1536,64,114]`。checkpoint 在 CPU 严格加载，模型参数 FP32，GPU inference autocast FP16；不允许用更小模型、
  更低分辨率或非官方实现替代。
- resource gate 固定为 start NVIDIA used `≤2,048 MiB`、sampled peak `≤22,528 MiB`、cgroup peak `≤80 GiB`、
  timeout=`900 s`、单 DINO 进程且 renderer 不并发。r004 只判 source/checkpoint/input/state-dict/output/resource；
  PCA、feature sidecar、renderer、method/quality/H/S/C、validation/test/KITTI 均不读，M2/M3 保持 `pending`。
- 下一步是在包含本预注册的 clean source commit 上执行唯一 r004。通过后才做 synthetic operator parity；OOM 或资源门失败
  必须写入统一 failure ledger，并按既定 M1 路线推进，不得事后缩小 backbone/输入，也不得停止整个 M1。

## V5.1 Stage B 已授权：U2/B3 fallback + M1 自动换路线进入冻结（2026-08-17）

- 用户已明确授权 Stage A 冻结 survivor=`U2/B3` 作为 fallback 进入 Stage B，并要求 M1 遇到单 arm、单 scene、
  单次工程错误或 paper idea 失败时留档后自动进入下一路线；`V51-F11` 的 governance 阻塞因此 `resolved`，历史冲突
  仍保留。授权 overlay=`configs/worldsim_v51/stage_b_authorization_v1.yaml`，绑定 normative plan、P0、H/S/C roles、
  Stage A closeout、freeze proposal 与 V5 formal30k batch 的原始 SHA，不改写历史 proposal 或 normative plan 字节。
- `WS-V51-M1-B-LUDVIG-UPLIFT-01=running`；当前只执行 freeze/asset identity。第一条路线固定为 faithful LUDVIG
  uplift→条件性 semantic graph，U2/B3 始终作为 matched baseline；原样迁移无效即 rejected 并进入 progressive
  propagation→super-primitive/anchor→Gaussian Grouping→Trace3D→BKI/graph-free，原样有效后才允许创新。
- canonical r001=`/root/autodl-tmp/runs/worldsim_v51/WS-V51-M1-B-LUDVIG-UPLIFT-01/20260817T141000Z__m1-stage-b-input-freeze-s20260814-r001`，
  source=`22149613b9fc958b2bb5351300dd53fdc0d3d221`，status=`done`。`240/240` 张 `1600×900` JPEG=
  `39,747,172 bytes`、8 个 V5 30k checkpoint 与 Gaussian counts 全部 exact；terminal/manifest 二次复核通过。
  machine freeze=`configs/worldsim_v51/stage_b_input_freeze_v1.yaml`。
- r001 summary/status/manifest/fingerprint/image-manifest/checkpoint-manifest SHA=`f6aae6f6.../8b4c9aec.../`
  `8c50882e.../88a4fa17.../be19da2e.../8b84bf9a...`；run=`129,809 bytes`、manifest inventory=`7 files /
  123,976 bytes`。checkpoint download、DINO/renderer、feature extraction、quality/validation/test/KITTI read 均为 false。
- 下一步使用 `configs/worldsim_v51/stage_b_dinov2_download_v1.yaml` 与
  `scripts/fetch_worldsim_v51_dinov2_asset.py`，在 `source /etc/network_turbo` 后下载官方 DINOv2 ViT-g/14
  registers checkpoint，先写 `.partial`，
  以 bytes=`4,546,140,349` 和完整 SHA-256 校验后原子发布；随后依次做单进程资源 smoke、synthetic operator parity、
  H→S→C matched A/B。DINO 与 renderer 不同进程且不在同卡并发常驻。
- r002 单连接在 `106 s` 后只有 `26,566,656 bytes`，执行者保留 prefix 并精确停止唯一 curl；terminal=`blocked`、
  curl exit=`-15`，不是方法失败，登记 `V51-F16`。下一 run 改用 14 个互不重叠 frozen ranges；逐段 bytes/SHA、
  assembled full SHA-256 与 8 MiB×542-part S3 ETag 必须全通过后才 atomic publish，配置/runner=
  `stage_b_dinov2_download_parallel_v1.yaml` / `fetch_worldsim_v51_dinov2_asset_parallel.py`。
- canonical r003=`20260817T142400Z__m1-stage-b-dinov2-parallel-s20260814-r003`，source=`de6221f`，14-range
  download=`1504.935 s`；asset bytes/SHA=`4,546,140,349 / 746ecb8c...a283`，本地 8 MiB×542 multipart ETag=
  `3d1b...-542` exact。逐段、terminal、manifest、final SHA/ETag 二次复核通过；r002 prefix +14 segments=
  `15 files / 4,546,140,349 bytes` 在发布后精确清理，`V51-F16=resolved`。
- r003 summary/status/manifest/fingerprint/asset/segments SHA=`6e79804a.../16cc40ed.../98ce7751.../06e34644.../`
  `b4913ccf.../459e698e...`；machine freeze=`configs/worldsim_v51/stage_b_dinov2_asset_freeze_v1.yaml`。
  下一门是冻结官方 DINOv2 source checkout 后做单图 ViT-g resource smoke；H quality 仍未解锁。
- M2/M3 保持 `pending`；validation/test/KITTI tuning 继续锁定。只有某个 M1 candidate 在 development confirmation
  稳定优于 U2/B3 并冻结后，才允许 exact-once fresh validation；test 只在最终候选上 exact-once。当前 source base=
  `de6221f6ec3a9fec08a620b7938800629226581e`；r001 failure delta=`V51-F11`，r002→r003 engineering delta=
  `V51-F16 resolved`，没有新增方法失败。

## V5.1 Stage B freeze-only proposal：资产/分母/operator/gate 已形成草案（2026-08-17）

- `WS-V51-M1-B-LUDVIG-UPLIFT-01` 仍为 `pending/locked`；新增
  `docs/WS_V51_STAGE_B_FREEZE_PROPOSAL.md` 与 `configs/worldsim_v51/stage_b_freeze_proposal_v1.yaml`，状态明确为
  `draft_freeze_only_not_authorized`。本轮没有 checkpoint 下载、方法实现、GPU/model run 或 feature/质量读取。
- official identity proposal：LUDVIG=`4461fc5`、Meta DINOv2=`7764ea0`、ViT-g/14 registers checkpoint URL/bytes=
  `4,546,140,349`/S3 version/ETag 已只读记录；ETag 是 multipart，不替代下载后的完整 SHA-256。
- 8 个开发 scene 沿用 H/S/C=`3/2/3`；每场固定 `5 evidence+5 evaluation frames × 3 front cameras=30` images，header
  audit=`240/240`、尺寸统一 `1600×900`。H patch population=`328,320<500,000`，因此无需上游随机 subsample；
  但 PCA solver/random state 与 CPU/GPU std 差异需显式冻结，登记 `V51-F14`。
- B0 proposal=per-view contribution aggregation + `1-exp(-mass)` view saturation；B1=同 support 上
  `sum(w*f)/(sum(w)+1e-8)` normalized transpose。DINO 与 renderer 分进程，40-D patch-grid sidecar + CPU chunked
  scatter，禁止 dense 900×1600×40 落盘和同卡并发。
- same-actor/actor-background 只用 frozen Rigid membership 作 evaluation proxy，不能进入 method，也不是真实 GT；
  same-Gaussian repeatability 与 heldout DINO reprojection 必须同表，登记 `V51-F15`。H/S/C gate 与 abstain denominator
  已在 proposal 预注册，但只有用户授权后才能转为 frozen。
- 当前下一门仍是用户裁决 `U2/B3 fallback`。若授权，第一步只做 P0 supersession + 240 image SHA freeze；
  validation/test/KITTI/Graph/backbone search 继续锁定。failure delta=`V51-F14/F15`，experiment delta=`none`。

## V5.1 Stage B 授权前预检：协议冲突与 DINOv2/24GB 门未闭合（2026-08-17）

- `WS-V51-M1-B-LUDVIG-UPLIFT-01=pending/locked`；本轮只读预检，没有下载/clone 上游、实现方法、启动 run 或读取
  C/validation/test/KITTI quality。预检合同见 `docs/WS_V51_STAGE_B_PREFLIGHT.md` 与
  `configs/worldsim_v51/stage_b_preflight_v1.yaml`。
- normative plan §10.8 允许 Stage A 全失败后以 U1/U2 进入 Stage B，但附录解锁条款要求 Stage A candidate 通过 S；
  r007 恰为 A1–A4 全 rejected、fallback=`U2/B3`。该 governance conflict 登记为 `V51-F11`，必须由用户独立选择
  fallback 授权或关闭 M1，执行者不得自行消解。
- official LUDVIG upstream 已只读冻结到 `4461fc515439bb498a75d71738a1e73cf7a452ed`；第一版应为 DINOv2
  ViT-g/14 registers、learning-free normalized renderer transpose、40-D PCA 参考点、immutable Gaussian + feature
  sidecar。当前服务器没有对应官方 checkpoint，登记 `V51-F12`。
- RTX 3090=`24576 MiB`；Stage A 单个 unary materialization 曾约 `20–22 GiB`。若获授权，执行序列必须拆成
  freeze-only→operator parity→离线 DINO feature sidecar→释放 DINO 进程→renderer uplift→one-shot separation gate；
  DINO 与 renderer 禁止同进程/同卡并发常驻。
- normative plan 已由 P0 config 按 SHA 冻结；preflight 回归发现 Stage A closeout `3d33262` 曾加入 5 行进展，造成
  inherited plan SHA drift 与 protocol regression=`2 failed / 1 passed`。本提交移除这 5 行、恢复原 SHA，并登记
  `V51-F13`；进展仍在 short plan/status/experiments。授权后的规范统一必须走显式 supersession/migration。
- 下一步只允许接收 Stage B 独立授权并做 freeze-only；checkpoint 下载、method implementation/run、Graph、backbone
  search、C/validation/test/KITTI quality 继续锁定。failure delta=`V51-F11/F12/F13`。

## V5.1 Stage A 正式收口：A1/A2 S rejected，冻结 U2/B3（2026-08-17）

- canonical r007=`/root/autodl-tmp/runs/worldsim_v51/WS-V51-M1-A-UNARY-OBSERVABILITY-01/20260817T140000Z__m1-stage-a-s-screening-s20260814-r007`，
  source=`dc24f28`，status=`done`，conclusion=`stage_a_screening_selected_u2_b3`，duration=`254.025 s`，peak GPU=`8393 MiB`。
- A1 S gate FAIL：0998/0359 `ΔBF1=-0.0000904944/+0.0000574359`；nonnegative=`1/2`、clearly positive=`0/2`、
  mean=`-0.0000165293`。mean IoU/Brier/ECE 虽分别为 `+0.000656494/-0.000159477/-0.000184098`，但预注册合取门
  不允许摘取局部正项，A1 rejected（`V51-F09`）。
- A2 selective gate FAIL：mean coverage=`0.557435<0.60`；accepted/abstained error=`0.0148416/0.134393`，错误集中
  成立但覆盖率不足，且 conditional 部分继承 A1 失败，A2 rejected（`V51-F10`）。A3/A4 原机制 rejection 保持。
- Stage A task=`done`，唯一 survivor=`U2/B3`；不得继续 Bayesian family。r007 checkpoint=`2/2 exact`，parameter search、
  C/validation/test/KITTI read/tuning 均为 false。closeout=`configs/worldsim_v51/stage_a_closeout_v1.yaml`。
- summary/status/manifest/fingerprint/diagnostics SHA=`094b4ae1.../03d319da.../7f60d097.../67f1001d.../15834f48...`；
  failure delta=`V51-F09/F10`。Stage B 仍按第一轮授权保持 pending/locked，需独立授权后才可冻结并启动。

## V5.1 Stage A S 输入已精确绑定，等待唯一 candidate screening run（2026-08-17）

- SAM r047/r048 与 B3/evidence materialization r049/r050 均已 `done`；0998/0359 的 evidence views=`15/15`、
  accepted evaluation views=`12/9`、abstained=`3/6`，两个 30k checkpoint 前后 SHA exact，未读 validation/heldout。
- canonical S binding=`configs/worldsim_v51/stage_a_screening_v1.yaml`；A1/A2 一次渲染筛选器=
  `scripts/run_worldsim_v51_stage_a_screening.py`，相关 gate/回归 `10 passed`。本条写入时尚未读取 A1/A2 的 S quality。
- PowerShell→SSH 后台 PID 转义使外层 wait 失效，但审计确认仅 r049 启动一次、r050 未启动；随后 r050 独立串行完成。
  wrapper 问题与单卡并发资源假设登记为 `V51-F08`，没有重复 run 或方法分母污染。
- 下一步只允许在 clean source commit 上执行一个 V5.1 screening run；候选/阈值/gate 不再改变，C、validation、test、
  KITTI tuning 继续锁定。

## V5.1 Stage A S screening freeze 完成，等待 one-shot SAM/evaluation（2026-08-17）

- retained candidates=`A1/A2`，S=`scene-0998/scene-0359`；A3/A4 rejected 结论保持。screening freeze=
  `configs/worldsim_v51/stage_a_screening_freeze_v1.yaml`，人类可读合同=`docs/WS_V51_STAGE_A_SCREENING_FREEZE.md`。
- gate 在 S quality read 前固定：`2/2` BF1 non-negative、至少 `1/2` BF1 delta>=`0.001`、mean BF1>0、
  mean FN delta<=`+0.02`、mean Brier/ECE delta 各<=`+0.005`；A2 另需 mean coverage>=`0.60` 与
  abstained error>accepted error。S 后最多保留一个 survivor。
- 两场 SAM configs 已绑定 V5 r030/r031 formal checkpoint/source/instances SHA，并逐项继承 H 的 split、prompt、SAM
  checkpoint 与 QC。计划 run 固定为 V5 task 下 r047/r048；执行前 S quality、heldout、C/validation/test/KITTI 均未读。
- freeze failure refs 到 `V51-F07`，delta=`none`。后续不得因 r047/r048 或 screening quality 改阈值、候选或重跑。

## V5.1 Stage A H 收口：A1/A2 保留，A3/A4 机制 rejected，下一步 freeze S screening（2026-08-17）

- A4 canonical r006=`/root/autodl-tmp/runs/worldsim_v51/WS-V51-M1-A-UNARY-OBSERVABILITY-01/20260817T122000Z__m1-a4-cif-identifiability-audit-s20260814-r006`，
  source=`cee8b66`，conclusion=`a4_cif_decoupling_rejected_no_independent_occupancy_observable`。
- 三场 A2 posterior 均无独立 occupancy field；现有 renderer 已分离 appearance base opacity 与 ownership sidecar，A1
  已分离 visibility eligibility，A2 已分离 UNKNOWN。constant occupancy=1 在三场都与现有 renderer bit-exact；把
  appearance opacity 再当 occupancy 则三场都 non-exact、等价于二次乘 alpha；用 visibility/count 又会混淆不存在与不可见。
- r006 只做绑定 source/evidence contract audit，未读 evaluation artifact/quality、未启动 GPU/training/search；A4 不形成
  quality arm，failure delta=`V51-F07`。summary/status/fingerprint/manifest/diagnostics SHA=
  `bb87357a.../a7eeef6c.../ae6953d2.../072ad60a.../ba43869b...`。
- Stage A H 决策：A1 visibility 与 A2 UNKNOWN 通过冻结 H gate；A3 Kish count 与 A4 CIF-lite 因机制不可识别/结构 no-op
  在 quality 前 rejected。最多保留两臂的约束正好落在 A1/A2；下一步必须先做 freeze-only commit，绑定 candidate source、
  threshold、S=`0998/0359`、一次性读取和停止条件，之后才允许读取 S quality。C/validation/test/KITTI 继续锁定。

## V5.1 Stage A A3 effective-count 机制预审 rejected，下一步只实现 A4 CIF 解耦（2026-08-17）

- A3 r004/r005 都只读取 H/A1 evidence observations 与绑定的 A2 posterior，不读取 evaluation artifact/quality，
  不启动 GPU renderer。r004 因近零分母的相对变化病理收口为 `done/inconclusive`；`V51-F06` 记录并由新 v2/r005
  以 absolute cap change>`1e-9` 修正，旧 terminal 不覆盖。
- canonical r005=`/root/autodl-tmp/runs/worldsim_v51/WS-V51-M1-A-UNARY-OBSERVABILITY-01/20260817T120000Z__m1-a3-effective-count-audit-v2-s20260814-r005`，
  source=`150b072`，conclusion=`a3_kish_cap_rejected_structural_noop_not_correlation_aware`。
- 45 份 evidence observation 对 A2 parent effective-count `3/3 float32 exact`；pooled positive-count Gaussian=
  `944,443`。无 epsilon 时 `n_eff<sum(r)` 数量=`0`，absolute cap change>`1e-9` 数量=`0`，证明 proposed cap
  是结构 no-op；直接用 `n_eff` 替换则放大 `940,762/944,443=99.6102%` Gaussian concentration。
- 该公式只有 `sum(r)` 与 `sum(r^2)`，没有 view-pair correlation observable，不能支持“相关视图不应按独立证据累计”
  的命题。A3 因机制不成立直接 rejected，不生成无归因质量数字，也不修改 A2/UNKNOWN 或 A1 visibility。
- r005 duration=`4.219 s`；summary/status/fingerprint/manifest/diagnostics SHA=
  `9e599c5f.../1cde89f2.../7e6c728b.../2f3d7f01.../a147843a...`；failure delta=`V51-F05/F06`。
  下一步只解锁 A4 visibility/occupancy/conditional-identity probabilistic decoupling；S 与 Stage B/C 继续锁定。

## V5.1 Stage A A2 UNKNOWN 通过 H gate，下一步只实现 A3 effective count（2026-08-17）

- `WS-V51-M1-A-UNARY-OBSERVABILITY-01=running`；A2 canonical r003=`done`，conclusion=
  `a2_unknown_passed_h_gate_candidate_for_stage_a`，source=`7e783f1`，run=
  `/root/autodl-tmp/runs/worldsim_v51/WS-V51-M1-A-UNARY-OBSERVABILITY-01/20260817T113000Z__m1-a2-unknown-h-s20260814-r003`。
- A2 thresholds 在 quality read 前由 `944,443` 个 pooled H/A1 positive-count Gaussian 冻结；规则为
  `high entropy AND (low effective count OR high cross-view disagreement)`，图像 abstain threshold=`0.5`。
  naive 全量 Gaussian inclusive quantile 会退化为 0 coverage，已登记 `V51-F04`，本轮未据 quality 调参。
- A1 conditional semantics 保持不变：12/12 当前 GPU renders 与 r002 A1 NPZ byte exact，7 项 conditional metric
  delta=`0`，三个 checkpoint 前后 SHA exact。因此 A2 相对 B3 的 BF1/IoU/FN/Brier/ECE 仍为 A1 的
  `+0.001155713/+0.000460310/+0.001105687/-0.000013972/-0.000144854`，UNKNOWN 的新增价值只由 selective
  endpoints 裁决，不能把 unchanged conditional quality 写成 A2 的额外质量增益。
- 0471/1087/0379 coverage=`46.4865%/73.6836%/95.8186%`，UNKNOWN recall-on-errors=
  `86.8498%/95.9160%/87.5421%`；scene-balanced coverage=`71.99625%`，accepted/abstained absolute error=
  `0.0148914/0.164250`，error separation=`+0.149358`，全部预注册 selective checks 通过。
- 重要边界：0471 单场 coverage 只有 `46.49%`，低于 60%；当前冻结 gate 要求的是 scene-balanced mean>=60%，
  所以 r003 合法 PASS 但不能声明逐场稳定 coverage。1087 仍只有 1 个 evaluation view；A2 只登记 H candidate，
  S/C/validation/test/KITTI 继续不可读。
- r003 duration=`192.314 s`，peak GPU=`8393 MiB`；summary/status/fingerprint/manifest/diagnostics SHA=
  `c9a82139.../cb67b786.../2c5f0600.../56bf0207.../206faa66...`；`failure_ledger_refs` 到 `V51-F04`，
  `failure_ledger_delta=none`。下一门只实现 A3 correlation-aware effective count；A4、S 与 Stage B/C 仍锁定。

## V5.1 Stage A A1 visibility 通过 H gate，下一步只实现 A2 UNKNOWN（2026-08-17）

- `WS-V51-M1-A-UNARY-OBSERVABILITY-01=running`；A1 canonical r002=`done`，conclusion=
  `a1_visibility_passed_h_gate_candidate_for_stage_a`，source=`38bc9b4`，run=
  `/root/autodl-tmp/runs/worldsim_v51/WS-V51-M1-A-UNARY-OBSERVABILITY-01/20260817T104000Z__m1-a1-visibility-h-s20260814-r002`。
- `visibility >= float32(0.01)` 只由 H evidence statistics 冻结，未读 evaluation quality；semantic-available observation
  的保留率在 0471/1087/0379 为 `85.7484%/96.7938%/82.6925%`，低 visibility 与 semantic-unavailable observation
  均执行 `Δalpha=Δbeta=0`。
- B3 当前 GPU 重渲染在 `8+1+3=12` 个 accepted evaluation views 上与 V5 canonical NPZ `12/12 byte exact`，
  7 项 aggregate metric delta 全为 0；三个 base checkpoint 前后 SHA exact。r002 耗时=`193.228 s`，峰值 GPU=
  `8382 MiB`。
- A1−B3 scene-balanced mean `ΔBoundary-F1/ΔIoU/ΔFN-mass/ΔBrier/ΔECE`=
  `+0.001155713/+0.000460310/+0.001105687/-0.000013972/-0.000144854`；BF1 正向=`2/3`，
  0471 仅 `+0.000020836`、1087=`0`、0379=`+0.003446302`。H gate 逐项通过，但效应很小且 1087 只有 1 个
  evaluation view，因此只登记 Stage-A candidate，不宣称 scene-disjoint 稳定或方法成功，也不提前进入 S。
- r002 summary/status/fingerprint/manifest SHA=`74246312.../c4dbac0d.../3cca740c.../fbec5f19...`；
  `failure_ledger_refs=V5-F20–F26/F29–F32 + V51-F01–F03`，`failure_ledger_delta=none`。
- 下一步只实现 A2 UNKNOWN/ABSTAIN，并以 A1 为 matched parent；A3/A4、S scenes、Stage B 和任何 Graph 仍锁定。

## V5.1 Stage A A0 exact replay 完成，下一门只解锁 A1 visibility（2026-08-17）

- `WS-V51-M1-A-UNARY-OBSERVABILITY-01=running`，A0 canonical r001=`done`，conclusion=
  `a0_v5_b0_b1_b3_posterior_and_gaussian_metrics_bit_exact`；source=`1e23616`，run=
  `/root/autodl-tmp/runs/worldsim_v51/WS-V51-M1-A-UNARY-OBSERVABILITY-01/20260817T102000Z__m1-a0-v5-unary-replay-s20260814-r001`。
- r037/r042/r043 的 `45` 份 evidence observation 被当前代码实际重算；`3 scenes × 3 arms × 6 fields=54`
  组 posterior/statistics 数组均为 `0` bit mismatch，9 个 arm×scene 的 Brier/ECE/IoU/NLL/FP/FN Gaussian
  metric delta 全为 `0.0`。
- 每个 canonical run 的 12 个核心生成源码与当前分支逐 SHA exact，159-file manifest inventory、checkpoint 与
  evaluation artifact 继续 exact。边界：本轮没有重新执行 GPU renderer；2D evidence 是 canonical bytes + generation
  source exact，不把它写成一次新的 2D render quality read。
- r001 summary/status/fingerprint/manifest SHA=`b9b33bbd.../5d695add.../6c7466b3.../c4a5e4fd...`；
  `failure_ledger_refs=V5-F20–F26/F29–F31 + V51-F01/F02`，`failure_ledger_delta=none`。
- A0 通过后只解锁 A1 `visibility / semantic decoupling`：不可见 observation 对 Beta posterior 的正负更新都必须为
  0。A2 UNKNOWN、A3 effective count、A4 CIF 与任何 Graph/uplift 仍未授权，必须逐机制 matched ablation。

## V5.1 M1-only P0/D0 冻结完成，Stage A A0 进入 exact replay（2026-08-17）

- `WS-V51-P0-M1-SCOPE-FREEZE-01=done`、`WS-V51-D0-DEV-ROLE-FREEZE-01=done`；实现提交=`58953a5`，
  canonical start audit=`/root/autodl-tmp/runs/worldsim_v51/WS-V51-P0-M1-SCOPE-FREEZE-01/20260817T101000Z__p0-start-audit-s0-r001`，
  conclusion=`v51_m1_scope_roles_and_v5_inputs_frozen`。
- V5.1 branch=`research/worldsim-v5.1-m1`，immutable V5 parent=`44d0e4a`；normative plan SHA=`3d7f7481...`，
  原 V5 cohort file/cohort SHA=`6d8caf19.../55337315...`。当前只授权 P0、D0 与 Stage A，M2/M3 保持 `pending`，
  Stage B 及 Graph/anchor/identity/BKI 等后续路线继续锁定。
- Development 角色按原 cohort 顺序 exact 冻结：H=`0471/1087/0379`，S=`0998/0359`，
  C=`0875/0535/0436`；8-scene validation 与 20-scene test quality 均未读，KITTI method tuning=false。
- r001 对 r037/r042/r043 的 `65+44+50=159` 个 manifest 文件、合计 `680,254,598 bytes` 逐 bytes/SHA 复核，
  三个 checkpoint、summary、status、fingerprint、manifest、diagnostics 与 resolved config 均 exact；r001
  summary/status/fingerprint/manifest SHA=`6d495ce2.../8a724b06.../b52b63d3.../8ab0ad66...`。
- `WS-V51-M1-A-UNARY-OBSERVABILITY-01=running`，当前唯一下一门是 A0：用冻结 observation 重算
  B0/B1/B3 posterior/统计并对 canonical table 做 bit-exact replay，再决定是否启动 A1 visibility mask。
- 首轮 pytest collection 因仓库根未注入 `sys.path` 被阻塞，修复后 `4 passed`；该工程教训登记为
  `V51-F01=resolved`，没有方法推理或质量读取，不能计入算法分母。

## V1–V5 统一 failure ledger 治理：done（2026-08-17）

- `DOC-FAILURE-LEDGER-01=done`，治理提交=`4e512d9`；`docs/RESEARCH_FAILURES.md` 现为唯一活跃失败事实源，
  archive 同名文档保持不可变快照，专项 `*_FAILURE_FORENSICS.md` 只保留为证据报告。
- 顶部新增渐进式读取/写入合同、最小条目 schema、版本总览与显式目录；V1/V2 的算法推翻、工程阻塞、资源和
  评测边界分别收敛为 `V1-F01`–`V1-F06` 与 `V2-F01`–`V2-F09`，V3–V5 继续链接现有详细账本。
- 旧 V4 追加段曾重复使用 `V4-F30`–`V4-F33`；live canonical 已校正为连续唯一的 `V4-F01`–`V4-F49`，
  并保留 historical→live 映射。旧 commit、run 和 archive 的历史编号不回写。
- 后续正式实验必须在启动前登记 `failure_ledger_refs`，收口时登记 `failure_ledger_delta`；出现 blocked/rejected、
  假设推翻、分母错误、工程/资源/协议失败或风险解除时，与 status/experiments 在同一逻辑提交中更新统一账本。
- 本次只做研究治理与历史整理，没有训练、推理、数据/quality read、split/seed/fingerprint 变化，也没有改写
  V1–V5 的科学终态或当前执行授权。定义条目审计=`207` 且重复 ID=`0`，V4 连续性=`49/49`，`docs` backup=`0`。

## V4 文档归档复核与临时产物清理：done（2026-08-17）

- `WS-V4-DOC-CLEANUP-02=done`，清理提交=`3598ef7a`；V4 终局包清理前后均通过 `78/78` SHA-256 校验，
  `M1 rejected / M2 done with geometry caveat / M3 confirmed` 及 `none_v4_closed` 均未改写。
- 已清空 `/root/autodl-tmp/motion_proj/tmp`（4 files / 8,985,579 bytes）和 `/root/autodl-tmp/tmp`
  （58 files / 402,855 bytes），两目录保留为空；已删除无仓库引用的 `/root/autodl-tmp/mnt`
  （154 files / 126,234,111 bytes）。
- `docs/` 内 6 个编辑恢复目录、233 个目录内副本及 50 个散落副本已删除；最终文件名/目录扫描为 `0`。
  后续 `docs/` 只保留 canonical 快照、实验凭证、清理清单与完整性 manifest，恢复依赖 Git 历史。
- `/root/autodl-tmp/motion_proj/work/codex-backups/` 因 V4 失败账本明确引用 partial scene 资产而保留；canonical
  run、冻结配置/源码/测试、KITTI、nuScenes、checkpoint 和环境均未清理。
- 完整绝对路径、清理前规模、`mnt` 内容摘要、不可恢复边界和保留项见
  `docs/archive/2026-08/worldsim-v4-cleanup-2026-08-17/CLEANUP_MANIFEST.md`。

## V5 M3 r001–r006 正式收口：rejected（2026-08-14）

- `WS-V5-M3-CONSTRAINT-PROJECTED-TEMPORAL-01=rejected`，r006 conclusion=`m3_rejected_constraint_projection_not_needed_on_frozen_requests`。r001 已冻结 T2–T5、物理 caps、REMOVE 隔离和 quality locks；V4 r238/r335 的 baseline=`FRAME_INDEPENDENT`，不得当作 V5 T2 comparator statistics。
- r003 使用 annotation metadata-only 在 8/8 fresh development scenes 冻结各一个 `7 keyframes / 3.0s` vehicle clip，0 abstain；图片、LiDAR blob、reconstruction/edit quality 均未读取。r002 因 YAML 缺 conclusion 在 streaming 前 blocked，terminal 保留。
- r004 初始 T2/T5 violation=`38/34`，但 T2 的 38 项全部来自 heading-velocity measurement；T5 把它们换成 `20 yaw-rate + 14 heading`，并对 2 个 T2-safe 请求回归。该结果为 insufficient，不解锁 renderer。
- r005 进行 result-aware measurement correction：speed<=`1m/s` 时 heading unobservable、允许 reverse、只有 zero residual violations 才称 convergence。exact replay 后 T2 safe=`15/16`、evaluable=`1/16`、T2/T5 violations=`2/1`；虽 reduction=`50%`，但远低于预注册 minimum evaluable=`8`，正式结论仍为 `m3_constraint_projection_insufficient_t2_violation_signal`。
- collision/render gates 未评估，method arm 未选择，validation/test/KITTI 继续锁定。完整归档见 `docs/WS_V5_M3_DEVELOPMENT.md` 与 `docs/archive/2026-08/worldsim-v5-m3/M3_R001_R006_METADATA.json`。
- r006 绑定 `4 completed + 1 blocked`，禁止 post-hoc stress-template search，并只把 V4 temporal positive result 保留为 historical baseline。summary/decision ledger SHA=`9dfc72a.../b904d0e5...`。

## V5 M2 正式拒绝收口，下一独立任务转入 M3 协议冻结（2026-08-14）

- `WS-V5-M2-GEOMETRY-FIRST-REPAIR-01=rejected`。r015 汇总 `8` 个 completed 与 `4` 个 blocked terminal，正式结论=`m2_rejected_no_absolute_geometry_safe_candidate`；method arm、router refit、validation unlock、parameter search 和 neural surface automatic unlock 均为 false。
- r013 G4 同相机跨时 scaffold 的 raw/post 改善数=`12/22`、`17/22`，raw absolute-safe=`0/22`，相对 raw 门失败；r014 G5 三相机 scaffold 的 raw/post 改善数=`15/22`、`19/22`，相对门成立，但 raw/post absolute-safe 只有 `1/22`、`0/22`，所以不能选择方法或进入 router/validation。
- G5 覆盖诊断为 any projection/direct/extrapolation/fallback mean=`60.40%/15.57%/47.99%/36.44%`，LiDAR 投影覆盖均值约 `0.8%`。相对改善是真实 model-proxy 证据，但不是 same-view hidden-background ground truth；禁止事后改 threshold、source grid、fusion 或外推半径。
- canonical closeout r015=`/root/autodl-tmp/runs/worldsim_v5/WS-V5-M2-GEOMETRY-FIRST-REPAIR-01/20260814T221000Z__m2-geometry-first-closeout-s0-r015`；summary SHA=`27a6613d4428fbadaf45a7e9f606bdb1819a9a5b712bb7f03171af0fe70a0c01`，decision ledger SHA=`63b9e36bfbf734017c625910af24e226eb174e79fd0da0a33ac1b8cc6c21d715`，manifest `8/8 exact`。
- `WS-V5-M2-GEOMETRY-FEASIBLE-ROUTER-01` 保持 `pending/locked`，不是 rejected 方法的补救入口。下一独立任务仅为 `WS-V5-M3-CONSTRAINT-PROJECTED-TEMPORAL-01` 的 result-blind 协议/证据审查；fresh validation/test 与 KITTI 方法调参仍未授权。
- 完整人类归档与机器元数据见 `docs/WS_V5_M2_GEOMETRY_FIRST_DEVELOPMENT.md` 和 `docs/archive/2026-08/worldsim-v5-m2/M2_R012_R015_CROSS_VIEW_CLOSEOUT_METADATA.json`。

## V5 M2 Gaussianization density 机制中间快照（已由 r015 收口取代，2026-08-14）

- 当时阶段为 `gaussianization_density_mechanism_supported_representation_repair_next`；该中间快照已由本文件首节的 r015 rejection 取代。全程只使用 frozen development `scene0471`；validation/test/KITTI quality、parameter search、method-arm selection 与 router refit 均为 false。
- r004 把 r036 的 actor-union mask 恢复为正式修复语义：`23 actor-view requests = 22 accepted + 1 rejected`，逐 actor union 与原 mask 逐像素 exact；r002/r003 的 union-mask 结果只保留为 staged/instrumentation 负证据，不参与最终 arm 选择。
- r005 的 G0 在 `22/22` 请求上 raw absolute gate fail，raw MAE mean/median=`8.5872/8.7151m`；Gaussianization primary=`16/22`，post MAE mean/median=`9.0056/9.0040m`。reference confidence mean/median=`0.0585/0.0582`，故全部数值仅是 base-background model proxy，不是独立 GT。
- 结果前冻结的 broad-support gate 要求 `>=14/22` 请求改善至少 `0.5m`，且 mean/median candidate−G0 都小于 0。G1/G2/G3 分别只有 `5/8/11` 请求改善；mean delta=`+3.658565/+3.005506/+0.103693m`，因此全部 `rejected`。G3 虽 median delta=`-1.489037m`，仍不得事后按中位数选臂。
- r011 的四因子臂对 r005 BASE 完成 `22/22 exact` replay，四臂共同评估 pixel loss=`0`。OPAQUE−BASE 改善=`0/22`、mean/median delta=`+0.059686/+0.065773m`；DENSE−BASE 改善=`20/22`、mean/median=`-0.424179/-0.480927m`；DENSE_OPAQUE−DENSE=`0/22`、mean=`+0.035533m`。因此 density 机制受支持，opacity/alpha mixing 不受支持；没有方法臂被选择。
- r001、r007、r010 合法保留为 `blocked`：依次是 unavailable-view denominator、serializer 变量遮蔽和 launcher 预建目录触发 overwrite guard。三者均不是方法质量失败，均以新 run ID 修复，不覆盖 terminal；r010 在模型加载前停止且没有 GPU/质量读数。
- 当前仍没有 geometry-safe candidate，`WS-V5-M2-GEOMETRY-FEASIBLE-ROUTER-01` 继续锁定。下一步只允许结果盲冻结一个 density representation repair，并重新通过逐 actor development gate；禁止进入 router、validation、神经 surface 或 KITTI 调参。
- 人类可读归档=`docs/WS_V5_M2_GEOMETRY_FIRST_DEVELOPMENT.md`；附录/机器元数据=`docs/archive/2026-08/worldsim-v5-m2/APPENDIX_INDEX.md`、`M2_R001_R009_METADATA.json` 与 `M2_R010_R011_GAUSSIANIZATION_METADATA.json`。

## V5 M1/M1B development 收口：rejected（2026-08-14）

- `WS-V5-M1-STRUCTURED-OWNERSHIP-01=rejected`：三场景 graph replication 只有 `3/6` Boundary F1 单元正向，未达 `>=4/6`；validation、formal arm selection 均未解锁。
- `WS-V5-M1B-D0-BOUNDARY-RESIDUAL-FORENSICS-01=done`：canonical r001 只读 r038/r045/r046 的 `12` 个 evaluation NPZ，boundary-primary=`0/6`，mean boundary classification/semantic-error share=`0.4020948014/0.2483529331`。summary SHA=`ddecf415bd71fcf920b6fab5f38ee74edea93aa085b48a73947480c3c186c35d`。
- 结论=`boundary_ambiguity_not_primary_semantic_split_remains_locked`；`WS-V5-M1B-REVERSIBLE-SEMANTIC-SPLIT-01` 的条件未成立，任务不启动。Transformer、split、M1 graph 调参与 validation quality 继续禁止。
- V5 当前转入 M2 geometry-feasible repair routing。后续 M2/M3 的任何成功都不得倒写 M1；M1 负结论与完整哈希见 `docs/WS_V5_M1_DEVELOPMENT_REPLICATION.md` 和 `docs/WS_V5_M1B_BOUNDARY_RESIDUAL_FORENSICS.md`。

## V5 M1 三场景 development replication 失败并进入 boundary-residual forensic（2026-08-14）

- `WS-V5-M1-STRUCTURED-OWNERSHIP-01=running`，当前阶段=`development_replication_rejected_boundary_residual_forensics_pending`。前三个 frozen development scenes=`scene0471/1087/0379` 按结果盲协议完成 SAM→unary→G0–G3 graph；选择时未读 quality，validation/test/KITTI quality 与参数搜索均未发生。
- 稀疏 SAM 分母为 `18/30、2/30、6/30` available；unary evaluation 为 `8+7、1+14、3+12` accepted+abstain。scene1087/0379 的 canonical unary 为 r042/r043，checkpoint 前后 SHA exact；scene0471 延用 r037。
- graph canonical 为 r038/r045/r046，G0 均逐像素 exact replay 所绑定 unary，base checkpoint 前后 exact。G3 对同 unary G0 的 Boundary F1 正向只有 `3/6`（要求 `>=4/6`）；mean ΔBoundary-F1=`+0.0016107723`、mean ΔFN-mass=`+0.0025676789`，但 topology `G3<G1` 仅 `2/3` 场景。
- 裁决=`physical_graph_development_replication_rejected_3of6_boundary_support`；formal arm selection=false，validation unlock=false，Transformer/semantic split=false。下一步只允许复用现有 development artifacts 做 boundary ambiguity residual forensic，先证明 boundary 是主要残差，才可能条件启动 M1B；不得直接实现 split 或继续调 graph。
- r041 的 SSH `BrokenPipeError` 与 r044 的硬编码 `8+7` 分母分别作为基础设施/合同失败保留；r042 与 r045 使用新编号完成。附录入口=`docs/WS_V5_M1_DEVELOPMENT_REPLICATION.md`。

## V5 M1 30k 正式基线与 scene0471 unary/graph 诊断闭环（2026-08-14）

- `WS-V5-M1-STRUCTURED-OWNERSHIP-01=running`，当前阶段=`structured_graph_mechanism_supported_development_replication_pending`。8-scene 30k reconstruction 已全部完成并由 r035 逐 run、逐 checkpoint 重哈希：`8/8 scenes × 30,000 steps`，总耗时=`18,392.900160 s`、checkpoint bytes=`2,920,094,512`，峰值 GPU/cgroup=`24,054 MiB / 27,706,277,888 bytes`；r035 summary SHA=`4a540d24cd8bfa18c9d63cdcbabe08dcded7a2de88de116695e431187cb6738b`。
- scene0471 冻结 SAM 证据 r036=`done`：`30` 个预注册视图中 `18` 个 available/accepted，`17` 个 prompt actors、`62` 个 prompt boxes、`61` 个 accepted boxes；network/held-out/method inference=`false/false/false`，summary SHA=`d66f04a05a5a0ee8fb94b423e20296ec019bc8fe1e56ebc6c57fb1c80495d487`。
- structured unary r037=`done`：不可变 checkpoint SHA 前后均为 `496356ca2d8f31b4e8593b294eebba1f068a0c7fddbd86ab1717c377a1793cfa`；`859,613` Gaussians、`15` evidence views、`8 accepted + 7 abstained` evaluation views；耗时=`555.199743 s`、峰值 GPU=`13,987 MiB`。summary/diagnostics SHA=`dd8b2a9e5f09f130f948c9de2b6b8eaa5bea9ab714278bed7fa56a633dd7a22d / 88e256b9f07149cdfbf94da26e7d59b83c2071cb4485e41cf80717f0eac0d755`。
- 相对冻结 B0，B1 的 2D `ΔIoU/ΔBoundary-F1/ΔBrier/ΔECE/ΔNLL/ΔFP-mass/ΔFN-mass`=`+0.116665/+0.107437/-0.101739/-0.110983/-0.214462/-0.142929/+0.091532`；B3=`+0.116161/+0.105079/-0.103467/-0.112431/-0.305104/-0.138453/+0.095477`。即 reliability-aware unary 有明确单场方向支持，但以显著 FN 增加换取 FP/calibration/boundary 改善。
- physical graph r038=`done`：`859,613` Gaussians、directed KNN `k=6`、`5,157,678` edges，G0 对 r037 的 B1/B3 float16 render 均逐像素 exact replay；checkpoint SHA 前后 exact。G3 相对同 unary G0 的 2D `ΔBoundary-F1/ΔIoU/ΔFN-mass` 为 B1=`+0.008585/+0.006530/+0.001491`、B3=`+0.006245/+0.006256/+0.001061`；cross-proxy affinity ratio 从 G1=`0.0083646` 降至 G2/G3=`0.0051204/0.0040198`。summary/diagnostics SHA=`c64e52a9de2a43cbb89564cbf61610746fdec210eb2a4f0efb32bc6463f7faf1 / ab81462375a2ea7faec73051aec807808f41426e872f954a249fdee19bfb9d2b`。
- 裁决=`physical_graph_direction_supported_single_scene_small_effect_replication_required`：G3 仅登记为 mechanism preference，不是 formal arm selection；不声明 validation 成功，不解锁 semantic split。下一门是 result-blind development replication protocol；validation/test/KITTI quality 与参数搜索仍未授权。附录入口=`docs/WS_V5_M1_FORMAL_BASE_UNARY_DIAGNOSTIC.md` 与 `docs/archive/2026-08/worldsim-v5-m1/APPENDIX_INDEX.md`。

## V5 KITTI 真实 adapter smoke 闭环（2026-08-14）

- `WS-V5-D1-KITTI-ADAPTER-01=done`：选择性抽取 0000/0001=`1805 files / 2,104,258,586 bytes`，raw manifest file/content SHA=`a3c77ab82bb29d2b34f615743a4e0393d611fb634826900c3756193127450928 / 98685653d4488b53a6ab94890629347b11cc43793b78b05c6b05738fcf46d83f`；原 ZIP 保留。
- adapter canonical r003 summary SHA=`3b27cb9fa9b06f563b690cc44b1466e622b578bc88294b450ed254e8192a970b`。0000 multimodal=`154/154`；0001=`443/447`，frames `177–180` 缺 LiDAR 并显式 abstain。mixed calibration、30-field OXTS、pose chain、PNG、LiDAR `N×4`/双目投影、label/track gates 全通过。
- metadata 入口=`docs/KITTI_TRACKING_ADAPTER_SMOKE_V5.md` 与 `.json`。本结论 supersede 早期 `adapter=blocked` 工程状态，但不改写 archive audit 当时事实；method quality/training/inference/search/cross-domain authorization=`false/false/false/false/false`。

## V5 M1 development 数据闭环与 base reconstruction 入口（历史执行快照；由上节取代）

- 8-scene `profile100` gate 已完成：r019–r026 全部 `done`，每场 checkpoint step=`100` 且 Gaussian means finite；总训练耗时=`463.532647 s`、checkpoint bytes=`2,592,731,152`，峰值 GPU/cgroup=`9142 MiB / 24,595,931,136 bytes`，8 份 checkpoint 全量 SHA 复核一致。source commit=`200ece4ebe59031b5546f285d2251482446ab162`，validation/test quality read=`false/false`。
- 30k 只能使用 `configs/worldsim_v5/m1_development_reconstruction_formal_v1.yaml`：它除 sky gate 外还绑定 8 份 profile summary/status/fingerprint/run-manifest/checkpoint identity；runner 对 formal 强制全量复核该 gate，缺失时 fail-closed。当前仍无 reconstruction quality 或 arm selection。
- 派生 sky-mask gate 已完成：8/8 scenes、`4704/4704` PNG、`14,058,820 bytes`，逐文件 bytes/SHA 全量复核一致；总推理耗时=`1067.213706 s`，加权 mean sky fraction=`0.0655167343`，峰值 GPU=`4168 MiB`。8 个 run 均来自 clean commit=`282ff528af5ca8455014271a3a492e8b9c344991`，network/test quality/method inference=`false/false/false`。
- 训练输入改用不可变 base 配置的 sky-bound overlay：`configs/worldsim_v5/m1_development_reconstruction_skybound_v1.yaml` 绑定原配置 SHA=`c55f39a089da1beaf8ba00a5eb9dda3c26f997486a8950e2002256e4f0dbc748`、8 份 summary/run-manifest/sky-manifest SHA；原配置不改写。当前下一门为 8-scene `profile100`，通过前不启动 30k formal 或 structured unary quality。

- `WS-V5-M1-STRUCTURED-OWNERSHIP-01=running`；精确 raw selective extraction 与 8-scene DriveStudio preprocess 均已完成，当前阶段=`development_base_reconstruction`。graph 继续 `disabled_until_unary_diagnostic`，尚无 development quality、arm selection 或参数搜索。
- raw formal run=`/root/autodl-tmp/runs/worldsim_v5/WS-V5-M1-STRUCTURED-OWNERSHIP-01/20260814T095000Z__m1-development-raw-extract-s0-r001`：六路相机 + `LIDAR_TOP` 完整 keyframe/sweep 时间链=`14,220/14,220 files`、`3,996,996,012 bytes`；batch manifest content/file SHA=`65fc5363ca13f9124fe6165a84a7857339943d64b87a756127950ab19c4611b6 / d244638a88943c3757a0fb00494766cf5f7ce30ba1bcd5d9e6c6bf18f0d06fd3`，summary SHA=`0c164e46873ecca4e2878d2d9937960d9b1df916ee25e92d075abc9d1ea0c213`。
- preprocess canonical run=`/root/autodl-tmp/runs/worldsim_v5/WS-V5-M1-STRUCTURED-OWNERSHIP-01/20260814T105300Z__m1-development-preprocess8-s0-r002`，source commit=`23e13e2201f757fc02eaf5ef0677219e564d85da`，`8/8 scenes`、`2,497,238,886 bytes`、耗时=`1148.674389 s`，summary/status/fingerprint/manifest SHA=`dcdd3450328669c26eed0316e2088e1f501fad965ed10ad8d344c37fda36f9c0 / 21702f7442824a2e7fd5e66b120511fc3c28121f68d8c711c521c7e858eacfa7 / 147c6d4d4024e12ea26b1e9f0cf7ebb9723d334aa83e1ec199a86c7529eb7a2c / c7140b4001bcfb108e83b5569a5d25a1176f61c66a3dcb1c5d17c3ddfa10f391`。
- timeline 分母按真实 scene metadata 保留：`scene-0379=191 frames`、`scene-0535=201 frames`、其余六场景=`196 frames`；对应最终文件数=`6120 / 6440 / 6280×6`。这不是截断或缺帧，训练合同不得强行改写成统一 196。
- 全过程 sensor payload 仅为抽取、预处理与 SHA/结构审计读取；`quality/training/model inference/validation/test read=false`。首个 StreetGS profile r003 在任何训练 step 前因派生 `sky_masks` 缺失合法 `blocked`，summary SHA=`a2802430984ab369143be609088df514e3ed0943563b23ee0a5b3bee02e214f7`；preprocess 8/8 结论不变。
- 当前执行结果盲 `development_derived_sky_masks`：只允许用本地 frozen SegFormer revision `2c6f153...`、三训练相机、offline/atomic 协议生成 `4704` masks，并绑定逐文件 SHA。sky-mask gate 完成后以新 run ID 重启 8-scene `profile100`；base checkpoint 就绪后才运行 structured unary diagnostic。

## V5 P0 forensic 正式证据注册（2026-08-14）

- freeze-only commit=`dfe7526c7a83ca12d7fa9f6c5a11a29ea7b27b19`，冻结文件仅为 P0/M1-D0/M2-D0 config、forensic runner、文档与测试；resolved plan SHA=`cc5f697357b9cc3a4051862563cd124e9fc3cc3a877096ab9f76e318e5e2f9b3`。
- `WS-V5-M1-D0-BAYES-FORENSICS-01=done`：canonical run=`/root/autodl-tmp/runs/worldsim_v5/WS-V5-M1-D0-BAYES-FORENSICS-01/20260814T090500Z__m1-d0-bayes-forensics-s0-r001`，summary SHA=`55006fff260d1bdacb8781492abc3b9f9c6f8bcb5351d2644ad9311c7034d82f`，conclusion=`blocked_evidence_missing_contract_frozen`。
- M1 runner 已重新校验 r200 三个 terminal JSON 与四份 state NPZ SHA，并逐份重算 O1-proxy recall、posterior extreme、low-uncertainty、unobserved 与 mixed-observation 分母；现有 state 仍缺 per-view observation、projected boundary、center/covariance 与 topology 字段。D0 完成只表示诊断和下一轮采集合同冻结，不表示缺失证据已恢复，也不改写 V4 M1=`rejected`。
- `WS-V5-M2-D0-GEOMETRY-FORENSICS-01=done`：canonical run=`/root/autodl-tmp/runs/worldsim_v5/WS-V5-M2-D0-GEOMETRY-FORENSICS-01/20260814T090600Z__m2-d0-geometry-forensics-s0-r001`，summary SHA=`33708f5165c04fb22a79bc985da36caf1b907fef8d038ac789e31b1debc5e0c0`，conclusion=`risk_saturation_and_blocked_evidence_missing_contract_frozen`。
- M2 runner 从 6 个 scene summary 重建 `154 requests / 214 candidates`，复现 saturation=`192/214`、same-risk/different-MAE collision=`57/130`、accepted geometry oracle=`62/83`、positive regret=`21/83`，并复现 accepted/risk-abstain/role-asset-blocked 的 full-denominator scene-balanced delta=`+3.390809623732304 m`。
- 两个 formal run 均来自同一 clean freeze commit，包含 resolved config、fingerprint、JSONL events、manifest、status、summary、source snapshot 与 run-local audit artifact；fresh/test quality、training、parameter/router search 均为 false。
- `WS-V5-P0-SCOPE-FREEZE-01=done`：formal closeout=`/root/autodl-tmp/runs/worldsim_v5/WS-V5-P0-SCOPE-FREEZE-01/20260814T091100Z__p0-scope-closeout-s0-r001`，source commit=`28c1d607de0a0ba72895806184348db4d3216de0`，summary SHA=`ca4248cff7085d8d5a57c842827b1a549b6d1d82fa95c2f81a2976c1192f5d38`，conclusion=`p0_scope_and_forensic_contracts_closed`，run 内合同测试通过。
- `WS-V5-D0-NUSCENES-FRESH-COHORT-01=done`：diagnostic r001 与 frozen replay r002 均由 metadata-only runner 生成；formal run=`/root/autodl-tmp/runs/worldsim_v5/WS-V5-D0-NUSCENES-FRESH-COHORT-01/20260814T100000Z__fresh-cohort-freeze-s2216484596-r002`，source commit=`8821bd9ad8c3f99b3b39829385728dc37533bb93`，summary SHA=`0ea5ff1f5fd16fc278269acbd11e9998c8e3e67d74245a55bdf89a5d09896aad`。
- fresh cohort=`8 development + 8 validation + 20 test`，cohort SHA=`553373159023218b44615be27aeeb5533a6c585be276e06425235fe09b6b48b1`，metadata inventory SHA=`63d0a70646615a5bc074faacee9838a8c7c4729a6e091a143435588ba53829f9`；850-scene metadata pool、V4 30-scene exclusion、role disjoint、official split 与 deterministic replay 全通过。
- 当前阶段=`post_fresh_cohort_development_only`：只开放冻结 development scenes 上的 M1 structured ownership、M2 geometry-first 与 evidence/reference instrumentation。validation/test quality 尚未读取；M1B semantic split、完整 M3、新模型大训练、validation 参数搜索、KITTI 方法调参仍禁止。
- `WS-V5-M1-STRUCTURED-OWNERSHIP-01=running`，第一子阶段只修复可观测性：已新增 deterministic chunked NPZ 合同，分别保存 per-Gaussian geometry/unary、per-view SAM/visibility/boundary/depth/view-angle observation 与 sparse per-edge topology；V4 aggregate state 不再作为 V5 唯一证据源。
- reliability-aware effective-count unary 已实现纯函数与 byte-stable schema tests；B0/B1/B3 只在 development 比较，B2 hierarchical 延后，graph/Transformer/semantic split 继续 disabled。当前没有读取任何 development quality，也没有选择 arm 或参数。
- 8 个 development scene 的 DriveStudio index=`382/827/296/756/276/663/425/350`；raw=`14,220/14,220`、processed=`8/8`。此前 `0/1280` 仅是三前向相机+LiDAR keyframe 的已废弃粗审计；精确六相机+完整 LiDAR 时间链已由 selective extraction 和逐文件 SHA 证据取代。

## V5 P0 / KITTI archive 审计（2026-08-14）

- 当前路线=`WorldSim V5 / StructDelta`，分支=`research/worldsim-v5-structdelta`；`WS-V5-P0-SCOPE-FREEZE-01=running`。当前只授权 P0、M1-D0、M2-D0 和结果前数据/适配器审计，不授权完整 M1/M2/M3、新模型大训练、fresh test quality 或 KITTI 参数搜索。
- `WS-V5-D1-KITTI-ARCHIVE-AUDIT-01=done`：7 个 ZIP 的 central directory、路径安全、序列集合、小包 payload CRC、全 archive SHA-256 和解压磁盘预算均已审计。审计实现 source HEAD=`1b64d668a90796666af7de8d53a6b8d4eaba7839`，manifest SHA=`56388fc64e36c77ebac5a6ee761aa1a17297faeb876347715e8c6e9d52ec23a7`。
- 权威证据：`docs/KITTI_TRACKING_ARCHIVE_AUDIT_V5.md`、`docs/KITTI_TRACKING_ARCHIVE_METADATA_V5.json`、`docs/KITTI_TRACKING_ARCHIVES_V5.sha256`。7 包总计=`67,746,799,901 bytes`（`63.094 GiB / 67.747 GB`）；training/testing=`21/29 sequences`、image frames=`8,008/11,095`。
- `WS-V5-D1-KITTI-ADAPTER-01=blocked`：唯一 archive gate failure 为 `sensor_frame_alignment`。`training/0001` 的 stereo frame=`447/447`，LiDAR=`443`，缺 `000177`–`000180`；不得静默取交集后把 coverage 写成完整序列。
- adapter 另有两项实现 blocker：官方 `R_rect/Tr_velo_cam/Tr_imu_velo` calibration 行无冒号，V4 parser 会漏读；官方 OXTS 每行是 30-field 导航记录，V4 loader 会误当 12-value `3×4` pose。二者必须在 V5 修复并通过 2-sequence 坐标/投影/pose/track-ID smoke。
- official testing split 不含 `label_02`；带 GT 的 V5 cross-domain formal pool 只能在结果前从 21 个 training sequences 冻结，testing 只允许无标签 engineering smoke。
- 本次 download/extraction/quality/training/parameter-search=`0/0/0/0/0`。当前 free=`99.800 GiB`，预计解压后 free=`36.710 GiB`；后续只允许同文件系统 `.partial` staging、post-extract 审计和原子发布，保留原 ZIP，不覆盖公共盘。
- 下一步：先完成 V5 P0 scope/failure forensics；KITTI 路线需冻结缺帧 common-frame/abstain 合同、修复 calibration/OXTS parser，再执行 2-sequence adapter smoke。上述门通过前不冻结 10-sequence formal，不运行 cross-domain quality。
- P0/forensic 输出已建立：`configs/worldsim_v5/p0_scope_v1.yaml`、`configs/worldsim_v5/nuscenes_fresh_cohort_v1.yaml`、`docs/WS_V5_M1_FAILURE_FORENSICS.md`、`docs/WS_V5_M2_GEOMETRY_FORENSICS.md`；四者当前分别为 `running/pending/running/running`，不是完成声明。
- M1-D0 已确认历史 r200 只有 `3 evaluable + 3 abstain` 且 `0/6` directional support；四个 canonical state 的 `18.82%–30.68%` Gaussian 没有正/负 observation count，state 又未持久化 per-view observation、投影 boundary distance、center/covariance 或 topology disagreement，因此现有产物不足以判定是 unary reliability 还是 topology 根因。
- M2-D0 已对 r222 全 `154` request 回溯：`130` 个有候选、`24` 个 role-asset blocked，共 `214` candidates；`192/214=89.72%` 的 geometry risk 饱和为 1，所有未归一化 rendered MAE `>=0.5 m` 的 `192/192` candidates 均失去 tail 排序。
- M2 accepted-only 的同请求比较并不退化：`83` accepted 上 router/TELEA scene-balanced geometry MAE=`1.62295/2.01453 m`，delta=`-0.39158 m`；总表 `+3.39081 m` 主要来自 `47` 个 risk-abstain 以 atomic no-op 保留在 denominator。V4 geometry caveat 继续成立，但不得误写成“accepted candidate 普遍比 TELEA 差”。
- 当前 forensic 结论只允许启动缺失字段采集与 evaluator/reference 审计；禁止根据 V4 validation 重调 0.5 m normalization、router weights/threshold 或直接实现 graph/geometry-first/full M1/M2。

## V4 终局归档与路线关闭（2026-08-14）

- `WorldSim V4 / EviDelta-GS` 已关闭；V4 分支当前执行授权=`none_v4_closed`，不得从旧计划或历史 terminal 恢复 pending 任务。
- 终局文档/轻量证据包=`docs/archive/2026-08/worldsim-v4-final/`，归档提交=`c7e4c969a95536d26d0a17a1c0d1d548f9a247dc`；包内 `79` 个文件，其中 `78` 个内容文件由 `SHA256SUMS` 覆盖并在 Linux 端逐条校验通过，清单 SHA=`947f045024e67d6fa1576fe2e8469adc6d9841e894860d7614ca1caeda820e21`。
- 归档冻结的 closeout source HEAD=`403c5703a755c999d42a5ec3eb063db6cc751761`；权威联合审计仍为 r336，M1=`rejected`、M2=`done_with_geometry_caveat`、M3 test=`confirmed`。
- 附录入口=`docs/archive/2026-08/worldsim-v4-final/TECHNICAL_REPORT_APPENDIX_INDEX.md`；M1 r200/r201、M2 r212/r222、M3 r238/r335/r336 的 summary/status/manifest、主要逐项指标与冻结配置均有轻量副本。
- `/root/autodl-tmp` 于 2026-08-13 到达 KITTI tracking 压缩包，但这不改变 V4 的 `blocked_local_dataset_missing` 历史事实，也不构成 V4 cross-domain 结果。压缩包仅作为 V5 新任务的待审计输入。
- 下一路线只能在新分支 `research/worldsim-v5-structdelta` 上按 `WS-V5-P0-SCOPE-FREEZE-01` 启动；P0 前禁止完整 M1/M2/M3 实现、新模型大训练、fresh test quality 和 KITTI 调参。

## V4 M1/M2/M3 与 18-scene exact-once test 收口（2026-08-13）

- M1 结论继续冻结为 `rejected`：canonical validation r200 为 `3 evaluable + 3 abstain`、方向支持 `0/6`，r201 禁止 M1 feature expansion；没有把后续 M2/M3 结果倒写成 M1 成功。
- M2 风险路由 r222 通过：冻结 `uncertainty_forward / threshold=1.0` 与 `TELEA` comparator，`83/154` accepted、`71/154` abstain；hole geometry MAE 仍明确退化 `+3.3908096237 m`。
- M3 validation r238 通过：`3 evaluable + 3 abstain`；warp L1 relative improvement=`0.3041063132`，temporal LPIPS relative improvement=`0.0264715072`；冻结参数=`4 control points / acceleration 0.10 / retention 0.50 / alpha 0.40`。
- 18-scene test canonical aggregate=`/root/autodl-tmp/runs/worldsim_v4/WS-V4-M3-TEMPORAL-DELTA-01/20260813T225624Z__m3-test-aggregate18-s0-r335`（`20260813T225624Z__m3-test-aggregate18-s0-r335`）；exact-once attempt/completion=`18/18`，完整 denominator=`12 evaluable + 6 abstain = 18`，结论=`confirmed`，test gate=`true`（`5/5` checks passed）。
- test aggregate：warp L1 baseline/candidate/relative improvement=`0.0618690015/0.0405895766/0.3439432407`；temporal LPIPS baseline/candidate/relative improvement=`0.0263505519/0.0220381359/0.1636556253`。
- 冻结与读取合同：source commit=`029d819e0abb63d2edacb811be9ea2153589e92f`，freeze-only commit=`83cb82872bf747c2b1c79fbc2a9982320f972413`，freeze SHA=`0b11d80057560202037d34a4eb9df10853461d7b5d3c6a333f9975d5105d1efb`；18 个 attempt 在读取前以 exclusive marker 消费，聚合器未重读 test source content，parameter/threshold search 均为 false。
- 结论边界：只覆盖冻结的 nuScenes 18 scenes、三前向相机、2–4 s 连续 clips 与单 RTX 3090；不外推 KITTI、闭环安全、长时序或 repair geometry dominance。aggregate summary SHA=`64d1a47c290c218cb3baecf66fd7a8eed2a7c65b793e30195b71e57af7f12519`，pre-closeout HEAD=`83cb82872bf747c2b1c79fbc2a9982320f972413`。

## V4 当前状态（2026-08-13，M2 validation 通过后）

- 当前路线：`WorldSim V4 / EviDelta-GS paper-first`，分支
  `research/worldsim-v4-evidelta`；M2 validation 聚合实现提交=`1cc90b1`。
- `WS-V4-M1-EVIDENCE-FIELD-01` 已冻结为 `rejected`。canonical validation=
  `/root/autodl-tmp/runs/worldsim_v4/WS-V4-M1-EVIDENCE-FIELD-01/20260812T204156Z__m1-validation-six-scene-confirmation-s0-r200`；
  `3 evaluable + 3 abstain`，方向支持=`0/6`（门槛 `>=4/6`），Boundary F1/Brier/ECE delta=
  `-0.0664623346/+0.0024487362/+0.0024972500`。base RGB/checkpoint exact，未重搜参数，test quality 未读；
  rejection audit=
  `/root/autodl-tmp/runs/worldsim_v4/WS-V4-M1-EVIDENCE-FIELD-01/20260812T210150Z__m1-validation-rejection-audit-s0-r201`。
- `WS-V4-M2-REPAIR-ROUTER-01` 已完成并通过 validation。development freeze=
  `/root/autodl-tmp/runs/worldsim_v4/WS-V4-M2-REPAIR-ROUTER-01/20260812T233139Z__m2-development-router-selection-s0-r212`；
  frozen router=`uncertainty_forward / threshold=1.0`，matched baseline=`TELEA`。
- M2 canonical validation=
  `/root/autodl-tmp/runs/worldsim_v4/WS-V4-M2-REPAIR-ROUTER-01/20260813T064330Z__m2-validation-confirmation-s0-r222`；
  六场 denominator=`3 evaluable + 3 scene abstain`，请求 denominator=`154`，其中 role-asset abstain=`24`；
  router accepted/abstain=`83/71`，coverage=`0.5389610390`。相对 frozen TELEA 的 scene-balanced delta：
  PSNR=`+0.0539729695 dB`、SSIM=`+0.0004358785`、LPIPS=`-0.0007536737`、hole PSNR=
  `+3.1797798583 dB`、static LiDAR MAE=`+0.0000499586 m`；abstain-error minus accepted-error=
  `+0.1241311528`。全部冻结门通过，summary/manifest/status SHA=
  `6bfeb3c6...b1a95 / 702cdb48...47dd / 4fcc7b6e...e75b`，test quality 未读。
- M2 的 hole geometry MAE 相对 TELEA 退化 `+3.3908096237 m`；由于预注册门只要求 hole PSNR/LPIPS/geometry
  至少一个主端点改善，M2 可以晋级但不能宣称 repair geometry dominance。
- 当前唯一授权：`WS-V4-M3-TEMPORAL-DELTA-01` 的 development 2–4 s 连续 clip 协议、实现与验证。
  M3 必须先比较 frame-independent/linear/cubic/evidence-memory/full arms；未通过 M3 前不得生成
  `V4_TEST_FREEZE.json`，不得读取 18-scene test quality。

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
