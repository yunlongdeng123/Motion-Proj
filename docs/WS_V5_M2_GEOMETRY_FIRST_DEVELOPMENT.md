# WorldSim V5 M2 Geometry-First Development Archive

- Task：`WS-V5-M2-GEOMETRY-FIRST-REPAIR-01`
- 日期：`2026-08-14`
- 当前状态：`running`
- 当前阶段：`gaussianization_density_mechanism_supported_representation_repair_next`
- 数据范围：仅 frozen development `scene-0471`；validation/test/KITTI quality 均未读取
- 参考范围：`base_background_depth_model_proxy_not_ground_truth`

## 1. 当前结论

逐 actor 修复单位上的非神经 surface 序列已经按计划完成 G0→G1→G2→G3。没有任何复杂 surface 通过相对 G0 的冻结广泛改善门：

| Arm | evaluable | `>=0.5m` 改善 | candidate−G0 mean MAE | candidate−G0 median MAE | 结论 |
|---|---:|---:|---:|---:|---|
| G0 robust plane | 22 | n/a | n/a | n/a | 22/22 raw absolute gate fail；只保留基准，不是 safe candidate |
| G1 piecewise plane | 22 | 5 | `+3.658565m` | `+2.300282m` | rejected |
| G2 moving least squares | 22 | 8 | `+3.005506m` | `+1.620793m` | rejected |
| G3 robust quadratic | 22 | 11 | `+0.103693m` | `-1.489037m` | rejected；稳定性与 mean gate 未通过 |

门槛在读取 arm 结果前固定为：至少 `18` 个可评请求；至少 `14` 个请求改善 `>=0.5m`；mean 与 median candidate−G0 raw MAE 都必须小于 `0`。G3 虽然中位数改善，但只有 `11/22` 广泛支持且均值仍轻微退化，不能按事后均值/中位数择优。

因此当前没有 geometry-safe candidate，`WS-V5-M2-GEOMETRY-FEASIBLE-ROUTER-01` 继续锁定。r011 已把 G0 的 raw→Gaussian asset→post-render 附加误差定位到 stride-2 稀疏采样；下一步只允许冻结一个 representation repair 并重新过独立 development gate，不得跳到 router、validation 或神经 surface model。

## 2. 请求单位修正

r002/r003 最初复用了 r036 的同 view actor-union mask，把多个 actor 合并为一个 hole。最大 union 达 `152,410` pixels，这不符合 V4 的“一次编辑一个 actor/request”语义，也会让单个大洞支配均值。

r004 以冻结 SAM prompt replay 重建逐 actor mask：

- `6` 个冻结 views；`4` 个 SAM available、`2` 个 unavailable；
- `23` 个 actor-view requests；`22` accepted、`1` rejected；
- 逐 actor mask 的 union 与 r036 原 union 逐像素 exact；
- 后续 r005–r009 统一使用 `one_actor_one_view_one_hole`，rejected request 以 `ABSTAIN_SAM_MASK_REJECTED` 留在分母。

r002/r003 仍是有价值的 staged instrumentation 与错误 request-unit 负证据，但不得进入最终 surface arm 选择。

## 3. Staged geometry 与 Gaussianization 证据

r005 在逐 actor G0 上得到：

- `23 requests = 22 evaluable + 1 abstain`；
- raw geometry MAE mean/median=`8.5872/8.7151m`，`22/22` absolute gate fail；
- post-Gaussianization MAE mean/median=`9.0056/9.0040m`；
- Gaussianization delta mean/median=`+0.4184/+1.5373m`；
- `16/22` 请求的主要附加误差来自 Gaussianization，故该层是下一轮独立 forensic 对象；
- reference confidence mean/median=`0.0585/0.0582`，范围=`0–0.1557`。

这证明 G0 raw builder 与 Gaussianization 都有问题，但不允许把与 base background proxy 的误差写成真实道路 GT 误差。低 reference confidence 必须与所有 MAE 同表出现；独立几何 claim 继续为 `false`。

r011 用 `BASE / OPAQUE / DENSE / DENSE_OPAQUE` 四个结果前冻结臂做 `2×2` 机制取证，并对 r005 的 mask、reference、raw、pre、post float16 与 BASE common mask 做 `22/22 exact` replay。四臂共同评估没有丢失任何 baseline pixel：

| 对比 | `>=0.1m` 改善 | candidate−BASE mean MAE | median | 机制解释 |
|---|---:|---:|---:|---|
| OPAQUE−BASE | `0/22` | `+0.059686m` | `+0.065773m` | 单独提高 opacity 退化，alpha/background mixing 不受支持 |
| DENSE−BASE | `20/22` | `-0.424179m` | `-0.480927m` | stride `2→1` 广泛改善，density arm 过冻结机制门 |
| DENSE_OPAQUE−BASE | `19/22` | `-0.388647m` | `-0.441758m` | 过门，但弱于 DENSE |
| DENSE_OPAQUE−DENSE | `0/22` | `+0.035533m` | `+0.039717m` | 在 dense 条件继续增 opacity 仍退化 |

标准 `2×2` 描述性对比的 density main effect mean/median=`-0.436256/-0.493625m`，opacity main effect=`+0.047609/+0.052537m`，interaction=`-0.024153/-0.026682m`。正式 summary 使用保守词汇 `multiple_gaussianization_factors_have_broad_mechanism_support`，因为 DENSE 与 DENSE_OPAQUE 两臂都过门；结合 OPAQUE 两个条件都退化，当前可归因解释是“density 必要、opacity 不受支持”，不是选择 DENSE 为正式方法。

## 4. 正式运行时间线

所有运行位于：

`/root/autodl-tmp/runs/worldsim_v5/WS-V5-M2-GEOMETRY-FIRST-REPAIR-01/`

| Run | 状态 | source | 作用与结论 |
|---|---|---|---|
| r001 `20260814T195110Z__m2-staged-geometry-scene0471-s0-r001` | blocked | `46226b45` | 错误要求 6/6 views 都有 SAM mask；实际 4 available+2 unavailable。工程合同失败，不是方法失败 |
| r002 `20260814T195500Z__m2-staged-geometry-scene0471-s0-r002` | done | `31a147d3` | union-mask staged G0；raw/pre mean=`21.7609m`，post mean=`6.3930m`；仅用于定位与解锁复杂 surface |
| r003 `20260814T200300Z__m2-g0-g3-surface-ablation-s0-r003` | done | `d9030f87` | union-mask G3 仅 `1/4` raw request 改善；后因 request-unit 不等价，不作为最终 G3 裁决 |
| r004 `20260814T201100Z__m2-scene0471-per-actor-masks-s0-r004` | done | `607479e9` | 23 个逐 actor requests materialized；22 accepted+1 rejected；union exact replay |
| r005 `20260814T202000Z__m2-scene0471-per-actor-g0-s0-r005` | done | `64c26c6c` | G0 raw `22/22` fail；Gaussianization primary `16/22` |
| r006 `20260814T203000Z__m2-scene0471-per-actor-g0-g1-raw-s0-r006` | done | `d5f4e647` | G1 改善 `5/22`，mean/median delta=`+3.6586/+2.3003m`，rejected |
| r007 `20260814T202900Z__m2-scene0471-per-actor-g0-g2-raw-s0-r007` | blocked | `8a729de6` | runner 局部指标字典覆盖 frozen arm tuple，artifact serialize `KeyError: 0`；无方法结论 |
| r008 `20260814T203200Z__m2-scene0471-per-actor-g0-g2-raw-s0-r008` | done | `cdf329eb` | G2 改善 `8/22`，mean/median delta=`+3.0055/+1.6208m`，rejected |
| r009 `20260814T203700Z__m2-scene0471-per-actor-g0-g3-raw-s0-r009` | done | `cc6c9058` | G3 改善 `11/22`，mean/median delta=`+0.1037/-1.4890m`，rejected |
| r010 `20260814T205600Z__m2-scene0471-gaussianization-factors-s0-r010` | blocked | `5daeaedc` | launcher 为日志重定向预建 run 目录，formal overwrite guard 在模型加载前拒绝；无 GPU、无方法读数 |
| r011 `20260814T205700Z__m2-scene0471-gaussianization-factors-s0-r011` | done | `5daeaedc` | BASE `22/22 exact`；DENSE/DENSE_OPAQUE 分别 `20/22`、`19/22` 改善，OPAQUE `0/22`；density 机制受支持，未选择方法臂 |

r007 的变量遮蔽由 commit `cdf329eb8721f7a7bb2f6ab0a5cf62d5f1e1c59e` 修复，并以独立 r008 重跑。r010 是启动器目录生命周期错误，r011 改为外部日志后完成；r001/r007/r010 目录和 terminal 均不覆盖、不复用。

## 5. 关键完整性哈希

| Run | summary SHA-256 | status SHA-256 | diagnostics/mask SHA-256 |
|---|---|---|---|
| r001 | n/a | `b748de0705b5681611f7129ed6ac463d24ab99c6b52f8a3c86b2cc5f5510b36b` | n/a |
| r002 | `094d1078401f790b05e4851735736088af8731dda52029433755b08d5ef348ca` | `bc7e685c887fc3c6ed114187af2336fb122e447fce6e84dbe15a1326d852e4f1` | `10f391dd72b3b3fed9879779f8d9bda1f5f49e43d849a3df7e69488834336f6d` |
| r003 | `bf81094a0d175c37ce306eb8e035e27548c7c9e1ec18f374476438d6341a1963` | `e71cb07f5527a4959c92001793939ea7dfb6479ffd271036d38b80af464f7449` | `530a36693f439c2dd8a8ba647bbf8bebb8d74bea5ca3688baf27a5834fa1714d` |
| r004 | `e931c09fa4c1f6ca34b9b302bc5902cf9d1b183d77f6eaaeecea1083d27911a5` | `0b8250529d85e09cd29fdaab0f1fa75e1282f35b1863b6a5e0a14c80daccade6` | `98b2d44d9bde0795a8956e6050a97ea5baefc89b57d278f06cd0ebe80f133ca1` |
| r005 | `b93c4be44c597762a899e46ef4c89c5a25ca1c54e09824ceca4d82c46f18eb15` | `1f0f31b3b861ab9575e59ca8d833ae4462bfdd21f62f40c44b5f3986b3dbbde3` | `29914287f2a34bc0d84565e74a9379ab458b7e4d696f0c9acee4c9c7819beed2` |
| r006 | `015a507471cb18fa728d98548d1714c8898125e4f165c0f662e797bdfca30fb9` | `5148458635aafcd7f4c47a81ea0a5ac4cf824602714c3a3eb57492efc5933955` | `f54f03b09ec9b817dc8bb68a45b1da4fa4b821c3a0f69b94d34b18eae0695bdd` |
| r007 | n/a | `876220e27cd38e49f620c01b61800a1b20449fb75c68d21a1281aa9e76462303` | n/a |
| r008 | `906bfc7b569265d6806e25b42bb855352b64b4e1cdb4562fb267aadf72efcf92` | `874c9f907365fbcbcccd1ad3073eb3495f962e70c0d491630431f2935e738fb3` | `2298fa0792805b12f01e8be661e9f16b86e203839ec764c35acbccbea0564252` |
| r009 | `8a253325f5b689239712b6de3e862ddfd1b1b10adb82ac6583286c755ef0d06c` | `a40727fb94dd64141cee37f83626811261c65222923f7897abaa33179f722426` | `d22fa44fa6a10cec29ed4bdd1481950211f069f73106469715914c038fce79b3` |
| r010 | n/a | `ae6724c2fba28d5c51b43fbe68a3a79754a0ee10acfcc09dabbadbd678d8f92d` | n/a |
| r011 | `47a08899b5e82e8297029b58d56aabfc0fee7afd1822d6b498d9a1734de87204` | `5a4a3ebb764afc50e4b715d0968acd3601c279e315bbab809d5780c1eb23d076` | `234adf931e6b0abc0808b0858e38c40f7b76f119ed8b32881d3a16150231b1cf` |

完整 fingerprint/manifest/resolved/events 哈希与机器可读字段见 [`archive/2026-08/worldsim-v5-m2/M2_R001_R009_METADATA.json`](archive/2026-08/worldsim-v5-m2/M2_R001_R009_METADATA.json) 和 [`archive/2026-08/worldsim-v5-m2/M2_R010_R011_GAUSSIANIZATION_METADATA.json`](archive/2026-08/worldsim-v5-m2/M2_R010_R011_GAUSSIANIZATION_METADATA.json)。r011 manifest `31/31` inventory files 已重哈希，无错误。

## 6. 技术报告附录边界

可以写：

- actor-union request 会改变几何诊断并产生大洞离群点；逐 actor request 是必要协议修正；
- frozen scene0471 model proxy 上，G1/G2/G3 均未稳定超过 G0；
- G0 本身在 22/22 请求上未过 absolute raw gate，且 16/22 存在主要 Gaussianization 附加误差；
- 当前证据支持继续诊断 representation/rendering，不支持选择新 surface 或改 router。
- stride-1 density 在 model proxy 上广泛减少 post-render representation gap；opacity `0.85→0.99` 在 sparse/dense 两个条件都退化。

不得写：

- G0 恢复了真实道路；
- G3 因中位数改善而通过；
- union-mask r002/r003 是逐 actor 方法比较；
- scene0471 development proxy 等价于 validation、KITTI 或独立 geometry GT。
- DENSE 已成为正式 geometry-safe method、router 已解锁，或应直接把 stride-1 带入 validation。
