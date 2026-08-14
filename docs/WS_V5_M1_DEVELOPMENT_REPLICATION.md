# WorldSim V5 M1：三场景 Result-Blind Development Replication

日期：2026-08-14

任务：`WS-V5-M1-STRUCTURED-OWNERSHIP-01`

用途：技术报告正文与附录的复制性结论入口；重型 mask、Gaussian、edge、posterior 与 checkpoint 仍保存在不可变 formal run 目录。

## 结论

按结果读取前冻结的顺序，复制场景固定为 development cohort 的前三场：`scene-0471 / scene-1087 / scene-0379`，不是按质量挑选。所有场景复用相同 SAM 视图、B0/B1/B3 unary、G0–G3 graph、`k=6`、两轮 `0.25` diffusion、固定阈值与资源合同。

G3 相对同一 unary 的 G0 在六个 `scene × unary` 单元中只有 `3/6` 个 Boundary F1 为正，低于预注册的 `>=4/6`。六单元均值 `+0.0016107723`，平均 FN semantic mass 增量 `+0.0025676789`，分别通过 `>0` 与 `<=+0.01`；但 scene1087 的 G1 cross-proxy affinity 已为精确 `0`，G3 为 `1.2800523e-29`，不满足每场景 `G3 < G1`。

正式裁决为：

> `physical_graph_development_replication_rejected_3of6_boundary_support`

这不是“graph 在所有场景都有害”：scene0471 两个单元为正，scene0379 的 B3/G3 也为正，且总体均值仍为正。失败点是方向不够稳定，不能选择 G3，不能读取 validation，不能解锁 Transformer 或 semantic split。semantic split 只有在独立的 boundary-residual forensic 证明 boundary ambiguity 是主要残差后才允许另立条件任务；当前 `semantic_split_allowed=false`。

## 冻结协议与门槛

- cohort：前三个 frozen development scenes，scene index=`382 / 827 / 296`；选择时未读 quality。
- 单元分母：`3 scenes × 2 unary inputs = 6`。
- comparator：每个场景、每个 unary 的 G3 对同一输入的 G0。
- 通过条件：Boundary F1 正向 `>=4/6`；平均 Boundary F1 `>0`；平均 FN 增量 `<=0.01`；每场 G3 cross-proxy affinity 严格低于 G1。
- 禁止：结果后改参数、validation/test/KITTI quality、formal arm selection、semantic split 自动解锁。
- 配置：`configs/worldsim_v5/m1_development_replication_v1.yaml`。

## 稀疏 SAM 与 unary 分母

| Scene | SAM run | Available/30 | Actors / accepted boxes | Unary run | Eval accepted + abstain | Gaussians |
|---|---|---:|---:|---|---:|---:|
| scene0471 | r036 | 18 | 17 / 61 | r037 | 8 + 7 | 859,613 |
| scene1087 | r039 | 2 | 1 / 2 | r042 | 1 + 14 | 931,223 |
| scene0379 | r040 | 6 | 5 / 6 | r043 | 3 + 12 | 1,187,291 |

| Scene / unary | ΔBoundary F1 vs B0 | ΔIoU | ΔBrier | ΔECE | ΔFN mass |
|---|---:|---:|---:|---:|---:|
| scene0471 / B1 | +0.1074366511 | +0.1166650769 | -0.1017390995 | -0.1109825384 | +0.0915315025 |
| scene0471 / B3 | +0.1050785689 | +0.1161613366 | -0.1034674387 | -0.1124314009 | +0.0954772599 |
| scene1087 / B1 | -0.0227915308 | -0.0800420964 | -0.0042454354 | -0.0165130574 | +0.2418383769 |
| scene1087 / B3 | -0.0391484586 | -0.1379522362 | -0.0020478637 | -0.0195231599 | +0.3117007940 |
| scene0379 / B1 | +0.1288147181 | +0.1447065180 | -0.0015423330 | -0.0041292365 | +0.1881035260 |
| scene0379 / B3 | +0.1218064992 | +0.1410337079 | -0.0014214774 | -0.0032385733 | +0.1881889094 |

unary 的 Boundary F1 方向在场景间不稳定，而且三场均存在明显 FN 代价。scene1087 只有一个可评估视图，不能把其负结果扩大解释为总体方法失败；但预注册分母要求保留该场景和 `14/15` abstain，不能删除或事后补 prompt。

## G3 对同 unary G0 的六单元结果

| Scene / unary | ΔBoundary F1 | ΔIoU | ΔBrier | ΔECE | ΔNLL | ΔFN mass | 方向 |
|---|---:|---:|---:|---:|---:|---:|---|
| scene0471 / B1 | +0.0085850810 | +0.0065298793 | -0.0014481246 | -0.0024063994 | -0.0331770412 | +0.0014910135 | positive |
| scene0471 / B3 | +0.0062453688 | +0.0062555836 | -0.0013714548 | -0.0024959492 | -0.0039721693 | +0.0010607626 | positive |
| scene1087 / B1 | -0.0004748720 | +0.0004804763 | -0.0000582786 | +0.0004748151 | -0.0002251314 | +0.0009583957 | negative |
| scene1087 / B3 | -0.0073722094 | -0.0006513279 | +0.0000145295 | -0.0000562215 | -0.0003849830 | -0.0003559098 | negative |
| scene0379 / B1 | -0.0002365704 | +0.0010458007 | -0.0000112711 | -0.0001873372 | -0.0000553587 | +0.0059328603 | negative |
| scene0379 / B3 | +0.0029178354 | +0.0023879564 | -0.0000630200 | -0.0004588126 | +0.0000632329 | +0.0063189512 | positive |

| Scene | G1 cross-proxy affinity | G3 cross-proxy affinity | `G3 < G1` |
|---|---:|---:|---|
| scene0471 | 0.0083646045 | 0.0040198140 | pass |
| scene1087 | 0 | 1.2800523e-29 | fail |
| scene0379 | 0.0003274808 | 0.0000761250 | pass |

每个完成 graph run 的 G0 均与其绑定 unary float16 render 逐像素 exact，且 base checkpoint 前后 SHA 一致。cross-proxy 只用于事后 topology audit，graph 从未消费 Background/RigidNodes proxy。

## Gate 逐项裁决

| Gate | 观测值 | 要求 | 结果 |
|---|---:|---:|---|
| Boundary F1 正向单元 | 3/6 | >=4/6 | fail |
| mean ΔBoundary F1 | +0.0016107723 | >0 | pass |
| mean ΔFN semantic mass | +0.0025676789 | <=+0.01 | pass |
| 每场 G3 affinity < G1 | 2/3 | 3/3 | fail |
| validation 自动解锁 | false | 必须保持 false | pass |
| formal arm selection | false | 必须保持 false | pass |

总门禁为 `rejected`。均值正不能覆盖 scene-level directional support 和 topology 条件失败。

## 保留的基础设施阻塞

| Run | 状态 | 原因 | 后继 |
|---|---|---|---|
| r041 scene1087 unary | blocked | SSH 输出通道关闭后，进程在计算末尾写日志触发 `BrokenPipeError` | 同配置以脱离式 r042 完成；r041 不覆盖 |
| r044 scene1087 graph | blocked | runner 把 scene0471 的 `8 accepted + 7 abstain` 硬编码为所有场景分母 | `d55a067` 改为绑定 summary/diagnostics 的动态双重校验；r045 完成 |

r044 暴露的是通用化合同缺陷，不是数据或方法质量失败。修复同时要求 accepted>0、B1/B3 `(frame,camera)` 键一致、accepted+abstain=总分母，并保留旧 replay 字段兼容。

## Canonical run 路径

- r039：`/root/autodl-tmp/runs/worldsim_v5/WS-V5-M1-STRUCTURED-OWNERSHIP-01/20260814T182142Z__m1-scene1087-sam-sparse-s0-r039`
- r040：`/root/autodl-tmp/runs/worldsim_v5/WS-V5-M1-STRUCTURED-OWNERSHIP-01/20260814T182202Z__m1-scene0379-sam-sparse-s0-r040`
- r041 blocked：`/root/autodl-tmp/runs/worldsim_v5/WS-V5-M1-STRUCTURED-OWNERSHIP-01/20260814T182655Z__m1-scene1087-unary-diagnostic-s0-r041`
- r042：`/root/autodl-tmp/runs/worldsim_v5/WS-V5-M1-STRUCTURED-OWNERSHIP-01/20260814T183210Z__m1-scene1087-unary-diagnostic-s0-r042`
- r043：`/root/autodl-tmp/runs/worldsim_v5/WS-V5-M1-STRUCTURED-OWNERSHIP-01/20260814T184610Z__m1-scene0379-unary-diagnostic-s0-r043`
- r044 blocked：`/root/autodl-tmp/runs/worldsim_v5/WS-V5-M1-STRUCTURED-OWNERSHIP-01/20260814T185922Z__m1-scene1087-graph-diagnostic-s0-r044`
- r045：`/root/autodl-tmp/runs/worldsim_v5/WS-V5-M1-STRUCTURED-OWNERSHIP-01/20260814T190405Z__m1-scene1087-graph-diagnostic-s0-r045`
- r046：`/root/autodl-tmp/runs/worldsim_v5/WS-V5-M1-STRUCTURED-OWNERSHIP-01/20260814T190715Z__m1-scene0379-graph-diagnostic-s0-r046`

## 完整性哈希

| Run | summary / status / fingerprint / manifest / diagnostic-or-mask / resolved config SHA-256 |
|---|---|
| r039 | `4b4f8d2b85809926adf02827203cfb9816fdabb49a9180617baee6fe06345f68 / cdd766f018415cbad36d44430fdbf2c68f29cc53b9c04f4c2812f99bf8da517e / 66da02fd436aa4bbbf21e06ebcb7b3f35cf297081bf0cf431c363d88bf9b741c / 00d1ad9d98e9673cdb64ecf10ba7a9bbe3aba2f8de9964dfd1c7ba6f5d5f5306 / da1bd44e0955fd04c29e33ed2ee71d4ba30fdf8e7f1002948d6e0dbaf13b334c / 6427f57cd4eb30e374151c650db4ea53317bc2293b73f6a08235416fa965486d` |
| r040 | `47f33cb4baa49251a7ffdc4ff7133273504934cc84656c3a405e90c9c7cddbb6 / 3ab01822211ab9c3db7154e29045672085c092fabac9df259f61a71b0e1f9fb8 / 67794a607a9615a9688330f147e5b800fae820ed5e1b74ffb1b997430db5f045 / eaa144e64d12f6f8ec9a50a5633c4b37778036471781321d8349bb93224e0c8f / a4f7e6955e2cf6b1580d76f00ecd7ae662778a267944758a3f4b5bdf921fc61d / c754da75b181655cee3f292f65d9d24e14d081a7604b12c5f520bca05e9465c5` |
| r042 | `d19cabd9a2bb48ded6e73ef6bf83a8073d3704198dd69ec628f39ddd98e47f8b / b35c7c89d8ad8520268b269cf6caa7db4c0628ccf01df130d66d98197cc79cd0 / 834a34f292bcf0be34856391f21ecd8e1408fa142d77301af8c8f55d9025b05d / 77b3965b4514c08b33f8ce70aed26aa661c9f14349559c992fc1f4729d73214a / e97e8d46fdff8a80792aec68280d08ca28d08ac854de860c8a0025215cbeb2db / ed069385ce6bd1393ab2ed789fe42d9b1bb08f9b031c6b75677eb686fb9c1057` |
| r043 | `1beff3d934c2628cb2b7a262d5e7ab75c4cf214db3bdf39cc51c59690140803d / 4887bb53a0665acff666fa072eb21eb0485bde45e016987655889f0edfb4f39b / 00b2e10d49fdb0b3f6dc37b980b25814a8a5d9d2e26ba22769fcd487d19430bc / 4cee48fdd0cc59f6ec2b95841db79ce32edfdb83a64c7d651ff22d49af458a05 / f539be1ce74bdd5eb10dc3d7525f4225299b1d98c7ff4e649144187caf876008 / 1561bbd065c8dbc535a3eb0d82283f02498f27ba51bd729979f2cd674bafb404` |
| r045 | `8e54d359ae97121bae5c3e6495126ddf94b60916ebe40023e32434dbb5762c78 / 11b22206b14428f5fec9f3ea17810c55d195e2b48006c1e6240520fd2dfa1609 / 15e69fac27739aa4e59174fef8802d0a302b19c0511347f78d65d5bea4037526 / 1b9f7c59457860552db3065daf6466245cb3655c35e2aa8f61f7893cd54f5b84 / ea43d11f5a8926ffbb99ef40065c2e159c9e95f7a07723ee30042f3116054e3c / 35aae83bc587d4d3baf8c4cd91d893224ed84a1229e1d3adeb6558e9db6d01bf` |
| r046 | `65a24024e2d9a62d3db5583f87eabf1edf57a402bed9650001ac43e1bf10eb34 / d813babaa6a148be948843dce233bc0bfb49551ede6a561fcdfeabf26b4c824c / d07f58d80bc6665c1718d19bfdb831468b5e9444a817d104aec6b0af1b29b2f1 / 4e148d11dbbb0c441a4423033b3c25820b487705d5fc168d67467d75de4ae0f9 / adbafa7e978a22642f15f0c47cfba9be8bce5c8e469c65482aa2b562bca2dc46 / df73a26184c53c5b7161a264042799505c1568db6ae95271a0dd37f1ad970605` |

r045 edges/posteriors SHA=`1ca4ad905ef23e6cf53b58c9b0a0566811ab61e0754eec272029dde9da69dc3f / 0e2fb2ecbc3b90dd65e64f945e7ed52b1ea50d6459a920952ac50f976bc89ffe`；r046=`09edc24d6427582068f717e2a4067463a6fb4093c0a71f7ccea625b56073aa18 / 611d33672f1eec14c309e6a5b0da39a35c083cf32c55d56d036129f17afc29ce`。

r041 blocked 的 resolved/events/status SHA=`ed069385ce6bd1393ab2ed789fe42d9b1bb08f9b031c6b75677eb686fb9c1057 / 846dfbaa21b084a526229e915f69ac778f75816a9ff98467605773d9e4d028ce / 3c2a810aa16cb775702f1620a250cb87c62f0353ceb93e3f0328c9cfcebc4191`；r044=`35aae83bc587d4d3baf8c4cd91d893224ed84a1229e1d3adeb6558e9db6d01bf / c3c1ded27d5507dc72b74b9645624c919592830a27f0683460569f6599ee88ad / f41a8b63f6cdab0265a2d00dc8585e4a19615f6699c4d43937f34f60ab4c7ad4`。

机器可读副本：`docs/archive/2026-08/worldsim-v5-m1/M1_R039_R046_REPLICATION_METADATA.json`。
