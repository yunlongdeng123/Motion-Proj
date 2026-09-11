# 历史原始记录 060

> 冻结历史，不是当前执行规则。当前规则见 [scaling law](../../../auto-research_scaling_law.md) 与 [AGENTS](../../../AGENTS.md)。
> 原文件：`docs/RESEARCH_FAILURES.md`；迁移前提交 `abbd431c`。原有相对链接按原目录解释，证据路径原样保留。
> 按索引读取相关小节；旧哈希、过度门控和资源指令均不重新生效。

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

