# 历史原始记录 076

> 冻结历史，不是当前执行规则。当前规则见 [scaling law](../../../auto-research_scaling_law.md) 与 [AGENTS](../../../AGENTS.md)。
> 原文件：`docs/RESEARCH_FAILURES.md`；迁移前提交 `abbd431c`。原有相对链接按原目录解释，证据路径原样保留。
> 按索引读取相关小节；旧哈希、过度门控和资源指令均不重新生效。

### V6-F54：独立 verifier arms 合格不等于 factor conjunction 有足够 usable coverage

H-R15-001 canonical run `20260821T145504Z__factorized-verification-s20260821-r1` 只消费 H-R14-001 独立通过的 P1 photo 与 P2 geometry decisions，按双 ACCEPT 才 ACCEPT、双 REJECT 才 REJECT、disagreement 全 ABSTAIN。28 cases 中只有 1 个 joint ACCEPT、13 个 ABSTAIN、14 个 REJECT；joint false-safe 为 `0`、相对 P0 reduction `0.89286`，但 accept coverage `0.03571` 未达到冻结 `0.05`，因此正式 `rejected`。

不得用 OR、放宽 verifier threshold、把 disagreement 计作 ACCEPT 或把门改成 1 case 来恢复。RGB-only ECC proposal 对 photo 优化，却没有约束 geometry，导致两个独立有效 arm 的接受集合错位。H-R16-001 保留同一 temporal source、28-case denominator、P0-P4 thresholds 与后续 conjunction，仅把 outside-mask alignment image 改为等权 RGB gray + robust normalized render inverse-depth；render depth 只用于 proposal alignment，不充当 P2 truth。

### V6-F55：无类型约束的双模型语义共识会形成相关性共同错误

H-R20-001 canonical run `20260821T152456Z__semantic-consensus-s20260821-r1` 在冻结的 12 个 R16 semantic-evidence cases 上，以 DeepLabV3-ResNet50 与 SegFormer-B0 的 hole 内 dynamic-mask IoU `>=0.70` 作为唯一决策证据。该机制成功拒绝了 R16 原先唯一的边缘 false-safe `scene-0048__ad_gs__f057__actor_removal_hole`，但两个模型在 `scene-0048__ad_gs__f052__disocclusion` 上以 IoU `0.88267` 共同预测了错误动态内容，使 2 个 ACCEPT 中 1 个为 false-safe，false-safe rate `0.50`、相对 P0 reduction 仅 `0.08333`，正式 `rejected`。

不得把架构不同等同于错误独立，也不得扫描 consensus threshold；相关模型会在编辑语义不同的 hole 上共同犯错。H-R21-001 保留相同模型、12-case denominator、truth 与质量门，新增由 compiler edit type 决定的 typed semantic contract：`actor_removal_hole` 沿用冻结的 `0.50` 双模型 dynamic IoU 门；`disocclusion` 只有两个模型在 hole 内都预测零 dynamic pixel 才可 ACCEPT。决策仍不读取 target dynamic truth，P4 保持 ABSTAIN。

### V6-F56：actor edit-mask 的轴对齐二阶矩不足以恢复可接受的纹理对应

H-R22-001 canonical run `20260821T153531Z__independent-arms-s20260821-r1` 对六个 `actor_removal_hole` 只使用源 actor-edit mask 与目标已知 hole mask 的中心和轴向二阶矩，估计 axis-aligned affine 后搬运邻帧 actor RGB。actor 子集 P2 geometry 得到 `1/6` ACCEPT、false-safe `0` 并通过独立 gate，但 P1 photo 为 `0/6` ACCEPT，导致要求 P1/P2 同时独立合格的正式 gate 拒绝。四个 photo truth-safe cases 的 masked RGB MAE 仍为 `0.06299–0.07655`，高于冻结 `0.05`；整体 P1/P2 通过只来自未改变的非 actor cases，不能覆盖 actor 子集失败。

不得放宽 P1 阈值或用整体分母掩盖 actor failure。轴对齐中心/尺度对齐无法表示邻帧 actor 的旋转、剪切和透视轮廓变化。H-R23-001 保留相同源/目标 masks、六个 actor 分母、非 actor RGB-D ECC、verifier 与全部 gates，只把 actor 配准改为：以二阶矩 affine 为初始化，在两个二值 edit-mask 的 signed-distance field 上执行一次冻结 homography ECC；仍不读取 target RGB/depth/dynamic。

### V6-F57：edit-mask 轮廓 homography 仍不能建立跨时刻 actor 纹理对应

H-R23-001 canonical run `20260821T154110Z__independent-arms-s20260821-r1` 在相同六个 actor cases 上，以 R22 moment affine 初始化 signed-distance-field homography ECC。actor P1 仍为 `0/6` ACCEPT，P2 仍只有 `1/6` ACCEPT；SDF warp 让若干 photo MAE 从 R22 的 `0.063–0.076` 恶化到 `0.069–0.116`，仅一个 case 改善至 `0.05748`，仍未过冻结 `0.05`。因此整体 P1/P2 合格仍不能覆盖 actor subset，正式 `rejected`。

不得继续扫描 SDF ROI、iteration 或 photo threshold。二值 actor-edit support 约束的是轮廓，不携带姿态、可见面与光照的跨时刻纹理对应；更灵活的 homography 会扭曲错误纹理。既有 H-R9-003 已独立证明 same-time cross-frontend proposal 在 actor 子集上 P1 `4/6`、P2 `1/6` 且 false-safe 均为 `0`。H-R24-001 因此不再拟合 mask，而按 typed asset route 复用已验证来源：static/disocclusion 保持 R16 temporal RGB-D，actor_removal 使用同帧 cross-frontend；所有 verifier 与 actor gates 不变。

### V6-F58：小 actor 分母上的固定 false-safe rate delta 可以数学上不可达

H-R24-001 canonical run `20260821T154603Z__independent-arms-s20260821-r1` 按 hole type 复用已接受的 R16 temporal RGB-D static source 与 H-R9-003 same-time cross-frontend actor source。actor 子集 P1 为 `4/6` ACCEPT、false-safe `0`，P2 为 `1/6` ACCEPT、false-safe `0`；两臂的绝对 coverage/risk 都通过。但 actor P1 的 P0 只有 `1/6` false-safe，最大可能 rate reduction 为 `0.16667`，无法达到从 28-case 全局臂继承的固定 `0.25`，因此要求 P1/P2 均合格的 H-R24-001 按原合同正式 `rejected`。

不得追认 R24 gate，也不得删除唯一 P0 错误或降低风险要求。H-R25-001 保持 R24 proposals、decisions、truth 与 P1/P2 阈值完全冻结，只预注册适合六例小分母的 actor Pareto gate：每臂 coverage `>=0.10`、false-safe rate `<=0.10`，并且 false-safe count 相对 P0 至少严格减少 `1`。该离散计数门在两个臂上均可实现且仍要求实际消除错误；旧 R24 run 继续保持 rejected。

### V6-F59：下游 factorized consumer 必须从拥有字段的冻结 artifact 读取 case metadata

H-R27-001 首个 formal run `20260821T160452Z__three-factor-s20260821-r1` 在组装六个 actor factorized decision 时因 `KeyError: mask_pixel_count` 失败，仅产生 failed `TERMINAL.json`，未构造 gate 或方法结论。R24 的 `verifier_worker/PER_CASE_ARMS.jsonl` 拥有 factor decisions、truth 与 proposal hash，但 `mask_pixel_count` 的 schema owner 是同一 run 的 `CASES.jsonl`；初版 consumer 错把该 metadata 当作 arm row 字段。

修复把冻结 R24 `CASES.jsonl` 及其 SHA256 加入 R27 source contract，并仅从 case-id index 读取 `mask_pixel_count`。不得改变 P1/P2/P3 decisions、truth、三因子合取规则、coverage/risk/count gate、R24 rejected 状态或确认集锁。该失败属于 artifact binding plumbing，不否定 H-R27-001，修复后以新 run 重试。

### V6-F60：factorized validity 的各因子 truth 应做逻辑积，不能要求逐例标签相等

H-R27-001 canonical retry `20260821T160717Z__three-factor-s20260821-r1` 在六个 actor cases 上得到预期决策：`1` ACCEPT、`1` REJECT、`4` ABSTAIN，全部 factor decision disagreement 都 ABSTAIN，ACCEPT 的 joint truth-safe 为真，false-safe 为 `0`，coverage 为 `1/6`。但预注册合同额外要求 photo、geometry、semantic 三种 truth label 逐例相等；实际有两个 case 的 factor truths 不同，因此 run 按合同正式 `rejected`。

不得删除这两个不一致 case 或追认 R27。photo truth 衡量 RGB 恢复、geometry truth 衡量 depth、semantic truth 衡量动态语义，它们本就可以独立真假。H-R28-001 保持全部 proposal、factor decisions、fusion 与风险门冻结，将 joint truth 明确定义为三种 factor truth 的逻辑 AND，并要求精确保留两个 cross-factor truth disagreements；R27 的唯一实质失败项必须仍是错误的 truth identity 假设。以后 factorized evaluator 必须区分 decision disagreement 与 truth-factor diversity。

### V6-F61：未展开的聚合布尔失败不能用于猜测精确 factor-diversity 分母

H-R28-001 canonical run `20260821T161052Z__three-factor-s20260821-r1` 正确使用 factor-truth product，并保持 `1` ACCEPT、`1` REJECT、`4` ABSTAIN、false-safe `0`；coverage、strict error removal、所有 disagreement ABSTAIN、source immutability 与 R27 rejection retention 全部通过。唯一失败是预注册把 cross-factor truth disagreement count 猜成 `2`，完整逐因子展开后实际为 `3`。

R27 只给出了 truth-identity 聚合布尔失败，不能推出失败 case 的精确数量。不得追认 R28 或把实际值 `3` 再硬编码成安全门。H-R29-001 保持全部六例、factor decisions、factor truths、product、fusion 与质量门冻结，要求 factor truth diversity 非零且逐例透明报告；精确 diversity count 作为描述性输出，不参与资格。以后只能从已冻结的逐例 artifact 预注册精确计数，不能从 aggregate failure 反推。

### V6-F62：aggregate actor-layer validity 不能下放给单 actor identity component

H-R32-001 canonical run `20260821T163038Z__identity-factor-s20260821-r1` 用此前接受的 H-R13-009 model-index-0 removal support 与 H-R13-011 `actor_0000` binding，在 R30/R31 layer 内得到 `4,792` 个 identity pixels，覆盖 resized actor effect support 的 `91.02%`。P1 photo 在该 support 上仍 ACCEPT（MAE `0.043739`），且 photo/geometry/semantic 三种 truth evaluation 均 safe；但 P2 geometry mean relative error 为 `0.212180`，超过冻结 `0.20`，P3 DeepLab/SegFormer dynamic IoU 仅 `0.098330`，低于冻结 `0.50`，因此 identity-specific conjunction ABSTAIN，正式 `rejected`。

不得用 aggregate R29 ACCEPT、truth-safe、接近 geometry 门或 target semantic IoU `0.9946` 覆盖独立 decision failure，也不得放宽阈值。R7/R30 的 actor layer 由 all-actor edit evidence 构成，整体可用不推出任一 identity component 可用。H-R33-001 不再修复 generated layer，而提取 H-R13-011 已接受的 observed-support SceneIR `actor_0000` Gaussian chunk 与 logged trajectory，作为明确标注的 baseline/runtime asset；generated identity route 保持 rejected。

### V6-F63：预注册记录的声明时间不得晚于正式 run，即使 source commit 已先冻结

H-R37-001 首个 formal run `20260821T170543Z__trajectory-edit-s20260821-r1` 的方法、阈值与源代码已在 run 前 commit/push，数值上也使两个 1m actor translations 都通过 compiled/native sensor equivalence；但 `HYPOTHESES.jsonl` 内手填的 `recorded_at_utc=2026-08-21T17:15:00Z` 晚于 run directory 的 `17:05:43Z`。该自相矛盾时间戳破坏了预注册审计的机器可验证顺序，因此该 run 不得作为 canonical acceptance，数值只可用于 failure diagnosis。

不得回写旧记录、追认首 run 或仅凭 Git commit 顺序忽略结构化时间字段。H-R37-002 保持相同代码路径、两个 interventions、thresholds、source denominator 与资源合同，在服务器 `date -u` 实时时钟下追加新的预注册记录并重新 commit/push 后复跑；只有 retry 可成为 R37 canonical authority。

### V6-F64：factor consumer 记录 intervention metadata 时必须从冻结 owner 绑定完整字段

H-R39-001 首个 formal attempt `20260821T172447Z__static-contact-s20260821-r1` 在 static KD-tree 和任何 decision 生成前因 `KeyError: translation_delta_m` 失败，仅产生 failed `TERMINAL.json`。R39 config 为两个 intervention 写了预期 contact decision，但 consumer 在输出 decision row 时还读取 delta；delta 的事实 owner 是冻结 R38 payload/decision，初版 config 没有显式重复绑定。

修复只在 R39 config 中补入与 R38 完全相同的 `[1,0,0]` 与 `[0,1,0]`，不得改动 static query、0.80 coverage、0.90 retention、directional control、资源合同或 source denominator。该失败属于 metadata plumbing，不读取或改变实验结果；H-R39-001 在新 commit/push 后按同一假设重试。

### V6-F65：background Gaussian 密度与 actor AABB 极值不能充当 ground-contact evidence

H-R39-001 canonical retry `20260821T172656Z__static-contact-s20260821-r1` 在 824,583 个 observed background Gaussians 上按冻结的 1.5m horizontal / 0.35m vertical / 3-point contract 查询 actor AABB bottom。logged support coverage 只有 `0.17857`，远低于 `0.80`；x+1m coverage `0.16837` 因绝对覆盖不足而 REJECT，y+1m coverage 反而为 `0.19388`，方向控制失败。run 正式 `rejected`。

不得放宽 coverage、vertical tolerance 或把 retention 单独当 ACCEPT。background splats 混合道路、立面与其他表面，Gaussian AABB minimum 又受少量低端 primitives 支配，两者组合不是 ground-contact 语义。H-R40-001 转向冻结的同前端三相机 logged LiDAR：排除 dynamic pixels 后提升到 world frame，以 actor world-Gaussian y 轴 5% 分位作为 robust support anchor、局部 LiDAR y 轴 10% 分位作为 ground proxy，仅评估 R37 实际执行的 frame57；x/y directional controls 与物理/semantic-road abstention保留。

### V6-F66：独立 runner 必须显式绑定仓库根目录后再导入项目包

H-R41-001 首次正式启动在创建 run directory 或读取任何冻结 artifact 前因 `ModuleNotFoundError: No module named 'motion_proj'` 退出。R41 runner 缺少其他 WorldSim V6 runner 已使用的仓库根目录 `sys.path` 引导；实验主体、预注册 factor decisions 与 fusion contract 均未执行，因而这不是方法拒绝。

修复只把 `scripts/worldsim_v6/run_r41_actor_edit_factor_fusion.py` 的仓库根目录插入 `sys.path`，不得修改 R37/R38/R40 hashes、两个 intervention、四因子 decisions、reject-dominates fusion、资源合同或 claim boundary。修复后必须新 commit/push，并以同一 H-R41-001 重跑；首次启动不得被追认为 canonical run。

### V6-F67：手工绑定 source digest 必须逐字符比对实际 SHA256

H-R41-001 第二次正式启动已进入 source verification，但在创建 run directory 或读取 factor decision rows 前拒绝 R37 `MANIFEST.json`。诊断显示配置值第 38 个字符误抄为 `f`：`...e8f42f5946...`，实际冻结 SHA256 为 `...e8f42c5946...`；两者长度均为 64，因此肉眼概览未能发现单字符漂移。源 artifact 本身未改变。

不得跳过或放宽 `_verify`。修复只把 R37 manifest digest 的错误字符改为实际 `c`，其余 R37/R38/R40 hashes、interventions、factor decisions、fusion contract、资源与 claim boundary 全部冻结不变。新 commit/push 后仍按 H-R41-001 重跑，第二次启动不具 canonical authority。

### V6-F68：负值向量 CLI 参数必须用 `--option=value` 绑定，避免 argparse 将其解释为新选项

H-R43-001 首个 formal run `20260821T175434Z__selected-sensor-s20260821-r1` 已完成全部 source/proposal 绑定并创建 run，但 native worker 在加载 checkpoint 前以 rc=2 退出。日志明确为 `argument --translation-delta-m: expected one argument`：选中 proposal 以负数开头的字符串 `-1.0,0.0,-0.5` 被 argparse 解释成新的 option；此前 R37 的正值向量没有暴露这个入口问题。run 只有 failed `TERMINAL.json` 与 worker log，没有 sensor 或 gate。

修复仅把调用形式从两个 argv token `--translation-delta-m`, `<negative-vector>` 改为单 token `--translation-delta-m=<negative-vector>`。不得修改 R42 proposal、translation、renderer worker、任何 sensor threshold、GPU 预算或 claim boundary。该错误不否定 H-R43-001；必须在新 commit/push 后按原假设重试，首 run 永不追认为 canonical。

### V6-F69：verified translation 不应通过破坏性 float32 world-means 重写来拥有 trajectory edit

H-R44-001 canonical run `20260821T180210Z__verified-bake-s20260821-r1` 成功生成自包含 68MB package，所有非 translation actor fields byte-exact、shifted means content-addressed、manifest 完整、双次 bake byte-exact，且 typed validity/abstention 全部保留。但把 `[-1,0,-0.5]` 直接加到原始 float32 world means 后，反算 translation 的最大误差为 `1.9073486328125e-6m`，超过预注册 `1e-6m`，因此 run 正式 `rejected`。

不得把阈值放宽到 2e-6，也不得用舍入后的数组冒充精确 trajectory ownership。H-R45-001 改变表示机制：R35 的全部 actor arrays（包括 base world means）原样 byte-exact 保存，proposal translation 由独立 content-addressed float64 `T_delta_world` trajectory 拥有；runtime 明确按齐次变换组合 base world means。这样 edit 是显式、持久、可验证的，又不迫使高精度 transform 被吸收到 float32 geometry。R44 rejected package 仅保留为失败证据，不得供 runtime 使用。

### V6-F70：trajectory event identity 不能要求每个 timestamp 的 state content hash 唯一

H-R46-001 canonical run `20260821T181019Z__detached-logsim-s20260821-r1` 从完整复制的 detached R45 package 独立加载，196 行/每行 12,390 primitives、组合误差 `0`、导数不变误差 `1.42e-14`、两次 replay aggregate SHA256 完全相同，且 source package 在 copy 后未被 loader 使用。但预注册错误要求 196 个 state content hashes 全部不同；实际只有 `142` 个唯一状态。诊断显示唯一重复组覆盖 `14.1s` 到 `19.5s` 共 `55` 个 timestamp，表示同一个 stationary geometry state 被多个合法 trajectory events 引用。run 因此正式 `rejected`。

不得给 state bytes 掺入 timestamp 以伪造不同 state，也不得删除静止尾段。H-R47-001 明确分离两种身份：`materialized_state_sha256` 继续只哈希几何内容、允许并精确报告 142 个唯一状态；`trajectory_event_sha256` 哈希 sequence index、timestamp、visibility、proposal id 与 state hash，必须对196个事件全部唯一。重复 state group 与55次静止尾段必须原样保留，detached replay、组合精度和 abstention 合同不变。

### V6-F71：actor geometry trajectory 必须同时拥有 native lifecycle，不能把 repeated terminal pose 当作 active actor

H-R49-001 canonical run `20260821T182444Z__multiframe-sensor-s20260821-r1` 在 frames `[0,57,140,141,195]` 上把 R47 detached package 与 R35+同一 delta 两条 runtime 路径逐数组比较，5/5 sensor NPZ 完全相同，runtime modes、event/state identity、repeat、state restoration、package/checkpoint immutability 与资源门均通过。但两条 compiled 路径共同遗漏 native `RigidNodes.instances_fv`：frames 141/195 的 native actor 已 inactive、opacity 为零，compiled package 仍使用固定 observed opacity，导致最大 opacity field error `0.99643`、RGB MAE `0.00501`、depth MAE `0.44633m`。因此 run 正式 `rejected`；cross-path equality 只能证明两个 consumer 同错。

不得删除141/195、放宽 sensor 阈值、把 actor effect 为0解释成无关，或继续把 stationary geometry state 等同于 active lifecycle。H-R50-001 从冻结 StreetGS native `instances_fv[:, actor_0000]` 提取完整196帧生命周期，预注册验证 frames0-140 active、141-195 inactive 的单次边界，并把 content-addressed bool lifecycle 作为独立字段 bake 进 transform-owned package。base geometry、proposal transform 与 R49 rejection 必须原样保留；后续 sensor runtime 必须用 lifecycle 乘 actor opacity。
### V6-F72：下游 perception adapter 必须显式绑定 sensor NPZ 的拥有字段

H-R53-001 首次正式启动 `20260821T185332Z__lifecycle-perception-s20260821-r1` 在产生任何感知输出前失败。冻结 R49/R51 sensor NPZ 使用 `native_rgb` 与 `compiled_rgb` 字段，而复用的旧 R13 worker 硬编码读取 `rgb`，因此抛出 `KeyError: rgb is not a file in the archive`；run 仅有输入 index 与错误日志，没有 label、gate 或方法结论。

修复新增 R53 专用隔离 worker，唯一变化是显式读取 `compiled_rgb`；冻结 R52/R49/R51/model hashes、frames57/141/195、双重复、active exact control、inactive label-change gate、资源和 claim boundary 均不变。不得把 `native_rgb` 偷换为输入或先读取标签结果调阈值。新 commit/push 后按原 H-R53-001 重试，首次启动不具 canonical authority。
### V6-F73：冻结 CUDA 感知 worker 必须在进程启动前绑定确定性 CuBLAS workspace

H-R53-001 第二次正式启动 `20260821T185611Z__lifecycle-perception-s20260821-r1` 已正确读取 `compiled_rgb` 并加载冻结 DeepLabV3，但在首个 forward、任何 label 输出前被 `torch.use_deterministic_algorithms(True)` 拒绝：CUDA>=10.2 的 CuBLAS 需要进程启动前设置 `CUBLAS_WORKSPACE_CONFIG=:4096:8`。run 仍只有输入 index 与错误日志，没有方法结果。

修复仅由 R53 主进程向隔离 worker 环境注入 `CUBLAS_WORKSPACE_CONFIG=:4096:8`，保持 deterministic algorithms 开启；不得关闭确定性模式。冻结 sources、模型、12 次推理分母、active/inactive gates、资源与 claim boundary 均不变。新 commit/push 后仍按原 H-R53-001 重试，第二次启动不具 canonical authority。
### V6-F74：跨 actor 复用 renderer-conformant translation 不保证全轨迹 interaction 可接受

H-R59-001 canonical run `20260821T192557Z__actor2-interaction-s20260821-r1` 将 R58 已通过 native renderer 的 `[-1,0,-0.5]m` translation 应用于 actor2 的完整196帧轨迹。self-kinematics 精确保持，最大 velocity/acceleration invariance error 仅 `8.88e-15/1.78e-13`；但相对 logged baseline 新增7个 AABB overlap events：actor0 在5.3--5.6s共4个，actor5 在6.2--6.4s共3个。因此 interaction factor 正式 `REJECT`，renderer conformance 不能提升为 edit validity。

不得删除发生 overlap 的帧、放宽 AABB gate、用删除4个旧 overlap 抵消新增7个，或因为 R58 sensor 通过就覆盖 R59。H-R60-001 保留 R58/R59 与完整27 actor x196帧 denominator，冻结 x/z 各 `[-2,-1.5,-1,-0.5,0,0.5,1,1.5,2]m` 的80个非零 translation 网格，逐候选要求 self-kinematics ACCEPT 且新增 overlap 为0；按与被拒候选的距离、再按字典序选择最近可接受方案。contact、road、physics 与 safety 继续 ABSTAIN。

### V6-F75：StreetGS 数据 support 提取必须由拥有完整前端依赖的冻结环境执行

H-R62-001 首次正式启动 `20260821T194222Z__actor2-lidar-contact-s20260821-r1` 在读取任何 frame98 support 或产生 contact decision 前失败，`TERMINAL.json` SHA256 为 `592ba630ce84d996ff478b7314a12bfe1d5e0aedb0762bc7270e99ceaa7565d7`。主实验从通用 `motionproj` 环境直接导入冻结 StreetGS `DrivingDataset`，其模型依赖链要求 `pytorch3d`，该环境未安装，因此抛出 `ModuleNotFoundError: No module named 'pytorch3d'`。这属于环境 ownership plumbing，不构成 contact 方法结果。

修复新增隔离的 LiDAR support worker，并用配置中冻结的 DriveStudio Python 环境执行两次 frame98 三相机提取；主实验只读取两个 worker artifact、核验逐数组 repeat-exact 后运行原 contact evaluator。不得改变 R60 proposal、R61/R56/R40 authority、frame98、13,490 primitive denominator、动态像素排除、R40 quantile/radius/0.35m 阈值、预期 ACCEPT 方向、资源上限或 claim boundary。新 commit/push 后按原 H-R62-001 重试，首次启动不具 canonical authority。

### V6-F76：隔离前端 worker 必须同时绑定项目包根目录

H-R62-001 第二次正式启动 `20260821T194557Z__actor2-lidar-contact-s20260821-r1` 已切换到拥有 `pytorch3d` 的冻结 DriveStudio Python，但仍在读取 frame98 前以 worker rc=1 失败，`TERMINAL.json` SHA256 为 `1dcea9a007b18df26c4ff420fd15e8e759a1b814a5ad09184a3a4d22001420b0`。独立复现显示冻结 `DrivingDataset` 还导入项目内 `motion_proj.worldsim_v3.drivestudio_compat`，而新 worker 只加入 checkpoint backup 与 upstream 路径，遗漏当前仓库根目录，触发 `ModuleNotFoundError: No module named 'motion_proj'`。

修复只给 worker 增加显式 `--repo-root` 并在导入冻结 dataset 前插入 `sys.path`，同时令主进程在检查 rc 前落盘 worker stderr。不得改变 Python 环境、数据配置、proposal、frame、接触协议、阈值、预期方向或任何方法分母。新 commit/push 后仍按原 H-R62-001 重试；前两次启动都不具 canonical authority。

### V6-F77：单帧投影 LiDAR 无局部点时不得把 contact 缺证据解释为 edit invalid

H-R62-001 canonical run `20260821T194813Z__actor2-lidar-contact-s20260821-r1` 在 frame98 从三相机重复提取出完全一致的 `7,183` 个静态 logged-LiDAR world points，actor2 的13,490 primitives、生命周期、R60 proposal 与全部 authority 均精确绑定。但冻结2m局部查询在 logged 与 `[-1,0,0]m` 编辑中心附近都得到0个候选，最近水平距离分别为 `4.1758m` 与 `3.9666m`，因此两者 contact error 都不可计算并正式 `REJECT`。这证明的是单帧稀疏 support 不足，不是编辑破坏地面接触。

不得放宽2m半径、降低32点分母、增大0.35m误差阈值、删除 logged baseline 控制，或把两个 REJECT 宣称为编辑无效。H-R63-001 固定使用 target frame98 前后各10帧的对称21帧窗口 `[88,108]`，逐帧排除 dynamic pixels、提升到同一 world frame，并沿用已接受 R13 的0.05m deterministic voxel union；随后完全复用 R40 的 quantile/radius/denominator/error 阈值评价同一个 logged/selected pair。semantic road、physics、planning 与 safety 继续 ABSTAIN。

### V6-F78：相机投影 LiDAR 子集的时间融合仍可能无法覆盖远距 actor 接触邻域

H-R63-001 canonical run `20260821T195625Z__temporal-lidar-contact-s20260821-r1` 精确保留 R62 frame98 support，并从冻结21帧三相机数据得到 `149,723` 个 raw projected static observations、0.05m 去重后 `25,798` 个 world points；worker 双提取、坐标、窗口和 source gates 全部通过。但 logged 与 selected actor2 的2m邻域仍各为0点，两个 contact 均正式 `REJECT`。actor2 是 `vehicle.car` 且88--108帧间移动约 `1.97m`，因此失败说明相机投影稀疏子集在远距/遮挡区域不能承担 contact map，而不是简单增加同类帧数即可修复。

不得扩大投影时间窗、放宽 contact gates 或用环带高度挑选靠近 actor anchor 的平面。H-R64-001 改用同一 processed scene 的21帧360度 raw LiDAR，按每帧全部标注3D box加0.10m固定边界排除动态点，再做0.05m world voxel union；45个 raw/pose/box 输入以预先计算的聚合 SHA256 冻结。contact evaluator、logged baseline、selected proposal与全部阈值保持不变，semantic road、physics、planning与safety继续ABSTAIN。

### V6-F79：中心圆查询与 Gaussian 低分位 anchor 不适用于大尺寸 actor 的 box-filtered contact

H-R64-001 canonical run `20260821T200653Z__raw-lidar-contact-s20260821-r1` 从21帧360度 LiDAR 获得 `729,568` 点，按全部标注 box 排除 `95,432` 个动态点并形成 `73,010` 个静态5cm voxels；source、变换、动态过滤与资源门均通过。但 actor2 logged 中心查询虽有37点，Gaussian y-5% anchor 与 ground proxy 相差 `2.415m`；selected 只有18点且误差 `0.934m`，两者正式 `REJECT`。审计发现 model actor2 唯一对应 processed instance7 `vehicle.truck`，其 local +z 映射到 world -y，所以 Gaussian y-5% 是上表面而非接触底面；同时12.153m长的 truck 在 box-filter 后中心区域本就形成观测空洞。

不得继续把 R40 针对 actor0 偶然通过的中心/低分位定义当作跨 actor ground owner，也不得降低32点门。H-R65-001 显式绑定最近且有大间隔的 instance7 box，使用标注 local -z 底面拥有 contact anchor，并在 oriented footprint 外固定1m边界环查询 raw static voxels；环内用 median world-y 抑制不同高度表面，要求每个 intervention 至少64点、误差仍不超过0.35m。box只拥有几何位置，不提供 ground 高度；logged 与 selected 均须独立通过。

### V6-F80：actor package schema 分母必须把独立 lifecycle 计入 base arrays

H-R67-001 canonical run `20260821T202013Z__actor2-transform-bake-s20260821-r1` 成功产生 transform-owned package：float64 composition error 为0、双 bake byte-exact、196 transforms、13,490 primitives、四因子与 abstention 均正确，且所有源数组 hash 实际逐项相等。唯一失败是预注册把“7个 Gaussian/trajectory arrays”误写成 package 的完整 base array count=7；R56 还拥有独立 `actor_frame_validity`，实际键为8个，因此 `all_seven_base_arrays_byte_exact` 按合同正式失败。

不得删除 lifecycle、追认 H-R67-001、忽略数量门或改变任何 package bytes。H-R67-002 保持相同 R66/R56、proposal、transform、composition、repeat、资源与 claim boundary，只把 schema 分母明确为8，并同时要求8个 hash 全等且 `actor_frame_validity` 必须存在；新 commit/push 后重跑，首 run 保持 rejected authority。

### V6-F81：本地 shell 不得展开远端 Git 预检的命令替换

H-R68-001 第一次启动尝试在创建 run directory、读取任何冻结 artifact 或启动 GPU worker 前退出。PowerShell 在 SSH 到达服务器前展开了双引号字符串中的 `$(git status --porcelain)` 与 `$(git rev-parse ...)`，本地当前目录不是 Git 仓库，继而使传给远端 bash 的引号不闭合。服务器只收到语法错误；没有 R68 run、sensor、gate 或方法结果产生。

修复只把 SSH 的远端命令改为 PowerShell 单引号字面量，并把可执行 runner 的物理文件模式同步为已提交的 `100755`；不改 R68 代码、配置、冻结 source、frame98、actor2、transform/lifecycle ownership、sensor exactness、阈值、资源或 claim boundary。H-R68-001 保持 active，在 clean 且已 push 的同一 source commit 后重新启动。该失败是 launcher plumbing，不构成假设 rejection。

### V6-F82：multi-actor sensor 证据帧必须让每个被编辑 actor 对相机具有独立可见支持

H-R71-001 canonical run `20260821T205641Z__two-actor-sensor-s20260821-r1` 在 frame98 正确加载 R70，向 native checkpoint 同时应用 actor0/actor2 两个 transform，并在 compiled path 替换两组 owned fields。两组 field error、共享 RGB/depth/opacity、repeat、state restore、package/checkpoint immutability、资源和全部 abstention 均通过；但 actor0 虽 lifecycle active 且12,390个 opacity primitives 非零，其 camera effect pixels 为0。联合 sensor SHA256 因而与 R61/R68 actor2-only sensor 完全相同，只有 actor2 的19,785 effect pixels，按预注册的双 actor 可见门正式 rejected。

不得删除 `both_actors_have_visible_effect`、把 active primitives 当作 camera evidence，或追认 frame98 为双 actor runtime 成功。冻结 R51 证据表明 actor0 在采样帧中只有 frame57 具有17,568 effect pixels；冻结 R57 表明 actor2 在相邻 frame49 和后续 frame98 分别具有17,290/19,700 effect pixels。H-R71-002 因此在任何新渲染前固定 frame57，并改用冻结 R36 frame57 logged sensor 作 counterfactual baseline；所有 runtime、field/sensor、每 actor>=32 pixels、joint>=256 pixels、资源和 abstention 门保持不变。若 actor2 在 frame57 仍不可见，则保留第二次 rejection 并转向独立的可见交集搜索，不得继续猜帧。

### V6-F83：手工复制 source SHA256 时不得遗漏重复的相邻字节组

H-R82-001 第一次正式启动在创建 run directory、复制 package 或开始任何 bake 前，被冻结输入校验拒绝。R82 配置把 R70 `MANIFEST.json` 的实际 SHA256 `1583baf70c760ab700992ef9573ceb6fe59f992527445ac5eb5eb99f7795e6fe` 误抄为少一个 `5e` 字节组的 `1583baf70c760ab700992ef9573ceb6fe59f992527445ac5eb99f7795e6fe`；实际 artifact 与 R71 已使用的冻结 authority 均未变化。本次没有 run、gate、package 或方法结果。

不得跳过 `_verify`、重新生成 R70 artifact、放宽 package denominator 或追认本次启动。H-R82-002 仅修正该 source digest，并保持 R70/R80/R81、三个 actor、45 payload、34,257 primitives、588 trajectory rows、双 bake exact、资源和 claim boundary 全部不变；必须在新 commit/push 后重试。

