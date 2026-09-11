# 历史原始记录 073

> 冻结历史，不是当前执行规则。当前规则见 [scaling law](../../../auto-research_scaling_law.md) 与 [AGENTS](../../../AGENTS.md)。
> 原文件：`docs/RESEARCH_FAILURES.md`；迁移前提交 `abbd431c`。原有相对链接按原目录解释，证据路径原样保留。
> 按索引读取相关小节；旧哈希、过度门控和资源指令均不重新生效。

### PIVOT-F30：原子发布目录不能把 `.partial` 绝对路径写进 manifest

S1 prompt preparer 首次在 `s1_prompt_v1.partial` 内生成绝对 `video_dir`，发布时将目录改名为
`s1_prompt_v1`，manifest 内部却仍指向已经不存在的 `.partial` 路径。SAM2 因此把它判定为既非 MP4 也非 JPEG
目录，r1 在真实 GPU 启动后立即失败。修复是 manifest 只保存相对 `sam_inputs/...`，消费者相对 manifest
父目录解析；原子 rename 后重新验证每个目录和 JPEG 链接。任何会整体 rename 的 staged asset 都不得在内部保存
staging 绝对路径。

### PIVOT-F31：SAM2 `reverse=True` 默认从最早 prompt 开始，可能合法地产生零帧

首次双向传播实现只设置 `reverse=True`，但官方 predictor 默认 `start_frame_idx=min(condition frames)`；train-only
block 的首个 prompt 常在 local frame 0，反向 processing order 因此为空。调用成功和进度条 `0it` 不能证明反向
覆盖。正确做法是显式用 block 内最晚 prompt 作为 reverse start，并按每个 object 自己的 prompt frame 过滤输出；
r5 中实际产生 13 个 prompt 之前的 mask，才构成双向证据。

### PIVOT-F32：mask QC 必须在同一像素坐标系比较

r4 将 SAM logits 从源图 `1600×900` resize 到模型原生 `800×450`，却直接与源图坐标的逐帧 3D box 比较，
造成 `235/263` 假拒绝。修复后 box 按 exact x/y 比例映射到 800×450，r5 为 `212 accepted / 51
fail-closed`，其中 43 个是近空 mask。以后任何 IoU、centroid、boundary 或 area ratio 门禁都必须同时记录 source
size、target size 与变换；不同尺度间的数值不得直接进入裁决。

### PIVOT-F33：大规模 Gaussian 重复索引累加不得使用逐元素 `np.add.at`

S1 r2 在每个视图的数百万 ray/Gaussian intersections 上多次使用 `np.add.at`，CPU 单核成为瓶颈；同时该版本
仍缺计划要求的 negative views、depth-consistency rate 和 boundary score，因此保留 250 个 mask 后以 exit 143
终止，不得作为完成证据。r5 改用 `np.bincount(minlength=total)` 和向量化 view count，263-view lift wall
`770.733s`，并保存完整 posterior schema。研究 runner 必须输出阶段进度；“CPU 持续运行”不能替代复杂度审计。

### V6-F06：数据 adapter 的运行环境必须覆盖写出阶段依赖

R3 首次正式目录
`20260821T085802Z__support-deviation-s20260821-r1` 在完成 scene-0242 图像、标定与点云聚合后，
于 `store_ply` 写出阶段因主环境缺少 `plyfile` 失败。该 run 保留 `failed` terminal，不改写为完成；
失败发生在任何 checkpoint 推理、质量读取或确认集读取之前，因此不是方法负结果，也不是 GPU/数据资源不足。
修复只把冻结的 adapter 命令路由到已经具备 AD-GS 依赖的 `/root/autodl-tmp/envs/adgs/bin/python`，
不改变数据分区、场景、checkpoint、support 假设、指标或门槛。以后环境 readiness 必须覆盖 adapter 的最终序列化依赖，
不能以脚本启动和主体循环成功代替端到端环境兼容性。

### V6-F07：只读 renderer 的数据 loader 仍可能强制读取训练期辅助字段

R3 第二次正式目录
`20260821T090109Z__support-deviation-s20260821-r1` 已完成 scene-0242 adapter 和 StreetGS 全部冻结渲染，
随后 AD-GS `Scene` 构造因 adapter 没有 `depth/000000.npy` 失败。V4 adapter 按设计只生成图像、语义、天空、
位姿与点云，而 AD-GS nuScenes loader 即使在只读 checkpoint 渲染时仍无条件加载每张训练期 depth；源码检索确认
`gaussian_renderer` 不读取 `viewpoint_camera.depth`，R3 指标也只使用 DriveStudio 导出的真实稀疏 LiDAR。
因此该 run 保留 `failed` terminal，不改写为方法负结果。

修复是在每个不可变新 run 的 adapter 内生成全零 float32 depth 占位文件，并写出独立 audit，明确标注
`loader_field_only=true`、renderer/指标不消费以及真实几何证据来源；不修改 AD-GS checkout、checkpoint、
开发/确认分区、指标或假设。以后复用训练代码做只读渲染时，必须区分 loader 的强制 schema 字段与实际计算依赖，
占位值只能用于经源码证明不被实验结果消费的字段。

### V6-F08：通用场景点云不能替代 checkpoint 对应的 object-aware loader 资产

R3 第三次正式目录
`20260821T090552Z__support-deviation-s20260821-r1` 已再次完成 scene-0242 adapter 与 StreetGS 渲染，
AD-GS 随后在 `readnuScenesInfo` 对 `obj_id[..., 0]` 索引时失败。通用 V4 adapter 生成的 `points3d.ply`
只有 xyz/rgb/time，没有 AD-GS 训练 adapter 的 `obj` property；即使 checkpoint 加载随后会覆盖 Gaussian 初始化，
`Scene` loader 仍先强制构造 object-aware point cloud。该 run 保留 `failed`，不是方法或资源负结果。

修复不伪造 object id，而是把同一场景、同一冻结 checkpoint 训练时使用的
`adgs_processed_v4/train/<scene>/points3d.ply` 复制进新 development adapter；绑定前验证 PLY header 的
`property float obj`，记录通用点云 hash、冻结训练点云路径/hash 和复制后 hash。开发图像、位姿与分区继续来自
新 adapter，checkpoint 与指标不变。以后给冻结模型换 evaluation camera 集时，应复用训练时与模型结构耦合的
初始化/registry 资产，只替换经协议允许的观测与相机字段。

### V6-F09：lazy camera 偏移前必须整体迁移设备，不能只新建 CUDA 外参

R3 第四次正式目录已成功完成 scene-0242 的 object-aware `Scene` 构造与 checkpoint restore，
在首个 novel camera 的 `full_proj_transform` 计算处失败：冻结 AD-GS 配置启用 `lazy_load_to_gpu`，
原 camera 的 `projection_matrix` 留在 CPU，而 R3 worker 直接把新 `world_view_transform` 建在 CUDA，导致 BMM
设备不一致。该 run 保留 `failed`，资源峰值远低于门槛，不属于方法或资源负结果。

修复在任何偏移或编辑前调用上游 `Camera.cuda()`，一次性迁移 image/depth/semantic/sky 与全部变换矩阵，
再深拷贝和修改外参；这与上游 `Camera.to()` 合同一致，不改变相机数值、checkpoint、renderer、指标或门槛。
以后 lazy evaluation path 必须把 camera 作为一个设备一致的整体处理，不能只迁移新创建的 tensor。

### V6-F10：structured array 拼接后必须使用字段索引，不能依赖 recarray 属性

R3 第五次正式目录
`20260821T091503Z__support-deviation-s20260821-r1` 已完成 2 场景 × 2 frontend 的全部 adapter、
checkpoint restore、横向/前向/actor-edit 渲染和 worker audit，共 80 个 render；汇总阶段把 `np.rec.fromarrays`
结果经 `np.concatenate` 拼接后得到 structured `ndarray`，代码仍以 `values.y/values.x` 访问字段，触发
`AttributeError`。该 run 的 terminal 保留 `failed`，不得倒写为 done；渲染与 checkpoint 证据本身完整。

修复统一改用 `values["y"]` 等 structured-array 字段索引。为避免无意义重跑冻结 renderer，建立独立
analysis-only recovery run：先逐个重算全部 render SHA、核对每个 worker 的 render count、checkpoint 前后 hash、
无训练/无确认集 audit 与 adapter 分区，再从只读失败目录计算指标；新目录记录原 run/commit/terminal hash、
分析 commit 与聚合 content hash，原目录不修改。以后 post-render 工程失败可复用已验证的不可变证据，
但必须新建 terminal 和完整 provenance，不能在原失败目录续写。

### V6-F11：Gaussian `source_indices` 是 chunk-local 身份，不能跨模型直接判全局唯一

R5 v0 正式目录 `20260821T094101Z__provenance-s20260821-r1` 已生成 provenance package，并达到
chunk=`24/24`、actor=`23/23`、primitive=`1,267,870/1,267,870` 覆盖；但 raw `source_indices` 的全局 unique
count 只有 `1,095,606`，因此 100% identity gate fail-closed。原因是 StreetGS Background 与各 Rigid model
分别维护局部 source-index 空间，actor 数值可与 Background 重叠；这不表示 primitive 丢失。

v0 run 与 config 保持 failed/frozen。v1 不放宽全局唯一门，而把 primitive identity 改为
`(chunk_id, source_index)` 复合键：先要求每个 chunk 内 source index 唯一，再要求 chunk id 唯一，二者合取形成
全局唯一身份。provenance 字段、source-type 分离、覆盖率和无 confirmation/训练边界均不变。以后任何跨 chunk
primitive registry 都必须显式携带命名空间，不能把上游局部索引误当全局主键。

### V6-F12：R7 source manifest 必须绑定实测完整 SHA，不能使用手工摘录值

R7 首次正式入口在创建 run 目录与读取 render payload 前 fail-closed：预注册配置中的 R3 recovery manifest
SHA 使用了手工摘录值 `e866dae3b84c...`，而冻结文件实测 SHA 为
`e866dae35a4ff17fb75791ff395f45504f8f779d57e75121354b8be388595acc`。两者不一致，因此程序按 source
identity contract 立即拒绝；没有 pseudo-hole、proposal、质量指标、训练或 confirmation 读取，也没有可写成
`rejected` 的方法结果。

修复只替换为 `sha256sum` 实测的完整 source manifest SHA；R7 hypothesis、cohort、hole 定义、verifier 阈值、
decoy、gate 与资源合同均不改变。以后冻结跨 run source identity 时，必须从机器可读 artifact 或现场哈希复制完整值，
不得从状态文档里的短写或人工记忆还原。

### V6-F13：R7 verifier 必须显式归一 frontend 的 singleton-channel 维度

R7 首个有 run 目录的正式实例
`20260821T100107Z__oracle-missing-world-s20260821-r1` 在第一个 pseudo-hole mask 构造时失败：StreetGS/AD-GS
冻结渲染对 depth/dynamic opacity 保留了 `H×W` 与 `H×W×1` 两种合法 singleton-channel 表示，初版代码无条件
取 `[...,0]`，把二维 opacity 错切成长度 `H` 的向量，随后与 `H×W` mask 广播失败。该目录 terminal 保持
`failed`；尚未形成完整 denominator、gate 或方法结论。

修复新增唯一的 plane normalization：只接受 `H×W`、`H×W×1` 或 `1×H×W`，统一返回二维数组，其他形状
继续 fail-closed；所有 depth/semantic verifier 和 usable-region 路径共同使用它。hypothesis、pseudo-hole、decoy、
阈值和 source render 均不改变。

### V6-F14：actor/disocclusion pseudo-hole 不能假设每个 frontend 都导出非空 dynamic opacity

R7 第二个有 run 目录的正式实例
`20260821T100228Z__oracle-missing-world-s20260821-r1` 已完成 scene-0242 与部分 scene-0048 cases，随后在
scene-0048/StreetGS 的 disocclusion mask 上 fail-closed：该冻结 renderer 的 `dynamic_opacity` 在此帧为空，初版
mask 得到 0 pixels，低于预注册 `256` denominator。该目录保持 `failed`，不汇报不完整 gate。

修复不降低 minimum pixels，也不读取 confirmation，而是使用本实验本来就冻结的 `base` 与
`actor_remove_all` 配对渲染：以 RGB 变化或同 frontend 可比 depth 变化，加上可用的 dynamic opacity，形成 actor
evidence；disocclusion 对其确定性膨胀，actor-removal hole 使用原 evidence。这样 StreetGS/AD-GS 都以“实际 actor
edit effect”而不是可选 buffer 的存在作为 denominator，其他 hole、verifier、decoy 与 gate 不变。

### V6-F15：actor pseudo-hole denominator 必须先满足非空 actor-effect 证据

R7 第三个失败目录 `20260821T100348Z__oracle-missing-world-s20260821-r1` 在使用 RGB/depth actor-edit evidence 后，
scene-0048/StreetGS/frame-52 的 disocclusion mask 仍严格为 0。R3 冻结 `ACTOR_EDIT_EFFECTS.jsonl` 证实该 frontend
在 scene-0048 的 frame 52/57 上，`actor_remove_all`、translate、time-shift 的 global effect 与 nonzero fraction
全部为 0；因此这四个 actor-removal/disocclusion Cartesian cases 没有可构造的真实 pseudo-hole denominator。

这不是 oracle verifier 的负结果。预注册 `WS-V6-H-R7-001` 因 32-case minimum experiment 无法实例化而标记
`invalidated_pre_gate`，不倒写 gate。替代假设 `WS-V6-H-R7-002` 在 proposal 评分前冻结 eligibility：route/side
仍保留全部 16 cases；actor/disocclusion 要求原冻结 evidence 至少 256 pixels，不合格的四项显式记录 structural
ABSTAIN；剩余 28 oracle + 28 decoy 才进入完全相同的 verifier/bake 门禁。

### V6-F16：冻结 LaMa 配置必须用 OmegaConf 解析字段引用

R8 首轮正式目录 `20260821T103805Z__frozen-generator-s20260821-r1` 中，Big-LaMa 在构造
FFC generator 时失败。官方 `config.yaml` 的 `downsample_conv_kwargs` 与
`resnet_conv_kwargs` 使用 `${generator...}` 字段引用；直接用 PyYAML 读取会把引用保留为
字符串，最终在通道比例转整数时触发 `ValueError`。该候选被如实记为 `failed`，首轮 gate
保持 `rejected`，不改写为方法结论。

修复只把 generator 子树改为 `OmegaConf.load` 后 `resolve=True`，仍使用同一官方配置、源码、
checkpoint、输入、seed、阈值与选择规则；不解析或消费任何训练/确认数据。以后复用带插值的
冻结配置时，必须在进入模型构造前解析并审计最终标量，不能把 YAML 语法读取成功当成配置已实例化。

### V6-F17：扩散 inpaint 输出尺寸必须显式绑定冻结输入尺寸

同一 R8 首轮中，SD-v1.5 pipeline 虽完成模型加载和推理，但未显式传入 `height/width`，输出采用
默认 `512×512`，与冻结输入及 mask 的 `512×288` 不一致，overlay 布尔索引因此触发
`IndexError`。该候选同样保留 `failed`，不是 GPU、权重或方法负结论。

修复是在冻结 pipeline 调用中显式设置 `height=image.shape[0]`、`width=image.shape[1]`；不 resize
实验输入、不改变 mask、prompt、steps、guidance、seed、候选或门槛。后续所有生成器 adapter
必须把空间尺寸当作调用合同，并在 compositing 前 fail-closed 核对 image/mask/proposal 三者形状。

