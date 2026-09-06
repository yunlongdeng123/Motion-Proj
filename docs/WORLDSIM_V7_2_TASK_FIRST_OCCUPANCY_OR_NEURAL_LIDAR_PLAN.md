# WorldSim V7.2：以任务与外部竞争为起点的研究计划

## 对象级占据／三维补全，或神经 LiDAR 仿真：先比较，再选主线

**计划载体：Markdown。研究交付：至少一份可编译的 CVPR 格式 LaTeX 主稿、PDF 和补充材料。**

**依据版本：** 本轮上传的 `Motion-Proj-research-worldsim-v7.1-learned-evidential-surface (2).zip`。用户报告对应 V7.1 提交 `1913ab0e`；ZIP 本身不含可验证的完整 Git 历史，因此启动时只确认一次实际父分支与提交，不借用其他版本结果。

**工作分支建议：** `research/worldsim-v7.2-task-first-completion-lidar`。

**目标：** CVPR／ICCV main track；先形成可信技术报告，再发布 arXiv，后提交会议。本文不承诺录用概率。

**研究性质：** 视觉、三维几何、占据或传感器生成研究。RL 不作为主贡献，不作为本版本的阻断项。

---

# 0. 当前 0 号原则：不再默认 Motion-Proj 的原主线是正确答案

V7.2 不是给 M39 继续加一个头，也不是默认把 M43 解释为“缺少传感器不变性”。

必须先检验：

> **在公平输入、统一算子和相近成本下，现有成熟补全／重建方法是否也存在有意义的“几何完整性—观测一致性”缺口？我们的方法是否比简单替代方案更有价值？**

允许出现三种合法结果：

1. 缺口普遍存在，V7.2 能提供稳定的新方法增量，继续主线。
2. 缺口主要来自自家候选点、数据生产或读出设计，改用更强的外部基座，不为保留旧架构而保留它。
3. 简单基线已经解决问题，或我们的创新没有增量，停止这项方法主张，交付负结果／测量改进报告，不无限启动新模型。

**允许重做任务定位，不允许重写旧实验结论。** M43 的同算子分类回波比较仍然失败；M8 的已暴露开发结果仍不是独立源域测试。

本计划只冻结研究顺序、数据角色和科学语义；不提前钦定最终网络、唯一数学形式或两条路线的胜者。

---

# 1. 从最新代码得到的起点，而不是从旧总结推断

配套文件 `V72_CURRENT_CODE_AUDIT.md` 保存了关键原代码和精确行号。以下是本次直接核对的事实。

| 项目 | 最新快照事实 | 对 V7.2 的影响 |
|---|---|---|
| M43 主比较 | M39 与单位权重 categorical 能量使用同一读出；all/hazard early 增加，hit 增加，冻结判定 1/3 | 保留跨域拒绝；不能在 M43 上调权重、阈值或归一化来救结果 |
| M43 几何描述 | `run_worldsim_v71_m43_m39_av2_zero_shot.py:252–278` 先生成 literal 几何计数，随后用 categorical 的 `baseline_early_count`、`baseline_hit_count` 覆盖；`summarize_surface_rows` 又读取这些字段 | “16.56%→50.15% 证明几何迁移失败”的旧解释不可沿用。先修正命名空间和描述性统计；不改变 M39 主 verdict |
| Chamfer 差值 | 是同一 AV2 样本上输出与基线的差值，不是两个数据域的误差漂移 | 不能说“几何完美泛化”，也不能单凭小 CD 差值确定是雷达线束引起失败 |
| M8 数据角色 | M8 加载 M7/M5，按 Actor 序号 stride 划分，显式写有 `pretrained_holdout_exposure=True` | 正式主表需新的独立日志级源域测试；新模型最好从可追溯的干净训练链重新训练 |
| 当前回波输出 | M39 在已知 Actor 框交区间内 softmax，并取插值中位数；只处理目标返回点提供的方向 | 这是“框内有回波”的条件测距，不是全扫描、无回波或全场景遮挡模拟 |
| 原训练 forward | `first_return_renderer.py` 的软深度近似用目标深度加 margin 确定采样终点 | 可保留为旧 train-only surrogate；新部署 forward 不得读待预测目标深度 |
| 旧 Actor 缓存 | 保存 canonical、anchors、candidate、target/origins 等，但缺真实逐点 frame ID、完整扫描有效性掩码、强度、未返回射线和背景 | 保留旧缓存用于重放；新任务必须建 v2 数据合同，不能把缺字段凭空补成真值 |
| 几何保留 | `evaluate_actor_surface` 的 retention 字段直接写 1.0 | 这是接口约定，不是独立测量。新版本从输入／输出 Actor 状态实际比较；与表面完整性分开 |
| 可复用资产 | 多数据集坐标转换、Actor 身份、规范空间、点生成模块、证据构造、类型隔离、论文工程 | 复用接口与基础设施，不继承未验证的优势主张 |
| 文档状态 | 最新 status 包含 M43，旧 `V71_RESEARCH_CLOSEOUT.md` 仍停在 M0/M1 | 新增简短 V7.1 终态索引与勘误；不重写整个历史账本 |

## 1.1 必须立即修正的旧判断

V7.2 不再默认以下推断成立：

- “M43 证明 Chamfer 泛化而几何首回波不泛化”；旧相关统计混用了算子。
- “观测锚点一定是无误差硬占据”；跨帧矛盾可能来自采样、轨迹／标注误差和遮挡。
- “primitive-union、SDF、体渲染在整个领域已经失败”；旧失败只针对当时的特定监督与参数化。
- “M49 本身就是强理论新颖性”；它是有用的归一化混合恒等式，不能替代方法竞争。
- “最终几何一定必须保留 M8”；M8 在本轮是必须战胜的历史候选之一，不是永远固定的主干。

---

# 2. 两条主路线及共同问题

共同研究问题暂定为：

> **给定稀疏多帧 LiDAR 和提供的刚性对象轨迹，如何恢复可复用的三维对象表示，使其在未参与构建的视角与时刻保持几何质量，并产生与真实传感器观测相符的结果？**

| 维度 | 路线 A：对象级 Occupancy／3D 补全 | 路线 B：Neural LiDAR Simulation |
|---|---|---|
| 首要预测对象 | 对象表面／有掩码的三维占据 | 传感器扫描：返回存在性、距离，条件允许时强度 |
| 模型最终输出 | 点集、网格或可查询场；不能只输出置信度 | 全场景查询可生成扫描，包含背景／Actor 遮挡与无回波处理 |
| 主要竞争者 | AdaPoinTr／PoinTr、对象中心占据补全、TSDF／surfel | DyNFL、LiDAR4D；选为主线后加入 LiDAR-RT 等较新直接方法 |
| 主要指标 | 表面 precision/recall/F-score、CD；有可靠体积标签时 IoU | range MAE/RMSE、ray-drop、点云距离、early–hit；强度单列 |
| 独立验证单位 | 行车日志／场景，Actor 为嵌套单位 | 场景与留出扫描；区分逐场景拟合和共享模型泛化 |
| 最小完整应用 | 完整场景中的前景占据／遮挡重建 | 完整场景 LiDAR 重建与留出视角／时刻合成 |
| 不默认要求 | 语义占据 SOTA、动态预测、闭环 RL | 相机照片级合成、强化学习、任意传感器零样本不变性 |

**不能把路线 A 的 IoU 与路线 B 的 PSNR 或误差直接相加排名。** 两条路线分别相对其最强公平基线判断收益与成本，再选更有独立证据和完整应用潜力的一条。

默认工作分配：前期共用数据与算子，A/B 各保留一个正式机制候选；选路之后约 80% GPU 预算给胜者。败者只保留相关对照，不继续长出几十个变体。

---

# 3. 文献与外部代码：先选直接竞争者，不以新年份代替可比性

本轮已核对官方论文／项目／仓库。引用见文末 [R1]–[R13]。

| 方法 | 定位与已核实信息 | V7.2 用法 |
|---|---|---|
| PoinTr／AdaPoinTr [R1] | 官方仓库含点补全训练、推理与评测；PoinTr 为 ICCV 2021，AdaPoinTr 为 TPAMI 扩展，不要写成 CVPR 2026 | A 的首选强补全基线；允许在当前同输入语料上训练，明确“数据适配版” |
| Object-Centric Occupancy Completion [R2] | NeurIPS 2024；官方代码含对象占据标注、轨迹数据、训练与测试；长序列和隐式解码器，原流程依赖 Waymo／SST／检测跟踪 | A 的最近邻任务基线；给定 GT 轨迹的比较版本须标注改动，不能宣称复现原下游检测表 |
| DyNFL [R3] | CVPR 2024；输入 LiDAR 与被跟踪动态框；显式静态背景与动态对象场；官方仓库说明测试过 RTX 3090 | B 的首选对齐基座；不是从零再写一个自家 global field |
| LiDAR4D [R4] | CVPR 2024；官方代码使用 KITTI-360，包含 range、intensity、ray-drop 及模拟入口 | B 的独立表示对照与可访问备选；逐场景优化成本需交代 |
| LiDAR-RT [R5] | CVPR 2025；官方已发布训练／评测代码，Gaussian ray tracing 与动态场景组合 | B 被选中后应加入的较新直接竞争者；先实际测安装、数据和耗时，不凭摘要数字填主表 |
| EvOcc [R6] | CVPR 2025；证据化占据目标与损失，处理未观测／矛盾 | 借鉴标签语义；不把当前 F/O/U 小头包装成完整 EvOcc 复现 |
| Gau-Occ [R7] | CVPR 2026；LiDAR＋图像的场景语义占据 | 主要用于任务定位／可选外部几何来源；不同输入不得直接公平排名 |
| DynamicVGGT [R8] | CVPR 2026；图像驱动动态点图／4D 重建 | 可选来源迁移或相关工作；不把它作为同信息量点云补全的唯一强基线 |
| SimULi [R9] | 作者预印本提供传感器建模与几何／外观分解思路 | 新近邻近工作核对项；未实际复现前只讨论方法，不写已胜过其结果 |

## 3.1 基线执行底线

- 主路线至少有一个直接外部学习方法和一个强简单基线。
- B 成为最终主路线时，至少完成两种有差别的神经 LiDAR 基线；优先 DyNFL 加 LiDAR-RT，工程不适配时用 LiDAR4D并说明原因。
- 官方默认训练／解析流程先跑通，不能把一个没收敛、单位错误或输入缺失的移植版称为强基线。
- 预训练和从头训练分表；额外 ShapeNet／Waymo 数据作为信息预算记录。
- 相同传感器、构建帧、Actor 先验、点数／分辨率和可用标签；训练时间不同时同时报告“趋于收敛性能”和“等时间预算性能”。
- 不要求把所有列出的模型都跑一遍。路线未被选择前，避免为一个重型基线搭建完整检测、跟踪、语义系统。

---

# 4. 研究决策只保留三次

## D0：问题是否真实、基线是否已解决？

先在开发数据上完成算子勘误、简单权重对照和至少两种几何来源的误差图谱。不得在这里改最终测试集。

结果解释：

- 缺口只在自家管线出现：优先替换管线／基座，关闭“普遍表征缺陷”主张。
- 跨两个不同方法、多个日志仍存在：可作为研究动机，但只主张已测试的范围。
- 简单观测支持率或一个标量头与三态证据相当：不再把 F/O/U 结构本身当核心新贡献。

## D1：A/B 开发竞赛后选主任务

两路都达到各自可运行、同信息量的强基线比较后，按第 11 节规则选路。选路只用开发集，不读最终源域／外域测试。

## D2：最终模型是否得到独立支持？

最终架构、训练配方、提取／读出和主要指标冻结后，读取独立源域与外部／第二公开协议。最终不通过，则收缩主张或关闭该主张，不再把这份测试加入训练后连续寻找下一份“全新确认”。

**不是每训练一个 epoch 就增加一道门。** 训练内验证、早停和学习率日程在开发角色下正常使用；不同层面的工程完成、方法收益、泛化和论文排版分别报告。

---

# 5. 数据角色：独立的是整个训练链，不只是最后一个 head

## 5.1 版本隔离与历史资产

- V7.1 所有已消费的测试、M43 20 logs、66-Actor 开发集合都保留原身份，在 V7.2 中最多作 `legacy_diagnostic`，不叫 fresh final。
- 旧训练数据可用于 V7.2 训练；旧开发数据可用于开发。不得把旧测试的质量用来重新挑“容易”的新测试。
- 旧 M8/M39 checkpoint 是 legacy reference。若其任何上游模型见过新 test 日志，该 checkpoint 不得作为该 test 上的干净“ours”主结果。
- 优先让最终候选和外部强基线使用同一干净训练链重新训练。必须使用外部预训练时列出预训练数据来源；不明来源的 checkpoint 只能作为补充。
- 不把 Source Final 已经被 M2 读取的事实遗忘，不通过新版本号恢复未读身份。

## 5.2 建议规模与可调整边界

以下是数据建设目标，不是对现有可用数量的声称。启动时仅依 metadata 与暴露记录形成实际清单。

| 角色 | 建议规模 | 用途 |
|---|---:|---|
| `legacy_diagnostic` | 现有 V7.1 训练／已消费开发资产 | 算子纠错、复现端点，不产出新泛化主张 |
| `train` | 120–200 个 source scenes，尽量覆盖至少 1,000 个合格刚性轨迹 | 训练 A/B 的共享模型；数量不足先报告，不把短序列切碎冒充独立 Actor |
| `dev` | 20 个日志级组 | 配方、checkpoint、误差诊断；允许开发迭代，但不称无偏评测 |
| `route_select` | 12 个 source 日志级组 | 只用于 A/B 定案和主指标确认；之后不升级成 final |
| `source_test` | 目标 30 个全依赖链未触及日志级组 | 最终源域主表；若不足，公开实际规模及区间，不伪造 |
| `external_test` | 目标 20 个未使用 AV2 Sensor logs，或可授权 Waymo | 零样本补充验证；M43 不复用为最终 |
| `native_lidar_protocol` | B 若入选：遵循选定官方仿真基线的公开切分，并增加多个独立日志 | 直接对照标准 LiDAR 仿真任务 |

先按官方 `log_id/segment_id` 分组，再分配 scenes／Actor／frames。同一连续采集日志中的相邻 scene 不得跨 train/test。

若剩余 nuScenes 独立日志不足，不无限放大数据分母：保留可用源域测试，在另一个有明确协议的公共数据集补独立验证。**数据不足是证据限制，不等于模型科学失败。**

## 5.3 两层划分必须同时存在

**模型层：** train/dev/test 日志隔离。

**场景内部：** 每个 Actor／场景的 `build_sweeps`、训练监督 sweeps 和最终 `query_sweeps` 角色明确。

对路线 A，训练 Actor 的监督来自其留出的真值扫帧；测试 Actor 的 target 从不作为模型输入。

对路线 B，逐场景优化是允许的正式任务设定：测试场景的 build scans 可以参与拟合，但 held-out target scans 不可参与优化／选 checkpoint。论文必须写“逐场景重建的新视角泛化”，不能称为“未见场景零样本”。

## 5.4 不预设旧数据已泄漏，也不默认旧数据干净

通过现有 scene/log 清单、训练路径和上游依赖做一次暴露合并表即可。无须建立哈希系统。

如果来源不确定，标为 `exposure_unknown`：可以用于机制开发，不进入正式独立主表。不要以“从没人工看过图像”替代“模型从没见过输入／标签”。

---

# 6. v2 数据合同：保留真实射线，不能从有回波点云假造全传感器

## 6.1 新的最小记录

```text
ActorBundleV2
  dataset, log_id, scene_id, actor_id, category
  size_lwh_m
  build_frame_ids, build_time_ns
  world_from_sensor[frame]
  world_from_actor[frame]
  build_points_actor_m, point_frame_id, point_sensor_id
  build_ray_origin_actor_m, build_ray_direction_actor
  build_range_m, build_intensity, build_intensity_valid
  sensor_model_ref, beam_or_ring_id, per_point_time_offset_ns
  evidence_free/occupied/unknown, conflict_count, opportunity_count
  build_only_surface, provenance

QueryRayBatch                 # 推理可见，禁止含 target depth
  ray_id, log_id, frame_id, time_ns, sensor_id
  origin, unit_direction, range_min_m, range_max_m
  query_pose, beam_parameters, valid_emission_mask

RayTargets                    # 训练监督／评估专用
  return_valid, return_index, range_m, intensity, intensity_valid
  censoring_or_missing_mask
  object_id_for_evaluation_only

SurfaceTargets                # target-only，不给模型
  surface_points, frame_id, normal_if_valid
  free_space_queries, occupied_queries, known_label_mask
  label_provenance, confidence_or_sensor_tolerance
```

不必逐字段全部另建文件；可以 NPZ＋JSON sidecar，按 Actor／log 分片。关键是角色在代码接口中分开。

## 6.2 必须区分四种情况

1. **FREE：** 有有效射线经过、且在真实返回之前的可证实自由区间。
2. **表面命中：** 传感器实际测得的 endpoint 附近，由固定噪声容差定义。
3. **UNKNOWN：** 未观测、被遮挡、相互矛盾或缺少可靠监督；不是 FREE。
4. **无回波／缺测：** 还需进一步分清“发射但未检测到返回”与“该角度根本未发射、被裁剪、数据丢失”。后者不能当无回波负例。

无回波不表示空间全为空；可能有低反射、量程、透明／半透明及传感器因素。V7.2 不以一个通用 UNKNOWN 标签混合这些语义。

## 6.3 nuScenes／AV2 与 Waymo 的现实区别

AV2 Sensor 官方提供有标注对象和 LiDAR 点的传感器数据；AV2 Lidar 是无对象标注的另一套数据，不得下载错 [R10]。

Waymo 官方提供 LiDAR range images、束倾角和返回通道，适合研究扫描与返回存在性；访问须遵守官方授权 [R11]。

对于只保存返回点的原始文件：

- 可以把返回点和原点换算成已知有效射线，做条件测距和几何评价。
- 可以按公开基线的 range-image 投影构造“投影栅格占用／ray-drop”任务，但必须叫**投影协议下的空 bin**，不能冒充实际硬件无回波真值。
- 若 B 要主张实际 no-return 建模，应选保留发射／有效性协议的数据；无法确认时降低主张，不编造标签。
- 目标方向在实际测得 ray 中由 endpoint 推导本身不等于深度泄漏，但整个推理深度区间必须与 target range 无关。全扫描任务的方向由 sensor firing grid 决定。

## 6.4 Actor-canonical 与观测误差

保留 Actor identity、给定 trajectory、size 的输入合同。原始观测与来源只读。

**“观测只读”不等于“每个观测点必须被当作无限置信的实体表面”。** 重建表面可以在训练期定义的传感器／配准误差容差内估计；不允许为了降低指标事后删除冲突观测。

规范变换使用完整旋转和平移，按时间插值／去畸变需求处理，不仅保存 xyz center。非刚性行人／车轮等暂不宣称已建模。

## 6.5 编译与运行时的确定性边界

编译／训练阶段允许学习式优化、生成式先验和随机增强。交付的对象资产、网络权重、表面抽取规则、射线范围与传感器配置必须冻结。

同一资产、位姿、传感器参数和查询输入应得到同一几何／回波分布，不能在运行时查看目标观测或临时重新拟合。神经场可作为冻结的确定性查询器，不要求为符合“显式”二字把一个成熟渲染器错误简化成点集。

若额外模拟传感器随机噪声，其采样属于独立传感器层：显式传入并记录种子，可重放，不改写物理几何；主定量优先使用固定读出或固定采样协议。未知状态继续保存在资产中，不能因某次未返回而把对象删除。

---

# 7. 统一评价：几何、回波、应用三张表，不能交叉覆盖

## 7.1 新结果结构

```json
{
  "geometry": {
    "operator": "beam_tube_min_v1",
    "baseline": {"early_count": 0, "hit_count": 0},
    "prediction": {"early_count": 0, "hit_count": 0},
    "ray_count": 0,
    "cd_l1_m": null,
    "surface_precision": null,
    "surface_recall": null
  },
  "return": {
    "operator": "registered_sensor_readout_v1",
    "baseline": {"early_count": 0, "hit_count": 0},
    "prediction": {"early_count": 0, "hit_count": 0},
    "valid_emission_count": 0,
    "return_count": 0,
    "no_return_count": 0
  },
  "support": {
    "input_actor_count": 0,
    "output_actor_count": 0,
    "eligible_count": 0,
    "fallback_count": 0,
    "missing_target_count": 0
  },
  "cost": {"train_gpu_hours": null, "latency_ms": null, "peak_vram_gib": null}
}
```

这是格式示意，0 不是实验结果。没有通用 `baseline_early_count` 跨算子复用。

## 7.2 几何指标

默认主表使用：

- 对称 CD-L1，以米／毫米明确报告；不能与外部 CD-L2 数字直接混表。
- 固定阈值 F-score、precision、recall；初始内部对照用已有 0.20 m，正式原生基准同时遵循其阈值。
- 同一预定义目标点数／表面采样数的密度受控结果，以及原生输出规模结果。
- literal beam-tube minimum 的 early/hit/miss，仅作为明确管半径下的几何可见性诊断。
- 有可信体积标签才报告 Occupancy IoU；只有稀疏 endpoint 时，报告“已知区域／表面壳占据 IoU”，不要宣称完整体积 IoU。
- unseen-by-input but observed-by-target 子集单列。完全未被任何目标观测到的背面，不能当有真值补全评测。

Chamfer 的 surface→target 项在 target 稀疏／遮挡时可能惩罚正确背面。主要结论应配合已知可见区 precision、target→surface completeness、真实 ray FREE violation，不依赖一个 CD 数字。

## 7.3 回波指标

- 对所有有效扫描机会报告返回存在性 precision/recall、Brier／NLL、无回波表现。
- 对有效返回的 range MAE/RMSE、hit-band recall、early/late 分开。
- 所有方法使用同一固定目标射线分母；漏报返回不能悄悄从距离误差分母里消失。条件距离结果必须与存在性结果并排。
- 使用 median／mean／mode 的方法按真实读出命名；它们都不自动等价于几何最早交点。
- 若官方通道是 strongest-return，就评价该通道；不可不加说明改成 first-return。
- 强度仅在有真实通道和有效性掩码时评价，不能使用图像亮度替代。

## 7.4 应用指标

- A：完整扫描对应的前景占据／遮挡重建；全场景与 Actor 区域分开，不用大量容易背景稀释对象失败。
- B：全 range image、完整点云、动态／静态区域及跨对象遮挡一致性。
- 碰撞只是可选扩展，必须有独立碰撞几何真值；标注 cuboid 不等于真实接触表面。

## 7.5 统计与可比性

主结果给绝对值、差值、按日志聚类的 95% 区间和逐日志散点。Actor 是日志内的嵌套单位，百万射线不能当百万独立样本。

固定输出密度分析、稀疏度条件和读出工作点集合由训练／开发阶段确定。最终画完整预定义权衡曲线是分析，不在测试曲线上挑一个“最佳点”作为主结果。

对学习方法的最终主配置与最强学习基线，可使用预先固定的 3 个种子估计训练波动，全部汇报，不择优。这不是 seed sweep。预算不足只能声称单次结果，不能假装已经稳定。

---

# 8. 阶段 P0：半天到一天完成最小基础修正，直接进入研究

## 8.1 要做

1. 在 V7.1 已完成分支新开 V7.2；保留 V7.1 原结果、旧代码和主稿。
2. 创建 `docs/WORLDSIM_V7_2_TASK_FIRST_PLAN.md`、一个简短结果表和数据角色清单。
3. 修正 M43 的字段命名污染，并生成 `M43_OPERATOR_ERRATUM.md`。
4. 如果保存的 partial rows 已丢失原 literal baseline 计数，只重算已消费 20 logs 的同算子 baseline；不训练、不调模型，不改变 M39 的原最终判定。
5. 原始数据当前不可用时，立即把旧几何迁移结论标为 invalid comparison，重算作为非阻塞待办；不要为了勘误占用几天研究时间。
6. 将 Actor retention 从硬编码常数改成一次真实的 ID／轨迹／尺寸比较。
7. 建立新 v2 缓存，不修改旧 v1 文件。

## 8.2 仅四类必要的微型语义检查

- 一个 early surface、一个 target hit，验证 literal 与 categorical 分别存储、互不覆盖。
- 无有效输入／无返回／背景挡住 Actor 的三种状态不得混为 FREE。
- 推理查询可在没有 target depth 的情况下运行。
- train 与 deploy 使用同一输出实现：不出现训练全保留、部署硬 UNKNOWN 删除，或采样范围偷偷读 GT。

这四类各用极小固定样例即可，不建立全仓回归矩阵，不新增哈希／指纹系统，不重复每个阶段跑。

## 8.3 第一批真实 GPU 工作

在 P0 数据转换时可并行建立 AdaPoinTr 或 DyNFL 独立环境。P0 完成后立即执行下面的 D0 实验，不先重构所有历史代码。

---

# 9. 阶段 P1：可证伪的基线图谱，优先排除简单解释

建议总时长 2–4 天；只消费开发数据。

## 9.1 几何来源至少包含

- `G0`：相同构建帧的原始 Actor-canonical 累积点／surfel。
- `G1`：Actor-local TSDF 或对象占据融合；FREE/UNKNOWN 处理明确。
- `G2`：现有 M8，标为 legacy learned，不充当独立源域性能。
- `G3`：AdaPoinTr 数据适配版；不能只用未适配 ShapeNet checkpoint 失败来说明它不适合。

数量不足的 Actor 不因缺少旧 COMPLETE candidates 被剔除；各方法使用同一输入对象集合。某方法不支持则显式 fallback，并报告失败覆盖。

先对 G0/G1/G2 得到完整性—early 曲线，同时训练 G3。开发图谱至少覆盖 10 个日志级组；需要时只从训练中选固定小规模样本加速，不碰 source_test。

## 9.2 同几何回波权重对照

固定 G0、G2 两种几何，固定同一 legacy categorical 读出，比较：

| 编号 | 权重来源 | 目的 |
|---|---|---|
| W0 | 单位权重 | 最简单基线 |
| W1 | build-only 观测支持比例 | 检查神经头是否只是重复计数规则 |
| W2 | 固定局部密度／采样机会归一化 | 检查密度不公平造成的收益 |
| W3 | 同特征、近似容量的单标量权重 MLP | 三态监督最关键的挑战者 |
| W4 | 三态 F/O/U head，读出使用其 O 分量 | 检查额外证据监督是否真的提供增量 |

W3/W4 先用相同回波损失比较，再单独增加 F/O/U 辅助监督；避免把“多一个监督源”错归因于“三个输出维度”。

所有监督仅来自 train 的目标射线。W1/W2 不能用 target support rate。

## 9.3 D0 如何解释

- 如果 G3 在普通监督下已经同时解决 CD 与 early，而 M8没有，换基座；不要继续论证全领域“不可兼得”。
- 如果 W3 与 W4 相当，主文不再宣称三态证据本身必需；可保留工程语义或附录。
- 如果某模型只有用自己的 median evaluator 才改善，几何算子无增量，归类为回波建模，不是修复几何。
- 如果点数匹配以后收益消失，归类为密度／资源收益，转效率问题或关闭原创新主张。

这一步产出一张简洁主表与一张权衡图，不产出新的“authority module”。

## 9.4 当前执行状态（2026-09-07）

D0 几何部分已在同一 66-Actor / 34-log `legacy_diagnostic` cohort 上完成。G0 raw fusion、G1 Actor-local TSDF、G2 历史 M8 与 G3 AdaPoinTr 均使用统一的 64/128/256/512/native 密度预算、query rays 与 evaluator。G3 已完成官方 zero-shot、预训练适配 600 epochs 和 scratch 600 epochs；两次训练均为 AMP batch 64，峰值 GPU 显存约 `19.96GiB`。

当前证据为：官方预训练适配稳定优于 scratch，但 G3 在所有固定密度预算下没有越过 G0/G1 简单前沿；G2 的 early 更低，但承担明显 CD/F-score 与 hit 代价。完整数值与 run ID 见 `docs/WORLDSIM_V7_2_D0_GPU_RESULTS.md`。

第 9.2 节也已完成：G0/G2 使用同一 categorical reader 比较 W0 单位权重、W1 build support、W2 密度／采样机会归一化、W3 单标量、W4 三态 response-only 与 W4 三态辅助监督。W3 与 W4 response-only 基本等价；F/O/U 辅助只在 G0 小幅改善，在 G2 同时恶化 early 与 hit，未形成跨几何稳定增量。单标量在两种几何上都改善 hit／深度误差，但付出小幅 early 代价。完整结果见 `docs/WORLDSIM_V7_2_D0_WEIGHT_RESULTS.md`。

因此 D0 的 legacy 机制筛选完成，原“普遍三态表征缺陷”主张关闭；更窄且仍有效的问题是观测约束下的 hit--early 权衡。该证据仍不触发 D1：干净 dev/route-select 数据和路线 B 的原生 capability 尚缺，`source_test` 与 `external_test` 保持未读。

---

# 10. 阶段 P2：两条有限规模研究路线

## 10A. 路线 A：可复用、观测约束的对象补全

### A0：主任务先叫对象表面补全

第一版输入与输出：

\[
B_i=\{(x,\text{local evidence},\text{frame support})\}_{\text{build}},
\qquad \widehat S_i=f_\theta(B_i,D_i).
\]

给定轨迹只负责规范坐标转换；不预测未知未来运动。输出是可评价的真实点集／表面，不是候选点筛选。

起点优先用适配后的 AdaPoinTr。V7.1 M8 是低容量对照，不强制所有后续模块依赖 M5→M7→M8。

### A1：唯一首轮创新假设

> **成熟集合补全网络已经具备形状先验；以真实观测自由区间和留出射线监督它，能否改善几何—可见性权衡，而不是只改变一个回波头？**

保留完整集合监督，增加几何位置的观测一致性。优先复用 baseline 的局部查询／点生成模块，不先新增任意容量的 ray decoder。

损失包络是：

\[
L_A=L_{\text{surface}}+\lambda_{\text{free}}L_{\text{known-free}}
+\lambda_{\text{hit}}L_{\text{observed-hit}}.
\]

含义：

- 表面监督同时约束完整性与 precision。
- FREE 只在可信已观测区间约束；后方及矛盾区不强制为空。
- hit 约束由同一个预测表面产生，不由能绕过几何的独立深度头代偿。
- 先把强基线训练到正常收敛，再以少量物理监督微调；不是从头把五六项 loss 同时堆给小 MLP。
- 最终输出策略在训练图和部署一致；不能把“多点保留”仅作为训练对象，推理再转 UNKNOWN。
- 观测记录不可改；允许重建表面在源域定义的观测误差容差内调整，不把每个稀疏 endpoint 硬当完美表面。

权重从原基线量纲与训练梯度确定一个初值；开发阶段最多一次有明确理由的调整。PCGrad 只在新基座上重复观测到持续梯度冲突后使用，并与同配方无 PCGrad 公平比较；不因 V7.1 用过就默认它是贡献。

### A2：对象 Occupancy 的条件解锁

只有当以下条件满足，才把标题扩为 Occupancy：

- 有独立可靠的 occupied/free/unknown 标注或可复现的公开对象占据协议。
- 使用完整对象空间查询，而非只在“已知会有点”的 candidates 上分类。
- unknown 使用真实掩码，不能把所有无点体素当 FREE，不能把 hit 后整段射线当实心物体。
- 与对象中心占据补全的官方数据生成／抽取协议对齐。

首选复用 ObjectCentricOccCompletion 的局部／对象占据基座，或使用已跑通的外部局部场，不重启旧 M1/M3/M4 global latent + 任意阈值抽面的家族。

若体积标签不足，路线 A 正式叫 **3D object surface completion**。不为了搭上 Occupancy 热点造全空间真值。

### A 的正式比较

主指标优先 F-score@固定阈值和 CD；已知自由空间侵入是重要副指标。若用官方对象占据基准，则 IoU 成为主指标，surface/ray 作为互补。

不能只跟 always-COMPLETE 比，也不能只要求“比自己的 M8 好 0.5 mm”。

第一轮希望看到的实质效果量：例如相对最强同信息基线 CD 降低约 5% 或 F-score 提高约 2 个百分点，并且没有靠 precision／known-free violation 明显恶化换得。数字是开发资源分配的参考尺度，不是 CVPR 录用条件。

### A 若失败

先问强基线是否有足够性能、标签是否可信、目标表面是否可辨识。只允许一项基于 D0 证据的结构替代（例如从点集换成有局部证据的公开占据基座）。不允许依次试全局 SDF、球体、圆盘、再路由到 UNKNOWN。

---

## 10B. 路线 B：从条件测距升级成可组合的 LiDAR 传感器输出

### B0：必须先跑通一个真正的场景级方法

首选官方 DyNFL；若当前 Waymo 数据授权不具备，先运行可获得的 LiDAR4D KITTI-360 官方协议建立完整扫描能力，同时保留 DyNFL 的数据适配任务。

不要用 M39 的 Actor 框内条件中位数替代 B0。

逐场景训练的已知轨迹／背景先验与我们方法对齐；官方基线会优化场景时，不能只让它零样本推理而让自己的方法拟合本场景。

### B1：唯一首轮创新假设

> **把真实 ray opportunity、未返回和前景／背景遮挡明确纳入观测模型，并用表面监督约束同一个几何源，是否比只拟合条件深度更准确、更能复用对象资产？**

共同概率接口：

\[
p(o_r,d_r,a_r\mid B,\xi_r),
\]

其中 `o_r` 是有效返回存在性，`d_r` 是测距，`a_r` 为可选强度，`ξ_r` 包括已知发射方向、时间、传感器参数。

一个允许的基线分解是：

\[
p(o,d,a\mid B,\xi)=p(o\mid B,\xi)\,p(d,a\mid o=1,B,\xi).
\]

它本身不是创新，必须由数据和外部对照证明增量。

### 两种可接受实现，不同时开发

**实现 B-D：判别式返回分布。** 在固定度量深度网格上预测 `{no-return,d_1,…,d_K}`；同一观测范围内总概率为 1。它可以保留 categorical 思路，但必须有 no-return 类、完整场景输入和固定量程。中位数只是条件测距读出，几何另评。

**实现 B-F：基于几何场的有序传感器渲染。** 复用 DyNFL／LiDAR-RT 的 forward、射线采样、丢失回波和场景组合；为几何部分加入公开可比的观测／表面监督。不得把 occupied belief 直接当消光系数。采样步长、返回概率和几何支持的量纲分别定义。

先用 B0 的成熟实现，只有 D0 证明当前读出限制是主因才选对应替代。不从零发明第三套体渲染理论。

### 场景组合必须真实执行

- 动态对象在给定刚体轨迹中移动；静态背景由 build scans 构建。
- 一条 ray 可以先经过背景遮挡、多个 Actor 或空空间；按真实世界深度顺序处理。
- 前景 Actor 未产生有效回波时，是否继续观察背景由选定传感器 forward 决定，不能永远在第一个框内强制返回。
- 背景也是模型输入／资产的一部分，不可使用 target scan 补背景。
- 相同 Actor ID 在不同动作中必须存在，但其**可见性**可以因遮挡变化；“不见”与“被删除”分开。

### 监督与目标泄漏边界

- 推理 `QueryRayBatch` 不含 target depth，也不以 target range±margin 设采样边界。
- 量程由传感器公开参数、固定场景区域和 build-only 包围盒决定。
- actor_id 的目标对应只用于评估，不提供“这束光一定命中哪辆车”的 oracle routing。
- no-return／无效发射掩码来自实际协议，不从模型失败自动生成。
- 原 V7.1 soft depth 使用 target-informed sampling 的函数不能直接移作 B 部署 forward。

### B 的主指标

全扫描 range MAE/RMSE、返回存在性／空 bin 指标、点云 F-score／CD 为主，early/hit 为定向物理误差。强度和速度是独立附表。

希望看到相对最强同协议基线约 5% 级别的主误差降低，且返回覆盖、early 和背景／Actor 区域没有明显反向交换；或者精度近似而有预先声明、实测的显著训练／渲染效率优势。

### B 若失败

如果只在 Actor 框条件返回成立、全场景遮挡／无回波崩溃，不能把 B 宣布成功。可以降为 A 的一个回波读出模块；不再写完整 Neural LiDAR Simulation 标题。

---

# 11. D1 的选路规则：比较证据，不比较不同任务的绝对指标

建议在第 2–3 周完成 D1。

| 判断项 | A 的问题 | B 的问题 |
|---|---|---|
| 任务有效性 | 是否真的输出更好的对象表面／占据？ | 是否真的输出完整扫描，而不是 target 条件测距？ |
| 外部竞争 | 是否超过 AdaPoinTr／对象占据或强融合？ | 是否超过 DyNFL／LiDAR4D／LiDAR-RT 的公平版本？ |
| 简单解释 | 点数、输入帧、后处理是否解释全部收益？ | 工作点、返回掩码、目标范围或强度是否解释全部收益？ |
| 日志稳定性 | 增益是否跨日志与观测稀疏度存在？ | 增益是否跨场景、动态／静态区域与距离存在？ |
| 完整应用 | 能否嵌入场景并产生正确的遮挡／占据？ | 能否重建真实新扫描并处理遮挡与无回波？ |
| 算力与篇幅 | 能否完成强比较并在 8 页讲清？ | 同左 |

决策：

- 只有 A 有清楚增量：A 主线；B 留作固定回波评价。
- 只有 B 有清楚增量：B 主线；A 留作几何约束及消融。
- 两路都有：选择相对其强基线增量更稳定、应用更完整、单位成本更好的路线。效应接近时优先 A，因为它与当前可复用数据和刚性 Actor 条件更一致，而非因为先验断定 A 更先进。
- 都没有：不添加无限第三路线。回到 D0 判断是任务／标签问题还是缺乏创新；交付可证伪诊断和基线研究报告，当前方法主张关闭。

**不能因为某方案在 5 个指标中 4 个过线就叫成功，也不能因一个次要指标少 0.001 就自动改出下一模型。** 主任务效果、重要副作用和不确定性整体判读；门槛不随 test 结果改动。

---

# 12. 阶段 P3：选定主路线后再做的核心消融

只保留下列能解释最终方法的对照，不做全排列。

## 12.1 几何 × 回波读出的 2×2

固定两种几何来源：强外部基础几何与选定改进几何。

固定两种回波：简单单标量／基线读出与证据／新传感器读出。

报告几何表与返回表。若只有换读出有用，主贡献就写回波建模；若换几何在不同读出下都改善，才支持几何贡献。

## 12.2 F/O/U 真的比标量有用吗？

W3 与 W4 在相同信息、容量和回波监督下比较；F/O/U 附加监督单独消融。若无可靠增量，删去“证据化”主贡献，不用一个 M49 公式强保它。

## 12.3 控制输出密度和额外数据

同一输出预算、相同 build 帧与相同预训练。

同时公布原生输出效率曲线。若方法就是用更多点或更多训练赢，明确说明成本权衡；不能一边匹配自己的设置、一边压缩对手设置。

## 12.4 跨几何来源

最终可复用模块至少在两种实际不同的几何来源上测试，例如多帧融合与 AdaPoinTr，或对象占据与神经场提取。

DynamicVGGT／Gau-Occ 输出可作为扩展，但不同模态、可用先验和训练数据另列。公开代码／权重不足时不阻塞主线，也不报告想象中的复现。

## 12.5 训练—部署一致性

只是一次机制对照，不再造“认证子项目”：新方法训练使用的表面、回波 forward 和最终部署输出必须可追溯到同一实现。硬最小、期望深度和条件中位数分别命名。

---

# 13. 阶段 P4：鲁棒性与完整应用，而不是堆新头

## 13.1 固定的三类压力测试

1. **输入稀疏度：** 100%／50%／25% 的 build 观测；规则在 source 开发时固定，同一物体同一 target。稀疏化训练增强需要给基线同等机会。
2. **时间／视角：** 未参与 build 的实测时刻；连续时间块留出与交错帧留出分别命名。相邻插值不叫长期预测。合成大偏移视角没有真实 GT 时只能作可视化／机制测试。
3. **轨迹／框扰动：** 给输入先验施加固定误差水平，以原始独立标注评价；训练、主表和鲁棒性表分开。不能把箱体真值当完整接触几何。

只在所选主路线做完整矩阵，另一条路线不额外扩研究预算。

## 13.2 A 的最小完整应用

**默认选择：真实留出扫描上的场景前景占据／遮挡重建。**

- 背景来自同一组 build scans，并按提供的动态轨迹移除融合残影。
- 将各 Actor 补全表面通过给定 pose 放回场景。
- 用相同世界坐标中的射线联合查询全部可见对象与背景。
- 对实际 held-out scans 报告前景占据／可见性、前后遮挡关系和命中。
- 画整场景鸟瞰／侧视、至少一个被遮挡车辆、动态序列和误差分布。

若有可信场景占据标签可补 foreground-only IoU；若只有回波标注，则只声称可观测几何／遮挡用途。

可选附加：固定 3D 检测器对补全前后点云的检测变化。不能同时重新训练检测器改变多个因素，除非作为独立完整应用预算。

## 13.3 B 的最小完整应用

**默认选择：完整场景新视角／时刻 LiDAR 重建。**

输出完整 range image、return mask、点云视频；有强度才输出强度图。

必须含静态背景、至少两个动态 Actor 的遮挡组合和远距稀疏区。只展示孤立车辆、或者把每 Actor 单独渲染再无序拼接，不算完成。

额外编辑轨迹不应被称为反应式驾驶模拟；未训练行为响应就明确轨迹已给定。

## 13.4 RL 与碰撞用途

本版本不训练 RL、不重新打开 P346/authority tree。已有规划／RL结果可放背景，但不得作为新的主要证据。

真正接触／碰撞查询需要独立网格、可验证模拟几何或同等真值。不能以“Actor 未删”“回波更准”直接推出碰撞更可靠。

---

# 14. 最终独立评测与外域策略

## 14.1 优先级

**独立源域正结果 > 同任务强外部基线 > 一个完整应用 > 第二数据集／零样本扩展。**

跨传感器零样本不是 V7.2 唯一成败条件。不能继续为救 M43 牺牲一个原本可以明确成立的任务。

## 14.2 独立源域测试

D1 选路后只保留一个主候选和预注册消融。训练与所有 checkpoint 选择结束，运行 source_test。

按日志 cluster bootstrap 给主指标差值区间；报告 hazard/clear、moving/static、near/far 的完整结果。分组只用于评价，不偷偷改变数据分母。

独立源域测试不通过：承认核心泛化证据不足，停止用新模型编号继续消费测试。需要新研究则另定版本和独立证据，不把本次称“几乎成功”。

## 14.3 外域测试的三种不同主张

| 模式 | 允许操作 | 正确名称 |
|---|---|---|
| 冻结共享模型直接推理 | 确定性单位／坐标／传感器元数据转换；不拟合目标域权重、阈值或统计量 | 零样本跨域 |
| 目标场景 build scans 优化 | 固定配方、固定步数／source 设定规则；不用 held-out scans 选模 | 逐场景重建泛化，不是零样本 |
| 用目标域训练集训练 | 官方 train/test 隔离 | 第二数据集实验，不是零样本 |

不能混在一个“generalization”标题下。

M43 已消费 AV2 logs只作历史诊断，不做 V7.2 主域训练／最终验证。新的 target cohort 从已知未使用日志中 metadata-only 冻结。若剩余量不够，采用授权 Waymo 或新的正式公开协议。

## 14.4 不强制“传感器回波数值不变”

换传感器后 beam width、发射模式、检测阈值等变化，正确观测本来就可能不同。

V7.2 要求的是：**相同几何经各自正确测量模型后符合各自观测**，而不是训练两个传感器输出完全相同。源域稀疏化与密度归一化是待验证机制，不是已经确定的 M43 病因。

---

# 15. 资源与执行合同

## 15.1 默认资源

从 1×RTX 3090 24 GB 开始，不人为限制永远单卡。上传 status 的最后磁盘记录约 82 GiB，只是历史记录，启动时读取当前 CPU RAM／磁盘／GPU 情况，不假设仍然相同。

- 开发和 baseline capability：1×3090。
- 多场景独立处理或正式重复：有资源时 2–4×3090；每个任务日志独立。
- 相同模型单卡可放但慢：数据并行优先。
- 超显存：先 microbatch、梯度累积、AMP、分块查询、激活检查点；仍超出才研究 FSDP／ZeRO。
- 不通过降低最终分辨率、缩短对手时域、删除难样本来适配硬件。

DyNFL 官方环境明确包含 RTX 3090 [R3]；不意味着所有现有配置都可直接无修改跑在当前环境。对象占据官方示例是多卡流程 [R2]，单卡能力应实测。

## 15.2 预算建议，不冒充预测耗时

- D0＋双路线小规模比较：初始配置总预算约 48–72 GPU 小时，允许先记录实测单 epoch／单 scene 时间再分配。
- 完整主路线＋强基线＋最终重复：预留约 250–450 GPU 小时，实际视选路与数据决定。
- 若 baseline 未收敛就用完探索预算，标 `budget-limited`，不能把它按弱模型拒绝。根据潜力增配或缩小该轮比较规模，正式共同 benchmark 不缩水。
- 训练多久由公开方法配方、训练／开发曲线与一致停止规则决定；禁止沿用“4 epochs 跑完就判一个新家族死亡”的惯性。

这些是资源预留，不要求把 GPU 填满，也不是总费用承诺。

## 15.3 I/O 与 GPU 并行

维持单个下载调度器，不等全部 archive EOF 才训练。

```text
下载／扫描 → log-ready → CPU 转换 → 原子发布缓存 → GPU 任务
                                  ↘ 完整场景背景／射线准备
```

有可用输入就训练／评估。下载停滞只检查实际写入与进程，不把旧 cohort `.complete` 总数算作新实验进度。

缓存按日志／Actor 分片，不让大量后台 GPU evaluator 各自预载同一 checkpoint 和整库。通常一个 evaluator 队列串行消费多个冻结模型即可；足够资源时再并行。

## 15.4 环境和权限

外部基线各用独立环境；优先复用已有兼容环境。不要为了一个库污染当前 Motion-Proj 环境。

Waymo 许可／认证由正式渠道解决；不能从未经授权的镜像绕过。授权未具备时标 `blocked_access`，继续 A 或可访问的 LiDAR4D 原生协议，不把它写成方法失败。

---

# 16. 代码落地：新增薄层，停止依赖实验脚本树

## 16.1 可复用与必须改写

| 现有路径 | 处理 |
|---|---|
| `worldsim_v71/actor_canonical.py`、`dataset_nuscenes.py`、`av2_adapter.py` | 复用坐标／读取逻辑，新增完整 ray/frame/sensor 字段 |
| `worldsim_v71/actor_corpus.py` | 旧 v1 只读；新增 v2 builder，不覆盖旧 NPZ |
| `worldsim_v71/evidence_volume.py` | 保留作为证据基线；重新明确“未观测”和“无回波”不等价 |
| `worldsim_v71/gaussian_anchor_relocation.py` | 暴露为历史 M8／低容量基线，不强制新模型继承 |
| `worldsim_v71/authority_contract.py` | 复用类型所有权与 SE(3)；接口名不能把未校准 energy 宣称碰撞概率 |
| `worldsim_v71/evaluate_surface.py` | 新建命名空间分离 evaluator；旧实现存档 |
| `worldsim_v71/first_return_renderer.py` | 旧 target-informed soft函数不进入新部署路径；literal 作为明确诊断算子 |
| `scripts/run_worldsim_v71_m39_*` | 抽出只接受几何与 QueryRayBatch 的 baseline inference，不带 dataset/训练脚本依赖 |
| `scripts/run_worldsim_v71_m43_*` | 添加勘误输出，不继续承载 V7.2 全任务 |
| `paper/` | 保留 V7.1 快照；新建独立 `paper_v72/` |

## 16.2 目标目录

```text
motion_proj/worldsim_v72/
  data/
    schema.py
    splits.py
    actor_dataset.py
    scan_dataset.py
    nuscenes_adapter.py
    av2_adapter.py
    native_lidar_adapter.py
  baselines/
    legacy_v71.py
    fusion.py
    pointr_adapter.py
    object_occ_adapter.py
    neural_lidar_adapter.py
  models/
    completion.py              # A 被选中后保留
    sensor_return.py           # B 被选中后保留
  render/
    ray_queries.py
    scene_compositor.py
  evaluation/
    surface_metrics.py
    return_metrics.py
    application_metrics.py
    clustered_report.py

scripts/
  prepare_worldsim_v72.py
  run_worldsim_v72_baselines.py
  train_worldsim_v72.py
  evaluate_worldsim_v72.py
  export_worldsim_v72_paper.py

configs/worldsim_v72/
  experiment.yaml
  data_roles.json
  baseline_map.yaml
  route_selection.yaml

paper_v72/
  main.tex
  arxiv.tex
  supplement.tex
  sections/
  figures/
  tables/
  results/results_macros.tex
  bibliography.bib
```

不要提前创建空的所有未来模型。先实现数据、评价、基线与一条训练入口；另一条通过同一接口接入。

## 16.3 最小配置示意

```yaml
version: worldsim_v72
parent: research/worldsim-v7.1-learned-evidential-surface
research:
  primary_route: undecided
  candidate_routes: [object_completion, neural_lidar]
  rl_primary: false
  final_model_selection_source: dev

data:
  roles: configs/worldsim_v72/data_roles.json
  split_unit: driving_log
  inference_target_access: false
  legacy_m43_role: diagnostic_only
  full_scan_required_for_lidar_claim: true

metrics:
  separate_geometry_and_return: true
  clustering_unit: driving_log
  geometry_cd: l1_meters
  point_beam_tube_diagnostic_m: 0.20
  report_no_output_and_no_return: true

training:
  config_origin: official_baseline_then_documented_adaptation
  fit_normalizer_on: full_train_role
  best_checkpoint_from: dev_only
  final_seeds: [0, 1, 2]
  report_all_final_seeds: true

resources:
  default_gpu: RTX3090
  default_gpu_count: 1
  multi_gpu_if_needed: true
  downloader_instances: 1
  preserve_benchmark_resolution: true

paper:
  root: paper_v72
  format: cvpr_official
  main_content_pages: 8
  include_supplement: true
```

`final_seeds` 只用于最终既定配方和对应基线，不用于开发择优。其他未定参数在强基线 capability 后固定，不在这里虚构最优值。

## 16.4 入口合同

以下是 **V7.2 需要实现的 CLI**，不是当前仓库已经存在的命令：

```bash
python scripts/prepare_worldsim_v72.py \
  --config configs/worldsim_v72/experiment.yaml --role train

python scripts/run_worldsim_v72_baselines.py \
  --config configs/worldsim_v72/experiment.yaml --suite diagnostic --role dev

python scripts/train_worldsim_v72.py \
  --config configs/worldsim_v72/experiment.yaml --route object_completion

python scripts/train_worldsim_v72.py \
  --config configs/worldsim_v72/experiment.yaml --route neural_lidar

python scripts/evaluate_worldsim_v72.py \
  --config configs/worldsim_v72/experiment.yaml --role route_select

# D1 后只评估选定候选；final flag 不自动触发训练／选模。
python scripts/evaluate_worldsim_v72.py \
  --config configs/worldsim_v72/experiment.yaml --role source_test --final

python scripts/export_worldsim_v72_paper.py \
  --config configs/worldsim_v72/experiment.yaml --paper-root paper_v72
```

本次 ZIP 中未发现 `scripts/compile_latex.py`，不能把历史日志中的外部编译脚本当作仓库现有入口。优先复用执行环境里已经验证的 LaTeX 编译流程；没有该脚本时，直接用 `latexmk -pdf -interaction=nonstopmode -halt-on-error` 编译新主稿与补充材料，不另造工具链。

---

# 17. 可写进主论文的贡献要由结果决定

## A 成为主线时

暂定标题方向：

**Observation-Consistent Object Completion from Sparse Driving LiDAR**

贡献最多三项：

1. 在明确输入与轨迹先验下，以同一对象表面连接集合补全与真实观测约束。
2. 相对成熟补全／占据方法，独立源域上改善表面—可见性权衡，并分析新增信息来自哪里。
3. 在完整场景前景占据／遮挡应用及第二几何来源上证明可复用性。

如果证据化 head 不优于标量，就不要出现在标题。M49 放补充解释，不当第一贡献。

## B 成为主线时

暂定标题方向：

**Geometry-Grounded LiDAR Re-simulation with Explicit Return and Occlusion Modeling**

贡献最多三项：

1. 从已知轨迹对象资产与背景形成同一观测前向模型，明确几何与传感器响应。
2. 在完整 ray opportunity、range／ray-drop 与动态遮挡上，比直接神经 LiDAR 方法有稳定增量或显著成本优势。
3. 留出真实时刻／视角与场景组合的完整扫描验证，而不只是框内条件中位数。

若只验证条件回波，标题必须缩窄，不写 full simulator。

## 原 C1/C2/C3 的处理

不机械保留三项旧标签：

- C1 保留为几何／占据或传感器主任务。
- C2 的类型隔离与已知刚体组合是方法约束；除非带来新的任务收益，不单独包装成发现动态世界模型。
- 原 C3 的连续任务代价密度及旧 authority 系统冻结为历史资产，不强行塞入新主稿。
- M49/M51 可解释测量／归一化行为，但不能替代强方法与独立正结果。

---

# 18. CVPR LaTeX／PDF 交付合同

## 18.1 从第一周开始写，但不把每次实验写成一页

- Draft 0：任务、输入输出、直接竞争者、基线表、待验证假设。
- Draft 1：D0 结果与选路；主方法按结果成形。
- Draft 2：强基线、关键消融、完整应用。
- Draft 3：独立 source/external 主表、失败范围、效率。
- arXiv v1：非匿名，完整证据，明确已支持和未支持主张。
- conference draft：匿名、正式模板、正文页数符合投稿年度要求。

本轮查到的官方 CVPR 2026 规范是八页正文、参考文献额外允许；arXiv 允许 [R12]。2027 投稿最终按当年官方要求核对，本计划不硬写尚未核实的会议截止日期。

## 18.2 主文版面建议

| 内容 | 页数预算 |
|---|---:|
| 摘要、问题与动机 | 1.1 |
| 相关工作、任务定义 | 0.8 |
| 最终方法 | 2.3 |
| 主结果、强基线、消融 | 2.2 |
| 完整应用、效率与鲁棒性 | 1.2 |
| 局限与结论 | 0.4 |
| 合计 | 8.0 |

参考文献另计。不是要求恰好填满；不要通过缩小字体或改变边距挤压。

## 18.3 最小主图表

- Figure 1：一个完整任务输入→输出→真实 held-out target，至少显示 3D 几何和实际观测，不只是彩色置信度。
- Figure 2：最终简洁网络／渲染图，不出现 M0→M51 历史树。
- Figure 3：相同几何下读出对比或几何—回波权衡，明确密度与覆盖。
- Figure 4：完整场景应用／视角时序，含一个有代表性的失败。
- Table 1：独立源域 + 强外部基线，绝对指标与成本。
- Table 2：关键 2×2／标量对照／监督消融。
- Table 3：第二协议或零样本＋固定压力测试。

负结果只保留最有解释力的两三类。其余放 supplementary，不能把 8 页写成失败账本。

## 18.4 交付文件

```text
paper_v72/main.tex
paper_v72/main.pdf
paper_v72/arxiv.tex
paper_v72/arxiv.pdf
paper_v72/supplement.tex
paper_v72/supplement.pdf
paper_v72/results/results_macros.tex
paper_v72/CONTRIBUTION_MAP.md
paper_v72/SUBMISSION_CHECKLIST.md
```

所有主表数字从最终结果导出；不能把旧 V7.1 数值硬写成 V7.2 成功结果。

阶段草稿允许明确 `PENDING`。最终称 paper-ready 时不得有未运行主实验占位、混算子表格或未标明的开发集结果。

一份 PDF 编译成功不等于科学投稿完成。

---

# 19. 建议时间线

从真正开始执行的 D0 计时，不承诺后台运行，也不把数据许可等待计算成科学失败。

| 阶段 | 建议时长 | 必须交付 |
|---|---:|---|
| P0 | 0.5–1 天 | M43 勘误、数据角色、v2 schema、论文骨架 |
| P1 | 2–4 天 | 强简单基线、标量权重对照、首个外部补全／LiDAR 方法跑通 |
| P2 / D1 | 7–12 天 | A/B 各一个机制候选与公平开发比较，明确选路 |
| P3 | 7–10 天 | 胜出路线完整训练、强基线、关键消融、第二几何来源 |
| P4 | 5–7 天 | 完整应用、固定压力测试、成本分析 |
| P5 / D2 | 5–7 天 | 独立源域／第二协议／外域测试及聚类区间 |
| P6 | 4–6 天 | 8 页主文、补充材料、arXiv 包 |

总量约 5–7 周，随真实算力和下载进度调整。若第二周仍然在建立日志／校验框架，而没有强基线曲线，说明执行偏离计划。

每完成一个有科学意义的里程碑，更新一次 status、experiments、failures 和论文。普通下载轮询、语法错误或每个 epoch 不新增长文档。

---

# 20. 禁止复活的范围与允许的新研究

## 20.1 不再开启

- CPSC-Lite、V6.3 已关闭 Surface-Mean／Max／CVaR 救援。
- 自家固定 candidates 上反复 classifier／visibility gate／UNKNOWN 删除。
- M1 双头后验 AND 抽面、M3/M4 全局 latent 场及其 band／offset sweep。
- M9–M17 仅修改球／椭球／圆盘／厚度来挽救原指标。
- M40–M51 的 opacity／family-mass／visibility attenuation／smooth-depth 调参循环。
- P4/P6-C/P346 的 source／target 校准救援。
- 用已消费 M43 做 V7.2 的 normalization fit 或 test-directed sensor correction。
- full IR-WM 解冻、换大模型或加 LoRA 作为没有诊断的默认答案。
- RL、语义、动态预测、视频生成与几何一轮全部混训。

## 20.2 允许且不等于复活旧家族

- 在公开成熟补全模型上，以新的完整输入／标签合同学习形状，并与其公平竞争。
- 对官方 DyNFL／LiDAR-RT 做任务匹配复现或明确的数据适配，使用其完整场景与传感器模型。
- 用真实完整射线构造 no-return／遮挡监督，而不是给旧 conditional head 增加阈值。
- 将旧误判结果用正确算子重算，保留原始实验与勘误，不把勘误当新方法收益。
- 有证据后做一次真正改变输入信息或最终预测对象的结构修正。

停止规则关闭的是具体机制、数据与输出合同，不是把“SDF”“Occupancy”“神经渲染”这些整个领域永远封禁。

---

# 21. 风险与应对

| 风险 | 必须采取的动作 |
|---|---|
| 外部基线适配很慢 | 隔离环境、优先原生小协议；记录 access/dependency 限制；不以失败移植版作弱对照 |
| 旧累计点被当完整 GT | 保留 unknown/visibility mask；切到真实表面完成度或官方对象占据协议 |
| 几何 baseline 被 categorical 计数覆盖 | 独立结果命名空间；一次性微型语义检查 |
| 节点数量变多才赢 | 点数匹配主表＋原生效率曲线 |
| 三态 evidence 不比标量好 | 移除 evidence 主贡献，保留更有效的简单设计 |
| source 最终没正结果 | 不读更多外域来找绿灯；收缩主张／关闭主方法 |
| 外域失败但 source 强 | 可以写源域／逐场景重建方法，诚实保留跨域边界，不自动要求另开传感器不变性版本 |
| 无回波真值不可得 | B 改为明确的 projected-range-image 协议，或请求正式 Waymo 授权；不伪造无回波物理标签 |
| 又在 Mxx 无限试验 | D1 后冻结主路线；超出一个结构 fallback 要先回答它解决哪项已证明的缺口 |
| 物理／外观混淆 | 不用图像梯度改变声称固定的物理几何；也不要求稀疏物理点替代完整外观容量 |
| 高指标但任务不完整 | A 必须有完整场景用途，B 必须全扫描／背景遮挡；否则缩窄标题 |

---

# 22. 最终完成标准

V7.2 只有满足下列证据条件，才叫“面向投稿的研究闭环”：

1. 明确选定 A 或 B 为主任务，输入、先验、输出和最终部署 forward 写清楚。
2. 主要学习式增益不再只来自已暴露的 66 个 Actor；有全依赖链隔离的独立源域证据。
3. 至少一个强外部学习基线、一个强简单基线，以及单标量／密度等最关键简单解释被公平检验。
4. 几何与回波的算子、分母、密度、训练成本独立清楚；不借改读出宣称几何修复。
5. 至少一个完整视觉／三维应用，有真实 held-out 观测或可靠 GT，而不只是类型残差为零。
6. 主论文与补充材料可编译，正文符合页数要求；结果、代码路径与主张一致。

不要求所有压力测试都赢、不要求未观测世界完美重建、不要求零样本全面成功。需要的是**所选任务中一个清楚、有竞争力、经独立验证的贡献**。

如果这些条件未达成，照实交付技术报告和失败范围，不能用“所有任务都跑完了”替代 paper-ready。

---

# 23. 给执行 Agent 的首轮指令

> 以本计划及已上传的 V7.1 `(2).zip` 为依据，在 V7.1 已完成分支上建立 V7.2；不要接入其他版本结果。先完成 M43 混算子勘误和新旧数据角色合并，保留 M39 外测拒绝。只做必要的语义检查，不做全仓回归／哈希／指纹。随后立即运行同几何的单位／支持率／标量／F-O-U 权重对照，并启动一个官方补全基线和一个完整 LiDAR 基线的能力运行。不能默认 Motion-Proj 是正确主线。第一轮报告必须包括：真实代码起点、实际可用数据／资源、D0 是否支持问题普遍性、A/B 哪一路能形成公平强对照、目前不能支持的主张。研究期间同步维护 `paper_v72`，但不要把每个实验编号搬进主文。直到 D1 选路前，不触及最终 source/external test；选路后只集中推进一条主任务及一个完整应用。

---

# 参考资料：一手来源与实施边界

以下链接在制定本计划时核对。官方代码可见不等于已经在本环境复现；所有性能必须实际运行后填入主表。

- **[R1] PoinTr／AdaPoinTr 官方代码与配置**：https://github.com/yuxumin/PoinTr ；论文：https://arxiv.org/abs/2301.04545
- **[R2] 对象中心占据补全，NeurIPS 2024**：https://arxiv.org/abs/2412.05154 ；官方代码：https://github.com/Ghostish/ObjectCentricOccCompletion
- **[R3] DyNFL，CVPR 2024**：https://arxiv.org/abs/2312.05247 ；项目：https://shengyuh.github.io/dynfl/ ；官方代码：https://github.com/prs-eth/Dynamic-LiDAR-Resimulation
- **[R4] LiDAR4D，CVPR 2024**：https://arxiv.org/abs/2404.02742 ；官方代码：https://github.com/ispc-lab/LiDAR4D
- **[R5] LiDAR-RT，CVPR 2025**：https://arxiv.org/abs/2412.15199 ；官方代码：https://github.com/zju3dv/LiDAR-RT
- **[R6] EvOcc，CVPR 2025**：https://openaccess.thecvf.com/content/CVPR2025/html/Kalble_EvOcc_Accurate_Semantic_Occupancy_for_Automated_Driving_Using_Evidence_Theory_CVPR_2025_paper.html
- **[R7] Gau-Occ，CVPR 2026**：https://arxiv.org/abs/2603.22852
- **[R8] DynamicVGGT，CVPR 2026**：https://arxiv.org/abs/2603.08254
- **[R9] SimULi 作者预印本**：https://arxiv.org/abs/2510.12901
- **[R10] Argoverse 官方 Sensor 数据说明／下载**：https://argoverse.github.io/user-guide/datasets/sensor.html ；https://argoverse.github.io/user-guide/getting_started.html
- **[R11] Waymo 官方数据与传感器说明**：https://waymo.com/open/about/ ；https://waymo.com/intl/it/open/data/perception/
- **[R12] CVPR 官方作者规范与模板**：https://cvpr.thecvf.com/Conferences/2026/AuthorGuidelines ；https://github.com/cvpr-org/author-kit
- **[R13] 直接源码依据**：本轮配套 `V72_CURRENT_CODE_AUDIT.md`；仅据最新上传 ZIP，不代表远端已修改、已跑新实验或已有 V7.2 结果。
