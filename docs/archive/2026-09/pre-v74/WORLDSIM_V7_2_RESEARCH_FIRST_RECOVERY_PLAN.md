> 历史归档（2026-09-10）：以下为原版本事实与指令快照；当前执行以 docs/RESEARCH_STATUS.md 为准。

# WorldSim V7.2：先检索、再迁移的研究推进计划

> **2026-09-07：本计划/交接的执行范围已被替代，status=`rejected`（目标不匹配，非全部方法被证伪）。** 当前使用 [EAS-VGGT Recovery Plan](WORLDSIM_V7_2_EAS_VGGT_RECOVERY_PLAN.md) 和 [最新状态](../../../RESEARCH_STATUS.md)。下文 A/B、R1–R7、外部补全/神经 LiDAR 队列及 shutdown 判据属于历史，不据此启动任务。已完成的代码、I/O、run 和各自正/负结果保留；纠偏见 `V71-F64/F65`。

日期：2026-09-07。任务：`WS-V72-R0-RESEARCH-PLAN-01`，状态：`done`（仅指本计划交付）。代码事实基线：`e6bb9a971114e7b60f234f9c2cf4766ee784973a`。后续实验均为 `pending`。本次依据用户“每次卡点先网络检索顶会或优秀开源，再结合项目迁移”的要求制定；2026-09-07 已重新连接开机后的 AutoDL，未启动新训练、未读取新的测试质量、未执行关机。

## 1. 先纠正研究边界

上次把 V7.2 整体宣布完成，收口过早。A1 的负结果成立，但路线 B 只完成了 LiDAR4D 公共场景能力运行，还没有方法候选的同协议比较；它是未完成工作，不能据此判为科学失败。原总计划第 4 节也要求两路完成同信息量比较后再选路。

直接代码证据是 `scripts/decide_worldsim_v72_d1.py::main`：`route_b_pass` 被固定写成 `False`，没有 B 的结果输入；A 的 dev gate 不通过时直接产生 `close_method_claim`。因此，该脚本能计算 A1 的既定门槛，不能完成原计划要求的 A/B 选路。旧 `D1_DEV_GATE.json` 保留原样，新计划在决策层纠正其解释。

本轮继续 V7.2 的未完工作，用 `WS-V72-R*` 标识恢复阶段。版本号不使旧 dev 重新独立，也不使旧失败消失。旧 A1 拒绝证据继续保留，停止的是无证据重复该候选，不是停止整个研究。

建议主问题继续保持：**给定稀疏多帧 LiDAR、传感器标定和已知刚性 Actor 轨迹，如何构建可复用对象与背景，在留出视角／时刻产生准确的完整扫描，并改善稀疏、遮挡区域的几何和回波？**

A 提供对象几何研究支线，B 提供完整场景任务。第一轮优先补齐 B 的直接竞争者和候选，同时做一个 A 的结构迁移；之后依据各自同协议增益选主线。不能只把“完整扫描能输出”算作方法贡献。

## 2. 已有资产、真实卡点和证据限制

| 维度 | 已核实事实 | 本轮处理 |
|---|---|---|
| A1 结果 | 4 dev logs、501 Actors、641,930 rays；512 点上限下，anchored A1 相对 G1：CD 恶化 9.17%，F-score 降 3.58pp，early 增 2.79pp，hit 降 2.76pp | 保留拒绝；不能称“接近通过”或所有补全方法失败 |
| A1 优化目标 | `_ray_losses` 按横向距离 softmax 后对深度求平均；没有深度顺序／透射率。评测使用 literal beam-tube 首交点 | 优先验证训练目标与首交点诊断的不一致；这是代码事实，但尚不是性能失败的已证实主因 |
| 训练与输出 | 593 个训练 Actors；100 epochs；每 Actor 使用 48 条监督射线；主候选是固定 75% TSDF、25% 补全点的输出组合 | 分析原生学习输出与混合／下采样损失，避免把训练收敛等同于几何改善 |
| B 的基础 | LiDAR4D 官方 KITTI-360 一段序列完整跑通；约 2.44 GPU 小时，峰值显存 22,506MiB | 复用环境、扫描数据与投影器；此结果仍是能力／开发证据 |
| 场景组合 | `compose_nearest_returns` 对背景与 Actor 的有限深度取最小值 | 保留作不透明最近交点基线；新增成熟传感器渲染接入，而非声称已具备概率遮挡组合 |
| 数据与 I/O | 已提取 3,882 个 keyframe LiDAR，约 2.70GB；已物化 dev 的 ActorBundleV2 | 复用逐点原点、帧号和姿态；完整场景背景、扫描掩码、附加 sweeps 仍需按任务补齐 |
| 独立数据 | 4 dev logs 已用；3 route-select logs 原始 LiDAR 已提取、质量未读；3 source-test candidates 未打开；外域未读 | dev 正常迭代；route-select 用于冻结后的选择；final 保持独立，不能将 501 Actors 当 501 个独立日志 |
| 资源 | RTX 3090 24GB，读取时空闲；数据盘剩约 55GB；主机 `free` 报告较大内存，但容器配额尚待读取 | 以实际 cgroup 配额设置缓存／worker；单卡串行重任务，取数和轻量预处理并行 |

本轮不再把问题定义、基线覆盖和实验说服力写成“已经完全解决”：目前有可执行合同和初步证据，完整应用、B 的方法对照、独立确认仍待完成。

相关账本：`V71-F43`（负迁移）、`V71-F52`（算子计数混用）、`V71-F54/F62`（真实数据可用性）、`V71-F55～F59`（环境与外部基线接入）、`V71-F63`（A1 负结果）、`V71-F64`（本次整体过早收口纠正）。

## 3. 本次检索到的论文／开源与迁移取舍

以下“迁移方案”是基于项目现状的设计判断，不是论文已经证明能改善本项目。来源优先使用正式论文、作者仓库及其安装文档；核对日期均为 2026-09-07。公开仓库可见不等于本机已复现。

| 论文／开源 | 可借鉴机制与实际可用性 | 对应卡点／落点 | 优先级与限制 |
|---|---|---|---|
| [NKSR，CVPR 2023](https://github.com/nv-tlabs/NKSR) | 稀疏／含噪点云到隐式表面；官方接口接受 `xyz + sensor origin`，支持分块及 CPU 暂存 | 直接接 ActorBundleV2，补一个强表面重建基线，判断点补全劣势是否来自表示；也可用于 build-only 背景 | **第一批**；先原接口推理。它不自动解决语义补全或任意未观测背面；预训练来源单列 |
| [Object-Centric Occupancy Completion，NeurIPS 2024](https://github.com/Ghostish/ObjectCentricOccCompletion) | 作者已发布训练／测试、对象占据标注和长轨迹隐式解码代码 | 保留逐帧观测，迁移对象局部编码／occupancy decoder，构建 A2 | **A 的结构候选**；原生依赖 Waymo、SST／MMDetection3D 和检测跟踪；本项目用给定 GT 轨迹替换前端，须命名“GT-track adapted”，不能冒充官方检测表复现 |
| [LiDAR-RT，CVPR 2025](https://github.com/zju3dv/LiDAR-RT) | Gaussian ray tracing、动态场景图；作者已发布训练评测，原生支持 KITTI-360／Waymo | 与已有 KITTI-360 对接，补齐更近期 B 基线；复用其 renderer、场景图和稀疏几何参数化构建 B1 | **B 的首选**；官方测试是 RTX 4090，3090 兼容性须实测。OptiX／图形驱动能力是明确工程风险 |
| [DyNFL，CVPR 2024](https://github.com/prs-eth/Dynamic-LiDAR-Resimulation) | 背景与对象分场、传感器感知的组合；作者记录 RTX 3090 测试 | B 的机制参考；若 LiDAR-RT 的宿主 OptiX 不可用，优先迁移这套组合实现 | **B 的替代实现**；当前已有源码，但官方 Waymo 数据链不能默认可用；nuScenes 适配版需明示协议变化 |
| [SplatAD，CVPR 2025](https://github.com/georghess/neurad-studio) | 作者发布滚动快门与 LiDAR 专用 gsplat 扩展、相机和 LiDAR 联合渲染 | 若 B 的 rasterization／时间建模更适合当前容器，作为第二选择；也提供未来相机扩展入口 | **备选**；默认多模态，额外图像必须单列信息预算，不与 LiDAR-only 方法直接声称公平胜出 |
| [Gau-Occ，CVPR 2026](https://arxiv.org/abs/2603.22852) | LCD 补全、Gaussian anchors、几何对齐图像融合；本次未定位到可确认作者身份的完整官方代码／权重入口 | 借鉴“先补几何、再构造紧凑表示”的阶段设计；用于检查 A2→场景输出的完整性 | **机制参考**；不是 GaussianOcc。先用已开放的对象占据代码实现相应子问题，不把图像语义或扩散模块直接全套叠入 |
| [DynamicVGGT，CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/papers/He_DynamicVGGT_Learning_Dynamic_Point_Maps_for_4D_Scene_Reconstruction_in_CVPR_2026_paper.pdf) | 动态点图、时间模块、分阶段学习以及点图到动态 Gaussian 的任务联动 | 学习它的动态任务组织与几何先验保持方式；若误差定位到时间不一致，再引入局部时间编码 | **机制参考**；图像模型且输入条件不同，当前没有证据支持重训整个基础模型。搜索命中的同名仓库 README 实为 VGGT，不能据名称当作作者实现 |
| [U4D，CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Xu_U4D_Uncertainty-Aware_4D_World_Modeling_from_LiDAR_Sequences_CVPR_2026_paper.html)／[作者代码](https://github.com/worldbench/U4D) | 以语义不确定区域引导分阶段 LiDAR 生成及时间建模；可见训练和生成入口 | 检查“难区域分配”和时间一致性的近期邻近工作，避免再把泛泛 uncertainty 当新颖性 | **相关工作／后续候选**；生成任务、语义熵、额外分割器与几何 FREE/UNKNOWN 不同；当前不把整套扩散列为必跑 |

两项立即可执行的工程迁移已经有直接文档依据：[NKSR 使用说明](https://github.com/nv-tlabs/NKSR/blob/public/NKSR-USAGE.md)给出原点输入与分块接口；[LiDAR-RT 安装文档](https://github.com/zju3dv/LiDAR-RT/blob/main/docs/INSTALL.md)给出 KITTI-360 数据布局、CMake 版本经验和容器驱动要求。后者支持现有数据意味着不必先等待 Waymo 才推进完整 B 实验。

## 4. 每遇卡点的固定工作流程

1. **先定义卡点。** 写清触发条件、错误或缺口、影响模块、现有证据；区分安装／数据／数值／标签／任务不适配／方法效果。一个候选失败不自动升级为路线失败。
2. **先上网查。** 工程问题先搜索报错、上游 issue／PR 和官方安装说明；方法问题搜索同任务、同失效机制的顶会论文和作者代码，再补最近两年的直接竞争者。已记录同一问题且条件没变时复用链接和补丁，避免重复搜索仪式。
3. **做一张迁移卡。** 至少给出首选与可行替代的机制差异、输入／监督、依赖、单卡成本和本项目接入点；找不到合适方案时写清检索范围，不能编造开源支持。
4. **做最小辨别实验。** 默认只跑能区分原因的实验：一个训练 batch 的必要数值检查、一个代表场景的完整链路、或已有预测的定向重读。通过后直接进入正式比较。
5. **实施或切换。** 可修复工程错误就修；方法缺口换可解释机制／公开基座；数据不可得则改用已经可访问的等价任务协议，并明确监督差异。当前实验达到资源估计后先诊断，不因计时自动结束项目。
6. **更新当前状态和统一账本。** 记录来源、改动与证据、下一步；质量未改善也写清下一项可执行工作。论文新颖性与工程完成度分别报告。

默认每个实质卡点先做 1 次首选迁移和必要的 1 次不同机制替代，避免漫无目的全排列。这是初始工作顺序，不是“试两次就关闭项目”的门槛；仍有有依据的可执行路径就继续。只有不可绕过的授权／宿主权限／新增算力需求，才把具体需求交给用户，并继续独立工作。

迁移卡放在对应 run 的说明中；统一失败事实源仍只有 `docs/RESEARCH_FAILURES.md`。模板：

```text
task_id / failure_ledger_refs / 当前卡点与证据
检索日期 / 官方论文、仓库、issue URL / 核实的 commit 或 release
首选与替代机制 / 额外数据与监督 / 本项目接入文件
最小辨别实验 / 预期能排除的原因 / GPU 与磁盘估计
实际结果 / 保留或切换理由 / 下一项动作
```

## 5. 第一轮 A：先证明表示问题，再迁移 A2

**A 的研究假设：** 对稀疏对象，保留逐帧支持和可见性、在观测约束下预测局部几何，比不分区域的点集补全加固定比例混合更适合留出射线。这一假设尚未验证。

第一步复用 A1 缓存，做两项有辨别力的诊断：比较平均深度代理与首交点的误差方向；比较纯补全、TSDF 及既有 anchored 输出，定位混合与下采样是否改变已观测表面。只作开发诊断，不重算旧 canonical 或改写旧 gate。若误差主要来自位姿／标签，先修复数据生成并重建新版本数据，不用网络掩盖错误。

第二步接入 NKSR。输入为 `build_points_actor_m` 和 `build_ray_origin_actor_m`，所有法向估计、尺度与可见支持仅用 build。输出 mesh／field，再用统一表面采样与已有 evaluator 对比 G0/G1/G3。参数只在开发数据上确定，正式表同时给原生输出和 512-cap，不能仅凭密度增加宣称收益。

第三步迁移 A2：从 Object-Centric Occupancy Completion 的轨迹特征与局部 implicit decoder 开始。先得到不加本项目机制的 adapted baseline，然后只加入“按观测支持约束的局部几何修正”这一项改动。TSDF 有效近表面区域用置信度加权的一致性约束；未知区域由学习先验生成，但不能将未观测区域标为空；冲突观测允许传感器容差内修正，不能宣称所有实测点永远正确。TSDF 未定义区域不做伪造的有符号距离残差。

A2 不使用 75/25 拼接比例作为主要结构，不新增 F/O/U 名义上的小头来代替几何。若接入需要大量检测跟踪系统，优先独立运行必要的对象编码／解码计算图；修改范围和官方版本差异一并记录。

训练可正常使用 dev 做有限的配方选择、早停和误差分析；本轮固定两项关键消融：去掉观测支持约束、去掉时间信息。若训练读出与主评价仍不匹配，则引入经过说明的有序几何读出；传感器均值深度和 literal 首交点仍分表，不能换算子后宣布旧几何指标被修复。

相较 `V71-F63`，此处改变的是表示、逐帧信息利用和修正范围；相较 `V71-F43/F52`，保留同算子比较并重新建立迁移证据。若仅公开 NKSR／adapted occupancy 已解释改善，则把它们作为更强基座，不将基座替换本身宣称为本项目新方法。

## 6. 第一轮 B：补齐正式比较，完成完整场景实验

**B 的研究假设：** 在成熟神经 LiDAR 渲染器内，对同一组场景几何参数加入 build-only、可见性约束的表面监督，可以减少稀疏／新视角区域的几何漂移，并改善完整扫描；收益应超过仅增加优化步数或仅调整 ray-drop 读出。

执行顺序：

1. 优先接 LiDAR-RT 的 KITTI-360 loader 和 renderer。复用现有 slice 做工程开发，按实际需求补 bbox XML、时间戳与标定；不得凭现有点云目录假设这些已齐。冻结一个真实动态场景，完成 build→训练→留出扫描导出。
2. LiDAR4D 旧运行继续保留；先逐项对齐扫描、构建帧、留出帧、量程、投影网格和评价，再决定旧权重能否复用。不能默认不同官方切分天然一致。
3. LiDAR-RT 基线跑到其协议要求的阶段；候选 B1 使用同一基座、数据与轨迹输入，只增加与同一几何参数相连的观测支持／表面约束。先审阅上游已有损失，若已有同义项，就不重复添加并宣称创新，回到误差定位选择真实缺口。
4. build-only 背景与 Actor 使用世界坐标、给定轨迹、真实时间统一渲染；保留原点、beam 参数、intensity 与有效性掩码。不能向 forward 提供 target range、target Actor 命中标签或 target 补建的背景。
5. 先实现完整场景，再谈跨域。第一轮必须输出整帧 range、返回／空 bin、intensity 和点云，以及动态 Actor／静态背景／遮挡边界分项结果。

公平性：B1 对未改动 LiDAR-RT 是主要配对比较。LiDAR4D 是独立表示参照；若它不能消费同等轨迹先验，应明确列为不同先验面板，不能以此单独证明公平优势。B 若最终入选，至少保留两种可运行外部神经 LiDAR 方法；必要时以 DyNFL 替代不适配基座，SplatAD 则需另列图像条件。

物理边界：接收器未检测到回波，不自动意味着光束穿透物体并命中背景；光线传播、几何遮挡和检测概率按上游传感器模型区分。仅有返回点的 KITTI-360／nuScenes 使用公开投影协议的空 bin，不冒充真实硬件 no-return 标签。B1 不依赖 A2 先胜出，可先使用 G1／NKSR 的 build-only 支持继续推进。

若 OptiX 失败：先依据作者文档检查 CMake、CUDA、`libnvoptix` 与容器 graphics capability，再查对应 issue／补丁。如果所缺库由宿主注入、容器内不能解决，则切换已有 DyNFL 的 CUDA 场实现与可访问数据适配，或 LiDAR4D 基座上的方法比较；同时明确对象编辑能力尚缺的部分，不能把替代运行伪称为完整可组合场景。

## 7. 让实验回答问题，而不是持续加门槛

| 实验 | 主要比较 | 回答的问题 | 必须同时给出的边界 |
|---|---|---|---|
| A 表面基线 | G0、G1、G3、NKSR、adapted occupancy | 当前数据上成熟表示能否直接解决缺口？ | 相同 build／target、额外预训练、原生点数与 cap；同一组样本 |
| A2 配对增量 | adapted occupancy vs A2 | 本项目的观测支持修正是否有效？ | 已观测／新可见区域、FREE violation、CD/F-score、early/hit |
| B 完整扫描 | LiDAR4D、LiDAR-RT、B1 | 是否有真实的任务级增益？ | 相同投影协议；轨迹先验差异显式标注；动态与静态分开 |
| 机制消融 | 固定几何换读出；固定读出换几何约束 | 收益来自几何、传感器读出还是工作点？ | 增益不能靠少报回波、更多点或更多训练步解释 |
| 稀疏与遮挡 | 预先固定的 build 帧减少、距离与遮挡分层 | 最困难的有效输入是否受益？ | 稀疏设置对所有方法相同；未知区域不捏造真值 |
| 应用 | 原场景留出扫描；Actor 移动／增删与遮挡重排 | 输出能否组成完整可用场景？ | 编辑后无实测 GT 的案例只评一致性／可视化，不能当真实传感器准确率 |
| 独立确认 | 3 route-select logs；最终 source 与第二公开协议 | 开发效果是否在未参与选择的数据上保持？ | 数量按日志／drive 统计；相邻片段不作为独立日志 |

主表采用日志等权宏平均及逐日志配对差值，Actor／ray 是嵌套观测。4 个 dev logs 只能支持开发判断；置信区间再窄也不能弥补独立日志少。第一轮一个固定种子；有明确增量的最终候选补到 3 个种子，避免为已被全面支配的候选重复烧卡。

保留原 A 的 CD 至少改善 5% 或 F-score 至少增加 2pp，以及 early 增量／hit 降幅不超过 1pp，作为开发效应参考；它不再控制整个研究是否结束。新协议在 route-select 前固定主指标、密度和副作用限值，不追溯修改 A1 阈值。B 默认以同协议完整扫描误差改善及返回覆盖为主；若选择效率主张，须先固定精度非劣范围、硬件和时延口径，再看结果。

只有在候选与外部基线都真正完成可比实验时，才可作对应路线的负判定。没有运行、没有数据或实现不完整，用任务 `pending/blocked` 和具体原因表示。D1 新脚本需显式接收 A/B 各自证据及 `comparison_complete`，缺 B 时不得自动输出“双方失败”。

route-select 读取后保留选择集身份；若根据其失败继续开发，就如实记为已消费，不再称新的独立确认。source/external final 在最后配方固定后读取；最终负结果不回流成反复挑测试集的过程。新的无标签 metadata 可用于筹备独立数据，但外部权重的训练暴露也须核对。

## 8. 实施顺序、代码落点和资源安排

下表是后续工作队列，不代表实现已完成。时长仅为单张 3090 的初始规划区间，包含方法差异的不确定性；正式运行以首个完整场景的实测吞吐重估。

| 顺序／任务 | 可交付物与代码落点 | 预计 GPU | 状态 |
|---|---|---:|---|
| R0 `WS-V72-R0-RESEARCH-PLAN-01` | 本计划；当前状态与 F64 纠正；AGENTS 固化检索优先流程 | 0 | done |
| R1 `WS-V72-R1-DECISION-AND-DIAGNOSIS-01` | 新决策 schema／独立脚本；A1 代理与首交点诊断。建议新建 `scripts/decide_worldsim_v72_recovery.py`，旧脚本保留历史身份 | ≤0.5h | pending |
| R2 `WS-V72-R2-NKSR-BASELINE-01` | 新 `baselines/nksr_adapter.py`，已有 ActorBundleV2→表面→统一评价 | 0.5–2h | pending |
| R3 `WS-V72-R3-LIDARRT-BASELINE-01` | 新 `baselines/lidarrt_adapter.py`、场景数据配置；至少一个完整动态场景 | 2–6h，另计编译 | pending |
| R4 `WS-V72-R4-LOCAL-OCCUPANCY-01` | adapted occupancy 基线与 A2，新增 `geometry/local_occupancy.py`；2 项机制消融 | 4–12h | pending |
| R5 `WS-V72-R5-FULL-SCAN-CANDIDATE-01` | B1 配对实验与完整扫描；新增 `render/neural_scene_renderer.py`，复用上游 forward | 4–12h | pending |
| R6 `WS-V72-R6-ROUTE-AND-APPLICATION-01` | 冻结候选后 route-select、跨场景表、完整应用演示 | 首轮实测后排期 | pending |
| R7 `WS-V72-R7-INDEPENDENT-EVIDENCE-01` | 胜出路线核心消融、最终源域与第二公开协议、论文与可重放交付 | 根据入选路线排期 | pending |

建议 R1 后先做 R3 的安装／数据准备，把 CPU 时间与 R2 的 GPU 推理交错；R3 通了尽快获得 B 的完整场景证据，再开展 A2／B1。只保持两条有依据的候选支线，不把表中所有参考模型全部训练。

单卡安排：

- 默认一个重训练占 GPU，已有 LiDAR4D 峰值约 22GB，不能叠跑另一重任务。GPU 稀疏小批任务允许在显存安全且吞吐实测改善时批处理。
- 数据按日志分片、build／target 分离；只补所需 sweeps／标注。复用已解决的选择性提取器，不能重新下载全量 nuScenes 或在 55GB 余量内堆多套大环境。
- 优先缓存解码后的数组、pinned memory 和有界预取；worker 数量由容器配额决定。关键 CUDA kernel／geometry accumulation 保留适当精度，AMP 用于支持的网络模块。
- 评价使用分块最近邻／射线查询，避免整帧 `torch.cdist` 的二次内存；输出按场景缓存，一次正式评测完成必要指标。
- 在一次有代表性的吞吐检查中记录 GPU 利用率、I/O wait、rays/s、峰值显存、wall time。利用率低才定位并改 batch／pipeline，不为追求仪表盘满载增加无价值运算。
- 环境失败先参考 `V71-F55～F59`；只在隔离环境／build copy 处理，不改用户全局 channel、shell 或已成功运行环境。

只做与风险对应的检查：决策缺失证据不能变成失败、推理没有 target depth、坐标与量程有效、一个小 batch 的有限 loss／梯度、一个完整输出可读。通过后进入真实实验，不反复全库测试、全量哈希或只为元数据小改重训。

## 9. 完成与停机规则

“计划交付”“候选被拒绝”“路线比较完成”“整体研究交付”分别记录。单个 gate、单个模型或安装失败，不触发整个项目完成，更不触发 shutdown。

用户此前“全部完成后关机”的要求继续有效，完成条件是：应执行的基线／方法／应用／独立证据已处理；可执行卡点已完成检索与合理迁移；负结论能指向完整可比证据；文档与产物一致；没有未结束训练和未保存结果。因真正外部阻塞无法推进时说明具体依赖，不能将其改写成科学失败再自动关机。

本轮仅交付计划与上述决策纠正，不开启长实验，也不把用户重新开机解释为“计划写完立即再次关机”。后续进入执行阶段时按 R1→R3/R2→R4/R5 推进。如果最终合理候选仍都没有增量，诚实报告研究结论和已探索范围，不保证一定能得到正结果或顶会录用。

## 10. 可追溯来源与本次核实边界

- A1 数据、训练与评测：`/root/autodl-tmp/runs/worldsim_v72/WS-V72-P2-A1-OBSERVATION-CONSTRAINED-TRAIN-01/20260906T223000Z__a1-observation-train-s7210-r1/summary.json`；`/root/autodl-tmp/runs/worldsim_v72/WS-V72-P2-A1-OBSERVATION-CONSTRAINED-DEV-01/20260906T230000Z__a1-observation-dev-s7210-r1/D1_DEV_GATE.json`。dev fingerprint=`1a64f864c665a277ff21571908f5b73622e282ab74c1af70377b2ceb62d953ec`。
- 本次只读代码：`scripts/run_worldsim_v72_a1_observation_constrained.py`、`scripts/decide_worldsim_v72_d1.py`、`motion_proj/worldsim_v72/data/schema.py`、`motion_proj/worldsim_v72/render/scene_compositor.py`。
- 对象占据作者配置：[ococcnet.py](https://github.com/Ghostish/ObjectCentricOccCompletion/blob/main/configs/ococc/ococcnet.py)；用来核对轨迹数据、隐式解码与 UNKNOWN 处理依赖，而非已移植声明。
- Gau-Occ 的补全预训练及场景输入：[正式补充材料](https://openaccess.thecvf.com/content/CVPR2026/supplemental/Lv_Gau-Occ_Geometry-Completed_Gaussians_CVPR_2026_supplemental.pdf)。多帧累积进入我们项目时须处理 Actor 运动，不能只做 ego 对齐而把运动拖影当密集真值。
- DynamicVGGT 的机制以正式论文为准；本次同名代码搜索命中的 [README](https://github.com/lifchrs/DynamicVGGT/blob/main/README.md)介绍 VGGT 原作，未据此认定作者身份、权重或复现能力。
- LiDAR-RT 的已发布状态与安装约束分别以上文作者仓库、安装文档为依据；NKSR 的分块与原点接口以上文使用文档为依据；没有引用它们的摘要速度作为本机实测值。

本计划中的模块路径和 GPU 时长是待实施设计。当前只有文献核对、远端代码／资源读取及计划文档完成；没有新增 benchmark、模型增益或测试结果。
