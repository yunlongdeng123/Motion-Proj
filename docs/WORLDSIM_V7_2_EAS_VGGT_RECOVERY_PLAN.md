# WorldSim V7.2 EAS-VGGT Recovery Plan：贡献与证据优先版

修订：2026-09-08；revision=`3`；任务 `WS-V72-E0-CONFERENCE-INTEGRATION-02`。E1–E5 已按本计划执行：官方 VGGT-1B/Pi3X 与 MapAnything 已接通，连续证据 late fusion、可靠性折扣、有序 blocking/detection 回波、物理/外观所有权桥和 SE(3) 组合均有正式 run。route-select 上证据 proper score 有小幅正结果，但几何无改善、Pi3X 为负，冻结 source cohort 因 0 个可观测候选而不可计算。完整数值、canonical run 与可写/禁写范围见 [EAS-VGGT 结果](WORLDSIM_V7_2_EAS_VGGT_RESULTS.md)。本计划后文保留原预注册目标，执行结果优先于其中的待验证措辞。

用户最新优先级是形成最有竞争力的主会研究，而非满足单卡、2GB 或极小参数量。资源影响执行安排，不决定研究边界。沿用 V7.2 和 EAS-VGGT 主线，继续继承 V7/V7.1；原外部补全 A/B 与 R1–R7 队列不恢复。唯一当前状态为 [RESEARCH_STATUS.md](RESEARCH_STATUS.md)，既有结果不被本次计划追溯修改。

## 1. 论文主问题与贡献组织

**研究主问题：在相同的稀疏测量信息下，如何把视觉几何先验适配成可由传感器查询、可随刚体轨迹编辑，并且与外观保持对应的动态场景表示？**

建议工作标题：**EAS-VGGT: Evidence-Conditioned Visual Geometry for Sensor-Consistent Dynamic Reconstruction**。EAS-VGGT 是主实现名称；只有冻结迁移证据成立后，才在标题/摘要中扩大为多基座通用 EAS。

主任务固定为**稀疏 build 观测条件下，未参与构建的射线/时刻上的动态场景重建与第一回波预测**。输入为 RGB 窗口、有限 build LiDAR、标定和已知 Actor 身份/刚体轨迹；输出物理表面、连续证据、独立外观状态，以及场景级第一回波距离分布和 no-return 概率。默认是离线重建与指定轨迹条件下的重放/编辑，不是未来轨迹预测。主实验的 target RGB、LiDAR 和 ray outcome 都不输入模型；若补充研究“给定 target RGB 的深度补全”，另设协议，所有基线共享该输入。

V7.1 的 Actor-box 条件 median 仍是正结果与起点，但单靠它不足以完成本任务。返回存在性、背景—对象遮挡、物理—外观对应关系升级为必需证据；实现从 EAS 对象状态组合展开，不依赖 AdaPoinTr 补齐全局表面或 LiDAR4D 代替 EAS 模型。

叙事链条固定为：**两个正常工作的视觉几何基座→排除尺度/位姿/动态对齐/表面转换误差→发现剩余的观测缺口→用 EAS 的专门机制处理→同信息基线、跨场景和跨几何来源验证**。缺口是否存在、由什么主导，必须先测；不能把预期写成已有发现。

| 拟主张 | 方法内容 | 必需证据 | 单独不足以支撑该主张的结果 |
|---|---|---|---|
| C1：观测条件下的视觉几何适配 | 局部三维/跨视图证据学习；区分 FREE、OCCUPIED、UNKNOWN 及观测支持量 | 超过同 LiDAR 预算的校正、融合、深度适配、scalar 和 LiDAR-only 模型 | 原视觉模型输给额外获得 LiDAR 的模型 |
| C2：可组合的传感器回波表示 | 物理表面证据与外观 opacity 分离；场景级有序返回/无返回模型 | surface 与 return 两表、完整有效 query 分母、对象/背景遮挡、密度和读出消融 | 条件 median 改善、NLL 下降或 CDF 单调 |
| C3：有对应关系的可编辑动态状态 | 参数归属隔离、物理到外观的前向联系、SE(3) push-forward | 实际表面深度/轮廓/遮挡对应，真实时间留出及轨迹干预，多来源复用 | RGB 完全未变导致 PSNR 不变；仅验证坐标恒等式 |

这三个贡献描述同一个 EAS 场景表示，不包装成三个互不相关模块。SE(3) 恒等式和梯度非干扰是结构基础；论文价值还需来自新观测建模、跨模态对应和实证收益。

措辞统一为 **sensor-observation consistency / return-order consistency**。不将同行概括为“纯图像自监督大模型”，不预先声称“首次发现因果性缺陷”“碰撞安全”“全链路小于 2GB”或“PSNR 不变所以世界自洽”。录用概率没有可计算保证；本计划优化的是新颖性边界、可证伪性、任务完整性和外部有效性。

| 相对上一版 | 本版实质变化 |
|---|---|
| VGGT+小头作为默认上限 | 结构化 evidence adapter 为起点；PEFT/部分 decoder/全量微调按证据开放 |
| 固定 M8 上的条件回波为主要终点 | 保留继承对照，最终必须包含场景第一回波存在性和遮挡组合 |
| 独立外观+非干扰 | 加入物理/外观可见表面对应，不能成为两套无关世界 |
| 单基座、现有少量日志确认 | 两个开发基座、第三来源冻结迁移；扩大独立训练和未见场景 |
| 等容量 scalar、G0/G1 | 补齐同测量校正/融合、CAPA/深度适配、LiDAR-only、神经 LiDAR 能力基线 |
| 先长时间做接口/资源前置 | 数据/模型可用性与一轮跨模型瓶颈诊断合并；随后进入四组实质实验 |

## 2. 正结果继承表：保留什么、还缺什么

以下数值来自现有账本及论文的 canonical 摘录，不是本轮重跑。不同算子、cohort 的绝对 early/hit 不跨行比较。

| 资产 | 已有正证据 | 继承方式 | 必须保留的边界 |
|---|---|---|---|
| V7 SceneIR / HARP-3D | Actor canonical 融合、逐 ray/frame 来源、KEEP/PROJECT/COMPLETE/UNKNOWN；matched-ray PROJECT 不落入该观测的 pre-hit free interval | 作为 EAS 输入合同和观测证据编译器；身份、轨迹和 UNKNOWN 语义保留 | 只对匹配观测成立；P20/P22/P23 已揭示 target-nearest 代理会低估真正 early，不能恢复成全表面安全证书 |
| V7 三维几何 | P2 在 433 Actors 上 CD `0.254→0.168m`、precision `79.59→97.67%`，Actor retention `100%` | 证明 canonical 观测处理和来源约束有价值，保留 compiler 基线 | 是当时 AV2 编译协议；不当作 EAS-VGGT 独立泛化结果，也不以 precision 代替首回波正确率 |
| V7.1 M7/M8 表面学习 | M8：66 dev Actors，CD `237.67→231.46mm`，literal hit `47.67→50.43%`；hazard early `27.80→26.37%`；相对 M7，moving frame-distance 再降 `7.01mm` | M8 为物理起点，保留固定 observed anchors、局部 child set 与逐帧覆盖监督；M7 为权衡参考 | dev 已暴露；M8 clear early `21.56→22.39%`，且 hazard 不优于 M7；不是简单基线的普遍支配者 |
| V7.1 M39 分类回波 | 固定 M8 几何，unit→learned categorical：all early `18.19→17.67%`，hit `62.28→64.46%`；hazard early `−0.62pp`、hit `+2.32pp` | 保留 M35/M38 训练出的证据头与 M39 reader 作为可回退端点；优先学习图像提供的证据增量 | 只是源域机制证据；M43 AV2 hit `+5.89pp`，early `+0.23pp`，hazard early `+0.54pp`，跨传感器 early 改善未成立 |
| M22/M28 刚体与所有权 | 12 Actors / 36 Actor-frames，5,791 物理 Gaussians；逆变换 query 残差 `3.40e−14`，刚体距离残差 `1.24e−14m`；物理接口不接受 visual state | 直接继承 `PhysicalActorField` / `ActorPose` 类型边界与 canonical inverse query | M28 是接口/理论实现，M22 是数值组合验证；都不是新轨迹预测能力或重建精度增益 |
| M25/M27 外观学习 | 只优化外观可提升画质且物理保持固定；M25 footprint PSNR `15.86→16.94dB`，M27 分层外观到 `17.73dB` | 继承分支隔离及外观独立扩容；让视觉头学习足够的专属 primitives | 与原 StreetGS `25.35dB` 仍有差距；M27 使用已暴露视图。309 物理载体并不够承担完整外观 |
| M49 连续测度分析 | 99,208 rays 上得到有限衰减精确恒等式；能解释降低 child 权重为什么反而增加前方概率 | 作为 reader/训练损失设计依据及归因工具 | 不能把“降权”叫作单调变安全；分析所用 GT 边界不能输入推理策略 |
| V7.2 数据与 I/O | 3,882 个 keyframe LiDAR 已物化；干净 dev 为 4 logs / 501 Actors；输入与 targets 分开、曝光记录、密度匹配已有实现 | 保留 source dispatch、ActorBundleV2、角色门控、G0/G1 简单基线与有限读取流程 | 尚未证明这些 logs 的 RGB 多相机素材及 VGGT 特征缓存已齐全；数据前置工作需要补这一部分 |

直接证据入口：`paper/results/eas_evidence.json`、`paper/results/results_macros.tex`、`paper/sections/supp_v7_restored.tex`、`paper/sections/05_experiments_appendix.tex`；run 以 [EXPERIMENTS.md](EXPERIMENTS.md) 的稳定 task ID 查找。关键 canonical：

- M8：`run://worldsim_v71/WS-V71-M8-TEMPORAL-FRAME-COVERAGE-01/20260904T202000Z__m8-temporal-frame-s71110-r2`。
- M39：`run://worldsim_v71/WS-V71-M39-CATEGORICAL-AUTHORITY-COMPOSITION-01/20260905T070000Z__m39-categorical-authority-r1`。
- M22：`run://worldsim_v71/WS-V71-M22-SE3-DYNAMIC-STATIC-COMPOSITION-01/20260904T161000Z__m22-se3-composition-r2`。
- M49：`run://worldsim_v71/WS-V71-M49-VISIBILITY-SIGN-BOUNDARY-01/20260905T121500Z__m49-visibility-sign-boundary-r1`。

## 3. 方法方案：以已有 EAS 为起点，补齐真正缺失的机制

### 3.1 强视觉先验与连续观测证据的学习

复用 V7 canonical compiler 的来源合同、M8 的表面学习及 M39 的读出端点。视觉基座提供 dense geometry 和可用特征；build LiDAR 提供有原点、时间、Actor pose 和来源的射线约束。采用保留局部三维邻域与跨视角 token 的 evidence transformer，而非预先压成少数全局统计量。输出物理局部残差、连续 F/O/U、支持量/冲突统计及回波所需的证据描述。

物理点图必须用 build-only 标定/尺度约束变换到 metric world，再按 Actor 轨迹运动补偿到 canonical frame；背景单独处理。先在同步多相机帧验证，再增加时间窗口。全局对齐、动态累积和点图转表面的误差分别记录，不把凸包/球支持/错误融合产生的侵入归给基座本身。

训练顺序仍为“固定几何学证据→单独开放几何→联合适配对照”，保留几何 set/plane/scale/frame 监督和独立 operator 表，防止 M19/M20 的补偿与 scale shortcut。observed anchors 的来源不覆盖；若需要修正带噪 build observations，显式建立观测噪声/校正版本，不暗改锚点。

三态 softmax 本身不是新贡献。reader 只读取 occupied scalar 时，其表达能力不比 scalar head 更强；F/O/U 的价值必须来自额外可辨别的观测监督、跨帧信息和可靠性输出。支持量也不自动等于校准后的认识不确定性。所有这些机制都需要同输入、同容量、同监督预算的消融。

### 3.2 从条件回波到场景级有序事件测度

**继承端点：** M39 的连续证据加权、全 ray 深度分类和 median 保留为同几何基线；M49 的有限衰减恒等式继续解释权重变化。其条件是 Actor box 内存在回波，不能直接升级为完整传感器模型。

**拟研究增量：** 在 canonical 物理表面上保存证据测度，经 SE(3) 放置后，沿 world ray 形成有几何位置与局部支持的表面事件，再对全部 Actor 和背景的事件统一排序。分开表面阻挡与传感器检出，不把 RGB opacity、primitive 数量或 UNKNOWN 当成回波概率。

第一实现采用有序表面事件的分类分布。设第 k 个事件的阻挡概率为 u_k，发生阻挡后检出回波的概率为 v_k：

\[
T_k=\prod_{\ell<k}(1-u_\ell),\qquad
q_k=T_k u_k v_k,\qquad
q_{\varnothing}=T_{K+1}+\sum_kT_k u_k(1-v_k).
\]

因此 `sum(q_k)+q_empty=1`，并能表达“前方表面阻挡却没有可检测回波”，而不是把它误作透明自由空间。`q_k` 的局部深度核给出连续深度/离散 bins 上的返回测度。训练采用同一 outcome distribution 的 likelihood/proper score；部署的存在性、距离与 depth-bin 概率读取同一分布，阈值与解码规则在 dev 训练前登记。

这是**不透明表面第一回波的近似模型**，不是完整波形、双程光学或多次散射理论。概率分解本身是已有观测建模思想，不单列为“首次”创新；研究增量是如何从视觉先验和连续 build 证据学习可定位、可组合、与外观有对应的表面事件。u/v 在稀疏监督下可能不可辨识，不能叫作真实反射率或已辨识材料参数；需要表面/自由区间监督及 `v=1`、单 hazard head 对照判断是否有实质价值。

不能重复 F22/M35/M38：事件深度与厚度受物理表面及传感器分辨率监督约束，不能把各向同性 Gaussian 的长前尾直接当体密度沿空域累积。事件支持过小又会重现 F20 的 coverage 损失，因此必须让视觉先验/物理头实际补足局部支持，并报告 support recall。只改善 query decoder、不改善所声称的表面，不算几何贡献。

同时引入显式 surface quadrature mass：分裂/重复一个 primitive 时分配原质量，而不是让总阻挡随点数翻倍。相同位置/事件参数且总质量守恒的离散化，只要求 **reader** 不变；这不是整个可学习 encoder 在任意重采样下天然不变。F40/F41 已说明总量守恒并不足以改善 early，因此它是消除混杂的约束，仍需定位和观测监督。

`no-return`、`unsupported`、`invalid/unfired` 分开：no-return 是有效测量机会的实际 outcome，UNKNOWN 是几何证据不足。没有事件支持造成的漏检仍计入完整 query 指标，不能用 unsupported 排除难例，也不能宣称该空间已知为空。事件式 reader 是否优于“原 M39+同容量 no-return head”和标准单 hazard 模型，由同几何实验决定；不预设复杂模型一定获胜。

### 3.3 物理与外观：梯度隔离，前向保持对应

状态为 `S_i=(P_i,E_i,A_i,T_i(t))`。物理 P/E 读取标定的几何/观测监督；appearance 有自己的 primitives、颜色、opacity 和优化器。RGB loss 不更新 physical geometry、evidence、Actor pose 或共享的可学习物理 backbone；PEFT/全量微调时必须显式分离可学习参数归属，不能只对最终 P 调一次 detach。

同时，外观必须是 `A_i = H_app(stopgrad(P_i), image_features, residual_state)`：物理到外观的输入保持实时关联，detach 只截断梯度，不把外观永远锁在过期物理快照。visual primitives 有父表面/Actor 来源并共享同一轨迹；允许独立容量与可学习 residual，避免重现 309 个物理载体兼做高质量外观的容量失败。

采用对外观单向生效的表面对应约束，结合可见范围内的深度、轮廓和遮挡顺序监督；其范围由真实可观测区域决定，不把 LiDAR 未见后表面全当空。必须与完全独立双分支、共享载体、隔离但无对应约束做同容量比较。物理修正后 PSNR 是否变化只是辅助指标；主要问题是同一可见车身、边界与前后遮挡是否仍能在两种输出中对应。

### 3.4 SE(3) 轨迹与状态组合

继承 `Q_{i,t}(x)=Q_i(T_i(t)^{-1}x)`，以及 `μ_world=Rμ+t`、`Σ_world=RΣRᵀ`。整体坐标变换 g 下满足 `Q_{gT}(gx)=Q_T(x)`；场景 ray 和全部 owners 一起变换时，距离参数及事件顺序保持。仅移动一个 Actor、固定传感器时，遮挡和回波应改变，其他 Actor 的 canonical 状态应保持。

连续时间用给定姿态的旋转/平移插值，先做真实留出时刻，再做声明范围内的轨迹编辑。由相同 `T_i(t)` 同时驱动物理与外观；新增 SE(3) 理论只声称表示/组合层的性质，不把 π³ 的 view permutation equivariance 混同于空间 SE(3)，也不宣称 VGGT image encoder 精确等变。

## 4. 四组核心实验

### A. 瓶颈真实存在：基座、对齐和转换分别诊断

选择有效官方 checkpoint 的 VGGT 与 π³ 作为默认开发基座；MapAnything 为第三种保留几何来源。E1 先核对权重、代码版本与原生能力，并固定选择；若候选缺有效 checkpoint，按可用性换位必须发生在质量比较之前，不能用差结果挑基座。DynamicVGGT 暂作方法参照，官方仓库未发布预训练权重；复现训练成功后可加入独立比较，随机动态头不算其论文模型。

同一预选日志、同一 RGB/build LiDAR，报告：原生深度/点图；加正确标定/尺度与运动补偿后；再经公共 surface/query adapter 后。输出跨基座的 geometry、可见 free-space intrusion、return-order 误差和射线剖面图，并逐项统计转换引入的增量。oracle pose/高质量表面只作诊断上界，不能混入主方法输入。

若误差主要被正确对齐或简单融合解决，就不以“基础模型观测缺陷”为论文主张。转而研究 EAS 中仍存在的证据/遮挡/跨模态对应问题，先查文献再定义有区别的候选；不重复一个没有剩余优势空间的小头。

### B. 同信息比较：排除多拿 LiDAR、更多容量和普通深度适配

| 编号 | 对照 | 主要排除的解释 |
|---|---|---|
| B0 | 视觉基座原生输出 | 输入较少的参考，不单凭它证明 EAS 机制有效 |
| B1 | 基座+相同 build LiDAR 的尺度/位姿校正 | 只是修正 gauge/alignment |
| B2 | 基座+相同 LiDAR 的融合/TSDF/标准 ray constraint | 普通观测处理已经足够；几何点数和支持预算匹配 |
| B3 | 同几何/特征、近似容量的 scalar response head | 三态名义或额外参数解释收益 |
| B4 | LiDAR-only+同规模物理/回波模型 | 视觉基座只是装饰；也保留 raw fusion/TSDF 简单端点 |
| B5 | CAPA；并以 Marigold-DC/TestPromptDC 补充强深度适配 | 只是稀疏深度适配；原生深度指标与公共 scene adapter 的结果分开 |
| B6 | 冻结 M8/M39、M39+同容量 no-return head、标准单 hazard、无 F/O/U、无来源/冲突、质量不守恒消融 | 增量是否来自新的 EAS 机制；逐项改变，不把多个开关一次合并 |
| B7 | EAS 主模型：不同表面/证据/外观对应开关 | 建立 geometry×reader 与 ownership×correspondence 的归因 |
| B8 | DyNFL / 现有 LiDAR4D 等成熟神经 LiDAR 同协议比较 | 任务完整性与性能位置；它们是竞争者，不是接入后代替 EAS |

所有核心方法共享 RGB/测量窗口、LiDAR 点/beam 预算、目标隔离、标定、轨迹、训练数据和合法 TTA 信息。B1–B7 的同信息组不能隐藏额外原始 sweeps、target RGB 或目标传感器标定监督。MapAnything 等原生可接收几何提示的模型还需**原生同提示输入**对照，不能故意只用它的无提示配置。

离线训练、测试时优化、无优化 forward 分列，报告 steps 与总时延；为每个强基线给到正常收敛和合理官方配置，不能用 EAS 已充分训练而对手未收敛制造差距。原生神经 LiDAR 通常用不同信息/逐场景优化，原生设置放参考表，严格匹配信息的版本标明 adaptation；不跨协议比较论文绝对数值。

### C. 改善发生在正确对象上，且两种世界保持对应

| 证据表/实验 | 必须输出 |
|---|---|
| 几何表 | CD、surface precision/recall、frame coverage、可观测自由空间侵入；固定 literal reader 的回波结果；不同密度预算 |
| 传感器表 | 有效 beam 上 return-existence NLL/Brier/precision/recall/F1；返回距离 MAE/RMSE、early/hit/late；漏检/虚假返回和对象/背景遮挡；条件与全分母分开 |
| 跨模态对应 | 同一相机下物理与外观可见深度偏差、轮廓距离、遮挡顺序错误；再用独立测量/标注分别锚定，避免两者一起错却一致 |
| 参数干预 | 外观颜色/容量/优化步数变化的 physical drift；物理修正后重新条件化外观的对应性；同容量基线和扩容成本 |
| 轨迹干预 | 全局 gauge、单 Actor 平移/旋转、多 Actor 遮挡变化、真实留出时刻；合成轨迹解析真值与真实传感器真值分开 |
| 表示干预 | 固定总质量的 primitive 分裂/重复、前方/后方事件增减、前遮挡无检出情形；证明正确排序/概率归一化不等于已经提高实际精度 |

相机深度/轮廓/遮挡比较只在实际有相应真值或可靠可见性标签的区域计算。仅有 box 不能当精细 silhouette GT；伪标签基准须标明来源和不确定性，并以独立人工标注或现有真值补充。若采用人工评测，执行前另交付完整盲评协议；本轮不要求用户评分。碰撞指标只有独立碰撞几何与 query GT 时才能加入，回波改善不外推为安全结论。

存在性以全部有效发射机会为分母；early/hit/late 以真实正回波 rays 为主分母，预测 miss 仍留在分母中。距离误差同时报告成功检出条件下的结果及包含漏检惩罚的完整 outcome score。不能通过增加无回波格子稀释 early，也不能通过少输出回波降低条件 MAE。固定范围、传感器分辨率和容差，在各方法之间共享。

统计单位为 driving log/独立 scene，报告 paired effect、逐 log 分布和 log bootstrap CI；原始 rays/Actors 保留完整分母。主终点预定为 scene-level outcome score 与 hazard early/all-hit 的联合变化，并设置 geometry/cross-modal 非劣条件；数值容限与预算在 E1 后、候选首次 quality read 前固定。M39-only 不能重新作为完整任务的结束线。

### D. 新场景、跨传感器、跨几何来源

训练/开发用两个基座的数据，冻结一个共享 EAS core 后在第三来源测零更新迁移。**原生 latent 维度和语义不同，不能直接说一个 projection head 自动通用。** 主迁移接口采用 canonical metric geometry、统一证据 schema 和同一个公共图像特征通道；特殊基座 token 可作为主实现增强臂，但其专属 projector 的训练成本与数据必须计入。

明确分三种结果：共享 adapter 直接迁移；每基座分别训练；允许 build-only TTA。第三来源上新训投影头、拟合 calibration 或调整阈值，都不算第一种。若增强臂依赖 VGGT token，而通用 core 通过共享图像编码器使用第三种几何，准确称“未见几何来源迁移”，不宣称任意未知 backbone 的 native feature 即插即用。

目标是在 source 未见场景和至少一个跨传感器数据集上分别验证。两个开发基座加一个第三来源是本项目通用性假设所需的设计，不是顶会录用的机械数量门槛。训练随机性在关键最终配对上用多个 seed 报告，不把所有消融反复重跑；数据/容量曲线选择少量预先固定点。

## 5. 数据与资源：为独立结论配置资源

| 数据资产 | 用途 | 约束 |
|---|---|---|
| 既有 nuScenes 593 train / 66 legacy dev 与 4 个已消费 dev logs | 继承端点、代码桥接、机制诊断 | 不因升级 backbone 或版本而变成独立测试 |
| Waymo Perception 官方数据 | 默认主任务候选：联合 RGB、metric LiDAR、range-image/outcome 与动态对象 | E1 核实 firing/return/invalid 语义和所有历史依赖，固定独立 split；采用官方训练池的充分规模，而非限于本地现成几段 |
| AV2 / 可合法恢复 firing 语义的其他传感器 | 外域几何/距离验证；有真实 outcome 时再做存在性验证 | 已暴露 80 个 AV2 logs 不洗白；positive-only 点云不能制造 no-return 标签 |
| 受控合成场景 | 表面事件遮挡、无检出阻挡、SE(3)/双分支对应的解析实验 | 不冒充真实新视角扫描/碰撞真值 |

主数据目标为数百个独立训练场景，以及数十到上百个未见评测场景；最终规模由官方可用数据和 log 依赖审计确定，不在看到质量之后挑场景。若原 nuScenes 只剩 3 个候选日志，就保留为有限补充并扩展独立数据，不能把它写成充分主验证。未知公开预训练曝光单独报告，不能保证基础模型训练集与测试域绝对无交集。

E1 已在任何 Waymo payload/quality 下载前，按固定 SHA-256 排序冻结 798 个 training contexts 为 `600 train / 99 development / 99 route-select`，并把 202 个 official validation contexts 整体冻结为 source-test。清单来源、摘要和成员写入 `configs/worldsim_v72/data_roles.json`。官方 bucket 需要 Waymo 注册与 gcloud 授权，当前只完成角色冻结，不能把未物化清单写成已验证数据。

existing LiDAR I/O、source dispatch、ActorBundleV2 和目标隔离继续复用。新增多相机 payload、标定/裁剪矩阵、原生 beam outcome、逐点时间/运动补偿和特征缓存；cache 只存合法输入，target mounts 分离。缺失字段先依据官方 schema 恢复，不能从“没有点”推断“发射且无回波”。GPU 推理/训练批处理与 CPU 解码预取并行，DDP/分片/cache 只优化吞吐，不改变数据和指标合同。

容量顺序为：有足够局部三维/跨视图信息的结构化 adapter→必要的 PEFT/decoder 微调→有数据支持的全量微调。冻结基座是解释性起点，不是方法宣言；若最终用了微调，标题和成本报告同步，不能继续声称完全冻结。RGB 梯度仍不得越过物理所有权边界。无需为了参数数字从头重训 1B，也不为小于 2GB 把有效特征删掉。

资源优先用于强基线收敛、独立训练/测试数据、必要容量、跨来源验证。报告 backbone 特征生成、离线训练、per-scene/TTA、联合推理、缓存/数据 I/O 五部分成本；adapter-only 显存不能写成全系统显存。当前 3090 可做起步工作，更大训练需要时按实际需求配置；实际训练资源在实施时按已明确的实验需求安排。

## 6. 四组实验与既有任务 ID 的执行映射

| 工作包 | 稳定任务 ID / 状态 | 完成产物与推进条件 |
|---|---|---|
| A：缺口与数据协议 | `WS-V72-E1-VGGT-EVIDENCE-IO-01` / done | VGGT/Pi3X/MapAnything 官方模型、RGB/cache/beam schema、nuScenes route/source I/O 已完成；Waymo 只保留冻结清单 |
| B：同信息机制学习 | `WS-V72-E2-LEARNED-VISUAL-EVIDENCE-01` / done-with-boundaries | late fusion 超过 no-visual，early/scalar/Pi3X 为负；有序 blocking/detection 机制在 controlled rays 成立，真实全场景 outcome 未完成 |
| C：对应与动态应用 | `WS-V72-E3-DECOUPLED-APPEARANCE-01`、`WS-V72-E4-SE3-RIGID-TRAJECTORY-01` / done | held-out rendering、父表面所有权与 RGB 梯度隔离、SE(3) 解析/真实 pose 组合完成；外观质量仍有明显差距 |
| D：冻结泛化与文稿 | `WS-V72-E5-FROZEN-CONFIRMATION-01` / route-supported, source-inconclusive | route proper score 支持；source 0 support，不作泛化主张；MapAnything 为第三来源诊断；稿件已更新 |

E1 的数据物化、基座能力与初步诊断交错进行，E3/E4 接口可在 E2 训练时推进；不要等某个 CD 门槛才开始整个场景实现。每个实质卡点先查顶会/官方开源，再做针对当前失败的新机制或工程修复；已有精确问题复用已知解决方案，不无限扫参，也不因单候选失败提前关机。

直接接入点：`motion_proj/worldsim_v71/authority_contract.py`；M8/M22/M25/M27/M39/M49 scripts；`motion_proj/worldsim_v72/data/`、`evaluation/`、`render/`。`eas_vggt/` 已包含双基座公共输出、target-free cache、Sim(3) 对齐和有序回波测度地基；E2 的 learned evidence/appearance core 尚未实现。旧 D1/A1、NKSR/LiDAR-RT 队列不恢复；神经 LiDAR 在新协议下作为竞争者独立运行，不充当 EAS 的方法实现。

## 7. 失败继承、贡献判定与稿件结构

| 相关记录 | 防止重复的要求 |
|---|---|
| `V71-F20`、`V71-F22`、`V71-F37`、`V71-F38`、`V71-F39` | 同时报告 support coverage 和 event localization；不能仅换成小 support 或重新累积弥散 density |
| `V71-F23`、`V71-F24` | 物理表面与 reader 两表分开；原生 3D 监督保留，拒绝 decoder 补偿/scale 膨胀解释 |
| `V71-F40`、`V71-F41`、`V71-F42`、`V71-F47` 与 M49 | 质量守恒、proper loss、衰减和单调 CDF 都不是性能证明；新增视觉/观测信息与分层收益必须实测 |
| `V71-F25`、`V71-F28`、M24–M28 | 正确 inverse query 和参数归属；同时补上前向对应与画质，不能只给接口残差 |
| `V71-F43`、`V71-F52`、D0 W0–W4 | 外域风险和 scalar 替代解释保留；严格区分 literal/categorical/scene outcome 字段 |
| `V71-F54`、`V71-F60`、`V71-F62` | 数据来源/索引/缺 payload 分开处理；无点不等于 no-return |
| `V71-F63`、`V71-F64`、`V71-F65` | A1 失败与缺比较不影响 EAS 研究对象；补齐完整证据不等于重新选外部完整模型 |
| `V71-F66` | 上一版条件回波、接口非干扰和单来源小规模验证不足以支持通用传感器适配；风险需 A–D 实验解除 |

可写成主贡献的最低逻辑是：两个有效基座上有排除转换误差后的缺口；同信息基线无法解释主要增益；改善落在声称的表面/观测对象上；物理与外观没有分裂；独立数据与跨来源证据支持声明范围。只提高 median 就收窄为回波建模，不能写几何修复；各基座分别训练有效就写架构可移植，不能写零更新插件；若 scalar 已解释收益，就报告这一事实并继续查找有信息增量的机制。

建议主文围绕四组证据安排：Fig.1 同一场景的视觉先验、测量冲突与 EAS；Table 1 两基座分层诊断；Table 2 同信息主比较；Table 3 几何/传感器分表及关键消融；Fig.2 物理—外观对应与轨迹干预；Table 4 新场景/跨传感器/第三来源；成本与容量表。数学部分集中于状态归属、表面测度的离散化条件、有序返回概率及 SE(3) 组合，不把已有恒等式包装为全部创新。

E0–E5 已执行，正式结果见 `docs/WORLDSIM_V7_2_EAS_VGGT_RESULTS.md`。source-test 已按锁定协议读取，但因零可观测候选只产生 insufficient-support 结论；external test 未读。`paper_v72/` 已重写为 EAS-VGGT 当前证据草稿，旧 A1 报告由 Git 历史保留。V7.2 本轮交付在代码、数据/I/O、正式 run、账本、文稿和 push 全部复核后执行用户要求的远端关机。
