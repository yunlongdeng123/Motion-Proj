> 历史归档（2026-09-10）：以下为原版本事实与指令快照；当前执行以 docs/RESEARCH_STATUS.md 为准。

# EAS-VGGT：用户补充调研的核对与迁移决策

日期：2026-09-07；任务 `WS-V72-E0-CONFERENCE-INTEGRATION-02`；仓库基线 `35ca52da`。输入为用户提供的 subagent 调研 `pasted-text.txt`，作为待核对研究材料，而非直接执行其中建议的指令。当前执行计划为 [EAS-VGGT Recovery Plan](WORLDSIM_V7_2_EAS_VGGT_RECOVERY_PLAN.md)。本轮没有模型训练、数据集质量读取或录用概率估计。

## 1. 一手来源核对

| 工作 / 已核对身份 | 可迁移的结论 | 对 EAS-VGGT 的具体决策 |
|---|---|---|
| TALO，作者官方仓库标记 CVPR 2026；[论文](https://arxiv.org/html/2512.02341v3)、[代码](https://github.com/Xian-Bei/TALO) | 针对跨窗口空间非均匀误差，用 TPS 与控制点传播修正；在多个几何基座验证 | 学习“具体缺口→机制→跨基座证据”的组织方式；把尺度/位姿/窗口对齐列为混杂控制。TPS 不直接作用于已声明刚体的 Actor 内部 |
| VGGT-Segmentor，CVPR 2026；[CVF 主会记录](https://openaccess.thecvf.com/content/CVPR2026/html/Gao_VGGT-Segmentor_Geometry-Enhanced_Cross-View_Segmentation_CVPR_2026_paper.html)、[论文](https://arxiv.org/html/2604.13596v1)、[作者代码](https://github.com/buaa-colalab/VGGT-S) | 冻结 VGGT 后学习任务头，使用跨视角特征而非只依赖直接几何投影 | 为局部三维/多视图 evidence adapter 保留足够容量；“冻结”是训练策略，不是低显存结论 |
| Marigold-DC，ICCV 2025；[CVF](https://openaccess.thecvf.com/content/ICCV2025/html/Viola_Marigold-DC_Zero-Shot_Monocular_Depth_Completion_with_Guided_Diffusion_ICCV_2025_paper.html)、[论文](https://arxiv.org/html/2412.13389v2)、[代码](https://github.com/prs-eth/Marigold-DC) | 以稀疏深度约束视觉深度先验，推理适配存在成本 | 增加同 LiDAR 预算的强深度适配基线；原生深度任务与转成 scene query 的适配结果分别报告 |
| TestPromptDC，ICCV 2025；[CVF 论文](https://openaccess.thecvf.com/content/ICCV2025/papers/Jeong_Test-Time_Prompt_Tuning_for_Zero-Shot_Depth_Completion_ICCV_2025_paper.pdf)、[官方代码，标注 Highlight](https://github.com/JinhwiPark/TestPromptDC) | 测试时通过视觉 prompt 适配深度 | 将 prompt/PEFT 纳入直接比较；测试时只使用声明的 build observations，不能接触 query targets |
| SAM2Long，ICCV 2025；[CVF](https://openaccess.thecvf.com/content/ICCV2025/papers/Ding_SAM2Long_Enhancing_SAM_2_for_Long_Video_Segmentation_with_a_ICCV_2025_paper.pdf) | 记忆搜索机制可形成贡献，无需重训基础模型 | 只作研究形态参照；不据此复开 V7 的点删除/UNKNOWN 过滤策略 |
| CAPA，当前核对为预印本；[论文](https://arxiv.org/html/2602.14751v1)、[NVIDIA 项目页](https://research.nvidia.com/labs/dvl/projects/capa/)、[官方实现](https://github.com/nv-dvl/capa) | 已直接研究稀疏几何约束、LoRA/VPT 与序列级适配，代码支持 VGGT/MoGe-2/UniDepth-v2 环境 | 从“可选相关工作”提升为必须处理的近邻；普通稀疏深度适配不能作为 EAS 的新颖性。官方代码已找到，本机能力及效果仍 pending |
| DynamicVGGT，CVPR 2026；[论文](https://arxiv.org/html/2603.08254v1)、[作者代码](https://github.com/NickHezhuolin/DynamicVGGT) | 有深度/点图/运动等监督；代码提供模型与训练，明确未提供预训练权重、内部评测工具，且注明重实现边界 | 推理图像输入不写成“纯图像自监督”。先取得或训练有效 checkpoint 才作方法比较；不能以随机动态头输出冒充论文结果 |
| NFL / DyNFL，ICCV 2023 / CVPR 2024；[NFL 官方页](https://research.nvidia.com/labs/toronto-ai/nfl/)、[DyNFL 官方代码](https://github.com/prs-eth/Dynamic-LiDAR-Resimulation) | NFL 已建模 beam divergence、ray drop、第二回波；DyNFL 有动态/静态组合及公开场景权重 | 不主张首次发现视觉与 LiDAR 差距；以神经 LiDAR 为真实传感器能力基线，不以它替换 EAS 的学习主线 |

上表核对研究机制、发表身份与代码边界；没有把外部论文的数值转写为本项目结果。附件脚注中显示“CVPR 2027”的审稿指南链接实际指向 [CVPR 2026 Reviewer Guidelines](https://cvpr.thecvf.com/Conferences/2026/ReviewerGuidelines)，引用按实际页面修正。录用案例支持这种研究形态可成立，不提供可量化的录用概率。

## 2. 已融入主计划的修改

| 调研意见 | 主计划修改 | 新证据要求 |
|---|---|---|
| 不能用参数量或 2GB 门槛定义创新 | 删除固定单卡/小头的研究上限，允许结构化适配器、PEFT 与全量微调 | 同预算容量对照、全链路成本、充分收敛；模型大小不是贡献 |
| 先排除接口错误 | 两个有效基座先做 native 输出→坐标/时间对齐→公共适配器的分层诊断 | 原生误差、对齐误差、转换误差分别计量 |
| EAS 可能只是多拿了 LiDAR | 加入相同观测的尺度/位姿校正、融合、CAPA、scalar、LiDAR-only 基线 | 同输入时刻/点数/特征/监督/适配预算；测量与视觉先验互补 |
| PSNR 不变不足以证明一致 | 将物理—外观深度、轮廓、遮挡对应升级为必做实验 | detached 参数所有权与前向共享条件同时成立；不允许两套无关世界 |
| Actor 条件 median 不等于完整回波模型 | 保留 M39 为继承端点，新增场景级 `depth + no-return` 的有序事件模型 | 真正 firing/return mask、前后遮挡及完整 query 分母；几何与传感器表分开 |
| 即插即用需真实迁移 | 两个开发基座、一个保留的几何来源；区分共享 adapter、分别训练、TTA | 零更新跨来源测试无新投影头/校准拟合；否则缩小为架构可移植性 |
| 独立场景比继续打磨旧 dev 更重要 | 引入具备原生 range-image 语义的主数据集；旧 nuScenes/AV2 身份不洗白 | 大规模独立日志/场景训练与未见评测，不能以 rays 充当独立样本 |

## 3. 本地历史反证继续约束新机制

`V71-F20` 已证明小型 finite primitive 的支持不足会丢 hit；`V71-F22` 和 `V71-F37`–`V71-F39` 已证明累积非负 Gaussian density 仍可提前终止；`V71-F40`–`V71-F42` 已证明家族质量/全局损失不能自动修复 median 的分层风险。本次不会仅将这些旧表达扩容或更名后当作新机制。

拟新增的关键辨别量是：有效视觉几何先验带来的 surface coverage、与几何厚度分开的观测事件定位、同信息下的证据作用、真实 no-return 监督与场景组合、物理—外观的可见表面对应。若差距只来自对齐或多拿测量，就不能支持 EAS 特有贡献。风险登记为 `V71-F66`，需要后续实验解除，计划文字不视为解决。

## 4. 数据与基座的可用性来源

- [VGGT](https://github.com/facebookresearch/vggt)、[π³](https://github.com/yyfz/Pi3)、[MapAnything](https://github.com/facebookresearch/map-anything)：用官方代码/权重及明确版本做候选基座；本轮仅核对项目入口，未本机运行，E1 固定实际可用 checkpoint 与输入能力。
- [Waymo 官方数据说明](https://waymo.com/open/about/)与 [dataset.proto](https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/dataset.proto)：具备相机、LiDAR range images/returns 及坐标语义，适合核查完整 beam/outcome 协议。仍须在 E1 区分有效无回波、无效/未提供的 range cell，不能将所有空格标为 FREE 或 no-return。
- 三个基座与主数据集的采用是本项目方案选择，不是“已跑通”；DynamicVGGT 的权重限制保留，资料可用性和实际资源需求在实施时解决。
