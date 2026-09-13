# V8.1 无卡阶段报告：候选图谱完成，模型失效待检验

2026-09-13；`WS-V81-CPU-01 / 20260913-cpu-r2`；seed 8101；状态 **CPU_COMPLETE_WAIT_GPU**。

![Architecture components](figures/worldsim_v81/architecture.png)

无卡条件下可执行的数据筛查、方法可用性审计、可视化、输入干预准备与推理接口检查已经完成。主要发现是**参考几何与 RGB 纹理可能来自不同深度层，以及干净四格对照不足**。模型推理为0，尚无已验证的 SOTA badcase；H1–H5 均 NOT_TESTED，V8.2 为 NO_GO_PENDING_EVIDENCE。人工 verdict=null。

## 数据与筛选口径

现有落盘数据含35个六相机完整场景、27个日志。r1先检查12日志、14场景、28窗口；发现混杂后，r2按同一 metadata 顺序扩大至全部27日志、34场景、68窗口、340个选中样本。另1场景因完整时序帧不足排除。没有根据模型误差选样本，也没有放宽阈值。全部已曝光数据仍为 DISCOVERY。

6120个固定320×180 ROI中，1527个通过几何支撑初筛，1311个同时通过极暗/过曝筛查。参考使用相邻四个 keyframe 的 LiDAR，当前样本只作 INPUT_PROMPT。全部1527份参考通过输入/参考样本不相交检查。RGB主模型不读取这些 LiDAR。

参考按扫描位姿 sensor→ego→world→camera 变换，膨胀剔除源帧和目标帧标注物体，要求至少两个扫描在8像素格内深度一致，拒绝深度边界与可检测的当前遮挡。原始点文件没有逐点时间戳，所以只称扫描位姿补偿，不声称精确逐点 deskew。平面初筛要求≥30点、6×6格覆盖≥0.25、RANSAC内点比例≥0.85、RMS≤0.12m；这不等于视觉或科学验收。

纹理由梯度能量和32-bin灰度熵的30%/70%分位联合定义；另记录特征/重复匹配密度、LiDAR二维/三维分散度与深度跨度。低重叠要求 calibrated frustum 上界≤0.1；高重叠要求留出几何可见支撑下界≥0.4且视差≥1°。中间区保留，缺少 raw prior、视差或模型预测时保留 null。

| 候选格 | ROI | 日志 | 地面 / 非地面候选 |
|---|---:|---:|---:|
| C00 高纹理×高重叠 | 18 | 9 | 11 / 7 |
| C10 低纹理×高重叠 | 28 | 12 | 20 / 8 |
| C01 高纹理×低重叠 | 215 | 26 | 139 / 76 |
| C11 低纹理×低重叠 | 182 | 22 | 123 / 59 |

这是几何/曝光/可见性筛查后的候选计数，尚未扣除逐图发现的混杂。四格图保留原始规则选中的案例，不能当作干净自然实验。

![四格候选；尚无模型预测](figures/worldsim_v81/evidence_grid.png)

## 当前真正发现的问题

全部46个高重叠候选已逐图复核，另检查平面招牌和2个低纹理低重叠示例，共49项记录。36项标为 CONFOUND_EXCLUDE_MAIN，包括栅栏、植被、多层遮挡、湿路反光、路缘深度边界。7个非地面 C00 均有明确混杂，其余也只称候选。

例如 scene-0535 的 CAM_FRONT_RIGHT_04、scene-0071 的 CAM_BACK_LEFT_14：前景网格贡献高频纹理，LiDAR主平面却可能属于后方墙面。不能据此说模型在有充分纹理的墙上表现如何。scene-0632_fd5b6a5c_CAM_FRONT_RIGHT_02 的低梯度墙板可作为下一步自然候选；scene-0800_a4354e58_CAM_BACK_LEFT_13 的暗墙板仍需排除阴影/曝光解释。平面招牌 scene-0626_9a9c05fe_CAM_BACK_LEFT_13 用于单列的合成输入干预。

**剔除已识别混杂后，按同一日志、地面/非地面、距离桶匹配，完整四格 block=0，独立匹配日志=0。** 当前数据不能估计 H1 四格交互效应。ROI/像素数量不能补足独立日志；未匹配组间差异不能称非加性退化。evaluator只允许完整冻结匹配组进入探索性统计，缺格返回不足。

开GPU不会自动解决这个数据缺口。先用冻结自然候选寻找可复现模型错误，再做同场景视角/纹理干预。自然H1主结论仍需补充干净高重叠对照和独立日志；新增数据或重定义视觉ROI时另建带来源队列，不能覆盖本轮冻结结果或按模型误差改阈值。

## CPU控制和干预

1527个几何候选均尝试当前 INPUT_PROMPT 的 RANSAC 平面控制，1516个产生有效指标，11个无有效预测。非地面候选595例中位 depth MAE=0.135m，地面921例=0.438m。这包含原始候选混杂，只衡量局部输入几何与独立参考一致性，不是 SOTA 性能、DriveMVS复现或科学 headroom。

![纹理、重叠与简单平面控制](figures/worldsim_v81/factor_controls.png)

已生成8张固定规则候选卡，展示 RGB、参考深度、输入平面控制误差及几何侧视；32组同点数 prompt placement 输入；1个视觉较干净平面招牌的跨视角纹理衰减输入。保持原图尺寸、位姿及上下文视图对应关系。r1含栅栏的2组纹理示范作为混杂历史保留，r2使用上述招牌。

![同场景纹理干预输入](figures/worldsim_v81/factor_escalation_inputs.png)
![相同点数、不同位置的输入](figures/worldsim_v81/prompt_placement.png)

合成mask明示使用参考平面，属于真值辅助 DIAGNOSTIC_ONLY，不进入自然发现主表，不证明真实分布的因果效应。当前没有SOTA goodcase/badcase，只有参考/控制一致与混杂候选。

## 模型准备与验证

DVGT-1、VGGT-1B、DGGT nuScenes 官方代码与完整权重已落盘，三个模型均通过 strict meta 参数名/形状加载。官方预处理也已在真实六相机窗口上通过：DVGT为[1,1,6,3,288,512]，VGGT/DGGT为[6,3,294,518]。没有模型前向。

7项针对性检查通过：相机z与射线距离、跨扫描支撑、缺失预测分母、日志bootstrap、nuScenes盒坐标轴、保留目标视角与时序相机顺序、拒绝混杂/缺格补足匹配组。空输出evaluator返回 WAIT_MODEL_OUTPUTS，没有生成假模型比较图。

隔离环境 /root/autodl-tmp/envs/worldsim-v81 继承现有基础包，Torch2.4.1+cu121/torchvision0.19.1与DVGT官方建议Torch2.8/CUDA12.8有差异；GPU前向与kernel尚未验证。继承环境pip check有3项既存mapanything/nuScenes-devkit冲突，本轮流式metadata路径不调用它们，不能称整个环境无冲突。大依赖安装曾中断，cgroup未记录OOM kill，归因未确定；之后用完整本地wheel和小包分步安装完成。

DVGT/VGGT为第一轮主模型，DGGT为条件扩展。DriveMVS、FocusGS、VGGD官方入口当前缺少可运行代码，不用代理冒充其输出。版本、一手来源和路径见[方法审计](WORLDSIM_V8_1_METHOD_AUDIT.md)。

## GPU接续与边界

冻结408个DVGT/VGGT任务（68窗口×full6/sparse3/sparse2×2模型），另有保留目标相机的干预及temporal18入口。各模型先一个full6窗口检查坐标、原生输出、峰值显存，再决定批量。VGGT保留原生深度及公开相机基线定尺度结果，不用ROI LiDAR拟合。未输入/未预测相机不当MISS，预测内部缺失保留coverage分母。

H1–H5均NOT_TESTED；PSNR/SSIM/LPIPS、prior recovery、geometry-vs-rendering mismatch均未产生。DGGT core depth/gs_map准备不等于完整renderer接通。10个AV2 reserve日志保持 quality sealed，本轮未用其质量选模型；独立确认前仍需核实原始RGB/标定合同。

完整证据：/root/autodl-tmp/runs/worldsim_v81/WS-V81-CPU-01/20260913-cpu-r2/，含registry、parquet、1527份参考NPZ、68份输入合同、控制/干预/队列、逐图复核、HTML与状态。r1保留历史。轻量记录在 docs/autoresearch/worldsim_v81/。cgroup实测0.5 CPU、2GiB；r1核心atlas约629s，r2增量atlas约335s，不包含全部准备时间。

**无卡阶段已停下，无自动续跑。建议2×48GB GPU分别运行两个主模型，或1×80GB顺序执行；主机≥64GB RAM、8 vCPU。** 这是首轮预算建议，实际显存待首个GPU窗口测量。详见[GPU交接](WORLDSIM_V8_1_CPU_HANDOFF.md)。

failure_ledger_refs：V74-H2-F11/F09/F10、V74-F01；failure_ledger_delta：V81-F01（公开方法边界）、V81-F02（参考混杂与零匹配组）。无新增SOTA scientific failure。
