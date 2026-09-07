# V7.3 原生DPT适配结果

## r2：扩大已有日志训练

M1 r2=`done`，code=`424743fc`，run=`20260907T151000Z__native-dpt-fit25-dev6-s7301-r2`。25fit/6dev场景、60epochs/1500更新，1130.19s。原生DPT 32,654,562参数，project最大变化0.004067；训练峰值1.348GiB、RSS8.817GiB。

按独立日志先平均场景MAE再等权汇总：fit 20日志 3.669→0.849m；development 5日志 2.870→2.485m，配对变化-0.384m，日志bootstrap95%区间[-0.889,0.395]m，4/5日志改善。6dev场景中5改善、scene1089 1.176→2.318m退化；scene0519 MAE改善而轴向early比例0.258→0.598。scene0994零Actor诊断点保留且不进入MAE均值；不能静默删除。

仍只是build点插值诊断，非新时刻表面或字面首交点。区间跨零、支持量不均、部分early变差，F05继续active；不把这次结果写成跨日志方法胜利。详表=`docs/autoresearch/worldsim_v73/m1/r2_summary.json`与`r2_log_analysis.json`。

# V7.3 M1 原生几何通路训练诊断

2026-09-07；实现commit=`29595e20`；run=`20260907T150300Z__native-dpt-fit4-dev2-s7301-r1`。

已真实训练 VGGT 原生 DPT **32,654,562 参数**，非冻结输出外挂。4fit/2development scenes，4build时刻×3相机、378×672，15epochs/60场景更新。第一步全DPT梯度范数832.50，原生第一层project卷积最大参数变化0.000526568。冻结前缀峰值4.578GiB，逐视图DPT训练峰值1.348GiB，RSS7.706GiB，总耗时118.18s。

| scene | role | Actor点数 | 原生MAE m | 训练后MAE m | 原生/训练后0.2m命中率 |
|---|---|---:|---:|---:|---|
| scene-0015 | fit | 124 | 6.134 | 1.466 | 2.42% / 8.06% |
| scene-0071 | fit | 3687 | 3.404 | 1.480 | 8.38% / 20.23% |
| scene-0100 | fit | 2130 | 1.144 | 0.417 | 3.29% / 52.91% |
| scene-0317 | fit | 860 | 0.762 | 0.250 | 8.95% / 61.74% |
| scene-0048 | development | 104 | 3.594 | 2.867 | 0.00% / 2.88% |
| scene-0359 | development | 11 | 5.043 | 5.095 | 0.00% / 0.00% |


该结果证明原生几何通路可学习、当前12视图DPT执行无需新增GPU；不证明完整Actor重建。数值为1/5留出的build点在相机像素处的轴向深度误差，**不是场景硬首交点**。输入LiDAR还用于固定build尺度，模型不使用held-out诊断点拟合尺度或梯度。稀疏LiDAR点的投影/扫描内运动不确定性保留；仅nearest-pixel不能保证完整相机可见性。

四个fit场景Actor MAE均下降，开发scene0048下降，但scene0359仅11个Actor诊断点且MAE5.043→5.095m，无法支持跨日志泛化。train早于目标0.2m的比例有增加，进一步说明MAE改善不等于free-space一致性。统一记录`V73-F05 active`；不是视觉几何适配全路线失败。

先检索后迁移：[CAPA](https://research.nvidia.com/labs/dvl/projects/capa/)以稀疏build信号作逐样本PEFT；[CVPR2024 TTA depth completion](https://openaccess.thecvf.com/content/CVPR2024/papers/Park_Test-Time_Adaptation_for_Depth_Completion_CVPR_2024_paper.pdf)以稀疏深度监督适配。当前先扩大共享DPT训练到全部已有25fit/6development scenes、60epochs，保持同输入/损失/lr1e-5；以更多独立日志和实际支持报告判别数据不足。若泛化问题持续，再比较build-only局部适配/显式传感器融合，不能以扩大step数直接宣称解决。

下一run=`20260907T151000Z__native-dpt-fit25-dev6-s7301-r2`；config=`configs/worldsim_v73/m1_native_geometry_full.yaml`。规范表面生成/自由空间/首事件、M2局部查询与完整场景读出仍待开展。单卡24GB当前只证明足够该DPT候选；上层LoRA与主查询联合反传仍需独立测量。所有checkpoint/优化器/输入/训练日志保存在原始run目录；report与summary纳入Git。
