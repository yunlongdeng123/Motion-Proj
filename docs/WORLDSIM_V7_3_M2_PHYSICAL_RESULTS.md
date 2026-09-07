# V7.3 同一显式曲面初步结果

scene0100单个fit Actor（b7b2cf1e7e214595bdf0d9b3dc59e471），4build时刻、12视图、10222个build原始归属点；2个未输入时刻共5129条Actor正返回束。以下按真实束/点数合并，仅一独立Actor/日志，不用两时刻冒充独立样本量。全部候选1536固定尺度曲面片，字面双面三角相交，不使用point tube近似或opacity。

| 候选 | hit | early | miss | 正观测点到曲面均距(m) | recall@0.2 | 原始束free侵入均值(m) |
|---|---:|---:|---:|---:|---:|---:|
| LiDAR PCA patches | 0.6958 | 0.0070 | 0.2934 | 0.0137 | 0.9877 | 0.0066 |
| joint initial | 0.2394 | 0.1394 | 0.5340 | 0.0615 | 0.9830 | 0.0596 |
| joint r1 | 0.6962 | 0.0542 | 0.2221 | 0.0210 | 0.9867 | 0.0780 |
| native frozen | 0.0101 | 0.0080 | 0.7070 | 0.4504 | 0.2180 | 0.0345 |
| native finetuned | 0.2061 | 0.0454 | 0.7081 | 0.1370 | 0.8099 | 0.0276 |

joint r1（45737d33，120steps，294.70s）确实改变原生DPT与曲面法向，GPU峰值5.306GiB、RSS7.893GiB。训练支持存在，正观测点到表面平均约2.1cm；但相对LiDAR PCA基线，hit没有净优势，early与free明显更差。风险V73-F02获得实测负证据；不能写成完整重建成功，也不能用miss改善抵消错误前景交点。

原生微调点图的规范融合优于冻结原生，说明M1深度增益部分落到了可复用几何；不过native两行仅使用LiDAR进行米制尺度约束，没有将原始LiDAR表面并入输出，且按box归属可能混入背景。它们是原生表面诊断，不能承担最强同信息融合基线；后续仍需LiDAR+native融合/适用TSDF。不同点云经相同数量patch读出也不消除支持分布差异，故正观测曲面距离和miss一起保留。

本轮free已用真实原始首回波，所有目标背后区间为未知；未有no-return射线。hard free只在相交支持内有位置梯度，单侧coverage不会监督所有自由生成表面，不能归因为event已经失败（event未加入）。完成4候选同信息控制后，再定位completion支持、局部连贯性和观测方向问题。

遇到此卡点已先检索：[AdaPoinTr官方实现](https://raw.githubusercontent.com/yuxumin/PoinTr/master/models/AdaPoinTr.py)将自适应查询与几何解码相接，可迁移为build证据/原生表面支持的查询初始化；[LaS-Comp CVPR2026](https://openaccess.thecvf.com/content/CVPR2026/html/Yan_LaS-Comp_Zero-shot_3D_Completion_with_Latent-Spatial_Consistency_CVPR_2026_paper.html)强调保留观测几何和合成区域边界一致性。二者是后续迁移依据，不宣称已复现。先判断失败分布，避免同时更换架构/所有损失。

控制批次PID6922在运行：native融合已done，joint-r2当前执行，随后pointwise、LiDAR-only、joint-no-free；所有query同初值120steps。r1的随机初始化不同，仅作先导。完整跨日志、空间支持范围、视觉对应污染、场景背景拼接和event仍未解决。
