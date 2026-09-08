# V7.3 已观测查询域的双向点集评价

状态：2026-09-08定义与实现，等待r5最终评价结束后运行同一旧开发cohort。该补充不改变训练目标、不读取外部20日志模型质量。

当前target→surface距离与0.2m召回是有效单向测量，不能称为Chamfer或完整表面precision。nuScenes/AV2稀疏扫描并没有覆盖Actor完整表面；将所有生成点直接匹配稀疏target，会把未扫描区域也当成误差。

核对[SS3DM NeurIPS2024官方基准](https://github.com/THU-LYJ-Lab/SS3DM-Benchmark)后，保留其区分可见性处理与双向点集指标的思路。该基准有稠密合成GT，官方实现还使用距离截断；本项目不能直接照搬其完整表面定义或声称复现其分数。

本轮补充严格限定为**原登记heldout时刻、首回波明确归属于此Actor的全部原始束**。每个方法在相同束上查询固定显式Actor表面的字面第一三角交点，得到预测点集P；原测量端点形成G。二者都使用已知轨迹下的Actor规范坐标，多次heldout测量直接合并。P不按其离GT的远近裁剪，不根据目标位置生成新采样；所有有限前交点，包括错误早交点和晚交点，都保留。

- precision：P中到G最近距离≤0.2m的比例。
- recall：G中到P最近距离≤0.2m的比例；这是**射线读出点集召回**，与原target→连续三角面召回分开。
- F-score：以上两者的调和平均。
- 双向距离：分别报告P→G与G→P非平方欧氏距离均值，`beam_point_chamfer_l1_sum_m`为两者之和，单位m，不做远离群值截断。
- 原始缺失率：固定评价束中没有表面首交点的比例，与以上分数并列。

没有任何有效Actor返回束的对象保留于cohort，指标不可用；有测量但完全没有预测时P/R/F按0记，距离不可定义，同时单列无预测对象数和缺失率。距离表始终附有效Actor/日志数，不能让全缺失模型因距离缺项获胜。每个Actor内完成点集统计，再按日志内Actor均值及独立日志等权汇总；配对区间只在至少两个独立日志时估计。

该域由原观测与归属定义，不能评价未观测完整Actor表面，也不包含未提供的no-return束。最近邻允许邻束互相匹配，因此点集分数可能掩盖某一条束的错误；原字面early/hit/free/miss和全场景遮挡评价仍是必要的并列证据。完整规范表面precision/Chamfer仍缺稠密独立GT，不能因新增这些字段而宣称该限制已解决。

入口`scripts/evaluate_worldsim_v73_observed_points.py`只做CPU BVH与最近邻；保存每方法每Actor的预测点、原测量点、返回mask/距离及全计数，未重新进行神经推理。PCA使用既有0.06m/20邻域/最多1536片定义，其他方法读取其原保存表面。CPU BVH与原GPU算子有数值实现差异，须披露。

登记`WS-V73-M4-OBSERVED-POINTS-01/20260908T043000Z__development-fixed-surfaces-r1`，覆盖全部75旧开发Actor/5日志，含23个无heldout归属返回对象。比较PCA、r5 joint、同短窗r6、full_track r9、M1 native fusion、CAPA r2与AdaPoinTr Y-up r2；不同标签/TTA预算与空输入路径仍明确保留，不能把所有差异归因于结构。r5−r6回答同短窗视觉联合路径，后续r10/r11补充同full_track的主方法/原生控制。
