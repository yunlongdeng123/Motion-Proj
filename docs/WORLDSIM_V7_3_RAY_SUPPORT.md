# 射线条件的显式曲面吸引

2026-09-09 06:37 UTC。实现与独立对照登记，检查和正式质量结果pending；基于r3结果提交8b012af1。任务WS-V73-Q-V2-01，run `20260909T063700Z__open-charts-lidar-ray-support-s7304-r4`。

![实际组件](autoresearch/worldsim_v73/ray_support/V73_RAY_SUPPORT_ARCHITECTURE.png)

r3开放chart与闭合LiDAR r2没有建立共同优势，对R8仍是missing下降/free侵入增加。根据用户策略二，保持同一开放chart，只新增真实返回条件的方向吸引；不扩大基座或near-boundary free。现有欧氏target→surface本来有位置梯度，这里检验加入射线方向后，正确沿束支持能否改善，不把所有miss都归因于没有梯度。

## 明确的目标及反传

对归属本Actor的原始首返回，读入只读的原点o、单位方向d、沿束距离r，目标x*=o+rd。对实际三角面上的x，令δ=x−x*，δ∥=d(dᵀδ)，δ⊥=δ−δ∥：

`L_ray = mean_r min_(x in generated triangles) || alpha*delta_perp + delta_parallel ||_2`。

固定alpha=20/3，等价于沿束.2m与横向.03m的相对权重；数值沿用当前优化尺度，它不是标定后的测量噪声，也不认证一个.03m占据圆盘。目标单位米，外层权重1，没有增加可学习置信度、opacity、截断饱和、UNKNOWN标签或positive thickness。完整目标为原coverage + .5*finite_beam_free + .05*box_envelope + 1*L_ray；native/event权重仍0。

实现`motion_proj/worldsim_v73/ray_support.py`：对每条射线，将目标平移到原点并按该方向线性拉伸空间，在每个三角形的面内部及三条边求精确最近点。双层分块64条射线×512面，不保存全射线×全部面反向图。选择最优面及最优重心权重时no_grad，随后重建原空间最近点，仅对这个最优距离向顶点反传；在离散选择不切换处使用包络定理。**重心权重须在完整各向异性度量中求最优**，不能仅按投影选择却任意忽略沿束项的权重导数。单位射线由原数据产生、只读，不重学位姿/轨迹。

这是ray-conditioned coverage候选，不是新first-event概率模型。对于非空曲面，未命中束仍能把已有面拉向真实返回；空网格不会因这个loss凭空生面。它也不保证投影面积、正面朝向、无chart重叠、首交点正确或连续排序梯度。更贴近target的后面可能被选中，前方错误面仍由原free约束；最终必须读同一真实三角面、原硬首交点，不能用最近target的面替换物理输出。

## 监督与对照协议

每个FIT step从原full_track `training_rays` 的 `positive_actor` 中独立抽最多1024条；只用原始首返回/既有归属，不把后方被遮Actor当首返回。既有归属是已知盒与membership代理，不能宣称精确分割或已解决逐点时间误差。旧coverage从去重后的端点抽最多1024，free仍从原全near-box束抽最多512，旧两项采样/权重/时段不变。新项按返回观测计数，重复位置的不同观测可能保留：因此**同时增加方向信息与返回观测加权的监督项**，不是相同样本上的纯度量消融；若有收益再拆分这两种来源。

额外抽样使用独立CUDA Generator seed7305，不消耗旧coverage/free的全局RNG；generator状态写入每轮checkpoint并在resume还原。正式r4从fresh seed7304开始，30轮预期11130实际更新，AdamW1e-5/global clip1。与r3相同64chart/4×4/固定尺寸scale.15、1024顶点1152面、LiDAR-only、371ready FIT/67ready DEV；完整414FIT/75DEV/51不可用保留。重新评价实际initial，最终只收口一次，对照r3为主要单因素问题，并附闭合r2/R8。

训练/评估输入只读build；FIT full_track额外时刻是损失标签，DEV无优化。5原DEV日志配对六指标，空预测和无owned分母保持；20新日志质量未读。raw JSONL记录新项米值、横向/沿束距离、可用/抽样owned数；这些随step抽样变化的值不作泛化证明。单GPU低资源仅是这个LiDAR对照的成本，不能代表联合基础模型成本或宣称视觉通路已训练。

## 一手依据及本机边界

[SoftRas，ICCV2019](https://openaccess.thecvf.com/content_ICCV_2019/html/Liu_Soft_Rasterizer_A_Differentiable_Renderer_for_Image-Based_3D_Reasoning_ICCV_2019_paper.html)及[官方函数接口](https://raw.githubusercontent.com/ShichenLiu/SoftRas/master/soft_renderer/functional/soft_rasterize.py)给出距离型软光栅化，说明轮廓之外也可通过距离获得梯度；本候选不复制其概率alpha聚合，也不把软像素解释成真实激光首返回。[DRC，CVPR2017作者页](https://shubhtuls.github.io/drc/)研究可微射线一致性，借鉴观测沿束语义，不引入其体占据表示。这两个来源不证明本各向异性最近面目标最优，也不构成同任务复现或论文新颖性证据。

一次必要检查入口`scripts/check_worldsim_v73_ray_support.sh`：离轴miss三角上比较原截断event的零梯度、新项的非零梯度/下降方向；alpha=1与已有欧氏最近面相符，并在非切换点检查一个坐标的有限差分。再用元数据首个ready FIT Actor，检验新项对位置/法向/高度的实际梯度和一次联合更新，不重复旧factory或其他回归。检查权重不用于正式训练。

正式入口`scripts/run_worldsim_v73_ray_support_r4.sh`。检查结果和实际运行状态另记，当前pending；收口入口`scripts/summarize_worldsim_v73_ray_support_r4.sh`，仅完整final结束后运行一次。若没有共同物理/覆盖收益，依据保存支持分解决定几何/沿束约束下一项，不进行alpha/weight网格，也不回到upper/DINOv3/full FT。

failure_ledger_refs=[V73-F01,V73-F02,V73-F03,V73-F04,V73-F05,V73-F06,V73-F09]；failure_ledger_delta=none at registration。旧风险保持，未证实解决F03，下一V73-F10；三本台账与计划同步，小步push、30分钟ACTIVE、完成不关机。
