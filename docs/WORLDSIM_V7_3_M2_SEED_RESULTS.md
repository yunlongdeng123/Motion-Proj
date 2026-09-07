# V7.3 表面种子结果与覆盖扩展

表面种子批次20260907T161000Z__surface-seeds全部done，code26646089。两个候选仍联合训练原生DPT+查询、120steps、seed7303、同一静止卡车与相同几何/free监督；只将任意体积completion初始化换成表面种子。native额外使当前原生depth输出进入种子位置梯度；FPS离散选择之外的所选深度保持反向路径。它和LiDAR种子均有后续无小位移硬上限的更新。

| 初始化 | hit | early | miss | 原始束free均值(m) |
|---|---:|---:|---:|---:|
| native | 0.7651 | 0.0287 | 0.1805 | 0.0325 |
| lidar | 0.8041 | 0.0431 | 0.1456 | 0.0461 |
| volume | 0.7543 | 0.0524 | 0.1651 | 0.1333 |

native相对volume的free平均侵入下降约75.6%，early约减半、hit小幅改善；LiDAR表面种子得到更高hit，但early/free比native差。两者均未支配LiDAR PCA强基线（early0.70%、free0.0066m），所以只确认表面支持组织有效缓解卡点，尚无方法优越性结论。native的原生depth输出梯度非零，训练峰值5.343GiB、RSS3.014GiB，最终633101个build原生候选中FPS取512个，无LiDAR回退；不能把多视图重复候选数当独立表面支持数。

覆盖复核：早期single-Actor机制对象为距相机最近7.48m、速度0的vehicle.truck，原始build归属点10222；不能代表动态稀疏重建。三相机输入共238个有投影观测Actor，41个速度>2m/s，169个<100个去重非诊断build点；六相机扩大到404/71/278。所有31场景实际均有24视图。新增166个观测Actor来自环视，说明先前前向配置不是有效完整覆盖上限。

进一步边界：六相机队列中的动态开发Actor仍只来自2个独立日志（另外3个开发日志没有满足>2m/s条件的当前窗口Actor），不能用更多射线补足独立性。后续应按build输入与只读轨迹重新组织有运动/观测的窗口，并保留静止、缺视觉或低点数对象的报告。窗口选择不能使用heldout几何质量，旧开发日志不升级为全新最终确认。

Actor点归属来自注释框+0.1m并排除重叠，是代理归属，不是逐点实例分割真值；需保留边界/地面污染风险。历史cohort.actor_count是整个scene的轨迹数，不能与短窗口observed_actor_count直接相除得到视觉覆盖率。已在新数据入口补充分母范围、window_actor_count和lidar_supported_actor_count；在跑r3是旧实现，分析时继续按明确范围解释，不追改历史数据。

M1环视r3当前PID9615 running，run20260907T161500Z__native-dpt-surround25-dev6-s7301-r3，codecf039715。24view冻结前缀已完成，已进入25fit×60epochs训练；前缀至少实测5.510GiB，最终峰值以run日志为准。当前无资源不足，不停机。

下一工作：在环视基线上接共享跨Actor查询训练与等容量逐点/LiDAR-only控制，扩展到运动和稀疏输入；继续处理free梯度的几何方向与表面连贯性。现有仅罚range-d的free项在挡住远背景时可能将几何沿束推向遥远终点，而不是退出局部已知free射线管，这是下一待测机制推断，不是已证明原因。检索依据为[DRC CVPR2017](https://openaccess.thecvf.com/content_cvpr_2017/html/Tulsiani_Multi-View_Supervision_for_CVPR_2017_paper.html)与[nvdiffrast可见性梯度说明](https://nvlabs.github.io/nvdiffrast/)；保持同一显式曲面的硬评价，不以外观透明度或学习置信度规避。
